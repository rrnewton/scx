//! Frida Stalker-based software RBC for deterministic preemption.
//!
//! This module provides an alternative to PMU-based preemption (see [`preempt`])
//! that uses [Frida Stalker](https://frida.re/docs/stalker/) for dynamic binary
//! instrumentation. Instead of relying on hardware performance counters to count
//! retired conditional branches, Stalker instruments the scheduler's `.so` at
//! runtime, inserting callouts before every conditional branch instruction.
//!
//! A thread-local software counter (`SOFTWARE_RBC_COUNTER`) is decremented at
//! each conditional branch. When it reaches zero, a `YIELD_PENDING` flag is set.
//! The actual yield happens at the next kfunc boundary, where
//! [`maybe_yield_preemptive`](crate::preempt::maybe_yield_preemptive) checks the
//! flag and performs the token-ring yield.
//!
//! ## Why deferred yield?
//!
//! Stalker's dynamic binary instrumentation translates ALL code on the thread,
//! not just the `.so`. Performing complex operations (futex_wait, heap alloc,
//! tracing) inside a Stalker callout corrupts memory because the Rust runtime
//! code is also being translated. By deferring the yield to a kfunc boundary
//! (which already runs safely under Stalker — proven by cooperative yields),
//! we avoid this corruption.
//!
//! ## Why software RBC?
//!
//! PMU-based RBC (`perf_event_open` with `PERF_COUNT_HW_BRANCH_INSTRUCTIONS`)
//! is not available in all environments (e.g., VMs without PMU passthrough,
//! containers without `CAP_PERFMON`). Software RBC provides the same
//! deterministic preemption without hardware support.
//!
//! ## Determinism
//!
//! Like hardware RBC, software RBC counts only conditional branches in the
//! scheduler's executable code. The same code path produces the same branch
//! count, making software RBC fully deterministic: same seed produces the same
//! interleaving and the same trace.
//!
//! ## Relationship to [`preempt`]
//!
//! - [`preempt`] uses PMU overflow signals (`SIGSTKFLT`) to trigger preemption.
//! - This module uses Stalker instrumentation callouts to set a yield-pending flag.
//! - Both use [`PreemptRing`] for token passing and futex-based parking.
//! - Both save/restore [`SimulatorState`] context across yields.
//! - The actual yield for Frida mode happens in [`preempt::maybe_yield_preemptive`]
//!   at kfunc boundaries, not inside the Stalker callout.
//!
//! [`preempt`]: crate::preempt
//! [`PreemptRing`]: crate::preempt::PreemptRing
//! [`SimulatorState`]: crate::kfuncs::SimulatorState

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use frida_gum::stalker::Transformer;
use frida_gum::Gum;

// ---------------------------------------------------------------------------
// Thread-local software RBC state
// ---------------------------------------------------------------------------

thread_local! {
    /// Software retired-branch-conditional counter.
    ///
    /// Decremented at each instrumented conditional branch. When it reaches
    /// zero, `YIELD_PENDING` is set. Initialized to `u64::MAX` (disarmed).
    static SOFTWARE_RBC_COUNTER: Cell<u64> = const { Cell::new(u64::MAX) };

    /// Whether Frida Stalker instrumentation is active on this thread.
    static FRIDA_ACTIVE: Cell<bool> = const { Cell::new(false) };

    /// Deferred yield flag. Set by the Stalker callout when the software RBC
    /// counter reaches zero. Checked and cleared by `maybe_yield_preemptive()`
    /// at kfunc boundaries.
    static YIELD_PENDING: Cell<bool> = const { Cell::new(false) };

    /// Address of the Jcc instruction that triggered the deferred yield.
    /// Captured in the callout and returned by `take_yield_pending()`.
    static YIELD_RIP: Cell<u64> = const { Cell::new(0) };

    /// Base address of the scheduler `.so`'s executable segment.
    ///
    /// Set by [`arm_software_rbc`] so that [`rip_to_offset`] can convert
    /// absolute RIPs to `.so`-relative offsets for ASLR-resilient
    /// determinism comparisons.
    static TEXT_BASE: Cell<u64> = const { Cell::new(0) };

    // -- Structop tracking --------------------------------------------------

    /// Per-CPU structop call counter (how many structops this thread has run).
    static STRUCTOP_CPU_COUNT: Cell<u64> = const { Cell::new(0) };

    /// Cumulative RBC count across timeslices within the current structop.
    static STRUCTOP_RBC_TOTAL: Cell<u64> = const { Cell::new(0) };

    /// Kfunc call count within the current structop.
    static STRUCTOP_KFUNC_COUNT: Cell<u64> = const { Cell::new(0) };

    /// The timeslice used for the current (or most recent) arming.
    /// Needed to compute consumed RBC when a timeslice expires.
    static CURRENT_TIMESLICE: Cell<u64> = const { Cell::new(0) };

    /// Whether we are currently inside a structop (ops_context != None).
    /// Used to detect structop boundaries at kfunc entry points.
    static IN_STRUCTOP: Cell<bool> = const { Cell::new(false) };
}

/// Global counter for total callouts executed (for diagnostics).
static TOTAL_CALLOUTS: AtomicU64 = AtomicU64::new(0);

/// Global counter for deferred yields consumed at kfunc boundaries.
static DEFERRED_YIELDS: AtomicU64 = AtomicU64::new(0);

/// Global structop call counter (monotonically increasing across all CPUs).
static STRUCTOP_GLOBAL_COUNT: AtomicU64 = AtomicU64::new(0);

// ---------------------------------------------------------------------------
// TextRange — address range of the scheduler .so's executable segment
// ---------------------------------------------------------------------------

/// Address range of the `.so`'s executable (`r-xp`) segment.
///
/// Used to restrict Stalker instrumentation to only the scheduler's code,
/// avoiding overhead from instrumenting libc, libpthread, or Rust runtime.
pub struct TextRange {
    /// Base address of the executable mapping.
    pub base: usize,
    /// Size in bytes of the executable mapping.
    pub size: usize,
}

impl TextRange {
    /// Check whether an address falls within this range.
    #[inline]
    pub fn contains(&self, addr: usize) -> bool {
        addr >= self.base && addr < self.base + self.size
    }
}

// ---------------------------------------------------------------------------
// discover_so_text_range — parse /proc/self/maps
// ---------------------------------------------------------------------------

/// Discover the executable (`r-xp`) segment of a loaded `.so` file.
///
/// Reads `/proc/self/maps` and finds the first line that contains `so_path`
/// with `r-xp` permissions, then parses the address range.
///
/// Returns `None` if the `.so` is not found or has no executable segment.
///
/// # Example maps line
/// ```text
/// 7f1234000000-7f1234010000 r-xp 00001000 08:01 12345  /path/to/libscx_simple.so
/// ```
pub fn discover_so_text_range(so_path: &str) -> Option<TextRange> {
    let maps = std::fs::read_to_string("/proc/self/maps").ok()?;

    for line in maps.lines() {
        if !line.contains(so_path) {
            continue;
        }
        if !line.contains("r-xp") {
            continue;
        }

        let addr_range = line.split_whitespace().next()?;
        let mut parts = addr_range.split('-');
        let start_str = parts.next()?;
        let end_str = parts.next()?;

        let start = usize::from_str_radix(start_str, 16).ok()?;
        let end = usize::from_str_radix(end_str, 16).ok()?;

        if end > start {
            return Some(TextRange {
                base: start,
                size: end - start,
            });
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Conditional branch detection (x86-64)
// ---------------------------------------------------------------------------

/// Check if an x86/x86-64 mnemonic is a conditional branch.
///
/// Matches the Jcc family (all conditional jumps) and the LOOPcc family.
/// Used in tests; the runtime `build_transformer` uses opcode-based detection.
#[cfg(test)]
fn is_conditional_branch(mnemonic: &str) -> bool {
    matches!(
        mnemonic,
        "jo" | "jno"
            | "jb"
            | "jnae"
            | "jc"
            | "jae"
            | "jnb"
            | "jnc"
            | "je"
            | "jz"
            | "jne"
            | "jnz"
            | "jbe"
            | "jna"
            | "ja"
            | "jnbe"
            | "js"
            | "jns"
            | "jp"
            | "jpe"
            | "jnp"
            | "jpo"
            | "jl"
            | "jnge"
            | "jge"
            | "jnl"
            | "jle"
            | "jng"
            | "jg"
            | "jnle"
            | "loop"
            | "loope"
            | "loopz"
            | "loopne"
            | "loopnz"
    )
}

/// Check if x86/x86-64 instruction bytes represent a conditional branch.
///
/// x86-64 conditional branch opcodes:
/// - `0x70..=0x7F`: Short Jcc (2-byte, e.g., `je rel8`)
/// - `0x0F 0x80..=0x0F 0x8F`: Near Jcc (6-byte, e.g., `je rel32`)
/// - `0xE0..=0xE3`: LOOPcc / JCXZ
fn is_conditional_branch_opcode(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return false;
    }

    // Skip legacy and REX prefixes.
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            0x26 | 0x2E | 0x36 | 0x3E | 0x64 | 0x65 | 0x66 | 0x67 | 0xF0 | 0xF2 | 0xF3 => {
                i += 1;
            }
            0x40..=0x4F => {
                i += 1;
            }
            _ => break,
        }
    }

    if i >= bytes.len() {
        return false;
    }

    match bytes[i] {
        0x70..=0x7F => true,
        0xE0..=0xE3 => true,
        0x0F => i + 1 < bytes.len() && matches!(bytes[i + 1], 0x80..=0x8F),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// build_transformer -- create a Stalker Transformer for software RBC
// ---------------------------------------------------------------------------

/// Build a Stalker [`Transformer`] that instruments conditional branches
/// within the scheduler's `.so` text segment.
///
/// The callout is intentionally minimal: it decrements a thread-local counter
/// and sets a `YIELD_PENDING` flag when the counter reaches zero. No blocking,
/// no allocation, no I/O. The actual yield is deferred to the next kfunc
/// boundary where [`maybe_yield_preemptive`](crate::preempt::maybe_yield_preemptive)
/// runs safely.
pub fn build_transformer<'a>(gum: &'a Gum, range: &TextRange) -> Transformer<'a> {
    let range_base = range.base;
    let range_size = range.size;

    Transformer::from_callback(gum, move |basic_block, _output| {
        for instr in basic_block {
            let addr = instr.instr().address() as usize;

            if addr >= range_base && addr < range_base + range_size {
                let bytes = instr.instr().bytes();
                if is_conditional_branch_opcode(bytes) {
                    let orig_addr = addr as u64;
                    instr.put_callout(move |_cpu_context| {
                        software_rbc_callout_inner(orig_addr);
                    });
                }
            }

            instr.keep();
        }
    })
}

// ---------------------------------------------------------------------------
// Software RBC callout -- decrements counter and sets yield flag
// ---------------------------------------------------------------------------

/// Inner logic for the software RBC callout.
///
/// This runs inside Stalker-translated code, so it must be minimal:
/// only thread-local reads/writes and an atomic increment. No blocking,
/// no heap allocation, no stdio, no tracing.
///
/// When the counter decrements to zero (1→0 transition), sets `YIELD_PENDING`
/// and records the instruction address in `YIELD_RIP` so the next
/// kfunc-boundary yield point will perform the actual yield.
#[inline]
fn software_rbc_callout_inner(orig_addr: u64) {
    TOTAL_CALLOUTS.fetch_add(1, Relaxed);

    SOFTWARE_RBC_COUNTER.with(|counter| {
        let current = counter.get();
        if current == u64::MAX || current == 0 {
            // Disarmed or already signaled — no-op.
            return;
        }
        let new = current - 1;
        counter.set(new);
        if new == 0 {
            // Counter just hit zero — signal deferred yield.
            YIELD_PENDING.with(|flag| flag.set(true));
            YIELD_RIP.with(|rip| rip.set(orig_addr));
        }
    });
}

// ---------------------------------------------------------------------------
// Deferred yield check (called from maybe_yield_preemptive)
// ---------------------------------------------------------------------------

/// Check and clear the deferred yield flag.
///
/// Called by [`maybe_yield_preemptive`](crate::preempt::maybe_yield_preemptive)
/// at kfunc boundaries. Returns `Some(rip)` if the Stalker callout set the
/// `YIELD_PENDING` flag (i.e., the software RBC counter expired), where `rip`
/// is the original instruction address of the Jcc that triggered the yield.
///
/// When this returns `Some`, the caller should perform a preemptive yield
/// (same as a PMU signal-triggered preemption) and then re-arm the counter
/// via [`rearm_software_rbc`].
pub fn take_yield_pending() -> Option<u64> {
    YIELD_PENDING.with(|flag| {
        if flag.get() {
            flag.set(false);
            DEFERRED_YIELDS.fetch_add(1, Relaxed);
            let rip = YIELD_RIP.with(|r| r.get());
            Some(rip)
        } else {
            None
        }
    })
}

/// Re-arm the software RBC counter with a new timeslice after a deferred yield.
///
/// Called by `maybe_yield_preemptive` after completing the yield to set the
/// counter for the next preemption interval. Also clears YIELD_PENDING in case
/// it was set redundantly.
pub fn rearm_software_rbc(timeslice: u64) {
    SOFTWARE_RBC_COUNTER.with(|counter| {
        counter.set(timeslice);
    });
    YIELD_PENDING.with(|flag| {
        flag.set(false);
    });
    CURRENT_TIMESLICE.with(|ts| {
        ts.set(timeslice);
    });
}

// ---------------------------------------------------------------------------
// Arm / disarm / query
// ---------------------------------------------------------------------------

/// Arm the software RBC counter with the given timeslice and `.so` text base.
///
/// The `text_base` is the base address of the scheduler `.so`'s executable
/// segment. It is used by [`rip_to_offset`] to convert absolute RIPs to
/// `.so`-relative offsets for ASLR-resilient determinism comparisons.
pub fn arm_software_rbc(timeslice: u64, text_base: u64) {
    SOFTWARE_RBC_COUNTER.with(|counter| {
        counter.set(timeslice);
    });
    FRIDA_ACTIVE.with(|active| {
        active.set(true);
    });
    TEXT_BASE.with(|base| {
        base.set(text_base);
    });
    CURRENT_TIMESLICE.with(|ts| {
        ts.set(timeslice);
    });
}

/// Disarm the software RBC counter.
pub fn disarm_software_rbc() {
    SOFTWARE_RBC_COUNTER.with(|counter| {
        counter.set(u64::MAX);
    });
    FRIDA_ACTIVE.with(|active| {
        active.set(false);
    });
    YIELD_PENDING.with(|flag| {
        flag.set(false);
    });
    TEXT_BASE.with(|base| {
        base.set(0);
    });
    CURRENT_TIMESLICE.with(|ts| {
        ts.set(0);
    });
}

/// Check whether Frida Stalker instrumentation is active on the current thread.
pub fn is_frida_active() -> bool {
    FRIDA_ACTIVE.with(|active| active.get())
}

/// Convert an absolute RIP to a `.so`-relative offset.
///
/// Subtracts the `.so` text segment base address (set by [`arm_software_rbc`])
/// from the absolute address. Returns 0 if the text base is not set (disarmed)
/// or if `rip` is below the base (shouldn't happen for instrumented code).
///
/// This makes RIP values ASLR-resilient: the `.so` loads at different virtual
/// addresses in each run, but the offset within the `.so` is constant.
pub fn rip_to_offset(rip: u64) -> u64 {
    let base = TEXT_BASE.with(|b| b.get());
    if base == 0 || rip < base {
        return rip; // Fallback: return raw address if base unknown
    }
    rip - base
}

/// Return the `.so` text segment base address for the current thread.
///
/// Returns 0 if Frida is not armed on this thread.
pub fn text_base() -> u64 {
    TEXT_BASE.with(|b| b.get())
}

/// Return the total number of callouts executed (diagnostic counter).
pub fn total_callouts() -> u64 {
    TOTAL_CALLOUTS.load(Relaxed)
}

/// Return the total number of deferred yields consumed.
pub fn deferred_yields() -> u64 {
    DEFERRED_YIELDS.load(Relaxed)
}

/// Reset all diagnostic counters.
pub fn reset_counters() {
    TOTAL_CALLOUTS.store(0, Relaxed);
    DEFERRED_YIELDS.store(0, Relaxed);
    STRUCTOP_GLOBAL_COUNT.store(0, Relaxed);
}

// ---------------------------------------------------------------------------
// Structop tracking — per-structop RBC and kfunc counters
// ---------------------------------------------------------------------------

/// Snapshot of structop tracking state for trace output.
pub struct StructopInfo {
    /// Per-CPU structop call number (1-based).
    pub cpu_count: u64,
    /// Global structop call number (1-based).
    pub global_count: u64,
    /// Cumulative RBC count within this structop.
    pub rbc_total: u64,
    /// Kfunc call count within this structop.
    pub kfunc_count: u64,
}

/// Begin a new structop on the current thread.
///
/// Increments the per-CPU and global structop counters and resets
/// per-structop accumulators (RBC total, kfunc count). Call this
/// before entering scheduler C code for an ops callback.
pub fn begin_structop() {
    STRUCTOP_CPU_COUNT.with(|c| c.set(c.get() + 1));
    STRUCTOP_GLOBAL_COUNT.fetch_add(1, Relaxed);
    STRUCTOP_RBC_TOTAL.with(|c| c.set(0));
    STRUCTOP_KFUNC_COUNT.with(|c| c.set(0));
    IN_STRUCTOP.with(|c| c.set(true));
}

/// Detect a structop boundary and begin a new structop if needed.
///
/// Called at kfunc entry points (via `maybe_yield_preemptive`). If
/// `in_ops` is true (ops_context != None) and we were not previously
/// inside a structop, this is a new structop boundary. The caller
/// passes `in_ops` based on the current `SimulatorState::ops_context`.
pub fn maybe_begin_structop(in_ops: bool) {
    if in_ops {
        let was_in = IN_STRUCTOP.with(|c| c.get());
        if !was_in {
            begin_structop();
        }
    } else {
        IN_STRUCTOP.with(|c| c.set(false));
    }
}

/// Record that the RBC timeslice expired within the current structop.
///
/// Adds the timeslice length to the cumulative RBC total, since the
/// counter reaching zero means exactly `timeslice` branches were retired.
pub fn record_rbc_expiry() {
    let ts = CURRENT_TIMESLICE.with(|c| c.get());
    STRUCTOP_RBC_TOTAL.with(|c| c.set(c.get() + ts));
}

/// Increment the kfunc counter for the current structop.
pub fn inc_structop_kfunc() {
    STRUCTOP_KFUNC_COUNT.with(|c| c.set(c.get() + 1));
}

/// Get the current structop tracking state for trace output.
pub fn structop_info() -> StructopInfo {
    StructopInfo {
        cpu_count: STRUCTOP_CPU_COUNT.with(|c| c.get()),
        global_count: STRUCTOP_GLOBAL_COUNT.load(Relaxed),
        rbc_total: STRUCTOP_RBC_TOTAL.with(|c| c.get()),
        kfunc_count: STRUCTOP_KFUNC_COUNT.with(|c| c.get()),
    }
}

/// Reset per-CPU structop counter (called when a worker finishes).
pub fn reset_structop_cpu_count() {
    STRUCTOP_CPU_COUNT.with(|c| c.set(0));
}

// ---------------------------------------------------------------------------
// SyncTransformer — thread-safe wrapper for Transformer
// ---------------------------------------------------------------------------

/// Wrapper around [`Transformer`] that implements `Sync`.
///
/// Frida's `Transformer` contains a raw `*mut GumStalkerTransformer`
/// which prevents auto-`Sync`. However, the transformer is a read-only
/// GObject reference that Stalker accesses immutably during `follow_me`.
/// Sharing the transformer across scoped worker threads is safe because
/// each worker only reads it.
pub struct SyncTransformer<'a>(pub Transformer<'a>);

// SAFETY: The GumStalkerTransformer GObject is ref-counted and the
// Transformer is only used via immutable `&self` in `follow_me`.
unsafe impl Sync for SyncTransformer<'_> {}
unsafe impl Send for SyncTransformer<'_> {}

impl<'a> SyncTransformer<'a> {
    /// Build a sync-safe transformer from a Gum instance and text range.
    pub fn new(gum: &'a Gum, range: &TextRange) -> Self {
        Self(build_transformer(gum, range))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_conditional_branch_jcc() {
        let jcc = [
            "jo", "jno", "jb", "jnae", "jc", "jae", "jnb", "jnc", "je", "jz", "jne", "jnz", "jbe",
            "jna", "ja", "jnbe", "js", "jns", "jp", "jpe", "jnp", "jpo", "jl", "jnge", "jge",
            "jnl", "jle", "jng", "jg", "jnle",
        ];
        for m in &jcc {
            assert!(
                is_conditional_branch(m),
                "{m} should be a conditional branch"
            );
        }
    }

    #[test]
    fn test_is_conditional_branch_loop() {
        let loops = ["loop", "loope", "loopz", "loopne", "loopnz"];
        for m in &loops {
            assert!(
                is_conditional_branch(m),
                "{m} should be a conditional branch"
            );
        }
    }

    #[test]
    fn test_is_not_conditional_branch() {
        let non_branch = [
            "jmp", "call", "ret", "nop", "mov", "add", "sub", "push", "pop", "lea", "cmp", "test",
            "xor", "and", "or", "shl", "shr",
        ];
        for m in &non_branch {
            assert!(
                !is_conditional_branch(m),
                "{m} should NOT be a conditional branch"
            );
        }
    }

    #[test]
    fn test_conditional_branch_opcode_short_jcc() {
        for opcode in 0x70u8..=0x7F {
            assert!(
                is_conditional_branch_opcode(&[opcode, 0x10]),
                "0x{opcode:02x} should be Jcc"
            );
        }
    }

    #[test]
    fn test_conditional_branch_opcode_near_jcc() {
        for second in 0x80u8..=0x8F {
            assert!(
                is_conditional_branch_opcode(&[0x0F, second, 0, 0, 0, 0]),
                "0x0F 0x{second:02x} should be Jcc"
            );
        }
    }

    #[test]
    fn test_conditional_branch_opcode_loop() {
        for opcode in 0xE0u8..=0xE3 {
            assert!(
                is_conditional_branch_opcode(&[opcode, 0x10]),
                "0x{opcode:02x} should be LOOPcc/JCXZ"
            );
        }
    }

    #[test]
    fn test_conditional_branch_opcode_with_rex_prefix() {
        assert!(is_conditional_branch_opcode(&[0x48, 0x74, 0x10]));
    }

    #[test]
    fn test_not_conditional_branch_opcode() {
        assert!(!is_conditional_branch_opcode(&[0xEB, 0x10])); // JMP rel8
        assert!(!is_conditional_branch_opcode(&[0xE8, 0, 0, 0, 0])); // CALL
        assert!(!is_conditional_branch_opcode(&[0xC3])); // RET
        assert!(!is_conditional_branch_opcode(&[0x90])); // NOP
        assert!(!is_conditional_branch_opcode(&[0x48, 0x89, 0xD8])); // MOV
        assert!(!is_conditional_branch_opcode(&[0x0F, 0x1F, 0x00])); // NOP
        assert!(!is_conditional_branch_opcode(&[])); // Empty
    }

    #[test]
    fn test_text_range_contains() {
        let range = TextRange {
            base: 0x1000,
            size: 0x500,
        };
        assert!(range.contains(0x1000));
        assert!(range.contains(0x14FF));
        assert!(!range.contains(0x0FFF));
        assert!(!range.contains(0x1500));
    }

    #[test]
    fn test_arm_disarm_software_rbc() {
        arm_software_rbc(42, 0x7f0000000000);
        assert!(is_frida_active());
        assert_eq!(SOFTWARE_RBC_COUNTER.with(|c| c.get()), 42);
        assert_eq!(text_base(), 0x7f0000000000);

        disarm_software_rbc();
        assert!(!is_frida_active());
        assert_eq!(SOFTWARE_RBC_COUNTER.with(|c| c.get()), u64::MAX);
        assert!(!YIELD_PENDING.with(|f| f.get()));
        assert_eq!(text_base(), 0);
    }

    #[test]
    fn test_callout_sets_yield_pending() {
        SOFTWARE_RBC_COUNTER.with(|c| c.set(2));

        // 2 → 1: no flag yet.
        software_rbc_callout_inner(0xdead);
        assert_eq!(SOFTWARE_RBC_COUNTER.with(|c| c.get()), 1);
        assert!(!YIELD_PENDING.with(|f| f.get()));

        // 1 → 0: YIELD_PENDING set.
        software_rbc_callout_inner(0xbeef);
        assert_eq!(SOFTWARE_RBC_COUNTER.with(|c| c.get()), 0);
        assert!(YIELD_PENDING.with(|f| f.get()));
        assert_eq!(YIELD_RIP.with(|r| r.get()), 0xbeef);

        // Counter == 0: already signaled, no-op.
        software_rbc_callout_inner(0xcafe);
        assert_eq!(SOFTWARE_RBC_COUNTER.with(|c| c.get()), 0);
        assert!(YIELD_PENDING.with(|f| f.get()));

        // take_yield_pending clears the flag and returns the RIP.
        assert_eq!(take_yield_pending(), Some(0xbeef));
        assert!(!YIELD_PENDING.with(|f| f.get()));

        // Second call returns None.
        assert_eq!(take_yield_pending(), None);

        disarm_software_rbc();
    }

    #[test]
    fn test_callout_disarmed_noop() {
        disarm_software_rbc();
        software_rbc_callout_inner(0x1234);
        assert_eq!(SOFTWARE_RBC_COUNTER.with(|c| c.get()), u64::MAX);
        assert!(!YIELD_PENDING.with(|f| f.get()));
    }

    #[test]
    fn test_rip_to_offset() {
        // With text base set, returns offset
        arm_software_rbc(100, 0x7f0000001000);
        assert_eq!(rip_to_offset(0x7f0000001500), 0x500);
        assert_eq!(rip_to_offset(0x7f0000001000), 0);

        // With base=0 (disarmed), returns raw address
        disarm_software_rbc();
        assert_eq!(rip_to_offset(0x7f0000001500), 0x7f0000001500);
    }
}
