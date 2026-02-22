//! Frida Stalker-based software RBC for deterministic preemption.
//!
//! This module provides an alternative to PMU-based preemption (see [`preempt`])
//! that uses [Frida Stalker](https://frida.re/docs/stalker/) for dynamic binary
//! instrumentation. Instead of relying on hardware performance counters to count
//! retired conditional branches, Stalker instruments the scheduler's `.so` at
//! runtime, inserting callouts before every conditional branch instruction.
//!
//! A thread-local software counter (`SOFTWARE_RBC_COUNTER`) is decremented at
//! each conditional branch. When it reaches zero, the thread yields its token
//! via the same [`PreemptRing`] mechanism used by the PMU signal handler,
//! producing identical deterministic interleaving behavior.
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
//! - This module uses Stalker instrumentation callouts to trigger preemption.
//! - Both use [`PreemptRing`] for token passing and futex-based parking.
//! - Both save/restore [`SimulatorState`] context across yields.
//!
//! [`preempt`]: crate::preempt
//! [`PreemptRing`]: crate::preempt::PreemptRing
//! [`SimulatorState`]: crate::kfuncs::SimulatorState

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use frida_gum::stalker::Transformer;
use frida_gum::Gum;

use crate::preempt::PREEMPT_CTX;

// ---------------------------------------------------------------------------
// Thread-local software RBC state
// ---------------------------------------------------------------------------

thread_local! {
    /// Software retired-branch-conditional counter.
    ///
    /// Decremented at each instrumented conditional branch. When it reaches
    /// zero, the thread yields its token. Initialized to `u64::MAX` (disarmed).
    static SOFTWARE_RBC_COUNTER: Cell<u64> = const { Cell::new(u64::MAX) };

    /// Whether Frida Stalker instrumentation is active on this thread.
    static FRIDA_ACTIVE: Cell<bool> = const { Cell::new(false) };
}

/// Global counter for total callouts executed (for diagnostics).
static TOTAL_CALLOUTS: AtomicU64 = AtomicU64::new(0);

/// Global counter for how many times do_software_yield() was entered.
static YIELD_ENTRIES: AtomicU64 = AtomicU64::new(0);

/// Global counter for how many times do_software_yield() found PREEMPT_CTX=None.
static YIELD_NO_CTX: AtomicU64 = AtomicU64::new(0);

/// Global counter for how many times do_software_yield() found sim_state_ptr=None.
static YIELD_NO_SIM: AtomicU64 = AtomicU64::new(0);

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
        // The line must reference our .so and have executable permissions.
        if !line.contains(so_path) {
            continue;
        }
        if !line.contains("r-xp") {
            continue;
        }

        // Parse "start-end r-xp ..."
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
/// These are the instructions that correspond to "retired conditional branches"
/// in PMU terminology.
/// Used in tests; the runtime `build_transformer` uses opcode-based detection.
#[cfg(test)]
fn is_conditional_branch(mnemonic: &str) -> bool {
    matches!(
        mnemonic,
        // Jcc family -- all conditional jumps
        "jo" | "jno"
        | "jb" | "jnae" | "jc"
        | "jae" | "jnb" | "jnc"
        | "je" | "jz"
        | "jne" | "jnz"
        | "jbe" | "jna"
        | "ja" | "jnbe"
        | "js" | "jns"
        | "jp" | "jpe"
        | "jnp" | "jpo"
        | "jl" | "jnge"
        | "jge" | "jnl"
        | "jle" | "jng"
        | "jg" | "jnle"
        // LOOPcc family
        | "loop" | "loope" | "loopz" | "loopne" | "loopnz"
    )
}

/// Check if x86/x86-64 instruction bytes represent a conditional branch.
///
/// Uses opcode-based detection rather than mnemonic string matching, which
/// avoids the need to access the private `cs_insn.mnemonic` field through
/// frida-gum's `Insn` wrapper.
///
/// x86-64 conditional branch opcodes:
/// - `0x70..=0x7F`: Short Jcc (2-byte instructions, e.g., `je rel8`)
/// - `0x0F 0x80..=0x0F 0x8F`: Near Jcc (6-byte instructions, e.g., `je rel32`)
/// - `0xE0`: LOOPNE/LOOPNZ
/// - `0xE1`: LOOPE/LOOPZ
/// - `0xE2`: LOOP
/// - `0xE3`: JCXZ/JECXZ/JRCXZ
fn is_conditional_branch_opcode(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return false;
    }

    // Skip any legacy prefixes (REX, segment overrides, etc.) that may
    // precede the opcode.
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            // Legacy prefixes (segment, operand-size, address-size, lock, rep)
            0x26 | 0x2E | 0x36 | 0x3E | 0x64 | 0x65 | 0x66 | 0x67 | 0xF0 | 0xF2 | 0xF3 => {
                i += 1;
            }
            // REX prefixes (x86-64 only)
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
        // Short Jcc: 0x70..=0x7F
        0x70..=0x7F => true,
        // LOOPcc and JCXZ: 0xE0..=0xE3
        0xE0..=0xE3 => true,
        // Two-byte opcode escape: 0x0F followed by 0x80..=0x8F (near Jcc)
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
/// For each basic block, the transformer iterates over instructions. If an
/// instruction's address falls within `range` and it is a conditional branch
/// (detected via opcode bytes), a callout is inserted that decrements the
/// software RBC counter and yields when exhausted.
///
/// All original instructions are preserved via `keep()`.
pub fn build_transformer<'a>(gum: &'a Gum, range: &TextRange) -> Transformer<'a> {
    let range_base = range.base;
    let range_size = range.size;

    Transformer::from_callback(gum, move |basic_block, _output| {
        for instr in basic_block {
            let addr = instr.instr().address() as usize;

            // Only instrument instructions within the scheduler .so text.
            if addr >= range_base && addr < range_base + range_size {
                let bytes = instr.instr().bytes();
                if is_conditional_branch_opcode(bytes) {
                    instr.put_callout(|_cpu_context| {
                        software_rbc_callout_inner();
                    });
                }
            }

            // Always keep the original instruction.
            instr.keep();
        }
    })
}

// ---------------------------------------------------------------------------
// Software RBC callout -- decrements counter and yields
// ---------------------------------------------------------------------------

/// Inner logic for the software RBC callout.
///
/// Decremented at each instrumented conditional branch. When the counter
/// reaches zero, triggers a yield via `do_software_yield`.
#[inline]
fn software_rbc_callout_inner() {
    TOTAL_CALLOUTS.fetch_add(1, Relaxed);

    SOFTWARE_RBC_COUNTER.with(|counter| {
        let current = counter.get();
        if current == 0 {
            // Counter exhausted — yield to another worker.
            do_software_yield();
        } else if current != u64::MAX {
            // Counter is armed and nonzero — decrement.
            counter.set(current - 1);
        }
        // If counter == u64::MAX, the counter is disarmed (no-op).
    });
}

/// Perform a software-RBC-triggered yield.
///
/// This replicates the save/restore pattern from the PMU signal handler
/// in [`preempt::preempt_handler`], but without signal context (since we
/// are called from a Stalker callout, not a signal handler).
///
/// Steps:
/// 1. Access the thread-local `PREEMPT_CTX` to get the ring and worker ID.
/// 2. Save `SimulatorState` per-callback context (current_cpu, ops_context,
///    waker_task_raw).
/// 3. Record the preemption for determinism verification.
/// 4. Yield the token via `ring.yield_token()` (blocks via futex_wait).
/// 5. On resume: restore `SimulatorState` context.
/// 6. Roll a new timeslice and set `SOFTWARE_RBC_COUNTER`.
fn do_software_yield() {
    YIELD_ENTRIES.fetch_add(1, Relaxed);

    let ctx = PREEMPT_CTX.with(|c| c.get());
    let ctx = match ctx {
        Some(ctx) => ctx,
        None => {
            YIELD_NO_CTX.fetch_add(1, Relaxed);
            return;
        }
    };

    let ring = unsafe { &*ctx.ring };

    // Get SimulatorState pointer.
    let sim_ptr = match crate::kfuncs::sim_state_ptr() {
        Some(p) => p,
        None => {
            YIELD_NO_SIM.fetch_add(1, Relaxed);
            return;
        }
    };

    // Save per-callback context from SimulatorState.
    let (saved_cpu, saved_ops_ctx, saved_waker) = unsafe {
        (
            (*sim_ptr).current_cpu,
            (*sim_ptr).ops_context,
            (*sim_ptr).waker_task_raw,
        )
    };

    // Record the preemption point for determinism verification.
    ring.record_preemption(
        0, // rbc_count: not available in software mode
        0, // instruction_pointer: not available without ucontext
        saved_cpu,
    );

    // Yield token (futex-based). Blocks until re-selected.
    ring.inc_signal_preempt();
    ring.yield_token(ctx.worker_id);

    // Resumed — restore SimulatorState context.
    unsafe {
        (*sim_ptr).current_cpu = saved_cpu;
        (*sim_ptr).ops_context = saved_ops_ctx;
        (*sim_ptr).waker_task_raw = saved_waker;
    }

    // Roll a new timeslice and re-arm the software counter.
    let timeslice = ring.roll_timeslice(ctx.timeslice_min, ctx.timeslice_max);
    SOFTWARE_RBC_COUNTER.with(|counter| {
        counter.set(timeslice);
    });
}

// ---------------------------------------------------------------------------
// Arm / disarm / query
// ---------------------------------------------------------------------------

/// Arm the software RBC counter with the given timeslice.
///
/// After this call, the next `timeslice` conditional branches will be
/// counted before triggering a yield. Call this before entering scheduler
/// C code on a worker thread.
pub fn arm_software_rbc(timeslice: u64) {
    SOFTWARE_RBC_COUNTER.with(|counter| {
        counter.set(timeslice);
    });
    FRIDA_ACTIVE.with(|active| {
        active.set(true);
    });
}

/// Disarm the software RBC counter.
///
/// Sets the counter to `u64::MAX` (effectively infinite — no preemption)
/// and marks Frida as inactive on this thread.
pub fn disarm_software_rbc() {
    SOFTWARE_RBC_COUNTER.with(|counter| {
        counter.set(u64::MAX);
    });
    FRIDA_ACTIVE.with(|active| {
        active.set(false);
    });
}

/// Check whether Frida Stalker instrumentation is active on the current thread.
///
/// Used by [`preempt::pause_timer`] / [`preempt::resume_timer`] to skip
/// PMU timer operations when software RBC is in use.
pub fn is_frida_active() -> bool {
    FRIDA_ACTIVE.with(|active| active.get())
}

/// Return the total number of callouts executed (diagnostic counter).
pub fn total_callouts() -> u64 {
    TOTAL_CALLOUTS.load(Relaxed)
}

/// Return yield diagnostic counters: (entries, no_ctx, no_sim).
pub fn yield_diagnostics() -> (u64, u64, u64) {
    (
        YIELD_ENTRIES.load(Relaxed),
        YIELD_NO_CTX.load(Relaxed),
        YIELD_NO_SIM.load(Relaxed),
    )
}

/// Reset the global callout counter (for testing).
pub fn reset_callout_counter() {
    TOTAL_CALLOUTS.store(0, Relaxed);
    YIELD_ENTRIES.store(0, Relaxed);
    YIELD_NO_CTX.store(0, Relaxed);
    YIELD_NO_SIM.store(0, Relaxed);
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
// Each worker thread creates its own Stalker instance and merely
// references the shared transformer.
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
            let bytes = [opcode, 0x10];
            assert!(
                is_conditional_branch_opcode(&bytes),
                "opcode 0x{opcode:02x} should be a conditional branch"
            );
        }
    }

    #[test]
    fn test_conditional_branch_opcode_near_jcc() {
        for second in 0x80u8..=0x8F {
            let bytes = [0x0F, second, 0x00, 0x00, 0x00, 0x00];
            assert!(
                is_conditional_branch_opcode(&bytes),
                "opcode 0x0F 0x{second:02x} should be a conditional branch"
            );
        }
    }

    #[test]
    fn test_conditional_branch_opcode_loop() {
        for opcode in 0xE0u8..=0xE3 {
            let bytes = [opcode, 0x10];
            assert!(
                is_conditional_branch_opcode(&bytes),
                "opcode 0x{opcode:02x} should be a conditional branch"
            );
        }
    }

    #[test]
    fn test_conditional_branch_opcode_with_rex_prefix() {
        let bytes = [0x48, 0x74, 0x10]; // REX.W + JE rel8
        assert!(is_conditional_branch_opcode(&bytes));
    }

    #[test]
    fn test_not_conditional_branch_opcode() {
        assert!(!is_conditional_branch_opcode(&[0xEB, 0x10])); // JMP rel8
        assert!(!is_conditional_branch_opcode(&[
            0xE8, 0x00, 0x00, 0x00, 0x00
        ])); // CALL
        assert!(!is_conditional_branch_opcode(&[0xC3])); // RET
        assert!(!is_conditional_branch_opcode(&[0x90])); // NOP
        assert!(!is_conditional_branch_opcode(&[0x48, 0x89, 0xD8])); // MOV rax, rbx
        assert!(!is_conditional_branch_opcode(&[0x0F, 0x1F, 0x00])); // NOP multi-byte
        assert!(!is_conditional_branch_opcode(&[])); // Empty
    }

    #[test]
    fn test_text_range_contains() {
        let range = TextRange {
            base: 0x1000,
            size: 0x500,
        };
        assert!(range.contains(0x1000));
        assert!(range.contains(0x1001));
        assert!(range.contains(0x14FF));
        assert!(!range.contains(0x0FFF));
        assert!(!range.contains(0x1500));
        assert!(!range.contains(0x2000));
    }

    #[test]
    fn test_arm_disarm_software_rbc() {
        assert!(!is_frida_active());
        assert_eq!(SOFTWARE_RBC_COUNTER.with(|c| c.get()), u64::MAX);

        arm_software_rbc(42);
        assert!(is_frida_active());
        assert_eq!(SOFTWARE_RBC_COUNTER.with(|c| c.get()), 42);

        disarm_software_rbc();
        assert!(!is_frida_active());
        assert_eq!(SOFTWARE_RBC_COUNTER.with(|c| c.get()), u64::MAX);
    }

    #[test]
    fn test_callout_inner_disarmed() {
        disarm_software_rbc();
        software_rbc_callout_inner();
        assert_eq!(SOFTWARE_RBC_COUNTER.with(|c| c.get()), u64::MAX);
    }

    #[test]
    fn test_callout_inner_decrement() {
        SOFTWARE_RBC_COUNTER.with(|c| c.set(5));

        software_rbc_callout_inner();
        assert_eq!(SOFTWARE_RBC_COUNTER.with(|c| c.get()), 4);

        software_rbc_callout_inner();
        assert_eq!(SOFTWARE_RBC_COUNTER.with(|c| c.get()), 3);

        software_rbc_callout_inner();
        assert_eq!(SOFTWARE_RBC_COUNTER.with(|c| c.get()), 2);

        software_rbc_callout_inner();
        assert_eq!(SOFTWARE_RBC_COUNTER.with(|c| c.get()), 1);

        software_rbc_callout_inner();
        assert_eq!(SOFTWARE_RBC_COUNTER.with(|c| c.get()), 0);

        // Calling with counter == 0 invokes do_software_yield,
        // but without PREEMPT_CTX it returns immediately.
        software_rbc_callout_inner();
        assert_eq!(SOFTWARE_RBC_COUNTER.with(|c| c.get()), 0);

        disarm_software_rbc();
    }
}
