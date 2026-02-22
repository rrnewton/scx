//! Preemptive interleaving via PMU RBC timer signals.
//!
//! Extends the cooperative kfunc-boundary interleaving (see [`interleave`]) with
//! mid-C-code preemption points. A PMU counter fires `SIGSTKFLT` after a random
//! number of retired conditional branches; the signal handler parks the worker
//! via futex and the [`PreemptRing`] passes execution to another worker.
//!
//! ## Determinism
//!
//! RBC (Retired Branch Conditionals) counts only *retired* (committed) branches,
//! not speculative ones. This makes RBC fully deterministic — the same code path
//! produces the same branch count. Tools like [rr](https://rr-project.org/) and
//! [Hermit](https://github.com/facebookexperimental/hermit) rely on this property
//! for record/replay.
//!
//! Combined with deterministic PRNG-driven timeslice selection, preemptive
//! interleaving is deterministic: same seed → same interleaving → same trace.
//!
//! See `ai_docs/DETERMINISM.md` for the full explanation.
//!
//! ## Signal safety
//!
//! The entire preemption path — signal handler, token passing, park/unpark —
//! uses only async-signal-safe primitives: atomics and raw `futex()` syscalls.
//! No `Mutex`, `Condvar`, or heap allocation in the hot path.
//!
//! ## Relationship to [`interleave`]
//!
//! - [`interleave::TokenRing`] uses `Mutex`/`Condvar` for cooperative yields at
//!   kfunc boundaries.
//! - [`PreemptRing`] uses atomics/futex for both cooperative and preemptive
//!   yields, making it safe to call from signal handlers.
//!
//! When preemptive interleaving is enabled, `PreemptRing` replaces `TokenRing`.
//! The existing [`interleave::maybe_yield`] cooperative yield points continue
//! to work — they call into `PreemptRing` instead of `TokenRing`.
//!
//! [`interleave`]: crate::interleave

use std::cell::Cell;
use std::io::{BufRead, Write};
use std::os::unix::io::RawFd;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering::SeqCst};
use std::sync::Mutex;

use crate::interleave::WorkerId;
use crate::kfuncs::SimulatorState;
use crate::types::CpuId;

// ---------------------------------------------------------------------------
// PreemptionRecord — instrumentation for verifying determinism
// ---------------------------------------------------------------------------

/// Maximum number of preemption records to store.
/// This is a fixed-size ring buffer to avoid allocation in signal handlers.
const MAX_PREEMPTION_RECORDS: usize = 4096;

/// A record of a single preemption point (PMU or cooperative kfunc yield).
///
/// Captures all relevant state at the moment of preemption for verifying
/// that RBC-based preemption is truly deterministic: same branch count,
/// same instruction, every time. Includes structop context for correlating
/// preemptions with scheduler ops callback invocations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreemptionRecord {
    /// The timeslice (retired branch conditionals) for this preemption.
    /// For cooperative (kfunc) yields this is 0.
    pub rbc_count: u64,
    /// The instruction pointer (RIP) at the preemption point.
    /// For cooperative (kfunc) yields this is 0.
    pub instruction_pointer: u64,
    /// The CPU ID of the worker that was preempted.
    pub cpu_id: CpuId,
    /// The worker that was preempted (for replay grouping).
    pub worker_id: WorkerId,
    /// Sequence number (monotonically increasing per PreemptRing).
    pub sequence: u64,
    /// Per-worker structop count (1-based) at the time of preemption.
    pub structop_local: u64,
    /// Global structop count (1-based) at the time of preemption.
    pub structop_global: u64,
    /// Cumulative RBC within the current structop at the time of preemption.
    pub structop_rbc: u64,
}

impl std::fmt::Display for PreemptionRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "seq={} structop={}:{} rbc={} rip=0x{:x} cpu={} worker={}",
            self.sequence,
            self.structop_local,
            self.structop_global,
            self.structop_rbc,
            self.instruction_pointer,
            self.cpu_id.0,
            self.worker_id.0
        )
    }
}

/// Number of AtomicU64 slots per preemption record.
const RECORD_FIELDS: usize = 8;

/// Fixed-size storage for preemption records (signal-safe).
///
/// Uses a fixed array with atomic index to avoid heap allocation in signal
/// handlers. Records beyond MAX_PREEMPTION_RECORDS are dropped.
struct PreemptionRecordStore {
    /// Fixed-size array of records (pre-allocated).
    records: Box<[AtomicU64; MAX_PREEMPTION_RECORDS * RECORD_FIELDS]>,
    /// Number of records stored (atomic for signal safety).
    count: AtomicUsize,
    /// Sequence counter for ordering records.
    sequence: AtomicU64,
}

impl PreemptionRecordStore {
    fn new() -> Self {
        // Initialize all slots to zero using a const array.
        let records: Box<[AtomicU64; MAX_PREEMPTION_RECORDS * RECORD_FIELDS]> = (0
            ..MAX_PREEMPTION_RECORDS * RECORD_FIELDS)
            .map(|_| AtomicU64::new(0))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        PreemptionRecordStore {
            records,
            count: AtomicUsize::new(0),
            sequence: AtomicU64::new(0),
        }
    }

    /// Add a record (signal-safe: uses only atomics).
    ///
    /// Returns the sequence number assigned, or None if the buffer is full.
    fn push(
        &self,
        rbc_count: u64,
        instruction_pointer: u64,
        cpu_id: CpuId,
        worker_id: WorkerId,
        sinfo: StructopInfo,
    ) -> Option<u64> {
        let idx = self.count.fetch_add(1, SeqCst);
        if idx >= MAX_PREEMPTION_RECORDS {
            // Buffer full, revert and drop.
            self.count.fetch_sub(1, SeqCst);
            return None;
        }
        let seq = self.sequence.fetch_add(1, SeqCst);
        let base = idx * RECORD_FIELDS;
        self.records[base].store(rbc_count, SeqCst);
        self.records[base + 1].store(instruction_pointer, SeqCst);
        self.records[base + 2].store(cpu_id.0 as u64, SeqCst);
        self.records[base + 3].store(worker_id.0 as u64, SeqCst);
        self.records[base + 4].store(seq, SeqCst);
        self.records[base + 5].store(sinfo.cpu_count, SeqCst);
        self.records[base + 6].store(sinfo.global_count, SeqCst);
        self.records[base + 7].store(sinfo.rbc_total, SeqCst);
        Some(seq)
    }

    /// Retrieve all records (not signal-safe, call after simulation).
    fn drain(&self) -> Vec<PreemptionRecord> {
        let count = self.count.load(SeqCst).min(MAX_PREEMPTION_RECORDS);
        let mut records = Vec::with_capacity(count);
        for i in 0..count {
            let base = i * RECORD_FIELDS;
            records.push(PreemptionRecord {
                rbc_count: self.records[base].load(SeqCst),
                instruction_pointer: self.records[base + 1].load(SeqCst),
                cpu_id: CpuId(self.records[base + 2].load(SeqCst) as u32),
                worker_id: WorkerId(self.records[base + 3].load(SeqCst) as usize),
                sequence: self.records[base + 4].load(SeqCst),
                structop_local: self.records[base + 5].load(SeqCst),
                structop_global: self.records[base + 6].load(SeqCst),
                structop_rbc: self.records[base + 7].load(SeqCst),
            });
        }
        // Sort by sequence number to ensure deterministic ordering.
        records.sort_by_key(|r| r.sequence);
        records
    }
}

// ---------------------------------------------------------------------------
// DeterminismCheckpoint — aggressive determinism verification
// ---------------------------------------------------------------------------

/// Event types that can trigger a determinism checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CheckpointEvent {
    /// PMU signal-triggered preemption.
    Preemption = 0,
    /// Cooperative yield at kfunc boundary.
    CooperativeYield = 1,
    /// ops.dispatch() callback invoked.
    Dispatch = 2,
    /// ops.enqueue() callback invoked.
    Enqueue = 3,
    /// ops.running() callback invoked.
    Running = 4,
    /// ops.stopping() callback invoked.
    Stopping = 5,
    /// ops.select_cpu() callback invoked.
    SelectCpu = 6,
    /// ops.tick() callback invoked.
    Tick = 7,
}

impl std::fmt::Display for CheckpointEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            CheckpointEvent::Preemption => "preemption",
            CheckpointEvent::CooperativeYield => "coop_yield",
            CheckpointEvent::Dispatch => "dispatch",
            CheckpointEvent::Enqueue => "enqueue",
            CheckpointEvent::Running => "running",
            CheckpointEvent::Stopping => "stopping",
            CheckpointEvent::SelectCpu => "select_cpu",
            CheckpointEvent::Tick => "tick",
        };
        write!(f, "{}", s)
    }
}

/// A determinism checkpoint capturing scheduler state at a key event.
///
/// Used for aggressive determinism verification: compare checkpoint sequences
/// from two runs with the same seed to detect divergence and pinpoint exactly
/// where execution differed (RIP, RBC, or memory state).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeterminismCheckpoint {
    /// Monotonically increasing sequence number.
    pub sequence: u64,
    /// The type of event that triggered this checkpoint.
    pub event: CheckpointEvent,
    /// Instruction pointer (RIP) at the checkpoint, if available.
    pub instruction_pointer: u64,
    /// RBC (retired branch conditional) count, if available.
    pub rbc_count: u64,
    /// FNV-1a hash of scheduler-visible memory state (DSQ contents, etc.).
    pub memory_hash: u64,
    /// CPU ID where the event occurred.
    pub cpu_id: CpuId,
}

impl std::fmt::Display for DeterminismCheckpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "seq={} event={} rip=0x{:016x} rbc={} hash=0x{:016x} cpu={}",
            self.sequence,
            self.event,
            self.instruction_pointer,
            self.rbc_count,
            self.memory_hash,
            self.cpu_id.0
        )
    }
}

/// Result of comparing two checkpoint sequences.
#[derive(Debug, Clone)]
pub struct CheckpointDivergence {
    /// Index of the first diverging checkpoint.
    pub checkpoint_index: usize,
    /// The expected checkpoint (from run 1).
    pub expected: DeterminismCheckpoint,
    /// The actual checkpoint (from run 2).
    pub actual: DeterminismCheckpoint,
    /// Which field(s) diverged.
    pub divergence_type: DivergenceType,
}

impl std::fmt::Display for CheckpointDivergence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Divergence at checkpoint {}: {}\n  expected: {}\n  actual:   {}",
            self.checkpoint_index, self.divergence_type, self.expected, self.actual
        )
    }
}

/// Which field(s) diverged between two checkpoints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DivergenceType {
    /// Instruction pointer (RIP) differs.
    Rip,
    /// RBC count differs.
    Rbc,
    /// Memory hash differs.
    MemoryHash,
    /// Event type differs.
    EventType,
    /// CPU ID differs.
    CpuId,
    /// Multiple fields differ.
    Multiple(Vec<DivergenceType>),
}

impl std::fmt::Display for DivergenceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DivergenceType::Rip => write!(f, "RIP differs"),
            DivergenceType::Rbc => write!(f, "RBC count differs"),
            DivergenceType::MemoryHash => write!(f, "memory hash differs"),
            DivergenceType::EventType => write!(f, "event type differs"),
            DivergenceType::CpuId => write!(f, "CPU ID differs"),
            DivergenceType::Multiple(types) => {
                let strs: Vec<_> = types.iter().map(|t| format!("{}", t)).collect();
                write!(f, "{}", strs.join(", "))
            }
        }
    }
}

/// Compare two checkpoint sequences and return the first divergence, if any.
///
/// Returns `None` if the sequences are identical, `Some(divergence)` otherwise.
pub fn compare_checkpoints(
    expected: &[DeterminismCheckpoint],
    actual: &[DeterminismCheckpoint],
) -> Option<CheckpointDivergence> {
    // First check for length mismatch
    if expected.len() != actual.len() {
        // Find the first index where they differ
        let min_len = expected.len().min(actual.len());
        for i in 0..min_len {
            if let Some(div) = compare_single_checkpoint(i, &expected[i], &actual[i]) {
                return Some(div);
            }
        }
        // If all common elements match, the divergence is at the shorter length
        if expected.len() > actual.len() {
            return Some(CheckpointDivergence {
                checkpoint_index: actual.len(),
                expected: expected[actual.len()],
                actual: DeterminismCheckpoint {
                    sequence: 0,
                    event: CheckpointEvent::Preemption,
                    instruction_pointer: 0,
                    rbc_count: 0,
                    memory_hash: 0,
                    cpu_id: CpuId(0),
                },
                divergence_type: DivergenceType::EventType,
            });
        } else {
            return Some(CheckpointDivergence {
                checkpoint_index: expected.len(),
                expected: DeterminismCheckpoint {
                    sequence: 0,
                    event: CheckpointEvent::Preemption,
                    instruction_pointer: 0,
                    rbc_count: 0,
                    memory_hash: 0,
                    cpu_id: CpuId(0),
                },
                actual: actual[expected.len()],
                divergence_type: DivergenceType::EventType,
            });
        }
    }

    // Compare element by element
    for i in 0..expected.len() {
        if let Some(div) = compare_single_checkpoint(i, &expected[i], &actual[i]) {
            return Some(div);
        }
    }
    None
}

fn compare_single_checkpoint(
    index: usize,
    expected: &DeterminismCheckpoint,
    actual: &DeterminismCheckpoint,
) -> Option<CheckpointDivergence> {
    let mut divergences = Vec::new();

    if expected.event != actual.event {
        divergences.push(DivergenceType::EventType);
    }
    if expected.cpu_id != actual.cpu_id {
        divergences.push(DivergenceType::CpuId);
    }
    if expected.instruction_pointer != actual.instruction_pointer {
        divergences.push(DivergenceType::Rip);
    }
    if expected.rbc_count != actual.rbc_count {
        divergences.push(DivergenceType::Rbc);
    }
    if expected.memory_hash != actual.memory_hash {
        divergences.push(DivergenceType::MemoryHash);
    }

    if divergences.is_empty() {
        None
    } else {
        let divergence_type = if divergences.len() == 1 {
            divergences.pop().unwrap()
        } else {
            DivergenceType::Multiple(divergences)
        };
        Some(CheckpointDivergence {
            checkpoint_index: index,
            expected: *expected,
            actual: *actual,
            divergence_type,
        })
    }
}

// ---------------------------------------------------------------------------
// FNV-1a hash for fast memory hashing
// ---------------------------------------------------------------------------

/// FNV-1a 64-bit hash offset basis.
const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x100000001b3;

/// FNV-1a hash implementation (fast, non-cryptographic).
///
/// This is a simple, fast hash suitable for determinism checking.
/// It has good distribution properties for detecting state changes.
#[inline]
pub fn fnv1a_hash_bytes(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Combine multiple hashes into one (order-dependent).
#[inline]
pub fn fnv1a_combine(h1: u64, h2: u64) -> u64 {
    let mut hash = h1;
    // Hash h2's bytes into the combined hash
    for i in 0..8 {
        let byte = ((h2 >> (i * 8)) & 0xff) as u8;
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Hash a u64 value.
#[inline]
pub fn fnv1a_hash_u64(value: u64) -> u64 {
    fnv1a_hash_bytes(&value.to_le_bytes())
}

// ---------------------------------------------------------------------------
// Global checkpoint collector (for aggressive determinism mode)
// ---------------------------------------------------------------------------

/// Maximum number of checkpoints to store.
const MAX_CHECKPOINTS: usize = 8192;

/// Global state for aggressive determinism mode.
struct CheckpointCollector {
    /// Whether aggressive determinism mode is enabled.
    enabled: bool,
    /// Collected checkpoints.
    checkpoints: Vec<DeterminismCheckpoint>,
    /// Sequence counter.
    sequence: u64,
}

impl CheckpointCollector {
    const fn new() -> Self {
        CheckpointCollector {
            enabled: false,
            checkpoints: Vec::new(),
            sequence: 0,
        }
    }
}

/// Global checkpoint collector.
static CHECKPOINT_COLLECTOR: Mutex<CheckpointCollector> = Mutex::new(CheckpointCollector::new());

/// Global flag for fast path checking (avoid lock on every callback).
static DETERMINISM_MODE_ENABLED: AtomicBool = AtomicBool::new(false);

/// Enable aggressive determinism mode.
///
/// Call this before running a simulation to start collecting checkpoints
/// at scheduling events. Checkpoints include memory hashes for detecting
/// state divergence.
pub fn enable_determinism_mode() {
    let mut guard = CHECKPOINT_COLLECTOR.lock().unwrap();
    guard.enabled = true;
    guard.checkpoints.clear();
    guard.checkpoints.reserve(MAX_CHECKPOINTS);
    guard.sequence = 0;
    DETERMINISM_MODE_ENABLED.store(true, SeqCst);
}

/// Disable aggressive determinism mode and drain collected checkpoints.
///
/// Returns the collected checkpoints sorted by sequence number.
pub fn drain_determinism_checkpoints() -> Vec<DeterminismCheckpoint> {
    let mut guard = CHECKPOINT_COLLECTOR.lock().unwrap();
    guard.enabled = false;
    DETERMINISM_MODE_ENABLED.store(false, SeqCst);
    let mut checkpoints = std::mem::take(&mut guard.checkpoints);
    checkpoints.sort_by_key(|c| c.sequence);
    checkpoints
}

/// Check if aggressive determinism mode is enabled (fast path).
#[inline]
pub fn is_determinism_mode_enabled() -> bool {
    DETERMINISM_MODE_ENABLED.load(SeqCst)
}

/// Record a determinism checkpoint.
///
/// Only records if aggressive determinism mode is enabled.
/// Returns the assigned sequence number, or None if disabled/full.
pub fn record_checkpoint(
    event: CheckpointEvent,
    instruction_pointer: u64,
    rbc_count: u64,
    memory_hash: u64,
    cpu_id: CpuId,
) -> Option<u64> {
    // Fast path: check atomic flag before taking lock
    if !DETERMINISM_MODE_ENABLED.load(SeqCst) {
        return None;
    }

    let mut guard = match CHECKPOINT_COLLECTOR.try_lock() {
        Ok(g) => g,
        Err(_) => return None, // Don't block if lock is held
    };

    if !guard.enabled || guard.checkpoints.len() >= MAX_CHECKPOINTS {
        return None;
    }

    let seq = guard.sequence;
    guard.sequence += 1;

    guard.checkpoints.push(DeterminismCheckpoint {
        sequence: seq,
        event,
        instruction_pointer,
        rbc_count,
        memory_hash,
        cpu_id,
    });

    Some(seq)
}

// ---------------------------------------------------------------------------
// Global preemption record collector (for test instrumentation)
// ---------------------------------------------------------------------------

/// Global collector for preemption records, accessible from tests.
///
/// This provides a way to capture preemption records across all `PreemptRing`
/// instances in a simulation run, without threading records through the engine.
static GLOBAL_PREEMPTION_COLLECTOR: Mutex<Option<Vec<PreemptionRecord>>> = Mutex::new(None);

/// Global monotonic sequence counter for preemption records.
///
/// Each `PreemptRing` has its own local sequence counter that resets per
/// batch. This global counter ensures unique, monotonic ordering across
/// all rings/batches for the record/replay feature.
static GLOBAL_PREEMPTION_SEQ: AtomicU64 = AtomicU64::new(0);

/// Enable global preemption record collection.
///
/// Call this before running a simulation to start collecting preemption records.
/// Records are accumulated until `drain_preemption_records()` is called.
pub fn enable_preemption_collection() {
    let mut guard = GLOBAL_PREEMPTION_COLLECTOR.lock().unwrap();
    *guard = Some(Vec::new());
    GLOBAL_PREEMPTION_SEQ.store(0, SeqCst);
}

/// Disable and drain all collected preemption records.
///
/// Returns the collected records sorted by sequence number, or an empty Vec
/// if collection was not enabled. Also disables further collection.
pub fn drain_preemption_records() -> Vec<PreemptionRecord> {
    let mut guard = GLOBAL_PREEMPTION_COLLECTOR.lock().unwrap();
    let mut records = guard.take().unwrap_or_default();
    records.sort_by_key(|r| r.sequence);
    records
}

/// Add a record to the global collector (signal-safe: uses try_lock).
///
/// Called from the signal handler after recording to the ring's local store.
/// This duplicates records to the global collector for test access.
/// Assigns a globally-unique monotonic sequence number.
fn maybe_collect_global(record: PreemptionRecord) {
    // Use try_lock to avoid blocking in signal handler.
    // If the lock is contended, we simply drop this record from global collection.
    if let Ok(mut guard) = GLOBAL_PREEMPTION_COLLECTOR.try_lock() {
        if let Some(ref mut records) = *guard {
            let global_seq = GLOBAL_PREEMPTION_SEQ.fetch_add(1, SeqCst);
            records.push(PreemptionRecord {
                sequence: global_seq,
                ..record
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Structop tracking — per-structop state for trace output
// ---------------------------------------------------------------------------

/// Snapshot of structop tracking for trace output.
///
/// A "structop" is one invocation of a scheduler ops callback (dispatch,
/// enqueue, select_cpu, etc.). This tracks per-worker and global call
/// counts, plus cumulative RBC within the current structop.
#[derive(Debug, Clone, Copy)]
pub struct StructopInfo {
    /// Per-worker structop call count (1-based).
    pub cpu_count: u64,
    /// Global structop call count across all workers (1-based).
    pub global_count: u64,
    /// Cumulative RBC count within this structop.
    pub rbc_total: u64,
    /// Number of kfunc (cooperative) yields within this structop.
    pub kfunc_count: u64,
}

thread_local! {
    static STRUCTOP_CPU_COUNT: Cell<u64> = const { Cell::new(0) };
    static STRUCTOP_RBC_TOTAL: Cell<u64> = const { Cell::new(0) };
    static STRUCTOP_KFUNC_COUNT: Cell<u64> = const { Cell::new(0) };
    static IN_STRUCTOP: Cell<bool> = const { Cell::new(false) };
}
static STRUCTOP_GLOBAL_COUNT: AtomicU64 = AtomicU64::new(0);

/// Begin a new structop: increment per-worker and global counts, reset
/// per-structop accumulators. Called when entering a new ops callback.
pub fn begin_structop() {
    STRUCTOP_CPU_COUNT.with(|c| c.set(c.get() + 1));
    STRUCTOP_GLOBAL_COUNT.fetch_add(1, SeqCst);
    STRUCTOP_RBC_TOTAL.with(|c| c.set(0));
    STRUCTOP_KFUNC_COUNT.with(|c| c.set(0));
}

/// Read current structop tracking state.
pub fn structop_info() -> StructopInfo {
    StructopInfo {
        cpu_count: STRUCTOP_CPU_COUNT.with(|c| c.get()),
        global_count: STRUCTOP_GLOBAL_COUNT.load(SeqCst),
        rbc_total: STRUCTOP_RBC_TOTAL.with(|c| c.get()),
        kfunc_count: STRUCTOP_KFUNC_COUNT.with(|c| c.get()),
    }
}

/// Record RBC consumed by a preemption within the current structop.
///
/// Called from the signal handler after preemption to accumulate the
/// timeslice into the per-structop RBC total.
pub fn record_rbc_preemption(timeslice: u64) {
    STRUCTOP_RBC_TOTAL.with(|c| c.set(c.get() + timeslice));
}

/// Increment the per-structop kfunc yield counter.
///
/// Called from `maybe_yield_preemptive()` on each cooperative yield.
pub fn inc_structop_kfunc() {
    STRUCTOP_KFUNC_COUNT.with(|c| c.set(c.get() + 1));
}

/// Reset the per-worker structop count (call when a worker finishes).
pub fn reset_structop_cpu_count() {
    STRUCTOP_CPU_COUNT.with(|c| c.set(0));
    STRUCTOP_RBC_TOTAL.with(|c| c.set(0));
    STRUCTOP_KFUNC_COUNT.with(|c| c.set(0));
    IN_STRUCTOP.with(|c| c.set(false));
}

/// Reset global structop count (call between dispatch rounds).
pub fn reset_structop_globals() {
    STRUCTOP_GLOBAL_COUNT.store(0, SeqCst);
}

/// Detect structop boundary transitions and call `begin_structop()` when
/// entering a new ops callback.
///
/// Matches the frida branch pattern: begin if newly in ops, clear flag
/// if not in ops.
pub fn maybe_begin_structop(in_ops: bool) {
    IN_STRUCTOP.with(|c| {
        if in_ops && !c.get() {
            c.set(true);
            begin_structop();
        } else if !in_ops {
            c.set(false);
        }
    });
}

// ---------------------------------------------------------------------------
// .so base address helper for RIP offset computation
// ---------------------------------------------------------------------------

/// Get the base address of the scheduler .so in the current process.
///
/// Parses `/proc/self/maps` looking for the first executable mapping from
/// a `libscx_*.so` file. Returns 0 if not found. The result can be
/// subtracted from an absolute RIP to get a .so-relative offset that
/// survives ASLR.
pub fn scheduler_so_base() -> u64 {
    let maps = match std::fs::read_to_string("/proc/self/maps") {
        Ok(s) => s,
        Err(_) => return 0,
    };
    for line in maps.lines() {
        // Executable mapping: look for 'r-xp' or 'r--xp' permission field
        // and a path containing "libscx_"
        if !line.contains("libscx_") {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            continue;
        }
        // Check permissions field for executable
        if !parts[1].contains('x') {
            continue;
        }
        // Parse start address from "start-end"
        if let Some(start_str) = parts[0].split('-').next() {
            if let Ok(addr) = u64::from_str_radix(start_str, 16) {
                return addr;
            }
        }
    }
    0
}

// ---------------------------------------------------------------------------
// Futex wrappers (async-signal-safe — raw syscalls only)
// ---------------------------------------------------------------------------

/// Atomically check `*futex == expected` and sleep until woken.
///
/// Returns immediately (spurious wakeup) if the value has changed.
fn futex_wait(futex: &AtomicU32, expected: u32) {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            futex as *const AtomicU32,
            libc::FUTEX_WAIT | libc::FUTEX_PRIVATE_FLAG,
            expected,
            std::ptr::null::<libc::timespec>(),
            std::ptr::null::<u32>(),
            0u32,
        );
    }
    // Return value intentionally ignored — spurious wakeups handled by caller.
}

/// Wake up to `count` threads blocked on `futex`.
fn futex_wake(futex: &AtomicU32, count: i32) {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            futex as *const AtomicU32,
            libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
            count,
            std::ptr::null::<libc::timespec>(),
            std::ptr::null::<u32>(),
            0u32,
        );
    }
}

// ---------------------------------------------------------------------------
// PreemptRing — futex-based token ring (signal-safe)
// ---------------------------------------------------------------------------

/// Per-worker state values.
const PARKED: u32 = 0;
const RUNNING: u32 = 1;

/// Signal-safe token ring using atomics and futex.
///
/// All methods are safe to call from signal handlers. The PRNG access is
/// serialized using a spinlock to ensure deterministic ordering.
///
/// NOTE: This intentionally uses a hand-rolled xorshift32 rather than
/// `rand::rngs::SmallRng` because the PRNG state must live in an `AtomicU32`
/// for signal-handler safety — standard library RNGs have multi-word state
/// that cannot be stored atomically.
pub struct PreemptRing {
    /// Per-worker state: `PARKED` or `RUNNING`.
    workers: Box<[AtomicU32]>,
    /// PRNG state (xorshift32). Access is serialized via CAS in `next_prng()`.
    prng: AtomicU32,
    /// Total number of workers.
    total: usize,
    /// Bitmask of finished workers (up to 64).
    finished_mask: AtomicU64,
    /// Orchestrator wake word: 0 = not all done, 1 = all done.
    all_done: AtomicU32,
    /// Count of signal-driven (PMU) preemptions (signal-safe increment).
    signal_preempt_count: AtomicU64,
    /// Count of cooperative yields at kfunc boundaries (safe Rust).
    cooperative_yield_count: AtomicU64,
    /// Storage for preemption records (instrumentation for determinism verification).
    preemption_records: PreemptionRecordStore,
}

impl PreemptRing {
    /// Create a new preemptive ring for `total` workers.
    ///
    /// # Panics
    /// Panics if `total` is 0 or exceeds 64.
    pub fn new(total: usize, seed: u32) -> Self {
        assert!(
            total > 0 && total <= 64,
            "PreemptRing supports 1–64 workers, got {total}"
        );
        let seed = if seed == 0 { 1 } else { seed };
        let workers: Box<[AtomicU32]> = (0..total).map(|_| AtomicU32::new(PARKED)).collect();
        PreemptRing {
            workers,
            prng: AtomicU32::new(seed),
            total,
            finished_mask: AtomicU64::new(0),
            all_done: AtomicU32::new(0),
            signal_preempt_count: AtomicU64::new(0),
            cooperative_yield_count: AtomicU64::new(0),
            preemption_records: PreemptionRecordStore::new(),
        }
    }

    /// Atomically advance the xorshift32 PRNG and return the new value.
    ///
    /// Uses CAS loop to ensure atomic read-modify-write, preventing races
    /// where two threads could load the same state and skip PRNG values.
    fn next_prng(&self) -> u32 {
        loop {
            let old = self.prng.load(SeqCst);
            let mut x = old;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            if self.prng.compare_exchange(old, x, SeqCst, SeqCst).is_ok() {
                return x;
            }
            // CAS failed - another thread updated the PRNG; retry.
        }
    }

    fn pick_next(&self) -> Option<WorkerId> {
        let mask = self.finished_mask.load(SeqCst);
        let n_finished = mask.count_ones() as usize;
        let n_remaining = self.total - n_finished;
        if n_remaining == 0 {
            return None;
        }
        let idx = (self.next_prng() as usize) % n_remaining;
        let mut count = 0;
        for i in 0..self.total {
            if mask & (1u64 << i) == 0 {
                if count == idx {
                    return Some(WorkerId(i));
                }
                count += 1;
            }
        }
        unreachable!()
    }

    /// Roll a random timeslice in `[min, max]` using the ring's PRNG.
    pub fn roll_timeslice(&self, min: u64, max: u64) -> u64 {
        debug_assert!(max >= min);
        let range = max - min;
        if range == 0 {
            return min;
        }
        min + (self.next_prng() as u64) % (range + 1)
    }

    /// Increment the signal-driven preemption counter.
    ///
    /// **Async-signal-safe**: uses only an atomic fetch_add.
    pub fn inc_signal_preempt(&self) {
        self.signal_preempt_count.fetch_add(1, SeqCst);
    }

    /// Increment the cooperative yield counter.
    pub fn inc_cooperative_yield(&self) {
        self.cooperative_yield_count.fetch_add(1, SeqCst);
    }

    /// Total signal-driven (PMU) preemptions since creation.
    pub fn signal_preemptions(&self) -> u64 {
        self.signal_preempt_count.load(SeqCst)
    }

    /// Total cooperative yields (kfunc boundary) since creation.
    pub fn cooperative_yields(&self) -> u64 {
        self.cooperative_yield_count.load(SeqCst)
    }

    /// Record a preemption point (signal-safe).
    ///
    /// Called from the signal handler or cooperative yield path to capture
    /// the preemption state including structop context.
    ///
    /// Returns the sequence number assigned, or None if the buffer is full.
    pub fn record_preemption(
        &self,
        rbc_count: u64,
        instruction_pointer: u64,
        cpu_id: CpuId,
        worker_id: WorkerId,
        sinfo: StructopInfo,
    ) -> Option<u64> {
        let seq =
            self.preemption_records
                .push(rbc_count, instruction_pointer, cpu_id, worker_id, sinfo);
        // Also collect to global store for test instrumentation.
        if let Some(s) = seq {
            maybe_collect_global(PreemptionRecord {
                rbc_count,
                instruction_pointer,
                cpu_id,
                worker_id,
                sequence: s,
                structop_local: sinfo.cpu_count,
                structop_global: sinfo.global_count,
                structop_rbc: sinfo.rbc_total,
            });
        }
        seq
    }

    /// Retrieve all preemption records collected during the simulation.
    ///
    /// Returns records sorted by sequence number. Call this after the
    /// simulation completes to verify determinism.
    pub fn preemption_records(&self) -> Vec<PreemptionRecord> {
        self.preemption_records.drain()
    }

    /// Orchestrator: select the first worker via PRNG and wake it.
    pub fn start(&self) {
        if let Some(first) = self.pick_next() {
            self.workers[first.0].store(RUNNING, SeqCst);
            futex_wake(&self.workers[first.0], 1);
        }
    }

    /// Worker: block until this worker is selected.
    pub fn wait_for_token(&self, my_id: WorkerId) {
        loop {
            if self.workers[my_id.0].load(SeqCst) == RUNNING {
                break;
            }
            futex_wait(&self.workers[my_id.0], PARKED);
        }
    }

    /// Worker: release token, select next worker via PRNG, block until
    /// re-selected.
    ///
    /// **Async-signal-safe**: safe to call from signal handlers.
    ///
    /// Ordering: parks self BEFORE waking next, preventing the race where
    /// the next worker yields back before we enter futex_wait.
    pub fn yield_token(&self, my_id: WorkerId) {
        // Park ourselves first to prevent wake-before-wait races.
        self.workers[my_id.0].store(PARKED, SeqCst);

        if let Some(next) = self.pick_next() {
            self.workers[next.0].store(RUNNING, SeqCst);
            futex_wake(&self.workers[next.0], 1);
        }

        // Wait until re-selected.
        loop {
            if self.workers[my_id.0].load(SeqCst) != PARKED {
                break;
            }
            futex_wait(&self.workers[my_id.0], PARKED);
        }
    }

    /// Worker: mark as finished and wake the next worker (or signal
    /// all-done to the orchestrator).
    pub fn finish(&self, my_id: WorkerId) {
        self.finished_mask.fetch_or(1u64 << my_id.0, SeqCst);
        if self.finished_mask.load(SeqCst).count_ones() as usize == self.total {
            self.all_done.store(1, SeqCst);
            futex_wake(&self.all_done, 1);
        } else if let Some(next) = self.pick_next() {
            self.workers[next.0].store(RUNNING, SeqCst);
            futex_wake(&self.workers[next.0], 1);
        }
    }

    /// Orchestrator: block until all workers have finished.
    pub fn wait_all_done(&self) {
        loop {
            if self.all_done.load(SeqCst) != 0 {
                break;
            }
            futex_wait(&self.all_done, 0);
        }
    }
}

// ---------------------------------------------------------------------------
// Thread-local preemptive interleave context
// ---------------------------------------------------------------------------

/// Thread-local context for a worker participating in preemptive interleaving.
#[derive(Clone, Copy)]
struct PreemptCtx {
    ring: *const PreemptRing,
    worker_id: WorkerId,
    /// Raw fd of the RBC timer (for disable/enable in signal handler).
    timer_fd: RawFd,
    /// Timeslice range for re-arming the timer after preemption.
    timeslice_min: u64,
    timeslice_max: u64,
}

// Raw pointer is Send — access serialized by token passing.
unsafe impl Send for PreemptCtx {}

thread_local! {
    static PREEMPT_CTX: Cell<Option<PreemptCtx>> = const { Cell::new(None) };
}

/// Install preemptive interleave context on the current worker thread.
pub fn install(
    ring: &PreemptRing,
    worker_id: WorkerId,
    timer_fd: RawFd,
    timeslice_min: u64,
    timeslice_max: u64,
) {
    PREEMPT_CTX.with(|c| {
        c.set(Some(PreemptCtx {
            ring: ring as *const PreemptRing,
            worker_id,
            timer_fd,
            timeslice_min,
            timeslice_max,
        }));
    });
}

/// Remove preemptive interleave context from the current thread.
pub fn uninstall() {
    PREEMPT_CTX.with(|c| c.set(None));
}

// ---------------------------------------------------------------------------
// Cooperative yield via PreemptRing (replaces interleave::maybe_yield)
// ---------------------------------------------------------------------------

/// Cooperative yield point for kfunc entry (using the PreemptRing).
///
/// Functionally identical to [`interleave::maybe_yield`] but uses the
/// futex-based `PreemptRing` instead of `Mutex`/`Condvar` `TokenRing`.
///
/// The PMU timer is disabled on entry and stays disabled on return.
/// The caller (via `with_sim()` → `resume_timer()`) is responsible for
/// re-arming the timer before returning to scheduler C code.
///
/// # Safety contract
///
/// Must be called BEFORE `with_sim()`, so no `&mut SimulatorState`
/// reference exists when the worker yields.
pub fn maybe_yield_preemptive() {
    let ctx = PREEMPT_CTX.with(|c| c.get());
    let ctx = match ctx {
        Some(ctx) => ctx,
        None => return,
    };

    let ring = unsafe { &*ctx.ring };

    // Disable the PMU timer during the cooperative yield to prevent
    // a preemptive signal from firing while we're in Rust/yield code.
    disable_timer(ctx.timer_fd);

    // Save per-callback context from SimulatorState.
    let sim_ptr: *mut SimulatorState = crate::kfuncs::sim_state_ptr()
        .expect("maybe_yield_preemptive called outside simulator context");

    let (saved_cpu, saved_ops_ctx, saved_waker) = unsafe {
        (
            (*sim_ptr).current_cpu,
            (*sim_ptr).ops_context,
            (*sim_ptr).waker_task_raw,
        )
    };

    // Detect structop boundary transitions.
    let in_ops = saved_ops_ctx != crate::kfuncs::OpsContext::None;
    maybe_begin_structop(in_ops);

    // Increment per-structop kfunc yield counter.
    inc_structop_kfunc();

    let sinfo = structop_info();

    // Release token and block until re-selected (futex-based).
    ring.inc_cooperative_yield();

    // Record the cooperative yield as a preemption point for record/replay.
    // The RIP is 0 (kfunc boundary marker) and rbc_count is 0 since this
    // is a cooperative yield, not a PMU-triggered preemption.
    ring.record_preemption(0, 0, saved_cpu, ctx.worker_id, sinfo);

    tracing::debug!(
        worker = ctx.worker_id.0,
        cpu = saved_cpu.0,
        "preempt: kfunc yield structop {}:{} kfunc {}",
        sinfo.cpu_count,
        sinfo.global_count,
        sinfo.kfunc_count,
    );
    ring.yield_token(ctx.worker_id);

    // Resumed — restore our context to SimulatorState.
    tracing::debug!(
        worker = ctx.worker_id.0,
        cpu = saved_cpu.0,
        "preempt: resumed after cooperative yield"
    );
    unsafe {
        (*sim_ptr).current_cpu = saved_cpu;
        (*sim_ptr).ops_context = saved_ops_ctx;
        (*sim_ptr).waker_task_raw = saved_waker;
    }

    // Timer stays disabled — with_sim() will re-arm via resume_timer().
}

// ---------------------------------------------------------------------------
// Timer pause/resume for with_sim() bracketing
// ---------------------------------------------------------------------------

/// Pause the preemption timer to prevent signals during kfunc execution.
///
/// Called by `with_sim()` before accessing `SimulatorState`. No-op if
/// preemptive interleaving is not active on this thread.
///
/// **Async-signal-safe**: uses only a thread-local read and an ioctl.
pub fn pause_timer() {
    if let Some(ctx) = PREEMPT_CTX.with(|c| c.get()) {
        disable_timer(ctx.timer_fd);
    }
}

/// Resume the preemption timer after kfunc execution completes.
///
/// Called by `with_sim()` just before returning to scheduler C code.
/// Re-arms the PMU timer with a fresh random timeslice. No-op if
/// preemptive interleaving is not active on this thread.
pub fn resume_timer() {
    if let Some(ctx) = PREEMPT_CTX.with(|c| c.get()) {
        let ring = unsafe { &*ctx.ring };
        rearm_timer(ring, &ctx);
    }
}

// ---------------------------------------------------------------------------
// Signal handler (preemptive yield)
// ---------------------------------------------------------------------------

/// The preemptive signal number. SIGSTKFLT is unused by the kernel.
pub const PREEMPT_SIGNAL: libc::c_int = libc::SIGSTKFLT;

/// Install the process-wide SIGSTKFLT signal handler.
///
/// Must be called before spawning worker threads. The handler is
/// async-signal-safe: it uses only atomics, futex, and raw ioctls.
pub fn install_signal_handler() {
    let sa = libc::sigaction {
        sa_sigaction: preempt_handler as *const () as libc::sighandler_t,
        sa_mask: unsafe { std::mem::zeroed() },
        // SA_SIGINFO so we get siginfo_t; SA_RESTART to not fail slow
        // syscalls (though we don't expect any during C scheduler code).
        sa_flags: libc::SA_SIGINFO | libc::SA_RESTART,
        sa_restorer: None,
    };
    let ret = unsafe { libc::sigaction(PREEMPT_SIGNAL, &sa, std::ptr::null_mut()) };
    assert_eq!(ret, 0, "failed to install SIGSTKFLT handler");
}

/// Remove the SIGSTKFLT signal handler, restoring default behavior.
pub fn uninstall_signal_handler() {
    let sa = libc::sigaction {
        sa_sigaction: libc::SIG_DFL,
        sa_mask: unsafe { std::mem::zeroed() },
        sa_flags: 0,
        sa_restorer: None,
    };
    unsafe {
        libc::sigaction(PREEMPT_SIGNAL, &sa, std::ptr::null_mut());
    }
}

/// Extract the instruction pointer (RIP) from a ucontext on x86-64.
///
/// Returns 0 if the context is null or on non-x86-64 platforms.
#[cfg(target_arch = "x86_64")]
fn extract_rip_from_ucontext(ctx: *mut libc::c_void) -> u64 {
    if ctx.is_null() {
        return 0;
    }
    unsafe {
        let uc = ctx as *const libc::ucontext_t;
        // REG_RIP is index 16 on x86-64 Linux (from sys/ucontext.h).
        (*uc).uc_mcontext.gregs[libc::REG_RIP as usize] as u64
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn extract_rip_from_ucontext(_ctx: *mut libc::c_void) -> u64 {
    0 // Not supported on non-x86-64 platforms.
}

/// Read the current RBC count from the timer fd (signal-safe).
///
/// Returns 0 if the fd is invalid or read fails.
fn read_rbc_count(timer_fd: RawFd) -> u64 {
    if timer_fd < 0 {
        return 0;
    }
    let mut count: u64 = 0;
    let ret = unsafe {
        libc::read(
            timer_fd,
            &mut count as *mut u64 as *mut libc::c_void,
            std::mem::size_of::<u64>(),
        )
    };
    if ret < 0 {
        0
    } else {
        count
    }
}

/// Signal handler for preemptive interleaving.
///
/// Called on SIGSTKFLT delivery (PMU counter overflow). All operations
/// here are async-signal-safe.
extern "C" fn preempt_handler(
    _signo: libc::c_int,
    _info: *mut libc::siginfo_t,
    ctx: *mut libc::c_void,
) {
    let pctx = PREEMPT_CTX.with(|c| c.get());
    let pctx = match pctx {
        Some(ctx) => ctx,
        None => return, // Not in a preemptive interleave context.
    };

    let ring = unsafe { &*pctx.ring };

    // 1. Disable PMU timer to prevent recursive signals.
    disable_timer(pctx.timer_fd);

    // 2. Capture preemption instrumentation BEFORE any other work.
    //    Extract RIP from ucontext and read current RBC count.
    let instruction_pointer = extract_rip_from_ucontext(ctx);
    let rbc_count = read_rbc_count(pctx.timer_fd);

    // 3. Save SimulatorState context to locals (on the signal stack frame).
    let sim_ptr = match crate::kfuncs::sim_state_ptr() {
        Some(p) => p,
        None => return,
    };

    let (saved_cpu, saved_ops_ctx, saved_waker) = unsafe {
        (
            (*sim_ptr).current_cpu,
            (*sim_ptr).ops_context,
            (*sim_ptr).waker_task_raw,
        )
    };

    // 4. Track structop RBC and emit trace message.
    record_rbc_preemption(rbc_count);
    let sinfo = structop_info();
    tracing::trace!(
        "preempt:rbc structop {}:{} rbc {} rip 0x{:x}",
        sinfo.cpu_count,
        sinfo.global_count,
        sinfo.rbc_total,
        instruction_pointer,
    );

    // 4a. Record the preemption point (with structop context).
    ring.record_preemption(
        rbc_count,
        instruction_pointer,
        saved_cpu,
        pctx.worker_id,
        sinfo,
    );

    // 5. Yield token (futex-based, signal-safe). Blocks until re-selected.
    ring.inc_signal_preempt(); // atomic, signal-safe
    ring.yield_token(pctx.worker_id);

    // 6. Resumed — restore SimulatorState context.
    unsafe {
        (*sim_ptr).current_cpu = saved_cpu;
        (*sim_ptr).ops_context = saved_ops_ctx;
        (*sim_ptr).waker_task_raw = saved_waker;
    }

    // 7. Re-arm PMU timer with a fresh timeslice.
    rearm_timer(ring, &pctx);
}

// ---------------------------------------------------------------------------
// Timer helpers (raw ioctls — async-signal-safe)
// ---------------------------------------------------------------------------

/// Disable the PMU timer. No-op if fd is -1 (no timer).
fn disable_timer(fd: RawFd) {
    if fd < 0 {
        return;
    }
    unsafe {
        libc::ioctl(fd, scx_perf::PERF_IOC_DISABLE, 0 as libc::c_ulong);
    }
}

/// Re-arm the PMU timer with a fresh random timeslice.
fn rearm_timer(ring: &PreemptRing, ctx: &PreemptCtx) {
    let fd = ctx.timer_fd;
    if fd < 0 {
        return;
    }
    let timeslice = ring.roll_timeslice(ctx.timeslice_min, ctx.timeslice_max);
    let mut period = timeslice;
    unsafe {
        // Reset counter to zero.
        libc::ioctl(fd, scx_perf::PERF_IOC_RESET, 0 as libc::c_ulong);
        // Set new period.
        libc::ioctl(fd, scx_perf::PERF_IOC_PERIOD, &mut period as *mut u64);
        // Enable.
        libc::ioctl(fd, scx_perf::PERF_IOC_ENABLE, 0 as libc::c_ulong);
    }
}

// ---------------------------------------------------------------------------
// PreemptionTrace — replayable preemption trace grouped by worker
// ---------------------------------------------------------------------------

/// A replayable preemption trace, grouped by worker.
///
/// During recording, preemption points are captured globally. For replay,
/// they're reorganized per-worker so each worker thread knows its own
/// sequence of target preemption points.
#[derive(Debug, Clone)]
pub struct PreemptionTrace {
    /// Per-worker preemption points, ordered by sequence number.
    /// Index = worker_id.0
    per_worker: Vec<Vec<PreemptionRecord>>,
}

impl PreemptionTrace {
    /// Build a trace from globally-collected records.
    ///
    /// Records are grouped by `worker_id` and sorted by sequence number
    /// within each group so that replay follows the original ordering.
    pub fn from_records(records: &[PreemptionRecord], num_workers: usize) -> Self {
        let mut per_worker: Vec<Vec<PreemptionRecord>> =
            (0..num_workers).map(|_| Vec::new()).collect();
        for rec in records {
            let wid = rec.worker_id.0;
            if wid < num_workers {
                per_worker[wid].push(*rec);
            }
        }
        // Sort each worker's records by sequence number.
        for bucket in &mut per_worker {
            bucket.sort_by_key(|r| r.sequence);
        }
        PreemptionTrace { per_worker }
    }

    /// Get the preemption points for a specific worker.
    pub fn worker_trace(&self, worker: WorkerId) -> &[PreemptionRecord] {
        self.per_worker
            .get(worker.0)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Total number of preemption points across all workers.
    pub fn total_preemptions(&self) -> usize {
        self.per_worker.iter().map(|v| v.len()).sum()
    }

    /// Number of workers in this trace.
    pub fn num_workers(&self) -> usize {
        self.per_worker.len()
    }

    /// Serialize the trace to a line-oriented text format.
    ///
    /// Format:
    /// ```text
    /// # scxsim preemption trace
    /// # workers: 2
    /// # total: 47
    /// seq=0 structop=1:2 rbc=10 timeslice=142 rip=0x7f3a rip_offset=0xc7c cpu=0 worker=0
    /// ```
    ///
    /// `rbc` is the cumulative RBC within the structop. `timeslice` is the
    /// raw PMU count for this preemption (used by the replay engine).
    /// `so_base` is used to compute `rip_offset` (ASLR-resilient).
    pub fn serialize(&self, w: &mut impl Write, so_base: u64) -> std::io::Result<()> {
        let total: usize = self.total_preemptions();
        writeln!(w, "# scxsim preemption trace")?;
        writeln!(w, "# workers: {}", self.per_worker.len())?;
        writeln!(w, "# total: {total}")?;

        // Flatten and sort by sequence for canonical output order.
        let mut all: Vec<&PreemptionRecord> =
            self.per_worker.iter().flat_map(|v| v.iter()).collect();
        all.sort_by_key(|r| r.sequence);

        for rec in all {
            let rip_offset = if so_base > 0 && rec.instruction_pointer >= so_base {
                rec.instruction_pointer - so_base
            } else {
                rec.instruction_pointer
            };
            writeln!(
                w,
                "seq={} structop={}:{} rbc={} timeslice={} rip=0x{:x} rip_offset=0x{:x} cpu={} worker={}",
                rec.sequence,
                rec.structop_local,
                rec.structop_global,
                rec.structop_rbc,
                rec.rbc_count,
                rec.instruction_pointer,
                rip_offset,
                rec.cpu_id.0,
                rec.worker_id.0,
            )?;
        }
        Ok(())
    }

    /// Deserialize a trace from the line-oriented text format.
    ///
    /// Parses the header to get the worker count, then each `seq=...` line
    /// into a `PreemptionRecord`. Uses `rip_offset` + `so_base` to
    /// reconstruct absolute RIPs if the current .so base differs from
    /// the recording.
    pub fn deserialize(r: &mut impl BufRead, so_base: u64) -> std::io::Result<Self> {
        let mut num_workers: usize = 0;
        let mut records = Vec::new();

        for line in r.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if line.starts_with('#') {
                // Parse header comments.
                if let Some(rest) = line.strip_prefix("# workers: ") {
                    num_workers = rest
                        .trim()
                        .parse()
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                }
                continue;
            }
            // Parse data line: seq=N rbc=N rip=0xN rip_offset=0xN cpu=N worker=N
            let rec = parse_preemption_line(line, so_base).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("bad preemption line: {e}: {line}"),
                )
            })?;
            records.push(rec);
        }

        if num_workers == 0 && !records.is_empty() {
            // Infer worker count from max worker_id.
            num_workers = records.iter().map(|r| r.worker_id.0).max().unwrap_or(0) + 1;
        }

        Ok(Self::from_records(&records, num_workers))
    }
}

/// Parse a single preemption trace line into a `PreemptionRecord`.
///
/// Uses `rip_offset` to reconstruct the absolute RIP relative to the
/// current `so_base`. Falls back to the stored `rip` if offset is missing.
/// Supports both old format (`rbc=` as timeslice) and new format
/// (`timeslice=` for raw count, `rbc=` for cumulative structop RBC).
fn parse_preemption_line(line: &str, so_base: u64) -> Result<PreemptionRecord, String> {
    let mut seq: u64 = 0;
    let mut rbc: Option<u64> = None;
    let mut timeslice: Option<u64> = None;
    let mut rip: u64 = 0;
    let mut rip_offset: Option<u64> = None;
    let mut cpu: u32 = 0;
    let mut worker: usize = 0;
    let mut structop_local: u64 = 0;
    let mut structop_global: u64 = 0;
    let mut structop_rbc: u64 = 0;

    for part in line.split_whitespace() {
        if let Some(val) = part.strip_prefix("seq=") {
            seq = val.parse().map_err(|e| format!("seq: {e}"))?;
        } else if let Some(val) = part.strip_prefix("timeslice=") {
            timeslice = Some(val.parse().map_err(|e| format!("timeslice: {e}"))?);
        } else if let Some(val) = part.strip_prefix("rbc=") {
            rbc = Some(val.parse().map_err(|e| format!("rbc: {e}"))?);
        } else if let Some(val) = part.strip_prefix("rip=") {
            rip = parse_hex_or_dec(val).map_err(|e| format!("rip: {e}"))?;
        } else if let Some(val) = part.strip_prefix("rip_offset=") {
            rip_offset = Some(parse_hex_or_dec(val).map_err(|e| format!("rip_offset: {e}"))?);
        } else if let Some(val) = part.strip_prefix("cpu=") {
            cpu = val.parse().map_err(|e| format!("cpu: {e}"))?;
        } else if let Some(val) = part.strip_prefix("worker=") {
            worker = val.parse().map_err(|e| format!("worker: {e}"))?;
        } else if let Some(val) = part.strip_prefix("structop=") {
            // Parse "L:G" format.
            if let Some((l, g)) = val.split_once(':') {
                structop_local = l.parse().map_err(|e| format!("structop local: {e}"))?;
                structop_global = g.parse().map_err(|e| format!("structop global: {e}"))?;
            }
        }
    }

    // Reconstruct absolute RIP: prefer rip_offset + so_base for ASLR resilience.
    let instruction_pointer = if let Some(offset) = rip_offset {
        if so_base > 0 {
            so_base + offset
        } else {
            rip // Fallback to stored absolute RIP
        }
    } else {
        rip
    };

    // New format: timeslice= is the raw PMU count, rbc= is cumulative structop RBC.
    // Old format: rbc= is the raw PMU count (no structop fields).
    let rbc_count = timeslice.or(rbc).unwrap_or(0);
    if timeslice.is_some() {
        structop_rbc = rbc.unwrap_or(0);
    }

    Ok(PreemptionRecord {
        sequence: seq,
        rbc_count,
        instruction_pointer,
        cpu_id: CpuId(cpu),
        worker_id: WorkerId(worker),
        structop_local,
        structop_global,
        structop_rbc,
    })
}

/// Parse a string as hex (0x prefix) or decimal.
fn parse_hex_or_dec(s: &str) -> Result<u64, String> {
    if let Some(hex) = s.strip_prefix("0x") {
        u64::from_str_radix(hex, 16).map_err(|e| e.to_string())
    } else if let Some(hex) = s.strip_prefix("0X") {
        u64::from_str_radix(hex, 16).map_err(|e| e.to_string())
    } else {
        s.parse::<u64>().map_err(|e| e.to_string())
    }
}

// ---------------------------------------------------------------------------
// Replay engine — hybrid PMU + hardware breakpoint
// ---------------------------------------------------------------------------

/// Margin (in retired conditional branches) before the target at which the
/// PMU timer fires, giving us time to arm the hardware breakpoint.
///
/// Must be well above typical PMU skid (~30-100 branches). 200 provides
/// comfortable headroom.
pub const REPLAY_MARGIN: u64 = 200;

/// The signal used by hardware breakpoints in replay mode.
pub const REPLAY_BP_SIGNAL: libc::c_int = libc::SIGTRAP;

/// Signal-safe cursor into a worker's replay trace.
///
/// Tracks which preemption target is next and accumulates signal-handler
/// overhead for rdpmc accounting. All mutable state uses atomics for
/// signal-handler safety.
pub struct ReplayCursor {
    /// The preemption targets for this worker.
    targets: Vec<PreemptionRecord>,
    /// Index of the next target (atomic for signal-handler access).
    next_idx: AtomicUsize,
    /// Accumulated signal handler branch overhead (for rdpmc accounting).
    total_overhead: AtomicU64,
}

impl ReplayCursor {
    /// Create a cursor from a worker's trace slice.
    pub fn new(targets: Vec<PreemptionRecord>) -> Self {
        ReplayCursor {
            targets,
            next_idx: AtomicUsize::new(0),
            total_overhead: AtomicU64::new(0),
        }
    }

    /// Get the next target, if any remain.
    pub fn current_target(&self) -> Option<&PreemptionRecord> {
        let idx = self.next_idx.load(SeqCst);
        self.targets.get(idx)
    }

    /// Advance to the next target. Returns true if there are more targets.
    fn advance(&self) -> bool {
        let idx = self.next_idx.fetch_add(1, SeqCst) + 1;
        idx < self.targets.len()
    }

    /// Number of targets in this cursor.
    pub fn len(&self) -> usize {
        self.targets.len()
    }

    /// Whether there are no targets.
    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }

    /// Accumulated overhead branches from signal-handler execution.
    pub fn overhead(&self) -> u64 {
        self.total_overhead.load(SeqCst)
    }
}

/// Thread-local context for a worker in REPLAY mode.
#[derive(Clone, Copy)]
struct ReplayCtx {
    ring: *const PreemptRing,
    worker_id: WorkerId,
    /// Raw fd of the RBC timer.
    timer_fd: RawFd,
    /// Raw fd of the hardware breakpoint.
    bp_fd: RawFd,
    /// Pointer to the per-worker replay cursor.
    /// Raw pointer because it must be accessible from a signal handler.
    cursor: *const ReplayCursor,
}

// Raw pointers are Send — access serialized by token passing.
unsafe impl Send for ReplayCtx {}

thread_local! {
    static REPLAY_CTX: Cell<Option<ReplayCtx>> = const { Cell::new(None) };
}

/// Install replay context on the current worker thread.
pub fn install_replay(
    ring: &PreemptRing,
    worker_id: WorkerId,
    timer_fd: RawFd,
    bp_fd: RawFd,
    cursor: &ReplayCursor,
) {
    REPLAY_CTX.with(|c| {
        c.set(Some(ReplayCtx {
            ring: ring as *const PreemptRing,
            worker_id,
            timer_fd,
            bp_fd,
            cursor: cursor as *const ReplayCursor,
        }));
    });
}

/// Remove replay context from the current thread.
pub fn uninstall_replay() {
    REPLAY_CTX.with(|c| c.set(None));
}

/// Install the process-wide replay signal handlers.
///
/// Installs both the PMU handler (SIGSTKFLT) and the breakpoint handler
/// (SIGTRAP). Must be called before spawning worker threads.
pub fn install_replay_signal_handlers() {
    // PMU handler: fires when we're within MARGIN of the target.
    let sa_pmu = libc::sigaction {
        sa_sigaction: replay_pmu_handler as *const () as libc::sighandler_t,
        sa_mask: unsafe { std::mem::zeroed() },
        sa_flags: libc::SA_SIGINFO | libc::SA_RESTART,
        sa_restorer: None,
    };
    let ret = unsafe { libc::sigaction(PREEMPT_SIGNAL, &sa_pmu, std::ptr::null_mut()) };
    assert_eq!(ret, 0, "failed to install replay PMU handler");

    // Breakpoint handler: fires when we hit the target instruction.
    let sa_bp = libc::sigaction {
        sa_sigaction: replay_bp_handler as *const () as libc::sighandler_t,
        sa_mask: unsafe { std::mem::zeroed() },
        sa_flags: libc::SA_SIGINFO | libc::SA_RESTART,
        sa_restorer: None,
    };
    let ret = unsafe { libc::sigaction(REPLAY_BP_SIGNAL, &sa_bp, std::ptr::null_mut()) };
    assert_eq!(ret, 0, "failed to install replay breakpoint handler");
}

/// Remove the replay signal handlers, restoring default behavior.
pub fn uninstall_replay_signal_handlers() {
    let sa_default = libc::sigaction {
        sa_sigaction: libc::SIG_DFL,
        sa_mask: unsafe { std::mem::zeroed() },
        sa_flags: 0,
        sa_restorer: None,
    };
    unsafe {
        libc::sigaction(PREEMPT_SIGNAL, &sa_default, std::ptr::null_mut());
        libc::sigaction(REPLAY_BP_SIGNAL, &sa_default, std::ptr::null_mut());
    }
}

/// Arm the PMU timer for the next replay target.
///
/// Sets the timer to fire at `target_rbc - REPLAY_MARGIN` branches from
/// the current counter position (which is reset to zero).
fn arm_replay_timer(timer_fd: RawFd, target_rbc: u64) {
    if timer_fd < 0 {
        return;
    }
    let mut period = target_rbc.saturating_sub(REPLAY_MARGIN).max(1);
    unsafe {
        libc::ioctl(timer_fd, scx_perf::PERF_IOC_RESET, 0 as libc::c_ulong);
        libc::ioctl(timer_fd, scx_perf::PERF_IOC_PERIOD, &mut period as *mut u64);
        libc::ioctl(timer_fd, scx_perf::PERF_IOC_ENABLE, 0 as libc::c_ulong);
    }
}

/// Public wrapper for [`arm_replay_timer`], used by the engine to arm the
/// first replay target before entering scheduler C code.
pub fn arm_replay_timer_pub(timer_fd: RawFd, target_rbc: u64) {
    arm_replay_timer(timer_fd, target_rbc);
}

/// Arm the hardware breakpoint at the given instruction pointer.
fn arm_breakpoint(bp_fd: RawFd, addr: u64) {
    if bp_fd < 0 || addr == 0 {
        return;
    }
    // Build a fresh perf_event_attr for the new address.
    let mut attr = perf_event_open_sys::bindings::perf_event_attr {
        type_: perf_event_open_sys::bindings::PERF_TYPE_BREAKPOINT,
        size: std::mem::size_of::<perf_event_open_sys::bindings::perf_event_attr>() as u32,
        bp_type: perf_event_open_sys::bindings::HW_BREAKPOINT_X,
        ..Default::default()
    };
    attr.__bindgen_anon_3.bp_addr = addr;
    attr.__bindgen_anon_4.bp_len = perf_event_open_sys::bindings::HW_BREAKPOINT_LEN_8 as u64;
    attr.__bindgen_anon_1.sample_period = 1;
    attr.__bindgen_anon_2.wakeup_events = 1;
    attr.set_disabled(0); // enable immediately after modify
    attr.set_exclude_kernel(1);
    attr.set_exclude_hv(1);
    attr.set_pinned(1);

    unsafe {
        libc::ioctl(
            bp_fd,
            scx_perf::PERF_IOC_MODIFY_ATTRIBUTES,
            &mut attr as *mut perf_event_open_sys::bindings::perf_event_attr,
        );
        libc::ioctl(bp_fd, scx_perf::PERF_IOC_ENABLE, 0 as libc::c_ulong);
    }
}

/// PMU signal handler for replay mode (SIGSTKFLT).
///
/// Fires when we're within REPLAY_MARGIN branches of the target. Arms
/// the hardware breakpoint at the target's instruction pointer.
///
/// All operations are async-signal-safe.
extern "C" fn replay_pmu_handler(
    _signo: libc::c_int,
    _info: *mut libc::siginfo_t,
    _ctx: *mut libc::c_void,
) {
    let rctx = REPLAY_CTX.with(|c| c.get());
    let rctx = match rctx {
        Some(ctx) => ctx,
        None => return,
    };

    // 1. Disable PMU timer to prevent recursive signals.
    disable_timer(rctx.timer_fd);

    // 2. Look up the next target from the cursor.
    let cursor = unsafe { &*rctx.cursor };
    let target = match cursor.current_target() {
        Some(t) => t,
        None => return, // No more targets.
    };

    // 3. Arm the hardware breakpoint at the target instruction pointer.
    arm_breakpoint(rctx.bp_fd, target.instruction_pointer);

    // 4. Return from signal handler — execution resumes with breakpoint armed.
}

/// Breakpoint signal handler for replay mode (SIGTRAP).
///
/// Fires when execution hits the target instruction pointer. Preempts
/// the worker by yielding the token, then arms the timer for the next
/// target (if any).
///
/// All operations are async-signal-safe.
extern "C" fn replay_bp_handler(
    _signo: libc::c_int,
    _info: *mut libc::siginfo_t,
    _ctx: *mut libc::c_void,
) {
    let rctx = REPLAY_CTX.with(|c| c.get());
    let rctx = match rctx {
        Some(ctx) => ctx,
        None => return,
    };

    let cursor = unsafe { &*rctx.cursor };
    let ring = unsafe { &*rctx.ring };

    // 1. Disable breakpoint to prevent re-firing immediately.
    disable_timer(rctx.bp_fd);

    // 2. Read the current target (should exist since the PMU handler armed us).
    let target = match cursor.current_target() {
        Some(t) => *t,
        None => return,
    };

    // 3. Save SimulatorState context.
    let sim_ptr = match crate::kfuncs::sim_state_ptr() {
        Some(p) => p,
        None => return,
    };
    let (saved_cpu, saved_ops_ctx, saved_waker) = unsafe {
        (
            (*sim_ptr).current_cpu,
            (*sim_ptr).ops_context,
            (*sim_ptr).waker_task_raw,
        )
    };

    // 4. Track structop RBC and emit trace message.
    record_rbc_preemption(target.rbc_count);
    let sinfo = structop_info();
    tracing::trace!(
        "preempt:rbc structop {}:{} rbc {} rip 0x{:x}",
        sinfo.cpu_count,
        sinfo.global_count,
        sinfo.rbc_total,
        target.instruction_pointer,
    );

    // 4a. Record the preemption point (with structop context).
    ring.record_preemption(
        target.rbc_count,
        target.instruction_pointer,
        saved_cpu,
        rctx.worker_id,
        sinfo,
    );

    // 5. Yield token (futex-based, signal-safe).
    ring.inc_signal_preempt();
    ring.yield_token(rctx.worker_id);

    // 6. Resumed — restore SimulatorState context.
    unsafe {
        (*sim_ptr).current_cpu = saved_cpu;
        (*sim_ptr).ops_context = saved_ops_ctx;
        (*sim_ptr).waker_task_raw = saved_waker;
    }

    // 7. Advance cursor and arm timer for next target.
    if cursor.advance() {
        if let Some(next) = cursor.current_target() {
            arm_replay_timer(rctx.timer_fd, next.rbc_count);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_worker_completes() {
        let ring = PreemptRing::new(1, 42);
        ring.start();
        ring.wait_for_token(WorkerId(0));
        ring.finish(WorkerId(0));
        ring.wait_all_done();
    }

    #[test]
    fn test_two_workers_interleave() {
        let ring = PreemptRing::new(2, 42);

        std::thread::scope(|s| {
            let ring_ref = &ring;

            s.spawn(move || {
                ring_ref.wait_for_token(WorkerId(0));
                ring_ref.yield_token(WorkerId(0));
                ring_ref.finish(WorkerId(0));
            });

            s.spawn(move || {
                ring_ref.wait_for_token(WorkerId(1));
                ring_ref.yield_token(WorkerId(1));
                ring_ref.finish(WorkerId(1));
            });

            ring.start();
            ring.wait_all_done();
        });
    }

    #[test]
    fn test_prng_determinism() {
        let order1 = run_and_record_order(3, 12345);
        let order2 = run_and_record_order(3, 12345);
        assert_eq!(order1, order2, "same seed must give same order");
    }

    #[test]
    fn test_different_seeds_may_differ() {
        let order1 = run_and_record_order(4, 100);
        let order2 = run_and_record_order(4, 999);
        let _ = (order1, order2); // Just verify no panics.
    }

    #[test]
    fn test_finish_without_yield() {
        let ring = PreemptRing::new(2, 42);

        std::thread::scope(|s| {
            let ring_ref = &ring;

            s.spawn(move || {
                ring_ref.wait_for_token(WorkerId(0));
                ring_ref.finish(WorkerId(0));
            });

            s.spawn(move || {
                ring_ref.wait_for_token(WorkerId(1));
                ring_ref.finish(WorkerId(1));
            });

            ring.start();
            ring.wait_all_done();
        });
    }

    #[test]
    fn test_multiple_yields() {
        let ring = PreemptRing::new(2, 42);

        std::thread::scope(|s| {
            let ring_ref = &ring;

            s.spawn(move || {
                ring_ref.wait_for_token(WorkerId(0));
                ring_ref.yield_token(WorkerId(0));
                ring_ref.yield_token(WorkerId(0));
                ring_ref.yield_token(WorkerId(0));
                ring_ref.finish(WorkerId(0));
            });

            s.spawn(move || {
                ring_ref.wait_for_token(WorkerId(1));
                ring_ref.yield_token(WorkerId(1));
                ring_ref.finish(WorkerId(1));
            });

            ring.start();
            ring.wait_all_done();
        });
    }

    #[test]
    fn test_many_workers_stress() {
        // Stress test with many workers and many yields.
        for seed in [1, 42, 12345, 999999] {
            let n = 8;
            let ring = PreemptRing::new(n, seed);

            std::thread::scope(|s| {
                let ring_ref = &ring;
                for i in 0..n {
                    s.spawn(move || {
                        ring_ref.wait_for_token(WorkerId(i));
                        for _ in 0..10 {
                            ring_ref.yield_token(WorkerId(i));
                        }
                        ring_ref.finish(WorkerId(i));
                    });
                }
                ring.start();
                ring.wait_all_done();
            });
        }
    }

    #[test]
    fn test_roll_timeslice() {
        let ring = PreemptRing::new(1, 42);
        // Must be in range.
        for _ in 0..100 {
            let ts = ring.roll_timeslice(50, 500);
            assert!(ts >= 50 && ts <= 500, "timeslice {ts} out of range");
        }
        // Degenerate range.
        assert_eq!(ring.roll_timeslice(100, 100), 100);
    }

    fn run_and_record_order(n: usize, seed: u32) -> Vec<WorkerId> {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let ring = PreemptRing::new(n, seed);
        let order: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(usize::MAX)).collect();
        let counter = AtomicUsize::new(0);

        std::thread::scope(|s| {
            for i in 0..n {
                let ring_ref = &ring;
                let order_ref = &order;
                let counter_ref = &counter;
                s.spawn(move || {
                    ring_ref.wait_for_token(WorkerId(i));
                    let seq = counter_ref.fetch_add(1, Ordering::SeqCst);
                    order_ref[i].store(seq, Ordering::SeqCst);
                    ring_ref.yield_token(WorkerId(i));
                    ring_ref.finish(WorkerId(i));
                });
            }
            ring.start();
            ring.wait_all_done();
        });

        let mut pairs: Vec<(usize, WorkerId)> = order
            .iter()
            .enumerate()
            .map(|(i, a)| (a.load(Ordering::SeqCst), WorkerId(i)))
            .collect();
        pairs.sort_by_key(|&(seq, _)| seq);
        pairs.into_iter().map(|(_, id)| id).collect()
    }

    #[test]
    fn test_preemption_trace_grouping() {
        let records = vec![
            PreemptionRecord {
                rbc_count: 100,
                instruction_pointer: 0x1000,
                cpu_id: CpuId(0),
                worker_id: WorkerId(0),
                sequence: 0,
                structop_local: 1,
                structop_global: 1,
                structop_rbc: 100,
            },
            PreemptionRecord {
                rbc_count: 200,
                instruction_pointer: 0x2000,
                cpu_id: CpuId(1),
                worker_id: WorkerId(1),
                sequence: 1,
                structop_local: 1,
                structop_global: 2,
                structop_rbc: 200,
            },
            PreemptionRecord {
                rbc_count: 300,
                instruction_pointer: 0x3000,
                cpu_id: CpuId(0),
                worker_id: WorkerId(0),
                sequence: 2,
                structop_local: 1,
                structop_global: 1,
                structop_rbc: 400,
            },
            PreemptionRecord {
                rbc_count: 400,
                instruction_pointer: 0x4000,
                cpu_id: CpuId(1),
                worker_id: WorkerId(1),
                sequence: 3,
                structop_local: 1,
                structop_global: 2,
                structop_rbc: 600,
            },
        ];

        let trace = PreemptionTrace::from_records(&records, 2);
        assert_eq!(trace.num_workers(), 2);
        assert_eq!(trace.total_preemptions(), 4);

        let w0 = trace.worker_trace(WorkerId(0));
        assert_eq!(w0.len(), 2);
        assert_eq!(w0[0].sequence, 0);
        assert_eq!(w0[1].sequence, 2);

        let w1 = trace.worker_trace(WorkerId(1));
        assert_eq!(w1.len(), 2);
        assert_eq!(w1[0].sequence, 1);
        assert_eq!(w1[1].sequence, 3);

        // Out-of-range worker returns empty slice.
        assert!(trace.worker_trace(WorkerId(99)).is_empty());
    }

    #[test]
    fn test_preemption_trace_empty() {
        let trace = PreemptionTrace::from_records(&[], 3);
        assert_eq!(trace.num_workers(), 3);
        assert_eq!(trace.total_preemptions(), 0);
        assert!(trace.worker_trace(WorkerId(0)).is_empty());
    }

    #[test]
    fn test_replay_cursor_basics() {
        let targets = vec![
            PreemptionRecord {
                rbc_count: 100,
                instruction_pointer: 0x1000,
                cpu_id: CpuId(0),
                worker_id: WorkerId(0),
                sequence: 0,
                structop_local: 0,
                structop_global: 0,
                structop_rbc: 0,
            },
            PreemptionRecord {
                rbc_count: 200,
                instruction_pointer: 0x2000,
                cpu_id: CpuId(0),
                worker_id: WorkerId(0),
                sequence: 1,
                structop_local: 0,
                structop_global: 0,
                structop_rbc: 0,
            },
        ];

        let cursor = ReplayCursor::new(targets);
        assert_eq!(cursor.len(), 2);
        assert!(!cursor.is_empty());
        assert_eq!(cursor.overhead(), 0);

        // First target.
        let t = cursor.current_target().unwrap();
        assert_eq!(t.rbc_count, 100);

        // Advance to second.
        assert!(cursor.advance());
        let t = cursor.current_target().unwrap();
        assert_eq!(t.rbc_count, 200);

        // Advance past end.
        assert!(!cursor.advance());
        assert!(cursor.current_target().is_none());
    }

    #[test]
    fn test_replay_cursor_empty() {
        let cursor = ReplayCursor::new(vec![]);
        assert!(cursor.is_empty());
        assert_eq!(cursor.len(), 0);
        assert!(cursor.current_target().is_none());
    }

    #[test]
    fn test_preemption_trace_serialize_roundtrip() {
        let records = vec![
            PreemptionRecord {
                rbc_count: 100,
                instruction_pointer: 0x7f000010c0,
                cpu_id: CpuId(0),
                worker_id: WorkerId(0),
                sequence: 0,
                structop_local: 1,
                structop_global: 1,
                structop_rbc: 100,
            },
            PreemptionRecord {
                rbc_count: 200,
                instruction_pointer: 0x7f000020d0,
                cpu_id: CpuId(1),
                worker_id: WorkerId(1),
                sequence: 1,
                structop_local: 1,
                structop_global: 2,
                structop_rbc: 200,
            },
            PreemptionRecord {
                rbc_count: 300,
                instruction_pointer: 0x7f000030e0,
                cpu_id: CpuId(0),
                worker_id: WorkerId(0),
                sequence: 2,
                structop_local: 1,
                structop_global: 1,
                structop_rbc: 400,
            },
        ];

        let trace = PreemptionTrace::from_records(&records, 2);
        let so_base: u64 = 0x7f00000000;

        // Serialize to buffer.
        let mut buf = Vec::new();
        trace.serialize(&mut buf, so_base).unwrap();
        let text = String::from_utf8(buf.clone()).unwrap();

        // Verify header.
        assert!(text.contains("# scxsim preemption trace"));
        assert!(text.contains("# workers: 2"));
        assert!(text.contains("# total: 3"));

        // Verify structop and rip_offset are present.
        assert!(text.contains("structop="));
        assert!(text.contains("rip_offset="));
        assert!(text.contains("timeslice="));

        // Deserialize with same so_base — should get same absolute RIPs.
        let mut cursor = std::io::Cursor::new(buf);
        let trace2 = PreemptionTrace::deserialize(&mut cursor, so_base).unwrap();
        assert_eq!(trace2.num_workers(), 2);
        assert_eq!(trace2.total_preemptions(), 3);

        let w0 = trace2.worker_trace(WorkerId(0));
        assert_eq!(w0.len(), 2);
        assert_eq!(w0[0].rbc_count, 100);
        assert_eq!(w0[0].instruction_pointer, 0x7f000010c0);
        assert_eq!(w0[0].structop_local, 1);
        assert_eq!(w0[0].structop_rbc, 100);
        assert_eq!(w0[1].rbc_count, 300);

        let w1 = trace2.worker_trace(WorkerId(1));
        assert_eq!(w1.len(), 1);
        assert_eq!(w1[0].rbc_count, 200);
        assert_eq!(w1[0].instruction_pointer, 0x7f000020d0);
        assert_eq!(w1[0].structop_rbc, 200);
    }

    #[test]
    fn test_preemption_trace_aslr_resilience() {
        // Record at one base address, replay at a different one.
        let records = vec![PreemptionRecord {
            rbc_count: 42,
            instruction_pointer: 0x1000_1000, // abs RIP
            cpu_id: CpuId(0),
            worker_id: WorkerId(0),
            sequence: 0,
            structop_local: 1,
            structop_global: 1,
            structop_rbc: 42,
        }];

        let trace = PreemptionTrace::from_records(&records, 1);
        let record_base: u64 = 0x1000_0000;

        // Serialize with record-time base.
        let mut buf = Vec::new();
        trace.serialize(&mut buf, record_base).unwrap();

        // Deserialize with a different replay-time base.
        let replay_base: u64 = 0x2000_0000;
        let mut cursor = std::io::Cursor::new(buf);
        let trace2 = PreemptionTrace::deserialize(&mut cursor, replay_base).unwrap();

        let w0 = trace2.worker_trace(WorkerId(0));
        assert_eq!(w0.len(), 1);
        // rip_offset = 0x1000_1000 - 0x1000_0000 = 0x1000
        // new rip = 0x2000_0000 + 0x1000 = 0x2000_1000
        assert_eq!(w0[0].instruction_pointer, 0x2000_1000);
    }

    #[test]
    fn test_structop_tracking() {
        // Reset state.
        reset_structop_globals();
        reset_structop_cpu_count();

        let info = structop_info();
        assert_eq!(info.cpu_count, 0);
        assert_eq!(info.global_count, 0);
        assert_eq!(info.rbc_total, 0);
        assert_eq!(info.kfunc_count, 0);

        // Begin a structop.
        begin_structop();
        let info = structop_info();
        assert_eq!(info.cpu_count, 1);
        assert_eq!(info.global_count, 1);

        // Record RBC.
        record_rbc_preemption(42);
        let info = structop_info();
        assert_eq!(info.rbc_total, 42);

        // Increment kfunc count.
        inc_structop_kfunc();
        inc_structop_kfunc();
        let info = structop_info();
        assert_eq!(info.kfunc_count, 2);

        // Begin another structop (resets rbc/kfunc).
        begin_structop();
        let info = structop_info();
        assert_eq!(info.cpu_count, 2);
        assert_eq!(info.global_count, 2);
        assert_eq!(info.rbc_total, 0);
        assert_eq!(info.kfunc_count, 0);

        // Cleanup.
        reset_structop_globals();
        reset_structop_cpu_count();
    }

    #[test]
    fn test_maybe_begin_structop() {
        reset_structop_globals();
        reset_structop_cpu_count();

        // Not in ops — should not begin.
        maybe_begin_structop(false);
        assert_eq!(structop_info().cpu_count, 0);

        // Enter ops — should begin.
        maybe_begin_structop(true);
        assert_eq!(structop_info().cpu_count, 1);

        // Still in ops — should NOT begin again.
        maybe_begin_structop(true);
        assert_eq!(structop_info().cpu_count, 1);

        // Leave ops.
        maybe_begin_structop(false);
        assert_eq!(structop_info().cpu_count, 1);

        // Re-enter — should begin again.
        maybe_begin_structop(true);
        assert_eq!(structop_info().cpu_count, 2);

        // Cleanup.
        reset_structop_globals();
        reset_structop_cpu_count();
    }

    #[test]
    fn test_parse_hex_or_dec() {
        assert_eq!(super::parse_hex_or_dec("42").unwrap(), 42);
        assert_eq!(super::parse_hex_or_dec("0x2a").unwrap(), 42);
        assert_eq!(super::parse_hex_or_dec("0X2A").unwrap(), 42);
        assert_eq!(super::parse_hex_or_dec("0xff").unwrap(), 255);
        assert!(super::parse_hex_or_dec("not_a_number").is_err());
    }
}
