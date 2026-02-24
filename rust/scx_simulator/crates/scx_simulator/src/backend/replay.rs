//! Hardware breakpoint replay preemption backend.
//!
//! Replays a recorded preemption trace using a hybrid PMU + hardware
//! breakpoint approach. The PMU timer fires when we're within
//! [`REPLAY_MARGIN`](crate::preempt::REPLAY_MARGIN) branches of the target,
//! then a hardware breakpoint catches the exact instruction pointer.
//!
//! Falls back to cooperative-only interleaving when PMU or hardware
//! breakpoints are unavailable (VMs, containers).

use std::os::unix::io::RawFd;

use tracing::debug;

use crate::backend::pmu::setup_pmu_timer;
use crate::backend::{PreemptionBackend, StructopDelta};
use crate::interleave::WorkerId;
use crate::perf;
use crate::preempt::trace::PreemptionTrace;
use crate::preempt::{self, PreemptRing, ReplayCursor};

/// Hardware breakpoint replay preemption backend.
///
/// Each worker follows its recorded trace of preemption points using the
/// hybrid PMU + hardware breakpoint approach: the PMU timer fires when
/// we're within REPLAY_MARGIN of the target RBC count, then a hardware
/// breakpoint catches the exact instruction pointer.
pub(crate) struct ReplayBackend {
    /// Per-worker cursors into the replay trace.
    cursors: Vec<ReplayCursor>,
    /// The PMU event type used during recording.
    break_on: perf::PmuEvent,
}

/// Per-worker state for the replay backend.
pub(crate) struct ReplayWorkerCtx {
    timer: Option<perf::RbcTimer>,
    timer_fd: RawFd,
    bp_fd: RawFd,
    worker_idx: usize,
}

impl ReplayBackend {
    /// Create a new replay backend from a recorded preemption trace.
    ///
    /// Builds per-worker cursors from the trace, one per dispatch CPU.
    pub fn new(trace: &PreemptionTrace, num_workers: usize) -> Self {
        let cursors = (0..num_workers)
            .map(|i| {
                let targets = trace.worker_trace(WorkerId(i)).to_vec();
                ReplayCursor::new(targets)
            })
            .collect();
        ReplayBackend {
            cursors,
            break_on: trace.break_on(),
        }
    }
}

impl PreemptionBackend for ReplayBackend {
    type WorkerCtx = ReplayWorkerCtx;

    fn global_setup(&self) {
        preempt::install_replay_signal_handlers();
    }

    fn global_teardown(&self) {
        preempt::uninstall_replay_signal_handlers();
    }

    fn worker_setup(&self, ring: &PreemptRing, worker_id: WorkerId) -> ReplayWorkerCtx {
        let i = worker_id.0;
        let cursor = &self.cursors[i];

        // Create per-thread PMU timer.
        let (timer, timer_fd) = setup_pmu_timer(false, self.break_on);

        // Create per-thread hardware breakpoint (at dummy addr 0x1).
        let bp_fd = {
            let bp = perf::try_create_hw_breakpoint(0x1);
            match bp {
                Some(b) => {
                    let fd = b.raw_fd();
                    // Leak the bp so the fd stays open; we manage
                    // the fd lifetime manually via raw ioctls.
                    std::mem::forget(b);
                    fd
                }
                None => {
                    tracing::warn!(
                        worker = i,
                        "replay: HW breakpoint unavailable, cooperative-only fallback"
                    );
                    -1
                }
            }
        };

        if timer_fd >= 0 && bp_fd >= 0 {
            debug!(
                worker = i,
                targets = cursor.len(),
                "replay: PMU + breakpoint armed"
            );
        } else {
            debug!(
                worker = i,
                targets = cursor.len(),
                "replay: cooperative-only (PMU or BP unavailable)"
            );
        }

        // Install replay context (replaces normal preempt context).
        preempt::install_replay(ring, worker_id, timer_fd, bp_fd, cursor);

        ReplayWorkerCtx {
            timer,
            timer_fd,
            bp_fd,
            worker_idx: i,
        }
    }

    fn arm(&self, ctx: &mut ReplayWorkerCtx, _ring: &PreemptRing) {
        let cursor = &self.cursors[ctx.worker_idx];

        // Arm PMU timer for the first replay target.
        if ctx.timer_fd >= 0 && ctx.bp_fd >= 0 {
            if let Some(first) = cursor.current_target() {
                preempt::arm_replay_timer_pub(ctx.timer_fd, first.rbc_count);
            }
        }
    }

    fn disarm(&self, ctx: &mut ReplayWorkerCtx) -> StructopDelta {
        // Disable timer.
        if let Some(ref t) = ctx.timer {
            let _ = t.disable();
        }
        // Disable and close breakpoint fd.
        if ctx.bp_fd >= 0 {
            unsafe {
                libc::ioctl(ctx.bp_fd, scx_perf::PERF_IOC_DISABLE, 0 as libc::c_ulong);
            }
        }

        StructopDelta {
            rbc_total: 0,
            interleave_count: preempt::structop_info().interleave_count,
        }
    }

    fn worker_teardown(&self, ctx: ReplayWorkerCtx) {
        // Close breakpoint fd (was leaked from HwBreakpoint via forget).
        if ctx.bp_fd >= 0 {
            unsafe { libc::close(ctx.bp_fd) };
        }
        preempt::uninstall_replay();
        // timer dropped here — closes the perf fd
    }

    fn log_completion(&self, ring: &PreemptRing) {
        debug!(
            signal_preemptions = ring.signal_preemptions(),
            cooperative_yields = ring.cooperative_yields(),
            "replay interleave: complete"
        );
    }
}
