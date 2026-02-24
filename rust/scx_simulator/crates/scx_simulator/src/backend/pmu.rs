//! PMU timer preemption backend.
//!
//! Uses hardware Performance Monitoring Unit (PMU) counters to fire a signal
//! after a random number of retired conditional branches. Falls back to
//! cooperative-only interleaving when PMU is unavailable (VMs, containers).

use std::os::unix::io::RawFd;

use tracing::debug;

use crate::backend::{PreemptionBackend, StructopDelta};
use crate::interleave::WorkerId;
use crate::perf::{self, PmuEvent, RbcTimer};
use crate::preempt::{self, is_determinism_mode_enabled, PreemptRing};

/// PMU timer preemption backend.
///
/// Each worker gets a per-thread PMU timer (fires `SIGSTKFLT` on RBC
/// overflow) and a separate RBC measurement counter (for structop
/// accounting). When the PMU is unavailable, workers still participate
/// in the PreemptRing but only yield at cooperative kfunc boundaries.
pub(crate) struct PmuBackend {
    pub timeslice_min: u64,
    pub timeslice_max: u64,
    pub cooperative_only: bool,
    pub break_on: PmuEvent,
}

/// Per-worker state for the PMU backend.
pub(crate) struct PmuWorkerCtx {
    timer: Option<RbcTimer>,
    timer_fd: RawFd,
    measure_counter: Option<perf::RbcCounter>,
}

impl PreemptionBackend for PmuBackend {
    type WorkerCtx = PmuWorkerCtx;

    fn global_setup(&self) {
        preempt::install_signal_handler();
    }

    fn global_teardown(&self) {
        preempt::uninstall_signal_handler();
    }

    fn worker_setup(&self, ring: &PreemptRing, worker_id: WorkerId) -> PmuWorkerCtx {
        // Create per-thread PMU timer (may be unavailable in VMs).
        // Skip if cooperative_only mode is requested.
        let (timer, timer_fd) = setup_pmu_timer(self.cooperative_only, self.break_on);

        // Create per-thread RBC measurement counter (separate from the
        // preemption timer). This counts cumulative C-code RBC with pauses
        // during kfuncs, matching non-preemptive mode.
        let measure_counter = perf::try_create_rbc_counter();
        let measure_fd = measure_counter.as_ref().map_or(-1, |c| c.raw_fd());

        if self.cooperative_only {
            debug!(
                worker = worker_id.0,
                "preempt: cooperative-only (by config)"
            );
        } else if timer_fd >= 0 {
            debug!(
                worker = worker_id.0,
                break_on = %self.break_on,
                "preempt: PMU timer armed"
            );
        } else if is_determinism_mode_enabled() {
            tracing::warn!(
                worker = worker_id.0,
                "WARNING: PMU unavailable in determinism mode. \
                 Falling back to cooperative-only interleaving. \
                 Rebuild with --features frida for exact counts."
            );
        } else {
            debug!(
                worker = worker_id.0,
                "preempt: PMU unavailable, cooperative-only"
            );
        }

        preempt::install(
            ring,
            worker_id,
            timer_fd,
            measure_fd,
            self.timeslice_min,
            self.timeslice_max,
        );

        PmuWorkerCtx {
            timer,
            timer_fd,
            measure_counter,
        }
    }

    fn arm(&self, ctx: &mut PmuWorkerCtx, ring: &PreemptRing) {
        // Arm the PMU timer before entering scheduler C code.
        if ctx.timer_fd >= 0 {
            let ts = ring.roll_timeslice(self.timeslice_min, self.timeslice_max);
            if let Some(ref t) = ctx.timer {
                let _ = t.reset();
                let _ = t.set_period(ts);
                let _ = t.enable();
            }
        }

        // Enable measurement counter before entering C code.
        if let Some(ref mc) = ctx.measure_counter {
            let _ = mc.reset();
            let _ = mc.enable();
        }
    }

    fn disarm(&self, ctx: &mut PmuWorkerCtx) -> StructopDelta {
        // Disable timer.
        if let Some(ref t) = ctx.timer {
            let _ = t.disable();
        }

        // Disable and read the measurement counter. This gives cumulative
        // C-code-only RBC (kfuncs were excluded by pause/resume in with_sim).
        let rbc = if let Some(ref mc) = ctx.measure_counter {
            let _ = mc.disable();
            mc.read().unwrap_or(0)
        } else {
            0
        };

        StructopDelta {
            rbc_total: rbc,
            interleave_count: preempt::structop_info().interleave_count,
        }
    }

    fn worker_teardown(&self, _ctx: PmuWorkerCtx) {
        preempt::uninstall();
        // timer + measure_counter dropped here — closes the perf fds
    }

    fn log_completion(&self, ring: &PreemptRing) {
        debug!(
            signal_preemptions = ring.signal_preemptions(),
            cooperative_yields = ring.cooperative_yields(),
            "preemptive interleave: complete"
        );
    }
}

/// Create a per-thread PMU timer for preemption signals.
///
/// Returns `(None, -1)` if `cooperative_only` is true or the PMU is
/// unavailable. Otherwise returns the timer and its raw fd.
pub(crate) fn setup_pmu_timer(
    cooperative_only: bool,
    break_on: PmuEvent,
) -> (Option<RbcTimer>, RawFd) {
    if cooperative_only {
        return (None, -1);
    }
    let timer = perf::try_create_pmu_timer(break_on);
    let timer_fd = match &timer {
        Some(t) => {
            let tid = unsafe { libc::syscall(libc::SYS_gettid) } as libc::pid_t;
            if let Err(e) = t.set_signal_delivery(tid, preempt::PREEMPT_SIGNAL) {
                tracing::warn!("preemptive: signal delivery setup failed: {e}");
                -1
            } else {
                t.raw_fd()
            }
        }
        None => -1,
    };
    (timer, timer_fd)
}
