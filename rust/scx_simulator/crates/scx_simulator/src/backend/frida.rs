//! Frida Stalker preemption backend.
//!
//! Uses Frida's Stalker dynamic binary instrumentation engine to count
//! conditional branches in the scheduler `.so` code in software. A
//! thread-local countdown is decremented at each instrumented branch;
//! when it reaches zero, the worker yields via the `PreemptRing`.
//!
//! This provides exact, deterministic branch counts without PMU hardware
//! support, making it suitable for VMs and containers.

use frida_gum::stalker::{NoneEventSink, Stalker};
use frida_gum::Gum;
use tracing::debug;

use crate::backend::{PreemptionBackend, StructopDelta};
use crate::interleave::WorkerId;
use crate::preempt::{self, PreemptRing};
use crate::stalker::{self, SyncTransformer, TextRange};

/// Frida Stalker preemption backend.
///
/// Each worker uses Frida's Stalker to instrument conditional branches
/// in the scheduler `.so`. A thread-local countdown fires when the
/// software branch counter expires, yielding via the `PreemptRing`.
/// No PMU timer or signal handler is needed.
pub(crate) struct FridaBackend<'a> {
    pub gum: &'a Gum,
    pub text_range: &'a TextRange,
    pub transformer: SyncTransformer<'a>,
    pub timeslice_min: u64,
    pub timeslice_max: u64,
}

/// Per-worker state for the Frida backend.
///
/// Wraps a `Stalker` instance. Stalker is not `Send` because it contains
/// raw pointers to Frida's internal thread state. However, each Stalker
/// is created on its worker thread in `worker_setup` and only used on
/// that same thread, so the `Send` impl is safe.
pub(crate) struct FridaWorkerCtx {
    stalker: Stalker,
}

// SAFETY: Each FridaWorkerCtx is created and used on a single worker
// thread. The `Send` bound is required by the `PreemptionBackend` trait
// but the value never actually crosses thread boundaries — it is created
// inside `worker_setup` which runs on the worker thread, and consumed
// by `worker_teardown` on the same thread.
unsafe impl Send for FridaWorkerCtx {}

impl<'a> FridaBackend<'a> {
    /// Create a new Frida backend.
    pub fn new(
        gum: &'a Gum,
        text_range: &'a TextRange,
        timeslice_min: u64,
        timeslice_max: u64,
    ) -> Self {
        let transformer = SyncTransformer::new(gum, text_range);
        FridaBackend {
            gum,
            text_range,
            transformer,
            timeslice_min,
            timeslice_max,
        }
    }
}

impl<'a> PreemptionBackend for FridaBackend<'a> {
    type WorkerCtx = FridaWorkerCtx;

    fn global_setup(&self) {
        stalker::reset_counters();
    }

    fn global_teardown(&self) {
        stalker::reset_counters();
    }

    fn worker_setup(&self, ring: &PreemptRing, worker_id: WorkerId) -> FridaWorkerCtx {
        let stalker_inst = Stalker::new(self.gum);

        // Install with timer_fd=-1, measure_fd=-1 (no PMU).
        preempt::install(
            ring,
            worker_id,
            -1,
            -1,
            self.timeslice_min,
            self.timeslice_max,
        );

        debug!(worker = worker_id.0, "frida: Stalker worker installed");

        FridaWorkerCtx {
            stalker: stalker_inst,
        }
    }

    fn arm(&self, ctx: &mut FridaWorkerCtx, ring: &PreemptRing) {
        let ts = ring.roll_timeslice(self.timeslice_min, self.timeslice_max);
        stalker::arm_software_rbc(ts, self.text_range.base as u64);
        ctx.stalker
            .follow_me::<NoneEventSink>(&self.transformer.0, None);
    }

    fn disarm(&self, ctx: &mut FridaWorkerCtx) -> StructopDelta {
        ctx.stalker.unfollow_me();
        stalker::disarm_software_rbc();

        StructopDelta {
            rbc_total: 0,
            interleave_count: preempt::structop_info().interleave_count,
        }
    }

    fn worker_teardown(&self, _ctx: FridaWorkerCtx) {
        preempt::uninstall();
        // Stalker dropped here
    }

    fn log_completion(&self, ring: &PreemptRing) {
        debug!(
            signal_preemptions = ring.signal_preemptions(),
            cooperative_yields = ring.cooperative_yields(),
            stalker_callouts = stalker::total_callouts(),
            "frida interleave: complete"
        );
    }
}
