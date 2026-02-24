//! Preemption backend trait and generic interleaving drivers.
//!
//! Provides a trait-based abstraction for different preemption backends
//! (PMU timer, hardware breakpoint replay, Frida Stalker). Each backend
//! implements [`PreemptionBackend`] to define how workers are instrumented;
//! the generic [`run_preemptive_dispatch`] and [`run_preemptive_batch`]
//! drivers handle the common worker lifecycle.

pub mod pmu;
pub mod replay;

use std::collections::HashMap;

use tracing::debug;

use crate::engine::{batch_worker_body, dispatch_worker_body, EventQueue, Simulator};
use crate::ffi::Scheduler;
use crate::interleave::WorkerId;
use crate::kfuncs::{self, OpsContext, SimulatorState};
use crate::preempt::PreemptRing;
use crate::types::CpuId;

/// Wrapper to send raw pointers across thread boundaries.
///
/// # Safety
///
/// Callers must ensure only one thread accesses the pointed-to data at a time
/// (enforced by PreemptRing / TokenRing token passing).
pub(crate) struct SendPtr<T>(pub *mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

/// Per-worker accounting delta merged into `structop_accum` after the worker
/// body runs. Backends populate this in [`PreemptionBackend::disarm`].
#[derive(Default)]
pub(crate) struct StructopDelta {
    /// Cumulative C-code-only retired branch conditional count (from a
    /// measurement counter that pauses during kfuncs).
    pub rbc_total: u64,
    /// Number of cooperative kfunc-boundary yields + preemptive signal yields.
    pub interleave_count: u64,
}

/// Trait for preemption interleaving backends.
///
/// A backend determines how worker threads are preempted during concurrent
/// scheduler execution. The generic drivers ([`run_preemptive_dispatch`],
/// [`run_preemptive_batch`]) call these methods at the appropriate lifecycle
/// points, ensuring consistent structop accounting, ops_context clearing,
/// and token ring protocol across all backends.
///
/// # Lifecycle (per worker)
///
/// 1. [`worker_setup`] — create instrumentation state, install TLS
/// 2. `ring.wait_for_token(worker_id)` — acquire execution token
/// 3. `enter_sim(state, cpu)` — enter simulation context
/// 4. [`arm`] — enable instrumentation (timer, stalker, etc.)
/// 5. Worker body runs (dispatch or batch event processing)
/// 6. [`disarm`] — disable instrumentation, return accounting deltas
/// 7. Common: drain structop, clear ops_context, finish, exit_sim
/// 8. [`worker_teardown`] — uninstall TLS, close fds
///
/// [`worker_setup`]: PreemptionBackend::worker_setup
/// [`arm`]: PreemptionBackend::arm
/// [`disarm`]: PreemptionBackend::disarm
/// [`worker_teardown`]: PreemptionBackend::worker_teardown
pub(crate) trait PreemptionBackend: Sync {
    /// Per-worker context created during setup, carried through arm/disarm.
    type WorkerCtx: Send;

    /// One-time global setup before spawning workers (e.g., install signal
    /// handlers). Default: no-op.
    fn global_setup(&self) {}

    /// One-time global teardown after all workers finish. Default: no-op.
    fn global_teardown(&self) {}

    /// Create per-worker instrumentation state and install preemption TLS.
    fn worker_setup(&self, ring: &PreemptRing, worker_id: WorkerId) -> Self::WorkerCtx;

    /// Arm instrumentation before entering scheduler code.
    /// Called after the worker acquires the token and enters sim.
    fn arm(&self, ctx: &mut Self::WorkerCtx, ring: &PreemptRing);

    /// Disarm instrumentation after scheduler code returns.
    /// Returns structop accounting deltas to merge into `structop_accum`.
    fn disarm(&self, ctx: &mut Self::WorkerCtx) -> StructopDelta;

    /// Per-worker cleanup: uninstall preemption TLS, close fds.
    /// Called after the worker has released the token and exited sim.
    fn worker_teardown(&self, ctx: Self::WorkerCtx);

    /// Log the completion summary after all workers finish.
    fn log_completion(&self, ring: &PreemptRing);
}

/// Run concurrent dispatch using a [`PreemptionBackend`].
///
/// Spawns one worker per CPU, each executing `dispatch_worker_body` inside
/// the PreemptRing token-passing protocol with backend-specific
/// instrumentation. The common lifecycle (structop drain, ops_context clear,
/// token protocol) is handled here.
pub(crate) fn run_preemptive_dispatch<S, B>(
    dispatch_cpus: &[CpuId],
    state_send: &SendPtr<SimulatorState>,
    sched_send: &SendPtr<S>,
    seed: u32,
    backend: &B,
) where
    S: Scheduler,
    B: PreemptionBackend,
{
    let ring = PreemptRing::new(dispatch_cpus.len(), seed);
    backend.global_setup();

    std::thread::scope(|s| {
        let ring_ref = &ring;
        let state_ref = state_send;
        let sched_ref = sched_send;

        for (i, &cpu) in dispatch_cpus.iter().enumerate() {
            let worker_id = WorkerId(i);

            s.spawn(move || {
                let sp = state_ref.0;
                let schp = sched_ref.0 as *const S;

                let mut ctx = backend.worker_setup(ring_ref, worker_id);
                ring_ref.wait_for_token(worker_id);

                // Enter sim AFTER acquiring the token to avoid racing on
                // SimulatorState.current_cpu with other workers.
                unsafe { kfuncs::enter_sim(&mut *sp, cpu) };

                backend.arm(&mut ctx, ring_ref);

                unsafe {
                    debug!(cpu = cpu.0, "enter:structop dispatch (preemptive)");
                    dispatch_worker_body(sp, schp, cpu);
                }

                let delta = backend.disarm(&mut ctx);

                // Drain per-worker structop deltas into the accumulator.
                unsafe {
                    let idx = cpu.0 as usize;
                    if idx < (*sp).structop_accum.len() {
                        let accum = &mut (&mut (*sp).structop_accum)[idx];
                        accum.rbc_total += delta.rbc_total;
                        accum.interleave_count += delta.interleave_count;
                    }
                }

                // Clear ops_context AFTER disabling instrumentation (so
                // pending signals still see the true callback context)
                // and BEFORE releasing the token (so the new token
                // holder's ops_context isn't clobbered by our exit_sim).
                unsafe { (*sp).ops_context = OpsContext::None };
                crate::preempt::set_current_ops_context(OpsContext::None);
                ring_ref.finish(worker_id);
                kfuncs::exit_sim_no_clear_ops();

                backend.worker_teardown(ctx);
            });
        }

        ring.start();
        ring.wait_all_done();
    });

    backend.log_completion(&ring);
    backend.global_teardown();
}

/// Run concurrent batch event processing using a [`PreemptionBackend`].
///
/// Like [`run_preemptive_dispatch`] but each worker processes a batch of
/// events for its CPU via `batch_worker_body`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_preemptive_batch<S, B>(
    per_cpu: &HashMap<CpuId, Vec<crate::engine::Event>>,
    cpu_ids: &[CpuId],
    sim_send: &SendPtr<Simulator<S>>,
    state_send: &SendPtr<SimulatorState>,
    tasks_send: &SendPtr<HashMap<crate::types::Pid, crate::task::SimTask>>,
    events_send: &SendPtr<EventQueue>,
    cgroup_send: &SendPtr<crate::cgroup::CgroupRegistry>,
    seed: u32,
    watchdog_timeout: Option<crate::types::TimeNs>,
    duration_ns: crate::types::TimeNs,
    max_cgroups: u32,
    backend: &B,
) where
    S: Scheduler,
    B: PreemptionBackend,
{
    let ring = PreemptRing::new(cpu_ids.len(), seed);
    backend.global_setup();

    std::thread::scope(|s| {
        let ring_ref = &ring;
        let sim_ref = sim_send;
        let state_ref = state_send;
        let tasks_ref = tasks_send;
        let events_ref = events_send;
        let cgroup_ref = cgroup_send;

        for (i, &cpu) in cpu_ids.iter().enumerate() {
            let worker_id = WorkerId(i);
            let cpu_events = per_cpu.get(&cpu).cloned().unwrap_or_default();

            s.spawn(move || {
                let simp = sim_ref.0 as *const Simulator<S>;
                let sp = state_ref.0;

                let mut ctx = backend.worker_setup(ring_ref, worker_id);
                ring_ref.wait_for_token(worker_id);

                unsafe { kfuncs::enter_sim(&mut *sp, cpu) };

                backend.arm(&mut ctx, ring_ref);

                unsafe {
                    batch_worker_body(
                        simp,
                        sp,
                        tasks_ref.0,
                        events_ref.0,
                        cgroup_ref.0,
                        cpu_events,
                        watchdog_timeout,
                        duration_ns,
                        max_cgroups,
                    );
                }

                let delta = backend.disarm(&mut ctx);

                unsafe {
                    let idx = cpu.0 as usize;
                    if idx < (*sp).structop_accum.len() {
                        let accum = &mut (&mut (*sp).structop_accum)[idx];
                        accum.rbc_total += delta.rbc_total;
                        accum.interleave_count += delta.interleave_count;
                    }
                }

                unsafe { (*sp).ops_context = OpsContext::None };
                crate::preempt::set_current_ops_context(OpsContext::None);
                ring_ref.finish(worker_id);
                kfuncs::exit_sim_no_clear_ops();

                backend.worker_teardown(ctx);
            });
        }

        ring.start();
        ring.wait_all_done();
    });

    backend.log_completion(&ring);
    backend.global_teardown();
}
