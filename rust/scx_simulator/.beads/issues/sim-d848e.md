---
title: Enforce OpsContext by construction via with_ops_callback helper
status: open
priority: 3
issue_type: task
labels:
- refactoring
- engine
created_at: 2026-02-23T16:20:27.853265682+00:00
updated_at: 2026-02-23T16:20:27.853265682+00:00
---

# Description

## Problem

The engine has ~30 open-coded sequences of enter_sim / set_ops_context /
start_rbc / scheduler.xxx() / charge_sched_time / exit_sim. It is easy
to add a new callback and forget set_ops_context, causing ops=none in
preemption traces (as we just fixed for tick/stopping/running/update_idle/
fire_timer/cpu_online/cpu_offline/runnable/quiescent/dequeue/enable).

Three runtime callbacks still lack set_ops_context: cpu_release,
cpu_acquire, and cgroup_move (these don't yet have OpsContext variants).

## Proposed Solution

Replace the open-coded pattern with a single helper that takes OpsContext
as a required parameter, making omission a compile error:

    fn with_ops_callback<R>(
        state: &mut SimulatorState,
        cpu: CpuId,
        ctx: OpsContext,
        label: &str,
        checkpoint: Option<CheckpointEvent>,
        f: impl FnOnce(&mut SimulatorState) -> R,
    ) -> R

Make enter_sim / exit_sim non-public to the engine module so all scheduler
callbacks must go through this helper.

## Challenges

~8 call sites do extra work between enter_sim and exit_sim:
- Setting waker_task_raw before select_cpu/runnable
- Clearing/resolving pending_dispatch after enqueue/select_cpu/dispatch
- Sharing one enter/exit bracket across multiple callbacks (dequeue+quiescent)
- Setting task_ops_state between callbacks
- cgroup_registry.prepare_css_iter_from_root() before fire_timer/cgroup_init

These need either pre/post closures, or refactoring into separate brackets.

## Also

Add OpsContext variants for CpuRelease, CpuAcquire, CgroupMove,
CgroupInit, CgroupExit and set them at their call sites.

## Relationship

Overlaps with the engine.rs refactoring plan (call_enqueue helper, etc.)
and could be done as part of that effort.
