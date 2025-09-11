/*
 * SPDX-License-Identifier: GPL-2.0
 * Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
 * Author: Changwoo Min <changwoo@igalia.com>
 */
#pragma once

#include <errno.h>

/**
 * Configs for cpu.max
 */
struct scx_cgroup_bw_config {
	/* The budget allocation from a parent cgroup to a child cgroup in nsec. */
	u64		budget_p2c;
	/* The budget allocation from a cgroup to its LLC context in nsec. */
	u64		budget_c2l;
	/* The number of LLC domains. LLC ID should be in [0, nr_llcs). */
	int		nr_llcs;
	/* verbose level */
	int		verbose;
};

/**
 * scx_cgroup_bw_lib_init - Initialize the library with a configuration.
 * @config: tunnables, see the struct definition.
 *
 * It should be called for the library initialization before calling any
 * other API.
 *
 * Return 0 for success, -errno for failure.
 */
int scx_cgroup_bw_lib_init(struct scx_cgroup_bw_config *config);

/**
 * scx_cgroup_bw_init - Initialize a cgroup for CPU bandwidth control.
 * @cgrp: cgroup being initialized.
 * @args: init arguments, see the struct definition.
 *
 * Either the BPF scheduler is being loaded or @cgrp created, initialize
 * @cgrp for CPU bandwidth control. When being loaded, cgroups are initialized
 * in a pre-order from the root. This operation may block.
 *
 * Return 0 for success, -errno for failure.
 */
int scx_cgroup_bw_init(struct cgroup *cgrp __arg_trusted, struct scx_cgroup_init_args *args __arg_trusted);

/**
 * scx_cgroup_bw_exit - Exit a cgroup.
 * @cgrp: cgroup being exited
 *
 * Either the BPF scheduler is being unloaded or @cgrp destroyed, exit
 * @cgrp for sched_ext. This operation my block.
 *
 * Return 0 for success, -errno for failure.
 */
int scx_cgroup_bw_exit(struct cgroup *cgrp __arg_trusted);

/**
 * scx_cgroup_bw_set - A cgroup's bandwidth is being changed.
 * @cgrp: cgroup whose bandwidth is being updated
 * @period_us: bandwidth control period
 * @quota_us: bandwidth control quota
 * @burst_us: bandwidth control burst
 *
 * Update @cgrp's bandwidth control parameters. This is from the cpu.max
 * cgroup interface.
 *
 * @quota_us / @period_us determines the CPU bandwidth @cgrp is entitled
 * to. For example, if @period_us is 1_000_000 and @quota_us is
 * 2_500_000. @cgrp is entitled to 2.5 CPUs. @burst_us can be
 * interpreted in the same fashion and specifies how much @cgrp can
 * burst temporarily. The specific control mechanism and thus the
 * interpretation of @period_us and burstiness is upto to the BPF
 * scheduler.
 *
 * Return 0 for success, -errno for failure.
 */
int scx_cgroup_bw_set(struct cgroup *cgrp __arg_trusted, u64 period_us, u64 quota_us, u64 burst_us);

/**
 * scx_cgroup_bw_reserve - Reserve a time budget for executing a task.
 * @cgrp: cgroup where a task belongs to.
 * @llc_id: caller's LLC id.
 * @slice_ns: amount of time budgeted.
 *
 * Return 0 for success in reserving the time slice,
 * -EAGAIN when the cgroup is throttled, and
 * -errno for some other failures.
 */
int scx_cgroup_bw_reserve(struct cgroup *cgrp __arg_trusted, int llc_id, u64 slice_ns);

/**
 * scx_cgroup_bw_consume - Consume the time actually used after the task execution.
 * @cgrp: cgroup where a task belongs to.
 * @llc_id: caller's LLC id.
 * @reserved_ns: amount of time budgeted.
 * @consumed_ns: amount of time actually used.
 *
 * Return 0 for success, -errno for failure.
 */
int scx_cgroup_bw_consume(struct cgroup *cgrp __arg_trusted, int llc_id, u64 reserved_ns, u64 consumed_ns);

/**
 * scx_cgroup_bw_put_aside - 
 * @p:
 * @vtime:
 * @cgrp:
 * @llc_id:
 *
 * Returns
 */
int scx_cgroup_bw_put_aside(struct task_struct *p __arg_trusted, u64 vtime, struct cgroup *cgrp __arg_trusted, int llc_id);

/**
 * REGISTER_SCX_CGROUP_BW_ENQUEUE_CB - Register an enqueue callback.
 * @enqueue_cb: A function name with a prototype of 'void fn(pid_t pid)'.
 *
 * @enqueue_cb enqueues a task with @pid following the BPF scheduler's
 * regular enqueue path. @enqueue_cb will be called when a throttled cgroup
 * becomes available again or when the cgroup is exiting for some reason.
 * @enqueue_cb MUST enqueue the task; otherwise, the task will be lost and
 * never be scheduled.
 */
#define REGISTER_SCX_CGROUP_BW_ENQUEUE_CB(enqueue_cb)				\
	__hidden int scx_group_bw_enqueue_cb(pid_t pid)				\
	{									\
		extern int enqueue_cb(pid_t pid);				\
		enqueue_cb(pid);						\
		return 0;							\
	}

extern int scx_group_bw_enqueue_cb(pid_t pid);
