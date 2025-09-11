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
 * scx_cgroup_bw_lib_init - 
 * @config:
 *
 * Returns
 */
int scx_cgroup_bw_lib_init(struct scx_cgroup_bw_config *config);

/**
 * scx_cgroup_bw_init - 
 * @cgrp:
 * @args:
 *
 * Returns
 */
int scx_cgroup_bw_init(struct cgroup *cgrp __arg_trusted, struct scx_cgroup_init_args *args __arg_trusted);

/**
 * scx_cgroup_bw_exit - 
 * @cgrp:
 *
 * Returns
 */
int scx_cgroup_bw_exit(struct cgroup *cgrp __arg_trusted);

/**
 * scx_cgroup_bw_set - 
 * @cgrp:
 *
 * Returns
 */
int scx_cgroup_bw_set(struct cgroup *cgrp __arg_trusted, u64 period_us, u64 quota_us, u64 burst_us);

/**
 * scx_cgroup_bw_reserve - 
 * @cgrp:
 * @llc_id:
 * @slice_ns:
 *
 * Returns
 */
int scx_cgroup_bw_reserve(struct cgroup *cgrp __arg_trusted, int llc_id, u64 slice_ns);

/**
 * scx_cgroup_bw_consume - 
 * @cgrp:
 * @llc_id:
 * @reserved_ns:
 * @consumed_ns:
 *
 * Returns
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
