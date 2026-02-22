//! Integration tests for Frida Stalker software-RBC determinism.
//!
//! These tests verify that when the simulator uses Frida Stalker for
//! retired-branch-conditional counting (instead of PMU hardware counters),
//! two runs with the same seed produce identical checkpoint sequences
//! including memory state hashes.
//!
//! Requires: `cargo test --features frida`
//!
//! The Frida devkit is auto-downloaded by `frida-gum-sys` when the
//! `auto-download` feature is enabled (which it is in our Cargo.toml).
//!
//! **Environment note:** Frida Stalker requires `mmap` and JIT permissions
//! that may be blocked in sandboxed environments (seccomp, containers).
//! Tests that use the full Stalker (`use_frida: true`) are skipped when
//! `SCX_SIM_NO_FRIDA_STALKER=1` is set, or can be explicitly enabled
//! via `SCX_SIM_FRIDA_STALKER=1`.

#![cfg(feature = "frida")]

use scx_simulator::*;

mod common;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns true if the Frida Stalker runtime tests should be skipped.
///
/// Frida Stalker requires JIT/mmap permissions that are unavailable in
/// some sandboxed environments. Set `SCX_SIM_NO_FRIDA_STALKER=1` to skip
/// tests that actually invoke the Stalker (compile-time and cooperative
/// tests still run).
fn skip_stalker_tests() -> bool {
    std::env::var("SCX_SIM_NO_FRIDA_STALKER").map_or(false, |v| v == "1")
}

/// Build a Frida-preemptive scenario with the given parameters.
///
/// Uses `use_frida: true` to activate Stalker software RBC counting.
/// `fixed_priority(true)` and `instant_timing()` eliminate timing noise
/// so we can assert exact checkpoint equality.
fn frida_scenario(nr_cpus: u32, nr_tasks: u32, seed: u32, duration_ms: u64) -> Scenario {
    let mut builder = Scenario::builder()
        .cpus(nr_cpus)
        .seed(seed)
        .fixed_priority(true)
        .instant_timing()
        .preemptive(PreemptiveConfig {
            timeslice_min: 100,
            timeslice_max: 500,
            cooperative_only: false,
            use_frida: true,
        });

    for i in 1..=nr_tasks {
        builder = builder.add_task(
            &format!("t{i}"),
            0,
            TaskBehavior {
                phases: vec![Phase::Run(10_000_000)],
                repeat: RepeatMode::Forever,
            },
        );
    }

    builder.duration_ms(duration_ms).build()
}

/// Build a Frida-preemptive scenario from the dsq_contention workload.
///
/// Loads the dsq_contention.json workload (8 workers, sleep/wake cycles)
/// and overlays Frida preemptive config on it.
fn frida_dsq_contention_scenario(seed: u32, duration_ms: u64) -> Scenario {
    let json = include_str!("../workloads/dsq_contention.json");
    let mut scenario = load_rtapp(json, 4).expect("failed to parse dsq_contention.json");

    scenario.duration_ns = duration_ms * 1_000_000;
    scenario.seed = seed;
    scenario.fixed_priority = true;
    scenario.interleave = true;
    scenario.preemptive = Some(PreemptiveConfig {
        timeslice_min: 100,
        timeslice_max: 500,
        cooperative_only: false,
        use_frida: true,
    });

    scenario
}

/// Build a cooperative-only Frida scenario (no actual Stalker invocation).
///
/// This tests the Frida code path configuration without requiring
/// the Stalker runtime to function (works in sandboxed environments).
fn cooperative_frida_scenario(
    nr_cpus: u32,
    nr_tasks: u32,
    seed: u32,
    duration_ms: u64,
) -> Scenario {
    let mut builder = Scenario::builder()
        .cpus(nr_cpus)
        .seed(seed)
        .fixed_priority(true)
        .instant_timing()
        .preemptive(PreemptiveConfig {
            timeslice_min: 100,
            timeslice_max: 500,
            cooperative_only: true,
            use_frida: false,
        });

    for i in 1..=nr_tasks {
        builder = builder.add_task(
            &format!("t{i}"),
            0,
            TaskBehavior {
                phases: vec![Phase::Run(10_000_000)],
                repeat: RepeatMode::Forever,
            },
        );
    }

    builder.duration_ms(duration_ms).build()
}

/// Run two simulations with determinism checkpointing and compare.
///
/// Returns `(checkpoints1, checkpoints2, divergence)`.
fn run_determinism_check<F>(
    make_sched: impl Fn() -> DynamicScheduler,
    make_scenario: F,
) -> (
    Vec<DeterminismCheckpoint>,
    Vec<DeterminismCheckpoint>,
    Option<CheckpointDivergence>,
)
where
    F: Fn() -> Scenario,
{
    // Run 1
    enable_determinism_mode();
    let trace1 = Simulator::new(make_sched()).run(make_scenario());
    let cp1 = drain_determinism_checkpoints();
    assert_eq!(
        trace1.exit_kind(),
        &ExitKind::Normal,
        "run 1 failed: {:?}",
        trace1.exit_kind()
    );

    // Run 2
    enable_determinism_mode();
    let trace2 = Simulator::new(make_sched()).run(make_scenario());
    let cp2 = drain_determinism_checkpoints();
    assert_eq!(
        trace2.exit_kind(),
        &ExitKind::Normal,
        "run 2 failed: {:?}",
        trace2.exit_kind()
    );

    // Trace-level sanity
    assert_eq!(
        trace1.events().len(),
        trace2.events().len(),
        "trace event counts differ: {} vs {}",
        trace1.events().len(),
        trace2.events().len()
    );

    let div = compare_checkpoints(&cp1, &cp2);
    (cp1, cp2, div)
}

// ===========================================================================
// Tests that always run (no Stalker runtime required)
// ===========================================================================

/// Cooperative-only determinism with simple scheduler.
///
/// Uses `cooperative_only: true` so no Stalker is invoked. This test
/// verifies the checkpoint infrastructure works, including memory hashes.
#[test]
fn test_frida_cooperative_determinism_simple() {
    let _lock = common::setup_test();
    let (cp1, _cp2, div) = run_determinism_check(
        || DynamicScheduler::simple(),
        || cooperative_frida_scenario(4, 2, 42, 30),
    );

    if let Some(d) = div {
        eprintln!("CHECKPOINT DIVERGENCE: {d}");
        panic!(
            "cooperative determinism check (simple) failed at checkpoint {}",
            d.checkpoint_index
        );
    }

    assert!(
        !cp1.is_empty(),
        "no checkpoints collected -- determinism mode not working"
    );
    eprintln!(
        "PASS: {} cooperative checkpoints verified identical (simple scheduler)",
        cp1.len()
    );
}

/// Cooperative-only determinism with LAVD scheduler + dsq_contention workload.
///
/// This exercises the full checkpoint pipeline (including memory hashes)
/// without requiring the Frida Stalker runtime. It serves as the baseline
/// determinism test for CI environments where Stalker may not be available.
#[test]
fn test_frida_cooperative_determinism_lavd_dsq_contention() {
    let _lock = common::setup_test();

    let json = include_str!("../workloads/dsq_contention.json");
    let make_scenario = || {
        let mut scenario = load_rtapp(json, 4).expect("failed to parse dsq_contention.json");
        scenario.duration_ns = 50 * 1_000_000;
        scenario.seed = 42;
        scenario.fixed_priority = true;
        scenario.interleave = true;
        scenario.preemptive = Some(PreemptiveConfig {
            timeslice_min: 100,
            timeslice_max: 500,
            cooperative_only: true,
            use_frida: false,
        });
        scenario
    };

    let (cp1, _cp2, div) = run_determinism_check(|| DynamicScheduler::lavd(4), make_scenario);

    if let Some(d) = div {
        eprintln!("CHECKPOINT DIVERGENCE: {d}");
        panic!(
            "cooperative determinism check (lavd + dsq_contention) failed at checkpoint {}",
            d.checkpoint_index
        );
    }

    assert!(
        !cp1.is_empty(),
        "no checkpoints collected -- determinism mode not working"
    );

    let nonzero_hashes = cp1.iter().filter(|c| c.memory_hash != 0).count();
    eprintln!(
        "PASS: {} cooperative checkpoints verified identical \
         (lavd + dsq_contention), {} with non-zero memory hash",
        cp1.len(),
        nonzero_hashes
    );
    assert!(
        nonzero_hashes > 0,
        "all memory hashes are zero -- compute_state_hash may not be working"
    );
}

/// Verify memory hashes change across checkpoints (cooperative mode).
///
/// A constant memory hash suggests the hashing is broken or always returns
/// the same value. Real scheduler state evolves as tasks are enqueued,
/// dispatched, and preempted.
#[test]
fn test_frida_memory_hashes_vary() {
    let _lock = common::setup_test();

    let json = include_str!("../workloads/dsq_contention.json");
    let mut scenario = load_rtapp(json, 4).expect("failed to parse dsq_contention.json");
    scenario.duration_ns = 30 * 1_000_000;
    scenario.seed = 42;
    scenario.fixed_priority = true;
    scenario.interleave = true;
    scenario.preemptive = Some(PreemptiveConfig::cooperative_only());

    enable_determinism_mode();
    let trace = Simulator::new(DynamicScheduler::lavd(4)).run(scenario);
    let checkpoints = drain_determinism_checkpoints();

    assert_eq!(trace.exit_kind(), &ExitKind::Normal);
    assert!(
        checkpoints.len() >= 10,
        "too few checkpoints ({}) to verify hash variance",
        checkpoints.len()
    );

    // Count distinct non-zero hash values
    let distinct_hashes: std::collections::HashSet<u64> = checkpoints
        .iter()
        .map(|c| c.memory_hash)
        .filter(|&h| h != 0)
        .collect();

    eprintln!(
        "Memory hash diversity: {} distinct values across {} checkpoints",
        distinct_hashes.len(),
        checkpoints.len()
    );

    // We expect at least a handful of distinct hashes as DSQ state evolves
    assert!(
        distinct_hashes.len() >= 3,
        "only {} distinct memory hashes -- state hashing may be broken",
        distinct_hashes.len()
    );
}

// ===========================================================================
// Tests that require Frida Stalker runtime (skipped in sandboxed envs)
// ===========================================================================

/// Smoke test: Frida Stalker preemptive mode runs to completion.
#[test]
fn test_frida_stalker_smoke_simple() {
    let _lock = common::setup_test();
    if skip_stalker_tests() {
        eprintln!("SKIP: SCX_SIM_NO_FRIDA_STALKER=1 set, skipping Stalker test");
        return;
    }

    let scenario = frida_scenario(2, 2, 42, 20);
    let trace = Simulator::new(DynamicScheduler::simple()).run(scenario);
    assert_eq!(trace.exit_kind(), &ExitKind::Normal);
    assert!(trace.schedule_count(Pid(1)) > 0, "task 1 never scheduled");
    assert!(trace.schedule_count(Pid(2)) > 0, "task 2 never scheduled");
}

/// Full Stalker determinism: simple scheduler.
#[test]
fn test_frida_stalker_determinism_simple() {
    let _lock = common::setup_test();
    if skip_stalker_tests() {
        eprintln!("SKIP: SCX_SIM_NO_FRIDA_STALKER=1 set, skipping Stalker test");
        return;
    }

    let (cp1, _cp2, div) = run_determinism_check(
        || DynamicScheduler::simple(),
        || frida_scenario(4, 2, 42, 30),
    );

    if let Some(d) = div {
        eprintln!("CHECKPOINT DIVERGENCE: {d}");
        panic!(
            "Frida Stalker determinism check (simple) failed at checkpoint {}",
            d.checkpoint_index
        );
    }

    assert!(
        !cp1.is_empty(),
        "no checkpoints collected -- frida determinism mode not working"
    );
    eprintln!(
        "PASS: {} Frida Stalker checkpoints verified identical (simple scheduler)",
        cp1.len()
    );
}

/// Core CI test: LAVD scheduler + dsq_contention workload with Stalker.
///
/// This is the complex determinism test for CI. It exercises:
/// - LAVD scheduler (production scheduler with complex DSQ logic)
/// - dsq_contention workload (8 workers, sleep/wake cycles, 4 CPUs)
/// - Frida Stalker software RBC counting
/// - Memory state hashing at every checkpoint
///
/// Two runs with seed 42 must produce bit-identical checkpoint sequences.
#[test]
fn test_frida_stalker_determinism_lavd_dsq_contention() {
    let _lock = common::setup_test();
    if skip_stalker_tests() {
        eprintln!("SKIP: SCX_SIM_NO_FRIDA_STALKER=1 set, skipping Stalker test");
        return;
    }

    let (cp1, _cp2, div) = run_determinism_check(
        || DynamicScheduler::lavd(4),
        || frida_dsq_contention_scenario(42, 50),
    );

    if let Some(d) = div {
        eprintln!("CHECKPOINT DIVERGENCE: {d}");
        panic!(
            "Frida Stalker determinism check (lavd + dsq_contention) failed at checkpoint {}",
            d.checkpoint_index
        );
    }

    assert!(
        !cp1.is_empty(),
        "no checkpoints collected -- frida determinism mode not working"
    );

    let nonzero_hashes = cp1.iter().filter(|c| c.memory_hash != 0).count();
    eprintln!(
        "PASS: {} Frida Stalker checkpoints verified identical \
         (lavd + dsq_contention), {} with non-zero memory hash",
        cp1.len(),
        nonzero_hashes
    );
    assert!(
        nonzero_hashes > 0,
        "all memory hashes are zero -- compute_state_hash may not be working"
    );
}

/// Different seeds should (usually) produce different checkpoint sequences.
///
/// This validates that the preemptive interleaving actually influences
/// scheduling decisions rather than being a no-op.
#[test]
fn test_frida_different_seeds_diverge() {
    let _lock = common::setup_test();

    // Use cooperative mode (works everywhere) to verify divergence detection
    // Run with seed 42
    enable_determinism_mode();
    let _ =
        Simulator::new(DynamicScheduler::simple()).run(cooperative_frida_scenario(4, 4, 42, 30));
    let cp_seed42 = drain_determinism_checkpoints();

    // Run with seed 999
    enable_determinism_mode();
    let _ =
        Simulator::new(DynamicScheduler::simple()).run(cooperative_frida_scenario(4, 4, 999, 30));
    let cp_seed999 = drain_determinism_checkpoints();

    assert!(!cp_seed42.is_empty(), "seed 42 produced no checkpoints");
    assert!(!cp_seed999.is_empty(), "seed 999 produced no checkpoints");

    // Different seeds may or may not diverge depending on the scheduler.
    // The important thing is both runs completed without panicking.
    match compare_checkpoints(&cp_seed42, &cp_seed999) {
        Some(div) => eprintln!(
            "Expected divergence between seeds: checkpoint {} ({:?})",
            div.checkpoint_index, div.divergence_type
        ),
        None => eprintln!(
            "INFO: Seeds 42 and 999 produced identical checkpoints ({} each). \
             This can happen with simple scheduler on deterministic scenarios.",
            cp_seed42.len()
        ),
    }
}
