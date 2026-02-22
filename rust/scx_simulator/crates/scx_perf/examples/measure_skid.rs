//! Empirical PMU skid measurement tool.
//!
//! Measures the difference between the *requested* overflow point and the
//! *actual* point at which the overflow signal is delivered. This difference
//! is called "skid" and is caused by CPU pipeline depth, interrupt delivery
//! latency, and microarchitecture.
//!
//! Supports two PMU event types:
//! - **rbc**: Retired Branch Conditionals (vendor-specific raw event)
//! - **insn**: Hardware Instructions Retired (PERF_COUNT_HW_INSTRUCTIONS)
//!
//! Usage:
//!   cargo run --release --example measure_skid
//!   cargo run --release --example measure_skid -- --event rbc
//!   cargo run --release --example measure_skid -- --event insn
//!   cargo run --release --example measure_skid -- --event all   (default)
//!
//! The tool tests several target periods and for each one reports the
//! distribution of skid values.

use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};

use scx_perf::{PmuConfig, PmuEvent, RbcTimer, PERF_IOC_DISABLE};

/// Number of trials per target period.
const TRIALS: usize = 500;

/// Target periods to test.
const TARGETS: &[u64] = &[1, 5, 10, 100, 1000, 2000];

// --- Globals for async-signal-safe communication with signal handler ---

/// The perf fd of the active timer (set before each trial).
static TIMER_FD: AtomicI32 = AtomicI32::new(-1);

/// The counter value read inside the signal handler.
static ACTUAL_COUNT: AtomicU64 = AtomicU64::new(0);

/// Flag: signal handler has fired for the current trial.
static SIGNAL_FIRED: AtomicU64 = AtomicU64::new(0);

/// Which events to measure, parsed from CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EventFilter {
    Rbc,
    Insn,
    All,
}

/// Signal handler -- reads counter and disables timer immediately.
///
/// SAFETY: Only uses async-signal-safe operations (read, ioctl, atomic store).
extern "C" fn handler(_signo: libc::c_int, _info: *mut libc::siginfo_t, _ctx: *mut libc::c_void) {
    let fd = TIMER_FD.load(Ordering::Relaxed);
    if fd < 0 {
        return;
    }

    // Disable timer first to prevent re-fire.
    unsafe {
        libc::ioctl(fd, PERF_IOC_DISABLE, 0usize);
    }

    // Read counter value -- this is the actual count at signal delivery.
    let mut count: u64 = 0;
    unsafe {
        libc::read(fd, &mut count as *mut u64 as *mut libc::c_void, 8);
    }

    ACTUAL_COUNT.store(count, Ordering::SeqCst);
    SIGNAL_FIRED.store(1, Ordering::SeqCst);
}

/// Generate conditional branches in a tight loop.
///
/// Each iteration has 2 conditional branches (the `if` conditions), so
/// `n` iterations produce roughly `2*n` conditional branches (plus loop
/// control). This is intentionally simple and predictable.
#[inline(never)]
fn branch_workload(n: u64) -> u64 {
    let mut sum = 0u64;
    for i in 0..n {
        if i % 2 == 0 {
            sum = sum.wrapping_add(i);
        }
        if i % 3 == 0 {
            sum = sum.wrapping_add(i >> 1);
        }
    }
    std::hint::black_box(sum)
}

/// Generate retired instructions via arithmetic and memory work.
///
/// Avoids conditional branches so the instruction count is predictable.
/// Each iteration does several arithmetic ops and a volatile-style memory
/// write, producing a consistent number of retired instructions per call.
#[inline(never)]
fn instruction_workload(n: u64) -> u64 {
    let mut a: u64 = 1;
    let mut b: u64 = 2;
    let mut c: u64 = 3;
    let mut d: u64 = 4;
    let mut i: u64 = 0;
    while i < n {
        // Arithmetic-heavy: each iteration retires multiple instructions
        // without conditional branches (the while condition is the only one).
        a = a.wrapping_mul(6364136223846793005).wrapping_add(1);
        b = b.wrapping_add(a >> 32);
        c ^= b.wrapping_mul(a);
        d = d.wrapping_add(c >> 16);
        a ^= d;
        i += 1;
    }
    std::hint::black_box(a ^ b ^ c ^ d)
}

struct SkidStats {
    target: u64,
    skids: Vec<i64>,
}

impl SkidStats {
    fn mean(&self) -> f64 {
        let sum: i64 = self.skids.iter().sum();
        sum as f64 / self.skids.len() as f64
    }

    fn stddev(&self) -> f64 {
        let m = self.mean();
        let variance: f64 = self
            .skids
            .iter()
            .map(|&s| {
                let d = s as f64 - m;
                d * d
            })
            .sum::<f64>()
            / self.skids.len() as f64;
        variance.sqrt()
    }

    fn min(&self) -> i64 {
        *self.skids.iter().min().unwrap_or(&0)
    }

    fn max(&self) -> i64 {
        *self.skids.iter().max().unwrap_or(&0)
    }

    fn median(&self) -> i64 {
        let mut sorted = self.skids.clone();
        sorted.sort();
        sorted[sorted.len() / 2]
    }

    fn p95(&self) -> i64 {
        let mut sorted = self.skids.clone();
        sorted.sort();
        sorted[(sorted.len() as f64 * 0.95) as usize]
    }

    fn p99(&self) -> i64 {
        let mut sorted = self.skids.clone();
        sorted.sort();
        sorted[(sorted.len() as f64 * 0.99) as usize]
    }
}

fn install_signal_handler() {
    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = handler as *const () as libc::sighandler_t;
        sa.sa_flags = libc::SA_SIGINFO | libc::SA_RESTART;
        libc::sigemptyset(&mut sa.sa_mask);
        let ret = libc::sigaction(libc::SIGSTKFLT, &sa, std::ptr::null_mut());
        assert_eq!(ret, 0, "failed to install SIGSTKFLT handler");
    }
}

fn measure_skid_for_target(config: &PmuConfig, event: PmuEvent, target: u64) -> SkidStats {
    let mut skids = Vec::with_capacity(TRIALS);
    let mut timeouts = 0usize;

    for _ in 0..TRIALS {
        // Create a fresh timer for each trial to avoid counter accumulation issues.
        let timer = RbcTimer::new_event(config, event, target).expect("failed to create PMU timer");

        let tid = unsafe { libc::syscall(libc::SYS_gettid) } as libc::pid_t;
        timer
            .set_signal_delivery(tid, libc::SIGSTKFLT)
            .expect("set_signal_delivery");

        TIMER_FD.store(timer.raw_fd(), Ordering::SeqCst);
        ACTUAL_COUNT.store(0, Ordering::SeqCst);
        SIGNAL_FIRED.store(0, Ordering::SeqCst);

        timer.reset().expect("reset");
        timer.enable().expect("enable");

        // Run enough work to guarantee overflow even with large skid.
        let iterations = (target + 50_000).max(100_000);
        match event {
            PmuEvent::RetiredBranchConditional => branch_workload(iterations),
            PmuEvent::InstructionsRetired => instruction_workload(iterations),
        };

        timer.disable().expect("disable");

        if SIGNAL_FIRED.load(Ordering::SeqCst) == 1 {
            let actual = ACTUAL_COUNT.load(Ordering::SeqCst);
            let skid = actual as i64 - target as i64;
            skids.push(skid);
        } else {
            timeouts += 1;
        }

        // Drop timer (closes fd).
        TIMER_FD.store(-1, Ordering::SeqCst);
    }

    if timeouts > 0 {
        eprintln!(
            "  WARNING: {timeouts}/{TRIALS} trials for target={target} did not fire a signal"
        );
    }

    SkidStats { target, skids }
}

fn print_stats_table(all_stats: &[SkidStats]) {
    println!(
        "{:>8} {:>8} {:>10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Target", "N", "Mean", "StdDev", "Min", "Median", "Max", "P95", "P99"
    );
    println!("{}", "-".repeat(86));

    for stats in all_stats {
        if stats.skids.is_empty() {
            println!(
                "{:>8} {:>8} {:>10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
                stats.target, 0, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"
            );
        } else {
            println!(
                "{:>8} {:>8} {:>10.1} {:>8.1} {:>8} {:>8} {:>8} {:>8} {:>8}",
                stats.target,
                stats.skids.len(),
                stats.mean(),
                stats.stddev(),
                stats.min(),
                stats.median(),
                stats.max(),
                stats.p95(),
                stats.p99(),
            );
        }
    }
}

fn event_label(event: PmuEvent) -> &'static str {
    match event {
        PmuEvent::RetiredBranchConditional => "Retired Branch Conditionals (RBC)",
        PmuEvent::InstructionsRetired => "Instructions Retired (INSN)",
    }
}

fn run_measurement(config: &PmuConfig, event: PmuEvent) {
    println!("--- {} ---", event_label(event));
    println!();

    let all_stats: Vec<SkidStats> = TARGETS
        .iter()
        .map(|&target| measure_skid_for_target(config, event, target))
        .collect();

    print_stats_table(&all_stats);
    println!();
}

fn parse_event_filter() -> EventFilter {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--event" {
            if i + 1 >= args.len() {
                eprintln!("error: --event requires a value (rbc, insn, or all)");
                std::process::exit(1);
            }
            return match args[i + 1].as_str() {
                "rbc" => EventFilter::Rbc,
                "insn" => EventFilter::Insn,
                "all" => EventFilter::All,
                other => {
                    eprintln!("error: unknown event type '{other}' (expected rbc, insn, or all)");
                    std::process::exit(1);
                }
            };
        }
        i += 1;
    }
    EventFilter::All
}

fn main() {
    let filter = parse_event_filter();

    // Detect CPU info for the header.
    let info = detect_cpu_info();
    println!("PMU Skid Measurement");
    println!("====================");
    println!("CPU: {info}");
    println!("Trials per target: {TRIALS}");
    println!("Event filter: {filter:?}");
    println!();

    install_signal_handler();

    // Pin to a single CPU to avoid cross-core migration noise.
    pin_to_cpu(0);

    let config = PmuConfig::detect().expect("CPU not supported for PMU counting");

    let events_to_measure: &[PmuEvent] = match filter {
        EventFilter::Rbc => &[PmuEvent::RetiredBranchConditional],
        EventFilter::Insn => &[PmuEvent::InstructionsRetired],
        EventFilter::All => &[
            PmuEvent::RetiredBranchConditional,
            PmuEvent::InstructionsRetired,
        ],
    };

    for &event in events_to_measure {
        run_measurement(&config, event);
    }

    println!("Skid = actual_count_at_signal - target_period");
    println!("Positive skid means the signal arrived AFTER the target (late delivery).");
    println!("Negative skid would mean the signal arrived BEFORE the target (should not happen).");
}

/// Pin the current thread to a specific CPU.
fn pin_to_cpu(cpu: usize) {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu, &mut set);
        let ret = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
        if ret != 0 {
            eprintln!(
                "WARNING: failed to pin to CPU {cpu}: {}",
                std::io::Error::last_os_error()
            );
        } else {
            println!("Pinned to CPU {cpu}");
        }
    }
}

/// Get a human-readable CPU identification string.
fn detect_cpu_info() -> String {
    // Read from /proc/cpuinfo for the model name.
    if let Ok(contents) = std::fs::read_to_string("/proc/cpuinfo") {
        for line in contents.lines() {
            if line.starts_with("model name") {
                if let Some(name) = line.split(':').nth(1) {
                    return name.trim().to_string();
                }
            }
        }
    }
    "unknown".to_string()
}
