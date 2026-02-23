# Survey: Controlled Concurrency & Deterministic Execution Systems

This document surveys systems that execute native (C/C++) code with controlled
concurrency — deterministic replay, concurrency fuzzing, and progress-counting
instrumentation. The goal is to inform our choice of preemption mechanism for
the scx_simulator.

## Our Use Case

We load compiled C scheduler code as `.so` files via `dlopen` and run them with
controlled, deterministic interleaving. We need a "progress counter" (e.g.,
retired branch conditionals) to inject preemption points at deterministic
locations. Current approaches under evaluation:

| Approach | Overhead | Debugger-compatible | Deterministic |
|----------|----------|---------------------|---------------|
| Hardware PMU (perf_events) | ~0% | Yes | No (skid) |
| Frida Stalker (DBI) | 10-100x | No (breaks gdb/rr) | Yes |
| SanCov (compiler) | 20-50% | Yes | Yes |

---

## 1. Deterministic Replay Systems

### rr (Mozilla / Robert O'Callahan)

- **URL**: https://rr-project.org / https://github.com/rr-debugger/rr
- **Status**: Actively maintained (2024-2026 commits)
- **License**: MIT + BSD
- **Approach**: Records non-deterministic inputs (syscalls, signals, scheduling
  decisions, RDTSC) during a recording pass, then replays them deterministically.
  Uses hardware performance counters (specifically `PERF_COUNT_HW_BRANCH_INSTRUCTIONS`
  or retired conditional branches) to count progress and inject preemption at
  exact instruction boundaries.
- **Key insight for us**: rr uses the same "retired branch conditional" counter
  we're trying to replicate. rr relies on the counter being precise — it fails
  on CPUs with excessive PMU skid. rr's `--chaos` mode randomizes scheduling to
  find concurrency bugs, similar to our preemptive interleaving.
- **Rust bindings**: None. rr is a standalone tool, not a library.
- **Applicability**: rr is a full-system record/replay tool, not embeddable. But
  its use of RBC counters validates our approach. rr's source code
  (`PerfCounters.cc`) documents which CPUs have reliable RBC counting.
- **Conflict with Frida**: rr cannot record a Frida-Stalker-instrumented process
  because Stalker's code translation changes hardware branch counts.

### Hermit (Meta)

- **URL**: https://github.com/facebookexperimental/hermit
- **Status**: Open-sourced 2022, limited maintenance since ~2023
- **License**: MIT
- **Approach**: Uses Linux `ptrace` + `seccomp` to intercept all syscalls and
  make them deterministic. Runs inside a "detcore" container that virtualizes
  time, PIDs, randomness, and scheduling. Uses hardware performance counters
  (like rr) for preemption counting.
- **Key insight**: Hermit's `detcore` scheduler uses RBC hardware counters with
  a configurable "preemption timeout" (number of RBCs before forced preemption).
  This is architecturally identical to what we're doing.
- **Rust**: Written entirely in Rust. The `detcore` crate could theoretically
  be reused, but it's tightly coupled to the ptrace/container architecture.
- **Applicability**: Hermit targets whole-program determinism via containers.
  Not directly usable for instrumenting a single `.so`, but its scheduling
  algorithm (RBC-counted preemption with PRNG-driven timeslices) is very
  relevant — it's essentially the same design as our PreemptRing.

### CHESS (Microsoft Research)

- **URL**: https://www.microsoft.com/en-us/research/project/chess-systematic-testing/
- **Status**: Unmaintained (last active ~2010-2012)
- **License**: Microsoft Research License
- **Approach**: Systematic concurrency testing via controlled scheduling.
  Intercepts synchronization primitives (locks, condition variables) and
  explores all possible interleavings up to a bound. Uses a "fair bounded
  scheduling" algorithm.
- **Key insight**: CHESS proves that many concurrency bugs can be found with
  a small number of preemption points (the "small scope hypothesis"). Our
  approach of random preemption at branch boundaries is a probabilistic
  version of this.
- **Applicability**: Windows-only, C#/.NET focused. Not directly usable.

### DThreads

- **URL**: https://github.com/emeryberger/DThreads
- **Status**: Research prototype, unmaintained (~2011)
- **License**: GPL
- **Approach**: Replaces pthreads with a deterministic threading library.
  Uses process isolation (each thread runs in a separate process with COW
  memory via `mmap`). Commits are serialized in a deterministic order.
- **Key insight**: Achieves determinism by making threads run in isolation
  and synchronizing at "commit points." Overhead is high for sharing-heavy
  workloads.
- **Applicability**: Unmaintained, Linux-only. The process-isolation approach
  is interesting but doesn't match our `.so` instrumentation use case.

### Kendo

- **Status**: Research paper (2009), no public implementation
- **Approach**: Deterministic multithreading via "deterministic logical time."
  Each thread has a logical clock that advances with retired instructions.
  Lock acquisitions are ordered by logical time.
- **Key insight**: Uses retired instruction counting (similar to RBC) as the
  progress metric. The logical-time ordering of lock acquisitions is
  conceptually similar to our token-ring approach.

### CoreDet

- **Status**: Research paper (2010), no maintained implementation
- **Approach**: Compiler-based deterministic execution. Instruments memory
  accesses at compile time and enforces a deterministic commit order.
- **Key insight**: Shows that compiler instrumentation can achieve determinism,
  but with 1.2-6x overhead for the instrumentation alone.

---

## 2. Concurrency Fuzzing / Testing Frameworks

### Loom (Tokio Project)

- **URL**: https://github.com/tokio-rs/loom
- **Status**: Actively maintained (2024-2026)
- **License**: MIT
- **Approach**: Model-checks Rust concurrent code by exploring all possible
  interleavings of atomic operations, thread spawns, and synchronization.
  Replaces `std::sync` and `std::thread` with instrumented versions.
- **Rust**: Pure Rust, first-class Rust API
- **Limitation**: Only works with Rust code that uses Loom's replacement
  types. Cannot instrument arbitrary C code or `.so` files.
- **Applicability**: Not applicable to our C `.so` use case, but its API
  design (replacement synchronization primitives) is a good reference.

### Shuttle (AWS)

- **URL**: https://github.com/awslabs/shuttle
- **Status**: Actively maintained (2024-2025)
- **License**: Apache 2.0
- **Approach**: Similar to Loom but uses randomized scheduling (PCT algorithm)
  instead of exhaustive exploration. Provides `shuttle::sync` replacements
  for `std::sync`.
- **Rust**: Pure Rust
- **Limitation**: Same as Loom — only instruments Rust code using Shuttle's
  types.
- **Applicability**: The PCT scheduling algorithm is relevant. Shuttle's
  approach of random-priority-based preemption is similar to our PRNG-driven
  timeslice approach.

### ThreadSanitizer (TSan)

- **URL**: Part of LLVM/Clang
- **Status**: Actively maintained (part of LLVM)
- **License**: Apache 2.0 (LLVM)
- **Approach**: Compile-time instrumentation of memory accesses + runtime
  shadow memory tracking. Detects data races by tracking happens-before
  relationships.
- **Key insight**: TSan instruments memory accesses at compile time (similar
  to SanCov instrumenting edges). It does NOT control scheduling — it
  observes whatever interleavings occur naturally.
- **Applicability**: TSan's compile-time instrumentation approach validates
  that compiler-inserted callbacks at memory accesses/branches are practical
  with acceptable overhead.

### PCT (Probabilistic Concurrency Testing)

- **Paper**: Burckhardt et al., "A Randomized Scheduler with Probabilistic
  Guarantees of Finding Bugs" (ASPLOS 2010)
- **Approach**: Assigns random priorities to threads and a small number of
  random "priority change points." Provides probabilistic guarantees of
  finding bugs with few preemption points.
- **Key insight**: PCT proves that O(n*d) random runs (where n = threads,
  d = bug depth) are sufficient to find most concurrency bugs with high
  probability. Our random-timeslice preemption is a continuous version
  of PCT.
- **Implementations**: Shuttle (Rust), CHESS (C#), various research tools.

---

## 3. Binary Instrumentation for Progress Counting

### Intel PIN

- **URL**: https://www.intel.com/content/www/us/en/developer/articles/tool/pin-a-dynamic-binary-instrumentation-tool.html
- **Status**: Actively maintained by Intel (2024-2026)
- **License**: Proprietary (free for non-commercial use)
- **Approach**: DBI framework similar to Frida. JIT-compiles instrumented
  code. Can count instructions, branches, memory accesses at arbitrary
  granularity.
- **Overhead**: 2-10x for instruction counting
- **Rust bindings**: None official. Community `pin-rs` crate exists but
  unmaintained.
- **Debugger compatibility**: Same issues as Frida — code runs from JIT
  cache, breaking breakpoints and source-level debugging.

### DynamoRIO

- **URL**: https://dynamorio.org / https://github.com/DynamoRIO/dynamorio
- **Status**: Actively maintained (2024-2026)
- **License**: BSD
- **Approach**: DBI framework. Includes `drcov` for code coverage and
  `drcount` for instruction counting. Client API for custom instrumentation.
- **Overhead**: 2-5x for basic block counting
- **Rust bindings**: `dynamorio-rs` crate exists but is very early-stage.
- **Debugger compatibility**: Same DBI issues as Frida/PIN.
- **Key feature**: DynamoRIO's `dr_insert_clean_call` is similar to Frida's
  `put_callout`. `dr_insert_cbr_instrumentation` specifically targets
  conditional branches.

### QEMU User-Mode Emulation

- **URL**: https://www.qemu.org
- **Status**: Actively maintained
- **License**: GPL v2
- **Approach**: Full CPU emulation via dynamic translation. Can count
  instructions/branches via TCG (Tiny Code Generator) plugins.
- **Overhead**: 5-20x
- **Rust bindings**: None for user-mode emulation API.
- **Key insight**: QEMU's TCG plugin API (`qemu_plugin_register_vcpu_tb_trans_cb`)
  can insert callbacks at translation block boundaries — similar to Stalker's
  transformer. AFL uses QEMU user-mode for instrumenting closed-source
  binaries.
- **Applicability**: Too heavy for our use case. We only need to instrument
  one `.so`, not emulate the entire process.

### Valgrind

- **URL**: https://valgrind.org
- **Status**: Actively maintained (2024-2026)
- **License**: GPL v2
- **Approach**: Translates all code to Valgrind IR (VEX), instruments it,
  then JIT-compiles back to native code. Callgrind counts instructions
  and branches precisely.
- **Overhead**: 10-50x (Memcheck), 20-100x (Callgrind)
- **Conflict with Frida**: Cannot coexist with Frida Stalker (both are DBI).
- **Applicability**: Too slow, and the full-process instrumentation is
  overkill for our single-`.so` use case.

---

## 4. Compiler-Based Progress Counting

### SanitizerCoverage (SanCov)

- **URL**: https://clang.llvm.org/docs/SanitizerCoverage.html
- **Status**: Part of LLVM, actively maintained
- **License**: Apache 2.0 (LLVM)
- **Approach**: `-fsanitize-coverage=trace-pc-guard` inserts a callback at
  every control-flow edge. The callback receives a "guard" pointer unique
  to each edge.
- **Overhead**: 20-50% (function call per edge)
- **Granularity**: Per-edge (covers Jcc, switch cases, function entries).
  Close to per-Jcc but includes unconditional edges too.
- **Debugger compatibility**: Full — instrumented code is normal native code
  with debug symbols intact.
- **Key advantage**: Zero build complexity (one compiler flag). The guard
  callback can be implemented in Rust via `#[no_mangle] extern "C"`.
- **Applicability**: Best fit for our use case. We control the scheduler
  build pipeline and can add the flag. The callback-per-edge approach
  maps directly onto our counter-decrement pattern.

### AFL-Style Instrumentation

- **URL**: https://github.com/AFLplusplus/AFLplusplus
- **Status**: Actively maintained (AFL++)
- **License**: Apache 2.0
- **Approach**: `afl-clang-fast` uses SanCov or a custom LLVM pass to
  instrument edges. Each edge increments a shared-memory coverage map
  entry: `shared_mem[cur_loc ^ prev_loc]++`.
- **Overhead**: ~5-15% (shared memory write is fast)
- **Key insight**: AFL's edge instrumentation is extremely lightweight
  because it uses a single indexed memory write instead of a function
  call. We could use a similar approach: `counter--; if (counter == 0)
  call_yield()` with the branch being highly predictable (almost never
  taken).
- **Applicability**: The AFL LLVM pass could be adapted for our use case.
  The `afl-compiler-rt` runtime is small and well-understood.

### Custom LLVM MachineFunctionPass

- **Approach**: Write a custom LLVM backend pass that inserts `dec + jz`
  inline at every Jcc instruction.
- **Overhead**: <5% (inline counter decrement, branch almost never taken)
- **Complexity**: 500-1000 lines of C++, needs updating for LLVM version
  changes.
- **Debugger compatibility**: Full — the inserted instructions are just
  normal x86 code with DWARF info.
- **Applicability**: Lowest overhead option, but highest implementation
  and maintenance cost. Good as a Phase 2 upgrade from SanCov if
  overhead matters.

---

## 5. Summary Comparison

| System | Type | Maintained | Rust API | Works with `.so` | Overhead | Debugger OK | Deterministic | Applicability |
|--------|------|-----------|----------|------------------|----------|-------------|---------------|---------------|
| **rr** | Replay | Yes | No | N/A (whole-process) | ~1.2x record | Yes (IS debugger) | Yes | Reference design |
| **Hermit** | Container | Limited | Yes (is Rust) | No (whole-container) | ~1.5-3x | Via rr | Yes | Architecture reference |
| **Loom** | Model checker | Yes | Yes | No (Rust only) | N/A | Yes | Yes | Not applicable |
| **Shuttle** | Fuzzer | Yes | Yes | No (Rust only) | N/A | Yes | Yes | PCT algorithm reference |
| **Frida Stalker** | DBI | Yes | Yes (`frida-gum`) | Yes | 10-100x | No | Yes | Current approach |
| **Intel PIN** | DBI | Yes | No | Yes | 2-10x | No | Yes | Alternative DBI |
| **DynamoRIO** | DBI | Yes | Minimal | Yes | 2-5x | No | Yes | Alternative DBI |
| **QEMU user** | Emulation | Yes | No | Yes | 5-20x | No | Yes | Too heavy |
| **Valgrind** | DBI | Yes | No | Yes | 20-100x | No | Yes | Too slow |
| **SanCov** | Compiler | Yes | N/A (flag) | Yes | 20-50% | **Yes** | Yes | **Recommended** |
| **AFL pass** | Compiler | Yes | No | Yes | 5-15% | Yes | Yes | Possible upgrade |
| **Custom LLVM** | Compiler | DIY | N/A | Yes | <5% | Yes | Yes | Future upgrade |
| **TSan** | Compiler | Yes | N/A (flag) | Yes | 5-15x | Yes | No (observer) | Not applicable |

## 6. Recommendations

### Primary: SanCov (`-fsanitize-coverage=trace-pc-guard`)

Best balance of simplicity, overhead, and debugger compatibility. One
compiler flag, full debug symbol preservation, compatible with gdb/lldb/rr.
Edge-level granularity is close enough to per-Jcc for our interleaving
use case.

### Upgrade Path: AFL-style inline instrumentation

If SanCov's per-edge function call overhead (~20-50%) is too high, switch
to an AFL-style approach where the counter decrement is inlined as a
direct memory operation instead of a function call. This brings overhead
down to ~5-15%.

### Future: Custom LLVM MachineFunctionPass

For maximum performance (<5% overhead), a custom backend pass that
inserts `dec [counter]; jz yield_func` at every Jcc. Only worth the
LLVM maintenance burden if the simulator is used in performance-critical
CI pipelines.

### Keep Frida as Fallback

Frida Stalker remains useful for instrumenting scheduler `.so` files
that we don't control the compilation of (e.g., pre-compiled third-party
schedulers). It works without recompilation, at the cost of debugger
compatibility and higher overhead.

---

## 7. Key Takeaway from rr and Hermit

Both rr and Hermit validate our fundamental approach:

1. **RBC counting** is the right progress metric for deterministic preemption
2. **PRNG-driven timeslices** provide good bug-finding coverage (PCT theory)
3. **Token-ring scheduling** (one thread active at a time) is the right
   concurrency model for deterministic interleaving

The main question is implementation: hardware counters (fast but skid),
DBI (precise but breaks tools), or compiler instrumentation (precise,
debugger-compatible, moderate overhead). The ecosystem strongly favors
compiler instrumentation for our use case where we control the build.
