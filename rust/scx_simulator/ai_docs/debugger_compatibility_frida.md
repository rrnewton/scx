# Debugger and Analysis Tool Compatibility with Frida Stalker DBI

This document analyzes the compatibility of common debugging, profiling, and
analysis tools with our Frida Stalker-based dynamic binary instrumentation
(DBI) approach to software RBC (Retired Branch Conditional) counting.

## 1. How Frida Stalker Works (Context)

Frida Stalker is a **dynamic code translator**. Understanding this architecture
is essential to predicting tool compatibility.

### Core mechanism

1. **Code cache / "slab"**: When `follow_me()` is called, Stalker takes over
   the calling thread. Every basic block that the thread is about to execute is
   first **copied** to a private code cache (the "slab"), which is an
   `mmap`-allocated region with `PROT_READ | PROT_WRITE | PROT_EXEC`
   permissions. The original code is **never executed** while Stalker is
   active.

2. **Translation**: During the copy, Stalker applies a user-supplied
   `Transformer` callback. In our case (`stalker.rs`), the transformer
   inspects each instruction for conditional branch opcodes (Jcc family:
   `0x70-0x7F`, `0x0F 0x80-0x8F`, and LOOPcc `0xE0-0xE3`) and inserts a
   callout before each one. The callout decrements a thread-local counter
   and sets `YIELD_PENDING` when it expires.

3. **Execution redirection**: After translation, the thread's instruction
   pointer (RIP) is redirected to the translated copy in the slab. All
   subsequent execution occurs in the slab, not in the original `.so` or
   in the Rust binary.

4. **Trust threshold**: Stalker has a "trust threshold" that controls
   re-translation. Once a block has been executed enough times, Stalker
   "trusts" it and backpatches direct jumps into the code cache to avoid
   re-translation overhead. Setting the trust threshold to `-1` forces
   re-translation every time (useful for self-modifying code, but very slow).

5. **Scope**: `follow_me()` instruments **the calling thread only**. In our
   architecture, each worker thread calls `follow_me()` before entering
   scheduler C code and `unfollow_me()` after it returns. Between these
   calls, ALL code on that thread runs translated — including the scheduler
   `.so`, libc, Rust runtime, etc. Our transformer only inserts callouts
   for addresses within the `.so`'s text segment (`TextRange`), but all
   code still runs from the slab.

6. **No ptrace**: Crucially, Frida's in-process mode (which we use via the
   `frida-gum` crate linked as a static library) does **not** use `ptrace`.
   It runs entirely in-process. Remote Frida injection on Linux does use
   `ptrace` briefly to inject a bootstrap, but our embedded/gadget mode
   avoids this entirely. This distinction is critical for debugger
   coexistence.

### Key implications for tool compatibility

- The instruction pointer points to slab addresses, not original code
  addresses.
- Software breakpoints (INT3 / `0xCC`) placed on original code will never
  be hit because that code is not executing.
- Stack frames and return addresses on the stack may contain a mix of slab
  addresses and original addresses (for code outside Stalker's scope).
- DWARF debug info and `.eh_frame` unwind tables reference original code
  addresses, not slab addresses.
- Stalker uses `mmap`/`mprotect` to manage the code cache (RWX pages).
- Stalker intercepts certain signals internally and uses them for its own
  control flow.

---

## 2. Per-Tool Compatibility Analysis

### 2.1 GDB

| Aspect | Compatibility | Notes |
|--------|--------------|-------|
| Attach to process | **Yes** | GDB uses `ptrace(PTRACE_ATTACH)`. Since our embedded Frida does not use `ptrace`, there is no "single tracer" conflict. GDB can attach. |
| Breakpoints in `.so` code | **No** | GDB sets software breakpoints by writing `INT3` (`0xCC`) into the original code. But Stalker has already copied the code to the slab — the original is not executing. The `INT3` is never hit. Hardware breakpoints (DR0-DR3) target *virtual addresses*, so they would also miss since the executing addresses are in the slab, not at the original location. |
| Breakpoints in Rust code | **Partial** | If the Rust code runs between `unfollow_me()` and the next `follow_me()` (e.g., in the engine, test harness, or any code outside the Stalker bracket), breakpoints work normally. While Stalker is active, even Rust code runs from the slab (Stalker translates everything on the followed thread). |
| Backtrace / stack unwinding | **Degraded** | DWARF `.eh_frame` and `.debug_frame` unwind information references original code addresses. When the IP points to slab addresses, the unwinder cannot find the unwind rules. Backtraces will show raw hex slab addresses for Stalker-translated frames. Frames from before `follow_me()` (which are on the stack but not currently executing) may still resolve correctly. Frida provides `gum_stalker_activate_experimental_unwind_support()` which improves this, but it is marked "experimental" and may not produce fully reliable backtraces. |
| Watchpoints | **Mostly yes** | Hardware data watchpoints (DR0-DR3 in data mode) trigger on memory *access addresses*, not instruction addresses. If the translated code in the slab accesses the same data address, the watchpoint fires. However, GDB may be confused by the instruction pointer being in the slab when it tries to display the triggering instruction. |
| Single-stepping | **Confusing** | GDB single-step (`stepi`) uses `PTRACE_SINGLESTEP` which sets the EFLAGS trap flag (TF). This will single-step through slab instructions, not original instructions. GDB will show disassembly of translated code, which includes Stalker's inserted callout trampolines, making it very hard to follow. Source-level stepping (`next`, `step`) will be completely broken since there is no debug info for slab addresses. |
| `info proc mappings` | **Yes** | This still works and is actually useful for identifying the Stalker slab regions (anonymous RWX mappings). |

**Verdict**: GDB can attach and is useful for inspecting state (memory, globals,
registers when paused outside Stalker), but breakpoints, stepping, and
backtraces are largely non-functional during active Stalker instrumentation.

### 2.2 LLDB

| Aspect | Compatibility | Notes |
|--------|--------------|-------|
| Attach to process | **Yes** | Same reasoning as GDB — no `ptrace` conflict with embedded Frida. |
| Breakpoints in `.so` code | **No** | Same issue as GDB — original code not executing. |
| Backtraces | **Partial** | We have empirical evidence that LLDB backtraces work to some degree — we used LLDB to debug the `Gum::obtain()` crash. However, that crash likely occurred before or after Stalker was active, not during active translation. During active Stalker instrumentation, LLDB faces the same slab-address unwinding problem as GDB. |
| Watchpoints | **Mostly yes** | Same as GDB. |

**Verdict**: Similar to GDB. Useful outside the Stalker bracket. The prior
success debugging the `Gum::obtain()` crash is consistent — that crash happened
during Frida initialization, before `follow_me()` was called, so all code was
still running from its original location.

### 2.3 rr (Record & Replay)

| Aspect | Compatibility | Notes |
|--------|--------------|-------|
| Recording | **Likely broken** | rr uses `ptrace` to control the tracee and intercepts all syscalls. More critically, rr relies on the **hardware Retired Conditional Branch counter** (`HW_RETIRED_CONDITIONAL_BRANCHES`) to track program progress deterministically. Stalker's translated code in the slab will generate *different* branch counts than the original code because: (1) Stalker inserts additional control flow (callout trampolines, backpatching jumps); (2) the translated code layout differs from the original. This makes the branch counts non-deterministic relative to the original program semantics, breaking rr's core invariant. |
| Dynamic code generation | **Problematic** | rr has known issues with JIT-compiled and dynamically generated code (see [rr issue #3461](https://github.com/rr-debugger/rr/issues/3461)). Stalker's code cache is exactly this: dynamically `mmap`'d, `mprotect`'d executable pages that change over the lifetime of the program. rr must track all such mappings and their contents. Stalker continuously allocates, writes, and makes executable new code slabs, which stresses rr's recording infrastructure. |
| Signal handling | **Conflicting** | rr interposes on all signals for deterministic replay. Stalker may use signals internally (e.g., SIGSEGV for code cache invalidation on some platforms). These would conflict with rr's signal interception. |
| Replay fidelity | **Broken** | Even if recording succeeded, replay would fail because rr would need to reproduce the exact sequence of `mmap`/`mprotect` calls for the code cache, and the branch counts would not match during replay. |

**Verdict**: rr and Frida Stalker are fundamentally incompatible during active
instrumentation. rr's determinism model depends on hardware branch counters
that Stalker invalidates by rewriting the instruction stream.

**Critical note**: Our software RBC and rr both count retired conditional
branches, but for different purposes. rr counts hardware RBC for progress
tracking; we count them in software via Stalker callouts. These two uses are
architecturally incompatible because Stalker's code translation changes the
hardware branch counts that rr depends on.

### 2.4 Valgrind (Memcheck, Helgrind, etc.)

| Aspect | Compatibility | Notes |
|--------|--------------|-------|
| Coexistence | **No — fundamentally incompatible** | Valgrind is itself a DBI framework. It works by the same basic principle: translating the program's code into an intermediate representation (VEX IR), inserting instrumentation (shadow memory checks, lock-order tracking), and running the translated code in a "sandbox." Two DBI tools cannot run simultaneously because each assumes it controls the instruction stream. |
| Specific conflicts | **Total** | (1) Valgrind translates all code — but Stalker has already translated it and is running copies in the slab. Valgrind would attempt to translate the slab code, which contains Stalker's internal trampolines, not the original program logic. (2) Both tools use `mmap`/`mprotect` to manage executable memory, and each tool's assumptions about memory layout would be violated by the other. (3) Both tools intercept signals and `clone`/`fork` syscalls. |
| Running order | **Cannot stack** | Unlike library interposition (`LD_PRELOAD`) where multiple libraries can stack, DBI tools each take complete control of the execution environment. There is no protocol for DBI tools to cooperate. This is a well-known limitation across all DBI frameworks (Valgrind, DynamoRIO, Intel Pin, Frida Stalker). |

**Verdict**: Completely incompatible. You cannot run Valgrind on a process that
uses Frida Stalker. This is a fundamental architectural constraint of DBI
tools, not a Frida-specific limitation.

### 2.5 Sanitizers (ASan, MSan, TSan, UBSan)

| Aspect | Compatibility | Notes |
|--------|--------------|-------|
| Compile-time instrumentation | **Partially compatible** | Sanitizers (ASan, TSan, etc.) insert instrumentation at compile time — every memory access gets a shadow memory check call inserted by the compiler. This instrumentation is part of the binary's code, so Stalker would translate it along with everything else. The instrumentation calls themselves should still work because they are just function calls to the sanitizer runtime. |
| Shadow memory | **Risk of conflict** | ASan reserves a large region of the address space (typically `0x7fff8000` onward, or via `mmap` at specific addresses) for shadow memory. Stalker's slab allocations could theoretically collide with these regions. In practice, both use `mmap` which should find non-conflicting regions, but there are no guarantees. |
| ASan on the `.so` | **Not applicable** | The scheduler `.so` files are compiled by the user (typically with a standard `cc` invocation, not with `-fsanitize=address`). We cannot add sanitizers to the `.so` without recompiling it. |
| ASan on the Rust binary | **Plausible** | The Rust binary (scx_simulator) could be compiled with sanitizers (`-Z sanitize=address`). The sanitizer instrumentation in the Rust code would be translated by Stalker but should still function because the shadow memory checks are ordinary memory accesses and function calls. However, this is **untested** and may have edge cases. |
| TSan (Thread Sanitizer) | **Higher risk** | TSan instruments synchronization operations and memory accesses to detect data races. Stalker's internal thread-local state management and the code cache may confuse TSan's happens-before analysis, producing false positives or missing real races. |
| UBSan | **Likely compatible** | UBSan inserts lightweight checks (integer overflow, null pointer dereference, etc.) that are simple conditional branches and function calls. These should survive Stalker translation without issues. |

**Verdict**: Compile-time sanitizers are *theoretically* compatible with
runtime DBI, but the combination is untested with Frida Stalker specifically.
ASan and UBSan are most likely to work; TSan is risky. The main limitation is
that sanitizers cannot be applied to the scheduler `.so` without recompiling it.

### 2.6 perf / Linux perf_events

| Aspect | Compatibility | Notes |
|--------|--------------|-------|
| `perf record` (sampling) | **Works, but results are confusing** | `perf record` uses hardware PMU sampling interrupts. These fire based on hardware events (cycles, instructions, branches) and capture the current instruction pointer. During Stalker instrumentation, the IP will point to slab addresses. `perf` will record these addresses, but they will not map to any known DSO or symbol. The samples will show up as `[unknown]` in `perf report`. |
| `perf stat` (counting) | **Works** | Hardware counters (cycles, instructions, branch-misses, cache-misses) will count events from the translated code in the slab. The absolute numbers are still meaningful for overall performance analysis, but they reflect the translated code's behavior, not the original code's behavior. The translated code has additional instructions (callout trampolines) that inflate instruction counts and may affect cache behavior. |
| Symbol resolution | **Broken for slab code** | `perf` resolves symbols via `/proc/<pid>/maps` and the ELF symbol tables. The slab is an anonymous `mmap` region with no associated ELF, so there are no symbols to resolve. `perf` supports a `jitdump` interface for JIT-compiled code (used by V8, SpiderMonkey, JVM) where the runtime writes a special file describing dynamically generated code regions. Frida does **not** write `jitdump` files, so `perf` cannot resolve slab addresses to meaningful symbols. |
| Hardware PMU counters | **Work correctly** | The PMU counts events from whatever code is actually executing, including slab code. This is fine for aggregate statistics but makes per-function attribution impossible for instrumented code. |
| `perf` on non-instrumented code | **Normal** | Code running outside the `follow_me()`/`unfollow_me()` bracket (engine, test harness, setup/teardown) profiles normally with correct symbol resolution. |

**Verdict**: `perf stat` is useful for aggregate performance numbers. `perf
record` produces results but cannot attribute samples to functions in the `.so`
or Rust code during active instrumentation. Profiling code outside the Stalker
bracket works normally.

### 2.7 strace / ltrace

| Aspect | Compatibility | Notes |
|--------|--------------|-------|
| strace | **Yes, with caveats** | `strace` uses `ptrace(PTRACE_SYSCALL)` to intercept syscalls. Since embedded Frida does not use `ptrace`, there is no "single tracer" conflict. `strace` will see all syscalls made by the process, including Stalker's `mmap`/`mprotect` calls for code cache management. Syscalls made from within Stalker-instrumented code are still real syscalls (Stalker does not intercept the `syscall` instruction itself — it just translates the surrounding code). The one concern is that `strace` pausing the process at every syscall may interact poorly with Stalker's timing assumptions, but there should be no correctness issue. |
| ltrace | **Broken for instrumented code** | `ltrace` intercepts library calls by overwriting PLT entries or using `LD_AUDIT`. Stalker's code translation means PLT calls are made from slab addresses, and Stalker may translate the PLT stubs themselves. This can confuse `ltrace`'s interception mechanism. |
| Overhead interaction | **Additive** | Both `strace` and Stalker add overhead. The combination may make the process very slow but should not cause correctness issues (for `strace`). |

**Verdict**: `strace` works and is a useful diagnostic tool for observing
Stalker's syscall behavior. `ltrace` is unreliable during active instrumentation.

### 2.8 Intel VTune / AMD uProf

| Aspect | Compatibility | Notes |
|--------|--------------|-------|
| Hardware sampling | **Same issues as perf** | VTune and uProf use the same PMU sampling infrastructure as `perf`. Samples collected during Stalker instrumentation will show slab addresses that cannot be resolved to symbols. |
| JIT profiling API | **Not integrated** | VTune provides the ITT/JIT API (`iJIT_NotifyEvent`) that allows runtimes to register dynamically generated code for symbol resolution. Frida does not call this API, so VTune cannot resolve slab addresses. A custom integration *could* be built (calling the ITT API from the Stalker transformer to register translated blocks), but this would be significant engineering effort and Stalker's block-level granularity would not map cleanly to function-level symbols. |
| User-mode sampling | **Works outside Stalker** | For code outside the Stalker bracket, VTune/uProf works normally with full source-level attribution. |

**Verdict**: Not useful during active Stalker instrumentation without custom
JIT profiling API integration. Works normally for non-instrumented code.

---

## 3. Practical Workarounds

### 3.1 Our architecture helps significantly

Our Stalker usage is **narrowly scoped**: `follow_me()` is called only on
worker threads, only during the dispatch/batch-processing phase, and only
while executing scheduler C code. The rest of the process — Rust engine code,
test harness, setup, teardown, trace analysis — runs natively without any
instrumentation.

This means:

- **Debugging engine bugs**: GDB/LLDB work perfectly for bugs in the Rust
  engine, scenario loading, trace analysis, etc. Set breakpoints in Rust code
  that runs outside the Stalker bracket.

- **Pre/post analysis**: `perf record` the full process and focus on samples
  from the setup/teardown phases where code runs natively.

- **Narrow the repro**: If a bug is suspected in the scheduler `.so` code,
  first reproduce it without Frida (using cooperative-only mode or PMU mode)
  to get clean debugger access. The bug is likely deterministic given the same
  seed regardless of the preemption mechanism.

### 3.2 Debugging strategies for instrumented code

1. **Debug outside the bracket**: Set breakpoints in `preempt::maybe_yield_preemptive()` or
   kfunc implementations. These are Rust functions called from the scheduler
   via our FFI layer. When these breakpoints hit, the call originated from
   translated code, but the breakpoint itself is in a kfunc that runs
   (in part) outside of the active translation context.
   Actually — correction: kfuncs run *while Stalker is active* (the thread
   is followed). The breakpoint on the kfunc original address would NOT be
   hit. However, the kfunc's effects (state changes, trace output) can be
   observed after `unfollow_me()`.

2. **Use tracing instead of breakpoints**: Our tracing infrastructure
   (`tracing` crate with `trace!`, `debug!`, etc.) works even during Stalker
   instrumentation (as long as the tracing calls do not involve stdio locks
   or complex allocations that could deadlock under Stalker). Structured
   tracing is the primary debugging tool during active instrumentation.
   **Warning**: As noted in `engine.rs`, avoid `eprintln!`/`println!` inside
   the Stalker bracket — stdio locks under Stalker DBI can deadlock.

3. **Conditional Frida disable**: Set `use_frida: false` in the preemptive
   configuration to fall back to PMU-based preemption or cooperative-only
   mode. This gives you clean GDB/LLDB access to the scheduler code.
   The command-line flag is `--preempt-mode=pmu` or `--preempt-mode=coop`
   (vs `--preempt-mode=frida`).

4. **Post-mortem with core dumps**: If the process crashes during Stalker
   instrumentation, a core dump will contain the slab addresses in the
   crashing frame. The core dump is still useful for inspecting memory,
   global state, and thread-local state, even if the backtrace is
   degraded. The thread-local `SOFTWARE_RBC_COUNTER`, `YIELD_PENDING`, and
   `FRIDA_ACTIVE` cells can be inspected in the core dump.

5. **Stalker diagnostics**: Use `stalker::total_callouts()` and
   `stalker::deferred_yields()` counters to verify Stalker is working
   correctly. These are logged at the end of each dispatch phase.

### 3.3 Frida `--disable-jit` mode

Frida has an option to disable JIT compilation of its JavaScript engine (V8 or
QuickJS). This flag affects the **Frida agent scripting runtime**, not Stalker.
Stalker is always a JIT-style code translator regardless of this flag. The
`--disable-jit` flag would only be relevant if we were using Frida's JavaScript
API, which we are not — we use the `frida-gum` C/Rust API directly.

**Conclusion**: `--disable-jit` does not help with debugger compatibility for
our use case.

### 3.4 Experimental unwind support

Frida provides `gum_stalker_activate_experimental_unwind_support()` which
attempts to make stack unwinding work during Stalker instrumentation. This
would improve GDB/LLDB backtraces. However:

- It is marked "experimental" — correctness is not guaranteed.
- It may add overhead to every translated block.
- It is unclear whether it generates synthetic `.eh_frame` data or uses
  some other mechanism.
- It may help with C++ exception handling (SEH on Windows, DWARF unwinding
  on Linux) — see [frida-gum issue #565](https://github.com/frida/frida-gum/issues/565).

This is worth testing but should not be relied upon for production debugging.

### 3.5 `gum_stalker_exclude()` for debugger-friendly regions

The Stalker API provides `gum_stalker_exclude(stalker, &memory_range)` to
exclude specific address ranges from translation. Code in excluded ranges
executes natively (from its original location), making it debugger-friendly.
However, in our current architecture, we do NOT use `exclude()` — instead,
our transformer only inserts callouts for addresses in the `.so` text range,
but all code is still translated (just without callouts for non-`.so` code).

Using `exclude()` for the Rust binary's text segment could potentially allow
breakpoints and backtraces to work for Rust code even while Stalker is active
on the `.so`. This would be a valuable improvement but has risks:

- Transitions between excluded and translated regions add overhead.
- The excluded code must not have inlined functions from the translated
  region (unlikely in our case since `.so` code is C and Rust code is
  separate).
- This is untested in our codebase.

---

## 4. Comparison with Static Instrumentation

If we used **static instrumentation** (compiler-inserted counters at
conditional branch sites) instead of Frida Stalker DBI, all of the above
compatibility issues would disappear.

| Tool | With Frida Stalker (DBI) | With Static Instrumentation |
|------|-------------------------|-----------------------------|
| **GDB/LLDB breakpoints** | Broken during active instrumentation | Fully working — code runs in place |
| **GDB/LLDB backtraces** | Degraded (slab addresses) | Fully working — DWARF info intact |
| **GDB/LLDB watchpoints** | Mostly working | Fully working |
| **GDB/LLDB stepping** | Confusing (slab code + trampolines) | Fully working — step through original + inserted counter code |
| **rr** | Fundamentally incompatible | **Compatible** — static instrumentation adds deterministic instructions that rr can record/replay normally. Branch counts change (counter checks add branches) but they are deterministic. |
| **Valgrind** | Fundamentally incompatible | **Compatible** — Valgrind translates statically-instrumented code just like any other code. Shadow memory checks work on the counter variables. |
| **ASan/TSan/UBSan** | Risky / untested | **Fully compatible** — both are compile-time instrumentation, they compose naturally. |
| **perf** | Samples show slab addresses | **Fully working** — symbols resolve normally. Inserted counter code appears in profiles (minor overhead visible). |
| **strace** | Works | Works |
| **VTune/uProf** | Slab addresses unresolvable | **Fully working** — source-level attribution works normally. |
| **rr + debugging** | Neither works | **Both work** — this is a major advantage. Record a failing seed with rr, replay with reverse debugging, set breakpoints in scheduler code. |

### Key tradeoff

Static instrumentation requires **recompiling the scheduler `.so`** with a
special compiler pass or wrapper. This means:

- The `.so` source code must be available (it is — we build the schedulers).
- A custom compiler pass or `__builtin_expect`-style wrapper must be
  maintained.
- The counter increment must be inserted at every conditional branch site,
  which is straightforward with a compiler pass but requires build system
  integration.
- The instrumentation is always present (no runtime toggle), though the
  counter can be initialized to `u64::MAX` (disarmed) for negligible
  overhead when not in use.

The DBI approach (Frida Stalker) avoids all build system complexity — it works
on any `.so` without recompilation. But the debugger compatibility cost is
severe, especially the loss of rr, which is arguably the most powerful tool
for debugging deterministic concurrency bugs (exactly the kind of bugs the
simulator is designed to find).

---

## 5. Summary and Recommendations

### Tools that work during active Stalker instrumentation

| Tool | Status |
|------|--------|
| strace | Works |
| perf stat (aggregate counters) | Works (measures translated code) |
| GDB/LLDB attach + memory inspection | Works (but breakpoints/backtraces broken) |
| Hardware data watchpoints | Mostly works |
| Tracing infrastructure (tracing crate) | Works (avoid stdio locks) |

### Tools that do NOT work during active Stalker instrumentation

| Tool | Status | Fundamental or fixable? |
|------|--------|------------------------|
| GDB/LLDB breakpoints in `.so` | Broken | Fundamental — original code not executing |
| GDB/LLDB source-level stepping | Broken | Fundamental — no debug info for slab |
| GDB/LLDB backtraces | Degraded | Partially fixable with experimental unwind support |
| rr | Broken | Fundamental — DBI invalidates rr's branch counting |
| Valgrind | Broken | Fundamental — two DBIs cannot coexist |
| perf record (per-function profiles) | Degraded | Fixable with jitdump integration (high effort) |
| VTune/uProf (per-function) | Degraded | Fixable with ITT API integration (high effort) |
| ltrace | Broken | Fundamental — PLT interception defeated by code translation |

### Practical recommendation

For day-to-day development and debugging:

1. **Use `--preempt-mode=coop` or `--preempt-mode=pmu`** when you need to
   attach a debugger to the scheduler code. The same seed should reproduce
   the same (or very similar) behavior.

2. **Use tracing** as the primary debugging tool during Frida-instrumented
   runs. Structure trace output with the `tracing` crate and filter with
   `RUST_LOG`.

3. **Use strace** to diagnose syscall-level issues in Frida-instrumented runs.

4. **Reserve Frida mode for production/CI determinism testing**, where you
   need exact software RBC counts but do not need interactive debugging.

5. **Seriously evaluate static instrumentation** if rr compatibility is
   important. The ability to record a failing seed with rr and replay it
   with full reverse-debugging capability is enormously valuable for
   debugging subtle concurrency bugs, and this capability is completely
   lost with DBI.

---

## References

- [Frida Stalker documentation](https://frida.re/docs/stalker/)
- [Frida Gadget documentation](https://frida.re/docs/gadget/)
- [rr project](https://rr-project.org/)
- [rr issue #3461 — JIT code interferes with reverse execution](https://github.com/rr-debugger/rr/issues/3461)
- [frida-gum issue #565 — Stalker and C++ SEH/unwind problems](https://github.com/frida/frida-gum/issues/565)
- [GDB JIT Interface documentation](https://sourceware.org/gdb/current/onlinedocs/gdb.html/JIT-Interface.html)
- [Intel ITT/JIT API documentation](https://intel.github.io/ittapi/src/jit-api-support.html)
- [rr performance counters (DeepWiki)](https://deepwiki.com/rr-debugger/rr/6.1-performance-counters)
- [Robert O'Callahan — Deterministic Hardware Performance Counters](https://robert.ocallahan.org/2017/03/deterministic-hardware-performance.html)
- [Robert O'Callahan — Exploiting Precognition in Binary Instrumentation of rr Replays](https://robert.ocallahan.org/2020/12/exploiting-precognition-in-binary.html)
- [perf jitdump interface (Firefox Source Docs)](https://firefox-source-docs.mozilla.org/performance/jit_profiling_with_perf.html)
- [Evaluating DBI Systems (ACM)](https://dl.acm.org/doi/fullHtml/10.1145/3478520)
