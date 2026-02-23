# Frida Stalker DBI + Debugger Interaction Investigation

**Date**: 2026-02-22
**Branch**: `simulator-frida`
**Investigated by**: AI analysis with experimental verification

## Executive Summary

Frida Stalker's dynamic binary instrumentation (DBI) fundamentally conflicts with
traditional debugger operation. When Stalker is active on a thread, the thread
executes JIT-compiled copies of basic blocks from an anonymous RWX slab, not the
original code. Breakpoints set on original addresses are ineffective for
Stalker-translated code. However, **breakpoints on Rust code that runs BEFORE
`follow_me` or AFTER `unfollow_me` work normally**, and breakpoints on kfunc
boundary code (`maybe_yield_preemptive`, `arm_software_rbc`) are fully functional.

**Severity assessment**: Manageable limitation. The simulator's architecture
(kfunc boundaries, deferred yields, structured logging) provides sufficient
debugging surface without needing to break into Stalker-translated code.

---

## 1. Can lldb Attach to a Running `scxsim --frida` Process?

### Experiment

Attached lldb to a running Frida-instrumented process (PID 1855573, release build):

```
(lldb) process attach --pid 1855573
Process 1855573 stopped
* thread #1, name = 'scxsim', stop reason = signal SIGSTOP
```

### Results

**Yes, lldb can attach.** All 5 threads are visible and stopped:

```
Process 1855573 stopped
* thread #1: tid = 1855573, name = 'scxsim', stop reason = signal SIGSTOP
  thread #2: tid = 1855585, name = 'scxsim', stop reason = signal SIGSTOP
  thread #3: tid = 1855586, name = 'scxsim', stop reason = signal SIGSTOP
  thread #4: tid = 1855587, name = 'scxsim', stop reason = signal SIGSTOP
  thread #5: tid = 1855588, name = 'scxsim', stop reason = signal SIGSTOP
```

However, **backtraces are unusable** for a release build (no debug symbols).
With a debug build, backtraces work for code outside the Stalker-instrumented
region but show raw addresses for Stalker-translated code.

### Breakpoints in the Scheduler `.so`

Breakpoints on `.so` functions (`simple_select_cpu`, `simple_enqueue`, etc.)
**DO resolve and hit** during the single-threaded (non-Stalker) processing phase:

```
(lldb) breakpoint set --shlib libscx_simple.so --name simple_select_cpu
Breakpoint 1: no locations (pending).  # Resolves after dlopen
(lldb) run
1 location added to breakpoint 1      # Resolves when .so is loaded
Process stopped
* thread #1, stop reason = breakpoint 1.1
    frame #0: libscx_simple.so`simple_select_cpu
    frame #1: scxsim`Scheduler::select_cpu at ffi.rs:813
    frame #2: scxsim`handle_task_wake at engine.rs:1900
```

These hits occur during the **non-Stalker single-threaded path** in
`run_internal` (Phase 0 / init phase). The same breakpoints are **NOT hit**
during the batch-concurrent Stalker phase because Stalker executes translated
copies, not the original code.

### Breakpoints in Rust Code

Breakpoints on Rust functions work with varying success depending on timing:

| Function | Hits? | Reason |
|----------|-------|--------|
| `maybe_yield_preemptive` | **Yes** | Called during init phase (non-Stalker) |
| `build_transformer` | **Yes** | Called before Stalker starts |
| `arm_software_rbc` | **Yes** | Called before `follow_me` |
| `software_rbc_callout_inner` | **No** | Called FROM Stalker-translated code |
| `take_yield_pending` | **No** | Called from within Stalker context |

### Backtrace from Pre-Stalker Code

Clean and fully symbolic:

```
* thread #5, stop reason = breakpoint (arm_software_rbc)
  frame #0: stalker::arm_software_rbc at stalker.rs:392
  frame #1: engine::process_batch_concurrent_frida at engine.rs:3541
  frame #2: std::backtrace::__rust_begin_short_backtrace
  frame #3: std::thread::Builder::spawn_unchecked_
  ...
  frame #13: libc.so.6`start_thread
```

### Backtrace After Stalker Takes Over

After `follow_me` is called and the thread starts executing in the Stalker slab:

```
* thread #5, stop reason = signal SIGTRAP
    frame #0: 0x00007ffff75b7095  (anonymous RWX region)
->  0x7ffff75b7095: addb   %cl, -0x75(%rax)   # Stalker JIT code
```

The debugger sees a SIGTRAP in anonymous memory with no symbol information. The
backtrace is broken because Stalker's out-of-place execution rewrites control
flow.

---

## 2. Is Frida Instrumentation Done in One Pass or JIT?

### Answer: Lazy JIT with Caching (Trust Threshold)

Frida Stalker uses a **lazy JIT model**:

1. **First execution**: When the thread first reaches a basic block, Stalker's
   transformer callback is invoked. The callback iterates over instructions,
   optionally inserting callouts (as `scx_simulator` does for conditional
   branches), and the transformed block is written to the **code slab** (an
   anonymous RWX memory region).

2. **Subsequent executions**: The translated block is **reused from the slab**.
   The `trust_threshold` parameter controls how many times a block must be
   executed before Stalker "trusts" it and removes the dynamic lookup overhead:

   ```rust
   // From frida-gum's Stalker API:
   pub fn set_trust_threshold(&mut self, threshold: i32)
   // -1 = never trust (slowest, always check)
   //  0 = trust immediately (fastest, no re-translation)
   //  N = trust after N executions (default: 1)
   ```

3. **The `Compile` event**: Stalker fires a `GUM_COMPILE` event each time it
   JIT-compiles a new basic block. This can be observed via the `EventSink`
   trait:

   ```rust
   enum EventMask {
       Compile = 1 << 4,  // Fired on first JIT compilation of a block
   }
   ```

### Evidence from the Codebase

The `scx_simulator` uses `build_transformer` which registers a callback:

```rust
Transformer::from_callback(gum, move |basic_block, _output| {
    for instr in basic_block {
        // Check if instruction is in the .so's text segment
        if addr >= range_base && addr < range_base + range_size {
            if is_conditional_branch_opcode(bytes) {
                instr.put_callout(move |_cpu_context| {
                    software_rbc_callout_inner(orig_addr);
                });
            }
        }
        instr.keep();  // Keep the original instruction (translated copy)
    }
});
```

This callback is invoked **lazily** when each basic block is first reached. The
`instr.keep()` call tells Stalker to emit a translated copy of the instruction.
The `instr.put_callout()` inserts a call to the Rust function at that point in
the translated block.

### Memory Layout Evidence

From `/proc/<pid>/maps` of a running Frida-instrumented process:

```
7fd199832000-7fd19987f000 rwxp 00000000 00:00 0   # Code slab (~316KB)
7fd199882000-7fd199889000 rwxp 00000000 00:00 0   # Slow slab (~28KB)
```

These anonymous RWX regions are Stalker's JIT code cache. The original `.so` is
mapped read-only:

```
7fd19980c000-7fd19980d000 r--p ... libscx_simple.so
```

Disassembly of the slab shows Stalker infrastructure code (AVX context
save/restore for callouts):

```
0x7fd19983505a: vextracti128 $0x1, %ymm4, 0x40(%rsp)
0x7fd199835062: vextracti128 $0x1, %ymm5, 0x50(%rsp)
...
0x7fd1998350d2: jmpq   *0x58(%rbx)    # Indirect jump to next block
```

### Key Implication

Because translation is lazy, blocks that are never executed are never translated.
The transformer callback captures the **original address** of each instruction
(which `scx_simulator` passes to `software_rbc_callout_inner` as `orig_addr`),
but execution happens at the **slab address**. This is why breakpoints on
original addresses are ineffective.

---

## 3. Frida + Debugger Documentation & Community Knowledge

### Official Frida Documentation

The [Stalker page](https://frida.re/docs/stalker/) describes Stalker as a "code
tracing engine" that "copies and instruments code just-in-time." The documentation
explicitly acknowledges the out-of-place execution model but does not discuss
debugger compatibility.

### Key Frida APIs for Debugging

1. **`Stalker::exclude(range)`**: Excludes a memory range from instrumentation.
   Code in excluded ranges runs at its original address, making breakpoints
   work. Could be used to selectively exclude Rust runtime code.

2. **`Stalker::activate(target)` / `Stalker::deactivate()`**: Pauses/resumes
   Stalker on the current thread. Code runs at original addresses when
   deactivated. This enables **conditional Stalker**: instrument only during
   scheduler `.so` calls, deactivate during Rust kfunc code.

3. **`Stalker::set_trust_threshold(threshold)`**: Controls re-translation.
   Setting to -1 (never trust) forces re-translation every time, which is
   slower but could be useful for debugging (ensures fresh callout data).

4. **`EventSink` with `Compile` events**: Allows observing which blocks get
   JIT-compiled and when. Useful for understanding code coverage under Stalker.

5. **`Stalker::invalidate(address)`**: Forces re-translation of a specific
   block. Could theoretically be used to "refresh" instrumentation.

6. **`gum_stalker_activate_experimental_unwind_support()`**: Experimental
   support for unwinding through Stalker-translated code. May improve
   backtraces but is marked experimental.

### Community Knowledge

The Frida community generally acknowledges that:

- **Debuggers and DBI frameworks are fundamentally incompatible** because both
  want to control instruction execution. Debuggers use `int 3` (software
  breakpoints) which require patching original code, but Stalker executes
  copies.

- **Hardware breakpoints might work** because they trigger on address access
  regardless of whether the code is original or translated. However, Stalker
  translates addresses, so hardware breakpoints on original `.so` addresses
  would not fire (the translated copy is at a different address).

- **Frida's own scripting (`put_callout`)** is the recommended way to "debug"
  instrumented code -- essentially replacing breakpoints with programmatic
  callouts that log state.

### No Frida GDB Server Mode

Frida does **not** provide a GDB server mode for Stalker. The `frida-server`
component (used for mobile instrumentation) provides a different kind of
debugging interface, not applicable to the Stalker use case.

---

## 4. Practical Debugging Strategies

### What WORKS

1. **Breakpoints on Rust code outside the Stalker window**: Code that runs
   before `follow_me` or after `unfollow_me` is fully debuggable. This includes:
   - `arm_software_rbc` / `disarm_software_rbc`
   - `build_transformer` / `SyncTransformer::new`
   - `PreemptRing::wait_for_token` / `PreemptRing::finish`
   - `kfuncs::enter_sim` / `kfuncs::exit_sim`

2. **Breakpoints during the init phase**: The init/single-threaded processing
   phase does NOT use Stalker. All `.so` callbacks (`init`, `select_cpu`,
   `enqueue`, `dispatch`) hit normally during this phase.

3. **Structured logging (RUST_LOG=trace)**: The simulator's trace logging
   provides detailed state at every kfunc boundary:
   ```
   preempt:frida rbc structop 3:15 rbc 261 rip 0x4a7
   preempt:frida kfunc structop 3:15 kfunc 4
   ```

4. **Callout-based "printf debugging"**: Stalker callouts can safely read
   thread-local state. The existing `software_rbc_callout_inner` demonstrates
   the pattern -- additional diagnostic callouts could be added.

### What DOES NOT WORK

1. **Breakpoints in `.so` code during the Stalker phase**: The translated copies
   do not go through original addresses.

2. **Stepping over `follow_me`**: lldb receives SIGTRAP from Stalker's JIT
   code and loses track of the thread.

3. **Backtraces through Stalker-translated code**: The backtrace shows raw
   addresses in anonymous RWX memory with no symbol information.

4. **`software_rbc_callout_inner` breakpoints**: Even though this is a Rust
   function, it is called from Stalker-translated code. The function's code
   may itself be translated by Stalker (since Stalker instruments ALL code on
   the thread, not just the `.so`).

### Conditional Stalker Strategy

The `activate/deactivate` API could enable a hybrid approach:

```rust
// Pseudocode: only instrument during .so calls
stalker_inst.follow_me(...);  // Start Stalker
stalker_inst.deactivate();     // Immediately pause

// When calling into .so:
stalker_inst.activate(so_entry_point);  // Resume for .so code
// ... .so executes under Stalker ...
// When .so returns to Rust, Stalker could be deactivated

stalker_inst.unfollow_me();  // Stop Stalker
```

This is not currently implemented in `scx_simulator` but could be added.
The challenge is that Stalker translates ALL code on the thread once active,
including the Rust runtime code that the `.so` calls back into (kfuncs). The
`exclude` API would need to exclude the entire Rust binary to make this work.

### Recommended Debugging Workflow

For developers debugging Frida-instrumented scheduler code:

1. **Start with non-Frida mode**: Debug without `--frida` first to validate
   logic. The PMU-based preemption mode allows full debugging.

2. **Use trace logging for Frida-specific issues**: Run with `RUST_LOG=trace`
   to capture per-structop RBC counts, kfunc counts, and yield points.

3. **Break on setup/teardown**: Set breakpoints on `arm_software_rbc`,
   `build_transformer`, or `PreemptRing` functions to inspect state before
   Stalker takes over.

4. **Use determinism checking**: Run `--determinism-check --frida` to verify
   that the same seed produces the same trace. Divergences indicate bugs.

5. **Add diagnostic callouts**: For deep Stalker debugging, add temporary
   `put_callout` calls that write to thread-local diagnostic buffers.

---

## 5. rr Compatibility

### Experimental Result

```
$ rr record target/debug/scxsim --scheduler simple --preemptive --frida ...
[FATAL] AMD CPU type 0x10f10 (ext family 0xb) unknown
```

rr failed due to **CPU incompatibility** (AMD EPYC 9D85, Zen 5), not due to
Frida. This CPU is too new for the installed rr version.

### Theoretical Analysis

Even on a supported CPU, rr would likely fail with Frida Stalker because:

1. **rr relies on hardware performance counters** (specifically retired
   conditional branches) for deterministic replay. Stalker's JIT-translated
   code changes the branch topology -- translated blocks contain different
   branch patterns than the original code (additional jumps between slab
   blocks, callout dispatch branches, etc.).

2. **rr uses `PTRACE_SINGLESTEP`** which conflicts with Stalker's control
   flow manipulation. When Stalker redirects execution to the slab,
   single-stepping would follow the translated code path, not the logical
   source-level path.

3. **Stalker uses RWX memory regions** which rr may flag as suspicious or
   handle incorrectly during replay.

4. **Frida's internal use of signals and mmap** for slab management would
   need to be replayed correctly, which rr may not handle for anonymous
   RWX regions that change at runtime.

### Conclusion

rr is almost certainly incompatible with Frida Stalker. This is consistent
with the general principle that DBI frameworks (Frida, DynamoRIO, Pin, Valgrind)
and record-replay debuggers (rr) are fundamentally incompatible because both
require exclusive control over the program's execution model.

---

## Appendix A: Memory Layout of a Frida-Instrumented Process

From `/proc/<pid>/maps`:

```
# Original binary
560659474000-56065974a000 r--p  ... scxsim     # Read-only data
56065974a000-560659b36000 r-xp  ... scxsim     # Executable code
56065a10f000-56065a279000 r--p  ... scxsim     # Relocations

# Scheduler .so (loaded via dlopen)
7fd19980c000-7fd19980d000 r--p  ... libscx_simple.so  # Read-only

# Frida Stalker JIT slabs (anonymous RWX)
7fd199832000-7fd19987f000 rwxp  (none)  # Code slab (~316KB)
7fd199882000-7fd199889000 rwxp  (none)  # Slow/data slab (~28KB)

# System libraries
7fd199600000-7fd19979e000 r-xp  ... libc.so.6
```

The RWX regions are the telltale sign of Frida Stalker. They contain:
- Block metadata (first ~4KB: pointers, counters)
- Translated basic blocks with callout dispatch code
- Context save/restore trampolines (AVX register spills)

---

## Appendix B: Glossary

| Term | Definition |
|------|-----------|
| **DBI** | Dynamic Binary Instrumentation -- modifying code at runtime |
| **Stalker** | Frida's DBI engine that translates and instruments code |
| **Slab** | Anonymous RWX memory region where Stalker stores JIT-compiled code |
| **Callout** | A function call inserted by Stalker into translated code |
| **Trust Threshold** | How many executions before Stalker caches a block permanently |
| **RBC** | Retired Branch Conditional -- hardware counter for branch instructions |
| **Software RBC** | Stalker-based equivalent: callouts decrement a counter at each Jcc |
| **Transformer** | Callback that Stalker invokes to instrument each basic block |

---

## Appendix C: Summary of Experimental Commands and Results

### Experiment 1: lldb Launch with Breakpoints

```bash
lldb target/debug/scxsim -- --scheduler simple --preemptive --frida --seed=42 \
  workloads/dsq_contention.json
(lldb) breakpoint set --name software_rbc_callout_inner  # Never hits
(lldb) breakpoint set --name maybe_yield_preemptive      # Hits (init phase)
(lldb) breakpoint set --name arm_software_rbc            # Hits (pre-Stalker)
```

### Experiment 2: .so Breakpoints

```bash
(lldb) breakpoint set --shlib libscx_simple.so --name simple_select_cpu
# Hits during init phase, NOT during Stalker batch-concurrent phase
```

### Experiment 3: Stepping Over follow_me

```bash
(lldb) breakpoint set --name arm_software_rbc
(lldb) run  # Hits at arm_software_rbc
(lldb) thread step-out  # Returns to engine.rs:3541
(lldb) thread step-over  # Steps to follow_me call at engine.rs:3543
(lldb) thread step-over  # SIGTRAP at 0x7ffff75b7095 (Stalker slab)
```

### Experiment 4: Attach to Running Process

```bash
lldb -p 1855573  # Release build, simulation complete
# All threads in futex_wait (preempt ring idle)
# Breakpoints fail to resolve (no debug symbols in release)
```

### Experiment 5: rr Recording

```bash
rr record target/debug/scxsim --scheduler simple --preemptive --frida ...
# [FATAL] AMD CPU type 0x10f10 unknown  (CPU incompatibility, not Frida-specific)
```

---

## Recommendations

1. **Do not attempt to debug inside the Stalker window.** Use logging,
   callouts, and determinism checking instead.

2. **Debug the init phase and setup code freely.** Breakpoints work normally
   before `follow_me` and after `unfollow_me`.

3. **For scheduler `.so` debugging**, run without `--frida` first. The
   single-threaded path calls `.so` functions at their original addresses
   where breakpoints work.

4. **Invest in trace-based debugging.** The existing `RUST_LOG=trace` output
   and structop tracking provide rich diagnostic data without needing a
   debugger.

5. **Consider `Stalker::exclude()` for future work.** Excluding the Rust
   binary's code range from Stalker instrumentation could allow breakpoints
   on kfunc code to work even during the Stalker phase. This would require
   architectural changes but is technically feasible.

6. **For rr, test on Intel hardware** with a supported CPU. Even then,
   expect Frida incompatibility due to JIT code cache and branch count
   distortion.
