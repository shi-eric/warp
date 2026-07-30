---
name: warp-compile-time-optimizer
description: Use when a Warp app stalls at first wp.launch or recompiles JIT modules each run. Not for runtime, memory, correctness, Warp source, or native builds.
license: Apache-2.0
allowed-tools: Bash, Read, Edit, Write, Glob, Grep
compatibility: >-
  Requires warp-lang and Python 3.10+. Runs the user's command as a subprocess;
  writes only to a temp directory. No network, MCP, or Warp checkout needed.
  CUDA mechanisms need a CUDA device.
metadata:
  version: "0.1.0"
  author: "Warp Team <warp-python@nvidia.com>"
  tags:
    - warp
    - compilation
    - cold-start
    - startup-latency
    - kernel-cache
    - gpu
---

# Warp cold-start compile time

## Purpose

You need a command that runs the application to completion; everything else
follows from measuring it.

## Prerequisites

Python 3.10+, `warp-lang`, and a runnable target command; CUDA-specific
mechanisms require a CUDA device.

## Available scripts

| Script | Purpose | Arguments |
| --- | --- | --- |
| `scripts/warp_compile_probe.py` | Measure isolated cold/warm compilation and launches. | `measure [OPTIONS] -- COMMAND...`; use `--help`. |

Use `run_script("scripts/warp_compile_probe.py", args=[...])` when supported;
otherwise use the Python command under Instructions.

## Compilation model

Warp compiles modules, not individual kernels. A module's identity is:

    (live kernel & function set) x (module options) x (CUDA block_dim) x (generic instances)

Each identity requires code generation and native compilation for the full
module.

Cold-start cost is roughly:

    number of distinct module identities you touch  x  size of each module

Reduce it in two ways:

1. Stop identity churn. A module that changes after loading compiles again.
2. Stop source duplication. A module used at three block dimensions compiles
   every kernel three times.

Deleting one kernel from a module that still builds saves only part of one
build. Removing an unnecessary module identity saves the full build.

When neither applies, the builds that remain are still serialized by default, and
overlapping them shortens the wait without compiling less (CS-13). That is a
different lever from the two above: it changes when the work happens, not how
much there is, so it composes with them and is judged on a different clock.

## When a module's options are fixed

Options are part of module identity, so when you set one decides whether it
takes effect at all. There are two deadlines, and missing them fails in opposite
ways.

**Module creation, which happens when its Python file is imported.** Six options
are copied out of `warp.config` as the `Module` object is built: `enable_backward`,
`max_unroll`, `lineinfo`, `deterministic`, `deterministic_max_records`, and
`compile_time_trace`. Assigning `wp.config.<option>` after that file has been
imported cannot reach it. Nothing raises. The module keeps the value it was born
with. (`default_grid_stride` is the exception, staying `None` and resolving from
config at use.)

**First load.** Until the module compiles, its options are still writable with
`wp.set_module_options()` or `wp.get_module(name).options`. Both work after
import; they mutate the module that already exists rather than a default. After
the module loads, changing an option gives it a new identity and rebuilds it
(CS-3).

| Missed deadline | Symptom | Cost |
| --- | --- | --- |
| `wp.config.*` set after import | hash unchanged, option silently absent | the entire benefit, invisibly |
| module options set after load | a second hash, module builds twice | one extra build, visible in the trace |

The first row is the dangerous one precisely because it is free of symptoms: a
partially applied global looks like a working change and a disappointing result.

So use the hash as the receipt. After changing any option, confirm the hash moved
for every module you meant to change. An unchanged hash means the option never
arrived, not that it made no difference.

## Preserve behavior

Change how Warp compiles the code, not the workload.

Do not delete or merge kernels to claim a compile-time gain. Apparently
redundant stages may preserve ownership, aliasing, retained outputs, numerical
boundaries, or API behavior. Merging kernels also does not remove
block-dimension variants unless it changes the launches. Fix duplication at
the module level.

Keep intact: every launch and its order, dimensions, dtypes, devices, and
block dimensions; gradients wherever a tape may traverse; numerical modes;
dynamic and plugin behavior; and public API signatures.

Keep kernel names when moving definitions to module scope. Names appear in
module logs, cache artifacts, and external tooling.

## Instructions

### 1. Find the real cost, and confirm it is compilation

Ask what command the user actually waits on, then measure it cold:

```bash
python scripts/warp_compile_probe.py measure --samples 3 \
    --json baseline.json -- <the user's command>
```

The probe runs the command in a fresh process with a private, empty Warp
cache, sets `WARP_CACHE_PATH`, `WARP_CACHE_ROOT`, and `CUDA_CACHE_PATH` to new
uniquely named directories for each sample, turns on Warp's module timers, and
records every kernel launch. Never make a cache cold by deleting it or by
calling `wp.clear_kernel_cache()` or `wp.clear_lto_cache()`: those operations
mutate the currently configured cache, can disrupt concurrent processes, and
do not isolate every cache layer.

Read the probe output before inspecting source. If compilation is a small part
of wall time, say so and stop; naming the real bottleneck is worth more to the
user than a compile-time change that cannot matter.

For libraries and test suites, measure the smallest command that compiles the
same modules as the real workload.

Modules that each compiled once, with no repeated hashes, block-dimension
variants, or LTO, have no structural churn to remove. That is a narrower finding
than it feels like: what remains is the size of each build, which cache
persistence (CS-2), the options in step 2, and CS-10/CS-11 all still reach.
Report that distinction rather than calling the remaining cost necessary — see
step 6.

Every sample also re-runs the command against the cache it just populated. If
that warm number is near zero, compilation is a cold-start cost. If it is not,
the application is discarding its own cache and no restructuring will help:
check cache reuse in `references/mechanisms.md` before touching code.

### 2. Ask about runtime tradeoffs when needed

Some fixes are free: grouping modules by lifecycle, defining kernels before
first load, and hoisting options above the first launch. Apply them without
asking.

Others buy compile time with runtime: relaxing or tightening `fast_math`,
lowering `max_unroll`, or falling back from a MathDx/tile implementation to a
scalar one. Ask before making one of these changes:

> Some of these knobs cut compile time but can make the compiled kernels
> slower or change numerics. Are you optimizing a fast edit-run loop (where
> slower kernels are usually fine), or production startup (where they usually
> are not)?

If the user is unavailable:

- Leave numerics-changing options such as `fast_math` or implementation swaps
  alone.
- For capability changes such as `enable_backward`, test whether the repository
  uses the capability. If not, state what the change removes, its measured
  benefit, and how to revert it.
- Do not set global `wp.config.*` options from library code. Recommend them at
  the application entry point instead.

Record each declined option and its measured benefit in the ledger described in
step 6.

#### Match the scope of the change to the scope of the evidence

A profile is evidence about one application. An option written into a shared
library's module changes every application that imports it — including consumers
outside this repository and consumers not written yet. Those are different sizes
of claim, and the larger one belongs to whoever owns the library's contract.

This matters most when the expensive modules are not yours. Suppose two programs
share a solver module: one differentiates through it, one does not. Profiling the
second does not license turning backward codegen off in the shared module,
because that also silently removes gradients from the first. Searching the
repository for `wp.Tape` does not close the gap either; it samples today's
in-tree callers.

Turn the capability off for the process you measured instead, at its entry point,
before the library is imported:

```python
import warp as wp

wp.config.enable_backward = False   # must precede the library import

import the_library
```

This is both correctly scoped and usually the bigger win, because it reaches
every module the application loads rather than the few files you thought to edit.
CS-10 has the measurement and the ordering trap, which is silent when you get it
wrong.

When the only effective fix really is inside the shared module, report it as an
upstream request with the measurement attached and let the library's maintainers
decide. A well-evidenced issue is a better deliverable than a fork.

### 3. Diagnose from the measurement, not from reading the source

The probe prints every compiled module identity with its name, hash, device,
and block dimension, then names which modules built more than once. Match what
you see:

| What the probe shows | What it means | Where to look |
| --- | --- | --- |
| One module name, several **hashes** | Identity churn: its kernel set, options, or generic instances changed after it first loaded | CS-1, CS-3, CS-6 |
| One module name, several **block_dim** values | The whole module is recompiled per block dimension (CUDA) | CS-5 |
| Many one-kernel modules in one feature | Fixed per-module cost repeated | CS-4 |
| A hash-named module per kernel | `module="unique"` used on stable kernels | CS-9 |
| Big gap between module time and native compile time, plus `.lto` artifacts | MathDx/LTO setup | CS-7 |
| `(compiled)` on a run that should have been warm | Cache is not being reused | CS-2 |
| Modules load, then "Failed to find module" | Concurrent CPU JIT first-use race | CS-8 |
| Large generated source, no rebuild problem | Unroll budget | CS-11 |
| Adjoint code in a module nothing differentiates | Backward codegen | CS-10 |
| Compiles slow across the board, or a few small modules on CUDA below toolkit 13 | The precompiled header is turned off, or is not paying for itself | CS-12 |
| Several independent modules, each built once, `overlap_factor` near 1.0 | Builds are running one at a time; parallel loading is off by default | CS-13 |
| An option you set changed nothing, and that module's hash is unchanged | It was assigned after the module was created, so it never arrived | "When a module's options are fixed" |
| No row above fires | Nothing is being built redundantly; the cost is the size of the builds themselves | Step 6 |

`references/mechanisms.md` has one section per mechanism: how to confirm it,
the fix, its limits, and its failure mode. Read only the sections selected by
the measurement.

### 4. Choose module boundaries deliberately

Group kernels in one module only when they share:

- lifecycle: they are defined, loaded, and invalidated together;
- option set: they need the same `fast_math`, `enable_backward`, `max_unroll`,
  and MathDx settings;
- stable block dimension on CUDA.

Kernels with the same lifecycle but different stable block dimensions should
not share a module because each would compile twice. Separate kernels with
independent lifecycles too.

Kernels whose block dimension varies at runtime (chosen from input size, say)
have no stable mapping, so keep them in their own module rather than dragging
a whole shared module into an extra variant.

Prefer the least invasive change that removes a build. Ordering fixes and
option hoists are cheaper and safer than re-architecting module layout;
regroup only when fixed per-module cost or block-dimension duplication
dominates.

Named-module options are easy to misapply. `wp.set_module_options()` targets
the calling Python module. A bare call does not configure kernels declared with
an explicit `module="pkg.name"`.

Two things that do work:

```python
# 1. Give the group a real Python module and configure it at module scope.
#    Kernels there need no explicit module= at all.
wp.set_module_options({"enable_backward": False})

# 2. Or configure the named module directly, while it is still empty.
wp.get_module("pkg.name").options.update({"enable_backward": False})
```

What does not work: `wp.set_module_options(opts, module=...)` expects an object
with `__name__`, which `wp.get_module()` does not return. Likewise,
`@wp.kernel(module_options={...})`, which Warp rejects unless the kernel is also
`module="unique"`. Per-kernel `@wp.kernel(..., enable_backward=False)` does work,
with one exception on tile modules noted in CS-10.

After changing options, confirm that the module hash changed.

### 5. Verify

```bash
python scripts/warp_compile_probe.py measure --samples 3 \
    --json candidate.json -- <the same command>
python scripts/warp_compile_probe.py compare baseline.json candidate.json
```

`compare` checks the launch topology first and refuses to credit any
compile-time change when the workload moved, printing which launches
appeared or disappeared. It also applies a noise band of
`max(1% of baseline, 2 x baseline MAD)`; a change inside that band is
inconclusive.

Cold module work sums per-module timers, so a change that builds the same
modules concurrently inflates it while the wall clock drops. The probe prints
`BUILDS OVERLAPPED` when it sees that, and `compare` judges such a change on
compile elapsed instead — which is why every sample runs a warm pass, and why
you cannot turn it off. `references/measurement.md` explains which of the two
clocks to headline for which kind of change.

Then check what the probe cannot see:

- Run the project's own tests or entry points and diff the numeric output
  against the baseline.
- If you touched `enable_backward` or module boundaries, exercise a gradient
  path and confirm it still produces correct adjoints.
- If you touched `fast_math`, `max_unroll`, MathDx, or swapped an
  implementation, benchmark the steady-state runtime. Compile-time measurement
  cannot detect a kernel you made slower.
- Re-run uncovered dynamic kernels, dtypes, profiles, and modes.

### 6. Report results

Report before/after medians, sample count, fixed and declined mechanisms,
tradeoffs, and unverified behavior. Call results inside the noise band
inconclusive.

Size the result against the complaint the user actually made. A verified 3% on a
startup they called painful is a true number and a misleading headline. Give the
reduction and the residual together, and say what the residual is made of.

#### Separate what you found from what is there

Two statements sound alike and are not:

- *"I did not find a further reduction."* A claim about your search. Your
  measurement supports it.
- *"This compiles as fast as it can."* A claim about the code. Nothing you
  measured supports it.

The probe establishes something narrow, and it is worth stating exactly: no
module compiled twice, no block-dimension variants, no LTO. That rules out
*redundant* builds. It says nothing about whether each build had to be that
large, because the kernel set, the adjoint code, and the unroll budget decide
that, and only CS-10 and CS-11 reach any of it.

So "no structural churn" is a conclusion you earned. "This needs the time it
takes" is not. Necessary, irreducible, optimal, and genuinely needed all assert
the second thing, and they close an investigation your evidence did not close.
Give the bound of the search instead: what you checked, the signal that ruled
each mechanism out, and what you never reached — including anything you ran out
of budget for. Carry the per-module breakdown so nobody has to remeasure, and
where one module dominates, name what would shrink it — fewer kernels reached per
launch, backward off where nothing differentiates, an upstream split — even when
you cannot do it from where you are standing.

Finding nothing is a real result and worth reporting well. It costs the reader
one page and saves them a week of looking. Inventing a small change so the report
has a number costs them a review, a merge, and a maintenance burden, and it
buries the answer they needed.

The converse overclaim is just as costly, so do not let a well-argued negative
stand in for a fix that was available. Before concluding that nothing is, check
that you reached past the mechanisms that remove work: CS-13 overlaps the builds
a workload genuinely needs, costs no library patch, and is therefore often the
last lever standing when everything else is ruled out.

#### Measure the levers you are not going to pull

The tempting shortcut, for an option that is blocked or that you have decided
against, is to reason about it and report the conclusion: *splitting those
modules would not help*, *backward is probably worth a couple of seconds*.
Predictions about compile cost are unreliable, and module-level ones fail in a
particular direction. The module is the compilation unit, so a run that launches
one kernel from a large module still pays for every kernel in it; splitting a
module along what a workload actually uses can remove most of that workload's
cost while removing no work at all. "It would not reduce total work" is true
across all workloads and irrelevant to the one you were asked about.

Measuring an option costs one probe run and a revert — you are not shipping it,
so the usual safety objections do not apply to measuring. Do that before telling
someone a door is not worth opening. When you genuinely cannot, label the number
an untested estimate and say why you did not get it.

When one module dominates and you are weighing whether to split it, measure the
fixed floor first: compile a throwaway one-kernel module with the same options
and see what it costs. That number is what any split has to pay again per new
module, so it bounds the answer before you restructure anything. One measured
case put the floor at 167 ms of a 467 ms module, which capped a three-way split
at roughly 20% and settled the question in a single probe run.

#### Leave a ledger of what you did not take

What you measured and declined is a result, and it is the part most easily lost.
Collect it in one table, with the same columns every time, so a reader can act on
it without reconstructing your reasoning:

| Option | Measured | Why not taken | To take it |
| --- | ---: | --- | --- |
| `enable_backward=False` on `pkg.solver` | −38% cold | a live tape traverses these kernels | set at the entry point, then re-check adjoints |
| `max_unroll=4` | −2%, inside noise | changes generated code for no measured gain | — |

A declined option carrying a number is worth more than an applied one without,
because that number is what lets somebody else make a call you were not placed to
make. The same table is where an option you *did* take belongs whenever you could
not verify it at the layer it affects: a runtime knob checked only against
compile time is an open question, not a result, and folding it into the headline
is how it stops being one.

#### When the user's constraints close every door

A pinned dependency, a required capability, or an environment with no persistent
storage can rule out every fix you would otherwise apply. That is not "nothing to
report," and it is not an invitation to relitigate a decision the user has
already made.

Price the constraints instead. Measure what relaxing each one would buy and put
the numbers side by side: persisting the cache is worth 99%, patching the pinned
library 25%, the capability you need 35%. Constraints are often set without
anyone knowing their cost, and a quantified menu respects the decision while
giving the user what they need to revisit it. Read each constraint for its intent
as well — a rule against persisting state between jobs may still permit a
prewarmed artifact built once and shipped read-only.

If nothing is available and no constraint looks worth reopening, say that
plainly, and say what would have to change for the answer to be different.

## Examples

Three shapes in the module log, in the columns the probe prints: time, device,
block dimension, hash, name.

**Identity churn** — one name, one block dimension, three hashes. Something
changed the module after it first loaded (CS-1, CS-3, CS-6):

```
  312.44  cuda:0   256   6eb476a   app.filters
  289.10  cuda:0   256   17e6601   app.filters
  244.87  cuda:0   256   851c7d0   app.filters
```

**Block-dimension duplication** — one name, one hash, three block dimensions.
Every kernel built three times, including those only ever launched at one of them
(CS-5):

```
  361.16  cuda:0   128   9004220   app.stages
  357.95  cuda:0   256   9004220   app.stages
  340.02  cuda:0   512   9004220   app.stages
```

**Nothing structural** — each module built once, one hash and one block dimension,
no LTO, whether that is one module or eleven:

```
 1772.27  cpu      1     b1faa06   __main__
```

The first two cost the same three builds and are different problems, which is why
the hash column decides between them. The third means no row of the diagnosis
table fires; step 6 covers what that does and does not let you conclude, and CS-2
covers it for ephemeral processes.

## Troubleshooting

Run the target command directly before debugging the probe. See
`references/measurement.md` for cache/noise issues and
`references/mechanisms.md` for mechanism-specific failures.

## Limitations

Two rules override any compile-time gain:

- Cold measurement is valid only with an isolated cache. The probe sets
  `WARP_CACHE_PATH` per run; measuring by hand without a fresh cache directory
  per cold sample can reuse old artifacts.
- Keep `max_workers <= 1` for any module load that can target the CPU:
  `device="cpu"`, `device=None`, or a mixed device list. Warp's native CPU JIT
  first-use path is not synchronized, so concurrent first loads can report
  success and then fail kernel lookup. Do not paper over it with retries.
  CUDA-only loading is not affected.

Measurements are environment-specific: cold times move with CPU, GPU, driver,
toolchain, and Warp version, so the mechanisms transfer but the numbers do not.
Block-dimension variants and MathDx/LTO costs are CUDA-only, and at least one
fix that helps on CUDA is a measured regression on CPU. Unproven areas are
listed at the end of `references/mechanisms.md`; do not present those as
established.

## Reference files

- `references/mechanisms.md`: the twelve compile-time mechanisms, each with
  its confirming signal, fix, applicability limits, and failure mode. Read the
  sections your measurement points to.
- `references/measurement.md`: measurement protocol, what each metric does
  and does not mean, and how to measure by hand where the probe does not fit.
