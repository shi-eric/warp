# NVSHMEM Automatic Differentiation

**Status**: Implemented on branch `ershi/nvshmem-device`, 2026-10-31.
Companion to [NVSHMEM Integration](nvshmem-integration.md), which covers the forward built-ins and the
runtime integration this document builds on.

## Summary

Warp kernels can move data between GPUs with the `wp.nvshmem_*` built-ins, but gradients do not flow
through them. Every NVSHMEM built-in is registered as non-differentiable, and Warp's code generator
emits no reverse statement for a non-differentiable built-in, so a kernel that pushes a halo row to a
neighbor compiles with `enable_backward=True`, runs its backward pass, and reports a zero gradient
through the exchange without any diagnostic.

This document designs reverse-mode differentiation for those built-ins. It has three parts.

1. **Values.** The data movers are copies between the memory of two PEs, and the adjoint of a copy is a
   copy in the other direction. A `put` from PE a into PE b differentiates as a pull of the destination's
   gradient from PE b into the source's gradient on PE a. Signals, waits, fences, barriers and PE queries
   carry no gradient.
2. **Placement.** A gradient array that another PE reads must live in NVSHMEM symmetric memory, like the
   forward array it belongs to. Users allocate it collectively and attach it as `array.grad`.
3. **Schedule.** The backward of a one-to-many signal is a many-to-one completion: the producer may read
   the ghost row's gradient only after every consumer thread on the other GPU has finished accumulating
   into it. No thread can announce that, and the only grid-level completion event Warp has is the end of
   a kernel. The schedule is therefore handled at launch granularity: Warp launches a one-thread barrier
   kernel before every adjoint launch whose generated code touches remote gradient memory, and the tape
   adds one after the last. Every in-kernel synchronization built-in then has a no-op adjoint, and
   signal-based forward protocols need no changes for the backward pass.

The pull family, `nvshmem_g` and `nvshmem_get`, stays non-differentiable in this version. Its adjoint
is a remote accumulation, NVSHMEM has no floating-point atomic add, and NVSHMEM atomics mixed with local
atomics on the same address are undefined behavior.
Section [The pull family](#the-pull-family-nvshmem_g-and-nvshmem_get) specifies the diagnostic users
get instead and the follow-up design.

Appendix A defends the reverse operation of every forward built-in. Appendix B proves the barrier
schedule sufficient and shows by counterexample that each barrier is necessary. Appendix C quotes the
NVSHMEM documentation this design relies on.

## The Running Example

A two-PE Jacobi iteration with one halo row on each side of the local block. Each PE owns `n_rows`
interior rows; row 0 and row `n_rows + 1` are ghost rows written by the neighbors. The kernel computes
the next field from the current one and pushes its first and last owned rows into the neighbors' ghost
rows with `wp.nvshmem_p`. The loss is the sum of squares of the final field's interior on every PE, and
the quantity wanted is the gradient of the loss with respect to the initial field.

```python
@wp.kernel
def jacobi_step(
    u: wp.array[wp.float32],  # field at step k, including ghost rows
    u_new: wp.array[wp.float32],  # field at step k + 1, symmetric
    nx: int,
    n_rows: int,
    top_pe: int,
    top_halo_row: int,
    bottom_pe: int,
    bottom_halo_row: int,
):
    tid = wp.tid()
    iy = tid // (nx - 2) + 1
    ix = tid % (nx - 2) + 1

    v = 0.25 * (u[(iy - 1) * nx + ix] + u[(iy + 1) * nx + ix] + u[iy * nx + ix - 1] + u[iy * nx + ix + 1])
    u_new[iy * nx + ix] = v

    if iy == 1:
        wp.nvshmem_p(u_new, top_halo_row * nx + ix, v, top_pe)
    if iy == n_rows:
        wp.nvshmem_p(u_new, bottom_halo_row * nx + ix, v, bottom_pe)


@wp.kernel
def interior_loss(u: wp.array[wp.float32], nx: int, n_rows: int, loss: wp.array[wp.float32]):
    tid = wp.tid()
    iy = tid // (nx - 2) + 1
    ix = tid % (nx - 2) + 1
    wp.atomic_add(loss, 0, u[iy * nx + ix] * u[iy * nx + ix])


@wp.kernel
def barrier_all():
    """One thread per PE, which is what the device-side collective requires."""
    wp.nvshmem_barrier_all()
```

The host side allocates one symmetric field per step, with a symmetric gradient for each, records the
forward pass on a tape, and runs it backward. The forward pass synchronizes between steps with a
stream-ordered barrier, which is the user's responsibility and is not recorded on the tape.

```python
def symmetric_array(shape, dtype, requires_grad=False):
    """Allocate a zeroed symmetric array. Every PE must call this in the same order."""
    nbytes = int(np.prod(shape)) * wp.types.type_size_in_bytes(dtype)
    buf = nvshmem.buffer(nbytes)
    arr = wp.array(ptr=int(buf.handle), shape=shape, dtype=dtype, device=device)
    arr.zero_()
    if requires_grad:
        grad_buf = nvshmem.buffer(nbytes)
        arr.grad = wp.array(ptr=int(grad_buf.handle), shape=shape, dtype=dtype, device=device)
        arr.grad.zero_()
    return arr


fields = [symmetric_array((alloc_rows * NX,), wp.float32, requires_grad=True) for _ in range(N_STEPS + 1)]
loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
init_field(fields[0])

with wp.Tape() as tape:
    for k in range(N_STEPS):
        wp.launch(jacobi_step, dim=n_rows * (NX - 2), inputs=[fields[k], fields[k + 1], NX, n_rows, *neighbors])
        wp.launch(barrier_all, dim=1)  # forward sync between steps, not recorded on the tape
    wp.launch(interior_loss, dim=n_rows * (NX - 2), inputs=[fields[N_STEPS], NX, n_rows, loss])

tape.backward(loss)
gradient = fields[0].grad.numpy()
```

With this design, `tape.backward(loss)` issues the following on every PE, in stream order. The lines
marked "inserted" are what Warp adds; nothing else changes for the user.

```
seed loss.grad = 1
adjoint of interior_loss                      (local only, no barrier)
barrier                                       (inserted: adjoint of jacobi_step pulls remote gradients)
adjoint of jacobi_step, k = N_STEPS - 1
barrier                                       (inserted)
adjoint of jacobi_step, k = N_STEPS - 2
...
barrier                                       (inserted)
adjoint of jacobi_step, k = 0
barrier                                       (inserted by Tape.backward: trailing)
```

The adjoint of `jacobi_step` on PE a, for the thread with `iy == 1`, contains the statement

```
adj_v += wp.nvshmem_g(adj_u_new, top_halo_row * nx + ix, top_pe)
```

which pulls from PE `top_pe` the gradient that its adjoint of step k + 1 accumulated into the ghost cell,
and adds it to the local gradient of `v`. From there the gradient flows into `adj_u` through the
stencil's own adjoint, exactly as on one GPU.

## Goals, Non-Goals, Anti-Goals

Goals:

- Correct gradients through push-based exchange (`nvshmem_p`, `nvshmem_put`, `nvshmem_put_signal`)
  under the usage rules in [Usage Rules](#usage-rules), on NVLink-only systems and on systems with
  InfiniBand.
- Forward kernels that synchronize with signals, waits, fences, quiet or barriers need no changes to be
  differentiated.
- Works through `Tape.backward()` and through manual `wp.launch(..., adjoint=True)`, and can be captured
  into a CUDA graph.
- Every case this design does not handle produces a diagnostic instead of a silent zero gradient.

Non-goals for this version:

- Differentiating the pull family, `nvshmem_g` and `nvshmem_get`. A follow-up design is sketched in
  [Future Work](#future-work).
- Cross-PE dependencies within a single launch. See [The Unsupported Pattern](#the-unsupported-pattern).
- Allocating symmetric memory on the user's behalf. The integration design rejected Warp-owned symmetric
  allocations because `nvshmem_malloc` and `nvshmem_free` are collectives that cannot follow Python
  object lifetimes; that reasoning applies unchanged to gradient arrays.
- Neighbor-only synchronization in the backward pass. The global barrier is the first version; the
  point-to-point refinement is in [Future Work](#future-work).
- Block-cooperative (`tile_nvshmem_*`) variants, which do not exist yet.
- Configurations where several PEs share one physical GPU, which NVSHMEM calls multiple processes per
  GPU (MPG), including MIG slices of one GPU. Without CUDA MPS, NVSHMEM supports only a limited API set
  there that excludes device-side synchronization. One PE per GPU is the target.

Anti-goals:

- Any configuration that is accepted without diagnostic and produces wrong gradients.
- Changing behavior or adding warnings for programs that never run a backward pass. Every diagnostic in
  this design fires at adjoint launch or during adjoint code generation of a kernel that actually uses a
  remote built-in.
- Hidden collective operations whose timing depends on Python object lifetimes.

## Background

### How Warp differentiates a kernel

Warp generates two device functions per kernel: the forward kernel, and an adjoint kernel that first
replays the forward body to recover intermediate values and then executes the reverse statements in
reverse program order. Each array argument has an adjoint twin: `adj_u` is `u.grad` when the array has
one and a null array otherwise. Reverse statements accumulate into adjoint arrays with atomics, and the
native adjoint helpers in `warp/native/array.h` skip the accumulation when the adjoint array is null:

```cpp
if (adj_buf.data)
    atomic_add(&index(adj_buf, i), adj_value);
```

Built-ins declare two flags that matter here. `skip_replay=True` comments the call out of the replay
section, so a side effect such as an atomic or a remote write happens once, in the forward kernel, and
not again in the adjoint kernel's replay. `is_differentiable=False` makes the code generator emit no
reverse statement for the call. All NVSHMEM built-ins carry `is_differentiable=False` today, and all
side-effecting ones carry `skip_replay=True`. When a built-in is differentiable, the generator emits a
call to a native function named `adj_<name>` with the forward arguments, the adjoints of the arguments,
and the adjoint of the return value, resolved by C++ overloading.

### What NVSHMEM guarantees

The facts below are verified against the NVSHMEM documentation; Appendix C has the quotes.

- `nvshmem_quiet` completes the operations issued before the call, and `nvshmem_fence` orders them
  per destination PE. Both enforce memory-store ordering only for the calling thread. Operations issued
  by other threads are covered only if they were issued before the call, which requires synchronizing
  those threads first.
- `nvshmem_barrier_all` is a collective that every PE must call. On the device it is thread-scoped: it
  must be called by exactly one thread per PE at a time, and concurrent calls from several threads in one
  PE are not allowed. Before synchronizing, it completes all previously issued stores and remote memory
  updates, so it subsumes `nvshmem_quiet`.
- NVSHMEM atomics are exclusive only with respect to other NVSHMEM atomics of the same datatype.
  Concurrent access to the same location by non-NVSHMEM operations, including local loads, stores and
  CUDA atomics, is undefined behavior. `atomic_add` is defined only for integer types; `float` and
  `double` have fetch, set and swap, but no add.
- On systems with InfiniBand, the synchronization operations guarantee visibility of updates to the local
  PE's symmetric objects only. On NVLink-only systems they guarantee global visibility.
- Kernels that use device-side synchronization or collectives should be launched with NVSHMEM's
  collective launch to guarantee deadlock-free progress; Warp launches with `cuLaunchKernel`.
- Collective operations on the same team must not execute in parallel, for example a host-side
  `nvshmem_malloc` concurrently with a barrier enqueued on a stream.

Two consequences shape the design. A multi-block Warp kernel cannot synchronize PEs from inside: only
one thread may call the barrier, Warp has no grid-wide synchronization to tell the other threads it
completed, and the barrier only covers operations the calling thread can see. And the natural adjoint of
a remote read, a remote accumulate, has no safe primitive for floating-point gradients.

## The Problem

Differentiating a kernel that communicates has three sub-problems.

**Values.** What is the reverse statement for each built-in. This is the easy part: the data movers are
copies, and Appendix A derives each adjoint from the chain rule.

**Placement.** Where the gradients live. The reverse of a remote write reads a gradient that belongs to
another PE, so that gradient must be addressable through NVSHMEM, which means symmetric memory.

**Schedule.** When the reverse statements may run. On one GPU, stream order guarantees that a backward
launch sees every gradient contribution made by the backward launches before it. Across PEs, stream
order on PE a says nothing about progress on PE b. The adjoint of a put on PE a reads a gradient that
threads on PE b are still accumulating, unless something makes PE a wait.

### Why mirroring the forward protocol fails

The first idea is to invert the synchronization along with the data: the adjoint of
`nvshmem_signal_wait_until` on the consumer notifies the producer, and the adjoint of
`nvshmem_put_signal` on the producer waits for that notification before pulling. It fails for two
independent reasons.

1. **The consumer does not know the producer.** A signal wait takes a local flag and a comparison. The
   flag was written by whichever PE chose to write it; the wait has no `pe` argument, so its adjoint has
   nowhere to send a notification. Polling the consumer's flag from the producer side avoids this, but
   not the second problem.
2. **Completion is many-to-one.** The forward wait is executed by every consumer thread, and each of
   those threads contributes to the ghost cell's gradient in the backward. The producer may pull only
   after the last of them is done. A thread that finishes its own accumulation cannot know whether the
   other consumer threads are done, or even scheduled. Counting schemes do not close the gap: a counter
   incremented when a thread registers in the replay and decremented when it completes in the reverse
   reads zero while unscheduled blocks have not registered yet. Only a grid-level event marks the end of
   the accumulation.

Monotone signal values make the mirror worse, since a protocol that waits for `flag >= iteration` in the
forward is satisfied immediately by stale values when the backward runs iterations in reverse order. The
conclusion is that the backward schedule must come from launch boundaries, not from in-kernel protocol.

## Design

### Adjoint rules

| Forward                                             | Reverse                                                                   | Replay   |
| --------------------------------------------------- | ------------------------------------------------------------------------- | -------- |
| `nvshmem_p(dest, i…, value, pe)`                    | `adj_value += nvshmem_g(adj_dest, i…, pe)`                                | skipped  |
| `nvshmem_put(dest, source, pe)`                     | for each element `n`: `adj_source[n] += nvshmem_g(adj_dest, n, pe)`       | skipped  |
| `nvshmem_put_signal(dest, source, sig, i, s, op, pe)` | as `nvshmem_put`; the signal arguments carry no gradient                | skipped  |
| `nvshmem_g(source, i…, pe)`                         | none in this version, see [The pull family](#the-pull-family-nvshmem_g-and-nvshmem_get) | replayed |
| `nvshmem_get(dest, source, pe)`                     | none in this version                                                      | skipped  |
| `nvshmem_signal_op`, `nvshmem_signal_wait_until`    | none                                                                      | skipped  |
| `nvshmem_quiet`, `nvshmem_fence`, `nvshmem_barrier_all` | none                                                                  | skipped  |
| `nvshmem_my_pe`, `nvshmem_n_pes`                    | none                                                                      | replayed |

Three built-ins change from `is_differentiable=False` to `True` and gain native adjoints in
`warp/native/nvshmem_device.h`. Their `skip_replay=True` stays, so the forward write happens once.
Everything else keeps `is_differentiable=False`, which is the correct no-op adjoint, and the pull family
gains a launch-time diagnostic.

The adjoint of `nvshmem_p` mirrors its four index overloads:

```cpp
template <typename T>
inline CUDA_CALLABLE_DEVICE void adj_nvshmem_p(
    const array_t<T>& dest, int i, T value, int pe,
    const array_t<T>& adj_dest, int& adj_i, T& adj_value, int& adj_pe)
{
    if (adj_dest.data)
        adj_value += nvshmem_detail::get_element(&index(adj_dest, i), pe);
}
```

`get_element` is the forward helper that `nvshmem_g` uses: it moves the element through the
unsigned-integer NVSHMEM routine sized to the element's alignment, in several parts for elements wider
than 8 bytes, so any dtype that the forward supports is supported here. The adjoint of an integer-typed
element adds remote zeros, which Warp's other adjoints also do for integer arrays; the null check mirrors
the convention in `array.h`. The blocking `g` returns when the value is local, so no completion step is
needed after it.

The adjoint of `nvshmem_put` pulls one element at a time and accumulates locally:

```cpp
template <typename T>
inline CUDA_CALLABLE_DEVICE void adj_nvshmem_put(
    const array_t<T>& dest, const array_t<T>& source, int pe,
    const array_t<T>& adj_dest, const array_t<T>& adj_source, int& adj_pe)
{
    if (!adj_dest.data || !adj_source.data)
        return;
    const size_t count = nvshmem_detail::element_count(source);
    for (size_t n = 0; n < count; ++n) {
        const T g = nvshmem_detail::get_element(nvshmem_detail::element_at(adj_dest, n), pe);
        atomic_add(nvshmem_detail::element_at(adj_source, n), g);
    }
}
```

The local accumulation is atomic because other threads of the same adjoint kernel may add to
`adj_source` too. The per-element pull is one round trip per element from a single thread, which is slow
over InfiniBand. A bulk `nvshmem_getmem` would overwrite rather than accumulate, and staging through
thread-local memory is not a valid NVSHMEM destination, so the first version accepts the cost. The
forward `nvshmem_put` from a single thread is already documented in the integration design as the
performance trap that block-cooperative variants will fix; their adjoints will pull cooperatively.

`adj_nvshmem_put_signal` has the signature of the forward plus the adjoints of all arguments, ignores the
signal arguments, and forwards the data arguments to `adj_nvshmem_put`.

### The pull family: `nvshmem_g` and `nvshmem_get`

The reverse of `x = nvshmem_g(source, i, pe)` is `adj_source[i] on pe += adj_x`, a remote accumulation
into another PE's gradient array. The reverse of `nvshmem_get` is the same per element. Three facts
make this unsafe to ship in this version.

1. **No primitive.** NVSHMEM defines `atomic_add` only for integer types. A floating-point accumulate
   has to be emulated with a compare-and-swap loop on the bit pattern, `nvshmem_int_atomic_compare_swap`
   for `float32` and `nvshmem_long_atomic_compare_swap` for `float64`, each attempt a fetch and a CAS
   round trip. Element types without an 8-byte-or-smaller bit pattern decompose into components.
2. **Undefined when mixed.** NVSHMEM atomics are exclusive only against other NVSHMEM atomics of the same
   datatype. If the owning PE also accumulates into the same gradient element with a local atomic in the
   same adjoint launch, the behavior is undefined. In a pull-based stencil that is the common case: PE b
   reads its own boundary row locally in the same kernel in which PE a fetches it.
3. **The owner does not know.** The gradient lands in an array that PE b later reads with a plain load.
   Only the barrier schedule can separate the remote writes from those loads, and it can do so only
   across launches, not within one.

Decision for this version: `nvshmem_g` and `nvshmem_get` stay non-differentiable. The code generator
records, per kernel, that the adjoint contains a call to one of them. When such a kernel is launched
with `adjoint=True`, Warp warns once per kernel with this text, which Warp's logger prints after its
`Warp UserWarning: ` prefix:

```
gradients do not flow through wp.nvshmem_g() or wp.nvshmem_get() in kernel "fetch_halo";
restructure the exchange so the owning PE pushes the data with wp.nvshmem_p() or wp.nvshmem_put().
```

The warning fires at adjoint launch rather than at code generation so that forward-only programs that
use `nvshmem_g` with the default `enable_backward=True` see nothing.

The restructuring the warning asks for is mechanical. A pull-based halo exchange

```python
@wp.kernel
def fetch_halo(u: wp.array[wp.float32], nx: int, n_rows: int, top_pe: int, top_last_row: int):
    ix = wp.tid() + 1
    u[0 * nx + ix] = wp.nvshmem_g(u, top_last_row * nx + ix, top_pe)  # pull neighbor's last row into my ghost row
```

becomes the neighbor pushing the same row into the same ghost cells:

```python
@wp.kernel
def push_halo(u: wp.array[wp.float32], nx: int, n_rows: int, bottom_pe: int):
    ix = wp.tid() + 1
    wp.nvshmem_p(u, 0 * nx + ix, u[n_rows * nx + ix], bottom_pe)  # push my last row into neighbor's ghost row
```

and the push is differentiable. The follow-up design for a true pull adjoint is in
[Future Work](#future-work); it needs the CAS emulation, an additional barrier after each launch that
performs remote accumulation, and a usage rule that no local accumulation into the same gradient array
happens in the same launch.

### Control built-ins

`nvshmem_signal_op`, `nvshmem_signal_wait_until`, `nvshmem_quiet`, `nvshmem_fence` and
`nvshmem_barrier_all` define no differentiable values. The wait returns the flag's value, an unsigned
integer, which Warp does not differentiate. What these built-ins do is constrain *when* operations
execute. The backward pass needs its own constraints, and the [barrier schedule](#the-schedule-launch-level-barriers)
provides them at a coarser granularity than any in-kernel protocol would, so the reverse statements are
empty. The existing `skip_replay=True` keeps the forward waits and barriers out of the adjoint kernel,
where they would otherwise block on flags that nobody sets a second time.

`nvshmem_my_pe` and `nvshmem_n_pes` are integer constants of the job and are replayed.

### Gradient placement

An array that is the `dest` of a `nvshmem_p`, `nvshmem_put` or `nvshmem_put_signal` must have a `.grad`
that is in symmetric memory on every PE, allocated in the same collective order as the forward array,
before the backward pass runs. `wp.array(ptr=..., requires_grad=True)` is wrong for such an array: it
allocates the gradient through Warp's allocator, which is not symmetric, and the remote pull in the
adjoint would translate a non-symmetric address. The pattern from the running example is the supported
one:

```python
buf = nvshmem.buffer(nbytes)
grad_buf = nvshmem.buffer(nbytes)  # same order on every PE
arr = wp.array(ptr=int(buf.handle), shape=shape, dtype=dtype, device=device)
arr.grad = wp.array(ptr=int(grad_buf.handle), shape=shape, dtype=dtype, device=device)
```

Warp validates this at adjoint launch for every flagged kernel, one whose adjoint contains any NVSHMEM
remote access, including a kernel whose only remote access is a replayed `nvshmem_g`. For each array
argument whose data pointer is symmetric, the gradient array the launch uses for the argument, that is,
the adjoint array passed for it, or else its `.grad`, must be symmetric too, otherwise the launch raises.
An argument with neither is skipped. Checking the array the launch passes matters for manual
`wp.launch(..., adjoint=True)` calls, whose `adj_inputs` take precedence over `.grad` in the native
adjoint. Symmetry is checked through the host library's `nvshmem_ptr(ptr, my_pe)`, which Warp binds with
`dlsym` next to the symbols it already resolves. The rule "symmetric data implies symmetric gradient" is
stricter than necessary, since a symmetric array used only locally in a kernel would be fine with a local
gradient, but it has no false negatives for the arrays that matter, and a user who allocated the data
symmetrically is already in the position to allocate the gradient the same way. The check is skipped when
NVSHMEM is not loaded and costs one host call per array argument otherwise.

The implementation confirmed that `nvshmem_ptr` returns null for an address outside the symmetric heap
([Open Questions](#open-questions), question 1), so the check ships.

Freeing follows the forward arrays: the user frees gradient buffers explicitly and collectively, after
the backward pass has completed on every PE. The trailing barrier that `Tape.backward()` enqueues
guarantees the completion once the user synchronizes the stream.

### The schedule: launch-level barriers

This section is the core of the design, and the part the user explicitly asked to see defended. It
states what must hold, derives the barrier placement from it, argues that the granularity cannot be
finer without machinery Warp does not have, and explains why the condition for inserting a barrier is as
broad as it is. Appendix B contains the proof and the counterexamples in full.

#### What must hold

Write x̄ for the gradient of the loss with respect to a value x. Two invariants make a backward pass
correct across PEs; on a single GPU, stream order gives both for free.

- **(R) Read after final.** When an adjoint statement on PE a reads x̄ held in PE b's memory, every
  contribution to x̄ has been made and is visible to PE a.
- **(W) Write before read.** When an adjoint statement on PE a writes x̄ held in PE b's memory, the write
  has completed and is visible before PE b, or the host, reads x̄.

This version only reads remote gradients (the adjoints of the push family pull), so (R) is the live
invariant; (W) becomes live with the pull-family follow-up and is covered by the same barriers.

#### Where contributions come from

The tape records launches L₁ … Lₙ in program order, and every PE records the same sequence (the SPMD
assumption, discussed below). The backward executes Aₙ … A₁, where Aₖ is the adjoint launch of Lₖ.

Suppose Lₖ on PE a puts a value into `dest` on PE b. Two usage rules, both of which single-GPU Warp
autodiff already imposes in spirit, pin down who contributes to that value's gradient:

- **U1, cross-launch consumption.** The value is read only by launches Lₘ with m > k. A read in the same
  launch is the unsupported pattern of the next section.
- **U2, write once.** The element is not overwritten before those reads. In practice: one array per
  time step, as every differentiable Warp simulation already does, because a reused buffer's gradient
  array conflates the gradients of different values on one GPU too.

Then every contribution to the ghost cell's gradient on PE b comes from the adjoints Aₘ with m > k,
executed on PE b (the reads were local loads on PE b). In backward program order, all of those precede
Aₖ on every PE. Invariant (R) for Aₖ on PE a therefore holds exactly when **every PE has finished
Aₖ₊₁ before PE a starts Aₖ**.

#### The barrier placement

A collective barrier enqueued on every PE's stream between Aₖ₊₁ and Aₖ gives precisely that: it
completes on PE a only when every PE has reached it, and reaching it in stream order means having
finished Aₖ₊₁ and everything before it. This yields the rule:

> **B1.** Before every adjoint launch of a *flagged* kernel, one whose generated adjoint contains a
> remote access, enqueue a barrier on the launch's stream.

The barrier before Aₖ is the same barrier as "after Aₖ₊₁", so placing it before each launch is the
minimal placement for the chain, and it has two properties the "after" form lacks. It covers the first
adjoint launch, whose remote reads depend on the other PEs having finished seeding and zeroing their
gradient arrays; those operations are stream-ordered before each PE's first barrier. And a manual
`wp.launch(..., adjoint=True)` gets the correct barrier without knowing what came before it.

One barrier is still missing at the end:

> **B2.** After the last adjoint launch of a backward pass, enqueue a barrier. `Tape.backward()` does
> this when any adjoint launch in the pass was of a flagged kernel. Programs that launch adjoints
> manually must do it themselves, and the documentation says so.

B2 exists for three reasons. First, after its own last adjoint launch, a PE goes on to zero or reseed
its gradient arrays for the next iteration, while the last adjoint launch of another PE may still be
pulling from them. The barrier before the last launch orders only the starts of the PEs' last launches,
not their ends. In the running example, PE a zeroes `fields[1].grad`, including the gradients of the
ghost cells that PE b's step 0 pushed into, while PE b's adjoint of step 0 may still be pulling them.
A replayed `nvshmem_g` in the last launch is exposed the same way to the next forward pass of the PE it
reads from.
Second, once the pull family gets an adjoint, remote writes from the last adjoint launch must land
before the host reads gradients. Third, freeing symmetric gradient buffers is collective and must follow
completion everywhere.

In the running example, the next forward pass does not race the replays. The adjoint of step k
replays the stencil's reads of `fields[k]`, not `fields[k + 1]`. The only adjoint launch whose replay
reads `fields[k + 1]`, where the next forward pass's step k puts, is the adjoint of step k + 1, and every
PE has finished it before the barrier ahead of the adjoint of step k completes, hence before any PE
starts its next forward pass. `fields[0]`, which the last adjoint launch replays, receives no puts.

Appendix B shows that B1 is also necessary: with any single B1 barrier removed, a slow PE b makes PE a
pull a gradient of zero in a two-PE, two-step instance of the running example.

#### Why launch granularity and a collective barrier

Three finer-grained alternatives were considered and rejected for this version.

- **Thread-level protocol in the adjoint kernel.** Rejected for the many-to-one reason in
  [Why mirroring fails](#why-mirroring-the-forward-protocol-fails): the event the producer needs is the
  completion of all consumer threads on another GPU, and no thread can announce it. The kernel boundary
  is the only grid-level completion event Warp has.
- **In-kernel `nvshmem_barrier_all` in the adjoint kernel.** The device barrier must be called by one
  thread per PE, that thread's `quiet` covers only operations it can see, and Warp has no grid-wide
  synchronization to make the other threads' accumulations visible to it first. Not implementable in a
  multi-block kernel.
- **Point-to-point progress polling.** The adjoint of a put knows its target PE, so it could poll a
  per-PE counter of completed adjoint launches on that PE and proceed when the neighbor has finished the
  previous one. This is sound for pull-only adjoints and deadlock-free, since each PE waits only on the
  other's *previous* launch, and it replaces the global barrier with neighbor-only traffic. It needs a
  symmetric counter per process, which Warp would have to allocate collectively at a well-defined point,
  and it does not cover remote writes. It is the first optimization to make once a measurement shows the
  global barrier matters; see [Future Work](#future-work).

A collective barrier is the right primitive for this version because it is the only one that covers both
invariants with no state, and because the forward passes this design targets already pay one barrier per
step.

#### Why the insertion condition is "every adjoint launch of a flagged kernel"

A natural instinct is to insert a barrier only when a given launch actually needs one. There are several
candidate conditions, and all but one are either unknowable at the host or unsound.

- *Per launch, by data flow.* Whether Aₖ needs the barrier depends on whether any PE's Aₖ₊₁ wrote a
  gradient that Aₖ reads remotely. That is a cross-PE dependency analysis over the tape, at runtime, on
  gradient arrays whose remote use is decided inside the kernel by data (`pe` is a kernel value). The
  host cannot compute it.
- *Per launch, by arguments.* Whether an array argument is a remote target is likewise decided inside the
  kernel. The host sees pointers.
- *Per module.* Too coarse: a module mixes NVSHMEM kernels with local ones, and local adjoint launches
  would pay for nothing.
- *Per kernel, by generated code.* The code generator knows statically whether the adjoint of a kernel
  contains any call that reads or writes remote gradient memory. That is exactly the set of launches
  that can violate (R) or (W), and it is the finest condition that is both sound and knowable before the
  launch.

So the condition is per kernel, decided at code generation, and persisted in the module's metadata so
that a kernel loaded from the kernel cache, where code generation does not run, still carries it. The
flag is "the adjoint kernel contains any NVSHMEM remote access", which covers two kinds of statement:
remote gradient accesses, which need B1 for invariant (R), and replayed remote reads of forward data
such as `nvshmem_g`, which need B2 so that the next forward pass on another PE does not overwrite what
the replay is still reading. One flag serves both rules; the barrier it adds before a pull-only kernel's
adjoint is unnecessary but harmless, and such kernels are already warned about. Within the flagged set
the barrier is unconditional: a flagged kernel's adjoint launch always gets one, whether or not this
particular launch's threads take the branch that communicates. A barrier that turns out to be
unnecessary costs a few microseconds; a missing one costs a silently wrong gradient, which is this
design's anti-goal.

A configuration flag, `wp.config.nvshmem_adjoint_barrier`, defaults to on and lets programs
that manage synchronization themselves, or that are not SPMD, turn the insertion off. The flag is read at
launch time so a program can scope it.

#### How the barrier is issued

The barrier is a one-thread Warp kernel, owned by Warp, that calls `wp.nvshmem_barrier_all()`, launched
with `dim=1` on the adjoint launch's stream immediately before it:

```python
@wp.kernel
def _nvshmem_barrier_all():
    wp.nvshmem_barrier_all()
```

This satisfies the one-thread-per-PE rule of the device collective, needs no new host binding, lives in
a module that the existing machinery already registers with NVSHMEM, and is capturable into a CUDA graph
because it is an ordinary kernel launch. A one-block grid is always resident, so the deadlock concern
that NVSHMEM's collective launch addresses does not apply to it. The device barrier completes operations
issued before it, and in stream order the previous adjoint kernel's operations were issued before the
barrier kernel started.

The alternative is the host library's `nvshmemx_barrier_all_on_stream`, bound with `dlsym`. It is
stream-ordered, but the documentation does not say whether it can be captured into a CUDA graph, and
`Tape.backward()` is routinely captured. The kernel form is the default; the host form is the fallback
if the kernel form shows a problem on a transport.

The barrier launch is a forward launch, so it does not recurse into the insertion logic. It runs on the
stream the adjoint launch was given, which is the device's current stream unless the caller passed one.

**Where the insertion lives.** Warp enqueues adjoint kernels from more than one place: the
`wp.launch(..., adjoint=True)` path that `Tape.backward()` uses, `Launch.launch()` for launch objects
created with `record_cmd=True`, and the deterministic-mode launcher. The insertion goes at the single
point where the adjoint kernel is actually enqueued on a stream, so every path gets it, and it happens
before the early return that `wp.launch` takes for a zero-sized `dim`. That last point matters: a PE
whose local block is empty in some step still has to arrive at the barrier, otherwise the other PEs hang.
Every PE enqueues the barrier for every flagged adjoint launch, regardless of its local work.

**Module loading and CUDA graph capture.** The barrier kernel lives in a Warp-owned module that is
compiled, loaded and registered with NVSHMEM like any other, and that work is deferred to the first
flagged adjoint launch, so that programs which never run a backward pass never pay for it. If that
first launch happens inside a stream capture, the module's load would have to run mid-capture, and
NVSHMEM's module registration performs synchronous device writes whose interaction with an active
capture is not something this design can settle from documentation. The first version takes the
conservative route: when the barrier module is not yet loaded and the launch stream is capturing, Warp
raises an error that tells the user to run one backward pass before capturing. That matches the
warm-up iteration most Warp graph-capture code already performs. Whether registration can run safely
under a relaxed capture mode is an open question to settle on hardware; if it can, the error becomes a
mid-capture load and no user code changes.

#### Assumptions and their failure modes

- **SPMD.** Every PE records the same sequence of adjoint launches of flagged kernels. If one PE launches
  a flagged kernel's adjoint and another does not, the barrier never completes and the program hangs,
  which is how every NVSHMEM collective fails. Programs with role-specialized PEs should turn the
  insertion off and synchronize by hand.
- **No concurrent collectives.** NVSHMEM forbids two collectives on the same team executing in parallel,
  for instance a host-side symmetric allocation while a barrier is pending on a stream. With this design
  a `tape.backward()` leaves barriers pending on the stream until it is synchronized, so symmetric
  allocation and freeing must wait for that synchronization. The user's own forward barriers already
  impose this; it is documented, not new.
- **Replay reads forward data.** The adjoint kernel's replay re-reads ghost cells with plain loads. U2
  keeps that data intact until B2 completes.
- **Validation after the barrier.** The barrier is enqueued before the launch's own validation, such as
  the grid-size checks, argument packing and the lookup of the backward kernel. A validation error that
  occurs on only one PE therefore leaves the other PEs waiting in the barrier instead of failing cleanly.
  Under SPMD every PE fails the same way.
- **Cost.** One one-thread kernel launch and one device-side all-PE barrier per flagged adjoint launch.
  On an NVLink node the barrier is expected to take on the order of ten microseconds; over InfiniBand
  tens. Backward launches of simulation kernels typically run tens to hundreds of microseconds, and the
  forward pass already pays one barrier per step. [Open Questions](#open-questions), question 2, records
  the measured number; it is not a design input.

### The unsupported pattern

A kernel in which a thread waits for data that another PE puts *in the same launch*:

```python
@wp.kernel
def fused_step(u: wp.array[wp.float32], sig: wp.array[wp.uint64], nx: int, n_rows: int, pe: int, step: int):
    tid = wp.tid()
    ...
    if iy == n_rows:
        wp.nvshmem_put_signal(u, ..., sig, 0, wp.uint64(step), wp.NVSHMEM_SIGNAL_SET, pe)   # write to neighbor
    wp.nvshmem_signal_wait_until(sig, 0, wp.NVSHMEM_CMP_EQ, wp.uint64(step))               # wait for neighbor
    v = ... u[0 * nx + ix] ...                                                             # read what neighbor wrote
```

In the forward this needs every block of every PE to be resident, otherwise a resident block spins on a
flag that a non-resident block on the other PE would have set; NVSHMEM's guidance is to use its
collective launch for such kernels, which Warp does not offer. In the backward, the gradient of the
ghost cell read in this launch is accumulated in this same adjoint launch, so the producer's adjoint
pulls a value that is still changing, and the barrier schedule cannot help because the dependency is
inside a launch.

This design declares the pattern unsupported for differentiation. The code generator records when a
kernel contains a remote write textually before a signal wait, which is the shape the fused pattern
takes, counting `nvshmem_p`, `nvshmem_put`, `nvshmem_put_signal` and `nvshmem_signal_op` as remote writes,
so it may also warn on a kernel that acknowledges with `nvshmem_signal_op` and then waits for data from
an earlier launch. Warp warns once per kernel when such a kernel is launched with `adjoint=True`:

```
kernel "fused_step" writes to a remote PE before waiting on a signal; a cross-PE dependency
within one launch is not supported by automatic differentiation. Move the wait to the start of the
next launch.
```

The warning is deferred to the adjoint launch for the same reason as the pull-family warning: a
forward-only program that runs the fused shape on a co-resident grid is legitimate and must not see an
autodiff diagnostic on every compile. The record is persisted in module metadata next to the other
per-kernel NVSHMEM records.

The pipelined form, in which the wait at the start of step k + 1 consumes the data that step k pushed, is
fully supported, and is also the form that does not need co-residency in the forward.

### Usage rules

Collected here; U1 to U6 are motivated above, and U7 after the table.

| Rule | Statement                                                                                       | Consequence of breaking it                                   |
| ---- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| U1   | Data a PE puts in launch k is read only in launches after k.                                    | Stale gradient; adjoint-launch warning for the common shape. |
| U2   | One array per time step; no element written twice before its reads.                             | Conflated gradients, as on one GPU.                          |
| U3   | Arrays that are remote write targets have symmetric `.grad` arrays, allocated collectively.     | Raised at adjoint launch.                                    |
| U4   | Every PE runs the same sequence of adjoint launches (SPMD), or turns the barrier insertion off. | Hang in the barrier.                                         |
| U5   | No host-side symmetric allocation or free while a backward pass is pending on the stream.       | Undefined behavior per NVSHMEM.                              |
| U6   | Manual adjoint launches add a barrier after the last one.                                       | Another PE's last launch pulls zeroed or freed gradients.    |
| U7   | Differentiable kernels call NVSHMEM built-ins in the kernel body, not in `@wp.func` helpers.    | The adjoint's replay runs the helper's calls again.          |

U7 follows from how Warp replays user functions. `skip_replay` keeps a built-in out of the replay only
where the call itself appears; the adjoint kernel's replay of a `@wp.func` call runs the helper's forward
body, built-ins included, as for any user function. A `nvshmem_signal_op` with `NVSHMEM_SIGNAL_ADD` inside
a helper would therefore be applied twice, and a wait or a push inside a helper would run again during the
backward pass.

### Diagnostics

| Situation                                                             | When              | Form                     |
| --------------------------------------------------------------------- | ----------------- | ------------------------ |
| Kernel adjoint contains `nvshmem_g` or `nvshmem_get`                  | adjoint launch    | warning, once per kernel |
| Symmetric array with non-symmetric gradient array in a flagged kernel | adjoint launch    | `RuntimeError`           |
| Remote write before a signal wait in one kernel                       | adjoint launch    | warning, once per kernel |
| Barrier module not loaded and the launch stream is capturing          | adjoint launch    | `RuntimeError`           |
| `enable_backward=False` on a recorded NVSHMEM kernel                  | `Tape.backward()` | existing warning         |

### Changes by component

The implementation plan will detail these; this list fixes the shape.

- `warp/_src/builtins.py`: `is_differentiable=True` for `nvshmem_p`, `nvshmem_put`, `nvshmem_put_signal`.
- `warp/native/nvshmem_device.h`: `adj_nvshmem_p` (four overloads), `adj_nvshmem_put`,
  `adj_nvshmem_put_signal`.
- `warp/_src/codegen.py`: per-kernel records of (a) any remote access in the adjoint kernel, (b) pull
  built-ins present, and (c) the remote-write-before-wait shape.
- `warp/_src/context.py`: persist the per-kernel records in module metadata next to the existing NVSHMEM
  record so cached modules carry them; barrier insertion at the single point where adjoint kernels are
  enqueued, ahead of the zero-size early return; the capture-time load check; the pull-family and
  fused-shape warnings; symmetric-gradient validation; the config flag.
- `warp/native/nvshmem.cpp`: bind `nvshmem_ptr` and `nvshmem_my_pe` from the host library and expose
  `wp_nvshmem_is_symmetric(ptr)`.
- `warp/_src/tape.py`: trailing barrier when a backward pass issued at least one.
- The internal barrier kernel in a lazily imported module, because `warp/_src/nvshmem.py` is imported by
  `context.py` at module level and cannot define kernels without a circular import.
- Tests, documentation (a section in the NVSHMEM docs page and the differentiability notes), a
  differentiable example next to the existing Jacobi example, and a changelog fragment.

## Testing

All multi-PE tests run through the existing `mpirun -np 2` harness in `warp/tests/distributed/`, which
skips when NVSHMEM, nvshmem4py, mpi4py or a second GPU is missing. They are new modules registered in
`default_suite`.

- **Adjoint rules.** PE 0 puts a slice into PE 1 with each of `nvshmem_p`, `nvshmem_put` (packed, strided
  and two-dimensional) and `nvshmem_put_signal`. PE 1 fills the destination's gradient with known values.
  Both PEs launch the adjoint. PE 0 checks that the source's gradient equals the seeded values plus its
  existing contents, with `np.testing.assert_allclose`. A variant leaves the destination gradient
  unattached and checks that the source gradient is untouched.
- **Signals are no-ops.** A kernel using `put_signal` and `signal_wait_until` in the pipelined form
  differentiates to the same gradient as the plain-put version.
- **Schedule.** The running example on two PEs for several steps, with the loss on the interior of each
  PE. The reference is the same stencil run on one GPU over the full domain with a plain Warp kernel; its
  gradient of the initial field is sliced per PE and compared with `assert_allclose`. The two-PE run adds
  an artificial delay kernel on PE 1 before each adjoint launch, so a missing or misplaced barrier shows
  up as a mismatch rather than passing by luck. The same script, run with the barrier flag off, is the
  negative control and is expected to fail; it is kept as a diagnostic script, not asserted in CI, since
  a race that does not manifest is not a test failure.
- **Graph capture.** After one eager backward pass, `tape.backward()` captured into a CUDA graph and
  replayed twice gives the same gradients as the eager run. Without the eager pass, capture raises the
  documented error.
- **Empty PE.** A two-PE run in which one PE's local block is empty for a step still completes, which
  exercises the barrier ahead of the zero-size early return.
- **Diagnostics.** The pull-family and fused-shape warnings fire exactly once per kernel at adjoint
  launch and not in a forward-only program; the symmetric-gradient check raises for a
  `requires_grad=True` symmetric array.

InfiniBand is not available in CI. The design's NVLink-versus-InfiniBand arguments rest on the
documented memory model, and the pull adjoint uses blocking gets, whose completion semantics do not
depend on the transport.

## Open Questions

1. Does `nvshmem_ptr` return null for a non-symmetric address, or is that undefined? Decides whether the
   U3 check ships or the rule is documentation only.
   **Answered:** with NVSHMEM 3.8.0, `nvshmem_ptr(ptr, my_pe)` returns null for a Warp-allocated array,
   for the gradient Warp allocates for a symmetric array with `requires_grad=True` and for a
   `wp.zeros_like()` array, and non-null for symmetric memory, without crashing, so the U3 check ships.
2. The latency of the one-thread device barrier on the test machines, which decides how soon the
   point-to-point refinement is worth building.
   **Answered:** on two RTX 3090 GPUs joined by an NVLink bridge, with NVSHMEM 3.8.0, 100 consecutive
   barrier launches from Python took 17.6 µs per barrier on each PE, about as long as enqueueing them
   without waiting took, so the host launch cost dominates; replayed from a CUDA graph, the same barrier
   took 4.3 µs.
3. Whether the barrier kernel behaves inside a CUDA graph over the InfiniBand proxy transport. Expected
   to, since it is a kernel, but unverified.
4. Whether NVSHMEM module registration can run under a relaxed stream-capture mode without invalidating
   the capture. If it can, the capture-time error in [How the barrier is issued](#how-the-barrier-is-issued)
   becomes a mid-capture load.
5. Whether `wp.array(ptr=..., requires_grad=True)` should warn when NVSHMEM is loaded and the pointer is
   symmetric. It is the most likely mistake and the U3 check would catch it later, at the adjoint launch.
6. The name of the configuration flag.
   **Answered:** `wp.config.nvshmem_adjoint_barrier`.

## Future Work

- **Point-to-point backward synchronization.** Each PE owns a symmetric counter of completed flagged
  adjoint launches, incremented by a one-thread kernel after each. The adjoint of a push polls
  `counter[pe] >= counter[me]` on its target PE before pulling. Each PE waits only on its neighbor's
  previous launch, so there is no cycle, and only communicating pairs synchronize. Needs a collective
  allocation at a well-defined point and keeps the global barrier for kernels with remote writes.
- **Pull-family adjoint.** CAS-emulated remote accumulation, a barrier after every launch that performs
  it, and the rule that the owning PE does not accumulate locally into the same gradient array in the same
  launch. Worth building only if a differentiable pull-based pattern turns out to matter.
- **Block-cooperative adjoints.** When `tile_nvshmem_put` exists, its adjoint pulls with a block-scoped
  get into a tile and accumulates cooperatively, removing the per-element round trips of
  `adj_nvshmem_put`.
- **A public stream-ordered barrier.** The forward passes this design targets want one too; the one-thread
  kernel is a candidate implementation for a `wp.nvshmem_barrier_all_on_stream`-style host helper.

## Appendix A: Correctness of Each Reverse Operation

Notation. ℓ is the scalar loss. For any value x, x̄ = ∂ℓ/∂x. The reverse-mode rule for a statement
y = f(x₁, …, xₙ) is x̄ᵢ += (∂f/∂xᵢ)ᵀ ȳ for each input, executed after every statement that used y has
been reversed. Symmetric arrays exist once per PE: `D` on PE b is written `D^b`, and its gradient array
is `D̄^b`. A statement executes on the PE that runs the kernel, and its reverse executes on that same PE,
because that is where the thread's local values and their adjoints live.

### `nvshmem_p(dest, i, value, pe)` on PE a

Forward: `D^pe[i] = v`, a copy of a thread-local value into one remote element. The Jacobian is 1 with
respect to `v` and 0 with respect to `i` and `pe`, which are integers. Reverse: `v̄ += D̄^pe[i]`. The
left side is thread-local on PE a; the right side is one element of PE pe's gradient array, which PE a
reads with `nvshmem_g(adj_dest, i, pe)`. The integer arguments receive no adjoint. The pull reads the
element of `D̄^pe` and leaves it unchanged, so several reverse statements may read the same element.
This differs from Warp's adjoint of a local array store, which zeroes the gradient element it reads when
the adjoint buffer it was given is the array's own `.grad`, or none was passed, and the array does not
retain its gradient; it leaves an external adjoint buffer intact. Under U2 no store adjoint on PE pe
competes for this element, because the push is its only writer.

Timing: `D̄^pe[i]` is complete when every statement that read `D^pe[i]` has been reversed. Under U1 those
statements are in later launches, and under U2 they read this value and no other, so Appendix B's
schedule guarantees completeness at the time of the read.

### `nvshmem_put(dest, source, pe)` on PE a

Forward: for every element index n, `D^pe[n] = S^a[n]`, the same copy as `p` applied to `count`
elements with a shape-preserving index map. Reverse: `S̄^a[n] += D̄^pe[n]` for every n. The left side is
PE a's own gradient array, accumulated with a local atomic because other threads of the adjoint kernel
may add to the same element. The right side is pulled element by element. Since `dest` and `source` have
the same shape (asserted in the forward), the element map is the identity on linear indices, and
`element_at` applies the arrays' own strides.

### `nvshmem_put_signal(dest, source, sig, i, s, op, pe)` on PE a

Forward: the data part is `nvshmem_put`; the signal part is `G^pe[i] = op(G^pe[i], s)` on an unsigned
integer flag with `op` either set or add. Integers are not differentiated in Warp, so the signal part has
Jacobian zero with respect to everything differentiable, and its reverse is empty. The reverse of the
whole is the reverse of the data part. The forward ordering guarantee that the flag is updated only after
the data is delivered has no counterpart in the reverse, because the reverse pulls synchronously.

### `nvshmem_g(source, i, pe)` on PE a (not differentiated in this version)

Forward: `x = S^pe[i]`, the mirror image of `p`. Reverse would be `S̄^pe[i] += x̄`, a remote accumulate.
The derivation is as sound as the one for `p`; what is missing is a primitive that performs it safely,
as [The pull family](#the-pull-family-nvshmem_g-and-nvshmem_get) explains. Until then the call has no
reverse and the launch warns. The call is replayed in the adjoint kernel because `x` is needed to
recompute downstream intermediates, and under U2 the remote value is unchanged since the forward.

### `nvshmem_get(dest, source, pe)` on PE a (not differentiated in this version)

Forward: `D^a[n] = S^pe[n]` for every n. Reverse would be `S̄^pe[n] += D̄^a[n]`. Same status as `g`.
The call is not replayed, so `dest` keeps the forward values; under U2 those are the values the
downstream statements saw.

### `nvshmem_signal_op(sig, i, s, op, pe)`

Forward: `G^pe[i] = op(G^pe[i], s)` on an unsigned integer. No differentiable value is defined or
consumed. Reverse: empty.

### `nvshmem_signal_wait_until(sig, i, cmp, s)`

Forward: blocks until `cmp(G^me[i], s)` holds and returns `G^me[i]`, an unsigned integer. It defines no
differentiable value. What it does define is a constraint on the forward schedule: every statement after
it executes after the remote update it waited for. The reverse program needs the mirrored constraint,
that the pull in the producer's reverse executes after the accumulations in the consumer's reverse.
Appendix B shows that the launch-level barriers establish that constraint for every consumer thread at
once, which no reverse of the wait could do for one thread. Reverse: empty.

### `nvshmem_quiet()`, `nvshmem_fence()`, `nvshmem_barrier_all()`

Forward: completion, ordering, and all-PE synchronization of the forward's remote operations. No
differentiable value. Their role in the forward is to make the forward's remote writes visible before
their consumers read; in the reverse, the remote reads are blocking pulls, which need no completion step,
and the visibility of the consumers' accumulations is guaranteed by the launch boundary plus the barrier
of Appendix B. Reverse: empty. The replay of each is skipped, because a forward barrier replayed inside
the adjoint kernel would have to be matched by every PE's adjoint kernel from exactly one thread, which
the generated code cannot arrange.

### `nvshmem_my_pe()`, `nvshmem_n_pes()`

Constants of the job. Replayed, no reverse.

## Appendix B: The Barrier Schedule

### Setting

P PEs run the same program. The tape holds launches L₁ … Lₙ. The backward issues Aₙ … A₁ on each PE's
stream, where Aₖ is the adjoint of Lₖ. A subset F of the launches is *flagged*: the adjoint of their
kernel contains a remote access, which includes every launch whose adjoint reads a remote gradient.
Rule B1 enqueues a barrier βₖ immediately before Aₖ for every k ∈ F, and rule B2 enqueues β₀ after A₁.
A barrier completes on a PE only after every PE has enqueued it and finished everything before it on
its stream.

Let G be a gradient array element on PE b and let the forward value it belongs to have been written by
Lₖ on PE a through a push. Under U1 the readers of that value are statements in launches Lₘ, m > k, and
under U2 there are no other writers of that element in the window of interest.

### Sufficiency of B1 for invariant (R)

Claim: when Aₖ on PE a reads G, every contribution to G has been made and is visible.

Contributions to G come from the reverse statements of the readers, which execute in Aₘ, m > k, on the
PEs that performed the reads, with local atomics. Each such Aₘ precedes βₖ on that PE's stream, since
m > k means Aₘ is issued before Aₖ in the backward and βₖ is issued immediately before Aₖ. βₖ completes
on PE a only after every PE has finished everything before βₖ on its stream, which includes every Aₘ
with m > k. Aₖ on PE a is issued after βₖ on PE a's stream, so it starts after βₖ completes, hence after
every contribution to G was made.

Visibility: the contributions are local stores and atomics on PE b, completed at PE b's kernel boundary
before PE b arrives at βₖ. The NVSHMEM barrier includes completion of PE b's previously issued memory
stores. PE a's `nvshmem_g` then reads PE b's memory over the transport and observes the completed value.
On systems with InfiniBand, the memory model's weaker visibility statement concerns remote updates being
visible at their *destination*; here the data being read is PE b's own local writes, which the barrier
makes visible at PE b, and the get is a read performed at PE b's memory, so the weaker statement is
sufficient.

Nothing changes G after the read: later launches in the backward are Aⱼ, j < k, which reverse readers of
*earlier* forward values, and under U2 those are different elements or different arrays.

### Sufficiency of B1 and B2 for invariant (W) (pull-family follow-up)

Claim: a remote write to G issued by Aₖ on PE a is complete and visible before any read of G by Aⱼ,
j < k, on PE b, or by the host.

Reads of G on PE b occur in Aⱼ, j < k, because G's forward value was produced before Lₖ consumed it. If
any Aⱼ with j < k is flagged, the first such βⱼ follows Aₖ on PE a's stream and includes PE a's `quiet`,
which completes PE a's remote writes; Aⱼ on PE b starts after βⱼ completes. If no later Aⱼ is flagged,
the reads of G in the remaining adjoint launches are local on PE b and happen after β₀, which again
follows Aₖ on PE a and completes PE a's writes. Host reads happen after the user synchronizes the stream,
which is after β₀.

### Necessity of each B1 barrier

Take the running example with two PEs and two steps: L₁ is `jacobi_step` from `fields[0]` to
`fields[1]`, L₂ is `jacobi_step` from `fields[1]` to `fields[2]`, L₃ is `interior_loss` on `fields[2]`.
A₃ is local. A₂ on PE b accumulates the gradient of the ghost cells of `fields[1]` that L₂ read. A₁ on
PE a pulls those ghost gradients, since L₁ on PE a wrote them.

Remove β₁, the barrier before A₁. Let PE b be delayed by δ relative to PE a, for instance because its
GPU is shared with another process. Stream order on each PE is then:

```
PE a:  A3 ──── A2 ──── A1 (pulls ghost gradients from PE b at time t)
PE b:        δ ──── A3 ──── A2 (writes those ghost gradients, finishing at time t' > t)
```

A₁ on PE a reads zeros, because PE b's A₂ has not run. The gradient of the initial field on PE a misses
every path that crosses the boundary into PE b. Nothing on PE a's stream can detect this, since every
operation on it completed normally. With β₁ in place, PE a's A₁ waits at β₁ until PE b has finished A₂.

The same construction removes any βₖ in a longer chain: delay PE b so that its Aₖ₊₁ finishes after PE
a's Aₖ starts. Hence each B1 barrier is necessary, not merely sufficient.

### Necessity of B2

Continue the example into a second training iteration without β₀. After A₁, PE a zeroes its gradient
arrays for the next iteration, in stream order. Among them is `fields[1].grad`, whose ghost-cell elements
PE b's A₁ pulls, since L₁ on PE b pushed into those ghost cells and PE a's A₂ accumulated their
gradients. β₁ ensured only that PE b *started* A₁ after every PE had finished A₂; nothing orders the end
of PE b's A₁ against PE a's later work. Let PE b be slow, for instance because its replay takes longer:

```
PE a:  β1 ── A1 ── zero fields[1].grad
PE b:  β1 ──────── A1 (pulls ghost-cell gradients from PE a after they were zeroed)
```

PE b's A₁ pulls zeros, and PE b's gradient of the initial field misses every path that crosses the
boundary into PE a. β₀ makes PE a's zeroing wait until PE b has finished A₁. Reseeding gradients and
freeing the gradient buffers race PE b's A₁ in the same way.

The next forward pass is not what β₀ protects here. A₁ replays the stencil's reads of `fields[0]`, which
no put writes, and A₂, whose replay reads `fields[1]`, finished on every PE before β₁ completed, so
before PE a's next L₁ puts into `fields[1]`.

The schedule test provokes this race: PE 1 spins inside the replay of its A₁, and PE 0 zeroes its
gradients for a second pass without a host synchronization in between, so the test fails when the
trailing barrier is removed. The single-PE test counts the barriers that `Tape.backward()` enqueues,
which pins β₀'s presence without depending on timing.

### Why the barrier is collective rather than pairwise

In the sufficiency argument, PE a needs only the PEs it pulls from to have finished Aₖ₊₁. A pairwise
wait would suffice for (R) and is the point-to-point refinement in Future Work. It does not suffice for
(W): the remote write lands in an array that PE b reads with plain loads, and PE b does not know which
PEs wrote to it, so PE b would have to wait for *all* PEs, which is the collective barrier again. Since
this version does not perform remote writes, the refinement is available later without changing the
rules above.

## Appendix C: NVSHMEM Facts Relied Upon

Verbatim quotes from the NVSHMEM API documentation at docs.nvidia.com/nvshmem/api/latest, retrieved for
this design. Page names refer to that tree.

**Ordering, from `gen/api/ordering.html`.**

- "The nvshmem_quiet routine ensures completion of all operations on symmetric data objects issued by the
  calling PE."
- "A call to nvshmem_quiet by a thread completes the operations posted before calling nvshmem_quiet. If
  the user intends to also complete operations issued by a thread that is not the thread calling
  nvshmem_quiet, the user must ensure that the operations are performed before the call to
  nvshmem_quiet."
- "The nvshmem_quiet routine can enforce memory store ordering only for the calling thread."
- "nvshmem_fence only provides per-PE ordering guarantees and does not guarantee completion of delivery."
- "nvshmem_quiet is not usually needed if nvshmem_barrier_all or nvshmem_barrier are called. The barrier
  routines wait for the completion of outstanding operations to symmetric data objects on all PEs."
- "Visibility is only guaranteed at the destination PE."

**Collectives, from `gen/api/collectives.html`.**

- "Collective operations on a given team (including the implicit team operating on all PEs) must be
  called by all PEs in the team concurrently."
- "Collective operations must be called by exactly one instance per PEs at a time. This means device APIs
  must be called by exactly one thread (for regular thread scoped APIs), warp (for warp scoped APIs) or
  block (for block scoped APIs) at a time."
- "In a multithreaded NVSHMEM program, only the calling thread is blocked, however, it may not be called
  concurrently by multiple threads in the same PE."
- "Before synchronizing with other PEs, nvshmem_barrier_all ensures completion of all previously issued
  memory stores and remote memory updates issued NVSHMEMAMOs and RMA routine calls such as
  nvshmem_int_add, nvshmem_put32, nvshmem_put_nbi, and nvshmem_get_nbi."
- "On systems with only NVLink, updates are globally visible, whereas on systems with both NVLink and
  InfiniBand, NVSHMEM only guarantees that updates to the memory of a given PE are visible to that PE."
- Synopsis: `void nvshmemx_barrier_all_on_stream(void, cudaStream_t stream)`. The page gives the
  on-stream variant no separate description and does not mention CUDA graph capture.

**Atomics, from `gen/api/amo.html` and `gen/mem-model.html`.**

- "The atomic routines guarantee that concurrent accesses by any of these routines to the same location,
  and using the same datatype (specified in Tables Standard AMO Types and Names and Extended AMO Types
  and Names) will be exclusive."
- "NVSHMEM atomic operations do not guarantee exclusivity in the following scenarios, all of which result
  in undefined behavior." Among the listed scenarios: "When NVSHMEM atomic operations and non-NVSHMEM
  operations (for example, load and store operations) are used to access the same location
  concurrently."
- `nvshmem_TYPENAME_atomic_add` is defined "where TYPE is one of the standard AMO types", which the page
  lists as `int`, `long`, `long long`, `size_t` and `ptrdiff_t`. `float` and `double` appear only in the
  extended table used by `atomic_fetch`, `atomic_set` and `atomic_swap`.
- "A call to a non-fetching atomic routine issues the atomic operation and may return before the
  operation executes on the target PE."

**Visibility, from `gen/mem-model.html`.**

- "On systems with both NVLink and InfiniBand, the NVSHMEM synchronization operations including
  nvshmem_barrier, nvshmem_barrier_all, nvshmem_quiet, nvshmem_wait_until_*, and nvshmem_test_* only
  guarantee visibility of updates to the local PE's symmetric objects. However, on systems with only
  NVLink, these operations guarantee global visibility of updates to symmetric objects."

**Launch and concurrency, from `using.html` and `cuda-interactions.html`.**

- "CUDA kernels that use synchronization or collective APIs must be launched by using the collective
  launch API to guarantee deadlock-free progress and completion."
- "NVSHMEM stream-based operations (for example, nvshmemx_putmem_on_stream) enqueue the corresponding
  operation on a CUDA stream and return immediately."
- "collective operations that use the same team must not be allowed to execute in parallel. This can
  happen when a collective operation (for example, nvshmem_malloc) is executed on the CPU in parallel
  with an operation on a stream that uses the same team (for example, nvshmem_barrier_all_on_stream)."
