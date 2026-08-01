# Keep an iterative spectral transform on CUDA

This synthetic card compares two ways to apply the same frequency-domain gain
to batches of complex signals over repeated iterations. The structured CUDA
claim below is limited to the exact measured scope; CPU impact remains
unverified.

## Before

The baseline moves the current state from CUDA to a newly allocated NumPy
array on every iteration, dispatches NumPy FFT and inverse FFT operations on
the host, stacks and casts the next complex-pair array, and constructs a new
CUDA array from it:

```python
host_pairs = state.numpy()
host_complex = host_pairs[..., 0] + 1j * host_pairs[..., 1]
spectrum = np.fft.fft(host_complex, axis=1)
filtered = np.fft.ifft(spectrum * host_gain, axis=1)
next_pairs = np.stack((filtered.real, filtered.imag), axis=-1).astype(np.float32)
state = wp.array(next_pairs, dtype=wp.vec2f, device=device)
```

`state.numpy()` implicitly waits for preceding CUDA work before copying data to
host memory. The NumPy expression creates complex input, spectrum, product,
inverse-transform, stack, and cast storage. `wp.array()` then allocates device
storage and copies the result from host to CUDA. These transfers,
synchronization points, host dispatches, and temporary allocations all remain
inside the repeated `run()` path intentionally.

## After

The candidate uploads the initial state and frequency gain before runtime,
preallocates two CUDA state arrays, and alternates them between iterations.
Each iteration launches one tiled kernel that loads one signal row, applies
`wp.tile_fft()`, multiplies by the gain, applies `wp.tile_ifft()`, and stores
the next row. No transform state returns to the host during `run()`.

Warp's public tile FFT tests establish that both transforms are unnormalized:
an FFT followed by an inverse FFT scales the values by the transform length.
The candidate therefore divides the frequency-domain product by 256 before
the inverse transform. That matches NumPy's normalized `ifft` convention and
preserves the baseline's per-iteration semantics.

Trial preparation copies the same initial CUDA state into the first ping-pong
array before either variant runs. The final ping-pong selection depends on
iteration parity, so both odd and even iteration counts observe the state
produced by the last transform.

## Runtime boundary

The harness performs at least three untimed warm-up trials, so kernel and
MathDx compilation are excluded from paired runtime measurements. Initial
uploads, gain construction, stable allocations, per-trial state reset, and
the final observation copy are also outside `run()`. Timed candidate work is
the repeated CUDA launch sequence plus the device synchronization at the
trial boundary. Timed baseline work includes every repeated host round trip
and its implicit synchronization.

## Supported scope

This card compiles a 256-point `wp.vec2f` transform and launches each tile with
64 threads. The transform is a power of two, has four elements per thread, and
meets both the MathDx path's element-per-thread requirement and the documented
CUDA fallback requirement that the transform length be divisible by the
block size. Correctness compares the real and imaginary components with
absolute and relative tolerance `3e-4`.

## Measured evidence

Evidence was generated from clean revision
`691fe17a9e9e15af9b2cbd2f8199366dba426eba` with 20 pairs and 10,000
bootstrap resamples at the exact default workload: size 256, batch 256, 20
iterations, and seed 20260730.

On an NVIDIA RTX PRO 6000 Blackwell Server Edition (`sm_120`) with Warp
`1.17.0.dev0`, CUDA Toolkit 13.0, and CUDA driver 13.0, the baseline median
was 10,058,128 ns with MAD 16,540.5 ns. The candidate median was 233,580 ns
with MAD 16,950 ns. The median paired ratio was `0.02312415714355634`, with
paired 95% confidence interval
`[0.021545257167803505, 0.024768975798170847]`. The complete interval is
below parity, so CUDA impact is `improved` and the card is `recommended` for
that exact device and workload. Record `895080285a5e4b12af64a5dac3218a05`
supports the CUDA claim. No CPU measurement was run, so CPU impact remains
`unverified`.

Do not apply this card directly when a transform size or dtype is unsupported,
when the real pipeline is CPU-resident, or when a host transfer occurs only
once outside the repeated path. Device residency also does not remove
application-level synchronization that a later host consumer genuinely
requires.
