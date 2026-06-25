# Performance, scalability & GPU

## Memory: the analysis is matrix-free

The expected adjacency matrix ``\hat{G}`` is dense and ``O(n^2)`` — about 512 MB at
``n = 8\,000`` and ``\approx 500`` GB at ``n = 250\,000`` in `Float64`. The package therefore does
**not** materialize it for standard analyses:

- **Metric values** (`degree`, `ANND`, `wedges`, V-motifs, projections, …) and **sampling**
  (`rand(m)` with the default `precomputed=false`) compute each entry on the fly from the reduced
  maximum-likelihood parameters via element accessors (`MaxEntropyGraphs.A(m, i, j)`), so they use
  ``O(n)`` memory. (Measured: `degree(m)` allocates ~0.25 MB at ``n = 8\,000``; a single element
  accessor allocates nothing.)
- `Ĝ(m)` / `set_Ĝ!`, `σˣ(m)` / `set_σ!`, and the `precomputed=true` code paths **do** materialize a
  dense ``n \times n`` matrix and are intended only for small networks — avoid them for large ``n``.
- The custom-metric variance via error propagation (`σₓ`) currently requires the dense `m.σ` (and
  `m.Ĝ`); for large networks prefer the sampling route to estimate variances. (Making this path
  matrix-free is a known follow-up.)

## Multithreading

Sampling `rand(m, n)` and the p-value computations parallelise over `Threads.@threads`; start Julia
with `-t auto` (or set `JULIA_NUM_THREADS`). `rand(m, n; rng = …)` is reproducible **and** independent
of the thread count — each sample draws its own seeded `Xoshiro` stream from the supplied `rng`.

## GPU acceleration (assessment)

GPU support is **not currently shipped**. Where it would (and would not) help:

| Kernel | GPU-amenable? | Verdict |
|---|---|---|
| Sampling Bernoulli draws / ``\hat{G}``-element evaluation | Yes — elementwise, embarrassingly parallel | Best (and only) candidate, and only at large ``n`` (``\gtrsim 10^5``) × many samples |
| Likelihood / gradient | No — the reduced loops are ``O(K^2)`` over *unique* degrees (``K`` small); kernel-launch overhead dominates | Keep on CPU |
| Metric p-values / motif counts | Partly — but the Poisson / Poisson–binomial CDFs are not trivially GPU-portable | Low priority |

Recommended design (future, optional): a package **extension** (`ext/…`) with the GPU backend as a
`[weakdeps]`, so the package stays installable without a GPU. Write kernels with
[`KernelAbstractions.jl`](https://github.com/JuliaGPU/KernelAbstractions.jl) for vendor portability
(CUDA / Metal / ROCm), detect hardware at run time (e.g. `CUDA.functional()`), default to `gpu=false`,
and keep a CPU fallback that is exercised in CI (which has no GPU). Ship it only if it clearly beats
the threaded CPU sampler at scale.
