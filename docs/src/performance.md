# Performance and scalability

## Memory: the analysis is matrix-free

The expected adjacency matrix ``\hat{G}`` is dense and ``O(n^2)`` — about 512 MB at
``n = 8\,000`` and ``\approx 500`` GB at ``n = 250\,000`` in `Float64`. The package therefore does
**not** materialize it for standard analyses:

- **Metric values** (`degree`, `ANND`, `wedges`, V-motifs, projections, …) and **sampling**
  (`rand(m)` with the default `precomputed=false`) compute each entry on the fly from the reduced
  maximum-likelihood parameters via element accessors (`MaxEntropyGraphs.A(m, i, j)`), so they use
  ``O(n)`` memory. (Measured: `degree(m)` allocates ~0.25 MB at ``n = 8\,000``; a single element
  accessor allocates nothing.)
- `Ĝ(m)` / `set_Ĝ!`, `σˣ(m)` / `set_σ!` (and `Ŵ`/`set_Ŵ!`, `σʷ`/`set_σʷ!` for the weighted models),
  and the `precomputed=true` code paths **do** materialize a dense ``n \times n`` matrix and are
  intended only for small networks — avoid them for large ``n``.
- The custom-metric variance via error propagation (`σₓ`) requires the dense expected matrix and the
  dense per-edge standard deviations; see the guidance below for its scaling and the alternatives.

## Variance (`σₓ`): backend choice and scaling

`σₓ(m, X)` evaluates the delta-method variance of a metric `X` by (auto)differentiating `X` at the
materialized expected matrix and weighting by the per-edge standard deviations. Benchmarks
(`performance/variance_benchmarks.jl`) give the following guidance:

- **Backend (`gradient_method`).** The default **`:ReverseDiff` is the right general choice.** For
  matrix-multiplication metrics (`triangles`, `motifs`) it is the *only* practical backend: `:Zygote`
  materializes ``O(n^2)`` BLAS pullback adjoints and blows up (tens of GB at ``n \approx 10^3``, TB at
  ``n \approx 3000``), and `:ForwardDiff` is already ~1000× slower at ``n = 100`` (its cost grows with
  the ``n^2`` inputs). For strictly linear metrics (`sum`, `degree`, `ANND`) all three agree and
  `:Zygote` is ~1.5–2× faster, but `:ReverseDiff` is safe everywhere.
- **Memory wall.** Materializing `Ĝ`+`σ` is ``O(n^2)`` — about 2 GB at ``n = 3000`` and **16 GB at
  ``n = 10\,000``** (`Float64`), plus the AD workspace. In practice the stored `σₓ` path is comfortable
  up to ``n \approx 5\,000\text{–}8\,000`` on a 16 GB machine. Materialize once (`set_Ĝ!`/`set_σ!`) and
  reuse it across metrics — that is ~100× cheaper per call than rebuilding the matrices each time.
- **Beyond the wall.** For larger networks you have two options:
  1. **Sampling** — estimate the variance from `rand(m, k)` (matrix-free, ``O(n)`` memory), the general
     route for any metric including those the delta method does not cover (e.g. weighted intensities).
  2. **A matrix-free analytic gradient** — for metrics whose ``\partial X/\partial g_{ij}`` is known in
     closed form (edge count, degree, ANND, triangles), streaming the delta-method sum with the element
     accessor `MaxEntropyGraphs.A(m, i, j)` is ``O(n)`` in memory: benchmarked at **0 bytes** for the
     edge count and **28 KiB at ``n = 3162``** for triangles, versus Zygote's 1.38 TB at the same size.
     A generic matrix-free `σₓ` fast path for these metrics is a planned addition; until then it can be
     written directly against `A(m, i, j)` (see `performance/variance_benchmarks.jl` for worked
     examples).

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
