# Changelog

## v0.5.3

Weighted, undirected models brought to full parity with the binary trio (UBCM/DBCM/BiCM).

### Added
- **`UECM`** — the Undirected Enhanced Configuration Model (constrains the degree **and** the integer
  strength sequence). Numerically-stable log-likelihood, branch-free SIMD gradient, seeded reproducible
  sampling, the full accessor/variance/information-criterion API, a NEMtropy (`ecm_exp`)
  performance/accuracy comparison, and documentation. Because the likelihood is only defined on the
  feasible region, the solver uses a `BackTracking` line search (`BFGS` default; the fixed point is
  unstable for this model).
- **`CReM`** — the Conditional Reconstruction Method (a two-step model for weighted, undirected networks
  with **continuous** positive weights: a binary UBCM layer supplies the edge probabilities `fᵢⱼ`,
  conditional on which the weights are exponential with rate `θᵢ+θⱼ`, constraining the strength
  sequence). Branch-free SIMD kernels, seeded reproducible sampling, the full accessor/variance/
  information-criterion API (`k = N` parameters), a NEMtropy (`crema`) performance/accuracy comparison,
  and documentation. The fixed-point recipe is stable and is the default; `BFGS`/`Newton` are also
  available.

## v0.5.2

Metric-computation performance.

### Performance
- Accelerated the metric kernels (algorithmic simplifications, BLAS-backed inner products, and AD- and
  memory-aware implementations); the kernel equivalences are documented.

### Changed
- Refreshed citation metadata (`CITATION.cff`, Zenodo concept DOI) and listed the network motifs in the
  API reference.

## v0.5.1

Solver fixes and robustness.

### Fixed
- `maxiters` is now forwarded to the gradient-based optimisers (BFGS/LBFGS/Newton); it was previously
  accepted by `solve_model!` but silently ignored for those methods.
- Made the BiCM automatic-differentiation gradient path robust to the current
  Zygote/DifferentiationInterface stack: the differentiated objective no longer captures the model's
  `status` dictionary (which triggered a `BoundsError` in the AD `dict_getindex` pullback).

### Added
- `g_tol` keyword for `solve_model!` (maps to Optim's gradient tolerance `g_abstol`), so a solve can
  stop before over-converging.

## v0.5.0

Modernization, correctness and performance release.

### Breaking
- **Minimum Julia is now 1.10 (LTS)** — Julia 1.9 (end-of-life) is no longer supported.
- **Dependency majors bumped**: Optimization 3 → 4, ForwardDiff 0.10 → 1, Zygote 0.6 → 0.7,
  OptimizationOptimJL 0.1 → 0.4.
- **Removed dependencies**: `Revise` and `Dates` (not used at run time; `Revise` as a hard dependency
  forced it onto every downstream user) and `OptimizationNLopt` (no NLopt optimizer was wired in).

### Added
- **Reproducible, thread-safe sampling**: `rand(m; rng=…)` and `rand(m, n; rng=…)` for UBCM/DBCM/BiCM.
  Batch sampling draws a per-sample seeded stream, so results are reproducible and independent of the
  thread schedule / thread count.
- **Low-precision guard**: `solve_model!` now warns when solving a `Float16`/`Float32` model (kept, but
  documented as storage-oriented since the solver may not converge at low precision).
- Quality assurance via **Aqua.jl** and solver-interface regression tests.
- An accurate performance/scalability/GPU page in the documentation.

### Changed / Performance
- **UBCM gradient is 8.6–11.7× faster** (branch-free, SIMD-friendly inner loop; exact to ~1e-15).
  DBCM/BiCM gradients simplified to be branch-free as well.
- Numerically stable `softplus` in the log-likelihoods (avoids overflow for hub nodes and precision
  loss at low precision).
- Modernized CI: matrix over Julia 1.10/1.11/pre × Linux/macOS/Windows (+ Apple Silicon), current
  GitHub Actions, CompatHelper, TagBot, a `[compat]`-downgrade job, and codecov configuration.

### Fixed
- The precompile workload (previously broken — wrong `project` kwarg, missing `set_Ĝ!`, and it disabled
  itself at load) now runs correctly and is enabled by default.
- `solve_model!(verbose=true)` no longer errors on Optimization 4 (`sol.solve_time` → `sol.stats.time`).
- Model type parameters are now properly constrained (the previous `<: AbstractMaxEntropyModel where {…}`
  form silently ignored the bounds); `ConvergenceError` is constructible with a `nothing` retcode.
- Removed the empty unused `src/Models/RCM.jl`.
