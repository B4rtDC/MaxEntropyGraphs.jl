# Changelog

## v0.6.0

Homogenized expectation & variance machinery across all eight models
(Squartini & Garlaschelli 2011, App. A.3/A.4/B; Saracco et al. 2015 SI for the BiCM),
together with the reciprocity-aware directed models and triadic statistics detailed below.

### Breaking
- **`σₓ` gained the within-dyad covariance term for undirected models** (UBCM, and the binary layer of
  UECM/CReM), so the variances and z-scores it returns for those models change (they were previously low
  by up to `√2`). This behavioural change — detailed under *Fixed* — is what warrants the minor bump; a
  `0.x` minor increment is a breaking release under Julia's SemVer convention.
- **The `:NelderMead` solver option was removed.** `solve_model!(m, method=:NelderMead)` now raises an
  `ArgumentError`; use the fixed-point default or a gradient-based method (`BFGS`/`LBFGS`/`Newton`).

### Fixed
- **Undirected delta-method `σₓ` was missing the within-dyad covariance term** (UBCM, and the binary
  layer of UECM/CReM). For an undirected model `aᵢⱼ` and `aⱼᵢ` are the *same* random variable, so the
  ordered-pair sum of Squartini & Garlaschelli Eq. B.16 requires the cross-term
  `Cov(aᵢⱼ,aⱼᵢ)·(∂X/∂aᵢⱼ)(∂X/∂aⱼᵢ)` with `Cov = σ²[aᵢⱼ]`. Without it, `σₓ` was low by exactly `√2`
  for metrics written on the full symmetric matrix (e.g. `sum`, ANND — Monte-Carlo confirmed on the
  karate club: 10.03 vs a sampled 14.24) and could even *overestimate* for direction-selective metrics
  (ANND of a single node: 2.52 vs a sampled 1.42). Metrics written on a single triangle were unaffected.
  The corrected form is now independent of which convention the metric uses. z-scores computed with
  `σₓ` for undirected models change accordingly (this is the reason for the 0.6.0 version bump).

### Added
- **UECM & CReM weighted-layer variance**: `Ŵ` is now stored via `set_Ŵ!`, and the new `σʷ`/`set_σʷ!`
  provide the per-edge weight standard deviations (UECM: Bernoulli–geometric mixture,
  `Var(wᵢⱼ) = pᵢⱼ(1 + yᵢyⱼ - pᵢⱼ)/(1 - yᵢyⱼ)²`; CReM: Bernoulli–exponential mixture,
  `Var(wᵢⱼ) = fᵢⱼ(2 - fᵢⱼ)/(θᵢ + θⱼ)²`). `σₓ` gained the `layer=:binary|:weighted` keyword (same API
  as the DCReM/CRWCM). Both formulas are derived symbolically and validated against ensemble sampling
  in `validation/`.
- **BiCM variance machinery**: `σˣ`/`set_σ!` (per-entry Bernoulli standard deviations of the
  biadjacency matrix) and the delta-method `σₓ` (independent entries, no covariance terms) now exist
  for the BiCM, closing the last gap in the common model API.
- **BiCM `Vn`/`Λn` motif families**: `Vn_motifs` (observed & expected `n`-fold co-occurrence counts),
  `Vn_sigma` and `Vn_zscore`, for any `n ≥ 2` and both layers. The default `method=:exact` evaluates
  the mean and variance *exactly* from the Poisson-binomial distribution of the random opposite-layer
  degrees (independent across nodes); `method=:delta` provides the closed forms of Saracco et al.
  (2015, SI Eqs. III.6-III.13), which are accurate when the opposite-layer degrees are large compared
  to `n` (they underestimate the variance for sparse layers — Monte-Carlo measured σ ratios down to
  0.15 at degrees ≈ n, hence the exact default).
- **`validation/`**: a standalone validation suite deriving every model's per-dyad moments
  (`⟨g⟩`, `Var[g]`, `Cov(gᵢⱼ,gⱼᵢ)`) symbolically with Symbolics.jl (state sums, probability generating
  functions, moment generating functions) and Monte-Carlo gates for all new/changed formulas; a fast
  distilled subset runs in CI (`test/symbolics.jl`).

Reciprocity-aware directed models and triadic statistics
(Squartini & Garlaschelli 2011; Di Vece, Pijpers & Garlaschelli 2023 — the model family of the NuMeTriS package).

### Added
- **`RBCM`** — the Reciprocal Binary Configuration Model (constrains, per node, the non-reciprocated
  out-degree `k→`, non-reciprocated in-degree `k←` and reciprocated degree `k↔`). Parameter reduction over
  the unique degree triples, a numerically-stable four-term log-sum-exp likelihood, branch-free SIMD
  analytical gradient, a stable fixed-point default, dyadic probability accessors, **exact** expected
  motif spectra evaluated from the dyadic probabilities (within a dyad `aᵢⱼ` and `aⱼᵢ` are correlated, so
  the `Ĝ`-based evaluation valid for the DBCM would be wrong), a covariance-aware delta-method `σₓ`,
  dyad-state sampling, `reciprocity` model methods (also for the `DBCM` baseline), and the full
  accessor/information-criterion API (`k = 3N`, same observation count as the DBCM, so the two are
  directly comparable).
- **`DCReM`** — the directed Conditional Reconstruction Method (CReM_A in the literature; `DBCM+CReMa` in
  NuMeTriS): a two-step model for weighted directed networks with continuous weights (internally solved
  DBCM topology + exponential conditional weights constraining the out/in-strengths). Includes the
  expected-weight machinery `Ŵ`/`set_Ŵ!`/`σʷ`/`set_σʷ!` (now exported) and a layer-aware `σₓ`
  (`layer=:binary`/`:weighted`).
- **`CRWCM`** — the Conditionally Reciprocal Weighted Configuration Model (Di Vece et al. 2023;
  `RBCM+CRWCM` in NuMeTriS): a two-step model constraining the four reciprocal strength sequences
  (`s→`, `s←`, `s↔out`, `s↔in`) conditional on an internally solved RBCM topology. The block-separable
  4N system is solved jointly; dead channels are pinned automatically; the within-dyad weight covariance
  `Cov(wᵢⱼ, wⱼᵢ) ≠ 0` is available analytically and enters the layer-aware `σₓ`.
- **Reciprocity metrics**: `reciprocity` (topological, `r_t`) and `weighted_reciprocity` (`r_w`), plus the
  reciprocal degree sequences (`nonreciprocated_outdegree`, `nonreciprocated_indegree`,
  `reciprocated_degree`) and reciprocal strength sequences (`nonreciprocated_outstrength`,
  `nonreciprocated_instrength`, `reciprocated_outstrength`, `reciprocated_instrength`), each with graph,
  matrix, single-node and model methods.
- **Triadic statistics**: `motif_fluxes`/`motif_flux` (the weight circulating on each of the 13 directed
  3-node motifs; BLAS-backed trace formulation with **exact** expected spectra for the DCReM and CRWCM),
  `motif_intensities` (Onnela geometric-mean intensities), and the sampling-based significance utilities
  `ensemble_zscores`/`motif_zscores`/`flux_zscores` (NuMeTriS `numerical_triadic_zscores` parity).
- A **"Which model when?"** documentation page guiding model selection (incl. the reciprocity
  diagnostics), plus model and API documentation pages for the three new models.

### Removed
- The `:NelderMead` solver option. It existed for testing purposes only: the derivative-free simplex
  rarely converges on these likelihoods and adds no value next to the fixed-point and gradient-based
  methods. `solve_model!(m, method=:NelderMead)` now raises an `ArgumentError` listing the supported
  methods.

### Fixed
- `Base.length(m::DBCM)` referenced a nonexistent field and would error when called.

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
