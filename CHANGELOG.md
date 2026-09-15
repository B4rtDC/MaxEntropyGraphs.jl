# Changelog

## Unreleased

### Fixed
- **`DECM`: `:Newton` aborted the whole Julia process when `Symbolics` was loaded.** `:Newton` needs second
  derivatives; handed a first-order ADtype, `OptimizationBase` wraps it as `SecondOrder(inner,
  AutoForwardDiff)` (it warns that it is doing so), and with our `:AutoZygote` default as the inner backend
  that nested HVP path dies with `signal 4: illegal instruction` inside `DifferentiationInterface`'s `hvp!`
  once `Symbolics` is in the same session. `:Newton` now computes its Hessian with `ForwardDiff` directly —
  no nesting, and what the upstream warning recommends anyway. It is also markedly more accurate: the
  degree residual on the rhesus macaques network drops from `~4e-9` to `3.6e-15`. An explicitly requested
  non-Zygote backend is still honoured as given.

  ⚠️ **The same crash affects `BiCM` and `UECM` with `method = :Newton`** and is *not* fixed here — their
  solvers are untouched by this release. `AD_method = :AutoForwardDiff` (or `:AutoReverseDiff`) is a working
  workaround for both. The crash is state-sensitive: it reproduces in `validation/` but not in the package's
  own test sandbox, on byte-identical dependency versions.

- **`DECM`: `:Newton` returned garbage from a far initial guess.** `L_DECM_reduced` is **exactly** invariant
  under `(α_out, α_in) → (α_out + c, α_in - c)` and `(β_out, β_in) → (β_out + c, β_in - c)`: the pair terms
  depend only on the sums `α_out,ᵢ + α_in,ⱼ` and `β_out,ᵢ + β_in,ⱼ`, and the linear part shifts by
  `-c·(Σ F·k_out - Σ F·k_in)` resp. `-c·(Σ F·s_out - Σ F·s_in)`, both identically zero (one counts the edges
  twice over, the other the total weight twice over). The Hessian is therefore singular — `‖H·g‖ ≈ 2e-19`
  for both gauge vectors. `:Newton` factorises that Hessian, so the degeneracy handed it a meaningless step:
  from a `:uniform` start it returned `-L ≈ 1.0e4` against a true optimum of `384.49`, on **every** network
  tested (8/8). Perturbing the start does not help — the degeneracy is structural, not an artifact of a
  symmetric starting point. `solve_model!` now adds a gauge-fixing term on the `:Newton` path, pinning the
  representative with `Σα_out = Σα_in`, `Σβ_out = Σβ_in`. Because the likelihood is *flat* along those
  directions this changes **no gauge-invariant quantity** — `Ĝ`, `Ŵ` and every metric are untouched.

  It is applied to `:Newton` **only**. `BFGS`/`LBFGS` keep a positive definite *approximation*, never invert
  the true Hessian, and (since `∇L` is exactly orthogonal to the gauge) never travel along it, so for them
  the extra curvature is pure trajectory perturbation that helps or hurts at random — measured, `λ = 1e-2`
  broke a `BFGS` case that both `λ = 0` and `λ = 1` solve, and `λ = 1` broke a different one.

- **`DECM`: `:LBFGS` reported failure on a converged fit.** It was not diverging but slow: at the old
  `maxiters = 1000` it was already within `7e-4` of the optimum yet reported `MaxIters`, which
  `solve_model!` turns into a hard `ConvergenceError`. The `DECM` default `maxiters` is now `10_000`; it is
  only a cap, so it costs the faster methods nothing. `:LBFGS` remains not recommended for this model.

### Added
- **`validation/symbolic/decm_gauge.jl`** (49 checks) — proves the gauge invariance and records the
  *degeneracy taxonomy* of the DECM Hessian. Beyond the two gauge modes it gains one near-null direction per
  constraint pinned at the edge of its feasible range, because the conjugate parameter runs away:

  | mechanism | condition | limit | handled |
  |---|---|---|---|
  | dead channel | `k = 0` | `α → +∞` | yes, `ind_inf` |
  | saturated degree | `k = N-1` | `α → -∞` (fitted `≈ -30…-55`) | no — runs away |
  | minimum strength | `s = k` | `β → +∞` (fitted `≈ +31`) | no — runs away |

  These runaways, not the gauge, are what make the DECM ill-conditioned (condition numbers `1e15`-`1e17` on
  affected networks versus `~1e3` on clean ones) and why first-order methods need many iterations there.
  They are a property of the **data** — such a constraint sits at the boundary of what any ensemble can
  realise, so its fitness is not identifiable — and not a defect: those fits still reproduce the constraints
  to `~1e-9`. The `rhesus_macaques` network shipped with the package has one `s = k` node.

  Recorded explicitly: the `UECM`'s box constraint (v0.7.1) must **not** be transplanted here. The DECM
  domain `β_out,ᵢ + β_in,ⱼ > 0` exempts the diagonal of singleton classes, and on a hub network the true
  optimum sits at `β_out,ᵢ + β_in,ᵢ = -0.204` — outside *every* box gauge — while all constrained `i≠j`
  pairs stay strictly feasible. A box would cut off the optimum.

- `DECM` solver-parity tests covering every recommended method × initial-guess combination, plus the gauge
  identities, so neither failure mode can regress silently.

### Changed
- `DECM` `solve_model!` now defaults to `maxiters = 10_000` (was `1000`), and `:Newton` defaults to a
  `ForwardDiff` Hessian rather than the Zygote `SecondOrder` path. Both documented on the method.

## v0.7.1

Compatibility with the modern SciML optimisation stack (`NLsolve` 5 / `Optim` 2).

### Compatibility
- **`NLsolve` is now `"4.5, 5"`.** This is a far larger change than it looks. `NLsolve` 4.5 requires
  `NLSolversBase` 7, `Optim` 1.13 requires `NLSolversBase` 7.9 and `Optim` 2 requires `NLSolversBase` 8,
  so the old bound transitively pinned the package to **`Optim` 1** — and with it `OptimizationOptimJL`
  0.4.8, even though our own `OptimizationOptimJL = "0.4"` already admitted 0.4.21 (which requires
  `Optim` 2). That pin was accidental rather than intended. Both stacks now solve correctly; `Optim` 1
  remains supported and is what the `downgrade` job exercises.
- Julia **1.13** added to the CI matrix (1.10 LTS through 1.13, plus `pre`). No source change was
  needed: `julia = "1.10"` already admits it.

### Fixed
- **Models whose initial guess contains `Inf` could not be solved by any gradient method.**
  `solve_model!` neutralises the `Inf` entries of `θ₀` (zero-degree / zero-strength classes) before
  solving, but on the `DBCM`, `BiCM`, `DECM` and `UECM` it did so only inside the `:fixedpoint` branch,
  while restoring `m.θᵣ[ind_inf] .= Inf` in **both**. The optimisation branch therefore handed a
  non-finite `θ₀` straight to `Optim`. `Optim` 1 absorbed that silently; `Optim` 2 validates every trial
  iterate (`accept_step!`) and aborts the solve, so `DBCM(maspalomas())` — which has zero-degree nodes
  and sits in the precompile workload — made the **package fail to precompile**, taking the docs build
  down with it. The neutralisation is now unconditional, matching what the `RBCM` already did, and the
  `DBCM` and `BiCM` now also **restore** those entries to `Inf` after an optimisation solve — previously
  that happened only by accident, because the `Inf` was handed to the optimiser and came back untouched,
  and the `DECM`/`RBCM`/`UECM` already restored it explicitly in both branches. Results are unchanged on
  `Optim` 1: degree residuals on *maspalomas* stay at `2.25e-9` (out) / `4.46e-9` (in), and the four
  solvers still agree on `Ĝ` to `8.1e-9`.
- **`UECM` first-order solves are now genuinely box-constrained instead of relying on `NaN`.** The
  likelihood is only defined on `yᵢyⱼ < 1`, and because the diagonal self-pair term
  (`om_c2 = -expm1(-2βᵢ)`) is evaluated for every class, that domain is *exactly* the open box
  `βᵢ > 0 ∀i`. It was previously solved unconstrained, leaving the out-of-domain `NaN` to repel the line
  search. `Optim` 2 (with `LineSearches` 7.8) turns a line search that cannot reach a finite point into a
  hard failure rather than muddling through, which pinned `BFGS` against the barrier: on the *rhesus
  macaques* network it stopped at `-L = 2189.7` with `min βᵢ ≈ 2e-14`, against a true optimum of
  `264.8103`. `BFGS` and `LBFGS` now carry the box `βᵢ ≥ 1e-10` and reach `264.8103` under **both**
  `Optim` 1 and `Optim` 2. `Newton` is unchanged (`Fminbox` does not accept it, and it converges
  unconstrained). Networks whose ML solution genuinely pushes a `βᵢ` onto the boundary now report it
  resting on that floor instead of at an arbitrary value produced by the barrier.

### CI
- **The `downgrade` job no longer fails on every CompatHelper PR.** `julia-runtest` leaves
  `force_latest_compatible_version` at `auto`, which it flips to `true` on CompatHelper/Dependabot
  branches — the exact opposite of what a floors job tests, and unsatisfiable in combination with it:
  forcing latest pulls `Symbolics` 7 (a test-only dep), which needs `Preferences ≥ 1.5`, while the
  downgrade pin holds `Preferences` at `~1.4`. It is now pinned to `false` for that job.

## v0.7.0

Convergence is now expressed in the units users actually care about.

### Breaking
- **The `CReM`, `DCReM` and `CRWCM` fixed-point solves now iterate in `log θ`, so `ftol` is a
  *relative* constraint tolerance rather than an absolute parameter-space one.** The fitted `θ` move
  at roughly the `1e-8` level, and for the `DCReM`/`CRWCM` they may also land on a different (equally
  valid) representative of their gauge orbit, so values compared bit-for-bit against stored 0.6.x
  output will differ. All gauge-invariant predictions (`Ĝ`, `Ŵ`, the dyadic probabilities, every
  metric) agree to `~1e-9`.

### Fixed
- **`ftol` silently failed to control accuracy on the two-step weighted models.** It is forwarded to
  `NLsolve`, which bounds the fixed-point *increment* `‖G(θ) - θ‖∞` in **parameter** space. Because
  the map obeys `Gᵢ = θᵢ⟨sᵢ⟩/sᵢ` exactly, the achieved constraint residual was
  `≈ ftol · max(sᵢ/θᵢ)`, and since `θ` scales like `1/s` that factor grows as the **square of the
  weight scale**. It is `~10` for the binary models (harmless) but `~5·10³` on the weighted layer of
  the *rhesus macaques* network, and `~4.5·10⁷` once its weights are scaled by 100. Concretely, a
  `DCReM` on a network with weights of order `10³` returned `retcode Success` with a strength
  constraint off by **61.7** in absolute units. Iterating in `log θ` makes the increment exactly
  `log(⟨sᵢ⟩/sᵢ)`, so the residual is now scale-invariant: measured at a constant `3.2e-9` relative
  across a 1000× range of weight scales, where it was previously `5.3e-5`, `0.45` and `61.7`.
  This also enforces `θ > 0` for free.
- **The `UBCM` was unsolvable at large scale: `exp` overflow made the default fixed point crash and
  the quasi-Newton path report false convergence.** Both kernels evaluated terms of the form
  `x/(1 + x·y)` with `x = exp(-θ)`, which overflows to `Inf/Inf = NaN` once a fitted parameter
  passes `θ < -710` (hub nodes on graphs with more than a few hundred distinct degrees). The fixed
  point then crashed with an `NLsolve.IsFiniteException`, while `BFGS` aborted its line search on
  the `NaN` gradient and reported `Success` at a garbage point (constraint residual `~4·10⁴`).
  The kernels now use the algebraically identical, overflow-safe forms `1/(exp(θⱼ) + exp(-θᵢ))`
  (fixed-point map) and `1/(1 + yᵢ·xⱼ)` (gradient) at the same cost, and the fixed-point solve
  retries once with damped Anderson acceleration (`m = 5`, `β = 0.5`) if the accelerator itself
  diverges. A 250,000-node scale-free graph (1,044 distinct degrees) now solves in ~0.2 s with a
  relative constraint residual of `9·10⁻⁹`; results on small graphs are bit-identical.
- The `θ` accessors of the two-step models (`strength`, `outstrength`, `instrength`) rebuilt the full
  per-node fitness vectors inside every per-node call, and their accumulator was type-unstable
  (`zero(precision(m))` does not infer). The vector forms are **60-64× faster** with ~200× fewer
  allocations; returned values are bit-identical.

### Added
- **`DECM`** — the Directed Enhanced Configuration Model (constrains the out-/in-degree **and** the
  integer out-/in-strength sequences, jointly — the directed counterpart of the `UECM`). Numerically-
  stable log-likelihood over the ordered pairs, branch-free SIMD gradient (verified against Zygote),
  reduction on unique `(k^out, k^in, s^out, s^in)` quadruples, seeded reproducible sampling
  (Bernoulli–geometric per directed channel), the full accessor/variance/information-criterion API
  (`k = 4N` parameters, `n = N(N-1)` observations), a NEMtropy (`decm_exp`) performance/accuracy
  comparison, and documentation. Because the likelihood is only defined on the feasible region
  `β^out_i + β^in_j > 0`, the solver uses a `BackTracking` line search (`BFGS` default; the fixed
  point is unstable for this model, as for the `UECM`). The delta-method `σₓ` carries **no**
  within-dyad covariance term: the two directions of a dyad are independent random variables
  (validated symbolically and by a 10k-sample Monte-Carlo gate in `validation/`).
- **`constraint_residual(m; relative=false)`**: what a solve actually achieved, in constraint units.
  It reuses each model's existing analytical gradient, which by ERGM stationarity *is* the constraint
  residual `⟨xᵢ⟩ - xᵢ`, so it is exact and costs well under 1% of a solve. Available for all nine
  models; the `relative` form masks zero-valued constraints (dead channels).

### Changed
- The `ftol` and `g_tol` docstrings now say what those knobs actually bound. `ftol` bounds the
  fixed-point increment in parameter space and is **not** the constraint residual; `g_tol` maps to
  Optim's `g_abstol`, which is a stopping criterion rather than a guarantee, since Optim may also
  stop on its function or parameter checks. Both point at `constraint_residual`.
- Passing `ftol` on a path that ignores it (for example the `UECM`'s default `:BFGS`, where it was
  silently discarded) now warns instead of doing nothing quietly.
- The `UECM`'s BackTracking optimizer instances were hoisted from `UECM.jl` into
  `backtracking_optimization_methods` (`src/Models/models.jl`) so the enhanced models (`UECM`/`DECM`)
  share them. Internal rename only; behaviour is unchanged.

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
