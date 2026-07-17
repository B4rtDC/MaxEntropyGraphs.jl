# Performance & experiment log — MaxEntropyGraphs.jl

This file tracks the iterative performance / precision experiments for the 0.5.0
modernization effort (branch `modernize-0.5.0`). Each entry records the hypothesis,
the exact config, and before/after numbers so the work is reproducible and auditable.

Conventions:
- Benchmarks use `BenchmarkTools` via the existing `performance/` harness
  (`benchmark_helpers.jl`, `UBCM_benchmarks.jl`, `BiCM_benchmarks.jl`), reference graphs
  karate / BA(1e4,4) / BA(2.5e5,30), fixed `seed=161`.
- The real cost driver for likelihood/gradient is the **unique-degree count K** (loops are
  O(K²)), not n. Sweep both n and K.
- Every result row records `versioninfo()` + Julia version + thread count + precision.
- Correctness is gated against the Float64 oracle fixture (see Phase 5) at `rtol≈1e-8` for
  parameters and exact for combinatorial counts.

---

## Baseline (Julia 1.10, pre-modernization deps)

- Date: 2026-06-25
- Julia: 1.10.11 (aarch64-apple-darwin), `JULIA_NUM_THREADS=2`
- Deps held at compat caps: Optimization v3.28, ForwardDiff v0.10.39, Zygote v0.6.77,
  OptimizationOptimJL v0.1.14, OptimizationNLopt v0.1.8.
- Finding: current `Project.toml` **resolves cleanly on 1.10** (old majors held by `[compat]`).
- Test-suite status: _(pending — baseline run in progress)_

---

## Experiments

### EXP-001  Debranch the UBCM gradient inner loop (Phase 6a)
- Hypothesis: the `if i==j` branch inside the O(K²) inner loop of `∇L_UBCM_reduced!` blocks SIMD;
  factoring `F[i]` out and handling the diagonal once should speed it up.
- Change: inner loop sums `Σⱼ F[j]·g(xᵢxⱼ)` branch-free (g(z)=z/(1+z)), then subtract one self term
  `g(xᵢ²)` to correct the diagonal `(F[i]-1)` vs `F[i]`; multiply by `F[i]` once. Same for the
  `_minus!` variant. `src/Models/UBCM.jl`.
- Correctness: vs the original branchy version, max relative diff ~1e-15 (machine epsilon) on
  BA graphs; full test suite (Zygote-vs-analytical `≈`) still green.
- Result (Julia 1.10, BA(n,6), `@elapsed` mean of 300 calls):
  | n | unique-deg K | old | new | speedup |
  |---|---|---|---|---|
  | 2 000   | 71  | 13.6µs | 1.6µs  | 8.6× |
  | 20 000  | 156 | 78.1µs | 7.1µs  | 11.0× |
  | 100 000 | 274 | 252µs  | 21.5µs | 11.7× |
- Decision: KEEP. The analytical gradient is called every solver iteration, so this directly
  speeds up `solve_model!(analytical_gradient=true)`. Follow-up: apply the same transform to the
  DBCM/BiCM gradients (different nz-index loop structure).

### EXP-002  Debranch/factor the DBCM + BiCM gradients (follow-up to EXP-001)
- Change: DBCM — fold the `if i≠j` diagonal correction into arithmetic `(F[j]-(i==j))` (branch-free)
  and factor `F[i]` out of the α-part inner loop. BiCM — already branch-free (bipartite, no diagonal);
  factor the outer-constant `f·x` / `f·y` out of the inner reduction. `_minus!` variants mirrored.
- Correctness: exact vs the original branchy code (max rel diff DBCM 3e-15, BiCM 5e-16); full test
  suite (Zygote-vs-analytical) green.
- Result: DBCM ≈1.03× (494 unique pairs), BiCM ≈1.17×. **Marginal** — unlike UBCM these loops are
  division-bound, and the DBCM `(i==j)` compare still blocks full SIMD vectorization.
- Decision: KEEP (correct, branch-free, cleaner, not slower). Future: a true SIMD debranch for DBCM
  would need the diagonal self-correction handled outside the inner loop (membership of nz_out∩nz_in),
  and reciprocal-approximation/`@fastmath` could help the division-bound divisions.

### EXP-003  Matrix-free Ĝ audit (Phase 6c)
- Question: the dense expected-adjacency matrix is O(n²) (~500 GB at n=250k Float64). How much of
  the package forces materializing it?
- Finding (code audit + measurement): **metric values and sampling are already matrix-free** — they
  use the element accessors `A(m,i,j)` / `f_UBCM`/`f_DBCM`/`f_BiCM` over the reduced parameters, never
  the matrix. Measured at n=8000: `degree(m)` allocates 0.25 MB, one `A(m,i,j)` call allocates 0 bytes
  (a dense Ĝ would be 512 MB). The dense path is hit ONLY by the opt-in `Ĝ(m)`/`set_Ĝ!`, `σˣ(m)`/
  `set_σ!`, the `precomputed=true` branches (gated, documented), and the custom-metric error-propagation
  `σₓ` (UBCM:995 `sqrt(sum((m.σ.*∇X)²))`), which needs the dense `m.σ`.
- Decision: NO large refactor needed for the common case (documented in `docs/src/performance.md`). Scoped
  remaining target: make `σₓ` variance propagation matrix-free (stream σ_ij·∇X_ij), and audit the
  triangle/square/motif std paths. Left as a focused follow-up (per-function correctness needed).

### EXP-004  NuMeTriS cross-validation of the reciprocity-aware models (RBCM / DCReM / CRWCM)
- Date: 2026-07-11, Julia 1.12.5 (aarch64-apple-darwin), BENCH_CORES=4; NuMeTriS 0.1.1 (added to
  requirements.txt / the uv venv), numpy 1.26.4, numba 0.60.0.
- Question: do the new reciprocity-aware models (RBCM, DCReM, CRWCM) converge to the same
  maximum-likelihood solution as the reference implementation (NuMeTriS, Di Vece et al. 2023), and do
  the triadic statistics agree across packages?
- Setup: `{RBCM,DCReM,CRWCM}_benchmarks.jl` + a generic `generate_NuMeTriS_python` template
  (`Graph(adjacency=W).solver(model='RBCM'|'DBCM+CReMa'|'RBCM+CRWCM', maxiter=1000, tol=1e-8)`);
  reference graph = rhesus_macaques (directed, r_t=0.757; binarised for the RBCM), larger problems by
  block-diagonal tiling. NuMeTriS does not expose expected sequences, so its own (gauge-invariant)
  constraint violations are reconstructed from its fitted parameters on the Python side and dumped to
  `accuracy/*_numetris.json` (consumed by accuracy_comparison.jl).
- Empirically verified NuMeTriS conventions (cost ~an hour; don't relearn):
  * `dseq_right/dseq_left/dseq_rec` == our k→/k←/k↔ and `stseq_*` == our reciprocal strengths (same
    node order, same definitions);
  * the binary Lagrange multipliers match ours exactly (exp(-γ_NuMeTriS) == our zᵣ to ~1e-5 —
    identical MLE, identical parameterisation); weighted blocks are ordered [β→; β←; β↔out; β↔in]
    (direct rates), identified only up to a per-block gauge (θᵒ+c, θⁱ−c) — never compare raw values;
  * the NuMeTriS solver is NOT re-entrant (re-solving a solved `Graph` can raise
    `numpy.linalg.LinAlgError`), so the pytest solve benchmark solves a fresh Graph per round
    (pedantic + setup, setup excluded from timing);
  * triadic conventions differ deterministically: we count LABELED ordered triples (× |Aut(m)|, the
    Squartini/NEMtropy convention — NEMtropy's DBCM motifs match ours to 1.6e-8) and total flux per
    occurrence; NuMeTriS counts each occurrence once and divides the flux by the motif's link count.
    Per-motif factors: Nm ×[2,1,1,2,1,2,1,2,3,1,2,1,6], Fm additionally ×#links [2,2,3,2,3,4,3,4,3,4,4,5,6].
- Result (rhesus, small problems; accuracy/accuracy_summary.json):
  * max constraint violation — RBCM: MEG 4.5e-8 vs NuMeTriS 8.7e-9; DCReM: MEG 5.3e-5 (fixed point at
    ftol=1e-8 on strengths summing to ~1300, rel ~1e-8) vs NuMeTriS 8.4e-9; CRWCM: MEG 2.1e-5 vs
    NuMeTriS 9.4e-9 — every implementation reproduces its constraints at its own solver tolerance;
  * empirical triadic statistics after convention alignment: Nm max rel.diff = 0.0 (exact),
    Fm max rel.diff = 1.9e-16 (machine epsilon) — both motif/flux implementations agree exactly.
- Decision: keep. Full-scale timing runs happen via `benchmarks.sh` (the three new models are wired
  in); the small-scale runs here only validate the pipeline end-to-end.

### EXP-005  Accelerate the triadic flux and intensity kernels (reciprocity metrics follow-up)
- Date: 2026-07-12, Julia 1.12.5 (aarch64-apple-darwin), BLAS threads = 1 (metrics_benchmarks.jl default).
- Question: after adding the reciprocity metrics (EXP-004 models), do the new kernels leave gains on
  the table, in the spirit of EXP-001..003?
- Audit: `motifs(m::RBCM)` already uses the shared-base-matrix BLAS form (13 matmuls, same as the DBCM
  path — nothing to gain); the O(N²) kernels (reciprocal sequences, reciprocity, Ĝ/Ŵ/σ/covariances)
  are loop-bound and fine. Two kernels had headroom:
- Change 1 — `_motif_fluxes` (`src/metrics.jl`): the flux term with the weighted factor at position q
  is rewritten by trace cyclicity as tr(W_q · F_a F_b) = dot(W_q, (F_a F_b)ᵀ), so only BINARY pair
  products appear as matrix multiplications, and they repeat across motifs/positions → computed once
  in a shared cache (≤12 distinct products serve all 33 weighted terms, vs one matmul per term
  before); the zpos diagonal corrections reduce to dots as well. Trade-off: the product cache
  (mutation) makes the value path non-Zygote-differentiable; flux z-scores are sampling-based anyway.
- Change 2 — `motif_intensities` (`src/metrics.jl`): the per-triple `Dict` lookup + symbol branching
  is replaced by precomputed per-pair state codes / weight products / link counts (O(N²)) and a
  constant 4×4×4 state→motif lookup table, so the O(N³) triple loop only reads arrays.
- Correctness: exact vs the previous implementations (rtol 1e-10 inline check for the fluxes; the
  full metrics suite incl. the naive triple-loop reference and the intensities==motifs-on-binary
  identity, plus the ensemble ⟨Fm⟩ assertions, all green).
- Result (dirsparse: dense directed real, ~35% link density; new head-to-head entries in
  metrics_benchmarks.jl):
  * motif_fluxes vs previous trace form: 2.2x at N=150/400/1000 (6.97→3.17 ms, 88.8→39.8 ms,
    1429→668 ms); vs the naive O(N³) triple loop: 60x (N=50) → 106x (N=150).
  * motif_intensities vs the Dict/symbol version: ~7x at N=50–150 (both O(N³); constant factor),
    at the cost of three O(N²) work arrays (~224 KiB at N=150).
- Decision: keep both. No further headroom identified: fluxes are now BLAS-3-bound like the motif
  spectrum; intensities are inherently O(N³) (geometric means do not factorise over dyads).

### EXP-013  Delta-method variance (σₓ): materialized vs matrix-free strategies (0.6.0)
- Hypothesis: `σₓ(m, X)` materializes two dense `n×n` matrices (`Ĝ`/`Ŵ` and `σ`/`σʷ`) plus the AD
  workspace, so it should hit a memory wall well below the metric-value paths; a matrix-free
  analytic-gradient streaming path (closed-form `∂X/∂g_ij` + the element accessor `A(m,i,j)`) should
  scale much further for the metrics whose gradient is known in closed form.
- Change / config: new `performance/variance_benchmarks.jl` (BenchmarkTools; `@belapsed` + `@allocated`;
  `BENCH_QUICK`, `BENCH_MAX_N`, per-strategy caps). Strategies per model/metric/size:
  `setup` (materialize once), `stored` (shipped path, per AD backend), `fresh` (rebuild the matrices
  every call), `analytic` (hand-derived streaming gradient). All strategies cross-checked `isapprox`
  at every N. Julia 1.12.5 (aarch64), BLAS threads = 1. Synthetic BA / bipartite / weighted graphs.
- Correctness: every strategy agrees to `rtol 1e-6` at every size (the cross-check is part of the run).
- Result (representative rows; time / allocation):
  * **AD backend choice.** For the **matmul metrics** (`triangles`, motifs) **ReverseDiff is the only
    viable backend**: UBCM triangles at N=100 is 188 µs / 0.8 MiB (ReverseDiff) vs 2.5 ms / 48 MiB
    (Zygote) vs **1.97 s / 835 MiB (ForwardDiff)**; Zygote's BLAS pullback then explodes —
    5.9 s / **44.8 GiB** at N=1000 and 241 s / **1.38 TiB** at N=3162. For **linear metrics**
    (`sum`, degree, ANND) all three agree and Zygote is ~1.5–2× faster than ReverseDiff (e.g. UBCM
    sum N=3162: 22 ms / 153 MiB Zygote vs 30 ms / 305 MiB ReverseDiff), while ForwardDiff is still
    ~1000× slower even at N=100. ⇒ the shipped default `:ReverseDiff` is correct across both classes.
  * **Dense materialization wall.** `setup` (`set_Ĝ!`+`set_σ!`) is O(n²): 152 MiB / 0.16 s at N=1000,
    2.16 GiB / 2.5 s at N=3162, **16.2 GiB / 17.7 s at N=10 000** — the practical ceiling for the
    stored path is ≈ n = 5–8 k on a 16 GB machine (two dense matrices + the AD tape).
  * **stored vs fresh.** Reusing the materialized matrices is ~100× faster per call than rebuilding
    them (UBCM sum N=1000: 2.4 ms stored vs 160 ms fresh). Materialize once if you evaluate `σₓ` for
    more than one or two metrics.
  * **Matrix-free analytic streaming wins decisively on memory** for closed-form-gradient metrics:
    UBCM `sum` **0 bytes** at N=3162; BiCM `sum` **0 bytes** at every N; UBCM `triangles`
    896 B (N=100) → 8 KiB (N=1000) → **28 KiB (N=3162)** — i.e. O(n) memory and O(n³) time, versus
    Zygote's 1.38 TiB at the same size. (The DBCM/DCReM analytic variants call the package accessor
    `A(m,i,j)` and still allocate ≈ n²·8 B because that accessor is not allocation-free — an
    independent minor follow-up.)
- Decision: **keep `:ReverseDiff` as the σₓ default** (documented in `docs/src/performance.md`). Ship the
  materialized path as-is for n ≲ 5 k. A **matrix-free `σₓ` fast path** (closed-form `∂X/∂g_ij` for
  degree / edge-count / ANND / triangles) is now benchmark-justified — it lifts the size ceiling by
  ~2 orders of magnitude — but is deferred to a follow-up because the gradients are metric-specific
  whereas the shipped AD path handles arbitrary user metrics. Recorded in `performance.md` as the recommended
  route for large-n variance.

### EXP-014  DECM added with NEMtropy `decm_exp` cross-validation (0.7.0)
- Hypothesis: the DECM (directed twin of the UECM: out/in-degrees + integer out/in-strengths, jointly)
  transfers the UECM recipe — NaN feasibility barrier + BackTracking BFGS, reduced quadruple
  compression — with DBCM-style ordered-pair sums, and reproduces NEMtropy's `DirectedGraph`/`decm_exp`
  solution on identical CSV edge lists.
- Change / config: `DECM_benchmarks.jl` (rhesus unsymmetrised + directed block-diagonal tilings,
  16 → 128 → 512 nodes), `generate_DECM_python` (weighted 3-tuple edge list, `decm_exp`,
  quasinewton/newton, `strengths` initial guess), `constraint_violation(::DECM)` + `DECM_small`
  reference entry in `accuracy_comparison.jl`. The NEMtropy dump stores the observed/expected
  sequences out;in-concatenated under the same `dseq`/`sseq` keys the UECM reader already handles.
- Correctness: analytic ∇L ≡ Zygote to 2e-13; default BFGS solve residual 3.7e-7 (Newton 3e-11) on
  rhesus; all four sequences recovered via Ĝ/Ŵ row AND column sums (rtol 1e-6); σʷ and the
  covariance-FREE directed σₓ validated by `validation/symbolic/decm.jl` (37 checks) and
  `validation/numeric/decm_weighted_sigma.jl` (33 checks, 10k samples).
- NEMtropy quirks (recorded for future readers, all NEMtropy 3.0.3, measured on `DECM_small`):
  1. A `DirectedGraph` built from a *weighted* edge list leaves `dseq_out`/`dseq_in` as **object-dtype**
     numpy arrays, and the numba `nopython` kernels (`loglikelihood_hessian_diag_decm_exp`) refuse
     them with a `TypingError`. The generated script coerces the four constraint sequences to
     `float64` right after construction (`coerce_sequences`), which fixes the typing.
  2. NEMtropy's `decm_exp` **quasinewton** (diagonal-hessian) recipe stalls at a `3.2e-2` max
     constraint violation on this graph, and its **fixed point diverges** (violation ≈ 49) — the
     latter consistent with the orientation bug below. Full **newton** reaches `3.7e-9`, so the
     accuracy dump uses the newton solution (the pytest timings still cover newton + quasinewton).
  3. `iterative_decm_exp` (the fixed-point map, `models_functions.py:3236`) computes its `fa_in`
     accumulator with `x[j + n]` (= a_in,j) where the in-degree equation requires `x[j]` (= a_out,j) —
     compare `iterative_decm_exp_2` (line 3312), which uses the correct orientation. Our
     `DECM_reduced_iter!` follows the mathematically consistent form (`iterative_decm_exp_2`'s).
- Decision: keep. DECM slots into `benchmarks.sh` between UECM and CReM at all three stages
  (Julia driver, Python wrapper, plots).

### EXP-015  First full-scale campaign on the M4 Max (0.7.0): all nine models, three scales

- Date: 2026-07-16/17, Julia 1.12.6 (aarch64-apple-darwin), CPython 3.12.13, BENCH_CORES=12.
  All published `figures/*_benchmark.pdf` now come from this single machine, toolchain and palette
  (previously: ubcm 2024/Julia 1.9.3, bicm 2024 hand-edited in Preview, crwcm 2026 medium-only).
- Protocol: selective runs via BENCH_MIN_SCALE/BENCH_MODELS; every Python job under a per-job
  wall-clock budget (BENCH_JOB_TIMEOUT, process-group kill, logged in `benchmarks/timeouts.log`).
  The BiCM_large projection variants run at min-rounds=5 (all eight measured, slowest 362 s/round);
  UBCM_large and DECM_medium/large Python run per-test so a slow solver cannot take the rest of its
  file's results with it.
- Two jobs exceeded their budget and are reported as such, not plotted:
  1. NEMtropy `decm_exp` quasinewton at N=512 (over 2 h; its newton finished in 15 min).
  2. The original single-file UBCM_large Python run (the split rerun then measured every method:
     create 8.9 s, fixed point 90.8 s, quasinewton 95.4 s, newton 87.5 s median over 30 rounds).
- The first execution of the large problems on the modern stack surfaced two real solver issues,
  both fixed in src (see commits a896b04, c17564f):
  1. UBCM at 250k nodes: exp overflow in x/(1+xy)-shaped terms made the Anderson fixed point throw
     IsFiniteException and made BFGS report a false success at constraint residual ~4e4 (caught by
     `constraint_residual`). Overflow-safe reformulation + damped-Anderson retry; the default solve
     now reaches relative residual 9.1e-9 in 0.18 s.
  2. CReM/DCReM/CRWCM BFGS at N=512 needs 1000-5000 iterations (converges in ~3 s at 5000); the
     large-scale rows now carry `:maxiters => 5000`.
- UBCM_large plots no quasi-newton row on either side: neither implementation converges there
  within the matched budget, and a non-converging solver's time is meaningless. The 2024 figure's
  BFGS point at 1051 constraints predates any residual check and was likely such an artifact.
- Headline at the largest scales, both sides honest: UBCM 1051 constraints, NEMtropy ~90 s per
  solve vs MaxEntropyGraphs fixed point ~0.03 s; CRWCM N=512, NuMeTriS ~1.3 s vs fixed point ~0.02 s.
- Decision: keep. Rerun cost after fixes: R1 (three weighted Julia larges) 9 min, R2 (UBCM Julia +
  split Python) ~2 h 40 m, zero kills.

<!-- Template:
### EXP-NNN  <short title>
- Hypothesis:
- Change / config:
- Correctness: (oracle rtol, pass/fail)
- Result: (before -> after; time, allocations, memory)
- Decision: keep / revert / iterate
-->
