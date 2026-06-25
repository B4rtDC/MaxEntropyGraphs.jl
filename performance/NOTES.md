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

<!-- Template:
### EXP-NNN  <short title>
- Hypothesis:
- Change / config:
- Correctness: (oracle rtol, pass/fail)
- Result: (before -> after; time, allocations, memory)
- Decision: keep / revert / iterate
-->
