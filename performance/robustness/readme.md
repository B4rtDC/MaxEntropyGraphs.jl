# Solver robustness

Does the answer depend on where the solve started? This harness measures accuracy, convergence and
cost across **model x method x initial guess**, which is the one axis the speed benchmarks in
[`../readme.md`](../readme.md) do not sweep: those pin `initial = :degrees` everywhere and vary the
method and the AD backend instead.

It also gives a home to the measurements behind the package's published robustness claims. Those
came from one-off scripts that were never committed, so the tables in `validation/*.md` and
`CHANGELOG.md` could be read but not reproduced. They can now be regenerated from a clean clone.

## Running it

```bash
julia --project=. robustness/sweep.jl      # the model x method x guess grid
julia --project=. robustness/ladder.jl     # the BiCM Anderson-accelerator study
julia --project=. robustness/report.jl     # render the newest of each into markdown + a figure
```

`sweep.jl` is controlled by environment variables:

| Variable | Default | Effect |
| --- | --- | --- |
| `ROB_MODELS` | all ten | Space-separated subset, e.g. `ROB_MODELS="UECM DECM"`. |
| `ROB_METHODS` | all four | Space-separated subset of `fixedpoint BFGS LBFGS Newton`. |
| `ROB_INITIALS` | each model's own set | Space-separated subset. Intersected with what each model accepts, so naming a guess a model does not know drops that cell rather than raising. |
| `ROB_NGRAPHS` | `50` | Graphs per corpus. The published tables used 150 (weighted) and 200 draws (bipartite). |
| `ROB_OUT` | timestamped file in `results/` | Output path. |
| `ROB_VERBOSE` | `0` | `1` prints every row as it lands. |

`ladder.jl` takes `ROB_NDRAWS` (default `200`) and `ROB_OUT`.

## The corpora

Every corpus is built from one seed and repairs isolated vertices deterministically. That repair is
not cosmetic: the framework models connected graphs, the `BiCM` rejects isolated vertices outright,
and `CReM`/`DCReM` raise `NLsolve.IsFiniteException` on a zero-strength node, so a corpus that
leaked one would measure input rejection rather than solver robustness.

| Corpus | Used by | Draw ranges | Seed |
| --- | --- | --- | --- |
| `binary_undirected` | UBCM | `nv` 8-30, `p` 0.15-0.60 | `20260918` |
| `binary_directed` | DBCM, RBCM | `nv` 8-24, `p` 0.15-0.60 | `20260918` |
| `bipartite` | BiCM | `N⊥`, `N⊤` 4-30, `p` 0.05-0.65 | `20260917` |
| `dibipartite` | DBiCM | `N⊥`, `N⊤` 4-20, `p⁺`, `p⁻` 0.1-0.5 | `20260918` |
| `weighted_undirected` | UECM, CReM | `nv` 6-18, `p` 0.15-0.60, `w` 1-8 | `2026` |
| `weighted_directed` | DECM, DCReM, CRWCM | `nv` 6-16, `p` 0.15-0.60, `w` 1-8 | `2026` |

Results are split into **well posed** and **runaway constraint** subsets. A runaway constraint
(`k = 0`, `k = N-1`, or `s = k`) puts the maximum-likelihood optimum at an infinite parameter, so
no solver can settle there at a finite tolerance. Those instances are not solver failures, and
folding them into a headline number hides which methods degrade gracefully and which do not.

## The two anchors

The harness is checked against the published results it is meant to reproduce. If either misses,
the harness is wrong, not the package.

**BiCM Anderson ladder** (`ladder.jl`, 200 draws, 183 constructible), from
`validation/bicm_uecm_solver_geometry.md`:

| strategy | accurate | non-finite aborts | total iterations |
| --- | --- | --- | --- |
| `plain` (default memory) | 158/183 | 25 | 3 612 |
| `m = 2` | 181/183 | 2 | 4 624 |
| `m = 20` | 108/183 | 75 | 2 120 |
| `picard` (`m = 0`) | 183/183 | 0 | 10 413 |
| `damped` retry | 169/183 | 14 | 4 071 |
| `gauge` projection | 127/183 | 56 | 2 851 |
| **`ladder`** (shipped) | **183/183** | 0 | **4 551** |

No strategy ever hits the iteration cap: every failure of the accelerated path is a non-finite
abort. Failures rise monotonically with accelerator memory, because the map is gauge-equivariant
and Anderson's least-squares is built from residual differences that all lie in the orthogonal
complement of the gauge direction.

**UECM/DECM cold start** (`sweep.jl`, 150 graphs each, `initial = :strengths`), from
`validation/uecm_decm_fixedpoint.md`:

| | n | `:fixedpoint` | `:BFGS` | median residual, fp / BFGS |
| --- | --- | --- | --- | --- |
| UECM, well posed | 100 | 98 | 86 | 3.1e-09 / 2.5e-07 |
| UECM, runaway | 50 | 8 | 38 | - / 1.2e-07 |
| DECM, well posed | 71 | 71 | 71 | 1.5e-09 / 3.1e-08 |
| DECM, runaway | 79 | 22 | 78 | - / 3.4e-08 |

Before the block-coordinate-ascent rewrite in v0.8.0 the `:fixedpoint` column was **0** in all four
rows.

## Four things that would silently corrupt a sweep

1. **`ftol` means four different things.** The parameter-space increment on the binary models'
   `:fixedpoint`; the absolute constraint residual on `UECM`/`DECM` `:fixedpoint`; the *relative*
   constraint residual on the two-step models' `:fixedpoint`; and nothing at all on every Optim
   path, where it is ignored with a warning. The only comparable success metric across the grid is
   `constraint_residual(m)`, so that is what is recorded, and `ftol` is treated as an input.
2. **The initial-guess vocabulary is per model.** `:degrees_minor` and `:chung_lu` exist only on
   the binary unipartite models, the bipartite models drop `:degrees_minor`, the weighted models
   speak `:strengths`, and the two-step models have no `:uniform`. Both `:degrees_minor` and
   `:chung_lu` additionally throw when the model was built from a degree sequence rather than a
   graph, since they need `ne(G)`.
3. **`:Newton` pins `AD_method = :AutoForwardDiff`.** With the `:AutoZygote` default,
   `OptimizationBase` wraps a `SecondOrder(AutoZygote, AutoForwardDiff)` whose nested HVP path
   **aborts the Julia process** with `signal 4: illegal instruction` when `Symbolics` is loaded.
   `BiCM`, `DBiCM`, `UECM` and `DECM` carry an internal override; `UBCM`, `DBCM` and `RBCM` do not,
   so an unpinned sweep takes the whole run down with it.
4. **`sol` has five shapes.** An `OptimizationSolution` on every Optim path; an NLsolve result for
   `UBCM`/`DBCM`/`RBCM`/`BiCM` and the two-step models; a `NamedTuple (zero, iterations, residual,
   converged)` for `UECM`/`DECM` `:fixedpoint`; and a per-channel `NamedTuple (out, in)` for the
   `DBiCM`, where an empty channel is `nothing`. Reading `.iterations` blindly records zeros.

## Output

`results/` holds the raw JSON (one file per run, gitignored), plus the rendered
`robustness_report.md` and `robustness_heatmap.pdf` that `report.jl` produces.
