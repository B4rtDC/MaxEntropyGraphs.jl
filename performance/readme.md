# Benchmarking

This folder lets a third party reproduce the performance and accuracy claims for
`MaxEntropyGraphs.jl`, comparing it against two Python packages: `NEMtropy` and `NuMeTriS`.

All eight models are benchmarked, each against the Python package that implements it:

| Model | Comparator | Model string in the comparator |
| --- | --- | --- |
| `UBCM` | `NEMtropy` | `cm_exp` |
| `DBCM` | `NEMtropy` | `dcm_exp` (plus `expected_dcm_3motif_*` for the motif spectra) |
| `BiCM` | `NEMtropy` | `BipartiteGraph` |
| `UECM` | `NEMtropy` | `ecm_exp` |
| `CReM` | `NEMtropy` | `crema` |
| `RBCM` | `NuMeTriS` | `RBCM` |
| `DCReM` | `NuMeTriS` | `DBCM+CReMa` |
| `CRWCM` | `NuMeTriS` | `RBCM+CRWCM` |

`NuMeTriS` is the reference implementation accompanying Di Vece et al., so it is the natural
comparator for the three reciprocity-aware models. Its model names differ from ours (see the table
above), and it solves both the binary and the weighted layer for the mixture models, which matches
our two-step `solve_model!`.

## What is compared

* **Speed** — model creation, parameter computation (fixed-point / quasi-Newton / Newton),
  and (for the BiCM) the validated bipartite projection, using
  [`BenchmarkTools.jl`](https://github.com/JuliaCI/BenchmarkTools.jl) on the Julia side and
  [`pytest-benchmark`](https://pytest-benchmark.readthedocs.io/) on the Python side.
* **Accuracy** — that both implementations converge to the *same, correct* maximum-likelihood
  solution, measured as how well the fitted model reproduces the imposed degree constraints
  (see [Accuracy comparison](#accuracy-comparison) below).

## Setup

### Julia environment

The Julia environment is described by `Project.toml` (pinned to the Julia 1.10 LTS baseline, see
`../NOTES.md`). The harness `dev`s the *local* package so the
benchmarks test this checkout's code, not a registry release:

```bash
julia --project=. -e 'using Pkg; Pkg.develop(path=".."); Pkg.instantiate()'
```

### Python environment (uv — cross-platform)

The Python side (both NEMtropy and NuMeTriS) uses the [uv](https://docs.astral.sh/uv/) package manager and works on both
macOS and Linux. **`uv` and `julia` are assumed to be on `PATH`.** `benchmarks.sh` sets this up
automatically, but to do it by hand:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
source .venv/bin/activate            # Windows: .venv\Scripts\activate
```

> `requirements.txt` pins `numpy==1.26.4` on purpose: NEMtropy 3.0.3 still uses `np.infty`, which was
> removed in NumPy 2.0.

`benchmarking.yml` is retained only as an optional, fully-locked **Linux** conda alternative
(`conda env create -f ./benchmarking.yml`).

## Running the benchmarks

The whole suite is driven by `benchmarks.sh`:

```bash
nohup ./benchmarks.sh >> benchmark.log 2>&1 &   # the full suite takes >24h on the large graphs
```

It is controlled by environment variables:

| Variable | Default | Effect |
| --- | --- | --- |
| `BENCH_CORES` | `4` | Core budget applied **fairly to both implementations**. Creation + parameter computation are single-threaded compute in both libraries, so this caps Julia's BLAS threads and Python's `OMP`/`OPENBLAS`/`MKL`/`NUMBA` threads to the same number (a same-core comparison); it also sets Julia's thread count and the NEMtropy sampler's `cpu_n`. Recorded as `system_info.bench_cores`/`blas_num_threads`. Run at `1` and `4` to show the comparison is fair either way. |
| `BENCH_MAX_SCALE` | `large` | Caps the problem size: `small` \| `medium` \| `large`. `medium` skips the >24h 250k-node UBCM graph, so `BENCH_MAX_SCALE=medium` is a good tractable run. |
| `BENCH_QUICK` | `0` | Back-compat alias: `1` is equivalent to `BENCH_MAX_SCALE=small` (karate-scale smoke test). |
| `BENCH_SKIP_PROJECTION` | `0` | `1` skips the (slow) BiCM projection benchmark. |
| `SKIP_PYTHON` | `0` | `1` skips the Python benchmarks, both NEMtropy and NuMeTriS (Julia only). |
| `SKIP_PLOTS` | `0` | `1` skips the plotting step. |

For example, a fair, tractable comparison across small + medium at a single core:
`BENCH_CORES=1 BENCH_MAX_SCALE=medium ./benchmarks.sh`.

### Sampling and reproducibility

The suite also times ensemble sampling, where the head-to-head shows an unusually large gap
(~10⁴–10⁵×). **This gap is architectural rather than purely algorithmic, and should be read that
way.** MaxEntropyGraphs.jl's `rand(model, n; rng)` generates each replicate *in memory* as a native
`Graphs.jl` object — exactly what a downstream analysis consumes — with a tight, allocation-light
sampler run thread-parallel across the `n` replicates and no I/O. NEMtropy's `ensemble_sampler`
instead **writes every sampled graph to disk** as an edge-list file and carries a large fixed
per-call overhead (Python/numba dispatch and array setup, plus a multiprocessing pool whose spawn
cost *exceeds the sampling work itself* at these sizes — which is why `cpu_n=4` is measured as
*slower* than `cpu_n=1`). Measured per graph, NEMtropy is ≈0.5 s for a 34-node graph versus
microseconds for MaxEntropyGraphs. So the reported ratio mostly reflects in-memory native-object
generation versus disk-backed, overhead-heavy file writing; a pure compute-only comparison would be
much closer. Because NEMtropy's sampler does not scale, its sampling benchmark is only emitted for
the small problem (MaxEntropyGraphs is benchmarked at every scale), and NEMtropy has **no bipartite
sampler at all** (BiCM sampling is Julia-only).

Both libraries reproduce a seeded sample independently of the core count; `reproducibility_check.py`
verifies this for NEMtropy (the MaxEntropyGraphs side is covered by the package's `test/solver.jl`
"sampling reproducibility" testset).

### How it works

1. The Julia drivers (one `<MODEL>_benchmarks.jl` per model) define the reference graphs, write
   them to edge lists in `data/` (so Python uses the *same* graphs), benchmark the Julia
   implementation, and **generate** the matching Python scripts from the templates in
   `benchmark_helpers.jl`.
2. The generated Python scripts (`<MODEL>_*.py`, each run via a generated `<MODEL>_script.sh`)
   benchmark NEMtropy or NuMeTriS, whichever the table above lists for that model. Both the `.py`
   and the `.sh` are generated artifacts, regenerated on every run and not tracked in git.
3. `accuracy_comparison.jl` checks that each implementation satisfies its own constraints.
4. The `<MODEL>_plots.jl` scripts render the comparison figures.

The Python *templates* live in `benchmark_helpers.jl` (`generate_python` for the NEMtropy models,
`generate_NuMeTriS_python` for the reciprocity-aware ones), so a reviewer can inspect exactly what
is run against each package without first executing Julia.

**Matched solver settings.** For an apples-to-apples comparison, both implementations use the same
convergence settings: a gradient tolerance of `1e-8` (`g_tol` in Julia / `tol` in NEMtropy) and an
iteration budget of `1000` (`maxiters` / `max_steps`). Without this, NEMtropy's default 100-step cap
would let it stop at a looser solution and look faster; with matched settings the timing differences
reflect the solvers, not their stopping criteria. (Even so, on the BiCM quasi-Newton solve NEMtropy
stalls at a ~10⁻⁷ solution while MaxEntropyGraphs.jl converges to ~10⁻⁹ — see `accuracy_comparison.jl`.)

## Solver-method correspondence

The libraries name their solvers differently. The benchmarks line them up as follows:

| MaxEntropyGraphs.jl (`solve_model!(method=…)`) | NEMtropy (`solve_tool(method=…)`) | Benchmark labels (Julia / NEMtropy) |
| --- | --- | --- |
| `:fixedpoint` (Anderson-accelerated fixed point) | `fixed-point` | `…-FP` / `…-fixed-point-degrees` |
| `:BFGS`, `:LBFGS` (quasi-Newton) | `quasinewton` | `…-QN-BFGS-AG` / `…-quasinewton-degrees` |
| `:Newton` | `newton` | `…-Newton-ADF` / `…-newton-degrees` |

Key implementation difference: NEMtropy uses a hard-coded analytical gradient/Hessian, whereas
MaxEntropyGraphs.jl can use either an analytical gradient (`analytical_gradient=true`, suffix `AG`)
or automatic differentiation (`AD_method=:AutoForwardDiff`, suffix `ADF`). This is why the Newton
method (which needs the Hessian) is relatively slower in Julia: the Hessian is obtained by AD.

`NuMeTriS` does not expose a comparable choice of solver method. It is driven through the single
entry point `G.solver(model=…)`, which the generated scripts call with the same convergence
settings used for the NEMtropy comparisons (`maxiter=1000`, `tol=1e-8`), while the Julia side is
still benchmarked across each of its own methods.

Two `NuMeTriS` conventions matter when reading the reciprocity results, and both are documented at
length in `NOTES.md` (EXP-004):

* **Gauge freedom.** The fitted parameters carry a per-block weighted gauge, so raw parameter values
  must never be compared directly between the two packages. The comparison is made on
  gauge-invariant reconstructions (expected sequences), each measured against its own package's
  observed sequences, which also removes any need to align node ordering.
* **Counting conventions.** The triadic motif and flux definitions differ by fixed per-motif
  factors, which `accuracy_comparison.jl` applies before comparing. Once aligned, the motif counts
  agree exactly and the fluxes agree to ~1e-16.

## Accuracy comparison

`accuracy_comparison.jl` validates that the solutions are *correct*, not just fast. For each model
it reports the maximum absolute difference between the expected and observed degree sequence (zero
at the maximum-likelihood solution), and writes `accuracy/accuracy_summary.json`. On the
representative small problems MaxEntropyGraphs.jl reproduces the constraints to ≈1e-7.

When the generated Python scripts run, they additionally write a best-effort solution dump for the
comparator they used: `accuracy/<name>_nemtropy.json` (keys: `dseq`/`expected_dseq` for unipartite
models, `rows_deg`/`cols_deg`/`expected_dseq_rows`/`expected_dseq_cols` for bipartite), or
`accuracy/<name>_numetris.json` for the reciprocity-aware models. When such a dump is present,
`accuracy_comparison.jl` reports the Python package's constraint violation alongside Julia's, so the
implementations are compared directly (each against its own observed sequences, so no node-ordering
alignment is required). NuMeTriS does not expose its expected sequences, so they are reconstructed
from the fitted parameters, which is valid because the reconstruction is gauge-invariant.

## Reproducibility notes

* **Graphs.** No reference graph needs to be downloaded or supplied: every one is either regenerated
  or tracked. `UBCM_medium`/`UBCM_large` are random Barabási-Albert graphs created with a fixed seed
  (`seed=161`); `UBCM_small` is the karate club; `BiCM_small`, `DBCM_small` and the `UECM`/`CReM`
  inputs are the deterministic `corporateclub()`, `maspalomas()` and `rhesus_macaques()` demo
  networks; and the `RBCM`/`DCReM`/`CRWCM` inputs are tiled from the last of these. All of these are
  regenerated rather than committed (the on-disk `UBCM_large.csv` alone is ~94 MB).
  `BiCM_medium`/`BiCM_large` are required input edge lists that cannot be regenerated, so **those two
  CSVs are tracked in git** (`data/BiCM_medium.csv`, `data/BiCM_large.csv`).
  Note that the seeded graphs are reproducible for a *fixed* Graphs.jl: the compat bound here allows
  any 1.x, so a future change to its generator or RNG would alter `UBCM_medium`/`UBCM_large`. Pin
  Graphs.jl exactly if you need to reproduce the published numbers bit-for-bit.
* **Threads.** Pin `JULIA_NUM_THREADS` and compare runs with the same value; the thread count is
  recorded in each result file.
* **Python pins.** `nemtropy==3.0.3`, `numetris==0.1.1`, `numpy==1.26.4`, `numba==0.60.0`,
  `scipy==1.14.0`, `networkx==3.3`, `bicm==3.3.0`, `pytest==9.0.3`, `pytest-benchmark==5.2.3`,
  Python 3.12. `requirements.txt` is the authoritative list. Note `numpy` must stay below 2.0
  (NEMtropy 3.0.3 still calls `np.infty`, which NumPy 2.0 removed), and `matplotlib==3.9.0` has no
  wheel for Python 3.13 or later, so stay on 3.12 (`benchmarks.sh` pins it via `uv venv --python 3.12`).

## Output layout

| Folder | Contents | Tracked in git? |
| --- | --- | --- |
| `data/` | Reference graphs as edge lists | only the `BiCM_medium`/`BiCM_large` inputs |
| `benchmarks/` | Raw benchmark results (JSON), per Julia/Python version | no |
| `accuracy/` | Constraint-violation summary + NEMtropy and NuMeTriS dumps | no |
| `samples/` | Sampled networks | no |
| `plots/` | Generated figures (PDF/PNG) | no |

## Kernel acceleration note

`metrics_acceleration.tex` is a separate LaTeX note, not part of the benchmark suite described
above. It documents how the graph-metric kernels in [`../src/metrics.jl`](../src/metrics.jl) were
accelerated, and *proves* that the accelerated forms compute the same thing as the naive loops they
replace: the average nearest-neighbour degree (`O(N³)` → `O(N²)`), the triangle count as `tr(A³)/6`
with `O(N)` peak memory, the thirteen directed three-node motifs in an inclusion–exclusion matrix
form, the count of "pure" squares under an exact 8× dihedral symmetry reduction, and the bipartite
V-motif marginal-sum closed form.

For each kernel it gives the exact identity behind the reformulation, a proof of computational
equivalence, the resulting complexity and peak-memory behaviour, and the two constraints that shape
the implementations: the kernels sit on the automatic-differentiation path used for metric
variances, and they are mapped over many sampled graphs inside a thread-parallel outer loop, so
per-call memory must not scale with the thread count. The equivalence proofs are stated for
**real-valued** matrices rather than merely `0/1` ones, because the kernels also run analytically on
the dense expected matrix `Ĝ`, whose entries are probabilities.

The matching regression tests are the `metrics acceleration` testset in
[`../test/metrics.jl`](../test/metrics.jl). Build the note with `pdflatex metrics_acceleration.tex`.
