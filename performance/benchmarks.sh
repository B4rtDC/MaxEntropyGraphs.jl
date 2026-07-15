#!/usr/bin/env bash
#
# Reproducible benchmark driver for MaxEntropyGraphs.jl vs. NEMtropy (Python).
#
# Cross-platform (macOS + Linux). The Python side uses the uv package manager to create an
# isolated virtual environment from requirements.txt, so no conda is needed.
#
# Requirements: julia and uv (https://docs.astral.sh/uv/) on PATH.
#
# The full suite takes a VERY long time (>24h on the large graphs), so run it in the background:
#   nohup ./benchmarks.sh >> benchmark.log 2>&1 &
#
# Environment variables (all optional):
#   BENCH_CORES         Core budget applied fairly to BOTH implementations (default: 4). It caps
#                       Julia's BLAS threads and Python's OMP/OPENBLAS/MKL/NUMBA threads (so the
#                       creation + parameter-computation comparison is same-core), sets Julia's
#                       thread count, and the NEMtropy sampler's cpu_n. Recorded in each result.
#   BENCH_MAX_SCALE     small | medium | large (default: large). Caps the problem size.
#   BENCH_QUICK         Alias: 1 == BENCH_MAX_SCALE=small.
#   BENCH_SKIP_PROJECTION  If 1, skip the (slow) BiCM projection benchmark.
#   SKIP_PYTHON         If 1, skip the NEMtropy/Python benchmarks (Julia only).
#   SKIP_PLOTS          If 1, skip the plotting step.
#
set -euo pipefail
cd "$(dirname "$0")"

export BENCH_CORES="${BENCH_CORES:-4}"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-$BENCH_CORES}"
export BENCH_MAX_SCALE="${BENCH_MAX_SCALE:-large}"
export BENCH_QUICK="${BENCH_QUICK:-0}"
[ "${BENCH_QUICK}" = "1" ] && export BENCH_MAX_SCALE="small"
# Exported so the generated Python projection test can honour it too (not just the Julia driver).
export BENCH_SKIP_PROJECTION="${BENCH_SKIP_PROJECTION:-0}"
# Cap the Python side's IMPLICIT threading (BLAS / numba) to the same core budget as Julia, so
# creation + parameter computation are compared on equal cores. The EXPLICIT parallelism
# (NEMtropy sampler cpu_n / projection threads_num) is controlled separately by the benchmarks.
export OMP_NUM_THREADS="$BENCH_CORES"
export OPENBLAS_NUM_THREADS="$BENCH_CORES"
export MKL_NUM_THREADS="$BENCH_CORES"
export NUMBA_NUM_THREADS="$BENCH_CORES"

echo "$(date) - config: BENCH_CORES=${BENCH_CORES}, BENCH_MAX_SCALE=${BENCH_MAX_SCALE}, SKIP_PYTHON=${SKIP_PYTHON:-0}, BENCH_SKIP_PROJECTION=${BENCH_SKIP_PROJECTION:-0}, SKIP_PLOTS=${SKIP_PLOTS:-0}"

## --- Tooling checks (fail early with a clear message) -----------------------
command -v julia >/dev/null 2>&1 || { echo "ERROR: 'julia' not found on PATH." >&2; exit 1; }
if [ "${SKIP_PYTHON:-0}" != "1" ]; then
    command -v uv >/dev/null 2>&1 || { echo "ERROR: 'uv' not found on PATH. Install uv (https://docs.astral.sh/uv/) or set SKIP_PYTHON=1." >&2; exit 1; }
fi

## --- Python environment (uv, cross-platform) --------------------------------
if [ "${SKIP_PYTHON:-0}" != "1" ]; then
    echo "$(date) - creating uv virtual environment (.venv) and installing NEMtropy"
    uv venv --python 3.12 .venv
    uv pip install --python .venv -r requirements.txt
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

## --- Julia environment ------------------------------------------------------
## `dev` the parent package so the benchmarks test THIS checkout's code rather than a registry release.
echo "$(date) - instantiating Julia environment (developing the local package)"
julia --project=. -e 'using Pkg; Pkg.develop(path=".."); Pkg.instantiate()'

## --- Julia benchmarks (also generate the Python scripts + shell wrappers) ---
echo "$(date) - Starting Julia benchmarks"
julia --project=. -t "${JULIA_NUM_THREADS}" ./UBCM_benchmarks.jl
julia --project=. -t "${JULIA_NUM_THREADS}" ./BiCM_benchmarks.jl
julia --project=. -t "${JULIA_NUM_THREADS}" ./DBCM_benchmarks.jl
julia --project=. -t "${JULIA_NUM_THREADS}" ./UECM_benchmarks.jl
julia --project=. -t "${JULIA_NUM_THREADS}" ./CReM_benchmarks.jl
julia --project=. -t "${JULIA_NUM_THREADS}" ./RBCM_benchmarks.jl
julia --project=. -t "${JULIA_NUM_THREADS}" ./DCReM_benchmarks.jl
julia --project=. -t "${JULIA_NUM_THREADS}" ./CRWCM_benchmarks.jl
echo "$(date) - Finished Julia benchmarks"

## --- Python (NEMtropy) benchmarks -------------------------------------------
if [ "${SKIP_PYTHON:-0}" != "1" ]; then
    echo "$(date) - Starting Python benchmarks"
    # Run with bash, not sh: the generated wrappers are #!/bin/bash and use `source` to activate
    # the venv. On macOS /bin/sh is bash in POSIX mode so `sh` happens to work, but on Debian and
    # Ubuntu /bin/sh is dash, where `source` does not exist and every wrapper would fail.
    bash ./UBCM_script.sh
    bash ./BiCM_script.sh
    bash ./DBCM_script.sh
    bash ./UECM_script.sh
    bash ./CReM_script.sh
    bash ./RBCM_script.sh
    bash ./DCReM_script.sh
    bash ./CRWCM_script.sh
    echo "$(date) - Finished Python benchmarks"
fi

## --- Accuracy comparison (parameter / constraint agreement) -----------------
echo "$(date) - Running accuracy comparison"
julia --project=. -t "${JULIA_NUM_THREADS}" ./accuracy_comparison.jl || \
    echo "WARNING: accuracy comparison did not complete (NEMtropy dumps may be missing)."

## --- Plots ------------------------------------------------------------------
if [ "${SKIP_PLOTS:-0}" != "1" ]; then
    echo "$(date) - Generating the plots"
    julia --project=. ./UBCM_plots.jl
    julia --project=. ./BiCM_plots.jl
    julia --project=. ./UECM_plots.jl
    julia --project=. ./CReM_plots.jl
    julia --project=. ./DBCM_plots.jl
    julia --project=. ./RBCM_plots.jl
    julia --project=. ./DCReM_plots.jl
    julia --project=. ./CRWCM_plots.jl
fi

echo "$(date) - Finished benchmarks"
