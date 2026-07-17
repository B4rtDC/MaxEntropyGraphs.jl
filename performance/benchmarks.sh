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
#   BENCH_MIN_SCALE     small | medium | large (default: small). Skips the problems below it,
#                       so a run can target only what is missing (e.g. BENCH_MIN_SCALE=large
#                       BENCH_MAX_SCALE=large re-runs just the large problems).
#   BENCH_MODELS        Space-separated subset of "UBCM BiCM DBCM UECM DECM CReM RBCM DCReM CRWCM"
#                       (default: all nine). Only the listed models are benchmarked and plotted.
#   BENCH_JOB_TIMEOUT   Per-job wall-clock budget in seconds for every generated Python benchmark
#                       job (default: 0 = no limit). A job over budget has its whole process group
#                       killed (see run_with_timeout.sh); the event lands in benchmarks/timeouts.log
#                       and the job's results are simply absent, the rest of the run continues.
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
export BENCH_MIN_SCALE="${BENCH_MIN_SCALE:-small}"
export BENCH_JOB_TIMEOUT="${BENCH_JOB_TIMEOUT:-0}"
BENCH_MODELS="${BENCH_MODELS:-UBCM BiCM DBCM UECM DECM CReM RBCM DCReM CRWCM}"
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

echo "$(date) - config: BENCH_CORES=${BENCH_CORES}, BENCH_MIN_SCALE=${BENCH_MIN_SCALE}, BENCH_MAX_SCALE=${BENCH_MAX_SCALE}, BENCH_MODELS=${BENCH_MODELS}, BENCH_JOB_TIMEOUT=${BENCH_JOB_TIMEOUT}, SKIP_PYTHON=${SKIP_PYTHON:-0}, BENCH_SKIP_PROJECTION=${BENCH_SKIP_PROJECTION:-0}, SKIP_PLOTS=${SKIP_PLOTS:-0}"

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
## Each driver is guarded: on an overnight multi-model run, one model's failure must not
## abort the models that come after it (the failure is loud in the log either way).
echo "$(date) - Starting Julia benchmarks"
for model in ${BENCH_MODELS}; do
    julia --project=. -t "${JULIA_NUM_THREADS}" "./${model}_benchmarks.jl" || \
        echo "WARNING: ${model}_benchmarks.jl did not complete; its Python script may be stale or missing."
done
echo "$(date) - Finished Julia benchmarks"

## --- Python (NEMtropy / NuMeTriS) benchmarks ---------------------------------
if [ "${SKIP_PYTHON:-0}" != "1" ]; then
    echo "$(date) - Starting Python benchmarks"
    # Run with bash, not sh: the generated wrappers are #!/bin/bash and use `source` to activate
    # the venv. On macOS /bin/sh is bash in POSIX mode so `sh` happens to work, but on Debian and
    # Ubuntu /bin/sh is dash, where `source` does not exist and every wrapper would fail.
    for model in ${BENCH_MODELS}; do
        bash "./${model}_script.sh" || \
            echo "WARNING: ${model}_script.sh did not complete (see benchmarks/timeouts.log for jobs killed by BENCH_JOB_TIMEOUT)."
    done
    echo "$(date) - Finished Python benchmarks"
fi

## --- Accuracy comparison (parameter / constraint agreement) -----------------
echo "$(date) - Running accuracy comparison"
julia --project=. -t "${JULIA_NUM_THREADS}" ./accuracy_comparison.jl || \
    echo "WARNING: accuracy comparison did not complete (NEMtropy dumps may be missing)."

## --- Plots ------------------------------------------------------------------
## Each plot script is guarded the same way the accuracy step is: the scripts `error()` out
## when a model has no results for one of the two languages, and under `set -e` a single such
## failure would abort the run before the later scripts write their paper figures. A missing
## plot must not cost the ones that come after it.
if [ "${SKIP_PLOTS:-0}" != "1" ]; then
    echo "$(date) - Generating the plots"
    for model in ${BENCH_MODELS}; do
        julia --project=. "./${model}_plots.jl" || \
            echo "WARNING: ${model}_plots.jl did not complete (benchmark results for ${model} may be missing)."
    done
fi

echo "$(date) - Finished benchmarks"
