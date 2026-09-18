#!/usr/bin/env bash
#
# EXP-016: the full v0.8.0 benchmark campaign.
#
# Runs, in order: a preflight that pins the provenance and proves the tree is green, the solver
# robustness sweep, and the full benchmark suite (which itself runs the Python comparators, the
# accuracy comparison and the plots). The order is deliberate: the cheap new work goes first, so
# that a failure overnight in the 24h+ benchmark stage does not cost the results that are quick to
# produce and have never been produced before.
#
# The whole thing takes one to two days. Launch it so it survives the SSH session going away:
#
#   ./campaign.sh --detach
#
# which is shorthand for
#
#   nohup caffeinate -dimsu ./campaign.sh > campaign_<date>.log 2>&1 &
#
# `nohup` covers the SIGHUP when the session ends, the background-and-disown detaches it from the
# launching shell, and `caffeinate -dimsu` holds the machine awake for the duration. That last part
# is not optional on a laptop: an earlier run on this box was killed mid-flight when the lid closed.
#
# Progress is readable from anywhere with a single `cat performance/campaign_status.json`, and the
# log carries a heartbeat line every five minutes so a hung stage is distinguishable from a slow
# one. Nothing about monitoring is required for the run to complete.
#
# Environment variables:
#   CAMPAIGN_FROM     first stage to run (default 0). Stages are 0 preflight, 1 robustness,
#                     2 benchmarks, 3 summary. Use this to resume after a failure.
#   CAMPAIGN_TO       last stage to run (default 3).
#   CAMPAIGN_SKIP_TESTS  1 to skip the test suite and validation scripts in the preflight.
#   ROB_NGRAPHS       graphs per corpus for the robustness sweep (default 150, the published size).
#   BENCH_*           passed straight through to benchmarks.sh.
#
# Defaults are pinned to EXP-015 so that every difference is attributable to the package rather
# than to the machine or the toolchain: same box, BENCH_CORES=12, Julia 1.12.6, same Python pins.

set -uo pipefail
cd "$(dirname "$0")"

REPO_ROOT="$(cd .. && pwd)"
STATUS="$(pwd)/campaign_status.json"
STAMP="$(date +%Y%m%d)"

## --- Self-detach ------------------------------------------------------------
if [ "${1:-}" = "--detach" ]; then
    LOG="$(pwd)/campaign_${STAMP}.log"
    echo "Detaching. Log: ${LOG}"
    echo "Status:        ${STATUS}"
    echo "Stop it with:  pkill -f campaign.sh"
    shift
    nohup caffeinate -dimsu "$0" "$@" >> "${LOG}" 2>&1 &
    disown
    exit 0
fi

export BENCH_CORES="${BENCH_CORES:-12}"
export BENCH_JOB_TIMEOUT="${BENCH_JOB_TIMEOUT:-7200}"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-$BENCH_CORES}"
export ROB_NGRAPHS="${ROB_NGRAPHS:-150}"
CAMPAIGN_FROM="${CAMPAIGN_FROM:-0}"
CAMPAIGN_TO="${CAMPAIGN_TO:-3}"

## --- Status file ------------------------------------------------------------
## One flat JSON, rewritten whole after every state change. A partial write is not worth guarding
## against here: the file is a progress report, not an input to anything.
## The current stage lives in a file, not only in a shell variable. The heartbeat runs in a
## subshell forked before the first stage begins, so a variable would leave it reporting (and
## writing) a stale stage forever, which is precisely the field anyone reading this file cares
## about.
STAGE_FILE="$(pwd)/.campaign_stage"
CURRENT_SINCE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "-" > "$STAGE_FILE"
current_stage() { cat "$STAGE_FILE" 2>/dev/null || echo "-"; }
STARTED="$CURRENT_SINCE"
declare -a DONE_STAGES=()

write_status() {
    local state="$1" detail="${2:-}"
    {
        printf '{\n'
        printf '  "campaign": "EXP-016",\n'
        printf '  "pid": %s,\n' "$$"
        printf '  "started": "%s",\n' "$STARTED"
        printf '  "updated": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf '  "stage": "%s",\n' "$(current_stage)"
        printf '  "stage_since": "%s",\n' "$CURRENT_SINCE"
        printf '  "state": "%s",\n' "$state"
        printf '  "detail": "%s",\n' "$detail"
        if [ "${#DONE_STAGES[@]}" -eq 0 ]; then
            printf '  "completed": []\n'
        else
            printf '  "completed": [%s]\n' "$(printf '"%s",' "${DONE_STAGES[@]}" | sed 's/,$//')"
        fi
        printf '}\n'
    } > "$STATUS"
}

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) | $*"; }

begin_stage() {
    echo "$1" > "$STAGE_FILE"
    CURRENT_SINCE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    write_status running
    log "=== BEGIN $(current_stage) ==="
}

end_stage() {
    local rc="$1" st; st="$(current_stage)"
    DONE_STAGES+=("${st}:$([ "$rc" -eq 0 ] && echo ok || echo "failed(${rc})")")
    write_status "$([ "$rc" -eq 0 ] && echo running || echo degraded)" "${st} exited ${rc}"
    log "=== END ${st} (exit ${rc}) ==="
}

## A stage that fails must not take the rest of the campaign with it: an overnight run that dies at
## hour 2 and reports nothing is the failure mode this guards against. Failures are loud in the log
## and recorded in the status file.
run_stage() {
    local name="$1"; shift
    begin_stage "$name"
    "$@"
    local rc=$?
    end_stage "$rc"
    return 0
}

want_stage() { [ "$1" -ge "$CAMPAIGN_FROM" ] && [ "$1" -le "$CAMPAIGN_TO" ]; }

## --- Heartbeat --------------------------------------------------------------
HEARTBEAT_PID=""
start_heartbeat() {
    ( while true; do sleep 300; log "heartbeat: still in $(current_stage)"; write_status running; done ) &
    HEARTBEAT_PID=$!
}
cleanup() {
    [ -n "$HEARTBEAT_PID" ] && kill "$HEARTBEAT_PID" 2>/dev/null
    write_status finished "campaign exited"
}
trap cleanup EXIT

## --- Stage 0: preflight -----------------------------------------------------
stage_preflight() {
    command -v julia >/dev/null || { log "ERROR: julia not on PATH"; return 1; }
    command -v uv    >/dev/null || log "WARNING: uv not on PATH; the Python side will fail"

    local sha; sha="$(git -C "$REPO_ROOT" rev-parse HEAD)"
    local dirty; dirty="$(git -C "$REPO_ROOT" status --porcelain | wc -l | tr -d ' ')"
    log "git ${sha} (${dirty} modified files)"
    log "julia $(julia --version)"

    # Resolve both environments from NO manifest. performance/Manifest.toml is gitignored, so a
    # stale local lockfile hides a compat pin that kills a fresh clone on its first command; that
    # has now happened twice, at 0.5 -> 0.6 and again at 0.7 -> 0.8.
    log "checking that the benchmark environment resolves from a clean slate"
    local tmp; tmp="$(mktemp -d)"
    cp Project.toml "$tmp/"
    ( cd "$tmp" && julia --project=. -e "using Pkg; Pkg.develop(path=\"$REPO_ROOT\"); Pkg.instantiate()" ) \
        || { log "ERROR: performance/Project.toml does not resolve against this package version"; return 1; }
    rm -rf "$tmp"

    julia --project=. -e 'using Pkg; Pkg.develop(path=".."); Pkg.instantiate()' || return 1

    # Provenance. Recorded before anything is measured, and including the resolved Graphs.jl
    # version: the seeded UBCM_medium/UBCM_large graphs are only reproducible for a fixed Graphs.jl,
    # whose compat bound allows any 1.x, so a change there means the EXP-015 comparison is between
    # different graphs rather than different code.
    julia --project=. -e '
        using Pkg, Dates, JSON, InteractiveUtils
        deps = Pkg.dependencies()
        ver(name) = begin
            for (_, p) in deps; p.name == name && return string(p.version); end
            "absent"
        end
        open("campaign_provenance.json", "w") do io
            JSON.print(io, Dict(
                "timestamp" => string(now()),
                "julia" => string(VERSION),
                "threads" => Threads.nthreads(),
                "bench_cores" => get(ENV, "BENCH_CORES", ""),
                "package" => ver("MaxEntropyGraphs"),
                "graphs_jl" => ver("Graphs"),
                "cpu" => string(Sys.cpu_info()[1].model),
                "machine" => string(Sys.MACHINE)), 2)
        end' || return 1
    log "provenance written: $(cat campaign_provenance.json | tr -d '\n ')"

    if [ "${CAMPAIGN_SKIP_TESTS:-0}" = "1" ]; then
        log "CAMPAIGN_SKIP_TESTS=1: skipping the test suite and validation scripts"
        return 0
    fi

    log "running the test suite at this SHA"
    ( cd "$REPO_ROOT" && julia --project=. -e 'using Pkg; Pkg.test()' ) || {
        log "ERROR: the test suite is not green; the campaign would measure an unknown tree"
        return 1
    }

    log "running the validation scripts"
    local failed=0
    for f in "$REPO_ROOT"/validation/symbolic/*.jl "$REPO_ROOT"/validation/numeric/*.jl; do
        case "$(basename "$f")" in common.jl) continue ;; esac
        if ( cd "$REPO_ROOT" && julia --project=validation "$f" > /dev/null 2>&1 ); then
            log "  PASS $(basename "$f")"
        else
            log "  FAIL $(basename "$f")"
            failed=$((failed + 1))
        fi
    done
    [ "$failed" -eq 0 ] || { log "ERROR: ${failed} validation script(s) failed"; return 1; }
    return 0
}

## --- Stage 1: solver robustness ---------------------------------------------
stage_robustness() {
    julia --project=. -t "$JULIA_NUM_THREADS" robustness/ladder.jl || return 1
    julia --project=. -t "$JULIA_NUM_THREADS" robustness/sweep.jl  || return 1
    julia --project=. robustness/report.jl > /dev/null || return 1
    log "robustness report: robustness/results/robustness_report.md"
    return 0
}

## --- Stage 2: the benchmark suite -------------------------------------------
## benchmarks.sh runs the Julia drivers, generates and runs the Python comparators, then the
## accuracy comparison and the plot scripts, so this one call covers the whole speed refresh.
stage_benchmarks() {
    ./benchmarks.sh
}

## --- Stage 3: summary -------------------------------------------------------
stage_summary() {
    julia --project=. campaign_summary.jl
}

## --- Run --------------------------------------------------------------------
log "EXP-016 campaign starting: BENCH_CORES=${BENCH_CORES}, ROB_NGRAPHS=${ROB_NGRAPHS}, stages ${CAMPAIGN_FROM}..${CAMPAIGN_TO}"
write_status starting
start_heartbeat

want_stage 0 && run_stage "0-preflight"  stage_preflight
want_stage 1 && run_stage "1-robustness" stage_robustness
want_stage 2 && run_stage "2-benchmarks" stage_benchmarks
want_stage 3 && run_stage "3-summary"    stage_summary

echo "done" > "$STAGE_FILE"
write_status finished "all requested stages attempted"
log "campaign finished. Stages: ${DONE_STAGES[*]+"${DONE_STAGES[*]}"}"
