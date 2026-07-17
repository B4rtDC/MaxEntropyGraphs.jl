#!/bin/bash
#
# run_with_timeout.sh <seconds> <command> [args...]
#
# Run a command under a wall-clock budget, killing its WHOLE PROCESS GROUP if it
# exceeds it. A budget of 0 means "no limit" (the command is exec'd untouched), so
# wrapping a command with a default budget of 0 changes nothing.
#
# Why this exists: macOS ships neither `timeout` nor `setsid`, and the benchmark
# jobs this wraps (pytest running NEMtropy/NuMeTriS) spawn numba threads and
# multiprocessing children. Killing just the pytest PID leaves those orphaned and
# still burning CPU, so the kill MUST target the process group. `set -m` (job
# control) makes the background child the leader of a fresh process group, which
# `kill -- -PID` can then take down as a unit.
#
# On timeout: SIGTERM to the group, a grace period, then SIGKILL; the event is
# appended to benchmarks/timeouts.log; the exit status is 0 ON PURPOSE, so a
# killed benchmark does not abort the generated benchmark script it is part of
# (the next job should still run). pytest-benchmark only writes its JSON at exit,
# so a killed job simply produces no result file, which downstream plotting
# already treats as "not benchmarked".
#
# Written against bash 3.2 (macOS system bash): no `wait -n`, no coreutils.

budget="${1:?usage: run_with_timeout.sh <seconds> <command> [args...]}"
shift

case "${budget}" in
    ''|*[!0-9]*) echo "run_with_timeout.sh: budget '${budget}' is not a number" >&2; exit 2 ;;
esac

if [ "${budget}" -eq 0 ]; then
    exec "$@"
fi

logfile="$(cd "$(dirname "$0")" && pwd)/benchmarks/timeouts.log"
mkdir -p "$(dirname "${logfile}")"

# Job control gives the child its own process group (pgid == its pid).
set -m
"$@" &
pid=$!
set +m

poll=15
waited=0
while kill -0 "${pid}" 2>/dev/null; do
    if [ "${waited}" -ge "${budget}" ]; then
        echo "$(date) - TIMEOUT after ${budget}s: $*" | tee -a "${logfile}" >&2
        kill -TERM -- "-${pid}" 2>/dev/null
        # grace period, then make sure
        grace=0
        while kill -0 "${pid}" 2>/dev/null && [ "${grace}" -lt 30 ]; do
            sleep 5; grace=$((grace + 5))
        done
        kill -KILL -- "-${pid}" 2>/dev/null
        wait "${pid}" 2>/dev/null
        echo "WARNING: job killed by run_with_timeout.sh (budget ${budget}s); its results were NOT saved: $*" >&2
        exit 0
    fi
    sleep "${poll}"
    waited=$((waited + poll))
done

wait "${pid}"
exit $?
