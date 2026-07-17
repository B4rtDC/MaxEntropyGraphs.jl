##############################################################################################
#   _   _ ____   ____ __  __     _                     _                          _        
#  | | | | __ ) / ___|  \/  |   | |__   ___ _ __   ___| |__  _ __ ___   __ _ _ __| | _____ 
#  | | | |  _ \| |   | |\/| |   | '_ \ / _ \ '_ \ / __| '_ \| '_ ` _ \ / _` | '__| |/ / __|
#  | |_| | |_) | |___| |  | |   | |_) |  __/ | | | (__| | | | | | | | | (_| | |  |   <\__ \
#   \___/|____/ \____|_|  |_|   |_.__/ \___|_| |_|\___|_| |_|_| |_| |_|\__,_|_|  |_|\_\___/
#                                                                                        
##############################################################################################

## Setup
cd(@__DIR__)
using Pkg
Pkg.activate(pwd())
using MaxEntropyGraphs
using Graphs
using BenchmarkTools
using JSON
using Dates
import LinearAlgebra

# Cap BLAS to the core budget so the (single-threaded) creation/solve benchmarks are a fair
# same-core comparison against the thread-limited Python side (see benchmarks.sh).
LinearAlgebra.BLAS.set_num_threads(parse(Int, get(ENV, "BENCH_CORES", string(Threads.nthreads()))))

@info """
$(now()) - Setting up UBCM benchmarks on $(Threads.nthreads()) threads (BLAS: $(LinearAlgebra.BLAS.get_num_threads())).
"""

## Load up helper functions
include(joinpath(@__DIR__, "benchmark_helpers.jl"))

## Reference graphs setup
## ----------------------
name_graphs = [("UBCM_small",  smallgraph(:karate),                             Dict(:include_fixed_point => true, :include_BFGS => true, :include_LBFGS => false, :include_newton => true)), # 11 unique degrees (34, 78) graph
               ("UBCM_medium", Graphs.barabasi_albert(10000, 4, seed=161),      Dict(:include_fixed_point => true, :include_BFGS => true, :include_LBFGS => false, :include_newton => true)), # 102 unique degrees (10000, 39984) graph
               ("UBCM_large",  Graphs.barabasi_albert(250000, 30, seed=161),    Dict(:include_fixed_point => true, :include_BFGS => false, :include_LBFGS => false, :include_newton => false))] # 1044 unique degrees (250000, 7499100 graph)  # fixed seed for reproducibility
# UBCM_large runs the fixed point only: at this scale NEITHER side's quasi-newton converges within
# the matched 1000-iteration budget (measured: Julia BFGS+BackTracking reaches residual ~1.8 from a
# chung_lu start and its default line search stalls into a false "Success" with residual ~4e4, while
# NEMtropy's quasinewton did not finish inside a 2h wall-clock budget). Benchmarking a solver that
# does not converge would publish a meaningless time, so the row is dropped rather than tuned.

# Scale limiter: BENCH_MAX_SCALE=small|medium|large (default large) caps the problem size, and
# BENCH_MIN_SCALE (default small) skips the smaller problems, so a run can target only what is
# missing (e.g. BENCH_MIN_SCALE=large BENCH_MAX_SCALE=large). BENCH_QUICK=1 aliases small.
let scale = lowercase(get(ENV, "BENCH_QUICK", "0") == "1" ? "small" : get(ENV, "BENCH_MAX_SCALE", "large")),
    minscale = lowercase(get(ENV, "BENCH_MIN_SCALE", "small"))

    ncap = scale == "small" ? 1 : scale == "medium" ? 2 : length(name_graphs)
    ncap = min(ncap, length(name_graphs))
    nfloor = minscale == "medium" ? 2 : minscale == "large" ? 3 : 1
    nfloor = min(nfloor, ncap) # a floor above the cap degrades to the cap, never to an empty run
    (nfloor > 1 || ncap < length(name_graphs)) && @info "BENCH_MIN_SCALE=$(minscale), BENCH_MAX_SCALE=$(scale): restricting UBCM benchmarks to problem(s) $(nfloor):$(ncap)."
    global name_graphs = name_graphs[nfloor:ncap]
end

# Write out the edgelists for the reference graphs (to use the same in python)
for (name, G, _) in name_graphs
    open(joinpath(@__DIR__, "data", "$(name).csv"), "w") do f
        for e in edges(G)
            write(f, "$(e.src), $(e.dst)\n")
        end
    end
    @info """$(now()) - benchmark "$(name)" will test G($(nv(G)), $(ne(G))) with $(length(unique(degree(G)))) unique degrees"""
end


## Generate python scripts and the associated shell script
for (name,G, _) in name_graphs
    generate_UBCM_python(name, n)
end
open(joinpath(@__DIR__, "UBCM_script.sh"), "w") do f
    println(f, "#!/bin/bash")
    println(f, "source \"$(joinpath(@__DIR__, ".venv", "bin", "activate"))\"")
    # Every pytest job runs under the process-group watchdog; BENCH_JOB_TIMEOUT=0 (the
    # default) runs it untouched, so this changes nothing unless a budget is set.
    watchdog = "\"$(joinpath(@__DIR__, "run_with_timeout.sh"))\" \"\${BENCH_JOB_TIMEOUT:-0}\" "
    for (name, G) in name_graphs
        cmd = readlines("$(name).py")[2][3:end]
        if name == "UBCM_large"
            # On the 250k-node graph NEMtropy's quasinewton did not finish inside a 2h
            # wall-clock budget and took the whole invocation's results with it (pytest-benchmark
            # saves at exit), so each test runs as its own pytest process, cheapest and most
            # valuable first: a slow quasinewton/newton can then be killed without losing
            # create/fixed-point. The parts are merged into one result file afterwards (the
            # plotting scripts read only the newest file per scale).
            for nodeid in ("UBCM_large.py::test_create_UBCM",
                           "UBCM_large.py::test_solve_UBCM[cm_exp-fixed-point-degrees]",
                           "UBCM_large.py::test_solve_UBCM[cm_exp-quasinewton-degrees]",
                           "UBCM_large.py::test_solve_UBCM[cm_exp-newton-degrees]")
                println(f, watchdog * replace(cmd, "pytest UBCM_large.py" => "pytest '$(nodeid)'"))
            end
            println(f, "python \"$(joinpath(@__DIR__, "merge_pytest_benchmarks.py"))\" \"$(joinpath(@__DIR__, "benchmarks"))\" UBCM_large")
        else
            println(f, watchdog * cmd)
        end
    end
end

@info "$(now()) - Reference graphs and python scripts written."


## Start benchmarking Julia
## ------------------------
for (name, G, kwargs) in name_graphs
    @info "$(now()) - Started benchmarking $(name)."

    # run benchmark
    results = Dict(
        "system_info" => get_system_info(),
        "benchmarks" => [test_create_UBCM(G);
                         test_solve_UBCM(G; kwargs...);
                         test_sample_UBCM(G, 10);
                         ]
        )
    # write out results
    open(joinpath(outpath, "$(Dates.format(now(), "YYYY_mm_dd_HH_MM"))_$(name).json"),"w") do f
        write(f, JSON.json(results))
    end

    @info "$(now()) - Benchmarking for $(name) done."
end

@info "$(now()) - All Julia benchmarks done."