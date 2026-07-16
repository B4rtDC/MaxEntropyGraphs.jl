##############################################################################################
#   ____  ____   ____ __  __     _                     _                          _
#  |  _ \| __ ) / ___|  \/  |   | |__   ___ _ __   ___| |__  _ __ ___   __ _ _ __| | _____
#  | | | |  _ \| |   | |\/| |   | '_ \ / _ \ '_ \ / __| '_ \| '_ ` _ \ / _` | '__| |/ / __|
#  | |_| | |_) | |___| |  | |   | |_) |  __/ | | | (__| | | | | | | | | (_| | |  |   <\__ \
#  |____/|____/ \____|_|  |_|   |_.__/ \___|_| |_|\___|_| |_|_| |_| |_|\__,_|_|  |_|\_\___/
#
#  Directed configuration model (DBCM): construction, solving, and the analytical directed 3-node
#  motif spectrum. The motif benchmark is the like-for-like counterpart of NEMtropy's
#  `expected_dcm_3motif_*` (both compute the ensemble-mean motif counts analytically); see
#  accuracy_comparison.jl for the cross-package correctness check.
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

# Cap BLAS to the core budget so the (single-threaded) motif/creation/solve benchmarks are a fair
# same-core comparison against the thread-limited Python side (see benchmarks.sh).
LinearAlgebra.BLAS.set_num_threads(parse(Int, get(ENV, "BENCH_CORES", string(Threads.nthreads()))))

@info """
$(now()) - Setting up DBCM benchmarks on $(Threads.nthreads()) threads (BLAS: $(LinearAlgebra.BLAS.get_num_threads())).
"""

## Load up helper functions
include(joinpath(@__DIR__, "benchmark_helpers.jl"))

## Reference graphs setup
## ----------------------
## `maspalomas` is a real directed food web (the validated anchor); the synthetic directed
## Erdos-Renyi graphs probe scaling. The motif spectrum is O(N^3) on both sides, so the sizes are
## capped well below the UBCM/BiCM scales.
name_graphs = [("DBCM_small",  MaxEntropyGraphs.maspalomas(),                              Dict(:include_fixed_point => true, :include_BFGS => true, :include_newton => true)),
               ("DBCM_medium", Graphs.erdos_renyi(300,  0.05, is_directed=true, seed=161), Dict(:include_fixed_point => true, :include_BFGS => true, :include_newton => true)),
               ("DBCM_large",  Graphs.erdos_renyi(1000, 0.015, is_directed=true, seed=161), Dict(:include_fixed_point => true, :include_BFGS => true, :include_newton => false))]

# Scale limiter: BENCH_MAX_SCALE=small|medium|large (default large) caps the problem size, and
# BENCH_MIN_SCALE (default small) skips the smaller problems, so a run can target only what is
# missing (e.g. BENCH_MIN_SCALE=large BENCH_MAX_SCALE=large). BENCH_QUICK=1 aliases small.
let scale = lowercase(get(ENV, "BENCH_QUICK", "0") == "1" ? "small" : get(ENV, "BENCH_MAX_SCALE", "large")),
    minscale = lowercase(get(ENV, "BENCH_MIN_SCALE", "small"))

    ncap = scale == "small" ? 1 : scale == "medium" ? 2 : length(name_graphs)
    ncap = min(ncap, length(name_graphs))
    nfloor = minscale == "medium" ? 2 : minscale == "large" ? 3 : 1
    nfloor = min(nfloor, ncap) # a floor above the cap degrades to the cap, never to an empty run
    (nfloor > 1 || ncap < length(name_graphs)) && @info "BENCH_MIN_SCALE=$(minscale), BENCH_MAX_SCALE=$(scale): restricting DBCM benchmarks to problem(s) $(nfloor):$(ncap)."
    global name_graphs = name_graphs[nfloor:ncap]
end

# Write out the edgelists for the reference graphs (to use the same in python)
for (name, G, _) in name_graphs
    open(joinpath(@__DIR__, "data", "$(name).csv"), "w") do f
        for e in edges(G)
            write(f, "$(e.src), $(e.dst)\n")
        end
    end
    @info """$(now()) - benchmark "$(name)" will test G($(nv(G)), $(ne(G))) [directed]"""
end

## Generate python scripts and the associated shell script
for (name, G, _) in name_graphs
    generate_DBCM_python(name)
end
open(joinpath(@__DIR__, "DBCM_script.sh"), "w") do f
    println(f, "#!/bin/bash")
    println(f, "source \"$(joinpath(@__DIR__, ".venv", "bin", "activate"))\"")
    for (name, G, _) in name_graphs
        # Wrap every pytest job in the process-group watchdog; BENCH_JOB_TIMEOUT=0 (the
        # default) runs it untouched, so this changes nothing unless a budget is set.
        println(f, "\"$(joinpath(@__DIR__, "run_with_timeout.sh"))\" \"\${BENCH_JOB_TIMEOUT:-0}\" " * readlines("$(name).py")[2][3:end])
    end
end

@info "$(now()) - Reference graphs and python scripts written."


## Start benchmarking Julia
## ------------------------
for (name, G, kwargs) in name_graphs
    @info "$(now()) - Started benchmarking $(name)."

    results = Dict(
        "system_info" => get_system_info(),
        "benchmarks" => [test_create_DBCM(G);
                         test_solve_DBCM(G; kwargs...);
                         test_motifs_DBCM(G);
                         ]
        )
    open(joinpath(outpath, "$(Dates.format(now(), "YYYY_mm_dd_HH_MM"))_$(name).json"),"w") do f
        write(f, JSON.json(results))
    end

    @info "$(now()) - Benchmarking for $(name) done."
end

@info "$(now()) - All Julia DBCM benchmarks done."
