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
               ("UBCM_large",  Graphs.barabasi_albert(250000, 30, seed=161),    Dict(:include_fixed_point => true, :include_BFGS => true, :include_LBFGS => false, :include_newton => false))] # 1051 unique degrees (250000, 7499100 graph)  # fixed seed for reproducibility

# Scale limiter: BENCH_MAX_SCALE=small|medium|large (default large). BENCH_QUICK=1 aliases small.
let scale = lowercase(get(ENV, "BENCH_QUICK", "0") == "1" ? "small" : get(ENV, "BENCH_MAX_SCALE", "large"))
    ncap = scale == "small" ? 1 : scale == "medium" ? 2 : length(name_graphs)
    ncap = min(ncap, length(name_graphs))
    ncap < length(name_graphs) && @info "BENCH_MAX_SCALE=$(scale): restricting UBCM benchmarks to the first $(ncap) problem(s)."
    global name_graphs = name_graphs[1:ncap]
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
    for (name, G) in name_graphs
        println(f, readlines("$(name).py")[2][3:end])
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