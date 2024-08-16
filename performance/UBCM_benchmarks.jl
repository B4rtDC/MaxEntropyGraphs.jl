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

@info """
$(now()) - Setting up UBCM benchmarks on $(Threads.nthreads()) threads.
"""

## Load up helper functions
include(joinpath(@__DIR__, "benchmark_helpers.jl"))

## Reference graphs setup
## ----------------------
name_graphs = [("UBCM_small",  smallgraph(:karate),                             Dict(:include_fixed_point => true, :include_BFGS => true, :include_LBFGS => false, :include_newton => true)), # 11 unique degrees (34, 78) graph
               ("UBCM_medium", Graphs.barabasi_albert(10000, 4, seed=161),      Dict(:include_fixed_point => true, :include_BFGS => true, :include_LBFGS => false, :include_newton => true)), # 102 unique degrees (10000, 39984) graph
               ("UBCM_large",  Graphs.barabasi_albert(250000, 30, seed=161),    Dict(:include_fixed_point => true, :include_BFGS => true, :include_LBFGS => false, :include_newton => false))] # 1051 unique degrees (250000, 7499100 graph)  # fixed seed for reproducibility

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
    println(f, "conda activate benchmarking")
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
                         ]
        )
    # write out results
    open(joinpath(outpath, "$(Dates.format(now(), "YYYY_mm_dd_HH_MM"))_$(name).json"),"w") do f
        write(f, JSON.json(results))
    end

    @info "$(now()) - Benchmarking for $(name) done."
end

@info "$(now()) - All Julia benchmarks done."