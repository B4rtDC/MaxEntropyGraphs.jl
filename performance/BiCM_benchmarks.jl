#######################################################################################
#  ____  _  ____ __  __     _                     _                          _        
# | __ )(_)/ ___|  \/  |   | |__   ___ _ __   ___| |__  _ __ ___   __ _ _ __| | _____ 
# |  _ \| | |   | |\/| |   | '_ \ / _ \ '_ \ / __| '_ \| '_ ` _ \ / _` | '__| |/ / __|
# | |_) | | |___| |  | |   | |_) |  __/ | | | (__| | | | | | | | | (_| | |  |   <\__ \
# |____/|_|\____|_|  |_|   |_.__/ \___|_| |_|\___|_| |_|_| |_| |_|\__,_|_|  |_|\_\___/
#######################################################################################

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
$(now()) - Setting up BiCM benchmarks on $(Threads.nthreads()) threads.
"""

## Load up helper functions
include(joinpath(@__DIR__, "benchmark_helpers.jl"))

## Reference graphs setup
## ----------------------
name_graphs = [("BiCM_small",  corporateclub(),                             Dict(:include_fixed_point => true, :include_BFGS => true, :include_LBFGS => false, :include_newton => true)), # 25 + 15 vertices, 6 + 6 unique degrees
               ("BiCM_medium",  Graphs.SimpleGraphFromIterator(Graphs.SimpleEdge.(map( x-> Tuple(parse.(Int,x)), split.(readlines(joinpath(@__DIR__, "data", "BiCM_medium.csv")),",")))),     
                                Dict(:include_fixed_point => true, :include_BFGS => true, :include_LBFGS => false, :include_newton => true)), # 500 + 250 vertices, 48 + 57 unique degrees
               ("BiCM_large",   Graphs.SimpleGraphFromIterator(Graphs.SimpleEdge.(map( x-> Tuple(parse.(Int,x)), split.(readlines(joinpath(@__DIR__, "data", "BiCM_large.csv")),",")))),
                                Dict(:include_fixed_point => true, :include_BFGS => true, :include_LBFGS => false, :include_newton => false))
                                ] # 850 + 1250 vertices, 101 + 97 unique degrees

# Write out the edgelists for the reference graphs (to use the same in python)
for (name, G, _) in name_graphs
    open(joinpath(@__DIR__, "data", "$(name).csv"), "w") do f
        for e in edges(G)
            write(f, "$(e.src), $(e.dst)\n")
        end
    end
    membership = bipartite_map(G)
    N_top = length(unique(degree(G,findall(membership .== 1))))
    N_bot = length(unique(degree(G,findall(membership .== 2))))
    @info """$(now()) - benchmark "$(name)" will test G($(nv(G)), $(ne(G))) with ($(N_top), $(N_bot)) unique degrees"""
end


## Generate python scripts and the associated shell script
for (name,G, _) in name_graphs
    generate_BiCM_python(name, n)
end
open(joinpath(@__DIR__, "BiCM_script.sh"), "w") do f
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
        "benchmarks" => [test_create_BiCM(G);
                         test_solve_BiCM(G; kwargs...);
                         test_project_BiCM(G)
                         ]
        )
    # write out results
    open(joinpath(outpath, "$(Dates.format(now(), "YYYY_mm_dd_HH_MM"))_$(name).json"),"w") do f
        write(f, JSON.json(results))
    end

    @info "$(now()) - Benchmarking for $(name) done."
end

@info "$(now()) - All Julia benchmarks done."