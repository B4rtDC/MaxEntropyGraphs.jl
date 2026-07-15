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
import LinearAlgebra

# Cap BLAS to the core budget so the (single-threaded) creation/solve benchmarks are a fair
# same-core comparison against the thread-limited Python side (see benchmarks.sh).
LinearAlgebra.BLAS.set_num_threads(parse(Int, get(ENV, "BENCH_CORES", string(Threads.nthreads()))))

@info """
$(now()) - Setting up BiCM benchmarks on $(Threads.nthreads()) threads (BLAS: $(LinearAlgebra.BLAS.get_num_threads())).
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
# Provenance: BiCM_small is the deterministic `corporateclub()` demo network shipped with the
# package; BiCM_medium / BiCM_large are committed canonical edge lists in ./data/ (loaded above),
# so every run uses the exact same graphs.

# Scale limiter: BENCH_MAX_SCALE=small|medium|large (default large). BENCH_QUICK=1 aliases small.
let scale = lowercase(get(ENV, "BENCH_QUICK", "0") == "1" ? "small" : get(ENV, "BENCH_MAX_SCALE", "large"))
    ncap = scale == "small" ? 1 : scale == "medium" ? 2 : length(name_graphs)
    ncap = min(ncap, length(name_graphs))
    ncap < length(name_graphs) && @info "BENCH_MAX_SCALE=$(scale): restricting BiCM benchmarks to the first $(ncap) problem(s)."
    global name_graphs = name_graphs[1:ncap]
end

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
    bench_list = Any[test_create_BiCM(G);
                     test_solve_BiCM(G; kwargs...);
                     test_sample_BiCM(G, 10)]
    # The projection benchmark is the slowest part; BENCH_SKIP_PROJECTION=1 skips it
    # (useful for quick core-sweep runs that only need creation/solve/sampling).
    if get(ENV, "BENCH_SKIP_PROJECTION", "0") != "1"
        push!(bench_list, test_project_BiCM(G))
    end
    results = Dict(
        "system_info" => get_system_info(),
        "benchmarks" => bench_list
        )
    # write out results
    open(joinpath(outpath, "$(Dates.format(now(), "YYYY_mm_dd_HH_MM"))_$(name).json"),"w") do f
        write(f, JSON.json(results))
    end

    @info "$(now()) - Benchmarking for $(name) done."
end

@info "$(now()) - All Julia benchmarks done."