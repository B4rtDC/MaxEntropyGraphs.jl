#######################################################################################
#  ____  ____  _  ____ __  __    _                     _                          _
# |  _ \| __ )(_)/ ___|  \/  |  | |__   ___ _ __   ___| |__  _ __ ___   __ _ _ __| | _____
# | | | |  _ \| | |   | |\/| |  | '_ \ / _ \ '_ \ / __| '_ \| '_ ` _ \ / _` | '__| |/ / __|
# | |_| | |_) | | |___| |  | |  | |_) |  __/ | | | (__| | | | | | | | | (_| | |  |   <\__ \
# |____/|____/|_|\____|_|  |_|  |_.__/ \___|_| |_|\___|_| |_|_| |_| |_|\__,_|_|  |_|\_\___/
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
using Random
import LinearAlgebra

LinearAlgebra.BLAS.set_num_threads(parse(Int, get(ENV, "BENCH_CORES", string(Threads.nthreads()))))

@info """
$(now()) - Setting up DBiCM benchmarks on $(Threads.nthreads()) threads (BLAS: $(LinearAlgebra.BLAS.get_num_threads())).
"""

include(joinpath(@__DIR__, "benchmark_helpers.jl"))
mkpath(outpath)

## Reference graphs setup
## ----------------------
## These are SYNTHETIC networks, generated here from a fixed seed. They are not empirical data and
## are not presented as such. No public directed bipartite benchmark network is shipped with the
## package, and the point of these instances is scale rather than realism.
##
## The two channels are given deliberately different densities (p⁻ = 0.6 p⁺): a model that quietly
## treated the two channels as one would look perfectly healthy on a symmetric instance.

"""
    synthetic_dibipartite(Nb, Nt, p⁺, p⁻; seed)

Directed bipartite graph on `1:Nb` (bottom) and `Nb+1:Nb+Nt` (top), with independent Bernoulli
channels. Isolated vertices are repaired on *total* degree, matching the DBiCM's own guard: a
vertex that only receives links is layer-assignable and perfectly legitimate.
"""
function synthetic_dibipartite(Nb, Nt, p⁺, p⁻; seed::Int=161)
    rng = Random.Xoshiro(seed)
    g = SimpleDiGraph(Nb + Nt)
    for b in 1:Nb, t in 1:Nt
        rand(rng) < p⁺ && add_edge!(g, b, Nb + t)
        rand(rng) < p⁻ && add_edge!(g, Nb + t, b)
    end
    for b in 1:Nb
        degree(g, b) == 0 && add_edge!(g, b, Nb + rand(rng, 1:Nt))
    end
    for t in 1:Nt
        degree(g, Nb + t) == 0 && add_edge!(g, rand(rng, 1:Nb), Nb + t)
    end
    g
end

"Drop the top-to-bottom channel, leaving a purely one-directional network."
function one_directional(g, Nb)
    h = SimpleDiGraph(nv(g))
    for e in edges(g)
        src(e) <= Nb && add_edge!(h, src(e), dst(e))
    end
    for b in 1:Nb
        # `outdegree` is exported by both Graphs and MaxEntropyGraphs (the latter for models)
        Graphs.outdegree(h, b) == 0 && add_edge!(h, b, Nb + 1)
    end
    h
end

const DBICM_LAYERS = [("DBiCM_small",   25,   15, 0.30, 0.18),
                      ("DBiCM_medium", 500,  250, 0.10, 0.06),
                      ("DBiCM_large",  850, 1250, 0.05, 0.03)]

name_graphs = [(name, synthetic_dibipartite(Nb, Nt, pp, pm),
                Dict(:include_fixed_point => true, :include_BFGS => true,
                     :include_LBFGS => false, :include_newton => name != "DBiCM_large"),
                Nb)
               for (name, Nb, Nt, pp, pm) in DBICM_LAYERS]

# Scale limiter, identical in behaviour to the other drivers.
let scale = lowercase(get(ENV, "BENCH_QUICK", "0") == "1" ? "small" : get(ENV, "BENCH_MAX_SCALE", "large")),
    minscale = lowercase(get(ENV, "BENCH_MIN_SCALE", "small"))

    ncap = scale == "small" ? 1 : scale == "medium" ? 2 : length(name_graphs)
    ncap = min(ncap, length(name_graphs))
    nfloor = minscale == "medium" ? 2 : minscale == "large" ? 3 : 1
    nfloor = min(nfloor, ncap)
    (nfloor > 1 || ncap < length(name_graphs)) && @info "BENCH_MIN_SCALE=$(minscale), BENCH_MAX_SCALE=$(scale): restricting DBiCM benchmarks to problem(s) $(nfloor):$(ncap)."
    global name_graphs = name_graphs[nfloor:ncap]
end

## Write the edge lists. Three files per problem: the directed graph itself (provenance), and one
## bipartite (bottom, top) edge list per channel, which is what the two-BiCM comparator consumes.
for (name, G, _, Nb) in name_graphs
    open(joinpath(@__DIR__, "data", "$(name).csv"), "w") do f
        for e in edges(G)
            write(f, "$(src(e)), $(dst(e))\n")
        end
    end
    open(joinpath(@__DIR__, "data", "$(name)_plus.csv"), "w") do f
        for e in edges(G)
            src(e) <= Nb && write(f, "$(src(e)), $(dst(e))\n")
        end
    end
    open(joinpath(@__DIR__, "data", "$(name)_minus.csv"), "w") do f
        for e in edges(G)
            # written bottom-first so NEMtropy sees an ordinary bipartite problem; the channel
            # direction is carried by which file the edge lands in, not by the tuple order.
            src(e) > Nb && write(f, "$(dst(e)), $(src(e))\n")
        end
    end
    m = DBiCM(G)
    @info """$(now()) - benchmark "$(name)" will test G($(nv(G)), $(ne(G))): """ *
          """⊥ $(m.status[:N⊥]) / ⊤ $(m.status[:N⊤]) vertices, unique degrees """ *
          """(⊥out $(length(m.d⊥ᵣ_out)), ⊤in $(length(m.d⊤ᵣ_in)), ⊥in $(length(m.d⊥ᵣ_in)), ⊤out $(length(m.d⊤ᵣ_out)))"""
end

## Generate the python scripts and the associated shell script
for (name, _, _, _) in name_graphs
    generate_DBiCM_python(name, n)
end
open(joinpath(@__DIR__, "DBiCM_script.sh"), "w") do f
    println(f, "#!/bin/bash")
    println(f, "source \"$(joinpath(@__DIR__, ".venv", "bin", "activate"))\"")
    watchdog = "\"$(joinpath(@__DIR__, "run_with_timeout.sh"))\" \"\${BENCH_JOB_TIMEOUT:-0}\" "
    for (name, _, _, _) in name_graphs
        println(f, watchdog * readlines("$(name).py")[2][3:end])
    end
end

@info "$(now()) - Reference graphs and python scripts written."

## Start benchmarking Julia
## ------------------------
for (name, G, kwargs, Nb) in name_graphs
    @info "$(now()) - Started benchmarking $(name)."

    bench_list = Any[test_create_DBiCM(G);
                     test_solve_DBiCM(G; kwargs...);
                     test_sample_DBiCM(G, 10)]
    # The empty-channel short circuit is only interesting once, at the smallest scale: it is a
    # structural property of the solve, not something that changes with size.
    if name == "DBiCM_small"
        push!(bench_list, test_solve_DBiCM_one_directional(one_directional(G, Nb)))
    end
    if get(ENV, "BENCH_SKIP_PROJECTION", "0") != "1"
        push!(bench_list, test_project_DBiCM(G))
    end
    results = Dict("system_info" => get_system_info(), "benchmarks" => bench_list)
    open(joinpath(outpath, "$(Dates.format(now(), "YYYY_mm_dd_HH_MM"))_$(name).json"), "w") do f
        write(f, JSON.json(results))
    end

    @info "$(now()) - Benchmarking for $(name) done."
end

@info "$(now()) - All Julia benchmarks done."
