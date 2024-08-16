##################################################
# General helper functions for benchmarking
##################################################

## Constants
const outpath =    joinpath(@__DIR__, "benchmarks", "Julia-$(VERSION)") # output path for benchmark results
const samplepath = joinpath(@__DIR__,"./samples/")                       # path to save sample networks  
const n = 100                                                           # sample size for network sampling

# check if the directories exist
isdir(joinpath(@__DIR__, "benchmarks")) || mkdir(joinpath(@__DIR__, "benchmarks"))
isdir(outpath) || mkdir(outpath)
isdir(samplepath) || mkdir(samplepath)

@info """
    benchmark data outpath: $(outpath)
    sample size: $(n)
    Sample networks will be saved to $(samplepath)
"""

"""
    get_system_info()

Function to get system information and return the information as a dictionary.
"""
function get_system_info()
    machine_info = Dict(
        "node" => gethostname(),
        "processor" => string(Base.Sys.ARCH),
        "machine" => string(Base.Sys.MACHINE),
        "julia_version" => string(VERSION),
        "system" => Base.Sys.isapple() ? "Darwin" : Base.Sys.islinux() ? "Linux" : Base.Sys.iswindows() ? "Windows" : "Unknown",
        "cpu" => Dict(
                        "arch" => string(Sys.ARCH),
                        "bits" => Base.Sys.WORD_SIZE,
                        "count" => Sys.CPU_THREADS
                    ))
    return machine_info
end

"""
    sample_networks(m, n)

helper function to replicate the python sampling approach (writing out edgelists to files)
"""
function sample_networks(m, n::Int, path::String=samplepath)
    # generate the sample
    S = rand(m, n)
    # writeout the sample
    for i in eachindex(S)
        open(joinpath(path, "sample_$(i).txt"), "w") do f
            for e in edges(S[i])
                write(f, "$(e.src) $(e.dst)\n")
            end
        end
    end
end

##################################################
# Helper functions for UBCM benchmarking
##################################################

"""
    test_create_UBCM(G)

Benchmark the creation of the UBCM model for the given graph `G`.
"""
function test_create_UBCM(G)
    # prepare the benchmark
    b = @benchmarkable UBCM($(G))
    tune!(b)
    # run the benchmark
    res = run(b)
    return Dict(
        "name" => "test_create_UBCM",
        "stats" => res
    )
end

"""
    test_solve_UBCM(G; include_fixed_point=true, include_BFGS=true, include_LBFGS=true, include_newton=true)

Benchmark the UBCM model for the given graph `G`. 
For large graphs, it might be useful to exclude the Newton method for performance reasons.
"""
function test_solve_UBCM(G; include_fixed_point=true, include_BFGS=true, include_LBFGS=true, include_newton=true)
    # create the UBCM
    model = UBCM(G)
    solve_model!(model)
    suite = BenchmarkGroup()
    ## prepare the benchmark
    # fixed point
    if include_fixed_point
        suite["test_solve_UBCM[cm_exp-FP]"] =               @benchmarkable solve_model!($(model), method=:fixedpoint, initial=:degrees)
    end
    # quasi-newton, BFGS with analytical gradient, AutoZygote, ForwardDiff and ReverseDiff
    if include_BFGS 
        suite["test_solve_UBCM[cm_exp-QN-BFGS-AG]"] =       @benchmarkable solve_model!($(model), method=:BFGS, initial=:degrees, analytical_gradient=true)
        #suite["test_solve_UBCM[cm_exp-QN-BFGS-ADZ]"] =      @benchmarkable solve_model!($(model), method=:BFGS, initial=:degrees, analytical_gradient=false, AD_method=:AutoZygote)
        suite["test_solve_UBCM[cm_exp-QN-BFGS-ADF]"] =      @benchmarkable solve_model!($(model), method=:BFGS, initial=:degrees, analytical_gradient=false, AD_method=:AutoForwardDiff)
        #suite["test_solve_UBCM[cm_exp-QN-BFGS-ADR]"] =      @benchmarkable solve_model!($(model), method=:BFGS, initial=:degrees, analytical_gradient=false, AD_method=:AutoReverseDiff)
    end
    # quasi-newton, L-BFGS with analytical gradient, AutoZygote, ForwardDiff and ReverseDiff
    if include_LBFGS
        suite["test_solve_UBCM[cm_exp-QN-LBFGS-AG]"] =      @benchmarkable solve_model!($(model), method=:LBFGS, initial=:degrees, analytical_gradient=true)
        #suite["test_solve_UBCM[cm_exp-QN-LBFGS-ADZ]"] =     @benchmarkable solve_model!($(model), method=:LBFGS, initial=:degrees, analytical_gradient=false, AD_method=:AutoZygote)
        suite["test_solve_UBCM[cm_exp-QN-LBFGS-ADF]"] =     @benchmarkable solve_model!($(model), method=:LBFGS, initial=:degrees, analytical_gradient=false, AD_method=:AutoForwardDiff)
        #suite["test_solve_UBCM[cm_exp-QN-LBFGS-ADR]"] =     @benchmarkable solve_model!($(model), method=:LBFGS, initial=:degrees, analytical_gradient=false, AD_method=:AutoReverseDiff)
    end
    # newton, with analytical gradient, AutoZygote, ForwardDiff and ReverseDiff
    if include_newton
        #suite["test_solve_UBCM[cm_exp-Newton-AG]"] =        @benchmarkable solve_model!($(model), method=:Newton, initial=:degrees, analytical_gradient=true)
        #suite["test_solve_UBCM[cm_exp-Newton-ADZ]"] =       @benchmarkable solve_model!($(model), method=:Newton, initial=:degrees, analytical_gradient=false, AD_method=:AutoZygote)
        suite["test_solve_UBCM[cm_exp-Newton-ADF]"] =       @benchmarkable solve_model!($(model), method=:Newton, initial=:degrees, analytical_gradient=false, AD_method=:AutoForwardDiff)
        #suite["test_solve_UBCM[cm_exp-Newton-ADR]"] =       @benchmarkable solve_model!($(model), method=:Newton, initial=:degrees, analytical_gradient=false, AD_method=:AutoReverseDiff)
    end

    tune!(suite)
    # run the benchmark
    res = run(suite)
    return Dict(
        "name" => "test_solve_UBCM",
        "stats" => res
    )
end

"""
    test_sample_UBCM(G, n::Int)

Benchmark the sampling of networks from the UBCM model for the given graph `G`.
"""
function test_sample_UBCM(G, n::Int)
    # create the UBCM
    model = UBCM(G)
    solve_model!(model)
    # prepare the benchmark
    b = @benchmarkable sample_networks($(model), n)
    tune!(b)
    # run the benchmark
    res = run(b)
    return Dict(
        "name" => "test_sample_UBCM",
        "stats" => res
    )
end

UBCM_python_template = """
# run in folder as:
# pytest {{scriptname}}.py --benchmark-save={{scriptname}} --benchmark-min-rounds=30 --benchmark-warmup-iterations=2 --benchmark-save-data --benchmark-storage='{{outfolder}}'

import numpy as np
import networkx as nx
from NEMtropy import UndirectedGraph
import pytest
import csv
import time

# set the path
NETPATH = '{{datafilename}}'

# loader function
def load_csv_file(filepath):
    with open(filepath, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            yield tuple(map(int, row))


## -------------- ##
## Objects to use ##
## -------------- ##
EDGE_LIST = [t for t in load_csv_file(NETPATH)]
M = UndirectedGraph(edgelist=EDGE_LIST)
M.solve_tool(model="cm_exp", method="fixed-point", initial_guess="degrees")
N_SAMPLE = {{n}}

## ---------------------- ##
## Functions to benchmark ##
## ---------------------- ##
def create_UBCM(edgelist):
    # Create the UBCM object
    return UndirectedGraph(edgelist=edgelist)

def solve_UBCM(m, model, method, initial_guess):
    m.solve_tool(model=model, method=method, initial_guess=initial_guess)


## ---------------------- ##
## Pytest benchmark tests ##
## ---------------------- ##
def test_create_UBCM(benchmark):
    benchmark(create_UBCM, EDGE_LIST)

# Parameterize the test to run with different sets of arguments
@pytest.mark.parametrize("model,method,initial_guess", [
    ("cm_exp", "newton",      "degrees"),
    ("cm_exp", "quasinewton", "degrees"),
    ("cm_exp", "fixed-point", "degrees")
])
def test_solve_UBCM(benchmark, model, method, initial_guess):
    benchmark(solve_UBCM, M, model, method, initial_guess)


# @pytest.mark.parametrize("cpu_n", [
#      (1),
#      (4),
#   ])
# def test_sample_UBCM(benchmark, cpu_n):
#      benchmark(M.ensemble_sampler, N_SAMPLE, cpu_n=cpu_n, output_dir="{{outfoldersamples}}/")

"""


"""
    generate_UBCM_python(name::String, n::Int)

Generate a python script for the UBCM model with the given name and number of samples `n`.
"""
function generate_UBCM_python(name::String, n::Int)
    network_data_path = joinpath(@__DIR__, "data", "$(name).csv")
    outfolder = joinpath(@__DIR__, "benchmarks")
    outfoldersamples = joinpath(@__DIR__, "samples")

    out = replace(UBCM_python_template, "{{scriptname}}" => name,
                                        "{{outfolder}}"  => outfolder,
                                        "{{datafilename}}"   => network_data_path,
                                        "{{n}}" => n,
                                        "{{outfoldersamples}}" => outfoldersamples)

    open(joinpath(@__DIR__, "$(name).py"), "w") do f
        write(f, out)
    end    

    @info "Python script for $(name) generated."

    return
end


"""
    test_create_BiCM(G) 

Benchmark the creation of the BiCM model for the given graph `G`.
"""
function test_create_BiCM(G) 
    # prepare the benchmark
    b = @benchmarkable BiCM($(G))
    tune!(b)
    # run the benchmark
    res = run(b)
    return Dict(
        "name" => "test_create_BiCM",
        "stats" => res
    )
end

"""
    test_solve_BiCM(G; include_fixed_point=true, include_BFGS=true, include_LBFGS=true, include_newton=true)

Benchmark the UBCM model for the given graph `G`. 
For large graphs, it might be useful to exclude the Newton method for performance reasons.
"""
function test_solve_BiCM(G; include_fixed_point=true, include_BFGS=true, include_LBFGS=true, include_newton=true)
    # create the BiCM
    model = BiCM(G)
    solve_model!(model)
    suite = BenchmarkGroup()
    ## prepare the benchmark
    # fixed point
    if include_fixed_point
        suite["test_solve_BiCM[cm_exp-FP]"] =               @benchmarkable solve_model!($(model), method=:fixedpoint, initial=:degrees)
    end
    # quasi-newton, BFGS with analytical gradient, AutoZygote, ForwardDiff and ReverseDiff
    if include_BFGS
        suite["test_solve_BiCM[cm_exp-QN-BFGS-AG]"] =       @benchmarkable solve_model!($(model), method=:BFGS, initial=:degrees, analytical_gradient=true)
        #suite["test_solve_BiCM[cm_exp-QN-BFGS-ADZ]"] =      @benchmarkable solve_model!($(model), method=:BFGS, initial=:degrees, analytical_gradient=false, AD_method=:AutoZygote)
        suite["test_solve_BiCM[cm_exp-QN-BFGS-ADF]"] =      @benchmarkable solve_model!($(model), method=:BFGS, initial=:degrees, analytical_gradient=false, AD_method=:AutoForwardDiff)
        #suite["test_solve_BiCM[cm_exp-QN-BFGS-ADR]"] =      @benchmarkable solve_model!($(model), method=:BFGS, initial=:degrees, analytical_gradient=false, AD_method=:AutoReverseDiff)
    end
    # quasi-newton, L-BFGS with analytical gradient, AutoZygote, ForwardDiff and ReverseDiff
    if include_LBFGS
        suite["test_solve_BiCM[cm_exp-QN-LBFGS-AG]"] =      @benchmarkable solve_model!($(model), method=:LBFGS, initial=:degrees, analytical_gradient=true)
        #suite["test_solve_BiCM[cm_exp-QN-LBFGS-ADZ]"] =     @benchmarkable solve_model!($(model), method=:LBFGS, initial=:degrees, analytical_gradient=false, AD_method=:AutoZygote)
        suite["test_solve_BiCM[cm_exp-QN-LBFGS-ADF]"] =     @benchmarkable solve_model!($(model), method=:LBFGS, initial=:degrees, analytical_gradient=false, AD_method=:AutoForwardDiff)
        #suite["test_solve_BiCM[cm_exp-QN-LBFGS-ADR]"] =     @benchmarkable solve_model!($(model), method=:LBFGS, initial=:degrees, analytical_gradient=false, AD_method=:AutoReverseDiff)
    end
    if include_newton
        # newton, with analytical gradient, AutoZygote, ForwardDiff and ReverseDiff
        #suite["test_solve_BiCM[cm_exp-Newton-AG]"] =        @benchmarkable solve_model!($(model), method=:Newton, initial=:degrees, analytical_gradient=true)
        #suite["test_solve_BiCM[cm_exp-Newton-ADZ]"] =       @benchmarkable solve_model!($(model), method=:Newton, initial=:degrees, analytical_gradient=false, AD_method=:AutoZygote)
        suite["test_solve_BiCM[cm_exp-Newton-ADF]"] =       @benchmarkable solve_model!($(model), method=:Newton, initial=:degrees, analytical_gradient=false, AD_method=:AutoForwardDiff)
        #suite["test_solve_BiCM[cm_exp-Newton-ADR]"] =       @benchmarkable solve_model!($(model), method=:Newton, initial=:degrees, analytical_gradient=false, AD_method=:AutoReverseDiff)
    end
    tune!(suite)
    # run the benchmark
    res = run(suite)
    return Dict(
        "name" => "test_solve_BiCM",
        "stats" => res
    )

end


"""
    test_project_BiCM(G)

Benchmark the projection of the BiCM model for the given graph `G`.
This benchmark includes all layers, precomputed options, distributions and multithreading options.
"""
function test_project_BiCM(G)
    # make model
    m = BiCM(G)
    # solve model
    solve_model!(m)
    # store biadjacency matrix
    set_Ĝ!(m)

    # prepare the benchmarks
    suite = BenchmarkGroup()
    ## prepare the benchmark
    for layer in [:bottom, :top]
        for precomputed in [true, false]
            for distribution in [:Poisson, :PoissonBinomial]
                for multithreaded in [true, false]
                    suite["test_project_BiCM[cm_exp-$(layer)-$(precomputed)-$(distribution)-$(multithreaded)]"] = @benchmarkable project($(m); layer=$(layer), precomputed=$(precomputed), distribution=$(distribution), multithreaded=$(multithreaded))
                end
            end
        end
    end
    tune!(suite)
    # run the benchmark
    res = run(suite)
    return Dict(
        "name" => "test_project_BiCM",
        "stats" => res
    )
end

BiCM_python_template = """
# run in folder as:
# pytest {{scriptname}}.py --benchmark-save={{scriptname}} --benchmark-min-rounds=30 --benchmark-warmup-iterations=2 --benchmark-save-data --benchmark-storage='{{outfolder}}'

import numpy as np
import networkx as nx
from NEMtropy import BipartiteGraph
import pytest
import csv
import time

# set the path
NETPATH = '{{datafilename}}'

# loader function
def load_csv_file(filepath):
    with open(filepath, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            yield tuple(map(int, row))


## -------------- ##
## Objects to use ##
## -------------- ##
EDGE_LIST = [t for t in load_csv_file(NETPATH)]
M = BipartiteGraph(edgelist=EDGE_LIST)
M.solve_tool(method="fixed-point", initial_guess="degrees")
N_SAMPLE = {{n}}

## ---------------------- ##
## Functions to benchmark ##
## ---------------------- ##
def create_BiCM(edgelist):
    # Create the BiCM object
    return BipartiteGraph(edgelist=edgelist)

def solve_BiCM(m, method, initial_guess):
    m.solve_tool(method=method, initial_guess=initial_guess)

def project_BiCM(m, method, rows, threads_num):
    m.compute_projection(method=method, rows=rows, threads_num=threads_num)

## ---------------------- ##
## Pytest benchmark tests ##
## ---------------------- ##
def test_create_BiCM(benchmark):
    benchmark(create_BiCM, EDGE_LIST)

# Parameterize the test to run with different sets of arguments
@pytest.mark.parametrize("method,initial_guess", [
    ("newton",      "degrees"),
    ("quasinewton", "degrees"),
    ("fixed-point", "degrees"),
])
def test_solve_BiCM(benchmark, method, initial_guess):
    benchmark(solve_BiCM, M, method, initial_guess)


@pytest.mark.parametrize("method,rows,threads_num", [
    ("poibin", True, 1),
    ("poibin", True, 4),
    ("poibin", False, 1),
    ("poibin", False, 4),
    ("poisson", True, 1),
    ("poisson", True, 4),
    ("poisson", False, 1),
    ("poisson", False, 4),
])
def test_project_BiCM(benchmark, method, rows, threads_num):
    benchmark(project_BiCM, M, method, rows, threads_num)

"""

"""
    generate_BiCM_python(name::String, n::Int)

Generate a python script for the BiCM model with the given name and number of samples `n`.
"""
function generate_BiCM_python(name::String, n::Int)
    network_data_path = joinpath(@__DIR__, "data", "$(name).csv")
    outfolder = joinpath(@__DIR__, "benchmarks")
    outfoldersamples = joinpath(@__DIR__, "samples")

    out = replace(BiCM_python_template, "{{scriptname}}" => name,
                                        "{{outfolder}}"  => outfolder,
                                        "{{datafilename}}"   => network_data_path,
                                        "{{n}}" => n,
                                        "{{outfoldersamples}}" => outfoldersamples)

    open(joinpath(@__DIR__, "$(name).py"), "w") do f
        write(f, out)
    end    

    @info "Python script for $(name) generated."

    return
end
