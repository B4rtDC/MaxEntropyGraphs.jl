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
        # Record the core budget used: timings are only comparable across machines for the
        # same number of cores. Creation + parameter computation are single-threaded compute,
        # so BENCH_CORES caps BLAS (Julia) and OMP/NUMBA (Python); the sampling/projection paths
        # additionally scale with Julia threads (`julia -t N`) / NEMtropy `cpu_n`.
        "julia_num_threads" => Threads.nthreads(),
        "blas_num_threads" => LinearAlgebra.BLAS.get_num_threads(),
        "bench_cores" => parse(Int, get(ENV, "BENCH_CORES", string(Threads.nthreads()))),
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
        suite["test_solve_UBCM[cm_exp-FP]"] =               @benchmarkable solve_model!($(model), method=:fixedpoint, initial=:degrees, maxiters=1000, g_tol=1e-8, ftol=1e-8)
    end
    # quasi-newton, BFGS with analytical gradient, AutoZygote, ForwardDiff and ReverseDiff
    if include_BFGS 
        suite["test_solve_UBCM[cm_exp-QN-BFGS-AG]"] =       @benchmarkable solve_model!($(model), method=:BFGS, initial=:degrees, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=true)
        #suite["test_solve_UBCM[cm_exp-QN-BFGS-ADZ]"] =      @benchmarkable solve_model!($(model), method=:BFGS, initial=:degrees, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=false, AD_method=:AutoZygote)
        suite["test_solve_UBCM[cm_exp-QN-BFGS-ADF]"] =      @benchmarkable solve_model!($(model), method=:BFGS, initial=:degrees, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=false, AD_method=:AutoForwardDiff)
        #suite["test_solve_UBCM[cm_exp-QN-BFGS-ADR]"] =      @benchmarkable solve_model!($(model), method=:BFGS, initial=:degrees, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=false, AD_method=:AutoReverseDiff)
    end
    # quasi-newton, L-BFGS with analytical gradient, AutoZygote, ForwardDiff and ReverseDiff
    if include_LBFGS
        suite["test_solve_UBCM[cm_exp-QN-LBFGS-AG]"] =      @benchmarkable solve_model!($(model), method=:LBFGS, initial=:degrees, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=true)
        #suite["test_solve_UBCM[cm_exp-QN-LBFGS-ADZ]"] =     @benchmarkable solve_model!($(model), method=:LBFGS, initial=:degrees, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=false, AD_method=:AutoZygote)
        suite["test_solve_UBCM[cm_exp-QN-LBFGS-ADF]"] =     @benchmarkable solve_model!($(model), method=:LBFGS, initial=:degrees, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=false, AD_method=:AutoForwardDiff)
        #suite["test_solve_UBCM[cm_exp-QN-LBFGS-ADR]"] =     @benchmarkable solve_model!($(model), method=:LBFGS, initial=:degrees, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=false, AD_method=:AutoReverseDiff)
    end
    # newton, with analytical gradient, AutoZygote, ForwardDiff and ReverseDiff
    if include_newton
        #suite["test_solve_UBCM[cm_exp-Newton-AG]"] =        @benchmarkable solve_model!($(model), method=:Newton, initial=:degrees, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=true)
        #suite["test_solve_UBCM[cm_exp-Newton-ADZ]"] =       @benchmarkable solve_model!($(model), method=:Newton, initial=:degrees, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=false, AD_method=:AutoZygote)
        suite["test_solve_UBCM[cm_exp-Newton-ADF]"] =       @benchmarkable solve_model!($(model), method=:Newton, initial=:degrees, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=false, AD_method=:AutoForwardDiff)
        #suite["test_solve_UBCM[cm_exp-Newton-ADR]"] =       @benchmarkable solve_model!($(model), method=:Newton, initial=:degrees, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=false, AD_method=:AutoReverseDiff)
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

Benchmark drawing `n` samples from the UBCM ensemble for `G`, seeded for reproducibility.
This times the in-memory sampling (`rand(model, n; rng)`); NEMtropy's counterpart
(`ensemble_sampler`) additionally writes each sample to disk, so that difference is noted
when the two are compared.
"""
function test_sample_UBCM(G, n::Int)
    model = UBCM(G)
    solve_model!(model)
    b = @benchmarkable rand($(model), $(n); rng = MaxEntropyGraphs.Xoshiro(161))
    tune!(b)
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
# tol / max_steps matched to the Julia side (g_tol=1e-8 / maxiters=1000) for an apples-to-apples comparison.
M.solve_tool(model="cm_exp", method="fixed-point", initial_guess="degrees", tol=1e-8, eps=1e-8, max_steps=1000)
N_SAMPLE = {{n}}

## Dump the observed + expected degree sequences for the accuracy comparison
## (best effort: wrapped so it can never fail the benchmark). The Julia side
## (accuracy_comparison.jl) reads accuracy/{{scriptname}}_nemtropy.json if present.
import json, os
try:
    _acc = '{{accfolder}}'
    os.makedirs(_acc, exist_ok=True)
    _dump = {}
    for _key, _cands in (("dseq", ("dseq",)),
                         ("expected_dseq", ("expected_dseq", "expected_degree_seq"))):
        for _a in _cands:
            if hasattr(M, _a):
                _dump[_key] = [float(_v) for _v in getattr(M, _a)]
                break
    with open(os.path.join(_acc, '{{scriptname}}_nemtropy.json'), 'w') as _f:
        json.dump(_dump, _f)
except Exception:
    pass

## ---------------------- ##
## Functions to benchmark ##
## ---------------------- ##
def create_UBCM(edgelist):
    # Create the UBCM object
    return UndirectedGraph(edgelist=edgelist)

def solve_UBCM(m, model, method, initial_guess):
    # tol / max_steps matched to the Julia side (g_tol=1e-8 / maxiters=1000) for a fair comparison.
    m.solve_tool(model=model, method=method, initial_guess=initial_guess, tol=1e-8, eps=1e-8, max_steps=1000)


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

def sample_UBCM(m, n, cpu_n, outdir):
    # NEMtropy writes each sampled graph to disk as an edge list (seeded for reproducibility).
    m.ensemble_sampler(n, cpu_n=cpu_n, output_dir=outdir, seed=42)

{{sample_test}}

"""


"""
    generate_UBCM_python(name::String, n::Int)

Generate a python script for the UBCM model with the given name and number of samples `n`.
"""
function generate_UBCM_python(name::String, n::Int)
    network_data_path = joinpath(@__DIR__, "data", "$(name).csv")
    outfolder = joinpath(@__DIR__, "benchmarks")
    outfoldersamples = joinpath(@__DIR__, "samples")
    accfolder = joinpath(@__DIR__, "accuracy")
    cores = parse(Int, get(ENV, "BENCH_CORES", "4"))
    # NEMtropy's dense, file-based sampler is impractical beyond small graphs, so the sampling
    # benchmark is only emitted for the small problem (the Julia side samples at every scale).
    sample_test = endswith(name, "_small") ? """
@pytest.mark.benchmark(min_rounds=3, warmup=False)
def test_sample_UBCM(benchmark):
    benchmark(sample_UBCM, M, 10, $(cores), "$(joinpath(outfoldersamples, "nemtropy_ubcm"))/")
""" : "# sampling benchmark omitted for non-small graphs (NEMtropy's dense sampler does not scale)"

    out = replace(UBCM_python_template, "{{scriptname}}" => name,
                                        "{{outfolder}}"  => outfolder,
                                        "{{accfolder}}"  => accfolder,
                                        "{{datafilename}}"   => network_data_path,
                                        "{{n}}" => n,
                                        "{{sample_test}}" => sample_test,
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
        suite["test_solve_BiCM[cm_exp-FP]"] =               @benchmarkable solve_model!($(model), method=:fixedpoint, initial=:degrees, maxiters=1000, g_tol=1e-8, ftol=1e-8)
    end
    # quasi-newton, BFGS with analytical gradient, AutoZygote, ForwardDiff and ReverseDiff
    if include_BFGS
        suite["test_solve_BiCM[cm_exp-QN-BFGS-AG]"] =       @benchmarkable solve_model!($(model), method=:BFGS, initial=:degrees, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=true)
        #suite["test_solve_BiCM[cm_exp-QN-BFGS-ADZ]"] =      @benchmarkable solve_model!($(model), method=:BFGS, initial=:degrees, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=false, AD_method=:AutoZygote)
        suite["test_solve_BiCM[cm_exp-QN-BFGS-ADF]"] =      @benchmarkable solve_model!($(model), method=:BFGS, initial=:degrees, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=false, AD_method=:AutoForwardDiff)
        #suite["test_solve_BiCM[cm_exp-QN-BFGS-ADR]"] =      @benchmarkable solve_model!($(model), method=:BFGS, initial=:degrees, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=false, AD_method=:AutoReverseDiff)
    end
    # quasi-newton, L-BFGS with analytical gradient, AutoZygote, ForwardDiff and ReverseDiff
    if include_LBFGS
        suite["test_solve_BiCM[cm_exp-QN-LBFGS-AG]"] =      @benchmarkable solve_model!($(model), method=:LBFGS, initial=:degrees, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=true)
        #suite["test_solve_BiCM[cm_exp-QN-LBFGS-ADZ]"] =     @benchmarkable solve_model!($(model), method=:LBFGS, initial=:degrees, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=false, AD_method=:AutoZygote)
        suite["test_solve_BiCM[cm_exp-QN-LBFGS-ADF]"] =     @benchmarkable solve_model!($(model), method=:LBFGS, initial=:degrees, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=false, AD_method=:AutoForwardDiff)
        #suite["test_solve_BiCM[cm_exp-QN-LBFGS-ADR]"] =     @benchmarkable solve_model!($(model), method=:LBFGS, initial=:degrees, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=false, AD_method=:AutoReverseDiff)
    end
    if include_newton
        # newton, with analytical gradient, AutoZygote, ForwardDiff and ReverseDiff
        #suite["test_solve_BiCM[cm_exp-Newton-AG]"] =        @benchmarkable solve_model!($(model), method=:Newton, initial=:degrees, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=true)
        #suite["test_solve_BiCM[cm_exp-Newton-ADZ]"] =       @benchmarkable solve_model!($(model), method=:Newton, initial=:degrees, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=false, AD_method=:AutoZygote)
        suite["test_solve_BiCM[cm_exp-Newton-ADF]"] =       @benchmarkable solve_model!($(model), method=:Newton, initial=:degrees, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=false, AD_method=:AutoForwardDiff)
        #suite["test_solve_BiCM[cm_exp-Newton-ADR]"] =       @benchmarkable solve_model!($(model), method=:Newton, initial=:degrees, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=false, AD_method=:AutoReverseDiff)
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

"""
    test_sample_BiCM(G, n::Int)

Benchmark drawing `n` samples from the BiCM ensemble for `G`, seeded for reproducibility
(in-memory `rand(model, n; rng)`; see the note on `test_sample_UBCM`).
"""
function test_sample_BiCM(G, n::Int)
    model = BiCM(G)
    solve_model!(model)
    b = @benchmarkable rand($(model), $(n); rng = MaxEntropyGraphs.Xoshiro(161))
    tune!(b)
    res = run(b)
    return Dict(
        "name" => "test_sample_BiCM",
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
# tol / max_steps matched to the Julia side (g_tol=1e-8 / maxiters=1000) for an apples-to-apples comparison.
M.solve_tool(method="fixed-point", initial_guess="degrees", tol=1e-8, eps=1e-8, max_steps=1000)
N_SAMPLE = {{n}}

## Dump the observed + expected degree sequences (per layer) for the accuracy
## comparison (best effort: wrapped so it can never fail the benchmark). The Julia
## side reads accuracy/{{scriptname}}_nemtropy.json if present.
import json, os
try:
    _acc = '{{accfolder}}'
    os.makedirs(_acc, exist_ok=True)
    _dump = {}
    for _key, _cands in (("rows_deg", ("rows_deg",)),
                         ("cols_deg", ("cols_deg",))):
        for _a in _cands:
            if hasattr(M, _a):
                _dump[_key] = [float(_v) for _v in getattr(M, _a)]
                break
    # NEMtropy 3.0.3 exposes no expected-degree attribute at all: the expected biadjacency
    # matrix is the only route. Earlier revisions probed avg_rows_deg / expected_rows_deg /
    # expected_dseq_rows, none of which exist, so the dump silently lost both expected
    # sequences and the Julia side then skipped the BiCM comparison entirely.
    _avg = None
    if hasattr(M, "get_bicm_matrix"):
        _avg = M.get_bicm_matrix()
    elif getattr(M, "avg_mat", None) is not None:
        _avg = M.avg_mat
    if _avg is not None:
        _avg = np.asarray(_avg, dtype=float)
        _dump["expected_dseq_rows"] = [float(_v) for _v in _avg.sum(axis=1)]
        _dump["expected_dseq_cols"] = [float(_v) for _v in _avg.sum(axis=0)]
    # Also record the quasi-Newton solution. The reported bipartite accuracy comparison is scoped to
    # that solver, so the harness has to produce the very pairing that is quoted, not just the
    # fixed-point one. A separate object is used so the benchmarked M is left untouched.
    _Mq = BipartiteGraph(edgelist=EDGE_LIST)
    _Mq.solve_tool(method="quasinewton", initial_guess="degrees", tol=1e-8, eps=1e-8, max_steps=1000)
    _avgq = _Mq.get_bicm_matrix() if hasattr(_Mq, "get_bicm_matrix") else None
    if _avgq is not None:
        _avgq = np.asarray(_avgq, dtype=float)
        _dump["expected_dseq_rows_quasinewton"] = [float(_v) for _v in _avgq.sum(axis=1)]
        _dump["expected_dseq_cols_quasinewton"] = [float(_v) for _v in _avgq.sum(axis=0)]
    with open(os.path.join(_acc, '{{scriptname}}_nemtropy.json'), 'w') as _f:
        json.dump(_dump, _f)
except Exception as _e:
    # Never fail the benchmark over the accuracy dump, but never hide the miss either:
    # a bare `pass` here is precisely why the BiCM comparison went missing unnoticed.
    print("WARNING: could not dump NEMtropy BiCM accuracy data: " + repr(_e))

## ---------------------- ##
## Functions to benchmark ##
## ---------------------- ##
def create_BiCM(edgelist):
    # Create the BiCM object
    return BipartiteGraph(edgelist=edgelist)

def solve_BiCM(m, method, initial_guess):
    # tol / max_steps matched to the Julia side (g_tol=1e-8 / maxiters=1000) for a fair comparison.
    m.solve_tool(method=method, initial_guess=initial_guess, tol=1e-8, eps=1e-8, max_steps=1000)

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


@pytest.mark.skipif(os.environ.get("BENCH_SKIP_PROJECTION") == "1", reason="BENCH_SKIP_PROJECTION=1")
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
    accfolder = joinpath(@__DIR__, "accuracy")

    out = replace(BiCM_python_template, "{{scriptname}}" => name,
                                        "{{outfolder}}"  => outfolder,
                                        "{{accfolder}}"  => accfolder,
                                        "{{datafilename}}"   => network_data_path,
                                        "{{n}}" => n,
                                        "{{outfoldersamples}}" => outfoldersamples)

    open(joinpath(@__DIR__, "$(name).py"), "w") do f
        write(f, out)
    end    

    @info "Python script for $(name) generated."

    return
end

##################################################
# Helper functions for DBCM benchmarking (directed 3-node motifs)
##################################################

"""
    test_create_DBCM(G)

Benchmark the creation of the DBCM model for the given directed graph `G`.
"""
function test_create_DBCM(G)
    b = @benchmarkable DBCM($(G))
    tune!(b)
    res = run(b)
    return Dict("name" => "test_create_DBCM", "stats" => res)
end

"""
    test_solve_DBCM(G; include_fixed_point=true, include_BFGS=true, include_LBFGS=false, include_newton=true)

Benchmark solving the DBCM for the given directed graph `G` (settings matched to the Python side).
"""
function test_solve_DBCM(G; include_fixed_point=true, include_BFGS=true, include_LBFGS=false, include_newton=true)
    model = DBCM(G)
    solve_model!(model)
    suite = BenchmarkGroup()
    if include_fixed_point
        suite["test_solve_DBCM[dcm_exp-FP]"] =         @benchmarkable solve_model!($(model), method=:fixedpoint, initial=:degrees, maxiters=1000, g_tol=1e-8, ftol=1e-8)
    end
    if include_BFGS
        suite["test_solve_DBCM[dcm_exp-QN-BFGS-AG]"] = @benchmarkable solve_model!($(model), method=:BFGS, initial=:degrees, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=true)
        suite["test_solve_DBCM[dcm_exp-QN-BFGS-ADF]"] = @benchmarkable solve_model!($(model), method=:BFGS, initial=:degrees, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=false, AD_method=:AutoForwardDiff)
    end
    if include_LBFGS
        suite["test_solve_DBCM[dcm_exp-QN-LBFGS-AG]"] = @benchmarkable solve_model!($(model), method=:LBFGS, initial=:degrees, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=true)
    end
    if include_newton
        suite["test_solve_DBCM[dcm_exp-Newton-ADF]"] = @benchmarkable solve_model!($(model), method=:Newton, initial=:degrees, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=false, AD_method=:AutoForwardDiff)
    end
    tune!(suite)
    res = run(suite)
    return Dict("name" => "test_solve_DBCM", "stats" => res)
end

"""
    test_motifs_DBCM(G)

Benchmark the analytical directed 3-node motif spectrum `motifs(m)` (expected counts M1..M13) for the DBCM
fitted to `G`. The model is solved and its expected adjacency matrix cached (`set_Ĝ!`) first, so this times the
motif kernel itself. This is the like-for-like counterpart of NEMtropy's `expected_dcm_3motif_*`, which
computes the same ensemble means analytically from the fitted parameters (`sol = concatenate((x, y))`).
Note: MEG amortises the one-time O(N²) `Ĝ` build across all metrics, whereas NEMtropy recomputes the
edge probabilities inside each motif loop.
"""
function test_motifs_DBCM(G)
    model = DBCM(G)
    solve_model!(model)
    set_Ĝ!(model)
    b = @benchmarkable motifs($(model))
    tune!(b)
    res = run(b)
    return Dict("name" => "test_motifs_DBCM", "stats" => res)
end

DBCM_python_template = """
# run in folder as:
# pytest {{scriptname}}.py --benchmark-save={{scriptname}} --benchmark-min-rounds=30 --benchmark-warmup-iterations=2 --benchmark-save-data --benchmark-storage='{{outfolder}}'

import numpy as np
from NEMtropy import DirectedGraph
import NEMtropy.ensemble_functions as ef
import pytest
import csv

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
M = DirectedGraph(edgelist=EDGE_LIST)
# tol / max_steps matched to the Julia side (g_tol=1e-8 / maxiters=1000) for an apples-to-apples comparison.
M.solve_tool(model="dcm_exp", method="fixed-point", initial_guess="degrees", tol=1e-8, eps=1e-8, max_steps=1000)

# NEMtropy's expected-motif functions take the concatenated solution vector, x (out) first then y (in),
# matching DirectedGraph.zscore_3motifs (sol = np.concatenate((self.x, self.y))).
SOL = np.concatenate((M.x, M.y))
MOTIF_FUNCS = [getattr(ef, "expected_dcm_3motif_%d" % k) for k in range(1, 14)]

def expected_3motifs(sol):
    # analytical expected counts of the 13 directed 3-node motifs (ensemble means), M1..M13
    return [f(sol) for f in MOTIF_FUNCS]

## Dump the expected motif spectrum for the cross-package correctness check (best effort: wrapped so it can
## never fail the benchmark). The Julia side (accuracy_comparison.jl) reads accuracy/{{scriptname}}_nemtropy.json
## and compares it against motifs(model). Motif counts are permutation-invariant, so no node-order matching
## is needed.
import json, os
try:
    _acc = '{{accfolder}}'
    os.makedirs(_acc, exist_ok=True)
    with open(os.path.join(_acc, '{{scriptname}}_nemtropy.json'), 'w') as _f:
        json.dump({"expected_3motifs": [float(v) for v in expected_3motifs(SOL)]}, _f)
except Exception:
    pass

## ---------------------- ##
## Functions to benchmark ##
## ---------------------- ##
def create_DBCM(edgelist):
    # Create the DBCM object
    return DirectedGraph(edgelist=edgelist)

def solve_DBCM(m, model, method, initial_guess):
    # tol / max_steps matched to the Julia side (g_tol=1e-8 / maxiters=1000) for a fair comparison.
    m.solve_tool(model=model, method=method, initial_guess=initial_guess, tol=1e-8, eps=1e-8, max_steps=1000)


## ---------------------- ##
## Pytest benchmark tests ##
## ---------------------- ##
def test_create_DBCM(benchmark):
    benchmark(create_DBCM, EDGE_LIST)

# Parameterize the test to run with different sets of arguments
@pytest.mark.parametrize("model,method,initial_guess", [
    ("dcm_exp", "newton",      "degrees"),
    ("dcm_exp", "quasinewton", "degrees"),
    ("dcm_exp", "fixed-point", "degrees"),
])
def test_solve_DBCM(benchmark, model, method, initial_guess):
    benchmark(solve_DBCM, M, model, method, initial_guess)

# analytical expected 3-node motif spectrum (the like-for-like counterpart of MEG's motifs(m))
def test_motifs_DBCM(benchmark):
    benchmark(expected_3motifs, SOL)

"""


"""
    generate_DBCM_python(name::String)

Generate a python (pytest-benchmark) script for the DBCM model with the given `name`.
"""
function generate_DBCM_python(name::String)
    network_data_path = joinpath(@__DIR__, "data", "$(name).csv")
    outfolder = joinpath(@__DIR__, "benchmarks")
    accfolder = joinpath(@__DIR__, "accuracy")

    out = replace(DBCM_python_template, "{{scriptname}}" => name,
                                        "{{outfolder}}"  => outfolder,
                                        "{{accfolder}}"  => accfolder,
                                        "{{datafilename}}"   => network_data_path)

    open(joinpath(@__DIR__, "$(name).py"), "w") do f
        write(f, out)
    end

    @info "Python script for $(name) generated."

    return
end


##############################################################################################
#  UECM (Undirected Enhanced Configuration Model) helpers.
#
#  The UECM constrains the degree AND the (integer) strength sequence of a weighted, undirected
#  network. NEMtropy solves it through the same `UndirectedGraph` class as the UBCM, but with the
#  `ecm`/`ecm_exp` model name and a strengths-based initial guess. Because the likelihood is only
#  defined on the feasible region (yᵢyⱼ < 1), the fixed-point recipe is unstable, so the Julia side
#  benchmarks BFGS/Newton only (matching the paper).
##############################################################################################

"""
    test_create_UECM(G)

Benchmark the creation of the UECM model for the given weighted, undirected graph `G`.
"""
function test_create_UECM(G)
    b = @benchmarkable UECM($(G))
    tune!(b)
    res = run(b)
    return Dict("name" => "test_create_UECM", "stats" => res)
end

"""
    test_solve_UECM(G; include_fixed_point=false, include_BFGS=true, include_LBFGS=false, include_newton=true)

Benchmark solving the UECM for the given weighted, undirected graph `G` (settings matched to the Python side).
The fixed point recipe is unstable for the UECM, so it is excluded by default.
"""
function test_solve_UECM(G; include_fixed_point=false, include_BFGS=true, include_LBFGS=false, include_newton=true)
    model = UECM(G)
    solve_model!(model, method=:BFGS)
    suite = BenchmarkGroup()
    if include_fixed_point
        suite["test_solve_UECM[ecm_exp-FP]"] =         @benchmarkable solve_model!($(model), method=:fixedpoint, initial=:strengths, maxiters=1000, g_tol=1e-8, ftol=1e-8)
    end
    if include_BFGS
        suite["test_solve_UECM[ecm_exp-QN-BFGS-AG]"] = @benchmarkable solve_model!($(model), method=:BFGS, initial=:strengths, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=true)
        suite["test_solve_UECM[ecm_exp-QN-BFGS-ADF]"] = @benchmarkable solve_model!($(model), method=:BFGS, initial=:strengths, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=false, AD_method=:AutoForwardDiff)
    end
    if include_LBFGS
        suite["test_solve_UECM[ecm_exp-QN-LBFGS-AG]"] = @benchmarkable solve_model!($(model), method=:LBFGS, initial=:strengths, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=true)
    end
    if include_newton
        suite["test_solve_UECM[ecm_exp-Newton-ADF]"] = @benchmarkable solve_model!($(model), method=:Newton, initial=:strengths, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=false, AD_method=:AutoForwardDiff)
    end
    tune!(suite)
    res = run(suite)
    return Dict("name" => "test_solve_UECM", "stats" => res)
end

"""
    test_sample_UECM(G, n::Int)

Benchmark drawing `n` samples from the UECM ensemble for `G`, seeded for reproducibility.
"""
function test_sample_UECM(G, n::Int)
    model = UECM(G)
    solve_model!(model, method=:BFGS)
    b = @benchmarkable rand($(model), $(n); rng = MaxEntropyGraphs.Xoshiro(161))
    tune!(b)
    res = run(b)
    return Dict("name" => "test_sample_UECM", "stats" => res)
end

UECM_python_template = """
# run in folder as:
# pytest {{scriptname}}.py --benchmark-save={{scriptname}} --benchmark-min-rounds=30 --benchmark-warmup-iterations=2 --benchmark-save-data --benchmark-storage='{{outfolder}}'

import numpy as np
from NEMtropy import UndirectedGraph
import pytest
import csv

# set the path
NETPATH = '{{datafilename}}'

# loader function (weighted edge list: source, target, weight)
def load_csv_file(filepath):
    with open(filepath, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            yield tuple(map(int, row))


## -------------- ##
## Objects to use ##
## -------------- ##
# NEMtropy auto-detects the weighted (3-tuple) edge list and builds both the degree and strength sequence.
EDGE_LIST = [t for t in load_csv_file(NETPATH)]
M = UndirectedGraph(edgelist=EDGE_LIST)
# UECM ('ecm_exp'), strengths initial guess; tol / max_steps matched to the Julia side (g_tol=1e-8 / maxiters=1000).
M.solve_tool(model="ecm_exp", method="quasinewton", initial_guess="strengths", tol=1e-8, eps=1e-8, max_steps=1000)

## Dump the observed + expected degree AND strength sequences for the accuracy comparison
## (best effort: wrapped so it can never fail the benchmark). The Julia side (accuracy_comparison.jl)
## reads accuracy/{{scriptname}}_nemtropy.json if present.
import json, os
try:
    _acc = '{{accfolder}}'
    os.makedirs(_acc, exist_ok=True)
    _dump = {}
    for _key, _cands in (("dseq", ("dseq",)),
                         ("expected_dseq", ("expected_dseq", "expected_degree_seq")),
                         ("sseq", ("strength_sequence", "stseq")),
                         ("expected_sseq", ("expected_strength_seq", "expected_strength_sequence"))):
        for _a in _cands:
            if hasattr(M, _a):
                _dump[_key] = [float(_v) for _v in getattr(M, _a)]
                break
    with open(os.path.join(_acc, '{{scriptname}}_nemtropy.json'), 'w') as _f:
        json.dump(_dump, _f)
except Exception:
    pass

## ---------------------- ##
## Functions to benchmark ##
## ---------------------- ##
def create_UECM(edgelist):
    # Create the UECM object (weighted undirected graph)
    return UndirectedGraph(edgelist=edgelist)

def solve_UECM(m, model, method, initial_guess):
    # tol / max_steps matched to the Julia side (g_tol=1e-8 / maxiters=1000) for a fair comparison.
    m.solve_tool(model=model, method=method, initial_guess=initial_guess, tol=1e-8, eps=1e-8, max_steps=1000)


## ---------------------- ##
## Pytest benchmark tests ##
## ---------------------- ##
def test_create_UECM(benchmark):
    benchmark(create_UECM, EDGE_LIST)

# Parameterize the test to run with different sets of arguments
@pytest.mark.parametrize("model,method,initial_guess", [
    ("ecm_exp", "newton",      "strengths"),
    ("ecm_exp", "quasinewton", "strengths"),
])
def test_solve_UECM(benchmark, model, method, initial_guess):
    benchmark(solve_UECM, M, model, method, initial_guess)

"""


"""
    generate_UECM_python(name::String)

Generate a python (pytest-benchmark) script for the UECM model with the given `name`.
"""
function generate_UECM_python(name::String)
    network_data_path = joinpath(@__DIR__, "data", "$(name).csv")
    outfolder = joinpath(@__DIR__, "benchmarks")
    accfolder = joinpath(@__DIR__, "accuracy")

    out = replace(UECM_python_template, "{{scriptname}}" => name,
                                        "{{outfolder}}"  => outfolder,
                                        "{{accfolder}}"  => accfolder,
                                        "{{datafilename}}"   => network_data_path)

    open(joinpath(@__DIR__, "$(name).py"), "w") do f
        write(f, out)
    end

    @info "Python script for $(name) generated."

    return
end


##############################################################################################
#  CReM (Conditional Reconstruction Method) helpers.
#
#  The CReM is a two-step model for weighted, undirected networks with CONTINUOUS positive weights:
#  a binary (UBCM) layer supplies the marginal edge probabilities fᵢⱼ, conditional on which the
#  weights are exponential (rate θᵢ+θⱼ), constraining the strength sequence. NEMtropy solves it
#  through the same `UndirectedGraph` class with the `crema` model name; the binary layer is supplied
#  via `adjacency="cm_exp"`. Unlike the UECM, the CReM fixed-point recipe is stable, so it is included.
##############################################################################################

"""
    test_create_CReM(G)

Benchmark the creation of the CReM model for the given weighted, undirected graph `G`.
"""
function test_create_CReM(G)
    b = @benchmarkable CReM($(G))
    tune!(b)
    res = run(b)
    return Dict("name" => "test_create_CReM", "stats" => res)
end

"""
    test_solve_CReM(G; include_fixed_point=true, include_BFGS=true, include_LBFGS=false, include_newton=true)

Benchmark solving the CReM for the given weighted, undirected graph `G` (settings matched to the Python
side). The two-step solve includes the internal binary (UBCM) layer, as it does on the NEMtropy side.
"""
function test_solve_CReM(G; include_fixed_point=true, include_BFGS=true, include_LBFGS=false, include_newton=true)
    model = CReM(G)
    solve_model!(model)
    suite = BenchmarkGroup()
    if include_fixed_point
        suite["test_solve_CReM[crema-FP]"] =           @benchmarkable solve_model!($(model), method=:fixedpoint, initial=:strengths, maxiters=1000, ftol=1e-8)
    end
    if include_BFGS
        suite["test_solve_CReM[crema-QN-BFGS-AG]"] =   @benchmarkable solve_model!($(model), method=:BFGS, initial=:strengths, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=true)
        suite["test_solve_CReM[crema-QN-BFGS-ADF]"] =  @benchmarkable solve_model!($(model), method=:BFGS, initial=:strengths, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=false, AD_method=:AutoForwardDiff)
    end
    if include_LBFGS
        suite["test_solve_CReM[crema-QN-LBFGS-AG]"] =  @benchmarkable solve_model!($(model), method=:LBFGS, initial=:strengths, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=true)
    end
    if include_newton
        suite["test_solve_CReM[crema-Newton-ADF]"] =   @benchmarkable solve_model!($(model), method=:Newton, initial=:strengths, maxiters=1000, g_tol=1e-8, ftol=1e-8, analytical_gradient=false, AD_method=:AutoForwardDiff)
    end
    tune!(suite)
    res = run(suite)
    return Dict("name" => "test_solve_CReM", "stats" => res)
end

"""
    test_sample_CReM(G, n::Int)

Benchmark drawing `n` samples from the CReM ensemble for `G`, seeded for reproducibility.
"""
function test_sample_CReM(G, n::Int)
    model = CReM(G)
    solve_model!(model)
    b = @benchmarkable rand($(model), $(n); rng = MaxEntropyGraphs.Xoshiro(161))
    tune!(b)
    res = run(b)
    return Dict("name" => "test_sample_CReM", "stats" => res)
end

CReM_python_template = """
# run in folder as:
# pytest {{scriptname}}.py --benchmark-save={{scriptname}} --benchmark-min-rounds=30 --benchmark-warmup-iterations=2 --benchmark-save-data --benchmark-storage='{{outfolder}}'

import numpy as np
from NEMtropy import UndirectedGraph
import pytest
import csv

# set the path
NETPATH = '{{datafilename}}'

# loader function (weighted edge list: source, target, weight). The CReM weights are CONTINUOUS, so the
# node ids are read as int but the weight is read as float (do NOT map(int, ...) the whole row).
def load_csv_file(filepath):
    with open(filepath, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            yield (int(row[0]), int(row[1]), float(row[2]))


## -------------- ##
## Objects to use ##
## -------------- ##
# NEMtropy auto-detects the weighted (3-tuple) edge list and builds both the degree and strength sequence.
EDGE_LIST = [t for t in load_csv_file(NETPATH)]
M = UndirectedGraph(edgelist=EDGE_LIST)
# CReM ('crema'); the binary layer is a UBCM ('cm_exp'). tol / max_steps matched to the Julia side.
M.solve_tool(model="crema", method="quasinewton", initial_guess="strengths",
             adjacency="cm_exp", method_adjacency="newton", initial_guess_adjacency="random",
             tol=1e-8, eps=1e-8, max_steps=1000)

## Dump the observed + expected degree AND strength sequences for the accuracy comparison
## (best effort: wrapped so it can never fail the benchmark). The Julia side (accuracy_comparison.jl)
## reads accuracy/{{scriptname}}_nemtropy.json if present. NOTE: crema stores the expected strengths in
## the (misspelled) attribute `expected_stregth_seq`.
import json, os
try:
    _acc = '{{accfolder}}'
    os.makedirs(_acc, exist_ok=True)
    _dump = {}
    for _key, _cands in (("dseq", ("dseq",)),
                         ("expected_dseq", ("expected_dseq", "expected_degree_seq")),
                         ("sseq", ("strength_sequence", "stseq")),
                         ("expected_sseq", ("expected_stregth_seq", "expected_strength_seq", "expected_strength_sequence"))):
        for _a in _cands:
            if hasattr(M, _a):
                _dump[_key] = [float(_v) for _v in getattr(M, _a)]
                break
    with open(os.path.join(_acc, '{{scriptname}}_nemtropy.json'), 'w') as _f:
        json.dump(_dump, _f)
except Exception:
    pass

## ---------------------- ##
## Functions to benchmark ##
## ---------------------- ##
def create_CReM(edgelist):
    # Create the CReM object (weighted undirected graph)
    return UndirectedGraph(edgelist=edgelist)

def solve_CReM(m, model, method, initial_guess):
    # tol / max_steps matched to the Julia side (g_tol=1e-8 / maxiters=1000) for a fair comparison.
    m.solve_tool(model=model, method=method, initial_guess=initial_guess,
                 adjacency="cm_exp", method_adjacency="newton", initial_guess_adjacency="random",
                 tol=1e-8, eps=1e-8, max_steps=1000)


## ---------------------- ##
## Pytest benchmark tests ##
## ---------------------- ##
def test_create_CReM(benchmark):
    benchmark(create_CReM, EDGE_LIST)

# Parameterize the test to run with different sets of arguments
@pytest.mark.parametrize("model,method,initial_guess", [
    ("crema", "newton",       "strengths"),
    ("crema", "quasinewton",  "strengths"),
    ("crema", "fixed-point",  "strengths"),
])
def test_solve_CReM(benchmark, model, method, initial_guess):
    benchmark(solve_CReM, M, model, method, initial_guess)

"""


"""
    generate_CReM_python(name::String)

Generate a python (pytest-benchmark) script for the CReM model with the given `name`.
"""
function generate_CReM_python(name::String)
    network_data_path = joinpath(@__DIR__, "data", "$(name).csv")
    outfolder = joinpath(@__DIR__, "benchmarks")
    accfolder = joinpath(@__DIR__, "accuracy")

    out = replace(CReM_python_template, "{{scriptname}}" => name,
                                        "{{outfolder}}"  => outfolder,
                                        "{{accfolder}}"  => accfolder,
                                        "{{datafilename}}"   => network_data_path)

    open(joinpath(@__DIR__, "$(name).py"), "w") do f
        write(f, out)
    end

    @info "Python script for $(name) generated."

    return
end


##############################################################################################
#   ____           _                        _ _           _                        _      _
#  |  _ \ ___  ___(_)_ __  _ __ ___   ___ (_) |_ _   _   | |     _ __ ___   ___  __| | ___| |___
#  | |_) / _ \/ __| | '_ \| '__/ _ \ / __|| | __| | | |  |/ __| '_ ` _ \ / _ \ / _` |/ _ \ / __|
#  |  _ <  __/ (__| | |_) | | | (_) | (__ | | |_| |_| |     | | | | | | | (_) | (_| |  __/ \__ \
#  |_| \_\___|\___|_| .__/|_|  \___/ \___||_|\__|\__, |     |_| |_| |_| |_|\___/ \__,_|\___|_|___/
#                   |_|                          |___/
#
#  Reciprocity-aware models (RBCM / DCReM / CRWCM). The reference Python implementation is
#  NuMeTriS (Di Vece et al. 2023, https://github.com/MarsMDK/NuMeTriS), which solves the same
#  models through `Graph(adjacency=W).solver(model='RBCM'|'DBCM+CReMa'|'RBCM+CRWCM')`.
#  Empirically verified conventions (rhesus_macaques, see NOTES.md):
#    - NuMeTriS dseq_right/dseq_left/dseq_rec == our k→/k←/k↔ and stseq_* == our reciprocal strengths;
#    - the binary Lagrange multipliers match ours exactly (exp(-γ_NuMeTriS) == our zᵣ to ~1e-5);
#    - the weighted parameter blocks are ordered [β→; β←; β↔out; β↔in] (direct rates, like ours),
#      identified only up to a per-block gauge (θᵒ+c, θⁱ-c), so the accuracy comparison uses the
#      gauge-invariant implied expected sequences (computed on the Python side, dumped to JSON).
##############################################################################################

function test_create_RBCM(G)
    b = @benchmarkable RBCM($(G))
    tune!(b)
    res = run(b)
    return Dict("name" => "test_create_RBCM", "stats" => res)
end

"""
    test_solve_RBCM(G; include_fixed_point=true, include_BFGS=true, include_LBFGS=false, include_newton=true)

Benchmark solving the RBCM for the given directed graph `G` (settings matched to the Python side).
"""
function test_solve_RBCM(G; include_fixed_point=true, include_BFGS=true, include_LBFGS=false, include_newton=true)
    model = RBCM(G)
    solve_model!(model)
    suite = BenchmarkGroup()
    if include_fixed_point
        suite["test_solve_RBCM[FP]"] =         @benchmarkable solve_model!($(model), method=:fixedpoint, initial=:degrees, maxiters=1000, ftol=1e-8)
    end
    if include_BFGS
        suite["test_solve_RBCM[QN-BFGS-AG]"] = @benchmarkable solve_model!($(model), method=:BFGS, initial=:degrees, maxiters=1000, g_tol=1e-8, analytical_gradient=true)
        suite["test_solve_RBCM[QN-BFGS-ADF]"] = @benchmarkable solve_model!($(model), method=:BFGS, initial=:degrees, maxiters=1000, g_tol=1e-8, analytical_gradient=false, AD_method=:AutoForwardDiff)
    end
    if include_LBFGS
        suite["test_solve_RBCM[QN-LBFGS-AG]"] = @benchmarkable solve_model!($(model), method=:LBFGS, initial=:degrees, maxiters=1000, g_tol=1e-8, analytical_gradient=true)
    end
    if include_newton
        suite["test_solve_RBCM[Newton-ADF]"] = @benchmarkable solve_model!($(model), method=:Newton, initial=:degrees, maxiters=1000, g_tol=1e-8, analytical_gradient=false, AD_method=:AutoForwardDiff)
    end
    tune!(suite)
    res = run(suite)
    return Dict("name" => "test_solve_RBCM", "stats" => res)
end

"""
    test_motifs_RBCM(G)

Benchmark the exact expected 3-node motif spectrum under the RBCM (evaluated from the dyadic
probability matrices; the RBCM's headline use case).
"""
function test_motifs_RBCM(G)
    model = RBCM(G)
    solve_model!(model)
    b = @benchmarkable motifs($(model))
    tune!(b)
    res = run(b)
    return Dict("name" => "test_motifs_RBCM", "stats" => res)
end

"""
    test_sample_RBCM(G, n::Int)

Benchmark drawing `n` samples from the RBCM ensemble for `G`, seeded for reproducibility.
"""
function test_sample_RBCM(G, n::Int)
    model = RBCM(G)
    solve_model!(model)
    b = @benchmarkable rand($(model), $(n); rng = MaxEntropyGraphs.Xoshiro(161))
    tune!(b)
    res = run(b)
    return Dict("name" => "test_sample_RBCM", "stats" => res)
end

function test_create_DCReM(G)
    b = @benchmarkable DCReM($(G))
    tune!(b)
    res = run(b)
    return Dict("name" => "test_create_DCReM", "stats" => res)
end

"""
    test_solve_DCReM(G; include_fixed_point=true, include_BFGS=true, include_newton=true)

Benchmark solving the DCReM (two-step: internal DBCM + weighted layer, as on the NuMeTriS side).
"""
function test_solve_DCReM(G; include_fixed_point=true, include_BFGS=true, include_newton=true)
    model = DCReM(G)
    solve_model!(model)
    suite = BenchmarkGroup()
    if include_fixed_point
        suite["test_solve_DCReM[FP]"] =         @benchmarkable solve_model!($(model), method=:fixedpoint, initial=:strengths, maxiters=1000, ftol=1e-8)
    end
    if include_BFGS
        suite["test_solve_DCReM[QN-BFGS-AG]"] = @benchmarkable solve_model!($(model), method=:BFGS, initial=:strengths, maxiters=1000, g_tol=1e-8, analytical_gradient=true)
    end
    if include_newton
        suite["test_solve_DCReM[Newton-ADF]"] = @benchmarkable solve_model!($(model), method=:Newton, initial=:strengths, maxiters=1000, g_tol=1e-8, analytical_gradient=false, AD_method=:AutoForwardDiff)
    end
    tune!(suite)
    res = run(suite)
    return Dict("name" => "test_solve_DCReM", "stats" => res)
end

"""
    test_sample_DCReM(G, n::Int)

Benchmark drawing `n` samples from the DCReM ensemble for `G`, seeded for reproducibility.
"""
function test_sample_DCReM(G, n::Int)
    model = DCReM(G)
    solve_model!(model)
    b = @benchmarkable rand($(model), $(n); rng = MaxEntropyGraphs.Xoshiro(161))
    tune!(b)
    res = run(b)
    return Dict("name" => "test_sample_DCReM", "stats" => res)
end

function test_create_CRWCM(G)
    b = @benchmarkable CRWCM($(G))
    tune!(b)
    res = run(b)
    return Dict("name" => "test_create_CRWCM", "stats" => res)
end

"""
    test_solve_CRWCM(G; include_fixed_point=true, include_BFGS=true, include_newton=true)

Benchmark solving the CRWCM (two-step: internal RBCM + the 4N weighted layer, as on the NuMeTriS side).
"""
function test_solve_CRWCM(G; include_fixed_point=true, include_BFGS=true, include_newton=true)
    model = CRWCM(G)
    solve_model!(model)
    suite = BenchmarkGroup()
    if include_fixed_point
        suite["test_solve_CRWCM[FP]"] =         @benchmarkable solve_model!($(model), method=:fixedpoint, initial=:strengths, maxiters=1000, ftol=1e-8)
    end
    if include_BFGS
        suite["test_solve_CRWCM[QN-BFGS-AG]"] = @benchmarkable solve_model!($(model), method=:BFGS, initial=:strengths, maxiters=1000, g_tol=1e-8, analytical_gradient=true)
    end
    if include_newton
        suite["test_solve_CRWCM[Newton-ADF]"] = @benchmarkable solve_model!($(model), method=:Newton, initial=:strengths, maxiters=1000, g_tol=1e-8, analytical_gradient=false, AD_method=:AutoForwardDiff)
    end
    tune!(suite)
    res = run(suite)
    return Dict("name" => "test_solve_CRWCM", "stats" => res)
end

"""
    test_fluxes_CRWCM(G)

Benchmark the exact expected triadic flux spectrum under the CRWCM.
"""
function test_fluxes_CRWCM(G)
    model = CRWCM(G)
    solve_model!(model)
    b = @benchmarkable motif_fluxes($(model))
    tune!(b)
    res = run(b)
    return Dict("name" => "test_fluxes_CRWCM", "stats" => res)
end

"""
    test_sample_CRWCM(G, n::Int)

Benchmark drawing `n` samples from the CRWCM ensemble for `G`, seeded for reproducibility.
"""
function test_sample_CRWCM(G, n::Int)
    model = CRWCM(G)
    solve_model!(model)
    b = @benchmarkable rand($(model), $(n); rng = MaxEntropyGraphs.Xoshiro(161))
    tune!(b)
    res = run(b)
    return Dict("name" => "test_sample_CRWCM", "stats" => res)
end


# Generic NuMeTriS benchmark script. `{{model}}` is one of 'RBCM', 'DBCM+CReMa' or 'RBCM+CRWCM'.
# NuMeTriS takes a DENSE weighted adjacency matrix; the script rebuilds it from the dumped edge list.
# The accuracy dump computes NuMeTriS's own (gauge-invariant) constraint violations from its fitted
# parameters, using the empirically verified block orders (see the section header above).
NuMeTriS_python_template = """
# run in folder as:
# pytest {{scriptname}}.py --benchmark-save={{scriptname}} --benchmark-min-rounds=10 --benchmark-warmup-iterations=1 --benchmark-save-data --benchmark-storage='{{outfolder}}'

import numpy as np
import io, contextlib, csv, json, os
from NuMeTriS import Graph

NETPATH = '{{datafilename}}'
MODEL = '{{model}}'

# loader (weighted edge list: source, target, weight) -> dense weighted adjacency matrix
def load_dense(filepath):
    rows = []
    with open(filepath, 'r') as csvfile:
        for row in csv.reader(csvfile):
            rows.append((int(row[0]), int(row[1]), float(row[2])))
    n = max(max(r[0], r[1]) for r in rows)
    W = np.zeros((n, n))
    for s, t, w in rows:
        W[s-1, t-1] = w
    return W

W = load_dense(NETPATH)
G = Graph(adjacency=W)
with contextlib.redirect_stdout(io.StringIO()):  # the NuMeTriS solver prints per iteration
    G.solver(model=MODEL, maxiter=1000, tol=1e-8)

## Accuracy dump (best effort: wrapped so it can never fail the benchmark). NuMeTriS does not expose
## expected sequences, so they are reconstructed from the fitted parameters (gauge-invariant) and
## compared against NuMeTriS's own observed sequences; accuracy_comparison.jl reads the result.
try:
    _acc = '{{accfolder}}'
    os.makedirs(_acc, exist_ok=True)
    n = W.shape[0]
    viol = []
    if MODEL == 'RBCM':
        tp = np.asarray(G.params)
    else:
        tp = np.asarray(G.topological_params)
        wp = np.asarray(G.weighted_params)
    if MODEL in ('RBCM', 'RBCM+CRWCM'):
        x, y, z = np.exp(-tp[:n]), np.exp(-tp[n:2*n]), np.exp(-tp[2*n:3*n])
        D = 1 + np.outer(x, y) + np.outer(x, y).T + np.outer(z, z)
        np.fill_diagonal(D, 1)
        Pf = np.outer(x, y) / D; Rf = np.outer(z, z) / D
        np.fill_diagonal(Pf, 0); np.fill_diagonal(Rf, 0)
        viol += [np.max(np.abs(Pf.sum(axis=1) - G.dseq_right)),
                 np.max(np.abs(Pf.sum(axis=0) - G.dseq_left)),
                 np.max(np.abs(Rf.sum(axis=1) - G.dseq_rec))]
        if MODEL == 'RBCM+CRWCM':
            br, bl, bro, bri = wp[:n], wp[n:2*n], wp[2*n:3*n], wp[3*n:4*n]
            with np.errstate(divide='ignore', invalid='ignore'):
                Wnr  = np.where(Pf > 0, Pf / np.add.outer(br, bl), 0.0)
                Wrec = np.where(Rf > 0, Rf / np.add.outer(bro, bri), 0.0)
            np.fill_diagonal(Wnr, 0); np.fill_diagonal(Wrec, 0)
            viol += [np.max(np.abs(Wnr.sum(axis=1) - G.stseq_right)),
                     np.max(np.abs(Wnr.sum(axis=0) - G.stseq_left)),
                     np.max(np.abs(Wrec.sum(axis=1) - G.stseq_rec_out)),
                     np.max(np.abs(Wrec.sum(axis=0) - G.stseq_rec_in))]
    elif MODEL == 'DBCM+CReMa':
        x, y = np.exp(-tp[:n]), np.exp(-tp[n:2*n])
        F = np.outer(x, y) / (1 + np.outer(x, y))
        np.fill_diagonal(F, 0)
        viol += [np.max(np.abs(F.sum(axis=1) - G.dseq_out)),
                 np.max(np.abs(F.sum(axis=0) - G.dseq_in))]
        bo, bi = wp[:n], wp[n:2*n]
        with np.errstate(divide='ignore', invalid='ignore'):
            Wexp = np.where(F > 0, F / np.add.outer(bo, bi), 0.0)
        np.fill_diagonal(Wexp, 0)
        viol += [np.max(np.abs(Wexp.sum(axis=1) - G.stseq_out)),
                 np.max(np.abs(Wexp.sum(axis=0) - G.stseq_in))]
    dump = {"model": MODEL, "norm": float(G.norm), "ll": float(G.ll),
            "max_violation": float(np.max(viol)), "mean_violation": float(np.mean(viol)),
            "Nm_emp": [float(v) for v in np.asarray(G.Nm_emp).ravel()]}
    if hasattr(G, 'Fm_emp'):
        dump["Fm_emp"] = [float(v) for v in np.asarray(G.Fm_emp).ravel()]
    with open(os.path.join(_acc, '{{scriptname}}_numetris.json'), 'w') as f:
        json.dump(dump, f)
except Exception:
    pass

## ---------------------- ##
## Functions to benchmark ##
## ---------------------- ##
def create_graph(Wm):
    return Graph(adjacency=Wm)

def solve_graph(g, model):
    with contextlib.redirect_stdout(io.StringIO()):
        g.solver(model=model, maxiter=1000, tol=1e-8)

## ---------------------- ##
## Pytest benchmark tests ##
## ---------------------- ##
def test_create_{{fname}}(benchmark):
    benchmark(create_graph, W)

# The NuMeTriS solver is not re-entrant (re-solving an already-solved Graph can hit a singular
# Jacobian), so each round solves a FRESH Graph; the per-round setup is excluded from the timing.
def test_solve_{{fname}}(benchmark):
    def _setup():
        return (Graph(adjacency=W), MODEL), {}
    benchmark.pedantic(solve_graph, setup=_setup, rounds=10, iterations=1, warmup_rounds=1)
"""

"""
    generate_NuMeTriS_python(name::String, model::String)

Generate a python (pytest-benchmark) script solving `model` ('RBCM', 'DBCM+CReMa' or 'RBCM+CRWCM')
with NuMeTriS on the weighted edge list `data/<name>.csv`.
"""
function generate_NuMeTriS_python(name::String, model::String)
    network_data_path = joinpath(@__DIR__, "data", "$(name).csv")
    outfolder = joinpath(@__DIR__, "benchmarks")
    accfolder = joinpath(@__DIR__, "accuracy")

    out = replace(NuMeTriS_python_template, "{{scriptname}}" => name,
                                            "{{outfolder}}"  => outfolder,
                                            "{{accfolder}}"  => accfolder,
                                            "{{datafilename}}" => network_data_path,
                                            "{{model}}" => model,
                                            "{{fname}}" => replace(name, "-" => "_"))

    open(joinpath(@__DIR__, "$(name).py"), "w") do f
        write(f, out)
    end

    @info "Python (NuMeTriS) script for $(name) generated."

    return
end
