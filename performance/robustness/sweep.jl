# Solver-robustness sweep: model x method x initial guess x graph.
#
# What this answers, and what `performance/` does not: `performance/` sweeps methods and AD
# backends but pins `initial = :degrees` everywhere, so the package's sensitivity to its starting
# point has never been measured systematically. The published robustness tables in
# validation/*.md and CHANGELOG.md came from one-off scripts that were never committed. This is
# their home.
#
# Usage:
#   julia --project=. robustness/sweep.jl
#
# Environment variables:
#   ROB_MODELS    space-separated subset of the ten models (default: all)
#   ROB_METHODS   space-separated subset of :fixedpoint :BFGS :LBFGS :Newton (default: all)
#   ROB_INITIALS  space-separated subset of the initial guesses (default: each model's own set)
#   ROB_NGRAPHS   graphs per corpus (default: 50; the published tables used 150 to 200)
#   ROB_OUT       output path (default: robustness/results/sweep_<julia>_<timestamp>.json)
#   ROB_VERBOSE   1 to print every row as it lands

using Dates, JSON, Logging, Printf, Statistics
include(joinpath(@__DIR__, "corpus.jl"))
include(joinpath(@__DIR__, "residual.jl"))

"""
Per-model sweep definition.

`initials` follows each model's own vocabulary, which differs: the binary unipartite models accept
`:degrees_minor` and `:chung_lu`, the bipartite ones drop `:degrees_minor`, the weighted models
speak `:strengths`, and the two-step models have no `:uniform` at all. Passing a guess a model does
not know is an `ArgumentError`, not a failure to converge, so the grid must be per-model.
"""
const MODEL_SPECS = [
    (name = :UBCM,  corpus = :binary_undirected,  build = g -> MEG.UBCM(g),
     methods = (:fixedpoint, :BFGS, :LBFGS, :Newton),
     initials = (:degrees, :degrees_minor, :random, :uniform, :chung_lu)),
    (name = :DBCM,  corpus = :binary_directed,    build = g -> MEG.DBCM(g),
     methods = (:fixedpoint, :BFGS, :LBFGS, :Newton),
     initials = (:degrees, :degrees_minor, :random, :uniform, :chung_lu)),
    (name = :RBCM,  corpus = :binary_directed,    build = g -> MEG.RBCM(g),
     methods = (:fixedpoint, :BFGS, :LBFGS, :Newton),
     initials = (:degrees, :degrees_minor, :random, :uniform, :chung_lu)),
    (name = :BiCM,  corpus = :bipartite,          build = g -> MEG.BiCM(g),
     methods = (:fixedpoint, :BFGS, :LBFGS, :Newton),
     initials = (:degrees, :random, :uniform, :chung_lu)),
    (name = :DBiCM, corpus = :dibipartite,        build = g -> MEG.DBiCM(g),
     methods = (:fixedpoint, :BFGS, :LBFGS, :Newton),
     initials = (:degrees, :random, :uniform, :chung_lu)),
    (name = :UECM,  corpus = :weighted_undirected, build = g -> MEG.UECM(g),
     methods = (:fixedpoint, :BFGS, :LBFGS, :Newton),
     initials = (:strengths, :strengths_minor, :random, :uniform)),
    (name = :DECM,  corpus = :weighted_directed,   build = g -> MEG.DECM(g),
     methods = (:fixedpoint, :BFGS, :LBFGS, :Newton),
     initials = (:strengths, :strengths_minor, :random, :uniform)),
    (name = :CReM,  corpus = :weighted_undirected, build = g -> MEG.CReM(g),
     methods = (:fixedpoint, :BFGS, :LBFGS, :Newton),
     initials = (:strengths, :strengths_minor, :random)),
    (name = :DCReM, corpus = :weighted_directed,   build = g -> MEG.DCReM(g),
     methods = (:fixedpoint, :BFGS, :LBFGS, :Newton),
     initials = (:strengths, :strengths_minor, :random)),
    (name = :CRWCM, corpus = :weighted_directed,   build = g -> MEG.CRWCM(g),
     methods = (:fixedpoint, :BFGS, :LBFGS, :Newton),
     initials = (:strengths, :strengths_minor, :random)),
]

"""
    solve_cell(spec, g, method, initial) -> NamedTuple

Run one cell and classify the outcome. Nothing here is allowed to throw: a solver that fails is a
measurement, not an error.

`:Newton` pins `AD_method = :AutoForwardDiff`. With the `:AutoZygote` default, `OptimizationBase`
wraps a `SecondOrder(AutoZygote, AutoForwardDiff)` whose nested HVP path **aborts the Julia
process** (`signal 4: illegal instruction`) when `Symbolics` is loaded. BiCM, DBiCM, UECM and DECM
carry an internal override; UBCM, DBCM and RBCM do not, so an unpinned sweep would take the whole
run down with it.
"""
function solve_cell(spec, g, method, initial)
    m = try
        spec.build(g)
    catch e
        return (status = "construct_failed", residual = Inf, iterations = -1,
                seconds = 0.0, runaway = false, error = _errname(e))
    end
    rw = try; runaway(m); catch; false; end
    kw = method === :Newton ? (AD_method = :AutoForwardDiff,) : NamedTuple()
    t0 = time()
    status, err = "converged", nothing
    sol = nothing
    # Warnings are silenced only around the solve itself. Every outcome is recorded explicitly
    # below, so nothing is lost, and the alternative is tens of thousands of identical
    # `SecondOrder ADtype was not provided` lines from OptimizationBase burying the run.
    try
        Logging.with_logger(Logging.ConsoleLogger(stderr, Logging.Error)) do
            _, sol = MEG.solve_model!(m; method = method, initial = initial, kw...)
        end
    catch e
        status = e isa MEG.ConvergenceError ? "not_converged" : "error"
        err = _errname(e)
    end
    secs = time() - t0
    # A cell that threw has no meaningful iteration count: recording 0 would drag the medians
    # of the failing subsets toward zero and make a solver that gives up instantly look cheap.
    its = status == "converged" ? solve_iterations(sol) : -1
    (status = status, residual = residual(m), iterations = its,
     seconds = secs, runaway = rw, error = err)
end

_errname(e) = string(nameof(typeof(e)))

function main()
    wanted = get(ENV, "ROB_MODELS", "")
    specs  = isempty(wanted) ? MODEL_SPECS :
             filter(s -> string(s.name) in split(wanted), MODEL_SPECS)
    ngraphs = parse(Int, get(ENV, "ROB_NGRAPHS", "50"))
    # Method and guess filters intersect with each model's own vocabulary, so a filter naming a
    # guess a model does not know simply drops that cell rather than raising.
    mfilter = split(get(ENV, "ROB_METHODS", ""))
    ifilter = split(get(ENV, "ROB_INITIALS", ""))
    keepm(x) = isempty(mfilter) || string(x) in mfilter
    keepi(x) = isempty(ifilter) || string(x) in ifilter
    verbose = get(ENV, "ROB_VERBOSE", "0") == "1"

    outdir = joinpath(@__DIR__, "results")
    mkpath(outdir)
    stamp = Dates.format(now(), "yyyymmdd-HHMMSS")
    outfile = get(ENV, "ROB_OUT", joinpath(outdir, "sweep_julia-$(VERSION)_$(stamp).json"))

    @info "robustness sweep" models=[s.name for s in specs] ngraphs julia=VERSION threads=Threads.nthreads()

    rows = Dict{String,Any}[]
    corpora = Dict{Symbol,Vector}()

    for spec in specs
        graphs = get!(corpora, spec.corpus) do
            corpus(spec.corpus, ngraphs)
        end
        # Warm up every (method, initial) once so the recorded seconds measure the solve rather
        # than Julia's first-call compilation.
        methods  = filter(keepm, collect(spec.methods))
        initials = filter(keepi, collect(spec.initials))
        for method in methods, initial in initials
            solve_cell(spec, graphs[1], method, initial)
        end
        flush(stdout)
        for method in methods, initial in initials
            t_cell = time()
            for (gi, g) in enumerate(graphs)
                r = solve_cell(spec, g, method, initial)
                push!(rows, Dict{String,Any}(
                    "model" => string(spec.name), "corpus" => string(spec.corpus),
                    "method" => string(method), "initial" => string(initial),
                    "graph" => gi, "nv" => MEG.Graphs.nv(g), "ne" => MEG.Graphs.ne(g),
                    "status" => r.status, "residual" => r.residual,
                    "iterations" => r.iterations, "seconds" => r.seconds,
                    "runaway" => r.runaway, "error" => r.error === nothing ? "" : r.error))
                verbose && @printf("%-6s %-11s %-16s g%-3d  %-14s %.2e\n",
                                   spec.name, method, initial, gi, r.status, r.residual)
            end
            @printf("%-6s %-11s %-16s  %d graphs in %.1f s\n",
                    spec.name, method, initial, length(graphs), time() - t_cell)
        end
    end

    meta = Dict{String,Any}(
        "julia" => string(VERSION), "threads" => Threads.nthreads(),
        "timestamp" => string(now()), "ngraphs" => ngraphs,
        "seeds" => Dict("bipartite" => SEED_BIPARTITE, "weighted" => SEED_WEIGHTED,
                        "binary" => SEED_BINARY),
        "package_version" => string(pkgversion(MaxEntropyGraphs)))
    open(outfile, "w") do io
        JSON.print(io, Dict("meta" => meta, "rows" => rows))
    end
    @info "wrote" outfile rows=length(rows)
    outfile
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
