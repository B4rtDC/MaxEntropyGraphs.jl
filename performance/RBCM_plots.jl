cd(joinpath(@__DIR__))
using Pkg
Pkg.activate(".")
using Plots
using Measures
using StatsPlots
using JSON
using Dates
using Statistics

include(joinpath(@__DIR__, "plot_helpers.jl"))

# The RBCM reference graphs are the (binarised) rhesus network and block-diagonal tilings of it
# (16 → 128 → 512). The Python reference is NuMeTriS (model 'RBCM'), which has a single solver and
# JIT-compiles through numba. The Julia side shows the fixed-point, quasi-newton (analytical gradient)
# and Newton (AD) variants.
# NOTE: the model token is matched with surrounding underscores ("_RBCM_") so it cannot collide with
# other model names in the shared benchmark folders.
const RBCM_positionmapper = Dict("small" => [1], "medium" => [2], "large" => [3])
const RBCM_xticklabels = ["16"; "128"; "512"]

"""
    find_latest_files(path, model, language)

Find the most recent benchmark files (json extension) for a given model token, and programming language.
"""
function find_latest_files(path::String, model, language)
    entries = readdir(path)
    subfolders = findall(x -> occursin(language, x) && isdir(joinpath(path, x)), entries)
    if length(subfolders) == 0
        error("No subfolder found for $language")
    end
    subnames = entries[subfolders]
    subfolder = joinpath(path, subnames[argmax([stat(joinpath(path, s)).mtime for s in subnames])])
    benchmark_files = filter(x -> occursin(model, x), readdir(subfolder))
    if length(benchmark_files) == 0
        error("No benchmark files found for $model")
    end
    @info benchmark_files
    res = Dict()
    categories = unique([x[1] for x in splitext.([split(file, "_")[end] for file in benchmark_files])])
    for cat in categories
        catfiles = filter(x -> occursin(cat, x), benchmark_files)
        latest_file = catfiles[argmax([stat(joinpath(subfolder, file)).mtime for file in catfiles if occursin(cat, file)])]
        res[cat] = JSON.parsefile(joinpath(path, subfolder, latest_file))
    end

    return res
end

# find the (single) python benchmark whose name starts with a prefix (the NuMeTriS test names carry the
# problem name, e.g. `test_solve_RBCM_small`, so they differ per scale)
py_index(bench, prefix) = findfirst(x -> startswith(x["name"], prefix), bench["benchmarks"])

# 2. Load the benchmark files
py_bench = find_latest_files(joinpath(@__DIR__,"benchmarks"), "_RBCM_", "Python")
ju_bench = find_latest_files(joinpath(@__DIR__,"benchmarks"), "_RBCM_", "Julia")


# 3.1. Creation times (NuMeTriS's Graph() also computes the empirical triadic statistics on construction)
p_create = begin
    p = plot()
    # Python part
    for scale in keys(py_bench)
        ind = py_index(py_bench[scale], "test_create_")
        ind === nothing && continue
        creation_times = Float64.(py_bench[scale]["benchmarks"][ind]["stats"]["data"])
        boxplot!(p, RBCM_positionmapper[scale], creation_times, label="", color=LIB_REFERENCE, alpha=0.25, linecolor=LIB_REFERENCE, outliers=false)
    end
    boxplot!(p,[],[], label="NuMeTriS", color=LIB_REFERENCE, alpha=0.25, linecolor=nothing)
    # Julia part
    for scale in keys(ju_bench)
        julia_index = findfirst(x -> x["name"] == "test_create_RBCM", ju_bench[scale]["benchmarks"])
        creation_times = Float64.(ju_bench[scale]["benchmarks"][julia_index]["stats"][2]["times"]) ./ 1e9
        boxplot!(p, RBCM_positionmapper[scale], creation_times, label="", color=LIB_MEG, alpha=0.25, linecolor=LIB_MEG, outliers=false)
    end
    boxplot!(p,[],[], label="MaxEntropyGraphs", color=LIB_MEG, alpha=0.25, linecolor=nothing)

    plot!(p, yscale=:log10, bar_width=0.5,
        xlabel="Number of nodes\n(problem scale)",
        ylabel="Time [s]",
        title="",
        titlefontsize=18,
        legendposition=:topleft,
        legendfontsize=12,
        tickfontsize=14,
        top_margin = 5mm,
        labelfontsize=18,
        xticks=(1:3, RBCM_xticklabels),
        yticks=10. .^ collect(-7:3),
        ylims=(1e-7,1e2), xlims=(0,4),
        grid=true,
        size=(800,400),
        left_margin = 5mm,
        bottom_margin = 8mm)

    isdir(joinpath(@__DIR__,"plots")) ? nothing : mkdir(joinpath(@__DIR__,"plots"))
    for ext in ["pdf", "png"]
        savefig(p, joinpath(@__DIR__,"plots", "RBCM_creation_comparison ($(Dates.format(now(), "YYYY_mm_dd_HH_MM"))).$ext"))
    end
    p
end

# 3.2. Median solve times (scatter). NuMeTriS has one solver for 'RBCM+RBCM' (both layers);
#      the Julia two-step solve is shown for its three weighted-layer methods.
p_solve = begin
    trans = 0.95
    p = plot()

    # Colour is the library, marker shape is the solver method (see plot_helpers.jl).
    approach_markers = METHOD_MARKERS

    # Python part (single solver). NuMeTriS exposes one solver for 'RBCM+RBCM' rather than a
    # choice of the three methods the shapes encode, so it keeps its own diamond and sits in the
    # centre of the reference cluster (an unknown approach dodges by 0.0).
    for scale in keys(py_bench)
        ind = py_index(py_bench[scale], "test_solve_")
        ind === nothing && continue
        solution_times = Float64.(py_bench[scale]["benchmarks"][ind]["stats"]["data"])
        scatter!(p, mark_x(RBCM_positionmapper[scale], LIB_REFERENCE, "single"), [median(solution_times)], label="",
        color=LIB_REFERENCE, alpha=trans, marker=:diamond, linecolor=LIB_REFERENCE,
        markerstrokecolor=MARK_STROKE, markerstrokewidth=MARK_STROKE_WIDTH, markersize=mark_size("single"))
    end
    scatter!(p,[],[], label="NuMeTriS (RBCM)", color=LIB_REFERENCE, alpha=trans, marker=:diamond, linecolor=nothing, markerstrokecolor=MARK_STROKE, markerstrokewidth=MARK_STROKE_WIDTH, markersize=mark_size("single"))

    # Julia part
    for (method, label, color, approach) in [ ("test_solve_RBCM[FP]", "MaxEntropyGraphs (fixed-point)", LIB_MEG, "fixed point");
                                              ("test_solve_RBCM[QN-BFGS-AG]", "MaxEntropyGraphs (quasi-newton)", LIB_MEG, "quasi-newton");
                                              ("test_solve_RBCM[Newton-ADF]", "MaxEntropyGraphs (newton)", LIB_MEG, "newton")]
        for scale in keys(ju_bench)
            benchind = findfirst(x -> x["name"] == "test_solve_RBCM", ju_bench[scale]["benchmarks"])
            if haskey(ju_bench[scale]["benchmarks"][benchind]["stats"][2]["data"], method)
                solution_times = ju_bench[scale]["benchmarks"][benchind]["stats"][2]["data"][method][2]["times"] ./ 1e9
                scatter!(p, mark_x(RBCM_positionmapper[scale], color, approach), [median(solution_times)], label="",
                color=color, alpha=trans, marker=approach_markers[approach], linecolor=color,
                markerstrokecolor=MARK_STROKE, markerstrokewidth=MARK_STROKE_WIDTH, markersize=mark_size(approach))
            end
        end
        scatter!(p,[],[], label=label, color=color, alpha=trans, marker=approach_markers[approach], linecolor=nothing, markerstrokecolor=MARK_STROKE, markerstrokewidth=MARK_STROKE_WIDTH, markersize=mark_size(approach))
    end

    plot!(p, yscale=:log10, bar_width=0.5,
            xlabel="Number of nodes\n(problem scale)",
            ylabel="Computation time [s]",
            title="",
            titlefontsize=18,
            legendposition=:topleft,
            legendfontsize=12,
            tickfontsize=14,
            labelfontsize=18,
            xticks=(1:3, RBCM_xticklabels),
            yticks=10. .^ collect(-5:4),
            ylims=(1e-5,1e4), xlims=(0,4),
            grid=true,
            size=(800,600),
            top_margin=3mm)

    isdir(joinpath(@__DIR__,"plots")) ? nothing : mkdir(joinpath(@__DIR__,"plots"))
    for ext in ["pdf", "png"]
        savefig(p, joinpath(@__DIR__,"plots", "RBCM_computation_comparison ($(Dates.format(now(), "YYYY_mm_dd_HH_MM"))).$ext"))
    end
    p
end
