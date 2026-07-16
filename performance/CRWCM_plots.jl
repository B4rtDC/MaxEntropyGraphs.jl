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

# The CRWCM reference graphs are the rhesus network and block-diagonal tilings of it (16 → 128 → 512).
# The Python reference is NuMeTriS (model 'RBCM+CRWCM'), which has a single solver, solves BOTH layers
# (like the Julia two-step solve_model!) and JIT-compiles through numba. The Julia side shows the
# fixed-point, quasi-newton (analytical gradient) and Newton (AD) variants.
# NOTE: the model token is matched with surrounding underscores ("_CRWCM_") so it cannot collide with
# other model names in the shared benchmark folders.
const CRWCM_positionmapper = Dict("small" => [1], "medium" => [2], "large" => [3])
const CRWCM_xticklabels = ["16"; "128"; "512"]

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
# problem name, e.g. `test_solve_CRWCM_small`, so they differ per scale)
py_index(bench, prefix) = findfirst(x -> startswith(x["name"], prefix), bench["benchmarks"])

# 2. Load the benchmark files
py_bench = find_latest_files(joinpath(@__DIR__,"benchmarks"), "_CRWCM_", "Python")
ju_bench = find_latest_files(joinpath(@__DIR__,"benchmarks"), "_CRWCM_", "Julia")


# 3.1. Creation times (NuMeTriS's Graph() also computes the empirical triadic statistics on construction)
p_create = begin
    p = plot()
    # Python part
    for scale in keys(py_bench)
        ind = py_index(py_bench[scale], "test_create_")
        ind === nothing && continue
        creation_times = Float64.(py_bench[scale]["benchmarks"][ind]["stats"]["data"])
        boxplot!(p, CRWCM_positionmapper[scale], creation_times, label="", color=:blue, alpha=0.25, linecolor=:blue, outliers=false)
    end
    boxplot!(p,[],[], label="NuMeTriS", color=:blue, alpha=0.25, linecolor=nothing)
    # Julia part
    for scale in keys(ju_bench)
        julia_index = findfirst(x -> x["name"] == "test_create_CRWCM", ju_bench[scale]["benchmarks"])
        creation_times = Float64.(ju_bench[scale]["benchmarks"][julia_index]["stats"][2]["times"]) ./ 1e9
        boxplot!(p, CRWCM_positionmapper[scale], creation_times, label="", color=:red, alpha=0.25, linecolor=:red, outliers=false)
    end
    boxplot!(p,[],[], label="MaxEntropyGraphs", color=:red, alpha=0.25, linecolor=nothing)

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
        xticks=(1:3, CRWCM_xticklabels),
        yticks=10. .^ collect(-7:3),
        ylims=(1e-7,1e2), xlims=(0,4),
        grid=true,
        size=(800,400),
        left_margin = 5mm,
        bottom_margin = 8mm)

    isdir(joinpath(@__DIR__,"plots")) ? nothing : mkdir(joinpath(@__DIR__,"plots"))
    for ext in ["pdf", "png"]
        savefig(p, joinpath(@__DIR__,"plots", "CRWCM_creation_comparison ($(Dates.format(now(), "YYYY_mm_dd_HH_MM"))).$ext"))
    end
    p
end

# 3.2. Median solve times (scatter). NuMeTriS has one solver for 'RBCM+CRWCM' (both layers);
#      the Julia two-step solve is shown for its three weighted-layer methods.
p_solve = begin
    trans = 0.7
    p = plot()

    approach_markers = Dict("fixed-point" => :circle, "quasi-newton" => :square, "newton" => :star5)

    # Python part (single solver)
    for scale in keys(py_bench)
        ind = py_index(py_bench[scale], "test_solve_")
        ind === nothing && continue
        solution_times = Float64.(py_bench[scale]["benchmarks"][ind]["stats"]["data"])
        scatter!(p, CRWCM_positionmapper[scale], [median(solution_times)], label="",
        color=:peru, alpha=trans, marker=:diamond, linecolor=:peru, markerstrokecolor=:peru, markersize=10)
    end
    scatter!(p,[],[], label="NuMeTriS (RBCM+CRWCM)", color=:peru, alpha=trans, marker=:diamond, linecolor=nothing, markerstrokecolor=:peru)

    # Julia part
    for (method, label, color, approach) in [ ("test_solve_CRWCM[FP]", "MaxEntropyGraphs (fixed-point)", :green4, "fixed-point");
                                              ("test_solve_CRWCM[QN-BFGS-AG]", "MaxEntropyGraphs (quasi-newton)", :dodgerblue4, "quasi-newton");
                                              ("test_solve_CRWCM[Newton-ADF]", "MaxEntropyGraphs (newton)", :dodgerblue, "newton")]
        for scale in keys(ju_bench)
            benchind = findfirst(x -> x["name"] == "test_solve_CRWCM", ju_bench[scale]["benchmarks"])
            if haskey(ju_bench[scale]["benchmarks"][benchind]["stats"][2]["data"], method)
                solution_times = ju_bench[scale]["benchmarks"][benchind]["stats"][2]["data"][method][2]["times"] ./ 1e9
                scatter!(p, CRWCM_positionmapper[scale], [median(solution_times)], label="",
                color=color, alpha=trans, marker=approach_markers[approach], linecolor=color, markerstrokecolor=color, markersize=10)
            end
        end
        scatter!(p,[],[], label=label, color=color, alpha=trans, marker=approach_markers[approach], linecolor=nothing, markerstrokecolor=color)
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
            xticks=(1:3, CRWCM_xticklabels),
            yticks=10. .^ collect(-5:4),
            ylims=(1e-5,1e4), xlims=(0,4),
            grid=true,
            size=(800,600),
            top_margin=3mm)

    isdir(joinpath(@__DIR__,"plots")) ? nothing : mkdir(joinpath(@__DIR__,"plots"))
    for ext in ["pdf", "png"]
        savefig(p, joinpath(@__DIR__,"plots", "CRWCM_computation_comparison ($(Dates.format(now(), "YYYY_mm_dd_HH_MM"))).$ext"))
    end
    p
end

# 3.3. Combined two-panel figure for the paper (creation left, median computation right),
#      mirroring figures/ubcm_benchmark.pdf
begin
    p = plot(p_create, p_solve, layout=(1,2), size=(1600,600),
             left_margin=10mm, bottom_margin=12mm, top_margin=5mm)
    mirror_to_figures(p, "crwcm_benchmark.pdf")
    p
end
