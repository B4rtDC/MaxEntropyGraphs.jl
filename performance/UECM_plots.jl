cd(joinpath(@__DIR__))
using Pkg
Pkg.activate(".")
using Plots
using Measures
using StatsPlots
using JSON
using Dates
using Statistics

# The UECM reference graphs are the (symmetrised) rhesus network and block-diagonal tilings of it, so
# the number of distinct {degree, strength} pairs stays constant (16) while N grows (16 → 128 → 512).
# The x-axis therefore shows N (the problem scale) rather than the unique-constraint count.
const UECM_positionmapper = Dict("small" => [1], "medium" => [2], "large" => [3])
const UECM_xticklabels = ["16"; "128"; "512"]

"""
    find_latest_files(path, model, language)

Find the most recent benchmark files (json extension) for a given model, and programming language.
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

# 2. Load the benchmark files
py_bench = find_latest_files(joinpath(@__DIR__,"benchmarks"), "UECM", "Python")
ju_bench = find_latest_files(joinpath(@__DIR__,"benchmarks"), "UECM", "Julia")


# 3. Try to extract the relevant information
## 3.1. Creation times
begin
    p = plot()
    # Python part
    for scale in keys(py_bench)
        python_index = findfirst(x -> x["name"] == "test_create_UECM", py_bench[scale]["benchmarks"])
        creation_times = Float64.(py_bench[scale]["benchmarks"][python_index]["stats"]["data"])
        boxplot!(p, UECM_positionmapper[scale], creation_times, label="", color=:blue, alpha=0.25, linecolor=:blue, outliers=false)
    end
    boxplot!(p,[],[], label="NEMtropy", color=:blue, alpha=0.25, linecolor=nothing)
    # Julia part
    for scale in keys(ju_bench)
        julia_index = findfirst(x -> x["name"] == "test_create_UECM", ju_bench[scale]["benchmarks"])
        creation_times = Float64.(ju_bench[scale]["benchmarks"][julia_index]["stats"][2]["times"]) ./ 1e9
        boxplot!(p, UECM_positionmapper[scale], creation_times, label="", color=:red, alpha=0.25, linecolor=:red, outliers=false)
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
        xticks=(1:3, UECM_xticklabels),
        yticks=10. .^ collect(-5:3),
        ylims=(1e-6,1e2), xlims=(0,4),
        grid=true,
        size=(800,400),
        left_margin = 5mm,
        bottom_margin = 8mm)

    isdir(joinpath(@__DIR__,"plots")) ? nothing : mkdir(joinpath(@__DIR__,"plots"))
    for ext in ["pdf", "png"]
        savefig(p, joinpath(@__DIR__,"plots", "UECM_creation_comparison ($(Dates.format(now(), "YYYY_mm_dd_HH_MM"))).$ext"))
    end
    p
end

## 3.2. Solve times (the UECM has no fixed-point recipe, so only quasi-newton and newton are shown)
begin
    trans = 0.5
    p = plot()
    # Python part
    for (method, label, color) in [ ("test_solve_UECM[ecm_exp-quasinewton-strengths]", "NEMtropy (quasi-newton)", :peru);
                                    ("test_solve_UECM[ecm_exp-newton-strengths]", "NEMtropy (newton)", :sandybrown)]
        for scale in keys(py_bench)
            ind = findfirst(x -> x["name"] == method, py_bench[scale]["benchmarks"])
            ind === nothing && continue
            solution_times = Float64.(py_bench[scale]["benchmarks"][ind]["stats"]["data"])
            boxplot!(p, UECM_positionmapper[scale], solution_times, label="", color=color, alpha=trans, outliers=false, linecolor=color)
        end
        boxplot!(p,[],[], label=label, color=color, alpha=trans, linecolor=nothing)
    end

    # Julia part
    for (method, label, color) in [ ("test_solve_UECM[ecm_exp-QN-BFGS-AG]", "MaxEntropyGraphs (quasi-newton)", :dodgerblue4);
                                    ("test_solve_UECM[ecm_exp-Newton-ADF]", "MaxEntropyGraphs (newton)", :deepskyblue)]
        for scale in keys(ju_bench)
            benchind = findfirst(x -> x["name"] == "test_solve_UECM", ju_bench[scale]["benchmarks"])
            if haskey(ju_bench[scale]["benchmarks"][benchind]["stats"][2]["data"], method)
                solution_times = ju_bench[scale]["benchmarks"][benchind]["stats"][2]["data"][method][2]["times"] ./ 1e9
                boxplot!(p, UECM_positionmapper[scale], solution_times, label="", color=color, alpha=trans, outliers=false, linecolor=color)
            end
        end
        boxplot!(p,[],[], label=label, color=color, alpha=trans, linecolor=nothing)
    end

    plot!(p, yscale=:log10, bar_width=0.5, xlabel="Number of nodes\n (problem scale)", ylabel="Time [s]",
            legendposition=:topleft, xticks=(1:3, UECM_xticklabels), title="Parameter computation (UECM)", yticks=10. .^ collect(-5:4), ylims=(1e-5,1e4))
    p
end


## 3.3. Median solve times (scatter)
begin
    trans = 0.7
    p = plot()

    approach_markers = Dict("quasi-newton" => :square, "newton" => :star5)

    # Python part
    for (method, label, color, approach) in [ ("test_solve_UECM[ecm_exp-quasinewton-strengths]", "NEMtropy (quasi-newton)", :peru, "quasi-newton");
                                              ("test_solve_UECM[ecm_exp-newton-strengths]", "NEMtropy (newton)", :sandybrown, "newton")]
        for scale in keys(py_bench)
            ind = findfirst(x -> x["name"] == method, py_bench[scale]["benchmarks"])
            ind === nothing && continue
            solution_times = Float64.(py_bench[scale]["benchmarks"][ind]["stats"]["data"])
            scatter!(p, UECM_positionmapper[scale], [median(solution_times)], label="",
            color=color, alpha=trans, marker=approach_markers[approach], outliers=false, linecolor=color, markerstrokecolor=color, markersize=10)
        end
        scatter!(p,[],[], label=label, color=color, alpha=trans, marker=approach_markers[approach], linecolor=nothing, markerstrokecolor=color)
    end

    # Julia part
    for (method, label, color, approach) in [ ("test_solve_UECM[ecm_exp-QN-BFGS-AG]", "MaxEntropyGraphs (quasi-newton)", :dodgerblue4, "quasi-newton");
                                              ("test_solve_UECM[ecm_exp-Newton-ADF]", "MaxEntropyGraphs (newton)", :dodgerblue, "newton")]
        for scale in keys(ju_bench)
            benchind = findfirst(x -> x["name"] == "test_solve_UECM", ju_bench[scale]["benchmarks"])
            if haskey(ju_bench[scale]["benchmarks"][benchind]["stats"][2]["data"], method)
                solution_times = ju_bench[scale]["benchmarks"][benchind]["stats"][2]["data"][method][2]["times"] ./ 1e9
                scatter!(p, UECM_positionmapper[scale], [median(solution_times)], label="",
                color=color, alpha=trans, marker=approach_markers[approach], outliers=false, linecolor=color, markerstrokecolor=color, markersize=10)
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
            xticks=(1:3, UECM_xticklabels),
            yticks=10. .^ collect(-5:4),
            ylims=(1e-5,1e4), xlims=(0,4),
            grid=true,
            size=(800,600),
            top_margin=3mm)

    isdir(joinpath(@__DIR__,"plots")) ? nothing : mkdir(joinpath(@__DIR__,"plots"))
    for ext in ["pdf", "png"]
        savefig(p, joinpath(@__DIR__,"plots", "UECM_computation_comparison ($(Dates.format(now(), "YYYY_mm_dd_HH_MM"))).$ext"))
    end
    p
end
