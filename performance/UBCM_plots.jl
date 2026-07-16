cd(joinpath(@__DIR__))
using Pkg
Pkg.activate(".")
using Plots
using Measures
using StatsPlots
using Measures
using JSON
using Dates
using Statistics

include(joinpath(@__DIR__, "plot_helpers.jl"))

const UBCM_positionmapper = Dict("small" => [1], "medium" => [2], "large" => [3])

"""
    find_latest_files(model, language)

Find the most recent benchmark files (json extension) for a given model, and programming language.
"""
function find_latest_files(path::String, model, language)
    # get the subfolder
    entries = readdir(path)
    subfolders = findall(x -> occursin(language, x) && isdir(joinpath(path, x)), entries)
    if length(subfolders) == 0
        error("No subfolder found for $language")
    end
    # If several platform/version subfolders match (e.g. Julia-1.10 and Julia-1.12), use the
    # most recently modified one rather than erroring out.
    subnames = entries[subfolders]
    subfolder = joinpath(path, subnames[argmax([stat(joinpath(path, s)).mtime for s in subnames])])
    # get the benchmark files in the subfolder
    benchmark_files = filter(x -> occursin(model, x), readdir(subfolder))
    if length(benchmark_files) == 0
        error("No benchmark files found for $model")
    end
    @info benchmark_files
    res = Dict()
    # identify the number of categories
    categories = unique([x[1] for x in splitext.([split(file, "_")[end] for file in benchmark_files])])
    for cat in categories
        # find the latest files for the category
        catfiles = filter(x -> occursin(cat, x), benchmark_files)
        # get the latest file for each category
        latest_file = catfiles[argmax([stat(joinpath(subfolder, file)).mtime for file in catfiles if occursin(cat, file)])]
        res[cat] = JSON.parsefile(joinpath(path, subfolder, latest_file))
    end
    
    return res
end

# 2. Load the benchmark files
py_bench = find_latest_files(joinpath(@__DIR__,"benchmarks"), "UBCM", "Python")
ju_bench = find_latest_files(joinpath(@__DIR__,"benchmarks"), "UBCM", "Julia")



# 3. Try to extract the relevant information
## 3.1. Creation times
begin
    p = plot()
    # Python part
    for scale in keys(py_bench)
        python_index = findfirst(x -> x["name"] == "test_create_UBCM", py_bench[scale]["benchmarks"])
        creation_times = Float64.(py_bench[scale]["benchmarks"][python_index]["stats"]["data"])
        boxplot!(p, UBCM_positionmapper[scale], creation_times, label="", color=LIB_REFERENCE, alpha=0.25, linecolor=LIB_REFERENCE, outliers=false)
    end
    boxplot!(p,[],[], label="NEMtropy", color=LIB_REFERENCE, alpha=0.25, linecolor=nothing)
    # Julia part
    for scale in keys(ju_bench)
        julia_index = findfirst(x -> x["name"] == "test_create_UBCM", ju_bench[scale]["benchmarks"])
        creation_times = Float64.(ju_bench[scale]["benchmarks"][julia_index]["stats"][2]["times"]) ./ 1e9
        boxplot!(p, UBCM_positionmapper[scale], creation_times, label="", color=LIB_MEG, alpha=0.25, linecolor=LIB_MEG, outliers=false)
    end
    boxplot!(p,[],[], label="MaxEntropyGraphs", color=LIB_MEG, alpha=0.25, linecolor=nothing)

    # finalize the plot
    #plot!(p, yscale=:log10, bar_width=0.5, xlabel="Number of unique constraints\n (scale of the problem)", ylabel="Time [s]",
    #legendposition=:topleft, xticks=(1:3, ["11"; "102"; "1051"]), title="Object creation (UBCM)", yticks=10. .^ collect(-5:3), ylims=(1e-6,1e2))

    plot!(p, yscale=:log10, bar_width=0.5, 
        xlabel="Number of unique constraints\n(problem scale)", 
        ylabel="Time [s]",
        title="",#"Object creation time (UBCM)", 
        titlefontsize=18,
        legendposition=:topleft, 
        legendfontsize=12,
        tickfontsize=14,
        top_margin = 5mm,
        labelfontsize=18,
        xticks=(1:3, ["11"; "102"; "1051"]), 
        yticks=10. .^ collect(-5:3), 
        ylims=(1e-6,1e2), xlims=(0,4), 
        grid=true, 
        size=(800,400),
        left_margin = 5mm,
        bottom_margin = 8mm)

    isdir(joinpath(@__DIR__,"plots")) ? nothing : mkdir(joinpath(@__DIR__,"plots"))
    for ext in ["pdf", "png"]
        savefig(p, joinpath(@__DIR__,"plots", "UBCM_creation_comparison ($(Dates.format(now(), "YYYY_mm_dd_HH_MM"))).$ext"))
    end
    p
end

## 3.2. Solve times
begin
    trans = 0.5
    p = plot()
    # Python part
    for (method, label, color) in [ ("test_solve_UBCM[cm_exp-fixed-point-degrees]", "NEMtropy (fixed point)", :sienna4);
                                    ("test_solve_UBCM[cm_exp-quasinewton-degrees]", "NEMtropy (quasi-newton)", :peru);
                                    ("test_solve_UBCM[cm_exp-newton-degrees]", "NEMtropy (newton)", :sandybrown)]
        for scale in keys(py_bench)
            # find the benchmark containing the method
            ind = findfirst(x -> x["name"] == method, py_bench[scale]["benchmarks"])
            solution_times = Float64.(py_bench[scale]["benchmarks"][ind]["stats"]["data"])
            boxplot!(p, UBCM_positionmapper[scale], solution_times, label="", color=color, alpha=trans, outliers=false, linecolor=color)
        end
        boxplot!(p,[],[], label=label, color=color, alpha=trans, linecolor=nothing)
    end
    
    # Julia part
    for (method, label, color) in [ ("test_solve_UBCM[cm_exp-FP]", "MaxEntropyGraphs (fixed point)", :navy);
                                    ("test_solve_UBCM[cm_exp-QN-BFGS-AG]", "MaxEntropyGraphs (quasi-newton)", :dodgerblue4);
                                    ("test_solve_UBCM[cm_exp-Newton-ADF]", "MaxEntropyGraphs (newton)", :deepskyblue)]
        for scale in keys(ju_bench)
            # find the relevant benchmarks
            benchind = findfirst(x -> x["name"] == "test_solve_UBCM", ju_bench[scale]["benchmarks"])
            if haskey(ju_bench[scale]["benchmarks"][benchind]["stats"][2]["data"], method)
                solution_times = ju_bench[scale]["benchmarks"][benchind]["stats"][2]["data"][method][2]["times"] ./ 1e9
                boxplot!(p, UBCM_positionmapper[scale], solution_times, label="", color=color, alpha=trans, outliers=false, linecolor=color)
            end
        end

        boxplot!(p,[],[], label=label, color=color, alpha=trans, linecolor=nothing)
    end

    # finalize the plot
    plot!(p, yscale=:log10, bar_width=0.5, xlabel="Number of unique constraints\n (scale of the problem)", ylabel="Time [s]",
            legendposition=:topleft, xticks=(1:3, ["11"; "102"; "1051"]), title="Parameter computation (UBCM)", yticks=10. .^ collect(-5:4), ylims=(1e-5,1e4))
    p
end



begin
    trans = 0.95
    p = plot()

    # Colour is the library, marker shape is the solver method (see plot_helpers.jl).
    approach_markers = METHOD_MARKERS

    # Python part
    for (method, label, color, approach) in [ ("test_solve_UBCM[cm_exp-fixed-point-degrees]", "NEMtropy (fixed point)", LIB_REFERENCE, "fixed point");
                                              ("test_solve_UBCM[cm_exp-quasinewton-degrees]", "NEMtropy (quasi-newton)", LIB_REFERENCE, "quasi-newton");
                                              ("test_solve_UBCM[cm_exp-newton-degrees]", "NEMtropy (newton)", LIB_REFERENCE, "newton")]
        for scale in keys(py_bench)
            # find the benchmark containing the method
            ind = findfirst(x -> x["name"] == method, py_bench[scale]["benchmarks"])
            solution_times = Float64.(py_bench[scale]["benchmarks"][ind]["stats"]["data"])
            scatter!(p, mark_x(UBCM_positionmapper[scale], color, approach), [median(solution_times)], label="",
            color=color, alpha=trans, marker=approach_markers[approach], outliers=false, linecolor=color,
            markerstrokecolor=MARK_STROKE, markerstrokewidth=MARK_STROKE_WIDTH, markersize=mark_size(approach))
        end
        scatter!(p,[],[], label=label, color=color, alpha=trans, marker=approach_markers[approach], linecolor=nothing, markerstrokecolor=MARK_STROKE, markerstrokewidth=MARK_STROKE_WIDTH, markersize=mark_size(approach)) # Dummy scatter for legend
    end

    # Julia part
    for (method, label, color, approach) in [ ("test_solve_UBCM[cm_exp-FP]", "MaxEntropyGraphs (fixed point)", LIB_MEG, "fixed point");
                                              ("test_solve_UBCM[cm_exp-QN-BFGS-AG]", "MaxEntropyGraphs (quasi-newton)", LIB_MEG, "quasi-newton");
                                              ("test_solve_UBCM[cm_exp-Newton-ADF]", "MaxEntropyGraphs (newton)", LIB_MEG, "newton")]
        for scale in keys(ju_bench)
            # find the relevant benchmarks
            benchind = findfirst(x -> x["name"] == "test_solve_UBCM", ju_bench[scale]["benchmarks"])
            if haskey(ju_bench[scale]["benchmarks"][benchind]["stats"][2]["data"], method)
                solution_times = ju_bench[scale]["benchmarks"][benchind]["stats"][2]["data"][method][2]["times"] ./ 1e9
                scatter!(p, mark_x(UBCM_positionmapper[scale], color, approach), [median(solution_times)], label="",
                color=color, alpha=trans, marker=approach_markers[approach], outliers=false, linecolor=color,
                markerstrokecolor=MARK_STROKE, markerstrokewidth=MARK_STROKE_WIDTH, markersize=mark_size(approach))
            end
        end
        scatter!(p,[],[], label=label, color=color, alpha=trans, marker=approach_markers[approach], linecolor=nothing, markerstrokecolor=MARK_STROKE, markerstrokewidth=MARK_STROKE_WIDTH, markersize=mark_size(approach)) # Dummy scatter for legend
    end

    # finalize the plot
    plot!(p, yscale=:log10, bar_width=0.5, 
            xlabel="Number of unique constraints\n(problem scale)", 
            ylabel="Computation time [s]",
            title="",#"Median computation time (UBCM)", 
            titlefontsize=18,
            legendposition=:topleft, 
            legendfontsize=12,
            tickfontsize=14,
            labelfontsize=18,
            xticks=(1:3, ["11"; "102"; "1051"]), 
            yticks=10. .^ collect(-5:4), 
            ylims=(1e-5,1e4), xlims=(0,4), 
            grid=true, 
            size=(800,600), 
            top_margin=3mm)

    isdir(joinpath(@__DIR__,"plots")) ? nothing : mkdir(joinpath(@__DIR__,"plots"))
    for ext in ["pdf", "png"]
        savefig(p, joinpath(@__DIR__,"plots", "UBCM_computation_comparison ($(Dates.format(now(), "YYYY_mm_dd_HH_MM"))).$ext"))
    end
    # This panel is figures/ubcm_benchmark.pdf in the paper (\autoref{fig:ubcm}).
    mirror_to_figures(p, "ubcm_benchmark.pdf")
    p
end


