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

const BiCM_positionmapper = Dict("small" => [1], "medium" => [2], "large" => [3])

"""
    find_latest_files(model, language)

Find the most recent benchmark files (json extension) for a given model, and programming language.
"""
function find_latest_files(path::String, model, language)
    # get the subfolder
    subfolders = findall(x -> occursin(language, x) && isdir(joinpath(path,x)), readdir(path))
    if length(subfolders) == 0
        error("No subfolder found for $language")
    elseif length(subfolders) > 1
        error("Multiple subfolders found for $language")
    end
    subfolder = joinpath(path, readdir(path)[subfolders[1]])
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
py_bench = find_latest_files(joinpath(@__DIR__,"benchmarks"), "BiCM", "Python")
ju_bench = find_latest_files(joinpath(@__DIR__,"benchmarks"), "BiCM", "Julia")



# 3. Try to extract the relevant information
## 3.1. Creation times
begin
    p = plot()
    # Python part
    for scale in keys(py_bench)
        python_index = findfirst(x -> x["name"] == "test_create_BiCM", py_bench[scale]["benchmarks"])
        creation_times = Float64.(py_bench[scale]["benchmarks"][python_index]["stats"]["data"])
        boxplot!(p, BiCM_positionmapper[scale], creation_times, label="", color=:blue, alpha=0.25, linecolor=:blue, outliers=false)
    end
    boxplot!(p,[],[], label="NEMtropy", color=:blue, alpha=0.25, linecolor=nothing)
    # Julia part
    for scale in keys(ju_bench)
        julia_index = findfirst(x -> x["name"] == "test_create_BiCM", ju_bench[scale]["benchmarks"])
        creation_times = Float64.(ju_bench[scale]["benchmarks"][julia_index]["stats"][2]["times"]) ./ 1e9
        boxplot!(p, BiCM_positionmapper[scale], creation_times, label="", color=:red, alpha=0.25, linecolor=:red, outliers=false)
    end
    boxplot!(p,[],[], label="MaxEntropyGraphs.jl", color=:red, alpha=0.25, linecolor=nothing)

    # finalize the plot
    #plot!(p, yscale=:log10, bar_width=0.5, xlabel="Number of unique constraints\n (scale of the problem)", ylabel="Time [s]",
    #legendposition=:topleft, xticks=(1:3, ["11"; "102"; "1051"]), title="Object creation (UBCM)", yticks=10. .^ collect(-5:3), ylims=(1e-6,1e2))

    plot!(p, yscale=:log10, bar_width=0.5, 
        xlabel="Number of unique constraints\n(problem scale)", 
        ylabel="Time [s]",
        title="Object creation time (BiCM)", 
        titlefontsize=18,
        legendposition=:topleft, 
        legendfontsize=12,
        tickfontsize=14,
        labelfontsize=18,
        xticks=(1:3, ["12"; "105"; "198"]), 
        yticks=10. .^ collect(-5:3), 
        ylims=(1e-6,1e2), xlims=(0,4), 
        grid=true, 
        size=(800,400),
        left_margin = 5mm,
        bottom_margin = 8mm)

    isdir(joinpath(@__DIR__,"plots")) ? nothing : mkdir(joinpath(@__DIR__,"plots"))
    for ext in ["pdf", "png"]
        savefig(p, joinpath(@__DIR__,"plots", "BiCM_creation_comparison ($(Dates.format(now(), "YYYY_mm_dd_HH_MM"))).$ext"))
    end
    p
end

## 3.3. Projection times
begin
    trans = 0.7
    p = plot()

    # Define markers for different distributions 
    approach_markers = Dict("Poisson" => :circle, "Poisson-Binomial" => :square)

    # Python part
    for (method, label, color, approach) in [   ("test_project_BiCM[poibin-True-1]", "NEMtropy (Poisson-Binomial, single thread)",:sienna4, "Poisson-Binomial");
                                                ("test_project_BiCM[poibin-True-4]", "NEMtropy (Poisson-Binomial, multithreaded)",:peru, "Poisson-Binomial");
                                                ("test_project_BiCM[poisson-True-1]", "NEMtropy (Poisson, multithreaded)",:sandybrown, "Poisson");
                                                ("test_project_BiCM[poisson-True-4]", "NEMtropy (Poisson, multithreaded)",:chocolate2, "Poisson");
        ]
        for scale in keys(py_bench)
            # find the benchmark containing the method
            ind = findfirst(x -> x["name"] == method, py_bench[scale]["benchmarks"])
            if ind != nothing
                projection_times = Float64.(py_bench[scale]["benchmarks"][ind]["stats"]["data"])
                scatter!(p, BiCM_positionmapper[scale], [median(projection_times)], label="",
                color=color, alpha=trans, marker=approach_markers[approach], outliers=false, linecolor=color, markerstrokecolor=color, markersize=10)
            end
        end
        scatter!(p,[],[], label=label, color=color, alpha=trans, marker=approach_markers[approach], linecolor=nothing, markerstrokecolor=color) # Dummy scatter for legend
    end

    # Julia part test_project_BiCM[cm_exp-$(layer)-$(precomputed)-$(distribution)-$(multithreaded)]
    for (method, label, color, approach) in [   ("test_project_BiCM[cm_exp-bottom-true-PoissonBinomial-false]", "MaxEntropyGraphs (Poisson-Binomial, single thread)", :navy, "Poisson-Binomial");
                                                ("test_project_BiCM[cm_exp-bottom-true-PoissonBinomial-true]", "MaxEntropyGraphs (Poisson-Binomial, multithreaded)", :dodgerblue4, "Poisson-Binomial");
                                                ("test_project_BiCM[cm_exp-bottom-true-Poisson-false]", "MaxEntropyGraphs (Poisson, single thread)", :dodgerblue, "Poisson");
                                                ("test_project_BiCM[cm_exp-bottom-true-Poisson-true]", "MaxEntropyGraphs (Poisson, multithreaded)", :deepskyblue, "Poisson")]
        for scale in keys(ju_bench)
            # find the relevant benchmarks
            benchind = findfirst(x -> x["name"] == "test_project_BiCM", ju_bench[scale]["benchmarks"])
            if haskey(ju_bench[scale]["benchmarks"][benchind]["stats"][2]["data"], method)
                projection_times = ju_bench[scale]["benchmarks"][benchind]["stats"][2]["data"][method][2]["times"] ./ 1e9
                scatter!(p, BiCM_positionmapper[scale], [median(projection_times)], label="", 
                color=color, alpha=trans, marker=approach_markers[approach], outliers=false, linecolor=color, markerstrokecolor=color, markersize=10)
            end
        end
        scatter!(p,[],[], label=label, color=color, alpha=trans, marker=approach_markers[approach], linecolor=nothing, markerstrokecolor=color) # Dummy scatter for legend
    end

    # finalize the plot
    plot!(p, yscale=:log10, bar_width=0.5, 
            xlabel="Number of unique constraints\n(problem scale)", 
            ylabel="Projection time [s]",
            title="Median projection time (BiCM)", 
            titlefontsize=18,
            legendposition=:topleft, 
            legendfontsize=12,
            tickfontsize=14,
            labelfontsize=18,
            xticks=(1:3, ["12"; "105"; "198"]), 
            yticks=10. .^ collect(-5:4), 
            ylims=(1e-5,1e4), xlims=(0,4), 
            grid=true, 
            size=(800,600))

    isdir(joinpath(@__DIR__,"plots")) ? nothing : mkdir(joinpath(@__DIR__,"plots"))
    for ext in ["pdf", "png"]
        savefig(p, joinpath(@__DIR__,"plots", "BiCM_projection_comparison ($(Dates.format(now(), "YYYY_mm_dd_HH_MM"))).$ext"))
    end
    p
end
begin
    trans = 0.7
    p = plot()

    # Define markers for different distributions 
    approach_markers = Dict("Poisson" => :circle, "Poisson-Binomial" => :square)

    # Python part
    for (method, label, color, approach) in [   ("test_project_BiCM[poibin-False-1]", "NEMtropy (Poisson-Binomial, single thread)",:sienna4, "Poisson-Binomial");
                                                ("test_project_BiCM[poibin-False-4]", "NEMtropy (Poisson-Binomial, multithreaded)",:peru, "Poisson-Binomial");
                                                ("test_project_BiCM[poisson-False-1]", "NEMtropy (Poisson, multithreaded)",:sandybrown, "Poisson");
                                                ("test_project_BiCM[poisson-False-4]", "NEMtropy (Poisson, multithreaded)",:chocolate2, "Poisson");
        ]
        for scale in keys(py_bench)
            # find the benchmark containing the method
            ind = findfirst(x -> x["name"] == method, py_bench[scale]["benchmarks"])
            if ind != nothing
                projection_times = Float64.(py_bench[scale]["benchmarks"][ind]["stats"]["data"])
                scatter!(p, BiCM_positionmapper[scale], [median(projection_times)], label="",
                color=color, alpha=trans, marker=approach_markers[approach], outliers=false, linecolor=color, markerstrokecolor=color, markersize=10)
            end
        end
        scatter!(p,[],[], label=label, color=color, alpha=trans, marker=approach_markers[approach], linecolor=nothing, markerstrokecolor=color) # Dummy scatter for legend
    end

    # Julia part test_project_BiCM[cm_exp-$(layer)-$(precomputed)-$(distribution)-$(multithreaded)]
    for (method, label, color, approach) in [   ("test_project_BiCM[cm_exp-top-true-PoissonBinomial-false]", "MaxEntropyGraphs (Poisson-Binomial, single thread)", :navy, "Poisson-Binomial");
                                                ("test_project_BiCM[cm_exp-top-true-PoissonBinomial-true]", "MaxEntropyGraphs (Poisson-Binomial, multithreaded)", :dodgerblue4, "Poisson-Binomial");
                                                ("test_project_BiCM[cm_exp-top-true-Poisson-false]", "MaxEntropyGraphs (Poisson, single thread)", :dodgerblue, "Poisson");
                                                ("test_project_BiCM[cm_exp-top-true-Poisson-true]", "MaxEntropyGraphs (Poisson, multithreaded)", :deepskyblue, "Poisson")]
        for scale in keys(ju_bench)
            # find the relevant benchmarks
            benchind = findfirst(x -> x["name"] == "test_project_BiCM", ju_bench[scale]["benchmarks"])
            if haskey(ju_bench[scale]["benchmarks"][benchind]["stats"][2]["data"], method)
                projection_times = ju_bench[scale]["benchmarks"][benchind]["stats"][2]["data"][method][2]["times"] ./ 1e9
                scatter!(p, BiCM_positionmapper[scale], [median(projection_times)], label="", 
                color=color, alpha=trans, marker=approach_markers[approach], outliers=false, linecolor=color, markerstrokecolor=color, markersize=10)
            end
        end
        scatter!(p,[],[], label=label, color=color, alpha=trans, marker=approach_markers[approach], linecolor=nothing, markerstrokecolor=color) # Dummy scatter for legend
    end

    # finalize the plot
    plot!(p, yscale=:log10, bar_width=0.5, 
            xlabel="Number of unique constraints\n(problem scale)", 
            ylabel="Projection time [s]",
            title="Median projection time (BiCM)", 
            titlefontsize=18,
            legendposition=:outertopright, 
            legendfontsize=12,
            tickfontsize=14,
            labelfontsize=18,
            left_margin = 5mm,
            bottom_margin = 8mm,
            xticks=(1:3, ["12"; "105"; "198"]), 
            yticks=10. .^ collect(-5:4), 
            ylims=(1e-5,1e4), xlims=(0.5,3.5), 
            grid=true, 
            size=(1200,600))

    isdir(joinpath(@__DIR__,"plots")) ? nothing : mkdir(joinpath(@__DIR__,"plots"))
    for ext in ["pdf", "png"]
        savefig(p, joinpath(@__DIR__,"plots", "BiCM_projection_comparison ($(Dates.format(now(), "YYYY_mm_dd_HH_MM"))).$ext"))
    end
    p
end


## 3.2. Computation times
begin
    trans = 0.7
    p = plot()

    # Define markers for different solution approaches
    approach_markers = Dict("fixed point" => :circle, "quasi-newton" => :square, "newton" => :star5)

    # Python part
    for (method, label, color, approach) in [ ("test_solve_BiCM[fixed-point-degrees]", "NEMtropy (fixed point)", :sienna4, "fixed point");
                                              ("test_solve_BiCM[quasinewton-degrees]", "NEMtropy (quasi-newton)", :peru, "quasi-newton");
                                              ("test_solve_BiCM[newton-degrees]", "NEMtropy (newton)", :sandybrown, "newton")]
        for scale in keys(py_bench)
            # find the benchmark containing the method
            ind = findfirst(x -> x["name"] == method, py_bench[scale]["benchmarks"])
            if ind != nothing
                solution_times = Float64.(py_bench[scale]["benchmarks"][ind]["stats"]["data"])
                scatter!(p, BiCM_positionmapper[scale], [median(solution_times)], label="", 
                color=color, alpha=trans, marker=approach_markers[approach], outliers=false, linecolor=color, markerstrokecolor=color, markersize=10)
            end
        end
        scatter!(p,[],[], label=label, color=color, alpha=trans, marker=approach_markers[approach], linecolor=nothing, markerstrokecolor=color) # Dummy scatter for legend
    end
    
    # Julia part
    for (method, label, color, approach) in [ ("test_solve_BiCM[cm_exp-FP]", "MaxEntropyGraphs (fixed point)", :navy, "fixed point");
                                              ("test_solve_BiCM[cm_exp-QN-BFGS-AG]", "MaxEntropyGraphs (quasi-newton)", :dodgerblue4, "quasi-newton");
                                              ("test_solve_BiCM[cm_exp-Newton-ADF]", "MaxEntropyGraphs (newton)", :dodgerblue, "newton")]
        for scale in keys(ju_bench)
            # find the relevant benchmarks
            benchind = findfirst(x -> x["name"] == "test_solve_BiCM", ju_bench[scale]["benchmarks"])
            if haskey(ju_bench[scale]["benchmarks"][benchind]["stats"][2]["data"], method)
                solution_times = ju_bench[scale]["benchmarks"][benchind]["stats"][2]["data"][method][2]["times"] ./ 1e9
                scatter!(p, BiCM_positionmapper[scale], [median(solution_times)], label="", 
                color=color, alpha=trans, marker=approach_markers[approach], outliers=false, linecolor=color, markerstrokecolor=color, markersize=10)
            end
        end
        scatter!(p,[],[], label=label, color=color, alpha=trans, marker=approach_markers[approach], linecolor=nothing, markerstrokecolor=color) # Dummy scatter for legend
    end

    # finalize the plot
    plot!(p, yscale=:log10, bar_width=0.5, 
            xlabel="Number of unique constraints\n(problem scale)", 
            ylabel="Computation time [s]",
            title="Median computation time (BiCM)", 
            titlefontsize=18,
            legendposition=:topleft, 
            legendfontsize=12,
            tickfontsize=14,
            labelfontsize=18,
            xticks=(1:3, ["12"; "105"; "198"]), 
            yticks=10. .^ collect(-5:4), 
            ylims=(1e-5,1e4), xlims=(0,4), 
            grid=true, 
            size=(800,600))

    isdir(joinpath(@__DIR__,"plots")) ? nothing : mkdir(joinpath(@__DIR__,"plots"))
    for ext in ["pdf", "png"]
        savefig(p, joinpath(@__DIR__,"plots", "BiCM_computation_comparison ($(Dates.format(now(), "YYYY_mm_dd_HH_MM"))).$ext"))
    end
    p
end