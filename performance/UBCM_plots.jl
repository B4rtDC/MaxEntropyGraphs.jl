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

const UBCM_positionmapper = Dict("small" => [1], "medium" => [2], "large" => [3])

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
        boxplot!(p, UBCM_positionmapper[scale], creation_times, label="", color=:blue, alpha=0.25, linecolor=:blue, outliers=false)
    end
    boxplot!(p,[],[], label="NEMtropy", color=:blue, alpha=0.25, linecolor=nothing)
    # Julia part
    for scale in keys(ju_bench)
        julia_index = findfirst(x -> x["name"] == "test_create_UBCM", ju_bench[scale]["benchmarks"])
        creation_times = Float64.(ju_bench[scale]["benchmarks"][julia_index]["stats"][2]["times"]) ./ 1e9
        boxplot!(p, UBCM_positionmapper[scale], creation_times, label="", color=:red, alpha=0.25, linecolor=:red, outliers=false)
    end
    boxplot!(p,[],[], label="MaxEntropyGraphs", color=:red, alpha=0.25, linecolor=nothing)

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
    trans = 0.7
    p = plot()

    # Define markers for different solution approaches
    approach_markers = Dict("fixed point" => :circle, "quasi-newton" => :square, "newton" => :star5)

    # Python part
    for (method, label, color, approach) in [ ("test_solve_UBCM[cm_exp-fixed-point-degrees]", "NEMtropy (fixed point)", :sienna4, "fixed point");
                                              ("test_solve_UBCM[cm_exp-quasinewton-degrees]", "NEMtropy (quasi-newton)", :peru, "quasi-newton");
                                              ("test_solve_UBCM[cm_exp-newton-degrees]", "NEMtropy (newton)", :sandybrown, "newton")]
        for scale in keys(py_bench)
            # find the benchmark containing the method
            ind = findfirst(x -> x["name"] == method, py_bench[scale]["benchmarks"])
            solution_times = Float64.(py_bench[scale]["benchmarks"][ind]["stats"]["data"])
            scatter!(p, UBCM_positionmapper[scale], [median(solution_times)], label="", 
            color=color, alpha=trans, marker=approach_markers[approach], outliers=false, linecolor=color, markerstrokecolor=color, markersize=10)
        end
        scatter!(p,[],[], label=label, color=color, alpha=trans, marker=approach_markers[approach], linecolor=nothing, markerstrokecolor=color) # Dummy scatter for legend
    end
    
    # Julia part
    for (method, label, color, approach) in [ ("test_solve_UBCM[cm_exp-FP]", "MaxEntropyGraphs (fixed point)", :navy, "fixed point");
                                              ("test_solve_UBCM[cm_exp-QN-BFGS-AG]", "MaxEntropyGraphs (quasi-newton)", :dodgerblue4, "quasi-newton");
                                              ("test_solve_UBCM[cm_exp-Newton-ADF]", "MaxEntropyGraphs (newton)", :dodgerblue, "newton")]
        for scale in keys(ju_bench)
            # find the relevant benchmarks
            benchind = findfirst(x -> x["name"] == "test_solve_UBCM", ju_bench[scale]["benchmarks"])
            if haskey(ju_bench[scale]["benchmarks"][benchind]["stats"][2]["data"], method)
                solution_times = ju_bench[scale]["benchmarks"][benchind]["stats"][2]["data"][method][2]["times"] ./ 1e9
                scatter!(p, UBCM_positionmapper[scale], [median(solution_times)], label="", 
                color=color, alpha=trans, marker=approach_markers[approach], outliers=false, linecolor=color, markerstrokecolor=color, markersize=10)
            end
        end
        scatter!(p,[],[], label=label, color=color, alpha=trans, marker=approach_markers[approach], linecolor=nothing, markerstrokecolor=color) # Dummy scatter for legend
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
    p
end


UBCM_solve_python = Dict(ob["name"] => ob for ob in py_bench[scale]["benchmarks"])
UBCM_solve_julia =  ju_bench[scale]["benchmarks"][findfirst(x-> x["name"] =="test_solve_UBCM", ju_bench[scale]["benchmarks"])]["stats"][2]["data"]

# Overall steps to consider:
# --------------------------
# 1. 


# Compare times
# ----------------------
# *Note*: python uses seconds, julia uses nanoseconds
UBCM_times_python = Dict{String, Vector{Float64}}()
UBCM_times_julia  = Dict{String, Vector{Float64}}()

# Creation times
python_index = findfirst(x -> x["name"] == "test_create_UBCM", UBCM_python_benchmarks["benchmarks"])
UBCM_times_python["test_create_UBCM"] = Float64.(UBCM_python_benchmarks["benchmarks"][python_index]["stats"]["data"])
julia_index = findfirst(x -> x["name"] == "test_create_UBCM", UBCM_julia_benchmarks["benchmarks"])
UBCM_times_julia["test_create_UBCM"]  = Float64.(UBCM_julia_benchmarks["benchmarks"][julia_index]["stats"][2]["times"]) ./ 1e9 # convert to seconds

# Compare solve times
# -------------------
# Start with common benchmark names
UBCM_solve_python = Dict(ob["name"] => ob for ob in UBCM_python_benchmarks["benchmarks"])
UBCM_solve_julia =  UBCM_julia_benchmarks["benchmarks"][findfirst(x-> x["name"] =="test_solve_UBCM", UBCM_julia_benchmarks["benchmarks"])]["stats"][2]["data"]
# All benchmark names
labels = collect(union(keys(UBCM_solve_python), keys(UBCM_solve_julia)))
# mapping for readability
labelmap = Dict("test_create_UBCM" => "Creation",
                "test_solve_UBCM[cm_exp-quasinewton-degrees]" => "Parameters\nquasi-Newton\n(degrees)",
                "test_solve_UBCM[cm-fixed-point-random]" => "Parameters\nfixed point\n(random)\nnon-exponential",
                "test_solve_UBCM[cm_exp-fixed-point-random]" => "Parameters\nfixed point\n(random)",
                "test_solve_UBCM[cm_exp-quasinewton-random]" => "Parameters\nquasi-Newton\n(random)",
                "test_solve_UBCM[cm_exp-newton-degrees]" => "Parameters\nNewton\n(degrees)",
                "test_solve_UBCM[cm_exp-newton-random]" => "Parameters\nNewton\n(random)",
                "test_solve_UBCM[cm-quasinewton-random]" => "Parameters\nquasi-Newton\n(random)\nnon-exponential",
                "test_sample_UBCM[1]" => "Sampling\n(n = 10,\n1 thread)",
                "test_solve_UBCM[cm_exp-fixed-point-degrees]" => "Parameters\nfixed point\n(degrees)",
                "test_solve_UBCM[cm-newton-random]" => "Parameters\nNewton\n(random)\nnon-exponential",
                "test_sample_UBCM[4]" => "Sampling\n(n = 10,\n4 threads)",
                )
clean_label_inds = sortperm([labelmap[l] for l in labels])
# Benchmarks in common
common_benchmark_names = intersect(keys(UBCM_solve_python), keys(UBCM_solve_julia))
for name in common_benchmark_names
    # python uses seconds, julia uses nanoseconds
    UBCM_times_python[name] = Float64.(UBCM_solve_python[name]["stats"]["data"])
    UBCM_times_julia[name]  = Float64.(UBCM_solve_julia[name][2]["times"]) ./ 1e9 # convert to seconds
end

# Benchmarks only existing in python
python_only_benchmark_names = setdiff(keys(UBCM_solve_python), keys(UBCM_times_python))
for name in python_only_benchmark_names
    UBCM_times_python[name] = Float64.(UBCM_solve_python[name]["stats"]["data"])
end
# Benchmarks only in julia
julia_only_benchmark_names = setdiff(keys(UBCM_solve_julia), keys(UBCM_times_julia))
for name in julia_only_benchmark_names
    UBCM_times_julia[name] = Float64.(UBCM_solve_julia[name][2]["times"]) ./ 1e9 # convert to seconds
end


## plot the results
# -----------------

# Initialize vectors to hold the combined data and group labels
combined_data = Float64[]
group_labels = String[]
positions = Float64[]
boxwidth = 0.5
# Populate the combined data and group labels
for (i, label) in enumerate(labels[clean_label_inds])
    @info "working on $label"
    if haskey(UBCM_times_python, label)
        @info "found $label in python"
        append!(combined_data, UBCM_times_python[label])
        append!(group_labels, fill("NEMtropy", length(UBCM_times_python[label])))
        append!(positions, fill(i, length(UBCM_times_python[label])))# .+ boxwidth/4)
    end
    if haskey(UBCM_times_julia, label)
        @info "found $label in julia"
        append!(combined_data, UBCM_times_julia[label])
        append!(group_labels, fill("MaxEntropyGraphs.jl", length(UBCM_times_julia[label])))
        append!(positions, fill(i, length(UBCM_times_julia[label])))# .- boxwidth/4)
    end
end


# Now we can plot the combined data with group labels
begin
    res = boxplot(positions, combined_data, group=group_labels, 
            layout=1, 
            legend=:bottomright, legendalpha=0.5,
            bar_width=boxwidth, outliers=false, yscale=:log10,
            xticks=(1:length(labels), [haskey(labelmap, label) ? labelmap[label] : "" for label in labels[clean_label_inds]]),
            xlims=(0,length(labels)+1),
            yticks=10.0 .^(-6:1:1),
            ylabel="Benchmark time for task [s]",
            xlabel="Task",
            xtickfontsize=6,
            fillalpha=0.5,
            linecolor=:match,
            title="UBCM performance (Zachary's Karate Club)",
            size=(900,400),
            left_margin = 5mm,
            )
    savefig(res, joinpath(@__DIR__,"plots", "$(res.subplots[1].attr[:title]) ($(Dates.format(now(), "YYYY_mm_dd_HH_MM"))).pdf"))
    annotate!(res, 1.75, 0.05, text("Note:\n\nNEMtropy uses the analytical Hessian\nMaxEntropyGraphs.jl computes the Hessian \nwith automated differentiation", 6,:left))
    savefig(res, joinpath(@__DIR__,"plots", "$(res.subplots[1].attr[:title])_comment ($(Dates.format(now(), "YYYY_mm_dd_HH_MM"))).pdf"))
end



begin
    combined_data = Float64[]
    group_labels = String[]
    positions = Float64[]
    
    # Julia Part
    begin
        # get all experiments that were run
        mynames = sort(collect(keys(UBCM_solve_julia)))
        # put the first name at the end
        push!(mynames, popfirst!(mynames))
        # loop over all names

        # group data for Analyical/ForwardDiff/ReverseDiff/AutoZygote
        for (i,name) in enumerate(mynames)
            # extract the part of the name that is between the brackets 
            # (e.g. test_solve_UBCM[cm_exp-quasinewton-degrees] -> cm_exp-quasinewton-degrees)
            shortname = match(r"\[cm_exp-(.*?)\]", name)[1]

            if occursin("ADF",shortname)
                group_label = "Julia, ForwardDiff" 
            elseif occursin("ADR",shortname)
                group_label = "Julia, ReverseDiff"
            elseif occursin("ADZ",shortname)
                group_label = "Julia, AutoZygote"
            else
                group_label = "Julia, Analytical (gradient)"
            end

            if occursin("Newton",shortname)
                position = 1
            elseif occursin("BFGS",shortname) && !occursin("LBFGS", shortname)
                position = 2
            elseif occursin("LBFGS",shortname)
                position = 3
            elseif occursin("FP",shortname)
                position = 4
            end


            append!(combined_data, Float64.(UBCM_solve_julia[name][2]["times"]) ./ 1e9)
            append!(group_labels, fill(group_label, length(UBCM_solve_julia[name][2]["times"])))
            append!(positions, fill(position, length(UBCM_solve_julia[name][2]["times"])))
        end
    end
    
    # Python Part
    begin
        # get all experiments that were run
        mynames = sort(collect(keys(UBCM_solve_python)))
        # put the first name at the end
        push!(mynames, popfirst!(mynames))
        @info mynames
        for name in mynames
            @info name
            if occursin("newton", name) && !occursin("quasinewton", name)
                position = 1
                @info "newton"
            elseif occursin("quasinewton", name)
                @info "wiener"
                position = 2.5
            elseif occursin("fixed-point", name)
                position = 4
                @info "fixed-point"
            end


            append!(combined_data, Float64.(UBCM_solve_python[name]["stats"]["data"]))
            append!(group_labels, fill("NEMtropy", length(UBCM_solve_python[name]["stats"]["data"])))
            append!(positions, fill(position, length(UBCM_solve_python[name]["stats"]["data"])))
        end
    end

    # plot part
    p = boxplot(positions, combined_data, group=group_labels, 
    layout=1, legend=:bottomleft, legendalpha=0.5,
    yscale=:log10, outliers=false, 
    xlims=(0,5),
    xticks=(1:4, ["Newton", "BFGS", "L-BFGS", "Fixed point"]), 
    yticks=10.0 .^(-4:1:2),
    ylims=(1e-4, 1e2),
    ylabel="Time [s]",
    linecolor=:match,fillalpha=0.5,
    xtickfontsize=10,legendfontsize=10)

    savefig(p,joinpath(@__DIR__,"plots", "UBCM_performance_julia_python_comparison ($(Dates.format(now(), "YYYY_mm_dd_HH_MM"))).pdf"))
    p
end