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

const DBiCM_positionmapper = Dict("small" => [1], "medium" => [2], "large" => [3])

"""
    find_latest_files(path, model, language)

Most recent benchmark file per scale, for a model and a language. `model` is matched as a
substring, so it must be anchored (`"_DBiCM_"`, not `"DBiCM"`) whenever one model's name contains
another's.
"""
function find_latest_files(path::String, model, language)
    entries = readdir(path)
    subfolders = findall(x -> occursin(language, x) && isdir(joinpath(path, x)), entries)
    length(subfolders) == 0 && error("No subfolder found for $language")
    subnames = entries[subfolders]
    subfolder = joinpath(path, subnames[argmax([stat(joinpath(path, s)).mtime for s in subnames])])
    benchmark_files = filter(x -> occursin(model, x), readdir(subfolder))
    length(benchmark_files) == 0 && error("No benchmark files found for $model")
    @info benchmark_files
    res = Dict()
    categories = unique([x[1] for x in splitext.([split(file, "_")[end] for file in benchmark_files])])
    for cat in categories
        catfiles = filter(x -> occursin(cat, x), benchmark_files)
        latest_file = catfiles[argmax([stat(joinpath(subfolder, file)).mtime for file in catfiles])]
        res[cat] = JSON.parsefile(joinpath(path, subfolder, latest_file))
    end
    return res
end

# The comparator is two NEMtropy BiCM solves, which is what a DBiCM decomposes into exactly. The
# Python file may be absent (Julia-only run), in which case only the Julia series is drawn.
ju_bench = find_latest_files(joinpath(@__DIR__, "benchmarks"), "_DBiCM_", "Julia")
py_bench = try
    find_latest_files(joinpath(@__DIR__, "benchmarks"), "_DBiCM_", "Python")
catch e
    @warn "No Python DBiCM benchmarks found; plotting the Julia series only." exception = e
    Dict()
end

scales = sort(collect(keys(ju_bench)), by = s -> DBiCM_positionmapper[s][1])
xtick_labels = [s for s in scales]
xtick_pos = [DBiCM_positionmapper[s][1] for s in scales]

py_series(scale, name) = begin
    ind = findfirst(x -> x["name"] == name, get(py_bench, scale, Dict("benchmarks" => []))["benchmarks"])
    ind === nothing ? nothing : Float64.(py_bench[scale]["benchmarks"][ind]["stats"]["data"])
end

"Julia timings live under the BenchmarkGroup entry as nanoseconds."
ju_group(scale, group, key) = begin
    bi = findfirst(x -> x["name"] == group, ju_bench[scale]["benchmarks"])
    bi === nothing && return nothing
    st = ju_bench[scale]["benchmarks"][bi]["stats"]
    haskey(st[2]["data"], key) || return nothing
    st[2]["data"][key][2]["times"] ./ 1e9
end

ju_single(scale, name) = begin
    bi = findfirst(x -> x["name"] == name, ju_bench[scale]["benchmarks"])
    bi === nothing ? nothing : ju_bench[scale]["benchmarks"][bi]["stats"][2]["times"] ./ 1e9
end

## 1. Creation times
begin
    trans = 0.95
    p = plot()
    for scale in scales
        t = py_series(scale, "test_create_DBiCM")
        t === nothing && continue
        scatter!(p, DBiCM_positionmapper[scale], [median(t)], label = "", color = LIB_REFERENCE,
                 alpha = trans, marker = :circle, markerstrokecolor = MARK_STROKE,
                 markerstrokewidth = MARK_STROKE_WIDTH, markersize = MARK_SIZE)
    end
    scatter!(p, [], [], label = "NEMtropy (two BiCM)", color = LIB_REFERENCE, marker = :circle,
             markerstrokecolor = MARK_STROKE, markerstrokewidth = MARK_STROKE_WIDTH, markersize = MARK_SIZE)
    for scale in scales
        t = ju_single(scale, "test_create_DBiCM")
        t === nothing && continue
        scatter!(p, DBiCM_positionmapper[scale], [median(t)], label = "", color = LIB_MEG,
                 alpha = trans, marker = :circle, markerstrokecolor = MARK_STROKE,
                 markerstrokewidth = MARK_STROKE_WIDTH, markersize = MARK_SIZE)
    end
    scatter!(p, [], [], label = "MaxEntropyGraphs (DBiCM)", color = LIB_MEG, marker = :circle,
             markerstrokecolor = MARK_STROKE, markerstrokewidth = MARK_STROKE_WIDTH, markersize = MARK_SIZE)
    plot!(p, yscale = :log10, xlabel = "Problem scale", ylabel = "Creation time [s]",
          legendposition = :topleft, legendfontsize = 12, tickfontsize = 14, labelfontsize = 18,
          xticks = (xtick_pos, xtick_labels), xlims = (0, length(scales) + 1), grid = true,
          size = (800, 600))
    isdir(joinpath(@__DIR__, "plots")) || mkdir(joinpath(@__DIR__, "plots"))
    for ext in ["pdf", "png"]
        savefig(p, joinpath(@__DIR__, "plots", "DBiCM_creation_comparison ($(Dates.format(now(), "YYYY_mm_dd_HH_MM"))).$ext"))
    end
    p
end

## 2. Computation times
begin
    trans = 0.95
    p = plot()
    for (key, label, approach) in [("test_solve_DBiCM[fixed-point-degrees]", "NEMtropy (fixed point, two BiCM)", "fixed point"),
                                   ("test_solve_DBiCM[quasinewton-degrees]", "NEMtropy (quasi-newton, two BiCM)", "quasi-newton"),
                                   ("test_solve_DBiCM[newton-degrees]",      "NEMtropy (newton, two BiCM)",      "newton")]
        for scale in scales
            t = py_series(scale, key)
            t === nothing && continue
            scatter!(p, mark_x(DBiCM_positionmapper[scale], LIB_REFERENCE, approach), [median(t)],
                     label = "", color = LIB_REFERENCE, alpha = trans, marker = METHOD_MARKERS[approach],
                     markerstrokecolor = MARK_STROKE, markerstrokewidth = MARK_STROKE_WIDTH,
                     markersize = mark_size(approach))
        end
        scatter!(p, [], [], label = label, color = LIB_REFERENCE, marker = METHOD_MARKERS[approach],
                 markerstrokecolor = MARK_STROKE, markerstrokewidth = MARK_STROKE_WIDTH,
                 markersize = mark_size(approach))
    end
    for (key, label, approach) in [("test_solve_DBiCM[two_bicm-FP]",         "MaxEntropyGraphs (fixed point)",  "fixed point"),
                                   ("test_solve_DBiCM[two_bicm-QN-BFGS-AG]", "MaxEntropyGraphs (quasi-newton)", "quasi-newton"),
                                   ("test_solve_DBiCM[two_bicm-Newton-ADF]", "MaxEntropyGraphs (newton)",       "newton")]
        for scale in scales
            t = ju_group(scale, "test_solve_DBiCM", key)
            t === nothing && continue
            scatter!(p, mark_x(DBiCM_positionmapper[scale], LIB_MEG, approach), [median(t)],
                     label = "", color = LIB_MEG, alpha = trans, marker = METHOD_MARKERS[approach],
                     markerstrokecolor = MARK_STROKE, markerstrokewidth = MARK_STROKE_WIDTH,
                     markersize = mark_size(approach))
        end
        scatter!(p, [], [], label = label, color = LIB_MEG, marker = METHOD_MARKERS[approach],
                 markerstrokecolor = MARK_STROKE, markerstrokewidth = MARK_STROKE_WIDTH,
                 markersize = mark_size(approach))
    end
    plot!(p, yscale = :log10, xlabel = "Problem scale", ylabel = "Computation time [s]",
          legendposition = :topleft, legendfontsize = 11, tickfontsize = 14, labelfontsize = 18,
          xticks = (xtick_pos, xtick_labels), xlims = (0, length(scales) + 1), grid = true,
          size = (800, 600))
    for ext in ["pdf", "png"]
        savefig(p, joinpath(@__DIR__, "plots", "DBiCM_computation_comparison ($(Dates.format(now(), "YYYY_mm_dd_HH_MM"))).$ext"))
    end
    p
end

@info "DBiCM plots written to $(joinpath(@__DIR__, "plots"))."
