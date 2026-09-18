# Stage 3 of the campaign: turn the raw results into a report that can be read against the paper.
#
# The important product is the speed delta. Every benchmark file is timestamped and nothing is
# overwritten, so the previous campaign's results are still on disk next to this one: the summary
# pairs the two most recent files per (model, scale) and reports the ratio. That makes the paper
# edit a transcription rather than a judgement, since each number that has to change appears
# beside the number it replaces.

cd(@__DIR__)
using Pkg
Pkg.activate(".")
using JSON, Dates, Printf, Statistics

const BENCHPATH = joinpath(@__DIR__, "benchmarks")
const ACCPATH   = joinpath(@__DIR__, "accuracy")
const OUT       = joinpath(@__DIR__, "campaign_report.md")

"Julia result files are named `YYYY_mm_dd_HH_MM_<model>_<scale>.json`."
function julia_result_files()
    dirs = filter(d -> startswith(d, "Julia-") && isdir(joinpath(BENCHPATH, d)), readdir(BENCHPATH))
    files = Tuple{String,String,String}[]   # (problem, path, dir)
    for d in dirs, f in readdir(joinpath(BENCHPATH, d))
        endswith(f, ".json") || continue
        parts = split(splitext(f)[1], "_")
        length(parts) >= 7 || continue
        problem = join(parts[6:end], "_")   # drop the YYYY_mm_dd_HH_MM prefix
        push!(files, (problem, joinpath(BENCHPATH, d, f), d))
    end
    files
end

"Median seconds per benchmark key in one result file, flattening the BenchmarkGroup entries."
function medians(path)
    res = Dict{String,Float64}()
    data = try
        JSON.parsefile(path)
    catch
        return res
    end
    for b in get(data, "benchmarks", [])
        name = get(b, "name", "")
        st = get(b, "stats", nothing)
        st === nothing && continue
        if st isa Vector && length(st) >= 2 && st[2] isa Dict && haskey(st[2], "data") && st[2]["data"] isa Dict
            for (k, v) in st[2]["data"]
                try
                    res[k] = median(Float64.(v[2]["times"])) / 1e9
                catch
                end
            end
        elseif st isa Vector && length(st) >= 2 && st[2] isa Dict && haskey(st[2], "times")
            try
                res[name] = median(Float64.(st[2]["times"])) / 1e9
            catch
            end
        end
    end
    res
end

function speed_deltas(io)
    files = julia_result_files()
    problems = sort(unique(first.(files)))
    println(io, "\n## Speed, this campaign against the previous one\n")
    println(io, "Each problem's two most recent result files are paired. `ratio` below 1 means ",
                "this campaign is faster. A key present in only one of the two files is listed ",
                "as new or as gone, since that is usually a deliberate change in what is ",
                "benchmarked rather than a measurement.\n")
    for problem in problems
        group = sort(filter(f -> f[1] == problem, files), by = f -> mtime(f[2]), rev = true)
        length(group) >= 1 || continue
        println(io, "### ", problem, "\n")
        if length(group) == 1
            println(io, "Only one result file; no previous campaign to compare against.\n")
            cur = medians(group[1][2])
            isempty(cur) && (println(io, "No parsable timings.\n"); continue)
            println(io, "| benchmark | median s |")
            println(io, "| --- | --- |")
            for k in sort(collect(keys(cur)))
                @printf(io, "| `%s` | %.4g |\n", k, cur[k])
            end
            println(io)
            continue
        end
        cur, prev = medians(group[1][2]), medians(group[2][2])
        println(io, "Current `", basename(group[1][2]), "` against previous `",
                basename(group[2][2]), "`.\n")
        println(io, "| benchmark | previous s | current s | ratio |")
        println(io, "| --- | --- | --- | --- |")
        for k in sort(collect(union(keys(cur), keys(prev))))
            if haskey(cur, k) && haskey(prev, k)
                @printf(io, "| `%s` | %.4g | %.4g | %.3g |\n", k, prev[k], cur[k], cur[k] / prev[k])
            elseif haskey(cur, k)
                @printf(io, "| `%s` | - | %.4g | new |\n", k, cur[k])
            else
                @printf(io, "| `%s` | %.4g | - | gone |\n", k, prev[k])
            end
        end
        println(io)
    end
end

function accuracy_section(io)
    f = joinpath(ACCPATH, "accuracy_summary.json")
    isfile(f) || return
    d = JSON.parsefile(f)
    println(io, "\n## Accuracy\n")
    println(io, "Maximum absolute constraint violation, each implementation measured against its ",
                "own observed sequences.\n")
    println(io, "| problem | MaxEntropyGraphs | comparator |")
    println(io, "| --- | --- | --- |")
    for (name, e) in sort(collect(get(d, "models", Dict())), by = first)
        comp = get(e, "nemtropy_max_violation", get(e, "numetris_max_violation", nothing))
        @printf(io, "| %s | %.3g | %s |\n", name, get(e, "julia_max_violation", NaN),
                comp === nothing ? "-" : @sprintf("%.3g", comp))
    end
    println(io)
    for (name, e) in sort(collect(get(d, "models", Dict())), by = first)
        haskey(e, "dbicm_channel_agreement") || continue
        ag = e["dbicm_channel_agreement"]
        pretty = join([string(k, " ", v isa Number ? @sprintf("%.3g", v) : v)
                       for (k, v) in sort(collect(ag), by = first)], ", ")
        println(io, "DBiCM channel agreement with two independent NEMtropy BiCM fits: ",
                pretty, ".\n")
    end
end

open(OUT, "w") do io
    println(io, "# EXP-016 campaign report\n")
    println(io, "Generated ", now(), " by `performance/campaign_summary.jl`.\n")
    pf = joinpath(@__DIR__, "campaign_provenance.json")
    if isfile(pf)
        println(io, "## Provenance\n")
        for (k, v) in sort(collect(JSON.parsefile(pf)), by = first)
            println(io, "- **", k, "**: ", v)
        end
        println(io)
    end
    rr = joinpath(@__DIR__, "robustness", "results", "robustness_report.md")
    println(io, "## Solver robustness\n")
    println(io, isfile(rr) ? "See `robustness/results/robustness_report.md`." :
                             "Not produced in this run.")
    accuracy_section(io)
    speed_deltas(io)
end

@info "campaign report written" OUT
