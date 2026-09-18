# Render a robustness sweep into the markdown tables and the figure the paper and the validation
# notes quote, so those numbers stop being hand-typed prose.
#
# Usage:
#   julia --project=. robustness/report.jl [sweep.json] [ladder.json]
#
# With no arguments it picks the newest of each in robustness/results/.

using JSON, Printf, Statistics, Plots

const ACCURATE = 1e-6   # the threshold the published tables use

"JSON.jl writes a non-finite residual as `null`; an unsolved cell is the worst outcome, not a gap."
_res(x) = x === nothing ? Inf : float(x)

_newest(dir, prefix) = begin
    c = filter(f -> startswith(f, prefix) && endswith(f, ".json"), readdir(dir))
    isempty(c) && return nothing
    joinpath(dir, last(sort(c)))
end

accurate(rows) = count(r -> _res(r["residual"]) < ACCURATE, rows)

function _median(f, rows; init = NaN)
    v = [f(r) for r in rows]
    v = filter(x -> x !== nothing && !isnan(x) && isfinite(x), v)
    isempty(v) ? init : median(v)
end

"""
    sweep_tables(io, data)

Per-model tables of accuracy, iterations and wall time across the method x initial-guess grid,
split into well-posed and runaway subsets. A runaway constraint (`k = 0`, `k = N-1`, `s = k`) puts
the optimum at an infinite parameter, so no solver can settle there at a finite tolerance; mixing
those instances into the headline number would understate every method equally and hide which ones
degrade gracefully.
"""
function sweep_tables(io, data)
    rows = data["rows"]
    models = unique(r["model"] for r in rows)
    for model in models
        mrows = filter(r -> r["model"] == model, rows)
        println(io, "\n### ", model, "\n")
        println(io, "Corpus `", first(mrows)["corpus"], "`, ",
                length(unique(r["graph"] for r in mrows)), " graphs.\n")
        for (label, want) in (("Well posed", false), ("Runaway constraint", true))
            sub = filter(r -> r["runaway"] == want, mrows)
            isempty(sub) && continue
            n = length(unique(r["graph"] for r in sub))
            println(io, "**", label, "** (", n, " graphs)\n")
            println(io, "| method | initial | accurate | median residual | median iterations | median s |")
            println(io, "| --- | --- | --- | --- | --- | --- |")
            for method in unique(r["method"] for r in sub)
                for initial in unique(r["initial"] for r in sub if r["method"] == method)
                    cell = filter(r -> r["method"] == method && r["initial"] == initial, sub)
                    isempty(cell) && continue
                    mr = _median(r -> _res(r["residual"]), cell)
                    mi = _median(r -> r["iterations"] < 0 ? NaN : float(r["iterations"]), cell)
                    ms = _median(r -> float(r["seconds"]), cell)
                    @printf(io, "| `:%s` | `:%s` | %d/%d | %.1e | %s | %.4f |\n",
                            method, initial, accurate(cell), length(cell), mr,
                            isnan(mi) ? "-" : string(round(Int, mi)), ms)
                end
            end
            println(io)
        end
    end
end

"""
    guess_sensitivity(io, data)

The question the sweep exists to answer: for each model and method, how much does the answer
depend on where the solve started? A method whose best and worst initial guess differ is one where
the documented default is load-bearing.
"""
function guess_sensitivity(io, data)
    rows = filter(r -> r["runaway"] == false, data["rows"])
    println(io, "\n## Sensitivity to the initial guess\n")
    println(io, "Well-posed instances only. `spread` is the gap between the best and the worst ",
                "initial guess for that method, in percentage points.\n")
    println(io, "| model | method | best guess | worst guess | spread |")
    println(io, "| --- | --- | --- | --- | --- |")
    for model in unique(r["model"] for r in rows)
        for method in unique(r["method"] for r in rows if r["model"] == model)
            cells = Tuple{String,Float64}[]
            sub = filter(r -> r["model"] == model && r["method"] == method, rows)
            for initial in unique(r["initial"] for r in sub)
                c = filter(r -> r["initial"] == initial, sub)
                push!(cells, (initial, 100 * accurate(c) / length(c)))
            end
            isempty(cells) && continue
            sort!(cells, by = last, rev = true)
            best, worst = first(cells), last(cells)
            @printf(io, "| %s | `:%s` | `:%s` %.0f%% | `:%s` %.0f%% | %.0f |\n",
                    model, method, best[1], best[2], worst[1], worst[2], best[2] - worst[2])
        end
    end
end

"""
    ladder_table(io, data)

The BiCM Anderson-ladder comparison. `total it` is the cost of the whole corpus, which is the
reason the shipped ladder is preferred over always-Picard: same 183/183, less than half the work.
"""
function ladder_table(io, data)
    rows = data["rows"]
    ngraphs = data["meta"]["constructible"]
    println(io, "\n## BiCM Anderson ladder\n")
    println(io, "| strategy | accurate | maxiters | nonfinite | total iterations | worst residual |")
    println(io, "| --- | --- | --- | --- | --- | --- |")
    for s in unique(r["strategy"] for r in rows)
        sub = filter(r -> r["strategy"] == s, rows)
        acc = filter(r -> r["status"] == "ok" && _res(r["residual"]) < ACCURATE, sub)
        @printf(io, "| `%s` | %d/%d | %d | %d | %d | %.2e |\n", s, length(acc), ngraphs,
                count(r -> r["status"] == "maxiters", sub),
                count(r -> r["status"] == "nonfinite", sub),
                sum(r -> r["iterations"], acc; init = 0),
                isempty(acc) ? NaN : maximum(r -> _res(r["residual"]), acc))
    end
end

"""
    heatmap_figure(data, path)

Fraction of well-posed instances solved to better than `ACCURATE`, over the whole
model x method x initial-guess grid. Blank cells are guesses the model does not accept.
"""
function heatmap_figure(data, path)
    rows = filter(r -> r["runaway"] == false, data["rows"])
    isempty(rows) && return nothing
    guesses = sort(unique(r["initial"] for r in rows))
    labels = String[]
    M = Union{Float64,Missing}[]
    for model in unique(r["model"] for r in rows)
        for method in unique(r["method"] for r in rows if r["model"] == model)
            push!(labels, string(model, "  ", method))
            for gname in guesses
                c = filter(r -> r["model"] == model && r["method"] == method &&
                                r["initial"] == gname, rows)
                push!(M, isempty(c) ? missing : 100 * accurate(c) / length(c))
            end
        end
    end
    Z = permutedims(reshape(M, length(guesses), length(labels)))
    p = heatmap(guesses, labels, Z; c = :viridis, clims = (0, 100),
                xlabel = "initial guess", colorbar_title = "% solved to < 1e-6",
                size = (760, 40 + 22 * length(labels)), yflip = true,
                title = "Convergence by starting point (well-posed instances)",
                titlefontsize = 10, tickfontsize = 7, left_margin = 12Plots.mm,
                bottom_margin = 6Plots.mm)
    savefig(p, path)
    path
end

function main(args)
    dir = joinpath(@__DIR__, "results")
    sweepfile  = length(args) >= 1 ? args[1] : _newest(dir, "sweep_")
    ladderfile = length(args) >= 2 ? args[2] : _newest(dir, "ladder_")
    out = joinpath(dir, "robustness_report.md")

    open(out, "w") do io
        println(io, "# Solver robustness\n")
        println(io, "Generated by `performance/robustness/report.jl`. ",
                    "A cell counts as accurate when `constraint_residual(m) < ", ACCURATE, "`.\n")
        if sweepfile !== nothing
            data = JSON.parsefile(sweepfile)
            m = data["meta"]
            println(io, "Sweep: MaxEntropyGraphs v", m["package_version"], ", Julia ", m["julia"],
                        ", ", m["ngraphs"], " graphs per corpus, ", m["timestamp"], ".")
            guess_sensitivity(io, data)
            println(io, "\n## Full grid")
            sweep_tables(io, data)
            fig = heatmap_figure(data, joinpath(dir, "robustness_heatmap.pdf"))
            fig === nothing || println(io, "\nFigure: `", basename(fig), "`.")
        end
        if ladderfile !== nothing
            ladder_table(io, JSON.parsefile(ladderfile))
        end
    end
    @info "wrote" out
    println(read(out, String))
    out
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
