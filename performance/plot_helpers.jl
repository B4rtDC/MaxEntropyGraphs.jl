##################################################################################
# plot_helpers.jl
#
# Shared helpers for the benchmark plotting scripts (<MODEL>_plots.jl).
##################################################################################

"""
    mirror_to_figures(p, filename)

Write the finished paper figure `p` to `../figures/filename`, which is where `paper.md`
reads it from.

The mirror only happens on a **full-scale** run. The plotting scripts always render from
the most recent benchmark results, so a `small`/`medium` sanity run would otherwise
silently overwrite a paper figure with a version that is missing its largest problem
(the x-axis still shows the category, with no data under it). The scale is read from the
same `BENCH_MAX_SCALE`/`BENCH_QUICK` pair the benchmark drivers use, so it reflects the
intent of the run that produced the results being plotted.

Set `BENCH_MIRROR_FIGURES=1` to force the mirror regardless of scale.
"""
function mirror_to_figures(p, filename::AbstractString)
    scale = get(ENV, "BENCH_QUICK", "0") == "1" ? "small" : lowercase(get(ENV, "BENCH_MAX_SCALE", "large"))
    forced = get(ENV, "BENCH_MIRROR_FIGURES", "0") == "1"
    if scale != "large" && !forced
        @info "$(filename): skipping the paper-figure mirror (BENCH_MAX_SCALE=$(scale) is not a full run; set BENCH_MIRROR_FIGURES=1 to override)"
        return nothing
    end
    figdir = joinpath(dirname(@__DIR__), "figures")
    if !isdir(figdir)
        @warn "$(figdir) does not exist: paper figure $(filename) was not written"
        return nothing
    end
    dest = joinpath(figdir, filename)
    savefig(p, dest)
    @info "paper figure written to $(dest)"
    return dest
end
