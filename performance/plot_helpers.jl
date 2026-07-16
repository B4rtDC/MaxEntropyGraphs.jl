##################################################################################
# plot_helpers.jl
#
# Shared styling and helpers for the benchmark plotting scripts (<MODEL>_plots.jl).
##################################################################################

##################################################################################
# Palette
#
# One encoding, used by every figure: COLOUR is the library, MARKER SHAPE is the
# solver method. Two hues, both at full strength.
#
# The earlier scheme split each library into three lightness steps *as well as*
# giving each method its own marker, which was redundant (the shape already said
# which method it was) and it pushed the third step out of the legible range:
# the lightest reference step (#f5c9a0) sat at OKLCH L=0.864 against a 0.43-0.77
# band, chroma 0.074 against a 0.10 floor, and 1.49:1 contrast on white. Worse,
# the OKLab dE between the second and third steps was 10.6 against a floor of 15,
# so even full-colour readers could not tell a reference quasi-newton marker from
# a reference newton one by colour. The two hues below score dE 24.7 under
# simulated protanopia/deuteranopia (target >= 8) and 33.6 unsimulated.
#
# The two panels of a paper figure previously disagreed with each other: the
# creation panel drew the reference library in blue and MaxEntropyGraphs in red,
# while the computation panel drew the reference in orange and MaxEntropyGraphs in
# blue. Both panels now read from these constants, so blue always means the same
# library.
##################################################################################

const LIB_REFERENCE = "#eb6834"   # NEMtropy / NuMeTriS, whichever comparator the model uses
const LIB_MEG       = "#2a78d6"   # MaxEntropyGraphs

# Marker shape carries the solver method, so it stays readable in greyscale and for
# any colour-vision deficiency.
const METHOD_MARKERS = Dict("fixed point"  => :circle,
                            "quasi-newton" => :square,
                            "newton"       => :star5)

# With one hue per library, two markers of the same library that land on a similar
# time would occlude each other, so each (library, method) gets its own slot inside
# the category: the library separates the two clusters, the method separates the
# marks within a cluster.
const METHOD_DODGE = Dict("fixed point"  => -0.09,
                          "quasi-newton" =>  0.00,
                          "newton"       =>  0.09)
const LIB_DODGE = Dict(LIB_REFERENCE => -0.16, LIB_MEG => 0.16)

"""
    mark_x(positions, color, approach)

Offset the categorical x `positions` so a (library, method) mark gets its own slot and
does not sit on top of its neighbours. `color` is one of [`LIB_REFERENCE`](@ref) or
[`LIB_MEG`](@ref); `approach` is a key of [`METHOD_MARKERS`](@ref).
"""
mark_x(positions, color, approach) = positions .+ get(LIB_DODGE, color, 0.0) .+ get(METHOD_DODGE, approach, 0.0)

# A surface-coloured ring keeps marks that still land close together from merging into
# one blob, and it reads on both the fills and the white page.
const MARK_STROKE = :white
const MARK_STROKE_WIDTH = 1
const MARK_SIZE = 8

# `markersize` is a radius, so shapes of equal size do not carry equal ink: a 5-pointed
# star covers roughly a third of the disc it is inscribed in, which made the newton marks
# read as subordinate to the fixed-point and quasi-newton ones rather than as a peer.
# Scale each shape back to a comparable visual weight.
const METHOD_SIZE = Dict("fixed point"  => 1.0,
                         "quasi-newton" => 0.9,   # a square already fills its bounding box
                         "newton"       => 1.5)

"""
    mark_size(approach)

Marker size for `approach`, corrected so the shapes carry comparable ink.
"""
mark_size(approach) = MARK_SIZE * get(METHOD_SIZE, approach, 1.0)

##################################################################################
# A third channel, for the panels that need one
#
# The BiCM projection panels cross two distributions with two thread counts, so once
# colour is the library and shape is the distribution there is nothing left to say which
# thread count a mark is. Fill state carries it: a solid mark is the multithreaded
# variant, a hollow one is the single-threaded variant. Fill survives greyscale and
# colour-vision deficiency, which a fourth hue would not.
##################################################################################

const THREAD_DODGE = Dict(true => 0.045, false => -0.045)

"""
    mark_fill(color, filled::Bool) -> (fillcolor, strokecolor)

Fill and stroke for a mark whose fill state carries a distinction. A `filled` mark is drawn
solid in `color` with the usual surface ring; a hollow one is drawn in the surface colour
and outlined in `color`, so it still reads as the same series.
"""
mark_fill(color, filled::Bool) = filled ? (color, MARK_STROKE) : (MARK_STROKE, color)

"""
    thread_x(positions, color, approach, threaded::Bool)

Like [`mark_x`](@ref), with an extra offset so the two thread variants of one series do not
land on top of each other.
"""
thread_x(positions, color, approach, threaded::Bool) = mark_x(positions, color, approach) .+ THREAD_DODGE[threaded]

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
