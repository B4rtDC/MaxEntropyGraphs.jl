#!/usr/bin/env julia

using Luxor

# Julia's established palette.
const JULIA_RED    = "#CB3C33"
const JULIA_BLUE   = "#4063D8"
const JULIA_GREEN  = "#389826"
const JULIA_PURPLE = "#9558B2"

"""
    draw_maxentropygraphs_logo(filename; size=1024, theme=:light,
                               background_color=nothing)

Draw a proposed logo for MaxEntropyGraphs.jl.

- `filename` may end in `.svg` or `.png`.
- `theme` is `:light` for charcoal scaffolding or `:dark` for an
  off-white scaffold intended for dark pages.
- `background_color=nothing` produces a transparent background.
  Pass a color string such as `"white"` or `"#101116"` for an opaque one.

The bold colored graph spells a compact M. The pale alternative edges
represent a maximum-entropy ensemble over a fixed node set; the outer ring
represents the constraints/normalization that bound that ensemble.
"""
function draw_maxentropygraphs_logo(
    filename::AbstractString;
    size::Integer = 1024,
    theme::Symbol = :light,
    background_color = nothing,
)
    size > 0 || throw(ArgumentError("size must be positive"))
    theme in (:light, :dark) ||
        throw(ArgumentError("theme must be :light or :dark"))

    ink = theme === :dark ? "#F2F2EF" : "#292B33"
    scale_factor = size / 512

    Drawing(size, size, filename)
    if isnothing(background_color)
        # Explicit alpha-zero paint works for both SVG and PNG output.
        background(0, 0, 0, 0)
    else
        background(background_color)
    end

    origin()
    setlinecap(:round)
    setlinejoin("round")

    # Coordinates are defined on a 512 x 512 design grid, then scaled.
    point(x, y) = Point((x - 256) * scale_factor, (y - 256) * scale_factor)
    nodes = [
        point(112, 356), # lower left
        point(156, 146), # upper left
        point(256, 340), # centre valley
        point(356, 146), # upper right
        point(400, 356), # lower right
    ]

    # Constraint / normalization boundary.
    setopacity(1.0)
    sethue(ink)
    setline(14 * scale_factor)
    circle(O, 205 * scale_factor, :stroke)

    # Alternative edges in the graph ensemble (all non-M edges of K5).
    ensemble_edges = [(1, 3), (1, 4), (1, 5), (2, 4), (2, 5), (3, 5)]
    setopacity(0.16)
    sethue(ink)
    setline(8 * scale_factor)
    for (i, j) in ensemble_edges
        line(nodes[i], nodes[j], :stroke)
    end

    # The M-shaped constrained/observed graph, with a dark or light keyline.
    m_edges = [(1, 2), (2, 3), (3, 4), (4, 5)]
    setopacity(1.0)
    sethue(ink)
    setline(31 * scale_factor)
    for (i, j) in m_edges
        line(nodes[i], nodes[j], :stroke)
    end

    # Julia-colored inner strokes.
    segment_colors = [JULIA_GREEN, JULIA_PURPLE, JULIA_BLUE, JULIA_RED]
    setline(13 * scale_factor)
    for ((i, j), color) in zip(m_edges, segment_colors)
        sethue(color)
        line(nodes[i], nodes[j], :stroke)
    end

    # Nodes sit on top of every edge.
    node_colors = [JULIA_GREEN, JULIA_PURPLE, JULIA_BLUE, JULIA_RED, JULIA_GREEN]
    node_radii = [27, 29, 29, 29, 27]
    for (p, color, radius) in zip(nodes, node_colors, node_radii)
        sethue(color)
        circle(p, radius * scale_factor, :fill)
        sethue(ink)
        setline(9 * scale_factor)
        circle(p, radius * scale_factor, :stroke)
    end

    finish()
    return filename
end

if abspath(PROGRAM_FILE) == @__FILE__
    outputs = [
        draw_maxentropygraphs_logo(
            "maxentropygraphs-logo-light.svg";
            size=512,
            theme=:light,
        ),
        draw_maxentropygraphs_logo(
            "maxentropygraphs-logo-light.png";
            size=1024,
            theme=:light,
        ),
        draw_maxentropygraphs_logo(
            "maxentropygraphs-logo-dark.svg";
            size=512,
            theme=:dark,
        ),
        draw_maxentropygraphs_logo(
            "maxentropygraphs-logo-dark.png";
            size=1024,
            theme=:dark,
        ),
    ]

    println("Wrote:")
    foreach(path -> println("  ", path), outputs)

    @info outputs
    # Move the generated logo-dark.png file to the `docs/assets` directory for inclusion in the documentation. 
    src = joinpath(@__DIR__, "maxentropygraphs-logo-dark.png")
    dest = joinpath(@__DIR__, "..", "docs", "src", "assets", "logo.jpeg")
    cp(src, dest; force=true)
    @info src "moved to" dest


end

