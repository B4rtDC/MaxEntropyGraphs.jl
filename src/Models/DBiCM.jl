"""
    DBiCM

Maximum entropy model for the Directed Bipartite Configuration Model (DBiCM).

A bipartite network whose edges carry a direction has **two disjoint link channels** between the ⊥ and
the ⊤ layer, each an `N⊥ × N⊤` binary matrix:

  * channel `⁺` (`:to_top`, `⊥ → ⊤`): `B⁺[i,α] = 1` iff `⊥ᵢ → ⊤α`
  * channel `⁻` (`:to_bottom`, `⊤ → ⊥`): `B⁻[i,α] = 1` iff `⊤α → ⊥ᵢ`

Constraining all four degree sequences — the ⊥ layer's out- and in-degrees `(d⊥_out, d⊥_in)` and the
⊤ layer's `(d⊤_out, d⊤_in)` — gives the Hamiltonian

``H = Σᵢ (αᵢ k^{out}_i + βᵢ k^{in}_i) + Σ_α (γ_α h^{out}_α + δ_α h^{in}_α)``

which, collected per entry, reads `Σ_{i,α} (αᵢ + δ_α) B⁺_{iα} + Σ_{i,α} (βᵢ + γ_α) B⁻_{iα}`. **No term
couples the two channels**, so the partition function factorises over all `2·N⊥·N⊤` entries and

  * channel `⁺` is exactly a [`BiCM`](@ref) on the sequence pair `(d⊥_out, d⊤_in)`, and
  * channel `⁻` is exactly a [`BiCM`](@ref) on the sequence pair `(d⊥_in, d⊤_out)`,

statistically independent of one another. The connection probabilities are

``p⁺_{iα} = \\frac{x⊥^{out}_i x⊤^{in}_α}{1 + x⊥^{out}_i x⊤^{in}_α}``, ``\\qquad p⁻_{iα} = \\frac{x⊥^{in}_i x⊤^{out}_α}{1 + x⊥^{in}_i x⊤^{out}_α}``

with `x = exp(-θ)`. In particular `B⁺_{iα}` and `B⁻_{iα}` are independent, so the model carries **no
reciprocity coupling**: it is the directed bipartite analogue of the [`DBCM`](@ref), not of the
[`RBCM`](@ref). A model that constrains reciprocated bipartite degrees would be a different object.

# Class reduction

Each of the four sequences reduces to unique-value classes **independently**. This is forced, not
merely convenient: the [`DBCM`](@ref) needs joint `(out, in)` classes only because its sums carry the
self-exclusion `i ≠ j`, so a node's out-equation depends on its own class. Bipartite sums range over
two disjoint vertex sets and have no such term, and the cross-channel second derivatives of the
log-likelihood vanish identically, so the joint pair `(k^{out}_i, k^{in}_i)` never enters at all. Four
independent reductions are therefore never coarser than a `DBCM`-style joint reduction, and usually
strictly finer.

# Field naming

`<quantity>⟨layer⟩[ᵣ]_⟨direction⟩`, composing the [`BiCM`](@ref)'s `⊥`/`⊤` with the [`DBCM`](@ref)'s
`out`/`in`. All four exponentiated parameter vectors share the `x` prefix — with four fitnesses there
is no natural `x`/`y` dichotomy, and `x⊥ᵣ_out * x⊤ᵣ_in` reads unambiguously.

The parameter vector is laid out as two channel blocks, each in [`BiCM`](@ref) order (⊥ side, then ⊤
side): `θᵣ = [α(⊥out) ; δ(⊤in) | β(⊥in) ; γ(⊤out)]`.
"""
mutable struct DBiCM{T<:Union{Graphs.AbstractGraph, Nothing}, N<:Real} <: AbstractMaxEntropyModel
    "Graph type, any bipartite directed graph; converted to SimpleDiGraph for the computation"
    const G::T
    "Maximum likelihood parameters, [α(⊥out) ; δ(⊤in) | β(⊥in) ; γ(⊤out)]"
    const θᵣ::Vector{N}

    # exponentiated parameters, one entry per reduced class of the matching sequence
    "Exponentiated parameters of the ⊥ layer's out-degree classes ( x = exp(-α) ), channel ⁺"
    const x⊥ᵣ_out::Vector{N}
    "Exponentiated parameters of the ⊤ layer's in-degree classes ( x = exp(-δ) ), channel ⁺"
    const x⊤ᵣ_in::Vector{N}
    "Exponentiated parameters of the ⊥ layer's in-degree classes ( x = exp(-β) ), channel ⁻"
    const x⊥ᵣ_in::Vector{N}
    "Exponentiated parameters of the ⊤ layer's out-degree classes ( x = exp(-γ) ), channel ⁻"
    const x⊤ᵣ_out::Vector{N}

    # full sequences, indexed by position within the layer
    const d⊥_out::Vector{Int}
    const d⊥_in::Vector{Int}
    const d⊤_out::Vector{Int}
    const d⊤_in::Vector{Int}
    # reduced sequences: four INDEPENDENT reductions, each sorted ascending
    const d⊥ᵣ_out::Vector{Int}
    const d⊥ᵣ_in::Vector{Int}
    const d⊤ᵣ_out::Vector{Int}
    const d⊤ᵣ_in::Vector{Int}
    # live (non-zero) class ranges; a UnitRange suffices because the reductions are sorted, so a zero
    # class can only ever be first
    const d⊥ᵣ_out_nz::UnitRange{Int}
    const d⊥ᵣ_in_nz::UnitRange{Int}
    const d⊤ᵣ_out_nz::UnitRange{Int}
    const d⊤ᵣ_in_nz::UnitRange{Int}
    # class multiplicities
    const f⊥_out::Vector{Int}
    const f⊥_in::Vector{Int}
    const f⊤_out::Vector{Int}
    const f⊤_in::Vector{Int}
    # node index within its layer -> reduced class index
    const d⊥ᵣ_out_ind::Vector{Int}
    const d⊥ᵣ_in_ind::Vector{Int}
    const d⊤ᵣ_out_ind::Vector{Int}
    const d⊤ᵣ_in_ind::Vector{Int}

    # layer bookkeeping (identical semantics to the BiCM; membership from the UNDIRECTED skeleton)
    "Graph vertex ids of the ⊥ layer"
    const ⊥nodes::Vector{Int}
    "Graph vertex ids of the ⊤ layer"
    const ⊤nodes::Vector{Int}
    "Layer membership of each graph vertex (true = ⊥)"
    const is⊥::Vector{Bool}
    "Graph vertex id -> row index in the biadjacency matrices"
    const ⊥map::Dict{Int, Int}
    "Graph vertex id -> column index in the biadjacency matrices"
    const ⊤map::Dict{Int, Int}

    "Expected biadjacency matrix of channel ⁺ (⊥ → ⊤), N⊥ × N⊤" # not always computed/required
    Ĝ⁺::Union{Nothing, Matrix{N}}
    "Expected biadjacency matrix of channel ⁻ (⊤ → ⊥), N⊥ × N⊤" # not always computed/required
    Ĝ⁻::Union{Nothing, Matrix{N}}
    "Entry-wise standard deviations of channel ⁺" # not always computed/required
    σ⁺::Union{Nothing, Matrix{N}}
    "Entry-wise standard deviations of channel ⁻" # not always computed/required
    σ⁻::Union{Nothing, Matrix{N}}

    "Status indicators: parameters computed, expected matrices computed, variances computed, etc."
    const status::Dict{Symbol, Any}
    "Function used to compute the log-likelihood of the (reduced) model"
    fun::Union{Nothing, Function}
end

Base.show(io::IO, m::DBiCM{T,N}) where {T,N} = print(io, """DBiCM{$(T), $(N)} ($(m.status[:N⊥]) + $(m.status[:N⊤]) vertices, $(length(m.θᵣ)) parameters, $(@sprintf("%.2f", m.status[:cᵣ])) compression ratio)""")

"""Return the reduced number of parameters of the DBiCM network (both channels combined)"""
Base.length(m::DBiCM) = length(m.θᵣ)

"""
    precision(m::DBiCM)

Return the compute precision of the DBiCM model `m`.
"""
precision(m::DBiCM) = typeof(m).parameters[2]


"""
    DBiCM(G::T; d⊥_out, d⊥_in, d⊤_out, d⊤_in, precision::Type{<:AbstractFloat}=Float64, kwargs...)
    DBiCM(; d⊥_out::Vector, d⊥_in::Vector, d⊤_out::Vector, d⊤_in::Vector, precision=Float64, kwargs...)

Construct a `DBiCM` model from a directed bipartite graph `G`, or from the four degree sequences
directly (pass `G = nothing`, or use the keyword-only form).

The layer membership is read off the **undirected skeleton** of `G`. This matters: `Graphs.bipartite_map`
traverses `outneighbors` only, so on a directed graph it explores just the out-reachable set of its seed
and leaves every other vertex at the default colour — while `Graphs.is_bipartite` still returns `true`,
because an empty traversal encounters no conflict.

# Arguments
- `d⊥_out`, `d⊥_in`: out- and in-degree sequences of the ⊥ layer (defaults read from `G`).
- `d⊤_out`, `d⊤_in`: out- and in-degree sequences of the ⊤ layer (defaults read from `G`).
- `precision`: compute precision, defaults to `Float64`.

# Rejected input
- a graph containing a vertex with **no edges in either direction** — its layer is not determined by
  the data. A vertex that only sends, or only receives, is fine.
- degree sequences that no directed bipartite graph realises: `Σd⊥_out == Σd⊤_in` and
  `Σd⊤_out == Σd⊥_in` are two separate link counts, one per channel, and both must hold.
- a vertex whose degree saturates the live opposite side of its channel, whose fitness diverges.

# Examples
```jldoctest DBiCM_creation
julia> using Graphs

julia> G = SimpleDiGraph(8);

julia> for (a,b) in ((1,5),(1,6),(2,6),(2,7),(3,7),(4,8),(5,2),(6,3),(7,4),(8,1)); add_edge!(G, a, b); end

julia> model = DBiCM(G)
DBiCM{Graphs.SimpleGraphs.SimpleDiGraph{Int64}, Float64} (4 + 4 vertices, 6 parameters, 0.38 compression ratio)

```
"""
function DBiCM(G::T;    d⊥_out::Union{Nothing, Vector}=nothing,
                        d⊥_in::Union{Nothing, Vector}=nothing,
                        d⊤_out::Union{Nothing, Vector}=nothing,
                        d⊤_in::Union{Nothing, Vector}=nothing,
                        precision::Type{N}=Float64,
                        kwargs...) where {T,N<:AbstractFloat}
    T <: Union{Graphs.AbstractGraph, Nothing} ? nothing : throw(TypeError(:DBiCM, "G must be a subtype of AbstractGraph or Nothing", Union{Graphs.AbstractGraph, Nothing}, T))
    ⊥nodes, ⊤nodes, is⊥ = Int[], Int[], Bool[]

    if T <: Graphs.AbstractGraph
        Graphs.nv(G) == 0 ? throw(ArgumentError("The graph is empty")) : nothing
        Graphs.nv(G) == 1 ? throw(ArgumentError("The graph has only one vertex")) : nothing

        if !Graphs.is_directed(G)
            @warn "The graph is undirected, while the DBiCM model is directed; each edge is read as present in both directions, so the two channels will be identical"
        end
        if T <: SimpleWeightedGraphs.AbstractSimpleWeightedGraph
            @warn "The graph is weighted, while the DBiCM model is unweighted, the weight information will be lost"
        end

        # The bipartition comes from the UNDIRECTED skeleton — see the docstring. `Graphs.SimpleGraph`
        # of a digraph is exactly that skeleton (reciprocated pairs collapse to one edge).
        Gᵤ = Graphs.is_directed(G) ? Graphs.SimpleGraph(G) : G
        Graphs.is_bipartite(Gᵤ) ? nothing : throw(ArgumentError("The graph is not bipartite"))

        # An isolated vertex has no determinable layer: nothing in the data says which side of the
        # bipartition it belongs to, and `Graphs.bipartite_map` would silently place all of them in the
        # ⊥ layer, changing |⊥| and |⊤| and therefore the ensemble this model represents. Note that
        # "isolated" here means no edges in EITHER direction: a pure sender or pure receiver is
        # perfectly well placed, it merely has a dead channel on one side.
        isolated = findall(v -> iszero(Graphs.degree(Gᵤ, v)), Graphs.vertices(Gᵤ))
        if !isempty(isolated)
            throw(ArgumentError("""
            The graph has $(length(isolated)) isolated vertex/vertices $(length(isolated) > 6 ? string(first(isolated, 6), " …") : string(isolated)), whose layer membership is not determined by the data.

            A vertex with no edges in either direction belongs to neither side of the bipartition as far as the graph is concerned, and `Graphs.bipartite_map` would place all of them in the ⊥ layer, silently changing |⊥| and |⊤|.

            Either drop the isolated vertices from the graph, or state the partition yourself by building the model from the degree sequences, which may contain zeros:

                DBiCM(nothing; d⊥_out = ..., d⊥_in = ..., d⊤_out = ..., d⊤_in = ...)
            """))
        end

        membership = Graphs.bipartite_map(Gᵤ)
        ⊥nodes, ⊤nodes = findall(membership .== 1), findall(membership .== 2)
        is⊥ = membership .== 1

        d⊥_out = isnothing(d⊥_out) ? Graphs.outdegree(G, ⊥nodes) : d⊥_out
        d⊥_in  = isnothing(d⊥_in)  ? Graphs.indegree(G,  ⊥nodes) : d⊥_in
        d⊤_out = isnothing(d⊤_out) ? Graphs.outdegree(G, ⊤nodes) : d⊤_out
        d⊤_in  = isnothing(d⊤_in)  ? Graphs.indegree(G,  ⊤nodes) : d⊤_in

        Graphs.nv(G) != length(d⊥_out) + length(d⊤_out) ? throw(DimensionMismatch("The number of vertices in the graph ($(Graphs.nv(G))) and the lengths of the degree sequences do not match")) : nothing
    end

    # coherence checks on the sequences themselves
    any(isnothing, (d⊥_out, d⊥_in, d⊤_out, d⊤_in)) && throw(ArgumentError("All four degree sequences (d⊥_out, d⊥_in, d⊤_out, d⊤_in) must be provided when no graph is given"))
    length(d⊥_out) == length(d⊥_in) ? nothing : throw(DimensionMismatch("The ⊥ layer's out- and in-degree sequences must have the same length ($(length(d⊥_out)) vs $(length(d⊥_in)))"))
    length(d⊤_out) == length(d⊤_in) ? nothing : throw(DimensionMismatch("The ⊤ layer's out- and in-degree sequences must have the same length ($(length(d⊤_out)) vs $(length(d⊤_in)))"))
    iszero(length(d⊥_out)) ? throw(ArgumentError("The degree sequences of the ⊥ layer are empty")) : nothing
    iszero(length(d⊤_out)) ? throw(ArgumentError("The degree sequences of the ⊤ layer are empty")) : nothing
    length(d⊥_out) == 1 ? throw(ArgumentError("The ⊥ layer only contains a single node")) : nothing
    length(d⊤_out) == 1 ? throw(ArgumentError("The ⊤ layer only contains a single node")) : nothing
    all(≥(0), d⊥_out) && all(≥(0), d⊥_in) && all(≥(0), d⊤_out) && all(≥(0), d⊤_in) ? nothing : throw(DomainError("Degree sequences must be non-negative"))

    # Link conservation, ONE CONDITION PER CHANNEL. `Σd⊥_out` and `Σd⊤_in` are two ways of counting the
    # ⊥ → ⊤ links, and `Σd⊤_out`/`Σd⊥_in` the ⊤ → ⊥ links. A channel whose two counts disagree describes
    # no graph: its likelihood has no stationary point, because the ⊥ block of the gradient wants one
    # total while the ⊤ block wants another, and the solve can only run to the iteration cap.
    sum(d⊥_out) == sum(d⊤_in) ? nothing : throw(DomainError("The ⊥ → ⊤ channel is not realisable: the links counted from the ⊥ layer (Σd⊥_out = $(sum(d⊥_out))) and from the ⊤ layer (Σd⊤_in = $(sum(d⊤_in))) disagree, but both count the same set of edges, so they must be equal"))
    sum(d⊤_out) == sum(d⊥_in) ? nothing : throw(DomainError("The ⊤ → ⊥ channel is not realisable: the links counted from the ⊤ layer (Σd⊤_out = $(sum(d⊤_out))) and from the ⊥ layer (Σd⊥_in = $(sum(d⊥_in))) disagree, but both count the same set of edges, so they must be equal"))

    # Saturation ceiling, per channel, against the number of LIVE vertices opposite in THAT channel: a
    # vertex adjacent to every available counterpart forces `p = 1` for all of them, so its fitness has
    # no finite maximiser. A dead vertex opposite can never be connected to, so it does not raise the bound.
    # A degree of zero never saturates anything, so the test only applies once the channel carries a
    # link at all — otherwise an entirely empty channel (a purely one-directional network, which is a
    # perfectly ordinary input) would be rejected by `0 >= 0`.
    _ceiling(d, opp, dname, oppname) = (!iszero(maximum(d)) && maximum(d) >= count(!iszero, opp)) ? throw(DomainError("The maximum of $(dname) ($(maximum(d))) is greater than or equal to the number of vertices with a live $(oppname) channel ($(count(!iszero, opp))), this is not allowed: such a vertex must connect to every available counterpart, so its fitness diverges")) : nothing
    _ceiling(d⊥_out, d⊤_in,  "d⊥_out", "d⊤_in")
    _ceiling(d⊤_in,  d⊥_out, "d⊤_in",  "d⊥_out")
    _ceiling(d⊥_in,  d⊤_out, "d⊥_in",  "d⊤_out")
    _ceiling(d⊤_out, d⊥_in,  "d⊤_out", "d⊥_in")

    if isnothing(G)
        ⊥nodes = collect(1:length(d⊥_out))
        ⊤nodes = collect(length(d⊥_out)+1:length(d⊥_out)+length(d⊤_out))
        is⊥ = vcat(ones(Bool, length(d⊥_out)), zeros(Bool, length(d⊤_out)))
    end

    # four INDEPENDENT reductions (see the type docstring for why joint (out,in) classes are not needed)
    _nz(dᵣ) = iszero(first(dᵣ)) ? (2:length(dᵣ)) : (1:length(dᵣ))
    d⊥ᵣ_out, _, d⊥ᵣ_out_ind, f⊥_out = np_unique_clone(d⊥_out, sorted=true)
    d⊥ᵣ_in,  _, d⊥ᵣ_in_ind,  f⊥_in  = np_unique_clone(d⊥_in,  sorted=true)
    d⊤ᵣ_out, _, d⊤ᵣ_out_ind, f⊤_out = np_unique_clone(d⊤_out, sorted=true)
    d⊤ᵣ_in,  _, d⊤ᵣ_in_ind,  f⊤_in  = np_unique_clone(d⊤_in,  sorted=true)

    n⁺ = length(d⊥ᵣ_out) + length(d⊤ᵣ_in)     # channel ⁺ block width, and the split point of θᵣ
    n⁻ = length(d⊥ᵣ_in)  + length(d⊤ᵣ_out)
    θᵣ = Vector{precision}(undef, n⁺ + n⁻)

    status = Dict{Symbol, Any}(:params_computed => false,
                               :out_params_computed => false,   # channel ⁺ (:to_top)
                               :in_params_computed => false,    # channel ⁻ (:to_bottom)
                               :G_computed => false,
                               :σ_computed => false,
                               :N⊥ => length(d⊥_out),
                               :N⊤ => length(d⊤_out),
                               :N => length(d⊥_out) + length(d⊤_out),
                               :n⁺ => n⁺,
                               :E⁺ => sum(d⊥_out),
                               :E⁻ => sum(d⊤_out),
                               :d⊥_out_unique => length(d⊥ᵣ_out),
                               :d⊥_in_unique  => length(d⊥ᵣ_in),
                               :d⊤_out_unique => length(d⊤ᵣ_out),
                               :d⊤_in_unique  => length(d⊤ᵣ_in),
                               :cᵣ => (n⁺ + n⁻) / (2 * (length(d⊥_out) + length(d⊤_out))))

    return DBiCM{T,precision}(G, θᵣ,
        Vector{precision}(undef, length(d⊥ᵣ_out)), Vector{precision}(undef, length(d⊤ᵣ_in)),
        Vector{precision}(undef, length(d⊥ᵣ_in)),  Vector{precision}(undef, length(d⊤ᵣ_out)),
        d⊥_out, d⊥_in, d⊤_out, d⊤_in,
        d⊥ᵣ_out, d⊥ᵣ_in, d⊤ᵣ_out, d⊤ᵣ_in,
        _nz(d⊥ᵣ_out), _nz(d⊥ᵣ_in), _nz(d⊤ᵣ_out), _nz(d⊤ᵣ_in),
        f⊥_out, f⊥_in, f⊤_out, f⊤_in,
        d⊥ᵣ_out_ind, d⊥ᵣ_in_ind, d⊤ᵣ_out_ind, d⊤ᵣ_in_ind,
        ⊥nodes, ⊤nodes, is⊥,
        Dict(node => i for (i,node) in enumerate(⊥nodes)),
        Dict(node => i for (i,node) in enumerate(⊤nodes)),
        nothing, nothing, nothing, nothing, status, nothing)
end

DBiCM(; d⊥_out::Vector{T}, d⊥_in::Vector{T}, d⊤_out::Vector{T}, d⊤_in::Vector{T}, precision::Type{N}=Float64, kwargs...) where {T<:Signed, N<:AbstractFloat} =
    DBiCM(nothing; d⊥_out=d⊥_out, d⊥_in=d⊥_in, d⊤_out=d⊤_out, d⊤_in=d⊤_in, precision=precision, kwargs...)


##############################################################################################
# Channel selection
##############################################################################################

"""
    _channel_id(channel::Symbol)

Normalise a `channel` selector to `:plus` (⊥ → ⊤) or `:minus` (⊤ → ⊥).

`channel` is **absolute** — it names the direction links travel — while the `kind` selector of the
projection functions is **relative to the projected layer**. Keeping the two vocabularies disjoint is
deliberate: under `layer = :top`, a layer-relative `:out` means "both ⊤ nodes send", which is channel
`⁻`, so a shared `:out`/`:in` spelling would have the same symbol select opposite matrices in the two
APIs. `:out`/`:in` are therefore rejected here with a pointer rather than silently accepted.
"""
function _channel_id(channel::Symbol)
    (channel === :to_top    || channel === :⁺ || channel === :plus)  && return :plus
    (channel === :to_bottom || channel === :⁻ || channel === :minus) && return :minus
    (channel === :out || channel === :in) && throw(ArgumentError("`channel` must be :to_top (⁺, ⊥ → ⊤) or :to_bottom (⁻, ⊤ → ⊥). `:out`/`:in` name V-motif kinds relative to the projected layer, not a direction of travel — see `V_motifs`."))
    throw(ArgumentError("Unknown channel $(channel); use :to_top (⁺, ⊥ → ⊤) or :to_bottom (⁻, ⊤ → ⊥)"))
end

"""
    _channel(m::DBiCM, channel::Symbol)

Everything the `BiCM` kernels need to evaluate one channel of `m`, as a `NamedTuple`: the `θᵣ` index
range of that channel's block, its own/opposite reduced sequences, multiplicities, live ranges, class
maps and exponentiated parameters, and its link count.

`own` is always the ⊥ side and `opp` the ⊤ side, so both channels present themselves to the `BiCM`
kernels in the same order and no kernel ever has to know which channel it is looking at.
"""
function _channel(m::DBiCM, channel::Symbol)
    n⁺ = m.status[:n⁺]::Int
    if _channel_id(channel) === :plus
        return (block = 1:n⁺,
                d_own = m.d⊥ᵣ_out, d_opp = m.d⊤ᵣ_in,
                f_own = m.f⊥_out,  f_opp = m.f⊤_in,
                nz_own = m.d⊥ᵣ_out_nz, nz_opp = m.d⊤ᵣ_in_nz,
                n_own = length(m.d⊥ᵣ_out),
                x_own = m.x⊥ᵣ_out, x_opp = m.x⊤ᵣ_in,
                ind_own = m.d⊥ᵣ_out_ind, ind_opp = m.d⊤ᵣ_in_ind,
                E = m.status[:E⁺]::Int, key = :out_params_computed, name = "⁺ (⊥ → ⊤)")
    else
        return (block = (n⁺ + 1):length(m.θᵣ),
                d_own = m.d⊥ᵣ_in, d_opp = m.d⊤ᵣ_out,
                f_own = m.f⊥_in,  f_opp = m.f⊤_out,
                nz_own = m.d⊥ᵣ_in_nz, nz_opp = m.d⊤ᵣ_out_nz,
                n_own = length(m.d⊥ᵣ_in),
                x_own = m.x⊥ᵣ_in, x_opp = m.x⊤ᵣ_out,
                ind_own = m.d⊥ᵣ_in_ind, ind_opp = m.d⊤ᵣ_out_ind,
                E = m.status[:E⁻]::Int, key = :in_params_computed, name = "⁻ (⊤ → ⊥)")
    end
end

##############################################################################################
# Likelihood, gradient and fixed-point map
#
# The DBiCM introduces NO new mathematics: its log-likelihood is `L_BiCM_reduced` evaluated on two
# disjoint blocks of θ and summed. Every function below is a dispatcher over the two channels.
#
# Each kernel is handed a view of its OWN block, never the whole vector: `L_BiCM_reduced` slices its
# second parameter group as `θ[n⊥ᵣ+1:end]`, so passing the full 4-block vector with an offset would
# silently make the ⁺ channel's ⊤ block swallow the entire ⁻ channel.
##############################################################################################

"""
    L_DBiCM_reduced(θ::AbstractVector, m::DBiCM)
    L_DBiCM_reduced(m::DBiCM)

Log-likelihood of the reduced DBiCM model, the sum of the two channels' `BiCM` log-likelihoods.
"""
function L_DBiCM_reduced(θ::AbstractVector, m::DBiCM)
    res = zero(eltype(θ))
    for ch in (:to_top, :to_bottom)
        c = _channel(m, ch)
        res += L_BiCM_reduced(view(θ, c.block), c.d_own, c.d_opp, c.f_own, c.f_opp, c.nz_own, c.nz_opp, c.n_own)
    end
    return res
end

L_DBiCM_reduced(m::DBiCM) = L_DBiCM_reduced(m.θᵣ, m)

"""
    ∇L_DBiCM_reduced!(∇L, θ, m::DBiCM, x⊥_out, x⊤_in, x⊥_in, x⊤_out)

Gradient of the reduced DBiCM log-likelihood, computed in place. The four trailing arguments are
scratch buffers for the exponentiated parameters, one per reduced sequence.
"""
function ∇L_DBiCM_reduced!(∇L::AbstractVector, θ::AbstractVector, m::DBiCM,
                           x⊥_out::AbstractVector, x⊤_in::AbstractVector,
                           x⊥_in::AbstractVector,  x⊤_out::AbstractVector)
    for (ch, xo, xp) in ((:to_top, x⊥_out, x⊤_in), (:to_bottom, x⊥_in, x⊤_out))
        c = _channel(m, ch)
        ∇L_BiCM_reduced!(view(∇L, c.block), view(θ, c.block), c.d_own, c.d_opp, c.f_own, c.f_opp,
                         c.nz_own, c.nz_opp, xo, xp, c.n_own)
    end
    return ∇L
end

"""
    ∇L_DBiCM_reduced_minus!(∇L, θ, m::DBiCM, x⊥_out, x⊤_in, x⊥_in, x⊤_out)

As [`∇L_DBiCM_reduced!`](@ref), returning the negative gradient (for minimisation).
"""
function ∇L_DBiCM_reduced_minus!(∇L::AbstractVector, θ::AbstractVector, m::DBiCM,
                                 x⊥_out::AbstractVector, x⊤_in::AbstractVector,
                                 x⊥_in::AbstractVector,  x⊤_out::AbstractVector)
    for (ch, xo, xp) in ((:to_top, x⊥_out, x⊤_in), (:to_bottom, x⊥_in, x⊤_out))
        c = _channel(m, ch)
        ∇L_BiCM_reduced_minus!(view(∇L, c.block), view(θ, c.block), c.d_own, c.d_opp, c.f_own, c.f_opp,
                               c.nz_own, c.nz_opp, xo, xp, c.n_own)
    end
    return ∇L
end

"""
    DBiCM_reduced_iter!(θ, m::DBiCM, x⊥_out, x⊤_in, x⊥_in, x⊤_out, G)

One fixed-point iteration of the reduced DBiCM model, written into the buffer `G`, which is returned.
Both channels are advanced independently.
"""
function DBiCM_reduced_iter!(θ::AbstractVector, m::DBiCM,
                             x⊥_out::AbstractVector, x⊤_in::AbstractVector,
                             x⊥_in::AbstractVector,  x⊤_out::AbstractVector,
                             G::AbstractVector)
    for (ch, xo, xp) in ((:to_top, x⊥_out, x⊤_in), (:to_bottom, x⊥_in, x⊤_out))
        c = _channel(m, ch)
        BiCM_reduced_iter!(view(θ, c.block), c.d_own, c.d_opp, c.f_own, c.f_opp, c.nz_own, c.nz_opp,
                           xo, xp, view(G, c.block), c.n_own)
    end
    return G
end

"""
    initial_guess(m::DBiCM; method::Symbol=:degrees)

Compute an initial guess for the parameters of the DBiCM model `m`.

Supported methods: `:degrees` (default), `:random`, `:uniform`, `:chung_lu`. Unlike the [`BiCM`](@ref)'s,
`:chung_lu` does not require the model to hold a graph — each channel's link count is already in
`m.status`.
"""
function initial_guess(m::DBiCM; method::Symbol=:degrees)
    N = precision(m)
    seqs = (m.d⊥ᵣ_out, m.d⊤ᵣ_in, m.d⊥ᵣ_in, m.d⊤ᵣ_out)
    if method == :degrees
        return vcat(map(d -> -log.(N.(d)), seqs)...)
    elseif method == :random
        return -log.(rand(N, length(m.θᵣ)))
    elseif method == :uniform
        return -log.(N(0.5) .* ones(N, length(m.θᵣ)))
    elseif method == :chung_lu
        E⁺, E⁻ = m.status[:E⁺]::Int, m.status[:E⁻]::Int
        scale = (max(E⁺, 1), max(E⁺, 1), max(E⁻, 1), max(E⁻, 1))
        return vcat(map((d, E) -> -log.(N.(d) ./ sqrt(N(E))), seqs, scale)...)
    else
        throw(ArgumentError("The initial guess method $(method) is not supported"))
    end
end


##############################################################################################
# Solver
##############################################################################################

"""
    _solve_dbicm_channel(m::DBiCM, ch::Symbol, θ_ch, method, ...)

Solve one channel of `m`, returning `(θ, sol)`. The channel is an ordinary `BiCM` problem, so this is
the `BiCM` solve recipe verbatim — including the `Newton`/Zygote second-order workaround and the
"bind the model fields to locals" precaution — applied to that channel's own sequences.
"""
function _solve_dbicm_channel(m::DBiCM, ch::Symbol, θ_ch::Vector{N}, method::Symbol,
                              maxiters::Int, verbose::Bool, ftol::Real,
                              abstol, reltol, g_tol, AD_method::Symbol,
                              analytical_gradient::Bool) where {N}
    c = _channel(m, ch)
    # Bind to locals so the differentiated closure captures plain values rather than the model (whose
    # `status` Dict access breaks Zygote's `dict_getindex` pullback — the defect that once affected only
    # the BiCM AD-gradient solve).
    d_own, d_opp, f_own, f_opp = c.d_own, c.d_opp, c.f_own, c.f_opp
    nz_own, nz_opp, n_own = c.nz_own, c.nz_opp, c.n_own
    x_buffer = zeros(N, length(d_own))
    y_buffer = zeros(N, length(d_opp))

    if method == :fixedpoint
        G_buffer = zeros(N, length(θ_ch))
        FP_model! = (θ::Vector) -> BiCM_reduced_iter!(θ, d_own, d_opp, f_own, f_opp, nz_own, nz_opp,
                                                      x_buffer, y_buffer, G_buffer, n_own)
        # Each channel carries exactly ONE gauge direction, which is the regime the ladder is measured
        # on. Solving both channels jointly would make the accelerator's least-squares rank-deficient
        # by two — an untested regime, and the reason this model does not do that.
        sol = _anderson_memory_ladder(FP_model!, θ_ch; ftol=ftol, maxiters=maxiters, verbose=verbose)
        NLsolve.converged(sol) || throw(ConvergenceError(method, nothing, c.name))
        verbose && @info "Fixed point iteration converged after $(sol.iterations) iterations (channel $(c.name))"
        return sol.zero, sol
    else
        method ∈ keys(optimization_methods) || throw(ArgumentError("The method $(method) is not supported (yet)"))
        AD_method ∈ keys(AD_methods) || throw(ArgumentError("The AD method $(AD_method) is not supported (yet)"))
        grad! = (G, θ, p) -> ∇L_BiCM_reduced_minus!(G, θ, d_own, d_opp, f_own, f_opp, nz_own, nz_opp,
                                                    x_buffer, y_buffer, n_own)
        # `Newton` needs second derivatives; with Zygote as the inner backend the nested HVP path aborts
        # the process once Symbolics is loaded, so second-order methods fall back to ForwardDiff.
        AD_for_method = (method === :Newton && AD_method === :AutoZygote) ? :AutoForwardDiff : AD_method
        f = Optimization.OptimizationFunction((θ, p) -> -L_BiCM_reduced(θ, d_own, d_opp, f_own, f_opp, nz_own, nz_opp, n_own),
                                              AD_methods[AD_for_method],
                                              grad = analytical_gradient ? grad! : nothing)
        prob = Optimization.OptimizationProblem(f, θ_ch)
        solve_kwargs = isnothing(g_tol) ? (; maxiters = maxiters, abstol = abstol, reltol = reltol) :
                                          (; maxiters = maxiters, abstol = abstol, reltol = reltol, g_abstol = g_tol)
        sol = Optimization.solve(prob, optimization_methods[method]; solve_kwargs...)
        Optimization.SciMLBase.successful_retcode(sol.retcode) || throw(ConvergenceError(method, sol.retcode, c.name))
        verbose && @info """$(method) optimisation converged after $(@sprintf("%1.2e", sol.stats.time)) seconds for channel $(c.name)"""
        return sol.u, sol
    end
end

"""
    solve_model!(m::DBiCM; kwargs...)

Compute the maximum-likelihood parameters of the DBiCM model `m`.

The two channels are **solved separately**. They share no parameter and their cross derivatives vanish
identically, so this is the exact decomposition rather than an approximation — and it keeps each solve
at one gauge direction, the regime the fixed-point accelerator is measured on, at a Newton cost of
`n⁺³ + n⁻³` instead of `(n⁺+n⁻)³`.

A channel with no links at all (a purely one-directional network, which is a perfectly ordinary input)
is not solved: its optimum is `θ ≡ Inf`, `p ≡ 0` exactly, and running the fixed-point map on it would
evaluate `-log(0/0)`.

# Arguments
- `method`: `:fixedpoint` (default), `:BFGS`, `:LBFGS` or `:Newton`.
- `initial`: `:degrees` (default), `:random`, `:uniform` or `:chung_lu`.
- `maxiters`, `verbose`, `ftol`, `abstol`, `reltol`, `g_tol`, `AD_method`, `analytical_gradient`: as
  for the other models.

Returns `(m, sol)`, where `sol` is a `NamedTuple` `(out = ..., in = ...)` holding each channel's
solution object, or `nothing` for a channel that carried no links.

# Examples
```jldoctest DBiCM_solve
julia> using Graphs

julia> G = SimpleDiGraph(8);

julia> for (a,b) in ((1,5),(1,6),(2,6),(2,7),(3,7),(4,8),(5,2),(6,3),(7,4),(8,1)); add_edge!(G, a, b); end

julia> model = DBiCM(G);

julia> solve_model!(model);

julia> model.status[:params_computed]
true

```
"""
function solve_model!(m::DBiCM;  method::Symbol=:fixedpoint,
                                 initial::Symbol=:degrees,
                                 maxiters::Int=1000,
                                 verbose::Bool=false,
                                 ftol::Union{Real, Nothing}=nothing,
                                 abstol::Union{Number, Nothing}=nothing,
                                 reltol::Union{Number, Nothing}=nothing,
                                 g_tol::Union{Number, Nothing}=nothing,
                                 AD_method::Symbol=:AutoZygote,
                                 analytical_gradient::Bool=true)
    N = precision(m)
    N <: Union{Float16, Float32} && @warn "Solving in $(N) precision is experimental and may not converge; low precision is intended for storage. Consider Float64 for the solve." maxlog=1
    method ≠ :fixedpoint && !isnothing(ftol) && @warn _ftol_unused_msg(method) maxlog=1
    ftol = isnothing(ftol) ? _DEFAULT_FTOL : ftol
    method ∈ [:fixedpoint; collect(keys(optimization_methods))] || throw(ArgumentError("The method $(method) is not supported (yet)"))

    θ₀ = initial_guess(m, method=initial)
    sols = Dict{Symbol, Any}(:out => nothing, :in => nothing)

    for (ch, key) in ((:to_top, :out), (:to_bottom, :in))
        c = _channel(m, ch)
        if iszero(c.E)
            # No link travels this way, so every class in the block is dead and the optimum is exactly
            # `θ = Inf` (`p ≡ 0`). Solving would divide zero by zero on the first iteration.
            m.θᵣ[c.block] .= N(Inf)
            m.status[c.key] = true
            verbose && @info "Channel $(c.name) carries no links; its parameters are set to their exact optimum without iterating"
            continue
        end
        θ_ch = collect(N, @view θ₀[c.block])
        # Dead channels come from the DATA, not from the initial guess: reading them off `isinf(θ₀)`
        # makes `:uniform`/`:random` guesses produce silently wrong fits.
        ind_inf = vcat(findall(iszero, c.d_own), c.n_own .+ findall(iszero, c.d_opp))
        θ_ch[ind_inf] .= zero(N)
        θ_sol, sol = _solve_dbicm_channel(m, ch, θ_ch, method, maxiters, verbose, ftol,
                                          abstol, reltol, g_tol, AD_method, analytical_gradient)
        θ_sol[ind_inf] .= N(Inf)
        m.θᵣ[c.block] .= θ_sol
        m.status[c.key] = true
        sols[key] = sol
    end

    m.status[:params_computed] = m.status[:out_params_computed] && m.status[:in_params_computed]
    set_xᵣ!(m)
    return m, (out = sols[:out], in = sols[:in])
end

"""
    set_xᵣ!(m::DBiCM)

Set the exponentiated maximum-likelihood parameters of all four reduced sequences of `m`.
"""
function set_xᵣ!(m::DBiCM)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    n⁺ = m.status[:n⁺]::Int
    n1 = length(m.d⊥ᵣ_out); n2 = length(m.d⊤ᵣ_in); n3 = length(m.d⊥ᵣ_in)
    m.x⊥ᵣ_out .= exp.(.-(@view m.θᵣ[1:n1]))
    m.x⊤ᵣ_in  .= exp.(.-(@view m.θᵣ[n1+1:n⁺]))
    m.x⊥ᵣ_in  .= exp.(.-(@view m.θᵣ[n⁺+1:n⁺+n3]))
    m.x⊤ᵣ_out .= exp.(.-(@view m.θᵣ[n⁺+n3+1:end]))
    return m
end


##############################################################################################
# Expected biadjacency matrices, variances and derived quantities
##############################################################################################

"""
    p⁺(m::DBiCM, i::Int, α::Int)

Probability of the link `⊥ᵢ → ⊤α` under `m`, with **layer-local** indices and no bounds checking.
"""
@inline p⁺(m::DBiCM, i::Int, α::Int) = f_BiCM(m.x⊥ᵣ_out[m.d⊥ᵣ_out_ind[i]] * m.x⊤ᵣ_in[m.d⊤ᵣ_in_ind[α]])

"""
    p⁻(m::DBiCM, i::Int, α::Int)

Probability of the link `⊤α → ⊥ᵢ` under `m`, with **layer-local** indices and no bounds checking.
"""
@inline p⁻(m::DBiCM, i::Int, α::Int) = f_BiCM(m.x⊥ᵣ_in[m.d⊥ᵣ_in_ind[i]] * m.x⊤ᵣ_out[m.d⊤ᵣ_out_ind[α]])

"""
    A(m::DBiCM, i::Int, α::Int; channel::Symbol=:to_top)

Expected value of a single biadjacency entry, with layer-local indices. `channel` selects the
direction: `:to_top` for `⊥ᵢ → ⊤α`, `:to_bottom` for `⊤α → ⊥ᵢ`.
"""
A(m::DBiCM, i::Int, α::Int; channel::Symbol=:to_top) = _channel_id(channel) === :plus ? p⁺(m, i, α) : p⁻(m, i, α)

"""
    Ĝ(m::DBiCM; channel::Symbol=:to_top)

Expected biadjacency matrix of `m`, `N⊥ × N⊤`, with rows indexed by the ⊥ layer and columns by the ⊤
layer in both cases.

`channel = :to_top` (default) gives `⟨B⁺⟩`, the `⊥ → ⊤` links; `:to_bottom` gives `⟨B⁻⟩`, the `⊤ → ⊥`
links. `channel = :both` returns the two matrices stacked vertically, `[⟨B⁺⟩ ; ⟨B⁻⟩]` (`2N⊥ × N⊤`),
which is the form [`σₓ`](@ref) expects for a metric that reads both channels at once.
"""
function Ĝ(m::DBiCM; channel::Symbol=:to_top)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    channel === :both && return vcat(Ĝ(m, channel=:to_top), Ĝ(m, channel=:to_bottom))
    c = _channel(m, channel)
    N⊥, N⊤ = m.status[:N⊥]::Int, m.status[:N⊤]::Int
    x = c.x_own[c.ind_own]
    y = c.x_opp[c.ind_opp]
    G = zeros(precision(m), N⊥, N⊤)
    @inbounds for i in 1:N⊥
        @simd for j in 1:N⊤
            xy = x[i] * y[j]
            G[i,j] = xy / (1 + xy)
        end
    end
    return G
end

"""
    set_Ĝ!(m::DBiCM)

Compute and store **both** expected biadjacency matrices of `m` (`m.Ĝ⁺` and `m.Ĝ⁻`).
"""
function set_Ĝ!(m::DBiCM)
    m.Ĝ⁺ = Ĝ(m, channel=:to_top)
    m.Ĝ⁻ = Ĝ(m, channel=:to_bottom)
    m.status[:G_computed] = true
    return m
end

"""
    σˣ(m::DBiCM; channel::Symbol=:to_top)

Entry-wise standard deviations of the expected biadjacency matrix of `m`, `σ[B] = sqrt(p(1-p))`.
`channel` is as for [`Ĝ`](@ref), including `:both`.
"""
function σˣ(m::DBiCM; channel::Symbol=:to_top)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    channel === :both && return vcat(σˣ(m, channel=:to_top), σˣ(m, channel=:to_bottom))
    G = Ĝ(m, channel=channel)
    return sqrt.(G .* (1 .- G))
end

"""
    set_σ!(m::DBiCM)

Compute and store **both** standard-deviation matrices of `m` (`m.σ⁺` and `m.σ⁻`).
"""
function set_σ!(m::DBiCM)
    m.σ⁺ = σˣ(m, channel=:to_top)
    m.σ⁻ = σˣ(m, channel=:to_bottom)
    m.status[:σ_computed] = true
    return m
end

"""
    σₓ(m::DBiCM, X::Function; channel::Symbol=:to_top, gradient_method::Symbol=:ReverseDiff)

Standard deviation of the metric `X` under the DBiCM model `m`, by the delta method.

Every biadjacency entry of a DBiCM is an independent Bernoulli variable — within a channel, across
channels, and within a `(i,α)` dyad — so there are no covariance terms anywhere and

``σ²[X] = Σ (σ_{iα} ∂X/∂B_{iα})²``

With `channel = :both`, `X` is a function of the stacked `2N⊥ × N⊤` matrix `[B⁺ ; B⁻]` and the sum
runs over both blocks; this is exact for the same reason, and is how metrics that mix the two
channels (total links, reciprocity, path motifs) are propagated.
"""
function σₓ(m::DBiCM, X::Function; channel::Symbol=:to_top, gradient_method::Symbol=:ReverseDiff)
    m.status[:G_computed] ? nothing : throw(ArgumentError("The expected values (m.Ĝ⁺/m.Ĝ⁻) must be computed for `m` first, see `set_Ĝ!`"))
    m.status[:σ_computed] ? nothing : throw(ArgumentError("The standard deviations (m.σ⁺/m.σ⁻) must be computed for `m` first, see `set_σ!`"))
    Ĝ_ = channel === :both ? vcat(m.Ĝ⁺, m.Ĝ⁻) : (_channel_id(channel) === :plus ? m.Ĝ⁺ : m.Ĝ⁻)
    σ_ = channel === :both ? vcat(m.σ⁺, m.σ⁻) : (_channel_id(channel) === :plus ? m.σ⁺ : m.σ⁻)
    if gradient_method == :ForwardDiff
        ∇X = ForwardDiff.gradient(X, Ĝ_)
    elseif gradient_method == :ReverseDiff
        ∇X = ReverseDiff.gradient(X, Ĝ_)
    elseif gradient_method == :Zygote
        ∇X = Zygote.gradient(X, Ĝ_)[1]
    else
        throw(ArgumentError("Invalid gradient method, only :ForwardDiff, :ReverseDiff and :Zygote are accepted"))
    end
    # independent entries throughout: no covariance terms
    return sqrt(sum((σ_ .* ∇X) .^ 2))
end

"""
    reciprocity(m::DBiCM)

Expected topological reciprocity of the DBiCM model `m`, as the ratio of expectations

``r = 2 Σ_{iα} p⁺_{iα} p⁻_{iα} / Σ_{iα} (p⁺_{iα} + p⁻_{iα})``

`B⁺_{iα}` and `B⁻_{iα}` are independent, so `⟨B⁺B⁻⟩ = p⁺p⁻` exactly and no approximation enters. The
factor of two makes this agree with `reciprocity(A::AbstractMatrix)` applied to the full expected
adjacency matrix `[0 ⟨B⁺⟩ ; ⟨B⁻⟩ᵀ 0]`, since each reciprocated ⊥–⊤ dyad contributes two ordered pairs.
"""
function reciprocity(m::DBiCM)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    N⊥, N⊤ = m.status[:N⊥]::Int, m.status[:N⊤]::Int
    num = zero(precision(m)); den = zero(precision(m))
    @inbounds for i in 1:N⊥, α in 1:N⊤
        a, b = p⁺(m, i, α), p⁻(m, i, α)
        num += a * b
        den += a + b
    end
    iszero(den) && throw(ArgumentError("The expected network has no links, reciprocity is undefined"))
    return 2 * num / den
end


##############################################################################################
# Degree accessors
##############################################################################################

"""
    _check_vertex(m::DBiCM, i::Int)

Validate that `m` is solved and that `i` is a vertex of it. Called before any layer lookup, so an
out-of-range vertex raises a clear `ArgumentError` rather than a `BoundsError` from inside `is⊥`.
"""
function _check_vertex(m::DBiCM, i::Int)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    (1 <= i <= m.status[:N]::Int) || throw(ArgumentError("Attempted to access node $i in a $(m.status[:N]) node graph"))
    return nothing
end

"""
    _dbicm_expected_degree(m::DBiCM, i::Int, channel::Symbol, method::Symbol)

Expected number of links of graph vertex `i` in one channel. `channel` is absolute, so for a ⊥ vertex
`:to_top` is its out-degree while for a ⊤ vertex `:to_top` is its in-degree.
"""
function _dbicm_expected_degree(m::DBiCM, i::Int, channel::Symbol, method::Symbol)
    _check_vertex(m, i)
    c = _channel(m, channel)
    N⊥, N⊤ = m.status[:N⊥]::Int, m.status[:N⊤]::Int
    if method == :reduced
        res = zero(precision(m))
        if m.is⊥[i]
            i_red = c.ind_own[m.⊥map[i]]
            @inbounds for j in eachindex(c.x_opp)
                res += f_BiCM(c.x_own[i_red] * c.x_opp[j]) * c.f_opp[j]
            end
        else
            i_red = c.ind_opp[m.⊤map[i]]
            @inbounds for j in eachindex(c.x_own)
                res += f_BiCM(c.x_own[j] * c.x_opp[i_red]) * c.f_own[j]
            end
        end
        return res
    elseif method == :full
        res = zero(precision(m))
        if m.is⊥[i]
            @inbounds for j in 1:N⊤; res += A(m, m.⊥map[i], j; channel=channel); end
        else
            @inbounds for j in 1:N⊥; res += A(m, j, m.⊤map[i]; channel=channel); end
        end
        return res
    elseif method == :adjacency
        m.status[:G_computed] ? nothing : throw(ArgumentError("The expected biadjacency matrices have not been computed yet"))
        M = _channel_id(channel) === :plus ? m.Ĝ⁺ : m.Ĝ⁻
        return m.is⊥[i] ? sum(@view M[m.⊥map[i], :]) : sum(@view M[:, m.⊤map[i]])
    else
        throw(ArgumentError("The method $(method) is not supported, use :reduced, :full or :adjacency"))
    end
end

"""
    outdegree(m::DBiCM, i::Int; method::Symbol=:reduced)
    outdegree(m::DBiCM, v::Vector{Int}=collect(1:m.status[:N]); method::Symbol=:reduced)

Expected out-degree of vertex `i` of the DBiCM model `m` (graph vertex ids, either layer). A ⊥ vertex
sends through channel `⁺`, a ⊤ vertex through channel `⁻`.

`method` is `:reduced` (default, from the reduced parameters), `:full` (summing the expected entries)
or `:adjacency` (from the stored matrices, which must have been computed with `set_Ĝ!`).
"""
function outdegree(m::DBiCM, i::Int; method::Symbol=:reduced)
    _check_vertex(m, i)
    return _dbicm_expected_degree(m, i, m.is⊥[i] ? :to_top : :to_bottom, method)
end
outdegree(m::DBiCM, v::Vector{Int}=collect(1:m.status[:N]); method::Symbol=:reduced) =
    [outdegree(m, i; method=method) for i in v]

"""
    indegree(m::DBiCM, i::Int; method::Symbol=:reduced)
    indegree(m::DBiCM, v::Vector{Int}=collect(1:m.status[:N]); method::Symbol=:reduced)

Expected in-degree of vertex `i` of the DBiCM model `m`. A ⊥ vertex receives through channel `⁻`, a ⊤
vertex through channel `⁺`. See [`outdegree`](@ref) for `method`.
"""
function indegree(m::DBiCM, i::Int; method::Symbol=:reduced)
    _check_vertex(m, i)
    return _dbicm_expected_degree(m, i, m.is⊥[i] ? :to_bottom : :to_top, method)
end
indegree(m::DBiCM, v::Vector{Int}=collect(1:m.status[:N]); method::Symbol=:reduced) =
    [indegree(m, i; method=method) for i in v]

"""
    degree(m::DBiCM, i::Int; method::Symbol=:reduced)
    degree(m::DBiCM, v::Vector{Int}=collect(1:m.status[:N]); method::Symbol=:reduced)

Expected total degree (out plus in) of vertex `i` of the DBiCM model `m`.
"""
degree(m::DBiCM, i::Int; method::Symbol=:reduced) = outdegree(m, i; method=method) + indegree(m, i; method=method)
degree(m::DBiCM, v::Vector{Int}=collect(1:m.status[:N]); method::Symbol=:reduced) =
    [degree(m, i; method=method) for i in v]

##############################################################################################
# Sampling
##############################################################################################

"""
    rand(m::DBiCM; precomputed::Bool=false, rng::AbstractRNG=default_rng())

Sample a directed bipartite graph from the ensemble of the DBiCM model `m`, as a `SimpleDiGraph` in the
**original vertex numbering** of `m.G`. Both channels are sampled independently.

Set `precomputed = true` to draw from the stored matrices (`set_Ĝ!` first) rather than recomputing the
probabilities on the fly.
"""
function rand(m::DBiCM; precomputed::Bool=false, rng::AbstractRNG=default_rng())
    if precomputed
        m.status[:G_computed] ? nothing : throw(ArgumentError("The expected biadjacency matrices have not been computed yet"))
        P⁺, P⁻ = m.Ĝ⁺, m.Ĝ⁻
    else
        m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
        P⁺, P⁻ = Ĝ(m, channel=:to_top), Ĝ(m, channel=:to_bottom)
    end
    edges = Graphs.SimpleDiGraphEdge{Int}[]
    @inbounds for (i, or⊥) in enumerate(m.⊥nodes), (j, or⊤) in enumerate(m.⊤nodes)
        rand(rng) < P⁺[i,j] && push!(edges, Graphs.SimpleDiGraphEdge(or⊥, or⊤))
        rand(rng) < P⁻[i,j] && push!(edges, Graphs.SimpleDiGraphEdge(or⊤, or⊥))
    end
    G = Graphs.SimpleDiGraphFromIterator(edges)
    # a vertex may end up with no sampled links at all, in which case it is missing from the iterator
    while Graphs.nv(G) < m.status[:N]::Int
        Graphs.add_vertex!(G)
    end
    return G
end

"""
    rand(m::DBiCM, n::Int; precomputed::Bool=false, rng::AbstractRNG=default_rng())

Sample `n` directed bipartite graphs from the ensemble of the DBiCM model `m`.
"""
function rand(m::DBiCM, n::Int; precomputed::Bool=false, rng::AbstractRNG=default_rng())
    res = Vector{Graphs.SimpleDiGraph{Int}}(undef, n)
    seeds = rand(rng, UInt64, n)    # per-sample seeds, so the result does not depend on the thread schedule
    Threads.@threads for i in 1:n
        res[i] = rand(m; precomputed=precomputed, rng=Xoshiro(seeds[i]))
    end
    return res
end

##############################################################################################
# Model selection
##############################################################################################

"""
    AIC(m::DBiCM)

Akaike information criterion of the DBiCM model `m`. The model has `2(N⊥ + N⊤)` parameters and
`2·N⊥·N⊤` observations — two independent matrices of Bernoulli entries. As for the [`BiCM`](@ref), the
gauge freedom is not discounted from the parameter count.
"""
function AIC(m::DBiCM)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    k = 2 * (m.status[:N⊥]::Int + m.status[:N⊤]::Int)
    n = 2 * m.status[:N⊥]::Int * m.status[:N⊤]::Int
    n/k < 40 && @warn """The number of observations is small with respect to the number of parameters (n/k < 40). Consider using the corrected AIC (AICc) instead."""
    return 2*k - 2*L_DBiCM_reduced(m)
end

"""
    AICc(m::DBiCM)

Corrected Akaike information criterion of the DBiCM model `m`, for a small number of observations.
"""
function AICc(m::DBiCM)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    k = 2 * (m.status[:N⊥]::Int + m.status[:N⊤]::Int)
    n = 2 * m.status[:N⊥]::Int * m.status[:N⊤]::Int
    return 2*k - 2*L_DBiCM_reduced(m) + (2*k*(k+1))/(n-k-1)
end

"""
    BIC(m::DBiCM)

Bayesian information criterion of the DBiCM model `m`.
"""
function BIC(m::DBiCM)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    k = 2 * (m.status[:N⊥]::Int + m.status[:N⊤]::Int)
    n = 2 * m.status[:N⊥]::Int * m.status[:N⊤]::Int
    n/k < 40 && @warn """The number of observations is small with respect to the number of parameters (n/k < 40). Consider using the corrected AIC (AICc) instead."""
    return k*log(n) - 2*L_DBiCM_reduced(m)
end
