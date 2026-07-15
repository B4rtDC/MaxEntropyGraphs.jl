
"""
    CRWCM

Maximum entropy model for the Conditionally Reciprocal Weighted Configuration Model (CRWCM).

The CRWCM (Di Vece, Pijpers & Garlaschelli (2023); `RBCM+CRWCM` in the NuMeTriS package) is the
reciprocity-aware counterpart of the [`DCReM`](@ref): a **two-step** null model for weighted, directed
networks with **continuous, positive** weights that accounts for the different nature of the links weights
sit on. The binary structure is supplied by an internally solved [`RBCM`](@ref) on the reciprocal degree
sequences, giving the dyadic probabilities `f⭢ᵢⱼ = p⭢ᵢⱼ` (single link `i→j`) and `f⭤ᵢⱼ = p⭤ᵢⱼ`
(reciprocated dyad). Conditional on the dyad state, the weights are exponential:

- a non-reciprocated link `i→j` carries a weight with rate `θ⭢ᵢ + θ⭠ⱼ`,
- a reciprocated pair carries two weights, `wᵢⱼ` with rate `θ⭤ᵒᵢ + θ⭤ⁱⱼ` and `wⱼᵢ` with rate `θ⭤ᵒⱼ + θ⭤ⁱᵢ`.

The `4N` parameters `θ = [θ⭢; θ⭠; θ⭤ᵒ; θ⭤ⁱ]` constrain the four reciprocal strength sequences
(`s→`, `s←`, `s↔out`, `s↔in`). The generalised (conditional) likelihood separates into two independent
sub-problems — the non-reciprocated `2N` system `{θ⭢, θ⭠}` and the reciprocated `2N` system `{θ⭤ᵒ, θ⭤ⁱ}` —
which are solved jointly as a single `4N` problem for API uniformity (the Hessian is block-diagonal, so
solvers benefit automatically).

*Note*: within a dyad the two weights `wᵢⱼ` and `wⱼᵢ` are **correlated** under the CRWCM (they are both
non-zero only in the reciprocated state): `Cov(wᵢⱼ, wⱼᵢ) ≠ 0`, unlike for the [`DCReM`](@ref). The
layer-aware `σₓ` accounts for this.
"""
mutable struct CRWCM{T<:Union{Graphs.AbstractGraph, Nothing}, N<:Real} <: AbstractMaxEntropyModel
    "Graph type, can be any subtype of AbstractGraph, but will be converted to SimpleWeightedDiGraph for the computation" # can also be empty
    const G::T
    "Maximum likelihood parameters of the weighted (CRWCM) layer, θ = [θ⭢; θ⭠; θ⭤ᵒ; θ⭤ⁱ] (four per node)"
    const θ::Vector{N}
    "Reduced maximum likelihood parameters of the binary (conditional RBCM) layer, αᵣ = [α; β; γ]"
    const αᵣ::Vector{N}
    "Exponentiated reduced binary parameters ( xᵢ = exp(-αᵢ) ), linked with the non-reciprocated out-degree"
    const xᵣ::Vector{N}
    "Exponentiated reduced binary parameters ( yᵢ = exp(-βᵢ) ), linked with the non-reciprocated in-degree"
    const yᵣ::Vector{N}
    "Exponentiated reduced binary parameters ( zᵢ = exp(-γᵢ) ), linked with the reciprocated degree"
    const zᵣ::Vector{N}
    "Non-reciprocated out-degree sequence of the graph (k→)"
    const d_out::Vector{Int}
    "Non-reciprocated in-degree sequence of the graph (k←)"
    const d_in::Vector{Int}
    "Reciprocated degree sequence of the graph (k↔)"
    const d_rec::Vector{Int}
    "Reduced non-reciprocated out-degree sequence of the graph"
    const dᵣ_out::Vector{Int}
    "Reduced non-reciprocated in-degree sequence of the graph"
    const dᵣ_in::Vector{Int}
    "Reduced reciprocated degree sequence of the graph"
    const dᵣ_rec::Vector{Int}
    "Frequency of each (k→, k←, k↔) triple in the graph"
    const f::Vector{Int}
    "Indices to reconstruct the degree sequences from the reduced degree sequences"
    const d_ind::Vector{Int}
    "Indices to reconstruct the reduced degree sequences from the degree sequences"
    const dᵣ_ind::Vector{Int}
    "Non-reciprocated out-strength sequence of the graph (s→, continuous, positive)"
    const s_out::Vector{N}
    "Non-reciprocated in-strength sequence of the graph (s←)"
    const s_in::Vector{N}
    "Reciprocated out-strength sequence of the graph (s↔out)"
    const s_rec_out::Vector{N}
    "Reciprocated in-strength sequence of the graph (s↔in)"
    const s_rec_in::Vector{N}
    "Indices of nodes with non-zero non-reciprocated out-strength"
    const s_out_nz::Vector{Int}
    "Indices of nodes with non-zero non-reciprocated in-strength"
    const s_in_nz::Vector{Int}
    "Indices of nodes with non-zero reciprocated strengths"
    const s_rec_nz::Vector{Int}
    "Expected (binary) adjacency matrix" # not always computed/required
    Ĝ::Union{Nothing, Matrix{N}}
    "Variance of the expected (binary) adjacency matrix" # not always computed/required
    σ::Union{Nothing, Matrix{N}}
    "Expected weighted adjacency matrix" # not always computed/required
    Ŵ::Union{Nothing, Matrix{N}}
    "Standard deviation of the weighted adjacency matrix" # not always computed/required
    σʷ::Union{Nothing, Matrix{N}}
    "Status indicators: parameters computed, expected adjacency matrix computed, variance computed, etc."
    const status::Dict{Symbol, Any}
end

Base.show(io::IO, m::CRWCM{T,N}) where {T,N} = print(io, """CRWCM{$(T), $(N)} ($(m.status[:N]) vertices, $(m.status[:d_unique]) unique degree triples, $(@sprintf("%.2f", m.status[:cᵣ])) compression ratio)""")

"""Return the number of vertices in the CRWCM network"""
Base.length(m::CRWCM) = length(m.d_out)


"""
    CRWCM(G::T; kwargs...) where {T}
    CRWCM(;d_out, d_in, d_rec, s_out, s_in, s_rec_out, s_rec_in, precision=Float64, kwargs...)

Constructor function for the `CRWCM` type.

By default the three reciprocal degree sequences (see [`nonreciprocated_outdegree`](@ref),
[`nonreciprocated_indegree`](@ref), [`reciprocated_degree`](@ref)) and the four reciprocal strength
sequences (see [`nonreciprocated_outstrength`](@ref), [`nonreciprocated_instrength`](@ref),
[`reciprocated_outstrength`](@ref), [`reciprocated_instrength`](@ref)) are computed from the weighted
directed graph `G`. You can also pass all seven sequences directly as keyword arguments.

The CRWCM allows **continuous, positive** weights. Because the weights are strictly positive, a node has a
zero strength in a channel *iff* its degree in that channel is zero; the constructor enforces this
consistency (`DomainError` otherwise).

# Examples
```jldoctest CRWCM_creation
# generating a model from a weighted directed graph
julia> model = CRWCM(MaxEntropyGraphs.rhesus_macaques())
CRWCM{SimpleWeightedGraphs.SimpleWeightedDiGraph{Int64, Float64}, Float64} (16 vertices, 15 unique degree triples, 0.94 compression ratio)

```
"""
function CRWCM(G::T;    d_out::Vector=Graphs.is_directed(G) ? nonreciprocated_outdegree(G) : zeros(Int, Graphs.nv(G)),
                        d_in::Vector=Graphs.is_directed(G)  ? nonreciprocated_indegree(G)  : zeros(Int, Graphs.nv(G)),
                        d_rec::Vector=Graphs.is_directed(G) ? reciprocated_degree(G)       : Graphs.degree(G),
                        s_out::Vector=Graphs.is_directed(G) ? nonreciprocated_outstrength(G) : zeros(Graphs.nv(G)),
                        s_in::Vector=Graphs.is_directed(G)  ? nonreciprocated_instrength(G)  : zeros(Graphs.nv(G)),
                        s_rec_out::Vector=Graphs.is_directed(G) ? reciprocated_outstrength(G) : strength(G),
                        s_rec_in::Vector=Graphs.is_directed(G)  ? reciprocated_instrength(G)  : strength(G),
                        precision::Type{<:AbstractFloat}=Float64, kwargs...) where {T}
    T <: Union{Graphs.AbstractGraph, Nothing} ? nothing : throw(TypeError(:CRWCM, "G must be a subtype of AbstractGraph or Nothing", Union{Graphs.AbstractGraph, Nothing}, T))
    # coherence checks
    if T <: Graphs.AbstractGraph # Graph specific checks
        if !Graphs.is_directed(G)
            @warn "The graph is undirected, while the CRWCM model is directed; every edge will be considered reciprocated (k→ = k← = 0)"
        end

        Graphs.nv(G) == 0 ? throw(ArgumentError("The graph is empty")) : nothing
        Graphs.nv(G) == 1 ? throw(ArgumentError("The graph has only one vertex")) : nothing
        Graphs.nv(G) != length(d_out) ? throw(DimensionMismatch("The number of vertices in the graph ($(Graphs.nv(G))) and the length of the degree sequences ($(length(d_out))) do not match")) : nothing
    end

    # coherence checks specific to the degree/strength sequences
    all(length(d_out) .== (length(d_in), length(d_rec), length(s_out), length(s_in), length(s_rec_out), length(s_rec_in))) ? nothing : throw(DimensionMismatch("The degree and strength sequences must all have the same length"))
    length(d_out) == 0 ? throw(ArgumentError("The degree sequences are empty")) : nothing
    length(d_out) == 1 ? throw(ArgumentError("The degree sequences only contain a single node")) : nothing
    (minimum(d_out) < 0 || minimum(d_in) < 0 || minimum(d_rec) < 0) ? throw(DomainError("The degree sequences must be non-negative")) : nothing
    (all(s_out .>= 0) && all(s_in .>= 0) && all(s_rec_out .>= 0) && all(s_rec_in .>= 0)) ? nothing : throw(DomainError("The strength sequences contain negative strengths"))
    sum(d_out) == sum(d_in) ? nothing : throw(DomainError("The total number of non-reciprocated out- and in-stubs must match (sum(d_out) == sum(d_in))"))
    iseven(sum(d_rec)) ? nothing : throw(DomainError("The total number of reciprocated stubs must be even (each reciprocated dyad contributes two)"))
    maximum(d_out .+ d_in .+ d_rec) >= length(d_out) ? throw(DomainError("A node's total number of connected dyads (k→ + k← + k↔) is greater or equal to the number of vertices, this is not allowed")) : nothing
    # positive weights ⟹ zero strength in a channel iff zero degree in that channel
    all((s_out .> 0) .== (d_out .> 0)) ? nothing : throw(DomainError("Inconsistent sequences: with positive weights, s→ᵢ > 0 ⟺ k→ᵢ > 0 must hold for every node"))
    all((s_in .> 0) .== (d_in .> 0)) ? nothing : throw(DomainError("Inconsistent sequences: with positive weights, s←ᵢ > 0 ⟺ k←ᵢ > 0 must hold for every node"))
    all((s_rec_out .> 0) .== (d_rec .> 0)) && all((s_rec_in .> 0) .== (d_rec .> 0)) ? nothing : throw(DomainError("Inconsistent sequences: with positive weights, s↔ᵢ > 0 ⟺ k↔ᵢ > 0 must hold for every node (in both directions)"))
    if iszero(sum(d_rec))
        @warn "The reciprocated degree sequence is all zeros: the CRWCM degenerates to a DCReM. Consider using the DCReM instead."
    elseif iszero(sum(d_out)) && iszero(sum(d_in))
        @warn "The non-reciprocated degree sequences are all zeros (fully reciprocal network): only the reciprocated parameters are identified."
    end

    # field generation (the binary layer reduces over the (k→, k←, k↔) triples, exactly like the RBCM)
    dᵣ, d_ind, dᵣ_ind, f = np_unique_clone(collect(zip(d_out, d_in, d_rec)), sorted=true)
    dᵣ_out = [d[1] for d in dᵣ]
    dᵣ_in  = [d[2] for d in dᵣ]
    dᵣ_rec = [d[3] for d in dᵣ]
    θ  = Vector{precision}(undef, 4*length(d_out))  # weighted (CRWCM) parameters, [θ⭢; θ⭠; θ⭤ᵒ; θ⭤ⁱ]
    αᵣ = Vector{precision}(undef, 3*length(dᵣ))     # reduced binary (RBCM) parameters, [α; β; γ]
    xᵣ = Vector{precision}(undef, length(dᵣ))
    yᵣ = Vector{precision}(undef, length(dᵣ))
    zᵣ = Vector{precision}(undef, length(dᵣ))
    s_out_nz = findall(!iszero, s_out)
    s_in_nz  = findall(!iszero, s_in)
    s_rec_nz = findall(!iszero, s_rec_out)
    status = Dict(  :conditional_params_computed => false,   # is the binary (RBCM) layer solved?
                    :params_computed             => false,   # is the weighted (CRWCM) layer solved?
                    :G_computed                  => false,   # is the expected adjacency matrix computed and stored?
                    :σ_computed                  => false,   # is the standard deviation computed and stored?
                    :W_computed                  => false,   # is the expected weight matrix computed and stored?
                    :σʷ_computed                 => false,   # is the weight standard deviation computed and stored?
                    :cᵣ       => length(dᵣ)/length(d_out),   # compression ratio of the reduced binary model
                    :d_unique => length(dᵣ),                 # number of unique (k→, k←, k↔) triples in the reduced model
                    :N        => length(d_out)               # number of vertices in the original graph
                )

    return CRWCM{T,precision}(G, θ, αᵣ, xᵣ, yᵣ, zᵣ, Int.(d_out), Int.(d_in), Int.(d_rec), dᵣ_out, dᵣ_in, dᵣ_rec, f, d_ind, dᵣ_ind,
                              Vector{precision}(s_out), Vector{precision}(s_in), Vector{precision}(s_rec_out), Vector{precision}(s_rec_in),
                              s_out_nz, s_in_nz, s_rec_nz, nothing, nothing, nothing, nothing, status)
end

CRWCM(;d_out::Vector, d_in::Vector, d_rec::Vector, s_out::Vector, s_in::Vector, s_rec_out::Vector, s_rec_in::Vector, precision::Type{<:AbstractFloat}=Float64, kwargs...) =
    CRWCM(nothing, d_out=d_out, d_in=d_in, d_rec=d_rec, s_out=s_out, s_in=s_in, s_rec_out=s_rec_out, s_rec_in=s_rec_in, precision=precision, kwargs...)


# dyadic state probabilities (on the fly, from the full per-node binary fitnesses)
@inline _CRWCM_f⭢(x::AbstractVector, y::AbstractVector, z::AbstractVector, i::Int, j::Int) = @inbounds x[i]*y[j] / (1 + x[i]*y[j] + x[j]*y[i] + z[i]*z[j])
@inline _CRWCM_f⭤(x::AbstractVector, y::AbstractVector, z::AbstractVector, i::Int, j::Int) = @inbounds z[i]*z[j] / (1 + x[i]*y[j] + x[j]*y[i] + z[i]*z[j])


"""
    L_CRWCM(θ::AbstractVector, s_out, s_in, s_rec_out, s_rec_in, nz_out, nz_in, nz_rec, x, y, z)

Compute the (generalised, conditional) log-likelihood of the CRWCM model given the weighted parameters
`θ = [θ⭢; θ⭠; θ⭤ᵒ; θ⭤ⁱ]`, the four reciprocal strength sequences, their non-zero index sets and the
per-node binary (RBCM) fitnesses `x`, `y`, `z`, computing the dyadic probabilities on the fly.

The log-likelihood is

``\\mathcal{G} = -\\sum_i \\left[ θ^{→}_i s^{→}_i + θ^{←}_i s^{←}_i + θ^{↔,o}_i s^{↔,out}_i + θ^{↔,i}_i s^{↔,in}_i \\right] + \\sum_{i≠j} \\left[ f^{→}_{ij} \\log(θ^{→}_i + θ^{←}_j) + f^{↔}_{ij} \\log(θ^{↔,o}_i + θ^{↔,i}_j) \\right]``

(Di Vece et al. (2023), Methods). It is **block-separable** into the non-reciprocated system `{θ⭢, θ⭠}` and
the reciprocated system `{θ⭤ᵒ, θ⭤ⁱ}`. The `log` is domain-guarded (returns `NaN` outside `θ + θ > 0`) so
the line search rejects infeasible steps. Channels whose strength constraint is zero have `f = 0` in every
pair term; they are excluded from the sums (their parameter is undetermined and pinned to `+Inf` after the solve).

This method is used when the dyadic probability matrices are not precomputed.
"""
function L_CRWCM(θ::AbstractVector, s_out::AbstractVector, s_in::AbstractVector, s_rec_out::AbstractVector, s_rec_in::AbstractVector,
                 nz_out::Vector, nz_in::Vector, nz_rec::Vector,
                 x::AbstractVector, y::AbstractVector, z::AbstractVector)
    n = length(s_out)
    θ⭢  = @view θ[1:n]
    θ⭠  = @view θ[n+1:2*n]
    θ⭤ᵒ = @view θ[2*n+1:3*n]
    θ⭤ⁱ = @view θ[3*n+1:4*n]
    res = zero(eltype(θ))
    # linear parts (restricted to non-zero strengths: dead channels contribute exactly zero)
    for i in nz_out
        @inbounds res -= θ⭢[i] * s_out[i]
    end
    for i in nz_in
        @inbounds res -= θ⭠[i] * s_in[i]
    end
    for i in nz_rec
        @inbounds res -= θ⭤ᵒ[i] * s_rec_out[i] + θ⭤ⁱ[i] * s_rec_in[i]
    end
    # pair parts over ordered pairs; pairs with a zero dyadic probability are skipped exactly
    @inbounds for i in 1:n
        acc = zero(eltype(θ))
        for j in 1:n
            if i ≠ j
                f⭢ = _CRWCM_f⭢(x, y, z, i, j)
                iszero(f⭢) || (acc += f⭢ * log_nan(θ⭢[i] + θ⭠[j]))
                f⭤ = _CRWCM_f⭤(x, y, z, i, j)
                iszero(f⭤) || (acc += f⭤ * log_nan(θ⭤ᵒ[i] + θ⭤ⁱ[j]))
            end
        end
        res += acc
    end

    return res
end


"""
    L_CRWCM(θ::AbstractVector, s_out, s_in, s_rec_out, s_rec_in, nz_out, nz_in, nz_rec, f⭢::AbstractMatrix, f⭤::AbstractMatrix)

Compute the log-likelihood of the CRWCM model, using the precomputed dyadic probability matrices `f⭢` and
`f⭤`. See also [`L_CRWCM(::AbstractVector, ::AbstractVector, ::AbstractVector, ::AbstractVector, ::AbstractVector, ::Vector, ::Vector, ::Vector, ::AbstractVector, ::AbstractVector, ::AbstractVector)`](@ref).
"""
function L_CRWCM(θ::AbstractVector, s_out::AbstractVector, s_in::AbstractVector, s_rec_out::AbstractVector, s_rec_in::AbstractVector,
                 nz_out::Vector, nz_in::Vector, nz_rec::Vector,
                 f⭢::AbstractMatrix, f⭤::AbstractMatrix)
    n = length(s_out)
    θ⭢  = @view θ[1:n]
    θ⭠  = @view θ[n+1:2*n]
    θ⭤ᵒ = @view θ[2*n+1:3*n]
    θ⭤ⁱ = @view θ[3*n+1:4*n]
    res = zero(eltype(θ))
    for i in nz_out
        @inbounds res -= θ⭢[i] * s_out[i]
    end
    for i in nz_in
        @inbounds res -= θ⭠[i] * s_in[i]
    end
    for i in nz_rec
        @inbounds res -= θ⭤ᵒ[i] * s_rec_out[i] + θ⭤ⁱ[i] * s_rec_in[i]
    end
    @inbounds for i in 1:n
        acc = zero(eltype(θ))
        for j in 1:n
            if i ≠ j
                iszero(f⭢[i,j]) || (acc += f⭢[i,j] * log_nan(θ⭢[i] + θ⭠[j]))
                iszero(f⭤[i,j]) || (acc += f⭤[i,j] * log_nan(θ⭤ᵒ[i] + θ⭤ⁱ[j]))
            end
        end
        res += acc
    end

    return res
end


"""
    L_CRWCM(m::CRWCM)

Return the log-likelihood of the CRWCM model `m` based on the computed maximum likelihood parameters.
"""
function L_CRWCM(m::CRWCM)
    m.status[:conditional_params_computed] ? nothing : throw(ArgumentError("The conditional parameters of the model have not been computed"))
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters of the model have not been computed"))
    return L_CRWCM(m.θ, m.s_out, m.s_in, m.s_rec_out, m.s_rec_in, m.s_out_nz, m.s_in_nz, m.s_rec_nz,
                   m.xᵣ[m.dᵣ_ind], m.yᵣ[m.dᵣ_ind], m.zᵣ[m.dᵣ_ind])
end


"""
    ∇L_CRWCM!(∇L::AbstractVector, θ::AbstractVector, s_out, s_in, s_rec_out, s_rec_in, nz_out, nz_in, nz_rec, x, y, z)

Compute the gradient of the log-likelihood of the CRWCM model in a non-allocating manner, computing the
dyadic probabilities on the fly.

`∂L/∂θ⭢ᵢ = -s→ᵢ + Σⱼ≠ᵢ f⭢ᵢⱼ/(θ⭢ᵢ + θ⭠ⱼ)`, `∂L/∂θ⭠ᵢ = -s←ᵢ + Σⱼ≠ᵢ f⭢ⱼᵢ/(θ⭠ᵢ + θ⭢ⱼ)` (note `f⭠ᵢⱼ = f⭢ⱼᵢ`),
`∂L/∂θ⭤ᵒᵢ = -s↔outᵢ + Σⱼ≠ᵢ f⭤ᵢⱼ/(θ⭤ᵒᵢ + θ⭤ⁱⱼ)` and `∂L/∂θ⭤ⁱᵢ = -s↔inᵢ + Σⱼ≠ᵢ f⭤ᵢⱼ/(θ⭤ⁱᵢ + θ⭤ᵒⱼ)`
(`f⭤` is symmetric). Rows of dead channels (zero strength) are left at zero.

See also [`∇L_CRWCM_minus!`](@ref).
"""
function ∇L_CRWCM!(∇L::AbstractVector, θ::AbstractVector, s_out::AbstractVector, s_in::AbstractVector, s_rec_out::AbstractVector, s_rec_in::AbstractVector,
                   nz_out::Vector, nz_in::Vector, nz_rec::Vector,
                   x::AbstractVector, y::AbstractVector, z::AbstractVector)
    n = length(s_out)
    θ⭢  = @view θ[1:n]
    θ⭠  = @view θ[n+1:2*n]
    θ⭤ᵒ = @view θ[2*n+1:3*n]
    θ⭤ⁱ = @view θ[3*n+1:4*n]
    ∇L .= zero(eltype(∇L))
    @inbounds for i in nz_out
        acc = zero(eltype(∇L))
        for j in 1:n
            if i ≠ j
                f⭢ = _CRWCM_f⭢(x, y, z, i, j)
                iszero(f⭢) || (acc += f⭢ / (θ⭢[i] + θ⭠[j]))
            end
        end
        ∇L[i] = -s_out[i] + acc
    end
    @inbounds for i in nz_in
        acc = zero(eltype(∇L))
        for j in 1:n
            if i ≠ j
                f⭠ = _CRWCM_f⭢(x, y, z, j, i) # f⭠ᵢⱼ = f⭢ⱼᵢ
                iszero(f⭠) || (acc += f⭠ / (θ⭠[i] + θ⭢[j]))
            end
        end
        ∇L[n+i] = -s_in[i] + acc
    end
    @inbounds for i in nz_rec
        acc_o = zero(eltype(∇L))
        acc_i = zero(eltype(∇L))
        for j in 1:n
            if i ≠ j
                f⭤ = _CRWCM_f⭤(x, y, z, i, j)
                if !iszero(f⭤)
                    acc_o += f⭤ / (θ⭤ᵒ[i] + θ⭤ⁱ[j])
                    acc_i += f⭤ / (θ⭤ⁱ[i] + θ⭤ᵒ[j])
                end
            end
        end
        ∇L[2*n+i] = -s_rec_out[i] + acc_o
        ∇L[3*n+i] = -s_rec_in[i]  + acc_i
    end

    return ∇L
end


"""
    ∇L_CRWCM_minus!(∇L::AbstractVector, θ::AbstractVector, s_out, s_in, s_rec_out, s_rec_in, nz_out, nz_in, nz_rec, x, y, z)

Compute minus the gradient of the log-likelihood of the CRWCM model (used for the minimisation carried out
by `Optimization.jl`), computing the dyadic probabilities on the fly. Non-allocating.

See also [`∇L_CRWCM!`](@ref).
"""
function ∇L_CRWCM_minus!(∇L::AbstractVector, θ::AbstractVector, s_out::AbstractVector, s_in::AbstractVector, s_rec_out::AbstractVector, s_rec_in::AbstractVector,
                         nz_out::Vector, nz_in::Vector, nz_rec::Vector,
                         x::AbstractVector, y::AbstractVector, z::AbstractVector)
    ∇L_CRWCM!(∇L, θ, s_out, s_in, s_rec_out, s_rec_in, nz_out, nz_in, nz_rec, x, y, z)
    ∇L .= .-∇L
    return ∇L
end


"""
    CRWCM_iter!(θ::AbstractVector, s_out, s_in, s_rec_out, s_rec_in, nz_out, nz_in, nz_rec, x, y, z, G::AbstractVector)

Compute the next fixed-point iteration for the CRWCM model, computing the dyadic probabilities on the fly.
The pre-allocated buffer `G` is updated in place. The consistency equations mirror the CReM family, e.g.
``θ^{→}_i = \\left( \\sum_{j≠i} f^{→}_{ij}/(1 + θ^{←}_j/θ^{→}_i) \\right)/s^{→}_i`` for the four channels.
Rows of dead channels (zero strength) are fixed at the current parameter value (residual zero).
"""
function CRWCM_iter!(θ::AbstractVector, s_out::AbstractVector, s_in::AbstractVector, s_rec_out::AbstractVector, s_rec_in::AbstractVector,
                     nz_out::Vector, nz_in::Vector, nz_rec::Vector,
                     x::AbstractVector, y::AbstractVector, z::AbstractVector, G::AbstractVector)
    n = length(s_out)
    θ⭢  = @view θ[1:n]
    θ⭠  = @view θ[n+1:2*n]
    θ⭤ᵒ = @view θ[2*n+1:3*n]
    θ⭤ⁱ = @view θ[3*n+1:4*n]
    G .= θ # dead channels: residual zero (their value is not updated)
    @inbounds for i in nz_out
        acc = zero(eltype(θ))
        for j in 1:n
            if i ≠ j
                f⭢ = _CRWCM_f⭢(x, y, z, i, j)
                iszero(f⭢) || (acc += f⭢ / (1 + θ⭠[j] / θ⭢[i]))
            end
        end
        G[i] = acc / s_out[i]
    end
    @inbounds for i in nz_in
        acc = zero(eltype(θ))
        for j in 1:n
            if i ≠ j
                f⭠ = _CRWCM_f⭢(x, y, z, j, i)
                iszero(f⭠) || (acc += f⭠ / (1 + θ⭢[j] / θ⭠[i]))
            end
        end
        G[n+i] = acc / s_in[i]
    end
    @inbounds for i in nz_rec
        acc_o = zero(eltype(θ))
        acc_i = zero(eltype(θ))
        for j in 1:n
            if i ≠ j
                f⭤ = _CRWCM_f⭤(x, y, z, i, j)
                if !iszero(f⭤)
                    acc_o += f⭤ / (1 + θ⭤ⁱ[j] / θ⭤ᵒ[i])
                    acc_i += f⭤ / (1 + θ⭤ᵒ[j] / θ⭤ⁱ[i])
                end
            end
        end
        G[2*n+i] = acc_o / s_rec_out[i]
        G[3*n+i] = acc_i / s_rec_in[i]
    end

    return G
end


"""
    initial_guess(m::CRWCM; method::Symbol=:strengths)

Compute an initial guess `θ₀ = [θ⭢₀; θ⭠₀; θ⭤ᵒ₀; θ⭤ⁱ₀]` for the maximum likelihood parameters of the
weighted (CRWCM) layer of the model `m`.

The CRWCM parameters `θ` are the **direct** rate parameters, so every method returns non-negative,
feasible values (dead channels get a zero placeholder; they are pinned to `+Inf` after the solve):
- `:strengths` (default): `θᵢ = 𝟙[sᵢ > 0] / Σ_channels Σⱼ sⱼ` per channel.
- `:strengths_minor`: `θᵢ = 𝟙[sᵢ > 0] / (sᵢ + 1)` per channel.
- `:random`: random values drawn from ``U(0,1)`` (zeroed on dead channels).
"""
function initial_guess(m::CRWCM; method::Symbol=:strengths)
    N = precision(m)
    mask = vcat(m.s_out .> 0, m.s_in .> 0, m.s_rec_out .> 0, m.s_rec_in .> 0)
    if isequal(method, :strengths)
        stot = sum(m.s_out) + sum(m.s_in) + sum(m.s_rec_out) + sum(m.s_rec_in)
        res = Vector{N}(mask ./ stot)
    elseif isequal(method, :strengths_minor)
        res = Vector{N}(mask ./ (vcat(m.s_out, m.s_in, m.s_rec_out, m.s_rec_in) .+ 1))
    elseif isequal(method, :random)
        res = Vector{N}(rand(N, 4*length(m.s_out)) .* mask)
    else
        throw(ArgumentError("The initial guess method $(method) is not supported"))
    end

    return res
end


"""
    precision(m::CRWCM)

Determine the compute precision of the CRWCM model `m`.
"""
precision(m::CRWCM) = typeof(m).parameters[2]


"""
    set_xᵣ!(m::CRWCM)

Set the value of `xᵣ` to `exp(-αᵣ[1:d_unique])` for the (binary layer of the) CRWCM model `m`.
"""
function set_xᵣ!(m::CRWCM)
    if m.status[:conditional_params_computed]
        αᵣ = @view m.αᵣ[1:m.status[:d_unique]]
        m.xᵣ .= exp.(-αᵣ)
    else
        throw(ArgumentError("The conditional parameters have not been computed yet"))
    end
end

"""
    set_yᵣ!(m::CRWCM)

Set the value of `yᵣ` to `exp(-αᵣ[d_unique+1:2*d_unique])` for the (binary layer of the) CRWCM model `m`.
"""
function set_yᵣ!(m::CRWCM)
    if m.status[:conditional_params_computed]
        βᵣ = @view m.αᵣ[m.status[:d_unique]+1:2*m.status[:d_unique]]
        m.yᵣ .= exp.(-βᵣ)
    else
        throw(ArgumentError("The conditional parameters have not been computed yet"))
    end
end

"""
    set_zᵣ!(m::CRWCM)

Set the value of `zᵣ` to `exp(-αᵣ[2*d_unique+1:end])` for the (binary layer of the) CRWCM model `m`.
"""
function set_zᵣ!(m::CRWCM)
    if m.status[:conditional_params_computed]
        γᵣ = @view m.αᵣ[2*m.status[:d_unique]+1:end]
        m.zᵣ .= exp.(-γᵣ)
    else
        throw(ArgumentError("The conditional parameters have not been computed yet"))
    end
end


"""
    p⭢(m::CRWCM, i::Int, j::Int)

Return the probability that the dyad `(i,j)` holds a single non-reciprocated link `i→j` under the binary
(RBCM) layer of the CRWCM model `m`.

❗ For performance reasons, the function does not check if the node pair is valid or if the conditional parameters have been computed.
"""
function p⭢(m::CRWCM, i::Int, j::Int)
    i == j && return zero(precision(m))
    @inbounds ri, rj = m.dᵣ_ind[i], m.dᵣ_ind[j]
    @inbounds xy = m.xᵣ[ri] * m.yᵣ[rj]
    @inbounds D = one(precision(m)) + xy + m.xᵣ[rj] * m.yᵣ[ri] + m.zᵣ[ri] * m.zᵣ[rj]
    return xy / D
end

"""
    p⭠(m::CRWCM, i::Int, j::Int)

Return the probability that the dyad `(i,j)` holds a single non-reciprocated link `j→i` under the binary
(RBCM) layer of the CRWCM model `m`.
"""
p⭠(m::CRWCM, i::Int, j::Int) = p⭢(m, j, i)

"""
    p⭤(m::CRWCM, i::Int, j::Int)

Return the probability that the dyad `(i,j)` holds a reciprocated link pair under the binary (RBCM) layer
of the CRWCM model `m`.

❗ For performance reasons, the function does not check if the node pair is valid or if the conditional parameters have been computed.
"""
function p⭤(m::CRWCM, i::Int, j::Int)
    i == j && return zero(precision(m))
    @inbounds ri, rj = m.dᵣ_ind[i], m.dᵣ_ind[j]
    @inbounds zz = m.zᵣ[ri] * m.zᵣ[rj]
    @inbounds D = one(precision(m)) + m.xᵣ[ri] * m.yᵣ[rj] + m.xᵣ[rj] * m.yᵣ[ri] + zz
    return zz / D
end

"""
    A(m::CRWCM, i::Int, j::Int)

Return the expected value of the (binary) adjacency matrix for the CRWCM model `m` at the node pair `(i,j)`
(``⟨a_{ij}⟩ = p^{→}_{ij} + p^{↔}_{ij}``, from the RBCM layer).

❗ For performance reasons, the function does not check:
- if the node pair is valid.
- if the conditional parameters of the model have been computed.
"""
A(m::CRWCM, i::Int, j::Int) = p⭢(m, i, j) + p⭤(m, i, j)


"""
    Ĝ(m::CRWCM)

Compute the expected (binary) **adjacency** matrix for the CRWCM model `m` from its RBCM layer
(``⟨a_{ij}⟩ = p^{→}_{ij} + p^{↔}_{ij}``).

*Note*: under the RBCM layer the entries `aᵢⱼ` and `aⱼᵢ` are correlated within a dyad, so `Ĝ` alone does
not characterise the dyadic joint distribution. The expected weights can be computed with [`Ŵ`](@ref MaxEntropyGraphs.Ŵ).
"""
function Ĝ(m::CRWCM)
    m.status[:conditional_params_computed] ? nothing : throw(ArgumentError("The conditional parameters have not been computed yet"))

    n = m.status[:N]
    G = zeros(precision(m), n, n)
    x = m.xᵣ[m.dᵣ_ind]
    y = m.yᵣ[m.dᵣ_ind]
    z = m.zᵣ[m.dᵣ_ind]
    o = one(precision(m))
    for i = 1:n
        @simd for j = 1:n
            if i ≠ j
                @inbounds xiyj = x[i]*y[j]
                @inbounds zizj = z[i]*z[j]
                @inbounds D = o + xiyj + x[j]*y[i] + zizj
                @inbounds G[i,j] = (xiyj + zizj) / D
            end
        end
    end

    return G
end

"""
    set_Ĝ!(m::CRWCM)

Set the expected (binary) adjacency matrix for the CRWCM model `m`.
"""
function set_Ĝ!(m::CRWCM)
    m.Ĝ = Ĝ(m)
    m.status[:G_computed] = true
    return m.Ĝ
end


"""
    Ŵ(m::CRWCM)

Compute the expected (unconditional) **weighted adjacency** matrix for the CRWCM model `m`, i.e.
``⟨w_{ij}⟩ = f^{→}_{ij}/(θ^{→}_i + θ^{←}_j) + f^{↔}_{ij}/(θ^{↔,o}_i + θ^{↔,i}_j)``, so that
`sum(Ŵ(m), dims=2) ≈ s→ + s↔out` (the total out-strength) and `sum(Ŵ(m), dims=1) ≈ s← + s↔in`.
"""
function Ŵ(m::CRWCM)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))

    n = m.status[:N]
    W = zeros(precision(m), n, n)
    x = m.xᵣ[m.dᵣ_ind]
    y = m.yᵣ[m.dᵣ_ind]
    z = m.zᵣ[m.dᵣ_ind]
    θ⭢  = @view m.θ[1:n]
    θ⭠  = @view m.θ[n+1:2*n]
    θ⭤ᵒ = @view m.θ[2*n+1:3*n]
    θ⭤ⁱ = @view m.θ[3*n+1:4*n]
    o = one(precision(m))
    for i = 1:n
        for j = 1:n
            if i ≠ j
                @inbounds xiyj = x[i]*y[j]
                @inbounds zizj = z[i]*z[j]
                @inbounds D = o + xiyj + x[j]*y[i] + zizj
                # dead channels have an exactly zero dyadic probability and an infinite rate; skip them
                # explicitly to avoid 0/Inf ambiguity in low precision
                @inbounds w = (iszero(xiyj) ? zero(precision(m)) : (xiyj / D) / (θ⭢[i] + θ⭠[j])) +
                              (iszero(zizj) ? zero(precision(m)) : (zizj / D) / (θ⭤ᵒ[i] + θ⭤ⁱ[j]))
                @inbounds W[i,j] = w
            end
        end
    end

    return W
end

"""
    set_Ŵ!(m::CRWCM)

Set the expected weighted adjacency matrix for the CRWCM model `m`.
"""
function set_Ŵ!(m::CRWCM)
    m.Ŵ = Ŵ(m)
    m.status[:W_computed] = true
    return m.Ŵ
end


"""
    σˣ(m::CRWCM)

Compute the standard deviation for the elements of the (binary) adjacency matrix for the CRWCM model `m`
(``σ[a_{ij}] = \\sqrt{⟨a_{ij}⟩(1-⟨a_{ij}⟩)}``, from the RBCM layer).

**Note:** the within-dyad correlation between `aᵢⱼ` and `aⱼᵢ` is captured separately and accounted for by
`σₓ`. Read as "sigma star".
"""
function σˣ(m::CRWCM)
    m.status[:conditional_params_computed] ? nothing : throw(ArgumentError("The conditional parameters have not been computed yet"))

    n = m.status[:N]
    σ = zeros(precision(m), n, n)
    x = m.xᵣ[m.dᵣ_ind]
    y = m.yᵣ[m.dᵣ_ind]
    z = m.zᵣ[m.dᵣ_ind]
    o = one(precision(m))
    for i = 1:n
        @simd for j = 1:n
            if i ≠ j
                @inbounds xiyj = x[i]*y[j]
                @inbounds zizj = z[i]*z[j]
                @inbounds D = o + xiyj + x[j]*y[i] + zizj
                @inbounds a = (xiyj + zizj) / D
                @inbounds σ[i,j] = sqrt(a * (o - a))
            end
        end
    end

    return σ
end

"""
    set_σ!(m::CRWCM)

Set the standard deviation for the elements of the (binary) adjacency matrix for the CRWCM model `m`.
"""
function set_σ!(m::CRWCM)
    m.σ = σˣ(m)
    m.status[:σ_computed] = true
    return m.σ
end


"""
    _cov_dyads(m::CRWCM)

Compute the within-dyad covariance matrix of the **binary** layer of the CRWCM model `m`
(``Cov(a_{ij}, a_{ji}) = p^{↔}_{ij} - ⟨a_{ij}⟩⟨a_{ji}⟩``, from the RBCM layer). Symmetric, zero diagonal.
"""
function _cov_dyads(m::CRWCM)
    m.status[:conditional_params_computed] ? nothing : throw(ArgumentError("The conditional parameters have not been computed yet"))
    n = m.status[:N]
    C = zeros(precision(m), n, n)
    x = m.xᵣ[m.dᵣ_ind]
    y = m.yᵣ[m.dᵣ_ind]
    z = m.zᵣ[m.dᵣ_ind]
    o = one(precision(m))
    for i = 1:n
        for j = i+1:n
            @inbounds xiyj = x[i]*y[j]
            @inbounds xjyi = x[j]*y[i]
            @inbounds zizj = z[i]*z[j]
            @inbounds D = o + xiyj + xjyi + zizj
            @inbounds aij = (xiyj + zizj) / D
            @inbounds aji = (xjyi + zizj) / D
            @inbounds c = zizj / D - aij * aji
            @inbounds C[i,j] = c
            @inbounds C[j,i] = c
        end
    end

    return C
end


"""
    σʷ(m::CRWCM)

Compute the standard deviation for the elements of the **weighted** adjacency matrix for the CRWCM model
`m`. The (unconditional) weight `wᵢⱼ` is a three-component mixture (zero; exponential with rate
`θ⭢ᵢ + θ⭠ⱼ` with probability `f⭢ᵢⱼ`; exponential with rate `θ⭤ᵒᵢ + θ⭤ⁱⱼ` with probability `f⭤ᵢⱼ`), so
``⟨w_{ij}^2⟩ = 2f^{→}_{ij}/(θ^{→}_i + θ^{←}_j)^2 + 2f^{↔}_{ij}/(θ^{↔,o}_i + θ^{↔,i}_j)^2`` and
``Var(w_{ij}) = ⟨w_{ij}^2⟩ - ⟨w_{ij}⟩^2``.

**Note:** within a dyad the weights `wᵢⱼ` and `wⱼᵢ` are correlated (see [`MaxEntropyGraphs._covʷ`](@ref));
`σₓ(m, X; layer=:weighted)` accounts for this.
"""
function σʷ(m::CRWCM)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))

    n = m.status[:N]
    σ = zeros(precision(m), n, n)
    x = m.xᵣ[m.dᵣ_ind]
    y = m.yᵣ[m.dᵣ_ind]
    z = m.zᵣ[m.dᵣ_ind]
    θ⭢  = @view m.θ[1:n]
    θ⭠  = @view m.θ[n+1:2*n]
    θ⭤ᵒ = @view m.θ[2*n+1:3*n]
    θ⭤ⁱ = @view m.θ[3*n+1:4*n]
    o = one(precision(m))
    for i = 1:n
        for j = 1:n
            if i ≠ j
                @inbounds xiyj = x[i]*y[j]
                @inbounds zizj = z[i]*z[j]
                @inbounds D = o + xiyj + x[j]*y[i] + zizj
                @inbounds w1 = iszero(xiyj) ? zero(precision(m)) : (xiyj / D) / (θ⭢[i] + θ⭠[j])
                @inbounds w2 = iszero(zizj) ? zero(precision(m)) : (zizj / D) / (θ⭤ᵒ[i] + θ⭤ⁱ[j])
                @inbounds m2 = iszero(xiyj) ? zero(precision(m)) : 2 * (xiyj / D) / (θ⭢[i] + θ⭠[j])^2
                @inbounds m2 += iszero(zizj) ? zero(precision(m)) : 2 * (zizj / D) / (θ⭤ᵒ[i] + θ⭤ⁱ[j])^2
                @inbounds σ[i,j] = sqrt(m2 - (w1 + w2)^2)
            end
        end
    end

    return σ
end

"""
    set_σʷ!(m::CRWCM)

Set the standard deviation for the elements of the weighted adjacency matrix for the CRWCM model `m`.
"""
function set_σʷ!(m::CRWCM)
    m.σʷ = σʷ(m)
    m.status[:σʷ_computed] = true
    return m.σʷ
end


"""
    _covʷ(m::CRWCM)

Compute the within-dyad covariance matrix of the **weighted** layer of the CRWCM model `m`. The weights
`wᵢⱼ` and `wⱼᵢ` are both non-zero only in the reciprocated dyad state (where they are conditionally
independent exponentials), so

``Cov(w_{ij}, w_{ji}) = \\frac{f^{↔}_{ij}}{(θ^{↔,o}_i + θ^{↔,i}_j)(θ^{↔,o}_j + θ^{↔,i}_i)} - ⟨w_{ij}⟩⟨w_{ji}⟩``.

Symmetric, zero diagonal. This term is what distinguishes the CRWCM's weighted uncertainty from the
[`DCReM`](@ref)'s (where it vanishes).
"""
function _covʷ(m::CRWCM)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    n = m.status[:N]
    C = zeros(precision(m), n, n)
    x = m.xᵣ[m.dᵣ_ind]
    y = m.yᵣ[m.dᵣ_ind]
    z = m.zᵣ[m.dᵣ_ind]
    θ⭢  = @view m.θ[1:n]
    θ⭠  = @view m.θ[n+1:2*n]
    θ⭤ᵒ = @view m.θ[2*n+1:3*n]
    θ⭤ⁱ = @view m.θ[3*n+1:4*n]
    o = one(precision(m))
    for i = 1:n
        for j = i+1:n
            @inbounds xiyj = x[i]*y[j]
            @inbounds xjyi = x[j]*y[i]
            @inbounds zizj = z[i]*z[j]
            @inbounds D = o + xiyj + xjyi + zizj
            @inbounds wij = (iszero(xiyj) ? zero(precision(m)) : (xiyj / D) / (θ⭢[i] + θ⭠[j])) +
                            (iszero(zizj) ? zero(precision(m)) : (zizj / D) / (θ⭤ᵒ[i] + θ⭤ⁱ[j]))
            @inbounds wji = (iszero(xjyi) ? zero(precision(m)) : (xjyi / D) / (θ⭢[j] + θ⭠[i])) +
                            (iszero(zizj) ? zero(precision(m)) : (zizj / D) / (θ⭤ᵒ[j] + θ⭤ⁱ[i]))
            @inbounds joint = iszero(zizj) ? zero(precision(m)) : (zizj / D) / ((θ⭤ᵒ[i] + θ⭤ⁱ[j]) * (θ⭤ᵒ[j] + θ⭤ⁱ[i]))
            c = joint - wij * wji
            @inbounds C[i,j] = c
            @inbounds C[j,i] = c
        end
    end

    return C
end


"""
    σₓ(m::CRWCM, X::Function; layer::Symbol=:binary, gradient_method::Symbol=:ReverseDiff)

Compute the standard deviation of metric `X` for the CRWCM model `m` via error propagation (the delta
method of Squartini & Garlaschelli (2011), Eq. B.16, including the within-dyad covariance cross-terms).

# Arguments
- `layer::Symbol`:
    - `:binary` (default): `X` is a function of the adjacency matrix; the gradient is evaluated at `m.Ĝ`,
      weighted by `m.σ`, plus the binary within-dyad covariance `Cov(aᵢⱼ, aⱼᵢ)` cross-term (requires
      `set_Ĝ!` and `set_σ!`).
    - `:weighted`: `X` is a function of the weight matrix; the gradient is evaluated at `m.Ŵ`, weighted by
      `m.σʷ`, plus the weighted within-dyad covariance `Cov(wᵢⱼ, wⱼᵢ)` cross-term (requires `set_Ŵ!` and
      `set_σʷ!`). This is the layer to use for weighted metrics such as the triadic fluxes.
- `gradient_method::Symbol`: `:ForwardDiff`, `:ReverseDiff` (default) or `:Zygote`.
"""
function σₓ(m::CRWCM, X::Function; layer::Symbol=:binary, gradient_method::Symbol=:ReverseDiff)
    if layer == :binary
        m.status[:G_computed] ? nothing : throw(ArgumentError("The expected values (m.Ĝ) must be computed for `m` before computing the standard deviation of metric `X`, see `set_Ĝ!`"))
        m.status[:σ_computed] ? nothing : throw(ArgumentError("The standard deviations (m.σ) must be computed for `m` before computing the standard deviation of metric `X`, see `set_σ!`"))
        M, S, C = m.Ĝ, m.σ, _cov_dyads(m)
    elseif layer == :weighted
        m.status[:W_computed] ? nothing : throw(ArgumentError("The expected weights (m.Ŵ) must be computed for `m` before computing the standard deviation of metric `X`, see `set_Ŵ!`"))
        m.status[:σʷ_computed] ? nothing : throw(ArgumentError("The weight standard deviations (m.σʷ) must be computed for `m` before computing the standard deviation of metric `X`, see `set_σʷ!`"))
        M, S, C = m.Ŵ, m.σʷ, _covʷ(m)
    else
        throw(ArgumentError("Invalid layer, only :binary and :weighted are accepted"))
    end

    if gradient_method == :ForwardDiff
        ∇X = ForwardDiff.gradient(X, M)
    elseif gradient_method == :ReverseDiff
        ∇X = ReverseDiff.gradient(X, M)
    elseif gradient_method == :Zygote
        ∇X = Zygote.gradient(X, M)[1]
    else
        throw(ArgumentError("Invalid gradient method, only :ForwardDiff, :ReverseDiff and :Zygote are accepted"))
    end

    # variance term + within-dyad covariance cross-term (both sums run over ordered pairs)
    return sqrt( sum((S .* ∇X) .^ 2) + sum(C .* ∇X .* transpose(∇X)) )
end


# ----------------------------------------------------------------------------------------------------------------------
# degree and strength accessors
# ----------------------------------------------------------------------------------------------------------------------

"""
    nonreciprocated_outdegree(m::CRWCM[, i]; method=:full)

Return the expected non-reciprocated out-degree of (node `i` of) the binary (RBCM) layer of the CRWCM
model `m` (``⟨k^{→}_i⟩ = \\sum_{j≠i} p^{→}_{ij}``).
"""
function nonreciprocated_outdegree(m::CRWCM, i::Int; method::Symbol=:full)
    m.status[:conditional_params_computed] ? nothing : throw(ArgumentError("The conditional parameters have not been computed yet"))
    i > m.status[:N] ? throw(ArgumentError("Attempted to access node $i in a $(m.status[:N]) node graph")) : nothing
    (method == :full || method == :reduced) || throw(ArgumentError("Unknown method $method"))
    return sum(p⭢(m, i, j) for j in 1:m.status[:N])
end
nonreciprocated_outdegree(m::CRWCM, v::Vector{Int}=collect(1:m.status[:N]); method::Symbol=:full) = [nonreciprocated_outdegree(m, i, method=method) for i in v]

"""
    nonreciprocated_indegree(m::CRWCM[, i]; method=:full)

Return the expected non-reciprocated in-degree of (node `i` of) the binary (RBCM) layer of the CRWCM
model `m` (``⟨k^{←}_i⟩ = \\sum_{j≠i} p^{←}_{ij}``).
"""
function nonreciprocated_indegree(m::CRWCM, i::Int; method::Symbol=:full)
    m.status[:conditional_params_computed] ? nothing : throw(ArgumentError("The conditional parameters have not been computed yet"))
    i > m.status[:N] ? throw(ArgumentError("Attempted to access node $i in a $(m.status[:N]) node graph")) : nothing
    (method == :full || method == :reduced) || throw(ArgumentError("Unknown method $method"))
    return sum(p⭠(m, i, j) for j in 1:m.status[:N])
end
nonreciprocated_indegree(m::CRWCM, v::Vector{Int}=collect(1:m.status[:N]); method::Symbol=:full) = [nonreciprocated_indegree(m, i, method=method) for i in v]

"""
    reciprocated_degree(m::CRWCM[, i]; method=:full)

Return the expected reciprocated degree of (node `i` of) the binary (RBCM) layer of the CRWCM model `m`
(``⟨k^{↔}_i⟩ = \\sum_{j≠i} p^{↔}_{ij}``).
"""
function reciprocated_degree(m::CRWCM, i::Int; method::Symbol=:full)
    m.status[:conditional_params_computed] ? nothing : throw(ArgumentError("The conditional parameters have not been computed yet"))
    i > m.status[:N] ? throw(ArgumentError("Attempted to access node $i in a $(m.status[:N]) node graph")) : nothing
    (method == :full || method == :reduced) || throw(ArgumentError("Unknown method $method"))
    return sum(p⭤(m, i, j) for j in 1:m.status[:N])
end
reciprocated_degree(m::CRWCM, v::Vector{Int}=collect(1:m.status[:N]); method::Symbol=:full) = [reciprocated_degree(m, i, method=method) for i in v]


"""
    nonreciprocated_outstrength(m::CRWCM[, i]; method=:full)

Return the expected non-reciprocated out-strength of (node `i` of) the CRWCM model `m`, i.e.
``⟨s^{→}_i⟩ = \\sum_{j≠i} f^{→}_{ij}/(θ^{→}_i + θ^{←}_j)``.
"""
function nonreciprocated_outstrength(m::CRWCM, i::Int; method::Symbol=:full)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    i > m.status[:N] ? throw(ArgumentError("Attempted to access node $i in a $(m.status[:N]) node graph")) : nothing
    (method == :full || method == :reduced) || throw(ArgumentError("Unknown method $method"))
    n = m.status[:N]
    θ⭢ = @view m.θ[1:n]
    θ⭠ = @view m.θ[n+1:2*n]
    res = zero(precision(m))
    @inbounds for j in 1:n
        if i ≠ j
            f = p⭢(m, i, j)
            iszero(f) || (res += f / (θ⭢[i] + θ⭠[j]))
        end
    end
    return res
end
nonreciprocated_outstrength(m::CRWCM, v::Vector{Int}=collect(1:m.status[:N]); method::Symbol=:full) = [nonreciprocated_outstrength(m, i, method=method) for i in v]

"""
    nonreciprocated_instrength(m::CRWCM[, i]; method=:full)

Return the expected non-reciprocated in-strength of (node `i` of) the CRWCM model `m`, i.e.
``⟨s^{←}_i⟩ = \\sum_{j≠i} f^{←}_{ij}/(θ^{←}_i + θ^{→}_j)``.
"""
function nonreciprocated_instrength(m::CRWCM, i::Int; method::Symbol=:full)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    i > m.status[:N] ? throw(ArgumentError("Attempted to access node $i in a $(m.status[:N]) node graph")) : nothing
    (method == :full || method == :reduced) || throw(ArgumentError("Unknown method $method"))
    n = m.status[:N]
    θ⭢ = @view m.θ[1:n]
    θ⭠ = @view m.θ[n+1:2*n]
    res = zero(precision(m))
    @inbounds for j in 1:n
        if i ≠ j
            f = p⭠(m, i, j) # = p⭢(m, j, i)
            iszero(f) || (res += f / (θ⭠[i] + θ⭢[j]))
        end
    end
    return res
end
nonreciprocated_instrength(m::CRWCM, v::Vector{Int}=collect(1:m.status[:N]); method::Symbol=:full) = [nonreciprocated_instrength(m, i, method=method) for i in v]

"""
    reciprocated_outstrength(m::CRWCM[, i]; method=:full)

Return the expected reciprocated out-strength of (node `i` of) the CRWCM model `m`, i.e.
``⟨s^{↔,out}_i⟩ = \\sum_{j≠i} f^{↔}_{ij}/(θ^{↔,o}_i + θ^{↔,i}_j)``.
"""
function reciprocated_outstrength(m::CRWCM, i::Int; method::Symbol=:full)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    i > m.status[:N] ? throw(ArgumentError("Attempted to access node $i in a $(m.status[:N]) node graph")) : nothing
    (method == :full || method == :reduced) || throw(ArgumentError("Unknown method $method"))
    n = m.status[:N]
    θ⭤ᵒ = @view m.θ[2*n+1:3*n]
    θ⭤ⁱ = @view m.θ[3*n+1:4*n]
    res = zero(precision(m))
    @inbounds for j in 1:n
        if i ≠ j
            f = p⭤(m, i, j)
            iszero(f) || (res += f / (θ⭤ᵒ[i] + θ⭤ⁱ[j]))
        end
    end
    return res
end
reciprocated_outstrength(m::CRWCM, v::Vector{Int}=collect(1:m.status[:N]); method::Symbol=:full) = [reciprocated_outstrength(m, i, method=method) for i in v]

"""
    reciprocated_instrength(m::CRWCM[, i]; method=:full)

Return the expected reciprocated in-strength of (node `i` of) the CRWCM model `m`, i.e.
``⟨s^{↔,in}_i⟩ = \\sum_{j≠i} f^{↔}_{ij}/(θ^{↔,i}_i + θ^{↔,o}_j)``.
"""
function reciprocated_instrength(m::CRWCM, i::Int; method::Symbol=:full)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    i > m.status[:N] ? throw(ArgumentError("Attempted to access node $i in a $(m.status[:N]) node graph")) : nothing
    (method == :full || method == :reduced) || throw(ArgumentError("Unknown method $method"))
    n = m.status[:N]
    θ⭤ᵒ = @view m.θ[2*n+1:3*n]
    θ⭤ⁱ = @view m.θ[3*n+1:4*n]
    res = zero(precision(m))
    @inbounds for j in 1:n
        if i ≠ j
            f = p⭤(m, i, j)
            iszero(f) || (res += f / (θ⭤ⁱ[i] + θ⭤ᵒ[j]))
        end
    end
    return res
end
reciprocated_instrength(m::CRWCM, v::Vector{Int}=collect(1:m.status[:N]); method::Symbol=:full) = [reciprocated_instrength(m, i, method=method) for i in v]


"""
    outstrength(m::CRWCM[, i]; method=:full)

Return the expected total out-strength of (node `i` of) the CRWCM model `m`
(``⟨s^{out}_i⟩ = ⟨s^{→}_i⟩ + ⟨s^{↔,out}_i⟩``, the row sum of [`Ŵ`](@ref MaxEntropyGraphs.Ŵ)).
"""
outstrength(m::CRWCM, i::Int; method::Symbol=:full) = nonreciprocated_outstrength(m, i, method=method) + reciprocated_outstrength(m, i, method=method)
outstrength(m::CRWCM, v::Vector{Int}=collect(1:m.status[:N]); method::Symbol=:full) = [outstrength(m, i, method=method) for i in v]

"""
    instrength(m::CRWCM[, i]; method=:full)

Return the expected total in-strength of (node `i` of) the CRWCM model `m`
(``⟨s^{in}_i⟩ = ⟨s^{←}_i⟩ + ⟨s^{↔,in}_i⟩``, the column sum of [`Ŵ`](@ref MaxEntropyGraphs.Ŵ)).
"""
instrength(m::CRWCM, i::Int; method::Symbol=:full) = nonreciprocated_instrength(m, i, method=method) + reciprocated_instrength(m, i, method=method)
instrength(m::CRWCM, v::Vector{Int}=collect(1:m.status[:N]); method::Symbol=:full) = [instrength(m, i, method=method) for i in v]


"""
    AIC(m::CRWCM)

Compute the Akaike Information Criterion (AIC) for the CRWCM model `m`. The parameters of the model must
be computed beforehand. The (conditional) CRWCM has `4N` parameters (four rate parameters ``θ`` per node;
the binary layer's dyadic probabilities are fixed inputs of the conditional likelihood).

See also [`AICc`](@ref MaxEntropyGraphs.AICc), [`L_CRWCM`](@ref MaxEntropyGraphs.L_CRWCM).
"""
function AIC(m::CRWCM)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    k = 4 * m.status[:N] # number of parameters (four θ per node)
    n = (m.status[:N] - 1) * m.status[:N] # number of observations (ordered node pairs)
    L = L_CRWCM(m) # log-likelihood

    if n/k < 40
        @warn """The number of observations is small with respect to the number of parameters (n/k < 40). Consider using the corrected AIC (AICc) instead."""
    end

    return 2*k - 2*L
end


"""
    AICc(m::CRWCM)

Compute the corrected Akaike Information Criterion (AICc) for the CRWCM model `m`.

See also [`AIC`](@ref MaxEntropyGraphs.AIC), [`L_CRWCM`](@ref MaxEntropyGraphs.L_CRWCM).
"""
function AICc(m::CRWCM)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    k = 4 * m.status[:N] # number of parameters
    n = (m.status[:N] - 1) * m.status[:N] # number of observations
    L = L_CRWCM(m) # log-likelihood

    return 2*k - 2*L + (2*k*(k+1)) / (n - k - 1)
end


"""
    BIC(m::CRWCM)

Compute the Bayesian Information Criterion (BIC) for the CRWCM model `m`.

See also [`AIC`](@ref MaxEntropyGraphs.AIC), [`L_CRWCM`](@ref MaxEntropyGraphs.L_CRWCM).
"""
function BIC(m::CRWCM)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    k = 4 * m.status[:N] # number of parameters
    n = (m.status[:N] - 1) * m.status[:N] # number of observations
    L = L_CRWCM(m) # log-likelihood

    return k * log(n) - 2*L
end


"""
    rand(m::CRWCM; precomputed=false, rng=Random.default_rng())

Generate a random weighted directed graph from the CRWCM model `m`.

Each dyad `(i,j)` is drawn from its four-state RBCM distribution (single link `i→j`, single link `j→i`,
reciprocated pair, absent); the realised links then get **continuous** exponential weights with the rates
of the corresponding channel (a reciprocated pair gets two conditionally independent weights).

# Arguments
- `precomputed::Bool`: not implemented for the CRWCM (the parameters are always used to generate the graph on the fly).
- `rng::AbstractRNG`: random number generator to use (defaults to `Random.default_rng()`).

# Examples
```jldoctest
julia> model = CRWCM(MaxEntropyGraphs.rhesus_macaques()); # generate a CRWCM model

julia> solve_model!(model); # compute the maximum likelihood parameters

julia> sample = rand(model); # sample a random weighted directed graph

julia> typeof(sample)
SimpleWeightedGraphs.SimpleWeightedDiGraph{Int64, Float64}
```
"""
function rand(m::CRWCM; precomputed::Bool=false, rng::AbstractRNG=default_rng())
    precomputed && throw(ArgumentError("This function is not implemented for CRWCM models (Ĝ does not capture the dyadic joint distribution)"))
    # check if possible to use parameters
    m.status[:conditional_params_computed] ? nothing : throw(ArgumentError("The conditional parameters have not been computed yet"))
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    # full per-node binary fitnesses + weighted parameters
    n = m.status[:N]
    x = m.xᵣ[m.dᵣ_ind]
    y = m.yᵣ[m.dᵣ_ind]
    z = m.zᵣ[m.dᵣ_ind]
    θ⭢  = @view m.θ[1:n]
    θ⭠  = @view m.θ[n+1:2*n]
    θ⭤ᵒ = @view m.θ[2*n+1:3*n]
    θ⭤ⁱ = @view m.θ[3*n+1:4*n]
    o = one(precision(m))
    # generate random graph edges per dyad
    sources = Vector{Int}();
    targets = Vector{Int}();
    weights = Vector{Float64}();
    for i in 1:n
        for j in i+1:n
            @inbounds xiyj = x[i]*y[j]
            @inbounds xjyi = x[j]*y[i]
            @inbounds zizj = z[i]*z[j]
            D = o + xiyj + xjyi + zizj
            u = rand(rng) * D # single uniform draw scaled by the normaliser
            if u < xiyj # single link i→j
                push!(sources, i); push!(targets, j)
                @inbounds push!(weights, rand(rng, Exponential(1 / (θ⭢[i] + θ⭠[j]))))
            elseif u < xiyj + xjyi # single link j→i
                push!(sources, j); push!(targets, i)
                @inbounds push!(weights, rand(rng, Exponential(1 / (θ⭢[j] + θ⭠[i]))))
            elseif u < xiyj + xjyi + zizj # reciprocated pair (two conditionally independent weights)
                push!(sources, i); push!(targets, j)
                @inbounds push!(weights, rand(rng, Exponential(1 / (θ⭤ᵒ[i] + θ⭤ⁱ[j]))))
                push!(sources, j); push!(targets, i)
                @inbounds push!(weights, rand(rng, Exponential(1 / (θ⭤ᵒ[j] + θ⭤ⁱ[i]))))
            end
        end
    end

    if length(sources) ≠ 0
        G = SimpleWeightedGraphs.SimpleWeightedDiGraph(sources, targets, weights)
    else
        G = SimpleWeightedGraphs.SimpleWeightedDiGraph(n)
    end

    # deal with edge case where no edges are generated for the last node(s) in the graph
    while Graphs.nv(G) < n
        Graphs.add_vertex!(G)
    end
    return G
end

"""
    rand(m::CRWCM, n::Int; precomputed=false, rng=Random.default_rng())

Generate `n` random weighted directed graphs from the CRWCM model `m`. If multithreading is available,
the graphs are generated in parallel; per-sample seeds are drawn from `rng` so the result is reproducible
and independent of the thread schedule.
"""
function rand(m::CRWCM, n::Int; precomputed::Bool=false, rng::AbstractRNG=default_rng())
    # pre-allocate
    res = Vector{SimpleWeightedGraphs.SimpleWeightedDiGraph}(undef, n)
    # per-sample seeds drawn from `rng` (reproducible, thread-schedule-independent)
    seeds = rand(rng, UInt64, n)
    # fill vector using threads
    Threads.@threads for i in 1:n
        res[i] = rand(m; precomputed=precomputed, rng=Xoshiro(seeds[i]))
    end

    return res
end


"""
    solve_model!(m::CRWCM; kwargs...)

Compute the likelihood maximising parameters of the CRWCM model `m`. This is a **two-step** process:
first the binary (conditional) RBCM layer is solved on the reciprocal degree sequences, then the weighted
CRWCM layer is solved on the four reciprocal strength sequences.

By default both layers are computed with the fixed-point method. Channels whose strength constraint is
zero have an undetermined parameter; after the solve those entries of `θ` are pinned to `+Inf` (an
infinite rate, i.e. an exactly zero weight — consistent with the dyadic probability being zero).

# Arguments (weighted CRWCM layer)
- `method::Symbol`: solution method, `:fixedpoint` (default) or any of :$(join(keys(MaxEntropyGraphs.optimization_methods), ", :", " and :")).
- `initial::Symbol`: initial guess, `:strengths` (default), `:strengths_minor` or `:random`.
- `AD_method::Symbol`: autodiff method, any of :$(join(keys(MaxEntropyGraphs.AD_methods), ", :", " and :")) (defaults to `:AutoZygote`).
- `analytical_gradient::Bool`: use the analytical gradient instead of autodiff (defaults to `false`).
- `store_adjacency::Bool`: cache the (binary) expected adjacency matrix `m.Ĝ` (defaults to `false`).

# Arguments (binary conditional RBCM layer)
- `method_conditional::Symbol`: solution method for the binary layer (defaults to `:fixedpoint`).
- `initial_conditional::Symbol`: initial guess for the binary layer (defaults to `:degrees`).
- `AD_method_conditional::Symbol`: autodiff method for the binary layer (defaults to `:AutoZygote`).
- `analytical_gradient_conditional::Bool`: analytical gradient for the binary layer (defaults to `false`).

# Common settings
- `maxiters::Int`: maximum number of iterations (defaults to 1000).
- `verbose::Bool`: show log messages (defaults to false).
- `ftol::Union{Real, Nothing}`: tolerance for the fixedpoint method (defaults to `nothing`, i.e. 1e-8), applied to the weighted layer and to the binary layer when either is solved with `:fixedpoint` (passing it when *neither* layer uses it warns). On the weighted layer it is a **relative** strength tolerance (see below); on the binary layer it bounds the fixed-point increment in parameter space.
- `abstol`, `reltol`: absolute/relative tolerances for the optimisation methods (default `nothing`).
- `g_tol::Union{Number, Nothing}`: gradient tolerance for the gradient-based methods (maps to Optim's `g_abstol`, default `nothing`). The gradient of the weighted layer *is* its constraint residual in strength units, but `g_abstol` is a stopping criterion rather than a guarantee: Optim can also stop on its function or parameter convergence checks and report success without the gradient ever reaching `g_tol`. Verify what was actually achieved with [`constraint_residual`](@ref).

!!! note "`ftol` is a relative strength tolerance on the weighted layer"
    The weighted layer is solved in **log-parameter** space (see `MaxEntropyGraphs._logspace_fixedpoint`),
    over the live channels only. The fixed-point map obeys ``G_i = θ_i⟨s_i⟩/s_i`` exactly, so the
    log-space increment is exactly ``\\log(⟨s_i⟩/s_i)``, and `ftol` therefore bounds the **relative**
    strength residual ``|⟨s_i⟩/s_i - 1|`` on each of the four channels. That makes it invariant under a
    rescaling of the weights: `ftol=1e-8` means eight significant digits on every strength whatever the
    units. (Solving in `θ` directly would instead bound ``|G_i - θ_i| = (θ_i/s_i)|⟨s_i⟩ - s_i|``, whose
    conversion factor ``s_i/θ_i`` grows as the *square* of the weight scale.) Use
    [`constraint_residual`](@ref) to measure the achieved residual in either absolute or relative form.

# Examples
```jldoctest CRWCM_solve
# default use
julia> model = CRWCM(MaxEntropyGraphs.rhesus_macaques());

julia> solve_model!(model);

```
"""
function solve_model!(m::CRWCM; # weighted (CRWCM) layer settings
                                method::Symbol=:fixedpoint,
                                initial::Symbol=:strengths,
                                AD_method::Symbol=:AutoZygote,
                                analytical_gradient::Bool=false,
                                store_adjacency::Bool=false,
                                # binary (conditional RBCM) layer settings
                                method_conditional::Symbol=:fixedpoint,
                                initial_conditional::Symbol=:degrees,
                                AD_method_conditional::Symbol=:AutoZygote,
                                analytical_gradient_conditional::Bool=false,
                                # common settings
                                maxiters::Int=1000,
                                verbose::Bool=false,
                                ftol::Union{Real, Nothing}=nothing,
                                abstol::Union{Number, Nothing}=nothing,
                                reltol::Union{Number, Nothing}=nothing,
                                g_tol::Union{Number, Nothing}=nothing)
    N = precision(m)
    N <: Union{Float16, Float32} && @warn "Solving in $(N) precision is experimental and may not converge; low precision is intended for storage. Consider Float64 for the solve." maxlog=1
    n = m.status[:N]
    # `ftol` reaches the fixed point solver of either layer, so it is only truly unused when neither
    # layer uses it: say so rather than ignoring it silently (only when it was actually passed)
    method ≠ :fixedpoint && method_conditional ≠ :fixedpoint && !isnothing(ftol) && @warn _ftol_unused_msg(method) maxlog=1
    ftol = isnothing(ftol) ? _DEFAULT_FTOL : ftol

    ## Part 1 - conditional binary layer (RBCM on the reciprocal degree sequences)
    cond_model = RBCM(d_out = m.d_out, d_in = m.d_in, d_rec = m.d_rec, precision = N)
    # only hand `ftol` down when the binary layer can act on it, so that it never warns on our behalf
    solve_model!(cond_model, method=method_conditional, initial=initial_conditional,
                             AD_method=AD_method_conditional, analytical_gradient=analytical_gradient_conditional,
                             maxiters=maxiters, ftol=(method_conditional == :fixedpoint ? ftol : nothing),
                             abstol=abstol, reltol=reltol, verbose=verbose)
    m.αᵣ .= cond_model.θᵣ
    m.xᵣ .= cond_model.xᵣ
    m.yᵣ .= cond_model.yᵣ
    m.zᵣ .= cond_model.zᵣ
    m.status[:conditional_params_computed] = true
    if store_adjacency
        set_Ĝ!(m)
    end

    ## Part 2 - weighted CRWCM layer
    θ₀ = initial_guess(m, method=initial)
    # dead channels (zero strength ⟺ zero dyadic probability everywhere): parameter undetermined,
    # kept at zero during the solve and pinned to +Inf afterwards
    ind_dead = findall(iszero, vcat(m.s_out .> 0, m.s_in .> 0, m.s_rec_out .> 0, m.s_rec_in .> 0))
    # full per-node binary fitnesses (used for the on-the-fly dyadic probabilities)
    x = m.xᵣ[m.dᵣ_ind]
    y = m.yᵣ[m.dᵣ_ind]
    z = m.zᵣ[m.dᵣ_ind]
    if method == :fixedpoint
        G_buffer = zeros(N, length(θ₀))
        FP_model! = (θ::Vector) -> CRWCM_iter!(θ, m.s_out, m.s_in, m.s_rec_out, m.s_rec_in, m.s_out_nz, m.s_in_nz, m.s_rec_nz, x, y, z, G_buffer)
        # solve in log-parameter space, where the increment is the *relative* strength residual and
        # `ftol` is scale-invariant (see `_logspace_fixedpoint`). Only the live channels are solved for:
        # a dead channel has no finite θ (`log(0) = -Inf`), and since its dyadic probabilities vanish
        # identically it never enters a live row, so holding it at its zero θ₀ leaves them untouched.
        ind_live = vcat(m.s_out_nz, n .+ m.s_in_nz, 2*n .+ m.s_rec_nz, 3*n .+ m.s_rec_nz)
        θ_sol, sol = _logspace_fixedpoint(FP_model!, θ₀, ind_live, ftol, maxiters)
        if NLsolve.converged(sol)
            verbose && @info "Fixed point iteration converged after $(sol.iterations) iterations"
            m.θ .= θ_sol
            m.θ[ind_dead] .= Inf
            m.status[:params_computed] = true
        else
            throw(ConvergenceError(method, nothing))
        end
    else
        if analytical_gradient
            grad! = (G, θ, p) -> ∇L_CRWCM_minus!(G, θ, m.s_out, m.s_in, m.s_rec_out, m.s_rec_in, m.s_out_nz, m.s_in_nz, m.s_rec_nz, x, y, z)
        end
        # objective (negative generalised log-likelihood)
        Lobj = (θ, p) -> -L_CRWCM(θ, m.s_out, m.s_in, m.s_rec_out, m.s_rec_in, m.s_out_nz, m.s_in_nz, m.s_rec_nz, x, y, z)
        f = AD_method ∈ keys(AD_methods) ? Optimization.OptimizationFunction(Lobj, AD_methods[AD_method], grad = analytical_gradient ? grad! : nothing) : throw(ArgumentError("The AD method $(AD_method) is not supported (yet)"))
        prob = Optimization.OptimizationProblem(f, θ₀)
        method ∈ keys(optimization_methods) || throw(ArgumentError("The method $(method) is not supported (yet)"))
        # use the BackTracking-line-search variants (see `CReM_optimization_methods`), falling back to
        # the package-wide optimizer for any method without one — the direct-rate positivity barrier requires it.
        opt = get(CReM_optimization_methods, method, optimization_methods[method])
        solve_kwargs = isnothing(g_tol) ? (; maxiters = maxiters, abstol = abstol, reltol = reltol) :
                                          (; maxiters = maxiters, abstol = abstol, reltol = reltol, g_abstol = g_tol)
        sol = Optimization.solve(prob, opt; solve_kwargs...)
        if Optimization.SciMLBase.successful_retcode(sol.retcode)
            verbose && @info """$(method) optimisation converged after $(@sprintf("%1.2e", sol.stats.time)) seconds (Optimization.jl return code: $("$(sol.retcode)"))"""
            m.θ .= sol.u
            m.θ[ind_dead] .= Inf
            m.status[:params_computed] = true
        else
            throw(ConvergenceError(method, sol.retcode))
        end
    end

    return m, sol
end
