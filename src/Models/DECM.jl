
"""
    DECM

Maximum entropy model for the Directed Enhanced Configuration Model (DECM).

The object holds the maximum likelihood parameters of the model (θ = [α_out; α_in; β_out; β_in]), the expected
adjacency matrix (Ĝ), the expected weight matrix (Ŵ), and the variance of their elements (σ, σʷ).

The DECM constrains, per node, both the out- and in-degree sequence and the (integer) out- and in-strength
sequence of a directed weighted network.

Note: this requires that the weights only assume (non-negative) integer values.
"""
mutable struct DECM{T<:Union{Graphs.AbstractGraph, Nothing}, N<:Real} <: AbstractMaxEntropyModel
    "Graph type, can be any subtype of AbstractGraph, but will be converted to SimpleWeightedDiGraph for the computation" # can also be empty
    const G::T
    "Maximum likelihood parameters for reduced model ([α_out; α_in; β_out; β_in])"
    const θᵣ::Vector{N}
    "Exponentiated maximum likelihood parameters for reduced model ( xᵢ_out = exp(-α_out,ᵢ) ) linked with out-degree"
    const xᵣ_out::Vector{N}
    "Exponentiated maximum likelihood parameters for reduced model ( xᵢ_in = exp(-α_in,ᵢ) ) linked with in-degree"
    const xᵣ_in::Vector{N}
    "Exponentiated maximum likelihood parameters for reduced model ( yᵢ_out = exp(-β_out,ᵢ) ) linked with out-strength"
    const yᵣ_out::Vector{N}
    "Exponentiated maximum likelihood parameters for reduced model ( yᵢ_in = exp(-β_in,ᵢ) ) linked with in-strength"
    const yᵣ_in::Vector{N}
    "Outdegree sequence of the graph"
    const d_out::Vector{Int}
    "Indegree sequence of the graph"
    const d_in::Vector{Int}
    "Outstrength sequence of the graph"
    const s_out::Vector{Int}
    "Instrength sequence of the graph"
    const s_in::Vector{Int}
    "Reduced outdegree sequence of the graph"
    const dᵣ_out::Vector{Int}
    "Reduced indegree sequence of the graph"
    const dᵣ_in::Vector{Int}
    "Reduced outstrength sequence of the graph"
    const sᵣ_out::Vector{Int}
    "Reduced instrength sequence of the graph"
    const sᵣ_in::Vector{Int}
    "Frequency of each (outdegree, indegree, outstrength, instrength) quadruple in the graph"
    const f::Vector{Int}
    "Indices to reconstruct the constraint sequences from the reduced sequences"
    const d_ind::Vector{Int}
    "Indices to reconstruct the reduced sequences from the constraint sequences"
    const dᵣ_ind::Vector{Int}
    "Indices of the reduced classes with a live out-channel (non-zero out-degree)"
    const dᵣ_out_nz::Vector{Int}
    "Indices of the reduced classes with a live in-channel (non-zero in-degree)"
    const dᵣ_in_nz::Vector{Int}
    "Expected adjacency matrix" # not always computed/required
    Ĝ::Union{Nothing, Matrix{N}}
    "Variance of the expected adjacency matrix" # not always computed/required
    σ::Union{Nothing, Matrix{N}}
    "Expected weighted adjacency matrix" # not always computed/required
    Ŵ::Union{Nothing, Matrix{N}}
    "Standard deviation of the expected weighted adjacency matrix" # not always computed/required
    σʷ::Union{Nothing, Matrix{N}}
    "Status indicators: parameters computed, expected adjacency matrix computed, variance computed, etc."
    const status::Dict{Symbol, Any}
    "Function used to computed the log-likelihood of the (reduced) model"
    fun::Union{Nothing, Function}
end

Base.show(io::IO, m::DECM{T,N}) where {T,N} = print(io, """DECM{$(T), $(N)} ($(m.status[:N]) vertices, $(m.status[:d_unique]) unique {out-degree, in-degree, out-strength, in-strength} quadruples, $(@sprintf("%.2f", m.status[:cᵣ])) compression ratio)""")

"""Return the reduced number of {out-degree, in-degree, out-strength, in-strength} quadruples in the DECM network"""
Base.length(m::DECM) = length(m.dᵣ_out)


"""
    DECM(G::T; d_out::Vector, d_in::Vector, s_out::Vector, s_in::Vector, precision::Type{<:AbstractFloat}=Float64, kwargs...) where {T}

Constructor function for the `DECM` type.

By default and depending on the graph type `T`, the definition of in- and out-degree from ``Graphs.jl`` and
in- and out-strength from ``SimpleWeightedGraphs`` is applied. If you want to use different definitions, you
can pass the sequences as keyword arguments (`d_out`, `d_in`, `s_out`, `s_in`).

If you want to generate a model directly from the degree and strength sequences without an underlying graph,
you can simply pass them all as keyword arguments. If you want to work from an adjacency/weight matrix, or an
edge list, you can use the (weighted) graph constructors from the ``JuliaGraphs`` ecosystem.

The strength sequences must be integer-valued (the DECM is defined for non-negative integer weights).

# Examples
```jldoctest DECM_creation
# generating a model from a directed weighted graph (the rhesus macaques grooming network)
julia> G = MaxEntropyGraphs.rhesus_macaques();

julia> model = DECM(G)
DECM{SimpleWeightedGraphs.SimpleWeightedDiGraph{Int64, Float64}, Float64} (16 vertices, 16 unique {out-degree, in-degree, out-strength, in-strength} quadruples, 1.00 compression ratio)

```
```jldoctest DECM_creation
# generating a model directly from the degree and strength sequences
julia> model = DECM(d_out=[1, 1, 2, 1], d_in=[1, 2, 1, 1], s_out=[3, 3, 5, 2], s_in=[3, 5, 3, 2])
DECM{Nothing, Float64} (4 vertices, 4 unique {out-degree, in-degree, out-strength, in-strength} quadruples, 1.00 compression ratio)

```
```jldoctest DECM_creation
# generating a model with a different precision
julia> model = DECM(d_out=[1, 1, 2, 1], d_in=[1, 2, 1, 1], s_out=[3, 3, 5, 2], s_in=[3, 5, 3, 2], precision=Float32);

julia> MaxEntropyGraphs.precision(model)
Float32

```

The degrees are taken from `Graphs.outdegree`/`Graphs.indegree` and the strengths from
`MaxEntropyGraphs.outstrength`/`MaxEntropyGraphs.instrength` (the `SimpleWeightedGraphs` weighted degrees).
"""
function DECM(G::T; d_out::Vector=Graphs.outdegree(G), d_in::Vector=Graphs.indegree(G),
                    s_out::Vector=outstrength(G),      s_in::Vector=instrength(G),
                    precision::Type{<:AbstractFloat}=Float64, kwargs...) where {T}
    T <: Union{Graphs.AbstractGraph, Nothing} ? nothing : throw(TypeError(:DECM, "G must be a subtype of AbstractGraph or Nothing", Union{Graphs.AbstractGraph, Nothing}, T))
    # coherence checks
    if T <: Graphs.AbstractGraph # Graph specific checks
        if !Graphs.is_directed(G)
            @warn "The graph is undirected, while the DECM model is directed, the out- and in-constraints will be identical"
        end

        Graphs.nv(G) == 0 ? throw(ArgumentError("The graph is empty")) : nothing
        Graphs.nv(G) == 1 ? throw(ArgumentError("The graph has only one vertex")) : nothing

        Graphs.nv(G) != length(d_out) ? throw(DimensionMismatch("The number of vertices in the graph ($(Graphs.nv(G))) and the length of the outdegree sequence ($(length(d_out))) do not match")) : nothing
        Graphs.nv(G) != length(d_in)  ? throw(DimensionMismatch("The number of vertices in the graph ($(Graphs.nv(G))) and the length of the indegree sequence ($(length(d_in))) do not match")) : nothing
        Graphs.nv(G) != length(s_out) ? throw(DimensionMismatch("The number of vertices in the graph ($(Graphs.nv(G))) and the length of the outstrength sequence ($(length(s_out))) do not match")) : nothing
        Graphs.nv(G) != length(s_in)  ? throw(DimensionMismatch("The number of vertices in the graph ($(Graphs.nv(G))) and the length of the instrength sequence ($(length(s_in))) do not match")) : nothing
    end

    # coherence checks specific to the degree/strength sequences
    (iszero(length(d_out)) || iszero(length(d_in)) || iszero(length(s_out)) || iszero(length(s_in))) ? throw(ArgumentError("The degree/strength sequences are empty")) : nothing
    (length(d_out) == 1 || length(d_in) == 1 || length(s_out) == 1 || length(s_in) == 1) ? throw(ArgumentError("The degree/strength sequences only contain a single node")) : nothing
    (length(d_out) == length(d_in) == length(s_out) == length(s_in)) ? nothing : throw(DimensionMismatch("The dimensions of the degree ($(length(d_out)), $(length(d_in))) and the strength sequences ($(length(s_out)), $(length(s_in))) do not match"))
    any(!isinteger, d_out) ? throw(DomainError("Some of the outdegree values are not integers, this is not allowed"))   : nothing
    any(!isinteger, d_in)  ? throw(DomainError("Some of the indegree values are not integers, this is not allowed"))    : nothing
    any(!isinteger, s_out) ? throw(DomainError("Some of the outstrength values are not integers, this is not allowed")) : nothing
    any(!isinteger, s_in)  ? throw(DomainError("Some of the instrength values are not integers, this is not allowed"))  : nothing
    maximum(d_out) >= length(d_out) ? throw(DomainError("The maximum outdegree in the graph is greater or equal to the number of vertices, this is not allowed")) : nothing
    maximum(d_in)  >= length(d_in)  ? throw(DomainError("The maximum indegree in the graph is greater or equal to the number of vertices, this is not allowed"))  : nothing

    if any(iszero, d_out)
        @warn "The graph has vertices with zero outdegree, this may lead to convergence issues."
    end
    if any(iszero, d_in)
        @warn "The graph has vertices with zero indegree, this may lead to convergence issues."
    end
    if any(iszero, s_out)
        @warn "The graph has vertices with zero outstrength, this may lead to convergence issues."
    end
    if any(iszero, s_in)
        @warn "The graph has vertices with zero instrength, this may lead to convergence issues."
    end
    # feasibility of the combined constraints (integer weights ≥ 1 per edge)
    if any(iszero.(d_out) .⊻ iszero.(s_out))
        @warn "The out-degree and out-strength sequences are inconsistent (one is zero where the other is not); no integer-weighted graph satisfies them, so the model is infeasible."
    end
    if any(iszero.(d_in) .⊻ iszero.(s_in))
        @warn "The in-degree and in-strength sequences are inconsistent (one is zero where the other is not); no integer-weighted graph satisfies them, so the model is infeasible."
    end
    if any(s_out .< d_out)
        @warn "Some out-strengths are smaller than the matching out-degrees; every edge carries an integer weight ≥ 1, so the model is infeasible."
    end
    if any(s_in .< d_in)
        @warn "Some in-strengths are smaller than the matching in-degrees; every edge carries an integer weight ≥ 1, so the model is infeasible."
    end

    # field generation
    dsᵣ, d_ind, dᵣ_ind, f = np_unique_clone(collect(zip(d_out, d_in, s_out, s_in)), sorted=true)
    dᵣ_out = Int.([q[1] for q in dsᵣ])
    dᵣ_in  = Int.([q[2] for q in dsᵣ])
    sᵣ_out = Int.([q[3] for q in dsᵣ])
    sᵣ_in  = Int.([q[4] for q in dsᵣ])
    dᵣ_out_nz = findall(!iszero, dᵣ_out)
    dᵣ_in_nz  = findall(!iszero, dᵣ_in)
    θᵣ = Vector{precision}(undef, 4*length(dᵣ_out))
    xᵣ_out = Vector{precision}(undef, length(dᵣ_out))
    xᵣ_in  = Vector{precision}(undef, length(dᵣ_out))
    yᵣ_out = Vector{precision}(undef, length(dᵣ_out))
    yᵣ_in  = Vector{precision}(undef, length(dᵣ_out))
    status = Dict{Symbol, Any}( :params_computed=>false,            # are the parameters computed?
                                :G_computed=>false,                 # is the expected adjacency matrix computed and stored?
                                :σ_computed=>false,                 # is the standard deviation computed and stored?
                                :W_computed=>false,                 # is the expected weighted adjacency matrix computed and stored?
                                :σʷ_computed=>false,                # is the weight standard deviation computed and stored?
                                :cᵣ => length(dᵣ_out)/length(d_out),# compression ratio of the reduced model
                                :d_unique => length(dᵣ_out),        # number of unique quadruples in the reduced model
                                :N => length(d_out)                 # number of vertices in the original graph
                )

    return DECM{T,precision}(G, θᵣ, xᵣ_out, xᵣ_in, yᵣ_out, yᵣ_in,
                             Int.(d_out), Int.(d_in), Int.(s_out), Int.(s_in),
                             dᵣ_out, dᵣ_in, sᵣ_out, sᵣ_in,
                             f, d_ind, dᵣ_ind, dᵣ_out_nz, dᵣ_in_nz,
                             nothing, nothing, nothing, nothing, status, nothing)
end

DECM(; d_out::Vector, d_in::Vector, s_out::Vector, s_in::Vector, precision::Type{<:AbstractFloat}=Float64, kwargs...) = DECM(nothing, d_out=d_out, d_in=d_in, s_out=s_out, s_in=s_in, precision=precision, kwargs...)


"""
    L_DECM_reduced(θ::AbstractVector, d_out::Vector, d_in::Vector, s_out::Vector, s_in::Vector, F::Vector, n::Int=length(d_out))

Compute the log-likelihood of the reduced DECM model using the exponential formulation in order to maintain convexity.

# Arguments
- `θ`: the maximum likelihood parameters of the model (`θ = [α_out; α_in; β_out; β_in]`)
- `d_out`: the reduced outdegree sequence
- `d_in`: the reduced indegree sequence
- `s_out`: the reduced outstrength sequence
- `s_in`: the reduced instrength sequence
- `F`: the frequency of each `(outdegree, indegree, outstrength, instrength)` quadruple
- `n`: the number of unique quadruples (defaults to `length(d_out)`)

The interaction term runs over the **ordered** pairs of reduced classes with multiplier `F[i]·(F[j] - (i==j))`
(a directed network has both `(i,j)` and `(j,i)` node pairs). The function is numerically stabilised
(`expm1`/`log1p`) so that the `1 - exp(-β_out,ᵢ-β_in,ⱼ)` denominator does not suffer from catastrophic
cancellation, and stays automatic-differentiation friendly.

# Examples
```jldoctest
julia> θ = collect(range(0.1, step=0.1, length=12));

julia> d_out = [1, 2, 1]; d_in = [2, 1, 1]; s_out = [2, 3, 1]; s_in = [3, 2, 1]; F = [1, 1, 1];

julia> L_DECM_reduced(θ, d_out, d_in, s_out, s_in, F);

```
"""
function L_DECM_reduced(θ::AbstractVector, d_out::Vector, d_in::Vector, s_out::Vector, s_in::Vector, F::Vector, n::Int=length(d_out))
    # split the parameters
    α_out = @view θ[1:n]
    α_in  = @view θ[n+1:2*n]
    β_out = @view θ[2*n+1:3*n]
    β_in  = @view θ[3*n+1:end]
    # initiate
    res = zero(eltype(θ))
    # compute
    for i in eachindex(d_out)
        @inbounds res -= F[i] * (d_out[i] * α_out[i] + d_in[i] * α_in[i] + s_out[i] * β_out[i] + s_in[i] * β_in[i])
        # ordered pairs (i→j): weight Fᵢ·(Fⱼ - (i==j)) (a class pairs with itself Fᵢ(Fᵢ-1) times)
        acc = zero(eltype(θ))
        @inbounds for j in eachindex(d_in)
            w = F[j] - (i == j)
            iszero(w) && continue                       # singleton diagonal: no pair exists, so no domain constraint either
            c1    = exp(-α_out[i] - α_in[j])
            c2    = exp(-β_out[i] - β_in[j])
            om_c2 = -expm1(-(β_out[i] + β_in[j]))       # == 1 - c2, computed without cancellation
            # the DECM is only defined for y_out,ᵢ·y_in,ⱼ < 1 (i.e. om_c2 > 0). Returning NaN for the whole
            # out-of-domain region (om_c2 ≤ 0) makes the line search reject those steps, instead of
            # letting the optimiser escape to β → -∞ where the objective is spuriously unbounded.
            acc  += w * (om_c2 > zero(om_c2) ? log1p(c1 * c2 / om_c2) : oftype(c1, NaN))
        end
        res -= F[i] * acc
    end

    return res
end

"""
    L_DECM_reduced(m::DECM)

Return the log-likelihood of the DECM model `m` based on the computed maximum likelihood parameters.

See also [`L_DECM_reduced(::AbstractVector, ::Vector, ::Vector, ::Vector, ::Vector, ::Vector)`](@ref)
"""
function L_DECM_reduced(m::DECM)
    if m.status[:params_computed]
        return L_DECM_reduced(m.θᵣ, m.dᵣ_out, m.dᵣ_in, m.sᵣ_out, m.sᵣ_in, m.f, m.status[:d_unique])
    else
        throw(ArgumentError("The parameters have not been computed yet"))
    end
end


"""
    ∇L_DECM_reduced!(∇L::AbstractVector, θ::AbstractVector, d_out::Vector, d_in::Vector, s_out::Vector, s_in::Vector, F::Vector, x_out::AbstractVector, x_in::AbstractVector, y_out::AbstractVector, y_in::AbstractVector, n=length(θ)÷4)

Compute the gradient of the log-likelihood of the reduced DECM model in a non-allocating manner.
The pre-allocated buffers `x_out`/`x_in` (`xᵢ = exp(-αᵢ)`) and `y_out`/`y_in` (`yᵢ = exp(-βᵢ)`) and the
gradient `∇L` are updated in place.

Each row `i` accumulates both orientations in a single inner loop: `i→j` feeds the out-blocks (`α_out`,
`β_out`) and `j→i` feeds the in-blocks (`α_in`, `β_in`). The inner loop is branch-free (the diagonal
correction is folded into the multiplier `F[j] - (i==j)`) so that it vectorises (`@simd`).

See also [`∇L_DECM_reduced_minus!`](@ref).
"""
function ∇L_DECM_reduced!(∇L::AbstractVector, θ::AbstractVector,
                          d_out::Vector, d_in::Vector, s_out::Vector, s_in::Vector,
                          F::Vector,
                          x_out::AbstractVector, x_in::AbstractVector,
                          y_out::AbstractVector, y_in::AbstractVector, n=length(θ)÷4)
    α_out = @view θ[1:n]
    α_in  = @view θ[n+1:2*n]
    β_out = @view θ[2*n+1:3*n]
    β_in  = @view θ[3*n+1:end]
    @inbounds @simd for i in eachindex(α_out) # to avoid the allocation of exp.(-θ)
        x_out[i] = exp(-α_out[i])
        x_in[i]  = exp(-α_in[i])
        y_out[i] = exp(-β_out[i])
        y_in[i]  = exp(-β_in[i])
    end

    for i in eachindex(α_out)
        @inbounds xᵢ_out = x_out[i]; xᵢ_in = x_in[i]
        @inbounds yᵢ_out = y_out[i]; yᵢ_in = y_in[i]
        accα_out = zero(eltype(∇L))
        accα_in  = zero(eltype(∇L))
        accβ_out = zero(eltype(∇L))
        accβ_in  = zero(eltype(∇L))
        @inbounds @simd for j in eachindex(α_out)
            w = F[j] - (i == j)                     # (F[j]-1) on the diagonal, F[j] elsewhere
            # orientation i→j (out-blocks of i)
            c1    = xᵢ_out * x_in[j]
            c2    = yᵢ_out * y_in[j]
            p     = (c1 * c2) / (1 + c1 * c2 - c2)
            accα_out += p * w
            accβ_out += (p / (1 - c2)) * w
            # orientation j→i (in-blocks of i)
            c1r   = x_out[j] * xᵢ_in
            c2r   = y_out[j] * yᵢ_in
            pr    = (c1r * c2r) / (1 + c1r * c2r - c2r)
            accα_in += pr * w
            accβ_in += (pr / (1 - c2r)) * w
        end
        @inbounds ∇L[i]       = -F[i] * d_out[i] + F[i] * accα_out
        @inbounds ∇L[i+n]     = -F[i] * d_in[i]  + F[i] * accα_in
        @inbounds ∇L[i+2*n]   = -F[i] * s_out[i] + F[i] * accβ_out
        @inbounds ∇L[i+3*n]   = -F[i] * s_in[i]  + F[i] * accβ_in
    end

    return ∇L
end


"""
    ∇L_DECM_reduced_minus!(args...)

Compute minus the gradient of the log-likelihood of the reduced DECM model (used for the minimisation carried
out by `Optimization.jl`). Non-allocating: updates the pre-allocated buffers `x_out`, `x_in`, `y_out`, `y_in`
and `∇L` in place.

See also [`∇L_DECM_reduced!`](@ref).
"""
function ∇L_DECM_reduced_minus!(∇L::AbstractVector, θ::AbstractVector,
                                d_out::Vector, d_in::Vector, s_out::Vector, s_in::Vector,
                                F::Vector,
                                x_out::AbstractVector, x_in::AbstractVector,
                                y_out::AbstractVector, y_in::AbstractVector, n=length(θ)÷4)
    α_out = @view θ[1:n]
    α_in  = @view θ[n+1:2*n]
    β_out = @view θ[2*n+1:3*n]
    β_in  = @view θ[3*n+1:end]
    @inbounds @simd for i in eachindex(α_out) # to avoid the allocation of exp.(-θ)
        x_out[i] = exp(-α_out[i])
        x_in[i]  = exp(-α_in[i])
        y_out[i] = exp(-β_out[i])
        y_in[i]  = exp(-β_in[i])
    end

    for i in eachindex(α_out)
        @inbounds xᵢ_out = x_out[i]; xᵢ_in = x_in[i]
        @inbounds yᵢ_out = y_out[i]; yᵢ_in = y_in[i]
        accα_out = zero(eltype(∇L))
        accα_in  = zero(eltype(∇L))
        accβ_out = zero(eltype(∇L))
        accβ_in  = zero(eltype(∇L))
        @inbounds @simd for j in eachindex(α_out)
            w = F[j] - (i == j)
            # orientation i→j (out-blocks of i)
            c1    = xᵢ_out * x_in[j]
            c2    = yᵢ_out * y_in[j]
            p     = (c1 * c2) / (1 + c1 * c2 - c2)
            accα_out += p * w
            accβ_out += (p / (1 - c2)) * w
            # orientation j→i (in-blocks of i)
            c1r   = x_out[j] * xᵢ_in
            c2r   = y_out[j] * yᵢ_in
            pr    = (c1r * c2r) / (1 + c1r * c2r - c2r)
            accα_in += pr * w
            accβ_in += (pr / (1 - c2r)) * w
        end
        @inbounds ∇L[i]       = F[i] * d_out[i] - F[i] * accα_out
        @inbounds ∇L[i+n]     = F[i] * d_in[i]  - F[i] * accα_in
        @inbounds ∇L[i+2*n]   = F[i] * s_out[i] - F[i] * accβ_out
        @inbounds ∇L[i+3*n]   = F[i] * s_in[i]  - F[i] * accβ_in
    end

    return ∇L
end


"""
    DECM_reduced_iter!(θ, d_out, d_in, s_out, s_in, F, nz_out, nz_in, x_out, x_in, y_out, y_in, G, n=length(θ)÷4)

Compute the next fixed-point iteration for the reduced DECM model. The pre-allocated buffers `x_out`, `x_in`,
`y_out`, `y_in` and `G` are updated in place. Only the live channels are iterated: the out-blocks over
`i ∈ nz_out` (with the inner sum over `j ∈ nz_in`) and the in-blocks over `i ∈ nz_in` (inner sum over
`j ∈ nz_out`) — a dead channel contributes an exact zero to every pair probability.

**Note**: the fixed-point recipe is unstable for the DECM (as it is for the UECM); `:BFGS`/`:Newton` are
preferred (see [`solve_model!`](@ref)).
"""
function DECM_reduced_iter!(θ::AbstractVector,
                            d_out::Vector, d_in::Vector, s_out::Vector, s_in::Vector,
                            F::Vector, nz_out::Vector, nz_in::Vector,
                            x_out::AbstractVector, x_in::AbstractVector,
                            y_out::AbstractVector, y_in::AbstractVector,
                            G::AbstractVector, n=length(θ)÷4)
    α_out = @view θ[1:n]
    α_in  = @view θ[n+1:2*n]
    β_out = @view θ[2*n+1:3*n]
    β_in  = @view θ[3*n+1:end]
    @inbounds @simd for i in eachindex(α_out) # to avoid the allocation of exp.(-θ)
        x_out[i] = exp(-α_out[i])
        x_in[i]  = exp(-α_in[i])
        y_out[i] = exp(-β_out[i])
        y_in[i]  = exp(-β_in[i])
    end
    G .= zero(eltype(G))

    # out-blocks (orientation i→j): xᵢ_out and yᵢ_out are divided out of the pair expressions
    for i in nz_out
        @inbounds xᵢ_out = x_out[i]
        @inbounds yᵢ_out = y_out[i]
        accα = zero(eltype(θ))
        accβ = zero(eltype(θ))
        @inbounds for j in nz_in
            wj    = F[j] - (i == j)
            c1    = xᵢ_out * x_in[j]
            c2    = yᵢ_out * y_in[j]
            denom = 1 - c2 + c1 * c2
            accα += wj * (x_in[j] * c2) / denom
            accβ += wj * (c1 * y_in[j]) / ((1 - c2) * denom)
        end
        @inbounds G[i]     = F[i] * accα
        @inbounds G[i+2*n] = F[i] * accβ
    end
    # in-blocks (orientation j→i): xᵢ_in and yᵢ_in are divided out of the pair expressions
    for i in nz_in
        @inbounds xᵢ_in = x_in[i]
        @inbounds yᵢ_in = y_in[i]
        accα = zero(eltype(θ))
        accβ = zero(eltype(θ))
        @inbounds for j in nz_out
            wj    = F[j] - (i == j)
            c1    = x_out[j] * xᵢ_in
            c2    = y_out[j] * yᵢ_in
            denom = 1 - c2 + c1 * c2
            accα += wj * (x_out[j] * c2) / denom
            accβ += wj * (c1 * y_out[j]) / ((1 - c2) * denom)
        end
        @inbounds G[i+n]   = F[i] * accα
        @inbounds G[i+3*n] = F[i] * accβ
    end

    for i in nz_out
        @inbounds G[i]     = - log_nan(F[i] * d_out[i] / G[i])
        @inbounds G[i+2*n] = - log_nan(F[i] * s_out[i] / G[i+2*n])
    end
    for i in nz_in
        @inbounds G[i+n]   = - log_nan(F[i] * d_in[i] / G[i+n])
        @inbounds G[i+3*n] = - log_nan(F[i] * s_in[i] / G[i+3*n])
    end

    for i in eachindex(G)
        @inbounds if iszero(G[i])
            G[i] = 1e8
        end
    end

    return G
end


"""
    initial_guess(m::DECM; method::Symbol=:strengths)

Compute an initial guess `θ₀ = [α_out; α_in; β_out; β_in]` for the maximum likelihood parameters of the DECM model `m`.

The methods available are:
- `:strengths` (default): degrees normalised by the number of edges, strengths normalised by the total (reduced) strength.
- `:strengths_minor`: `1/(dᵣ+1)` and `1/(sᵣ+1)` per block.
- `:random`: random values drawn from ``U(0,1)``.
- `:uniform`: uniformly set to `-log(0.001)`.
"""
function initial_guess(m::DECM; method::Symbol=:strengths)
    N = precision(m)
    if isequal(method, :strengths)
        isnothing(m.G) ? throw(ArgumentError("Cannot compute the number of edges because the model has no underlying graph (m.G == nothing)")) : nothing
        res = Vector{N}(-log.(vcat( m.dᵣ_out ./ (Graphs.ne(m.G) + 1),
                                    m.dᵣ_in  ./ (Graphs.ne(m.G) + 1),
                                    m.sᵣ_out ./ (sum(m.sᵣ_out) + 1),
                                    m.sᵣ_in  ./ (sum(m.sᵣ_in) + 1)))) # will return Inf initial guesses for zero values
    elseif isequal(method, :strengths_minor)
        res = Vector{N}(-log.(vcat( ones(N, length(m.dᵣ_out)) ./ (m.dᵣ_out .+ 1),
                                    ones(N, length(m.dᵣ_in))  ./ (m.dᵣ_in  .+ 1),
                                    ones(N, length(m.sᵣ_out)) ./ (m.sᵣ_out .+ 1),
                                    ones(N, length(m.sᵣ_in))  ./ (m.sᵣ_in  .+ 1))))
    elseif isequal(method, :random)
        res = Vector{N}(-log.(rand(N, 4*length(m.dᵣ_out))))
    elseif isequal(method, :uniform)
        res = Vector{N}(-log.(0.001 .* ones(N, 4*length(m.dᵣ_out))))
    else
        throw(ArgumentError("The initial guess method $(method) is not supported"))
    end

    return res
end


"""
    set_xᵣ!(m::DECM)

Set the values of the degree-channel parameters xᵣ_out to exp(-α_out) and xᵣ_in to exp(-α_in) for the DECM model `m`
"""
function set_xᵣ!(m::DECM)
    if m.status[:params_computed]
        n = length(m)
        α_out = @view m.θᵣ[1:n]
        α_in  = @view m.θᵣ[n+1:2*n]
        m.xᵣ_out .= exp.(-α_out)
        m.xᵣ_in  .= exp.(-α_in)
    else
        throw(ArgumentError("The parameters have not been computed yet"))
    end
end

"""
    set_yᵣ!(m::DECM)

Set the values of the strength-channel parameters yᵣ_out to exp(-β_out) and yᵣ_in to exp(-β_in) for the DECM model `m`
"""
function set_yᵣ!(m::DECM)
    if m.status[:params_computed]
        n = length(m)
        β_out = @view m.θᵣ[2*n+1:3*n]
        β_in  = @view m.θᵣ[3*n+1:end]
        m.yᵣ_out .= exp.(-β_out)
        m.yᵣ_in  .= exp.(-β_in)
    else
        throw(ArgumentError("The parameters have not been computed yet"))
    end
end


"""
    precision(m::DECM)

Determine the compute precision of the DECM model `m`.
"""
precision(m::DECM) = typeof(m).parameters[2]


"""
    f_DECM(xixj::T, yiyj::T) where {T}

Helper for the DECM model computing the expected adjacency entry `pᵢⱼ = (xᵢxⱼ·yᵢyⱼ)/(1 - yᵢyⱼ + xᵢxⱼ·yᵢyⱼ)` for
the ordered pair `i→j` from the products `xᵢxⱼ = xᵢ_out·xⱼ_in` and `yᵢyⱼ = yᵢ_out·yⱼ_in` of the maximum
likelihood parameters.
"""
f_DECM(xixj::T, yiyj::T) where {T} = (xixj * yiyj) / (one(T) - yiyj + xixj * yiyj)


"""
    A(m::DECM, i::Int, j::Int)

Return the expected value of the adjacency matrix for the DECM model `m` at the (ordered) node pair `(i,j)`.

❗ For performance reasons, the function does not check:
- if the node pair is valid.
- if the parameters of the model have been computed.
"""
function A(m::DECM, i::Int, j::Int)
    return i == j ? zero(precision(m)) : @inbounds f_DECM(m.xᵣ_out[m.dᵣ_ind[i]] * m.xᵣ_in[m.dᵣ_ind[j]], m.yᵣ_out[m.dᵣ_ind[i]] * m.yᵣ_in[m.dᵣ_ind[j]])
end


"""
    Ĝ(m::DECM)

Compute the expected **adjacency** matrix for the DECM model `m`. The matrix is asymmetric: entry `(i,j)` is
the probability of the directed edge `i→j`.

Note: The expected weights can be computed separately with [`Ŵ`](@ref MaxEntropyGraphs.Ŵ).
"""
function Ĝ(m::DECM)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))

    # get network size => this is the full size
    n = m.status[:N]::Int
    # initiate G
    G = zeros(precision(m), n, n)
    # initiate parameter vectors
    x_out = m.xᵣ_out[m.dᵣ_ind]
    x_in  = m.xᵣ_in[m.dᵣ_ind]
    y_out = m.yᵣ_out[m.dᵣ_ind]
    y_in  = m.yᵣ_in[m.dᵣ_ind]
    # compute G over the ordered pairs
    for i = 1:n
        @simd for j = 1:n
            if i ≠ j
                @inbounds xixj = x_out[i] * x_in[j]
                @inbounds yiyj = y_out[i] * y_in[j]
                @inbounds G[i,j] = (xixj * yiyj) / (1 - yiyj + xixj * yiyj)
            end
        end
    end

    return G
end

"""
    set_Ĝ!(m::DECM)

Set the expected adjacency matrix for the DECM model `m`
"""
function set_Ĝ!(m::DECM)
    m.Ĝ = Ĝ(m)
    m.status[:G_computed] = true
    return m.Ĝ
end


"""
    Ŵ(m::DECM)

Compute the expected (unconditional) **weighted adjacency** matrix for the DECM model `m`, i.e.
`⟨wᵢⱼ⟩ = pᵢⱼ / (1 - yᵢ_out·yⱼ_in)`, so that `sum(Ŵ(m), dims=2) ≈ outstrength` and
`sum(Ŵ(m), dims=1)' ≈ instrength`.
"""
function Ŵ(m::DECM)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))

    # get network size => this is the full size
    n = m.status[:N]::Int
    # initiate W
    W = zeros(precision(m), n, n)
    # initiate parameter vectors
    x_out = m.xᵣ_out[m.dᵣ_ind]
    x_in  = m.xᵣ_in[m.dᵣ_ind]
    y_out = m.yᵣ_out[m.dᵣ_ind]
    y_in  = m.yᵣ_in[m.dᵣ_ind]
    # compute W over the ordered pairs: unconditional expected weight
    for i = 1:n
        @simd for j = 1:n
            if i ≠ j
                @inbounds xixj = x_out[i] * x_in[j]
                @inbounds yiyj = y_out[i] * y_in[j]
                @inbounds pij  = (xixj * yiyj) / (1 - yiyj + xixj * yiyj)
                @inbounds W[i,j] = pij / (1 - yiyj)
            end
        end
    end

    return W
end

"""
    set_Ŵ!(m::DECM)

Set the expected weighted adjacency matrix for the DECM model `m`.
"""
function set_Ŵ!(m::DECM)
    m.Ŵ = Ŵ(m)
    m.status[:W_computed] = true
    return m.Ŵ
end


"""
    σˣ(m::DECM{T,N}) where {T,N}

Compute the standard deviation for the elements of the (binary) adjacency matrix for the DECM model `m`, i.e.
`sqrt(pᵢⱼ(1 - pᵢⱼ))` (the adjacency entries are Bernoulli distributed).

**Note:** this is the standard deviation of the *binary* layer; the standard deviation of the weights is
available via [`σʷ`](@ref MaxEntropyGraphs.σʷ). Read as "sigma star".
"""
function σˣ(m::DECM)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    # network size => full size
    n = m.status[:N]::Int
    # initiate σ
    σ = zeros(precision(m), n, n)
    # initiate parameter vectors
    x_out = m.xᵣ_out[m.dᵣ_ind]
    x_in  = m.xᵣ_in[m.dᵣ_ind]
    y_out = m.yᵣ_out[m.dᵣ_ind]
    y_in  = m.yᵣ_in[m.dᵣ_ind]
    # compute σ over the ordered pairs
    for i = 1:n
        @simd for j = 1:n
            if i ≠ j
                @inbounds xixj = x_out[i] * x_in[j]
                @inbounds yiyj = y_out[i] * y_in[j]
                @inbounds pij  = (xixj * yiyj) / (1 - yiyj + xixj * yiyj)
                @inbounds σ[i,j] = sqrt(pij * (1 - pij))
            end
        end
    end

    return σ
end

"""
    set_σ!(m::DECM)

Set the standard deviation for the elements of the (binary) adjacency matrix for the DECM model `m`
"""
function set_σ!(m::DECM)
    m.σ = σˣ(m)
    m.status[:σ_computed] = true
    return m.σ
end


"""
    σʷ(m::DECM)

Compute the standard deviation for the elements of the **weighted** adjacency matrix for the DECM model `m`.
The weight `wᵢⱼ` of the directed edge `i→j` follows a Bernoulli–geometric mixture: with `pᵢⱼ` the connection
probability and `yᵢ_out·yⱼ_in` the geometric parameter, ``⟨w_{ij}⟩ = p_{ij}/(1 - y_iy_j)``,
``⟨w_{ij}^2⟩ = p_{ij}(1 + y_iy_j)/(1 - y_iy_j)^2`` and

``Var(w_{ij}) = \\frac{p_{ij}(1 + y_iy_j - p_{ij})}{(1 - y_iy_j)^2}``

As the network is directed, `wᵢⱼ` and `wⱼᵢ` are *distinct, independent* random variables (unlike the UECM).
"""
function σʷ(m::DECM)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    # network size => full size
    n = m.status[:N]::Int
    # initiate σ
    σ = zeros(precision(m), n, n)
    # initiate parameter vectors
    x_out = m.xᵣ_out[m.dᵣ_ind]
    x_in  = m.xᵣ_in[m.dᵣ_ind]
    y_out = m.yᵣ_out[m.dᵣ_ind]
    y_in  = m.yᵣ_in[m.dᵣ_ind]
    # compute σ over the ordered pairs
    for i = 1:n
        @simd for j = 1:n
            if i ≠ j
                @inbounds xixj = x_out[i] * x_in[j]
                @inbounds yiyj = y_out[i] * y_in[j]
                @inbounds pij  = (xixj * yiyj) / (1 - yiyj + xixj * yiyj)
                @inbounds σ[i,j] = sqrt(pij * (1 + yiyj - pij)) / (1 - yiyj)
            end
        end
    end

    return σ
end

"""
    set_σʷ!(m::DECM)

Set the standard deviation for the elements of the weighted adjacency matrix for the DECM model `m`.
"""
function set_σʷ!(m::DECM)
    m.σʷ = σʷ(m)
    m.status[:σʷ_computed] = true
    return m.σʷ
end


"""
    σₓ(m::DECM, X::Function; layer::Symbol=:binary, gradient_method::Symbol=:ReverseDiff)

Compute the standard deviation of metric `X` for the DECM model `m` via error propagation (the delta
method of Squartini & Garlaschelli (2011), Eq. B.16).

# Arguments
- `layer::Symbol`:
    - `:binary` (default): propagate over the **binary adjacency matrix** — `X` is a function of the
      adjacency matrix, the gradient is evaluated at `m.Ĝ` and weighted by `m.σ` (requires `set_Ĝ!` and `set_σ!`).
    - `:weighted`: propagate over the **weighted adjacency matrix** — `X` is a function of the weight
      matrix, the gradient is evaluated at `m.Ŵ` and weighted by `m.σʷ` (requires `set_Ŵ!` and `set_σʷ!`).
- `gradient_method::Symbol`: `:ForwardDiff`, `:ReverseDiff` (default) or `:Zygote`.

As the network is directed, the entries `(i,j)` and `(j,i)` of either layer are *independent* random
variables, so the delta method carries no within-dyad covariance cross-term (unlike the undirected UECM).

Metrics mixing the two layers (functions of both the adjacency and the weight matrix) are not supported
by this per-layer propagation; use ensemble sampling instead.
"""
function σₓ(m::DECM, X::Function; layer::Symbol=:binary, gradient_method::Symbol=:ReverseDiff)
    # checks
    if layer == :binary
        m.status[:G_computed] ? nothing : throw(ArgumentError("The expected values (m.Ĝ) must be computed for `m` before computing the standard deviation of metric `X`, see `set_Ĝ!`"))
        m.status[:σ_computed] ? nothing : throw(ArgumentError("The standard deviations (m.σ) must be computed for `m` before computing the standard deviation of metric `X`, see `set_σ!`"))
        M, S = m.Ĝ, m.σ
    elseif layer == :weighted
        m.status[:W_computed] ? nothing : throw(ArgumentError("The expected weights (m.Ŵ) must be computed for `m` before computing the standard deviation of metric `X`, see `set_Ŵ!`"))
        m.status[:σʷ_computed] ? nothing : throw(ArgumentError("The weight standard deviations (m.σʷ) must be computed for `m` before computing the standard deviation of metric `X`, see `set_σʷ!`"))
        M, S = m.Ŵ, m.σʷ
    else
        throw(ArgumentError("Invalid layer, only :binary and :weighted are accepted"))
    end

    # gradient
    if gradient_method == :ForwardDiff
        ∇X = ForwardDiff.gradient(X, M)
    elseif gradient_method == :ReverseDiff
        ∇X = ReverseDiff.gradient(X, M)
    elseif gradient_method == :Zygote
        ∇X = Zygote.gradient(X, M)[1]
    else
        throw(ArgumentError("Invalid gradient method, only :ForwardDiff, :ReverseDiff and :Zygote are accepted"))
    end

    # delta method over the ordered pairs; the (i,j) and (j,i) entries are independent (directed)
    return sqrt( sum((S .* ∇X) .^ 2) )
end


"""
    outdegree(m::DECM, i::Int; method=:reduced)

Return the expected outdegree for node `i` of the DECM model `m`.

# Arguments
- `method::Symbol`:
    - `:reduced` (default) uses the reduced model parameters for performance reasons.
    - `:full` uses all elements of the expected adjacency matrix.
    - `:adjacency` uses the precomputed adjacency matrix `m.Ĝ` of the model.
"""
function outdegree(m::DECM, i::Int; method::Symbol=:reduced)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    i > m.status[:N] ? throw(ArgumentError("Attempted to access node $i in a $(m.status[:N]) node graph")) : nothing

    if method == :reduced
        res = zero(precision(m))
        i_red = m.dᵣ_ind[i] # find matching index in reduced model
        for j in eachindex(m.xᵣ_in)
            @inbounds pij = f_DECM(m.xᵣ_out[i_red] * m.xᵣ_in[j], m.yᵣ_out[i_red] * m.yᵣ_in[j])
            if i_red ≠ j
                @inbounds res += pij * m.f[j]
            else
                @inbounds res += pij * (m.f[j] - 1) # subtract 1 because the diagonal is not counted
            end
        end
    elseif method == :full
        res = zero(precision(m))
        for j in eachindex(m.d_out)
            res += A(m, i, j)
        end
    elseif method == :adjacency
        m.status[:G_computed] ? nothing : throw(ArgumentError("The adjacency matrix has not been computed yet"))
        res = sum(@view m.Ĝ[i,:])
    else
        throw(ArgumentError("Unknown method $method"))
    end

    return res
end

"""
    outdegree(m::DECM[, v]; method=:reduced)

Return a vector corresponding to the expected outdegree of each node of the DECM model `m`. If `v` is
specified, only return outdegrees for nodes in `v`.
"""
outdegree(m::DECM, v::Vector{Int}=collect(1:m.status[:N]); method::Symbol=:reduced) = [outdegree(m, i, method=method) for i in v]


"""
    indegree(m::DECM, i::Int; method=:reduced)

Return the expected indegree for node `i` of the DECM model `m`.

# Arguments
- `method::Symbol`:
    - `:reduced` (default) uses the reduced model parameters for performance reasons.
    - `:full` uses all elements of the expected adjacency matrix.
    - `:adjacency` uses the precomputed adjacency matrix `m.Ĝ` of the model.
"""
function indegree(m::DECM, i::Int; method::Symbol=:reduced)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    i > m.status[:N] ? throw(ArgumentError("Attempted to access node $i in a $(m.status[:N]) node graph")) : nothing

    if method == :reduced
        res = zero(precision(m))
        i_red = m.dᵣ_ind[i] # find matching index in reduced model
        for j in eachindex(m.xᵣ_out)
            @inbounds pji = f_DECM(m.xᵣ_out[j] * m.xᵣ_in[i_red], m.yᵣ_out[j] * m.yᵣ_in[i_red])
            if i_red ≠ j
                @inbounds res += pji * m.f[j]
            else
                @inbounds res += pji * (m.f[j] - 1) # subtract 1 because the diagonal is not counted
            end
        end
    elseif method == :full
        res = zero(precision(m))
        for j in eachindex(m.d_out)
            res += A(m, j, i)
        end
    elseif method == :adjacency
        m.status[:G_computed] ? nothing : throw(ArgumentError("The adjacency matrix has not been computed yet"))
        res = sum(@view m.Ĝ[:,i])
    else
        throw(ArgumentError("Unknown method $method"))
    end

    return res
end

"""
    indegree(m::DECM[, v]; method=:reduced)

Return a vector corresponding to the expected indegree of each node of the DECM model `m`. If `v` is
specified, only return indegrees for nodes in `v`.
"""
indegree(m::DECM, v::Vector{Int}=collect(1:m.status[:N]); method::Symbol=:reduced) = [indegree(m, i, method=method) for i in v]


"""
    degree(m::DECM, i::Int; method=:reduced)

In alignment with `Graphs.jl`, returns the sum of the expected out- and indegree for node `i` of the DECM model `m`.

# Arguments
- `method::Symbol`: `:reduced` (default), `:full` or `:adjacency` (see [`outdegree`](@ref MaxEntropyGraphs.outdegree)).
"""
degree(m::DECM, i::Int; method::Symbol=:reduced) = outdegree(m, i, method=method) + indegree(m, i, method=method)

"""
    degree(m::DECM[, v]; method=:reduced)

In alignment with `Graphs.jl`, returns a vector corresponding to the sum of the expected out- and indegree of
each node of the DECM model `m`. If `v` is specified, only return degrees for nodes in `v`.
"""
degree(m::DECM, v::Vector{Int}=collect(1:m.status[:N]); method::Symbol=:reduced) = outdegree(m, v, method=method) + indegree(m, v, method=method)


"""
    outstrength(m::DECM, i::Int; method=:reduced)

Return the expected (unconditional) outstrength for node `i` of the DECM model `m`, i.e. `Σⱼ pᵢⱼ/(1 - yᵢ_out·yⱼ_in)`.

# Arguments
- `method::Symbol`:
    - `:reduced` (default) uses the reduced model parameters for performance reasons.
    - `:full` uses all node pairs.
    - `:adjacency` reuses the precomputed adjacency matrix `m.Ĝ` (plus the `yᵣ` parameters).
"""
function outstrength(m::DECM, i::Int; method::Symbol=:reduced)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    i > m.status[:N] ? throw(ArgumentError("Attempted to access node $i in a $(m.status[:N]) node graph")) : nothing

    if method == :reduced
        res = zero(precision(m))
        i_red = m.dᵣ_ind[i]
        for j in eachindex(m.xᵣ_in)
            @inbounds yiyj = m.yᵣ_out[i_red] * m.yᵣ_in[j]
            @inbounds wij  = f_DECM(m.xᵣ_out[i_red] * m.xᵣ_in[j], yiyj) / (1 - yiyj)
            if i_red ≠ j
                @inbounds res += wij * m.f[j]
            else
                @inbounds res += wij * (m.f[j] - 1)
            end
        end
    elseif method == :full
        res = zero(precision(m))
        for j in eachindex(m.d_out)
            if i ≠ j
                @inbounds yiyj = m.yᵣ_out[m.dᵣ_ind[i]] * m.yᵣ_in[m.dᵣ_ind[j]]
                res += A(m, i, j) / (1 - yiyj)
            end
        end
    elseif method == :adjacency
        m.status[:G_computed] ? nothing : throw(ArgumentError("The adjacency matrix has not been computed yet"))
        y_out = m.yᵣ_out[m.dᵣ_ind]
        y_in  = m.yᵣ_in[m.dᵣ_ind]
        res = zero(precision(m))
        for j in 1:m.status[:N]
            if i ≠ j
                @inbounds res += m.Ĝ[i,j] / (1 - y_out[i] * y_in[j])
            end
        end
    else
        throw(ArgumentError("Unknown method $method"))
    end

    return res
end

"""
    outstrength(m::DECM[, v]; method=:reduced)

Return a vector corresponding to the expected outstrength of each node of the DECM model `m`. If `v` is
specified, only return outstrengths for nodes in `v`.
"""
outstrength(m::DECM, v::Vector{Int}=collect(1:m.status[:N]); method::Symbol=:reduced) = [outstrength(m, i, method=method) for i in v]


"""
    instrength(m::DECM, i::Int; method=:reduced)

Return the expected (unconditional) instrength for node `i` of the DECM model `m`, i.e. `Σⱼ pⱼᵢ/(1 - yⱼ_out·yᵢ_in)`.

# Arguments
- `method::Symbol`:
    - `:reduced` (default) uses the reduced model parameters for performance reasons.
    - `:full` uses all node pairs.
    - `:adjacency` reuses the precomputed adjacency matrix `m.Ĝ` (plus the `yᵣ` parameters).
"""
function instrength(m::DECM, i::Int; method::Symbol=:reduced)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    i > m.status[:N] ? throw(ArgumentError("Attempted to access node $i in a $(m.status[:N]) node graph")) : nothing

    if method == :reduced
        res = zero(precision(m))
        i_red = m.dᵣ_ind[i]
        for j in eachindex(m.xᵣ_out)
            @inbounds yjyi = m.yᵣ_out[j] * m.yᵣ_in[i_red]
            @inbounds wji  = f_DECM(m.xᵣ_out[j] * m.xᵣ_in[i_red], yjyi) / (1 - yjyi)
            if i_red ≠ j
                @inbounds res += wji * m.f[j]
            else
                @inbounds res += wji * (m.f[j] - 1)
            end
        end
    elseif method == :full
        res = zero(precision(m))
        for j in eachindex(m.d_out)
            if i ≠ j
                @inbounds yjyi = m.yᵣ_out[m.dᵣ_ind[j]] * m.yᵣ_in[m.dᵣ_ind[i]]
                res += A(m, j, i) / (1 - yjyi)
            end
        end
    elseif method == :adjacency
        m.status[:G_computed] ? nothing : throw(ArgumentError("The adjacency matrix has not been computed yet"))
        y_out = m.yᵣ_out[m.dᵣ_ind]
        y_in  = m.yᵣ_in[m.dᵣ_ind]
        res = zero(precision(m))
        for j in 1:m.status[:N]
            if i ≠ j
                @inbounds res += m.Ĝ[j,i] / (1 - y_out[j] * y_in[i])
            end
        end
    else
        throw(ArgumentError("Unknown method $method"))
    end

    return res
end

"""
    instrength(m::DECM[, v]; method=:reduced)

Return a vector corresponding to the expected instrength of each node of the DECM model `m`. If `v` is
specified, only return instrengths for nodes in `v`.
"""
instrength(m::DECM, v::Vector{Int}=collect(1:m.status[:N]); method::Symbol=:reduced) = [instrength(m, i, method=method) for i in v]


"""
    strength(m::DECM, i::Int; method=:reduced)

In alignment with the `dir=:both` convention, returns the sum of the expected out- and instrength for node
`i` of the DECM model `m`.

# Arguments
- `method::Symbol`: `:reduced` (default), `:full` or `:adjacency` (see [`outstrength`](@ref MaxEntropyGraphs.outstrength)).
"""
strength(m::DECM, i::Int; method::Symbol=:reduced) = outstrength(m, i, method=method) + instrength(m, i, method=method)

"""
    strength(m::DECM[, v]; method=:reduced)

Returns a vector corresponding to the sum of the expected out- and instrength of each node of the DECM model
`m`. If `v` is specified, only return strengths for nodes in `v`.
"""
strength(m::DECM, v::Vector{Int}=collect(1:m.status[:N]); method::Symbol=:reduced) = outstrength(m, v, method=method) + instrength(m, v, method=method)


"""
    AIC(m::DECM)

Compute the Akaike Information Criterion (AIC) for the DECM model `m`. The parameters of the model must be
computed beforehand. The DECM has `4N` parameters (an ``\\alpha^{out}``, ``\\alpha^{in}``, ``\\beta^{out}``
and ``\\beta^{in}`` per node).

See also [`AICc`](@ref MaxEntropyGraphs.AICc), [`L_DECM_reduced`](@ref MaxEntropyGraphs.L_DECM_reduced).
"""
function AIC(m::DECM)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    # compute AIC components
    k = 4 * m.status[:N] # number of parameters (α_out, α_in, β_out and β_in per node)
    n = (m.status[:N] - 1) * m.status[:N] # number of observations (ordered node pairs)
    L = L_DECM_reduced(m) # log-likelihood

    if n/k < 40
        @warn """The number of observations is small with respect to the number of parameters (n/k < 40). Consider using the corrected AIC (AICc) instead."""
    end

    return 2*k - 2*L
end


"""
    AICc(m::DECM)

Compute the corrected Akaike Information Criterion (AICc) for the DECM model `m`.

See also [`AIC`](@ref MaxEntropyGraphs.AIC), [`L_DECM_reduced`](@ref MaxEntropyGraphs.L_DECM_reduced).
"""
function AICc(m::DECM)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    # compute AIC components
    k = 4 * m.status[:N] # number of parameters
    n = (m.status[:N] - 1) * m.status[:N] # number of observations (ordered node pairs)
    L = L_DECM_reduced(m) # log-likelihood

    return 2*k - 2*L + (2*k*(k+1)) / (n - k - 1)
end


"""
    BIC(m::DECM)

Compute the Bayesian Information Criterion (BIC) for the DECM model `m`.

See also [`AIC`](@ref MaxEntropyGraphs.AIC), [`L_DECM_reduced`](@ref MaxEntropyGraphs.L_DECM_reduced).
"""
function BIC(m::DECM)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    # compute AIC components
    k = 4 * m.status[:N] # number of parameters
    n = (m.status[:N] - 1) * m.status[:N] # number of observations (ordered node pairs)
    L = L_DECM_reduced(m) # log-likelihood

    return k * log(n) - 2*L
end


"""
    rand(m::DECM; precomputed=false, rng=Random.default_rng())

Generate a random directed weighted graph from the DECM model `m`.

# Arguments
- `precomputed::Bool`: not implemented yet for the DECM (the parameters are always used to generate the graph on the fly).
- `rng::AbstractRNG`: random number generator to use (defaults to `Random.default_rng()`).

# Examples
```jldoctest
julia> model = DECM(MaxEntropyGraphs.rhesus_macaques()); # generate a DECM model

julia> solve_model!(model, method=:BFGS); # compute the maximum likelihood parameters

julia> sample = rand(model); # sample a random directed weighted graph

julia> typeof(sample)
SimpleWeightedGraphs.SimpleWeightedDiGraph{Int64, Int64}
```
"""
function rand(m::DECM; precomputed::Bool=false, rng::AbstractRNG=default_rng())
    if precomputed
        throw(ArgumentError("This function is not implemented yet for DECM models"))
    else
        # check if possible to use parameters
        m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
        # generate the parameter vectors
        x_out = m.xᵣ_out[m.dᵣ_ind]
        x_in  = m.xᵣ_in[m.dᵣ_ind]
        y_out = m.yᵣ_out[m.dᵣ_ind]
        y_in  = m.yᵣ_in[m.dᵣ_ind]
        # generate random graph edges over the ordered pairs
        sources = Vector{Int}();
        targets = Vector{Int}();
        weights = Vector{Int}();
        for i in 1:m.status[:N]
            for j in 1:m.status[:N]
                i == j && continue
                # check if the directed edge i→j exists
                @inbounds xixj = x_out[i] * x_in[j]
                @inbounds yiyj = y_out[i] * y_in[j]
                p_ij = (xixj * yiyj) / (1 - yiyj + xixj * yiyj)
                if rand(rng) ≤ p_ij
                    push!(sources, i)
                    push!(targets, j)
                    # weight = 1 + Geometric(1 - yᵢ_out·yⱼ_in); mean excess weight = yᵢyⱼ/(1 - yᵢyⱼ)
                    push!(weights, rand(rng, Geometric(1 - yiyj)) + 1)
                end
            end
        end

        if length(sources) ≠ 0
            G = SimpleWeightedGraphs.SimpleWeightedDiGraph(sources, targets, weights)
        else
            G = SimpleWeightedGraphs.SimpleWeightedDiGraph(m.status[:N])
        end

        # deal with edge case where no edges are generated for the last node(s) in the graph
        while Graphs.nv(G) < m.status[:N]
            Graphs.add_vertex!(G)
        end
        return G
    end
end

"""
    rand(m::DECM, n::Int; precomputed=false, rng=Random.default_rng())

Generate `n` random directed weighted graphs from the DECM model `m`. If multithreading is available, the
graphs are generated in parallel; per-sample seeds are drawn from `rng` so the result is reproducible and
independent of the thread schedule.
"""
function rand(m::DECM, n::Int; precomputed::Bool=false, rng::AbstractRNG=default_rng())
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
    solve_model!(m::DECM; kwargs...)

Compute the likelihood maximising parameters of the DECM model `m`.

By default the parameters are computed using the BFGS method with the strength sequences as initial guess.

# Arguments
- `method::Symbol`: solution method, `:BFGS` (default) or any of :$(join(keys(MaxEntropyGraphs.optimization_methods), ", :", " and :")), plus `:fixedpoint`.
- `initial::Symbol`: initial guess, `:strengths` (default), `:strengths_minor`, `:random`, or `:uniform`.
- `maxiters::Int`: maximum number of iterations (defaults to 1000).
- `verbose::Bool`: show log messages (defaults to false).
- `ftol::Union{Real, Nothing}`: tolerance for the fixedpoint method (defaults to `nothing`, i.e. 1e-8). It bounds the fixed-point *increment* ``\\|G(\\theta) - \\theta\\|_\\infty`` in **parameter** space; it is **not** the constraint residual. ❗ It applies to the `:fixedpoint` method only, and so is **ignored on this model's default `:BFGS` path** (passing it there warns). Use [`constraint_residual`](@ref) to measure how well the expected degrees and strengths actually match the observed ones.
- `abstol`, `reltol`: absolute/relative tolerances for the optimisation methods (default `nothing`).
- `g_tol::Union{Number, Nothing}`: gradient tolerance for the gradient-based methods (maps to Optim's `g_abstol`, default `nothing`). The gradient of this model *is* its constraint residual (up to the multiplicities), and it is the tolerance to reach for on the default path, but `g_abstol` is a stopping criterion rather than a guarantee: Optim can also stop on its function or parameter convergence checks and report success without the gradient ever reaching `g_tol`. Verify what was actually achieved with [`constraint_residual`](@ref).
- `AD_method::Symbol`: autodiff method, any of :$(join(keys(MaxEntropyGraphs.AD_methods), ", :", " and :")) (defaults to `:AutoZygote`).
- `analytical_gradient::Bool`: use the analytical gradient instead of autodiff (defaults to `false`).

**Notes**
- the fixed-point method is very unstable for this model and should not be used. From an acceptable solution it can be used to fine-tune an existing one.
- the L-BFGS method is known to be unstable for this model and should not be used.
"""
function solve_model!(m::DECM;  # common settings
                                method::Symbol=:BFGS,
                                initial::Symbol=:strengths,
                                maxiters::Int=1000,
                                verbose::Bool=false,
                                # NLsolve.jl specific settings (fixed point method)
                                ftol::Union{Real, Nothing}=nothing,
                                # optimisation.jl specific settings (optimisation methods)
                                abstol::Union{Number, Nothing}=nothing,
                                reltol::Union{Number, Nothing}=nothing,
                                g_tol::Union{Number, Nothing}=nothing,
                                AD_method::Symbol=:AutoZygote,
                                analytical_gradient::Bool=false)
    N = precision(m)
    n = length(m)
    N <: Union{Float16, Float32} && @warn "Solving in $(N) precision is experimental and may not converge; low precision is intended for storage. Consider Float64 for the solve." maxlog=1
    # `ftol` is accepted on every path but only ever reaches the fixed point solver: say so rather than
    # ignoring it silently (only when it was actually passed, so a default solve stays quiet)
    method ≠ :fixedpoint && !isnothing(ftol) && @warn _ftol_unused_msg(method) maxlog=1
    ftol = isnothing(ftol) ? _DEFAULT_FTOL : ftol
    # initial guess
    θ₀ = initial_guess(m, method=initial)
    # find Inf values (dead channels: zero degrees/strengths)
    ind_inf = findall(isinf, θ₀)
    if method==:fixedpoint
        @warn "The fixed point method is very unstable for this model and should not be used. `BFGS` is prefered for quasinewton methods."
        # initiate buffers
        x_out_buffer = zeros(N, n); # buffer for x_out = exp(-α_out)
        x_in_buffer  = zeros(N, n); # buffer for x_in  = exp(-α_in)
        y_out_buffer = zeros(N, n); # buffer for y_out = exp(-β_out)
        y_in_buffer  = zeros(N, n); # buffer for y_in  = exp(-β_in)
        G_buffer     = zeros(N, length(m.θᵣ)); # buffer for G(x)
        # define fixed point function
        FP_model! = (θ::Vector) -> DECM_reduced_iter!(θ, m.dᵣ_out, m.dᵣ_in, m.sᵣ_out, m.sᵣ_in, m.f, m.dᵣ_out_nz, m.dᵣ_in_nz, x_out_buffer, x_in_buffer, y_out_buffer, y_in_buffer, G_buffer, n);
        # obtain solution
        θ₀[ind_inf] .= zero(N);
        sol = NLsolve.fixedpoint(FP_model!, θ₀, method=:anderson, ftol=ftol, iterations=maxiters);
        if NLsolve.converged(sol)
            if verbose
                @info "Fixed point iteration converged after $(sol.iterations) iterations"
            end
            m.θᵣ .= sol.zero;
            m.θᵣ[ind_inf] .= N(Inf);
            m.status[:params_computed] = true;
            set_xᵣ!(m);
            set_yᵣ!(m);
        else
            throw(ConvergenceError(method, nothing))
        end
    else
        if analytical_gradient
            # initiate buffers
            x_out_buffer = zeros(N, n); # buffer for x_out = exp(-α_out)
            x_in_buffer  = zeros(N, n); # buffer for x_in  = exp(-α_in)
            y_out_buffer = zeros(N, n); # buffer for y_out = exp(-β_out)
            y_in_buffer  = zeros(N, n); # buffer for y_in  = exp(-β_in)
            # define gradient function for optimisation.jl
            grad! = (G, θ, p) -> ∇L_DECM_reduced_minus!(G, θ, m.dᵣ_out, m.dᵣ_in, m.sᵣ_out, m.sᵣ_in, m.f, x_out_buffer, x_in_buffer, y_out_buffer, y_in_buffer, n);
        end
        # define objective function and its AD method
        f = AD_method ∈ keys(AD_methods) ? Optimization.OptimizationFunction( (θ, p) -> - L_DECM_reduced(θ, m.dᵣ_out, m.dᵣ_in, m.sᵣ_out, m.sᵣ_in, m.f, n),
                                                                                        AD_methods[AD_method],
                                                                                        grad = analytical_gradient ? grad! : nothing)                      : throw(ArgumentError("The AD method $(AD_method) is not supported (yet)"))

        prob = Optimization.OptimizationProblem(f, θ₀);
        # obtain solution
        method ∈ keys(optimization_methods) || throw(ArgumentError("The method $(method) is not supported (yet)"))
        # use the BackTracking-line-search variants (see `backtracking_optimization_methods` in
        # models.jl), falling back to the package-wide optimizer for any method without a BackTracking variant.
        opt = get(backtracking_optimization_methods, method, optimization_methods[method])
        # `maxiters` is forwarded; `g_tol` (when set) maps to Optim's gradient tolerance so the solve can
        # stop before over-converging.
        solve_kwargs = isnothing(g_tol) ? (; maxiters = maxiters, abstol = abstol, reltol = reltol) :
                                          (; maxiters = maxiters, abstol = abstol, reltol = reltol, g_abstol = g_tol)
        sol = Optimization.solve(prob, opt; solve_kwargs...)
        # check convergence
        if Optimization.SciMLBase.successful_retcode(sol.retcode)
            if verbose
                @info """$(method) optimisation converged after $(@sprintf("%1.2e", sol.stats.time)) seconds (Optimization.jl return code: $("$(sol.retcode)"))"""
            end
            m.θᵣ .= sol.u;
            m.θᵣ[ind_inf] .= N(Inf);
            m.status[:params_computed] = true;
            set_xᵣ!(m);
            set_yᵣ!(m);
        else
            throw(ConvergenceError(method, sol.retcode))
        end
    end

    return m, sol
end
