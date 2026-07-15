
"""
    RBCM

Maximum entropy model for the Reciprocal Binary Configuration Model (RBCM).

The model constrains, for every node, the *non-reciprocated out-degree* ``k^{→}_i = \\sum_{j≠i} a_{ij}(1-a_{ji})``,
the *non-reciprocated in-degree* ``k^{←}_i = \\sum_{j≠i} a_{ji}(1-a_{ij})`` and the *reciprocated degree*
``k^{↔}_i = \\sum_{j≠i} a_{ij}a_{ji}`` (Squartini & Garlaschelli (2011), App. C.1, there called the reciprocal
configuration model RCM; also used in Di Vece et al. (2023) and the NuMeTriS package).

The object holds the maximum likelihood parameters of the model (θᵣ = [α; β; γ]), and optionally the expected
adjacency matrix (Ĝ) and the variance of its elements (σ). All settings and other metadata are stored in the
`status` field.

*Note*: within a dyad the entries ``a_{ij}`` and ``a_{ji}`` are **not** independent under the RBCM (unlike the
DBCM): their covariance is ``p^{↔}_{ij} - ⟨a_{ij}⟩⟨a_{ji}⟩ ≠ 0``. Metrics that combine both directions of a
dyad (e.g. motifs) must therefore be computed from the dyadic probabilities, not from Ĝ alone; the package's
`σₓ` and motif methods for the RBCM take care of this.
"""
mutable struct RBCM{T<:Union{Graphs.AbstractGraph, Nothing}, N<:Real} <: AbstractMaxEntropyModel
    "Graph type, can be any subtype of AbstractGraph, but will be converted to SimpleDiGraph for the computation" # can also be empty
    const G::T
    "Vector holding all maximum likelihood parameters for reduced model (α ; β ; γ)"
    const θᵣ::Vector{N}
    "Exponentiated maximum likelihood parameters for reduced model ( xᵢ = exp(-αᵢ) ) linked with the non-reciprocated out-degree"
    const xᵣ::Vector{N}
    "Exponentiated maximum likelihood parameters for reduced model ( yᵢ = exp(-βᵢ) ) linked with the non-reciprocated in-degree"
    const yᵣ::Vector{N}
    "Exponentiated maximum likelihood parameters for reduced model ( zᵢ = exp(-γᵢ) ) linked with the reciprocated degree"
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
    "Indices of non-zero elements in the reduced non-reciprocated out-degree sequence"
    const dᵣ_out_nz::Vector{Int}
    "Indices of non-zero elements in the reduced non-reciprocated in-degree sequence"
    const dᵣ_in_nz::Vector{Int}
    "Indices of non-zero elements in the reduced reciprocated degree sequence"
    const dᵣ_rec_nz::Vector{Int}
    "Frequency of each (k→, k←, k↔) triple in the graph"
    const f::Vector{Int}
    "Indices to reconstruct the degree sequences from the reduced degree sequences"
    const d_ind::Vector{Int}
    "Indices to reconstruct the reduced degree sequences from the degree sequences"
    const dᵣ_ind::Vector{Int}
    "Expected adjacency matrix" # not always computed/required
    Ĝ::Union{Nothing, Matrix{N}}
    "Variance of the expected adjacency matrix" # not always computed/required
    σ::Union{Nothing, Matrix{N}}
    "Status indicators: parameters computed, expected adjacency matrix computed, variance computed, etc."
    const status::Dict{Symbol, Real}
    "Function used to computed the log-likelihood of the (reduced) model"
    fun::Union{Nothing, Function}
end


Base.show(io::IO, m::RBCM{T,N}) where {T,N} = print(io, """RBCM{$(T), $(N)} ($(m.status[:d]) vertices, $(m.status[:d_unique]) unique degree triples, $(@sprintf("%.2f", m.status[:cᵣ])) compression ratio)""")

"""Return the reduced number of nodes in the RBCM network"""
Base.length(m::RBCM) = length(m.dᵣ_out)


"""
    RBCM(G::T; precision::N=Float64, kwargs...) where {T<:Graphs.AbstractGraph, N<:Real}
    RBCM(;d_out::Vector{T}, d_in::Vector{T}, d_rec::Vector{T}, precision::Type{<:AbstractFloat}=Float64, kwargs...)

Constructor function for the `RBCM` type.

By default the three reciprocal degree sequences are computed from the directed graph `G`
(see [`nonreciprocated_outdegree`](@ref), [`nonreciprocated_indegree`](@ref) and [`reciprocated_degree`](@ref)).
You can also pass the degree sequences directly as keyword arguments (`d_out` = k→, `d_in` = k←, `d_rec` = k↔),
either alongside the graph or without one.

# Examples
```jldoctest RBCM_creation
# generating a model from a graph
julia> G = MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques());

julia> model = RBCM(G)
RBCM{Graphs.SimpleGraphs.SimpleDiGraph{Int64}, Float64} (16 vertices, 15 unique degree triples, 0.94 compression ratio)
```
```jldoctest RBCM_creation
# generating a model directly from degree sequences
julia> model = RBCM(d_out=nonreciprocated_outdegree(G), d_in=nonreciprocated_indegree(G), d_rec=reciprocated_degree(G))
RBCM{Nothing, Float64} (16 vertices, 15 unique degree triples, 0.94 compression ratio)
```
"""
function RBCM(G::T; d_out::Vector=Graphs.is_directed(G) ? nonreciprocated_outdegree(G) : zeros(Int, Graphs.nv(G)),
                    d_in::Vector=Graphs.is_directed(G)  ? nonreciprocated_indegree(G)  : zeros(Int, Graphs.nv(G)),
                    d_rec::Vector=Graphs.is_directed(G) ? reciprocated_degree(G)       : Graphs.degree(G),
                    precision::Type{N}=Float64,
                    kwargs...) where {T,N<:AbstractFloat}
    T <: Union{Graphs.AbstractGraph, Nothing} ? nothing : throw(TypeError("G must be a subtype of AbstractGraph or Nothing"))
    (length(d_out) == length(d_in) && length(d_out) == length(d_rec)) ? nothing : throw(DimensionMismatch("The degree sequences must have the same length"))
    # coherence checks
    if T <: Graphs.AbstractGraph # Graph specific checks
        if !Graphs.is_directed(G)
            @warn "The graph is undirected, while the RBCM model is directed; every edge will be considered reciprocated (k→ = k← = 0)"
        end

        if T <: SimpleWeightedGraphs.AbstractSimpleWeightedGraph
            @warn "The graph is weighted, while the RBCM model is unweighted, the weight information will be lost"
        end

        Graphs.nv(G) == 0 ? throw(ArgumentError("The graph is empty")) : nothing
        Graphs.nv(G) == 1 ? throw(ArgumentError("The graph has only one vertex")) : nothing
        Graphs.nv(G) != length(d_out) ? throw(DimensionMismatch("The number of vertices in the graph ($(Graphs.nv(G))) and the length of the degree sequences ($(length(d_out))) do not match")) : nothing
    end
    # coherence checks specific to the degree sequences
    iszero(length(d_out)) ? throw(ArgumentError("The degree sequences are empty")) : nothing
    length(d_out) == 1 ? throw(ArgumentError("The degree sequences only contain a single node")) : nothing
    (minimum(d_out) < 0 || minimum(d_in) < 0 || minimum(d_rec) < 0) ? throw(DomainError("The degree sequences must be non-negative")) : nothing
    sum(d_out) == sum(d_in) ? nothing : throw(DomainError("The total number of non-reciprocated out- and in-stubs must match (sum(d_out) == sum(d_in))"))
    iseven(sum(d_rec)) ? nothing : throw(DomainError("The total number of reciprocated stubs must be even (each reciprocated dyad contributes two)"))
    maximum(d_out .+ d_in .+ d_rec) >= length(d_out) ? throw(DomainError("A node's total number of connected dyads (k→ + k← + k↔) is greater or equal to the number of vertices, this is not allowed")) : nothing
    # informative warnings for degenerate regimes
    if iszero(sum(d_rec))
        @warn "The reciprocated degree sequence is all zeros: the RBCM degenerates to a DBCM with an additional (unidentified) parameter set. Consider using the DBCM instead."
    elseif iszero(sum(d_out)) && iszero(sum(d_in))
        @warn "The non-reciprocated degree sequences are all zeros (fully reciprocal network): only the reciprocated parameters are identified."
    end

    # field generation
    dᵣ, d_ind , dᵣ_ind, f = np_unique_clone(collect(zip(d_out, d_in, d_rec)), sorted=true)
    dᵣ_out = [d[1] for d in dᵣ]
    dᵣ_in  = [d[2] for d in dᵣ]
    dᵣ_rec = [d[3] for d in dᵣ]
    dᵣ_out_nz = findall(!iszero, dᵣ_out)
    dᵣ_in_nz  = findall(!iszero, dᵣ_in)
    dᵣ_rec_nz = findall(!iszero, dᵣ_rec)
    Θᵣ = Vector{precision}(undef, 3*length(dᵣ))
    xᵣ = Vector{precision}(undef, length(dᵣ))
    yᵣ = Vector{precision}(undef, length(dᵣ))
    zᵣ = Vector{precision}(undef, length(dᵣ))
    status = Dict{Symbol, Real}(:params_computed=>false,            # are the parameters computed?
                                :G_computed=>false,                 # is the expected adjacency matrix computed and stored?
                                :σ_computed=>false,                 # is the standard deviation computed and stored?
                                :cᵣ => length(dᵣ)/length(d_out),    # compression ratio of the reduced model
                                :d_unique => length(dᵣ),            # number of unique (k→, k←, k↔) triples in the reduced model
                                :d => length(d_out)                 # number of vertices in the original graph
                )

    return RBCM{T,precision}(G, Θᵣ, xᵣ, yᵣ, zᵣ, Vector{Int}(d_out), Vector{Int}(d_in), Vector{Int}(d_rec), dᵣ_out, dᵣ_in, dᵣ_rec, dᵣ_out_nz, dᵣ_in_nz, dᵣ_rec_nz, f, d_ind, dᵣ_ind, nothing, nothing, status, nothing)
end

RBCM(; d_out::Vector{T}, d_in::Vector{T}, d_rec::Vector{T}, precision::Type{N}=Float64, kwargs...) where {T<:Signed, N<:AbstractFloat} = RBCM(nothing; d_out=d_out, d_in=d_in, d_rec=d_rec, precision=precision, kwargs...)


"""
    L_RBCM_reduced(θ::AbstractVector, k_out::Vector, k_in::Vector, k_rec::Vector, F::Vector, nz_out::Vector, nz_in::Vector, nz_rec::Vector, n::Int=length(k_out))

Compute the log-likelihood of the reduced RBCM model using the exponential formulation in order to maintain convexity.

The RBCM likelihood is (Squartini & Garlaschelli (2011), Eq. C.14, written in terms of θ = [α; β; γ])

``\\mathcal{L} = -\\sum_i \\left[ k^{→}_i α_i + k^{←}_i β_i + k^{↔}_i γ_i \\right] - \\sum_{i<j} \\ln\\left( 1 + x_iy_j + x_jy_i + z_iz_j \\right)``

with ``x = e^{-α}, y = e^{-β}, z = e^{-γ}``. The dyadic normaliser is evaluated with a numerically stable
four-term log-sum-exp ([`MaxEntropyGraphs.log1pexpsum`](@ref)). Channels whose constraint is zero (e.g. ``k^{↔}_i = 0``)
are treated as being at their analytical optimum (parameter `+Inf`, fitness exactly zero), so the likelihood does not
depend on those coordinates.

# Arguments
- `θ`: the maximum likelihood parameters of the model ([α; β; γ])
- `k_out`: the reduced non-reciprocated out-degree sequence (k→)
- `k_in`: the reduced non-reciprocated in-degree sequence (k←)
- `k_rec`: the reduced reciprocated degree sequence (k↔)
- `F`: the frequency of each (k→, k←, k↔) triple in the degree sequences
- `nz_out`: the indices of non-zero elements in `k_out`
- `nz_in`: the indices of non-zero elements in `k_in`
- `nz_rec`: the indices of non-zero elements in `k_rec`
- `n`: the number of unique node classes in the reduced model

# Examples
```jldoctest
# Use with RBCM model:
julia> G = MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques());

julia> model = RBCM(G);

julia> model_fun = θ -> L_RBCM_reduced(θ, model.dᵣ_out, model.dᵣ_in, model.dᵣ_rec, model.f, model.dᵣ_out_nz, model.dᵣ_in_nz, model.dᵣ_rec_nz, model.status[:d_unique]);

julia> model_fun(ones(size(model.θᵣ)))
-167.0688758052793
```
"""
function L_RBCM_reduced(θ::AbstractVector, k_out::Vector, k_in::Vector, k_rec::Vector, F::Vector,
                        nz_out::Vector, nz_in::Vector, nz_rec::Vector, n::Int=length(k_out))
    T = eltype(θ)
    α = @view θ[1:n]
    β = @view θ[n+1:2*n]
    γ = @view θ[2*n+1:3*n]
    ninf = T(-Inf)
    res = zero(T)
    # linear parts (restricted to non-zero constraints: zero-valued constraints contribute exactly zero)
    for i ∈ nz_out
        @inbounds res -= F[i] * k_out[i] * α[i]
    end
    for i ∈ nz_in
        @inbounds res -= F[i] * k_in[i] * β[i]
    end
    for i ∈ nz_rec
        @inbounds res -= F[i] * k_rec[i] * γ[i]
    end
    # pair part over unordered class pairs; dead channels (zero constraint) enter with exponent -Inf,
    # i.e. as exact zeros in the dyadic normaliser (their parameter sits at its +Inf optimum)
    for i in 1:n
        # within class: F[i](F[i]-1)/2 node pairs; both non-reciprocated exponents coincide
        @inbounds begin
            a = ifelse(!iszero(k_out[i]) && !iszero(k_in[i]), -α[i] - β[i], ninf)
            c = ifelse(!iszero(k_rec[i]), -γ[i] - γ[i], ninf)
            res -= ((F[i] * (F[i] - 1)) ÷ 2) * log1pexpsum(a, a, c)
        end
        # across classes
        for j in i+1:n
            @inbounds begin
                a = ifelse(!iszero(k_out[i]) && !iszero(k_in[j]), -α[i] - β[j], ninf)
                b = ifelse(!iszero(k_out[j]) && !iszero(k_in[i]), -α[j] - β[i], ninf)
                c = ifelse(!iszero(k_rec[i]) && !iszero(k_rec[j]), -γ[i] - γ[j], ninf)
                res -= F[i] * F[j] * log1pexpsum(a, b, c)
            end
        end
    end

    return res
end


"""
    L_RBCM_reduced(m::RBCM)

Return the log-likelihood of the RBCM model `m` based on the computed maximum likelihood parameters.

# Examples
```jldoctest
# Use with RBCM model:
julia> G = MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques());

julia> model = RBCM(G);

julia> solve_model!(model);

julia> L_RBCM_reduced(model);

```

See also [`L_RBCM_reduced(::Vector, ::Vector, ::Vector, ::Vector, ::Vector, ::Vector, ::Vector, ::Vector)`](@ref)
"""
function L_RBCM_reduced(m::RBCM)
    if m.status[:params_computed]
        return L_RBCM_reduced(m.θᵣ, m.dᵣ_out, m.dᵣ_in, m.dᵣ_rec, m.f, m.dᵣ_out_nz, m.dᵣ_in_nz, m.dᵣ_rec_nz, m.status[:d_unique])
    else
        throw(ArgumentError("The maximum likelihood parameters have not been computed yet"))
    end
end


# fill the masked fitness buffers x = exp(-α), y = exp(-β), z = exp(-γ), forcing channels with a
# zero-valued constraint to an exact zero fitness (their parameter sits at its +Inf optimum)
@inline function _RBCM_fitnesses!(x::AbstractVector, y::AbstractVector, z::AbstractVector, θ::AbstractVector,
                                  k_out::AbstractVector, k_in::AbstractVector, k_rec::AbstractVector, n::Int)
    α = @view θ[1:n]
    β = @view θ[n+1:2*n]
    γ = @view θ[2*n+1:3*n]
    @simd for i in eachindex(α)
        @inbounds x[i] = ifelse(iszero(k_out[i]), zero(eltype(x)), exp(-α[i]))
        @inbounds y[i] = ifelse(iszero(k_in[i]),  zero(eltype(y)), exp(-β[i]))
        @inbounds z[i] = ifelse(iszero(k_rec[i]), zero(eltype(z)), exp(-γ[i]))
    end
    return nothing
end


"""
    ∇L_RBCM_reduced!(∇L::AbstractVector, θ::AbstractVector, k_out::AbstractVector, k_in::AbstractVector, k_rec::AbstractVector, F::AbstractVector, nz_out::Vector, nz_in::Vector, nz_rec::Vector, x::AbstractVector, y::AbstractVector, z::AbstractVector, n::Int)

Compute the gradient of the log-likelihood of the reduced RBCM model using the exponential formulation in order to maintain convexity.

For the optimisation, this function will be used to generate an anonymous function associated with a specific model.
The function will update pre-allocated vectors (`∇L`, `x`, `y` and `z`) for speed. The gradient is non-allocating.

# Arguments
- `∇L`: the gradient of the log-likelihood of the reduced model
- `θ`: the maximum likelihood parameters of the model ([α; β; γ])
- `k_out`: the reduced non-reciprocated out-degree sequence (k→)
- `k_in`: the reduced non-reciprocated in-degree sequence (k←)
- `k_rec`: the reduced reciprocated degree sequence (k↔)
- `F`: the frequency of each triple in the degree sequences
- `nz_out`, `nz_in`, `nz_rec`: the indices of non-zero elements in the respective reduced degree sequences
- `x`, `y`, `z`: buffers for the exponentiated maximum likelihood parameters
- `n`: the number of unique node classes in the reduced model

# Examples
```jldoctest
# Use with RBCM model:
julia> G = MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques());

julia> model = RBCM(G);

julia> ∇L = zeros(length(model.θᵣ));

julia> x, y, z = zeros(length(model.xᵣ)), zeros(length(model.yᵣ)), zeros(length(model.zᵣ));

julia> ∇model_fun! = θ -> ∇L_RBCM_reduced!(∇L, θ, model.dᵣ_out, model.dᵣ_in, model.dᵣ_rec, model.f, model.dᵣ_out_nz, model.dᵣ_in_nz, model.dᵣ_rec_nz, x, y, z, model.status[:d_unique]);

julia> ∇model_fun!(ones(size(model.θᵣ)));

```
"""
function ∇L_RBCM_reduced!(  ∇L::AbstractVector, θ::AbstractVector,
                            k_out::AbstractVector, k_in::AbstractVector, k_rec::AbstractVector,
                            F::AbstractVector,
                            nz_out::Vector, nz_in::Vector, nz_rec::Vector,
                            x::AbstractVector, y::AbstractVector, z::AbstractVector,
                            n::Int)
    # set pre-allocated values (masked: dead channels have zero fitness)
    _RBCM_fitnesses!(x, y, z, θ, k_out, k_in, k_rec, n)
    # reset gradient to zero
    ∇L .= zero(eltype(∇L))
    o = one(eltype(∇L))

    # part related to α (branch-free inner reduction: (i==j) folds in the diagonal correction)
    for i ∈ nz_out
        @inbounds xᵢ, yᵢ, zᵢ = x[i], y[i], z[i]
        fx = zero(eltype(∇L))
        @inbounds @simd for j in 1:n
            D = o + xᵢ * y[j] + x[j] * yᵢ + zᵢ * z[j]
            fx += (F[j] - (i == j)) * y[j] / D
        end
        @inbounds ∇L[i] = xᵢ * F[i] * fx - F[i] * k_out[i]
    end
    # part related to β
    for i ∈ nz_in
        @inbounds xᵢ, yᵢ, zᵢ = x[i], y[i], z[i]
        fy = zero(eltype(∇L))
        @inbounds @simd for j in 1:n
            D = o + xᵢ * y[j] + x[j] * yᵢ + zᵢ * z[j]
            fy += (F[j] - (i == j)) * x[j] / D
        end
        @inbounds ∇L[n+i] = yᵢ * F[i] * fy - F[i] * k_in[i]
    end
    # part related to γ
    for i ∈ nz_rec
        @inbounds xᵢ, yᵢ, zᵢ = x[i], y[i], z[i]
        fz = zero(eltype(∇L))
        @inbounds @simd for j in 1:n
            D = o + xᵢ * y[j] + x[j] * yᵢ + zᵢ * z[j]
            fz += (F[j] - (i == j)) * z[j] / D
        end
        @inbounds ∇L[2*n+i] = zᵢ * F[i] * fz - F[i] * k_rec[i]
    end

    return ∇L
end


"""
    ∇L_RBCM_reduced_minus!(args...)

Compute minus the gradient of the log-likelihood of the reduced RBCM model using the exponential formulation in order to maintain convexity. Used for optimisation in a non-allocating manner.

See also [`∇L_RBCM_reduced!`](@ref)
"""
function ∇L_RBCM_reduced_minus!(∇L::AbstractVector, θ::AbstractVector,
                                k_out::AbstractVector, k_in::AbstractVector, k_rec::AbstractVector,
                                F::AbstractVector,
                                nz_out::Vector, nz_in::Vector, nz_rec::Vector,
                                x::AbstractVector, y::AbstractVector, z::AbstractVector,
                                n::Int)
    # set pre-allocated values (masked: dead channels have zero fitness)
    _RBCM_fitnesses!(x, y, z, θ, k_out, k_in, k_rec, n)
    # reset gradient to zero
    ∇L .= zero(eltype(∇L))
    o = one(eltype(∇L))

    # part related to α (branch-free inner reduction: (i==j) folds in the diagonal correction)
    for i ∈ nz_out
        @inbounds xᵢ, yᵢ, zᵢ = x[i], y[i], z[i]
        fx = zero(eltype(∇L))
        @inbounds @simd for j in 1:n
            D = o + xᵢ * y[j] + x[j] * yᵢ + zᵢ * z[j]
            fx += (F[j] - (i == j)) * y[j] / D
        end
        @inbounds ∇L[i] = -xᵢ * F[i] * fx + F[i] * k_out[i]
    end
    # part related to β
    for i ∈ nz_in
        @inbounds xᵢ, yᵢ, zᵢ = x[i], y[i], z[i]
        fy = zero(eltype(∇L))
        @inbounds @simd for j in 1:n
            D = o + xᵢ * y[j] + x[j] * yᵢ + zᵢ * z[j]
            fy += (F[j] - (i == j)) * x[j] / D
        end
        @inbounds ∇L[n+i] = -yᵢ * F[i] * fy + F[i] * k_in[i]
    end
    # part related to γ
    for i ∈ nz_rec
        @inbounds xᵢ, yᵢ, zᵢ = x[i], y[i], z[i]
        fz = zero(eltype(∇L))
        @inbounds @simd for j in 1:n
            D = o + xᵢ * y[j] + x[j] * yᵢ + zᵢ * z[j]
            fz += (F[j] - (i == j)) * z[j] / D
        end
        @inbounds ∇L[2*n+i] = -zᵢ * F[i] * fz + F[i] * k_rec[i]
    end

    return ∇L
end


"""
    RBCM_reduced_iter!(θ::AbstractVector, k_out::AbstractVector, k_in::AbstractVector, k_rec::AbstractVector, F::AbstractVector, nz_out::Vector, nz_in::Vector, nz_rec::Vector, x::AbstractVector, y::AbstractVector, z::AbstractVector, G::AbstractVector, n::Int)

Compute the next fixed-point iteration for the RBCM model using the exponential formulation in order to maintain convexity.
The function is non-allocating and will update pre-allocated vectors (`x`, `y`, `z` and `G`) for speed.

# Arguments
- `θ`: the maximum likelihood parameters of the model ([α; β; γ])
- `k_out`, `k_in`, `k_rec`: the reduced degree sequences (k→, k←, k↔)
- `F`: the frequency of each triple in the degree sequences
- `nz_out`, `nz_in`, `nz_rec`: the indices of non-zero elements in the respective reduced degree sequences
- `x`, `y`, `z`: buffers for the exponentiated maximum likelihood parameters
- `G`: buffer for computations
- `n`: the number of unique node classes in the reduced model

# Examples
```jldoctest
# Use with RBCM model:
julia> G = MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques());

julia> model = RBCM(G);

julia> buf = zeros(length(model.θᵣ));

julia> x, y, z = zeros(length(model.xᵣ)), zeros(length(model.yᵣ)), zeros(length(model.zᵣ));

julia> RBCM_FP! = θ -> RBCM_reduced_iter!(θ, model.dᵣ_out, model.dᵣ_in, model.dᵣ_rec, model.f, model.dᵣ_out_nz, model.dᵣ_in_nz, model.dᵣ_rec_nz, x, y, z, buf, model.status[:d_unique]);

julia> RBCM_FP!(ones(size(model.θᵣ)));

```
"""
function RBCM_reduced_iter!(θ::AbstractVector,
                            k_out::AbstractVector, k_in::AbstractVector, k_rec::AbstractVector,
                            F::AbstractVector,
                            nz_out::Vector, nz_in::Vector, nz_rec::Vector,
                            x::AbstractVector, y::AbstractVector, z::AbstractVector,
                            G::AbstractVector, n::Int)
    # set pre-allocated values (masked: dead channels have zero fitness)
    _RBCM_fitnesses!(x, y, z, θ, k_out, k_in, k_rec, n)
    G .= zero(eltype(G))
    o = one(eltype(G))
    # part related to α
    for i ∈ nz_out
        @inbounds xᵢ, yᵢ, zᵢ = x[i], y[i], z[i]
        @inbounds for j in 1:n
            D = o + xᵢ * y[j] + x[j] * yᵢ + zᵢ * z[j]
            G[i] += (F[j] - (i == j)) * y[j] / D
        end
        @inbounds G[i] = -log(k_out[i] / G[i])
    end
    # part related to β
    for i ∈ nz_in
        @inbounds xᵢ, yᵢ, zᵢ = x[i], y[i], z[i]
        @inbounds for j in 1:n
            D = o + xᵢ * y[j] + x[j] * yᵢ + zᵢ * z[j]
            G[n+i] += (F[j] - (i == j)) * x[j] / D
        end
        @inbounds G[n+i] = -log(k_in[i] / G[n+i])
    end
    # part related to γ
    for i ∈ nz_rec
        @inbounds xᵢ, yᵢ, zᵢ = x[i], y[i], z[i]
        @inbounds for j in 1:n
            D = o + xᵢ * y[j] + x[j] * yᵢ + zᵢ * z[j]
            G[2*n+i] += (F[j] - (i == j)) * z[j] / D
        end
        @inbounds G[2*n+i] = -log(k_rec[i] / G[2*n+i])
    end

    return G
end


"""
    initial_guess(m::RBCM; method::Symbol=:degrees)

Compute an initial guess for the maximum likelihood parameters of the RBCM model `m` using the method `method`.

The methods available are:
- `:degrees` (default): the initial guess is computed using the degree sequences, i.e. ``\\theta = [-\\log(k^{→}); -\\log(k^{←}); -\\log(k^{↔})]``
- `:degrees_minor`: the initial guess is computed using the degree sequences and the number of edges, i.e. ``\\theta_{ch} = -\\log(k_{ch}/(\\sqrt{E} + 1))``
- `:random`: the initial guess is computed using random values between 0 and 1, i.e. ``\\theta_{i} = -\\log(r_{i})`` where ``r_{i} \\sim U(0,1)``
- `:uniform`: the initial guess is uniformily set to 0.5, i.e. ``\\theta_{i} = -\\log(0.5)``
- `:chung_lu`: the initial guess is computed using the degree sequences and the number of edges, i.e. ``\\theta_{ch} = -\\log(k_{ch}/(2E))``

Entries whose constraint is zero get an infinite parameter (their fitness is exactly zero); the solution methods
handle these separately.

# Examples
```jldoctest
julia> model = RBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()));

julia> initial_guess(model, method=:random);

julia> initial_guess(model, method=:uniform);

julia> initial_guess(model, method=:degrees_minor);

julia> initial_guess(model, method=:chung_lu);

julia> initial_guess(model);

```
"""
function initial_guess(m::RBCM; method::Symbol=:degrees)
    if isequal(method, :degrees)
        return Vector{precision(m)}(vcat(-log.(m.dᵣ_out), -log.(m.dᵣ_in), -log.(m.dᵣ_rec)))
    elseif isequal(method, :degrees_minor)
        isnothing(m.G) ? throw(ArgumentError("Cannot compute the number of edges because the model has no underlying graph (m.G == nothing)")) : nothing
        return Vector{precision(m)}(vcat(-log.(m.dᵣ_out ./ (sqrt(Graphs.ne(m.G)) + 1)), -log.(m.dᵣ_in ./ (sqrt(Graphs.ne(m.G)) + 1)), -log.(m.dᵣ_rec ./ (sqrt(Graphs.ne(m.G)) + 1))))
    elseif isequal(method, :random)
        return Vector{precision(m)}(-log.(rand(precision(m), 3*length(m.dᵣ_out))))
    elseif isequal(method, :uniform)
        return Vector{precision(m)}(-log.(0.5 .* ones(precision(m), 3*length(m.dᵣ_out))))
    elseif isequal(method, :chung_lu)
        isnothing(m.G) ? throw(ArgumentError("Cannot compute the number of edges because the model has no underlying graph (m.G == nothing)")) : nothing
        return Vector{precision(m)}(vcat(-log.(m.dᵣ_out ./ (2 * Graphs.ne(m.G))), -log.(m.dᵣ_in ./ (2 * Graphs.ne(m.G))), -log.(m.dᵣ_rec ./ (2 * Graphs.ne(m.G)))))
    else
        throw(ArgumentError("The initial guess method $(method) is not supported"))
    end
end


"""
    set_xᵣ!(m::RBCM)

Set the value of xᵣ to exp(-αᵣ) for the RBCM model `m`
"""
function set_xᵣ!(m::RBCM)
    if m.status[:params_computed]
        αᵣ = @view m.θᵣ[1:m.status[:d_unique]]
        m.xᵣ .= exp.(-αᵣ)
    else
        throw(ArgumentError("The parameters have not been computed yet"))
    end
end

"""
    set_yᵣ!(m::RBCM)

Set the value of yᵣ to exp(-βᵣ) for the RBCM model `m`
"""
function set_yᵣ!(m::RBCM)
    if m.status[:params_computed]
        βᵣ = @view m.θᵣ[m.status[:d_unique]+1:2*m.status[:d_unique]]
        m.yᵣ .= exp.(-βᵣ)
    else
        throw(ArgumentError("The parameters have not been computed yet"))
    end
end

"""
    set_zᵣ!(m::RBCM)

Set the value of zᵣ to exp(-γᵣ) for the RBCM model `m`
"""
function set_zᵣ!(m::RBCM)
    if m.status[:params_computed]
        γᵣ = @view m.θᵣ[2*m.status[:d_unique]+1:end]
        m.zᵣ .= exp.(-γᵣ)
    else
        throw(ArgumentError("The parameters have not been computed yet"))
    end
end


"""
    solve_model!(m::RBCM)

Compute the likelihood maximising parameters of the RBCM model `m`.

# Arguments
- `method::Symbol`: solution method to use, can be `:fixedpoint` (default), or :$(join(keys(MaxEntropyGraphs.optimization_methods), ", :", " and :")).
- `initial::Symbol`: initial guess for the parameters ``\\Theta``, can be :degrees (default), :degrees_minor, :random, :uniform, or :chung_lu.
- `maxiters::Int`: maximum number of iterations for the solver (defaults to 1000).
- `verbose::Bool`: set to show log messages (defaults to false).
- `ftol::Union{Real, Nothing}`: tolerance for the fixedpoint method (defaults to `nothing`, i.e. 1e-8). It bounds the fixed-point *increment* ``\\|G(\\theta) - \\theta\\|_\\infty`` in **parameter** space; it is **not** the constraint residual, and it is ignored by every other method. Use [`constraint_residual`](@ref) to measure how well the expected degrees actually match the observed ones.
- `abstol::Union{Number, Nothing}`: absolute function tolerance for convergence with the other methods (defaults to `nothing`).
- `reltol::Union{Number, Nothing}`: relative function tolerance for convergence with the other methods (defaults to `nothing`).
- `g_tol::Union{Number, Nothing}`: gradient tolerance for the gradient-based methods (maps to Optim's `g_abstol`); set e.g. `1e-5` to stop before over-converging (defaults to `nothing`, i.e. Optim's tight default). The gradient of this model *is* its constraint residual (up to the degree multiplicities), but `g_abstol` is a stopping criterion rather than a guarantee: Optim can also stop on its function or parameter convergence checks and report success without the gradient ever reaching `g_tol`. Verify what was actually achieved with [`constraint_residual`](@ref).
- `AD_method::Symbol`: autodiff method to use, can be any of :$(join(keys(MaxEntropyGraphs.AD_methods), ", :", " and :")). Performance depends on the size of the problem (defaults to `:AutoZygote`),
- `analytical_gradient::Bool`: set the use the analytical gradient instead of the one generated with autodiff (defaults to `false`)

!!! note
    The fixed-point method (with Anderson acceleration) is stable for the RBCM on typical networks.
    On *degenerate* inputs — in particular fully reciprocal networks (k→ = k← = 0 everywhere), where only
    the γ-channel is identified — the accelerated fixed point can overshoot to non-finite values; use a
    gradient-based method (e.g. `method=:BFGS`) in that case.

# Examples
```jldoctest RBCM_solve
# default use
julia> model = RBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()));

julia> solve_model!(model);

```
"""
function solve_model!(m::RBCM;  # common settings
                                method::Symbol=:fixedpoint,
                                initial::Symbol=:degrees,
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
    N <: Union{Float16, Float32} && @warn "Solving in $(N) precision is experimental and may not converge; low precision is intended for storage. Consider Float64 for the solve." maxlog=1
    # `ftol` is accepted on every path but only ever reaches the fixed point solver: say so rather than
    # ignoring it silently (only when it was actually passed, so a default solve stays quiet)
    method ≠ :fixedpoint && !isnothing(ftol) && @warn _ftol_unused_msg(method) maxlog=1
    ftol = isnothing(ftol) ? _DEFAULT_FTOL : ftol
    # initial guess
    θ₀ = initial_guess(m, method=initial)
    # channels with a zero-valued constraint sit at their analytical +Inf optimum. The set is derived
    # from the CONSTRAINTS (not from the initial guess: e.g. :uniform/:random give finite values there,
    # and the likelihood is deliberately flat in those coordinates, so the solver would leave whatever
    # the guess contained). They are solved at a finite placeholder and pinned to +Inf afterwards.
    nᵣ = m.status[:d_unique]
    ind_inf = vcat(findall(iszero, m.dᵣ_out), nᵣ .+ findall(iszero, m.dᵣ_in), 2*nᵣ .+ findall(iszero, m.dᵣ_rec))
    θ₀[ind_inf] .= zero(N)
    if method == :fixedpoint
        # initiate buffers
        x_buffer = zeros(N, m.status[:d_unique]); # buffer for x = exp(-α)
        y_buffer = zeros(N, m.status[:d_unique]); # buffer for y = exp(-β)
        z_buffer = zeros(N, m.status[:d_unique]); # buffer for z = exp(-γ)
        G_buffer = zeros(N, length(m.θᵣ));        # buffer for G(x)
        # define fixed point function
        FP_model! = (θ::Vector) -> RBCM_reduced_iter!(θ, m.dᵣ_out, m.dᵣ_in, m.dᵣ_rec, m.f, m.dᵣ_out_nz, m.dᵣ_in_nz, m.dᵣ_rec_nz, x_buffer, y_buffer, z_buffer, G_buffer, m.status[:d_unique])
        # obtain solution
        sol = NLsolve.fixedpoint(FP_model!, θ₀, method=:anderson, ftol=ftol, iterations=maxiters);
        if NLsolve.converged(sol)
            if verbose
                @info "Fixed point iteration converged after $(sol.iterations) iterations"
            end
            m.θᵣ .= sol.zero;
            m.θᵣ[ind_inf] .= Inf;
            m.status[:params_computed] = true;
            set_xᵣ!(m);
            set_yᵣ!(m);
            set_zᵣ!(m);
        else
            throw(ConvergenceError(method, nothing))
        end
    else
        if analytical_gradient
            # initiate buffers
            x_buffer = zeros(N, m.status[:d_unique]); # buffer for x = exp(-α)
            y_buffer = zeros(N, m.status[:d_unique]); # buffer for y = exp(-β)
            z_buffer = zeros(N, m.status[:d_unique]); # buffer for z = exp(-γ)
            # define gradient function for optimisation.jl
            grad! = (G, θ, p) -> ∇L_RBCM_reduced_minus!(G, θ, m.dᵣ_out, m.dᵣ_in, m.dᵣ_rec, m.f, m.dᵣ_out_nz, m.dᵣ_in_nz, m.dᵣ_rec_nz, x_buffer, y_buffer, z_buffer, m.status[:d_unique]);
        end
        # define objective function and its AD method
        f = AD_method ∈ keys(AD_methods)            ? Optimization.OptimizationFunction( (θ, p) ->   -L_RBCM_reduced(θ, m.dᵣ_out, m.dᵣ_in, m.dᵣ_rec, m.f, m.dᵣ_out_nz, m.dᵣ_in_nz, m.dᵣ_rec_nz, m.status[:d_unique]),
                                                                                         AD_methods[AD_method],
                                                                                         grad = analytical_gradient ? grad! : nothing)                      : throw(ArgumentError("The AD method $(AD_method) is not supported (yet)"))
        prob = Optimization.OptimizationProblem(f, θ₀);
        # obtain solution
        method ∈ keys(optimization_methods) || throw(ArgumentError("The method $(method) is not supported (yet)"))
        solve_kwargs = isnothing(g_tol) ? (; maxiters = maxiters, abstol = abstol, reltol = reltol) :
                                          (; maxiters = maxiters, abstol = abstol, reltol = reltol, g_abstol = g_tol)
        sol = Optimization.solve(prob, optimization_methods[method]; solve_kwargs...)
        # check convergence
        if Optimization.SciMLBase.successful_retcode(sol.retcode)
            if verbose
                @info """$(method) optimisation converged after $(@sprintf("%1.2e", sol.stats.time)) seconds (Optimization.jl return code: $("$(sol.retcode)"))"""
            end
            m.θᵣ .= sol.u;
            m.θᵣ[ind_inf] .= Inf;
            m.status[:params_computed] = true;
            set_xᵣ!(m);
            set_yᵣ!(m);
            set_zᵣ!(m);
        else
            throw(ConvergenceError(method, sol.retcode))
        end
    end

    return m, sol
end


"""
    precision(m::RBCM)

Determine the compute precision of the RBCM model `m`.

# Examples
```jldoctest
julia> model = RBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()));

julia> MaxEntropyGraphs.precision(model)
Float64
```
"""
precision(m::RBCM) = typeof(m).parameters[2]


# ----------------------------------------------------------------------------------------------------------------------
# dyadic probabilities
#
# Under the RBCM each unordered dyad (i,j) is an independent categorical variable over four mutually exclusive
# states with probabilities p⭢ (i→j only), p⭠ (j→i only), p⭤ (both) and p∅ (absent), normalised by
# Dᵢⱼ = 1 + xᵢyⱼ + xⱼyᵢ + zᵢzⱼ. Note that ⟨aᵢⱼ⟩ = p⭢ᵢⱼ + p⭤ᵢⱼ and that aᵢⱼ, aⱼᵢ are correlated within a dyad:
# Cov(aᵢⱼ, aⱼᵢ) = p⭤ᵢⱼ - ⟨aᵢⱼ⟩⟨aⱼᵢ⟩ (Squartini & Garlaschelli (2011), Eq. C.17).
# ----------------------------------------------------------------------------------------------------------------------

"""
    p⭢(m::RBCM, i::Int, j::Int)

Return the probability that the dyad `(i,j)` holds a single non-reciprocated link `i→j` under the RBCM model `m`
(``p^{→}_{ij} = x_iy_j/D_{ij}`` with ``D_{ij} = 1 + x_iy_j + x_jy_i + z_iz_j``).

❗ For performance reasons, the function does not check if the node pair is valid or if the parameters have been computed.
"""
function p⭢(m::RBCM, i::Int, j::Int)
    i == j && return zero(precision(m))
    @inbounds ri, rj = m.dᵣ_ind[i], m.dᵣ_ind[j]
    @inbounds xy = m.xᵣ[ri] * m.yᵣ[rj]
    @inbounds D = one(precision(m)) + xy + m.xᵣ[rj] * m.yᵣ[ri] + m.zᵣ[ri] * m.zᵣ[rj]
    return xy / D
end

"""
    p⭠(m::RBCM, i::Int, j::Int)

Return the probability that the dyad `(i,j)` holds a single non-reciprocated link `j→i` under the RBCM model `m`
(``p^{←}_{ij} = x_jy_i/D_{ij} = p^{→}_{ji}``).

❗ For performance reasons, the function does not check if the node pair is valid or if the parameters have been computed.
"""
p⭠(m::RBCM, i::Int, j::Int) = p⭢(m, j, i)

"""
    p⭤(m::RBCM, i::Int, j::Int)

Return the probability that the dyad `(i,j)` holds a reciprocated link pair (`i→j` and `j→i`) under the RBCM model `m`
(``p^{↔}_{ij} = z_iz_j/D_{ij}``).

❗ For performance reasons, the function does not check if the node pair is valid or if the parameters have been computed.
"""
function p⭤(m::RBCM, i::Int, j::Int)
    i == j && return zero(precision(m))
    @inbounds ri, rj = m.dᵣ_ind[i], m.dᵣ_ind[j]
    @inbounds zz = m.zᵣ[ri] * m.zᵣ[rj]
    @inbounds D = one(precision(m)) + m.xᵣ[ri] * m.yᵣ[rj] + m.xᵣ[rj] * m.yᵣ[ri] + zz
    return zz / D
end

"""
    p∅(m::RBCM, i::Int, j::Int)

Return the probability that the dyad `(i,j)` holds no links under the RBCM model `m` (``p^{∅}_{ij} = 1/D_{ij}``).

❗ For performance reasons, the function does not check if the node pair is valid or if the parameters have been computed.
"""
function p∅(m::RBCM, i::Int, j::Int)
    i == j && return zero(precision(m))
    @inbounds ri, rj = m.dᵣ_ind[i], m.dᵣ_ind[j]
    @inbounds D = one(precision(m)) + m.xᵣ[ri] * m.yᵣ[rj] + m.xᵣ[rj] * m.yᵣ[ri] + m.zᵣ[ri] * m.zᵣ[rj]
    return one(precision(m)) / D
end

"""
    A(m::RBCM,i::Int,j::Int)

Return the expected value of the adjacency matrix for the RBCM model `m` at the node pair `(i,j)`
(``⟨a_{ij}⟩ = p^{→}_{ij} + p^{↔}_{ij}``).

❗ For performance reasons, the function does not check:
- if the node pair is valid.
- if the parameters of the model have been computed.
"""
function A(m::RBCM, i::Int, j::Int)
    i == j && return zero(precision(m))
    @inbounds ri, rj = m.dᵣ_ind[i], m.dᵣ_ind[j]
    @inbounds xy = m.xᵣ[ri] * m.yᵣ[rj]
    @inbounds zz = m.zᵣ[ri] * m.zᵣ[rj]
    @inbounds D = one(precision(m)) + xy + m.xᵣ[rj] * m.yᵣ[ri] + zz
    return (xy + zz) / D
end


"""
    _dyadic_probability_matrices(m::RBCM)

Compute the three dyadic probability matrices `(P̂, R̂, Ẑ)` of the RBCM model `m`, where `P̂[i,j] = p⭢(m,i,j)`
(single link i→j), `R̂[i,j] = p⭤(m,i,j)` (reciprocated dyad) and `Ẑ[i,j] = p∅(m,i,j)` (absent dyad).
The matrix of `p⭠` probabilities is `transpose(P̂)`. To match the conventions of the motif counting kernels
(`_motif_base_matrices`/`_motif_count`), `Ẑ` carries a **unit diagonal** while `P̂` and `R̂` have zero diagonals.

These matrices are the correct expected counterparts of the motif base matrices `P/Q/R/Z` under the RBCM:
because `aᵢⱼ` and `aⱼᵢ` are correlated within a dyad, they can **not** be obtained from the expected adjacency
matrix `Ĝ` (e.g. `⟨aᵢⱼaⱼᵢ⟩ = p⭤ᵢⱼ ≠ ĜᵢⱼĜⱼᵢ`).
"""
function _dyadic_probability_matrices(m::RBCM)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    n = m.status[:d]
    P̂ = zeros(precision(m), n, n)
    R̂ = zeros(precision(m), n, n)
    Ẑ = zeros(precision(m), n, n)
    x = m.xᵣ[m.dᵣ_ind]
    y = m.yᵣ[m.dᵣ_ind]
    z = m.zᵣ[m.dᵣ_ind]
    o = one(precision(m))
    for i = 1:n
        @inbounds Ẑ[i,i] = o # unit diagonal, matching the Z = (1 .- A).*(1 .- Aᵀ) convention
        @simd for j = 1:n
            if i ≠ j
                @inbounds xiyj = x[i] * y[j]
                @inbounds zizj = z[i] * z[j]
                @inbounds D = o + xiyj + x[j] * y[i] + zizj
                @inbounds P̂[i,j] = xiyj / D
                @inbounds R̂[i,j] = zizj / D
                @inbounds Ẑ[i,j] = o / D
            end
        end
    end
    return P̂, R̂, Ẑ
end


"""
    Ĝ(m::RBCM)

Compute the expected adjacency matrix for the RBCM model `m` (``⟨a_{ij}⟩ = p^{→}_{ij} + p^{↔}_{ij}``).

*Note*: under the RBCM the entries `aᵢⱼ` and `aⱼᵢ` are correlated within a dyad, so `Ĝ` alone does not fully
characterise the dyadic joint distribution (see [`MaxEntropyGraphs._dyadic_probability_matrices`](@ref)).
"""
function Ĝ(m::RBCM)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))

    # get network size => this is the full size
    n = m.status[:d]
    # initiate G
    G = zeros(precision(m), n, n)
    # initiate x, y and z
    x = m.xᵣ[m.dᵣ_ind]
    y = m.yᵣ[m.dᵣ_ind]
    z = m.zᵣ[m.dᵣ_ind]
    o = one(precision(m))
    # compute G
    for i = 1:n
        @simd for j = 1:n
            if i ≠ j
                @inbounds xiyj = x[i] * y[j]
                @inbounds zizj = z[i] * z[j]
                @inbounds D = o + xiyj + x[j] * y[i] + zizj
                @inbounds G[i,j] = (xiyj + zizj) / D
            end
        end
    end

    return G
end


"""
    set_Ĝ!(m::RBCM)

Set the expected adjacency matrix for the RBCM model `m`
"""
function set_Ĝ!(m::RBCM)
    m.Ĝ = Ĝ(m)
    m.status[:G_computed] = true
    return m.Ĝ
end


"""
    σˣ(m::RBCM{T,N}) where {T,N}

Compute the standard deviation for the elements of the adjacency matrix for the RBCM model `m`
(``σ[a_{ij}] = \\sqrt{⟨a_{ij}⟩(1-⟨a_{ij}⟩)}``).

**Note:** read as "sigma star". The `aᵢⱼ` remain Bernoulli marginally; the within-dyad correlation between
`aᵢⱼ` and `aⱼᵢ` is captured separately (see [`MaxEntropyGraphs._cov_dyads`](@ref)) and is accounted for by `σₓ`.
"""
function σˣ(m::RBCM)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    # check network size => this is the full size
    n = m.status[:d]
    # initiate σ
    σ = zeros(precision(m), n, n)
    # initiate x, y and z
    x = m.xᵣ[m.dᵣ_ind]
    y = m.yᵣ[m.dᵣ_ind]
    z = m.zᵣ[m.dᵣ_ind]
    o = one(precision(m))
    # compute σ
    for i = 1:n
        @simd for j = 1:n
            if i ≠ j
                @inbounds xiyj = x[i] * y[j]
                @inbounds zizj = z[i] * z[j]
                @inbounds D = o + xiyj + x[j] * y[i] + zizj
                @inbounds a = (xiyj + zizj) / D
                @inbounds σ[i,j] = sqrt(a * (o - a))
            end
        end
    end

    return σ
end

"""
    set_σ!(m::RBCM)

Set the standard deviation for the elements of the adjacency matrix for the RBCM model `m`
"""
function set_σ!(m::RBCM)
    m.σ = σˣ(m)
    m.status[:σ_computed] = true
    return m.σ
end


"""
    _cov_dyads(m::RBCM)

Compute the within-dyad covariance matrix `C` of the RBCM model `m`, with
``C_{ij} = Cov(a_{ij}, a_{ji}) = p^{↔}_{ij} - ⟨a_{ij}⟩⟨a_{ji}⟩`` for `i ≠ j` and zero diagonal
(Squartini & Garlaschelli (2011), Eq. C.17). The matrix is symmetric; distinct dyads are independent.
"""
function _cov_dyads(m::RBCM)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    n = m.status[:d]
    C = zeros(precision(m), n, n)
    x = m.xᵣ[m.dᵣ_ind]
    y = m.yᵣ[m.dᵣ_ind]
    z = m.zᵣ[m.dᵣ_ind]
    o = one(precision(m))
    for i = 1:n
        for j = i+1:n
            @inbounds xiyj = x[i] * y[j]
            @inbounds xjyi = x[j] * y[i]
            @inbounds zizj = z[i] * z[j]
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
    rand(m::RBCM; precomputed::Bool=false)

Generate a random graph from the RBCM model `m`. Each dyad `(i,j)` is sampled independently from its
four-state distribution (single link `i→j`, single link `j→i`, reciprocated pair, absent) using a single
uniform draw against the cumulative probabilities (p⭢, p⭠, p⭤).

# Arguments:
- `precomputed::Bool`: only `false` is supported for the RBCM. The expected adjacency matrix `Ĝ` does not
  characterise the dyadic joint distribution (the states of `aᵢⱼ` and `aⱼᵢ` are correlated), so sampling from
  `Ĝ` would not reproduce the reciprocated degree sequence. An `ArgumentError` is thrown when `precomputed=true`.

# Examples
```jldoctest
# generate a RBCM model of the macaques network
julia> G = MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques());

julia> model = RBCM(G);

julia> solve_model!(model); # compute the maximum likelihood parameters

julia> typeof(rand(model))
Graphs.SimpleGraphs.SimpleDiGraph{Int64}

```
"""
function rand(m::RBCM; precomputed::Bool=false, rng::AbstractRNG=default_rng())
    precomputed && throw(ArgumentError("Sampling from the precomputed expected adjacency matrix is not supported for the RBCM: Ĝ does not capture the within-dyad correlation between aᵢⱼ and aⱼᵢ. Use precomputed=false."))
    # check if possible to use parameters
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    # initiate x, y and z
    x = m.xᵣ[m.dᵣ_ind]
    y = m.yᵣ[m.dᵣ_ind]
    z = m.zᵣ[m.dᵣ_ind]
    o = one(precision(m))
    # generate random edges per dyad
    edges = Vector{Graphs.SimpleGraphs.SimpleEdge{Int}}()
    n = m.status[:d]
    for i = 1:n
        for j = i+1:n
            @inbounds xiyj = x[i] * y[j]
            @inbounds xjyi = x[j] * y[i]
            @inbounds zizj = z[i] * z[j]
            D = o + xiyj + xjyi + zizj
            u = rand(rng) * D # single uniform draw scaled by the normaliser (avoids three divisions)
            if u < xiyj
                push!(edges, Graphs.SimpleGraphs.SimpleEdge(i, j))
            elseif u < xiyj + xjyi
                push!(edges, Graphs.SimpleGraphs.SimpleEdge(j, i))
            elseif u < xiyj + xjyi + zizj
                push!(edges, Graphs.SimpleGraphs.SimpleEdge(i, j))
                push!(edges, Graphs.SimpleGraphs.SimpleEdge(j, i))
            end
        end
    end
    G = Graphs.SimpleDiGraphFromIterator(edges)

    # deal with edge case where no edges are generated for the last node(s) in the graph
    while Graphs.nv(G) < m.status[:d]
        Graphs.add_vertex!(G)
    end

    return G
end


"""
    rand(m::RBCM, n::Int; precomputed::Bool=false)

Generate `n` random graphs from the RBCM model `m`. If multithreading is available, the graphs are generated in parallel.

# Arguments:
- `precomputed::Bool`: only `false` is supported for the RBCM (see [`rand(::RBCM)`](@ref)).

# Examples
```jldoctest
# generate a RBCM model of the macaques network
julia> G = MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques());

julia> model = RBCM(G);

julia> solve_model!(model); # compute the maximum likelihood parameters

julia> typeof(rand(model, 10))
Vector{SimpleDiGraph{Int64}} (alias for Array{Graphs.SimpleGraphs.SimpleDiGraph{Int64}, 1})

```
"""
function rand(m::RBCM, n::Int; precomputed::Bool=false, rng::AbstractRNG=default_rng())
    # pre-allocate
    res = Vector{Graphs.SimpleDiGraph{Int}}(undef, n)
    # per-sample seeds for reproducible, thread-schedule-independent sampling
    seeds = rand(rng, UInt64, n)
    # fill vector using threads
    Threads.@threads for i in 1:n
        res[i] = rand(m; precomputed=precomputed, rng=Xoshiro(seeds[i]))
    end

    return res
end


# ----------------------------------------------------------------------------------------------------------------------
# degree accessors
# ----------------------------------------------------------------------------------------------------------------------

"""
    nonreciprocated_outdegree(m::RBCM, i::Int; method=:reduced)

Return the expected non-reciprocated out-degree ``⟨k^{→}_i⟩ = \\sum_{j≠i} p^{→}_{ij}`` for node `i` of the RBCM model `m`.

# Arguments
- `m::RBCM`: the RBCM model
- `i::Int`: the node for which to compute the degree.
- `method::Symbol`: the method to use for computing the degree. Can be any of the following:
    - `:reduced` (default) uses the reduced model parameters for performance reasons.
    - `:full` sums the dyadic probabilities over all node pairs.
    - `:adjacency` is **not** supported for the reciprocal degree split: the expected adjacency matrix `Ĝ`
      only holds ``⟨a_{ij}⟩ = p^{→}_{ij} + p^{↔}_{ij}`` and cannot separate the two contributions.

# Examples
```jldoctest
julia> model = RBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()));

julia> solve_model!(model);

julia> typeof([nonreciprocated_outdegree(model, 1), nonreciprocated_outdegree(model, 1, method=:full)])
Vector{Float64} (alias for Array{Float64, 1})

```
"""
function nonreciprocated_outdegree(m::RBCM, i::Int; method::Symbol=:reduced)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    i > m.status[:d] ? throw(ArgumentError("Attempted to access node $i in a $(m.status[:d]) node graph")) : nothing

    if method == :reduced
        res = zero(precision(m))
        i_red = m.dᵣ_ind[i] # find matching index in reduced model
        o = one(precision(m))
        for j in eachindex(m.xᵣ)
            @inbounds xiyj = m.xᵣ[i_red] * m.yᵣ[j]
            @inbounds D = o + xiyj + m.xᵣ[j] * m.yᵣ[i_red] + m.zᵣ[i_red] * m.zᵣ[j]
            res += @inbounds xiyj / D * (m.f[j] - (i_red == j)) # subtract 1 within class because the diagonal is not counted
        end
    elseif method == :full
        # using all dyadic probabilities
        res = zero(precision(m))
        for j in eachindex(m.d_out)
            res += p⭢(m, i, j)
        end
    elseif method == :adjacency
        throw(ArgumentError("The reciprocal degree split cannot be recovered from the expected adjacency matrix Ĝ (it only holds ⟨aᵢⱼ⟩ = p⭢ + p⭤); use method=:reduced or :full"))
    else
        throw(ArgumentError("Unknown method $method"))
    end

    return res
end

"""
    nonreciprocated_outdegree(m::RBCM[, v]; method=:reduced)

Return a vector corresponding to the expected non-reciprocated out-degree of the RBCM model `m` for each node.
If v is specified, only return the values for the nodes in v.

# Examples
```jldoctest
julia> model = RBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()));

julia> solve_model!(model);

julia> typeof(nonreciprocated_outdegree(model))
Vector{Float64} (alias for Array{Float64, 1})

```
"""
nonreciprocated_outdegree(m::RBCM, v::Vector{Int}=collect(1:m.status[:d]); method::Symbol=:reduced) = [nonreciprocated_outdegree(m, i, method=method) for i in v]


"""
    nonreciprocated_indegree(m::RBCM, i::Int; method=:reduced)

Return the expected non-reciprocated in-degree ``⟨k^{←}_i⟩ = \\sum_{j≠i} p^{←}_{ij}`` for node `i` of the RBCM model `m`.

# Arguments
- `m::RBCM`: the RBCM model
- `i::Int`: the node for which to compute the degree.
- `method::Symbol`: `:reduced` (default), `:full`, see [`nonreciprocated_outdegree`](@ref) (`:adjacency` is not supported).

# Examples
```jldoctest
julia> model = RBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()));

julia> solve_model!(model);

julia> typeof([nonreciprocated_indegree(model, 1), nonreciprocated_indegree(model, 1, method=:full)])
Vector{Float64} (alias for Array{Float64, 1})

```
"""
function nonreciprocated_indegree(m::RBCM, i::Int; method::Symbol=:reduced)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    i > m.status[:d] ? throw(ArgumentError("Attempted to access node $i in a $(m.status[:d]) node graph")) : nothing

    if method == :reduced
        res = zero(precision(m))
        i_red = m.dᵣ_ind[i] # find matching index in reduced model
        o = one(precision(m))
        for j in eachindex(m.xᵣ)
            @inbounds xjyi = m.xᵣ[j] * m.yᵣ[i_red]
            @inbounds D = o + m.xᵣ[i_red] * m.yᵣ[j] + xjyi + m.zᵣ[i_red] * m.zᵣ[j]
            res += @inbounds xjyi / D * (m.f[j] - (i_red == j)) # subtract 1 within class because the diagonal is not counted
        end
    elseif method == :full
        # using all dyadic probabilities
        res = zero(precision(m))
        for j in eachindex(m.d_out)
            res += p⭠(m, i, j)
        end
    elseif method == :adjacency
        throw(ArgumentError("The reciprocal degree split cannot be recovered from the expected adjacency matrix Ĝ (it only holds ⟨aᵢⱼ⟩ = p⭢ + p⭤); use method=:reduced or :full"))
    else
        throw(ArgumentError("Unknown method $method"))
    end

    return res
end

"""
    nonreciprocated_indegree(m::RBCM[, v]; method=:reduced)

Return a vector corresponding to the expected non-reciprocated in-degree of the RBCM model `m` for each node.
If v is specified, only return the values for the nodes in v.

# Examples
```jldoctest
julia> model = RBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()));

julia> solve_model!(model);

julia> typeof(nonreciprocated_indegree(model))
Vector{Float64} (alias for Array{Float64, 1})

```
"""
nonreciprocated_indegree(m::RBCM, v::Vector{Int}=collect(1:m.status[:d]); method::Symbol=:reduced) = [nonreciprocated_indegree(m, i, method=method) for i in v]


"""
    reciprocated_degree(m::RBCM, i::Int; method=:reduced)

Return the expected reciprocated degree ``⟨k^{↔}_i⟩ = \\sum_{j≠i} p^{↔}_{ij}`` for node `i` of the RBCM model `m`.

# Arguments
- `m::RBCM`: the RBCM model
- `i::Int`: the node for which to compute the degree.
- `method::Symbol`: `:reduced` (default), `:full`, see [`nonreciprocated_outdegree`](@ref) (`:adjacency` is not supported).

# Examples
```jldoctest
julia> model = RBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()));

julia> solve_model!(model);

julia> typeof([reciprocated_degree(model, 1), reciprocated_degree(model, 1, method=:full)])
Vector{Float64} (alias for Array{Float64, 1})

```
"""
function reciprocated_degree(m::RBCM, i::Int; method::Symbol=:reduced)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    i > m.status[:d] ? throw(ArgumentError("Attempted to access node $i in a $(m.status[:d]) node graph")) : nothing

    if method == :reduced
        res = zero(precision(m))
        i_red = m.dᵣ_ind[i] # find matching index in reduced model
        o = one(precision(m))
        for j in eachindex(m.xᵣ)
            @inbounds zizj = m.zᵣ[i_red] * m.zᵣ[j]
            @inbounds D = o + m.xᵣ[i_red] * m.yᵣ[j] + m.xᵣ[j] * m.yᵣ[i_red] + zizj
            res += @inbounds zizj / D * (m.f[j] - (i_red == j)) # subtract 1 within class because the diagonal is not counted
        end
    elseif method == :full
        # using all dyadic probabilities
        res = zero(precision(m))
        for j in eachindex(m.d_out)
            res += p⭤(m, i, j)
        end
    elseif method == :adjacency
        throw(ArgumentError("The reciprocal degree split cannot be recovered from the expected adjacency matrix Ĝ (it only holds ⟨aᵢⱼ⟩ = p⭢ + p⭤); use method=:reduced or :full"))
    else
        throw(ArgumentError("Unknown method $method"))
    end

    return res
end

"""
    reciprocated_degree(m::RBCM[, v]; method=:reduced)

Return a vector corresponding to the expected reciprocated degree of the RBCM model `m` for each node.
If v is specified, only return the values for the nodes in v.

# Examples
```jldoctest
julia> model = RBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()));

julia> solve_model!(model);

julia> typeof(reciprocated_degree(model))
Vector{Float64} (alias for Array{Float64, 1})

```
"""
reciprocated_degree(m::RBCM, v::Vector{Int}=collect(1:m.status[:d]); method::Symbol=:reduced) = [reciprocated_degree(m, i, method=method) for i in v]


"""
    outdegree(m::RBCM, i::Int; method=:reduced)

Return the expected out-degree (in the `Graphs.jl` sense, i.e. ``⟨k^{out}_i⟩ = ⟨k^{→}_i⟩ + ⟨k^{↔}_i⟩``) for node `i`
of the RBCM model `m`.

# Arguments
- `m::RBCM`: the RBCM model
- `i::Int`: the node for which to compute the degree.
- `method::Symbol`: the method to use for computing the degree. Can be any of the following:
    - `:reduced` (default) uses the reduced model parameters for performance reasons.
    - `:full` uses all elements of the expected adjacency matrix.
    - `:adjacency` uses the precomputed adjacency matrix `m.Ĝ` of the model.

# Examples
```jldoctest
julia> model = RBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()));

julia> solve_model!(model);

julia> set_Ĝ!(model);

julia> typeof([outdegree(model, 1), outdegree(model, 1, method=:full), outdegree(model, 1, method=:adjacency)])
Vector{Float64} (alias for Array{Float64, 1})

```
"""
function outdegree(m::RBCM, i::Int; method::Symbol=:reduced)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    i > m.status[:d] ? throw(ArgumentError("Attempted to access node $i in a $(m.status[:d]) node graph")) : nothing

    if method == :reduced
        res = nonreciprocated_outdegree(m, i, method=:reduced) + reciprocated_degree(m, i, method=:reduced)
    elseif method == :full
        # using all elements of the adjacency matrix
        res = zero(precision(m))
        for j in eachindex(m.d_out)
            res += A(m, i, j)
        end
    elseif method == :adjacency
        #  using the precomputed adjacency matrix
        m.status[:G_computed] ? nothing : throw(ArgumentError("The adjacency matrix has not been computed yet"))
        res = sum(@view m.Ĝ[i,:])
    else
        throw(ArgumentError("Unknown method $method"))
    end

    return res
end

"""
    outdegree(m::RBCM[, v]; method=:reduced)

Return a vector corresponding to the expected out-degree (``⟨k^{→}⟩ + ⟨k^{↔}⟩``) of the RBCM model `m` for each node.
If v is specified, only return the values for the nodes in v.

# Examples
```jldoctest
julia> model = RBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()));

julia> solve_model!(model);

julia> set_Ĝ!(model);

julia> typeof(outdegree(model, method=:adjacency))
Vector{Float64} (alias for Array{Float64, 1})

```
"""
outdegree(m::RBCM, v::Vector{Int}=collect(1:m.status[:d]); method::Symbol=:reduced) = [outdegree(m, i, method=method) for i in v]


"""
    indegree(m::RBCM, i::Int; method=:reduced)

Return the expected in-degree (in the `Graphs.jl` sense, i.e. ``⟨k^{in}_i⟩ = ⟨k^{←}_i⟩ + ⟨k^{↔}_i⟩``) for node `i`
of the RBCM model `m`.

# Arguments
- `m::RBCM`: the RBCM model
- `i::Int`: the node for which to compute the degree.
- `method::Symbol`: `:reduced` (default), `:full` or `:adjacency`, see [`outdegree`](@ref).

# Examples
```jldoctest
julia> model = RBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()));

julia> solve_model!(model);

julia> set_Ĝ!(model);

julia> typeof([indegree(model, 1), indegree(model, 1, method=:full), indegree(model, 1, method=:adjacency)])
Vector{Float64} (alias for Array{Float64, 1})

```
"""
function indegree(m::RBCM, i::Int; method::Symbol=:reduced)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    i > m.status[:d] ? throw(ArgumentError("Attempted to access node $i in a $(m.status[:d]) node graph")) : nothing

    if method == :reduced
        res = nonreciprocated_indegree(m, i, method=:reduced) + reciprocated_degree(m, i, method=:reduced)
    elseif method == :full
        # using all elements of the adjacency matrix
        res = zero(precision(m))
        for j in eachindex(m.d_out)
            res += A(m, j, i)
        end
    elseif method == :adjacency
        #  using the precomputed adjacency matrix
        m.status[:G_computed] ? nothing : throw(ArgumentError("The adjacency matrix has not been computed yet"))
        res = sum(@view m.Ĝ[:,i])
    else
        throw(ArgumentError("Unknown method $method"))
    end

    return res
end

"""
    indegree(m::RBCM[, v]; method=:reduced)

Return a vector corresponding to the expected in-degree (``⟨k^{←}⟩ + ⟨k^{↔}⟩``) of the RBCM model `m` for each node.
If v is specified, only return the values for the nodes in v.

# Examples
```jldoctest
julia> model = RBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()));

julia> solve_model!(model);

julia> set_Ĝ!(model);

julia> typeof(indegree(model, method=:adjacency))
Vector{Float64} (alias for Array{Float64, 1})

```
"""
indegree(m::RBCM, v::Vector{Int}=collect(1:m.status[:d]); method::Symbol=:reduced) = [indegree(m, i, method=method) for i in v]


"""
    degree(m::RBCM, i::Int; method=:reduced)

In alignment with `Graphs.jl`, returns the sum of the expected out- and in-degree for node `i` of the RBCM model `m`.

# Arguments
- `m::RBCM`: the RBCM model
- `i::Int`: the node for which to compute the degree.
- `method::Symbol`: `:reduced` (default), `:full` or `:adjacency`, see [`outdegree`](@ref).

# Examples
```jldoctest
julia> model = RBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()));

julia> solve_model!(model);

julia> set_Ĝ!(model);

julia> typeof([degree(model, 1), degree(model, 1, method=:full), degree(model, 1, method=:adjacency)])
Vector{Float64} (alias for Array{Float64, 1})

```
"""
degree(m::RBCM, i::Int; method::Symbol=:reduced) = outdegree(m, i, method=method) + indegree(m, i, method=method)

"""
    degree(m::RBCM[, v]; method=:reduced)

In alignment with `Graphs.jl`, returns a vector corresponding to the sum of the expected out- and in-degree
of the RBCM model `m` for each node. If v is specified, only return the values for the nodes in v.

# Examples
```jldoctest
julia> model = RBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()));

julia> solve_model!(model);

julia> set_Ĝ!(model);

julia> typeof(degree(model, method=:adjacency))
Vector{Float64} (alias for Array{Float64, 1})

```
"""
degree(m::RBCM, v::Vector{Int}=collect(1:m.status[:d]); method::Symbol=:reduced) = outdegree(m, v, method=method) + indegree(m, v, method=method)


"""
    reciprocity(m::RBCM)

Compute the expected topological reciprocity of the RBCM model `m` as the ratio of expectations
``\\sum_{i≠j} p^{↔}_{ij} / \\sum_{i≠j} ⟨a_{ij}⟩`` (the standard approximation of ``⟨r⟩`` used in the
Squartini & Garlaschelli framework). As the RBCM constrains the reciprocal degree sequences, this
reproduces the observed reciprocity of the network it was fitted to.

# Examples
```jldoctest
julia> model = RBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()));

julia> solve_model!(model);

julia> reciprocity(model) ≈ reciprocity(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()))
true

```
"""
function reciprocity(m::RBCM)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    n = m.status[:d]
    num = zero(precision(m))
    den = zero(precision(m))
    for i = 1:n
        for j = 1:n
            if i ≠ j
                pr = p⭤(m, i, j)
                num += pr
                den += pr + p⭢(m, i, j)
            end
        end
    end
    return num / den
end


"""
    reciprocity(m::DBCM)

Compute the expected topological reciprocity of the DBCM model `m` as the ratio of expectations
``\\sum_{i≠j} p_{ij}p_{ji} / \\sum_{i≠j} p_{ij}`` (under the DBCM the directions are independent, so
``⟨a_{ij}a_{ji}⟩ = p_{ij}p_{ji}``). Comparing this baseline with the observed [`reciprocity`](@ref) of the
network indicates whether reciprocity is a significant feature that warrants the [`RBCM`](@ref).

# Examples
```jldoctest
julia> model = DBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()));

julia> solve_model!(model);

julia> reciprocity(model) < reciprocity(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()))
true

```
"""
function reciprocity(m::DBCM)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    n = m.status[:d]
    num = zero(precision(m))
    den = zero(precision(m))
    for i = 1:n
        for j = 1:n
            if i ≠ j
                num += A(m, i, j) * A(m, j, i)
                den += A(m, i, j)
            end
        end
    end
    return num / den
end


"""
    AIC(m::RBCM)

Compute the Akaike Information Criterion (AIC) for the RBCM model `m`. The parameters of the models most be computed beforehand.
If the number of empirical observations becomes too small with respect to the number of parameters, you will get a warning. In
that case, the corrected AIC (AICc) should be used instead.

The number of parameters is `3N` and the number of observations `N(N-1)` (the ordered node pairs), identical to
the DBCM convention, so AIC values of a DBCM and an RBCM fitted to the same network are directly comparable.

# Examples
```julia-repl
julia> model = RBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()));

julia> solve_model!(model);

julia> AIC(model);
┌ Warning: The number of observations is small with respect to the number of parameters (n/k < 40). Consider using the corrected AIC (AICc) instead.
[...]

```

See also [`AICc`](@ref MaxEntropyGraphs.AICc), [`L_RBCM_reduced`](@ref MaxEntropyGraphs.L_RBCM_reduced).
"""
function AIC(m::RBCM)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    # compute AIC components
    k = m.status[:d] * 3 # number of parameters (3 per node)
    n = (m.status[:d] - 1) * m.status[:d]  # number of observations (N-1)*N
    L = L_RBCM_reduced(m) # log-likelihood

    if n/k < 40
        @warn """The number of observations is small with respect to the number of parameters (n/k < 40). Consider using the corrected AIC (AICc) instead."""
    end

    return 2*k - 2*L
end


"""
    AICc(m::RBCM)

Compute the corrected Akaike Information Criterion (AICc) for the RBCM model `m`. The parameters of the models most be computed beforehand.

# Examples
```jldoctest
julia> model = RBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()));

julia> solve_model!(model);

julia> AICc(model);

```

See also [`AIC`](@ref MaxEntropyGraphs.AIC), [`L_RBCM_reduced`](@ref MaxEntropyGraphs.L_RBCM_reduced).
"""
function AICc(m::RBCM)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    # compute AIC components
    k = m.status[:d] * 3 # number of parameters (3 per node)
    n = (m.status[:d] - 1) * m.status[:d]   # number of observations (N-1)*N
    L = L_RBCM_reduced(m) # log-likelihood

    return 2*k - 2*L + (2*k*(k+1)) / (n - k - 1)
end


"""
    BIC(m::RBCM)

Compute the Bayesian Information Criterion (BIC) for the RBCM model `m`. The parameters of the models most be computed beforehand.
BIC is believed to be more restrictive than AIC, as the former favors models with a lower number of parameters than those favored by the latter.

# Examples
```jldoctest
julia> model = RBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()));

julia> solve_model!(model);

julia> BIC(model);

```

See also [`AIC`](@ref MaxEntropyGraphs.AIC), [`L_RBCM_reduced`](@ref MaxEntropyGraphs.L_RBCM_reduced).
"""
function BIC(m::RBCM)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    # compute BIC components
    k = m.status[:d] * 3 # number of parameters
    n = (m.status[:d] - 1) * m.status[:d]  # number of observations
    L = L_RBCM_reduced(m) # log-likelihood

    return k * log(n) - 2*L
end


"""
    σₓ(m::RBCM, X::function)

Compute the standard deviation of metric `X` for the RBCM model `m` using the delta method of
Squartini & Garlaschelli (2011) (Eqs. B.16/C.17), including the within-dyad covariance term:

``(σ^{*}[X])^2 = \\sum_{i,j} \\left[ σ^2[a_{ij}] \\left(\\frac{∂X}{∂a_{ij}}\\right)^2 + Cov(a_{ij},a_{ji}) \\frac{∂X}{∂a_{ij}}\\frac{∂X}{∂a_{ji}} \\right]_{A = ⟨A⟩}``

Under the RBCM the covariance ``Cov(a_{ij},a_{ji}) = p^{↔}_{ij} - ⟨a_{ij}⟩⟨a_{ji}⟩`` is non-zero, unlike for
the DBCM. This requires that both the expected values (m.Ĝ) and standard deviations (m.σ) are computed for `m`.
"""
function σₓ(m::RBCM, X::Function; gradient_method::Symbol=:ReverseDiff)
    # checks
    m.status[:G_computed] ? nothing : throw(ArgumentError("The expected values (m.Ĝ) must be computed for `m` before computing the standard deviation of metric `X`, see `set_Ĝ!`"))
    m.status[:σ_computed] ? nothing : throw(ArgumentError("The standard deviations (m.σ) must be computed for `m` before computing the standard deviation of metric `X`, see `set_σ!`"))

    # gradient
    if gradient_method == :ForwardDiff
        ∇X = ForwardDiff.gradient(X, m.Ĝ)
    elseif gradient_method == :ReverseDiff
        ∇X = ReverseDiff.gradient(X, m.Ĝ)
    elseif gradient_method == :Zygote
        ∇X = Zygote.gradient(X, m.Ĝ)[1]
    else
        throw(ArgumentError("Invalid gradient method, only :ForwardDiff, :ReverseDiff and :Zygote are accepted"))
    end

    # within-dyad covariance matrix (recomputed on demand)
    C = _cov_dyads(m)

    # return value: variance term + covariance cross-term (both sums run over ordered pairs)
    return sqrt( sum((m.σ .* ∇X) .^ 2) + sum(C .* ∇X .* transpose(∇X)) )
end
