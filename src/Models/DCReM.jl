
"""
    DCReM

Maximum entropy model for the directed Conditional Reconstruction Method (DCReM).

The DCReM is the **directed** counterpart of the [`CReM`](@ref): a **two-step** null model for weighted,
directed networks with **continuous, positive** weights (Parisi, Squartini & Garlaschelli (2020); the model
is called CReM_A in the literature and `DBCM+CReMa` in the NuMeTriS package). The binary structure
(topology) is supplied by a prior binary model — here an internally solved [`DBCM`](@ref) on the out- and
in-degree sequences, giving the marginal edge probability `fᵢⱼ = xᵢyⱼ/(1 + xᵢyⱼ)` with `xᵢ = e^{-αᵢ}`,
`yᵢ = e^{-βᵢ}`. Conditional on an edge `i→j` existing, its weight follows an exponential distribution with
rate `θᵒᵢ + θⁱⱼ` (mean `1/(θᵒᵢ + θⁱⱼ)`); the parameters `θ = [θᵒ; θⁱ]` (two per node) constrain the
out- and in-strength sequences.

The object holds the maximum likelihood parameters of the weighted layer (`θ`), the reduced binary
(conditional) DBCM parameters (`αᵣ`) and their exponentiated forms (`xᵣ`, `yᵣ`).
"""
mutable struct DCReM{T<:Union{Graphs.AbstractGraph, Nothing}, N<:Real} <: AbstractMaxEntropyModel
    "Graph type, can be any subtype of AbstractGraph, but will be converted to SimpleWeightedDiGraph for the computation" # can also be empty
    const G::T
    "Maximum likelihood parameters of the weighted (DCReM) layer, θ = [θᵒ; θⁱ] (two per node)"
    const θ::Vector{N}
    "Reduced maximum likelihood parameters of the binary (conditional DBCM) layer, αᵣ = [α; β]"
    const αᵣ::Vector{N}
    "Exponentiated reduced binary parameters ( xᵢ = exp(-αᵢ) ), linked with the out-degree"
    const xᵣ::Vector{N}
    "Exponentiated reduced binary parameters ( yᵢ = exp(-βᵢ) ), linked with the in-degree"
    const yᵣ::Vector{N}
    "Outdegree sequence of the graph"
    const d_out::Vector{Int}
    "Indegree sequence of the graph"
    const d_in::Vector{Int}
    "Reduced outdegree sequence of the graph"
    const dᵣ_out::Vector{Int}
    "Reduced indegree sequence of the graph"
    const dᵣ_in::Vector{Int}
    "Frequency of each (outdegree, indegree) pair in the graph"
    const f::Vector{Int}
    "Outstrength sequence of the graph (continuous, positive)"
    const s_out::Vector{N}
    "Instrength sequence of the graph (continuous, positive)"
    const s_in::Vector{N}
    "Indices to reconstruct the degree sequences from the reduced degree sequences"
    const d_ind::Vector{Int}
    "Indices to reconstruct the reduced degree sequences from the degree sequences"
    const dᵣ_ind::Vector{Int}
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

Base.show(io::IO, m::DCReM{T,N}) where {T,N} = print(io, """DCReM{$(T), $(N)} ($(m.status[:N]) vertices, $(m.status[:d_unique]) unique degree pairs, $(@sprintf("%.2f", m.status[:cᵣ])) compression ratio)""")

"""Return the number of vertices in the DCReM network"""
Base.length(m::DCReM) = length(m.d_out)


"""
    DCReM(G::T; d_out::Vector, d_in::Vector, s_out::Vector, s_in::Vector, precision::Type{<:AbstractFloat}=Float64, kwargs...) where {T}

Constructor function for the `DCReM` type.

By default and depending on the graph type `T`, the definition of out/in-degree from ``Graphs.jl`` and
out/in-strength from ``SimpleWeightedGraphs`` is applied. If you want to use different definitions, you can
pass the vectors as keyword arguments. If you want to generate a model directly from degree and strength
sequences without an underlying graph you can simply pass them as keyword arguments.

The DCReM allows **continuous, positive** weights (the strength sequences need not be integer-valued).

# Examples
```jldoctest DCReM_creation
# generating a model from a weighted directed graph
julia> model = DCReM(MaxEntropyGraphs.rhesus_macaques())
DCReM{SimpleWeightedGraphs.SimpleWeightedDiGraph{Int64, Float64}, Float64} (16 vertices, 15 unique degree pairs, 0.94 compression ratio)

```
```jldoctest DCReM_creation
# generating a model directly from degree and strength sequences
julia> model = DCReM(d_out=[1, 1, 2, 1], d_in=[2, 1, 1, 1], s_out=[3.0, 5.0, 4.0, 2.0], s_in=[4.0, 2.0, 5.0, 3.0])
DCReM{Nothing, Float64} (4 vertices, 3 unique degree pairs, 0.75 compression ratio)

```
"""
function DCReM(G::T;    d_out::Vector=Graphs.outdegree(G),
                        d_in::Vector=Graphs.indegree(G),
                        s_out::Vector=strength(G, dir=:out),
                        s_in::Vector=strength(G, dir=:in),
                        precision::Type{<:AbstractFloat}=Float64, kwargs...) where {T}
    T <: Union{Graphs.AbstractGraph, Nothing} ? nothing : throw(TypeError(:DCReM, "G must be a subtype of AbstractGraph or Nothing", Union{Graphs.AbstractGraph, Nothing}, T))
    # coherence checks
    if T <: Graphs.AbstractGraph # Graph specific checks
        if !Graphs.is_directed(G)
            @warn "The graph is undirected, while the DCReM model is directed, the out- and in-quantities will be the same"
        end

        Graphs.nv(G) == 0 ? throw(ArgumentError("The graph is empty")) : nothing
        Graphs.nv(G) == 1 ? throw(ArgumentError("The graph has only one vertex")) : nothing

        (Graphs.nv(G) != length(d_out) || Graphs.nv(G) != length(d_in)) ? throw(DimensionMismatch("The number of vertices in the graph ($(Graphs.nv(G))) and the length of the degree sequences do not match")) : nothing
        (Graphs.nv(G) != length(s_out) || Graphs.nv(G) != length(s_in)) ? throw(DimensionMismatch("The number of vertices in the graph ($(Graphs.nv(G))) and the length of the strength sequences do not match")) : nothing
    end

    # coherence checks specific to the degree/strength sequences
    (length(d_out) == length(d_in) && length(s_out) == length(s_in) && length(d_out) == length(s_out)) ? nothing : throw(DimensionMismatch("The degree and strength sequences must all have the same length"))
    length(d_out) == 0 ? throw(ArgumentError("The degree sequences are empty")) : nothing
    length(d_out) == 1 ? throw(ArgumentError("The degree sequences only contain a single node")) : nothing
    if any(iszero, d_out) || any(iszero, d_in)
        @warn "The graph has vertices with zero out- or in-degree, this may lead to convergence issues."
    end
    if any(iszero, s_out) || any(iszero, s_in)
        @warn "The graph has vertices with zero out- or in-strength, this may lead to convergence issues."
    end
    (any(!isinteger, d_out) || any(!isinteger, d_in)) ? throw(DomainError("Some of the degree values are not integers, this is not allowed")) : nothing
    maximum(d_out) >= length(d_out) ? throw(DomainError("The maximum outdegree in the graph is greater or equal to the number of vertices, this is not allowed")) : nothing
    maximum(d_in)  >= length(d_in)  ? throw(DomainError("The maximum indegree in the graph is greater or equal to the number of vertices, this is not allowed")) : nothing
    (all(d_out .>= 0) && all(d_in .>= 0)) ? nothing : throw(DomainError("The degree sequences contain negative degrees"))
    (all(s_out .>= 0) && all(s_in .>= 0)) ? nothing : throw(DomainError("The strength sequences contain negative strengths"))

    # field generation (the binary layer reduces over the (outdegree, indegree) pairs, exactly like the DBCM)
    dᵣ, d_ind, dᵣ_ind, f = np_unique_clone(collect(zip(d_out, d_in)), sorted=true)
    dᵣ_out = [d[1] for d in dᵣ]
    dᵣ_in  = [d[2] for d in dᵣ]
    θ  = Vector{precision}(undef, 2*length(d_out))  # weighted (DCReM) parameters, [θᵒ; θⁱ]
    αᵣ = Vector{precision}(undef, 2*length(dᵣ))     # reduced binary (DBCM) parameters, [α; β]
    xᵣ = Vector{precision}(undef, length(dᵣ))       # exponentiated reduced binary parameters (out)
    yᵣ = Vector{precision}(undef, length(dᵣ))       # exponentiated reduced binary parameters (in)
    status = Dict(  :conditional_params_computed => false,   # is the binary (DBCM) layer solved?
                    :params_computed             => false,   # is the weighted (DCReM) layer solved?
                    :G_computed                  => false,   # is the expected adjacency matrix computed and stored?
                    :σ_computed                  => false,   # is the standard deviation computed and stored?
                    :W_computed                  => false,   # is the expected weight matrix computed and stored?
                    :σʷ_computed                 => false,   # is the weight standard deviation computed and stored?
                    :cᵣ       => length(dᵣ)/length(d_out),   # compression ratio of the reduced binary model
                    :d_unique => length(dᵣ),                 # number of unique (outdegree, indegree) pairs in the reduced model
                    :N        => length(d_out)               # number of vertices in the original graph
                )

    return DCReM{T,precision}(G, θ, αᵣ, xᵣ, yᵣ, Int.(d_out), Int.(d_in), dᵣ_out, dᵣ_in, f, Vector{precision}(s_out), Vector{precision}(s_in), d_ind, dᵣ_ind, nothing, nothing, nothing, nothing, status)
end

DCReM(;d_out::Vector, d_in::Vector, s_out::Vector, s_in::Vector, precision::Type{<:AbstractFloat}=Float64, kwargs...) = DCReM(nothing, d_out=d_out, d_in=d_in, s_out=s_out, s_in=s_in, precision=precision, kwargs...)


"""
    L_DCReM(θ::AbstractVector, s_out::AbstractVector, s_in::AbstractVector, x::AbstractVector, y::AbstractVector)

Compute the log-likelihood of the DCReM model given the weighted parameters `θ = [θᵒ; θⁱ]`, the strength
sequences `s_out`/`s_in` and the per-node binary fitnesses `x`, `y`, computing the marginal edge probability
`fᵢⱼ = xᵢyⱼ/(1 + xᵢyⱼ)` on the fly.

The (conditional) log-likelihood is
``\\mathcal{L} = -\\sum_i \\left[ θ^{o}_i s^{out}_i + θ^{i}_i s^{in}_i \\right] + \\sum_{i≠j} f_{ij} \\log(θ^{o}_i + θ^{i}_j)``.
The `log` is domain-guarded (returns `NaN` for `θᵒᵢ + θⁱⱼ ≤ 0`) so that the line search rejects steps that
leave the feasible region; this keeps it automatic-differentiation friendly.

This method is used when the marginal probability matrix is not precomputed (i.e. `model.Ĝ` is `nothing`).
"""
function L_DCReM(θ::AbstractVector, s_out::AbstractVector, s_in::AbstractVector, x::AbstractVector, y::AbstractVector)
    n = length(s_out)
    θᵒ = @view θ[1:n]
    θⁱ = @view θ[n+1:end]
    res = zero(eltype(θ))
    @inbounds for i in eachindex(s_out)
        res -= θᵒ[i] * s_out[i] + θⁱ[i] * s_in[i]
        xᵢ = x[i]; θᵒᵢ = θᵒ[i]
        acc = zero(eltype(θ))
        for j in eachindex(s_in)
            if i ≠ j
                fij = xᵢ * y[j] / (1 + xᵢ * y[j])
                acc += fij * log_nan(θᵒᵢ + θⁱ[j])
            end
        end
        res += acc
    end

    return res
end


"""
    L_DCReM(θ::AbstractVector, s_out::AbstractVector, s_in::AbstractVector, f::AbstractMatrix)

Compute the log-likelihood of the DCReM model, using the precomputed marginal probability matrix `f`
(i.e. `model.Ĝ`). See also [`L_DCReM(::AbstractVector, ::AbstractVector, ::AbstractVector, ::AbstractVector, ::AbstractVector)`](@ref).
"""
function L_DCReM(θ::AbstractVector, s_out::AbstractVector, s_in::AbstractVector, f::AbstractMatrix)
    n = length(s_out)
    θᵒ = @view θ[1:n]
    θⁱ = @view θ[n+1:end]
    res = zero(eltype(θ))
    @inbounds for i in eachindex(s_out)
        res -= θᵒ[i] * s_out[i] + θⁱ[i] * s_in[i]
        θᵒᵢ = θᵒ[i]
        acc = zero(eltype(θ))
        for j in eachindex(s_in)
            if i ≠ j
                acc += f[i,j] * log_nan(θᵒᵢ + θⁱ[j])
            end
        end
        res += acc
    end

    return res
end


"""
    L_DCReM(m::DCReM)

Return the log-likelihood of the DCReM model `m` based on the computed maximum likelihood parameters.
Depending on the status of the model, the precomputed marginal probability matrix (`m.Ĝ`) is used, or
the marginal probability is computed on the fly.

See also [`L_DCReM(::AbstractVector, ::AbstractVector, ::AbstractVector, ::AbstractVector, ::AbstractVector)`](@ref).
"""
function L_DCReM(m::DCReM)
    m.status[:conditional_params_computed] ? nothing : throw(ArgumentError("The conditional parameters of the model have not been computed"))
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters of the model have not been computed"))
    if m.status[:G_computed]
        return L_DCReM(m.θ, m.s_out, m.s_in, m.Ĝ)
    else
        return L_DCReM(m.θ, m.s_out, m.s_in, m.xᵣ[m.dᵣ_ind], m.yᵣ[m.dᵣ_ind])
    end
end


"""
    ∇L_DCReM!(∇L::AbstractVector, θ::AbstractVector, s_out::AbstractVector, s_in::AbstractVector, x::AbstractVector, y::AbstractVector)

Compute the gradient of the log-likelihood of the DCReM model in a non-allocating manner, computing the
marginal edge probability on the fly (`x`, `y` = per-node binary fitnesses).

The inner loop is branch-free (the spurious `j == i` self-term is folded out afterwards) so that it
vectorises (`@simd`). `∂L/∂θᵒᵢ = -s^{out}_i + Σⱼ≠ᵢ fᵢⱼ/(θᵒᵢ + θⁱⱼ)` and
`∂L/∂θⁱᵢ = -s^{in}_i + Σⱼ≠ᵢ fⱼᵢ/(θⁱᵢ + θᵒⱼ)`.

See also [`∇L_DCReM_minus!`](@ref).
"""
function ∇L_DCReM!(∇L::AbstractVector, θ::AbstractVector, s_out::AbstractVector, s_in::AbstractVector, x::AbstractVector, y::AbstractVector)
    n = length(s_out)
    θᵒ = @view θ[1:n]
    θⁱ = @view θ[n+1:end]
    @inbounds for i in eachindex(s_out)
        θᵒᵢ = θᵒ[i]; θⁱᵢ = θⁱ[i]; xᵢ = x[i]; yᵢ = y[i]
        acc_out = zero(eltype(∇L))
        acc_in  = zero(eltype(∇L))
        @simd for j in eachindex(s_in)
            fij = xᵢ * y[j] / (1 + xᵢ * y[j])
            fji = x[j] * yᵢ / (1 + x[j] * yᵢ)
            acc_out += fij / (θᵒᵢ + θⁱ[j])
            acc_in  += fji / (θⁱᵢ + θᵒ[j])
        end
        # subtract the spurious j == i self-terms that the branch-free sums included
        fii = xᵢ * yᵢ / (1 + xᵢ * yᵢ)
        acc_out -= fii / (θᵒᵢ + θⁱᵢ)
        acc_in  -= fii / (θⁱᵢ + θᵒᵢ)
        ∇L[i]   = -s_out[i] + acc_out
        ∇L[n+i] = -s_in[i]  + acc_in
    end

    return ∇L
end


"""
    ∇L_DCReM!(∇L::AbstractVector, θ::AbstractVector, s_out::AbstractVector, s_in::AbstractVector, f::AbstractMatrix)

Compute the gradient of the log-likelihood of the DCReM model, using the precomputed marginal probability
matrix `f` (i.e. `model.Ĝ`). The `f[i,i] = 0` diagonal makes the self-terms vanish automatically.
"""
function ∇L_DCReM!(∇L::AbstractVector, θ::AbstractVector, s_out::AbstractVector, s_in::AbstractVector, f::AbstractMatrix)
    n = length(s_out)
    θᵒ = @view θ[1:n]
    θⁱ = @view θ[n+1:end]
    @inbounds for i in eachindex(s_out)
        θᵒᵢ = θᵒ[i]; θⁱᵢ = θⁱ[i]
        acc_out = zero(eltype(∇L))
        acc_in  = zero(eltype(∇L))
        @simd for j in eachindex(s_in)
            acc_out += f[i,j] / (θᵒᵢ + θⁱ[j])
            acc_in  += f[j,i] / (θⁱᵢ + θᵒ[j])
        end
        ∇L[i]   = -s_out[i] + acc_out
        ∇L[n+i] = -s_in[i]  + acc_in
    end

    return ∇L
end


"""
    ∇L_DCReM_minus!(∇L::AbstractVector, θ::AbstractVector, s_out::AbstractVector, s_in::AbstractVector, x::AbstractVector, y::AbstractVector)

Compute minus the gradient of the log-likelihood of the DCReM model (used for the minimisation carried
out by `Optimization.jl`), computing the marginal edge probability on the fly. Non-allocating.

See also [`∇L_DCReM!`](@ref).
"""
function ∇L_DCReM_minus!(∇L::AbstractVector, θ::AbstractVector, s_out::AbstractVector, s_in::AbstractVector, x::AbstractVector, y::AbstractVector)
    n = length(s_out)
    θᵒ = @view θ[1:n]
    θⁱ = @view θ[n+1:end]
    @inbounds for i in eachindex(s_out)
        θᵒᵢ = θᵒ[i]; θⁱᵢ = θⁱ[i]; xᵢ = x[i]; yᵢ = y[i]
        acc_out = zero(eltype(∇L))
        acc_in  = zero(eltype(∇L))
        @simd for j in eachindex(s_in)
            fij = xᵢ * y[j] / (1 + xᵢ * y[j])
            fji = x[j] * yᵢ / (1 + x[j] * yᵢ)
            acc_out += fij / (θᵒᵢ + θⁱ[j])
            acc_in  += fji / (θⁱᵢ + θᵒ[j])
        end
        fii = xᵢ * yᵢ / (1 + xᵢ * yᵢ)
        acc_out -= fii / (θᵒᵢ + θⁱᵢ)
        acc_in  -= fii / (θⁱᵢ + θᵒᵢ)
        ∇L[i]   = s_out[i] - acc_out
        ∇L[n+i] = s_in[i]  - acc_in
    end

    return ∇L
end


"""
    ∇L_DCReM_minus!(∇L::AbstractVector, θ::AbstractVector, s_out::AbstractVector, s_in::AbstractVector, f::AbstractMatrix)

Compute minus the gradient of the log-likelihood of the DCReM model, using the precomputed marginal
probability matrix `f` (i.e. `model.Ĝ`). Non-allocating.
"""
function ∇L_DCReM_minus!(∇L::AbstractVector, θ::AbstractVector, s_out::AbstractVector, s_in::AbstractVector, f::AbstractMatrix)
    n = length(s_out)
    θᵒ = @view θ[1:n]
    θⁱ = @view θ[n+1:end]
    @inbounds for i in eachindex(s_out)
        θᵒᵢ = θᵒ[i]; θⁱᵢ = θⁱ[i]
        acc_out = zero(eltype(∇L))
        acc_in  = zero(eltype(∇L))
        @simd for j in eachindex(s_in)
            acc_out += f[i,j] / (θᵒᵢ + θⁱ[j])
            acc_in  += f[j,i] / (θⁱᵢ + θᵒ[j])
        end
        ∇L[i]   = s_out[i] - acc_out
        ∇L[n+i] = s_in[i]  - acc_in
    end

    return ∇L
end


"""
    DCReM_iter!(θ::AbstractVector, s_out::AbstractVector, s_in::AbstractVector, x::AbstractVector, y::AbstractVector, G::AbstractVector)

Compute the next fixed-point iteration for the DCReM model, computing the marginal edge probability on
the fly (`x`, `y` = per-node binary fitnesses). The pre-allocated buffer `G` is updated in place.

The consistency equations are ``θ^{o}_i = \\left( \\sum_{j\\neq i} f_{ij}/(1 + θ^{i}_j/θ^{o}_i) \\right) / s^{out}_i``
and ``θ^{i}_i = \\left( \\sum_{j\\neq i} f_{ji}/(1 + θ^{o}_j/θ^{i}_i) \\right) / s^{in}_i``.
"""
function DCReM_iter!(θ::AbstractVector, s_out::AbstractVector, s_in::AbstractVector, x::AbstractVector, y::AbstractVector, G::AbstractVector)
    n = length(s_out)
    θᵒ = @view θ[1:n]
    θⁱ = @view θ[n+1:end]
    @inbounds for i in eachindex(s_out)
        θᵒᵢ = θᵒ[i]; θⁱᵢ = θⁱ[i]; xᵢ = x[i]; yᵢ = y[i]
        acc_out = zero(eltype(θ))
        acc_in  = zero(eltype(θ))
        @simd for j in eachindex(s_in)
            fij = xᵢ * y[j] / (1 + xᵢ * y[j])
            fji = x[j] * yᵢ / (1 + x[j] * yᵢ)
            acc_out += fij / (1 + θⁱ[j] / θᵒᵢ)
            acc_in  += fji / (1 + θᵒ[j] / θⁱᵢ)
        end
        # subtract the spurious j == i self-terms
        fii = xᵢ * yᵢ / (1 + xᵢ * yᵢ)
        acc_out -= fii / (1 + θⁱᵢ / θᵒᵢ)
        acc_in  -= fii / (1 + θᵒᵢ / θⁱᵢ)
        G[i]   = acc_out / s_out[i]
        G[n+i] = acc_in  / s_in[i]
    end

    return G
end


"""
    DCReM_iter!(θ::AbstractVector, s_out::AbstractVector, s_in::AbstractVector, f::AbstractMatrix, G::AbstractVector)

Compute the next fixed-point iteration for the DCReM model, using the precomputed marginal probability
matrix `f` (i.e. `model.Ĝ`). The pre-allocated buffer `G` is updated in place.
"""
function DCReM_iter!(θ::AbstractVector, s_out::AbstractVector, s_in::AbstractVector, f::AbstractMatrix, G::AbstractVector)
    n = length(s_out)
    θᵒ = @view θ[1:n]
    θⁱ = @view θ[n+1:end]
    @inbounds for i in eachindex(s_out)
        θᵒᵢ = θᵒ[i]; θⁱᵢ = θⁱ[i]
        acc_out = zero(eltype(θ))
        acc_in  = zero(eltype(θ))
        @simd for j in eachindex(s_in)
            acc_out += f[i,j] / (1 + θⁱ[j] / θᵒᵢ)
            acc_in  += f[j,i] / (1 + θᵒ[j] / θⁱᵢ)
        end
        G[i]   = acc_out / s_out[i]
        G[n+i] = acc_in  / s_in[i]
    end

    return G
end


"""
    initial_guess(m::DCReM; method::Symbol=:strengths)

Compute an initial guess `θ₀ = [θᵒ₀; θⁱ₀]` for the maximum likelihood parameters of the weighted (DCReM)
layer of the model `m`.

The DCReM parameters `θ` are the **direct** rate parameters (they appear as `log(θᵒᵢ + θⁱⱼ)` in the
log-likelihood), so every method returns strictly positive, feasible values — a negative `θᵒᵢ + θⁱⱼ` would
put the objective outside its domain. The forms mirror NEMtropy's `crema` initial guesses:
- `:strengths` (default): `θᵒᵢ = 𝟙[s^{out}_i > 0] / (Σⱼ s^{out}_j + Σⱼ s^{in}_j)` and the analog for `θⁱᵢ`.
- `:strengths_minor`: `θᵒᵢ = 𝟙[s^{out}_i > 0] / (s^{out}_i + 1)` and the analog for `θⁱᵢ`.
- `:random`: random values drawn from ``U(0,1)``.
"""
function initial_guess(m::DCReM; method::Symbol=:strengths)
    N = precision(m)
    if isequal(method, :strengths)
        stot = sum(m.s_out) + sum(m.s_in)
        res = Vector{N}(vcat((m.s_out .> 0) ./ stot, (m.s_in .> 0) ./ stot))
    elseif isequal(method, :strengths_minor)
        res = Vector{N}(vcat((m.s_out .> 0) ./ (m.s_out .+ 1), (m.s_in .> 0) ./ (m.s_in .+ 1)))
    elseif isequal(method, :random)
        res = Vector{N}(rand(N, 2*length(m.s_out)))
    else
        throw(ArgumentError("The initial guess method $(method) is not supported"))
    end

    return res
end


"""
    precision(m::DCReM)

Determine the compute precision of the DCReM model `m`.
"""
precision(m::DCReM) = typeof(m).parameters[2]


"""
    set_xᵣ!(m::DCReM)

Set the value of `xᵣ` to `exp(-αᵣ[1:d_unique])` for the (binary layer of the) DCReM model `m`.
"""
function set_xᵣ!(m::DCReM)
    if m.status[:conditional_params_computed]
        αᵣ = @view m.αᵣ[1:m.status[:d_unique]]
        m.xᵣ .= exp.(-αᵣ)
    else
        throw(ArgumentError("The conditional parameters have not been computed yet"))
    end
end

"""
    set_yᵣ!(m::DCReM)

Set the value of `yᵣ` to `exp(-αᵣ[d_unique+1:end])` for the (binary layer of the) DCReM model `m`.
"""
function set_yᵣ!(m::DCReM)
    if m.status[:conditional_params_computed]
        βᵣ = @view m.αᵣ[m.status[:d_unique]+1:end]
        m.yᵣ .= exp.(-βᵣ)
    else
        throw(ArgumentError("The conditional parameters have not been computed yet"))
    end
end


"""
    A(m::DCReM, i::Int, j::Int)

Return the expected value of the (binary) adjacency matrix for the DCReM model `m` at the node pair `(i,j)`
(`fᵢⱼ = xᵢyⱼ/(1 + xᵢyⱼ)`).

❗ For performance reasons, the function does not check:
- if the node pair is valid.
- if the parameters of the model have been computed.
"""
function A(m::DCReM, i::Int, j::Int)
    return i == j ? zero(precision(m)) : @inbounds f_DBCM(m.xᵣ[m.dᵣ_ind[i]] * m.yᵣ[m.dᵣ_ind[j]])
end


"""
    Ĝ(m::DCReM)

Compute the expected (binary) **adjacency** matrix for the DCReM model `m`, i.e. `fᵢⱼ = xᵢyⱼ/(1 + xᵢyⱼ)`,
so that `sum(Ĝ(m), dims=2) ≈ outdegree` and `sum(Ĝ(m), dims=1) ≈ indegree`.

Note: The expected weights can be computed separately with [`Ŵ`](@ref MaxEntropyGraphs.Ŵ).
"""
function Ĝ(m::DCReM)
    m.status[:conditional_params_computed] ? nothing : throw(ArgumentError("The conditional parameters have not been computed yet"))

    n = m.status[:N]::Int
    G = zeros(precision(m), n, n)
    x = m.xᵣ[m.dᵣ_ind]
    y = m.yᵣ[m.dᵣ_ind]
    for i = 1:n
        @simd for j = 1:n
            if i ≠ j
                @inbounds xiyj = x[i]*y[j]
                @inbounds G[i,j] = xiyj / (1 + xiyj)
            end
        end
    end

    return G
end

"""
    set_Ĝ!(m::DCReM)

Set the expected (binary) adjacency matrix for the DCReM model `m`.
"""
function set_Ĝ!(m::DCReM)
    m.Ĝ = Ĝ(m)
    m.status[:G_computed] = true
    return m.Ĝ
end


"""
    Ŵ(m::DCReM)

Compute the expected (unconditional) **weighted adjacency** matrix for the DCReM model `m`, i.e.
`⟨wᵢⱼ⟩ = fᵢⱼ / (θᵒᵢ + θⁱⱼ)`, so that `sum(Ŵ(m), dims=2) ≈ outstrength` and `sum(Ŵ(m), dims=1) ≈ instrength`.
"""
function Ŵ(m::DCReM)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))

    n = m.status[:N]::Int
    W = zeros(precision(m), n, n)
    x = m.xᵣ[m.dᵣ_ind]
    y = m.yᵣ[m.dᵣ_ind]
    θᵒ = @view m.θ[1:n]
    θⁱ = @view m.θ[n+1:end]
    for i = 1:n
        @simd for j = 1:n
            if i ≠ j
                @inbounds xiyj = x[i]*y[j]
                @inbounds W[i,j] = (xiyj / (1 + xiyj)) / (θᵒ[i] + θⁱ[j])
            end
        end
    end

    return W
end

"""
    set_Ŵ!(m::DCReM)

Set the expected weighted adjacency matrix for the DCReM model `m`.
"""
function set_Ŵ!(m::DCReM)
    m.Ŵ = Ŵ(m)
    m.status[:W_computed] = true
    return m.Ŵ
end


"""
    σˣ(m::DCReM)

Compute the standard deviation for the elements of the (binary) adjacency matrix for the DCReM model `m`,
i.e. `sqrt(fᵢⱼ(1 - fᵢⱼ))` (the adjacency entries are Bernoulli distributed and, under the conditional DBCM
layer, independent).

**Note:** this is the standard deviation of the *binary* layer; the standard deviation of the weights is
available via [`σʷ`](@ref MaxEntropyGraphs.σʷ). Read as "sigma star".
"""
function σˣ(m::DCReM)
    m.status[:conditional_params_computed] ? nothing : throw(ArgumentError("The conditional parameters have not been computed yet"))

    n = m.status[:N]::Int
    σ = zeros(precision(m), n, n)
    x = m.xᵣ[m.dᵣ_ind]
    y = m.yᵣ[m.dᵣ_ind]
    for i = 1:n
        @simd for j = 1:n
            if i ≠ j
                @inbounds xiyj = x[i]*y[j]
                @inbounds pij  = xiyj / (1 + xiyj)
                @inbounds σ[i,j] = sqrt(pij * (1 - pij))
            end
        end
    end

    return σ
end

"""
    set_σ!(m::DCReM)

Set the standard deviation for the elements of the (binary) adjacency matrix for the DCReM model `m`.
"""
function set_σ!(m::DCReM)
    m.σ = σˣ(m)
    m.status[:σ_computed] = true
    return m.σ
end


"""
    σʷ(m::DCReM)

Compute the standard deviation for the elements of the **weighted** adjacency matrix for the DCReM model
`m`. The (unconditional) weight `wᵢⱼ` is a mixture of an exponential (with probability `fᵢⱼ`) and zero,
so ``⟨w_{ij}^2⟩ = 2f_{ij}/(θ^{o}_i + θ^{i}_j)^2`` and
``Var(w_{ij}) = f_{ij}(2 - f_{ij})/(θ^{o}_i + θ^{i}_j)^2``.

Under the conditional DBCM layer the weights of distinct ordered pairs are independent
(`Cov(wᵢⱼ, wⱼᵢ) = 0`), unlike for the [`CRWCM`](@ref).
"""
function σʷ(m::DCReM)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))

    n = m.status[:N]::Int
    σ = zeros(precision(m), n, n)
    x = m.xᵣ[m.dᵣ_ind]
    y = m.yᵣ[m.dᵣ_ind]
    θᵒ = @view m.θ[1:n]
    θⁱ = @view m.θ[n+1:end]
    for i = 1:n
        @simd for j = 1:n
            if i ≠ j
                @inbounds xiyj = x[i]*y[j]
                @inbounds fij  = xiyj / (1 + xiyj)
                @inbounds σ[i,j] = sqrt(fij * (2 - fij)) / (θᵒ[i] + θⁱ[j])
            end
        end
    end

    return σ
end

"""
    set_σʷ!(m::DCReM)

Set the standard deviation for the elements of the weighted adjacency matrix for the DCReM model `m`.
"""
function set_σʷ!(m::DCReM)
    m.σʷ = σʷ(m)
    m.status[:σʷ_computed] = true
    return m.σʷ
end


"""
    σₓ(m::DCReM, X::Function; layer::Symbol=:binary, gradient_method::Symbol=:ReverseDiff)

Compute the standard deviation of metric `X` for the DCReM model `m` via error propagation (the delta
method of Squartini & Garlaschelli (2011)).

# Arguments
- `layer::Symbol`:
    - `:binary` (default): propagate over the **binary adjacency matrix** — `X` is a function of the
      adjacency matrix, the gradient is evaluated at `m.Ĝ` and weighted by `m.σ` (requires `set_Ĝ!` and `set_σ!`).
    - `:weighted`: propagate over the **weighted adjacency matrix** — `X` is a function of the weight
      matrix, the gradient is evaluated at `m.Ŵ` and weighted by `m.σʷ` (requires `set_Ŵ!` and `set_σʷ!`).
      This is the layer to use for weighted metrics such as the triadic fluxes.
- `gradient_method::Symbol`: `:ForwardDiff`, `:ReverseDiff` (default) or `:Zygote`.

Under the conditional DBCM binary layer all matrix entries are independent (no covariance terms).
"""
function σₓ(m::DCReM, X::Function; layer::Symbol=:binary, gradient_method::Symbol=:ReverseDiff)
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

    if gradient_method == :ForwardDiff
        ∇X = ForwardDiff.gradient(X, M)
    elseif gradient_method == :ReverseDiff
        ∇X = ReverseDiff.gradient(X, M)
    elseif gradient_method == :Zygote
        ∇X = Zygote.gradient(X, M)[1]
    else
        throw(ArgumentError("Invalid gradient method, only :ForwardDiff, :ReverseDiff and :Zygote are accepted"))
    end

    return sqrt( sum((S .* ∇X) .^ 2) )
end


"""
    outdegree(m::DCReM, i::Int; method=:reduced)

Return the expected (binary) out-degree for node `i` of the DCReM model `m`.

# Arguments
- `method::Symbol`:
    - `:reduced` (default) uses the reduced binary model parameters.
    - `:full` uses all elements of the expected adjacency matrix.
    - `:adjacency` uses the precomputed adjacency matrix `m.Ĝ` of the model.
"""
function outdegree(m::DCReM, i::Int; method::Symbol=:reduced)
    m.status[:conditional_params_computed] ? nothing : throw(ArgumentError("The conditional parameters have not been computed yet"))
    i > m.status[:N] ? throw(ArgumentError("Attempted to access node $i in a $(m.status[:N]) node graph")) : nothing

    if method == :reduced
        res = zero(precision(m))
        i_red = m.dᵣ_ind[i] # find matching index in reduced model
        for j in eachindex(m.xᵣ)
            @inbounds pij = f_DBCM(m.xᵣ[i_red] * m.yᵣ[j])
            @inbounds res += pij * (m.f[j] - (i_red == j)) # subtract 1 within class because the diagonal is not counted
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
    outdegree(m::DCReM[, v]; method=:reduced)

Return a vector corresponding to the expected (binary) out-degree of each node of the DCReM model `m`.
If `v` is specified, only return outdegrees for nodes in `v`.
"""
outdegree(m::DCReM, v::Vector{Int}=collect(1:m.status[:N]); method::Symbol=:reduced) = [outdegree(m, i, method=method) for i in v]


"""
    indegree(m::DCReM, i::Int; method=:reduced)

Return the expected (binary) in-degree for node `i` of the DCReM model `m`.

# Arguments
- `method::Symbol`: `:reduced` (default), `:full` or `:adjacency`, see [`outdegree`](@ref).
"""
function indegree(m::DCReM, i::Int; method::Symbol=:reduced)
    m.status[:conditional_params_computed] ? nothing : throw(ArgumentError("The conditional parameters have not been computed yet"))
    i > m.status[:N] ? throw(ArgumentError("Attempted to access node $i in a $(m.status[:N]) node graph")) : nothing

    if method == :reduced
        res = zero(precision(m))
        i_red = m.dᵣ_ind[i] # find matching index in reduced model
        for j in eachindex(m.xᵣ)
            @inbounds pji = f_DBCM(m.xᵣ[j] * m.yᵣ[i_red])
            @inbounds res += pji * (m.f[j] - (i_red == j)) # subtract 1 within class because the diagonal is not counted
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
    indegree(m::DCReM[, v]; method=:reduced)

Return a vector corresponding to the expected (binary) in-degree of each node of the DCReM model `m`.
If `v` is specified, only return indegrees for nodes in `v`.
"""
indegree(m::DCReM, v::Vector{Int}=collect(1:m.status[:N]); method::Symbol=:reduced) = [indegree(m, i, method=method) for i in v]

"""
    degree(m::DCReM, i::Int; method=:reduced)

In alignment with `Graphs.jl`, returns the sum of the expected (binary) out- and in-degree for node `i`
of the DCReM model `m`.
"""
degree(m::DCReM, i::Int; method::Symbol=:reduced) = outdegree(m, i, method=method) + indegree(m, i, method=method)

"""
    degree(m::DCReM[, v]; method=:reduced)

In alignment with `Graphs.jl`, returns a vector corresponding to the sum of the expected (binary) out- and
in-degree of each node of the DCReM model `m`.
"""
degree(m::DCReM, v::Vector{Int}=collect(1:m.status[:N]); method::Symbol=:reduced) = outdegree(m, v, method=method) + indegree(m, v, method=method)


"""
    _DCReM_outstrength(θᵒ, θⁱ, x, y, i::Int, n::Int)

Kernel for the expected out-strength of node `i`, given the per-node binary fitnesses `x`, `y` and the
weighted parameters `θᵒ`, `θⁱ`.

Split out of [`outstrength`](@ref) so that the vector form can materialise the fitness vectors **once**
instead of once per node: they do not depend on the node being queried, and rebuilding them inside every
per-node call made the vector form allocate `2n` vectors of length `n`.
"""
function _DCReM_outstrength(θᵒ::AbstractVector, θⁱ::AbstractVector, x::AbstractVector, y::AbstractVector, i::Int, n::Int)
    res = zero(eltype(x))
    @inbounds for j in 1:n
        if i ≠ j
            xiyj = x[i]*y[j]
            res += (xiyj / (1 + xiyj)) / (θᵒ[i] + θⁱ[j])
        end
    end
    return res
end

"""
    _DCReM_outstrength_adj(θᵒ, θⁱ, Ĝ, i::Int, n::Int)

Kernel for the expected out-strength of node `i` reusing the precomputed adjacency matrix `Ĝ`.

See also [`_DCReM_outstrength`](@ref).
"""
function _DCReM_outstrength_adj(θᵒ::AbstractVector, θⁱ::AbstractVector, Ĝ::AbstractMatrix, i::Int, n::Int)
    res = zero(eltype(Ĝ))
    @inbounds for j in 1:n
        if i ≠ j
            res += Ĝ[i,j] / (θᵒ[i] + θⁱ[j])
        end
    end
    return res
end

"""
    outstrength(m::DCReM, i::Int; method=:reduced)

Return the expected (unconditional) out-strength for node `i` of the DCReM model `m`, i.e.
`Σⱼ≠ᵢ fᵢⱼ/(θᵒᵢ + θⁱⱼ)`.

# Arguments
- `method::Symbol`:
    - `:reduced` (default) / `:full` sum over all node pairs (the weighted parameters `θ` are not
      reducible, so these two are identical for the DCReM).
    - `:adjacency` reuses the precomputed adjacency matrix `m.Ĝ` (plus the `θ` parameters).
"""
function outstrength(m::DCReM, i::Int; method::Symbol=:reduced)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    i > m.status[:N] ? throw(ArgumentError("Attempted to access node $i in a $(m.status[:N]) node graph")) : nothing

    n = m.status[:N]::Int
    θᵒ = @view m.θ[1:n]
    θⁱ = @view m.θ[n+1:end]
    if method == :reduced || method == :full
        return _DCReM_outstrength(θᵒ, θⁱ, m.xᵣ[m.dᵣ_ind], m.yᵣ[m.dᵣ_ind], i, n)
    elseif method == :adjacency
        m.status[:G_computed] ? nothing : throw(ArgumentError("The adjacency matrix has not been computed yet"))
        return _DCReM_outstrength_adj(θᵒ, θⁱ, m.Ĝ, i, n)
    else
        throw(ArgumentError("Unknown method $method"))
    end
end

"""
    outstrength(m::DCReM[, v]; method=:reduced)

Return a vector corresponding to the expected out-strength of each node of the DCReM model `m`. If `v` is
specified, only return out-strengths for nodes in `v`.
"""
function outstrength(m::DCReM, v::Vector{Int}=collect(1:m.status[:N]); method::Symbol=:reduced)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    n = m.status[:N]::Int
    for i in v
        i > n ? throw(ArgumentError("Attempted to access node $i in a $(n) node graph")) : nothing
    end
    θᵒ = @view m.θ[1:n]
    θⁱ = @view m.θ[n+1:end]
    if method == :reduced || method == :full
        # the fitnesses do not depend on the node being queried: build them once for the whole vector
        x = m.xᵣ[m.dᵣ_ind]
        y = m.yᵣ[m.dᵣ_ind]
        return [_DCReM_outstrength(θᵒ, θⁱ, x, y, i, n) for i in v]
    elseif method == :adjacency
        m.status[:G_computed] ? nothing : throw(ArgumentError("The adjacency matrix has not been computed yet"))
        return [_DCReM_outstrength_adj(θᵒ, θⁱ, m.Ĝ, i, n) for i in v]
    else
        throw(ArgumentError("Unknown method $method"))
    end
end


"""
    _DCReM_instrength(θᵒ, θⁱ, x, y, i::Int, n::Int)

Kernel for the expected in-strength of node `i`, given the per-node binary fitnesses `x`, `y` and the
weighted parameters `θᵒ`, `θⁱ`.

See also [`_DCReM_outstrength`](@ref).
"""
function _DCReM_instrength(θᵒ::AbstractVector, θⁱ::AbstractVector, x::AbstractVector, y::AbstractVector, i::Int, n::Int)
    res = zero(eltype(x))
    @inbounds for j in 1:n
        if i ≠ j
            xjyi = x[j]*y[i]
            res += (xjyi / (1 + xjyi)) / (θⁱ[i] + θᵒ[j])
        end
    end
    return res
end

"""
    _DCReM_instrength_adj(θᵒ, θⁱ, Ĝ, i::Int, n::Int)

Kernel for the expected in-strength of node `i` reusing the precomputed adjacency matrix `Ĝ`.

See also [`_DCReM_outstrength`](@ref).
"""
function _DCReM_instrength_adj(θᵒ::AbstractVector, θⁱ::AbstractVector, Ĝ::AbstractMatrix, i::Int, n::Int)
    res = zero(eltype(Ĝ))
    @inbounds for j in 1:n
        if i ≠ j
            res += Ĝ[j,i] / (θⁱ[i] + θᵒ[j])
        end
    end
    return res
end

"""
    instrength(m::DCReM, i::Int; method=:reduced)

Return the expected (unconditional) in-strength for node `i` of the DCReM model `m`, i.e.
`Σⱼ≠ᵢ fⱼᵢ/(θⁱᵢ + θᵒⱼ)`.

# Arguments
- `method::Symbol`: `:reduced` (default) / `:full` / `:adjacency`, see [`outstrength`](@ref).
"""
function instrength(m::DCReM, i::Int; method::Symbol=:reduced)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    i > m.status[:N] ? throw(ArgumentError("Attempted to access node $i in a $(m.status[:N]) node graph")) : nothing

    n = m.status[:N]::Int
    θᵒ = @view m.θ[1:n]
    θⁱ = @view m.θ[n+1:end]
    if method == :reduced || method == :full
        return _DCReM_instrength(θᵒ, θⁱ, m.xᵣ[m.dᵣ_ind], m.yᵣ[m.dᵣ_ind], i, n)
    elseif method == :adjacency
        m.status[:G_computed] ? nothing : throw(ArgumentError("The adjacency matrix has not been computed yet"))
        return _DCReM_instrength_adj(θᵒ, θⁱ, m.Ĝ, i, n)
    else
        throw(ArgumentError("Unknown method $method"))
    end
end

"""
    instrength(m::DCReM[, v]; method=:reduced)

Return a vector corresponding to the expected in-strength of each node of the DCReM model `m`. If `v` is
specified, only return in-strengths for nodes in `v`.
"""
function instrength(m::DCReM, v::Vector{Int}=collect(1:m.status[:N]); method::Symbol=:reduced)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    n = m.status[:N]::Int
    for i in v
        i > n ? throw(ArgumentError("Attempted to access node $i in a $(n) node graph")) : nothing
    end
    θᵒ = @view m.θ[1:n]
    θⁱ = @view m.θ[n+1:end]
    if method == :reduced || method == :full
        # the fitnesses do not depend on the node being queried: build them once for the whole vector
        x = m.xᵣ[m.dᵣ_ind]
        y = m.yᵣ[m.dᵣ_ind]
        return [_DCReM_instrength(θᵒ, θⁱ, x, y, i, n) for i in v]
    elseif method == :adjacency
        m.status[:G_computed] ? nothing : throw(ArgumentError("The adjacency matrix has not been computed yet"))
        return [_DCReM_instrength_adj(θᵒ, θⁱ, m.Ĝ, i, n) for i in v]
    else
        throw(ArgumentError("Unknown method $method"))
    end
end


"""
    AIC(m::DCReM)

Compute the Akaike Information Criterion (AIC) for the DCReM model `m`. The parameters of the model must
be computed beforehand. The (conditional) DCReM has `2N` parameters (one out- and one in-rate ``θ`` per
node; the binary layer's `fᵢⱼ` are fixed inputs of the conditional likelihood).

See also [`AICc`](@ref MaxEntropyGraphs.AICc), [`L_DCReM`](@ref MaxEntropyGraphs.L_DCReM).
"""
function AIC(m::DCReM)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    k = 2 * m.status[:N] # number of parameters (two θ per node)
    n = (m.status[:N] - 1) * m.status[:N] # number of observations (ordered node pairs)
    L = L_DCReM(m) # log-likelihood

    if n/k < 40
        @warn """The number of observations is small with respect to the number of parameters (n/k < 40). Consider using the corrected AIC (AICc) instead."""
    end

    return 2*k - 2*L
end


"""
    AICc(m::DCReM)

Compute the corrected Akaike Information Criterion (AICc) for the DCReM model `m`.

See also [`AIC`](@ref MaxEntropyGraphs.AIC), [`L_DCReM`](@ref MaxEntropyGraphs.L_DCReM).
"""
function AICc(m::DCReM)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    k = 2 * m.status[:N] # number of parameters
    n = (m.status[:N] - 1) * m.status[:N] # number of observations
    L = L_DCReM(m) # log-likelihood

    return 2*k - 2*L + (2*k*(k+1)) / (n - k - 1)
end


"""
    BIC(m::DCReM)

Compute the Bayesian Information Criterion (BIC) for the DCReM model `m`.

See also [`AIC`](@ref MaxEntropyGraphs.AIC), [`L_DCReM`](@ref MaxEntropyGraphs.L_DCReM).
"""
function BIC(m::DCReM)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    k = 2 * m.status[:N] # number of parameters
    n = (m.status[:N] - 1) * m.status[:N] # number of observations
    L = L_DCReM(m) # log-likelihood

    return k * log(n) - 2*L
end


"""
    rand(m::DCReM; precomputed=false, rng=Random.default_rng())

Generate a random weighted directed graph from the DCReM model `m`.

The binary structure is drawn from the (binary) DBCM layer (`fᵢⱼ = xᵢyⱼ/(1 + xᵢyⱼ)`); for each realised
edge `i→j` a **continuous** weight is drawn from an exponential distribution with rate `θᵒᵢ + θⁱⱼ`.

# Arguments
- `precomputed::Bool`: not implemented yet for the DCReM (the parameters are always used to generate the graph on the fly).
- `rng::AbstractRNG`: random number generator to use (defaults to `Random.default_rng()`).

# Examples
```jldoctest
julia> model = DCReM(MaxEntropyGraphs.rhesus_macaques()); # generate a DCReM model

julia> solve_model!(model); # compute the maximum likelihood parameters

julia> sample = rand(model); # sample a random weighted directed graph

julia> typeof(sample)
SimpleWeightedGraphs.SimpleWeightedDiGraph{Int64, Float64}
```
"""
function rand(m::DCReM; precomputed::Bool=false, rng::AbstractRNG=default_rng())
    if precomputed
        throw(ArgumentError("This function is not implemented yet for DCReM models"))
    else
        # check if possible to use parameters
        m.status[:conditional_params_computed] ? nothing : throw(ArgumentError("The conditional parameters have not been computed yet"))
        m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
        # full per-node binary fitnesses + weighted parameters
        n = m.status[:N]::Int
        x = m.xᵣ[m.dᵣ_ind]
        y = m.yᵣ[m.dᵣ_ind]
        θᵒ = @view m.θ[1:n]
        θⁱ = @view m.θ[n+1:end]
        # generate random graph edges
        sources = Vector{Int}();
        targets = Vector{Int}();
        weights = Vector{Float64}();
        for i in 1:n
            for j in 1:n
                i == j && continue
                @inbounds xiyj = x[i]*y[j]
                p_ij = xiyj / (1 + xiyj)
                if rand(rng) ≤ p_ij
                    push!(sources, i)
                    push!(targets, j)
                    # continuous weight: exponential with rate θᵒᵢ+θⁱⱼ (mean 1/(θᵒᵢ+θⁱⱼ))
                    @inbounds push!(weights, rand(rng, Exponential(1 / (θᵒ[i] + θⁱ[j]))))
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
end

"""
    rand(m::DCReM, n::Int; precomputed=false, rng=Random.default_rng())

Generate `n` random weighted directed graphs from the DCReM model `m`. If multithreading is available,
the graphs are generated in parallel; per-sample seeds are drawn from `rng` so the result is reproducible
and independent of the thread schedule.
"""
function rand(m::DCReM, n::Int; precomputed::Bool=false, rng::AbstractRNG=default_rng())
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
    solve_model!(m::DCReM; kwargs...)

Compute the likelihood maximising parameters of the DCReM model `m`. This is a **two-step** process:
first the binary (conditional) DBCM layer is solved on the out/in-degree sequences, then the weighted
DCReM layer is solved on the out/in-strength sequences.

By default the weighted layer is computed with the fixed-point method using the strength sequences as
initial guess (the CReM-family fixed-point recipe is stable).

# Arguments (weighted DCReM layer)
- `method::Symbol`: solution method, `:fixedpoint` (default) or any of :$(join(keys(MaxEntropyGraphs.optimization_methods), ", :", " and :")).
- `initial::Symbol`: initial guess, `:strengths` (default), `:strengths_minor` or `:random`.
- `AD_method::Symbol`: autodiff method, any of :$(join(keys(MaxEntropyGraphs.AD_methods), ", :", " and :")) (defaults to `:AutoZygote`).
- `analytical_gradient::Bool`: use the analytical gradient instead of autodiff (defaults to `false`).
- `store_adjacency::Bool`: cache the (binary) expected adjacency matrix `m.Ĝ` and use it in the weighted solve (defaults to `false`).

# Arguments (binary conditional DBCM layer)
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
    The weighted layer is solved in **log-parameter** space (see `MaxEntropyGraphs._logspace_fixedpoint`).
    The fixed-point map obeys ``G_i = θ_i⟨s_i⟩/s_i`` exactly, so the log-space increment is exactly
    ``\\log(⟨s_i⟩/s_i)``, and `ftol` therefore bounds the **relative** strength residual
    ``|⟨s_i⟩/s_i - 1|`` on both the out- and the in-strengths. That makes it invariant under a rescaling
    of the weights: `ftol=1e-8` means eight significant digits on every strength whatever the units.
    (Solving in `θ` directly would instead bound ``|G_i - θ_i| = (θ_i/s_i)|⟨s_i⟩ - s_i|``, whose
    conversion factor ``s_i/θ_i`` grows as the *square* of the weight scale.) Use
    [`constraint_residual`](@ref) to measure the achieved residual in either absolute or relative form.

# Examples
```jldoctest DCReM_solve
# default use
julia> model = DCReM(MaxEntropyGraphs.rhesus_macaques());

julia> solve_model!(model);

```
"""
function solve_model!(m::DCReM; # weighted (DCReM) layer settings
                                method::Symbol=:fixedpoint,
                                initial::Symbol=:strengths,
                                AD_method::Symbol=:AutoZygote,
                                analytical_gradient::Bool=false,
                                store_adjacency::Bool=false,
                                # binary (conditional DBCM) layer settings
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
    # `ftol` reaches the fixed point solver of either layer, so it is only truly unused when neither
    # layer uses it: say so rather than ignoring it silently (only when it was actually passed)
    method ≠ :fixedpoint && method_conditional ≠ :fixedpoint && !isnothing(ftol) && @warn _ftol_unused_msg(method) maxlog=1
    ftol = isnothing(ftol) ? _DEFAULT_FTOL : ftol

    ## Part 1 - conditional binary layer (DBCM on the out/in-degree sequences)
    cond_model = DBCM(d_out = m.d_out, d_in = m.d_in, precision = N)
    # only hand `ftol` down when the binary layer can act on it, so that it never warns on our behalf
    solve_model!(cond_model, method=method_conditional, initial=initial_conditional,
                             AD_method=AD_method_conditional, analytical_gradient=analytical_gradient_conditional,
                             maxiters=maxiters, ftol=(method_conditional == :fixedpoint ? ftol : nothing),
                             abstol=abstol, reltol=reltol, verbose=verbose)
    m.αᵣ .= cond_model.θᵣ
    m.xᵣ .= cond_model.xᵣ
    m.yᵣ .= cond_model.yᵣ
    m.status[:conditional_params_computed] = true
    if store_adjacency
        set_Ĝ!(m)
    end

    ## Part 2 - weighted DCReM layer
    θ₀ = initial_guess(m, method=initial)
    # full per-node binary fitnesses (used for the on-the-fly fᵢⱼ unless the adjacency is cached)
    x = m.xᵣ[m.dᵣ_ind]
    y = m.yᵣ[m.dᵣ_ind]
    if method == :fixedpoint
        G_buffer = zeros(N, length(θ₀))
        FP_model! = m.status[:G_computed] ? (θ::Vector) -> DCReM_iter!(θ, m.s_out, m.s_in, m.Ĝ, G_buffer) :
                                            (θ::Vector) -> DCReM_iter!(θ, m.s_out, m.s_in, x, y, G_buffer)
        # solve in log-parameter space, where the increment is the *relative* strength residual and
        # `ftol` is scale-invariant (see `_logspace_fixedpoint`). Every node is live: a zero-strength
        # node has no finite θᵢ, so it is left to fail loudly rather than being masked away.
        θ_sol, sol = _logspace_fixedpoint(FP_model!, θ₀, eachindex(θ₀), ftol, maxiters)
        if NLsolve.converged(sol)
            verbose && @info "Fixed point iteration converged after $(sol.iterations) iterations"
            m.θ .= θ_sol
            m.status[:params_computed] = true
        else
            throw(ConvergenceError(method, nothing))
        end
    else
        if analytical_gradient
            grad! = m.status[:G_computed] ? (G, θ, p) -> ∇L_DCReM_minus!(G, θ, m.s_out, m.s_in, m.Ĝ) :
                                            (G, θ, p) -> ∇L_DCReM_minus!(G, θ, m.s_out, m.s_in, x, y)
        end
        # objective (negative log-likelihood); use the cached adjacency when available
        Lobj = m.status[:G_computed] ? ((θ, p) -> -L_DCReM(θ, m.s_out, m.s_in, m.Ĝ)) :
                                       ((θ, p) -> -L_DCReM(θ, m.s_out, m.s_in, x, y))
        f = AD_method ∈ keys(AD_methods) ? Optimization.OptimizationFunction(Lobj, AD_methods[AD_method], grad = analytical_gradient ? grad! : nothing) : throw(ArgumentError("The AD method $(AD_method) is not supported (yet)"))
        prob = Optimization.OptimizationProblem(f, θ₀)
        method ∈ keys(optimization_methods) || throw(ArgumentError("The method $(method) is not supported (yet)"))
        # use the BackTracking-line-search variants (see `CReM_optimization_methods`), falling back to
        # the package-wide optimizer for any method without one — the θᵒᵢ + θⁱⱼ > 0 barrier requires it.
        opt = get(CReM_optimization_methods, method, optimization_methods[method])
        solve_kwargs = isnothing(g_tol) ? (; maxiters = maxiters, abstol = abstol, reltol = reltol) :
                                          (; maxiters = maxiters, abstol = abstol, reltol = reltol, g_abstol = g_tol)
        sol = Optimization.solve(prob, opt; solve_kwargs...)
        if Optimization.SciMLBase.successful_retcode(sol.retcode)
            verbose && @info """$(method) optimisation converged after $(@sprintf("%1.2e", sol.stats.time)) seconds (Optimization.jl return code: $("$(sol.retcode)"))"""
            m.θ .= sol.u
            m.status[:params_computed] = true
        else
            throw(ConvergenceError(method, sol.retcode))
        end
    end

    return m, sol
end
