
"""
    CReM

Maximum entropy model for the Conditional Reconstruction Method (CReM).

The CReM is a **two-step** null model for weighted, undirected networks with **continuous, positive**
weights. The binary structure (topology) is supplied by a prior binary model — here an internally
solved [`UBCM`](@ref) on the degree sequence, giving the marginal edge probability
`fᵢⱼ = xᵢxⱼ/(1 + xᵢxⱼ)` with `xᵢ = e^{-αᵢ}`. Conditional on an edge existing, the weight follows an
exponential distribution with rate `θᵢ + θⱼ` (mean `1/(θᵢ + θⱼ)`); the parameters `θ` (one per node)
constrain the strength sequence.

The object holds the maximum likelihood parameters of the weighted layer (`θ`), the reduced binary
(conditional) UBCM parameters (`αᵣ`) and their exponentiated form (`xᵣ`).
"""
mutable struct CReM{T<:Union{Graphs.AbstractGraph, Nothing}, N<:Real} <: AbstractMaxEntropyModel
    "Graph type, can be any subtype of AbstractGraph, but will be converted to SimpleWeightedGraph for the computation" # can also be empty
    const G::T
    "Maximum likelihood parameters of the weighted (CReM) layer, θ (one per node)"
    const θ::Vector{N}
    "Reduced maximum likelihood parameters of the binary (conditional UBCM) layer, αᵣ"
    const αᵣ::Vector{N}
    "Exponentiated reduced binary parameters ( xᵢ = exp(-αᵢ) )"
    const xᵣ::Vector{N}
    "Degree sequence of the graph"
    const d::Vector{Int}
    "Reduced degree sequence of the graph"
    const dᵣ::Vector{Int}
    "Frequency of each degree in the degree sequence"
    const f::Vector{Int}
    "Strength sequence of the graph (continuous, positive)"
    const s::Vector{N}
    "Indices to reconstruct the degree sequence from the reduced degree sequence"
    const d_ind::Vector{Int}
    "Indices to reconstruct the reduced degree sequence from the degree sequence"
    const dᵣ_ind::Vector{Int}
    "Expected (binary) adjacency matrix" # not always computed/required
    Ĝ::Union{Nothing, Matrix{N}}
    "Variance of the expected (binary) adjacency matrix" # not always computed/required
    σ::Union{Nothing, Matrix{N}}
    "Expected weighted adjacency matrix" # not always computed/required
    Ŵ::Union{Nothing, Matrix{N}}
    "Standard deviation of the expected weighted adjacency matrix" # not always computed/required
    σʷ::Union{Nothing, Matrix{N}}
    "Status indicators: parameters computed, expected adjacency matrix computed, variance computed, etc."
    const status::Dict{Symbol, Any}
end

Base.show(io::IO, m::CReM{T,N}) where {T,N} = print(io, """CReM{$(T), $(N)} ($(m.status[:N]) vertices, $(m.status[:d_unique]) unique degrees, $(@sprintf("%.2f", m.status[:cᵣ])) compression ratio)""")

"""Return the number of vertices in the CReM network"""
Base.length(m::CReM) = length(m.d)


"""
    CReM(G::T; d::Vector, s::Vector, precision::Type{<:AbstractFloat}=Float64, kwargs...) where {T}

Constructor function for the `CReM` type.

By default and depending on the graph type `T`, the definition of degree from ``Graphs.jl`` and strength from ``SimpleWeightedGraphs`` is applied.
If you want to use a different definition of degree or strength, you can pass the vectors as keyword arguments.

If you want to generate a model directly from a degree and strength sequence without an underlying graph you can simply pass them as keyword arguments (`d` and `s`).
If you want to work from an adjacency/weight matrix, or edge list, you can use the (weighted) graph constructors from the ``JuliaGraphs`` ecosystem.

The CReM allows **continuous, positive** weights (the strength sequence need not be integer-valued).

# Examples
```jldoctest CReM_creation
# generating a model from a weighted graph (here: the symmetrised rhesus macaques network)
julia> G = MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(MaxEntropyGraphs.rhesus_macaques());

julia> model = CReM(G)
CReM{SimpleWeightedGraphs.SimpleWeightedGraph{Int64, Float64}, Float64} (16 vertices, 16 unique degrees, 1.00 compression ratio)

```
```jldoctest CReM_creation
# generating a model directly from a degree and strength sequence
julia> model = CReM(d=[1, 2, 2, 1], s=[3.0, 5.0, 4.0, 2.0])
CReM{Nothing, Float64} (4 vertices, 3 unique degrees, 0.75 compression ratio)

```
```jldoctest CReM_creation
# generating a model with a different precision
julia> model = CReM(d=[1, 2, 2, 1], s=[3.0, 5.0, 4.0, 2.0], precision=Float32);

julia> MaxEntropyGraphs.precision(model)
Float32

```
"""
function CReM(G::T; d::Vector=Graphs.degree(G), s::Vector=strength(G), precision::Type{<:AbstractFloat}=Float64, kwargs...) where {T}
    T <: Union{Graphs.AbstractGraph, Nothing} ? nothing : throw(TypeError(:CReM, "G must be a subtype of AbstractGraph or Nothing", Union{Graphs.AbstractGraph, Nothing}, T))
    # coherence checks
    if T <: Graphs.AbstractGraph # Graph specific checks
        if Graphs.is_directed(G)
            @warn "The graph is directed, the CReM model is undirected, the directional information will be lost"
        end

        Graphs.nv(G) == 0 ? throw(ArgumentError("The graph is empty")) : nothing
        Graphs.nv(G) == 1 ? throw(ArgumentError("The graph has only one vertex")) : nothing

        Graphs.nv(G) != length(d) ? throw(DimensionMismatch("The number of vertices in the graph ($(Graphs.nv(G))) and the length of the degree sequence ($(length(d))) do not match")) : nothing
        Graphs.nv(G) != length(s) ? throw(DimensionMismatch("The number of vertices in the graph ($(Graphs.nv(G))) and the length of the strength sequence ($(length(s))) do not match")) : nothing
        length(d) != length(s) ? throw(DimensionMismatch("The dimensions of the degree ($(length(d))) and the strength sequence ($(length(s))) do not match")) : nothing
    end

    # coherence checks specific to the degree/strength sequence
    if any(iszero, d)
        @warn "The graph has vertices with zero degree, this may lead to convergence issues."
    end
    if any(iszero, s)
        @warn "The graph has vertices with zero strength, this may lead to convergence issues."
    end
    any(!isinteger, d) ? throw(DomainError("Some of the degree values are not integers, this is not allowed")) : nothing
    length(d) == 0 ? throw(ArgumentError("The degree sequence is empty")) : nothing
    length(d) == 1 ? throw(ArgumentError("The degree sequence has only one degree")) : nothing
    length(d) != length(s) ? throw(DimensionMismatch("The dimensions of the degree ($(length(d))) and the strength sequence ($(length(s))) do not match")) : nothing
    maximum(d) >= length(d) ? throw(DomainError("The maximum degree in the graph is greater or equal to the number of vertices, this is not allowed")) : nothing
    all(d .>= 0) ? nothing : throw(DomainError("The degree sequence contains negative degrees"))
    all(s .>= 0) ? nothing : throw(DomainError("The strength sequence contains negative strengths"))

    # field generation (the binary layer reduces over the degree sequence, exactly like the UBCM)
    dᵣ, d_ind, dᵣ_ind, f = np_unique_clone(d, sorted=true)
    θ  = Vector{precision}(undef, length(d))    # weighted (CReM) parameters, one per node
    αᵣ = Vector{precision}(undef, length(dᵣ))   # reduced binary (UBCM) parameters
    xᵣ = Vector{precision}(undef, length(dᵣ))   # exponentiated reduced binary parameters
    status = Dict(  :conditional_params_computed => false,   # is the binary (UBCM) layer solved?
                    :params_computed             => false,   # is the weighted (CReM) layer solved?
                    :G_computed                  => false,   # is the expected adjacency matrix computed and stored?
                    :σ_computed                  => false,   # is the standard deviation computed and stored?
                    :W_computed                  => false,   # is the expected weighted adjacency matrix computed and stored?
                    :σʷ_computed                 => false,   # is the weight standard deviation computed and stored?
                    :cᵣ       => length(dᵣ)/length(d),       # compression ratio of the reduced binary model
                    :d_unique => length(dᵣ),                 # number of unique degrees in the reduced model
                    :N        => length(d)                   # number of vertices in the original graph
                )

    return CReM{T,precision}(G, θ, αᵣ, xᵣ, Int.(d), dᵣ, f, Vector{precision}(s), d_ind, dᵣ_ind, nothing, nothing, nothing, nothing, status)
end

CReM(;d::Vector, s::Vector, precision::Type{<:AbstractFloat}=Float64, kwargs...) = CReM(nothing, d=d, s=s, precision=precision, kwargs...)



"""
    L_CReM(θ::AbstractVector, s::AbstractVector, f::AbstractVector)

Compute the log-likelihood of the CReM model given the weighted parameters `θ`, the strength sequence
`s` and the per-node binary fitness `f` (`fᵢ = xᵢ`), computing the marginal edge probability
`fᵢⱼ = fᵢfⱼ/(1 + fᵢfⱼ)` on the fly.

The log-likelihood is ``\\mathcal{L} = -\\sum_i θ_i s_i + \\sum_{i} \\sum_{j<i} f_{ij} \\log(θ_i + θ_j)``.
The `log` is domain-guarded (returns `NaN` for `θᵢ + θⱼ ≤ 0`) so that the line search rejects steps that
leave the feasible region; this keeps it automatic-differentiation friendly (used for the AD cross-check).

This method is used when the marginal probability matrix is not precomputed (i.e. `model.Ĝ` is `nothing`).
"""
function L_CReM(θ::AbstractVector, s::AbstractVector, f::AbstractVector)
    res = zero(eltype(θ))
    @inbounds for i in eachindex(θ)
        res -= θ[i] * s[i]
        fᵢ = f[i]; θᵢ = θ[i]
        acc = zero(eltype(θ))
        for j in 1:i-1
            fij = fᵢ * f[j] / (1 + fᵢ * f[j])
            acc += fij * log_nan(θᵢ + θ[j])
        end
        res += acc
    end

    return res
end


"""
    L_CReM(θ::AbstractVector, s::AbstractVector, f::AbstractMatrix)

Compute the log-likelihood of the CReM model, using the precomputed marginal probability matrix `f`
(i.e. `model.Ĝ`). See also [`L_CReM(::AbstractVector, ::AbstractVector, ::AbstractVector)`](@ref).
"""
function L_CReM(θ::AbstractVector, s::AbstractVector, f::AbstractMatrix)
    res = zero(eltype(θ))
    @inbounds for i in eachindex(θ)
        res -= θ[i] * s[i]
        θᵢ = θ[i]
        acc = zero(eltype(θ))
        for j in 1:i-1
            acc += f[i,j] * log_nan(θᵢ + θ[j])
        end
        res += acc
    end

    return res
end


"""
    L_CReM(m::CReM)

Return the log-likelihood of the CReM model `m` based on the computed maximum likelihood parameters.
Depending on the status of the model, the precomputed marginal probability matrix (`m.Ĝ`) is used, or
the marginal probability is computed on the fly.

See also [`L_CReM(::AbstractVector, ::AbstractVector, ::AbstractVector)`](@ref).
"""
function L_CReM(m::CReM)
    m.status[:conditional_params_computed] ? nothing : throw(ArgumentError("The conditional parameters of the model have not been computed"))
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters of the model have not been computed"))
    if m.status[:G_computed]
        return L_CReM(m.θ, m.s, m.Ĝ)
    else
        return L_CReM(m.θ, m.s, m.xᵣ[m.dᵣ_ind])
    end
end


"""
    ∇L_CReM!(∇L::AbstractVector, θ::AbstractVector, s::AbstractVector, f::AbstractVector)

Compute the gradient of the log-likelihood of the CReM model in a non-allocating manner, computing the
marginal edge probability on the fly (`f` = per-node binary fitness).

The inner loop is branch-free (the spurious `j == i` self-term is folded out afterwards) so that it
vectorises (`@simd`). `∂L/∂θᵢ = -sᵢ + Σⱼ≠ᵢ fᵢⱼ/(θᵢ + θⱼ)`.

See also [`∇L_CReM_minus!`](@ref).
"""
function ∇L_CReM!(∇L::AbstractVector, θ::AbstractVector, s::AbstractVector, f::AbstractVector)
    @inbounds for i in eachindex(θ)
        θᵢ = θ[i]; fᵢ = f[i]
        acc = zero(eltype(∇L))
        @simd for j in eachindex(θ)
            fij = fᵢ * f[j] / (1 + fᵢ * f[j])
            acc += fij / (θᵢ + θ[j])
        end
        # subtract the spurious j == i self-term (fᵢᵢ/(2θᵢ)) that the branch-free sum included
        fii = fᵢ * fᵢ / (1 + fᵢ * fᵢ)
        acc -= fii / (2 * θᵢ)
        ∇L[i] = -s[i] + acc
    end

    return ∇L
end


"""
    ∇L_CReM!(∇L::AbstractVector, θ::AbstractVector, s::AbstractVector, f::AbstractMatrix)

Compute the gradient of the log-likelihood of the CReM model, using the precomputed marginal probability
matrix `f` (i.e. `model.Ĝ`). The `f[i,i] = 0` diagonal makes the self-term vanish automatically.
"""
function ∇L_CReM!(∇L::AbstractVector, θ::AbstractVector, s::AbstractVector, f::AbstractMatrix)
    @inbounds for i in eachindex(θ)
        θᵢ = θ[i]
        acc = zero(eltype(∇L))
        @simd for j in eachindex(θ)
            acc += f[i,j] / (θᵢ + θ[j])
        end
        ∇L[i] = -s[i] + acc
    end

    return ∇L
end


"""
    ∇L_CReM_minus!(∇L::AbstractVector, θ::AbstractVector, s::AbstractVector, f::AbstractVector)

Compute minus the gradient of the log-likelihood of the CReM model (used for the minimisation carried
out by `Optimization.jl`), computing the marginal edge probability on the fly. Non-allocating.

See also [`∇L_CReM!`](@ref).
"""
function ∇L_CReM_minus!(∇L::AbstractVector, θ::AbstractVector, s::AbstractVector, f::AbstractVector)
    @inbounds for i in eachindex(θ)
        θᵢ = θ[i]; fᵢ = f[i]
        acc = zero(eltype(∇L))
        @simd for j in eachindex(θ)
            fij = fᵢ * f[j] / (1 + fᵢ * f[j])
            acc += fij / (θᵢ + θ[j])
        end
        fii = fᵢ * fᵢ / (1 + fᵢ * fᵢ)
        acc -= fii / (2 * θᵢ)
        ∇L[i] = s[i] - acc
    end

    return ∇L
end


"""
    ∇L_CReM_minus!(∇L::AbstractVector, θ::AbstractVector, s::AbstractVector, f::AbstractMatrix)

Compute minus the gradient of the log-likelihood of the CReM model, using the precomputed marginal
probability matrix `f` (i.e. `model.Ĝ`). Non-allocating.
"""
function ∇L_CReM_minus!(∇L::AbstractVector, θ::AbstractVector, s::AbstractVector, f::AbstractMatrix)
    @inbounds for i in eachindex(θ)
        θᵢ = θ[i]
        acc = zero(eltype(∇L))
        @simd for j in eachindex(θ)
            acc += f[i,j] / (θᵢ + θ[j])
        end
        ∇L[i] = s[i] - acc
    end

    return ∇L
end


"""
    CReM_iter!(θ::AbstractVector, s::AbstractVector, f::AbstractVector, G::AbstractVector)

Compute the next fixed-point iteration for the CReM model, computing the marginal edge probability on
the fly (`f` = per-node binary fitness). The pre-allocated buffer `G` is updated in place.

The consistency equation is ``θ_i = \\left( \\sum_{j\\neq i} f_{ij}/(1 + θ_j/θ_i) \\right) / s_i``.
"""
function CReM_iter!(θ::AbstractVector, s::AbstractVector, f::AbstractVector, G::AbstractVector)
    @inbounds for i in eachindex(θ)
        θᵢ = θ[i]; fᵢ = f[i]
        acc = zero(eltype(θ))
        @simd for j in eachindex(θ)
            fij = fᵢ * f[j] / (1 + fᵢ * f[j])
            acc += fij / (1 + θ[j] / θᵢ)
        end
        # subtract the spurious j == i self-term (fᵢᵢ/2)
        fii = fᵢ * fᵢ / (1 + fᵢ * fᵢ)
        acc -= fii / 2
        G[i] = acc / s[i]
    end

    return G
end


"""
    CReM_iter!(θ::AbstractVector, s::AbstractVector, f::AbstractMatrix, G::AbstractVector)

Compute the next fixed-point iteration for the CReM model, using the precomputed marginal probability
matrix `f` (i.e. `model.Ĝ`). The pre-allocated buffer `G` is updated in place.
"""
function CReM_iter!(θ::AbstractVector, s::AbstractVector, f::AbstractMatrix, G::AbstractVector)
    @inbounds for i in eachindex(θ)
        θᵢ = θ[i]
        acc = zero(eltype(θ))
        @simd for j in eachindex(θ)
            acc += f[i,j] / (1 + θ[j] / θᵢ)
        end
        G[i] = acc / s[i]
    end

    return G
end


"""
    initial_guess(m::CReM; method::Symbol=:strengths)

Compute an initial guess `θ₀` for the maximum likelihood parameters of the weighted (CReM) layer of the
model `m`.

The CReM parameters `θ` are the **direct** rate parameters (they appear as `log(θᵢ + θⱼ)` in the
log-likelihood), so every method returns strictly positive, feasible values — a negative `θᵢ + θⱼ` would
put the objective outside its domain. The forms mirror NEMtropy's `crema` initial guesses:
- `:strengths` (default): `θᵢ = 𝟙[sᵢ > 0] / Σⱼ sⱼ`.
- `:strengths_minor`: `θᵢ = 𝟙[sᵢ > 0] / (sᵢ + 1)`.
- `:random`: random values drawn from ``U(0,1)``.
"""
function initial_guess(m::CReM; method::Symbol=:strengths)
    N = precision(m)
    if isequal(method, :strengths)
        res = Vector{N}((m.s .> 0) ./ sum(m.s))
    elseif isequal(method, :strengths_minor)
        res = Vector{N}((m.s .> 0) ./ (m.s .+ 1))
    elseif isequal(method, :random)
        res = Vector{N}(rand(N, length(m.s)))
    else
        throw(ArgumentError("The initial guess method $(method) is not supported"))
    end

    return res
end


"""
    precision(m::CReM)

Determine the compute precision of the CReM model `m`.
"""
precision(m::CReM) = typeof(m).parameters[2]


"""
    set_xᵣ!(m::CReM)

Set the value of `xᵣ` to `exp(-αᵣ)` for the (binary layer of the) CReM model `m`.
"""
function set_xᵣ!(m::CReM)
    if m.status[:conditional_params_computed]
        m.xᵣ .= exp.(-m.αᵣ)
    else
        throw(ArgumentError("The conditional parameters have not been computed yet"))
    end
end


"""
    f_CReM(xixj::T) where {T}

Helper for the CReM model computing the (binary) expected adjacency entry `fᵢⱼ = xᵢxⱼ/(1 + xᵢxⱼ)` from
the product `xᵢxⱼ` of the binary maximum likelihood parameters.
"""
f_CReM(xixj::T) where {T} = xixj / (one(T) + xixj)


"""
    A(m::CReM, i::Int, j::Int)

Return the expected value of the (binary) adjacency matrix for the CReM model `m` at the node pair `(i,j)`.

❗ For performance reasons, the function does not check:
- if the node pair is valid.
- if the parameters of the model have been computed.
"""
function A(m::CReM, i::Int, j::Int)
    return i == j ? zero(precision(m)) : @inbounds f_CReM(m.xᵣ[m.dᵣ_ind[i]] * m.xᵣ[m.dᵣ_ind[j]])
end


"""
    Ĝ(m::CReM)

Compute the expected (binary) **adjacency** matrix for the CReM model `m`, i.e. `fᵢⱼ = xᵢxⱼ/(1 + xᵢxⱼ)`,
so that `sum(Ĝ(m), dims=2) ≈ degree`.

Note: The expected weights can be computed separately with [`Ŵ`](@ref MaxEntropyGraphs.Ŵ).
"""
function Ĝ(m::CReM)
    m.status[:conditional_params_computed] ? nothing : throw(ArgumentError("The conditional parameters have not been computed yet"))

    n = m.status[:N]
    G = zeros(precision(m), n, n)
    x = m.xᵣ[m.dᵣ_ind]
    for i = 1:n
        @simd for j = i+1:n
            @inbounds xixj = x[i]*x[j]
            @inbounds pij  = xixj / (1 + xixj)
            @inbounds G[i,j] = pij
            @inbounds G[j,i] = pij
        end
    end

    return G
end

"""
    set_Ĝ!(m::CReM)

Set the expected (binary) adjacency matrix for the CReM model `m`.
"""
function set_Ĝ!(m::CReM)
    m.Ĝ = Ĝ(m)
    m.status[:G_computed] = true
    return m.Ĝ
end


"""
    Ŵ(m::CReM)

Compute the expected (unconditional) **weighted adjacency** matrix for the CReM model `m`, i.e.
`⟨wᵢⱼ⟩ = fᵢⱼ / (θᵢ + θⱼ)`, so that `sum(Ŵ(m), dims=2) ≈ strength`.
"""
function Ŵ(m::CReM)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))

    n = m.status[:N]
    W = zeros(precision(m), n, n)
    x = m.xᵣ[m.dᵣ_ind]
    θ = m.θ
    for i = 1:n
        @simd for j = i+1:n
            @inbounds xixj = x[i]*x[j]
            @inbounds pij  = xixj / (1 + xixj)
            @inbounds wij  = pij / (θ[i] + θ[j])
            @inbounds W[i,j] = wij
            @inbounds W[j,i] = wij
        end
    end

    return W
end

"""
    set_Ŵ!(m::CReM)

Set the expected weighted adjacency matrix for the CReM model `m`.
"""
function set_Ŵ!(m::CReM)
    m.Ŵ = Ŵ(m)
    m.status[:W_computed] = true
    return m.Ŵ
end


"""
    σˣ(m::CReM)

Compute the standard deviation for the elements of the (binary) adjacency matrix for the CReM model `m`,
i.e. `sqrt(fᵢⱼ(1 - fᵢⱼ))` (the adjacency entries are Bernoulli distributed).

**Note:** this is the standard deviation of the *binary* layer; the standard deviation of the weights is
available via [`σʷ`](@ref MaxEntropyGraphs.σʷ). Read as "sigma star".
"""
function σˣ(m::CReM)
    m.status[:conditional_params_computed] ? nothing : throw(ArgumentError("The conditional parameters have not been computed yet"))

    n = m.status[:N]
    σ = zeros(precision(m), n, n)
    x = m.xᵣ[m.dᵣ_ind]
    for i = 1:n
        @simd for j = i+1:n
            @inbounds xixj = x[i]*x[j]
            @inbounds pij  = xixj / (1 + xixj)
            @inbounds sij  = sqrt(pij * (1 - pij))
            @inbounds σ[i,j] = sij
            @inbounds σ[j,i] = sij
        end
    end

    return σ
end

"""
    set_σ!(m::CReM)

Set the standard deviation for the elements of the (binary) adjacency matrix for the CReM model `m`.
"""
function set_σ!(m::CReM)
    m.σ = σˣ(m)
    m.status[:σ_computed] = true
    return m.σ
end


"""
    σʷ(m::CReM)

Compute the standard deviation for the elements of the **weighted** adjacency matrix for the CReM model
`m`. The (unconditional) weight `wᵢⱼ` is a mixture of an exponential (with probability `fᵢⱼ`) and zero,
so ``⟨w_{ij}^2⟩ = 2f_{ij}/(θ_i + θ_j)^2`` and

``Var(w_{ij}) = \\frac{f_{ij}(2 - f_{ij})}{(θ_i + θ_j)^2}``

As the network is undirected, `wᵢⱼ` and `wⱼᵢ` denote the same random variable; the corresponding
covariance is accounted for by [`σₓ`](@ref MaxEntropyGraphs.σₓ).
"""
function σʷ(m::CReM)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    m.status[:conditional_params_computed] ? nothing : throw(ArgumentError("The conditional parameters have not been computed yet"))

    n = m.status[:N]
    σ = zeros(precision(m), n, n)
    x = m.xᵣ[m.dᵣ_ind]
    θ = m.θ
    for i = 1:n
        @simd for j = i+1:n
            @inbounds xixj = x[i]*x[j]
            @inbounds fij  = xixj / (1 + xixj)
            @inbounds sij  = sqrt(fij * (2 - fij)) / (θ[i] + θ[j])
            @inbounds σ[i,j] = sij
            @inbounds σ[j,i] = sij
        end
    end

    return σ
end

"""
    set_σʷ!(m::CReM)

Set the standard deviation for the elements of the weighted adjacency matrix for the CReM model `m`.
"""
function set_σʷ!(m::CReM)
    m.σʷ = σʷ(m)
    m.status[:σʷ_computed] = true
    return m.σʷ
end


"""
    σₓ(m::CReM, X::Function; layer::Symbol=:binary, gradient_method::Symbol=:ReverseDiff)

Compute the standard deviation of metric `X` for the CReM model `m` via error propagation (the delta
method of Squartini & Garlaschelli (2011), Eq. B.16).

# Arguments
- `layer::Symbol`:
    - `:binary` (default): propagate over the **binary adjacency matrix** — `X` is a function of the
      adjacency matrix, the gradient is evaluated at `m.Ĝ` and weighted by `m.σ` (requires `set_Ĝ!` and `set_σ!`).
    - `:weighted`: propagate over the **weighted adjacency matrix** — `X` is a function of the weight
      matrix, the gradient is evaluated at `m.Ŵ` and weighted by `m.σʷ` (requires `set_Ŵ!` and `set_σʷ!`).
- `gradient_method::Symbol`: `:ForwardDiff`, `:ReverseDiff` (default) or `:Zygote`.

As the network is undirected, the entries `(i,j)` and `(j,i)` of either layer denote the *same* random
variable, so the delta method over ordered pairs includes the within-dyad covariance cross-term
(`Cov(g_ij, g_ji) = σ²[g_ij]`). This makes the result independent of whether `X` is written using one or
both triangles of the matrix.

Metrics mixing the two layers (functions of both the adjacency and the weight matrix) are not supported
by this per-layer propagation; use ensemble sampling instead.
"""
function σₓ(m::CReM, X::Function; layer::Symbol=:binary, gradient_method::Symbol=:ReverseDiff)
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

    # delta method over ordered pairs; g_ij ≡ g_ji (undirected), so Cov(g_ij, g_ji) = σ²[g_ij]:
    # σ²[X] = Σ σ²∇² + Σ σ²∇∇ᵀ = Σ σ² ∇ (∇ + ∇ᵀ), fused into a single broadcast
    return sqrt( sum(S .^ 2 .* ∇X .* (∇X .+ transpose(∇X))) )
end


"""
    degree(m::CReM, i::Int; method=:reduced)

Return the expected (binary) degree for node `i` of the CReM model `m`.

# Arguments
- `method::Symbol`:
    - `:reduced` (default) uses the reduced binary model parameters `xᵣ`.
    - `:full` uses all elements of the expected adjacency matrix.
    - `:adjacency` uses the precomputed adjacency matrix `m.Ĝ` of the model.
"""
function degree(m::CReM, i::Int; method::Symbol=:reduced)
    m.status[:conditional_params_computed] ? nothing : throw(ArgumentError("The conditional parameters have not been computed yet"))
    i > m.status[:N] ? throw(ArgumentError("Attempted to access node $i in a $(m.status[:N]) node graph")) : nothing

    if method == :reduced
        res = zero(precision(m))
        i_red = m.dᵣ_ind[i] # find matching index in reduced model
        for j in eachindex(m.xᵣ)
            @inbounds pij = f_CReM(m.xᵣ[i_red] * m.xᵣ[j])
            if i_red ≠ j
                @inbounds res += pij * m.f[j]
            else
                @inbounds res += pij * (m.f[j] - 1) # subtract 1 because the diagonal is not counted
            end
        end
    elseif method == :full
        res = zero(precision(m))
        for j in eachindex(m.d)
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
    degree(m::CReM[, v]; method=:reduced)

Return a vector corresponding to the expected (binary) degree of each node of the CReM model `m`. If `v`
is specified, only return degrees for nodes in `v`.
"""
degree(m::CReM, v::Vector{Int}=collect(1:m.status[:N]); method::Symbol=:reduced) = [degree(m, i, method=method) for i in v]


"""
    strength(m::CReM, i::Int; method=:reduced)

Return the expected (unconditional) strength for node `i` of the CReM model `m`, i.e.
`Σⱼ≠ᵢ fᵢⱼ/(θᵢ + θⱼ)`.

# Arguments
- `method::Symbol`:
    - `:reduced` (default) / `:full` sum over all node pairs (the weighted parameters `θ` are not
      reducible, so these two are identical for the CReM).
    - `:adjacency` reuses the precomputed adjacency matrix `m.Ĝ` (plus the `θ` parameters).
"""
function strength(m::CReM, i::Int; method::Symbol=:reduced)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    i > m.status[:N] ? throw(ArgumentError("Attempted to access node $i in a $(m.status[:N]) node graph")) : nothing

    θ = m.θ
    if method == :reduced || method == :full
        x = m.xᵣ[m.dᵣ_ind]
        res = zero(precision(m))
        @inbounds for j in 1:m.status[:N]
            if i ≠ j
                xixj = x[i]*x[j]
                pij  = xixj / (1 + xixj)
                res += pij / (θ[i] + θ[j])
            end
        end
    elseif method == :adjacency
        m.status[:G_computed] ? nothing : throw(ArgumentError("The adjacency matrix has not been computed yet"))
        res = zero(precision(m))
        @inbounds for j in 1:m.status[:N]
            if i ≠ j
                res += m.Ĝ[i,j] / (θ[i] + θ[j])
            end
        end
    else
        throw(ArgumentError("Unknown method $method"))
    end

    return res
end

"""
    strength(m::CReM[, v]; method=:reduced)

Return a vector corresponding to the expected strength of each node of the CReM model `m`. If `v` is
specified, only return strengths for nodes in `v`.
"""
strength(m::CReM, v::Vector{Int}=collect(1:m.status[:N]); method::Symbol=:reduced) = [strength(m, i, method=method) for i in v]


"""
    AIC(m::CReM)

Compute the Akaike Information Criterion (AIC) for the CReM model `m`. The parameters of the model must
be computed beforehand. The (conditional) CReM has `N` parameters (one weighted parameter ``θ`` per node;
the binary layer's `fᵢⱼ` are fixed inputs of the conditional likelihood).

See also [`AICc`](@ref MaxEntropyGraphs.AICc), [`L_CReM`](@ref MaxEntropyGraphs.L_CReM).
"""
function AIC(m::CReM)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    k = m.status[:N] # number of parameters (one θ per node)
    n = (m.status[:N] - 1) * m.status[:N] / 2 # number of observations (node pairs)
    L = L_CReM(m) # log-likelihood

    if n/k < 40
        @warn """The number of observations is small with respect to the number of parameters (n/k < 40). Consider using the corrected AIC (AICc) instead."""
    end

    return 2*k - 2*L
end


"""
    AICc(m::CReM)

Compute the corrected Akaike Information Criterion (AICc) for the CReM model `m`.

See also [`AIC`](@ref MaxEntropyGraphs.AIC), [`L_CReM`](@ref MaxEntropyGraphs.L_CReM).
"""
function AICc(m::CReM)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    k = m.status[:N] # number of parameters
    n = (m.status[:N] - 1) * m.status[:N] / 2 # number of observations
    L = L_CReM(m) # log-likelihood

    return 2*k - 2*L + (2*k*(k+1)) / (n - k - 1)
end


"""
    BIC(m::CReM)

Compute the Bayesian Information Criterion (BIC) for the CReM model `m`.

See also [`AIC`](@ref MaxEntropyGraphs.AIC), [`L_CReM`](@ref MaxEntropyGraphs.L_CReM).
"""
function BIC(m::CReM)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    k = m.status[:N] # number of parameters
    n = (m.status[:N] - 1) * m.status[:N] / 2 # number of observations
    L = L_CReM(m) # log-likelihood

    return k * log(n) - 2*L
end


"""
    rand(m::CReM; precomputed=false, rng=Random.default_rng())

Generate a random weighted graph from the CReM model `m`.

The binary structure is drawn from the (binary) UBCM layer (`fᵢⱼ = xᵢxⱼ/(1 + xᵢxⱼ)`); for each realised
edge a **continuous** weight is drawn from an exponential distribution with rate `θᵢ + θⱼ`.

# Arguments
- `precomputed::Bool`: not implemented yet for the CReM (the parameters are always used to generate the graph on the fly).
- `rng::AbstractRNG`: random number generator to use (defaults to `Random.default_rng()`).

# Examples
```jldoctest
julia> model = CReM(MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(MaxEntropyGraphs.rhesus_macaques())); # generate a CReM model

julia> solve_model!(model); # compute the maximum likelihood parameters

julia> sample = rand(model); # sample a random weighted graph

julia> typeof(sample)
SimpleWeightedGraphs.SimpleWeightedGraph{Int64, Float64}
```
"""
function rand(m::CReM; precomputed::Bool=false, rng::AbstractRNG=default_rng())
    if precomputed
        throw(ArgumentError("This function is not implemented yet for CReM models"))
    else
        # check if possible to use parameters
        m.status[:conditional_params_computed] ? nothing : throw(ArgumentError("The conditional parameters have not been computed yet"))
        m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
        # full per-node binary fitness + weighted parameters
        x = m.xᵣ[m.dᵣ_ind]
        θ = m.θ
        # generate random graph edges
        sources = Vector{Int}();
        targets = Vector{Int}();
        weights = Vector{Float64}();
        for i in 1:m.status[:N]
            for j in 1:i-1
                @inbounds xixj = x[i]*x[j]
                p_ij = xixj / (1 + xixj)
                if rand(rng) ≤ p_ij
                    push!(sources, i)
                    push!(targets, j)
                    # continuous weight: exponential with rate θᵢ+θⱼ (mean 1/(θᵢ+θⱼ))
                    @inbounds push!(weights, rand(rng, Exponential(1 / (θ[i] + θ[j]))))
                end
            end
        end

        if length(sources) ≠ 0
            G = SimpleWeightedGraphs.SimpleWeightedGraph(sources, targets, weights)
        else
            G = SimpleWeightedGraphs.SimpleWeightedGraph(m.status[:N])
        end

        # deal with edge case where no edges are generated for the last node(s) in the graph
        while Graphs.nv(G) < m.status[:N]
            Graphs.add_vertex!(G)
        end
        return G
    end
end

"""
    rand(m::CReM, n::Int; precomputed=false, rng=Random.default_rng())

Generate `n` random weighted graphs from the CReM model `m`. If multithreading is available, the graphs
are generated in parallel; per-sample seeds are drawn from `rng` so the result is reproducible and
independent of the thread schedule.
"""
function rand(m::CReM, n::Int; precomputed::Bool=false, rng::AbstractRNG=default_rng())
    # pre-allocate
    res = Vector{SimpleWeightedGraphs.SimpleWeightedGraph}(undef, n)
    # per-sample seeds drawn from `rng` (reproducible, thread-schedule-independent)
    seeds = rand(rng, UInt64, n)
    # fill vector using threads
    Threads.@threads for i in 1:n
        res[i] = rand(m; precomputed=precomputed, rng=Xoshiro(seeds[i]))
    end

    return res
end



# The CReM weighted layer requires `θᵢ + θⱼ > 0` (the argument of the `log(θᵢ + θⱼ)` term). The default
# `:fixedpoint` recipe starts from a positive guess and stays positive, so it sidesteps this barrier
# entirely; the gradient-based methods, however, can step out of the feasible region, where the
# (domain-guarded) objective is `NaN`. A BackTracking line search halves the step until the objective is
# finite, keeping the iterates feasible — so the gradient methods get their own optimizer instances.
const CReM_optimization_methods = Dict( :LBFGS  => OptimizationOptimJL.LBFGS( linesearch = OptimizationOptimJL.Optim.LineSearches.BackTracking()),
                                        :BFGS   => OptimizationOptimJL.BFGS(  linesearch = OptimizationOptimJL.Optim.LineSearches.BackTracking()),
                                        :Newton => OptimizationOptimJL.Newton(linesearch = OptimizationOptimJL.Optim.LineSearches.BackTracking()))

"""
    solve_model!(m::CReM; kwargs...)

Compute the likelihood maximising parameters of the CReM model `m`. This is a **two-step** process:
first the binary (conditional) UBCM layer is solved on the degree sequence, then the weighted CReM layer
is solved on the strength sequence.

By default the weighted layer is computed with the fixed-point method using the strength sequence as
initial guess (the CReM fixed-point recipe is stable, unlike the UECM's).

# Arguments (weighted CReM layer)
- `method::Symbol`: solution method, `:fixedpoint` (default) or any of :$(join(keys(MaxEntropyGraphs.optimization_methods), ", :", " and :")).
- `initial::Symbol`: initial guess, `:strengths` (default), `:strengths_minor` or `:random`.
- `AD_method::Symbol`: autodiff method, any of :$(join(keys(MaxEntropyGraphs.AD_methods), ", :", " and :")) (defaults to `:AutoZygote`).
- `analytical_gradient::Bool`: use the analytical gradient instead of autodiff (defaults to `false`).
- `store_adjacency::Bool`: cache the (binary) expected adjacency matrix `m.Ĝ` and use it in the weighted solve (defaults to `false`).

# Arguments (binary conditional UBCM layer)
- `method_conditional::Symbol`: solution method for the binary layer (defaults to `:fixedpoint`).
- `initial_conditional::Symbol`: initial guess for the binary layer (defaults to `:degrees`).
- `AD_method_conditional::Symbol`: autodiff method for the binary layer (defaults to `:AutoZygote`).
- `analytical_gradient_conditional::Bool`: analytical gradient for the binary layer (defaults to `false`).

# Common settings
- `maxiters::Int`: maximum number of iterations (defaults to 1000).
- `verbose::Bool`: show log messages (defaults to false).
- `ftol::Real`: function tolerance for the fixedpoint method (defaults to 1e-8).
- `abstol`, `reltol`: absolute/relative tolerances for the optimisation methods (default `nothing`).
- `g_tol::Union{Number, Nothing}`: gradient tolerance for the gradient-based methods (maps to Optim's `g_abstol`, default `nothing`).
"""
function solve_model!(m::CReM;  # weighted (CReM) layer settings
                                method::Symbol=:fixedpoint,
                                initial::Symbol=:strengths,
                                AD_method::Symbol=:AutoZygote,
                                analytical_gradient::Bool=false,
                                store_adjacency::Bool=false,
                                # binary (conditional UBCM) layer settings
                                method_conditional::Symbol=:fixedpoint,
                                initial_conditional::Symbol=:degrees,
                                AD_method_conditional::Symbol=:AutoZygote,
                                analytical_gradient_conditional::Bool=false,
                                # common settings
                                maxiters::Int=1000,
                                verbose::Bool=false,
                                ftol::Real=1e-8,
                                abstol::Union{Number, Nothing}=nothing,
                                reltol::Union{Number, Nothing}=nothing,
                                g_tol::Union{Number, Nothing}=nothing)
    N = precision(m)
    N <: Union{Float16, Float32} && @warn "Solving in $(N) precision is experimental and may not converge; low precision is intended for storage. Consider Float64 for the solve." maxlog=1

    ## Part 1 - conditional binary layer (UBCM on the degree sequence)
    cond_model = UBCM(d = m.d, precision = N)
    solve_model!(cond_model, method=method_conditional, initial=initial_conditional,
                             AD_method=AD_method_conditional, analytical_gradient=analytical_gradient_conditional,
                             maxiters=maxiters, ftol=ftol, abstol=abstol, reltol=reltol, verbose=verbose)
    m.αᵣ .= cond_model.θᵣ
    m.xᵣ .= cond_model.xᵣ
    m.status[:conditional_params_computed] = true
    if store_adjacency
        set_Ĝ!(m)
    end

    ## Part 2 - weighted CReM layer
    θ₀ = initial_guess(m, method=initial)
    # full per-node binary fitness (used for the on-the-fly fᵢⱼ unless the adjacency is cached)
    x = m.xᵣ[m.dᵣ_ind]
    if method == :fixedpoint
        G_buffer = zeros(N, length(θ₀))
        FP_model! = m.status[:G_computed] ? (θ::Vector) -> CReM_iter!(θ, m.s, m.Ĝ, G_buffer) :
                                            (θ::Vector) -> CReM_iter!(θ, m.s, x, G_buffer)
        sol = NLsolve.fixedpoint(FP_model!, θ₀, method=:anderson, ftol=ftol, iterations=maxiters)
        if NLsolve.converged(sol)
            verbose && @info "Fixed point iteration converged after $(sol.iterations) iterations"
            m.θ .= sol.zero
            m.status[:params_computed] = true
        else
            throw(ConvergenceError(method, nothing))
        end
    else
        if analytical_gradient
            grad! = m.status[:G_computed] ? (G, θ, p) -> ∇L_CReM_minus!(G, θ, m.s, m.Ĝ) :
                                            (G, θ, p) -> ∇L_CReM_minus!(G, θ, m.s, x)
        end
        # objective (negative log-likelihood); use the cached adjacency when available
        Lobj = m.status[:G_computed] ? ((θ, p) -> -L_CReM(θ, m.s, m.Ĝ)) :
                                       ((θ, p) -> -L_CReM(θ, m.s, x))
        f = AD_method ∈ keys(AD_methods) ? Optimization.OptimizationFunction(Lobj, AD_methods[AD_method], grad = analytical_gradient ? grad! : nothing) : throw(ArgumentError("The AD method $(AD_method) is not supported (yet)"))
        prob = Optimization.OptimizationProblem(f, θ₀)
        method ∈ keys(optimization_methods) || throw(ArgumentError("The method $(method) is not supported (yet)"))
        # use the BackTracking-line-search variants (see `CReM_optimization_methods` above), falling
        # back to the package-wide optimizer for any method without a BackTracking variant.
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
