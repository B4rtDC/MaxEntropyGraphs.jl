
"""
    UECM

Maximum entropy model for the Undirected Enhanced Configuration Model (UECM).

The object holds the maximum likelihood parameters of the model (θ = [α; β]), the expected adjacency matrix (Ĝ),
and the variance for the elements of the adjacency matrix (σ).

The UECM constrains both the degree sequence and the (integer) strength sequence of an undirected weighted network.

Note: this requires that the weights only assume (non-negative) integer values.
"""
mutable struct UECM{T<:Union{Graphs.AbstractGraph, Nothing}, N<:Real} <: AbstractMaxEntropyModel
    "Graph type, can be any subtype of AbstractGraph, but will be converted to SimpleWeightedGraph for the computation" # can also be empty
    const G::T
    "Maximum likelihood parameters for reduced model"
    const θᵣ::Vector{N}
    "Exponentiated maximum likelihood parameters for reduced model ( xᵢ = exp(-αᵢ) )"
    const xᵣ::Vector{N}
    "Exponentiated maximum likelihood parameters for reduced model ( yᵢ = exp(-βᵢ) )"
    const yᵣ::Vector{N}
    "Degree sequence of the graph" # evaluate usefulness of this field later on
    const d::Vector{Int}
    "Reduced degree sequence of the graph"
    const dᵣ::Vector{Int}
    "Strength sequence of the graph"
    const s::Vector{Int}
    "Reduced strength sequence of the graph"
    const sᵣ::Vector{Int}
    "Frequency of each (degree, strength) pair in the graph"
    const f::Vector{Int}
    "Indices to reconstruct the degree sequence from the reduced degree sequence"
    const d_ind::Vector{Int}
    "Indices to reconstruct the reduced degree sequence from the degree sequence"
    const dᵣ_ind::Vector{Int}
    "Non-zero constraints"
    const nz::Vector{Int}
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

Base.show(io::IO, m::UECM{T,N}) where {T,N} = print(io, """UECM{$(T), $(N)} ($(m.status[:N]) vertices, $(m.status[:d_unique]) unique {degree,strength} pairs, $(@sprintf("%.2f", m.status[:cᵣ])) compression ratio)""")

"""Return the reduced number of {degree, strength} pairs in the UECM network"""
Base.length(m::UECM) = length(m.dᵣ)



"""
    UECM(G::T; d::Vector, s::Vector, precision::Type{<:AbstractFloat}=Float64, kwargs...) where {T}

Constructor function for the `UECM` type.

By default and depending on the graph type `T`, the definition of degree from ``Graphs.jl`` and strength from ``SimpleWeightedGraphs`` is applied.
If you want to use a different definition of degree or strength, you can pass the vectors as keyword arguments.

If you want to generate a model directly from a degree and strength sequence without an underlying graph you can simply pass them as keyword arguments (`d` and `s`).
If you want to work from an adjacency/weight matrix, or edge list, you can use the (weighted) graph constructors from the ``JuliaGraphs`` ecosystem.

The strength sequence must be integer-valued (the UECM is defined for non-negative integer weights).

# Examples
```jldoctest UECM_creation
# generating a model from a weighted graph (here: the symmetrised rhesus macaques network)
julia> G = MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(MaxEntropyGraphs.rhesus_macaques());

julia> model = UECM(G)
UECM{SimpleWeightedGraphs.SimpleWeightedGraph{Int64, Float64}, Float64} (16 vertices, 16 unique {degree,strength} pairs, 1.00 compression ratio)

```
```jldoctest UECM_creation
# generating a model directly from a degree and strength sequence
julia> model = UECM(d=[1, 2, 2, 1], s=[3, 5, 4, 2])
UECM{Nothing, Float64} (4 vertices, 4 unique {degree,strength} pairs, 1.00 compression ratio)

```
```jldoctest UECM_creation
# generating a model with a different precision
julia> model = UECM(d=[1, 2, 2, 1], s=[3, 5, 4, 2], precision=Float32);

julia> MaxEntropyGraphs.precision(model)
Float32

```

The degree is taken from `Graphs.degree` and the strength from `MaxEntropyGraphs.strength` (the
`SimpleWeightedGraphs` weighted-degree).
"""
function UECM(G::T; d::Vector=Graphs.degree(G), s::Vector=strength(G), precision::Type{<:AbstractFloat}=Float64, kwargs...) where {T}
    T <: Union{Graphs.AbstractGraph, Nothing} ? nothing : throw(TypeError(:UECM, "G must be a subtype of AbstractGraph or Nothing", Union{Graphs.AbstractGraph, Nothing}, T))
    # coherence checks
    if T <: Graphs.AbstractGraph # Graph specific checks
        if Graphs.is_directed(G)
            @warn "The graph is directed, the UECM model is undirected, the directional information will be lost"
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
    any(!isinteger, s) ? throw(DomainError("Some of the strength values are not integers, this is not allowed")) : nothing
    length(d) == 0 ? throw(ArgumentError("The degree sequence is empty")) : nothing
    length(d) == 1 ? throw(ArgumentError("The degree sequence has only one degree")) : nothing
    length(d) != length(s) ? throw(DimensionMismatch("The dimensions of the degree ($(length(d))) and the strength sequence ($(length(s))) do not match")) : nothing
    maximum(d) >= length(d) ? throw(DomainError("The maximum degree in the graph is greater or equal to the number of vertices, this is not allowed")) : nothing

    # field generation
    dsᵣ, d_ind , dᵣ_ind, f = np_unique_clone(collect(zip(d, s)), sorted=true)
    dᵣ = Int.([d[1] for d in dsᵣ])
    sᵣ = Int.([d[2] for d in dsᵣ])
    θᵣ = Vector{precision}(undef, 2*length(dᵣ))
    xᵣ = Vector{precision}(undef, length(dᵣ))
    yᵣ = Vector{precision}(undef, length(dᵣ))
    nz = findall(!iszero, dᵣ)
    status = Dict(  :params_computed=>false,        # are the parameters computed?
                    :G_computed=>false,             # is the expected adjacency matrix computed and stored?
                    :σ_computed=>false,             # is the standard deviation computed and stored?
                    :W_computed=>false,             # is the expected weighted adjacency matrix computed and stored?
                    :σʷ_computed=>false,            # is the weight standard deviation computed and stored?
                    :cᵣ => length(dᵣ)/length(d),    # compression ratio of the reduced model
                    :d_unique => length(dᵣ),        # number of unique {degree,strength} pairs in the reduced model
                    :N => length(d)                 # number of vertices in the original graph
                )

    return UECM{T,precision}(G, θᵣ, xᵣ, yᵣ, Int.(d), dᵣ, Int.(s), sᵣ, f, d_ind, dᵣ_ind, nz, nothing, nothing, nothing, nothing, status, nothing)
end

UECM(;d::Vector, s::Vector, precision::Type{<:AbstractFloat}=Float64, kwargs...) = UECM(nothing, d=d, s=s, precision=precision, kwargs...)



"""
    L_UECM_reduced(θ::Vector, d::Vector, s::Vector, F::Vector, n::Int=length(d))

Compute the log-likelihood of the reduced UECM model using the exponential formulation in order to maintain convexity.

# Arguments
- `θ`: the maximum likelihood parameters of the model (`θ = [α; β]`)
- `d`: the reduced degree sequence
- `s`: the reduced strength sequence
- `F`: the frequency of each `(degree, strength)` pair
- `n`: the number of unique `(degree, strength)` pairs (defaults to `length(d)`)

The function is numerically stabilised (`expm1`/`log1p`) so that the `1 - exp(-βᵢ-βⱼ)` denominator does not
suffer from catastrophic cancellation, and stays automatic-differentiation friendly.

# Examples
```jldoctest
julia> θ = [1.0, 2.0, 3.0, 4.0, 1.5, 2.5];

julia> d = [1, 2, 1]; s = [2, 3, 1]; F = [1, 1, 1];

julia> L_UECM_reduced(θ, d, s, F);

```
"""
function L_UECM_reduced(θ::AbstractVector, d::Vector, s::Vector, F::Vector, n::Int=length(d))
    # split the parameters
    α = @view θ[1:n]
    β = @view θ[n+1:end]
    # initiate
    res = zero(eltype(θ))
    # compute
    for i in eachindex(d)
        @inbounds αᵢ = α[i]; βᵢ = β[i]; Fᵢ = F[i]
        @inbounds res -= Fᵢ * (d[i] * αᵢ + βᵢ * s[i])
        # off-diagonal pairs (j < i): weight Fᵢ·F[j]
        acc = zero(eltype(θ))
        @inbounds for j in 1:i-1
            c1    = exp(-αᵢ - α[j])
            c2    = exp(-βᵢ - β[j])
            om_c2 = -expm1(-(βᵢ + β[j]))                # == 1 - c2, computed without cancellation
            # the UECM is only defined for yᵢyⱼ < 1 (i.e. om_c2 > 0). Returning NaN for the whole
            # out-of-domain region (om_c2 ≤ 0) makes the line search reject those steps, instead of
            # letting the optimiser escape to β → -∞ where the objective is spuriously unbounded.
            acc  += F[j] * (om_c2 > zero(om_c2) ? log1p(c1 * c2 / om_c2) : oftype(c1, NaN))
        end
        res -= Fᵢ * acc
        # diagonal self-pairs (within class i): Fᵢ·(Fᵢ-1)/2 pairs
        @inbounds begin
            c1    = exp(-2αᵢ)
            c2    = exp(-2βᵢ)
            om_c2 = -expm1(-2βᵢ)
            contrib = om_c2 > zero(om_c2) ? log1p(c1 * c2 / om_c2) : oftype(c1, NaN)
            res  -= Fᵢ * (Fᵢ - 1) * contrib * 0.5
        end
    end

    return res
end

"""
    L_UECM_reduced(m::UECM)

Return the log-likelihood of the UECM model `m` based on the computed maximum likelihood parameters.

See also [`L_UECM_reduced(::AbstractVector, ::Vector, ::Vector, ::Vector)`](@ref)
"""
function L_UECM_reduced(m::UECM)
    if m.status[:params_computed]
        return L_UECM_reduced(m.θᵣ, m.dᵣ, m.sᵣ, m.f, m.status[:d_unique])
    else
        throw(ArgumentError("The parameters have not been computed yet"))
    end
end


"""
    ∇L_UECM_reduced!(∇L::AbstractVector, θ::AbstractVector, d::Vector, s::Vector, F::Vector, x::AbstractVector, y::AbstractVector, n=length(θ)÷2)

Compute the gradient of the log-likelihood of the reduced UECM model in a non-allocating manner.
The pre-allocated buffers `x` (`xᵢ = exp(-αᵢ)`) and `y` (`yᵢ = exp(-βᵢ)`) and the gradient `∇L` are updated in place.

The inner loop is branch-free (the diagonal correction is folded into the multiplier `F[j] - (i==j)`) so that it
vectorises (`@simd`).

See also [`∇L_UECM_reduced_minus!`](@ref).
"""
function ∇L_UECM_reduced!(∇L::AbstractVector, θ::AbstractVector, d::Vector, s::Vector, F::Vector, x::AbstractVector, y::AbstractVector, n=length(θ)÷2)
    α = @view θ[1:n]
    β = @view θ[n+1:end]
    @inbounds @simd for i in eachindex(α) # to avoid the allocation of exp.(-θ)
        x[i] = exp(-α[i])
        y[i] = exp(-β[i])
    end

    for i in eachindex(α)
        @inbounds xᵢ = x[i]
        @inbounds yᵢ = y[i]
        accα = zero(eltype(∇L))
        accβ = zero(eltype(∇L))
        @inbounds @simd for j in eachindex(α)
            c1    = xᵢ * x[j]
            c2    = yᵢ * y[j]
            denom = 1 + c1 * c2 - c2
            p     = (c1 * c2) / denom
            w     = F[j] - (i == j)                     # (F[j]-1) on the diagonal, F[j] elsewhere
            accα += p * w
            accβ += (p / (1 - c2)) * w
        end
        @inbounds ∇L[i]   = -F[i] * d[i] + F[i] * accα
        @inbounds ∇L[i+n] = -F[i] * s[i] + F[i] * accβ
    end

    return ∇L
end


"""
    ∇L_UECM_reduced_minus!(args...)

Compute minus the gradient of the log-likelihood of the reduced UECM model (used for the minimisation carried out by
`Optimization.jl`). Non-allocating: updates the pre-allocated buffers `x`, `y` and `∇L` in place.

See also [`∇L_UECM_reduced!`](@ref).
"""
function ∇L_UECM_reduced_minus!(∇L::AbstractVector, θ::AbstractVector, d::Vector, s::Vector, F::Vector, x::AbstractVector, y::AbstractVector, n=length(θ)÷2)
    α = @view θ[1:n]
    β = @view θ[n+1:end]
    @inbounds @simd for i in eachindex(α) # to avoid the allocation of exp.(-θ)
        x[i] = exp(-α[i])
        y[i] = exp(-β[i])
    end

    for i in eachindex(α)
        @inbounds xᵢ = x[i]
        @inbounds yᵢ = y[i]
        accα = zero(eltype(∇L))
        accβ = zero(eltype(∇L))
        @inbounds @simd for j in eachindex(α)
            c1    = xᵢ * x[j]
            c2    = yᵢ * y[j]
            denom = 1 + c1 * c2 - c2
            p     = (c1 * c2) / denom
            w     = F[j] - (i == j)
            accα += p * w
            accβ += (p / (1 - c2)) * w
        end
        @inbounds ∇L[i]   = F[i] * d[i] - F[i] * accα
        @inbounds ∇L[i+n] = F[i] * s[i] - F[i] * accβ
    end

    return ∇L
end


"""
    UECM_reduced_iter!(θ, d, s, F, x, y, G, nz, n=length(θ)÷2)

Compute the next fixed-point iteration for the reduced UECM model. The pre-allocated buffers `x`, `y` and `G` are
updated in place. Only the non-zero constraints `nz` are iterated.

**Note**: the fixed-point recipe is unstable for the UECM; `:BFGS`/`:Newton` are preferred (see [`solve_model!`](@ref)).
"""
function UECM_reduced_iter!(θ::AbstractVector, d::Vector, s::Vector, F::Vector, x::AbstractVector, y::AbstractVector, G::AbstractVector, nz::Vector, n=length(θ)÷2)
    α = @view θ[1:n]
    β = @view θ[n+1:end]
    @inbounds @simd for i in eachindex(α) # to avoid the allocation of exp.(-θ)
        x[i] = exp(-α[i])
        y[i] = exp(-β[i])
    end

    # compute
    for i in nz
        @inbounds xᵢ = x[i]
        @inbounds yᵢ = y[i]
        accα = zero(eltype(θ))
        accβ = zero(eltype(θ))
        @inbounds for j in nz
            c1    = xᵢ * x[j]
            c2    = yᵢ * y[j]
            denom = 1 + c1 * c2 - c2
            wj    = F[j] - (i == j)
            accα += (x[j] * c2) / denom * wj
            accβ += (c1 * y[j]) / ((1 - c2) * denom) * wj
        end
        @inbounds G[i]   = F[i] * accα
        @inbounds G[i+n] = F[i] * accβ
    end

    for i in nz
        @inbounds G[i]   = - log_nan(F[i] * d[i] / G[i])
        @inbounds G[i+n] = - log_nan(F[i] * s[i] / G[i+n])
    end

    for i in eachindex(G)
        @inbounds if iszero(G[i])
            G[i] = 1e8
        end
    end

    return G
end


"""
    initial_guess(m::UECM; method::Symbol=:strengths)

Compute an initial guess `θ₀ = [α₀; β₀]` for the maximum likelihood parameters of the UECM model `m`.

The methods available are:
- `:strengths` (default): degrees/strengths normalised by the number of edges / total strength.
- `:strengths_minor`: `1/(dᵣ+1)` and `1/(sᵣ+1)`.
- `:random`: random values drawn from ``U(0,1)``.
- `:uniform`: uniformly set to `-log(0.001)`.
"""
function initial_guess(m::UECM; method::Symbol=:strengths)
    N = precision(m)
    if isequal(method, :strengths )
        isnothing(m.G) ? throw(ArgumentError("Cannot compute the number of edges because the model has no underlying graph (m.G == nothing)")) : nothing
        res = Vector{N}(-log.(vcat( m.dᵣ ./ (Graphs.ne(m.G) + 1),
                                    m.sᵣ ./ (sum(m.sᵣ) + 1)))) # will return Inf initial guesses for zero values
    elseif isequal(method, :strengths_minor)
        res = Vector{N}(-log.(vcat( ones(N, length(m.dᵣ)) ./ (m.dᵣ .+ 1),
                                    ones(N, length(m.sᵣ)) ./ (m.sᵣ .+ 1))))
    elseif isequal(method, :random)
        res =  Vector{N}(-log.(rand(N, 2*length(m.dᵣ))))
    elseif isequal(method, :uniform)
        res = Vector{N}(-log.(0.001 .* ones(N, 2*length(m.dᵣ))))
    else
        throw(ArgumentError("The initial guess method $(method) is not supported"))
    end

    return res
end


"""
    set_xᵣ!(m::UECM)

Set the value of xᵣ to exp(-αᵣ) for the UECM model `m`
"""
function set_xᵣ!(m::UECM)
    if m.status[:params_computed]
        αᵣ = @view m.θᵣ[1:m.status[:d_unique]]
        m.xᵣ .= exp.(-αᵣ)
    else
        throw(ArgumentError("The parameters have not been computed yet"))
    end
end

"""
    set_yᵣ!(m::UECM)

Set the value of yᵣ to exp(-βᵣ) for the UECM model `m`
"""
function set_yᵣ!(m::UECM)
    if m.status[:params_computed]
        βᵣ = @view m.θᵣ[m.status[:d_unique]+1:end]
        m.yᵣ .= exp.(-βᵣ)
    else
        throw(ArgumentError("The parameters have not been computed yet"))
    end
end


"""
    precision(m::UECM)

Determine the compute precision of the UECM model `m`.
"""
precision(m::UECM) = typeof(m).parameters[2]


"""
    f_UECM(xixj::T, yiyj::T) where {T}

Helper for the UECM model computing the expected adjacency entry `pᵢⱼ = (xᵢxⱼ·yᵢyⱼ)/(1 - yᵢyⱼ + xᵢxⱼ·yᵢyⱼ)` from the
products `xᵢxⱼ` and `yᵢyⱼ` of the maximum likelihood parameters.
"""
f_UECM(xixj::T, yiyj::T) where {T} = (xixj * yiyj) / (one(T) - yiyj + xixj * yiyj)


"""
    A(m::UECM, i::Int, j::Int)

Return the expected value of the adjacency matrix for the UECM model `m` at the node pair `(i,j)`.

❗ For performance reasons, the function does not check:
- if the node pair is valid.
- if the parameters of the model have been computed.
"""
function A(m::UECM, i::Int, j::Int)
    return i == j ? zero(precision(m)) : @inbounds f_UECM(m.xᵣ[m.dᵣ_ind[i]] * m.xᵣ[m.dᵣ_ind[j]], m.yᵣ[m.dᵣ_ind[i]] * m.yᵣ[m.dᵣ_ind[j]])
end


"""
    Ĝ(m::UECM)

Compute the expected **adjacency** matrix for the UECM model `m`.

Note: The expected weights can be computed separately with [`Ŵ`](@ref MaxEntropyGraphs.Ŵ).
"""
function Ĝ(m::UECM)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))

    # get network size => this is the full size
    n = m.status[:N]::Int
    # initiate G
    G = zeros(precision(m), n, n)
    # initiate x and y
    x = m.xᵣ[m.dᵣ_ind]
    y = m.yᵣ[m.dᵣ_ind]
    # compute G (symmetric, branch-free)
    for i = 1:n
        @simd for j = i+1:n
            @inbounds xixj = x[i]*x[j]
            @inbounds yiyj = y[i]*y[j]
            @inbounds pij  = (xixj * yiyj) / (1 - yiyj + xixj * yiyj)
            @inbounds G[i,j] = pij
            @inbounds G[j,i] = pij
        end
    end

    return G
end

"""
    set_Ĝ!(m::UECM)

Set the expected adjacency matrix for the UECM model `m`
"""
function set_Ĝ!(m::UECM)
    m.Ĝ = Ĝ(m)
    m.status[:G_computed] = true
    return m.Ĝ
end


"""
    Ŵ(m::UECM)

Compute the expected (unconditional) **weighted adjacency** matrix for the UECM model `m`, i.e.
`⟨wᵢⱼ⟩ = pᵢⱼ / (1 - yᵢyⱼ)`, so that `sum(Ŵ(m), dims=2) ≈ strength`.
"""
function Ŵ(m::UECM)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))

    # get network size => this is the full size
    n = m.status[:N]::Int
    # initiate W
    W = zeros(precision(m), n, n)
    # initiate x and y
    x = m.xᵣ[m.dᵣ_ind]
    y = m.yᵣ[m.dᵣ_ind]
    # compute W (symmetric, branch-free): unconditional expected weight
    for i = 1:n
        @simd for j = i+1:n
            @inbounds xixj = x[i]*x[j]
            @inbounds yiyj = y[i]*y[j]
            @inbounds pij  = (xixj * yiyj) / (1 - yiyj + xixj * yiyj)
            @inbounds wij  = pij / (1 - yiyj)
            @inbounds W[i,j] = wij
            @inbounds W[j,i] = wij
        end
    end

    return W
end

"""
    set_Ŵ!(m::UECM)

Set the expected weighted adjacency matrix for the UECM model `m`.
"""
function set_Ŵ!(m::UECM)
    m.Ŵ = Ŵ(m)
    m.status[:W_computed] = true
    return m.Ŵ
end


"""
    σˣ(m::UECM{T,N}) where {T,N}

Compute the standard deviation for the elements of the (binary) adjacency matrix for the UECM model `m`, i.e.
`sqrt(pᵢⱼ(1 - pᵢⱼ))` (the adjacency entries are Bernoulli distributed).

**Note:** this is the standard deviation of the *binary* layer; the standard deviation of the weights is
available via [`σʷ`](@ref MaxEntropyGraphs.σʷ). Read as "sigma star".
"""
function σˣ(m::UECM)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    # network size => full size
    n = m.status[:N]::Int
    # initiate σ
    σ = zeros(precision(m), n, n)
    # initiate x and y
    x = m.xᵣ[m.dᵣ_ind]
    y = m.yᵣ[m.dᵣ_ind]
    # compute σ (symmetric, branch-free)
    for i = 1:n
        @simd for j = i+1:n
            @inbounds xixj = x[i]*x[j]
            @inbounds yiyj = y[i]*y[j]
            @inbounds pij  = (xixj * yiyj) / (1 - yiyj + xixj * yiyj)
            @inbounds sij  = sqrt(pij * (1 - pij))
            @inbounds σ[i,j] = sij
            @inbounds σ[j,i] = sij
        end
    end

    return σ
end

"""
    set_σ!(m::UECM)

Set the standard deviation for the elements of the (binary) adjacency matrix for the UECM model `m`
"""
function set_σ!(m::UECM)
    m.σ = σˣ(m)
    m.status[:σ_computed] = true
    return m.σ
end


"""
    σʷ(m::UECM)

Compute the standard deviation for the elements of the **weighted** adjacency matrix for the UECM model `m`.
The weight `wᵢⱼ` follows a Bernoulli–geometric mixture: with `pᵢⱼ` the connection probability and
`yᵢyⱼ` the geometric parameter, ``⟨w_{ij}⟩ = p_{ij}/(1 - y_iy_j)``,
``⟨w_{ij}^2⟩ = p_{ij}(1 + y_iy_j)/(1 - y_iy_j)^2`` and

``Var(w_{ij}) = \\frac{p_{ij}(1 + y_iy_j - p_{ij})}{(1 - y_iy_j)^2}``

As the network is undirected, `wᵢⱼ` and `wⱼᵢ` denote the same random variable; the corresponding
covariance is accounted for by [`σₓ`](@ref MaxEntropyGraphs.σₓ).
"""
function σʷ(m::UECM)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    # network size => full size
    n = m.status[:N]::Int
    # initiate σ
    σ = zeros(precision(m), n, n)
    # initiate x and y
    x = m.xᵣ[m.dᵣ_ind]
    y = m.yᵣ[m.dᵣ_ind]
    # compute σ (symmetric, branch-free)
    for i = 1:n
        @simd for j = i+1:n
            @inbounds xixj = x[i]*x[j]
            @inbounds yiyj = y[i]*y[j]
            @inbounds pij  = (xixj * yiyj) / (1 - yiyj + xixj * yiyj)
            @inbounds sij  = sqrt(pij * (1 + yiyj - pij)) / (1 - yiyj)
            @inbounds σ[i,j] = sij
            @inbounds σ[j,i] = sij
        end
    end

    return σ
end

"""
    set_σʷ!(m::UECM)

Set the standard deviation for the elements of the weighted adjacency matrix for the UECM model `m`.
"""
function set_σʷ!(m::UECM)
    m.σʷ = σʷ(m)
    m.status[:σʷ_computed] = true
    return m.σʷ
end


"""
    σₓ(m::UECM, X::Function; layer::Symbol=:binary, gradient_method::Symbol=:ReverseDiff)

Compute the standard deviation of metric `X` for the UECM model `m` via error propagation (the delta
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
function σₓ(m::UECM, X::Function; layer::Symbol=:binary, gradient_method::Symbol=:ReverseDiff)
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

    # delta method over ordered pairs; g_ij ≡ g_ji (undirected), so Cov(g_ij, g_ji) = σ²[g_ij]:
    # σ²[X] = Σ σ²∇² + Σ σ²∇∇ᵀ = Σ σ² ∇ (∇ + ∇ᵀ), fused into a single broadcast
    return sqrt( sum(S .^ 2 .* ∇X .* (∇X .+ transpose(∇X))) )
end


"""
    degree(m::UECM, i::Int; method=:reduced)

Return the expected degree for node `i` of the UECM model `m`.

# Arguments
- `method::Symbol`:
    - `:reduced` (default) uses the reduced model parameters `xᵣ`/`yᵣ`.
    - `:full` uses all elements of the expected adjacency matrix.
    - `:adjacency` uses the precomputed adjacency matrix `m.Ĝ` of the model.
"""
function degree(m::UECM, i::Int; method::Symbol=:reduced)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    i > m.status[:N] ? throw(ArgumentError("Attempted to access node $i in a $(m.status[:N]) node graph")) : nothing

    if method == :reduced
        res = zero(precision(m))
        i_red = m.dᵣ_ind[i] # find matching index in reduced model
        for j in eachindex(m.xᵣ)
            @inbounds pij = f_UECM(m.xᵣ[i_red] * m.xᵣ[j], m.yᵣ[i_red] * m.yᵣ[j])
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
    degree(m::UECM[, v]; method=:reduced)

Return a vector corresponding to the expected degree of each node of the UECM model `m`. If `v` is specified, only
return degrees for nodes in `v`.
"""
degree(m::UECM, v::Vector{Int}=collect(1:m.status[:N]); method::Symbol=:reduced) = [degree(m, i, method=method) for i in v]


"""
    strength(m::UECM, i::Int; method=:reduced)

Return the expected (unconditional) strength for node `i` of the UECM model `m`, i.e. `Σⱼ pᵢⱼ/(1 - yᵢyⱼ)`.

# Arguments
- `method::Symbol`:
    - `:reduced` (default) uses the reduced model parameters `xᵣ`/`yᵣ`.
    - `:full` uses all node pairs.
    - `:adjacency` reuses the precomputed adjacency matrix `m.Ĝ` (plus the `yᵣ` parameters).
"""
function strength(m::UECM, i::Int; method::Symbol=:reduced)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    i > m.status[:N] ? throw(ArgumentError("Attempted to access node $i in a $(m.status[:N]) node graph")) : nothing

    if method == :reduced
        res = zero(precision(m))
        i_red = m.dᵣ_ind[i]
        for j in eachindex(m.xᵣ)
            @inbounds yiyj = m.yᵣ[i_red] * m.yᵣ[j]
            @inbounds wij  = f_UECM(m.xᵣ[i_red] * m.xᵣ[j], yiyj) / (1 - yiyj)
            if i_red ≠ j
                @inbounds res += wij * m.f[j]
            else
                @inbounds res += wij * (m.f[j] - 1)
            end
        end
    elseif method == :full
        res = zero(precision(m))
        for j in eachindex(m.d)
            if i ≠ j
                @inbounds yiyj = m.yᵣ[m.dᵣ_ind[i]] * m.yᵣ[m.dᵣ_ind[j]]
                res += A(m, i, j) / (1 - yiyj)
            end
        end
    elseif method == :adjacency
        m.status[:G_computed] ? nothing : throw(ArgumentError("The adjacency matrix has not been computed yet"))
        y = m.yᵣ[m.dᵣ_ind]
        res = zero(precision(m))
        for j in 1:m.status[:N]
            if i ≠ j
                @inbounds res += m.Ĝ[i,j] / (1 - y[i] * y[j])
            end
        end
    else
        throw(ArgumentError("Unknown method $method"))
    end

    return res
end

"""
    strength(m::UECM[, v]; method=:reduced)

Return a vector corresponding to the expected strength of each node of the UECM model `m`. If `v` is specified, only
return strengths for nodes in `v`.
"""
strength(m::UECM, v::Vector{Int}=collect(1:m.status[:N]); method::Symbol=:reduced) = [strength(m, i, method=method) for i in v]


"""
    AIC(m::UECM)

Compute the Akaike Information Criterion (AIC) for the UECM model `m`. The parameters of the model must be computed
beforehand. The UECM has `2N` parameters (an ``\\alpha`` and a ``\\beta`` per node).

See also [`AICc`](@ref MaxEntropyGraphs.AICc), [`L_UECM_reduced`](@ref MaxEntropyGraphs.L_UECM_reduced).
"""
function AIC(m::UECM)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    # compute AIC components
    k = 2 * m.status[:N] # number of parameters (α and β per node)
    n = (m.status[:N] - 1) * m.status[:N] / 2 # number of observations (node pairs)
    L = L_UECM_reduced(m) # log-likelihood

    if n/k < 40
        @warn """The number of observations is small with respect to the number of parameters (n/k < 40). Consider using the corrected AIC (AICc) instead."""
    end

    return 2*k - 2*L
end


"""
    AICc(m::UECM)

Compute the corrected Akaike Information Criterion (AICc) for the UECM model `m`.

See also [`AIC`](@ref MaxEntropyGraphs.AIC), [`L_UECM_reduced`](@ref MaxEntropyGraphs.L_UECM_reduced).
"""
function AICc(m::UECM)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    # compute AIC components
    k = 2 * m.status[:N] # number of parameters
    n = (m.status[:N] - 1) * m.status[:N] / 2 # number of observations
    L = L_UECM_reduced(m) # log-likelihood

    return 2*k - 2*L + (2*k*(k+1)) / (n - k - 1)
end


"""
    BIC(m::UECM)

Compute the Bayesian Information Criterion (BIC) for the UECM model `m`.

See also [`AIC`](@ref MaxEntropyGraphs.AIC), [`L_UECM_reduced`](@ref MaxEntropyGraphs.L_UECM_reduced).
"""
function BIC(m::UECM)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    # compute AIC components
    k = 2 * m.status[:N] # number of parameters
    n = (m.status[:N] - 1) * m.status[:N] / 2 # number of observations
    L = L_UECM_reduced(m) # log-likelihood

    return k * log(n) - 2*L
end


"""
    rand(m::UECM; precomputed=false, rng=Random.default_rng())

Generate a random weighted graph from the UECM model `m`.

# Arguments
- `precomputed::Bool`: not implemented yet for the UECM (the parameters are always used to generate the graph on the fly).
- `rng::AbstractRNG`: random number generator to use (defaults to `Random.default_rng()`).

# Examples
```jldoctest
julia> model = UECM(MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(MaxEntropyGraphs.rhesus_macaques())); # generate a UECM model

julia> solve_model!(model, method=:BFGS); # compute the maximum likelihood parameters

julia> sample = rand(model); # sample a random weighted graph

julia> typeof(sample)
SimpleWeightedGraphs.SimpleWeightedGraph{Int64, Int64}
```
"""
function rand(m::UECM; precomputed::Bool=false, rng::AbstractRNG=default_rng())
    if precomputed
        throw(ArgumentError("This function is not implemented yet for UECM models"))
    else
        # check if possible to use parameters
        m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
        # generate x and y vectors
        x = m.xᵣ[m.dᵣ_ind]
        y = m.yᵣ[m.dᵣ_ind]
        # generate random graph edges
        sources = Vector{Int}();
        targets = Vector{Int}();
        weights = Vector{Int}();
        for i in 1:m.status[:N]
            for j in 1:i-1
                # check if edge exists
                @inbounds xixj = x[i]*x[j]
                @inbounds yiyj = y[i]*y[j]
                p_ij = (xixj * yiyj) / (1 - yiyj + xixj * yiyj)
                if rand(rng) ≤ p_ij
                    push!(sources, i)
                    push!(targets, j)
                    # weight = 1 + Geometric(1 - yᵢyⱼ); mean excess weight = yᵢyⱼ/(1 - yᵢyⱼ)
                    push!(weights, rand(rng, Geometric(1 - yiyj)) + 1)
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
    rand(m::UECM, n::Int; precomputed=false, rng=Random.default_rng())

Generate `n` random weighted graphs from the UECM model `m`. If multithreading is available, the graphs are generated
in parallel; per-sample seeds are drawn from `rng` so the result is reproducible and independent of the thread schedule.
"""
function rand(m::UECM, n::Int; precomputed::Bool=false, rng::AbstractRNG=default_rng())
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



# The UECM feasible region is `yᵢyⱼ < 1` (`βᵢ + βⱼ > 0`); outside it the likelihood is not
# defined (it evaluates to `NaN`). The default HagerZhang / (Strong)Wolfe line searches cannot
# cope with that barrier and stall almost immediately, whereas a BackTracking line search (halve
# the step until the objective is finite and satisfies the Armijo condition) stays in the feasible
# interior and converges — this is exactly the backtracking recipe of Vallarano et al. (2021).
# We therefore give the UECM its own optimizer instances (the other models keep the package-wide
# `optimization_methods`, which work well for their unconstrained domain).
const UECM_optimization_methods = Dict( :LBFGS  => OptimizationOptimJL.LBFGS( linesearch = OptimizationOptimJL.Optim.LineSearches.BackTracking()),
                                        :BFGS   => OptimizationOptimJL.BFGS(  linesearch = OptimizationOptimJL.Optim.LineSearches.BackTracking()),
                                        :Newton => OptimizationOptimJL.Newton(linesearch = OptimizationOptimJL.Optim.LineSearches.BackTracking()))

"""
    solve_model!(m::UECM; kwargs...)

Compute the likelihood maximising parameters of the UECM model `m`.

By default the parameters are computed using the BFGS method with the strength sequence as initial guess.

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
function solve_model!(m::UECM;  # common settings
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
    N <: Union{Float16, Float32} && @warn "Solving in $(N) precision is experimental and may not converge; low precision is intended for storage. Consider Float64 for the solve." maxlog=1
    # `ftol` is accepted on every path but only ever reaches the fixed point solver: say so rather than
    # ignoring it silently (only when it was actually passed, so a default solve stays quiet)
    method ≠ :fixedpoint && !isnothing(ftol) && @warn _ftol_unused_msg(method) maxlog=1
    ftol = isnothing(ftol) ? _DEFAULT_FTOL : ftol
    # initial guess
    θ₀ = initial_guess(m, method=initial)
    # find Inf values (zero-degree/zero-strength nodes)
    ind_inf = findall(isinf, θ₀)
    if method==:fixedpoint
        @warn "The fixed point method is very unstable for this model and should not be used. `BFGS` is prefered for quasinewton methods."
        # initiate buffers
        x_buffer = zeros(N, length(m.dᵣ)); # buffer for x = exp(-α)
        y_buffer = zeros(N, length(m.sᵣ));  # buffer for y = exp(-β)
        G_buffer = zeros(N, length(m.θᵣ)); # buffer for G(x)
        # define fixed point function
        FP_model! = (θ::Vector) -> UECM_reduced_iter!(θ, m.dᵣ, m.sᵣ, m.f, x_buffer, y_buffer, G_buffer, m.nz, length(m.dᵣ));
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
            x_buffer = zeros(N, length(m.dᵣ)); # buffer for x = exp(-α)
            y_buffer = zeros(N, length(m.sᵣ));  # buffer for y = exp(-β)
            # define gradient function for optimisation.jl
            grad! = (G, θ, p) -> ∇L_UECM_reduced_minus!(G, θ, m.dᵣ, m.sᵣ, m.f, x_buffer, y_buffer, length(m.dᵣ));
        end
        # define objective function and its AD method
        f = AD_method ∈ keys(AD_methods) ? Optimization.OptimizationFunction( (θ, p) -> - L_UECM_reduced(θ, m.dᵣ, m.sᵣ, m.f, length(m.dᵣ)),
                                                                                        AD_methods[AD_method],
                                                                                        grad = analytical_gradient ? grad! : nothing)                      : throw(ArgumentError("The AD method $(AD_method) is not supported (yet)"))

        prob = Optimization.OptimizationProblem(f, θ₀);
        # obtain solution
        method ∈ keys(optimization_methods) || throw(ArgumentError("The method $(method) is not supported (yet)"))
        # use the BackTracking-line-search variants (see `UECM_optimization_methods` above), falling
        # back to the package-wide optimizer for any method without a BackTracking variant.
        opt = get(UECM_optimization_methods, method, optimization_methods[method])
        # `maxiters` is forwarded (it was previously silently ignored); `g_tol` (when set) maps to
        # Optim's gradient tolerance so the solve can stop before over-converging.
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
