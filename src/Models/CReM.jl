"""
    CReM

Maximum entropy model for the Conditional Reconstruction Model (CReM). 
    
The object holds the maximum likelihood parameters of the model (θ) and those required for the conditional reconstruction (α / αᵣ).
Weight are continuous and positive, and the model is undirected.

This is a two-step process: first we consider the marginal probability of an edge existing between nodes using the UBCM model, 
then we use this in for the CReM model.
"""
mutable struct CReM{T,N} <: AbstractMaxEntropyModel where {T<:Union{Graphs.AbstractGraph, Nothing}, N<:Real}
    "Graph type, can be any subtype of AbstractGraph, but will be converted to SimpleGraph for the computation" # can also be empty
    const G::T 
    "Maximum likelihood parameters for CReM model"
    const Θ::Vector{N}
    "Maximum likelihood parameters for conditional model"
    const α::Vector{N} 
    "Maximum likelihood parameters for reduced conditional model"
    const αᵣ::Vector{N} 
    "Exponentiated maximum likelihood parameters for reduced model ( xᵢ = exp(-θᵢ) )"
    const x::Vector{N}
    "Degree sequence of the graph" # evaluate usefulness of this field later on
    const d::Vector{Int}
    "Reduced degree sequence of the graph"
    const dᵣ::Vector{Int}
    "Frequency of each degree in the degree sequence"
    const f::Vector{Int}
    "Strength sequence of the graph"
    const s::Vector{N}
    "Indices to reconstruct the degree sequence from the reduced degree sequence"
    const d_ind::Vector{Int}
    "Indices to reconstruct the reduced degree sequence from the degree sequence"
    const dᵣ_ind::Vector{Int}
    "Expected adjacency matrix" # not always computed/required
    Ĝ::Union{Nothing, Matrix{N}}
    "Expected weighted adjacency matrix" # not always computed/required
    Ĝw::Union{Nothing, Matrix{N}}
    "Status indicators: parameters computed, expected adjacency matrix computed, variance computed, etc."
    const status::Dict{Symbol, Any}
end

Base.show(io::IO, m::CReM{T,N}) where {T,N} = print(io,  """CReM{$(T), $(N)} ($(m.status[:d]) vertices)""")

Base.length(m::CReM) = length(m.d)

function CReM(G::T; d::Vector=Graphs.degree(G), s::Vector=strength(G), precision::Type{<:AbstractFloat}=Float64, kwargs...) where {T}
    T <: Union{Graphs.AbstractGraph, Nothing} ? nothing : throw(TypeError("G must be a subtype of AbstractGraph or Nothing"))

    # coherence checks
    if T <: Graphs.AbstractGraph # Graph specific checks
        if Graphs.is_directed(G)
            @warn "The graph is directed, the CReM model is undirected, the directional information will be lost"
        end

        if zero(eltype(d)) ∈ d
            @warn "The graph has vertices with zero degree, this may lead to convergence issues."
        end

        Graphs.nv(G) == 0 ? throw(ArgumentError("The graph is empty")) : nothing
        Graphs.nv(G) == 1 ? throw(ArgumentError("The graph has only one vertex")) : nothing

        Graphs.nv(G) != length(d) ? throw(DimensionMismatch("The number of vertices in the graph ($(Graphs.nv(G))) and the length of the degree sequence ($(length(d))) do not match")) : nothing
        Graphs.nv(G) != length(s) ? throw(DimensionMismatch("The number of vertices in the graph ($(Graphs.nv(G))) and the length of the strength sequence ($(length(s))) do not match")) : nothing
    end
    # coherence checks specific to the degree/strength sequence
    length(d) == 0 ? throw(ArgumentError("The degree sequence is empty")) : nothing
    length(d) == 1 ? throw(ArgumentError("The degree sequence has only one degree")) : nothing
    maximum(d) >= length(d) ? throw(DomainError("The maximum degree in the graph is greater or equal to the number of vertices, this is not allowed")) : nothing
    all(d .>= 0) ? nothing : throw(DomainError("The degree sequence contains negative degrees"))
    all(s .>= 0) ? nothing : throw(DomainError("The strength sequence contains negative strengths"))

    # field generation
    dᵣ, d_ind , dᵣ_ind, f = np_unique_clone(d, sorted=true)
    Θ = Vector{precision}(undef, length(d))     # CReM parameters
    α = Vector{precision}(undef, length(d))     # UBCM parameters
    αᵣ = Vector{precision}(undef, length(dᵣ))   # UBCM reduced parameters
    x = Vector{precision}(undef, length(dᵣ))    # exponentiated UBCM parameters 
    status = Dict(  :conditional_params_computed=>false,    # are the conditional parameters computed?
                    :params_computed=>false,                # are the parameters computed?
                    :G_computed=>false,                     # is the expected adjacency matrix computed and stored?
                    :Gw_computed=>false,                    # is the expected weighted adjacency matrix computed and stored?
                    :cᵣ => length(dᵣ)/length(d),            # compression ratio of the reduced conditional model
                    :d_unique => length(dᵣ),                # number of unique degrees in the reduced model
                    :d => length(d)                         # number of vertices in the original graph 
                )

    return CReM{T, precision}(G, Θ, α, αᵣ, x, d, dᵣ, f, s, d_ind, dᵣ_ind, nothing, nothing, status)
end

CReM(; d::Vector{T}, s::Vector{S}, precision::Type{<:AbstractFloat}=Float64, kwargs...) where {T<:Signed, S<:Real} = UBCM(nothing, d=d, s=s, precision=precision, kwargs...)

"""
    L_CReM(θ::Vector, s::Vector, f::Vector)

Compute the loglikelihood of the CReM model given the parameters θ, the strength sequence s and conditional parameters f.

This is the function this is used when the marginal probability of an edge existing between nodes using the UBCM model is not precomputed (i.e. `model.Ĝ` is `nothing`)
"""
function L_CReM(θ::Vector, s::Vector, f::Vector)
    # initialise result
    res = zero(eltype(θ))
    for i in eachindex(θ)
        res -= θ[i] * s[i]
        for j in 1:i-1
            # expected conditional probability of an edge between i and j
            fij = f[i]*f[j] / (1 + f[i]*f[j])
            # update result
            res += fij * log(θ[i] + θ[j])
        end
    end
    
    return res
end


"""
    L_CReM(θ::Vector, s::Vector, f::Matrix)

Compute the loglikelihood of the CReM model given the parameters θ, the strength sequence s and conditional parameters f.

This is the function this is used when the marginal probability of an edge existing between nodes using the UBCM model is precomputed (i.e. `model.Ĝ` is `Matrix`)
"""
function L_CReM(θ::Vector, s::Vector, f::Matrix)
    # initialise result
    res = zero(eltype(θ))
    for i in eachindex(θ)
        res -= θ[i] * s[i]
        for j in 1:i-1
            # update result
            res += f[i,j] * log(θ[i] + θ[j])
        end
    end
    
    return res
end


"""
L_CReM(m::CReM)

Compute the loglikelihood of the CReM model `m`. Depending on the status of the model, the function will use the precomputed marginal probability matrix 
of an edge existing between nodes or compute the marginal probability on the fly.
"""
function L_CReM(m::CReM)
    # check if the parameters are computed
    m.status[:conditional_params_computed] ? nothing : throw(ArgumentError("The conditional parameters of the model have not been computed"))
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters of the model have not been computed"))
    if m.status[:G_computed]
        return L_CReM(m.Θ, m.s, m.Ĝ)
    else
        return L_CReM(m.Θ, m.s, m.α)
    end
end


"""
    ∇L_CReM!(∇L::AbstractVector, θ::AbstractVector, s::Vector, f::Vector) 

Compute the gradient of the log-likelihood of the CReM model, computing the marginal probability on the fly.
"""
function ∇L_CReM!(∇L::AbstractVector, θ::AbstractVector, s::Vector, f::Vector) 
    # reset gradient
    ∇L .= zero(eltype(θ))
    # compute
    for i in eachindex(θ)
        ∇L[i] -= s[i]
        for j in eachindex(θ)
            if i≠j
                fij = f[i]*f[j] / (1 + f[i]*f[j])
                ∇L[i] += fij / (θ[i] + θ[j])
            end
        end
    end

    return ∇L
end


"""
    ∇L_CReM!(∇L::AbstractVector, θ::AbstractVector, s::Vector, f::Maxtrix) 

Compute the gradient of the log-likelihood of the CReM model, using the precomputed the marginal probability.
"""
function ∇L_CReM!(∇L::AbstractVector, θ::AbstractVector, s::Vector, f::Matrix) 
    # reset gradient
    ∇L .= zero(eltype(θ))
    # compute
    for i in eachindex(θ)
        ∇L[i] -= s[i]
        for j in eachindex(θ)
            if i≠j
                ∇L[i] += f[i,j] / (θ[i] + θ[j])
            end
        end
    end

    return ∇L
end


"""
    ∇L_CReM!(∇L::AbstractVector, θ::AbstractVector, s::Vector, f::Vector) 

Compute minus the gradient of the log-likelihood of the CReM model, computing the marginal probability on the fly. Used for optimisation in a non-allocating manner.
"""
function ∇L_CReM_minus!(∇L::AbstractVector, θ::AbstractVector, s::Vector, f::Vector) 
    # reset gradient
    ∇L .= zero(eltype(θ))
    # compute
    for i in eachindex(θ)
        ∇L[i] += s[i]
        for j in eachindex(θ)
            if i≠j
                fij = f[i]*f[j] / (1 + f[i]*f[j])
                ∇L[i] -= fij / (θ[i] + θ[j])
            end
        end
    end

    return ∇L
end


"""
    ∇L_CReM!(∇L::AbstractVector, θ::AbstractVector, s::Vector, f::Maxtrix) 

Compute minus the gradient of the log-likelihood of the CReM model, using the precomputed the marginal probability. Used for optimisation in a non-allocating manner.
"""
function ∇L_CReM_minus!(∇L::AbstractVector, θ::AbstractVector, s::Vector, f::Matrix) 
    # reset gradient
    ∇L .= zero(eltype(θ))
    # compute
    for i in eachindex(θ)
        ∇L[i] += s[i]
        for j in eachindex(θ)
            if i≠j
                ∇L[i] -= f[i,j] / (θ[i] + θ[j])
            end
        end
    end

    return ∇L
end


"""
    CReM_iter!(θ::AbstractVector, s::Vector, f::Vector, G::AbstractVector)

Computer the next fixed-point iteration for the CReM model, computing the marginal probability on the fly.
The function will update pre-allocated vector (`G`) for speed.
"""
function CReM_iter!(θ::AbstractVector, s::Vector, f::Vector, G::AbstractVector)
    # reset buffer
    G .= zero(eltype(θ))
    # compute
    for i in eachindex(θ)
        for j in eachindex(θ)
            if i≠j
                fij = f[i]*f[j] / (1 + f[i]*f[j])
                G[i] += fij / (1 + θ[j] / θ[i])
            end
        end
        G[i] = G[i] / s[i]
    end

    return G
end


"""
    CReM_iter!(θ::AbstractVector, s::Vector, f::Matrix, G::AbstractVector)

Computer the next fixed-point iteration for the CReM model, using the precomputed the marginal probability.
The function will update pre-allocated vector (`G`) for speed.
"""
function CReM_iter!(θ::AbstractVector, s::Vector, f::Matrix, G::AbstractVector)
    # reset buffer
    G .= zero(eltype(θ))
    # compute
    for i in eachindex(θ)
        for j in eachindex(θ)
            if i≠j
                G[i] += f[i,j] / (1 + θ[j] / θ[i])
            end
        end
        G[i] = G[i] / s[i]
    end

    return G
end


"""
    initial_guess(m::CReM, method::Symbol=:degrees)

Compute an initial guess for the maximum likelihood parameters of the CReM model `m` using the method `method`.

The methods available are: `:strengths` (default), `:strengths_minor`, `:random`.
"""
function initial_guess(m::CReM{T,N}; method::Symbol=:degrees) where {T,N}
    if isequal(method, :strengths)
        return Vector{N}(-log.(m.s ./ sqrt(2*sum(m.s))))
    elseif isequal(method, :strengths_minor)
        return Vector{N}(-log.(ones(N, length(m.s)) ./ ( m.s .+ 1)))
    elseif isequal(method, :random)
        return Vector{N}(-log.(rand(N, length(m.s))))
    else
        throw(ArgumentError("The initial guess method $(method) is not supported"))
    end
end



function rand(m::CReM; precomputed::Bool=false)
    if precomputed
        throw(ArgumentError("The precomputed option is not supported yet for the CReM model"))
        return
    else
        # check if possible
        m.status[:conditional_params_computed] ? nothing : throw(ArgumentError("The conditional parameters of the model have not been computed"))
        m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters of the model have not been computed"))
        # initialise
        sources = Vector{Int}(); 
        targets = Vector{Int}();
        weights = Vector{Int}();
        x = m.x
        θ = m.Θ
        # generate edges
        for i in eachindex(m.θ)
            for j in 1:i-1
                if rand() <= x[i]*x[j] / (1 + x[i]*x[j])
                    push!(sources, i)
                    push!(targets, j)
                    push!(weights, rand(Distributions.exponential(θ[i]+θ[j])))
                end
            end
        end
        
        # build graph
        if length(sources) ≠ 0
            G = SimpleWeightedGraphs.SimpleWeightedGraph(sources, targets, weights)
        else
            G = SimpleWeightedGraphs.SimpleWeightedGraph(length(θ))
        end

        # deal with edge case where no edges are generated for the last node(s) in the graph
        while Graphs.nv(G) < length(θ)
            Graphs.add_vertex!(G)
        end

        return G
    end
end


function rand(m::CReM, n::Int; precomputed::Bool=false)
    # pre-allocate
    res = Vector{SimpleWeightedGraphs.SimpleWeightedGraph}(undef, n)
    # fill vector using threads
    Threads.@threads for i in 1:n
        res[i] = rand(m; precomputed=precomputed)
    end

    return res
end



function solve_model!(m::CReM{T,N}; # related to CReM
                                    initial::Symbol=:strengths,  
                                    method::Symbol=:fixedpoint,
                                    AD_method::Symbol=:AutoZygote,
                                    analytical_gradient::Bool=false,
                                    # related to UBCM
                                    initial_conditional::Symbol=:degrees,
                                    method_conditional::Symbol=:fixedpoint,
                                    AD_method_conditional::Symbol=:AutoZygote,
                                    analytical_gradient_conditional::Bool=false,
                                    # common settings
                                    maxiters::Int=1000, 
                                    verbose::Bool=false,
                                    ftol::Real=1e-8,                        # NLsolve.jl specific settings (fixed point method)
                                    abstol::Union{Number, Nothing}=nothing, # optimisation.jl specific settings
                                    reltol::Union{Number, Nothing}=nothing, 
                                    ) where {T,N}
    ## Part 1 - Conditional UBCM
    cond_model = UBCM(d=m.d, precision=N)
    solve_model!(cond_model,initial=initial_conditional, method=method_conditional, 
                            AD_method=AD_method_conditional, analytical_gradient=analytical_gradient_conditional,
                            ftol=ftol, abstol=abstol, reltol=reltol, maxiters=maxiters, verbose=verbose)
    
    ## Part 2 - CReM compute
    θ₀ = initial_guess(m; method=initial)


    return m
end