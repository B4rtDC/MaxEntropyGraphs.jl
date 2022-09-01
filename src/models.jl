"""
Idea: starting from models with known parameters:
- obtain expected values and variances for adjacency/weight matrix elements
- sample networks, returning 
    1. Adjacency matrix (dense/ sparse (ULT)) 
    2. Graph 
    3. Adjacency List & node number
- compute z-scores of different metrics by 
    1. "exact" method 
    2. sampling method
"""
#= run this once at startup
if Sys.islinux()
    ENV["GRDIR"] = "" # for headless plotting
    using Pkg; Pkg.build("GR")
    # sudo apt install xvfb
    # https://gr-framework.org/julia.html#installation
    import GR:inline
    GR.inline("pdf")
    GR.inline("png")
end
=#


# ----------------------------------------------------------------------------------------------------------------------
#
#                                               General model
#
# ----------------------------------------------------------------------------------------------------------------------

"""
    AbstractMaxEntropyModel

An abstract type for a MaxEntropyModel. Each model has one or more structural constraints  
that are fixed while the rest of the network is completely random. 

The different functions below should be implemented in the subtypes.
"""
abstract type AbstractMaxEntropyModel end

"""
    σ(::AbstractMaxEntropyModel)

Compute variance for elements of the adjacency matrix for the specific `AbstractMaxEntropyModel` based on the ML parameters.
"""
σ(::AbstractMaxEntropyModel) = nothing


"""
    Ĝ(::AbstractMaxEntropyModel)

Compute expected adjacency and/or weight matrix for a given `AbstractMaxEntropyModel`
"""
Ĝ(::AbstractMaxEntropyModel) = nothing


"""
    rand(::AbstractMaxEntropyModel)

Sample a random network from the `AbstractMaxEntropyModel`
"""
Base.rand(::AbstractMaxEntropyModel) = nothing

"""
    ∇X(X::Function, M::T)

Compute the gradient of a property `X` with respect to the expected adjacency matrix associated with the model `M`.
"""
∇X(X::Function, M::T) where T <: AbstractMaxEntropyModel = ReverseDiff.gradient(X, M.G)


"""
    σˣ(X::Function, M::T)

Compute the standard deviation of a property `X` with respect to the expected adjacency matrix associated with the model `M`.
"""
σˣ(X::Function, M::T) where T <: AbstractMaxEntropyModel = nothing

# ----------------------------------------------------------------------------------------------------------------------
#
#                                               UBCM model
#
# ----------------------------------------------------------------------------------------------------------------------

"""
    UBCM

Maximum entropy model for the Undirected Binary Configuration Model (UBCM). 
    
The object holds the maximum likelihood parameters of the model (x), the expected adjacency matrix (G), 
and the variance for the elements of the adjacency matrix (σ).

"""
struct UBCM{T} <: AbstractMaxEntropyModel where {T<:Real}
    x::Vector{T}
    G::Matrix{T}
    σ::Matrix{T}
end

Base.show(io::IO, m::UBCM{T}) where T = print(io, "$(T) UBCM model ($(length(m)) vertices)")

"""Return the number of nodes in the UBCM network"""
Base.length(m::UBCM) = length(m.x)

"""
    UBCM(x::Vector{T}; compute::Bool=true) where {T<:Real}

Constructor for the `UBCM` type.
"""
function UBCM(x::Vector{T}) where {T<:Real}
    G = Ĝ(x, UBCM{T})  # expected adjacency matrix
    σ = σˣ(x, UBCM{T}) # expected standard deviation matrix

    return UBCM(x, G, σ)
end

"""
    UBCM(G::T) where T<:SimpleGraph

Constructor for the `UBCM` type based on a `SimpleGraph`. 
"""
function UBCM(G::T; method="fixed-point", initial_guess="degrees", max_steps=5000, tol=1e-12, kwargs...) where T<:Graphs.SimpleGraph
    NP = PyCall.pyimport("NEMtropy")
    G_nem = NP.UndirectedGraph(degree_sequence=Graphs.degree(G))
    G_nem.solve_tool(model="cm_exp", method=method, initial_guess=initial_guess, max_steps=max_steps, tol=tol, kwargs...);
    if G_nem.error > 1e-7
        @warn "The model did not converge, maybe try some other options (solution error $(G_nem.error))"
    end
    return UBCM(G_nem.x)
end

"""
    Ĝ(::UBCM, x::Vector{T}) where {T<:Real}

Compute the expected adjacency matrix for the UBCM model with maximum likelihood parameters `x`.
"""
function Ĝ(x::Vector{T}, ::Type{UBCM{T}}) where T
    n = length(x)
    G = zeros(T, n, n)
    for i = 1:n
        @simd for j = i+1:n
            @inbounds xij = x[i]*x[j]
            @inbounds G[i,j] = xij/(1 + xij)
            @inbounds G[j,i] = xij/(1 + xij)
        end
    end
    
    return G
end

"""
    σˣ(x::Vector{T}, ::Type{UBCM{T}}) where T

Compute the standard deviation for the elements of the adjacency matrix for the UBCM model using the maximum likelihood parameters `x`.

**Note:** read as "sigma star"
"""
function σˣ(x::Vector{T}, ::Type{UBCM{T}}) where T
    n = length(x)
    res = zeros(T, n, n)
    for i = 1:n
        @simd for j = i+1:n
            @inbounds xij =  x[i]*x[j]
            @inbounds res[i,j] = sqrt(xij)/(1 + xij)
            @inbounds res[j,i] = sqrt(xij)/(1 + xij)
        end
    end

    return res
end

"""
    rand(m::UBCM)

Generate a random graph from the UBCM model. The function returns a `Graphs.AbstractGraph` object.
"""
function Base.rand(m::UBCM)
    n = length(m)
    g = Graphs.SimpleGraph(n)
    for i = 1:n
        for j = i+1:n
            if rand() < m.G[i,j]
                Graphs.add_edge!(g, i, j)
            end
        end
    end

    return g
end


"""
    σˣ(X::Function, M::UBCM{T})

Compute the standard deviation of a property `X` with respect to the expected adjacency matrix associated with the UBCM model `M`.
"""
σˣ(X::Function, M::UBCM{T}) where T = sqrt( sum((M.σ .* ∇X(X, M)) .^ 2) )



# ----------------------------------------------------------------------------------------------------------------------
#
#                                               DBCM model
#
# ----------------------------------------------------------------------------------------------------------------------

"""
    DBCM

Maximum entropy model for the Directed Binary Configuration Model (DBCM). 
    
The object holds the maximum likelihood parameters of the model (x, y), the expected adjacency matrix (G), 
and the variance for the elements of the adjacency matrix (σ).

"""
struct DBCM{T} <: AbstractMaxEntropyModel where {T<:Real}
    x::Vector{T}
    y::Vector{T}
    G::Matrix{T}
    σ::Matrix{T}
end

Base.show(io::IO, m::DBCM{T}) where T = print(io, "$(T) DBCM model ($(length(m)) vertices)")

"""Return the number of nodes in the DBCM network"""
Base.length(m::DBCM) = length(m.x)

"""
    DBCM(x::Vector{T}, y::Vector{T}; compute::Bool=true) where {T<:Real}

Constructor for the `DBCM` type. If `compute` is true, the expected adjacency matrix and variance are computed. 
Otherwise the memory is allocated but not initialized. (TBC)
"""
function DBCM(x::Vector{T}, y::Vector{T}; compute::Bool=true) where {T<:Real}
    G = Ĝ( x, y, DBCM{T}) # expected adjacency matrix
    σ = σˣ(x, y, DBCM{T}) # expected standard deviation matrix

    return DBCM(x, y, G, σ)
end

"""
    DBCM(G::T) where T<:SimpleDiGraph

Constructor for the `DBCM` type based on a `SimpleDiGraph`. 
"""
function DBCM(G::T; method="fixed-point", initial_guess="degrees", max_steps=5000, tol=1e-12, kwargs...) where T<:Graphs.SimpleDiGraph
    NP = PyCall.pyimport("NEMtropy")
    G_nem =  NP.DirectedGraph(degree_sequence=vcat(Graphs.outdegree(G), Graphs.indegree(G)))
    G_nem.solve_tool(model="dcm_exp"; method=method, initial_guess=initial_guess, max_steps=max_steps, tol=tol, kwargs...);
    if G_nem.error > 1e-7
        @warn "The model did not converge, maybe try some other options (solution error $(G_nem.error))"
    end
    return DBCM(G_nem.x, G_nem.y)
end

"""
    Ĝ(x::Vector{T}, y::Vector{T}, ::Type{DBCM{T}}) where {T<:Real}

Compute the expected adjacency matrix for the `DBCM` model with maximum likelihood parameters `x` and `y`.
"""
function Ĝ(x::Vector{T}, y::Vector{T}, ::Type{DBCM{T}}) where T
    n = length(x)
    G = zeros(T, n, n)
    for i = 1:n
        @simd for j = i+1:n
            @inbounds xiyj = x[i]*y[j]
            @inbounds xjyi = x[j]*y[i]
            @inbounds G[i,j] = xiyj/(1 + xiyj)
            @inbounds G[j,i] = xjyi/(1 + xjyi)
        end
    end
    
    return G
end

"""
    σˣ(x::Vector{T}, y::Vector{T}, ::Type{DBCM{T}}) where T

Compute the standard deviation for the elements of the adjacency matrix for the `DBCM` model using the maximum likelihood parameters `x` and `y`.

**Note:** read as "sigma star"
"""
function σˣ(x::Vector{T}, y::Vector{T}, ::Type{DBCM{T}}) where T
    n = length(x)
    res = zeros(T, n, n)
    for i = 1:n
        @simd for j = i+1:n
            @inbounds xiyj =  x[i]*y[j]
            @inbounds xjyi =  x[j]*y[i]
            @inbounds res[i,j] = sqrt(xiyj)/(1 + xiyj)
            @inbounds res[j,i] = sqrt(xjyi)/(1 + xjyi)
        end
    end

    return res
end

"""
    rand(m::DBCM)

Generate a random graph from the `DBCM` model. The function returns a `Graphs.AbstractGraph` object.
"""
function Base.rand(m::DBCM)
    n = length(m)
    g = Graphs.SimpleDiGraph(n)
    for i = 1:n
        for j = i+1:n
            if rand() < m.G[i,j]
                Graphs.add_edge!(g, i, j)
            end
            if rand() < m.G[j,i]
                Graphs.add_edge!(g, j, i)
            end
        end
    end

    return g
end

"""
    σˣ(X::Function, M::DBCM{T})

Compute the standard deviation of a property `X` with respect to the expected adjacency matrix associated with the `DBCM` model `M`.
"""
σˣ(X::Function, M::DBCM{T}) where T = sqrt( sum((M.σ .* ∇X(X, M)) .^ 2) )

