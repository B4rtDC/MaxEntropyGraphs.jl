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

import Graphs                   # network interface for graphs
import PyCall                   # for calling NEMtropy package in Python
import ReverseDiff              # for gradient
import Statistics: std, mean    # for statistics
import Printf: @sprintf         # for specific printing
using Plots                     # for plotting
using Measures                  # for margin settings
using LaTeXStrings              # for LaTeX printing
import Dates: now, Day, Minute  # for illustration printing
using GraphIO                   # to read and write external graphs
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

Constructor for the `UBCM` type. If `compute` is true, the expected adjacency matrix and variance are computed. 
Otherwise the memory is allocated but not initialized. (TBC)
"""
function UBCM(x::Vector{T}; compute::Bool=true) where {T<:Real}
    G = Ĝ(x, UBCM{T})  # expected adjacency matrix
    σ = σˣ(x, UBCM{T}) # expected standard deviation matrix

    return UBCM(x, G, σ)
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


# ----------------------------------------------------------------------------------------------------------------------
#
#                                               Supporting network functions
#
# Note: the function working on matrices need to be defined in without contraining the types too much
#       otherwise there will be a problem when using the autodiff package.
# ----------------------------------------------------------------------------------------------------------------------

## Binary networks
# degree metric
degree(A, i::Int)   = sum(@view A[:,i])             # degree of node i
degree(A)           = reshape(sum(A, dims=1), :)    # degree vector for the entire network
degree(m::UBCM, i::Int)     = degree(m.G, i)                # degree of node i
degree(m::UBCM)             = reshape(sum(m.G, dims=1), :)  # degree vector for the entire network
# ANND metric
ANND(G::Graphs.SimpleGraph, i)  = iszero(Graphs.degree(G,i)) ? zero(Float64) : sum(map( n -> Graphs.degree(G,n), Graphs.neighbors(G,i))) / Graphs.degree(G,i)
ANND(G::Graphs.SimpleGraph)     = map(i -> ANND(G,i), 1:Graphs.nv(G))
ANND(A::T, i::Int) where T<: AbstractArray = sum(A[i,j] * degree(A,j) for j=1:size(A,1) if j≠i) / degree(A,i)
ANND(A::T) where T<: AbstractArray         = map(i -> ANND(A,i), 1:size(A,1))
ANND(m::UBCM, i::Int)           = ANND(m.G, i)
ANND(m::UBCM)                   = ANND(m.G)
# motifs
M₁(A::T) where T<: AbstractArray = sum(A[i,j]*A[j,k]*(1 - A[k,i]) for i = axes(A,1) for j=i+1:size(A,1) for k=j+1:size(A,1))   # v-motifs metric
M₁(m::UBCM)                      = M₁(m.G)
M₁(G::Graphs.SimpleGraph)        = M₁(Graphs.adjacency_matrix(G))
M₂(A::T) where T<: AbstractArray = sum(A[i,j]*A[j,k]*A[k,i] for i = axes(A,1) for j=i+1:size(A,1) for k=j+1:size(A,1))         # triangles metric
M₂(m::UBCM)                      = M₂(m.G)
M₂(G::Graphs.SimpleGraph)        = M₂(Graphs.adjacency_matrix(G))

## Directed binary networks
# degree metrics
outdegree(A::T, i::Int) where T<: AbstractArray        = sum(@view A[i,:])             # out-degree of node i
outdegree(A::T)         where T<: AbstractArray       = reshape(sum(A, dims=2), :)    # out-degree vector for the entire network
outdegree(M::DBCM, i::Int)  = outdegree(M.G, i)
outdegree(M::DBCM)          = outdegree(M.G)
indegree(A::T, i::Int) where T<: AbstractArray        = sum(@view A[:,i])             # out-degree of node i
indegree(A::T)         where T<: AbstractArray        = reshape(sum(A, dims=1), :)    # out-degree vector for the entire network
indegree(M::DBCM, i::Int)  = indegree(M.G, i)
indegree(M::DBCM)          = indegree(M.G)
# ANND metric
ANND_in(G::T, i) where T<:Graphs.SimpleGraphs.SimpleDiGraph = iszero(Graphs.indegree(G,i)) ? zero(Float64) : sum(map( n -> Graphs.indegree(G,n), Graphs.inneighbors(G,i))) / Graphs.indegree(G,i)
ANND_in(G::T)       where T<:Graphs.SimpleGraphs.SimpleDiGraph = map(i -> ANND_in(G,i), 1:Graphs.nv(G))
ANND_in(A::T, i::Int) where T<: AbstractArray = sum(A[j,i] * indegree(A,j) for j=1:size(A,1) if j≠i) / indegree(A,i)
ANND_in(A::T) where T<: AbstractArray         = map(i -> ANND_in(A,i), 1:size(A,1))
ANND_in(m::DBCM, i::Int)           = ANND_in(m.G, i)
ANND_in(m::DBCM)                   = ANND_in(m.G)

ANND_out(G::T, i) where T<:Graphs.SimpleGraphs.SimpleDiGraph = iszero(Graphs.outdegree(G,i)) ? zero(Float64) : sum(map( n -> Graphs.outdegree(G,n), Graphs.outneighbors(G,i))) / Graphs.outdegree(G,i)
ANND_out(G::T)       where T<:Graphs.SimpleGraphs.SimpleDiGraph = map(i -> ANND_out(G,i), 1:Graphs.nv(G))
ANND_out(A::T, i::Int) where T<: AbstractArray = sum(A[i,j] * outdegree(A,j) for j=1:size(A,1) if j≠i) / outdegree(A,i)
ANND_out(A::T) where T<: AbstractArray         = map(i -> ANND_out(A,i), 1:size(A,1))
ANND_out(m::DBCM, i::Int)           = ANND_out(m.G, i)
ANND_out(m::DBCM)                   = ANND_out(m.G)
# motifs
# - scaffolding
a⭢(A::T, i::Int, j::Int) where T<:AbstractArray = @inbounds A[i,j] * (one(eltype(T)) - A[j,i])                    # directed link from i to j and not from j to i A[i,j] *A[j,i]#A
a⭠(A::T, i::Int, j::Int) where T<:AbstractArray = @inbounds (one(eltype(T)) - A[i,j]) * A[j,i]                    # directed link from j to i and not from i to j A[i,j] *A[j,i]#
a⭤(A::T, i::Int, j::Int) where T<:AbstractArray = @inbounds A[i,j]*A[j,i]                                         # recipocrated link between i and j
a̲(A::T, i::Int, j::Int)   where T<:AbstractArray = @inbounds (one(eltype(T)) - A[i,j])*(one(eltype(T)) - A[j,i])  # no links between i and j  A[i,j] *A[j,i]#
# - actual motifs (cf. original 2011 paper by Squartini et al. for definitions)
motif_functions = [ (a⭠, a⭢, a̲);
                    (a⭠, a⭠, a̲);
                    (a⭠, a⭤, a̲);
                    (a⭠, a̲, a⭢);
                    (a⭠, a⭢,a⭢);
                    (a⭠, a⭤, a⭢);
                    (a⭢, a⭤, a̲);
                    (a⭤, a⭤, a̲);
                    (a⭢, a⭢, a⭢);
                    (a⭤, a⭢, a⭢);
                    (a⭤, a⭠, a⭢);
                    (a⭤, a⭤, a⭢);
                    (a⭤, a⭤, a⭤);
                    ]
for i = 1:13 # mapping to different functions for adjacency matrix, DBCM model and graph
    fname = Symbol('M' * prod(map(x -> Char(x+48+8272),map(v -> reverse(digits(v)), i))))
    @eval begin
        """
            $($fname)(A::T) where T<:AbstractArray
        
        Compute the motif $($fname) (Σ_{i,j,k} $(motif_functions[$i][1])(i,j) $(motif_functions[$i][2])(j,k) $(motif_functions[$i][3])(k,i) ) from the adjacency matrix.
        """
        function $(fname)(A::T)  where T<:AbstractArray
            res = zero(eltype(A))
            for i = axes(A,1)
                for j = axes(A,1)
                    @simd for k = axes(A,1)
                        if i ≠ j && j ≠ k && k ≠ i
                            res += $(motif_functions[i][1])(A,i,j) * $(motif_functions[i][2])(A,j,k) *   $(motif_functions[i][3])(A,k,i)
                        end
                    end
                end
            end
            return res
        end

        """
            $($fname)(M::DBCM)
        
        Compute the motif $($fname) (Σ_{i,j,k} $(motif_functions[$i][1])(i,j) $(motif_functions[$i][2])(j,k) $(motif_functions[$i][3])(k,i) ) from the `DBCM` model.
        """
        $(fname)(M::DBCM) = $(fname)(M.G)

        """
            $($fname)(G::SimpleDiGraph)
        
        Compute the motif $($fname) (Σ_{i,j,k} $(motif_functions[$i][1])(i,j) $(motif_functions[$i][2])(j,k) $(motif_functions[$i][3])(k,i) ) from the `SimpleDiGraph`.
        """
        $(fname)(G::Graphs.SimpleDiGraph) = $(fname)(Graphs.adjacency_matrix(G))
    end
end

const DBCM_motif_functions = eval.([Symbol('M' * prod(map(x -> Char(x+48+8272),map(v -> reverse(digits(v)), i)))) for i = 1:13])

"""
    motifs(M::DBCM, n::Int...)

Compute the number of occurrences of motif `n` in the `DBCM` model. If no `n` is given, compute the number of occurrences of all motifs.


# Examples
```julia-repl
julia> motifs(model, 13)
[37]

julia> motifs(model, 1,2,3)
[36; 1; 19]

julia> motifs(model, 1:13...)
[36;  1;  19;  24;  13;  14;  32;  44;  16;  3;  36;  26;  37]
```
"""
function motifs(M::DBCM, n::Int...)
    iszero(length(n)) && return nothing
    # generate function names
    fnames = [Symbol('M' * prod(map(x -> Char(x+48+8272),map(v -> reverse(digits(v)), i)))) for i in n]
    # apply function
    eval.(map(f -> :($(f)($M.G)), fnames))
end

motifs(M::DBCM) = motifs(M, 1:13...)

"""
    motifs(G::SimpleDiGraph, n::Int...; full::Bool=false))

Compute the number of occurrences of motif `n` in the `SimpleDiGraph`. If no `n` is given, compute the number of occurrences of all motifs.
The keyword `full` allows you to choose between using a sparse or dense representation for the adjacency matrix. For small networks, a full representation is faster.

# Examples
```julia-repl
julia> motifs(G, 13)
[37]

julia> motifs(G, 1,2,3)
[36; 1; 19]

julia> motifs(G, 1:13...)
[36;  1;  19;  24;  13;  14;  32;  44;  16;  3;  36;  26;  37]
```
"""
function motifs(G::Graphs.SimpleDiGraph, n::Int...; full::Bool=false)
    iszero(length(n)) && return nothing
    # generate function names
    fnames = [Symbol('M' * prod(map(x -> Char(x+48+8272),map(v -> reverse(digits(v)), i)))) for i in n]
    # generate adjacency matrix
    A = full ? Array(Graphs.adjacency_matrix(G)) : Graphs.adjacency_matrix(G)
    # apply function
    res = Vector{Int64}(undef, length(n)) # fixed type for performance reasons (x35 faster)
    for i = 1:length(n)
        res[i] = eval(:($(fnames[i])($A)))
    end
    #eval.(map(f -> :($(f)($A)), fnames))
    return res
end

motifs(G::Graphs.SimpleDiGraph; full::Bool=false) = motifs(G, 1:13...; full=full)

# ----------------------------------------------------------------------------------------------------------------------
#
#                                               model testing
#
# ----------------------------------------------------------------------------------------------------------------------

function UBCM_test()
    NP = PyCall.pyimport("NEMtropy")

    mygraph = Graphs.smallgraph(:karate) # quick demo - Zachary karate club
    mygraph_nem = NP.UndirectedGraph(degree_sequence=Graphs.degree(mygraph))
    mygraph_nem.solve_tool(model="cm", method="fixed-point", initial_guess="random");

    M = UBCM(mygraph_nem.x)         # generate the model
    S = [rand(M) for _ in 1:10000]; # sample 10000 graphs from the model

    ## degree metric testing
    dˣ = Graphs.degree(mygraph)                         # observed degree sequence ("d star")
    inds = sortperm(dˣ)
    # Squartini method
    d̂  = degree(M)                                      # expected degree sequence from the model
    σ_d = map(n -> σˣ(G -> degree(G, n), M), 1:length(M))      # standard deviation for each degree in the sequence
    z_d = (dˣ - d̂) ./ σ_d
    # Sampling method
    S_d = hcat(map( g -> Graphs.degree(g), S)...);       # degree sequences from the sample
    # illustration 
    # - acceptable domain
    p1 = scatter(dˣ[inds], dˣ[inds], label="observed", color=:black, marker=:xcross, 
                xlabel="observed node degree", ylabel="node degree", legend_position=:topleft,
                bottom_margin=5mm, left_margin=5mm)
    scatter!(p1, dˣ[inds], d̂[inds], label="expected (analytical)", linestyle=:dash, markerstrokecolor=:steelblue,markercolor=:white)
    plot!(p1, dˣ[inds], d̂[inds] .+  2* σ_d[inds], linestyle=:dash,color=:steelblue,label="acceptance domain (analytical)")
    plot!(p1, dˣ[inds], d̂[inds] .-  2* σ_d[inds], linestyle=:dash,color=:steelblue,label="")
    # - z-scores (NOTE: manual log computation to be able to deal with zero values)
    p2 = scatter(log10.(abs.(z_d)), label="analytical", xlabel="node ID", ylabel="|degree z-score|", color=:steelblue,legend=:bottom)
    for ns in [100;1000;10000]
        μ̂ = reshape(mean( S_d[:,1:ns], dims=2), :)
        ŝ = reshape(std(  S_d[:,1:ns], dims=2), :)
        z_s = (dˣ - μ̂) ./ ŝ
        scatter!(p1, dˣ[inds], μ̂[inds], label="sampled (n = $(ns))", color=Int(log10(ns)), marker=:cross)
        plot!(p1, dˣ[inds], μ̂[inds] .+ 2*ŝ[inds], linestyle=:dash, color=Int(log10(ns)),  label="acceptance domain, sampled (n=$(ns))")
        plot!(p1, dˣ[inds], μ̂[inds] .- 2*ŝ[inds], linestyle=:dash, color=Int(log10(ns)), label="")
        scatter!(p2, log10.(abs.(z_s)), label="sampled (n = $(ns))", color=Int(log10(ns)))
    end
    plot!(p2, [1; length(M)], log10.([3;3]), color=:red, label="statistical significance threshold", linestyle=:dash)
    yrange = -12:2:0
    yticks!(p2,yrange, ["1e$(x)" for x in yrange])
    plot(p1,p2,layout=(1,2), size=(1200,600), title="Zachary karate club")
    savefig("""ZKC_degree_$("$(round(now(), Day))"[1:10]).pdf""")

    ## ANND metric testing
    # Squartini method
    ANNDˣ = ANND(mygraph)                                   # observed ANND sequence ("ANND star")
    ANND_hat = ANND(M)                                      # expected ANND sequence
    σ_ANND = map(n -> σˣ(G -> ANND(G, n), M), 1:length(M))  # standard deviation for each ANND in the sequence
    z_ANND = (ANNDˣ - ANND_hat) ./ σ_ANND
    # Sampling method
    S_ANND = hcat(map( g -> ANND(g), S)...);                # ANND sequences from the sample
    # illustration 
    # - acceptable domain
    p1 = scatter(dˣ[inds], ANNDˣ[inds], label="observed", color=:black, marker=:xcross, 
                xlabel="observed node degree", ylabel="ANND", legend_position=:topleft,
                bottom_margin=5mm, left_margin=5mm)
    scatter!(p1, dˣ[inds], ANND_hat[inds], label="expected (analytical)", linestyle=:dash, markerstrokecolor=:steelblue,markercolor=:white)
    plot!(p1, dˣ[inds], ANND_hat[inds] .+  2* σ_ANND[inds], linestyle=:dash,color=:steelblue,label="acceptance domain (analytical)")
    plot!(p1, dˣ[inds], ANND_hat[inds] .-  2* σ_ANND[inds], linestyle=:dash,color=:steelblue,label="")
    # - z-scores (NOTE: manual log computation to be able to deal with zero values)
    p2 = scatter(log10.(abs.(z_ANND)), label="analytical", xlabel="node ID", ylabel="|ANND z-score|", color=:steelblue,legend=:bottom)
    for ns in [100;1000;10000]
        μ̂ = reshape(mean( S_ANND[:,1:ns], dims=2), :)
        ŝ = reshape(std(  S_ANND[:,1:ns], dims=2), :)
        z_s = (ANNDˣ - μ̂) ./ ŝ
        scatter!(p1, dˣ[inds], μ̂[inds], label="sampled (n = $(ns))", color=Int(log10(ns)), marker=:cross)
        plot!(p1, dˣ[inds], μ̂[inds] .+ 2*ŝ[inds], linestyle=:dash, color=Int(log10(ns)),  label="acceptance domain, sampled (n=$(ns))")
        plot!(p1, dˣ[inds], μ̂[inds] .- 2*ŝ[inds], linestyle=:dash, color=Int(log10(ns)), label="")
        scatter!(p2, log10.(abs.(z_s)), label="sampled (n = $(ns))", color=Int(log10(ns)))
    end
    plot!(p2, [1; length(M)], log10.([3;3]), color=:red, label="statistical significance threshold", linestyle=:dash)
    yrange = -3:1:1
    ylims!(p2,-3,1)
    yticks!(p2,yrange, [L"$10^{ %$(x)}$" for x in yrange])
    plot(p1,p2,layout=(1,2), size=(1200,600), title="Zachary karate club")
    savefig("""ZKC_ANND_$("$(round(now(), Day))"[1:10]).pdf""")

    ## motifs testing
    # Squartini method
    motifs = [M₁; M₂]                  # different motifs that will be computed
    motifnames = Dict(M₁ => "M₁, v-motif", M₂ => "M₂, triangle")
    Mˣ  = map(f -> f(mygraph), motifs) # observed motifs sequence ("M star")
    M̂   = map(f -> f(M), motifs)       # expected motifs sequence
    σ_M = map(f -> σˣ(f, M), motifs)   # standard deviation for each motif in the sequence
    z_M̂ = (Mˣ - M̂) ./ σ_M
    # Sampling method
    Ns = [100;1000;10000]
    z_M = zeros(length(motifs), length(Ns))
    for i in eachindex(Ns)
        res = hcat(map(s -> map(f -> f(s), motifs),S[1:Ns[i]])...)
        μ̂ = mean(res, dims=2)
        ŝ = std(res, dims=2)
        z_M[:,i] = (Mˣ - μ̂) ./ ŝ
    end
    # illustration
    plot(Ns, abs.(permutedims(repeat(z_M̂, 1,3))), color=[:steelblue :sienna], label=reshape(["$(motifnames[f]) (analytical)" for f in motifs],1,2), linestyle=[:dash :dot], 
        xlabel="Sample size", ylabel="|motif z-score|", xscale=:log10)
    plot!(Ns, abs.(permutedims(z_M)), color=[:steelblue :sienna], label=reshape(["$(motifnames[f]) (sampled)" for f in motifs],1,2), marker=:circle, linestyle=[:dash :dot])
    savefig("""ZKC_motifs_$("$(round(now(), Day))"[1:10]).pdf""")
end

function DBCM_test()
    NP = PyCall.pyimport("NEMtropy")
    mygraph = Graphs.erdos_renyi(183,2494, is_directed=true, seed=161)
    mygraph_nem = NP.DirectedGraph(degree_sequence=vcat(Graphs.outdegree(mygraph), Graphs.indegree(mygraph)))
    mygraph_nem.solve_tool(model="dcm", method="fixed-point", initial_guess="random");
    
    # quick checks functionality
    @assert outdegree(Graphs.adjacency_matrix(mygraph)) == Graphs.outdegree(mygraph)
    @assert indegree( Graphs.adjacency_matrix(mygraph)) == Graphs.indegree(mygraph)
    @assert mygraph_nem.dseq_out == Graphs.outdegree(mygraph)
    @assert mygraph_nem.dseq_in  == Graphs.indegree(mygraph)
    
    M = UBCM(mygraph_nem.x, mygraph_nem.y) # generate the model
    # quick checks convergence
    @assert Graphs.outdegree(mygraph) ≈ outdegree(M)
    @assert Graphs.indegree(mygraph)  ≈ indegree(M)
         
    S = [rand(M) for _ in 1:10000]; # sample 10000 graphs from the model
    
    ## degree metric testing
    dˣ_in, dˣ_out = Graphs.indegree(mygraph), Graphs.outdegree(mygraph) # observed degree sequence ("d star")
    inds_in, inds_out = sortperm(dˣ_in), sortperm(dˣ_out)
    # Squartini method
    d̂_in, d̂_out  = indegree(M), outdegree(M)                            # expected degree sequence from the model
    σ_d_in =  map(n -> σˣ(G -> indegree(G, n), M),  1:length(M))        # standard deviation for each degree in the sequence
    σ_d_out = map(n -> σˣ(G -> outdegree(G, n), M), 1:length(M))      
    z_d_in, z_d_out = (dˣ_in - d̂_in) ./ σ_d_in, (dˣ_out - d̂_out) ./ σ_d_out
    # Sampling method
    S_d_in, S_d_out = hcat(map(g -> Graphs.indegree(g), S)...), hcat(map(g -> Graphs.outdegree(g), S)...);       # degree sequences from the sample
    
    # illustration
    # - acceptable domain
    # --indegree
    p1 = scatter(dˣ_in[inds_in], dˣ_in[inds_in], label="observed", color=:black, marker=:xcross, 
                    xlabel="observed node indegree", ylabel="node indegree", legend_position=:topleft,
                    bottom_margin=5mm, left_margin=5mm)
    scatter!(p1, dˣ_in[inds_in], d̂_in[inds_in], label="expected (analytical)", linestyle=:dash, markerstrokecolor=:steelblue,markercolor=:white)                
    plot!(p1, dˣ_in[inds_in], d̂_in[inds_in] .+  2* σ_d_in[inds_in], linestyle=:dash,color=:steelblue,label="acceptance domain (analytical)")
    plot!(p1, dˣ_in[inds_in], d̂_in[inds_in] .-  2* σ_d_in[inds_in], linestyle=:dash,color=:steelblue,label="")
    # --outdegree
    p3 = scatter(dˣ_out[inds_out], dˣ_out[inds_out], label="observed", color=:black, marker=:xcross, 
                    xlabel="observed node outdegree", ylabel="node outdegree", legend_position=:topleft,
                    bottom_margin=5mm, left_margin=5mm)
    scatter!(p3, dˣ_out[inds_out], d̂_out[inds_out], label="expected (analytical)", linestyle=:dash, markerstrokecolor=:steelblue,markercolor=:white)                
    plot!(p3, dˣ_out[inds_out], d̂_out[inds_out] .+  2* σ_d_out[inds_out], linestyle=:dash,color=:steelblue,label="acceptance domain (analytical)")
    plot!(p3, dˣ_out[inds_out], d̂_out[inds_out] .-  2* σ_d_out[inds_out], linestyle=:dash,color=:steelblue,label="")
    # - z-scores (NOTE: manual log computation to be able to deal with zero values)
    p2 = scatter(log10.(abs.(z_d_in)),  label="analytical", xlabel="node ID", ylabel="|indegree z-score|",  color=:steelblue,legend=:bottom)
    p4 = scatter(log10.(abs.(z_d_out)), label="analytical", xlabel="node ID", ylabel="|outdegree z-score|", color=:steelblue,legend=:bottom)
    for ns in [100;1000;10000]
        # --indegree
        μ̂_in = reshape(mean( S_d_in[:,1:ns], dims=2), :)
        ŝ_in = reshape(std(  S_d_in[:,1:ns], dims=2), :)
        z_s_in = (dˣ_in - μ̂_in) ./ ŝ_in
        scatter!(p1, dˣ_in[inds_in], μ̂_in[inds_in], label="sampled (n = $(ns))", color=Int(log10(ns)), marker=:cross)
        plot!(p1, dˣ_in[inds_in], μ̂_in[inds_in] .+ 2*ŝ_in[inds_in], linestyle=:dash, color=Int(log10(ns)),  label="acceptance domain, sampled (n=$(ns))")
        plot!(p1, dˣ_in[inds_in], μ̂_in[inds_in] .- 2*ŝ_in[inds_in], linestyle=:dash, color=Int(log10(ns)), label="")
        scatter!(p2, log10.(abs.(z_s_in)), label="sampled (n = $(ns))", color=Int(log10(ns)))
        # --outdegree
        μ̂_out = reshape(mean( S_d_out[:,1:ns], dims=2), :)
        ŝ_out = reshape(std(  S_d_out[:,1:ns], dims=2), :)
        z_s_out = (dˣ_out - μ̂_out) ./ ŝ_out
        scatter!(p3, dˣ_out[inds_out], μ̂_out[inds_out], label="sampled (n = $(ns))", color=Int(log10(ns)), marker=:cross)
        plot!(p3, dˣ_out[inds_out], μ̂_out[inds_out] .+ 2*ŝ_out[inds_out], linestyle=:dash, color=Int(log10(ns)),  label="acceptance domain, sampled (n=$(ns))")
        plot!(p3, dˣ_out[inds_out], μ̂_out[inds_out] .- 2*ŝ_out[inds_out], linestyle=:dash, color=Int(log10(ns)), label="")
        scatter!(p4, log10.(abs.(z_s_out)), label="sampled (n = $(ns))", color=Int(log10(ns)))
    end
    yrange = -10:2:0
    ylims!(p2,-10,0); ylims!(p4,-10,0)
    yticks!(p2,yrange, ["1e$(x)" for x in yrange])
    yticks!(p4,yrange, ["1e$(x)" for x in yrange])
    p_in  = plot(p1,p2,layout=(1,2), size=(1200,600), title="random Erdos-Renyi graph")
    p_out = plot(p3,p4,layout=(1,2), size=(1200,600), title="random Erdos-Renyi graph")
    savefig(p_in,  """ER_indegree_$("$(round(now(), Day))"[1:10]).pdf""")
    savefig(p_out, """ER_outdegree_$("$(round(now(), Day))"[1:10]).pdf""")
    
    ## motif testing
    # Squartini method
    mˣ = motifs(mygraph)
    m̂  = motifs(M)
    mfuns = [Symbol('M' * prod(map(x -> Char(x+48+8272),map(v -> reverse(digits(v)), i)))) for i = 1:13]
    σ_m = Dict{Int, Float64}()
    for i = 1:13
        @info "working on $(i)"
        σ_m[i] = σˣ(eval(mfuns[i]), M)
    end
    z_m = [(mˣ[i] - m̂[i]) ./ σ_m[i] for i = 1:13]
    # simulation method
    S_m = motifs.(S, full=true);
    S_M = hcat(S_m...)
    
    # illustration
    p_mot = scatter(z_m, xlabel="Motif ID", ylabel="z-score", color=:steelblue, label="analytical", 
                    size=(1200,600), bottom_margin=5mm, left_margin=5mm, legend=:bottomleft, legendfontsize=6)
    xticks!(collect(1:13))
    for ns in [100;1000;10000]
        m̂_sim = mean(S_M[:,1:ns], dims=2)
        σ̂_sim = std( S_M[:,1:ns], dims=2) 
        z_sim = (mˣ - m̂_sim) ./ σ̂_sim
        scatter!(p_mot, z_sim, color=Int(log10(ns)), label="sampled (n = $(ns))", markeralpha=0.5)
    end
    plot!(p_mot, collect(1:13), -2*ones(13), color=:red, label="statistical significance threshold", linestyle=:dash)
    plot!(p_mot, collect(1:13), 2*ones(13), color=:red, label="", linestyle=:dash)
    
    title!("random Erdos-Renyi graph")
    xlims!(0,14)
    savefig(p_mot, """ER_motifs_$("$(round(now(), Day))"[1:10]).pdf""")
end


#ylims!(-10,0)

#σˣ($(f), $(M)
#=

"""
∇X(X::Function, G::Matrix{T})

Compute the gradient of a metric X given an adjacency/weight matrix G
"""
∇X(::Function, ::Matrix)

function ∇X(X::Function, G::Matrix{T}) where T<:Real
    return ReverseDiff.gradient(X, G)
end

function ∇X(X::Function, m::UBCM{T}) where T<:Real
    G = Ĝ(m)
    return ReverseDiff.gradient(X, G)
end

"""
σₓ(::Function)

Compute variance of a metric X given the expected adjacency/weight matrix
"""
σₓ(::Function, ::Matrix)

function σX(X::Function, Ĝ::Matrix{T}, σ̂::Matrix{T}) where T<:Real
    return sqrt( sum((σ̂ .* ∇X(X, Ĝ)) .^ 2) )
end




## some metrics 
d(A,i) = sum(@view A[:,i]) # degree node i
ANND(G::Graphs.SimpleGraph,i) where T = iszero(Graphs.degree(G,i)) ? zero(Float64) : sum(map( n -> Graphs.degree(G,n), Graphs.neighbors(G,i))) / Graphs.degree(G,i)
ANND(G::Graphs.SimpleGraph) where T = map( i -> ANND(G,i), 1:Graphs.nv(G))
ANND(A,i) =  sum(A[i,j] * d(A,j) for j=1:size(A,1) if j≠i) / d(A,i)






# sampling the model
S = [rand(M) for _ in 1:10000];
dS = hcat(map( g -> Graphs.degree(g), S)...);
d_sample_mean_estimate(V,n) = mean(V[:,1:n], dims=2);
d_sample_var_estimate(V,n) =  std(V[:,1:n], dims=2);

# illustration for degree - acceptable domain
p1 = scatter(d_star[inds], d_star[inds], label="observed", legend=:topleft)
plot!(d_star[inds], map(n -> d(G_exp,n), 1:34)[inds], label="expected", linestyle=:dash, color=:steelblue)
plot!(d_star[inds], map(n -> d(G_exp,n), 1:34)[inds] .+  2* map(n-> σX(A->d(A,n), G_exp, σ_exp), 1:34)[inds],linestyle=:dash,color=:black,label="acceptance domain (analytical)")
plot!(d_star[inds], map(n -> d(G_exp,n), 1:34)[inds] .-  2* map(n-> σX(A->d(A,n), G_exp, σ_exp), 1:34)[inds],linestyle=:dash,color=:black,label="")
for ns in [10;100;1000;10000]
    plot!(d_star[inds], d_sample_mean_estimate(dS,ns)[inds] .+ 2* d_sample_var_estimate(dS,ns)[inds],linestyle=:dot,label="acceptance domain, sample (n=$(ns))")
    plot!(d_star[inds], d_sample_mean_estimate(dS,ns)[inds] .- 2* d_sample_var_estimate(dS,ns)[inds],linestyle=:dot,label="")
end
xlabel!("observed degree")
ylabel!("expected degree")
title!("UBCM - Zachary karate club")

# illustration for degree - z-scores per node
p2 = scatter(abs.((d_star - map(n -> d(G_exp,n), 1:34)) ./ map(n-> σX(A->d(A,n), G_exp, σ_exp), 1:34)), label="analytical", yscale=:log10)
for ns in [10;100;1000;10000]
    z = abs.((d_star - d_sample_mean_estimate(dS,ns)) ./ d_sample_var_estimate(dS,ns))
    lab = @sprintf "sample (n= %1.0e)" ns
    scatter!(z[z.>0], label=lab, yscale=:log10)
end
plot!([1; Graphs.nv(mygraph)], [3;3], linestyle=:dash,color=:red,label="")
xlabel!("node id")
ylabel!("|z-score degree|")
title!("UBCM - Zachary karate club")
ylims!(1e-10,1)

plot!(p1,p2, size=(1200,600), bottom_margin=5mm)
savefig("ZKC - degree.pdf")
nothing



##
## ANND part ##
##
X_star = ANND(mygraph)  # observed value (real network)
X_exp  = map(n->ANND(G_exp,n), 1:Graphs.nv(mygraph))    # expected value (analytical)
X_std  = map(n-> σX(A->ANND(A,n), G_exp, σ_exp), 1:34)  # standard deviation (analytical)
z_anal = (X_star - X_exp) ./ X_std

xS = hcat(map( g -> ANND(g), S)...);  #segmentation fault!!! => solved, tpye unstable


AP1 = scatter(d_star[inds], X_star[inds], label="observed", legend=:topright)
plot!(d_star[inds], X_exp[inds], label="expected", linestyle=:dash, color=:steelblue)
plot!(d_star[inds], X_exp[inds] .+ 2*X_std[inds], linestyle=:dash,color=:black,label="acceptance domain (analytical)")
plot!(d_star[inds], X_exp[inds] .- 2*X_std[inds], linestyle=:dash,color=:black,label="")
for ns in [10;100;1000;10000]
    plot!(d_star[inds], d_sample_mean_estimate(xS,ns)[inds] .+ 2* d_sample_var_estimate(xS,ns)[inds],linestyle=:dot,label="acceptance domain, sample (n=$(ns))")
    plot!(d_star[inds], d_sample_mean_estimate(xS,ns)[inds] .- 2* d_sample_var_estimate(xS,ns)[inds],linestyle=:dot,label="")
end
xlabel!("observed degree")
ylabel!("ANND")
title!("UBCM - Zachary karate club")

# illustration for degree - z-scores per node
AP2 = scatter(abs.(z_anal), label="analytical", yscale=:log10)
for ns in [10;100;1000;10000]
    z = abs.((X_star - d_sample_mean_estimate(xS,ns)) ./ d_sample_var_estimate(xS,ns))
    lab = @sprintf "sample (n= %1.0e)" ns
    scatter!(z[z.>0], label=lab, yscale=:log10)
end
plot!([1; Graphs.nv(mygraph)], [3;3], linestyle=:dash,color=:red,label="", legend=:bottomleft)
xlabel!("node id")
ylabel!("|z-score ANND|")
title!("UBCM - Zachary karate club")
ylims!(1e-3,1e1)

plot(AP1,AP2, size=(1200,600), bottom_margin=5mm)
savefig("ZKC - ANND.pdf")






N₂_obs = M₂(Graphs.adjacency_matrix(mygraph))
N₂_exp = M₂(G_exp)
N₂_σ   = σX(M₂, G_exp, σ_exp) # standard deviation (analytical)
z_N₂ = (N₂_obs- N₂_exp) / N₂_σ
# from the sample
N₂_sampled = map(g -> M₂(Graphs.adjacency_matrix(g)), S)
# computed values:
N₂_hat = [mean(N₂_sampled[1:n]) for n ∈ [10;100;1000;10000]]
N₂_std_hat = [std(N₂_sampled[1:n]) for n ∈ [10;100;1000;10000]]
N₂_z_hat = (N₂_obs .- N₂_hat) ./ N₂_std_hat

scatter([10;100;1000;10000],abs.(N₂_z_hat), xscale=:log10, label="sampled")
plot!([10;100;1000;10000], abs.(z_N₂ .* ones(4)), linestyle=:dash, label="analytical")
ylims!(0,2)
xticks!([10;100;1000;10000])
xlabel!("sample size")
ylabel!("|z-score triangle count|")
title!("UBCM - Zachary karate club")
Plots.savefig("ZKC - Triangles.pdf")

N₁_obs = M₁(Graphs.adjacency_matrix(mygraph))
N₁_exp = M₁(G_exp)
N₁_σ   = σX(M₁, G_exp, σ_exp) # standard deviation (analytical)
z_N₁ = (N₁_obs- N₁_exp) / N₁_σ
# from the sample
N₁_sampled = map(g -> M₁(Graphs.adjacency_matrix(g)), S)
# computed values:
N₁_hat = [mean(N₁_sampled[1:n]) for n ∈ [10;100;1000;10000]]
N₁_std_hat = [std(N₁_sampled[1:n]) for n ∈ [10;100;1000;10000]]
N₁_z_hat = (N₁_obs .- N₁_hat) ./ N₁_std_hat

scatter([10;100;1000;10000],abs.(N₁_z_hat), xscale=:log10, label="sampled")
plot!([10;100;1000;10000], abs.(z_N₁ .* ones(4)), linestyle=:dash, label="analytical")
ylims!(0,2)
xticks!([10;100;1000;10000])
xlabel!("sample size")
ylabel!("|z-score v-motif count|")
title!("UBCM - Zachary karate club")
savefig("ZKC - v-motifs.pdf")


# scaffolding for the directed network motif elements
"""
    a⭢(A::Matrix{T}, i, j)

directed link from i to j and not from j to i
"""
a⭢(A::Matrix{T}, i, j) where T = A[i,j] * (one(T) - A[j,i]) 

"""
    a⭠(A::Matrix{T}, i, j)

directed link from j to i and not from i to j
"""
a⭠(A::Matrix{T}, i, j) where T = (one(T) - A[i,j]) * A[j,i]

"""
    a⭤(A::Matrix{T}, i, j)

recipocrated link between i and j
"""
a⭤(A,i,j) = A[i,j]*A[j,i]

"""
    a_̸(A::Matrix{T}, i, j)

no link between i and j
"""
a_̸(A::Matrix{T},i,j) where T = (one(T) - A[j,i]) * (one(T) - A[i,j])

# motifs itself

M_13(A) = sum(a⭤(A,i,j)*a⭤(A,j,k)*a⭤(A,k,i) for i = 1:size(A,1) for j=1:size(A,1) for k=1:size(A,1) if i≠j && j≠k && i≠k)

=#


# helper function to load up a network from a file
load_network_from_edges(f::String) = Graphs.loadgraph(f, EdgeListFormat())


"""
    run a complete analysis (with plots and storage (?))

"""
function DBCM_analysis(G::Graphs.SimpleDiGraph, name::String; 
                        N::Int=10000, subsamples::Vector{Int64}=[100;1000;10000],
                        nodefunctions= [(indegree, Graphs.indegree, Graphs.indegree); 
                                        (outdegree, Graphs.outdegree, Graphs.outdegree);
                                        (ANND_in, ANND_in, Graphs.indegree);
                                        (ANND_out, ANND_out, Graphs.outdegree)])
    NP = PyCall.pyimport("NEMtropy")
    
    # for plotting
    sample_labels = permutedims(map(l -> "sampled (n = $(l))", subsamples))
    sample_colors = permutedims(Int.(log10.(round.(Int,subsamples))))

    @info "$(round(now(), Minute)) - Starting analysis for $(name)"
    @info "$(round(now(), Minute)) - Computing ML parameters for $(name)"
    G_nem =  NP.DirectedGraph(degree_sequence=vcat(Graphs.outdegree(G), Graphs.indegree(G)))
    G_nem.solve_tool(model="dcm", method="fixed-point", initial_guess="random")
    model = DBCM(G_nem.x, G_nem.y)
    @info "$(round(now(), Minute)) - Generating sample (n = $(N)) for $(name)"
    S = [rand(model) for _ in 1:N]

    ### Different METRICS ###
    ## Node metrics ##
    #=
    @info "$(round(now(), Minute)) - Computing node metrics for $(name)"
    for (f, f_graphs, f_ref) in nodefunctions
        @info "$(round(now(), Minute)) - Computing $(f) for $(name)"
        Xˣ, X̂, σ̂_X̂, X_S = node_metric(f, G, model, S; graph_fun=f_graphs)
        ref_val = f_ref(G)
        inds = sortperm(ref_val) # sorted indices based on oberved value (not based on degree)
        # compute z-score (Squartini)
        z_X_a = (Xˣ .- X̂) ./ σ̂_X̂
        # compute z-score (Simulation)
        X̂_S = hcat(map(n -> reshape(mean(X_S[:,1:n], dims=2),:), subsamples)...)
        σ̂_X̂_S = hcat(map(n -> reshape( std(X_S[:,1:n], dims=2),:), subsamples)...)
        z_X_S = (Xˣ .- X̂_S) ./ σ̂_X̂_S

        # illustration
        p1 = scatter(ref_val[inds], Xˣ[inds], label="observed", color=:red, marker=:circle, 
                     xlabel="observed node $(f_ref)", ylabel="node $(f)", legend_position=:topleft,
                     bottom_margin=5mm, left_margin=5mm)
        scatter!(p1, ref_val[inds], X̂[inds], label="expected (analytical - DBCM)", markerstrokecolor=:steelblue, markercolor=:white)  
        plot!(p1,ref_val[inds],X̂[inds] .+ 2* σ̂_X̂[inds], label="expected ± 2σ (analytical - DBCM)", linestyle=:dash, color=:steelblue)              
        plot!(p1,ref_val[inds],X̂[inds] .- 2* σ̂_X̂[inds], label="", linestyle=:dash, color=:steelblue)
        
        # maybe re-evaluate this
        p2 = scatter(log10.(abs.(z_X_a)),  label="analytical - BDCM)", xlabel="node ID", ylabel="|$(f) z-score|",
                     color=:steelblue,legend=:bottom)
        scatter!(p2, log10.(abs.(z_X_S)), label=sample_labels, color=sample_colors)
        plot!(p2, )
        p = plot(p1,p2,layout=(1,2), size=(1200,600))
        savefig(p, """$(name)_$("$(round(now(), Day))"[1:10])_$(f)_z-score.pdf""")
    end
=#
    ## Graph metrics (e.g. motifs) ##
    @info "$(round(now(), Minute)) - Computing graph metrics for $(name)"
    @info "$(round(now(), Minute)) - Computing triadic motifs for $(name)"
    mˣ = motifs(G)
    m̂  = motifs(model)
    σ̂_m̂  = Vector{eltype(model.G)}(undef, length(m̂))
    for i = 1:length(m̂)
        @info "$(round(now(), Minute)) - Computing standard deviation for motif $(i)"
        σ̂_m̂[i] = σˣ(DBCM_motif_functions[i], model)
    end
    # compute z-score (Squartini)
    z_m_a = [(mˣ[i] - m̂[i]) ./ σ̂_m̂[i] for i = 1:length(m̂)]
    # compute z-score (Simulation)
    @info "$(round(now(), Minute)) - Computing motifs for the sample"
    S_m = hcat(motifs.(S,full=true)...); # computed values from sample
    m̂_S =   hcat(map(n -> reshape(mean(S_m[:,1:n], dims=2),:), subsamples)...)
    σ̂_m̂_S = hcat(map(n -> reshape( std(S_m[:,1:n], dims=2),:), subsamples)...)
    z_m_S = (mˣ .- m̂_S) ./ σ̂_m̂_S
    #=
    # illustration
    p_mot = scatter(z_m_a, xlabel="Motif ID", ylabel="z-score", color=:steelblue, label="analytical (BDCM)", 
                    size=(1200,600), bottom_margin=5mm, left_margin=5mm, legend=:topleft, legendfontsize=6, xticks=collect(1:13))


    scatter!(p_mot, z_m_S, label=sample_labels, color=sample_colors, markeralpha=0.5)
    plot!(p_mot, collect(1:13), -2*ones(13), color=:red, label="statistical significance threshold", linestyle=:dash)
    plot!(p_mot, collect(1:13), 2*ones(13), color=:red, label="", linestyle=:dash)
    title!("$(name)")
    xlims!(0,14)
    savefig(p_mot, """$(name)_$("$(round(now(), Day))"[1:10])_motifs_z-score.pdf""")
    =#

    return hcat(z_m_a, z_m_S), m̂, m̂_S, σ̂_m̂_S
end
   

"""
node_metric(X::Function, G::Graphs.SimpleDiGraph, M::DBCM, S::Vector{T}; graph_fun::Function=X) where T<: Graphs.SimpleDiGraph

For the metric `X`, compute the observed value from the graph `G`, the expected value under the DBCM `M` and the values for the 
elements of the sample `S`. The function `graph_fun` is used to indicate the function that will be used to compute the metric `X`
on the `::Graphs.SimpleDiGraph` object (some metrics such as degree are available in the `Graphs` package and are prefered for speed). 
The default is to use the function `X` itself.

By default, this function assumes that the metric `X` needs to be computed for each node in the graph.

The return tuple is (Xˣ, X̂, σ̂_X̂, X_S) where
- `Xˣ`: the observed value of the metric `X`
- `X̂`: the expected value of the metrix `X` under the null model
- `σ̂_X̂`: the standard deviation of the metric `X` under the null model
- `X_S`: the values of the metric `X` for the elements of the sample

Can be used in combination with: (indegree, Graphs.indegree), (outdegree, Graphs.outdegree), ANND

"""
function node_metric(X::Function, G::Graphs.SimpleDiGraph, M::DBCM, S::Vector{T}; graph_fun::Function=X) where T<: Graphs.SimpleDiGraph
    Xˣ =  graph_fun(G)                        # compute observed value
    X̂  =  X(M)                                # compute value for the DBCM
    σ̂_X̂  =  map(n -> σˣ(g -> indegree(g, n), M),  1:length(M)) # compute variance for the DBCM
    X_S = hcat(map(g -> graph_fun(g), S)...) # compute value for the sample

    return Xˣ, X̂, σ̂_X̂, X_S
end

#=
G_nem =  NP.DirectedGraph(degree_sequence=vcat(Graphs.outdegree(G), Graphs.indegree(G)))
    G_nem.solve_tool(model="dcm", method="fixed-point", initial_guess="random")
    model = UBCM(mygraph_nem.x, mygraph_nem.y)

    model = UBCM(mygraph_nem.x, mygraph_nem.y)


LRFW_nem = NP.DirectedGraph(degree_sequence=vcat(Graphs.outdegree(LRFW), Graphs.indegree(LRFW)))
LRFW_nem.solve_tool(model="dcm", method="fixed-point", initial_guess="random")
LRFW_M = DBCM(LRFW_nem.x, LRFW_nem.y)
LRFB_sample = [rand(LRFW_M) for i = 1:100]
res = computevals(ANND_in, LRFB, LRFB_M, LRFB_sample)
inds = sortperm(Graphs.indegree(LRFB))


LRFW = load_network_from_edges("./data/foodweb_little_rock/edges.csv")
LRFW_nem =  NP.DirectedGraph(degree_sequence=vcat(Graphs.outdegree(LRFW), Graphs.indegree(LRFW)))
LRFW_nem.solve_tool(model="dcm", method="fixed-point", initial_guess="random")
LRFW_model = DBCM(mygraph_nem.x, mygraph_nem.y)
LRFW_sample = [rand(LRFW_model) for i = 1:100]
res = computevals(ANND_in, LRFW, LRFW_model, LRFW_sample)

inds = sortperm(Graphs.indegree(LRFW))
x = Graphs.indegree(LRFW)
scatter(x[inds], res[1][inds], label="analytical", color=:red)
plot!(x[inds], res[2][inds], color=:black)
plot!(x[inds], res[2][inds] .+ 2*res[3][inds], color=:blue,line=:dash)
plot!(x[inds], res[2][inds] .- 2*res[3][inds], color=:blue,line=:dash)

sss = reshape(std(res[4], dims=2),:)
#(res[1] .- reshape(mean(res[4], dims=2),:)) ./ reshape(std(res[4], dims=2),:)

plot!(x[inds], res[2][inds] .+ 2*sss[inds], label="sample", color=:green)
xticks!(0:20:100)

=#

#mijnetwerk = Graphs.SimpleDiGraph( Graphs.smallgraph(:karate))
#DBCM_analysis(mijnetwerk, "karate")
#Gbis = Graphs.erdos_renyi(70, 2500, is_directed=true)
#DBCM_analysis(Gbis, "randomerdosrenyi")


# Figure 5 reproduction from the paper
G_chesapeake = readpajek("./data/Chesapeake_bay.txt");
z_chesapeake, r_chesapeake... = DBCM_analysis(G_chesapeake, "Chesapeake_bay")

G_foodnet = load_network_from_edges("./data/foodweb_little_rock/edges.csv")
z_little_rock, r_little_rock... = DBCM_analysis(foodnet, "Little_rock")

G_maspalomas = readpajek("./data/Maspalomas_lagoon.txt");
z_maspalomas, r_maspalomas... = DBCM_analysis(G_maspalomas, "Maspalomas")

G_florida_bay = readpajek("./data/Florida_bay.txt");
z_florida_bay, r_florida_bay... = DBCM_analysis(G_florida_bay, "Floriday_bay")

G_stmark = readpajek("./data/StMarks_seagrass.txt");
z_stmark, z_stmark... = DBCM_analysis(G_stmark, "St_mark")


data = (z_chesapeake, z_little_rock, z_maspalomas, z_florida_bay, z_stmark)
plotnames = ["Chesapeake Bay";"Little Rock Lake"; "Maspalomas Lagoon";"Florida Bay"; "St Marks Seagrass"]
mycolors = [:crimson; :steelblue; :seagreen; :indigo; :goldenrod]; 
markershapes = [:circle; :utriangle; :rect; :star5]

myplot = plot(size=(800,600), bottom_margin=5mm)
for i in 1:length(plotnames)
    #@info Int((i-1)*13+1):Int((i-1)*13+13)
    #data[Int((i-1)*1+1):Int((i-1)+13),1]
    plot!(myplot, x, data[i][:,1]   , color=mycolors[i], marker=markershapes[1], label="$(plotnames[i])", markeralpha=0.5, markersize=2, markerstrokewidth=0)
    scatter!(myplot, x, data[i][:,2:end], color=mycolors[i], marker=permutedims(markershapes[2:end]), label="", markeralpha=0.5, markersize=2,markerstrokewidth=0)
end
plot!(myplot,[0;14], [-2 2;-2 2], label="", color=:black, legend_position=:bottomright)
xlabel!(myplot,"motif \n∘: analytical\n △: n=100, □: n=1000, ⋆:n=10000")
ylabel!(myplot,"z-score")
xlims!(myplot,0,14)
xticks!(myplot,x)
ylims!(myplot,-17,15)
yticks!(myplot,-15:5:15)
title!(myplot,"Analytical vs simulation results for motifs (DBCM model)")
savefig(myplot, "comparison_motifs_dbcm_$("$(round(now(), Minute))").pdf")
