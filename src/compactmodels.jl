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
import Dates: now, Day          # for illustration printing

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
function UBCM(x::Vector{T}, y::Vector{T}; compute::Bool=true) where {T<:Real}
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
indegree(M::DBCM, i::Int)   = indegree(M.G, i)
indegree(M::DBCM)           = indegree(M.G)
# ANND metric

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
    eval.(map(f -> :($(f)(M.G)), fnames))
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
    eval.(map(f -> :($(f)($A)), fnames))
end

motifs(G::Graphs.SimpleDiGraph; full::Bool=false) = motifs(G, 1:13...; full=false)

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
end

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
mˣ = motifs(mygraph)
m̂  = motifs(M)

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
