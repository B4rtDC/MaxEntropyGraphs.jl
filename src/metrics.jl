# ----------------------------------------------------------------------------------------------------------------------
#
#                                               Supporting network functions
#
# Note: the function working on matrices need to be defined in without contraining the types too much
#       otherwise there will be a problem when using the autodiff package.
# ----------------------------------------------------------------------------------------------------------------------

## Undirected binary networks
# degree metric
Graphs.degree(A::T, i::Int) where T<: AbstractArray         = sum(@view A[:,i])             # degree of node i
Graphs.degree(A::T)         where T<: AbstractArray         = reshape(sum(A, dims=1), :)    # degree vector for the entire network
Graphs.degree(m::UBCM, i::Int)                              = Graphs.degree(m.G, i)                # degree of node i
Graphs.degree(m::UBCM)                                      = reshape(sum(m.G, dims=1), :)  # degree vector for the entire network
# ANND metric
ANND(G::Graphs.SimpleGraph, i)              = iszero(Graphs.degree(G,i)) ? zero(Float64) : sum(map( n -> Graphs.degree(G,n), Graphs.neighbors(G,i))) / Graphs.degree(G,i)
ANND(G::Graphs.SimpleGraph)                 = map(i -> ANND(G,i), 1:Graphs.nv(G))
ANND(A::T, i::Int) where T<: AbstractArray  = sum(A[i,j] * Graphs.degree(A,j) for j=1:size(A,1) if j≠i) / Graphs.degree(A,i)
ANND(A::T) where T<: AbstractArray          = map(i -> ANND(A,i), 1:size(A,1))
ANND(m::UBCM, i::Int)                       = ANND(m.G, i)
ANND(m::UBCM)                               = ANND(m.G)
# motifs
#M₁(A::T) where T<: AbstractArray = sum(A[i,j]*A[j,k]*(1 - A[k,i]) for i = axes(A,1) for j=i+1:size(A,1) for k=j+1:size(A,1))   # v-motifs metric
M₁(m::UBCM)                      = M₁(m.G)
M₁(G::Graphs.SimpleGraph)        = M₁(Graphs.adjacency_matrix(G))
#M₂(A::T) where T<: AbstractArray = sum(A[i,j]*A[j,k]*A[k,i] for i = axes(A,1) for j=i+1:size(A,1) for k=j+1:size(A,1))         # triangles metric
M₂(m::UBCM)                      = M₂(m.G)
M₂(G::Graphs.SimpleGraph)        = M₂(Graphs.adjacency_matrix(G))

"""
    degree_dist(m::UBCM, i::Int)

Compute the Poisson-Binomial distribution for the indegree of node `i` for the `UBCM` model `m`.
"""
degree_dist(m::UBCM, i::Int) = Distributions.PoissonBinomial(m.G[axes(m.G, 1) .!= i, i])

"""
    indegree_dist(m::UBCM)

Compute the Poisson-Binomial distribution for the indegree for all nodes for the `UBCM` model `m`.
"""
degree_dist(m::UBCM) = map(i -> degree_dist(m, i), axes(m.G,1))



## Directed binary networks
# degree metrics
Graphs.outdegree(A::T, i::Int) where T<: AbstractArray     = sum(@view A[i,:])             # out-degree of node i
Graphs.outdegree(A::T)         where T<: AbstractArray     = reshape(sum(A, dims=2), :)    # out-degree vector for the entire network
Graphs.outdegree(M::DBCM, i::Int)                          = outdegree(M.G, i)
Graphs.outdegree(M::DBCM)                                  = outdegree(M.G)
Graphs.indegree(A::T, i::Int) where T<: AbstractArray      = sum(@view A[:,i])             # out-degree of node i
Graphs.indegree(A::T)         where T<: AbstractArray      = reshape(sum(A, dims=1), :)    # out-degree vector for the entire network
Graphs.indegree(M::DBCM, i::Int)                           = indegree(M.G, i)
Graphs.indegree(M::DBCM)                                   = indegree(M.G)
# ANND metric
ANND_in(G::T, i) where T<:Graphs.SimpleGraphs.SimpleDiGraph     = iszero(Graphs.indegree(G,i)) ? zero(Float64) : sum(map( n -> Graphs.indegree(G,n), Graphs.inneighbors(G,i))) / Graphs.indegree(G,i)
ANND_in(G::T)       where T<:Graphs.SimpleGraphs.SimpleDiGraph  = map(i -> ANND_in(G,i), 1:Graphs.nv(G))
ANND_in(A::T, i::Int) where T<: AbstractArray                   = sum(A[j,i] * indegree(A,j) for j=1:size(A,1) if j≠i) / indegree(A,i)
ANND_in(A::T) where T<: AbstractArray                           = map(i -> ANND_in(A,i), 1:size(A,1))
ANND_in(m::DBCM, i::Int)                                        = ANND_in(m.G, i)
ANND_in(m::DBCM)                                                = ANND_in(m.G)
ANND_out(G::T, i) where T<:Graphs.SimpleGraphs.SimpleDiGraph    = iszero(Graphs.outdegree(G,i)) ? zero(Float64) : sum(map( n -> Graphs.outdegree(G,n), Graphs.outneighbors(G,i))) / Graphs.outdegree(G,i)
ANND_out(G::T)       where T<:Graphs.SimpleGraphs.SimpleDiGraph = map(i -> ANND_out(G,i), 1:Graphs.nv(G))
ANND_out(A::T, i::Int) where T<: AbstractArray                  = sum(A[i,j] * outdegree(A,j) for j=1:size(A,1) if j≠i) / outdegree(A,i)
ANND_out(A::T) where T<: AbstractArray                          = map(i -> ANND_out(A,i), 1:size(A,1))
ANND_out(m::DBCM, i::Int)                                       = ANND_out(m.G, i)
ANND_out(m::DBCM)                                               = ANND_out(m.G)
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

const DBCM_motif_function_names = [Symbol('M' * prod(map(x -> Char(x+48+8272),map(v -> reverse(digits(v)), i)))) for i = 1:13]

for i = 1:13 # mapping to different functions for adjacency matrix, DBCM model and graph
    fname = DBCM_motif_function_names[i]
    @eval begin 
        """
            $($fname)(A::T) where T<:AbstractArray
        
        Compute the motif $($fname) (Σ_{i≠j≠k} $(motif_functions[$i][1])(i,j) $(motif_functions[$i][2])(j,k) $(motif_functions[$i][3])(k,i) ) from the adjacency matrix.
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
        
        Compute the motif $($fname) (Σ_{i≠j≠k} $(motif_functions[$i][1])(i,j) $(motif_functions[$i][2])(j,k) $(motif_functions[$i][3])(k,i) ) from the `DBCM` model.
        """
        $(fname)(M::DBCM) = $(fname)(M.G)

        """
            $($fname)(G::SimpleDiGraph; full::Bool=false))
        
        Compute the motif $($fname) (Σ_{i≠j≠k} $(motif_functions[$i][1])(i,j) $(motif_functions[$i][2])(j,k) $(motif_functions[$i][3])(k,i) ) from the `SimpleDiGraph`.
        """
        $(fname)(G::Graphs.SimpleDiGraph; full::Bool=false) = $(fname)(full ? Array(Graphs.adjacency_matrix(G)) : Graphs.adjacency_matrix(G))
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

    return eval.(map(f -> :($(f)($M.G)), DBCM_motif_functions)) 
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
    # generate adjacency matrix
    A = full ? Array(Graphs.adjacency_matrix(G)) : Graphs.adjacency_matrix(G)
    # apply function
    res = Vector{Int64}(undef, length(n)) # fixed type for performance reasons (x35 faster)
    for i = 1:length(n)
        res[i] = eval(:($(DBCM_motif_function_names[i])($A)))
    end

    return res
end

motifs(G::Graphs.SimpleDiGraph; full::Bool=false) = motifs(G, 1:13...; full=full)


"""
    indegree_dist(m::DBCM, i::Int)

Compute the Poisson-Binomial distribution for the indegree of node `i` for the `DBCM` model `m`.
"""
indegree_dist(m::DBCM, i::Int) = Distributions.PoissonBinomial(m.G[axes(m.G, 1) .!= i, i])

"""
    indegree_dist(m::DBCM)

Compute the Poisson-Binomial distribution for the indegree for all nodes for the `DBCM` model `m`.
"""
indegree_dist(m::DBCM) = map(i -> indegree_dist(m, i), axes(m.G,1))

"""
    outdegree_dist(m::DBCM, i::Int)

Compute the Poisson-Binomial distribution for node `i` for the `DBCM` model `m`.
"""
outdegree_dist(m::DBCM, i::Int) = PoissonBinomial(m.G[i, axes(m.G, 2) .!= i])

"""
    outdegree_dist(m::DBCM)

Compute the Poisson-Binomial distribution for the outdegree for all nodes for the `DBCM` model `m`.
"""
outdegree_dist(m::DBCM) = map(i -> outdegree_dist(m, i), axes(m.G,1))




#=
        """
            $($fdistname)(A::T) where T<:AbstractArray
        
        Compute the Poisson Binomial distribution for motif $($fname) (Σ_{i,j,k} $(motif_functions[$i][1])(i,j) $(motif_functions[$i][2])(j,k) $(motif_functions[$i][3])(k,i) ) from the adjacency matrix.
        """
        function $(fdistname)(A::T)  where T<:AbstractArray
            res = eltype(A)[] #zero(eltype(A))
            for i = axes(A,1)
                for j = axes(A,1)
                    @simd for k = axes(A,1)
                        if i ≠ j && j ≠ k && k ≠ i
                            push!(res, $(motif_functions[i][1])(A,i,j) * $(motif_functions[i][2])(A,j,k) *   $(motif_functions[i][3])(A,k,i))
                        end
                    end
                end
            end
            return PoissonBinomial(res)
        end 

=#