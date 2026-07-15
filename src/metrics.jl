# ----------------------------------------------------------------------------------------------------------------------
#
#                                               Supporting network functions
#
# Note: the function working on matrices need to be defined without contraining the types too much
#       otherwise there will be a problem when using the autodiff package.
#
# Readability note: several matrix-based metrics (triangles, the directed motifs M1..M13, squares, V_motifs)
# are implemented as linear-algebra identities rather than the graph-intuitive neighbour loops. They are
# provably equal to the naive counts but that equivalence is NOT obvious from the code. The derivations and
# proofs (valid for real-valued matrices, not just 0/1) live in `performance/metrics_acceleration.tex`; the
# `ref_*` implementations in `test/metrics.jl` are the naive versions they are checked against.
# ----------------------------------------------------------------------------------------------------------------------

"""
    strength(G, T; dir)

Construct the strength vector for the graph `G`, filled with element type `T` and considering edge direction `dir ∈ [:in, :out, :both]` (default is `:out`).
"""
function strength(G::SimpleWeightedGraphs.AbstractSimpleWeightedGraph, T::DataType=SimpleWeightedGraphs.weighttype(G); dir::Symbol=:out)
    if Graphs.is_directed(G)
        if dir == :out
            d = vec(sum(G.weights; dims=1))
        elseif dir == :in
            d = vec(sum(G.weights; dims=2))
        elseif dir == :both
            d = vec(sum(G.weights; dims=1)) + vec(sum(G.weights; dims=2))
        else
            throw(DomainError(dir, "invalid argument, only accept :in, :out and :both"))
        end
    else
        d = vec(sum(G.weights; dims=1))
    end
    
    return T.(d)
end

# TO DO: extend to other graph type (Graphs.jl) & add docs
"""
    instrength(G, T; dir)

Construct the instrength vector for the graph `G`, filled with element type `T`.
"""
instrength(G::SimpleWeightedGraphs.AbstractSimpleWeightedGraph, T::DataType=SimpleWeightedGraphs.weighttype(G); dir::Symbol=:in)   = strength(G, T, dir=dir)

"""
    instrength(G, T; dir)

Construct the outstrength vector for the graph `G`, filled with element type `T`.
"""
outstrength(G::SimpleWeightedGraphs.AbstractSimpleWeightedGraph, T::DataType=SimpleWeightedGraphs.weighttype(G); dir::Symbol=:out) = strength(G, T, dir=dir)


"""
    strength(G, i, T; dir)

Construct the strength of node `i` for the graph `G`, filled with element type `T` and considering edge direction `dir ∈ [:in, :out, :both]` (default is `:out`).
"""
function strength(G::SimpleWeightedGraphs.AbstractSimpleWeightedGraph, i::N, T::DataType=SimpleWeightedGraphs.weighttype(G); dir::Symbol=:out) where N<:Integer
    # single-node strength: index only node `i`'s marginal instead of building the whole strength vector
    # (SimpleWeightedGraphs stores weights[dst, src], so column `i` = out-strength and row `i` = in-strength)
    if Graphs.is_directed(G)
        if dir == :out
            d = sum(@view G.weights[:, i])
        elseif dir == :in
            d = sum(@view G.weights[i, :])
        elseif dir == :both
            d = sum(@view G.weights[:, i]) + sum(@view G.weights[i, :])
        else
            throw(DomainError(dir, "invalid argument, only accept :in, :out and :both"))
        end
    else
        d = sum(@view G.weights[:, i])
    end

    return T(d)
end


# ----------------------------------------------------------------------------------------------------------------------
# Reciprocity metrics (directed graphs)
#
# Dyadic decomposition following Squartini & Garlaschelli (2011) New J. Phys. 13 083001, App. C.1
# (see also Di Vece et al. (2023)): every ordered pair (i,j), i≠j, is classified as
#   - non-reciprocated outgoing (i→j present, j→i absent):  a⭢ᵢⱼ = aᵢⱼ(1-aⱼᵢ)
#   - non-reciprocated incoming (j→i present, i→j absent):  a⭠ᵢⱼ = (1-aᵢⱼ)aⱼᵢ
#   - reciprocated (both present):                          a⭤ᵢⱼ = aᵢⱼaⱼᵢ
#   - absent (neither present).
# The *degree* matrix methods are purely algebraic (no binarisation): on a 0/1 adjacency matrix they count
# dyads, on a probability matrix under dyadic independence (e.g. a DBCM's Ĝ) they yield expected values.
# The *strength* variants and `weighted_reciprocity` use presence indicators (`!iszero`) on the reverse
# direction, so they are empirical quantities; model-expected counterparts are provided by the model methods.
# All definitions exclude the diagonal (self-loops are not considered).
# ----------------------------------------------------------------------------------------------------------------------

"""
    nonreciprocated_outdegree(A::AbstractMatrix, i::Int)
    nonreciprocated_outdegree(A::AbstractMatrix)

Compute the non-reciprocated outdegree of node `i` from the adjacency matrix `A`, defined as
``k^{→}_i = \\sum_{j≠i} a_{ij}(1 - a_{ji})``, i.e. the number of nodes `j` that `i` points to without `j` pointing back.
Without the node argument, the whole sequence is returned.

The computation is purely algebraic: for a 0/1 adjacency matrix this counts dyads, for a probability matrix
under dyadic independence it yields the expected value. Weighted adjacency matrices should be binarised first.

# Examples
```jldoctest
julia> A = [0 1 0; 1 0 1; 0 0 0];

julia> nonreciprocated_outdegree(A, 2)
1

julia> nonreciprocated_outdegree(A)
3-element Vector{Int64}:
 0
 1
 0
```
"""
function nonreciprocated_outdegree(A::T, i::Int) where T<:AbstractMatrix
    size(A,1) == size(A,2) || throw(DimensionMismatch("The adjacency matrix must be square"))
    (i < 1 || i > size(A,1)) && throw(ArgumentError("Attempted to access node $i in a $(size(A,1)) node graph"))
    o = one(eltype(A))
    res = zero(promote_type(eltype(A), Int))
    @inbounds for j in axes(A,2)
        j ≠ i && (res += A[i,j] * (o - A[j,i]))
    end
    return res
end

nonreciprocated_outdegree(A::T) where T<:AbstractMatrix = [nonreciprocated_outdegree(A, i) for i in axes(A,1)]

"""
    nonreciprocated_outdegree(G::Graphs.AbstractGraph, i::Int)
    nonreciprocated_outdegree(G::Graphs.AbstractGraph)

Compute the non-reciprocated outdegree of node `i` in the directed graph `G` (``k^{→}_i = \\sum_{j≠i} a_{ij}(1 - a_{ji})``).
Without the node argument, the whole sequence is returned.

# Examples
```jldoctest
julia> G = maspalomas();

julia> nonreciprocated_outdegree(G, 1)
4
```
"""
function nonreciprocated_outdegree(G::T, i::Int) where T<:Graphs.AbstractGraph
    Graphs.is_directed(G) || throw(ArgumentError("The graph must be directed for reciprocity-based metrics"))
    return count(j -> j ≠ i && !Graphs.has_edge(G, j, i), Graphs.outneighbors(G, i))
end

nonreciprocated_outdegree(G::T) where T<:Graphs.AbstractGraph = [nonreciprocated_outdegree(G, i) for i in Graphs.vertices(G)]


"""
    nonreciprocated_indegree(A::AbstractMatrix, i::Int)
    nonreciprocated_indegree(A::AbstractMatrix)

Compute the non-reciprocated indegree of node `i` from the adjacency matrix `A`, defined as
``k^{←}_i = \\sum_{j≠i} a_{ji}(1 - a_{ij})``, i.e. the number of nodes `j` that point to `i` without `i` pointing back.
Without the node argument, the whole sequence is returned.

The computation is purely algebraic: for a 0/1 adjacency matrix this counts dyads, for a probability matrix
under dyadic independence it yields the expected value. Weighted adjacency matrices should be binarised first.

# Examples
```jldoctest
julia> A = [0 1 0; 1 0 1; 0 0 0];

julia> nonreciprocated_indegree(A, 3)
1
```
"""
function nonreciprocated_indegree(A::T, i::Int) where T<:AbstractMatrix
    size(A,1) == size(A,2) || throw(DimensionMismatch("The adjacency matrix must be square"))
    (i < 1 || i > size(A,1)) && throw(ArgumentError("Attempted to access node $i in a $(size(A,1)) node graph"))
    o = one(eltype(A))
    res = zero(promote_type(eltype(A), Int))
    @inbounds for j in axes(A,2)
        j ≠ i && (res += A[j,i] * (o - A[i,j]))
    end
    return res
end

nonreciprocated_indegree(A::T) where T<:AbstractMatrix = [nonreciprocated_indegree(A, i) for i in axes(A,1)]

"""
    nonreciprocated_indegree(G::Graphs.AbstractGraph, i::Int)
    nonreciprocated_indegree(G::Graphs.AbstractGraph)

Compute the non-reciprocated indegree of node `i` in the directed graph `G` (``k^{←}_i = \\sum_{j≠i} a_{ji}(1 - a_{ij})``).
Without the node argument, the whole sequence is returned.

# Examples
```jldoctest
julia> G = maspalomas();

julia> nonreciprocated_indegree(G, 1)
1
```
"""
function nonreciprocated_indegree(G::T, i::Int) where T<:Graphs.AbstractGraph
    Graphs.is_directed(G) || throw(ArgumentError("The graph must be directed for reciprocity-based metrics"))
    return count(j -> j ≠ i && !Graphs.has_edge(G, i, j), Graphs.inneighbors(G, i))
end

nonreciprocated_indegree(G::T) where T<:Graphs.AbstractGraph = [nonreciprocated_indegree(G, i) for i in Graphs.vertices(G)]


"""
    reciprocated_degree(A::AbstractMatrix, i::Int)
    reciprocated_degree(A::AbstractMatrix)

Compute the reciprocated degree of node `i` from the adjacency matrix `A`, defined as
``k^{↔}_i = \\sum_{j≠i} a_{ij}a_{ji}``, i.e. the number of nodes `j` with links in both directions between `i` and `j`.
Without the node argument, the whole sequence is returned.

The computation is purely algebraic: for a 0/1 adjacency matrix this counts dyads, for a probability matrix
under dyadic independence it yields the expected value. Weighted adjacency matrices should be binarised first.

# Examples
```jldoctest
julia> A = [0 1 0; 1 0 1; 0 0 0];

julia> reciprocated_degree(A, 1)
1
```
"""
function reciprocated_degree(A::T, i::Int) where T<:AbstractMatrix
    size(A,1) == size(A,2) || throw(DimensionMismatch("The adjacency matrix must be square"))
    (i < 1 || i > size(A,1)) && throw(ArgumentError("Attempted to access node $i in a $(size(A,1)) node graph"))
    res = zero(promote_type(eltype(A), Int))
    @inbounds for j in axes(A,2)
        j ≠ i && (res += A[i,j] * A[j,i])
    end
    return res
end

reciprocated_degree(A::T) where T<:AbstractMatrix = [reciprocated_degree(A, i) for i in axes(A,1)]

"""
    reciprocated_degree(G::Graphs.AbstractGraph, i::Int)
    reciprocated_degree(G::Graphs.AbstractGraph)

Compute the reciprocated degree of node `i` in the directed graph `G` (``k^{↔}_i = \\sum_{j≠i} a_{ij}a_{ji}``).
Without the node argument, the whole sequence is returned.

# Examples
```jldoctest
julia> G = MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques());

julia> reciprocated_degree(G, 1)
2
```
"""
function reciprocated_degree(G::T, i::Int) where T<:Graphs.AbstractGraph
    Graphs.is_directed(G) || throw(ArgumentError("The graph must be directed for reciprocity-based metrics"))
    return count(j -> j ≠ i && Graphs.has_edge(G, j, i), Graphs.outneighbors(G, i))
end

reciprocated_degree(G::T) where T<:Graphs.AbstractGraph = [reciprocated_degree(G, i) for i in Graphs.vertices(G)]


"""
    nonreciprocated_outstrength(W::AbstractMatrix, i::Int)
    nonreciprocated_outstrength(W::AbstractMatrix)

Compute the non-reciprocated outstrength of node `i` from the weighted adjacency matrix `W`
(convention: `W[i,j]` is the weight of the link `i→j`), defined as
``s^{→}_i = \\sum_{j≠i} w_{ij} \\, \\mathbb{1}[a_{ij}(1-a_{ji})]``, i.e. the total weight on links from `i`
to nodes that do not point back. Without the node argument, the whole sequence is returned.

Note: the reverse-direction indicator uses `iszero`, so this is an *empirical* quantity;
for model-expected counterparts use the corresponding model methods.

# Examples
```jldoctest
julia> W = [0.0 2.5 0.0; 1.0 0.0 3.0; 0.0 0.0 0.0];

julia> nonreciprocated_outstrength(W, 2)
3.0
```
"""
function nonreciprocated_outstrength(W::T, i::Int) where T<:AbstractMatrix
    size(W,1) == size(W,2) || throw(DimensionMismatch("The adjacency matrix must be square"))
    (i < 1 || i > size(W,1)) && throw(ArgumentError("Attempted to access node $i in a $(size(W,1)) node graph"))
    res = zero(promote_type(eltype(W), Int))
    @inbounds for j in axes(W,2)
        j ≠ i && (res += W[i,j] * iszero(W[j,i]))
    end
    return res
end

nonreciprocated_outstrength(W::T) where T<:AbstractMatrix = [nonreciprocated_outstrength(W, i) for i in axes(W,1)]

"""
    nonreciprocated_outstrength(G::SimpleWeightedGraphs.AbstractSimpleWeightedGraph, i::Int)
    nonreciprocated_outstrength(G::SimpleWeightedGraphs.AbstractSimpleWeightedGraph)

Compute the non-reciprocated outstrength of node `i` in the weighted directed graph `G`
(``s^{→}_i``: total weight on links from `i` to nodes that do not point back).
Without the node argument, the whole sequence is returned.

# Examples
```jldoctest
julia> G = rhesus_macaques();

julia> nonreciprocated_outstrength(G, 6)
19.0
```
"""
function nonreciprocated_outstrength(G::T, i::Int) where T<:SimpleWeightedGraphs.AbstractSimpleWeightedGraph
    Graphs.is_directed(G) || throw(ArgumentError("The graph must be directed for reciprocity-based metrics"))
    S = G.weights # S[dst, src] = w(src → dst)
    res = zero(SimpleWeightedGraphs.weighttype(G))
    @inbounds for j in Graphs.outneighbors(G, i)
        j ≠ i && iszero(S[i,j]) && (res += S[j,i])
    end
    return res
end

nonreciprocated_outstrength(G::T) where T<:SimpleWeightedGraphs.AbstractSimpleWeightedGraph = [nonreciprocated_outstrength(G, i) for i in Graphs.vertices(G)]


"""
    nonreciprocated_instrength(W::AbstractMatrix, i::Int)
    nonreciprocated_instrength(W::AbstractMatrix)

Compute the non-reciprocated instrength of node `i` from the weighted adjacency matrix `W`
(convention: `W[i,j]` is the weight of the link `i→j`), defined as
``s^{←}_i = \\sum_{j≠i} w_{ji} \\, \\mathbb{1}[a_{ji}(1-a_{ij})]``, i.e. the total weight on links towards `i`
from nodes that `i` does not point back to. Without the node argument, the whole sequence is returned.

Note: the reverse-direction indicator uses `iszero`, so this is an *empirical* quantity;
for model-expected counterparts use the corresponding model methods.

# Examples
```jldoctest
julia> W = [0.0 2.5 0.0; 1.0 0.0 3.0; 0.0 0.0 0.0];

julia> nonreciprocated_instrength(W, 3)
3.0
```
"""
function nonreciprocated_instrength(W::T, i::Int) where T<:AbstractMatrix
    size(W,1) == size(W,2) || throw(DimensionMismatch("The adjacency matrix must be square"))
    (i < 1 || i > size(W,1)) && throw(ArgumentError("Attempted to access node $i in a $(size(W,1)) node graph"))
    res = zero(promote_type(eltype(W), Int))
    @inbounds for j in axes(W,2)
        j ≠ i && (res += W[j,i] * iszero(W[i,j]))
    end
    return res
end

nonreciprocated_instrength(W::T) where T<:AbstractMatrix = [nonreciprocated_instrength(W, i) for i in axes(W,1)]

"""
    nonreciprocated_instrength(G::SimpleWeightedGraphs.AbstractSimpleWeightedGraph, i::Int)
    nonreciprocated_instrength(G::SimpleWeightedGraphs.AbstractSimpleWeightedGraph)

Compute the non-reciprocated instrength of node `i` in the weighted directed graph `G`
(``s^{←}_i``: total weight on links towards `i` from nodes that `i` does not point back to).
Without the node argument, the whole sequence is returned.

# Examples
```jldoctest
julia> G = rhesus_macaques();

julia> nonreciprocated_instrength(G, 6)
0.0
```
"""
function nonreciprocated_instrength(G::T, i::Int) where T<:SimpleWeightedGraphs.AbstractSimpleWeightedGraph
    Graphs.is_directed(G) || throw(ArgumentError("The graph must be directed for reciprocity-based metrics"))
    S = G.weights # S[dst, src] = w(src → dst)
    res = zero(SimpleWeightedGraphs.weighttype(G))
    @inbounds for j in Graphs.inneighbors(G, i)
        j ≠ i && iszero(S[j,i]) && (res += S[i,j])
    end
    return res
end

nonreciprocated_instrength(G::T) where T<:SimpleWeightedGraphs.AbstractSimpleWeightedGraph = [nonreciprocated_instrength(G, i) for i in Graphs.vertices(G)]


"""
    reciprocated_outstrength(W::AbstractMatrix, i::Int)
    reciprocated_outstrength(W::AbstractMatrix)

Compute the reciprocated outstrength of node `i` from the weighted adjacency matrix `W`
(convention: `W[i,j]` is the weight of the link `i→j`), defined as
``s^{↔,out}_i = \\sum_{j≠i} w_{ij} \\, \\mathbb{1}[a_{ij}a_{ji}]``, i.e. the total weight on links from `i`
to nodes that also point back. Without the node argument, the whole sequence is returned.

Note: the reverse-direction indicator uses `iszero`, so this is an *empirical* quantity;
for model-expected counterparts use the corresponding model methods.

# Examples
```jldoctest
julia> W = [0.0 2.5 0.0; 1.0 0.0 3.0; 0.0 0.0 0.0];

julia> reciprocated_outstrength(W, 1)
2.5
```
"""
function reciprocated_outstrength(W::T, i::Int) where T<:AbstractMatrix
    size(W,1) == size(W,2) || throw(DimensionMismatch("The adjacency matrix must be square"))
    (i < 1 || i > size(W,1)) && throw(ArgumentError("Attempted to access node $i in a $(size(W,1)) node graph"))
    res = zero(promote_type(eltype(W), Int))
    @inbounds for j in axes(W,2)
        j ≠ i && (res += W[i,j] * !iszero(W[j,i]))
    end
    return res
end

reciprocated_outstrength(W::T) where T<:AbstractMatrix = [reciprocated_outstrength(W, i) for i in axes(W,1)]

"""
    reciprocated_outstrength(G::SimpleWeightedGraphs.AbstractSimpleWeightedGraph, i::Int)
    reciprocated_outstrength(G::SimpleWeightedGraphs.AbstractSimpleWeightedGraph)

Compute the reciprocated outstrength of node `i` in the weighted directed graph `G`
(``s^{↔,out}_i``: total weight on links from `i` to nodes that also point back).
Without the node argument, the whole sequence is returned.

# Examples
```jldoctest
julia> G = rhesus_macaques();

julia> reciprocated_outstrength(G, 6)
20.0
```
"""
function reciprocated_outstrength(G::T, i::Int) where T<:SimpleWeightedGraphs.AbstractSimpleWeightedGraph
    Graphs.is_directed(G) || throw(ArgumentError("The graph must be directed for reciprocity-based metrics"))
    S = G.weights # S[dst, src] = w(src → dst)
    res = zero(SimpleWeightedGraphs.weighttype(G))
    @inbounds for j in Graphs.outneighbors(G, i)
        j ≠ i && !iszero(S[i,j]) && (res += S[j,i])
    end
    return res
end

reciprocated_outstrength(G::T) where T<:SimpleWeightedGraphs.AbstractSimpleWeightedGraph = [reciprocated_outstrength(G, i) for i in Graphs.vertices(G)]


"""
    reciprocated_instrength(W::AbstractMatrix, i::Int)
    reciprocated_instrength(W::AbstractMatrix)

Compute the reciprocated instrength of node `i` from the weighted adjacency matrix `W`
(convention: `W[i,j]` is the weight of the link `i→j`), defined as
``s^{↔,in}_i = \\sum_{j≠i} w_{ji} \\, \\mathbb{1}[a_{ij}a_{ji}]``, i.e. the total weight on links towards `i`
from nodes that `i` also points to. Without the node argument, the whole sequence is returned.

Note: the reverse-direction indicator uses `iszero`, so this is an *empirical* quantity;
for model-expected counterparts use the corresponding model methods.

# Examples
```jldoctest
julia> W = [0.0 2.5 0.0; 1.0 0.0 3.0; 0.0 0.0 0.0];

julia> reciprocated_instrength(W, 1)
1.0
```
"""
function reciprocated_instrength(W::T, i::Int) where T<:AbstractMatrix
    size(W,1) == size(W,2) || throw(DimensionMismatch("The adjacency matrix must be square"))
    (i < 1 || i > size(W,1)) && throw(ArgumentError("Attempted to access node $i in a $(size(W,1)) node graph"))
    res = zero(promote_type(eltype(W), Int))
    @inbounds for j in axes(W,2)
        j ≠ i && (res += W[j,i] * !iszero(W[i,j]))
    end
    return res
end

reciprocated_instrength(W::T) where T<:AbstractMatrix = [reciprocated_instrength(W, i) for i in axes(W,1)]

"""
    reciprocated_instrength(G::SimpleWeightedGraphs.AbstractSimpleWeightedGraph, i::Int)
    reciprocated_instrength(G::SimpleWeightedGraphs.AbstractSimpleWeightedGraph)

Compute the reciprocated instrength of node `i` in the weighted directed graph `G`
(``s^{↔,in}_i``: total weight on links towards `i` from nodes that `i` also points to).
Without the node argument, the whole sequence is returned.

# Examples
```jldoctest
julia> G = rhesus_macaques();

julia> reciprocated_instrength(G, 6)
30.0
```
"""
function reciprocated_instrength(G::T, i::Int) where T<:SimpleWeightedGraphs.AbstractSimpleWeightedGraph
    Graphs.is_directed(G) || throw(ArgumentError("The graph must be directed for reciprocity-based metrics"))
    S = G.weights # S[dst, src] = w(src → dst)
    res = zero(SimpleWeightedGraphs.weighttype(G))
    @inbounds for j in Graphs.inneighbors(G, i)
        j ≠ i && !iszero(S[j,i]) && (res += S[i,j])
    end
    return res
end

reciprocated_instrength(G::T) where T<:SimpleWeightedGraphs.AbstractSimpleWeightedGraph = [reciprocated_instrength(G, i) for i in Graphs.vertices(G)]


"""
    reciprocity(A::AbstractMatrix)
    reciprocity(G::Graphs.AbstractGraph)

Compute the topological reciprocity of the directed network, defined as the ratio of reciprocated links to
the total number of links: ``r = L^{↔}/L = \\sum_{i≠j} a_{ij}a_{ji} / \\sum_{i≠j} a_{ij}``
(Di Vece et al. (2023), Eq. 1).

The matrix method is purely algebraic: for a 0/1 adjacency matrix it returns the observed reciprocity, for a
probability matrix under dyadic independence it returns the ratio-of-expectations approximation of the
expected reciprocity. Weighted adjacency matrices should be binarised first (or use [`weighted_reciprocity`](@ref)).

*Note*: `Graphs.jl` also exports a `reciprocity` function. When both packages are loaded with `using`,
the function must be qualified (`MaxEntropyGraphs.reciprocity` / `Graphs.reciprocity`).

# Examples
```jldoctest
julia> G = MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques());

julia> reciprocity(G)
0.7567567567567568
```
"""
function reciprocity(A::T) where T<:AbstractMatrix
    size(A,1) == size(A,2) || throw(DimensionMismatch("The adjacency matrix must be square"))
    num = zero(promote_type(eltype(A), Int))
    den = zero(promote_type(eltype(A), Int))
    @inbounds for i in axes(A,1)
        for j in axes(A,2)
            if i ≠ j
                num += A[i,j] * A[j,i]
                den += A[i,j]
            end
        end
    end
    iszero(den) && throw(ArgumentError("The network has no links, reciprocity is undefined"))
    return num / den
end

function reciprocity(G::T) where T<:Graphs.AbstractGraph
    Graphs.is_directed(G) || throw(ArgumentError("The graph must be directed for reciprocity-based metrics"))
    iszero(Graphs.ne(G)) && throw(ArgumentError("The network has no links, reciprocity is undefined"))
    L_recip = count(e -> Graphs.src(e) ≠ Graphs.dst(e) && Graphs.has_edge(G, Graphs.dst(e), Graphs.src(e)), Graphs.edges(G))
    return L_recip / Graphs.ne(G)
end


"""
    weighted_reciprocity(W::AbstractMatrix)
    weighted_reciprocity(G::SimpleWeightedGraphs.AbstractSimpleWeightedGraph)

Compute the weighted reciprocity of the directed network, defined as the share of the total weight that sits
on reciprocated links: ``r_w = W^{↔}/W_{tot} = \\sum_{i≠j} w_{ij}\\mathbb{1}[a_{ij}a_{ji}] / \\sum_{i≠j} w_{ij}``
(Di Vece et al. (2023), Eq. 2).

*Note*: this "share of total weight on reciprocated links" definition differs from the fully weighted
reciprocity of Squartini et al. (2013) ("Reciprocity of weighted networks"), which is based on the mutual
weight ``\\min(w_{ij}, w_{ji})``.

# Examples
```jldoctest
julia> G = rhesus_macaques();

julia> weighted_reciprocity(G)
0.8995363214837713
```
"""
function weighted_reciprocity(W::T) where T<:AbstractMatrix
    size(W,1) == size(W,2) || throw(DimensionMismatch("The adjacency matrix must be square"))
    num = zero(promote_type(eltype(W), Int))
    den = zero(promote_type(eltype(W), Int))
    @inbounds for i in axes(W,1)
        for j in axes(W,2)
            if i ≠ j
                num += W[i,j] * !iszero(W[j,i])
                den += W[i,j]
            end
        end
    end
    iszero(den) && throw(ArgumentError("The network has no links, reciprocity is undefined"))
    return num / den
end

function weighted_reciprocity(G::T) where T<:SimpleWeightedGraphs.AbstractSimpleWeightedGraph
    Graphs.is_directed(G) || throw(ArgumentError("The graph must be directed for reciprocity-based metrics"))
    S = G.weights # S[dst, src] = w(src → dst)
    num = zero(SimpleWeightedGraphs.weighttype(G))
    den = zero(SimpleWeightedGraphs.weighttype(G))
    @inbounds for i in Graphs.vertices(G)
        for j in Graphs.outneighbors(G, i)
            if j ≠ i
                num += S[j,i] * !iszero(S[i,j])
                den += S[j,i]
            end
        end
    end
    iszero(den) && throw(ArgumentError("The network has no links, reciprocity is undefined"))
    return num / den
end


"""
    ANND(G::T, i::Int; check_directed::Bool=true) where {T<:Graphs.AbstractGraph}

Compute the average nearest neighbor degree (ANND) for node `i` in graph `G`. The ANND for a node `i` is defined as
``
ANND_i(A^{*}) = \\frac{\\sum_{j=1}^{N} a_{ij}k_j }{k_i}
``
where ``a_{ij}`` denotes the element of the adjacency matrix ``A`` at row ``i`` and column ``j``, and ``k_i`` denotes the degree of node ``i``.

**Notes:** 
- the ANND is only defined for nodes with nonzero degree. If `degree(G,i) = 0`, then `ANND(G,i) = 0`.
- if `G` is a directed graph, by default an error is thrown because the `degree` function returns the incoming plus outgoing edges for node `i` in this case.
    This can be turned off by setting `check_directed=false`.


# Examples
```jldoctest ANND_graph_node_docs
julia> using Graphs

julia> G = smallgraph(:karate);

julia> ANND(G,1)
4.3125

```
```jldoctest ANND_graph_node_docs
julia> add_vertex!(G);

julia> ANND(G, nv(G))
0.0
```
```jldoctest ANND_graph_node_docs
julia> Gd = SimpleDiGraph(G);

julia> ANND(Gd,1)
ERROR: ArgumentError: The graph is directed. The degree function returns the incoming plus outgoing edges for node `i`. Consider using ANND_in or ANND_out instead.
[...]
```
```jldoctest ANND_graph_node_docs
julia> ANND(Gd,1, check_directed=false)
4.3125
```

See also: [`ANND_in`](@ref), [`ANND_out`](@ref), [`Graphs.degree`](https://juliagraphs.org/Graphs.jl/stable/core_functions/core/#Graphs.degree)
"""
function ANND(G::T, i::Int; check_directed::Bool=true, kwargs...) where {T<:Graphs.AbstractGraph}
    if check_directed && Graphs.is_directed(G)
        throw(ArgumentError("The graph is directed. The degree function returns the incoming plus outgoing edges for node `i`. Consider using ANND_in or ANND_out instead."))
    end

    if iszero(Graphs.degree(G,i))
        return zero(Float64)
    else
        return mapreduce(x -> Graphs.degree(G, x), +, Graphs.neighbors(G, i), init=zero(Float64)) / Graphs.degree(G, i)
    end
end


"""
    ANND(G::T, vs=vertices(G); check_directed::Bool=true) where {T<:Graphs.AbstractGraph}

Return a vector correcponding to the average nearest neighbor degree (ANND) all nodes in the graph `G`. 
If v is specified, only return the ANND for nodes in v. The ANND for a node `i` is defined as 
``
ANND_i(A^{*}) = \\frac{\\sum_{j=1}^{N} a_{ij}k_j }{k_i}
``
where ``a_{ij}`` denotes the element of the adjacency matrix ``A`` at row ``i`` and column ``j``, and ``k_i`` denotes the degree of node ``i``.

**Notes:** 
- the ANND is only defined for nodes with nonzero degree. If `degree(G,i) = 0`, then `ANND(G,i) = 0`.
- if `G` is a directed graph, by default an error is thrown because the `degree` function returns the incoming plus outgoing edges for node `i` in this case.
This can be turned off by setting `check_directed=false`. This check is only performed once the actual computing.


# Examples
```jldoctest ANND_graph_docs
julia> using Graphs

julia> G = smallgraph(:karate);

julia> ANND(G,[10; 20; 30])
3-element Vector{Float64}:
 13.5
 14.0
  9.0

```
```jldoctest ANND_graph_docs
julia> Gd = SimpleDiGraph(G);

julia> ANND(Gd,[10; 20; 30]);
ERROR: ArgumentError: The graph is directed. The degree function returns the incoming plus outgoing edges for node `i`. Consider using ANND_in or ANND_out instead.
[...]
```
```jldoctest ANND_graph_docs
julia> ANND(Gd,[10; 20; 30], check_directed=false)
3-element Vector{Float64}:
 13.5
 14.0
  9.0

```

See also: `ANND_in`, `ANND_out`, [`Graphs.degree`](https://juliagraphs.org/Graphs.jl/stable/core_functions/core/#Graphs.degree)
"""
function ANND(G::T, vs=Graphs.vertices(G); check_directed::Bool=true, kwargs...) where {T<:Graphs.AbstractGraph}
    # check only once before computing the rest
    if check_directed && Graphs.is_directed(G)
        throw(ArgumentError("The graph is directed. The degree function returns the incoming plus outgoing edges for node `i`. Consider using ANND_in or ANND_out instead."))
    end

    return [ANND(G,i, check_directed=false) for i in vs]
end




# Elementwise ANND ratio `num[i]/deg[i]` following the convention that a zero-degree node maps to zero.
# Non-mutating and using instance-based `zero`/`oneunit` (not type-level) so it stays exact for integer
# matrices and differentiable (ForwardDiff/ReverseDiff/Zygote) on the `σₓ` autodiff path; for a solved model
# all degrees are > 0, so the guard branch is never taken there.
_annd_ratio(num::AbstractVector, deg::AbstractVector, vs) = [_ratio_or_zero(num[i], deg[i]) for i in vs]
_ratio_or_zero(n, d) = iszero(d) ? zero(n) / oneunit(d) : n / d

"""
    ANND(A::T, i::Int; check_dimensions::Bool=true, check_directed::Bool=true) where {T<:AbstractMatrix}

Compute the average nearest neighbor degree (ANND) for node `i` using adjacency matrix `A`. The ANND for a node `i` is defined as
``
ANND_i(A^{*}) = \\frac{\\sum_{j=1}^{N} a_{ij}k_j }{k_i}
``
where ``a_{ij}`` denotes the element of the adjacency matrix ``A`` at row ``i`` and column ``j``, and ``k_i`` denotes the degree of node ``i``.

**Notes:** 
- this function is intented for use with the expected adjacency matrix of a `::AbstractMaxEntropyModel` model. A separate method exists for `::AbstractGraph` objects.
- if `A` is not symmetrical, you have a directed graph, and this will throw an error by default. This can be turned off by setting `check_directed=false`.
- the adjacency matrix should be square, if not, this will throw an error by default. This can be turned off by setting `check_dimensions=false`.

# Examples
```jldoctest ANND_mat_node_docs
julia> using Graphs

julia> G = smallgraph(:karate);

julia> A = adjacency_matrix(G);

julia> ANND(A, 1)
4.3125

```
```jldoctest ANND_mat_node_docs
julia> Gd = SimpleDiGraph(G);

julia> add_vertex!(Gd); add_edge!(Gd, 1, nv(Gd));

julia> Ad = adjacency_matrix(Gd);

julia> ANND(Ad, 1)
ERROR: ArgumentError: The matrix is not symmetrical. Consider using ANND_in or ANND_out instead.
[...]
```
```jldoctest ANND_mat_node_docs
julia> ANND(Ad, 1, check_directed=false)
4.375

```
```jldoctest ANND_mat_node_docs
julia> ANND(rand(2,3),1)
ERROR: DimensionMismatch: `A` must be a square matrix.
[...]
```

See also: `ANND_in`, `ANND_out`, [`Graphs.degree`](https://juliagraphs.org/Graphs.jl/stable/core_functions/core/#Graphs.degree)
"""
function ANND(A::T, i::Int; check_dimensions::Bool=true, check_directed::Bool=true) where {T<:AbstractMatrix}
    # checks
    if check_dimensions && !isequal(size(A)...) 
        throw(DimensionMismatch("`A` must be a square matrix."))
    end
    if check_directed && !issymmetric(A) 
        throw(ArgumentError( "The matrix is not symmetrical. Consider using ANND_in or ANND_out instead."))
    end

    # compute
    if iszero(sum(@view A[:,i]))
        return zero(Float64)
    else
        return mapreduce(x -> A[i,x] * sum(@view A[:,x]), +, 1:size(A,1), init=zero(Float64)) / sum(@view A[:,i])
    end
end

"""
    ANND(A::T, vs=1:size(A,1); check_dimensions::Bool=true, check_directed::Bool=true) where {T<:AbstractMatrix}

Return a vector correcponding to the average nearest neighbor degree (ANND) all nodes in the graph with adjacency matrix `A`. 
If v is specified, only return the ANND for nodes in v. The ANND for a node `i` is defined as 
``
ANND_i(A^{*}) = \\frac{\\sum_{j=1}^{N} a_{ij}k_j }{k_i}
``
where ``a_{ij}`` denotes the element of the adjacency matrix ``A`` at row ``i`` and column ``j``, and ``k_i`` denotes the degree of node ``i``.

**Notes:** 
- this function is intented for use with the expected adjacency matrix of a `::AbstractMaxEntropyModel` model. A separate method exists for `::AbstractGraph` objects.
- if `A` is not symmetrical, you have a directed graph, and this will throw an error by default. This can be turned off by setting `check_directed=false`.
- the adjacency matrix should be square, if not, this will throw an error by default. This can be turned off by setting `check_dimensions=false`.

# Examples
```jldoctest ANND_mat_docs
julia> using Graphs

julia> G = smallgraph(:karate);

julia> A = adjacency_matrix(G);

julia> ANND(A);

```
```jldoctest ANND_mat_docs
julia> Gd = SimpleDiGraph(G);

julia> add_vertex!(Gd); add_edge!(Gd, 1, nv(Gd));

julia> Ad = adjacency_matrix(Gd);

julia> ANND(Ad)
ERROR: ArgumentError: The matrix is not symmetrical. Consider using ANND_in or ANND_out instead.
[...]
```
```jldoctest ANND_mat_docs
julia> ANND(Ad, check_directed=false)[1]
4.375

```
```jldoctest ANND_mat_docs
julia> ANND(rand(2,3))
ERROR: DimensionMismatch: `A` must be a square matrix.
[...]
```

See also: `ANND_in`, `ANND_out`, [`Graphs.degree`](https://juliagraphs.org/Graphs.jl/stable/core_functions/core/#Graphs.degree)
"""
function ANND(A::T, vs=1:size(A,1); check_dimensions::Bool=true, check_directed::Bool=true) where {T<:AbstractMatrix}
    # checks
    if check_dimensions && !isequal(size(A)...)
        throw(DimensionMismatch("`A` must be a square matrix."))
    end
    if check_directed && !issymmetric(A)
        throw(ArgumentError( "The matrix is not symmetrical. Consider using ANND_in or ANND_out instead."))
    end

    # compute: ANND_i = (Σ_j A[i,j] k_j) / k_i with k = column sums (degrees) -> a single gemv (O(N²))
    k = vec(sum(A, dims=1))
    return _annd_ratio(A * k, k, vs)
end



function ANND_out(G::T, i::Int; kwargs...) where {T<:Graphs.AbstractGraph}
    if iszero(Graphs.outdegree(G,i))
        return zero(Float64)
    else
        return mapreduce(x -> Graphs.outdegree(G, x), +, Graphs.neighbors(G, i), init=zero(Float64)) / Graphs.outdegree(G, i)
    end
end

ANND_out(G::T, v::Vector{Int}=collect(Graphs.vertices(G)); kwargs...) where {T<:Graphs.AbstractGraph} = [ANND_out(G, i; kwargs...) for i in v]


"""
ANND_out(A::T, vs=1:size(A,1); check_dimensions::Bool=true, check_directed::Bool=true) where {T<:AbstractMatrix}

Return a vector corresponding to the average nearest neighbor outdegree (ANND) all nodes in the graph with adjacency matrix `A`. 
If v is specified, only return the ANND_out for nodes in v. 


See also: [`ANND_in`](@ref), [`ANND`](@ref)
"""
function ANND_out(A::T, i::Int; check_dimensions::Bool=true) where {T<:AbstractMatrix}
    # checks
    if check_dimensions && !isequal(size(A)...) 
        throw(DimensionMismatch("`A` must be a square matrix."))
    end

    # compute
    if iszero(sum(@view A[i,:]))
        return zero(Float64)
    else
        return mapreduce(x -> A[i,x] * sum(@view A[x,:]), +, 1:size(A,1), init=zero(Float64)) / sum(@view A[i,:])
    end
end

function ANND_out(A::T, vs=1:size(A,1); check_dimensions::Bool=true) where {T<:AbstractMatrix}
    # checks
    if check_dimensions && !isequal(size(A)...)
        throw(DimensionMismatch("`A` must be a square matrix."))
    end

    # compute: ANND_out uses out-degrees (row sums) both in numerator and denominator -> a single gemv
    k = vec(sum(A, dims=2))
    return _annd_ratio(A * k, k, vs)
end


function ANND_in(G::T, i::Int; kwargs...) where {T<:Graphs.AbstractGraph}
    if iszero(Graphs.indegree(G,i))
        return zero(Float64)
    else
        return mapreduce(x -> Graphs.indegree(G, x), +, Graphs.neighbors(G, i), init=zero(Float64)) / Graphs.indegree(G, i)
    end
end

ANND_in(G::T, v::Vector{Int}=collect(Graphs.vertices(G)); kwargs...) where {T<:Graphs.AbstractGraph} = [ANND_in(G, i; kwargs...) for i in v]


"""
ANND_in(A::T, vs=1:size(A,1); check_dimensions::Bool=true, check_directed::Bool=true) where {T<:AbstractMatrix}

Return a vector corresponding to the average nearest neighbor indegree (ANND) all nodes in the graph with adjacency matrix `A`. 
If v is specified, only return the ANND_in for nodes in v. 


See also: [`ANND_out`](@ref), [`ANND`](@ref)
"""
function ANND_in(A::T, i::Int; check_dimensions::Bool=true) where {T<:AbstractMatrix}
    # checks
    if check_dimensions && !isequal(size(A)...) 
        throw(DimensionMismatch("`A` must be a square matrix."))
    end

    # compute
    if iszero(sum(@view A[:,i]))
        return zero(Float64)
    else
        return mapreduce(x -> A[i,x] * sum(@view A[:,x]), +, 1:size(A,1), init=zero(Float64)) / sum(@view A[:,i])
    end
end

function ANND_in(A::T, vs=1:size(A,1); check_dimensions::Bool=true) where {T<:AbstractMatrix}
    # checks
    if check_dimensions && !isequal(size(A)...)
        throw(DimensionMismatch("`A` must be a square matrix."))
    end

    # compute: ANND_in uses in-degrees (column sums) both in numerator and denominator -> a single gemv
    k = vec(sum(A, dims=1))
    return _annd_ratio(A * k, k, vs)
end

function ANND(m::UBCM)
    # checks
    m.status[:G_computed] ? nothing : throw(ArgumentError("The expected values (m.Ĝ) must be computed for `m` before computing the standard deviation of metric `X`, see `set_Ĝ!`"))

    # compute (m.Ĝ is square & symmetric by construction, so skip the redundant O(N²) checks)
    return ANND(m.Ĝ, check_dimensions=false, check_directed=false)
end

function ANND_in(m::DBCM)
    # checks
    m.status[:G_computed] ? nothing : throw(ArgumentError("The expected values (m.Ĝ) must be computed for `m` before computing the standard deviation of metric `X`, see `set_Ĝ!`"))

    # compute
    return ANND_in(m.Ĝ, check_dimensions=false)
end

function ANND_out(m::DBCM)
    # checks
    m.status[:G_computed] ? nothing : throw(ArgumentError("The expected values (m.Ĝ) must be computed for `m` before computing the standard deviation of metric `X`, see `set_Ĝ!`"))

    # compute
    return ANND_out(m.Ĝ, check_dimensions=false)
end


"""
    wedges(G::Graphs.SimpleGraph)
    wedges(A::T; check_dimensions::Bool=true, check_directed::Bool=true) where {T<:AbstractMatrix}
    wedges(m::UBCM)

Compute the number of (expected) wedges for an undirected graph. Can be done directly from the graph, based on the adjacency matrix or a UBCM model.

# Arguments
For the adjacency matrix `A`, the following arguments can be passed:
- `check_dimensions`: if true, check that `A` is a square matrix, otherwise throw an error.
- `check_directed`: if true, check that `A` is symmetrical, otherwise throw an error.

# Examples
```jldoctest wedges_graph_docs
julia> G = MaxEntropyGraphs.Graphs.smallgraph(:karate);

julia> model = UBCM(G);

julia> solve_model!(model);

julia> set_Ĝ!(model);

julia> (wedges(G), wedges(MaxEntropyGraphs.Graphs.adjacency_matrix(G)), wedges(model))
(528.0, 528.0, 528.0000011499742)
```
"""
function wedges end

function wedges(G::Graphs.SimpleGraph) 
    d =  Graphs.degree(G)
    return sum(d .* (d .- one(eltype(d))) ./ 2)
end

function wedges(A::T; check_dimensions::Bool=true, check_directed::Bool=true) where {T<:AbstractMatrix}
    # checks
    if check_dimensions && !isequal(size(A)...) 
        throw(DimensionMismatch("`A` must be a square matrix."))
    end
    if check_directed && !issymmetric(A) 
        throw(ArgumentError( "The matrix is not symmetrical. Consider using ANND_in or ANND_out instead."))
    end

    # compute
    d = sum(A, dims=1)

    return sum(d .* (d .- one(eltype(A))) ./ 2)
end

function wedges(m::UBCM)
    # checks
    m.status[:G_computed] ? nothing : throw(ArgumentError("The expected values (m.Ĝ) must be computed for `m` before computing the standard deviation of metric `X`, see `set_Ĝ!`"))

    # compute
    return wedges(m.Ĝ)
end

"""
    triangles(G::Graphs.SimpleGraph)
    triangles(A::T; check_dimensions::Bool=true, check_directed::Bool=true) where {T<:AbstractMatrix}
    triangles(m::UBCM)

Compute the number of (expected) triangles for an undirected graph. Can be done directly from the graph, based on the adjacency matrix or a UBCM model.

# Arguments
For the adjacency matrix `A`, the following arguments can be passed:
- `check_dimensions`: if true, check that `A` is a square matrix, otherwise throw an error.
- `check_directed`: if true, check that `A` is symmetrical, otherwise throw an error.
These checks can be turned off for perfomance reasons.

# Examples
```jldoctest triangles_doc_all
julia> G = MaxEntropyGraphs.Graphs.smallgraph(:karate);

julia> model = UBCM(G);

julia> solve_model!(model);

julia> set_Ĝ!(model);

julia> (triangles(G), triangles(MaxEntropyGraphs.Graphs.adjacency_matrix(G)), round(triangles(model), digits=6))
(45, 45.0, 52.849301)
```
"""
function triangles end 

triangles(G::Graphs.SimpleGraph) = sum(Graphs.triangles(G)) ÷ 3

function triangles(A::T; check_dimensions::Bool=true, check_directed::Bool=true) where {T<:AbstractMatrix}
    # checks
    if check_dimensions && !isequal(size(A)...)
        throw(DimensionMismatch("`A` must be a square matrix."))
    end
    if check_directed && !issymmetric(A)
        throw(ArgumentError( "The matrix is not symmetrical. Consider using `M13` instead."))
    end

    return _triangles(A)
end

# Number of (expected) triangles = tr(A³)/6. For a zero-diagonal matrix every index coincidence in
# tr(A³)=Σ_{i,j,k} A_ij A_jk A_ki forces a diagonal factor A_ii = 0, so this equals the distinct-index
# sum exactly. `dot(A, A*A)` = Σ_{i,k} A_ik (A²)_ik = tr(A³) (A symmetric). This generic form is BLAS-backed
# for `Matrix{Float64}` and differentiable (ReverseDiff/ForwardDiff/Zygote) for tracked/Dual eltypes — it is
# the autodiff path used by `σₓ`.
_triangles(A::AbstractMatrix) = dot(A, A * A) / 6

# Memory-frugal path for concrete BLAS floats: stream tr(A³) = Σ_i A[i,:]·(A·A[:,i]) column by column, so the
# peak extra memory is O(N) (one gemv result) instead of the O(N²) of a materialised A*A — this is what keeps
# the batch/threaded-over-graphs workload from multiplying an N×N temporary by the thread count. The gemv is
# non-mutating (no `mul!`) so Zygote — which differentiates the concrete method directly — stays happy; the
# form is also correct without symmetry. (ReverseDiff/ForwardDiff use tracked eltypes and take the generic path.)
function _triangles(A::AbstractMatrix{T}) where {T<:BLAS.BlasFloat}
    res = zero(T)
    @inbounds for i in axes(A, 2)
        res += dot(A[i, :], A * A[:, i])
    end
    return res / 6
end

# does this need additional checks/allow for on-the-fly computation?
function triangles(m::UBCM)
    # check if Ĝ is computed
    m.status[:G_computed] ? nothing : throw(ArgumentError("The expected values (m.Ĝ) must be computed for `m` before computing triangles, see `set_Ĝ!`"))
    
    return triangles(m.Ĝ, check_dimensions=false, check_directed=false)
end

"""
    squares(G::Graphs.SimpleGraph)
    squares(A::T; check_dimensions::Bool=true, check_directed::Bool=true) where {T<:AbstractMatrix}
    squares(m::UBCM)

Compute the number of (expected) squares for an undirected graph. Can be done directly from the graph, based on the adjacency matrix or a UBCM model.

# Notes:
In this function, by ``square``, a \'pure\' square is understood, without any diagonals inside. This explains the difference with the induced subgraph count, which counts all squares, including those with triangles inside. 

# Arguments
For the adjacency matrix `A`, the following arguments can be passed:
- `check_dimensions`: if true, check that `A` is a square matrix, otherwise throw an error.
- `check_directed`: if true, check that `A` is symmetrical, otherwise throw an error.
These checks can be turned off for perfomance reasons.

# Examples
```jldoctest squares_doc_all
julia> G = MaxEntropyGraphs.Graphs.smallgraph(:karate);

julia> model = UBCM(G);

julia> solve_model!(model);

julia> set_Ĝ!(model);

julia> (squares(G), squares(MaxEntropyGraphs.Graphs.adjacency_matrix(G)), round(squares(model), digits=6))
(36.0, 36.0, 45.644737)
```
"""
function squares end

function squares(G::Graphs.SimpleGraph) 
    res = 0
    for i in Graphs.vertices(G)
        # only valid candidates
        if Graphs.degree(G, i) >= 2
            for (k,j) in combinations(Graphs.neighbors(G, i), 2)
                # edge betweenj and k should not exist
                if !Graphs.has_edge(G, j, k)
                    # search for common neighbors
                    for l in intersect(Graphs.neighbors(G, j), Graphs.neighbors(G, k))
                        if l ≠ i && !Graphs.has_edge(G, i, l)
                            res += 1
                        end
                    end
                end
            end
        end
    end
    return res / 4
end

function squares(A::T; check_dimensions::Bool=true, check_directed::Bool=true) where {T<:AbstractMatrix}
    # checks
    if check_dimensions && !isequal(size(A)...)
        throw(DimensionMismatch("`A` must be a square matrix."))
    end
    if check_directed && !issymmetric(A)
        throw(ArgumentError( "The matrix is not symmetrical, ``squares`` is only defined for undirected graphs."))
    end

    return _squares(A)
end

# Number of (expected) "pure" squares (4-cycles with both chords absent):
#   (1/8) Σ_{distinct i,j,k,l} A_ij A_jk A_kl A_li (1-A_ik)(1-A_lj).
# The summand is invariant under the labelled 4-cycle's 8-fold (dihedral D₄) symmetry, so we sum exactly one
# representative per orbit — i as the minimum index, its two cycle-neighbours j<l, and the opposite corner k —
# and drop the 1/8. This is ~8× fewer iterations than the naive quadruple loop, uses O(1) extra memory, is
# integer-exact for 0/1 matrices, and stays differentiable: the branches depend only on indices or on *exact*
# zeros, and on the σₓ path Ĝ ∈ (0,1) so no term is ever pruned (no gradient contribution is dropped).
# NOTE: no sub-O(N⁴) closed form exists (the count contains a K4-homomorphism term), so this is a constant-
# factor speedup; large *sparse* 0/1 graphs are handled by the neighbour-enumeration fast path below.
# Dihedral (D₄) orbit-reduction proof: performance/metrics_acceleration.tex §6.1.
function _squares(A::AbstractMatrix)
    n = size(A, 1)
    o = one(eltype(A))
    res = zero(eltype(A))
    @inbounds for i in 1:n
        for j in (i+1):n
            Aij = A[i, j]
            for l in (j+1):n
                base = Aij * A[l, i] * (o - A[l, j])   # factors independent of k
                iszero(base) && continue
                for k in (i+1):n
                    (k == j || k == l) && continue
                    res += base * A[j, k] * A[k, l] * (o - A[i, k])
                end
            end
        end
    end
    return res
end

# Sparse 0/1 fast path: the dense kernel's scalar indexing is O(log) per access on a sparse matrix, so instead
# reuse the neighbour-enumeration graph algorithm (equivalent: squares(G) == squares(adjacency_matrix(G))).
_squares(A::SparseMatrixCSC{<:Union{Bool,Integer}}) = squares(Graphs.SimpleGraph(A))

# does this need additional checks/allow for on-the-fly computation?
function squares(m::UBCM)
    # check if Ĝ is computed
    m.status[:G_computed] ? nothing : throw(ArgumentError("The expected values (m.Ĝ) must be computed for `m` before computing squares, see `set_Ĝ!`"))

    return squares(m.Ĝ, check_dimensions=false, check_directed=false)
end


########################################################################################################
# directed network motifs and helper functions
########################################################################################################
# building blocks used for the motif computation (adjecency matrix based)
"""
    a⭢(A::T, i::Int, j::Int) where T<:AbstractArray

Compute non-recipocrated directed link from i to j and not from j to i.
"""
a⭢(A::T, i::Int, j::Int) where T<:AbstractArray = @inbounds A[i,j] * (one(eltype(T)) - A[j,i])

"""
    a⭠(A::T, i::Int, j::Int) where T<:AbstractArray

Computed non-recipocrated directed link not from i to j and  from j to i.
"""
a⭠(A::T, i::Int, j::Int) where T<:AbstractArray = @inbounds (one(eltype(T)) - A[i,j]) * A[j,i]

"""
    a⭤(A::T, i::Int, j::Int) where T<:AbstractArray

Computed recipocrated directed link between i and j.
"""
a⭤(A::T, i::Int, j::Int) where T<:AbstractArray = @inbounds A[i,j]*A[j,i]

"""
    a̲(A::T, i::Int, j::Int) where T<:AbstractArray

Compute absence of link between i and j.
"""
a̲(A::T, i::Int, j::Int)   where T<:AbstractArray = @inbounds (one(eltype(T)) - A[i,j])*(one(eltype(T)) - A[j,i])

# The 13 directed 3-node motifs based on the combination of (i,j,k)
const directed_graph_motif_functions = [ (a⭠, a⭢, a̲);
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
const directed_graph_motif_function_names = [Symbol("M$(i)") for i = 1:13]

# ---- fast matrix reformulation of the directed 3-node motif counts --------------------------------------
# A motif count is Σ_{i≠j≠k} f₁(i,j)·f₂(j,k)·f₃(k,i) with fₓ ∈ {a⭢, a⭠, a⭤, a̲}. Elementwise these indicators
# are entries of four matrices built from A and its transpose:
#   P = A .* (1 .- Aᵀ)      (a⭢, i→j only)          Q = transpose(P)          (a⭠, j→i only)
#   R = A .* Aᵀ            (a⭤, reciprocated)       Z = (1 .- A) .* (1 .- Aᵀ)  (a̲, absent)
# A has zero diagonal, so only Z has a nonzero (unit) diagonal. The distinct-index inclusion–exclusion gives
# Σ_{i≠j≠k} F1_ij F2_jk F3_ki = tr(F1 F2 F3) − (a single correction trace at the a̲ factor, if any); no motif
# has two a̲ factors, so exactly one correction (or none) applies. These forms are BLAS-backed for Float64 and
# differentiable (ReverseDiff/ForwardDiff/Zygote), serving both the value path and the σₓ gradient path.
# Proof of equivalence to the naive triple loop: performance/metrics_acceleration.tex §5 (and Table 1 there).
function _motif_base_matrices(A::AbstractMatrix)
    At = transpose(A)
    o  = one(eltype(A))
    P  = A .* (o .- At)
    R  = A .* At
    Z  = (o .- A) .* (o .- At)
    return P, R, Z
end

# tr(F1 F2 F3) minus the diagonal correction. `zpos` ∈ (0,1,2,3) is the position of the a̲ (absent-link)
# factor whose diagonal is 1; 0 means none. tr(XY) = dot(X, transpose(Y)); tr(XYW) = dot(X*Y, transpose(W)).
function _motif_count(F1, F2, F3, zpos::Int)
    main = dot(F1 * F2, transpose(F3))
    zpos == 1 && return main - dot(F2, transpose(F3))   # − tr(F2 F3)
    zpos == 2 && return main - dot(F3, transpose(F1))   # − tr(F3 F1)
    zpos == 3 && return main - dot(F1, transpose(F2))   # − tr(F1 F2)
    return main
end

const _motif_label = IdDict(a⭢ => :P, a⭠ => :Q, a⭤ => :R, a̲ => :Z)
_motif_select(lbl::Symbol, P, Q, R, Z) = lbl === :P ? P : lbl === :Q ? Q : lbl === :R ? R : Z

# use metaprogramming to generate the functions for the 13 directed 3-node motifs for a directed graph
for i = 1:13
    fname = directed_graph_motif_function_names[i]
    (l1, l2, l3) = map(f -> _motif_label[f], directed_graph_motif_functions[i])
    zp = something(findfirst(==(:Z), (l1, l2, l3)), 0)
    @eval begin
        # method based on matrix
        """
            $($fname)(A::T) where T<:AbstractArray
        
        Count the occurence of motif $($fname) (Σ_{i≠j≠k} $(directed_graph_motif_functions[$i][1])(i,j) $(directed_graph_motif_functions[$i][2])(j,k) $(directed_graph_motif_functions[$i][3])(k,i) ) from the adjacency matrix.
        """
        function $(fname)(A::T)  where T<:AbstractArray
            P, R, Z = _motif_base_matrices(A)
            Q = transpose(P)
            return _motif_count($(l1), $(l2), $(l3), $(zp))
        end

        # method for DBCM > refer to underlying matrix
        """
            $($fname)(m::DBCM)
        
        Count the occurence of motif $($fname) (Σ_{i≠j≠k} $(directed_graph_motif_functions[$i][1]) (i,j) × $(directed_graph_motif_functions[$i][2]) (j,k) × $(directed_graph_motif_functions[$i][3]) (k,i) ) from the `DBCM` model.
        """
        function $(fname)(m::DBCM)
            m.status[:G_computed] ? nothing : throw(ArgumentError("The expected values (m.Ĝ) must be computed for `m` before counting the occurence of motif $($fname), see `set_Ĝ!`"))
            
            return $(fname)(m.Ĝ)
        end

        # method for RBCM > use the dyadic probability matrices, NOT the expected adjacency matrix: within a
        # dyad aᵢⱼ and aⱼᵢ are correlated under the RBCM, so evaluating the motif on the expected adjacency
        # matrix would be wrong. With the dyadic matrices the result is the EXACT expectation (Squartini &
        # Garlaschelli (2011), Eq. C.16): a motif's three dyads are distinct, and distinct dyads are independent.
        """
            $($fname)(m::RBCM)

        Compute the exact expected occurrence of motif $($fname) (Σ_{i≠j≠k} $(directed_graph_motif_functions[$i][1]) (i,j) × $(directed_graph_motif_functions[$i][2]) (j,k) × $(directed_graph_motif_functions[$i][3]) (k,i) ) under the `RBCM` model, evaluated from the dyadic probability matrices (p⭢, p⭠, p⭤, p∅).
        """
        function $(fname)(m::RBCM)
            m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters of `m` must be computed before counting the occurence of motif $($fname), see `solve_model!`"))
            P, R, Z = _dyadic_probability_matrices(m)
            Q = transpose(P)
            return _motif_count($(l1), $(l2), $(l3), $(zp))
        end
    end
end

# (label₁, label₂, label₃, zpos) for each motif, used by the batched `motifs` spectrum below
const _motif_specs = Tuple(begin
    (l1, l2, l3) = map(f -> _motif_label[f], directed_graph_motif_functions[i])
    (l1, l2, l3, something(findfirst(==(:Z), (l1, l2, l3)), 0))
end for i in 1:13)

"""
    motifs(A::AbstractMatrix)
    motifs(m::DBCM)
    motifs(m::RBCM)

Count all 13 directed 3-node motifs at once, returning the vector `[M1, M2, …, M13]`.

The four base matrices (`P, Q, R, Z`) are built once and shared across the whole spectrum, so this is faster
than calling `M1`…`M13` individually (which each rebuild them). Results are identical to the individual methods.

For the `RBCM` the base matrices are the model's dyadic probability matrices (p⭢, p⭠, p⭤, p∅) rather than
functions of `Ĝ` (within a dyad `aᵢⱼ` and `aⱼᵢ` are correlated under the RBCM), which makes the result the
**exact** expected motif spectrum ⟨M1⟩, …, ⟨M13⟩ (Squartini & Garlaschelli (2011), Eq. C.16).

# Examples
```jldoctest motifs_doc
julia> model = DBCM(MaxEntropyGraphs.maspalomas());

julia> solve_model!(model); set_Ĝ!(model);

julia> motifs(model) == [Mᵢ(model) for Mᵢ in (M1,M2,M3,M4,M5,M6,M7,M8,M9,M10,M11,M12,M13)]
true
```
"""
function motifs(A::AbstractMatrix)
    P, R, Z = _motif_base_matrices(A)
    Q = transpose(P)
    return [_motif_count(_motif_select(s[1], P, Q, R, Z),
                         _motif_select(s[2], P, Q, R, Z),
                         _motif_select(s[3], P, Q, R, Z), s[4]) for s in _motif_specs]
end

function motifs(m::DBCM)
    m.status[:G_computed] ? nothing : throw(ArgumentError("The expected values (m.Ĝ) must be computed for `m` before counting motifs, see `set_Ĝ!`"))

    return motifs(m.Ĝ)
end

function motifs(m::RBCM)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters of `m` must be computed before counting motifs, see `solve_model!`"))
    P, R, Z = _dyadic_probability_matrices(m)
    Q = transpose(P)
    return [_motif_count(_motif_select(s[1], P, Q, R, Z),
                         _motif_select(s[2], P, Q, R, Z),
                         _motif_select(s[3], P, Q, R, Z), s[4]) for s in _motif_specs]
end


# ---- triadic fluxes -----------------------------------------------------------------------------------
# The triadic flux of motif m is the total weight sitting on the links of all its occurrences
# (Di Vece et al. (2023), the Fₘ statistic):
#   F_m = Σ_{i≠j≠k} f₁(i,j) f₂(j,k) f₃(k,i) × [w(dyad₁) + w(dyad₂) + w(dyad₃)]
# where an a⭢ dyad carries wᵢⱼ, an a⭠ dyad carries wⱼᵢ, an a⭤ dyad carries wᵢⱼ + wⱼᵢ, and an absent
# dyad carries nothing. Because the bracket splits by position, F_m is a sum of at most three trace terms,
# each equal to `_motif_count` with exactly ONE binary factor replaced by its weighted counterpart:
#   Pw = W ∘ (1 - Aᵀ)  (weight on an a⭢ factor),  Qw = transpose(Pw)  (an a⭠(i,j) factor carries wⱼᵢ),
#   Rw = W ∘ Aᵀ + Wᵀ ∘ A  (both directions of an a⭤ factor);   the a̲ position is skipped (zero weight).
# The zpos diagonal correction inside `_motif_count` stays valid (the weighted matrices share the binary
# matrices' zero diagonal, Z keeps its unit diagonal). For the model methods the same trace forms hold with
# the expected matrices — since a motif's three dyads are distinct and dyads are independent, the
# expectation factorises per position and the result is the EXACT ⟨F_m⟩:
#   - DCReM: entries are independent, so P̂w = Ŵ ∘ (1 - Ĝᵀ), R̂w = Ŵ ∘ Ĝᵀ + Ŵᵀ ∘ Ĝ (the same algebra
#     applied to the expected matrices),
#   - CRWCM: within-dyad correlation requires the dyadic forms P̂w[i,j] = f⭢ᵢⱼ/(θ⭢ᵢ+θ⭠ⱼ) = E[wᵢⱼ·𝟙⭢] and
#     R̂w[i,j] = f⭤ᵢⱼ·[1/(θ⭤ᵒᵢ+θ⭤ⁱⱼ) + 1/(θ⭤ᵒⱼ+θ⭤ⁱᵢ)] = E[(wᵢⱼ+wⱼᵢ)·𝟙⭤].
# --------------------------------------------------------------------------------------------------------

# binary + weighted base matrices from a weight matrix `W` and its (probability or 0/1) support `A`
function _flux_base_matrices(W::AbstractMatrix, A::AbstractMatrix)
    At = transpose(A)
    o  = one(eltype(A))
    P  = A .* (o .- At)
    R  = A .* At
    Z  = (o .- A) .* (o .- At)
    Pw = W .* (o .- At)
    Rw = W .* At .+ transpose(W) .* A
    return P, R, Z, Pw, Rw
end

# the 13 triadic fluxes from precomputed base matrices (shared by the empirical and model methods).
# By the cyclic property of the trace, the flux term with the weighted factor at position q is
#   tr(W_q · B_q) = dot(W_q, transpose(B_q)),
# where B_q is the PRODUCT OF THE OTHER TWO BINARY FACTORS in cyclic order (q=1: F2F3, q=2: F3F1,
# q=3: F1F2). The zpos diagonal correction of `_motif_count` carries over as a cheap dot as well
# (e.g. for zp=2 and the weight at q=1 it subtracts tr(F3·W1) = dot(W1, transpose(F3))). Only the
# BINARY pair products appear as matrix multiplications, and they repeat across motifs and weighted
# positions, so they are computed once and shared: ≤12 distinct products serve all 33 flux terms
# (vs. one product per term in the naive evaluation, a ~3x reduction in BLAS-3 work).
# NOTE: the product cache makes this value path non-Zygote-differentiable (mutation); ForwardDiff
# still traces through it. The z-scores of the fluxes are sampling-based anyway (`flux_zscores`).
function _motif_fluxes(P, Q, R, Z, Pw, Qw, Rw)
    T = eltype(Pw)
    binmat = lbl -> _motif_select(lbl, P, Q, R, Z)
    wmat   = lbl -> _motif_select(lbl, Pw, Qw, Rw, Z)
    # shared cache of binary pair products (materialised as plain matrices)
    prods = Dict{Tuple{Symbol,Symbol}, Matrix{T}}()
    pairprod!(a::Symbol, b::Symbol) = get!(() -> Matrix{T}(binmat(a) * binmat(b)), prods, (a, b))

    res = Vector{T}(undef, 13)
    for (k, s) in enumerate(_motif_specs)
        (l1, l2, l3, zp) = s
        lbls = (l1, l2, l3)
        acc = zero(T)
        for q in 1:3
            lbls[q] === :Z && continue                        # an absent dyad carries no weight
            W_q = wmat(lbls[q])
            a, b = lbls[mod1(q + 1, 3)], lbls[mod1(q + 2, 3)] # the other two factors, in cyclic order
            acc += dot(W_q, transpose(pairprod!(a, b)))       # tr(W_q F_a F_b)
            # diagonal correction (zp is never q: the a̲ factor carries no weight): subtract the trace
            # over the two non-Z POSITIONS of the ordered triple, with F_q replaced by W_q
            if zp != 0
                o1, o2 = mod1(zp + 1, 3), mod1(zp + 2, 3)
                other = o1 == q ? o2 : o1 # the non-Z position that is not the weighted one
                acc -= dot(W_q, transpose(binmat(lbls[other]))) # tr(W_q · F_other) (= tr(F_other · W_q))
            end
        end
        res[k] = acc
    end
    return res
end

"""
    motif_fluxes(W::AbstractMatrix)
    motif_fluxes(G::SimpleWeightedGraphs.AbstractSimpleWeightedGraph)
    motif_fluxes(m::DCReM)
    motif_fluxes(m::CRWCM)

Compute the 13 triadic fluxes `[F1, F2, …, F13]` of the weighted directed network: the total weight sitting
on the links of all occurrences of each 3-node motif (Di Vece et al. (2023)). The motif numbering matches
[`motifs`](@ref).

For the matrix/graph methods the binary support is obtained by binarising the weight matrix
(`W[i,j] ≠ 0` ⟺ link `i→j`; matrix convention: `W[i,j]` is the weight of the link `i→j`). For the model
methods the result is the **exact** expected flux spectrum ⟨F1⟩, …, ⟨F13⟩ (a motif's three dyads are
distinct and dyads are independent, so the expectation factorises per position; for the [`CRWCM`](@ref)
the within-dyad correlation between the two directions is handled through the dyadic expectations).

# Examples
```jldoctest motif_fluxes_doc
julia> G = rhesus_macaques();

julia> model = CRWCM(G);

julia> solve_model!(model);

julia> length(motif_fluxes(model)) == length(motif_fluxes(G)) == 13
true
```
"""
function motif_fluxes(W::T) where T<:AbstractMatrix
    size(W,1) == size(W,2) || throw(DimensionMismatch("The weight matrix must be square"))
    A = (!iszero).(W) .* one(eltype(W)) # numeric 0/1 support, same eltype as W
    P, R, Z, Pw, Rw = _flux_base_matrices(W, A)
    return _motif_fluxes(P, transpose(P), R, Z, Pw, transpose(Pw), Rw)
end

motif_fluxes(G::T) where T<:SimpleWeightedGraphs.AbstractSimpleWeightedGraph = Graphs.is_directed(G) ? motif_fluxes(Matrix(transpose(G.weights))) : throw(ArgumentError("The graph must be directed for the triadic fluxes"))

function motif_fluxes(m::DCReM)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters of `m` must be computed before computing the triadic fluxes, see `solve_model!`"))
    Ghat = Ĝ(m)
    What = Ŵ(m)
    # Ĝ has a zero diagonal, so the (1-A)∘(1-Aᵀ) form automatically carries the unit diagonal required by
    # the `_motif_count` inclusion–exclusion convention
    P, R, Z, Pw, Rw = _flux_base_matrices(What, Ghat)
    return _motif_fluxes(P, transpose(P), R, Z, Pw, transpose(Pw), Rw)
end

function motif_fluxes(m::CRWCM)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters of `m` must be computed before computing the triadic fluxes, see `solve_model!`"))
    n = m.status[:N]
    x = m.xᵣ[m.dᵣ_ind]
    y = m.yᵣ[m.dᵣ_ind]
    z = m.zᵣ[m.dᵣ_ind]
    θ⭢  = @view m.θ[1:n]
    θ⭠  = @view m.θ[n+1:2*n]
    θ⭤ᵒ = @view m.θ[2*n+1:3*n]
    θ⭤ⁱ = @view m.θ[3*n+1:4*n]
    o = one(precision(m))
    P̂  = zeros(precision(m), n, n)
    R̂  = zeros(precision(m), n, n)
    Ẑ  = zeros(precision(m), n, n)
    P̂w = zeros(precision(m), n, n)
    R̂w = zeros(precision(m), n, n)
    for i in 1:n
        @inbounds Ẑ[i,i] = o
        for j in 1:n
            if i ≠ j
                @inbounds xiyj = x[i]*y[j]
                @inbounds zizj = z[i]*z[j]
                @inbounds D = o + xiyj + x[j]*y[i] + zizj
                @inbounds P̂[i,j] = xiyj / D
                @inbounds R̂[i,j] = zizj / D
                @inbounds Ẑ[i,j] = o / D
                @inbounds P̂w[i,j] = iszero(xiyj) ? zero(precision(m)) : (xiyj / D) / (θ⭢[i] + θ⭠[j])
                @inbounds R̂w[i,j] = iszero(zizj) ? zero(precision(m)) : (zizj / D) * (o/(θ⭤ᵒ[i] + θ⭤ⁱ[j]) + o/(θ⭤ᵒ[j] + θ⭤ⁱ[i]))
            end
        end
    end
    return _motif_fluxes(P̂, transpose(P̂), R̂, Ẑ, P̂w, transpose(P̂w), R̂w)
end

"""
    motif_flux(x, k::Int)

Compute the triadic flux of motif `k` (`k ∈ 1:13`) only. Accepts the same arguments as
[`motif_fluxes`](@ref). When several fluxes are needed, `motif_fluxes` is cheaper than repeated calls.
"""
function motif_flux(x, k::Int)
    1 <= k <= 13 || throw(ArgumentError("The motif number must be in 1:13"))
    return motif_fluxes(x)[k]
end


## dyad-state machinery for the intensity kernel: state codes 1=:P (i→j only), 2=:Q (j→i only),
## 3=:R (both), 4=:Z (absent), and a 4×4×4 lookup mapping every connected state triple to its motif
## index (0 = disconnected pattern). Precomputing per-pair states/weight-products/link-counts turns
## the O(N³) triple loop into pure array reads (no per-triple hashing, tuple building or symbol
## branching), which is a large constant-factor gain.
const _state_code = Dict(:P => 1, :Q => 2, :R => 3, :Z => 4)
const _motif_state_lut = let lut = zeros(Int8, 4, 4, 4)
    for (k, s) in enumerate(_motif_specs)
        lut[_state_code[s[1]], _state_code[s[2]], _state_code[s[3]]] = Int8(k)
    end
    lut
end

"""
    motif_intensities(W::AbstractMatrix)
    motif_intensities(G::SimpleWeightedGraphs.AbstractSimpleWeightedGraph)

Compute the 13 triadic intensities `[I1, I2, …, I13]` of the weighted directed network: for every
occurrence of a 3-node motif, its intensity is the *geometric mean* of the weights on its links
(Onnela et al. (2005)), and `I_m` sums the intensities of all occurrences of motif `m` (the `Im`
statistic of NuMeTriS / Di Vece et al. (2023)).

*Note*: unlike the fluxes, the intensities have no closed-form model expectation (the geometric mean does
not factorise over dyads); model-induced z-scores can be obtained by sampling via [`ensemble_zscores`](@ref).

# Examples
```jldoctest
julia> length(motif_intensities(rhesus_macaques())) == 13
true
```
"""
function motif_intensities(W::T) where T<:AbstractMatrix
    size(W,1) == size(W,2) || throw(DimensionMismatch("The weight matrix must be square"))
    n = size(W,1)
    F = float(eltype(W))
    res = zeros(F, 13)
    # precompute, per ordered pair, the dyad-state code, the product of the weights the dyad carries
    # in this orientation (1 for an absent dyad: multiplicative identity), and its link count — O(N²)
    S  = Matrix{Int8}(undef, n, n)
    WP = Matrix{F}(undef, n, n)
    NL = Matrix{Int8}(undef, n, n)
    @inbounds for j in 1:n, i in 1:n
        fwd = !iszero(W[i,j]); bwd = !iszero(W[j,i])
        if fwd & bwd
            S[i,j] = Int8(3); WP[i,j] = F(W[i,j]) * F(W[j,i]); NL[i,j] = Int8(2)
        elseif fwd
            S[i,j] = Int8(1); WP[i,j] = F(W[i,j]);             NL[i,j] = Int8(1)
        elseif bwd
            S[i,j] = Int8(2); WP[i,j] = F(W[j,i]);             NL[i,j] = Int8(1)
        else
            S[i,j] = Int8(4); WP[i,j] = one(F);                NL[i,j] = Int8(0)
        end
    end
    # triple loop over array reads only
    @inbounds for i in 1:n, j in 1:n
        i == j && continue
        Sij = S[i,j]; Wij = WP[i,j]; Lij = NL[i,j]
        for k in 1:n
            (k == i || k == j) && continue
            idx = _motif_state_lut[Sij, S[j,k], S[k,i]]
            iszero(idx) && continue
            res[idx] += (Wij * WP[j,k] * WP[k,i])^(1 / (Lij + NL[j,k] + NL[k,i]))
        end
    end
    return res
end

motif_intensities(G::T) where T<:SimpleWeightedGraphs.AbstractSimpleWeightedGraph = Graphs.is_directed(G) ? motif_intensities(Matrix(transpose(G.weights))) : throw(ArgumentError("The graph must be directed for the triadic intensities"))


# ---- model-expected reciprocity for the weighted directed models --------------------------------------

"""
    reciprocity(m::DCReM)

Compute the expected topological reciprocity of the (binary DBCM layer of the) DCReM model `m` as the
ratio of expectations ``\\sum_{i≠j} f_{ij}f_{ji} / \\sum_{i≠j} f_{ij}`` (directions independent under the DBCM).
"""
function reciprocity(m::DCReM)
    m.status[:conditional_params_computed] ? nothing : throw(ArgumentError("The conditional parameters have not been computed yet"))
    n = m.status[:N]
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
    reciprocity(m::CRWCM)

Compute the expected topological reciprocity of the (binary RBCM layer of the) CRWCM model `m` as the
ratio of expectations ``\\sum_{i≠j} p^{↔}_{ij} / \\sum_{i≠j} ⟨a_{ij}⟩``. As the RBCM layer constrains the
reciprocal degree sequences, this reproduces the observed reciprocity of the network it was fitted to.
"""
function reciprocity(m::CRWCM)
    m.status[:conditional_params_computed] ? nothing : throw(ArgumentError("The conditional parameters have not been computed yet"))
    n = m.status[:N]
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
    weighted_reciprocity(m::DCReM)

Compute the expected weighted reciprocity of the DCReM model `m` as the ratio of expectations
``\\sum_{i≠j} ⟨w_{ij}⟩ f_{ji} / \\sum_{i≠j} ⟨w_{ij}⟩`` (the weight of `i→j` and the presence of `j→i` are
independent under the DBCM layer).
"""
function weighted_reciprocity(m::DCReM)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    n = m.status[:N]
    θᵒ = @view m.θ[1:n]
    θⁱ = @view m.θ[n+1:end]
    num = zero(precision(m))
    den = zero(precision(m))
    for i = 1:n
        for j = 1:n
            if i ≠ j
                w = A(m, i, j) / (θᵒ[i] + θⁱ[j])
                num += w * A(m, j, i)
                den += w
            end
        end
    end
    return num / den
end

"""
    weighted_reciprocity(m::CRWCM)

Compute the expected weighted reciprocity of the CRWCM model `m` as the ratio of expectations
``\\sum_{i≠j} E[w_{ij} 𝟙^{↔}_{ij}] / \\sum_{i≠j} ⟨w_{ij}⟩``, where the numerator terms are
``f^{↔}_{ij}/(θ^{↔,o}_i + θ^{↔,i}_j)`` (weight sits on a reciprocated link iff the dyad is reciprocated).
As the CRWCM constrains the reciprocal strength sequences, this reproduces the observed weighted
reciprocity of the network it was fitted to.
"""
function weighted_reciprocity(m::CRWCM)
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    n = m.status[:N]
    θ⭢  = @view m.θ[1:n]
    θ⭠  = @view m.θ[n+1:2*n]
    θ⭤ᵒ = @view m.θ[2*n+1:3*n]
    θ⭤ⁱ = @view m.θ[3*n+1:4*n]
    num = zero(precision(m))
    den = zero(precision(m))
    for i = 1:n
        for j = 1:n
            if i ≠ j
                f1 = p⭢(m, i, j)
                f2 = p⭤(m, i, j)
                wrec = iszero(f2) ? zero(precision(m)) : f2 / (θ⭤ᵒ[i] + θ⭤ⁱ[j])
                num += wrec
                den += wrec + (iszero(f1) ? zero(precision(m)) : f1 / (θ⭢[i] + θ⭠[j]))
            end
        end
    end
    return num / den
end


# ---- sampling-based ensemble z-scores ------------------------------------------------------------------

# dense weight matrix of a weighted graph in the W[i,j] = w(i→j) convention
_weight_matrix(G::SimpleWeightedGraphs.AbstractSimpleWeightedGraph) = Matrix(transpose(G.weights))
_weight_matrix(G) = throw(ArgumentError("Weighted metrics require a weighted graph (got $(typeof(G)))"))

# dense BINARY (0/1, numeric) adjacency matrix. NOTE: `Graphs.adjacency_matrix` on a
# SimpleWeightedGraphs graph returns the WEIGHTS, so it must be binarised explicitly — feeding
# weights into a binary metric (e.g. `motifs`) would silently produce garbage.
_binary_matrix(G) = Float64.((!iszero).(Matrix(Graphs.adjacency_matrix(G))))

"""
    ensemble_zscores(m::AbstractMaxEntropyModel, X::Function; n::Int=500, rng=Random.default_rng(), weighted::Bool=false)

Compute sampling-based z-scores of the metric `X` under the model `m`: sample `n` graphs from the model,
evaluate `X` on each (and on the observed graph `m.G`), and standardise the observed value with the
ensemble mean and standard deviation. `X` maps a matrix to a scalar **or a vector** (e.g. [`motifs`](@ref)
or [`motif_fluxes`](@ref)); with `weighted=true` `X` receives the dense weight matrix, otherwise the
binary adjacency matrix.

Returns a `NamedTuple` `(obs = X(G_obs), μ = ensemble mean, σ = ensemble std, z = (obs - μ)/σ)`.

This mirrors the numerical procedure used in Di Vece et al. (2023) and the NuMeTriS package
(`numerical_triadic_zscores`, default 500-sample ensembles), which is required when no analytical
expectation or standard deviation is available (e.g. for triadic fluxes and intensities) or when the
metric's ensemble distribution is not normal.

See also [`motif_zscores`](@ref), [`flux_zscores`](@ref).

# Examples
```jldoctest
julia> model = RBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()));

julia> solve_model!(model);

julia> res = ensemble_zscores(model, motifs, n=100, rng=MaxEntropyGraphs.Xoshiro(1));

julia> length(res.z)
13
```
"""
function ensemble_zscores(m::AbstractMaxEntropyModel, X::Function; n::Int=500, rng::AbstractRNG=default_rng(), weighted::Bool=false)
    isnothing(m.G) && throw(ArgumentError("The model must hold an observed graph (m.G) to compute z-scores"))
    n > 1 || throw(ArgumentError("The ensemble size must be larger than one"))
    # observed value
    obs = weighted ? X(_weight_matrix(m.G)) : X(_binary_matrix(m.G))
    # ensemble values
    sample = rand(m, n; rng=rng)
    vals = [weighted ? X(_weight_matrix(g)) : X(_binary_matrix(g)) for g in sample]
    # elementwise mean / std / z (works for scalar- and vector-valued metrics alike)
    μ = sum(vals) ./ n
    σ = sqrt.(sum(v -> (v .- μ) .^ 2, vals) ./ (n - 1))
    z = (obs .- μ) ./ σ
    return (obs = obs, μ = μ, σ = σ, z = z)
end

"""
    motif_zscores(m::AbstractMaxEntropyModel; n::Int=500, rng=Random.default_rng())

Sampling-based z-scores of the 13 binary triadic motif counts under the model `m`
(`ensemble_zscores(m, motifs; ...)`). For the [`RBCM`](@ref), analytical z-scores are also available
through the exact `motifs(m)` expectations and `σₓ(m, Mk)`.
"""
motif_zscores(m::AbstractMaxEntropyModel; n::Int=500, rng::AbstractRNG=default_rng()) = ensemble_zscores(m, motifs; n=n, rng=rng)

"""
    flux_zscores(m::AbstractMaxEntropyModel; n::Int=500, rng=Random.default_rng())

Sampling-based z-scores of the 13 triadic fluxes under the (weighted) model `m`
(`ensemble_zscores(m, motif_fluxes; weighted=true, ...)`).
"""
flux_zscores(m::AbstractMaxEntropyModel; n::Int=500, rng::AbstractRNG=default_rng()) = ensemble_zscores(m, motif_fluxes; n=n, rng=rng, weighted=true)

function M13(G::T) where T <: Graphs.AbstractGraph
    # check if the graph is directed
    Graphs.is_directed(G) ? nothing : throw(ArgumentError("The graph must be directed"))

    # initialise the count
    res = 0 
    # iterate
    for i in Graphs.vertices(G)
        # outgoing edges i -> j
        for j in Graphs.outneighbors(G, i)
            # verify recipocrated edge j -> i
            if j ≠ i && Graphs.has_edge(G, j, i) 
                # outgoing edges j -> k
                for k in Graphs.outneighbors(G, j)
                    # verify recipocrated edge k -> j
                    if k ≠ j && Graphs.has_edge(G, k, j)
                        # check edges i -> k and k -> i
                        if Graphs.has_edge(G, i, k) && Graphs.has_edge(G, k, i)
                            res += 1
                        end
                    end
                end
            end
        end
    end

    # Each triangle is counted 6 times (once for each permutation of the vertices), but we accept this for coherence with other packages
    return res
end

function M12(G::T) where T <: Graphs.AbstractGraph
    # check if the graph is directed
    Graphs.is_directed(G) ? nothing : throw(ArgumentError("The graph must be directed"))

    # initialise the count
    res = 0 
    # iterate
    for i in Graphs.vertices(G)
        # outgoing edges i -> j
        for j in Graphs.outneighbors(G, i)
            # verify recipocrated edge j -> i
            if j ≠ i && Graphs.has_edge(G, j, i) 
                # outgoing edges j -> k
                for k in Graphs.outneighbors(G, j)
                    # verify recipocrated edge k -> j
                    if k ≠ j && Graphs.has_edge(G, k, j)
                        # check edges i -> k and k -> i
                        if !Graphs.has_edge(G, i, k) && Graphs.has_edge(G, k, i)
                            res += 1
                        end
                    end
                end
            end
        end
    end

    # Each triangle is counted 6 times (once for each permutation of the vertices), but we accept this for coherence with other packages
    return res
end

function M11(G::T) where T <: Graphs.AbstractGraph
    # check if the graph is directed
    Graphs.is_directed(G) ? nothing : throw(ArgumentError("The graph must be directed"))

    # initialise the count
    res = 0 
    # iterate
    for i in Graphs.vertices(G)
        # outgoing edges i -> j
        for j in Graphs.outneighbors(G, i)
            # verify recipocrated edge j -> i
            if j ≠ i && Graphs.has_edge(G, j, i) 
                # incoming edges k -> j
                for k in Graphs.inneighbors(G, j)
                    # verify non-recipocrated edge j -> k
                    if k ≠ i && k ≠ j && !Graphs.has_edge(G, j, k)
                        # check edges i -> k and k -> i
                        if !Graphs.has_edge(G, i, k) && Graphs.has_edge(G, k, i)
                            res += 1
                        end
                    end
                end
            end
        end
    end

    # Each triangle is counted 6 times (once for each permutation of the vertices), but we accept this for coherence with other packages
    return res
end

function M10(G::T) where T <: Graphs.AbstractGraph
    # check if the graph is directed
    Graphs.is_directed(G) ? nothing : throw(ArgumentError("The graph must be directed"))

    # initialise the count
    res = 0 
    # iterate
    for i in Graphs.vertices(G)
        # outgoing edges i -> j
        for j in Graphs.outneighbors(G, i)
            # verify recipocrated edge j -> i
            if j ≠ i && Graphs.has_edge(G, j, i) 
                # outgoing edges j -> k
                for k in Graphs.outneighbors(G, j)
                    # verify non-recipocrated edge k -> j
                    if k ≠ i && !Graphs.has_edge(G, k, j)
                        # check edges i -> k and k -> i
                        if !Graphs.has_edge(G, i, k) && Graphs.has_edge(G, k, i)
                            res += 1
                        end
                    end
                end
            end
        end
    end

    # Each triangle is counted 6 times (once for each permutation of the vertices), but we accept this for coherence with other packages
    return res
end

function M9(G::T) where T <: Graphs.AbstractGraph
    # check if the graph is directed
    Graphs.is_directed(G) ? nothing : throw(ArgumentError("The graph must be directed"))

    # initialise the count
    res = 0 
    # iterate
    for i in Graphs.vertices(G)
        # outgoing edges i -> j
        for j in Graphs.outneighbors(G, i)
            # verify non-recipocrated edge j -> i
            if j ≠ i && !Graphs.has_edge(G, j, i) 
                # outgoing edges j -> k
                for k in Graphs.outneighbors(G, j)
                    # verify non-recipocrated edge k -> j
                    if k ≠ j && !Graphs.has_edge(G, k, j)
                        # check edges i -> k and k -> i
                        if !Graphs.has_edge(G, i, k) && Graphs.has_edge(G, k, i)
                            res += 1
                        end
                    end
                end
            end
        end
    end

    # Each triangle is counted 6 times (once for each permutation of the vertices), but we accept this for coherence with other packages
    return res
end

function M8(G::T) where T <: Graphs.AbstractGraph
    # check if the graph is directed
    Graphs.is_directed(G) ? nothing : throw(ArgumentError("The graph must be directed"))

    # initialise the count
    res = 0 
    # iterate
    for i in Graphs.vertices(G)
        # outgoing edges i -> j
        for j in Graphs.outneighbors(G, i)
            # verify recipocrated edge j -> i
            if j ≠ i && Graphs.has_edge(G, j, i) 
                # outgoing edges j -> k
                for k in Graphs.outneighbors(G, j)
                    # verify recipocrated edge k -> j
                    if k ≠ i && k ≠ j && Graphs.has_edge(G, k, j)
                        # check edges i -> k and k -> i
                        if !Graphs.has_edge(G, i, k) && !Graphs.has_edge(G, k, i)
                            res += 1
                        end
                    end
                end
            end
        end
    end

    # Each triangle is counted 6 times (once for each permutation of the vertices), but we accept this for coherence with other packages
    return res
end

function M7(G::T) where T <: Graphs.AbstractGraph
    # check if the graph is directed
    Graphs.is_directed(G) ? nothing : throw(ArgumentError("The graph must be directed"))

    # initialise the count
    res = 0 
    # iterate
    for i in Graphs.vertices(G)
        # outgoing edges i -> j
        for j in Graphs.outneighbors(G, i)
            # verify non-recipocrated edge j -> i
            if j ≠ i && !Graphs.has_edge(G, j, i) 
                # outgoing edges j -> k
                for k in Graphs.outneighbors(G, j)
                    # verify recipocrated edge k -> j
                    if k ≠ i && k ≠ j && Graphs.has_edge(G, k, j)
                        # check edges i -> k and k -> i
                        if !Graphs.has_edge(G, i, k) && !Graphs.has_edge(G, k, i)
                            res += 1
                        end
                    end
                end
            end
        end
    end

    # Each triangle is counted 6 times (once for each permutation of the vertices), but we accept this for coherence with other packages
    return res
end

function M6(G::T) where T <: Graphs.AbstractGraph
    # check if the graph is directed
    Graphs.is_directed(G) ? nothing : throw(ArgumentError("The graph must be directed"))

    # initialise the count
    res = 0 
    # iterate
    for i in Graphs.vertices(G)
        # incoming edges j -> i
        for j in Graphs.inneighbors(G, i)
            # verify non-recipocrated edge i -> j
            if j ≠ i && !Graphs.has_edge(G, i, j) 
                # outgoing edges j -> k
                for k in Graphs.outneighbors(G, j)
                    # verify recipocrated edge k -> j
                    if k ≠ j && Graphs.has_edge(G, k, j)
                        # check edges i -> k and k -> i
                        if !Graphs.has_edge(G, i, k) && Graphs.has_edge(G, k, i)
                            res += 1
                        end
                    end
                end
            end
        end
    end

    # Each triangle is counted 6 times (once for each permutation of the vertices), but we accept this for coherence with other packages
    return res
end

function M5(G::T) where T <: Graphs.AbstractGraph
    # check if the graph is directed
    Graphs.is_directed(G) ? nothing : throw(ArgumentError("The graph must be directed"))

    # initialise the count
    res = 0 
    # iterate
    for i in Graphs.vertices(G)
        # incoming edges j -> i
        for j in Graphs.inneighbors(G, i)
            # verify non-recipocrated edge i -> j
            if j ≠ i && !Graphs.has_edge(G, i, j) 
                # outgoing edges j -> k
                for k in Graphs.outneighbors(G, j)
                    # verify non-recipocrated edge k -> j
                    if k ≠ j && !Graphs.has_edge(G, k, j)
                        # check edges i -> k and k -> i
                        if !Graphs.has_edge(G, i, k) && Graphs.has_edge(G, k, i)
                            res += 1
                        end
                    end
                end
            end
        end
    end

    # Each triangle is counted 6 times (once for each permutation of the vertices), but we accept this for coherence with other packages
    return res
end

function M4(G::T) where T <: Graphs.AbstractGraph
    # check if the graph is directed
    Graphs.is_directed(G) ? nothing : throw(ArgumentError("The graph must be directed"))

    # initialise the count
    res = 0 
    # iterate
    for i in Graphs.vertices(G)
        # incoming edges j -> i
        for (j,k) in combinations(Graphs.inneighbors(G,i), 2) #Graphs.inneighbors(G, i)
            if !Graphs.has_edge(G, i, j) && !Graphs.has_edge(G, i, k) && !Graphs.has_edge(G, j, k) && !Graphs.has_edge(G, k, j)
                res += 1
            end
        end
    end

    # we multiply by two, because by using combinations, we only count half of the wedges
    return res * 2
end

function M3(G::T) where T <: Graphs.AbstractGraph
    # check if the graph is directed
    Graphs.is_directed(G) ? nothing : throw(ArgumentError("The graph must be directed"))

    # initialise the count
    res = 0 
    # iterate
    for i in Graphs.vertices(G)
        # outgoing edges i -> j
        for j in Graphs.inneighbors(G, i)
            # verify non-recipocrated edge i -> j
            if j ≠ i && !Graphs.has_edge(G, i, j) 
                # outgoing edges j -> k
                for k in Graphs.outneighbors(G, j)
                    # verify recipocrated edge k -> j
                    if k ≠ i && k ≠ j && Graphs.has_edge(G, k, j)
                        # check edges i -> k and k -> i
                        if !Graphs.has_edge(G, i, k) && !Graphs.has_edge(G, k, i)
                            res += 1
                        end
                    end
                end
            end
        end
    end

    # Each triangle is counted 6 times (once for each permutation of the vertices), but we accept this for coherence with other packages
    return res
end

function M2(G::T) where T <: Graphs.AbstractGraph
    # check if the graph is directed
    Graphs.is_directed(G) ? nothing : throw(ArgumentError("The graph must be directed"))

    # initialise the count
    res = 0 
    # iterate
    for i in Graphs.vertices(G)
        # outgoing edges i -> j
        for j in Graphs.inneighbors(G, i)
            # verify non-recipocrated edge i -> j
            if j ≠ i && !Graphs.has_edge(G, i, j) 
                # outgoing edges j -> k
                for k in Graphs.inneighbors(G, j)
                    # verify non-recipocrated edge j -> k
                    if k ≠ i && k ≠ j && !Graphs.has_edge(G, j, k)
                        # check edges i -> k and k -> i
                        if !Graphs.has_edge(G, i, k) && !Graphs.has_edge(G, k, i)
                            res += 1
                        end
                    end
                end
            end
        end
    end

    # Each triangle is counted 6 times (once for each permutation of the vertices), but we accept this for coherence with other packages
    return res
end

function M1(G::T) where T <: Graphs.AbstractGraph
    # check if the graph is directed
    Graphs.is_directed(G) ? nothing : throw(ArgumentError("The graph must be directed"))

    # initialise the count
    res = 0 
    # iterate
    for i in Graphs.vertices(G)
        # outgoing edges i -> j
        for j in Graphs.inneighbors(G, i)
            # verify non-recipocrated edge i -> j
            if j ≠ i && !Graphs.has_edge(G, i, j) 
                # outgoing edges j -> k
                for k in Graphs.outneighbors(G, j)
                    # verify non-recipocrated edge k -> j
                    if k ≠ i && k ≠ j && !Graphs.has_edge(G, k, j)
                        # check edges i -> k and k -> i
                        if !Graphs.has_edge(G, i, k) && !Graphs.has_edge(G, k, i)
                            res += 1
                        end
                    end
                end
            end
        end
    end

    # Each triangle is counted 6 times (once for each permutation of the vertices), but we accept this for coherence with other packages
    return res
end


########################################################################################################
# Bipartite networks motifs and helper functions
########################################################################################################

"""
    biadjacency_matrix(G::Graphs.SimpleGraph; skipchecks::Bool=false)

Return the biadjacency matrix of the bipartite graph `G`.

If the graph is not bipartite, an error is thrown.
If the adjacency matrix can be written as:
A = [O B; B' O] where O is the null matrix, then the returned biadjacency matrix is B and and B' is the transposed biadjacency matrix.

# Arguments
- `skipchecks`: if true, skip the check for the graph being bipartite.

# Examples
```jldoctest biadjacency_matrix
julia> A = [0 0 0 0 1 0 0;
0 0 0 0 1 1 0;
0 0 0 0 0 0 1;
0 0 0 0 0 1 1;
1 1 0 0 0 0 0;
0 1 0 1 0 0 0;
0 0 1 1 0 0 0];

julia> G = MaxEntropyGraphs.Graphs.SimpleGraph(A);

julia> Array(biadjacency_matrix(A))
4x3 Matrix{Int64}:
 1  0  0
 1  1  0
 0  0  1
 0  1  1

```
"""
function biadjacency_matrix(G::Graphs.SimpleGraph; skipchecks::Bool=false)
    if !skipchecks
        Graphs.is_bipartite(G) || throw(ArgumentError("The graph `G` must be bipartite."))  
    end
    # get the bipartite map
    membership = Graphs.bipartite_map(G)
    # get the adjacency_matrix
    A = Graphs.adjacency_matrix(G)
    # get the biadjacency-matrix
    return A[membership .== 1, membership .== 2]
end

"""
    project(G::Graphs.SimpleGraph;  membership::Vector=Graphs.bipartite_map(G), 
                                    bottom::Vector=findall(membership .== 1), 
                                    top::Vector=findall(membership .== 2); 
                                    layer::Symbol=:bottom)

Project the bipartite graph `G` onto one of its layers and return the projected graph.

# Arguments 
- `membership`: the bipartite mapping of the graphs. This can be computed using `Graphs.bipartite_map(G)`.
- `bottom`: the nodes in the bottom layer. This can be computed using `findall(membership .== 1)`.
- `top`: the nodes in the top layer. This can be computed using `findall(membership .== 2)`.
- `layer`: the layer can be specified by passing `layer=:bottom` or `layer=:top`. Layer membership is determined by the bipartite map of the graph.
- `method`: the method used to compute the adjacency matrix of the projected graph. This can be `:simple` or `:weighted`. Both methods compute 
    the product of the biadjacency matrix with its transposed, but the `:weighted` method uses the weights of the edges in the projected graph.


# Examples
```jldoctest project_bipartite_to_simple
julia> using Graphs

julia> G = SimpleGraph(5); add_edge!(G, 1, 4); add_edge!(G, 2, 4); add_edge!(G, 3, 4); add_edge!(G, 3, 5);

julia> project(G, layer=:bottom)
{3, 3} undirected simple Int64 graph

```
```jldoctest project_bipartite_to_simple
julia> project(G, layer=:top)
{2, 1} undirected simple Int64 graph

```
"""
function project(G::Graphs.SimpleGraph, membership::Vector=Graphs.bipartite_map(G), 
                                        bottom::Vector=findall(membership .== 1), 
                                        top::Vector=findall(membership .== 2); 
                                        layer::Symbol=:bottom, method::Symbol=:simple, skipchecks::Bool=false)
    
    if !skipchecks
        # check if bipartite
        Graphs.is_bipartite(G) || throw(ArgumentError("The graph `G` must be bipartite."))  
    end
    # define graph type
    if method == :simple
        Gtype = Graphs.SimpleGraph
    elseif method ==:weighted
        Gtype = SimpleWeightedGraphs.SimpleWeightedGraph
    else
        throw(ArgumentError("The method $(method) is not yet implemented."))
    end

    # get the adjacency_matrix
    A = Graphs.adjacency_matrix(G)
    # get the biadjacency-matrix
    B = @view A[bottom, top]
    # get projected adjacency_matrix
    if layer ∈ [:bottom; :⊥]
        Aproj = B * B'
    elseif layer ∈ [:top; :⊤]
        Aproj = B' * B
    else
        throw(ArgumentError("The layer must be one of [:bottom, :⊥] for the bottom layer or [:top, :⊤] for the top layer."))
    end
    # avoid self-loops
    Aproj[diagind(Aproj)] .= zero(eltype(Aproj))
    # dropzeros
    dropzeros!(Aproj)

    return Gtype(Aproj)
end


"""
    project(B::T; layer::Symbol=:bottom, method::Symbol=:simple) where {T<:AbstractMatrix}

Project the biadjacency matrix `B` onto one of its layers.

# Arguments 
- `layer`: the layer can be specified by passing `layer=:bottom` or `layer=:top`. Layer membership is determined by the bipartite map of the graph.
- `method`: the method used to compute the adjacency matrix of the projected graph. This can be `:simple` or `:weighted`. Both methods compute 
    the product of the biadjacency matrix with its transposed, but the `:weighted` method uses the weights of the edges in the projected graph.


# Examples
```jldoctest project_bipartite_matrix
julia> B = [0 0 0 1 1; 0 0 0 1 1; 0 0 0 1 0];

julia> project(B, layer=:bottom)
3×3 Matrix{Bool}:
 0  1  1
 1  0  1
 1  1  0

```
```jldoctest project_bipartite_matrix
julia> project(B, layer=:top)
5×5 Matrix{Bool}:
 0  0  0  0  0
 0  0  0  0  0
 0  0  0  0  0
 0  0  0  0  1
 0  0  0  1  0

```
```jldoctest project_bipartite_matrix
julia> B = [0 0 0 1 1; 0 0 0 1 1; 0 0 0 1 0];

julia> project(B, layer=:bottom, method=:weighted)
3×3 Matrix{Int64}:
 0  2  1
 2  0  1
 1  1  0

```
```jldoctest project_bipartite_matrix
julia> project(B, layer=:top, method=:weighted)
5×5 Matrix{Int64}:
 0  0  0  0  0
 0  0  0  0  0
 0  0  0  0  0
 0  0  0  0  2
 0  0  0  2  0

```
"""
function project(B::T; layer::Symbol=:bottom, method::Symbol=:simple) where {T<:AbstractMatrix}
    if isequal(size(B)...)
        @warn "The matrix `B` is square, make sure it is a biadjacency matrix."
    end
    # define return type
    if method == :simple
        returnmethod =  B -> map(x -> !iszero(x), B)
    elseif method ==:weighted
        returnmethod = identity
    else
        throw(ArgumentError("The method $(method) is not yet implemented."))
    end

    # project
    if layer ∈ [:bottom; :⊥]
        Aproj = B * B'
    elseif layer ∈ [:top; :⊤]
        Aproj = B' * B
    else
        throw(ArgumentError("The layer must be one of [:bottom, :⊥] for the bottom layer or [:top, :⊤] for the top layer."))
    end
    # avoid self-loops
    Aproj[diagind(Aproj)] .= zero(eltype(Aproj))
    # dropzeros
    issparse(Aproj) ? dropzeros!(Aproj) : nothing

    return returnmethod(Aproj)
end


"""
    V_motifs(G::Graphs.SimpleGraph; membership::Vector=Graphs.bipartite_map(G), layer::Symbol=:bottom)

Count the total number of V-motif occurences in graph `G` for one of its layers.

*Note*: the bipartiteness of the graph is not explicitely checked.

# Arguments 
- `membership`: the bipartite mapping of the graphs. This can be computed using `Graphs.bipartite_map(G)`.
- `layer`: the layer can be specified by passing `layer=:bottom` or `layer=:top`. Layer membership is determined by the bipartite map of the graph.

# Examples
```jldoctest V_motifs_bipartite
julia> using Graphs

julia> G = SimpleGraph(5); add_edge!(G, 1, 4); add_edge!(G, 2, 4); add_edge!(G, 3, 4); add_edge!(G, 3, 5);

julia> V_motifs(G, layer=:bottom)
3

```
```jldoctest V_motifs_bipartite
julia> V_motifs(G, layer=:top)
1

```
"""
function V_motifs(G::Graphs.SimpleGraph; membership::Vector=Graphs.bipartite_map(G), layer::Symbol=:bottom)
    counted = zeros(Bool, Graphs.nv(G))
    # define the layer identification value
    if layer ∈ [:bottom; :⊥]
        layerid = 1
    elseif layer ∈ [:top; :⊤]
        layerid = 2
    else
        throw(ArgumentError("The layer must be one of [:bottom, :⊥] for the bottom layer or [:top, :⊤] for the top layer."))
    end

    res = 0
    # go over all nodes in the layer
    for i in Graphs.vertices(G)
        if membership[i] == layerid
            # go over all its neighbors
            for j in Graphs.neighbors(G, i)
                if membership[j] ≠ layerid
                    # go over all neighbors of j
                    for k in Graphs.neighbors(G, j)
                        if k≠i && membership[k] == layerid
                            if !counted[k]
                                res += 1
                            end
                        end
                    end
                end
            end
            # all links with i have been counted
            counted[i] = true
        end
    end
    return res
end


# Σ_{i<j} (A Aᵀ)_{ij} (the strict upper triangle of the Gram matrix), computed from the marginal sums `s`
# without ever materialising A*Aᵀ: (‖s‖² − Σ_row‖·‖²)/2 = (dot(s,s) − sum(abs2, A))/2. The numerator is always
# even, so integer inputs stay exact and Int-typed (`÷2`); float/tracked inputs use `/2` and remain differentiable.
function _half_gram_offdiag(s::AbstractVector, A::AbstractMatrix)
    num = dot(s, s) - sum(abs2, A)
    return eltype(A) <: Integer ? num ÷ 2 : num / 2
end

"""
    V_motifs(A::T; layer::Symbol=:bottom, skipchecks::Bool=false) where {T<:AbstractMatrix}

Count the total number of V-motif occurence in the biadjacency matrix `A` for one of its layers.

# Arguments 
- `layer`: the layer can be specified by passing `layer=:bottom` or `layer=:top`. Layer membership is determined by the bipartite map of the graph.
- `skipchecks`: if true, skip the dimension check on `A`

# Examples
```jldoctest V_motifs_bipartite_matrix
julia> A = [1 1; 1 1; 1 0];

julia> V_motifs(A, layer=:bottom)
4

```
```jldoctest V_motifs_bipartite_matrix
julia> V_motifs(A, layer=:top)
2

```
"""
function V_motifs(A::T; layer::Symbol=:bottom, skipchecks::Bool=false) where {T<:AbstractMatrix}
    # check dimensions
    if !skipchecks && isequal(size(A)...)
        @warn "The matrix `A` is square, make sure it is a biadjacency matrix."
    end
    # closed form per layer: sum the strict upper triangle of the Gram matrix from marginal sums only
    if layer ∈ [:bottom; :⊥]
        return _half_gram_offdiag(vec(sum(A, dims=1)), A)   # rows share bottom-layer neighbours (columns)
    elseif layer ∈ [:top; :⊤]
        return _half_gram_offdiag(vec(sum(A, dims=2)), A)   # columns share top-layer neighbours (rows)
    else
        throw(ArgumentError("The layer must be one of [:bottom, :⊥] for the bottom layer or [:top, :⊤] for the top layer."))
    end
end


"""
    V_motifs(m::BiCM; layer::Symbol=:bottom)

Count the total number of V-motif occurences in the BiCM `m` for one of its layers.

# Arguments 
- `layer`: the layer can be specified by passing `layer=:bottom` or `layer=:top`. Layer membership is determined by the bipartite map of the graph.
- `precomputed`: if true, the expected values of the biadjacency matrix are used, otherwise the parameters are computed from the model parameters.

# Examples
```jldoctest V_motifs_bicm
julia> model = BiCM(corporateclub());

julia> solve_model!(model);

julia> set_Ĝ!(model);

julia> round.((V_motifs(model, layer=:bottom), V_motifs(model, layer=:bottom, precomputed=false)), digits=4)
(449.257, 449.257)

```
```jldoctest V_motifs_bicm
julia> round.((V_motifs(model, layer=:top), V_motifs(model, layer=:top, precomputed=false)), digits=4)
(180.257, 180.257)
```
"""
function V_motifs(m::BiCM; layer::Symbol=:bottom, precomputed::Bool=true)
    # checks
    m.status[:params_computed] ? nothing : throw(ArgumentError("The likelihood maximising parameters must be computed for `m` first, see `solve_model!`"))

    if precomputed
        # check
        m.status[:G_computed] ? nothing : throw(ArgumentError("The expected values (m.Ĝ) must be computed first see `set_Ĝ!`"))

        # compute
        return V_motifs(m.Ĝ, layer=layer, skipchecks=true)

    else
        res = zero(precision(m))
        for i in (layer == :bottom ? eachindex(m.⊥nodes) : eachindex(m.⊤nodes))
            for j in (layer == :bottom ? eachindex(m.⊥nodes) : eachindex(m.⊤nodes))
                if j > i
                    res += V_motifs(m, i, j; layer=layer, precomputed=false)
                end
            end
        end
        return res
    end
end



"""
    V_motifs(G::Graphs.SimpleGraph, i::Int, j::Int; membership::Vector=Graphs.bipartite_map(G))

Count the number of V-motif occurences in graph `G` between nodes `i` and `j` of the graph.

*Notes*: 
1. the bipartiteness of the graph is not explicitely checked.
2. this uses the actual node numbers in the graph, not their numbering in the specific layer.

# Arguments 
- `membership`: the bipartite mapping of the graphs. This can be computed using `Graphs.bipartite_map(G)`.

# Examples
```jldoctest V_motifs_nodes
julia> using Graphs

julia> G = SimpleGraph(5); add_edge!(G, 1, 4); add_edge!(G, 1, 5); add_edge!(G, 2, 4); add_edge!(G, 2, 5); add_edge!(G, 3, 4);

julia> V_motifs(G, 1, 2)
2

```

"""
function V_motifs(G::Graphs.SimpleGraph, i::Int, j::Int; membership::Vector=Graphs.bipartite_map(G))
    # check membership
    membership[i] == membership[j] || throw(ArgumentError("The nodes `i` and `j` must be in the same layer."))
    # do the count
    res = 0
    for k in Graphs.neighbors(G, i)
        if Graphs.has_edge(G, k, j) && membership[k] ≠ membership[i]
            res += 1
        end
    end

    return res
end


"""
    V_motifs(B::T, i::Int, j::Int; layer::Symbol=:bottom, skipchecks::Bool=false) where {T<:AbstractMatrix}

Count the number of V-motif occurences in the biadjacency matrix `B` between nodes `i` and `j` of a `layer`.

# Arguments
- `layer`: the layer can be specified by passing `layer=:bottom` or `layer=:top`.
- `skipchecks`: if true, skip the dimension check on `B`

*Notes*: depending on the layer, the tuple (`i`, `j`) denotes rows (:bottom) or columns (:top) of the biadjacency matrix `B`.

# Examples
```jldoctest V_motifs_bipartite_matrix_local
julia> B = [1 1; 1 1; 1 0];

julia> V_motifs(B, 1, 2, layer=:bottom)
2

```
```jldoctest V_motifs_bipartite_matrix_local
julia> V_motifs(B, 1, 2, layer=:top)
2

```
"""
function V_motifs(B::T, i::Int, j::Int; layer::Symbol=:bottom, skipchecks::Bool=false) where {T<:AbstractMatrix}
    # check dimensions
    if !skipchecks && isequal(size(B)...)
        @warn "The matrix `B` is square, make sure it is a biadjacency matrix."
    end
    # do the count
    if layer ∈ [:bottom; :⊥]
        return dot(B[i,:], B[j,:])
    elseif layer ∈ [:top; :⊤]
        return dot(B[:,i], B[:,j])
    else
        throw(ArgumentError("The layer must be one of [:bottom, :⊥] for the bottom layer or [:top, :⊤] for the top layer."))
    end
end


"""
    V_motifs(m::BiCM, i::Int, j::Int; layer::Symbol=:bottom, precomputed::Bool=false)

Count the number of expected V-motif occurences in the BiCM `m` between nodes `i` and `j` of a `layer`.

# Arguments
- `layer`: the layer can be specified by passing `layer=:bottom` or `layer=:top`.
- `precomputed`: if true, the expected values of the biadjacency matrix are used, otherwise the parameters are computed from the model parameters.

*Notes*: depending on the layer, the tuple (`i`, `j`) denotes rows (:bottom) or columns (:top) of the expected biadjacency matrix `A`.

# Examples
```jldoctest V_motifs_bicm_local
julia> model = BiCM(corporateclub());

julia> solve_model!(model);

julia> set_Ĝ!(model);

julia> V_motifs(model, 16, 13, layer=:bottom), V_motifs(model, 16, 13, layer=:bottom, precomputed=false)
(3.385652998856112, 3.3856529988561115)

```
```jldoctest V_motifs_bicm_local
julia> V_motifs(model, 5, 1, layer=:top), V_motifs(model, 5, 1, layer=:top, precomputed=false)
(9.46988024296453, 9.469880242964528)

```
"""
function V_motifs(m::BiCM, i::Int, j::Int; layer::Symbol=:bottom, precomputed::Bool=false)
    # checks
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters (m.Θᵣ) must be computed for `m` first, see `solve_model!`"))
    
    if precomputed
        # checks
        m.status[:G_computed] ? nothing : throw(ArgumentError("The expected values (m.Ĝ) of the *biadjecency matrix* must be computed for `m` first, see `set_Ĝ!`"))

        # compute
        return V_motifs(m.Ĝ, i, j; layer=layer, skipchecks=true)

    else
        # compute
        res = zero(eltype(m.θᵣ)) 
        if layer ∈ [:bottom; :⊥]
            i_red = m.d⊥ᵣ_ind[i]
            j_red = m.d⊥ᵣ_ind[j]
            for α in eachindex(m.yᵣ)
                res += m.f⊤[α] * f_BiCM(m.xᵣ[i_red] * m.yᵣ[α]) * f_BiCM(m.xᵣ[j_red] * m.yᵣ[α])
            end
        elseif layer ∈ [:top; :⊤]
            α_red = m.d⊤ᵣ_ind[i]
            β_red = m.d⊤ᵣ_ind[j]
            for k in eachindex(m.xᵣ)
                res += m.f⊥[k] * f_BiCM(m.xᵣ[k] * m.yᵣ[α_red]) * f_BiCM(m.xᵣ[k] * m.yᵣ[β_red])
            end
        else
            throw(ArgumentError("The layer must be one of [:bottom, :⊥] for the bottom layer or [:top, :⊤] for the top layer."))
        end

        return res
    end
end


"""
    V_PB_parameters(m::BiCM, i::Int, j::Int; layer::Symbol=:bottom, precomputed::Bool=false)

Compute the parameters of the Poisson-Binomial distribution for the number of V-motifs between nodes `i` and `j` for the `BiCM` model `m`.

# Arguments
- `layer`: the layer can be specified by passing `layer=:bottom` or `layer=:top`.
- `precomputed`: if true, the expected values of the biadjacency matrix are used, otherwise the parameters are computed from the model parameters.

*Notes*: 
1. depending on the layer,  the tuple (`i`, `j`) denotes rows (:bottom) or columns (:top) of the biadjacency matrix `m.Ĝ`.

# Examples
```jldoctest V_PB_parameters_bicm_local
julia> model = BiCM(corporateclub());

julia> solve_model!(model);

julia> MaxEntropyGraphs.V_PB_parameters(model, 1, 1; layer=:bottom, precomputed=false);

```
"""
function V_PB_parameters(m::BiCM, i::Int, j::Int; layer::Symbol=:bottom, precomputed::Bool=false)
    # checks
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters (m.Θᵣ) must be computed for `m` first, see `solve_model!`"))

    if precomputed
        # checks
        m.status[:G_computed] ? nothing : throw(ArgumentError("The expected values (m.Ĝ) of the *biadjecency matrix* must be computed for `m` first, see `set_Ĝ!`"))
        
        # compute
        if layer ∈ [:bottom; :⊥]
            return m.Ĝ[i,:] .* m.Ĝ[j,:]
        elseif layer ∈ [:top; :⊤]
            return m.Ĝ[:,i] .* m.Ĝ[:,j]
        else
            throw(ArgumentError("The layer must be one of [:bottom, :⊥] for the bottom layer or [:top, :⊤] for the top layer."))
        end
    else
        if layer ∈ [:bottom; :⊥]
            # initialize
            res = zeros(eltype(m.θᵣ), m.status[:d⊤_unique])
            i_red = m.d⊥ᵣ_ind[i]
            j_red = m.d⊥ᵣ_ind[j]
            for α in eachindex(res) # instead of all, go only over the reduced ones and impute them => allocations and compute time reduced
                res[α] = f_BiCM(m.xᵣ[i_red] * m.yᵣ[α]) * f_BiCM(m.xᵣ[j_red] * m.yᵣ[α])
            end

            return res[m.d⊤ᵣ_ind]
        elseif layer ∈ [:top; :⊤]
            # initialise
            res = zeros(eltype(m.θᵣ), m.status[:d⊥_unique])
            α_red = m.d⊤ᵣ_ind[i]
            β_red = m.d⊤ᵣ_ind[j]
            for k in eachindex(res) # instead of all, go only over the reduced ones and impute them => allocations and compute time reduced
                res[k] = f_BiCM(m.xᵣ[k] * m.yᵣ[α_red]) * f_BiCM(m.xᵣ[k] * m.yᵣ[β_red])
            end

            return res[m.d⊥ᵣ_ind]
        else
            throw(ArgumentError("The layer must be one of [:bottom, :⊥] for the bottom layer or [:top, :⊤] for the top layer."))
        end
    end
end


# falling-factorial form of binomial(u, n), valid for real u (needed for expected/plug-in values).
# Delegates to Base's generic binomial(x::Number, k::Integer), which divides term-by-term and thus
# avoids the OverflowError that factorial(n) throws for n > 20.
_binomial_poly(u::T, n::Int) where {T<:Real} = binomial(float(u), n)

# harmonic-number difference H_u - H_{u-n} = Σ_{i=u-n+1}^{u} 1/i (zero when u < n), cf. Saracco et al. (2015), SI Eq. III.7
_harmonic_diff(u::Int, n::Int) = u < n ? 0.0 : sum(1/i for i in (u-n+1):u)

"""
    _Vn_aggregation_classes(m::BiCM, layer::Symbol)

Internal helper for the Vn/Λn motif family. `Vn`-motifs between nodes of `layer` aggregate over the
degrees of the *opposite* layer: return `(own_r, own_f, own_d, opp_r, opp_f)` where `own_r/own_f/own_d`
are the reduced parameters, frequencies and reduced degrees of the aggregated (opposite) layer and
`opp_r/opp_f` those generating the connection probabilities (one per node of `layer`).
"""
function _Vn_aggregation_classes(m::BiCM, layer::Symbol)
    if layer ∈ [:bottom; :⊥]
        return m.yᵣ, m.f⊤, m.d⊤ᵣ, m.xᵣ, m.f⊥
    elseif layer ∈ [:top; :⊤]
        return m.xᵣ, m.f⊥, m.d⊥ᵣ, m.yᵣ, m.f⊤
    else
        throw(ArgumentError("The layer must be one of [:bottom, :⊥] for the bottom layer or [:top, :⊤] for the top layer."))
    end
end

"""
    _Vn_exact_moments(m::BiCM, n::Int; layer::Symbol=:bottom)

Internal helper computing the **exact** mean and variance of ``N_{Vn} = \\sum_α \\binom{U_α}{n}`` under
the BiCM, where the sum runs over the nodes `α` of the layer opposite to `layer` and
``U_α = \\sum_k a_{kα}`` is the (random) degree of `α`. Each `U_α` follows a Poisson-binomial
distribution (the biadjacency entries are independent Bernoulli variables) and the `U_α` of distinct
nodes involve disjoint entry sets, hence are independent:

``⟨N_{Vn}⟩ = \\sum_α ⟨\\binom{U_α}{n}⟩, \\qquad Var[N_{Vn}] = \\sum_α Var[\\binom{U_α}{n}]``

The Poisson-binomial pmf of each `U_α` is obtained by direct convolution over the connection
probabilities, at cost `O(n_{opp}^2)` per unique degree class.
"""
function _Vn_exact_moments(m::BiCM, n::Int; layer::Symbol=:bottom)
    own_r, own_f, _, opp_r, opp_f = _Vn_aggregation_classes(m, layer)
    N = precision(m)
    nopp = sum(opp_f)
    μ, v = zero(N), zero(N)
    pmf = Vector{N}(undef, nopp + 1)
    for β in eachindex(own_r)
        # Poisson-binomial pmf of U_β by convolution over all opposite-layer nodes
        fill!(pmf, zero(N))
        pmf[1] = one(N)
        cnt = 0
        for k in eachindex(opp_r)
            p = N(f_BiCM(opp_r[k] * own_r[β]))
            q = one(N) - p
            for _ in 1:opp_f[k]
                cnt += 1
                @inbounds for u in cnt:-1:1
                    pmf[u+1] = pmf[u+1] * q + pmf[u] * p
                end
                @inbounds pmf[1] *= q
            end
        end
        # moments of binomial(U_β, n)
        m1, m2 = zero(N), zero(N)
        for u in n:nopp
            g = _binomial_poly(N(u), n)
            @inbounds w = pmf[u+1]
            m1 += w * g
            m2 += w * g * g
        end
        μ += own_f[β] * m1
        v += own_f[β] * (m2 - m1^2)
    end
    return μ, v
end

"""
    _Vn_observed(m::BiCM, n::Int, layer::Symbol)

Internal helper: the observed count ``N_{Vn} = \\sum_α \\binom{u_α}{n}`` computed from the observed
degree sequence of the aggregated (opposite) layer stored in the model (no graph needed).
"""
function _Vn_observed(m::BiCM, n::Int, layer::Symbol)
    _, own_f, own_d, _, _ = _Vn_aggregation_classes(m, layer)
    N = precision(m)
    return sum(own_f[β] * (own_d[β] < n ? zero(N) : N(_binomial_poly(N(own_d[β]), n))) for β in eachindex(own_d))
end

"""
    _Vn_delta_inputs(m::BiCM, layer::Symbol)

Internal helper for the closed-form (`method=:delta`) Vn machinery: return `(uᵣ, fᵣ, s2ᵣ)` with the
reduced observed degrees `uᵣ` of the aggregated (opposite) layer, their frequencies `fᵣ` and the
variance of each random degree ``σ^2[U_α] = \\sum_k p_{kα}(1 - p_{kα})`` per reduced class.
"""
function _Vn_delta_inputs(m::BiCM, layer::Symbol)
    own_r, own_f, own_d, opp_r, opp_f = _Vn_aggregation_classes(m, layer)
    N = precision(m)
    s2 = zeros(N, length(own_r))
    for β in eachindex(own_r)
        acc = zero(N)
        for k in eachindex(opp_r)
            p = N(f_BiCM(opp_r[k] * own_r[β]))
            acc += opp_f[k] * p * (1 - p)
        end
        s2[β] = acc
    end
    return own_d, own_f, s2
end

"""
    Vn_motifs(A::T, n::Int; layer::Symbol=:bottom, skipchecks::Bool=false) where {T<:AbstractMatrix}

Count the total number of `Vn`-motifs (`n`-fold co-occurrences, i.e. `n` nodes of a layer sharing a
common neighbour in the other layer) in the biadjacency matrix `A` for one of its layers:
``N_{Vn} = \\sum_α \\binom{u_α}{n}`` where the `u_α` are the degrees of the *opposite* layer.
For `n = 2` this is the V-motif count ([`V_motifs`](@ref)); following Saracco et al. (2015), the
family for the bottom (top) layer is also known as the `Vn` (`Λn`) family.

# Arguments
- `n`: motif order (`n ≥ 2`).
- `layer`: the layer between whose nodes the co-occurrences are counted (`:bottom` or `:top`).
- `skipchecks`: if true, skip the dimension check on `A`.

# Examples
```jldoctest Vn_motifs_bipartite_matrix
julia> A = [1 1; 1 1; 1 0];

julia> Vn_motifs(A, 2, layer=:bottom)
4.0

julia> Vn_motifs(A, 3, layer=:bottom)
1.0

```
"""
function Vn_motifs(A::T, n::Int; layer::Symbol=:bottom, skipchecks::Bool=false) where {T<:AbstractMatrix}
    n ≥ 2 || throw(ArgumentError("The motif order `n` must be at least 2"))
    if !skipchecks && isequal(size(A)...)
        @warn "The matrix `A` is square, make sure it is a biadjacency matrix."
    end
    if layer ∈ [:bottom; :⊥]
        u = vec(sum(A, dims=1))     # degrees of the ⊤ layer (columns)
    elseif layer ∈ [:top; :⊤]
        u = vec(sum(A, dims=2))     # degrees of the ⊥ layer (rows)
    else
        throw(ArgumentError("The layer must be one of [:bottom, :⊥] for the bottom layer or [:top, :⊤] for the top layer."))
    end
    return sum(uα < n ? zero(float(eltype(A))) : _binomial_poly(float(uα), n) for uα in u)
end

"""
    Vn_motifs(m::BiCM, n::Int; layer::Symbol=:bottom, method::Symbol=:exact)

Compute the expected total number of `Vn`-motifs ``⟨N_{Vn}⟩`` between the nodes of `layer` under the
BiCM model `m` (cf. [`Vn_motifs(::AbstractMatrix, ::Int)`](@ref) for the definition).

# Arguments
- `n`: motif order (`n ≥ 2`).
- `layer`: the layer between whose nodes the co-occurrences are counted (`:bottom` or `:top`).
- `method`:
    - `:exact` (default): exact expectation from the Poisson-binomial distribution of the random
      opposite-layer degrees (see [`_Vn_exact_moments`](@ref MaxEntropyGraphs._Vn_exact_moments)); any `n ≥ 2`.
    - `:delta`: closed-form Taylor expansion around the observed degrees (Saracco et al. (2015), SI
      Eqs. III.9-III.13); only `n ∈ {2, 3, 4}` (exact for `n = 2`). Accurate when the opposite-layer
      degrees are large compared to `n`; for sparse layers prefer `:exact`.

# Examples
```jldoctest Vn_motifs_bicm
julia> model = BiCM(corporateclub());

julia> solve_model!(model);

julia> Vn_motifs(model, 2, layer=:bottom) ≈ V_motifs(model, layer=:bottom, precomputed=false)
true

```
"""
function Vn_motifs(m::BiCM, n::Int; layer::Symbol=:bottom, method::Symbol=:exact)
    n ≥ 2 || throw(ArgumentError("The motif order `n` must be at least 2"))
    m.status[:params_computed] ? nothing : throw(ArgumentError("The likelihood maximising parameters must be computed for `m` first, see `solve_model!`"))
    if method == :exact
        return _Vn_exact_moments(m, n; layer=layer)[1]
    elseif method == :delta
        uᵣ, fᵣ, s2 = _Vn_delta_inputs(m, layer)
        obs = _Vn_observed(m, n, layer)
        if n == 2
            shift = sum(fᵣ[β] * s2[β] / 2 for β in eachindex(uᵣ))
        elseif n == 3
            shift = sum(fᵣ[β] * s2[β] * (uᵣ[β] - 1) / 2 for β in eachindex(uᵣ))
        elseif n == 4
            shift = sum(fᵣ[β] * (3s2[β]^2 + s2[β] * (6uᵣ[β]^2 - 18uᵣ[β] + 11)) / 24 for β in eachindex(uᵣ))
        else
            throw(ArgumentError("The closed-form (:delta) expectation is only available for n ∈ {2, 3, 4}, use method=:exact instead"))
        end
        return obs + shift
    else
        throw(ArgumentError("Invalid method, only :exact and :delta are accepted"))
    end
end

"""
    Vn_sigma(m::BiCM, n::Int; layer::Symbol=:bottom, method::Symbol=:exact)

Compute the standard deviation ``σ[N_{Vn}]`` of the total number of `Vn`-motifs between the nodes of
`layer` under the BiCM model `m`.

# Arguments
- `n`: motif order (`n ≥ 2`).
- `layer`: the layer between whose nodes the co-occurrences are counted (`:bottom` or `:top`).
- `method`:
    - `:exact` (default): exact variance from the Poisson-binomial distribution of the random
      opposite-layer degrees (independent across nodes, see
      [`_Vn_exact_moments`](@ref MaxEntropyGraphs._Vn_exact_moments)); any `n ≥ 2`.
    - `:delta`: first-order delta method around the observed degrees (Saracco et al. (2015), SI
      Eqs. III.6-III.7): ``σ[N_{Vn}] = \\sqrt{\\sum_α [\\binom{u_α}{n}(H_{u_α} - H_{u_α-n})]^2 σ^2[U_α]}``.
      Accurate when the opposite-layer degrees are large compared to `n`; it *underestimates* the
      variance for sparse layers — prefer `:exact`.
"""
function Vn_sigma(m::BiCM, n::Int; layer::Symbol=:bottom, method::Symbol=:exact)
    n ≥ 2 || throw(ArgumentError("The motif order `n` must be at least 2"))
    m.status[:params_computed] ? nothing : throw(ArgumentError("The likelihood maximising parameters must be computed for `m` first, see `solve_model!`"))
    if method == :exact
        return sqrt(_Vn_exact_moments(m, n; layer=layer)[2])
    elseif method == :delta
        uᵣ, fᵣ, s2 = _Vn_delta_inputs(m, layer)
        N = precision(m)
        acc = zero(N)
        for β in eachindex(uᵣ)
            uᵣ[β] < n && continue
            deriv = _binomial_poly(N(uᵣ[β]), n) * _harmonic_diff(uᵣ[β], n)
            acc += fᵣ[β] * deriv^2 * s2[β]
        end
        return sqrt(acc)
    else
        throw(ArgumentError("Invalid method, only :exact and :delta are accepted"))
    end
end

"""
    Vn_zscore(m::BiCM, n::Int; layer::Symbol=:bottom, method::Symbol=:exact)

Compute the z-score of the observed total number of `Vn`-motifs between the nodes of `layer` with
respect to the BiCM model `m`:
``z = (N_{Vn}^{obs} - ⟨N_{Vn}⟩)/σ[N_{Vn}]``.
The observed count is obtained from the observed degree sequence stored in the model (no graph needed).

!!! note
    Because the constrained degrees fully determine the observed count, these z-scores have a definite
    sign (cf. Saracco et al. (2015), SI): under the BiCM, ``⟨N_{Vn}⟩ ≥ N_{Vn}^{obs}``. The associated
    significance tests are one-sided.

# Arguments
- `n`: motif order (`n ≥ 2`).
- `layer`: the layer between whose nodes the co-occurrences are counted (`:bottom` or `:top`).
- `method`: `:exact` (default, any `n ≥ 2`) or `:delta` (`n ∈ {2, 3, 4}`); see
  [`Vn_motifs`](@ref) and [`Vn_sigma`](@ref).
"""
function Vn_zscore(m::BiCM, n::Int; layer::Symbol=:bottom, method::Symbol=:exact)
    n ≥ 2 || throw(ArgumentError("The motif order `n` must be at least 2"))
    m.status[:params_computed] ? nothing : throw(ArgumentError("The likelihood maximising parameters must be computed for `m` first, see `solve_model!`"))
    obs = _Vn_observed(m, n, layer)
    if method == :exact
        # single pass: the Poisson-binomial convolution yields mean and variance together
        μ, v = _Vn_exact_moments(m, n; layer=layer)
        return (obs - μ) / sqrt(v)
    else
        return (obs - Vn_motifs(m, n; layer=layer, method=method)) / Vn_sigma(m, n; layer=layer, method=method)
    end
end


"""
    project(m::BiCM;   α::Float64=0.05, layer::Symbol=:bottom, precomputed::Bool=true,
                        distribution::Symbol=:Poisson, adjustment::PValueAdjustment=BenjaminiHochberg(),
                            multithreaded::Bool=false)

Obtain the statistically validated projected graph of the BiCM model `m` onto the layer `layer` using the V-motifs and the significance level `α` combined with the p-value adjustment method `adjustment`.

# Arguments
- `α`: the significance level.
- `layer`: the layer can be specified by passing `layer=:bottom` or `layer=:top`.
- `precomputed`: if true, the expected values of the biadjacency matrix are used, otherwise the parameters are computed from the model parameters.
- `distribution`: the distribution used to compute the p-values. This can be `:Poisson` or `:PoissonBinomial`.
- `adjustment`: the method used to adjust the p-values for multiple testing. This can be any of the methods in the `PValueAdjustment` type (see `MultipleTesting.jl`). By default, the Benjamini-Hochberg method is used.
- `multithreaded`: if true, the p-values are computed using multithreading.

# Examples
```jldoctest project_bicm
julia> model = BiCM(corporateclub());

julia> solve_model!(model);

julia> project(model, layer=:bottom)
{25, 0} undirected simple Int64 graph

```
```jldoctest project_bicm
julia> project(model, layer=:top)
{15, 0} undirected simple Int64 graph

```
"""
function project(m::BiCM;   α::Float64=0.05, layer::Symbol=:bottom, precomputed::Bool=false, 
                            distribution::Symbol=:Poisson, adjustment::MultipleTesting.PValueAdjustment=MultipleTesting.BenjaminiHochberg(),
                            multithreaded::Bool=false)
    # checks
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters (m.Θᵣ) must be computed for `m` first, see `solve_model!`"))
    if precomputed
        m.status[:G_computed] ? nothing : throw(ArgumentError("The expected values (m.Ĝ) must be computed first when using `precomputed=true` see `set_Ĝ!`"))
    end
    0. ≤ α ≤ 1. || throw(ArgumentError("The significance level `α` must be between 0 and 1."))
    if !(layer ∈ [:bottom; :⊥; :top; :⊤])
        throw(ArgumentError("The layer must be one of [:bottom, :⊥] for the bottom layer or [:top, :⊤] for the top layer."))
    end
    if !(distribution ∈ [:Poisson; :PoissonBinomial])
        throw(ArgumentError("The distribution must be one of [:Poisson, :PoissonBinomial]."))
    end

    # compute the observed V-motifs (upper triangular part)
    V_ij_obs = triu!(project(biadjacency_matrix(m.G), layer=layer, method=:weighted))
    (row_indices, col_indices) = findnz(V_ij_obs)

    # determine the distributions and compute the p-values
    if multithreaded
        # initiate p-values
        pvals = zeros(precision(m), length(V_ij_obs.nzval))
        # compute p-values
        Threads.@threads for i in eachindex(pvals)
            if distribution == :Poisson
                @inbounds pvals[i] = 1 - cdf(Poisson(V_motifs(m, row_indices[i], col_indices[i], layer=layer, precomputed=precomputed)), V_ij_obs.nzval[i] - 1)
            elseif distribution == :PoissonBinomial
                # get pvalues
                @inbounds pvals[i] = 1 - cdf(PoissonBinomial(V_PB_parameters(m, row_indices[i], col_indices[i], layer=layer, precomputed=precomputed)), V_ij_obs.nzval[i] - 1)
            end
        end
    else
        if distribution == :Poisson
            # get pvalues
            pvals = [1 - cdf(Poisson(V_motifs(m, pair...; layer=layer, precomputed=precomputed)), V_ij_obs[pair...]-1) for pair in zip(row_indices, col_indices)]
        elseif distribution == :PoissonBinomial
            # get pvalues
            pvals = [1 - cdf(PoissonBinomial(V_PB_parameters(m, pair...; layer=layer, precomputed=precomputed)), V_ij_obs[pair...]-1) for pair in zip(row_indices, col_indices)]
        end
    end

    # due to the way the cdf is computed, the cdf values can be larger than 1 (up to machine precision),
    # so we need to retify this, to avoid having negative p-values (up to machine precision e.g. -7.771561172376096e-15)
    pvals = max.(pvals, 0.)

    # adjust p-values for multiple testing
    pvals_adj = MultipleTesting.adjust(pvals, adjustment)

    # get the significant ones
    sig_idx = pvals_adj .< α

    # make edge iterator
    edge_iter = (Graphs.SimpleEdge(e[1], e[2]) for e in zip(row_indices[sig_idx], col_indices[sig_idx]))

    # generate return graph
    G = Graphs.SimpleGraphFromIterator(edge_iter)

    while Graphs.nv(G) < (layer ∈ (:bottom, :⊥) ? m.status[:N⊥] : m.status[:N⊤])
        Graphs.add_vertex!(G)
    end

    return G

end