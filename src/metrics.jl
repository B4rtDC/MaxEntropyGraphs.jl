# ----------------------------------------------------------------------------------------------------------------------
#
#                                               Supporting network functions
#
# Note: the function working on matrices need to be defined without contraining the types too much
#       otherwise there will be a problem when using the autodiff package.
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

    # compute
    return [ANND(A,i, check_dimensions=false, check_directed=false) for i in vs]
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

    # compute
    return [ANND_out(A,i, check_dimensions=false) for i in vs]
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

    # compute
    return [ANND_in(A,i, check_dimensions=false) for i in vs]
end

function ANND(m::UBCM)
    # checks
    m.status[:G_computed] ? nothing : throw(ArgumentError("The expected values (m.Ĝ) must be computed for `m` before computing the standard deviation of metric `X`, see `set_Ĝ!`"))

    # compute
    return ANND(m.Ĝ)
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

julia> (triangles(G), triangles(MaxEntropyGraphs.Graphs.adjacency_matrix(G)), triangles(model))
(45, 45.0, 52.849301363026846)
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

    # compute
    res = zero(eltype(A))
    for i = axes(A,1)
        for j = axes(A,1)
            @simd for k = axes(A,1)
                if i ≠ j && j ≠ k && k ≠ i
                    res += A[i,j] * A[j,k] * A[k,i]
                end
            end
        end
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

julia> (squares(G), squares(MaxEntropyGraphs.Graphs.adjacency_matrix(G)), squares(model))
(36.0, 36.0, 45.644736823949344)
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

    # compute
    res = zero(eltype(A))
    for i = axes(A,1)
        for j = axes(A,1)
            if j≠i
                for k = axes(A,1)
                    if k≠i && k≠j
                        @simd for l in axes(A,1)
                            if l≠i && l≠j && l≠k
                                res += A[i,j] * A[j,k] * A[k,l] * A[l,i] * (one(eltype(A)) - A[i,k]) * (one(eltype(A)) -  A[l,j])
                            end
                        end
                    end
                end
            end
        end
    end

    return res / 8
end

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
# use metaprogramming to generate the functions for the 13 directed 3-node motifs for a directed graph
for i = 1:13
    fname = directed_graph_motif_function_names[i]
    @eval begin
        # method based on matrix
        """
            $($fname)(A::T) where T<:AbstractArray
        
        Count the occurence of motif $($fname) (Σ_{i≠j≠k} $(directed_graph_motif_functions[$i][1])(i,j) $(directed_graph_motif_functions[$i][2])(j,k) $(directed_graph_motif_functions[$i][3])(k,i) ) from the adjacency matrix.
        """
        function $(fname)(A::T)  where T<:AbstractArray
            res = zero(eltype(A))
            for i = axes(A,1)
                for j = axes(A,1)
                    @simd for k = axes(A,1)
                        if i ≠ j && j ≠ k && k ≠ i
                            res += $(directed_graph_motif_functions[i][1])(A,i,j) * $(directed_graph_motif_functions[i][2])(A,j,k) *   $(directed_graph_motif_functions[i][3])(A,k,i)
                        end
                    end
                end
            end
            return res
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
    end
end

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
    res = zero(eltype(A))
    # determine 
    if layer ∈ [:bottom; :⊥]
        for i in axes(A, 1)
            for j in axes(A, 1)
                if j > i
                    res += @views dot(A[i,:], A[j,:]) # possible perfomance gain with custom dot function
                end
            end
        end
    elseif layer ∈ [:top; :⊤]
        for i in axes(A, 2)
            for j in axes(A, 2)
                if j > i
                    res += @views dot(A[:,i], A[:,j]) # possible perfomance gain with custom dot function
                end
            end
        end
    else
        throw(ArgumentError("The layer must be one of [:bottom, :⊥] for the bottom layer or [:top, :⊤] for the top layer."))
    end
    
    return res
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

julia> V_motifs(model, layer=:bottom), V_motifs(model, layer=:bottom, precomputed=false)
(449.2569925909879, 449.2569925909879)

```
```jldoctest V_motifs_bicm
julia> V_motifs(model, layer=:top), V_motifs(model, layer=:top, precomputed=false)
(180.2569926636081, 180.2569926636081)
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
    V_motifs(m::BiCM, i::Int, j::Int; layer::Symbol=:bottom, precomputed::Bool=true)

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