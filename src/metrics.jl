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
        throw(ArgumentError( "The matrix is not symmetrical. Consider using ANND_in or ANND_out instead."))
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

triangles(m::UBCM) = triangles(m.Ĝ,check_dimensions=false, check_directed=false)

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
        throw(ArgumentError( "The matrix is not symmetrical. Consider using ANND_in or ANND_out instead."))
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

squares(m::UBCM) = squares(m.Ĝ, check_dimensions=false, check_directed=false)


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


#= OLDER VERSION
#For undirected, unweighted, and unsigned networks, four types of triads exist: (1) triads without ties/edges (empty triads); (2) triads with one tie present, and two ties absent (one edge triads); (3) triads with one edge absent, and two edges present, referred to in the literature as two-path, two-star, or open triads (or forbidden triads in weighted networks when present edges are strong); and (4) triads with all edges present (triangles, closed triads) (Triads should not be confused with triplets. 

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

    return eval.(map(f -> :($(f)($M.G)), [DBCM_motif_functions[i] for i in n])) 
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
outdegree_dist(m::DBCM, i::Int) = Distributions.PoissonBinomial(m.G[i, axes(m.G, 2) .!= i])

"""
    outdegree_dist(m::DBCM)

Compute the Poisson-Binomial distribution for the outdegree for all nodes for the `DBCM` model `m`.
"""
outdegree_dist(m::DBCM) = map(i -> outdegree_dist(m, i), axes(m.G,1))


=#