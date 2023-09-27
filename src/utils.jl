#############################################################################
# utils.jl
#
# This file contains utility functions for the MaxEntropyGraphs.jl package
#############################################################################


"""
    np_unique_clone(x::Vector; sorted::Bool=false)

Julia replication of the numpy.unique(a, return_counts=True, return_index=True, return_inverse=True) function from Python.

Returns a tuple of:
 - vector of unique values in x
 - vector of indices of the first occurence of each unique value in x. Follows the same order as the unique values.
 - vector of inverse indices of the original data in the unique values
 - vector of the counts of the unique values in x. Follows the same order as the unique values.

 If sorted is true, the unique values are sorted by size and the other vectors are sorted accordingly.

# Examples     
```jldoctest
julia> x = [1;2;2;4;1];

julia> np_unique_clone(x)
([1, 2, 4], [1, 2, 4], [1, 2, 2, 3, 1], [2, 2, 1])

julia> x = [10;9;9;8];

julia> np_unique_clone(x, sorted=true)
([8, 9, 10], [4, 2, 1], [3, 2, 2, 1], [1, 2, 1])
```
"""
function np_unique_clone(x::Vector; sorted::Bool=true)
    T = eltype(x)
    unique_x =  Vector{T}()         # unique values
    seen =      Set{T}()            # unique values set
    index =     Dict{T,Int}()       # first occurence of unique values (index)
    reverse_index = Dict{T,Int}()   # position of value in unique values
    counts =    Dict{T, Int}()      # unique values counts
    
    for i in eachindex(x)
        if !in(x[i], seen)
            push!(seen, x[i])
            push!(unique_x, x[i])
            index[x[i]] = i
            reverse_index[x[i]] = length(unique_x) #
            counts[x[i]] = 1
        else
            counts[x[i]] += 1
        end

    end

    first_occ = [index[v] for v in unique_x]
    inverse_index = [reverse_index[v] for v in x]
    freqs = [counts[v] for v in unique_x]
    
    if sorted
        # sorted indices
        inds = sortperm(unique_x)
        # sort vectors
        unique_x = unique_x[inds]
        first_occ = first_occ[inds]
        freqs = freqs[inds]
        
        # Create a map from old indices to new indices
        new_indices_map = Dict(inds[i] => i for i in 1:length(inds))
        
        # Update inverse_index using the map
        inverse_index = [new_indices_map[i] for i in inverse_index]
    end

    return unique_x, first_occ, inverse_index, freqs
end


"""
    log_nan(x::T)

Same as `log(x)`, but returns `NaN` if `x <= 0`. Inspired by `NaNMath.jl` and https://github.com/JuliaMath/NaNMath.jl/issues/63. This methods is prefered
over the ones from `NaNMath.jl` version because it does not require a foreign call expression to be evaluated, hence autodiff methods can be used with this.

# Examples     
```jldoctest
julia> MaxEntropyGraphs.log_nan(10.)
2.302585092994046

julia> MaxEntropyGraphs.log_nan(-10.)
NaN
```
"""
function log_nan(x::T)::T where {T<:Real}
    x <= T(0) && return T(NaN)
    return log(x)
end



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

instrength(G::SimpleWeightedGraphs.AbstractSimpleWeightedGraph, T::DataType=SimpleWeightedGraphs.weighttype(G); dir::Symbol=:in)   = strength(G, T, dir=dir)
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





#     DBCM_analysis

# Compute the z-scores etc. for all motifs and the degrees for a `SimpleDiGraph`. Returns a Dict for storage of the computed results

# G: the network
# N_min: minimum sample length used for computing metrics
# N_max: maximum sample length used for computing metrics
# n_sample_lengths: number of values in the domain [N_min, N_max]
# """
# function DBCM_analysis(  G::T;
#                                 N_min::Int=100, 
#                                 N_max::Int=10000, 
#                                 n_sample_lengths::Int=3,
#                                 subsamples::Vector{Int64}=round.(Int,exp10.(range(log10(N_min),log10(N_max), length=n_sample_lengths))), kwargs...) where T<:Graphs.SimpleDiGraph
#     @info "$(round(now(), Minute)) - Started DBCM motif analysis"
#     NP = PyCall.pyimport("NEMtropy")
#     G_nem =  NP.DirectedGraph(degree_sequence=vcat(Graphs.outdegree(G), Graphs.indegree(G)))
#     G_nem.solve_tool(model="dcm_exp", method="fixed-point", initial_guess="degrees", max_steps=3000)
#     if abs(G_nem.error) > 1e-6
#         @warn "Method did not converge"
#     end
#     # generate the model
#     model = DBCM(G_nem.x, G_nem.y)
#     @assert Graphs.indegree(model)  ≈ Graphs.indegree(G)
#     @assert Graphs.outdegree(model) ≈ Graphs.outdegree(G)
#     # generate the sample
#     @info "$(round(now(), Minute)) - Generating sample"
#     S = [rand(model) for _ in 1:N_max]

#     #################################
#     # motif part
#     #################################

#     # compute motif data
#     @info "$(round(now(), Minute)) - Computing observed motifs in the observed network"
#     mˣ = motifs(G)
#     @info "$(round(now(), Minute)) - Computing expected motifs for the DBCM model"
#     m̂  = motifs(model)
#     σ̂_m̂  = Vector{eltype(model.G)}(undef, length(m̂))
#     for i = 1:length(m̂)
#         @info "$(round(now(), Minute)) - Computing standard deviation for motif $(i)"
#         σ̂_m̂[i] = σˣ(DBCM_motif_functions[i], model)
#     end
#     # compute z-score (Squartini)
#     z_m_a = (mˣ - m̂) ./ σ̂_m̂  
#     @info "$(round(now(), Minute)) - Computing expected motifs for the sample"
#     #S_m = hcat(motifs.(S,full=true)...); # computed values from sample
#     S_m = zeros(13, length(S))
#     Threads.@threads for i in eachindex(S)
#         S_m[:,i] .= motifs(S[i], full=true)
#     end
#     m̂_S =   hcat(map(n -> reshape(mean(S_m[:,1:n], dims=2),:), subsamples)...)
#     σ̂_m̂_S = hcat(map(n -> reshape( std(S_m[:,1:n], dims=2),:), subsamples)...)
#     z_m_S = (mˣ .- m̂_S) ./ σ̂_m̂_S

#     #################################       
#     # degree part
#     #################################
#     # compute degree sequence
#     @info "$(round(now(), Minute)) - Computing degrees in the observed network"
#     d_inˣ, d_outˣ = Graphs.indegree(G), Graphs.outdegree(G)
#     @info "$(round(now(), Minute)) - Computing expected degrees for the DBCM model"
#     d̂_in, d̂_out = Graphs.indegree(model), Graphs.outdegree(model)
#     @info "$(round(now(), Minute)) - Computing standard deviations for the degrees for the DBCM model"
#     σ̂_d̂_in, σ̂_d̂_out = map(j -> σˣ(m -> Graphs.indegree(m, j), model), 1:length(model)), map(j -> σˣ(m -> Graphs.outdegree(m, j), model), 1:length(model))
#     # compute degree z-score (Squartini)
#     z_d_in_sq, z_d_out_sq = (d_inˣ - d̂_in) ./ σ̂_d̂_in, (d_outˣ - d̂_out) ./ σ̂_d̂_out
#     @info "$(round(now(), Minute)) - Computing distributions for degree sequences"
#     d_in_dist, d_out_dist = indegree_dist(model), outdegree_dist(model)
#     z_d_in_dist, z_d_out_dist = (d_inˣ - mean.(d_in_dist)) ./ std.(d_in_dist), (d_outˣ - mean.(d_out_dist)) ./ std.(d_out_dist)

#     # compute data for the sample
#     @info "$(round(now(), Minute)) - Computing degree sequences for the sample"
#     d_in_S, d_out_S = hcat(Graphs.indegree.(S)...), hcat(Graphs.outdegree.(S)...)
#     d̂_in_S, d̂_out_S = hcat(map(n -> reshape(mean(d_in_S[:,1:n], dims=2),:), subsamples)...), hcat(map(n -> reshape(mean(d_out_S[:,1:n], dims=2),:), subsamples)...)
#     σ̂_d_in_S, σ̂_d_out_S = hcat(map(n -> reshape(std( d_in_S[:,1:n], dims=2),:), subsamples)...), hcat(map(n -> reshape( std(d_out_S[:,1:n], dims=2),:), subsamples)...)
#     # compute degree z-score (sample)
#     z_d_in_S, z_d_out_S = (d_inˣ .- d̂_in_S) ./ σ̂_d_in_S, (d_outˣ .- d̂_out_S) ./ σ̂_d_out_S
    

#     @info "$(round(now(), Minute)) - Finished"
#     return Dict(:network => G,
#                 :model => model,
#                 :error => G_nem.error,
#                 # motif information
#                 :mˣ => mˣ,          # observed
#                 :m̂ => m̂,            # expected squartini
#                 :σ̂_m̂ => σ̂_m̂,        # standard deviation squartini
#                 :z_m_a => z_m_a,    # z_motif squartini
#                 :S_m => S_m,        # sample data
#                 :m̂_S => m̂_S,        # expected motifs in sample
#                 :σ̂_m̂_S => σ̂_m̂_S,    # standard deviation motifs sample
#                 :z_m_S => z_m_S,    # z_motif sample
#                 # in/outdegree information
#                 :d_inˣ => d_inˣ,                # observed
#                 :d_outˣ => d_outˣ,              # observed
#                 :d̂_in => d̂_in,                  # expected squartini
#                 :d̂_out => d̂_out,                # expected squartini
#                 :σ̂_d̂_in => σ̂_d̂_in,              # standard deviation squartini
#                 :σ̂_d̂_out => σ̂_d̂_out,            # standard deviation squartini
#                 :z_d_in_sq => z_d_in_sq,        # z_degree squartini
#                 :z_d_out_sq => z_d_out_sq,      # z_degree squartini
#                 :d̂_in_S => d̂_in_S,              # expected sample
#                 :d̂_out_S => d̂_out_S,            # expected sample
#                 :σ̂_d_in_S => σ̂_d_in_S,          # standard deviation sample
#                 :σ̂_d_out_S => σ̂_d_out_S,        # standard deviation sample
#                 :z_d_in_S => z_d_in_S,          # z_degree sample
#                 :z_d_out_S => z_d_out_S,        # z_degree sample
#                 :z_d_in_dist => z_d_in_dist,    # z_degree distribution (analytical PoissonBinomial)
#                 :z_d_out_dist => z_d_out_dist,  # z_degree distribution (analytical PoissonBinomial)
#                 :d_in_dist => d_in_dist,        # distribution (analytical PoissonBinomial)
#                 :d_out_dist => d_out_dist,      # distribution (analytical PoissonBinomial)
#                 :d_in_S => d_in_S,              # indegree sample
#                 :d_out_S => d_out_S             # indegree sample
#                 ) 
                
# end


# """
#     write_result(outfile::String, label::Union{String, Symbol}, data)

# write out data to jld file. Checks for existance of the file and appends if it exists.

# outfile::String - path to output file
# label - label for the data in the file
# data - actual data to write to the file
# """
# function write_result(outfile::String, label::Union{String, SubString{String}, Symbol}, data)
#     outfile = endswith(outfile, ".jld") ? outfile : outfile * ".jld"
#     # append or create file
#     JLD2.jldopen(outfile, isfile(outfile) ? "r+" : "w") do file
#         write(file, String(label), data)
#     end
# end

# """
#     produce_squartini_dbcm_data

# utility function to reproduce the data from the original 2011 Squartini paper (https://arxiv.org/abs/1103.0701)
# """
# function produce_squartini_dbcm_data(output = "./data/computed_results/DBCM_result_more.jld",
#                                      netdata = "./data/networks")
#     for network in filter(x-> occursin("_directed", x), joinpath.(netdata, readdir(netdata)))
#         @info "working on $(network)"
#         # reference name for storage
#         refname = split(split(network,"/")[end],"_")[1]
#         @info refname
#         # load network
#         G = Graphs.loadgraph(network)
#         # compute motifs
#         res = DBCM_analysis(G)
#         # write out results
#         write_result(output, refname, res)
#     end
# end

# function readpajek(f::String; is_directed::Bool=true)
#     G = is_directed ? Graphs.SimpleDiGraph() : Graphs.SimpleGraph()
#     nv = r"\*vertices\s(\d+)"
#     arcs = r"\*arcs"
#     arc = r"(\d+)\s+(\d+)"
#     arcing, finished = false, false
#     for line in readlines(f)
#         if !isnothing(match(nv, line))
#             N = parse(Int,(match(nv, line).captures[1]))
#             Graphs.nv(G) == 0 && Graphs.add_vertices!(G, N) 
#         end
#         if !isnothing(match(arcs, line))
#             arcing = true
#             continue
#         end

#         if arcing 
#             # check empty line
#             if isempty(line)
#                 arcing = false
#                 return G
#             end

#             src, dst = parse.(Int, match(arc, line).captures)
#             Graphs.add_edge!(G, src, dst)
#         end

#     end
    
# end