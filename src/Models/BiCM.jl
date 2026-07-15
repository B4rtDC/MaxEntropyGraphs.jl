
"""
    BiCM

Maximum entropy model for the Undirected Bipartite Configuration Model (BiCM). 
    
The object holds the maximum likelihood parameters of the model (θ), the expected bi-adjacency matrix (Ĝ), 
and the variance for the elements of the adjacency matrix (σ).
"""
mutable struct BiCM{T<:Union{Graphs.AbstractGraph, Nothing}, N<:Real} <: AbstractMaxEntropyModel
    "Graph type, can be any bipartite subtype of AbstractGraph, but will be converted to SimpleGraph for the computation" # can also be empty
    const G::T 
    "Maximum likelihood parameters for reduced model" 
    const θᵣ::Vector{N}
    "Exponentiated maximum likelihood parameters for reduced model ( xᵢ = exp(-αᵢ) )"
    const xᵣ::Vector{N}
    "Exponentiated maximum likelihood parameters for reduced model ( yᵢ = exp(-βᵢ) )"
    const yᵣ::Vector{N}
    "Degree sequence of the ⊥ layer" # evaluate usefulness of this field later on
    const d⊥::Vector{Int}
    "Degree sequence of the ⊤ layer" # evaluate usefulness of this field later on
    const d⊤::Vector{Int}
    "Reduced degree sequence of the ⊥ layer"
    const d⊥ᵣ::Vector{Int}
    "Reduced degree sequence of the ⊤ layer"
    const d⊤ᵣ::Vector{Int}
    "Non-zero elements of the reduced degree sequence of the ⊥ layer"
    const d⊥ᵣ_nz::UnitRange{Int}
    "Non-zero elements of the reduced degree sequence of the ⊤ layer"
    const d⊤ᵣ_nz::UnitRange{Int}
    "Frequency of each degree in the ⊥ layer"
    const f⊥::Vector{Int}
    "Frequency of each degree in the ⊤ layer"
    const f⊤::Vector{Int}
    "Indices to reconstruct the degree sequence from the reduced degree sequence of the ⊥ layer"
    const d⊥_ind::Vector{Int}
    "Indices to reconstruct the degree sequence from the reduced degree sequence of the ⊤ layer"
    const d⊤_ind::Vector{Int}
    "Indices to reconstruct the reduced degree sequence from the degree sequence of the ⊥ layer"
    const d⊥ᵣ_ind::Vector{Int}
    "Indices to reconstruct the reduced degree sequence from the degree sequence of the ⊤ layer"
    const d⊤ᵣ_ind::Vector{Int}
    "membership of the ⊥ layer"
    const ⊥nodes::Vector{Int}
    "membership of the ⊤ layer"
    const ⊤nodes::Vector{Int}
    "Expected bi-adjacency matrix" # not always computed/required
    Ĝ::Union{Nothing, Matrix{N}}
    "Variance of the expected bi-adjacency matrix" # not always computed/required
    σ::Union{Nothing, Matrix{N}}
    "Status indicators: parameters computed, expected adjacency matrix computed, variance computed, etc."
    const status::Dict{Symbol, Any}
    "Function used to computed the log-likelihood of the (reduced) model"
    fun::Union{Nothing, Function}
    "Membership vector of ⊥ layer (boolean)"
    const is⊥::Vector{Bool}
    "mapping between node id and index in ⊥-node degree sequence"
    const ⊥map::Dict{Int, Int}
    "mapping between node id and index in ⊤-node degree sequence"
    const ⊤map::Dict{Int, Int}
end

Base.show(io::IO, m::BiCM{T,N}) where {T,N} = print(io, """BiCM{$(T), $(N)} ($(m.status[:N⊥]) + $(m.status[:N⊤]) vertices, $(m.status[:d⊥_unique]) + $(m.status[:d⊤_unique]) unique degrees, $(@sprintf("%.2f", m.status[:cᵣ])) compression ratio)""")

"""Return the reduced number of nodes in the UBCM network"""
Base.length(m::BiCM) = length(m.d⊥_ᵣ) + length(m.d⊤_ᵣ)


"""
    BiCM(G::T; precision::N=Float64, kwargs...) where {T<:Graphs.AbstractGraph, N<:Real}
    BiCM(;d⊥::Vector{T}, d⊤::Vector{T}, precision::Type{<:AbstractFloat}=Float64, kwargs...)

Constructor function for the `BiCM` type. The graph you provide should be bipartite
    
By default and dependng on the graph type `T`, the definition of degree from ``Graphs.jl`` is applied. 
If you want to use a different definition of degrees, you can pass vectors of degrees sequences as keyword arguments (`d⊥`, `d⊤`).
If you want to generate a model directly from degree sequences without an underlying graph , you can simply pass the degree sequences as arguments (`d⊥`, `d⊤`).
If you want to work from an adjacency matrix, or edge list, you can use the graph constructors from the ``JuliaGraphs`` ecosystem.

Zero degree nodes have a zero probability of being connected to other nodes, so they are skipped in the computation of the model.

# Examples     
```jldoctest
# generating a model from a graph
julia> G = corporateclub();

julia> model =  BiCM(G)
BiCM{Graphs.SimpleGraphs.SimpleGraph{Int64}, Float64} (25 + 15 vertices, 6 + 6 unique degrees, 0.30 compression ratio)

```
```jldoctest
# generating a model directly from a degree sequence
julia> model = model = BiCM(d⊥=[1,1,2,2,2,3,3,1,1,2], d⊤=[3,4,5,2,5,6,6,1,1,2])
BiCM{Nothing, Float64} (10 + 10 vertices, 3 + 6 unique degrees, 0.45 compression ratio)

```
```jldoctest
# generating a model directly from a degree sequence with a different precision
julia> model = model = BiCM(d⊥=[1,1,2,2,2,3,3,1,1,2], d⊤=[3,4,5,2,5,6,6,1,1,2], precision=Float32)
BiCM{Nothing, Float32} (10 + 10 vertices, 3 + 6 unique degrees, 0.45 compression ratio)

```
```jldoctest
# generating a model from an adjacency matrix
julia> A = [0 0 0 1 0;0 0 0 1 0;0 0 0 0 1;1 1 0 0 0;0 0 1 0 0];

julia> G = MaxEntropyGraphs.Graphs.SimpleGraph(A);

julia> @assert MaxEntropyGraphs.Graphs.is_bipartite(G); # check if the graph is bipartite

julia> model = BiCM(G) # generating the model
BiCM{Graphs.SimpleGraphs.SimpleGraph{Int64}, Float64} (3 + 2 vertices, 1 + 2 unique degrees, 0.60 compression ratio)

```
```jldoctest
# generating a model from a biadjacency matrix
julia> biadjacency = [1 0;1 0; 0 1];

julia> N⊥,N⊤ = size(biadjacency); # layer dimensions

julia> adjacency = [zeros(Int, N⊥,N⊥) biadjacency; biadjacency' zeros(Int,N⊤,N⊤)];

julia> G = MaxEntropyGraphs.Graphs.SimpleGraph(adjacency); # generate graph

julia> model = BiCM(G) # generate model
BiCM{Graphs.SimpleGraphs.SimpleGraph{Int64}, Float64} (3 + 2 vertices, 1 + 2 unique degrees, 0.60 compression ratio)

```
```jldoctest
# generating a model from an edge list
julia> edges = MaxEntropyGraphs.Graphs.SimpleEdge.([(1,4); (2,4); (3,5)]);

julia> G = MaxEntropyGraphs.Graphs.SimpleGraph(edges); # generate graph

julia> model = BiCM(G) # generate model
BiCM{Graphs.SimpleGraphs.SimpleGraph{Int64}, Float64} (3 + 2 vertices, 1 + 2 unique degrees, 0.60 compression ratio)

```
"""
function BiCM(G::T; d⊥::Union{Nothing, Vector}=nothing, 
                    d⊤::Union{Nothing, Vector}=nothing, 
                    precision::Type{N}=Float64, 
                    kwargs...) where {T,N<:AbstractFloat}
    T <: Union{Graphs.AbstractGraph, Nothing} ? nothing : throw(TypeError("G must be a subtype of AbstractGraph or Nothing"))
    

    # coherence checks
    if T <: Graphs.AbstractGraph # Graph specific checks
        # check if the graph is empty or has only one vertex
        Graphs.nv(G) == 0 ? throw(ArgumentError("The graph is empty")) : nothing
        Graphs.nv(G) == 1 ? throw(ArgumentError("The graph has only one vertex")) : nothing
        # check if the graph is bipartite
        Graphs.is_bipartite(G) ? nothing : throw(ArgumentError("The graph is not bipartite"))
        
        if Graphs.is_directed(G)
            @warn "The graph is directed, while the BiCM model is undirected, the directional information will be lost"
        end

        if T <: SimpleWeightedGraphs.AbstractSimpleWeightedGraph
            @warn "The graph is weighted, while the BiCM model is unweighted, the weight information will be lost"
        end

        # get layer membership
        membership = Graphs.bipartite_map(G)
        ⊥nodes, ⊤nodes = findall(membership .== 1), findall(membership .== 2)
        is⊥ = membership .== 1 # keep track of the membership of each node for later use (e.g. degree etc.)
        # degree sequences
        d⊥ = isnothing(d⊥) ? Graphs.degree(G, ⊥nodes) : d⊥
        d⊤ = isnothing(d⊤) ? Graphs.degree(G, ⊤nodes) : d⊤

        Graphs.nv(G) != length(d⊥) + length(d⊤) ? throw(DimensionMismatch("The number of vertices in the graph ($(Graphs.nv(G))) and the length of the degree sequences do not match")) : nothing
    end
    # coherence checks specific to the degree sequences
    !isnothing(d⊥) && length(d⊥) == 0 ? throw(ArgumentError("The degree sequences d⊥ is empty")) : nothing
    !isnothing(d⊤) && length(d⊤) == 0 ? throw(ArgumentError("The degree sequences d⊤ is empty")) : nothing
    !isnothing(d⊥) && length(d⊥) == 1 ? throw(ArgumentError("The degree sequences d⊥ only contains a single node")) : nothing
    !isnothing(d⊤) && length(d⊤) == 1 ? throw(ArgumentError("The degree sequences d⊤ only contains a single node")) : nothing    
    maximum(d⊥) >= length(d⊤) ? throw(DomainError("The maximum outdegree in the layer d⊥ is greater or equal to the number of vertices in layer d⊤, this is not allowed")) : nothing
    maximum(d⊤) >= length(d⊥) ? throw(DomainError("The maximum outdegree in the layer d⊤ is greater or equal to the number of vertices in layer d⊥, this is not allowed")) : nothing
    if isnothing(G)
        ⊥nodes = collect(1:length(d⊥))
        ⊤nodes = collect(length(d⊥)+1:length(d⊥)+length(d⊤))
        is⊥ = vcat(ones(Bool, length(d⊥)), zeros(Bool, length(d⊤)))
    end

    # field generation
    d⊥ᵣ, d⊥_ind, d⊥ᵣ_ind, f⊥ = np_unique_clone(d⊥, sorted=true)
    d⊥ᵣ_nz = iszero(first(d⊥ᵣ)) ? (2:length(d⊥ᵣ)) : (1:length(d⊥ᵣ)) # precomputed indices for the reduced degree sequence (works because sorted and unique values)
    d⊤ᵣ, d⊤_ind, d⊤ᵣ_ind, f⊤ = np_unique_clone(d⊤, sorted=true)
    d⊤ᵣ_nz = iszero(first(d⊤ᵣ)) ? (2:length(d⊤ᵣ)) : (1:length(d⊤ᵣ)) # precomputed indices for the reduced degree sequence
    ⊥map = Dict(node => i for (i,node) in enumerate(⊥nodes)) # for retrieval of the row/column in the biadjacency matrix matching node i of the graph
    ⊤map = Dict(node => i for (i,node) in enumerate(⊤nodes))

    # initiate parameters
    θᵣ = Vector{precision}(undef, length(d⊥ᵣ) + length(d⊤ᵣ))
    xᵣ = Vector{precision}(undef, length(d⊥ᵣ))
    yᵣ = Vector{precision}(undef, length(d⊤ᵣ)) 
    status = Dict{Symbol, Real}(:params_computed=>false,            # are the parameters computed?
                                :G_computed=>false,                 # is the expected adjacency matrix computed and stored?
                                :σ_computed=>false,                 # is the standard deviation computed and stored?
                                :cᵣ => (length(d⊥ᵣ) + length(d⊤ᵣ))/(length(d⊥)+length(d⊤)),    # compression ratio of the reduced model TODO: consider not counting the zero values?
                                :N⊥ => length(d⊥),                  # number of nodes in layer ⊥
                                :N⊤ => length(d⊤),                  # number of nodes in layer ⊤
                                :d⊥_unique => length(d⊥ᵣ),         # number of unique degrees in layer ⊥
                                :d⊤_unique => length(d⊤ᵣ),         # number of unique degrees in layer ⊤
                                :N => length(d⊥) + length(d⊤)       # number of vertices in the original graph 
                )   
    
    return BiCM{T,precision}(G, θᵣ, xᵣ, yᵣ, d⊥, d⊤, d⊥ᵣ, d⊤ᵣ, d⊥ᵣ_nz, d⊤ᵣ_nz, f⊥, f⊤, d⊥_ind, d⊤_ind, d⊥ᵣ_ind, d⊤ᵣ_ind, ⊥nodes, ⊤nodes, nothing, nothing, status, nothing,is⊥,⊥map,⊤map)
end

BiCM(; d⊥::Vector{T}, d⊤::Vector{T}, precision::Type{N}=Float64, kwargs...) where {T<:Signed, N<:AbstractFloat} = BiCM(nothing; d⊥=d⊥, d⊤=d⊤, precision=precision, kwargs...)


"""
    L_BiCM_reduced(θ::Vector, k⊥::Vector, k⊤::Vector, f⊥::Vector, f⊤::Vector, nz⊥::UnitRange, nz⊤::UnitRange, n⊥ᵣ::Int)


Compute the log-likelihood of the reduced BiCM model using the exponential formulation in order to maintain convexity.

# Arguments
- `θ`: the maximum likelihood parameters of the model ([α; β])
- `k⊥`: the reduced degree sequence of the ⊥ layer
- `k⊤`: the reduced degree sequence of the ⊤ layer
- `f⊥`: the frequency of each degree in the ⊥ layer
- `f⊤`: the frequency of each degree in the ⊤ layer
- `nz⊥`: the indices of non-zero elements in the reduced ⊥ layer degree sequence
- `nz⊤`: the indices of non-zero elements in the reduced ⊤ layer degree sequence
- `n⊥ᵣ`: the number unique values in the reduced ⊥ layer degree sequence

The function returns the log-likelihood of the reduced model. For the optimisation, this function will be used to
generate an anonymous function associated with a specific model.

# Examples
```jldoctest
# Generic use:
julia> k⊥ = [1, 2, 3, 4];

julia> k⊤  = [1, 2, 4];

julia> f⊥  = [1; 3; 1; 1];

julia> f⊤  = [4; 2; 1];

julia> nz⊥ = 1:length(k⊥);

julia> nz⊤ = 1:length(k⊤);

julia> n⊥ᵣ = length(k⊥);

julia> θ   = collect(range(0.1, step=0.1, length=length(k⊥) + length(k⊤)));

julia> L_BiCM_reduced(θ, k⊥, k⊤, f⊥, f⊤, nz⊥, nz⊤, n⊥ᵣ)
-26.7741690720244
```
```jldoctest
# Use with DBCM model:
julia> G = corporateclub();

julia> model = BiCM(G);

julia> model_fun = θ -> L_BiCM_reduced(θ, model.d⊥ᵣ, model.d⊤ᵣ, model.f⊥, model.f⊤, model.d⊥ᵣ_nz, model.d⊤ᵣ_nz, model.status[:d⊥_unique]);

julia> model_fun(ones(size(model.θᵣ)))
-237.5980041411147
```
"""
function L_BiCM_reduced(θ::AbstractVector, k⊥::Vector, k⊤::Vector, f⊥::Vector, f⊤::Vector, nz⊥::UnitRange, nz⊤::UnitRange, n⊥ᵣ::Int)
    # set pre-allocated values
    α = @view θ[1:n⊥ᵣ]
    β = @view θ[n⊥ᵣ+1:end]
    res = zero(eltype(θ))
    # actual compute
    for i in nz⊥
        res -=  f⊥[i]*k⊥[i]*α[i] 
        for j in nz⊤
            res -= f⊥[i] * f⊤[j] * softplus(-α[i] - β[j])
        end
    end
    for j in nz⊤
        res -= f⊤[j]*k⊤[j]*β[j]
    end
    return res
end


"""
    L_BiCM_reduced(m::BiCM)

Return the log-likelihood of the BiCM model `m` based on the computed maximum likelihood parameters.

# Examples
```jldoctest
# Use with DBCM model:
julia> G = corporateclub();

julia> model = BiCM(G);

julia> solve_model!(model);

julia> L_BiCM_reduced(model);

```

See also [`L_BiCM_reduced(::Vector, ::Vector, ::Vector, ::Vector, ::Vector, ::UnitRange, ::UnitRange, ::Int)`](@ref)
"""
L_BiCM_reduced(m::BiCM) = L_BiCM_reduced(m.θᵣ, m.d⊥ᵣ, m.d⊤ᵣ, m.f⊥, m.f⊤, m.d⊥ᵣ_nz, m.d⊤ᵣ_nz, m.status[:d⊥_unique])


"""
    ∇L_BiCM_reduced!(∇L::AbstractVector, θ::AbstractVector, k⊥::Vector, k⊤::Vector, f⊥::Vector, f⊤::Vector,  nz⊥::UnitRange{T}, nz⊤::UnitRange{T}, x::AbstractVector, y::AbstractVector, n⊥::Int) where {T<:Signed}

Compute the gradient of the log-likelihood of the reduced DBCM model using the exponential formulation in order to maintain convexity.

For the optimisation, this function will be used togenerate an anonymous function associated with a specific model. The function 
will update pre-allocated vectors (`∇L`,`x` and `y`) for speed. The gradient is non-allocating.

# Arguments
- `∇L`: the gradient of the log-likelihood of the reduced model
- `θ`: the maximum likelihood parameters of the model ([α; β])
- `k⊥`: the reduced degree sequence of the ⊥ layer
- `k⊤`: the reduced degree sequence of the ⊤ layer
- `f⊥`: the frequency of each degree in the ⊥ layer
- `f⊤`: the frequency of each degree in the ⊤ layer
- `nz⊥`: the indices of non-zero elements in the reduced ⊥ layer degree sequence
- `nz⊤`: the indices of non-zero elements in the reduced ⊤ layer degree sequence
- `x`: the exponentiated maximum likelihood parameters of the model ( xᵢ = exp(-αᵢ) )
- `y`: the exponentiated maximum likelihood parameters of the model ( yᵢ = exp(-βᵢ) )
- `n⊥`: the number unique values in the reduced ⊥ layer degree sequence

# Examples
```jldoctest ∇L_BiCM_reduced
# Explicit use with BiCM model:
julia> G = corporateclub();

julia> model = BiCM(G);

julia> ∇L = zeros(Real, length(model.θᵣ));

julia> x  = zeros(Real, length(model.xᵣ));

julia> y  = zeros(Real, length(model.yᵣ));

julia> ∇model_fun! = θ -> ∇L_BiCM_reduced!(∇L, θ, model.d⊥ᵣ, model.d⊤ᵣ, model.f⊥, model.f⊤, model.d⊥ᵣ_nz, model.d⊤ᵣ_nz, x, y, model.status[:d⊥_unique]);

julia> ∇model_fun!(model.θᵣ);

```
"""
function ∇L_BiCM_reduced!(  ∇L::AbstractVector, θ::AbstractVector,
                            k⊥::Vector, k⊤::Vector,
                            f⊥::Vector, f⊤::Vector, 
                            nz⊥::UnitRange{T}, nz⊤::UnitRange{T}, 
                            x::AbstractVector, y::AbstractVector,
                            n⊥::Int) where {T<:Signed}
    # set pre-allocated values # keep these in model as views for more efficiency?
    α = @view θ[1:n⊥]
    β = @view θ[n⊥+1:end]
    
    for i in nz⊥ # non-allocating version of x .= exp.(-α)
        x[i] = exp(-α[i])
    end
    for j in nz⊤
        y[j] = exp(-β[j])
    end

    # reset gradient to zero
    ∇L .= zero(eltype(θ))
    
    # actual compute (bipartite: no diagonal term; factor the outer-constant f·x / f·y out of
    # the inner reduction so each inner step is a single multiply-add)
    for i in nz⊥
        @inbounds xᵢ = x[i]
        s = zero(eltype(∇L))
        @inbounds @simd for j in nz⊤
            s += f⊤[j] * y[j] / (1 + xᵢ * y[j])
        end
        @inbounds ∇L[i] = -f⊥[i] * k⊥[i] + f⊥[i] * xᵢ * s
    end
    for j in nz⊤
        @inbounds yⱼ = y[j]
        s = zero(eltype(∇L))
        @inbounds @simd for i in nz⊥
            s += f⊥[i] * x[i] / (1 + x[i] * yⱼ)
        end
        @inbounds ∇L[n⊥+j] = -f⊤[j] * k⊤[j] + f⊤[j] * yⱼ * s
    end

    return ∇L
end


"""
    ∇L_BiCM_reduced_minus!(args...)

Compute minus the gradient of the log-likelihood of the reduced BiCM model using the exponential formulation in order to maintain convexity. Used for optimisation in a non-allocating manner.

See also [`∇L_BiCM_reduced!`](@ref)
"""
function ∇L_BiCM_reduced_minus!(∇L::AbstractVector, θ::AbstractVector,
                                k⊥::Vector, k⊤::Vector,
                                f⊥::Vector, f⊤::Vector, 
                                nz⊥::UnitRange{T}, nz⊤::UnitRange{T}, 
                                x::AbstractVector, y::AbstractVector,
                                n⊥::Int) where {T<:Signed}
    # set pre-allocated values
    α = @view θ[1:n⊥]
    β = @view θ[n⊥+1:end]
    
    for i in nz⊥ # non-allocating version of x .= exp.(-α)
        x[i] = exp(-α[i])
    end
    for j in nz⊤
        y[j] = exp(-β[j])
    end

    # reset gradient to zero
    ∇L .= zero(eltype(θ))

    # actual compute (bipartite: no diagonal term; factor the outer-constant f·x / f·y out of
    # the inner reduction so each inner step is a single multiply-add)
    for i in nz⊥
        @inbounds xᵢ = x[i]
        s = zero(eltype(∇L))
        @inbounds @simd for j in nz⊤
            s += f⊤[j] * y[j] / (1 + xᵢ * y[j])
        end
        @inbounds ∇L[i] = f⊥[i] * k⊥[i] - f⊥[i] * xᵢ * s
    end
    for j in nz⊤
        @inbounds yⱼ = y[j]
        s = zero(eltype(∇L))
        @inbounds @simd for i in nz⊥
            s += f⊥[i] * x[i] / (1 + x[i] * yⱼ)
        end
        @inbounds ∇L[n⊥+j] = f⊤[j] * k⊤[j] - f⊤[j] * yⱼ * s
    end

    return ∇L
end

"""
    BiCM_reduced_iter!(θ::AbstractVector, k⊥::Vector, k⊤::Vector, f⊥::Vector, f⊤::Vector, nz⊥::UnitRange{T}, nz⊤::UnitRange{T}, x::AbstractVector, y::AbstractVector, G::AbstractVector, n⊥::Int) where {T<:Signed}

Compute the next fixed-point iteration for the BiCM model using the exponential formulation in order to maintain convexity.
The function is non-allocating and will update pre-allocated vectors (`θ`, `x`, `y` and `G`) for speed.

# Arguments
- `θ`: the maximum likelihood parameters of the model ([α; β])
- `k⊥`: the reduced degree sequence of the ⊥ layer
- `k⊤`: the reduced degree sequence of the ⊤ layer
- `f⊥`: the frequency of each degree in the ⊥ layer
- `f⊤`: the frequency of each degree in the ⊤ layer
- `nz⊥`: the indices of non-zero elements in the reduced ⊥ layer degree sequence
- `nz⊤`: the indices of non-zero elements in the reduced ⊤ layer degree sequence
- `x`: the exponentiated maximum likelihood parameters of the model ( xᵢ = exp(-αᵢ) )
- `y`: the exponentiated maximum likelihood parameters of the model ( yᵢ = exp(-βᵢ) )
- `G`: buffer for computations
- `n⊥`: the number unique values in the reduced ⊥ layer degree sequence


# Examples
```jldoctest
# Use with BiCM model:
julia> G = corporateclub();

julia> model = BiCM(G);

julia> G = zeros(eltype(model.θᵣ), length(model.θᵣ));

julia> x = zeros(eltype(model.θᵣ), length(model.xᵣ));

julia> y = zeros(eltype(model.θᵣ), length(model.yᵣ));


julia> BiCM_FP! = θ -> BiCM_reduced_iter!(θ, model.d⊥ᵣ, model.d⊤ᵣ, model.f⊥, model.f⊤, model.d⊥ᵣ_nz, model.d⊤ᵣ_nz, x, y, G, model.status[:d⊥_unique]);

julia> BiCM_FP!(model.θᵣ);

```
"""
function BiCM_reduced_iter!(θ::AbstractVector, k⊥::Vector, k⊤::Vector,
                            f⊥::Vector, f⊤::Vector, 
                            nz⊥::UnitRange{T}, nz⊤::UnitRange{T}, 
                            x::AbstractVector, y::AbstractVector, G::AbstractVector,
                            n⊥::Int) where {T<:Signed}
    # set pre-allocated values
    α = @view θ[1:n⊥]
    β = @view θ[n⊥+1:end]
    # non-allocating version of x .= exp.(-α)
    for i in nz⊥
        x[i] = exp(-α[i])
    end
    for j in nz⊤
        y[j] = exp(-β[j])
    end
    # actual compute
    for i in nz⊥
        G[i] = zero(eltype(θ))
        for j in nz⊤
            G[i] += f⊤[j] * y[j]/(1 + x[i]*y[j])
        end
        G[i] = - log( k⊥[i] / G[i] )
    end
    for j in nz⊤
        G[n⊥+j] = zero(eltype(θ))
        for i in nz⊥
            G[n⊥+j] += f⊥[i] * x[i]/(1 + x[i]*y[j])
        end
        G[n⊥+j] = - log( k⊤[j] / G[n⊥+j] )
    end

    return G
end


"""
    precision(m::BiCM)

Determine the compute precision of the BiCM model `m`.

# Examples
```jldoctest
julia> model = BiCM(corporateclub());

julia> MaxEntropyGraphs.precision(model)
Float64
```

```jldoctest
julia> model = BiCM(corporateclub(), precision=Float32);

julia> MaxEntropyGraphs.precision(model)
Float32
```
"""
precision(m::BiCM) = typeof(m).parameters[2]


"""
    initial_guess(m::BiCM; method::Symbol=:degrees)

Compute an initial guess for the maximum likelihood parameters of the BiCM model `m` using the method `method`.

The methods available are: 
- `:degrees` (default): the initial guess is computed using the degrees of the graph, i.e. ``\\theta = [-\\log(d_{\bot}); -\\log(d_{\top})]`` 
- `:random`: the initial guess is computed using random values between 0 and 1, i.e. ``\\theta_{i} = -\\log(r_{i})`` where ``r_{i} \\sim U(0,1)``
- `:uniform`: the initial guess is uniformily set to 0.5, i.e. ``\\theta_{i} = -\\log(0.5)``
- `:chung_lu`: the initial guess is computed using the degrees of the graph and the number of edges, i.e. ``\\theta = [-\\log(d_{\bot}/(2E)); -\\log(d_{\top}/(2E))]``

# Examples
```jldoctest
julia> model = BiCM(corporateclub());

julia> initial_guess(model, method=:random);

julia> initial_guess(model, method=:uniform);

julia> initial_guess(model, method=:chung_lu);

julia> initial_guess(model);

```
"""
function initial_guess(m::BiCM; method::Symbol=:degrees)
    if isequal(method, :degrees)
        return Vector{precision(m)}(vcat(-log.(m.d⊥ᵣ), -log.(m.d⊤ᵣ)))
    elseif isequal(method, :random)
        return Vector{precision(m)}(-log.(rand(length(m.θᵣ))))
    elseif isequal(method, :uniform)
        return Vector{precision(m)}(-log.(0.5 .* ones(length(m.θᵣ))))
    elseif isequal(method, :chung_lu)
        isnothing(m.G) ? throw(ArgumentError("Cannot compute the number of edges because the model has no underlying graph (m.G == nothing)")) : nothing
        return Vector{precision(m)}(vcat(-log.(m.d⊥ᵣ ./ sqrt(Graphs.ne(m.G)) ), -log.(m.d⊤ᵣ ./ sqrt(Graphs.ne(m.G)) ) ))
    else
        throw(ArgumentError("The initial guess method $(method) is not supported"))
    end
end


"""
    set_xᵣ!(m::BiCM)

Set the value of xᵣ to exp(-αᵣ) for the BiCM model `m`
"""
function set_xᵣ!(m::BiCM)
    if m.status[:params_computed]
        αᵣ = @view m.θᵣ[1:m.status[:d⊥_unique]]
        m.xᵣ .= exp.(-αᵣ)
    else
        throw(ArgumentError("The parameters have not been computed yet"))
    end
end


"""
    set_yᵣ!(m::BiCM)

Set the value of yᵣ to exp(-βᵣ) for the BiCM model `m`
"""
function set_yᵣ!(m::BiCM)
    if m.status[:params_computed]
        βᵣ = @view m.θᵣ[m.status[:d⊥_unique]+1:end]
        m.yᵣ .= exp.(-βᵣ)
    else
        throw(ArgumentError("The parameters have not been computed yet"))
    end
end


"""
    Ĝ(m::BiCM)

Compute the expected **biadjacency matrix** for the BiCM model `m`

!!! note
    Please note that this generates a bi-adjacency matrix, not an adjacency matrix.
"""
function Ĝ(m::BiCM)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    
    # get layer sizes => this is the full size of the network
    n⊥, n⊤ = m.status[:N⊥], m.status[:N⊤] 

    # initiate Ĝ
    G = zeros(precision(m), n⊥, n⊤)

    # initiate x and y
    x = m.xᵣ[m.d⊥ᵣ_ind]
    y = m.yᵣ[m.d⊤ᵣ_ind]

    # compute Ĝ
    for i in 1:n⊥
        for j in 1:n⊤
            G[i,j] = x[i]*y[j]/(1 + x[i]*y[j])
        end
    end

    return G
end


"""
    set_Ĝ!(m::BiCM)

Set the expected **biadjacency matrix** for the BiCM model `m`
"""
function set_Ĝ!(m::BiCM)
    m.Ĝ = Ĝ(m)
    m.status[:G_computed] = true
    return m.Ĝ
end

"""
    σˣ(m::BiCM)

Compute the standard deviation for the elements of the **biadjacency matrix** for the BiCM model `m`,
i.e. `sqrt(pᵢα(1 - pᵢα))` (the biadjacency entries are independent Bernoulli distributed variables).

**Note:** read as "sigma star". The result is an `n⊥ × n⊤` matrix, like the biadjacency matrix itself.
"""
function σˣ(m::BiCM)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))

    # get layer sizes => this is the full size of the network
    n⊥, n⊤ = m.status[:N⊥], m.status[:N⊤]

    # initiate σ
    σ = zeros(precision(m), n⊥, n⊤)

    # initiate x and y
    x = m.xᵣ[m.d⊥ᵣ_ind]
    y = m.yᵣ[m.d⊤ᵣ_ind]

    # compute σ
    for i in 1:n⊥
        for j in 1:n⊤
            piα = x[i]*y[j]/(1 + x[i]*y[j])
            σ[i,j] = sqrt(piα * (1 - piα))
        end
    end

    return σ
end

"""
    set_σ!(m::BiCM)

Set the standard deviation for the elements of the biadjacency matrix for the BiCM model `m`.
"""
function set_σ!(m::BiCM)
    m.σ = σˣ(m)
    m.status[:σ_computed] = true
    return m.σ
end


"""
    σₓ(m::BiCM, X::Function; gradient_method::Symbol=:ReverseDiff)

Compute the standard deviation of metric `X` for the BiCM model `m` via error propagation (the delta
method of Squartini & Garlaschelli (2011); see also Saracco et al. (2015)).

`X` must be a function of the **biadjacency matrix** (`n⊥ × n⊤`, cf. [`Ĝ`](@ref MaxEntropyGraphs.Ĝ)),
not of an adjacency matrix. The biadjacency entries are independent Bernoulli variables, so no
covariance terms occur:

``σ^{2}[X] = \\sum_{i,α} \\left( σ[m_{iα}] \\frac{∂X}{∂m_{iα}} \\right)^{2}``

This requires that both the expected values (`m.Ĝ`) and standard deviations (`m.σ`) are computed for `m`.

For the statistical significance of projection-related quantities, the exact Poisson-binomial machinery
(cf. [`project`](@ref MaxEntropyGraphs.project) and [`V_PB_parameters`](@ref MaxEntropyGraphs.V_PB_parameters))
is preferable to the delta method.
"""
function σₓ(m::BiCM, X::Function; gradient_method::Symbol=:ReverseDiff)
    # checks
    m.status[:G_computed] ? nothing : throw(ArgumentError("The expected values (m.Ĝ) must be computed for `m` before computing the standard deviation of metric `X`, see `set_Ĝ!`"))
    m.status[:σ_computed] ? nothing : throw(ArgumentError("The standard deviations (m.σ) must be computed for `m` before computing the standard deviation of metric `X`, see `set_σ!`"))

    # gradient
    if gradient_method == :ForwardDiff
        ∇X = ForwardDiff.gradient(X, m.Ĝ)
    elseif gradient_method == :ReverseDiff
        ∇X = ReverseDiff.gradient(X, m.Ĝ)
    elseif gradient_method == :Zygote
        ∇X = Zygote.gradient(X, m.Ĝ)[1]
    else
        throw(ArgumentError("Invalid gradient method, only :ForwardDiff, :ReverseDiff and :Zygote are accepted"))
    end

    # independent entries: no covariance terms
    return sqrt( sum((m.σ .* ∇X) .^ 2) )
end


"""
    rand(m::BiCM; precomputed::Bool=false)

Generate a random graph from the BiCM model `m`.

# Arguments:
- `precomputed::Bool`: if `true`, the precomputed expected **biadjacency matrix** (`m.Ĝ`) is used to generate the random graph, otherwise the maximum likelihood parameters are used to generate the random graph on the fly. For larger networks, it is 
  recommended to not precompute the expected adjacency matrix to limit memory pressure.

**Note**: The generated graph will also be bipartite and respect the layer membership of the original graph used to define the model.
"""
function rand(m::BiCM; precomputed::Bool=false, rng::AbstractRNG=default_rng())
    if precomputed
        # check if possible to use precomputed Ĝ
        m.status[:G_computed] ? nothing : throw(ArgumentError("The expected adjacency matrix has not been computed yet"))
        # generate random graph
        G = Graphs.SimpleDiGraphFromIterator( Graphs.Edge.([(or⊥,or⊤) for (i,or⊥) in enumerate(m.⊥nodes) for (j,or⊤) in enumerate(m.⊤nodes) if rand(rng)<m.Ĝ[i,j]]))
    else
        # check if possible to use parameters
        m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
        # initiate x and y
        x = m.xᵣ[m.d⊥ᵣ_ind]
        y = m.yᵣ[m.d⊤ᵣ_ind]
        G = Graphs.SimpleGraphFromIterator( Graphs.Edge.([(or⊥,or⊤) for (i,or⊥) in enumerate(m.⊥nodes) for (j,or⊤) in enumerate(m.⊤nodes) if rand(rng)< x[i]*y[j]/(1 + x[i]*y[j]) ]))
    end

    # deal with edge case where no edges are generated for the last node(s) in the graph
    while Graphs.nv(G) < m.status[:N]
        Graphs.add_vertex!(G)
    end

    return G
end


"""
    rand(m::BiCM, n::Int; precomputed::Bool=false)

    Generate `n` random graphs from the BiCM model `m`. If multithreading is available, the graphs are generated in parallel.

# Arguments:
- `precomputed::Bool`: if `true`, the precomputed expected **biadjacency matrix** (`m.Ĝ`) is used to generate the random graph, otherwise the maximum likelihood parameters are used to generate the random graph on the fly. For larger networks, it is 
  recommended to not precompute the expected adjacency matrix to limit memory pressure.

**Note**: The generated graph will also be bipartite and respect the layer membership of the original graph used to define the model.
"""
function rand(m::BiCM, n::Int; precomputed::Bool=false, rng::AbstractRNG=default_rng())
    # pre-allocate
    res = Vector{Graphs.SimpleGraph{Int}}(undef, n)
    # per-sample seeds for reproducible, thread-schedule-independent sampling
    seeds = rand(rng, UInt64, n)
    # fill vector using threads
    Threads.@threads for i in 1:n
        res[i] = rand(m; precomputed=precomputed, rng=Xoshiro(seeds[i]))
    end

    return res
end

"""
    solve_model!(m::BiCM)

Compute the likelihood maximising parameters of the BiCM model `m`. 

# Arguments
- `method::Symbol`: solution method to use, can be `:fixedpoint` (default), or :$(join(keys(MaxEntropyGraphs.optimization_methods), ", :", " and :")).
- `initial::Symbol`: initial guess for the parameters ``\\Theta``, can be :degrees (default), :random, :uniform, or :chung_lu.
- `maxiters::Int`: maximum number of iterations for the solver (defaults to 1000). 
- `verbose::Bool`: set to show log messages (defaults to false).
- `ftol::Real`: function tolerance for convergence with the fixedpoint method (defaults to 1e-8).
- `abstol::Union{Number, Nothing}`: absolute function tolerance for convergence with the other methods (defaults to `nothing`).
- `reltol::Union{Number, Nothing}`: relative function tolerance for convergence with the other methods (defaults to `nothing`).
- `g_tol::Union{Number, Nothing}`: gradient tolerance for the gradient-based methods (maps to Optim's `g_abstol`); set e.g. `1e-5` to stop before over-converging (defaults to `nothing`, i.e. Optim's tight default).
- `AD_method::Symbol`: autodiff method to use, can be any of :$(join(keys(MaxEntropyGraphs.AD_methods), ", :", " and :")). Performance depends on the size of the problem (defaults to `:AutoZygote`),
- `analytical_gradient::Bool`: set the use the analytical gradient instead of the one generated with autodiff (defaults to `false`)

# Examples
```jldoctest BiCM_solve
# default use
julia> model = BiCM(corporateclub());

julia> solve_model!(model);

```
```jldoctest BiCM_solve
# using analytical gradient and uniform initial guess
julia> solve_model!(model, method=:BFGS, analytical_gradient=true, initial=:uniform)
(BiCM{Graphs.SimpleGraphs.SimpleGraph{Int64}, Float64} (25 + 15 vertices, 6 + 6 unique degrees, 0.30 compression ratio), retcode: Success
u: [1.449571644621672, 0.8231752829683303, 0.34755085972479766, -0.04834480708852856, -0.3984299800917503, -0.7223268299919358, 1.6090554004279671, 1.2614196476197532, 0.9762560461922147, 0.11406188481061938, -0.24499004480426345, -2.2646067641037333]
Final objective value:     171.15095803718134
)

```
"""
function solve_model!(m::BiCM;  # common settings
                                method::Symbol=:fixedpoint, 
                                initial::Symbol=:degrees,
                                maxiters::Int=1000, 
                                verbose::Bool=false,
                                # NLsolve.jl specific settings (fixed point method)
                                ftol::Real=1e-8,
                                # optimisation.jl specific settings (optimisation methods)
                                abstol::Union{Number, Nothing}=nothing,
                                reltol::Union{Number, Nothing}=nothing,
                                g_tol::Union{Number, Nothing}=nothing,
                                AD_method::Symbol=:AutoZygote,
                                analytical_gradient::Bool=true)
    N = precision(m)
    N <: Union{Float16, Float32} && @warn "Solving in $(N) precision is experimental and may not converge; low precision is intended for storage. Consider Float64 for the solve." maxlog=1
    # initial guess
    θ₀ = initial_guess(m, method=initial)
    # find Inf values
    ind_inf = findall(isinf, θ₀)
    if method == :fixedpoint
        # initiate buffers
        x_buffer = zeros(N, length(m.d⊥ᵣ));  # buffer for x = exp(-α)
        y_buffer = zeros(N, length(m.d⊤ᵣ));  # buffer for y = exp(-β)
        G_buffer = zeros(N, length(m.θᵣ));   # buffer for G(x)
        # define fixed point function
        FP_model! = (θ::Vector) -> BiCM_reduced_iter!(θ, m.d⊥ᵣ, m.d⊤ᵣ, m.f⊥, m.f⊤, m.d⊥ᵣ_nz, m.d⊤ᵣ_nz, x_buffer, y_buffer, G_buffer, m.status[:d⊥_unique]);
        # obtain solution
        θ₀[ind_inf] .= zero(N);
        sol = NLsolve.fixedpoint(FP_model!, θ₀, method=:anderson, ftol=ftol, iterations=maxiters);
        if NLsolve.converged(sol)
            if verbose 
            @info "Fixed point iteration converged after $(sol.iterations) iterations"
            end
            m.θᵣ .= sol.zero; 
            m.θᵣ[ind_inf] .= Inf;
            m.status[:params_computed] = true;
            set_xᵣ!(m);
            set_yᵣ!(m);
        else
            throw(ConvergenceError(method, nothing))
        end
    else
        if analytical_gradient
            # initiate buffers
            x_buffer = zeros(N, length(m.d⊥ᵣ)); # buffer for x = exp(-α)
            y_buffer = zeros(N, length(m.d⊤ᵣ)); # buffer for y = exp(-β)
            
            # define gradient function for optimisation.jl
            #θ ->               ∇L_BiCM_reduced_minus!(∇L, θ, m.d⊥ᵣ, m.d⊤ᵣ, m.f⊥, m.f⊤, m.d⊥ᵣ_nz, m.d⊤ᵣ_nz, x_buffer, y_buffer, m.status[:d⊥_unique]);
            grad! = (G, θ, p) ->∇L_BiCM_reduced_minus!(G,  θ, m.d⊥ᵣ, m.d⊤ᵣ, m.f⊥, m.f⊤, m.d⊥ᵣ_nz, m.d⊤ᵣ_nz, x_buffer, y_buffer, m.status[:d⊥_unique]);
        end
        # define objective function and its AD method.
        # Bind the model fields (and especially the `status` Dict lookup) to locals first, so the
        # differentiated objective closure captures plain values rather than the model/`status` Dict:
        # on the current AD stack, Zygote's `dict_getindex` pullback errors when a Dict access sits
        # inside the AD path (this is why only the BiCM AD-gradient solve was affected).
        d⊥ᵣ, d⊤ᵣ, f⊥, f⊤, d⊥ᵣ_nz, d⊤ᵣ_nz = m.d⊥ᵣ, m.d⊤ᵣ, m.f⊥, m.f⊤, m.d⊥ᵣ_nz, m.d⊤ᵣ_nz
        d⊥_unique = m.status[:d⊥_unique]
        f = AD_method ∈ keys(AD_methods)            ? Optimization.OptimizationFunction( (θ, p) ->   -L_BiCM_reduced(θ, d⊥ᵣ, d⊤ᵣ, f⊥, f⊤, d⊥ᵣ_nz, d⊤ᵣ_nz, d⊥_unique),
                                                                                            AD_methods[AD_method],
                                                                                            grad = analytical_gradient ? grad! : nothing)                      : throw(ArgumentError("The AD method $(AD_method) is not supported (yet)"))
        prob = Optimization.OptimizationProblem(f, θ₀);
        # obtain solution
        method ∈ keys(optimization_methods) || throw(ArgumentError("The method $(method) is not supported (yet)"))
        # `maxiters` was previously not forwarded (silently ignored for the optimisation methods);
        # `g_tol` (when set) maps to Optim's gradient tolerance so the solve can stop before
        # over-converging (the default is Optim's tight g_abstol ≈ 1e-8).
        solve_kwargs = isnothing(g_tol) ? (; maxiters = maxiters, abstol = abstol, reltol = reltol) :
                                          (; maxiters = maxiters, abstol = abstol, reltol = reltol, g_abstol = g_tol)
        sol = Optimization.solve(prob, optimization_methods[method]; solve_kwargs...)
        # check convergence
        if Optimization.SciMLBase.successful_retcode(sol.retcode)
            if verbose 
                @info """$(method) optimisation converged after $(@sprintf("%1.2e", sol.stats.time)) seconds (Optimization.jl return code: $("$(sol.retcode)"))"""
            end
            m.θᵣ .= sol.u;
            m.status[:params_computed] = true;
            set_xᵣ!(m);
            set_yᵣ!(m);
        else
            throw(ConvergenceError(method, sol.retcode))
        end
    end

    return m, sol
    
end

"""
    f_BiCM(x::T)

Helper function for the BiCM model to compute the expected value of the biadjacency matrix. The function computes the expression `x / (1 + x)`.
As an argument you need to pass the product of the maximum likelihood parameters `xᵣ[i] * yᵣ[j]` from a BiCM model.
"""
f_BiCM(xiyj::T) where {T} = xiyj/(1 + xiyj)

"""
    A(m::BiCM,i::Int,j::Int)

Return the expected value of the **biadjacency matrix** for the BiCM model `m` at the node pair `(i,j)`.

❗ For perfomance reasons, the function does not check:
- if the node pair is valid.
- if the parameters of the model have been computed.
"""
function A(m::BiCM,i::Int,j::Int)
    return @inbounds f_BiCM(m.xᵣ[m.d⊥ᵣ_ind[i]] * m.yᵣ[m.d⊤ᵣ_ind[j]])
end


"""
    degree(m::BiCM, i::Int; method=:reduced)

Return the expected degree for node `i` of the BiCM model `m`.
Uses the reduced model parameters `xᵣ` for perfomance reasons.

# Arguments
- `m::BiCM`: the BiCM model
- `i::Int`: the node for which to compute the degree. This can be any of the nodes in the original graph used to define the model.
- `method::Symbol`: the method to use for computing the degree. Can be any of the following:
    - `:reduced` (default) uses the reduced model parameters `xᵣ`, `yᵣ`, `f⊥` and `f⊤` for perfomance reasons.
    - `:full` uses all elements of the expected adjacency matrix.
    - `:adjacency` uses the precomputed adjacency matrix `m.Ĝ` of the model.

# Examples
```jldoctest
julia> model = BiCM(corporateclub());

julia> solve_model!(model);

julia> set_Ĝ!(model);

julia> typeof([degree(model, 1), degree(model, 1, method=:full), degree(model, 1, method=:adjacency)])
Vector{Float64} (alias for Array{Float64, 1})

``` 
"""
function degree(m::BiCM, i::Int; method::Symbol=:reduced)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    i > m.status[:N] ? throw(ArgumentError("Attempted to access node $i in a $(m.status[:N]) node graph")) : nothing

    if method == :reduced
        res = zero(precision(m))
        if m.is⊥[i]
            i_red = m.d⊥ᵣ_ind[m.⊥map[i]]
            for j in eachindex(m.yᵣ)
                res += f_BiCM(m.xᵣ[i_red] * m.yᵣ[j]) * m.f⊤[j]
            end
        else
            i_red = m.d⊤ᵣ_ind[m.⊤map[i]]
            for j in eachindex(m.xᵣ)
                res += f_BiCM(m.xᵣ[j] * m.yᵣ[i_red]) * m.f⊥[j]
            end
        end
    elseif method == :full
        # using all elements of the adjacency matrix
        res = zero(precision(m))
        if m.is⊥[i]
            for j in eachindex(m.d⊤)
                res += A(m, m.⊥map[i], j)
            end
        else
            for j in eachindex(m.d⊥)
                res += A(m, j, m.⊤map[i]) 
            end
        end
    elseif method == :adjacency
        #  using the precomputed biadjacency matrix 
        m.status[:G_computed] ? nothing : throw(ArgumentError("The adjacency matrix has not been computed yet"))
        # check layer membership
        res = m.is⊥[i] ? sum(@view m.Ĝ[m.⊥map[i],:]) : sum(@view m.Ĝ[:, m.⊤map[i]])
    else
        throw(ArgumentError("Unknown method $method"))
    end

    return res
end


"""
    degree(m::BiCM[, v]; method=:reduced)

Return a vector corresponding to the expected degree of the BiCM model `m` each node. If v is specified, only return degrees for nodes in v.

# Arguments
- `m::BiCM`: the BiCM model
- `v::Vector{Int}`: the nodes for which to compute the degree. Default is all nodes.
- `method::Symbol`: the method to use for computing the degree. Can be any of the following:
    - `:reduced` (default) uses the reduced model parameters `xᵣ` for perfomance reasons.
    - `:full` uses all elements of the expected adjacency matrix.
    - `:adjacency` uses the precomputed adjacency matrix `m.Ĝ` of the model.

# Examples
```jldoctest
julia> model = BiCM(corporateclub());

julia> solve_model!(model);

julia> set_Ĝ!(model);

julia> typeof(degree(model, method=:adjacency)) 
Vector{Float64} (alias for Array{Float64, 1})

``` 
"""
degree(m::BiCM, v::Vector{Int}=collect(1:m.status[:N]); method::Symbol=:reduced) = [degree(m, i, method=method) for i in v]



"""
    AIC(m::BiCM)

Compute the Akaike Information Criterion (AIC) for the BiCM model `m`. The parameters of the models most be computed beforehand. 
If the number of empirical observations becomes too small with respect to the number of parameters, you will get a warning. In 
that case, the corrected AIC (AICc) should be used instead.

# Examples
```julia
julia> model = BiCM(corporateclub());

julia> solve_model!(model);

julia> AIC(model);
[...]

```

See also [`AICc`](@ref MaxEntropyGraphs.AICc), [`L_BiCM_reduced`](@ref MaxEntropyGraphs.L_BiCM_reduced).
"""
function AIC(m::BiCM)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    # compute AIC components
    k = m.status[:N⊥] + m.status[:N⊤] # number of parameters
    n = m.status[:N⊥] *  m.status[:N⊤] # number of observations
    L = L_BiCM_reduced(m) # log-likelihood

    if n/k < 40
        @warn """The number of observations is small with respect to the number of parameters (n/k < 40). Consider using the corrected AIC (AICc) instead."""
    end

    return 2*k - 2*L
end


"""
    AICc(m::BiCM)

Compute the corrected Akaike Information Criterion (AICc) for the BiCM model `m`. The parameters of the models most be computed beforehand. 


# Examples
```jldoctest
julia> model = BiCM(corporateclub());

julia> solve_model!(model);

julia> AICc(model)
432.12227535579956

```

See also [`AIC`](@ref MaxEntropyGraphs.AIC), [`L_BiCM_reduced`](@ref MaxEntropyGraphs.L_BiCM_reduced).
"""
function AICc(m::BiCM)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    # compute AIC components
    k = m.status[:N⊥] + m.status[:N⊤] # number of parameters
    n = m.status[:N⊥] *  m.status[:N⊤] # number of observations
    L = L_BiCM_reduced(m) # log-likelihood

    return 2*k - 2*L + (2*k*(k+1)) / (n - k - 1)
end


"""
    BIC(m::BiCM)

Compute the Bayesian Information Criterion (BIC) for the BiCM model `m`. The parameters of the models most be computed beforehand. 
BIC is believed to be more restrictive than AIC, as the former favors models with a lower number of parameters than those favored by the latter.

# Examples
```julia-repl
julia> model = BiCM(corporateclub());

julia> solve_model!(model);

julia> BIC(model)
579.3789571131789

```

See also [`AIC`](@ref MaxEntropyGraphs.AIC), [`L_BiCM_reduced`](@ref MaxEntropyGraphs.L_BiCM_reduced).
"""
function BIC(m::BiCM)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    # compute AIC components
    k = m.status[:N⊥] + m.status[:N⊤] # number of parameters
    n = m.status[:N⊥] *  m.status[:N⊤] # number of observations
    L = L_BiCM_reduced(m) # log-likelihood

    return k * log(n) - 2*L
end