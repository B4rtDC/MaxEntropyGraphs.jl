
"""
    UBCM

Maximum entropy model for the Undirected Binary Configuration Model (UBCM). 
    
The object holds the maximum likelihood parameters of the model (θ) and optionally the expected adjacency matrix (G), 
and the variance for the elements of the adjacency matrix (σ). All settings and other metadata are stored in the `status` field.
"""
mutable struct UBCM{T,N} <: AbstractMaxEntropyModel where {T<:Union{Graphs.AbstractGraph, Nothing}, N<:Real}
    "Graph type, can be any subtype of AbstractGraph, but will be converted to SimpleGraph for the computation" # can also be empty
    const G::T 
    "Maximum likelihood parameters for reduced model"
    const θᵣ::Vector{N} 
    "Exponentiated maximum likelihood parameters for reduced model ( xᵢ = exp(-θᵢ) )"
    const xᵣ::Vector{N}
    "Degree sequence of the graph" # evaluate usefulness of this field later on
    const d::Vector{Int}
    "Reduced degree sequence of the graph"
    const dᵣ::Vector{Int}
    "Frequency of each degree in the degree sequence"
    const f::Vector{Int}
    "Indices to reconstruct the degree sequence from the reduced degree sequence"
    const d_ind::Vector{Int}
    "Indices to reconstruct the reduced degree sequence from the degree sequence"
    const dᵣ_ind::Vector{Int}
    "Expected adjacency matrix" # not always computed/required
    Ĝ::Union{Nothing, Matrix{N}}
    "Variance of the expected adjacency matrix" # not always computed/required
    σ::Union{Nothing, Matrix{N}}
    "Status indicators: parameters computed, expected adjacency matrix computed, variance computed, etc."
    const status::Dict{Symbol, Any}
    "Function used to computed the log-likelihood of the (reduced) model"
    fun::Union{Nothing, Function}
end

Base.show(io::IO, m::UBCM{T,N}) where {T,N} = print(io, """UBCM{$(T), $(N)} ($(m.status[:d]) vertices, $(m.status[:d_unique]) unique degrees, $(@sprintf("%.2f", m.status[:cᵣ])) compression ratio)""")

"""Return the reduced number of nodes in the UBCM network"""
Base.length(m::UBCM) = length(m.dᵣ)



"""
    UBCM(G::T; d::Vector=Graphs.degree(G), precision::N=Float64, kwargs...) where {T<:Graphs.AbstractGraph, N<:Real}
    UBCM(d::Vector{T}, precision::Type{<:AbstractFloat}=Float64, kwargs...) 

Constructor function for the `UBCM` type. 
    
By default and dependng on the graph type `T`, the definition of degree from `Graphs.jl` is applied. 
If you want to use a different definition of degree, you can pass a vector of degrees as the second argument.
If you want to generate a model directly from a degree sequence without an underlying graph, you can simply pass the degree sequence as an argument.
If you want to work from an adjacency matrix, or edge list, you can use the graph constructors from the `JuliaGraphs` ecosystem.

# Examples     
```jldoctest
# generating a model from a graph
julia> G = MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate)
{34, 78} undirected simple Int64 graph
julia> model = UBCM(G)
UBCM{Graphs.SimpleGraphs.SimpleGraph{Int64}, Float64} (34 vertices, 11 unique degrees, 0.32 compression ratio)
```
```jldoctest
# generating a model directly from a degree sequence
julia> model = UBCM(d=[4;3;3;3;2])
UBCM{Nothing, Float64} (5 vertices, 3 unique degrees, 0.60 compression ratio)
```
```jldoctest
# generating a model directly from a degree sequence with a different precision
julia> model = UBCM(d=[4;3;3;3;2], precision=Float16)
UBCM{Nothing, Float16} (5 vertices, 3 unique degrees, 0.60 compression ratio)
```
```jldoctest
# generating a model from an adjacency matrix
julia> A = [0 1 1;1 0 0;1 0 0];

julia> G = MaxEntropyGraphs.Graphs.SimpleGraph(A)
{3, 2} undirected simple Int64 graph
julia> model = UBCM(G)
UBCM{Graphs.SimpleGraphs.SimpleGraph{Int64}, Float64} (3 vertices, 2 unique degrees, 0.67 compression ratio)
```
```jldoctest
# generating a model from an edge list
julia> E = [(1,2),(1,3),(2,3)];

julia> edgelist = [MaxEntropyGraphs.Graphs.Edge(x,y) for (x,y) in E];

julia> G = MaxEntropyGraphs.Graphs.SimpleGraphFromIterator(edgelist)
{3, 3} undirected simple Int64 graph
julia> model = UBCM(G)
UBCM{Graphs.SimpleGraphs.SimpleGraph{Int64}, Float64} (3 vertices, 1 unique degrees, 0.33 compression ratio)
```

See also [`Graphs.degree`](https://juliagraphs.org/Graphs.jl/stable/core_functions/core/#Graphs.degree), [`SimpleWeightedGraphs.inneighbors`](https://juliagraphs.org/SimpleWeightedGraphs.jl/stable/api/#Graphs.inneighbors-Tuple{SimpleWeightedDiGraph,%20Integer}).
"""
function UBCM(G::T; d::Vector=Graphs.degree(G), precision::Type{<:AbstractFloat}=Float64, kwargs...) where {T}
    T <: Union{Graphs.AbstractGraph, Nothing} ? nothing : throw(TypeError("G must be a subtype of AbstractGraph or Nothing"))
    # coherence checks
    if T <: Graphs.AbstractGraph # Graph specific checks
        if Graphs.is_directed(G)
            @warn "The graph is directed, the UBCM model is undirected, the directional information will be lost"
        end

        if T <: SimpleWeightedGraphs.AbstractSimpleWeightedGraph
            @warn "The graph is weighted, the UBCM model is unweighted, the weight information will be lost"
        end

        if zero(eltype(d)) ∈ d
            @warn "The graph has vertices with zero degree, this may lead to convergence issues."
        end

        Graphs.nv(G) == 0 ? throw(ArgumentError("The graph is empty")) : nothing
        Graphs.nv(G) == 1 ? throw(ArgumentError("The graph has only one vertex")) : nothing

        Graphs.nv(G) != length(d) ? throw(DimensionMismatch("The number of vertices in the graph ($(Graphs.nv(G))) and the length of the degree sequence ($(length(d))) do not match")) : nothing
    end
    # coherence checks specific to the degree sequence
    length(d) == 0 ? throw(ArgumentError("The degree sequence is empty")) : nothing
    length(d) == 1 ? throw(ArgumentError("The degree sequence has only one degree")) : nothing
    maximum(d) >= length(d) ? throw(DomainError("The maximum degree in the graph is greater or equal to the number of vertices, this is not allowed")) : nothing

    # field generation
    dᵣ, d_ind , dᵣ_ind, f = np_unique_clone(d, sorted=true)
    θᵣ = Vector{precision}(undef, length(dᵣ))
    xᵣ = Vector{precision}(undef, length(dᵣ))
    status = Dict(  :params_computed=>false,        # are the parameters computed?
                    :G_computed=>false,             # is the expected adjacency matrix computed and stored?
                    :σ_computed=>false,             # is the standard deviation computed and stored?
                    :cᵣ => length(dᵣ)/length(d),    # compression ratio of the reduced model
                    :d_unique => length(dᵣ),        # number of unique degrees in the reduced model
                    :d => length(d)                 # number of vertices in the original graph 
                )
    
    return UBCM{T,precision}(G, θᵣ, xᵣ, d, dᵣ, f, d_ind, dᵣ_ind, nothing, nothing, status, nothing)
end

UBCM(;d::Vector{T}, precision::Type{<:AbstractFloat}=Float64, kwargs...) where {T<:Signed} = UBCM(nothing, d=d, precision=precision, kwargs...)


"""
    L_UBCM_reduced(θ::Vector, K::Vector, F::Vector)

Compute the log-likelihood of the reduced UBCM model using the exponential formulation in order to maintain convexity.

# Arguments
- `θ`: the maximum likelihood parameters of the model
- `K`: the reduced degree sequence
- `F`: the frequency of each degree in the degree sequence

The function returns the log-likelihood of the reduced model. For the optimisation, this function will be used to
generate an anonymous function associated with a specific model.

# Examples
```jldoctest
# Generic use:
julia> θ = [1.0, 2.0, 3.0, 4.0, 5.0];

julia> K = [1, 2, 3, 4, 5];

julia> F = [1, 2, 3, 4, 5];

julia> L_UBCM_reduced(θ, K, F)
-225.3065566905141
```
"""
function L_UBCM_reduced(θ::Vector, K::Vector, F::Vector)
    res = - sum(θ .* K .* F)
    for k in eachindex(K)
        @simd for k′ in eachindex(K)
            if k′ ≤ k
                if k == k′
                    @inbounds res -= F[k] * (F[k] - 1) * log(1 + exp(- θ[k] - θ[k′]) ) * .5 # to avoid counting it twice
                else
                    @inbounds res -= F[k] * F[k′]      * log(1 + exp(- θ[k] - θ[k′]) )
                end
                #@inbounds res -= F[k] * (F[k′] - (k==k′ ? 1. : 0.)) * log(1 + exp(- θ[k] - θ[k′]) ) * (k==k′ ? .5 : 1.) 
            end
        end
    end

    return res
end

"""
    L_UBCM_reduced(m::UBCM)

Return the log-likelihood of the UBCM model `m` based on the computed maximum likelihood parameters.

# Examples
```jldoctest
# Use with UBCM model:
julia> G = MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate);

julia> model = UBCM(G);

julia> solve_model!(model);

julia> L_UBCM_reduced(model)
-168.68325136302832
```

See also [`L_UBCM_reduced(::Vector, ::Vector, ::Vector)`](@ref)
"""
function L_UBCM_reduced(m::UBCM) 
    if m.status[:params_computed]
        return L_UBCM_reduced(m.θᵣ, m.dᵣ, m.f)
    else
        throw(ArgumentError("The parameters have not been computed yet"))
    end
end

"""
    ∇L_UBCM_reduced!(∇L::Vector, θ::Vector, K::Vector, F::Vector, x::Vector)

Compute the gradient of the log-likelihood of the reduced UBCM model using the exponential formulation (to maintain convexity).

For the optimisation, this function will be used togenerate an anonymous function associated with a specific model. 
The gradient is non-allocating and will update pre-allocated vectors (`∇L` and `x`) for speed. 

# Arguments
- `∇L`: the gradient of the log-likelihood of the reduced model
- `θ`: the maximum likelihood parameters of the model
- `K`: the reduced degree sequence
- `F`: the frequency of each degree in the degree sequence
- `x`: the exponentiated maximum likelihood parameters of the model ( xᵢ = exp(-θᵢ) )

# Examples
```jldoctest
# Explicit use with UBCM model:
julia> G = MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate);

julia> model = UBCM(G);

julia> ∇L = zeros(Real, length(model.θᵣ));

julia> x  = zeros(Real, length(model.θᵣ));

julia> ∇model_fun! = θ -> ∇L_UBCM_reduced!(∇L, θ, model.dᵣ, model.f, x);

julia> ∇model_fun!(model.θᵣ);

```
```jldoctest
# Use within optimisation.jl framework:
julia> model = UBCM(MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate));

julia> fun =  (θ, p) ->  - L_UBCM_reduced(θ, model.dᵣ, model.f);

julia> x  = zeros(Real, length(model.θᵣ)); # initialise gradient buffer

julia> ∇fun! = (∇L, θ, p) -> ∇L_UBCM_reduced!(∇L, θ, model.dᵣ, model.f, x); # define gradient

julia> θ₀ = initial_guess(model); # initial condition

julia> foo = MaxEntropyGraphs.Optimization.OptimizationFunction(fun, grad=∇fun!); # define target function 

julia> prob  = MaxEntropyGraphs.Optimization.OptimizationProblem(foo, θ₀); # define the optimisation problem

julia> method = MaxEntropyGraphs.OptimizationOptimJL.LBFGS(); # set the optimisation method

julia> MaxEntropyGraphs.Optimization.solve(prob, method); # solve it

```
"""
function ∇L_UBCM_reduced!(∇L::AbstractVector, θ::AbstractVector, K::Vector, F::Vector, x::AbstractVector)
    @simd for i in eachindex(x) # to avoid the allocation of exp.(-θ)
        @inbounds x[i] = exp(-θ[i])
    end

    for i in eachindex(K)
        ∇L[i] = - F[i] * K[i]
        for j in eachindex(K)
            if i == j
                aux = x[i] ^ 2
                ∇L[i] += F[i] * (F[i] - 1) * (aux / (1 + aux))
            else
                aux = x[i] * x[j]
                ∇L[i] += F[i] * F[j]       * (aux / (1 + aux))
            end
        end
    end

    return ∇L
end


"""
    ∇L_UBCM_reduced_minus!(args...)

Compute minus the gradient of the log-likelihood of the reduced UBCM model using the exponential formulation in order to maintain convexity. Used for optimisation in a non-allocating manner.

See also [`∇L_UBCM_reduced!`](@ref)
"""
function ∇L_UBCM_reduced_minus!(∇L::AbstractVector, θ::AbstractVector, K::Vector, F::Vector, x::AbstractVector)
    @simd for i in eachindex(x) # to avoid the allocation of exp.(-θ)
        @inbounds x[i] = exp(-θ[i])
    end
    @simd for i in eachindex(K)
        @inbounds ∇L[i] =  F[i] * K[i]
        for j in eachindex(K)
            if i == j
                aux = x[i] ^ 2
                @inbounds ∇L[i] -= F[i] * (F[i] - 1) * (aux / (1 + aux))
            else
                aux = x[i] * x[j]
                @inbounds ∇L[i] -= F[i] * F[j]       * (aux / (1 + aux))
            end
        end
    end

    return ∇L
end


"""
    UBCM_reduced_iter!(θ, K, F, x, G)

Computer the next fixed-point iteration for the UBCM model using the exponential formulation in order to maintain convexity.
The function will update pre-allocated vectors (`G` and `x`) for speed.

# Arguments
- `θ`: the maximum likelihood parameters of the model
- `K`: the reduced degree sequence
- `F`: the frequency of each degree in the degree sequence
- `x`: the exponentiated maximum likelihood parameters of the model ( xᵢ = exp(-θᵢ) ) (pre-allocated)
- `G`: the next fixed-point iteration for the UBCM model (pre-allocated)


# Examples
```jldoctest
julia> model = UBCM(MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate));

julia> G = zeros(eltype(model.θᵣ), length(model.θᵣ));

julia> x = zeros(eltype(model.θᵣ), length(model.θᵣ));

julia> UBCM_FP! = θ -> UBCM_reduced_iter!(θ, model.dᵣ, model.f, x, G);

julia> UBCM_FP!(initial_guess(model));

```
"""
function UBCM_reduced_iter!(θ::AbstractVector, K::AbstractVector, F::AbstractVector, x::AbstractVector, G::AbstractVector)
    @simd for i in eachindex(θ) # to avoid the allocation of exp.(-θ)
        @inbounds x[i] = exp(-θ[i])
    end
    G .= zero(eltype(G))
    @simd for i in eachindex(K)
        for j in eachindex(K)
            if i == j
                @inbounds G[i] += (F[j] - 1) * (x[j] / (1 + x[j] * x[i]))
            else
                @inbounds G[i] += (F[j]) *     (x[j] / (1 + x[j] * x[i]))
            end
        end

        if !iszero(G[i])
            @inbounds G[i] = -log(K[i] / G[i])
        end
    end
    return G
end

#=
"""
# results seem to vary with FP iterartion , not to be used
"""
function UBCM_reduced_iter_turbo!(θ::AbstractVector, K::AbstractVector, F::AbstractVector, x::AbstractVector, G::AbstractVector)
    @tturbo for i in eachindex(θ) # to avoid the allocation of exp.(-θ)
        x[i] = exp(-θ[i])
    end
    G .= zero(eltype(G))#, length(G))
    @tturbo for i in eachindex(K)
        for j in eachindex(K)
            # add "normal" contribution
            @inbounds G[i] += (F[j]) *     (x[j] / (1 + x[j] * x[i]))
        end
        # retify
        @inbounds G[i] -= x[i] / (1 + x[i] * x[i])
        #     if i == j
        #         @inbounds G[i] += (F[j] - 1) * (x[j] / (1 + x[j] * x[i]))
        #     else
        #         @inbounds G[i] += (F[j]) *     (x[j] / (1 + x[j] * x[i]))
        #     end
        # end

        #if !iszero(G[i])
        @inbounds G[i] = -log(K[i] / G[i])
        #end
    end
    # rectify the log of zero operation
    @simd for i in eachindex(G)
        if iszero(K[i])
            @inbounds G[i] = Inf
        end
    end
    return G
end
=#

"""
    initial_guess(m::UBCM, method::Symbol=:degrees)

Compute an initial guess for the maximum likelihood parameters of the UBCM model `m` using the method `method`.

The methods available are: 
- `:degrees` (default): the initial guess is computed using the degrees of the graph, i.e. ``\\theta_{i} = -\\log(d_{i})`` 
- `:degrees_minor`: the initial guess is computed using the degrees of the graph and the number of edges, i.e. ``\\theta_{i} = -\\log(d_{i}/(\\sqrt{E} + 1))``
- `:random`: the initial guess is computed using random values between 0 and 1, i.e. ``\\theta_{i} = -\\log(r_{i})`` where ``r_{i} \\sim U(0,1)``
- `:uniform`: the initial guess is uniformily set to 0.5, i.e. ``\\theta_{i} = -\\log(0.5)``
- `:chung_lu`: the initial guess is computed using the degrees of the graph and the number of edges, i.e. ``\\theta_{i} = -\\log(d_{i}/(2E))``

# Examples
```jldoctest
julia> model = UBCM(MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate));

julia> initial_guess(model, method=:random);

julia> initial_guess(model, method=:uniform);

julia> initial_guess(model, method=:degrees_minor);

julia> initial_guess(model, method=:chung_lu);

julia> initial_guess(model)
11-element Vector{Float64}:
 -0.0
 -0.6931471805599453
 -1.0986122886681098
 -1.3862943611198906
 -1.6094379124341003
 -1.791759469228055
 -2.1972245773362196
 -2.302585092994046
 -2.4849066497880004
 -2.772588722239781
 -2.833213344056216

```
"""
function initial_guess(m::UBCM{T,N}; method::Symbol=:degrees) where {T,N}
    #N = typeof(m).parameters[2]
    if isequal(method, :degrees)
        return Vector{N}(-log.(m.dᵣ))
    elseif isequal(method, :degrees_minor)
        isnothing(m.G) ? throw(ArgumentError("Cannot compute the number of edges because the model has no underlying graph (m.G == nothing)")) : nothing
        return Vector{N}(-log.(m.dᵣ ./ (sqrt(Graphs.ne(m.G)) + 1)))
    elseif isequal(method, :random)
        return Vector{N}(-log.(rand(N, length(m.dᵣ))))
    elseif isequal(method, :uniform)
        return Vector{N}(-log.(0.5 .* ones(N, length(m.dᵣ))))
    elseif isequal(method, :chung_lu)
        isnothing(m.G) ? throw(ArgumentError("Cannot compute the number of edges because the model has no underlying graph (m.G == nothing)")) : nothing
        return Vector{N}(-log.(m.dᵣ ./ (2 * Graphs.ne(m.G))))
    else
        throw(ArgumentError("The initial guess method $(method) is not supported"))
    end
end


"""
    set_xᵣ!(m::UBCM)

Set the value of xᵣ to exp(-θᵣ) for the UBCM model `m`
"""
function set_xᵣ!(m::UBCM)
    if m.status[:params_computed]
        m.xᵣ .= exp.(-m.θᵣ)
    else
        throw(ArgumentError("The parameters have not been computed yet"))
    end
end


"""
    Ĝ(m::UBCM)

Compute the expected adjacency matrix for the UBCM model `m`
"""
function Ĝ(m::UBCM{T,N}) where {T,N}
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    
    # get network size => this is the full size
    n = m.status[:d]
    # initiate G
    G = zeros(N, n, n)
    # initiate x
    x = m.xᵣ[m.dᵣ_ind]
    # compute G
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
    set_Ĝ!(m::UBCM)

Set the expected adjacency matrix for the UBCM model `m`
"""
function set_Ĝ!(m::UBCM)
    m.Ĝ = Ĝ(m)
    m.status[:G_computed] = true
    return m.Ĝ
end


"""
    σˣ(m::UBCM{T,N}) where {T,N}

Compute the standard deviation for the elements of the adjacency matrix for the UBCM model `m`.

**Note:** read as "sigma star"
"""
function σˣ(m::UBCM{T,N}) where {T,N}
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    # check network size => this is the full size
    n = m.status[:d]
    # initiate G
    σ = zeros(N, n, n)
    # initiate x
    x = m.xᵣ[m.dᵣ_ind]
    # compute σ
    for i = 1:n
        @simd for j = i+1:n
            @inbounds xij =  x[i]*x[j]
            @inbounds σ[i,j] = sqrt(xij)/(1 + xij)
            @inbounds σ[j,i] = sqrt(xij)/(1 + xij)
        end
    end

    return σ
end

"""
    set_σ!(m::UBCM)

Set the standard deviation for the elements of the adjacency matrix for the UBCM model `m`
"""
function set_σ!(m::UBCM)
    m.σ = σˣ(m)
    m.status[:σ_computed] = true
    return m.σ
end

"""
    rand(m::UBCM; precomputed=false)

Generate a random graph from the UBCM model `m`.

# Arguments
- `precomputed::Bool`: if `true`, the precomputed expected adjacency matrix (`m.Ĝ`) is used to generate the random graph, otherwise the maximum likelihood parameters are used to generate the random graph on the fly. For larger networks, it is 
  recommended to not precompute the expected adjacency matrix to limit memory pressure.

# Examples
```jldoctest
julia> model = UBCM(MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate)); # generate a UBCM model from the karate club network

julia> solve_model!(model); # compute the maximum likelihood parameters

julia> sample = rand(model); # sample a random graph

julia> typeof(sample)
Graphs.SimpleGraphs.SimpleGraph{Int64}
```
"""
function rand(m::UBCM; precomputed::Bool=false)
    if precomputed
        # check if possible to use precomputed Ĝ
        m.status[:G_computed] ? nothing : throw(ArgumentError("The expected adjacency matrix has not been computed yet"))
        # generate random graph
        G = Graphs.SimpleGraphFromIterator(Graphs.Edge.([(i,j) for i = 1:m.status[:d] for j in i+1:m.status[:d] if rand()<m.Ĝ[i,j]]))
    else
        # check if possible to use parameters
        m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
        # generate x vector
        x = m.xᵣ[m.dᵣ_ind]
        # generate random graph
        G = Graphs.SimpleGraphFromIterator(Graphs.Edge.([(i,j) for i = 1:m.status[:d] for j in i+1:m.status[:d] if rand()< (x[i]*x[j])/(1 + x[i]*x[j]) ]))
    end

    # deal with edge case where no edges are generated for the last node(s) in the graph
    while Graphs.nv(G) < m.status[:d]
        Graphs.add_vertex!(G)
    end

    return G
end


"""
    rand(m::UBCM, n::Int; precomputed=false)

Generate `n` random graphs from the UBCM model `m`. If multithreading is available, the graphs are generated in parallel.

# Arguments
- `precomputed::Bool`: if `true`, the precomputed expected adjacency matrix (`m.Ĝ`) is used to generate the random graph, otherwise the maximum likelihood parameters are used to generate the random graph on the fly. For larger networks, it is 
  recommended to not precompute the expected adjacency matrix to limit memory pressure.

# Examples
```jldoctest
julia> model = UBCM(MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate)); # generate a UBCM model from the karate club network

julia> solve_model!(model); # compute the maximum likelihood parameters

julia> sample = rand(model, 10); # sample a set of random graphs

julia> typeof(sample)
Vector{SimpleGraph{Int64}} (alias for Array{Graphs.SimpleGraphs.SimpleGraph{Int64}, 1})
```
"""
function rand(m::UBCM, n::Int; precomputed::Bool=false)
    # pre-allocate
    res = Vector{Graphs.SimpleGraph{Int}}(undef, n)
    # fill vector using threads
    Threads.@threads for i in 1:n
        res[i] = rand(m; precomputed=precomputed)
    end

    return res
end


"""
    solve_model!(m::UBCM; kwargs...)

Compute the likelihood maximising parameters of the UBCM model `m`. 

# Arguments
- `method::Symbol`: solution method to use, can be `:fixedpoint` (default), or :$(join(keys(MaxEntropyGraphs.optimization_methods), ", :", " and :")).
- `initial::Symbol`: initial guess for the parameters ``\\Theta``, can be :degrees, :degrees_minor, :random, :uniform, or :chung_lu.
- `maxiters::Int`: maximum number of iterations for the solver (defaults to 1000). 
- `verbose::Bool`: set to show log messages (defaults to false).
- `ftol::Real`: function tolerance for convergence with the fixedpoint method (defaults to 1e-8).
- `abstol::Union{Number, Nothing}`: absolute function tolerance for convergence with the other methods (defaults to `nothing`).
- `reltol::Union{Number, Nothing}`: relative function tolerance for convergence with the other methods (defaults to `nothing`).
- `AD_method::Symbol`: autodiff method to use, can be any of :$(join(keys(MaxEntropyGraphs.AD_methods), ", :", " and :")). Performance depends on the size of the problem (defaults to `:AutoZygote`),
- `analytical_gradient::Bool`: set the use the analytical gradient instead of the one generated with autodiff (defaults to `false`)

# Examples
```jldoctest
# default use
julia> model = UBCM(MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate));

julia> solve_model!(model);

```
```jldoctest
# default use
julia> model = UBCM(MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate));

julia> solve_model!(model, method=:BFGS, analytical_gradient=true, initial=:degrees_minor)
(UBCM{Graphs.SimpleGraphs.SimpleGraph{Int64}, Float64} (34 vertices, 11 unique degrees, 0.32 compression ratio), retcode: Success
u: [2.851659905903854, 2.053008374573552, 1.5432639513870743, 1.152360118212239, 0.8271267490690292, 0.5445045274064909, -0.1398726818076551, -0.3293252270659469, -0.6706207459338859, -1.2685575582149227, -1.410096540372487]
Final objective value:     168.68325136302835
)

```

See also: [`initial_guess`](@ref MaxEntropyGraphs.initial_guess(::UBCM)), [`∇L_UBCM_reduced!`](@ref MaxEntropyGraphs.∇L_UBCM_reduced!)
"""
function solve_model!(m::UBCM{T,N};  # common settings
                                method::Symbol=:fixedpoint, 
                                initial::Symbol=:degrees,
                                maxiters::Int=1000, 
                                verbose::Bool=false,
                                # NLsolve.jl specific settings (fixed point method)
                                ftol::Real=1e-8,
                                # optimisation.jl specific settings (optimisation methods)
                                abstol::Union{Number, Nothing}=nothing,
                                reltol::Union{Number, Nothing}=nothing,
                                AD_method::Symbol=:AutoZygote,
                                analytical_gradient::Bool=false) where {T,N}
    # initial guess
    θ₀ = initial_guess(m, method=initial)
    if method==:fixedpoint
        # initiate buffers
        x_buffer = zeros(N,length(m.dᵣ)); # buffer for x = exp(-θ)
        G_buffer = zeros(N,length(m.dᵣ)); # buffer for G(x)
        # define fixed point function
        FP_model! = (θ::Vector) -> UBCM_reduced_iter!(θ, m.dᵣ, m.f, x_buffer, G_buffer);
        # obtain solution
        sol = NLsolve.fixedpoint(FP_model!, θ₀, method=:anderson, ftol=ftol, iterations=maxiters);
        if NLsolve.converged(sol)
            if verbose 
                @info "Fixed point iteration converged after $(sol.iterations) iterations"
            end
            m.θᵣ .= sol.zero;
            m.status[:params_computed] = true;
            set_xᵣ!(m);
        else
            throw(ConvergenceError(method, nothing))
        end
    else
        if analytical_gradient
            # initialise gradient buffer
            x_buffer = zeros(N,length(m.dᵣ));
            # define gradient function for optimisation.jl
            grad! = (G, θ, p) -> ∇L_UBCM_reduced_minus!(G, θ, m.dᵣ, m.f, x_buffer);
        end
        # define objective function and its AD method
        f = AD_method ∈ keys(AD_methods)            ? Optimization.OptimizationFunction( (θ, p) ->   -L_UBCM_reduced(θ, m.dᵣ, m.f), AD_methods[AD_method],
                                                                                         grad = analytical_gradient ? grad! : nothing)                      : throw(ArgumentError("The AD method $(AD_method) is not supported (yet)"))
        prob = Optimization.OptimizationProblem(f, θ₀);
        # obtain solution
        sol = method ∈ keys(optimization_methods)   ? Optimization.solve(prob, optimization_methods[method], abstol=abstol, reltol=reltol)                                                : throw(ArgumentError("The method $(method) is not supported (yet)"))
        # check convergence
        if Optimization.SciMLBase.successful_retcode(sol.retcode)
            if verbose 
                @info """$(method) optimisation converged after $(@sprintf("%1.2e", sol.solve_time)) seconds (Optimization.jl return code: $("$(sol.retcode)"))"""
            end
            m.θᵣ .= sol.u;
            m.status[:params_computed] = true;
            set_xᵣ!(m);
        else
            throw(ConvergenceError(method, sol.retcode))
        end
    end

    return m, sol
end


"""
    precision(m::UBCM)

Determine the compute precision of the UBCM model `m`.

# Examples
```jldoctest
julia> model = UBCM(MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate));

julia> MaxEntropyGraphs.precision(model)
Float64
```

```jldoctest
julia> model = UBCM(MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate), precision=Float32);

julia> MaxEntropyGraphs.precision(model)
Float32
```
"""
precision(m::UBCM) = typeof(m).parameters[2]


"""
    f_UBCM(x::T)

Helper function for the UBCM model to compute the expected value of the adjacency matrix. The function compute the expression `x / (1 + x)`.
As an argument you need to pass the product of the maximum likelihood parameters `xᵣ[i] * xᵣ[j]` from a UBCM model.
"""
f_UBCM(xixj::T) where {T} = xixj / (one(T) + xixj)


"""
    A(m::UBCM,i::Int,j::Int)

Return the expected value of the adjacency matrix for the UBCM model `m` at the node pair `(i,j)`.

❗ For perfomance reasons, the function does not check:
- if the node pair is valid.
- if the parameters of the model have been computed.
"""
function A(m::UBCM,i::Int,j::Int)
    return i == j ? zero(precision(m)) : @inbounds f_UBCM(m.xᵣ[m.dᵣ_ind[i]] * m.xᵣ[m.dᵣ_ind[j]])
end

"""
    degree(m::UBCM, i::Int; method=:reduced)

Return the expected degree vector for node `i` of the UBCM model `m`.
Uses the reduced model parameters `xᵣ` for perfomance reasons.

# Arguments
- `m::UBCM`: the UBCM model
- `i::Int`: the node for which to compute the degree.
- `method::Symbol`: the method to use for computing the degree. Can be any of the following:
    - `:reduced` (default) uses the reduced model parameters `xᵣ` for perfomance reasons.
    - `:full` uses all elements of the expected adjacency matrix.
    - `:adjacency` uses the precomputed adjacency matrix `m.Ĝ` of the model.

# Examples
```jldoctest
julia> model = UBCM(MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate));

julia> solve_model!(model);

julia> set_Ĝ!(model);

julia> typeof([degree(model, 1), degree(model, 1, method=:full), degree(model, 1, method=:adjacency)])
Vector{Float64} (alias for Array{Float64, 1})

``` 
"""
function degree(m::UBCM, i::Int; method::Symbol=:reduced)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    i > m.status[:d] ? throw(ArgumentError("Attempted to access node $i in a $(m.status[:d]) node graph")) : nothing
    
    if method == :reduced
        res = zero(precision(m))
        i_red = m.dᵣ_ind[i] # find matching index in reduced model
        for j in eachindex(m.xᵣ)
            if i_red ≠ j 
                res += @inbounds f_UBCM(m.xᵣ[i_red] * m.xᵣ[j]) * m.f[j]
            else
                res += @inbounds f_UBCM(m.xᵣ[i_red] * m.xᵣ[i_red]) * (m.f[j] - 1) # subtract 1 because the diagonal is not counted
            end
        end
    elseif method == :full
        # using all elements of the adjacency matrix
        res = zero(precision(m))
        for j in eachindex(m.d)
            res += A(m, i, j)
        end
    elseif method == :adjacency
        #  using the precomputed adjacency matrix 
        m.status[:G_computed] ? nothing : throw(ArgumentError("The adjacency matrix has not been computed yet"))
        res = sum(@view m.Ĝ[i,:])  
    else
        throw(ArgumentError("Unknown method $method"))
    end

    return res
end


"""
    degree(m::UBCM[, v]; method=:reduced)

Return a vector corresponding to the expected degree of the UBCM model `m` each node. If v is specified, only return degrees for nodes in v.

# Arguments
- `m::UBCM`: the UBCM model
- `v::Vector{Int}`: the nodes for which to compute the degree. Default is all nodes.
- `method::Symbol`: the method to use for computing the degree. Can be any of the following:
    - `:reduced` (default) uses the reduced model parameters `xᵣ` for perfomance reasons.
    - `:full` uses all elements of the expected adjacency matrix.
    - `:adjacency` uses the precomputed adjacency matrix `m.Ĝ` of the model.

# Examples
```jldoctest
julia> model = UBCM(MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate));

julia> solve_model!(model);

julia> set_Ĝ!(model);

julia> typeof(degree(model, method=:adjacency)) 
Vector{Float64} (alias for Array{Float64, 1})

``` 
"""
degree(m::UBCM, v::Vector{Int}=collect(1:m.status[:d]); method::Symbol=:reduced) = [degree(m, i, method=method) for i in v]

"""
    AIC(m::UBCM)

Compute the Akaike Information Criterion (AIC) for the UBCM model `m`. The parameters of the models most be computed beforehand. 
If the number of empirical observations becomes too small with respect to the number of parameters, you will get a warning. In 
that case, the corrected AIC (AICc) should be used instead.

# Examples
```julia-repl
julia> model = UBCM(MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate));

julia> solve_model!(model);

julia> AIC(model);

```

See also [`AICc`](@ref MaxEntropyGraphs.AICc), [`L_UBCM_reduced`](@ref MaxEntropyGraphs.L_UBCM_reduced).
"""
function AIC(m::UBCM)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    # compute AIC components
    k = m.status[:d] # number of parameters
    n = (m.status[:d] - 1) * m.status[:d] / 2 # number of observations
    L = L_UBCM_reduced(m) # log-likelihood

    if n/k < 40
        @warn """The number of observations is small with respect to the number of parameters (n/k < 40). Consider using the corrected AIC (AICc) instead."""
    end

    return 2*k - 2*L
end


"""
    AICc(m::UBCM)

Compute the corrected Akaike Information Criterion (AICc) for the UBCM model `m`. The parameters of the models most be computed beforehand. 


# Examples
```jldoctest
julia> model = UBCM(MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate));

julia> solve_model!(model);

julia> AICc(model)
409.891217554954

```

See also [`AIC`](@ref MaxEntropyGraphs.AIC), [`L_UBCM_reduced`](@ref MaxEntropyGraphs.L_UBCM_reduced).
"""
function AICc(m::UBCM)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    # compute AIC components
    k = m.status[:d] # number of parameters
    n = (m.status[:d] - 1) * m.status[:d] / 2 # number of observations
    L = L_UBCM_reduced(m) # log-likelihood

    return 2*k - 2*L + (2*k*(k+1)) / (n - k - 1)
end


"""
    BIC(m::UBCM)

Compute the Bayesian Information Criterion (BIC) for the UBCM model `m`. The parameters of the models most be computed beforehand. 
BIC is believed to be more restrictive than AIC, as the former favors models with a lower number of parameters than those favored by the latter.

# Examples
```julia-repl
julia> model = UBCM(MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate));

julia> solve_model!(model);

julia> BIC(model)
552.5770135138283

```

See also [`AIC`](@ref MaxEntropyGraphs.AIC), [`L_UBCM_reduced`](@ref MaxEntropyGraphs.L_UBCM_reduced).
"""
function BIC(m::UBCM)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    # compute AIC components
    k = m.status[:d] # number of parameters
    n = (m.status[:d] - 1) * m.status[:d] / 2 # number of observations
    L = L_UBCM_reduced(m) # log-likelihood

    return k * log(n) - 2*L
end


