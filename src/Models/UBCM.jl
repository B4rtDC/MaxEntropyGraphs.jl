
"""
    UBCM

Maximum entropy model for the Undirected Binary Configuration Model (UBCM). 
    
The object holds the maximum likelihood parameters of the model (θ), the expected adjacency matrix (G), 
and the variance for the elements of the adjacency matrix (σ).
"""
mutable struct UBCM{T,N} <: AbstractMaxEntropyModel where {T<:Union{Graphs.AbstractGraph, Nothing}, N<:Real}
    "Graph type, can be any subtype of AbstractGraph, but will be converted to SimpleGraph for the computation" # can also be empty
    const G::T 
    "Maximum likelihood parameters for reduced model"
    const Θᵣ::Vector{N}
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

Constructor function for the `UBCM` type. 
    
By default and dependng on the graph type `T`, the definition of degree from `Graphs.jl` is applied. 
If you want to use a different definition of degree, you can pass a vector of degrees as the second argument.
If you want to generate a model directly from a degree sequence without an underlying graph , you can simply pass the degree sequence as an argument.
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
    Θᵣ = Vector{precision}(undef, length(dᵣ))
    xᵣ = Vector{precision}(undef, length(dᵣ))
    status = Dict(  :params_computed=>false,        # are the parameters computed?
                    :G_computed=>false,             # is the expected adjacency matrix computed and stored?
                    :σ_computed=>false,             # is the standard deviation computed and stored?
                    :cᵣ => length(dᵣ)/length(d),    # compression ratio of the reduced model
                    :d_unique => length(dᵣ),        # number of unique degrees in the reduced model
                    :d => length(d)                 # number of vertices in the original graph 
                )
    
    return UBCM{T,precision}(G, Θᵣ, xᵣ, d, dᵣ, f, d_ind, dᵣ_ind, nothing, nothing, status, nothing)
end

UBCM(;d::Vector{T}, precision::Type{<:AbstractFloat}=Float64, kwargs...) where {T<:Signed} = UBCM(nothing, d=d, precision=precision, kwargs...)


"""
    L_UBCM_reduced(θ::Vector, K::Vector, F::Vector)

Compute the log-likelihood of the reduced UBCM model using the exponential formulation in order to maintain convexity.

The arguments of the function are:
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
```jldoctest
# Use with UBCM model:
julia> G = MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate);

julia> model = UBCM(G);

julia> model_fun = θ -> L_UBCM_reduced(θ, model.dᵣ, model.f)

julia> model_fun(model.Θᵣ)
-388.8555682941297
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

See also [`L_UBCM_reduced(::Vector, ::Vector, ::Vector)`](@ref)
"""
function L_UBCM_reduced(m::UBCM) 
    if m.status[:params_computed]
        return L_UBCM_reduced(m.Θᵣ, m.dᵣ, m.f)
    else
        throw(ArgumentError("The parameters have not been computed yet"))
    end
    return L_UBCM_reduced(m.θᵣ, m.dᵣ, m.f)
end

"""
    ∇L_UBCM_reduced!( ∇L::Vector, θ::Vector, K::Vector, F::Vector, x::Vector)

Compute the gradient of the log-likelihood of the reduced UBCM model using the exponential formulation in order to maintain convexity.

For the optimisation, this function will be used togenerate an anonymous function associated with a specific model. The function 
will update pre-allocated vectors (`∇L` and `x`) for speed. The gradient is non-allocating.

The arguments of the function are:
    - `∇L`: the gradient of the log-likelihood of the reduced model
    - `θ`: the maximum likelihood parameters of the model
    - `K`: the reduced degree sequence
    - `F`: the frequency of each degree in the degree sequence
    - `x`: the exponentiated maximum likelihood parameters of the model ( xᵢ = exp(-θᵢ) )

# Examples
```julia-repl
# Explicit use with UBCM model:
julia> G = Graphs.SimpleGraphs.smallgraph(:karate);
julia> model = UBCM(G);
julia> ∇L = zeros(Real, length(model.Θᵣ);
julia> x  = zeros(Real, length(model.Θᵣ);
julia> ∇model_fun! = θ -> ∇L_UBCM_reduced!(θ::AbstractVector, K, F, ∇L, x);
julia> ∇model_fun!(model.Θᵣ)
# Use within optimisation.jl framework:
julia> fun =   (θ, p) ->  - MaxEntropyGraphs.L_UBCM_reduced(θ, model.dᵣ, model.f)
julia> ∇fun! = (∇L, θ, p) -> MaxEntropyGraphs.∇L_UBCM_reduced!(∇L, θ, K, F, x);
julia> θ₀ = -log.( model.dᵣ ./ maximum(model.dᵣ)) # initial condition
julia> foo = MaxEntropyGraphs.Optimization.OptimizationProblem(fun, grad=∇fun!)
julia> prob  = MaxEntropyGraphs.Optimization.OptimizationFunction(prob, θ₀)
julia> method = MaxEntropyGraphs.OptimizationOptimJL.NLopt.LD_LBFGS()
julia> solve(prob, method)
...
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

The arguments of the function are:
    - `θ`: the maximum likelihood parameters of the model
    - `K`: the reduced degree sequence
    - `F`: the frequency of each degree in the degree sequence
    - `x`: the exponentiated maximum likelihood parameters of the model ( xᵢ = exp(-θᵢ) )
    - `G`: the next fixed-point iteration for the UBCM model


# Examples
```julia-repl
# Use with UBCM model:
julia> G = Graphs.SimpleGraphs.smallgraph(:karate);
julia> model = UBCM(G);
julia> G = zeros(eltype(model.Θᵣ), length(model.Θᵣ);
julia> x = zeros(eltype(model.Θᵣ), length(model.Θᵣ);
julia> UBCM_FP! = θ -> UBCM_reduced_iter!(θ::AbstractVector, K, F, x, G);
julia> UBCM_FP!(model.Θᵣ)
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

The methods available are: `:degrees` (default), `:degrees_minor`, `:random`, `:uniform`, `:chung_lu`.
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
        m.xᵣ .= exp.(-m.Θᵣ)
    else
        throw(ArgumentError("The parameters have not been computed yet"))
    end
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
    set_σ!(m::UBCM)

Set the standard deviation for the elements of the adjacency matrix for the UBCM model `m`
"""
function set_σ!(m::UBCM)
    m.σ = σˣ(m)
    m.status[:σ_computed] = true
    return m.σ
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
    rand(m::UBCM; precomputed=false)

Generate a random graph from the UBCM model `m`.

Keyword arguments:
- `precomputed::Bool`: if `true`, the precomputed expected adjacency matrix (`m.Ĝ`) is used to generate the random graph, otherwise the maximum likelihood parameters are used to generate the random graph on the fly. For larger networks, it is 
  recommended to not precompute the expected adjacency matrix to limit memory pressure.

# Examples
```julia-repl
julia> using MaxEntropyGraphs
# generate a UBCM model from the karate club network
julia> G = MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate);
julia> model = MaxEntropyGraphs.UBCM(G);
# compute the maximum likelihood parameters
using NLsolve
x_buffer = zeros(length(model.dᵣ));G_buffer = zeros(length(model.dᵣ));
FP_model! = (θ::Vector) -> MaxEntropyGraphs.UBCM_reduced_iter!(θ, model.dᵣ, model.f, x_buffer, G_buffer);
sol = fixedpoint(FP_model!, θ₀, method=:anderson, ftol=1e-12, iterations=1000);
model.Θᵣ .= sol.zero;
model.status[:params_computed] = true;
set_xᵣ!(model);
# set the expected adjacency matrix
MaxEntropyGraphs.set_Ĝ!(model);
# sample a random graph
julia> rand(model)
{34, 78} undirected simple Int64 graph
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

Keyword arguments:
- `precomputed::Bool`: if `true`, the precomputed expected adjacency matrix (`m.Ĝ`) is used to generate the random graph, otherwise the maximum likelihood parameters are used to generate the random graph on the fly. For larger networks, it is 
  recommended to not precompute the expected adjacency matrix to limit memory pressure.

# Examples
```julia-repl
julia> using MaxEntropyGraphs
## generate a UBCM model from the karate club network
julia> G = MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate);
julia> model = MaxEntropyGraphs.UBCM(G);
## compute the maximum likelihood parameters
using NLsolve
x_buffer = zeros(length(model.dᵣ));G_buffer = zeros(length(model.dᵣ));
FP_model! = (θ::Vector) -> MaxEntropyGraphs.UBCM_reduced_iter!(θ, model.dᵣ, model.f, x_buffer, G_buffer);
sol = fixedpoint(FP_model!, θ₀, method=:anderson, ftol=1e-12, iterations=1000);
model.Θᵣ .= sol.zero;
model.status[:params_computed] = true;
set_xᵣ!(model);
# set the expected adjacency matrix
MaxEntropyGraphs.set_Ĝ!(model);
# sample a random graph
julia> rand(model, 10)
10-element Vector{Graphs.SimpleGraphs.SimpleGraph{Int64}}
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
    solve_model!(m::UBCM)

Compute the likelihood maximising parameters of the UBCM model `m`. 

By default the parameters are computed using the fixed point iteration method with the degree sequence as initial guess.
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
            m.Θᵣ .= sol.zero;
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
            m.Θᵣ .= sol.u;
            m.status[:params_computed] = true;
            set_xᵣ!(m);
        else
            throw(ConvergenceError(method, sol.retcode))
        end
    end

    return m, sol
end

precision(m::UBCM) = typeof(m).parameters[2]
