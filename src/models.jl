##################################################################################
# models.jl
#
# This file contains model types and methods for the MaxEntropyGraphs.jl package
##################################################################################

"""
    AbstractMaxEntropyModel

An abstract type for a MaxEntropyModel. Each model has one or more structural constraints  
that are fixed while the rest of the network is completely random. 
"""
abstract type AbstractMaxEntropyModel end


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

#"""E"""
#Base.precision(m::UBCM) = eltype(m.Θᵣ)


"""
    UBCM(G::T; precision::N=Float64, kwargs...) where {T<:Graphs.AbstractGraph, N<:Real}

Constructor function for the `UBCM` type. 
    
By default and dependng on the graph type `T`, the definition of degree from ``Graphs.jl`` is applied. 
If you want to use a different definition of degree, you can pass a vector of degrees as the second argument.
If you want to generate a model directly from a degree sequence without an underlying graph , you can simply pass the degree sequence as an argument.
If you want to work from an adjacency matrix, or edge list, you can use the graph constructors from the ``JuliaGraphs`` ecosystem.

# Examples     
```jldoctest
# generating a model from a graph
julia> G = MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate)
{34, 78} undirected simple Int64 graph
julia> model = UBCM(G)
UBCM{SimpleGraph{Int64}, Float64} (34 vertices, 11 unique degrees, 0.32 compression ratio)

# generating a model directly from a degree sequence
julia> model = UBCM([4;3;3;3;2])
UBCM{Nothing, Float64} (5 vertices, 3 unique degrees, 0.60 compression ratio)

# generating a model directly from a degree sequence with a different precision
julia> model = UBCM([4;3;3;3;2], precision=Float16)
UBCM{Nothing, Float16} (5 vertices, 3 unique degrees, 0.60 compression ratio)

# generating a model from an adjacency matrix
julia> A = [0 1 1;1 0 0;1 0 0];
julia> G = MaxEntropyGraphs.Graphs.SimpleGraph(A)
{3, 2} undirected simple Int64 graph
julia> model = UBCM(G)
UBCM{SimpleGraph{Int64}, Float64} (3 vertices, 2 unique degrees, 0.67 compression ratio)

# generating a model from an edge list
julia> E = [(1,2),(1,3),(2,3)];
julia> edgelist = [MaxEntropyGraphs.Graphs.Edge(x,y) for (x,y) in E];
julia> G = MaxEntropyGraphs.Graphs.SimpleGraphFromIterator(edgelist)
{3, 3} undirected simple Int64 graph
julia> model = UBCM(G)
UBCM{SimpleGraph{Int64}, Float64} (3 vertices, 1 unique degrees, 0.33 compression ratio)
```

See also [`Graphs.degree`](@ref), [`SimpleWeightedGraphs.degree`](@ref).
"""
function UBCM(G::T, d::Vector=Graphs.degree(G); precision::Type{<:AbstractFloat}=Float64, kwargs...) where {T}
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

UBCM(d::Vector{T}; precision::Type{<:AbstractFloat}=Float64, kwargs...) where {T<:Signed} = UBCM(nothing, d; precision=precision, kwargs...)


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

# Use with UBCM model:
julia> G = MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate);
julia> model = UBCM(G);
julia> model_fun = θ -> L_UBCM_reduced(θ, model.dᵣ, model.f)
julia> model_fun(model.Θᵣ)
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
    ∇L_UBCM_reduced!(θ::Vector, K::Vector, F::Vector, ∇L::Vector, x::Vector)

Compute the gradient of the log-likelihood of the reduced UBCM model using the exponential formulation in order to maintain convexity.

For the optimisation, this function will be used togenerate an anonymous function associated with a specific model. The function 
will update pre-allocated vectors (`∇L` and `x`) for speed.

The arguments of the function are:
    - `θ`: the maximum likelihood parameters of the model
    - `K`: the reduced degree sequence
    - `F`: the frequency of each degree in the degree sequence
    - `∇L`: the gradient of the log-likelihood of the reduced model
    - `x`: the exponentiated maximum likelihood parameters of the model ( xᵢ = exp(-θᵢ) )

# Examples
```jldoctest
# Explicit use with UBCM model:
julia> G = Graphs.SimpleGraphs.smallgraph(:karate);
julia> model = UBCM(G);
julia> ∇L = zeros(Real, length(model.Θᵣ);
juliaW x  = zeros(Real, length(model.Θᵣ);
julia> ∇model_fun! = θ -> ∇L_UBCM_reduced!(θ::AbstractVector, K, F, ∇L, x);
julia> ∇model_fun!(model.Θᵣ)
# Use within optimisation.jl framework:
julia> fun =   (θ, p) ->  - MaxEntropyGraphs.L_UBCM_reduced(θ, model.dᵣ, model.f)
julia> ∇fun! = (θ, p) -> MaxEntropyGraphs.∇L_UBCM_reduced!(θ::AbstractVector, K, F, ∇L, x);
julia> θ₀ = -log.( model.dᵣ ./ maximum(model.dᵣ)) # initial condition
julia> prob = MaxEntropyGraphs.Optimization.OptimizationProblem(fun, θ₀)
julia> fun  = MaxEntropyGraphs.Optimization.OptimizationFunction(prob, grad=∇fun!)
julia> method = MaxEntropyGraphs.OptimizationOptimJL.NLopt.LD_LBFGS()
julia> solve(fun, method)
...
```
"""
function ∇L_UBCM_reduced!(θ::AbstractVector, K::Vector, F::Vector, ∇L::AbstractVector, x::AbstractVector)
    x .= exp.(-θ)
    @simd for i in eachindex(K)
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
```jldoctest
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
    x .= exp.(-θ)
    G .= zero(eltype(G))#, length(G))
    @simd for i in eachindex(K)
        for j in eachindex(K)
            if i == j
                G[i] += (F[j] - 1) * (x[j] / (1 + x[j] * x[i]))
            else
                G[i] += (F[j]) *     (x[j] / (1 + x[j] * x[i]))
            end
        end

        if !iszero(G[i])
            G[i] = -log(K[i] / G[i])
        end
    end
    return G
end

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
        return Vector{N}(-log.(m.dᵣ ./ (sqrt(Graphs.ne(m.G)) + 1)))
    elseif isequal(method, :random)
        return Vector{N}(-log.(rand(N, length(m.dᵣ))))
    elseif isequal(method, :uniform)
        return Vector{N}(-log.(0.5 .* ones(N, length(m.dᵣ))))
    elseif isequal(method, :chung_lu)
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
```jldoctest
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
```jldoctest
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
    ConvergenceError

Exception thrown when the optimisation method does not converge. 

When using and optimisation method from the `optimisation.jl` framework, the return code of the optimisation method is stored in the `retcode` field.
When using the fixed point iteration method, the `retcode` field is set to `nothing`.
"""
struct ConvergenceError{T} <: Exception where {T<:Optimization.SciMLBase.EnumX.Enum{Int32}}
    method::Symbol
    retcode::Union{Nothing, T}
end

Base.showerror(io::IO, e::ConvergenceError) = print(io, """method `$(e.method)` did not converge $(isnothing(e.retcode) ? "" : "(Optimization.jl return code: $(e.retcode))")""")

const optimization_methods = Dict(  :LBFGS      => OptimizationOptimJL.LBFGS(),
                                    :BFGS       => OptimizationOptimJL.BFGS(),
                                    :Newton     => OptimizationOptimJL.Newton(),
                                    :NelderMead => OptimizationOptimJL.NelderMead())

const AD_methods = Dict(:AutoZygote         => Optimization.AutoZygote(),
                        :AutoForwardDiff    => Optimization.AutoForwardDiff(),
                        :AutoReverseDiff    => Optimization.AutoReverseDiff(),
                        :AutoFiniteDiff     => Optimization.AutoFiniteDiff())

"""
    solve_model!(m::UBCM)

Compute the likelihood maximising parameters of the UBCM model `m`. By default the parameters are computed using the fixed point iteration method with the degree sequence as initial guess.
"""
function solve_model!(m::UBCM;  # common settings
                                method::Symbol=:fixedpoint, 
                                initial_method::Symbol=:degrees,
                                maxiters::Int=1000, 
                                verbose::Bool=false,
                                # NLsolve.jl specific settings (fixed point method)
                                ftol::Real=1e-8,
                                # optimisation.jl specific settings (optimisation methods)
                                abstol::Union{Number, Nothing}=nothing,
                                reltol::Union{Number, Nothing}=nothing,
                                AD_method::Symbol=:AutoZygote)
    # initial guess
    θ₀ = initial_guess(m, method=initial_method)
    if method==:fixedpoint
        # initiate buffers
        x_buffer = zeros(length(m.dᵣ)); # buffer for x = exp(-θ)
        G_buffer = zeros(length(m.dᵣ)); # buffer for G(x)
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
        # define objective function and with its AD method
        f = AD_method ∈ keys(AD_methods)            ? Optimization.OptimizationFunction( (θ, p) ->  - L_UBCM_reduced(θ, m.dᵣ, m.f), AD_methods[AD_method])  : throw(ArgumentError("The AD method $(AD_method) is not supported (yet)"))
        prob = Optimization.OptimizationProblem(f, θ₀);
        # obtain solution
        sol = method ∈ keys(optimization_methods)   ? Optimization.solve(prob, optimization_methods[method])                                                : throw(ArgumentError("The method $(method) is not supported (yet)"))
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

    return m
end




"""
    solve_model(m::UBCM, method::Symbol)

solve_model(::T, method::Symbol) where {T<:AbstractMaxEntropyModel} = throw(MethodError(solve_model, method))
jebroer() = @info "dag brave jongens, waar is je moeder?"
vake() = @info "dag brave jongens"
jemoeder() = @warn "Wee je gebeente!"
jevader() = @warn "Je vader zal er van horen!"

test(m::UBCM{T}) where T = @info T
newtest() = @warn "Succes?"






mutable struct DBCM{T,N} <: AbstractMaxEntropyModel where {T<:Union{Graphs.AbstractGraph, Nothing}, N<:Real}
    "Graph type, can be any subtype of AbstractGraph, but will be converted to SimpleDiGraph for the computation" # can also be empty
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



"""
#    Ĝ(m::UBCM)


# G = MODELS.Graphs.SimpleGraphs.smallgraph(:karate)
# model = MODELS.UBCM(G,)

# G = Graphs.SimpleDiGraph(5)
# nv(G)
# res = MODELS.UBCM(G, precision=Float16)
# res = MODELS.UBCM(G, [4;3;3;3;2])
# res.Θᵣ, res.dᵣ

# res.dᵣ[res.dᵣ_ind] == res.d
# res.Θᵣ[res.dᵣ_ind]
# res.Θᵣ
# """
# Idea: starting from models with known parameters:
# - obtain expected values and variances for adjacency/weight matrix elements
# - sample networks, returning 
#     1. Adjacency matrix (dense/ sparse (ULT)) 
#     2. Graph 
#     3. Adjacency List & node number
# - compute z-scores of different metrics by 
#     1. "exact" method 
#     2. sampling method
# """
# #= run this once at startup
# if Sys.islinux()
#     ENV["GRDIR"] = "" # for headless plotting
#     using Pkg; Pkg.build("GR")
#     # sudo apt install xvfb
#     # https://gr-framework.org/julia.html#installation
#     import GR:inline
#     GR.inline("pdf")
#     GR.inline("png")
# end
# =#


# # ----------------------------------------------------------------------------------------------------------------------
# #
# #                                               General model
# #
# # ----------------------------------------------------------------------------------------------------------------------



# """
#     σ(::AbstractMaxEntropyModel)

# Compute variance for elements of the adjacency matrix for the specific `AbstractMaxEntropyModel` based on the ML parameters.
# """
# σ(::AbstractMaxEntropyModel) = nothing


# """
#     Ĝ(::AbstractMaxEntropyModel)

# Compute expected adjacency and/or weight matrix for a given `AbstractMaxEntropyModel`
# """
# Ĝ(::AbstractMaxEntropyModel) = nothing


# """
#     rand(::AbstractMaxEntropyModel)

# Sample a random network from the `AbstractMaxEntropyModel`
# """
# Base.rand(::AbstractMaxEntropyModel) = nothing

# """
#     ∇X(X::Function, M::T)

# Compute the gradient of a property `X` with respect to the expected adjacency matrix associated with the model `M`.
# """
# ∇X(X::Function, M::T) where T <: AbstractMaxEntropyModel = ReverseDiff.gradient(X, M.G)


# """
#     σˣ(X::Function, M::T)

# Compute the standard deviation of a property `X` with respect to the expected adjacency matrix associated with the model `M`.
# """
# σˣ(X::Function, M::T) where T <: AbstractMaxEntropyModel = nothing

# # ----------------------------------------------------------------------------------------------------------------------
# #
# #                                               UBCM model
# #
# # ----------------------------------------------------------------------------------------------------------------------

# """
#     UBCM

# Maximum entropy model for the Undirected Binary Configuration Model (UBCM). 
    
# The object holds the maximum likelihood parameters of the model (x), the expected adjacency matrix (G), 
# and the variance for the elements of the adjacency matrix (σ).

# """
# struct UBCM{T} <: AbstractMaxEntropyModel where {T<:Real}
#     x::Vector{T}
#     G::Matrix{T}
#     σ::Matrix{T}
# end



# """
#     UBCM(x::Vector{T}; compute::Bool=true) where {T<:Real}

# Constructor for the `UBCM` type.
# """
# function UBCM(x::Vector{T}) where {T<:Real}
#     G = Ĝ(x, UBCM{T})  # expected adjacency matrix
#     σ = σˣ(x, UBCM{T}) # expected standard deviation matrix

#     return UBCM(x, G, σ)
# end

# """
#     UBCM(G::T) where T<:SimpleGraph

# Constructor for the `UBCM` type based on a `SimpleGraph`. 
# """
# function UBCM(G::T; method="fixed-point", initial_guess="degrees", max_steps=5000, tol=1e-12, kwargs...) where T<:Graphs.SimpleGraph
#     NP = PyCall.pyimport("NEMtropy")
#     G_nem = NP.UndirectedGraph(degree_sequence=Graphs.degree(G))
#     G_nem.solve_tool(model="cm_exp", method=method, initial_guess=initial_guess, max_steps=max_steps, tol=tol, kwargs...);
#     if G_nem.error > 1e-7
#         @warn "The model did not converge, maybe try some other options (solution error $(G_nem.error))"
#     end
#     return UBCM(G_nem.x)
# end

# """
#     Ĝ(::UBCM, x::Vector{T}) where {T<:Real}

# Compute the expected adjacency matrix for the UBCM model with maximum likelihood parameters `x`.
# """
# function Ĝ(x::Vector{T}, ::Type{UBCM{T}}) where T
#     n = length(x)
#     G = zeros(T, n, n)
#     for i = 1:n
#         @simd for j = i+1:n
#             @inbounds xij = x[i]*x[j]
#             @inbounds G[i,j] = xij/(1 + xij)
#             @inbounds G[j,i] = xij/(1 + xij)
#         end
#     end
    
#     return G
# end

# """
#     σˣ(x::Vector{T}, ::Type{UBCM{T}}) where T

# Compute the standard deviation for the elements of the adjacency matrix for the UBCM model using the maximum likelihood parameters `x`.

# **Note:** read as "sigma star"
# """
# function σˣ(x::Vector{T}, ::Type{UBCM{T}}) where T
#     n = length(x)
#     res = zeros(T, n, n)
#     for i = 1:n
#         @simd for j = i+1:n
#             @inbounds xij =  x[i]*x[j]
#             @inbounds res[i,j] = sqrt(xij)/(1 + xij)
#             @inbounds res[j,i] = sqrt(xij)/(1 + xij)
#         end
#     end

#     return res
# end

# """
#     rand(m::UBCM)

# Generate a random graph from the UBCM model. The function returns a `Graphs.AbstractGraph` object.
# """
# function Base.rand(m::UBCM)
#     n = length(m)
#     g = Graphs.SimpleGraph(n)
#     for i = 1:n
#         for j = i+1:n
#             if rand() < m.G[i,j]
#                 Graphs.add_edge!(g, i, j)
#             end
#         end
#     end

#     return g
# end


# """
#     σˣ(X::Function, M::UBCM{T})

# Compute the standard deviation of a property `X` with respect to the expected adjacency matrix associated with the UBCM model `M`.
# """
# σˣ(X::Function, M::UBCM{T}) where T = sqrt( sum((M.σ .* ∇X(X, M)) .^ 2) )



# # ----------------------------------------------------------------------------------------------------------------------
# #
# #                                               DBCM model
# #
# # ----------------------------------------------------------------------------------------------------------------------

# """
#     DBCM

# Maximum entropy model for the Directed Binary Configuration Model (DBCM). 
    
# The object holds the maximum likelihood parameters of the model (x, y), the expected adjacency matrix (G), 
# and the variance for the elements of the adjacency matrix (σ).

# """
# struct DBCM{T} <: AbstractMaxEntropyModel where {T<:Real}
#     x::Vector{T}
#     y::Vector{T}
#     G::Matrix{T}
#     σ::Matrix{T}
# end

# Base.show(io::IO, m::DBCM{T}) where T = print(io, "$(T) DBCM model ($(length(m)) vertices)")

# """Return the number of nodes in the DBCM network"""
# Base.length(m::DBCM) = length(m.x)

# """
#     DBCM(x::Vector{T}, y::Vector{T}; compute::Bool=true) where {T<:Real}

# Constructor for the `DBCM` type. If `compute` is true, the expected adjacency matrix and variance are computed. 
# Otherwise the memory is allocated but not initialized. (TBC)
# """
# function DBCM(x::Vector{T}, y::Vector{T}; compute::Bool=true) where {T<:Real}
#     G = Ĝ( x, y, DBCM{T}) # expected adjacency matrix
#     σ = σˣ(x, y, DBCM{T}) # expected standard deviation matrix

#     return DBCM(x, y, G, σ)
# end

# """
#     DBCM(G::T) where T<:SimpleDiGraph

# Constructor for the `DBCM` type based on a `SimpleDiGraph`. 
# """
# function DBCM(G::T; method="fixed-point", initial_guess="degrees", max_steps=5000, tol=1e-12, kwargs...) where T<:Graphs.SimpleDiGraph
#     NP = PyCall.pyimport("NEMtropy")
#     G_nem =  NP.DirectedGraph(degree_sequence=vcat(Graphs.outdegree(G), Graphs.indegree(G)))
#     G_nem.solve_tool(model="dcm_exp"; method=method, initial_guess=initial_guess, max_steps=max_steps, tol=tol, kwargs...);
#     if G_nem.error > 1e-7
#         @warn "The model did not converge, maybe try some other options (solution error $(G_nem.error))"
#     end
#     return DBCM(G_nem.x, G_nem.y)
# end

# """
#     Ĝ(x::Vector{T}, y::Vector{T}, ::Type{DBCM{T}}) where {T<:Real}

# Compute the expected adjacency matrix for the `DBCM` model with maximum likelihood parameters `x` and `y`.
# """
# function Ĝ(x::Vector{T}, y::Vector{T}, ::Type{DBCM{T}}) where T
#     n = length(x)
#     G = zeros(T, n, n)
#     for i = 1:n
#         @simd for j = i+1:n
#             @inbounds xiyj = x[i]*y[j]
#             @inbounds xjyi = x[j]*y[i]
#             @inbounds G[i,j] = xiyj/(1 + xiyj)
#             @inbounds G[j,i] = xjyi/(1 + xjyi)
#         end
#     end
    
#     return G
# end

# """
#     σˣ(x::Vector{T}, y::Vector{T}, ::Type{DBCM{T}}) where T

# Compute the standard deviation for the elements of the adjacency matrix for the `DBCM` model using the maximum likelihood parameters `x` and `y`.

# **Note:** read as "sigma star"
# """
# function σˣ(x::Vector{T}, y::Vector{T}, ::Type{DBCM{T}}) where T
#     n = length(x)
#     res = zeros(T, n, n)
#     for i = 1:n
#         @simd for j = i+1:n
#             @inbounds xiyj =  x[i]*y[j]
#             @inbounds xjyi =  x[j]*y[i]
#             @inbounds res[i,j] = sqrt(xiyj)/(1 + xiyj)
#             @inbounds res[j,i] = sqrt(xjyi)/(1 + xjyi)
#         end
#     end

#     return res
# end

# """
#     rand(m::DBCM)

# Generate a random graph from the `DBCM` model. The function returns a `Graphs.AbstractGraph` object.
# """
# function Base.rand(m::DBCM)
#     n = length(m)
#     g = Graphs.SimpleDiGraph(n)
#     for i = 1:n
#         for j = i+1:n
#             if rand() < m.G[i,j]
#                 Graphs.add_edge!(g, i, j)
#             end
#             if rand() < m.G[j,i]
#                 Graphs.add_edge!(g, j, i)
#             end
#         end
#     end

#     return g
# end

# """
#     σˣ(X::Function, M::DBCM{T})

# Compute the standard deviation of a property `X` with respect to the expected adjacency matrix associated with the `DBCM` model `M`.
# """
# σˣ(X::Function, M::DBCM{T}) where T = sqrt( sum((M.σ .* ∇X(X, M)) .^ 2) )

