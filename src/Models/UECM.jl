
"""
    UECM

Maximum entropy model for the Undirected Enhanced Configuration Model (UECM). 
    
The object holds the maximum likelihood parameters of the model (θ = [α;β]), the expected adjacency matrix (G), 
and the variance for the elements of the adjacency matrix (σ).


Note: this requires that the weights only assume (non-negative) integer values.   
"""
mutable struct UECM{T,N} <: AbstractMaxEntropyModel where {T<:Union{Graphs.AbstractGraph, Nothing}, N<:Real}
    "Graph type, can be any subtype of AbstractGraph, but will be converted to SimpleWeightedGraph for the computation" # can also be empty
    const G::T 
    "Maximum likelihood parameters for reduced model"
    const θᵣ::Vector{N}
    "Exponentiated maximum likelihood parameters for reduced model ( xᵢ = exp(-αᵢ) )"
    const xᵣ::Vector{N}
    "Exponentiated maximum likelihood parameters for reduced model ( yᵢ = exp(-βᵢ) )"
    const yᵣ::Vector{N}
    "Degree sequence of the graph" # evaluate usefulness of this field later on
    const d::Vector{Int}
    "Reduced degree sequence of the graph"
    const dᵣ::Vector{Int}
    "Strength sequence of the graph"
    const s::Vector{Int}
    "Reduced strength sequence of the graph"
    const sᵣ::Vector{Int}
    "Frequency of each (degree, strength) pair in the graph"
    const f::Vector{Int}
    "Indices to reconstruct the degree sequence from the reduced degree sequence"
    const d_ind::Vector{Int}
    "Indices to reconstruct the reduced degree sequence from the degree sequence"
    const dᵣ_ind::Vector{Int}
    "Non-zero constraints"
    const nz::Vector{Int}
    "Expected adjacency matrix" # not always computed/required
    Ĝ::Union{Nothing, Matrix{N}}
    "Variance of the expected adjacency matrix" # not always computed/required
    σ::Union{Nothing, Matrix{N}}
    "Status indicators: parameters computed, expected adjacency matrix computed, variance computed, etc."
    const status::Dict{Symbol, Any}
    "Function used to computed the log-likelihood of the (reduced) model"
    fun::Union{Nothing, Function}
end

Base.show(io::IO, m::UECM{T,N}) where {T,N} = print(io, """UECM{$(T), $(N)} ($(m.status[:N]) vertices, $(m.status[:d_unique]) unique {degree,strength} pairs, $(@sprintf("%.2f", m.status[:cᵣ])) compression ratio)""")

"""Return the reduced number of nodes in the UBCM network"""
Base.length(m::UECM) = length(m.dᵣ)



"""
    UECM(G::T; d::Vector, s::Vector, precision::N=Float64, kwargs...) where {T<:Graphs.AbstractGraph, N<:Real}

Constructor function for the `UECM` type. 
    
By default and dependng on the graph type `T`, the definition of degree and strength from ``Graphs.jl`` and ``SimpleWeigthedGraphs`` is applied. 
If you want to use a different definition of degree or strength, you can pass the vectors as keyword arguments.

If you want to generate a model directly from a degree and strength sequence without an underlying graph  you can simply pass them as keyword arguments.
If you want to work from an adjacency matrix, or edge list, you can use the graph constructors from the ``JuliaGraphs`` ecosystem.

# Examples     
```jldoctest
# generating a model from a graph


# generating a model directly from a degree sequence


# generating a model directly from a degree sequence with a different precision


# generating a model from an adjacency matrix


# generating a model from an edge list

```

See also [`Graphs.degree`](@ref), [`SimpleWeightedGraphs.strength`](@ref).
"""
function UECM(G::T; d::Vector=Graphs.degree(G), s::Vector=strength(G), precision::Type{<:AbstractFloat}=Float64, kwargs...) where {T}
    T <: Union{Graphs.AbstractGraph, Nothing} ? nothing : throw(TypeError("G must be a subtype of AbstractGraph or Nothing"))
    # coherence checks
    if T <: Graphs.AbstractGraph # Graph specific checks
        if Graphs.is_directed(G)
            @warn "The graph is directed, the UECM model is undirected, the directional information will be lost"
        end

        if zero(eltype(d)) ∈ d
            @warn "The graph has vertices with zero degree, this may lead to convergence issues."
        end

        if zero(eltype(s)) ∈ s
            @warn "The graph has vertices with zero strength, this may lead to convergence issues."
        end

        Graphs.nv(G) == 0 ? throw(ArgumentError("The graph is empty")) : nothing
        Graphs.nv(G) == 1 ? throw(ArgumentError("The graph has only one vertex")) : nothing

        Graphs.nv(G) != length(d) ? throw(DimensionMismatch("The number of vertices in the graph ($(Graphs.nv(G))) and the length of the degree sequence ($(length(d))) do not match")) : nothing
        Graphs.nv(G) != length(s) ? throw(DimensionMismatch("The number of vertices in the graph ($(Graphs.nv(G))) and the length of the strength sequence ($(length(s))) do not match")) : nothing
        length(d) != length(s) ? throw(DimensionMismatch("The dimensions of the degree ($(length(s))) and the strength sequence ($(length(s))) do not match")) : nothing
    end

    # coherence checks specific to the degree sequence
    any(!isinteger, d) ? throw(DomainError("Some of the degree values are not integers, this is not allowed")) : nothing
    any(!isinteger, s) ? throw(DomainError("Some of the strength values are not integers, this is not allowed")) : nothing
    length(d) == 0 ? throw(ArgumentError("The degree sequence is empty")) : nothing
    length(d) == 1 ? throw(ArgumentError("The degree sequence has only one degree")) : nothing
    maximum(d) >= length(d) ? throw(DomainError("The maximum degree in the graph is greater or equal to the number of vertices, this is not allowed")) : nothing

    # field generation
    dsᵣ, d_ind , dᵣ_ind, f = np_unique_clone(collect(zip(d, s)), sorted=true)
    dᵣ = Int.([d[1] for d in dsᵣ])
    sᵣ = Int.([d[2] for d in dsᵣ])
    θᵣ = Vector{precision}(undef, 2*length(dᵣ)) 
    xᵣ = Vector{precision}(undef, length(dᵣ))
    yᵣ = Vector{precision}(undef, length(dᵣ))
    nz = findall(!iszero, dᵣ)
    status = Dict(  :params_computed=>false,        # are the parameters computed?
                    :G_computed=>false,             # is the expected adjacency matrix computed and stored?
                    :σ_computed=>false,             # is the standard deviation computed and stored?
                    :cᵣ => length(dᵣ)/length(d),    # compression ratio of the reduced model
                    :d_unique => length(dᵣ),        # number of unique degrees in the reduced model
                    :N => length(d)                 # number of vertices in the original graph 
                )
    
    return UECM{T,precision}(G, θᵣ, xᵣ, yᵣ, d, dᵣ, s, sᵣ, f, d_ind, dᵣ_ind, nz, nothing, nothing, status, nothing)
end

UECM(;d::Vector, s::Vector, precision::Type{<:AbstractFloat}=Float64, kwargs...) = UECM(nothing, d=d, s=s, precision=precision, kwargs...)



function L_UECM_reduced(θ::Vector, d::Vector, s::Vector, F::Vector, n::Int=length(d))
    # split the parameters
    α = @view θ[1:n]
    β = @view θ[n+1:end]
    # initiate
    res = zero(eltype(θ))
    # compute
    for i in eachindex(d)
        res -= F[i] *  (d[i] * α[i] + β[i] * s[i])
        for j in 1:i
            contrib = log_nan( 1 + exp(-α[i] - α[j]) * exp(-β[i] - β[j]) / (1 - exp(-β[i] - β[j])) ) # for optimisation we use lognan (cf utils.jl)
            if i == j
                res -=  F[i] * (F[j] - 1) * contrib * 0.5 # to avoid double counting
            else
                res -=  F[i] * F[j] * contrib
            end
        end
    end

    return res
end


function ∇L_UECM_reduced!(∇L::AbstractVector, θ::AbstractVector, d::Vector, s::Vector, F::Vector, x::AbstractVector, y::AbstractVector, n=length(θ)÷2)
    α = @view θ[1:n]
    β = @view θ[n+1:end]
    for i in eachindex(α) # to avoid the allocation of exp.(-θ)
        x[i] = exp(-α[i])
        y[i] = exp(-β[i])
    end

    # reset gradient
    ∇L .= zero(eltype(θ))
    # compute gradient
    for i in eachindex(α)
        # ∂L/∂αᵢ
        ∇L[i]   -= F[i] * d[i]
        # ∂L/∂βᵢ
        ∇L[i+n] -= F[i] * s[i]
        for j in eachindex(α)
            c1 = x[i] * x[j]
            c2 = y[i] * y[j]
            if i == j
                # ∂L/∂αᵢ
                ∇L[i]     += (c1 * c2) / (1 + c1 * c2 - c2)            * F[i] * (F[j] - 1)
                # ∂L/∂βᵢ
                ∇L[i + n] += (c1 * c2) / ((1- c2) * (1 +c1 * c2 - c2)) * F[i] * (F[j] - 1) 
            else
                # ∂L/∂αᵢ
                ∇L[i]     += (c1 * c2) / (1 + c1 * c2 - c2)            * F[i] * F[j]
                # ∂L/∂βᵢ
                ∇L[i + n] += (c1 * c2) / ((1- c2) * (1 +c1 * c2 - c2)) * F[i] * F[j]
            end
        end
    end

    return ∇L
end


function ∇L_UECM_reduced_minus!(∇L::AbstractVector, θ::AbstractVector, d::Vector, s::Vector, F::Vector, x::AbstractVector, y::AbstractVector, n=length(θ)÷2)
    α = @view θ[1:n]
    β = @view θ[n+1:end]
    for i in eachindex(α) # to avoid the allocation of exp.(-θ)
        x[i] = exp(-α[i])
        y[i] = exp(-β[i])
    end

    # reset gradient
    ∇L .= zero(eltype(θ))
    # compute gradient
    for i in eachindex(α)
        # ∂L/∂αᵢ
        ∇L[i]   += F[i] * d[i]
        # ∂L/∂βᵢ
        ∇L[i+n] += F[i] * s[i]
        for j in eachindex(α)
            c1 = x[i] * x[j]
            c2 = y[i] * y[j]
            if i == j
                # ∂L/∂αᵢ
                ∇L[i]     -= (c1 * c2) / (1 + c1 * c2 - c2)            * F[i] * (F[j] - 1)
                # ∂L/∂βᵢ
                ∇L[i + n] -= (c1 * c2) / ((1- c2) * (1 +c1 * c2 - c2)) * F[i] * (F[j] - 1) 
            else
                # ∂L/∂αᵢ
                ∇L[i]     -= (c1 * c2) / (1 + c1 * c2 - c2)            * F[i] * F[j]
                # ∂L/∂βᵢ
                ∇L[i + n] -= (c1 * c2) / ((1- c2) * (1 +c1 * c2 - c2)) * F[i] * F[j]
            end
        end
    end

    return ∇L
end


function UECM_reduced_iter!(θ::AbstractVector, d::Vector, s::Vector, F::Vector, x::AbstractVector, y::AbstractVector, G::AbstractVector, nz::Vector, n=length(θ)÷2)
    α = @view θ[1:n]
    β = @view θ[n+1:end]
    for i in eachindex(α) # to avoid the allocation of exp.(-θ)
        x[i] = exp(-α[i])
        y[i] = exp(-β[i])
    end
    
    # compute
    for i in nz
        # reset buffer
        G[i]   = zero(eltype(θ))
        G[i+n] = zero(eltype(θ))
        for j in nz
            c1 = x[i] * x[j]
            c2 = y[i] * y[j]
            if i == j
                G[i]   += (x[j] * c2) / (1 + c1 * c2 - c2) * F[i]     * (F[j] - 1)
                G[i+n] += (c1 * y[j]) / ((1- c2) * (1 +c1 * c2 - c2)) * F[i] * (F[j] - 1) 
            else
                G[i]   += (x[j] * c2) / (1 + c1 * c2 - c2) * F[i] * F[j]
                G[i+n] += (c1 * y[j]) / ((1- c2) * (1 +c1 * c2 - c2)) * F[i] * F[j] 
            end
        end
    end

    for i in nz
        G[i]   =  - log_nan(F[i] * d[i] / G[i])
        G[i+n] =  - log_nan(F[i] * s[i] / G[i+n])
    end

    for i in eachindex(G)
        if iszero(G[i])
            G[i] = 1e8
        end
    end
    
    return G
end


function initial_guess(m::UECM{T,N}; method::Symbol=:strengths) where {T,N}
    if isequal(method, :strengths )
        isnothing(m.G) ? throw(ArgumentError("Cannot compute the number of edges because the model has no underlying graph (m.G == nothing)")) : nothing
        res = Vector{N}(-log.(vcat( m.dᵣ ./ (Graphs.ne(m.G) + 1), 
                                    m.sᵣ ./ (sum(m.sᵣ) + 1)))) # will return Inf initial guesses for zero values 
    elseif isequal(method, :strengths_minor)
        isnothing(m.G) ? throw(ArgumentError("Cannot compute the number of edges because the model has no underlying graph (m.G == nothing)")) : nothing
        res = Vector{N}(-log.(vcat( ones(N, length(m.dᵣ)) ./ (m.dᵣ .+ 1),
                                    ones(N, length(m.sᵣ)) ./ (m.sᵣ .+ 1))))
    elseif isequal(method, :random)
        res =  Vector{N}(-log.(rand(N, 2*length(m.dᵣ))))
    elseif isequal(method, :uniform)
        res = Vector{N}(-log.(0.001 .* ones(N, 2*length(m.dᵣ))))
    else
        throw(ArgumentError("The initial guess method $(method) is not supported"))
    end

    # replace Inf values with 1e8
    #res[findall(isinf, res)] .= 1e8 
    return res
end


"""
    set_xᵣ!(m::UECM)

Set the value of xᵣ to exp(-αᵣ) for the UECM model `m`
"""
function set_xᵣ!(m::UECM)
    if m.status[:params_computed]
        αᵣ = @view m.θᵣ[1:m.status[:d_unique]]
        m.xᵣ .= exp.(-αᵣ)
    else
        throw(ArgumentError("The parameters have not been computed yet"))
    end
end

"""
    set_yᵣ!(m::UECM)

Set the value of yᵣ to exp(-βᵣ) for the UECM model `m`
"""
function set_yᵣ!(m::UECM)
    if m.status[:params_computed]
        βᵣ = @view m.θᵣ[m.status[:d_unique]+1:end]
        m.yᵣ .= exp.(-βᵣ)
    else
        throw(ArgumentError("The parameters have not been computed yet"))
    end
end


"""
    Ĝ(m::UECM)

Compute the expected **adjacency** matrix for the UECM model `m`. 

Note: The expected weights need to be computed separately.
"""
function Ĝ(m::UECM{T,N}) where {T,N}
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    
    # get network size => this is the full size
    n = m.status[:N] 
    # initiate G
    G = zeros(N, n, n)
    # initiate x and y
    x = m.xᵣ[m.dᵣ_ind]
    y = m.yᵣ[m.dᵣ_ind]
    # compute G
    for i = 1:n
        @simd for j = 1:n
            if i≠j
                @inbounds xixj = x[i]*x[j]
                @inbounds yiyj = y[i]*y[j]
                @inbounds G[i,j] = (xixj * yiyj) / (1 - yiyj + xixj * yiyj)
            end
        end
    end

    return G    
end


"""
    Ŵ(m::UECM)

Compute the expected ** weigthed adjacency** matrix for the UECM model `m`. 
"""
function Ŵ(m::UECM{T,N}) where {T,N}
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    
    # get network size => this is the full size
    n = m.status[:N] 
    # initiate W
    W = zeros(N, n, n)
    # initiate x and y
    #x = m.xᵣ[m.dᵣ_ind]
    y = m.yᵣ[m.dᵣ_ind]
    # compute G
    for i = 1:n
        @simd for j = 1:n
            if i≠j
                @inbounds W[i,j] = 1 / (1 - y[i]*y[j])
            end
        end
    end

    return W    
end


"""
    set_σ!(m::UECM)

Set the standard deviation for the elements of the adjacency matrix for the UECM model `m`
"""
function set_σ!(m::UECM)
    throw(MethodError("This function is not implemented yet for UECM models"))
end

"""
    σˣ(m::UECM{T,N}) where {T,N}

Compute the standard deviation for the elements of the adjacency matrix for the UECM model `m`.

**Note:** read as "sigma star"
"""
function σˣ(m::UECM{T,N}) where {T,N}
    throw(MethodError("This function is not implemented yet for UECM models"))
end


"""
    rand(m::UECM; precomputed=false)

Generate a random graph from the UECM model `m`.

Keyword arguments:
- `precomputed::Bool`: if `true`, the precomputed expected adjacency matrix (`m.Ĝ`) is used to generate the random graph, otherwise the maximum likelihood parameters are used to generate the random graph on the fly. For larger networks, it is 
  recommended to not precompute the expected adjacency matrix to limit memory pressure.

# Examples
```jldoctest
# generate a UECM model from the karate club network
julia> G = MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate);
julia> model = MaxEntropyGraphs.UECM(G);
# compute the maximum likelihood parameters
using NLsolve
x_buffer = zeros(length(model.dᵣ));G_buffer = zeros(length(model.dᵣ));
FP_model! = (θ::Vector) -> MaxEntropyGraphs.UECM(θ, model.dᵣ, model.f, x_buffer, G_buffer);
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
function rand(m::UECM; precomputed::Bool=false)
    if precomputed
        throw(ArgumentError("This function is not implemented yet for UECM models"))
        # check if possible to use precomputed Ĝ
        #m.status[:G_computed] ? nothing : throw(ArgumentError("The expected adjacency matrix has not been computed yet"))
        # generate random graph
        #G = Graphs.SimpleGraphFromIterator(Graphs.Edge.([(i,j) for i = 1:m.status[:d] for j in i+1:m.status[:d] if rand()<m.Ĝ[i,j]]))
    else
        # check if possible to use parameters
        m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
        # generate x vector
        x = m.xᵣ[m.dᵣ_ind]
        y = m.yᵣ[m.dᵣ_ind]
        # generate random graph edges
        sources = Vector{Int}(); 
        targets = Vector{Int}();
        weights = Vector{Int}();
        for i in 1:m.status[:N]
            for j in 1:i-1
                # check if edge exists
                @inbounds xixj = x[i]*x[j]
                @inbounds yiyj = y[i]*y[j]
                p_ij = (xixj * yiyj) / (1 - yiyj + xixj * yiyj)
                if rand() ≤ p_ij
                    push!(sources, i)
                    push!(targets, j)
                    push!(weights, rand(Distributions.Geometric(1-yiyj)+1))
                end
            end
        end

        if length(sources) ≠ 0
            G = SimpleWeightedGraphs.SimpleWeightedGraph(sources, targets, weights)
        else
            G = SimpleWeightedGraphs.SimpleWeightedGraph(m.status[:N])
        end
        
        
        # deal with edge case where no edges are generated for the last node(s) in the graph
        while Graphs.nv(G) < m.status[:N]
            Graphs.add_vertex!(G)
        end
        return G
    end
end

function rand(m::UECM, n::Int; precomputed::Bool=false)
    # pre-allocate
    res = Vector{SimpleWeightedGraphs.SimpleWeightedGraph}(undef, n)
    # fill vector using threads
    Threads.@threads for i in 1:n
        res[i] = rand(m; precomputed=precomputed)
    end

    return res
end



"""
    solve_model!(m::UECM)

Compute the likelihood maximising parameters of the UECM model `m`. 

By default the parameters are computed using the BFGS method with Zygote.jl using the strength sequence as initial guess.

**Notes**
- the fixed-point method is very unstable for this model and should not be used. From an acceptable solution it can be used to fine-tune an existing one
- the L-BFGS method is known to be unstable for this model and should not be used.
"""
function solve_model!(m::UECM{T,N};  # common settings
                                method::Symbol=:BFGS, 
                                initial::Symbol=:strengths,
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
    # find Inf values
    ind_inf = findall(isinf, θ₀)
    if method==:fixedpoint
        @warn "The fixed point method is very unstable for this model and should not be used. `BFGS` is prefered for quasinewton methods."
        # initiate buffers
        x_buffer = zeros(N, length(m.dᵣ)); # buffer for x = exp(-α)
        y_buffer = zeros(N, length(m.sᵣ));  # buffer for y = exp(-β)
        G_buffer = zeros(N, length(m.θᵣ)); # buffer for G(x)
        # define fixed point function
        FP_model! = (θ::Vector) -> UECM_reduced_iter!(θ, m.dᵣ, m.sᵣ, m.f, x_buffer, y_buffer, G_buffer, m.nz, length(m.dᵣ));
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
            x_buffer = zeros(N, length(m.dᵣ)); # buffer for x = exp(-α)
            y_buffer = zeros(N, length(m.sᵣ));  # buffer for y = exp(-β)
            # define gradient function for optimisation.jl
            grad! = (G, θ, p) -> MaxEntropyGraphs.∇L_UECM_reduced_minus!(G, θ, m.dᵣ, m.sᵣ, m.f, x_buffer, y_buffer, length(m.dᵣ));
        end
        # define objective function and its AD method
        f = AD_method ∈ keys(AD_methods) ? Optimization.OptimizationFunction( (θ, p) -> - L_UECM_reduced(θ, m.dᵣ, m.sᵣ, m.f, length(m.dᵣ)),
                                                                                        AD_methods[AD_method],
                                                                                        grad = analytical_gradient ? grad! : nothing)                      : throw(ArgumentError("The AD method $(AD_method) is not supported (yet)"))
        
        prob = Optimization.OptimizationProblem(f, θ₀);

        # obtain solution
        sol = method ∈ keys(optimization_methods)   ? Optimization.solve(prob, optimization_methods[method], abstol=abstol, reltol=reltol)                                                : throw(ArgumentError("The method $(method) is not supported (yet)"))
        # check convergence
        if Optimization.SciMLBase.successful_retcode(sol.retcode)
            if verbose 
                @info """$(method) optimisation converged after $(@sprintf("%1.2e", sol.solve_time)) seconds (Optimization.jl return code: $("$(sol.retcode)"))\n$(sol.original)"""
            end
            m.θᵣ .= sol.u;
            m.θᵣ[ind_inf] .= N(Inf);
            m.status[:params_computed] = true;
            set_xᵣ!(m);
            set_yᵣ!(m);
        else
            throw(ConvergenceError(method, sol.retcode))
        end
    end

    return m,sol
end
