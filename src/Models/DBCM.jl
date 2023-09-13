
"""
    DBCM

Maximum entropy model for the Directed Binary Configuration Model (UBCM).

The object holds the maximum likelihood parameters of the model (θ) and optionally the expected adjacency matrix (G), 
and the variance for the elements of the adjacency matrix (σ). All settings and other metadata are stored in the `status` field.
"""
mutable struct DBCM{T,N} <: AbstractMaxEntropyModel where {T<:Union{Graphs.AbstractGraph, Nothing}, N<:Real}
    "Graph type, can be any subtype of AbstractGraph, but will be converted to SimpleDiGraph for the computation" # can also be empty
    const G::T 
    "Vector holding all maximum likelihood parameters for reduced model (α ; β)"
    const θᵣ::Vector{N}
    "Exponentiated maximum likelihood parameters for reduced model ( xᵢ = exp(-αᵢ) ) linked with out-degree"
    const xᵣ::Vector{N}
    "Exponentiated maximum likelihood parameters for reduced model ( yᵢ = exp(-βᵢ) ) linked with in-degree"
    const yᵣ::Vector{N}
    "Outdegree sequence of the graph" # evaluate usefulness of this field later on
    const d_out::Vector{Int}
    "Indegree sequence of the graph" # evaluate usefulness of this field later on
    const d_in::Vector{Int}
    "Reduced outdegree sequence of the graph"
    const dᵣ_out::Vector{Int}
    "Reduced indegree sequence of the graph"
    const dᵣ_in::Vector{Int}
    "Indices of non-zero elements in the reduced outdegree sequence"
    const dᵣ_out_nz::Vector{Int}
    "Indices of non-zero elements in the reduced indegree sequence"
    const dᵣ_in_nz::Vector{Int}
    "Frequency of each (outdegree, indegree) pair in the graph"
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
    const status::Dict{Symbol, Real}
    "Function used to computed the log-likelihood of the (reduced) model"
    fun::Union{Nothing, Function}
end


Base.show(io::IO, m::DBCM{T,N}) where {T,N} = print(io, """DBCM{$(T), $(N)} ($(m.status[:d]) vertices, $(m.status[:d_unique]) unique degree pairs, $(@sprintf("%.2f", m.status[:cᵣ])) compression ratio)""")

"""Return the reduced number of nodes in the UBCM network"""
Base.length(m::DBCM) = length(m.dᵣ)


"""
    DBCM(G::T; precision::N=Float64, kwargs...) where {T<:Graphs.AbstractGraph, N<:Real}
    DBCM(;d_out::Vector{T}, d_in::Vector{T}, precision::Type{<:AbstractFloat}=Float64, kwargs...)

Constructor function for the `DBCM` type. 
    
By default and dependng on the graph type `T`, the definition of in- and outdegree from ``Graphs.jl`` is applied. 
If you want to use a different definition of degrees, you can pass vectors of degrees sequences as keyword arguments (`d_out`, `d_in`).
If you want to generate a model directly from degree sequences without an underlying graph, you can simply pass the degree sequences as arguments (`d_out`, `d_in`).
If you want to work from an adjacency matrix, or edge list, you can use the graph constructors from the ``JuliaGraphs`` ecosystem.

# Examples     
```jldoctest DBCM_creation
# generating a model from a graph
julia> G = MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques())
{16, 111} directed simple Int64 graph
julia> model = DBCM(G)
DBCM{Graphs.SimpleGraphs.SimpleDiGraph{Int64}, Float64} (16 vertices, 15 unique degree pairs, 0.94 compression ratio)
```
```jldoctest DBCM_creation
# generating a model directly from a degree sequence
julia> model = DBCM(d_out=MaxEntropyGraphs.Graphs.outdegree(G), d_in=MaxEntropyGraphs.Graphs.indegree(G))
DBCM{Nothing, Float64} (16 vertices, 15 unique degree pairs, 0.94 compression ratio)
```
```jldoctest DBCM_creation
# generating a model directly from a degree sequence with a different precision
julia>  model = DBCM(d_out=MaxEntropyGraphs.Graphs.outdegree(G), d_in=MaxEntropyGraphs.Graphs.indegree(G), precision=Float32)
DBCM{Nothing, Float32} (16 vertices, 15 unique degree pairs, 0.94 compression ratio)
```
```jldoctest DBCM_creation
# generating a model from an adjacency matrix
julia> A = [0 1 1;1 0 0;1 1 0];

julia> G = MaxEntropyGraphs.Graphs.SimpleDiGraph(A)
{3, 5} directed simple Int64 graph
julia> model = DBCM(G)
DBCM{Graphs.SimpleGraphs.SimpleDiGraph{Int64}, Float64} (3 vertices, 3 unique degree pairs, 1.00 compression ratio)
```
```jldoctest DBCM_creation
# generating a model from an edge list
julia> E = [(1,2),(1,3),(2,3)];

julia> edgelist = [MaxEntropyGraphs.Graphs.Edge(x,y) for (x,y) in E];

julia> G = MaxEntropyGraphs.Graphs.SimpleDiGraphFromIterator(edgelist)
{3, 3} directed simple Int64 graph
julia> model = DBCM(G)
DBCM{Graphs.SimpleGraphs.SimpleDiGraph{Int64}, Float64} (3 vertices, 3 unique degree pairs, 1.00 compression ratio)

```

See also [`Graphs.outdegree`](https://juliagraphs.org/Graphs.jl/stable/core_functions/core/#Graphs.outdegree-Tuple{AbstractGraph,%20Integer}), [`Graphs.indegree`](https://juliagraphs.org/Graphs.jl/stable/core_functions/core/#Graphs.indegree-Tuple{AbstractGraph,%20Integer}).
"""
function DBCM(G::T; d_out::Vector=Graphs.outdegree(G), 
                    d_in::Vector=Graphs.indegree(G), 
                    precision::Type{N}=Float64, 
                    kwargs...) where {T,N<:AbstractFloat}
    T <: Union{Graphs.AbstractGraph, Nothing} ? nothing : throw(TypeError("G must be a subtype of AbstractGraph or Nothing"))
    length(d_out) == length(d_in) ? nothing : throw(DimensionMismatch("The outdegree and indegree sequences must have the same length"))
    # coherence checks
    if T <: Graphs.AbstractGraph # Graph specific checks
        if !Graphs.is_directed(G)
            @warn "The graph is undirected, while the DBCM model is directed, the in- and out-degree will be the same"
        end

        if T <: SimpleWeightedGraphs.AbstractSimpleWeightedGraph
            @warn "The graph is weighted, while DBCM model is unweighted, the weight information will be lost"
        end

        Graphs.nv(G) == 0 ? throw(ArgumentError("The graph is empty")) : nothing
        Graphs.nv(G) == 1 ? throw(ArgumentError("The graph has only one vertex")) : nothing
        Graphs.nv(G) != length(d_out) ? throw(DimensionMismatch("The number of vertices in the graph ($(Graphs.nv(G))) and the length of the degree sequence ($(length(d))) do not match")) : nothing
    end
    # coherence checks specific to the degree sequences
    length(d_out) == 0 ? throw(ArgumentError("The degree sequences are empty")) : nothing
    length(d_out) == 1 ? throw(ArgumentError("The degree sequences only contain a single node")) : nothing
    maximum(d_out) >= length(d_out) ? throw(DomainError("The maximum outdegree in the graph is greater or equal to the number of vertices, this is not allowed")) : nothing
    maximum(d_in)  >= length(d_in)  ? throw(DomainError("The maximum indegree in the graph is greater or equal to the number of vertices, this is not allowed")) : nothing

    # field generation
    dᵣ, d_ind , dᵣ_ind, f = np_unique_clone(collect(zip(d_out, d_in)), sorted=true)
    dᵣ_out = [d[1] for d in dᵣ]
    dᵣ_in =  [d[2] for d in dᵣ]
    dᵣ_out_nz = findall(!iszero, dᵣ_out)
    dᵣ_in_nz  = findall(!iszero, dᵣ_in)
    Θᵣ = Vector{precision}(undef, 2*length(dᵣ))
    xᵣ = Vector{precision}(undef, length(dᵣ))
    yᵣ = Vector{precision}(undef, length(dᵣ))
    status = Dict{Symbol, Real}(:params_computed=>false,            # are the parameters computed?
                                :G_computed=>false,                 # is the expected adjacency matrix computed and stored?
                                :σ_computed=>false,                 # is the standard deviation computed and stored?
                                :cᵣ => length(dᵣ)/length(d_out),    # compression ratio of the reduced model
                                :d_unique => length(dᵣ),            # number of unique (outdegree, indegree) pairs in the reduced model
                                :d => length(d_out)                 # number of vertices in the original graph 
                )
    
    return DBCM{T,precision}(G, Θᵣ, xᵣ, yᵣ, d_out, d_in, dᵣ_out, dᵣ_in, dᵣ_out_nz, dᵣ_in_nz, f, d_ind, dᵣ_ind, nothing, nothing, status, nothing)
end

DBCM(; d_out::Vector{T}, d_in::Vector{T}, precision::Type{N}=Float64, kwargs...) where {T<:Signed, N<:AbstractFloat} = DBCM(nothing; d_out=d_out, d_in=d_in, precision=precision, kwargs...)


"""
    L_DBCM_reduced(θ::Vector, k_out::Vector, k_in::Vector, F::Vector, nz_out::Vector, nz_in::Vector, n::Int=length(k_out))

Compute the log-likelihood of the reduced DBCM model using the exponential formulation in order to maintain convexity.

# Arguments
    - `θ`: the maximum likelihood parameters of the model ([α; β])
    - `k_out`: the reduced outdegree sequence
    - `k_in`: the reduced indegree sequence
    - `F`: the frequency of each pair in the degree sequence
    - `nz_out`: the indices of non-zero elements in the reduced outdegree sequence
    - `nz_in`: the indices of non-zero elements in the reduced indegree sequence
    - `n`: the number of nodes in the reduced model

The function returns the log-likelihood of the reduced model. For the optimisation, this function will be used to
generate an anonymous function associated with a specific model.

# Examples
```jldoctest
# Generic use:
julia> k_out  = [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5];

julia> k_in   = [2, 3, 4, 1, 3, 5, 2, 4, 1, 2, 4, 0, 4];

julia> F      = [2, 2, 1, 1, 1, 2, 3, 1, 1, 2, 2, 1, 1];

julia> θ      = collect(range(0.1, step=0.1, length=length(k_out)));

julia> nz_out = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];

julia> nz_in  = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13];

julia> n      = length(k_out);

julia> L_DBCM_reduced(θ, k_out, k_in, F, nz_out, nz_in, n)
-200.48153981203262
```
```jldoctest
# Use with DBCM model:
julia> G = MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques());

julia> model = DBCM(G);

julia> model_fun = θ -> L_DBCM_reduced(θ, model.dᵣ_out, model.dᵣ_in, model.f, model.dᵣ_out_nz, model.dᵣ_in_nz, model.status[:d_unique]);

julia> model_fun(ones(size(model.θᵣ)))
-252.4627226503138
```
"""
function L_DBCM_reduced(θ::Vector, k_out::Vector, k_in::Vector, F::Vector, nz_out::Vector, nz_in::Vector, n::Int=length(k_out))
    α = @view θ[1:n]
    β = @view θ[n+1:end]
    res = zero(eltype(θ))
    for i ∈ nz_out
        @inbounds res -= F[i] * k_out[i] * α[i]
        for j ∈ nz_in
            if i ≠ j 
                @inbounds res -= F[i] * F[j]       * log(1 + exp(-α[i] - β[j]))
            else
                @inbounds res -= F[i] * (F[i] - 1) * log(1 + exp(-α[i] - β[j]))
            end
        end
    end

    for j ∈ nz_in
        @inbounds res -= F[j] * k_in[j]  * β[j]
    end

    return res
end


"""
    L_DBCM_reduced(m::DBCM)

Return the log-likelihood of the DBCM model `m` based on the computed maximum likelihood parameters.

# Examples
```jldoctest
# Use with DBCM model:
julia> G = MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques());

julia> model = DBCM(G);

julia> solve_model!(model);

julia> L_DBCM_reduced(model)
-120.15942408828172
```

See also [`L_DBCM_reduced(::Vector, ::Vector, ::Vector, ::Vector, ::Vector, ::Vector)`](@ref)
"""
function L_DBCM_reduced(m::DBCM)
    if m.status[:params_computed]
        return L_DBCM_reduced(m.θᵣ, m.dᵣ_out, m.dᵣ_in, m.f, m.dᵣ_out_nz, m.dᵣ_in_nz, m.status[:d_unique])
    else
        throw(ArgumentError("The maximum likelihood parameters have not been computed yet"))
    end
end


"""
    ∇L_DBCM_reduced!(∇L::AbstractVector, θ::AbstractVector, k_out::AbstractVector, k_in::AbstractVector, F::AbstractVector, nz_out::Vector, nz_in::Vector, x::AbstractVector, y::AbstractVector,n::Int)

Compute the gradient of the log-likelihood of the reduced DBCM model using the exponential formulation in order to maintain convexity.

For the optimisation, this function will be used togenerate an anonymous function associated with a specific model. The function 
will update pre-allocated vectors (`∇L`,`x` and `y`) for speed. The gradient is non-allocating.

# Arguments
- `∇L`: the gradient of the log-likelihood of the reduced model
- `θ`: the maximum likelihood parameters of the model ([α; β])
- `k_out`: the reduced outdegree sequence
- `k_in`: the reduced indegree sequence
- `F`: the frequency of each pair in the degree sequence
- `nz_out`: the indices of non-zero elements in the reduced outdegree sequence
- `nz_in`: the indices of non-zero elements in the reduced indegree sequence
- `x`: the exponentiated maximum likelihood parameters of the model ( xᵢ = exp(-αᵢ) )
- `y`: the exponentiated maximum likelihood parameters of the model ( yᵢ = exp(-βᵢ) )
- `n`: the number of nodes in the reduced model

# Examples
```jldoctest ∇L_DBCM_reduced
# Explicit use with DBCM model:
julia> G = MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques());

julia> model = DBCM(G);

julia> ∇L = zeros(Real, length(model.θᵣ));

julia> x  = zeros(Real, length(model.xᵣ));

julia> y  = zeros(Real, length(model.yᵣ));

julia> ∇model_fun! = θ -> ∇L_DBCM_reduced!(∇L, θ, model.dᵣ_out, model.dᵣ_in, model.f, model.dᵣ_out_nz, model.dᵣ_in_nz, x, y, model.status[:d_unique]);

julia> ∇model_fun!(model.θᵣ);

```
```jldoctest ∇L_DBCM_reduced
# Use within optimisation.jl framework:
julia> fun = (θ,p) -> -L_DBCM_reduced(θ, model.dᵣ_out, model.dᵣ_in, model.f, model.dᵣ_out_nz, model.dᵣ_in_nz, model.status[:d_unique]);

julia> x  = zeros(Real, length(model.xᵣ)); # initialise  buffer

julia> y  = zeros(Real, length(model.yᵣ));#  initialise  buffer

julia> ∇fun! = (∇L, θ, p) -> ∇L_DBCM_reduced!(∇L, θ, model.dᵣ_out, model.dᵣ_in, model.f, model.dᵣ_out_nz, model.dᵣ_in_nz, x, y, model.status[:d_unique]);

julia> θ₀ = initial_guess(model); # initial condition

julia> foo = MaxEntropyGraphs.Optimization.OptimizationFunction(fun, grad=∇fun!); # define target function 

julia> prob  = MaxEntropyGraphs.Optimization.OptimizationProblem(foo, θ₀); # define the optimisation problem

julia> method = MaxEntropyGraphs.OptimizationOptimJL.LBFGS(); # set the optimisation method

julia> MaxEntropyGraphs.Optimization.solve(prob, method); # solve it

```
"""
function ∇L_DBCM_reduced!(  ∇L::AbstractVector, θ::AbstractVector, 
                            k_out::AbstractVector, k_in::AbstractVector, 
                            F::AbstractVector, 
                            nz_out::Vector, nz_in::Vector,
                            x::AbstractVector, y::AbstractVector,
                            n::Int)
    # set pre-allocated values
    α = @view θ[1:n]
    β = @view θ[n+1:end]
    @simd for i in eachindex(α) # to obtain a non-allocating function <> x .= exp.(-α), y .= exp.(-β)
        @inbounds x[i] = exp(-α[i])
        @inbounds y[i] = exp(-β[i])
    end
    # reset gradient to zero
    ∇L .= zero(eltype(∇L))
    
    # part related to α
    @simd for i ∈ nz_out
        fx = zero(eltype(∇L))
        for j ∈ nz_in
            if i ≠ j
                @inbounds c = F[i] * F[j]
            else
                @inbounds c = F[i] * (F[j] - 1)
            end
            @inbounds fx += c * y[j] / (1 + x[i] * y[j])
        end
        @inbounds ∇L[i] = x[i] * fx - F[i] * k_out[i]
    end
    # part related to β
    @simd for j ∈ nz_in
        fy = zero(eltype(∇L))
        for i ∈ nz_out
            if i≠j
                @inbounds c = F[i] * F[j]
            else
                @inbounds c = F[i] * (F[j] - 1)
            end
            @inbounds fy += c * x[i] / (1 + x[i] * y[j])
        end
        @inbounds ∇L[n+j] = y[j] * fy - F[j] * k_in[j]
    end

    return ∇L
end


"""
    ∇L_DBCM_reduced_minus!(args...)

Compute minus the gradient of the log-likelihood of the reduced DBCM model using the exponential formulation in order to maintain convexity. Used for optimisation in a non-allocating manner.

See also [`∇L_DBCM_reduced!`](@ref)
"""
function ∇L_DBCM_reduced_minus!(∇L::AbstractVector, θ::AbstractVector,
                                k_out::AbstractVector, k_in::AbstractVector, 
                                F::AbstractVector, 
                                nz_out::Vector, nz_in::Vector,
                                x::AbstractVector, y::AbstractVector,
                                n::Int)
    # set pre-allocated values
    α = @view θ[1:n]
    β = @view θ[n+1:end]
    @simd for i in eachindex(α) # to obtain a non-allocating function <> x .= exp.(-α), y .= exp.(-β)
        @inbounds x[i] = exp(-α[i])
        @inbounds y[i] = exp(-β[i])
    end
    # reset gradient to zero
    ∇L .= zero(eltype(∇L))

    # part related to α
    @simd for i ∈ nz_out
        fx = zero(eltype(∇L))
        for j ∈ nz_in
            if i ≠ j
                @inbounds c = F[i] * F[j]
            else
                @inbounds c = F[i] * (F[j] - 1)
            end
            @inbounds fx -= c * y[j] / (1 + x[i] * y[j])
        end
        @inbounds ∇L[i] = x[i] * fx + F[i] * k_out[i]
    end
    # part related to β
    @simd for j ∈ nz_in
        fy = zero(eltype(∇L))
        for i ∈ nz_out
            if i≠j
                @inbounds c = F[i] * F[j]
            else
                @inbounds c = F[i] * (F[j] - 1)
            end
            @inbounds fy -= c * x[i] / (1 + x[i] * y[j])
        end
        @inbounds ∇L[n+j] = y[j] * fy + F[j] * k_in[j]
    end

    return ∇L
end


"""
    DBCM_reduced_iter!(θ::AbstractVector, k_out::AbstractVector, k_in::AbstractVector, F::AbstractVector, nz_out::Vector, nz_in::Vector,x::AbstractVector, y::AbstractVector, G::AbstractVector, H::AbstractVector, n::Int)

Computer the next fixed-point iteration for the DBCM model using the exponential formulation in order to maintain convexity.
The function is non-allocating and will update pre-allocated vectors (`θ`, `x`, `y` and `G`) for speed.

# Arguments
- `θ`: the maximum likelihood parameters of the model ([α; β])
- `k_out`: the reduced outdegree sequence
- `k_in`: the reduced indegree sequence
- `F`: the frequency of each pair in the degree sequence
- `nz_out`: the indices of non-zero elements in the reduced outdegree sequence
- `nz_in`: the indices of non-zero elements in the reduced indegree sequence
- `x`: the exponentiated maximum likelihood parameters of the model ( xᵢ = exp(-αᵢ) )
- `y`: the exponentiated maximum likelihood parameters of the model ( yᵢ = exp(-βᵢ) )
- `G`: buffer for computations
- `n`: the number of nodes in the reduced model


# Examples
```jldoctest
# Use with DBCM model:
julia> G = MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques());

julia> model = DBCM(G);

julia> G = zeros(eltype(model.θᵣ), length(model.θᵣ));

julia> H = zeros(eltype(model.θᵣ), length(model.yᵣ));

julia> x = zeros(eltype(model.θᵣ), length(model.xᵣ));

julia> y = zeros(eltype(model.θᵣ), length(model.yᵣ));

julia> DBCM_FP! = θ -> DBCM_reduced_iter!(θ, model.dᵣ_out, model.dᵣ_in, model.f, model.dᵣ_out_nz, model.dᵣ_in_nz, x, y, G, model.status[:d_unique]);

julia> DBCM_FP!(model.θᵣ);

```
"""
function DBCM_reduced_iter!(θ::AbstractVector, 
                            k_out::AbstractVector, k_in::AbstractVector, 
                            F::AbstractVector, 
                            nz_out::Vector, nz_in::Vector,
                            x::AbstractVector, y::AbstractVector, 
                            G::AbstractVector,  n::Int) # H::AbstractVector,
    α = @view θ[1:n]
    β = @view θ[n+1:end]
    @simd for i in eachindex(α) # to obtain a non-allocating function <> x .= exp.(-α), y .= exp.(-β) (1.8μs, 6 allocs -> 1.2μs, 0 allocs)
        @inbounds x[i] = exp(-α[i])
        @inbounds y[i] = exp(-β[i])
    end
    G .= zero(eltype(G))
    #H .= zero(eltype(H))
    # part related to α
    @simd for i ∈ nz_out
        for j ∈ nz_in
            if i ≠ j
                @inbounds G[i] += F[j]        * y[j] / (1 + x[i] * y[j])
            else
                @inbounds G[i] += (F[j] - 1)  * y[j] / (1 + x[i] * y[j])
            end
        end
        @inbounds G[i] = -log(k_out[i] / G[i])
    end
    # part related to β
    @simd for j ∈ nz_in
        for i ∈ nz_out
            if i ≠ j
                @inbounds G[j+n] += F[i]        * x[i] / (1 + x[i] * y[j])
                #@inbounds H[j] += F[i]        * x[i] / (1 + x[i] * y[j])
            else
                @inbounds G[j+n] += (F[i] - 1)  * x[i] / (1 + x[i] * y[j])
                #@inbounds H[j] += (F[i] - 1)  * x[i] / (1 + x[i] * y[j])
            end
        end
        @inbounds G[n+j] = -log(k_in[j] / G[j+n])
        #@inbounds θ[n+j] = -log(k_in[j] / H[j])
    end

    return G
    #return θ
end



"""
    initial_guess(m::DBCM, method::Symbol=:degrees)

Compute an initial guess for the maximum likelihood parameters of the DBCM model `m` using the method `method`.

The methods available are: 
- `:degrees` (default): the initial guess is computed using the degrees of the graph, i.e. ``\\theta = [-\\log(d_{out}); -\\log(d_{in})]`` 
- `:degrees_minor`: the initial guess is computed using the degrees of the graph and the number of edges, i.e. ``\\theta = [-\\log(d_{out}/(\\sqrt{E} + 1)); -\\log(d_{in}//(\\sqrt{E} + 1) )]``
- `:random`: the initial guess is computed using random values between 0 and 1, i.e. ``\\theta_{i} = -\\log(r_{i})`` where ``r_{i} \\sim U(0,1)``
- `:uniform`: the initial guess is uniformily set to 0.5, i.e. ``\\theta_{i} = -\\log(0.5)``
- `:chung_lu`: the initial guess is computed using the degrees of the graph and the number of edges, i.e. ``\\theta = [-\\log(d_{out}/(2E)); -\\log(d_{in}/(2E))]``

# Examples
```jldoctest
julia> model = DBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()));

julia> initial_guess(model, method=:random);

julia> initial_guess(model, method=:uniform);

julia> initial_guess(model, method=:degrees_minor);

julia> initial_guess(model, method=:chung_lu);

julia> initial_guess(model);

```
"""
function initial_guess(m::DBCM; method::Symbol=:degrees)
    #N = typeof(m).parameters[2]
    if isequal(method, :degrees)
        return Vector{precision(m)}(vcat(-log.(m.dᵣ_out), -log.(m.dᵣ_in)))
    elseif isequal(method, :degrees_minor)
        isnothing(m.G) ? throw(ArgumentError("Cannot compute the number of edges because the model has no underlying graph (m.G == nothing)")) : nothing
        return Vector{precision(m)}(vcat(-log.(m.dᵣ_out ./ (sqrt(Graphs.ne(m.G)) + 1)), -log.(m.dᵣ_in ./ (sqrt(Graphs.ne(m.G)) + 1)) ))
    elseif isequal(method, :random)
        return Vector{precision(m)}(-log.(rand(precision(m), 2*length(m.dᵣ_out))))
    elseif isequal(method, :uniform)
        return Vector{precision(m)}(-log.(0.5 .* ones(precision(m), 2*length(m.dᵣ_out))))
    elseif isequal(method, :chung_lu)
        isnothing(m.G) ? throw(ArgumentError("Cannot compute the number of edges because the model has no underlying graph (m.G == nothing)")) : nothing
        return Vector{precision(m)}(vcat(-log.(m.dᵣ_out ./ (2 * Graphs.ne(m.G))), -log.(m.dᵣ_in ./ (2 * Graphs.ne(m.G)))))
    else
        throw(ArgumentError("The initial guess method $(method) is not supported"))
    end
end


"""
    set_xᵣ!(m::DBCM)

Set the value of xᵣ to exp(-αᵣ) for the DBCM model `m`
"""
function set_xᵣ!(m::DBCM)
    if m.status[:params_computed]
        αᵣ = @view m.θᵣ[1:m.status[:d_unique]]
        m.xᵣ .= exp.(-αᵣ)
    else
        throw(ArgumentError("The parameters have not been computed yet"))
    end
end

"""
    set_yᵣ!(m::DBCM)

Set the value of yᵣ to exp(-βᵣ) for the DBCM model `m`
"""
function set_yᵣ!(m::DBCM)
    if m.status[:params_computed]
        βᵣ = @view m.θᵣ[m.status[:d_unique]+1:end]
        m.yᵣ .= exp.(-βᵣ)
    else
        throw(ArgumentError("The parameters have not been computed yet"))
    end
end


"""
    Ĝ(m::DBCM)

Compute the expected adjacency matrix for the DBCM model `m`
"""
function Ĝ(m::DBCM)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    
    # get network size => this is the full size
    n = m.status[:d] 
    # initiate G
    G = zeros(precision(m), n, n)
    # initiate x and y
    x = m.xᵣ[m.dᵣ_ind]
    y = m.yᵣ[m.dᵣ_ind]
    # compute G
    for i = 1:n
        @simd for j = 1:n
            if i≠j
                @inbounds xiyj = x[i]*y[j]
                @inbounds G[i,j] = xiyj/(1 + xiyj)
            end
        end
    end

    return G    
end


"""
    set_Ĝ!(m::DBCM)

Set the expected adjacency matrix for the DBCM model `m`
"""
function set_Ĝ!(m::DBCM)
    m.Ĝ = Ĝ(m)
    m.status[:G_computed] = true
    return m.Ĝ
end


"""
    σˣ(m::DBCM{T,N}) where {T,N}

Compute the standard deviation for the elements of the adjacency matrix for the DBCM model `m`.

**Note:** read as "sigma star"
"""
function σˣ(m::DBCM)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    # check network size => this is the full size
    n = m.status[:d]
    # initiate G
    σ = zeros(precision(m), n, n)
    # initiate x and y
    x = m.xᵣ[m.dᵣ_ind]
    y = m.yᵣ[m.dᵣ_ind]
    # compute σ
    for i = 1:n
        @simd for j = i+1:n
            @inbounds xiyj =  x[i]*y[j]
            @inbounds xjyi =  x[j]*y[i]
            @inbounds res[i,j] = sqrt(xiyj)/(1 + xiyj)
            @inbounds res[j,i] = sqrt(xjyi)/(1 + xjyi)
        end
    end

    return σ
end

"""
    set_σ!(m::DBCM)

Set the standard deviation for the elements of the adjacency matrix for the DBCM model `m`
"""
function set_σ!(m::DBCM)
    m.σ = σˣ(m)
    m.status[:σ_computed] = true
    return m.σ
end


"""
    rand(m::DBCM; precomputed=false)

Generate a random graph from the DBCM model `m`.

# Arguments:
- `precomputed::Bool`: if `true`, the precomputed expected adjacency matrix (`m.Ĝ`) is used to generate the random graph, otherwise the maximum likelihood parameters are used to generate the random graph on the fly. For larger networks, it is 
  recommended to not precompute the expected adjacency matrix to limit memory pressure.

# Examples
```jldoctest
# generate a DBCM model macaques network
julia> G = MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques());

julia> model = DBCM(G); 

julia> solve_model!(model); # compute the maximum likelihood parameters

julia> typeof(rand(model))
Graphs.SimpleGraphs.SimpleDiGraph{Int64}

```
"""
function rand(m::DBCM; precomputed::Bool=false)
    if precomputed
        # check if possible to use precomputed Ĝ
        m.status[:G_computed] ? nothing : throw(ArgumentError("The expected adjacency matrix has not been computed yet"))
        # generate random graph
        #G = Graphs.SimpleGraphFromIterator(  Graphs.Edge.([(i,j) for i = 1:m.status[:d] for j in i+1:m.status[:d] if rand()<m.Ĝ[i,j]]))
        G = Graphs.SimpleDiGraphFromIterator( Graphs.Edge.([(i,j) for i = 1:m.status[:d] for j in 1:m.status[:d] if (rand()<m.Ĝ[i,j] && i≠j)  ]))
    else
        # check if possible to use parameters
        m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
        # initiate x and y
        x = m.xᵣ[m.dᵣ_ind]
        y = m.yᵣ[m.dᵣ_ind]
        # generate random graph
        # G = Graphs.SimpleGraphFromIterator(Graphs.Edge.([(i,j) for i = 1:m.status[:d] for j in i+1:m.status[:d] if rand()< (x[i]*x[j])/(1 + x[i]*x[j]) ]))
        G = Graphs.SimpleDiGraphFromIterator(Graphs.Edge.([(i,j) for i = 1:m.status[:d] for j in   1:m.status[:d] if (rand() < (x[i]*y[j])/(1 + x[i]*y[j]) && i≠j) ]))
    end

    # deal with edge case where no edges are generated for the last node(s) in the graph
    while Graphs.nv(G) < m.status[:d]
        Graphs.add_vertex!(G)
    end

    return G
end


"""
    rand(m::DBCM, n::Int; precomputed=false)

Generate `n` random graphs from the DBCM model `m`. If multithreading is available, the graphs are generated in parallel.

# Arguments
- `precomputed::Bool`: if `true`, the precomputed expected adjacency matrix (`m.Ĝ`) is used to generate the random graph, otherwise the maximum likelihood parameters are used to generate the random graph on the fly. For larger networks, it is 
  recommended to not precompute the expected adjacency matrix to limit memory pressure.

# Examples
# Examples
```jldoctest
# generate a DBCM model macaques network
julia> G = MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques());

julia> model = DBCM(G); 

julia> solve_model!(model); # compute the maximum likelihood parameters

julia> typeof(rand(model, 10))
Vector{SimpleDiGraph{Int64}} (alias for Array{Graphs.SimpleGraphs.SimpleDiGraph{Int64}, 1})

```
"""
function rand(m::DBCM, n::Int; precomputed::Bool=false)
    # pre-allocate
    res = Vector{Graphs.SimpleDiGraph{Int}}(undef, n)
    # fill vector using threads
    Threads.@threads for i in 1:n
        res[i] = rand(m; precomputed=precomputed)
    end

    return res
end



"""
    solve_model!(m::DBCM)

Compute the likelihood maximising parameters of the DBCM model `m`. 

# Arguments
- `method::Symbol`: solution method to use, can be `:fixedpoint` (default), or :$(join(keys(MaxEntropyGraphs.optimization_methods), ", :", " and :")).
- `initial::Symbol`: initial guess for the parameters ``\\Theta``, can be :degrees (default), :degrees_minor, :random, :uniform, or :chung_lu.
- `maxiters::Int`: maximum number of iterations for the solver (defaults to 1000). 
- `verbose::Bool`: set to show log messages (defaults to false).
- `ftol::Real`: function tolerance for convergence with the fixedpoint method (defaults to 1e-8).
- `abstol::Union{Number, Nothing}`: absolute function tolerance for convergence with the other methods (defaults to `nothing`).
- `reltol::Union{Number, Nothing}`: relative function tolerance for convergence with the other methods (defaults to `nothing`).
- `AD_method::Symbol`: autodiff method to use, can be any of :$(join(keys(MaxEntropyGraphs.AD_methods), ", :", " and :")). Performance depends on the size of the problem (defaults to `:AutoZygote`),
- `analytical_gradient::Bool`: set the use the analytical gradient instead of the one generated with autodiff (defaults to `false`)

# Examples
```jldoctest DBCM_solve
# default use
julia> model = DBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()));

julia> solve_model!(model);

```
```jldoctest DBCM_solve
# using analytical gradient and degrees minor initial guess
julia> solve_model!(model, method=:BFGS, analytical_gradient=true, initial=:degrees_minor)
(DBCM{Graphs.SimpleGraphs.SimpleDiGraph{Int64}, Float64} (16 vertices, 15 unique degree pairs, 0.94 compression ratio), retcode: Success
u: [3.118482950362848, 2.2567400402511617, 2.2467332710940333, 0.8596258292464105, 0.4957550197436504, 0.3427782029923598, 0.126564995232929, -0.3127732185244699, -0.3967757456352901, -0.43450987676209596  …  -0.5626916621021604, 1.223396713832784, 0.10977479732876981, -1.0367565290851806, -2.0427364999923148, -0.650376357149203, -1.5165614611776657, 0.7532475835319463, 0.39856890694767605, -0.6704522097652438]
Final objective value:     120.15942408828177
)

```
"""
function solve_model!(m::DBCM;  # common settings
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
                                analytical_gradient::Bool=false)
    N = precision(m)
    # initial guess
    θ₀ = initial_guess(m, method=initial)
    # find Inf values
    ind_inf = findall(isinf, θ₀)
    if method==:fixedpoint
        # initiate buffers
        x_buffer = zeros(N, length(m.dᵣ_out)); # buffer for x = exp(-α)
        y_buffer = zeros(N, length(m.dᵣ_in));  # buffer for y = exp(-β)
        G_buffer = zeros(N, length(m.θᵣ)); # buffer for G(x)
        # define fixed point function
        FP_model! = (θ::Vector) -> DBCM_reduced_iter!(θ, m.dᵣ_out, m.dᵣ_in, m.f, m.dᵣ_out_nz, m.dᵣ_in_nz, x_buffer, y_buffer, G_buffer, m.status[:d_unique])
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
            x_buffer = zeros(N, length(m.dᵣ_out)); # buffer for x = exp(-α)
            y_buffer = zeros(N, length(m.dᵣ_in));  # buffer for y = exp(-β)
            # initialise gradient buffer
            #gx_buffer = similar(θ₀)
            # define gradient function for optimisation.jl
            grad! = (G, θ, p) -> ∇L_DBCM_reduced_minus!(G, θ, m.dᵣ_out, m.dᵣ_in, m.f, m.dᵣ_out_nz, m.dᵣ_in_nz, x_buffer, y_buffer, m.status[:d_unique]);
        end
        # define objective function and its AD method
        f = AD_method ∈ keys(AD_methods)            ? Optimization.OptimizationFunction( (θ, p) ->   -L_DBCM_reduced(θ, m.dᵣ_out, m.dᵣ_in, m.f, m.dᵣ_out_nz, m.dᵣ_in_nz, m.status[:d_unique]),
                                                                                         AD_methods[AD_method],
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
            set_yᵣ!(m);
        else
            throw(ConvergenceError(method, sol.retcode))
        end
    end

    return m, sol
end


"""
    precision(m::DBCM)

Determine the compute precision of the UBCM model `m`.

# Examples
```jldoctest
julia> model = DBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()));

julia> MaxEntropyGraphs.precision(model)
Float64
```

```jldoctest
julia> model = DBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()), precision=Float32);

julia> MaxEntropyGraphs.precision(model)
Float32
```
"""
precision(m::DBCM) = typeof(m).parameters[2]


"""
    f_DBCM(x::T)

Helper function for the DBCM model to compute the expected value of the adjacency matrix. The function computes the expression `x / (1 + x)`.
As an argument you need to pass the product of the maximum likelihood parameters `xᵣ[i] * yᵣ[j]` from a DBCM model.
"""
f_DBCM(xiyj::T) where {T} = xiyj/(1 + xiyj)


"""
    A(m::DBCM,i::Int,j::Int)

Return the expected value of the adjacency matrix for the DBCM model `m` at the node pair `(i,j)`.

❗ For perfomance reasons, the function does not check:
- if the node pair is valid.
- if the parameters of the model have been computed.
"""
function A(m::DBCM,i::Int,j::Int)
    return i == j ? zero(precision(m)) : @inbounds f_DBCM(m.xᵣ[m.dᵣ_ind[i]] * m.yᵣ[m.dᵣ_ind[j]])
end


"""
    outdegree(m::DBCM, i::Int; method=:reduced)

Return the expected degree vector for node `i` of the DBCM model `m`.
Uses the reduced model parameters `xᵣ` for perfomance reasons.

# Arguments
- `m::DBCM`: the DBCM model
- `i::Int`: the node for which to compute the degree.
- `method::Symbol`: the method to use for computing the degree. Can be any of the following:
    - `:reduced` (default) uses the reduced model parameters `xᵣ` for perfomance reasons.
    - `:full` uses all elements of the expected adjacency matrix.
    - `:adjacency` uses the precomputed adjacency matrix `m.Ĝ` of the model.

# Examples
```jldoctest
julia> model = DBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()));

julia> solve_model!(model);

julia> set_Ĝ!(model);

julia> typeof([outdegree(model, 1), outdegree(model, 1, method=:full), outdegree(model, 1, method=:adjacency)])
Vector{Float64} (alias for Array{Float64, 1})

``` 
"""
function outdegree(m::DBCM, i::Int; method::Symbol=:reduced)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    i > m.status[:d] ? throw(ArgumentError("Attempted to access node $i in a $(m.status[:d]) node graph")) : nothing
    
    if method == :reduced
        res = zero(precision(m))
        i_red = m.dᵣ_ind[i] # find matching index in reduced model
        for j in eachindex(m.xᵣ)
            if i_red ≠ j 
                res += @inbounds f_DBCM(m.xᵣ[i_red] * m.yᵣ[j]) * m.f[j]
            else
                res += @inbounds f_DBCM(m.xᵣ[i_red] * m.yᵣ[i_red]) * (m.f[j] - 1) # subtract 1 because the diagonal is not counted
            end
        end
    elseif method == :full
        # using all elements of the adjacency matrix
        res = zero(precision(m))
        for j in eachindex(m.d_out)
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
    outdegree(m::DBCM[, v]; method=:reduced)

Return a vector corresponding to the expected outdegree of the DBCM model `m` each node. If v is specified, only return outdegrees for nodes in v.

# Arguments
- `m::DBCM`: the DBCM model
- `v::Vector{Int}`: the nodes for which to compute the outdegree. Default is all nodes.
- `method::Symbol`: the method to use for computing the outdegree. Can be any of the following:
    - `:reduced` (default) uses the reduced model parameters `xᵣ` for perfomance reasons.
    - `:full` uses all elements of the expected adjacency matrix.
    - `:adjacency` uses the precomputed adjacency matrix `m.Ĝ` of the model.

# Examples
```jldoctest
julia> model = DBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()));

julia> solve_model!(model);

julia> set_Ĝ!(model);

julia> typeof(outdegree(model, method=:adjacency)) 
Vector{Float64} (alias for Array{Float64, 1})

``` 
"""
outdegree(m::DBCM, v::Vector{Int}=collect(1:m.status[:d]); method::Symbol=:reduced) = [outdegree(m, i, method=method) for i in v]


"""
    indegree(m::DBCM, i::Int; method=:reduced)

Return the expected degree vector for node `i` of the DBCM model `m`.
Uses the reduced model parameters `xᵣ` for perfomance reasons.

# Arguments
- `m::DBCM`: the DBCM model
- `i::Int`: the node for which to compute the degree.
- `method::Symbol`: the method to use for computing the degree. Can be any of the following:
    - `:reduced` (default) uses the reduced model parameters `xᵣ` for perfomance reasons.
    - `:full` uses all elements of the expected adjacency matrix.
    - `:adjacency` uses the precomputed adjacency matrix `m.Ĝ` of the model.

# Examples
```jldoctest
julia> model = DBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()));

julia> solve_model!(model);

julia> set_Ĝ!(model);

julia> typeof([indegree(model, 1), indegree(model, 1, method=:full), indegree(model, 1, method=:adjacency)])
Vector{Float64} (alias for Array{Float64, 1})

``` 
"""
function indegree(m::DBCM, i::Int; method::Symbol=:reduced)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    i > m.status[:d] ? throw(ArgumentError("Attempted to access node $i in a $(m.status[:d]) node graph")) : nothing
    
    if method == :reduced
        res = zero(precision(m))
        i_red = m.dᵣ_ind[i] # find matching index in reduced model
        for j in eachindex(m.xᵣ)
            if i_red ≠ j 
                res += @inbounds f_DBCM(m.xᵣ[j] * m.yᵣ[i_red]) * m.f[j]
            else
                res += @inbounds f_DBCM(m.xᵣ[j] * m.yᵣ[i_red]) * (m.f[j] - 1) # subtract 1 because the diagonal is not counted
            end
        end
    elseif method == :full
        # using all elements of the adjacency matrix
        res = zero(precision(m))
        for j in eachindex(m.d_out)
            res += A(m, j, i)
        end
    elseif method == :adjacency
        #  using the precomputed adjacency matrix 
        m.status[:G_computed] ? nothing : throw(ArgumentError("The adjacency matrix has not been computed yet"))
        res = sum(@view m.Ĝ[:,i])  
    else
        throw(ArgumentError("Unknown method $method"))
    end

    return res
end


"""
    indegree(m::DBCM[, v]; method=:reduced)

Return a vector corresponding to the expected indegree of the DBCM model `m` each node. If v is specified, only return indegrees for nodes in v.

# Arguments
- `m::DBCM`: the DBCM model
- `v::Vector{Int}`: the nodes for which to compute the indegree. Default is all nodes.
- `method::Symbol`: the method to use for computing the indegree. Can be any of the following:
    - `:reduced` (default) uses the reduced model parameters `xᵣ` for perfomance reasons.
    - `:full` uses all elements of the expected adjacency matrix.
    - `:adjacency` uses the precomputed adjacency matrix `m.Ĝ` of the model.

# Examples
```jldoctest
julia> model = DBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()));

julia> solve_model!(model);

julia> set_Ĝ!(model);

julia> typeof(indegree(model, method=:adjacency)) 
Vector{Float64} (alias for Array{Float64, 1})

``` 
"""
indegree(m::DBCM, v::Vector{Int}=collect(1:m.status[:d]); method::Symbol=:reduced) = [indegree(m, i, method=method) for i in v]

"""
    degree(m::DBCM, i::Int; method=:reduced)

In alignment with `Graphs.jl`, returns the sum of the expected out- and indegree for node `i` of the DBCM model `m`.
Uses the reduced model parameters `xᵣ` for perfomance reasons.

# Arguments
- `m::DBCM`: the DBCM model
- `i::Int`: the node for which to compute the degree.
- `method::Symbol`: the method to use for computing the degree. Can be any of the following:
    - `:reduced` (default) uses the reduced model parameters `xᵣ` for perfomance reasons.
    - `:full` uses all elements of the expected adjacency matrix.
    - `:adjacency` uses the precomputed adjacency matrix `m.Ĝ` of the model.

# Examples
```jldoctest
julia> model = DBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()));

julia> solve_model!(model);

julia> set_Ĝ!(model);

julia> typeof([degree(model, 1), degree(model, 1, method=:full), degree(model, 1, method=:adjacency)])
Vector{Float64} (alias for Array{Float64, 1})

``` 
"""
degree(m::DBCM, i::Int; method::Symbol=:reduced) = outdegree(m, i, method=method) + indegree(m, i, method=method)

"""
    degree(m::DBCM[, v]; method=:reduced)

In alignment with `Graphs.jl`, returns a vector corresponding to the sum of the expected out- and indegree of the DBCM model `m` each node. If v is specified, only return indegrees for nodes in v.

# Arguments
- `m::DBCM`: the DBCM model
- `v::Vector{Int}`: the nodes for which to compute the degree. Default is all nodes.
- `method::Symbol`: the method to use for computing the degree. Can be any of the following:
    - `:reduced` (default) uses the reduced model parameters `xᵣ` for perfomance reasons.
    - `:full` uses all elements of the expected adjacency matrix.
    - `:adjacency` uses the precomputed adjacency matrix `m.Ĝ` of the model.

# Examples
```jldoctest
julia> model = DBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()));

julia> solve_model!(model);

julia> set_Ĝ!(model);

julia> typeof(degree(model, method=:adjacency)) 
Vector{Float64} (alias for Array{Float64, 1})

``` 
"""
degree(m::DBCM, v::Vector{Int}=collect(1:m.status[:d]); method::Symbol=:reduced) = outdegree(m, v, method=method) + indegree(m, v, method=method)

"""
    AIC(m::DBCM)

Compute the Akaike Information Criterion (AIC) for the DBCM model `m`. The parameters of the models most be computed beforehand. 
If the number of empirical observations becomes too small with respect to the number of parameters, you will get a warning. In 
that case, the corrected AIC (AICc) should be used instead.

# Examples
```julia-repl
julia> model = DBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()));

julia> solve_model!(model);

julia> AIC(model);
┌ Warning: The number of observations is small with respect to the number of parameters (n/k < 40). Consider using the corrected AIC (AICc) instead.
[...]

```

See also [`AICc`](@ref MaxEntropyGraphs.AICc), [`L_DBCM_reduced`](@ref MaxEntropyGraphs.L_DBCM_reduced).
"""
function AIC(m::DBCM)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    # compute AIC components
    k = m.status[:d] * 2 # number of parameters ( = 2 * number of in/out degree pairs)
    n = (m.status[:d] - 1) * m.status[:d]  # number of observations (N-1)*N
    L = L_DBCM_reduced(m) # log-likelihood

    if n/k < 40
        @warn """The number of observations is small with respect to the number of parameters (n/k < 40). Consider using the corrected AIC (AICc) instead."""
    end

    return 2*k - 2*L
end



"""
    AICc(m::DBCM)

Compute the corrected Akaike Information Criterion (AICc) for the DBCM model `m`. The parameters of the models most be computed beforehand. 


# Examples
```jldoctest
julia> model = DBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()));

julia> solve_model!(model);

julia> AICc(model)
314.5217467272881

```

See also [`AIC`](@ref MaxEntropyGraphs.AIC), [`L_DBCM_reduced`](@ref MaxEntropyGraphs.L_DBCM_reduced).
"""
function AICc(m::DBCM)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    # compute AIC components
    k = m.status[:d] * 2 # number of parameters ( = 2 * number of in/out degree pairs)
    n = (m.status[:d] - 1) * m.status[:d]   # number of observations (N-1)*N
    L = L_DBCM_reduced(m) # log-likelihood

    return 2*k - 2*L + (2*k*(k+1)) / (n - k - 1)
end


"""
    BIC(m::DBCM)

Compute the Bayesian Information Criterion (BIC) for the DBCM model `m`. The parameters of the models most be computed beforehand. 
BIC is believed to be more restrictive than AIC, as the former favors models with a lower number of parameters than those favored by the latter.

# Examples
```jldoctest
julia> model = DBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques()));

julia> solve_model!(model);

julia> BIC(model)
415.69929372350714

```

See also [`AIC`](@ref MaxEntropyGraphs.AIC), [`L_DBCM_reduced`](@ref MaxEntropyGraphs.L_DBCM_reduced).
"""
function BIC(m::DBCM)
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    # compute AIC components
    k = m.status[:d] * 2 # number of parameters
    n = (m.status[:d] - 1) * m.status[:d]  # number of observations
    L = L_DBCM_reduced(m) # log-likelihood

    return k * log(n) - 2*L
end