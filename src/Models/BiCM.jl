
"""
    BiCM

Maximum entropy model for the Undirected Bipartite Configuration Model (BiCM). 
    
The object holds the maximum likelihood parameters of the model (θ), the expected bi-adjacency matrix (Ĝ), 
and the variance for the elements of the adjacency matrix (σ).

cf. "Inferring monopartite projections of bipartite networks: an entropy-based approach", Fabio Saracco et al 2017 New J. Phys. 19 053022
"""
mutable struct BiCM{T,N} <: AbstractMaxEntropyModel where {T<:Union{Graphs.AbstractGraph, Nothing}, N<:Real}
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


DETAIL how zero degree nodes are treated

# Examples     
```jldoctest
# generating a model from a graph


# generating a model directly from a degree sequence


# generating a model directly from a degree sequence with a different precision


# generating a model from an adjacency matrix


# generating a model from an edge list


```

See also [`Graphs.degree`](@ref)
"""
function BiCM(G::T; d⊥::Union{Nothing, Vector}=nothing, 
                    d⊤::Union{Nothing, Vector}=nothing, 
                    precision::Type{N}=Float64, 
                    kwargs...) where {T,N<:AbstractFloat}
    T <: Union{Graphs.AbstractGraph, Nothing} ? nothing : throw(TypeError("G must be a subtype of AbstractGraph or Nothing"))
    

    # coherence checks
    if T <: Graphs.AbstractGraph # Graph specific checks
        # check if the graph is bipartite
        Graphs.is_bipartite(G) ? nothing : throw(ArgumentError("The graph is not bipartite"))
        
        if Graphs.is_directed(G)
            @warn "The graph is directed, while the BiCM model is undirected, the directional information will be lost"
        end

        if T <: SimpleWeightedGraphs.AbstractSimpleWeightedGraph
            @warn "The graph is weighted, while BiCM model is unweighted, the weight information will be lost"
        end

        # get layer membership
        membership = Graphs.bipartite_map(G)
        ⊥nodes, ⊤nodes = findall(membership .== 1), findall(membership .== 2)
        # degree sequences
        d⊥ = isnothing(d⊥) ? Graphs.degree(G, ⊥nodes) : d⊥
        d⊤ = isnothing(d⊤) ? Graphs.degree(G, ⊤nodes) : d⊤


        Graphs.nv(G) == 0 ? throw(ArgumentError("The graph is empty")) : nothing
        Graphs.nv(G) == 1 ? throw(ArgumentError("The graph has only one vertex")) : nothing
        Graphs.nv(G) != length(d⊥) + length(d⊤) ? throw(DimensionMismatch("The number of vertices in the graph ($(Graphs.nv(G))) and the length of the degree sequence ($(length(d))) do not match")) : nothing
    end
    # coherence checks specific to the degree sequences
    !isnothing(d⊥) && length(d⊥) == 0 ? throw(ArgumentError("The degree sequences d⊥ is empty")) : nothing
    !isnothing(d⊤) && length(d⊤) == 0 ? throw(ArgumentError("The degree sequences d⊤ is empty")) : nothing
    !isnothing(d⊥) && length(d⊥) == 1 ? throw(ArgumentError("The degree sequences d⊥ only contains a single node")) : nothing
    !isnothing(d⊤) && length(d⊤) == 1 ? throw(ArgumentError("The degree sequences d⊤ only contains a single node")) : nothing    
    maximum(d⊥) >= length(d⊤) ? throw(DomainError("The maximum outdegree in the layer d⊥ is greater or equal to the number of vertices in layer d⊤, this is not allowed")) : nothing
    maximum(d⊤) >= length(d⊥) ? throw(DomainError("The maximum outdegree in the layer d⊤ is greater or equal to the number of vertices in layer d⊥, this is not allowed")) : nothing


    # field generation
    d⊥ᵣ, d⊥_ind, d⊥ᵣ_ind, f⊥ = np_unique_clone(d⊥, sorted=true)
    d⊥ᵣ_nz = iszero(first(d⊥ᵣ)) ? (2:length(d⊥ᵣ)) : (1:length(d⊥ᵣ)) # precomputed indices for the reduced degree sequence (works because sorted and unique values)
    d⊤ᵣ, d⊤_ind, d⊤ᵣ_ind, f⊤ = np_unique_clone(d⊤, sorted=true)
    d⊤ᵣ_nz = iszero(first(d⊤ᵣ)) ? (2:length(d⊤ᵣ)) : (1:length(d⊤ᵣ)) # precomputed indices for the reduced degree sequence

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
    
    return BiCM{T,precision}(G, θᵣ, xᵣ, yᵣ, d⊥, d⊤, d⊥ᵣ, d⊤ᵣ, d⊥ᵣ_nz, d⊤ᵣ_nz, f⊥, f⊤, d⊥_ind, d⊤_ind, d⊥ᵣ_ind, d⊤ᵣ_ind, ⊥nodes, ⊤nodes, nothing, nothing, status, nothing)
end

BiCM(; d⊥::Vector{T}, d⊤::Vector{T}, precision::Type{N}=Float64, kwargs...) where {T<:Signed, N<:AbstractFloat} = BiCM(nothing; d⊥=d⊥, d⊤=d⊤, precision=precision, kwargs...)



function L_BiCM_reduced(θ::Vector, k⊥::Vector, k⊤::Vector, f⊥::Vector, f⊤::Vector, nz⊥::UnitRange, nz⊤::UnitRange, n⊥ᵣ::Int)
    # set pre-allocated values
    α = @view θ[1:n⊥ᵣ]
    β = @view θ[n⊥ᵣ+1:end]
    res = zero(eltype(θ))
    # actual compute
    for i in nz⊥
        res -=  f⊥[i]*k⊥[i]*α[i] 
        for j in nz⊤
            res -= f⊥[i] * f⊤[j] * log(1 + exp(-α[i] - β[j]))
        end
    end
    for j in nz⊤
        res -= f⊤[j]*k⊤[j]*β[j]
    end
    return res
end


L_BiCM_reduced(m::BiCM) = L_BiCM_reduced(m.θᵣ, m.d⊥ᵣ, m.d⊤ᵣ, m.f⊥, m.f⊤, m.d⊥ᵣ_nz, m.d⊤ᵣ_nz, m.status[:d⊥_unique])


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
    
    # actual compute
    for i in nz⊥
        ∇L[i] = - f⊥[i] * k⊥[i]
        for j in nz⊤
            ∇L[i]    += f⊥[i] * f⊤[j] * x[i]*y[j]/(1 + x[i]*y[j])
        end
    end
    for j in nz⊤
        ∇L[n⊥+j] = - f⊤[j] * k⊤[j]
        for i in nz⊥
            ∇L[n⊥+j] += f⊥[i] * f⊤[j] * x[i]*y[j]/(1 + x[i]*y[j])
        end
    end

    return ∇L
end



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

    # actual compute
    for i in nz⊥
        ∇L[i] = f⊥[i] * k⊥[i]
        for j in nz⊤
            ∇L[i]    -= f⊥[i] * f⊤[j] * x[i]*y[j]/(1 + x[i]*y[j])
        end
    end
    for j in nz⊤
        ∇L[n⊥+j] = f⊤[j] * k⊤[j]
        for i in nz⊥
            ∇L[n⊥+j] -= f⊥[i] * f⊤[j] * x[i]*y[j]/(1 + x[i]*y[j])
        end
    end

    return ∇L
end

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

function initial_guess(m::BiCM{T,N}; method::Symbol=:degrees) where {T,N}
    if isequal(method, :degrees)
        return Vector{N}(vcat(-log.(m.d⊥ᵣ), -log.(m.d⊤ᵣ)))
    elseif isequal(method, :random)
        return Vector{N}(-log.(rand(N, length(m.θᵣ))))
    elseif isequal(method, :uniform)
        return Vector{N}(-log.(0.5 .* ones(N, length(m.θᵣ))))
    elseif isequal(method, :chung_lu)
        isnothing(m.G) ? throw(ArgumentError("Cannot compute the number of edges because the model has no underlying graph (m.G == nothing)")) : nothing
        return Vector{N}(vcat(-log.(m.d⊥ᵣ ./ sqrt(Graphs.ne(m.G)) ), -log.(m.d⊤ᵣ ./ sqrt(Graphs.ne(m.G)) ) ))
    else
        throw(ArgumentError("The initial guess method $(method) is not supported"))
    end
end

function set_xᵣ!(m::BiCM)
    if m.status[:params_computed]
        αᵣ = @view m.θᵣ[1:m.status[:d⊥_unique]]
        m.xᵣ .= exp.(-αᵣ)
    else
        throw(ArgumentError("The parameters have not been computed yet"))
    end
end

function set_yᵣ!(m::BiCM)
    if m.status[:params_computed]
        βᵣ = @view m.θᵣ[m.status[:d⊥_unique]+1:end]
        m.yᵣ .= exp.(-βᵣ)
    else
        throw(ArgumentError("The parameters have not been computed yet"))
    end
end

# NOTE: this will compute the bi-adjacency matrix, and not the adjacency matrix
function Ĝ(m::BiCM{T,N}) where {T,N}
    # check if possible
    m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
    
    # get layer sizes => this is the full size of the network
    n⊥, n⊤ = m.status[:N⊥], m.status[:N⊤] 

    # initiate Ĝ
    G = zeros(N, n⊥, n⊤)

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

function set_Ĝ!(m::BiCM)
    m.Ĝ = Ĝ(m)
    m.status[:G_computed] = true
    return m.Ĝ
end

"""
    rand(m::BiCM; precomputed::Bool=false)

Generate a random graph from the BiCM model `m`.

**Note**: the generated matrix from the `BiCM` is a bi-adjacency matrix, not an adjacency matrix. The generated graph
wil also be bipartite and respect the layer membership of the original graph.
"""
function rand(m::BiCM; precomputed::Bool=false)
    if precomputed
        # check if possible to use precomputed Ĝ
        m.status[:G_computed] ? nothing : throw(ArgumentError("The expected adjacency matrix has not been computed yet"))
        # generate random graph
        G = Graphs.SimpleDiGraphFromIterator( Graphs.Edge.([(or⊥,or⊤) for (i,or⊥) in enumerate(m.⊥nodes) for (j,or⊤) in enumerate(m.⊤nodes) if rand()<m.Ĝ[i,j]]))
    else
        # check if possible to use parameters
        m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
        # initiate x and y
        x = m.xᵣ[m.d⊥ᵣ_ind]
        y = m.yᵣ[m.d⊤ᵣ_ind]
        G = Graphs.SimpleGraphFromIterator( Graphs.Edge.([(or⊥,or⊤) for (i,or⊥) in enumerate(m.⊥nodes) for (j,or⊤) in enumerate(m.⊤nodes) if rand()< x[i]*y[j]/(1 + x[i]*y[j]) ]))
    end

    # deal with edge case where no edges are generated for the last node(s) in the graph
    while Graphs.nv(G) < m.status[:N]
        Graphs.add_vertex!(G)
    end

    return G
end

function rand(m::BiCM, n::Int; precomputed::Bool=false)
    # pre-allocate
    res = Vector{Graphs.SimpleGraph{Int}}(undef, n)
    # fill vector using threads
    Threads.@threads for i in 1:n
        res[i] = rand(m; precomputed=precomputed)
    end

    return res
end


function solve_model!(m::BiCM{T,N}) where {T,N}
end