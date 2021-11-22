abstract type AbstractMaxEntropyModel end

# Large version UBCM
struct UBCM{T} <: AbstractMaxEntropyModel where {T<:Real}
    x0::Vector{T}         # initial solution estimate
    x::Vector{T}          # solution if converged
    F::Vector{T}          # vector for values of the system of equations
    f!::Function          # sytem that needs to be solved
end

show(io::IO, model::UBCM{T}) where {T} = print(io, "UBCM ($(T)) model ($(length(model.x0)) nodes)")

function UBCM(G::T) where {T<:AbstractGraph}
    # assert validity of the method
    !is_connected(G) && throw(DomainError("Graph is not connected"))
    has_self_loops(G) && throw(DomainError("Graph has self loops"))
    # get degree sequence
    k = degree(G)
    # estimate for x0
    x0 = k ./ maximum(k)
    return UBCM(x0, similar(x0), similar(x0), (F::Vector, x::Vector) -> UBCM!(F, x, k))
end

function UBCM!(F::Vector, x::Vector, k::Vector)
    Threads.@threads for i in eachindex(x)
        @inbounds F[i] = -f(x[i]*x[i]) - k[i]
        @simd for j in eachindex(x)
            @inbounds F[i] += f(x[i]*x[j])
        end
    end
    return F
end

function solve!(model::UBCM)
    res =  nlsolve(model.f!, model.x0, autodiff=:forward)
    !res.f_converged && @warn "Solver did not converge"
    if res.f_converged
        any(res.zero .< 0) && @warn "Negative values in solution"
        @inbounds for i in eachindex(model.x)
            model.x[i] = res.zero[i]
        end
    end
    return res
end

# Compact version UBCM
mutable struct UBCMCompact{T, I} <: AbstractMaxEntropyModel where {T<:Real, I<:Integer}
    x0::Vector{T}         # initial solution estimate
    xs::Vector{T}         # solution if converged
    x::Union{IndirectArray, Nothing}      # expanded vector (using IndirectArrays)
    F::Vector{T}          # vector for values of the system of equations
    f!::Function          # sytem that needs to be solved
    #freqs::Dict{Int, Int} # dict holding the mapping (not required, but nice to have)
    #nodemap::Dict{Int, Int} # dict holding the mapping
    revmap::Vector{I}        # dict holding the reverse mapping
end

show(io::IO, model::UBCMCompact{T,I}) where {T,I} = print(io, "Compact UBCM ($(T),$(I)) model ($(length(model.revmap)) nodes reduced to $(length(model.x0)))")

function UBCMCompact(G::T) where {T<:AbstractGraph}
    # assert validity of the method
    !is_connected(G) && throw(DomainError("Graph is not connected"))
    has_self_loops(G) && throw(DomainError("Graph has self loops"))
    # get unique degrees and frequencies
    d = degree(G)
    freqs = countmap(d)
    κ = collect(keys(freqs))
    # get reverse mapping
    #revmap = Dict(i => findall(x->isequal(x, κ[i]), d) for i in eachindex(κ))
    #nodemap = Dict(keys => val for (key, val) in revmap)
    indtype = length(d) < typemax(UInt8) ? UInt8 : length(d) < typemax(UInt16) ? UInt16 : length(d) < typemax(UInt32) ? UInt32 : UInt64
    revmap = Vector{indtype}(undef, length(d))
	for i in eachindex(κ)
		revmap[findall(x->isequal(x, κ[i]), d)] .= indtype(i) # link values of large vector to compact one
	end

    # estimate for x0
    x0 = sqrt.(κ) ./ sum(κ)

    return UBCMCompact(x0, similar(x0), nothing, similar(x0), (F::Vector, x::Vector) -> UBCMCompact!(F, x, κ, freqs), revmap)
end

function UBCMCompact!(F::Vector, x::Vector, κ::Vector, freqs::Dict)
    Threads.@threads for i in eachindex(x)
        @inbounds F[i] = -f(x[i]*x[i]) - κ[i]
        @simd for j in eachindex(x)
            @inbounds F[i] += freqs[κ[j]]*f(x[i]*x[j])
        end
    end
    return F
end

function solve!(model::UBCMCompact)
    res =  nlsolve(model.f!, model.x0, autodiff=:forward)
    !res.f_converged && @warn "Solver did not converge"
    if res.f_converged
        # expansion of compact solution to large one happens here
        any(res.zero .< 0) && @warn "Negative values in solution"
        @inbounds for i in eachindex(model.x0)
            model.xs[i] = res.zero[i]
        end
        # generation of large solution happens here (using IndirectArrays)
        model.x = IndirectArray(model.revmap, model.xs)
    end
    return res
end


# supporting functions for computations of UBCM
f(x) = x / (1 + x) # helper function 
a(i::Int,j::Int,x::T) where {T<:AbstractVector} = @inbounds f(x[i]*x[j]) # single element of adjacency matrix
"""Obtain full adjacency matrix from a model"""
function adjacency_matrix(model::UBCM)
    A = zeros(length(model.x), length(model.x))
    for i in eachindex(model.x)
        for j in eachindex(model.x)
            if i≠j
                A[i,j] = f(model.x[i]*model.x[j])
            end
        end
    end
    return A
end

"""helper function for adjacency matrix"""
function indmap(model::UBCMCompact,T::DataType,i,j)
    return model.revmap[i] <= model.revmap[j] ? T(model.revmap[i]+model.revmap[j]*(model.revmap[j]-1)÷2) : T(model.revmap[j]+model.revmap[i]*(model.revmap[i]-1)÷2)
end

"""Obtain full adjacency matrix from a compact model (as an IndirectArray)"""
function adjacency_matrix(model::UBCMCompact)
    # generate vector of all possible unique values in the adjacency matrix
    values = @inbounds [f(model.xs[i]*model.xs[j]) for j = 1:length(model.xs) for i = 1:j]
    push!(values,zero(eltype(values))) # include the zero for later
    # infer storage index type
    T = length(values) < typemax(UInt8) ? UInt8 : length(values) < typemax(UInt16) ? UInt16 : UInt32
    # generate indices in matrix from xs
    indices = [indmap(model,T,i,j) for i in 1:length(model.revmap), j in 1:length(model.revmap)]
    # set diagonal to zero
    indices[diagind(indices)] .= T(length(values))
    # return indirectarray
    return IndirectArray(indices, values)
end

