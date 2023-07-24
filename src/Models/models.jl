##################################################################################
# models.jl
#
# This file contains model types and methods for the MaxEntropyGraphs.jl package
##################################################################################


# solver function constants
const optimization_methods = Dict(  :LBFGS      => OptimizationOptimJL.LBFGS(),
                                    :BFGS       => OptimizationOptimJL.BFGS(),
                                    :Newton     => OptimizationOptimJL.Newton(),
                                    :NelderMead => OptimizationOptimJL.NelderMead())

const AD_methods = Dict(:AutoZygote         => Optimization.AutoZygote(),
                        :AutoForwardDiff    => Optimization.AutoForwardDiff(),
                        :AutoReverseDiff    => Optimization.AutoReverseDiff(),
                        :AutoFiniteDiff     => Optimization.AutoFiniteDiff())


"""
    AbstractMaxEntropyModel

An abstract type for a MaxEntropyModel. Each model has one or more structural constraints  
that are fixed while the rest of the network is completely random. 
"""
abstract type AbstractMaxEntropyModel end


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
#     σˣ(X::Function, M::DBCM{T})

# Compute the standard deviation of a property `X` with respect to the expected adjacency matrix associated with the `DBCM` model `M`.
# """
# σˣ(X::Function, M::DBCM{T}) where T = sqrt( sum((M.σ .* ∇X(X, M)) .^ 2) )

