##################################################################################
# models.jl
#
# This file contains model types and methods for the MaxEntropyGraphs.jl package
##################################################################################


# solver function constants
const optimization_methods = Dict(  :LBFGS      => OptimizationOptimJL.LBFGS(),
                                    :BFGS       => OptimizationOptimJL.BFGS(),
                                    :Newton     => OptimizationOptimJL.Newton())

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

When using and optimisation method from the `Optimisation.jl` framework, the return code of the optimisation method is stored in the `retcode` field.
When using the fixed point iteration method, the `retcode` field is set to `nothing`.
"""
struct ConvergenceError <: Exception
    method::Symbol
    retcode::Any  # Optimization.jl return code, or `nothing` for the fixed-point method
end

Base.showerror(io::IO, e::ConvergenceError) = print(io, """method `$(e.method)` did not converge $(isnothing(e.retcode) ? "" : "(Optimization.jl return code: $(e.retcode))")""")


"""
    softplus(x)

Numerically stable evaluation of `log(1 + exp(x))` (the softplus function).

Computed as `max(x, 0) + log1p(exp(-abs(x)))`, which is mathematically identical to
`log(1 + exp(x))` but avoids overflow for large positive `x` (hub nodes / strongly
attached vertices) and precision loss for large negative `x`. This matters in particular
for low-precision (`Float32`/`Float16`) solves. Its derivative is the logistic sigmoid
`1 / (1 + exp(-x))`, matching the analytical gradients used by the models.
"""
@inline softplus(x::T) where {T<:Real} = max(x, zero(T)) + log1p(exp(-abs(x)))


"""
    log1pexpsum(a, b, c)

Numerically stable evaluation of `log(1 + exp(a) + exp(b) + exp(c))` (a four-term log-sum-exp with an
implicit unit term). This is the analog of `softplus` for the RBCM's dyadic normaliser:
``\\ln D_{ij} = \\ln(1 + x_iy_j + x_jy_i + z_iz_j) =`` `log1pexpsum(-(αᵢ+βⱼ), -(αⱼ+βᵢ), -(γᵢ+γⱼ))`.

Computed by factoring out `m = max(0, a, b, c)`, which avoids overflow for large positive arguments and
degrades gracefully for `-Inf` arguments (channels pinned at their analytical optimum contribute an exact
zero: `exp(-Inf) = 0`, and `m ≥ 0` remains finite).
"""
@inline function log1pexpsum(a::T, b::T, c::T) where {T<:Real}
    m = max(zero(T), a, b, c)
    return m + log(exp(-m) + exp(a - m) + exp(b - m) + exp(c - m))
end

