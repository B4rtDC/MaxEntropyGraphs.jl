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

When using and optimisation method from the `Optimisation.jl` framework, the return code of the optimisation method is stored in the `retcode` field.
When using the fixed point iteration method, the `retcode` field is set to `nothing`.
"""
struct ConvergenceError{T} <: Exception
    method::Symbol
    retcode::Union{Nothing, T}
end

# The fixed-point path has no SciMLBase return code, so allow construction with `nothing`
# (the typed path infers `T` from the supplied return code automatically).
ConvergenceError(method::Symbol, ::Nothing) = ConvergenceError{Union{}}(method, nothing)

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

