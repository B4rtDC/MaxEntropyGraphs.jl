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
    _DEFAULT_FTOL

Default value of the `ftol` keyword of the `solve_model!` methods, used when the caller leaves it at
`nothing` (see [`_ftol_unused_msg`](@ref) for when it does not apply at all).

It bounds the fixed-point increment, which on the binary models lives in parameter space, and on the
weighted `CReM`/`DCReM`/`CRWCM` layers is the relative constraint residual (they are solved in
log-parameter space, see [`_logspace_fixedpoint`](@ref)).
"""
const _DEFAULT_FTOL = 1e-8


"""
    _ftol_unused_msg(method::Symbol)

Message warning that an explicitly passed `ftol` is silently ignored by the chosen solution `method`.

`ftol` is an `NLsolve` setting that only ever reaches the `:fixedpoint` path; the `Optimization.jl`
methods discard it without a trace. The `solve_model!` methods therefore default `ftol` to `nothing`, so
that an explicit value (worth warning about) can be told apart from an untouched default (never warn).
"""
_ftol_unused_msg(method::Symbol) = """`ftol` applies to the `:fixedpoint` method only and is ignored by `:$(method)`, so it has no effect on this solve. Use `g_tol` to bound the gradient instead, which for these models is the constraint residual (see `constraint_residual`)."""


"""
    _logspace_fixedpoint(FP!, θ₀, live, ftol, maxiters)

Solve the fixed point `θ = G(θ)` of the (in-place, buffer-returning) map `FP!` in **log-parameter**
space, i.e. solve `u = log(G(exp(u)))` for `u = log(θ)` over the index set `live`, and return the
`(θ, sol)` pair with `θ` back in linear space.

This is the recipe used by the weighted `CReM`/`DCReM`/`CRWCM` layers, whose maps all satisfy
``G_i = θ_i⟨x_i⟩/x_i`` exactly (`x` being the constrained sequence). That identity makes the log-space
increment

```
log(G(exp(u)))_i - u_i = log(⟨x_i⟩/x_i)
```

which is (to first order) the **relative** constraint residual ``⟨x_i⟩/x_i - 1``. `NLsolve`'s `ftol`
bounds the infinity norm of that increment, so in log space `ftol` becomes a relative constraint
tolerance: it is invariant under a rescaling of the weights, unlike the linear-space increment
``|G_i - θ_i| = (θ_i/x_i)|⟨x_i⟩ - x_i|``, whose conversion factor ``x_i/θ_i`` grows as the square of the
weight scale. Working in `u` also enforces `θ > 0` for free, keeping the iterates feasible.

Entries outside `live` are **dead channels** (a zero constrained value, for which `log(θ₀) = -Inf` is not
a usable starting point). They are held at their `θ₀` value throughout and returned unchanged, leaving
the caller to pin them to their analytical optimum. The caller is responsible for `live` covering every
index whose value influences the live rows of `FP!`.
"""
function _logspace_fixedpoint(FP!, θ₀::Vector{N}, live::AbstractVector{<:Integer}, ftol::Real, maxiters::Int) where {N}
    θ = copy(θ₀)                    # working parameter vector (dead channels keep their θ₀ value)
    u₀ = N[log(θ₀[i]) for i in live]
    G_log = similar(u₀)             # returned buffer of the log-space map
    FP_log = function (u::Vector)
        @inbounds for (k, i) in enumerate(live)
            θ[i] = exp(u[k])
        end
        G = FP!(θ)
        @inbounds for (k, i) in enumerate(live)
            G_log[k] = log(G[i])
        end
        return G_log
    end
    sol = NLsolve.fixedpoint(FP_log, u₀, method=:anderson, ftol=ftol, iterations=maxiters)
    θ_sol = copy(θ₀)
    @inbounds for (k, i) in enumerate(live)
        θ_sol[i] = exp(sol.zero[k])
    end

    return θ_sol, sol
end


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

