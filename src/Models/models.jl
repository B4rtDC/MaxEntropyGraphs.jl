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

# The enhanced models (UECM/DECM) have a feasible region `yᵢyⱼ < 1` (`βᵢ + βⱼ > 0`); outside it the
# likelihood is not defined (it evaluates to `NaN`). The default HagerZhang / (Strong)Wolfe line searches
# cannot cope with that barrier and stall almost immediately, whereas a BackTracking line search (halve
# the step until the objective is finite and satisfies the Armijo condition) stays in the feasible
# interior and converges — this is exactly the backtracking recipe of Vallarano et al. (2021).
# The enhanced models therefore use these optimizer instances (the other models keep the package-wide
# `optimization_methods`, which work well for their unconstrained domain).
const backtracking_optimization_methods = Dict( :LBFGS  => OptimizationOptimJL.LBFGS( linesearch = OptimizationOptimJL.Optim.LineSearches.BackTracking()),
                                                :BFGS   => OptimizationOptimJL.BFGS(  linesearch = OptimizationOptimJL.Optim.LineSearches.BackTracking()),
                                                :Newton => OptimizationOptimJL.Newton(linesearch = OptimizationOptimJL.Optim.LineSearches.BackTracking()))


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
    _UECM_β_FLOOR

Lower bound imposed on the `β` entries of **degree/strength classes of multiplicity `Fᵢ ≥ 2`** when the
[`UECM`](@ref) is solved with a first-order method.

The UECM is defined wherever every pair that actually exists satisfies `yᵢyⱼ < 1`, i.e. `βᵢ + βⱼ > 0` — a
*pairwise* condition, not a per-coordinate one. The only per-coordinate case is the **same-class** pair,
which needs `2βᵢ > 0`; a class of multiplicity `Fᵢ = 1` contains a single vertex and therefore has no
same-class pair, so nothing in the model bounds the sign of its own `βᵢ`.

Applying this floor to every class — as versions up to `v0.7.0` did — is therefore too tight, and it is
not a harmless over-restriction: on 5 of 128 random weighted networks the ML optimum genuinely has a
`βᵢ < 0` on a singleton class, and a solver pinned at the floor reported `Success` with a degree/strength
residual between `0.17` and `2.31`. Singleton classes now get `-Inf` and are held inside the domain by
the objective itself (`L_UECM_reduced` returns `NaN` off it), exactly as the [`DECM`](@ref) does.

The value is deliberately tiny — a feasibility guard, not a regularisation, so it does not move a
well-posed solution. A class whose ML solution genuinely pushes `βᵢ` to the boundary will report that
`βᵢ` sitting at this floor.
"""
const _UECM_β_FLOOR = 1e-10


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
    _ECM_LOGCAP

Cap on `|log x|` and `|log y|` inside the `UECM`/`DECM` `:fixedpoint` block solvers.

A degenerate constraint (a saturated degree, or a class whose links all carry weight 1) puts the optimum
at an infinite parameter, and the two-dimensional Newton polish will march towards it. `exp(±80)` is
`10^±35`, which keeps every pair product `xᵢxⱼ·yᵢyⱼ` representable in `Float64` while being far enough
out that the constraint it is chasing is already satisfied to well below `eps()`.
"""
const _ECM_LOGCAP = 80.0


"""
    _monotone_root(φ, lo, hi, target, v0; maxit)

Solve `φ(v) = target` for a function that is **strictly increasing** on `(lo, hi)` and brackets the
target there, using Newton steps safeguarded by bisection. `φ` returns the tuple `(value, derivative)`.

Used by the `UECM`/`DECM` `:fixedpoint` solvers for the one-dimensional block updates, where the
monotonicity and the bracket are both established analytically (see `UECM_reduced_coordinate_iter!`).
Because every accepted iterate stays inside the initial bracket, the returned root is feasible by
construction — which is the property the Picard recipe those solvers replaced did not have.
"""
function _monotone_root(φ, lo::N, hi::N, target::N, v0::N; maxit::Int=100) where {N<:Real}
    lo < hi || return lo
    v = clamp(v0, nextfloat(lo), prevfloat(hi))
    for _ in 1:maxit
        val, der = φ(v)
        (!isfinite(val) || val > target) ? (hi = v) : (lo = v)
        hi - lo <= eps(N) * max(one(N), abs(hi)) && break
        vn = (isfinite(der) && der > zero(N)) ? v - (val - target) / der : (lo + hi) / 2
        v = (lo < vn < hi) ? vn : (lo + hi) / 2
    end
    return (lo + hi) / 2
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
    _ecm_runaway_message(model, iters, residual, ftol, offenders)

Message explaining a `UECM`/`DECM` `:fixedpoint` non-convergence caused by a **runaway constraint**.

A constraint at the edge of its feasible range — a saturated degree (`k = N-1`), or a node whose links all
carry weight `1` (`s = k`) — has its maximum-likelihood parameter at `±∞`. The block-coordinate map
approaches such a point only geometrically (measured rate `0.9998` per sweep), so it cannot reach `ftol`
in any practical number of sweeps, and raising `maxiters` does not help. The gradient methods do reach a
usable answer on these networks, because they are free to travel a long way in `θ` per step.

`offenders` is a list of `description => indices` pairs, already restricted to the classes at fault.
"""
function _ecm_runaway_message(model::AbstractString, iters::Int, residual::Real, ftol::Real,
                              offenders::Vector{<:Pair{<:AbstractString,<:AbstractVector}})
    lines = ["`:fixedpoint` stopped after $(iters) sweeps; best constraint residual reached was $(residual) (ftol = $(ftol)).",
             "This $(model) has a runaway constraint, whose maximum-likelihood parameter is infinite:"]
    for (what, idx) in offenders
        isempty(idx) && continue
        push!(lines, "  • $(what): reduced class(es) $(idx)")
    end
    push!(lines, "The fixed point approaches an infinite parameter only geometrically, so it does not settle")
    push!(lines, "at `ftol`, and raising `maxiters` will not help — it only makes the failure slower. Use")
    push!(lines, "`method = :BFGS` for this network; if the best residual above is good enough for your")
    push!(lines, "purpose, a looser `ftol` will accept it.")
    return join(lines, "\n")
end


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


"""
    _gauge_fixedpoint_ladder(FP!, θ₀; ftol, maxiters, verbose=false)

Run `NLsolve.fixedpoint` on the Anderson-accelerated map `FP!`, retrying down a ladder of shorter
memories (default → `m=2` → `m=0`) whenever the accelerator produces non-finite values. Returns the
`NLsolve` solution object. Shared by the bipartite models, whose maps are gauge-equivariant.

WHY: a bipartite fixed-point map is *gauge-equivariant*. With `g = (1…1, -1…-1)` over the live
entries, `G(θ + c·g) = G(θ) + c·g` exactly — the shift leaves every product `xᵢ·yⱼ` alone and
multiplies the inner sum by `e^c`, which the outer `-log` turns back into `+c`. Two consequences:
`g` is an eigenvector of the Jacobian with eigenvalue **exactly 1** (measured `‖J·g - g‖ ≈ 2e-16`,
next eigenvalue `|λ-1| ≈ 0.06`), and the residual `G(θ) - θ` is completely **blind** to the gauge
component. The residual Jacobian is therefore singular along `g` by construction — measured
`rank = 8` of `9` on a small model.

Anderson acceleration solves a least-squares problem built from residual *differences*, and every
one of those lies in `gᗮ`. The more history it keeps, the sooner that system is rank-deficient and
its internal solve emits `NaN`, which NLsolve reports as an `IsFiniteException`. Measured over 183
random bipartite graphs, failures rise monotonically with the memory: 0 at `m=0`, 2 at `m=2`, 25 at
the default, 75 at `m=20` — and *every* failure is that `NaN`, never a failure to converge in time.

So the remedy is to shrink the least-squares, not to damp it. Damping (`beta=0.5`), which is what
`UBCM` does for its own — different — overflow problem, makes this one WORSE (148/183 against
158/183 for the plain path). Stepping the memory down does work, and `m=0` is plain Picard, which
has no least-squares at all and so cannot hit this failure mode. Measured on the same 183 graphs:
this ladder solves 183/183 in 4551 total iterations, against 10413 for always-Picard (robust but
slow) and 158/183 for the accelerated path alone.

A model with `n` independent gauge directions makes the least-squares rank-deficient by `n`, which
is a *different* regime from the one measured above. The `DBiCM` therefore solves its two channels
separately, so that each call here sees exactly one gauge direction.
"""
function _gauge_fixedpoint_ladder(FP!, θ₀::Vector; ftol::Real, maxiters::Int, verbose::Bool=false)
    return try
        NLsolve.fixedpoint(FP!, θ₀, method=:anderson, ftol=ftol, iterations=maxiters)
    catch e
        e isa NLsolve.IsFiniteException || rethrow()
        verbose && @info "Anderson acceleration produced non-finite values (its least-squares is rank-deficient along the gauge direction); retrying with a shorter memory (m=2)"
        try
            NLsolve.fixedpoint(FP!, θ₀, method=:anderson, m=2, ftol=ftol, iterations=maxiters)
        catch e2
            e2 isa NLsolve.IsFiniteException || rethrow()
            verbose && @info "still non-finite; falling back to un-accelerated Picard iteration (m=0), which has no least-squares to go singular"
            NLsolve.fixedpoint(FP!, θ₀, method=:anderson, m=0, ftol=ftol, iterations=maxiters)
        end
    end
end
