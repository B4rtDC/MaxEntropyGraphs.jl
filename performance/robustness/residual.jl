# Outcome extraction for the solver-robustness sweep: how well a fit reproduces its constraints,
# how many iterations it took, and whether the instance was well posed to begin with.

using MaxEntropyGraphs
const MEG = MaxEntropyGraphs

# ---------------------------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------------------------
#
# `constraint_residual` is the only success metric comparable across the whole grid. `ftol` is not:
# it means the parameter-space increment on the binary models' `:fixedpoint`, the absolute
# constraint residual on UECM/DECM `:fixedpoint`, the relative constraint residual on the two-step
# models' `:fixedpoint`, and nothing at all on every Optim path, where it is ignored with a
# warning. So `ftol` is recorded as an input and never read back as an outcome.

"""
    residual(m) -> Float64

Maximum absolute constraint violation of a solved model, via `constraint_residual`. Returns `Inf`
for a model whose parameters were never computed, so an unsolved cell sorts as the worst outcome
rather than being dropped.
"""
function residual(m)
    get(m.status, :params_computed, false) || return Inf
    try
        return float(MEG.constraint_residual(m))
    catch
        return Inf
    end
end

# ---------------------------------------------------------------------------------------------
# Iteration counts
# ---------------------------------------------------------------------------------------------
#
# `solve_model!` returns `(m, sol)` for every model, but `sol` has five shapes: an
# `OptimizationSolution` on every Optim path; an NLsolve result for UBCM/DBCM/RBCM/BiCM and the
# two-step models' `:fixedpoint`; a `NamedTuple (zero, iterations, residual, converged)` for
# UECM/DECM `:fixedpoint`; and a per-channel `NamedTuple (out, in)` for the DBiCM, in which an
# empty channel is `nothing`. Reading `.iterations` blindly silently records zeros.

"""
    solve_iterations(sol) -> Int

Iteration count from any of the solution shapes `solve_model!` returns. `-1` means the shape
carried no count.
"""
function solve_iterations(sol)
    sol === nothing && return 0
    if sol isa NamedTuple && haskey(sol, :out) && haskey(sol, :in)
        a = solve_iterations(sol.out)
        b = solve_iterations(sol.in)
        return (a < 0 || b < 0) ? -1 : a + b
    end
    if hasproperty(sol, :iterations)
        it = getproperty(sol, :iterations)
        it isa Integer && return Int(it)
    end
    if hasproperty(sol, :stats)
        st = getproperty(sol, :stats)
        if hasproperty(st, :iterations)
            it = getproperty(st, :iterations)
            it isa Integer && return Int(it)
        end
    end
    -1
end

# ---------------------------------------------------------------------------------------------
# Well-posedness
# ---------------------------------------------------------------------------------------------
#
# A "runaway" constraint puts the maximum-likelihood optimum at an infinite parameter, so no
# solver can settle at a finite tolerance. Those instances are not failures of the solver and are
# reported as their own subset rather than mixed in; the published UECM/DECM tables are split the
# same way. The three kinds are a zero constraint, a saturated one (`k = N-1`, or the bipartite
# equivalent `k = ` the number of live counterparts), and an all-weights-one node (`s = k`), where
# the weighted layer collapses onto the binary one.

_has_zero(v)        = any(iszero, v)
_has_saturated(v,N) = any(==(N - 1), v)
_has_unit_weights(d, s) = any(i -> s[i] == d[i], eachindex(d))

runaway(m::MEG.UBCM) = (N = length(m.d); _has_zero(m.d) || _has_saturated(m.d, N))
runaway(m::MEG.DBCM) = (N = length(m.d_out);
                        _has_zero(m.d_out) || _has_zero(m.d_in) ||
                        _has_saturated(m.d_out, N) || _has_saturated(m.d_in, N))
runaway(m::MEG.RBCM) = (N = length(m.d_out);
                        _has_saturated(m.d_out, N) || _has_saturated(m.d_in, N) ||
                        _has_saturated(m.d_rec, N))
runaway(m::MEG.BiCM) = (maximum(m.d⊥) >= count(!iszero, m.d⊤) ||
                        maximum(m.d⊤) >= count(!iszero, m.d⊥))
function runaway(m::MEG.DBiCM)
    sat(d, opp) = !iszero(maximum(d)) && maximum(d) >= count(!iszero, opp)
    sat(m.d⊥_out, m.d⊤_in) || sat(m.d⊤_in, m.d⊥_out) ||
    sat(m.d⊥_in, m.d⊤_out) || sat(m.d⊤_out, m.d⊥_in)
end
runaway(m::MEG.UECM) = (N = length(m.d);
                        _has_zero(m.d) || _has_saturated(m.d, N) || _has_unit_weights(m.d, m.s))
runaway(m::MEG.DECM) = (N = length(m.d_out);
                        _has_zero(m.d_out) || _has_zero(m.d_in) ||
                        _has_saturated(m.d_out, N) || _has_saturated(m.d_in, N) ||
                        _has_unit_weights(m.d_out, m.s_out) || _has_unit_weights(m.d_in, m.s_in))
runaway(m::MEG.CReM)  = _has_zero(m.s)
runaway(m::MEG.DCReM) = _has_zero(m.s_out) || _has_zero(m.s_in)
runaway(m::MEG.CRWCM) = _has_zero(m.s_out) || _has_zero(m.s_in)
