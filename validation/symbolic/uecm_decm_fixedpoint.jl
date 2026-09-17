###############################################################################
# uecm_decm_fixedpoint.jl
#
# Why the UECM/DECM Picard recipe could not be used from a cold start, and why the
# block-coordinate map that replaced it can.
#
# ---------------------------------------------------------------------------
# The domain
# ---------------------------------------------------------------------------
# Both models give a pair the Bernoulli-geometric weight distribution
#
#     P(w) ∝ (x_i x_j)^{1[w>0]} (y_i y_j)^w ,   w = 0,1,2,...
#
# whose normaliser 1 + g·t/(1-t) — with g = x_i x_j and t = y_i y_j — converges only
# for t < 1. So the feasible set is
#
#     UECM:  y_i y_j < 1          (β_i + β_j > 0)   over the pairs that EXIST
#     DECM:  y_out,i y_in,j < 1   (β_out,i + β_in,j > 0)
#
# a PAIRWISE condition. Note what it is not: a per-coordinate box. The only
# per-coordinate case is the same-class pair of a UECM class of multiplicity F ≥ 2,
# which needs 2β_i > 0; a SINGLETON class has no same-class pair, so nothing in the
# model bounds the sign of its own β_i. Section 5 below is about what that costs.
#
# ---------------------------------------------------------------------------
# The Picard recipe, and the asymmetry that kills it
# ---------------------------------------------------------------------------
# The classical recipe (NEMtropy's `iterative_ecm`, shipped here as
# `UECM_reduced_iter!` / `DECM_reduced_iter!`) factors one parameter out of each
# constraint and inverts with the rest frozen:
#
#     ⟨k_i⟩ = x_i·A_i   ⇒   x_i ← d_i / A_i(θ_old)
#     ⟨s_i⟩ = y_i·B_i   ⇒   y_i ← s_i / B_i(θ_old)
#
# These two steps behave in OPPOSITE ways, and that is the whole story.
#
#   A_i = Σ_j w_j·c_j/(d₀_j + c_j x_i)     is strictly DECREASING in x_i
#       (∂/∂x_i = -c²/(d₀+c x_i)² < 0, §1)
#     ⇒ if x_i < x_i*, then A_i(x_i) > A_i(x_i*), so the update lands BELOW the root:
#       x_i < x_new < x_i*. The degree step is a safe, monotone UNDERSHOOT. It is
#       also unconstrained above, so it can never be infeasible.
#
#   B_i = Σ_j w_j·g_j y_j/D(y_i y_j),  D(t) = (1-t)(1-t+g t),  is strictly INCREASING
#       in y_i whenever D'(0) = g - 2 < 0, i.e. x_i x_j < 2 — the whole sparse regime (§1)
#     ⇒ if y_i < y_i*, then B_i(y_i) < B_i(y_i*), so the update lands ABOVE the root.
#       The strength step is a monotone OVERSHOOT, and it is unbounded: as y_i → 0,
#       B_i → x_i Σ_j w_j x_j y_j, so y_new → s_i/(x_i Σ_j w_j x_j y_j) = O(1/(x²y)).
#
# Any cold start has small x and y, so the strength step overshoots by orders of
# magnitude and lands outside t < 1 immediately. Measured (§2): on the symmetrised
# rhesus network the first strength step overshoots its own root by 208x-1383x on
# EVERY class. There is no slack to absorb it — the optimum there sits at
# max y_i y_j = 0.974, i.e. 97% of the way to the domain wall.
#
# Over 150 random weighted networks and 150 random weighted digraphs, the
# Picard/Anderson path converged on 0 of each.
#
# ---------------------------------------------------------------------------
# The replacement
# ---------------------------------------------------------------------------
# Do not freeze anything: solve each block EXACTLY. Both blocks are monotone, so the
# one-dimensional problems are unconditionally well posed (§3):
#
#   ⟨k_i⟩(x_i) rises from 0 to Σ_j w_j, so a root exists iff d_i is under that ceiling
#     (it is not for a saturated degree — a genuine runaway) and is then unique;
#   ⟨s_i⟩(y_i) rises from 0 to ∞ on (0, ȳ_i) with ȳ_i = min_j 1/y_j, because
#     d/dt[g t/D] = g(1-(1-g)t²)/D² > 0 for every t ∈ (0,1) and every g > 0, and
#     D(t) → 0 as t → 1. So the strength root ALWAYS exists, is unique, and is
#     FEASIBLE BY CONSTRUCTION — precisely the property the Picard step lacked.
#
# A safeguarded 2x2 Newton polish on (log x_i, log y_i) then follows runaway ridges
# that alternating 1-D moves only crawl along (§4).
#
# Run:  julia --project=validation validation/symbolic/uecm_decm_fixedpoint.jl
###############################################################################

using Symbolics
include(joinpath(@__DIR__, "common.jl"))
using MaxEntropyGraphs
const MEG = MaxEntropyGraphs
const SWG = MEG.SimpleWeightedGraphs
const G_  = MEG.Graphs

# ===========================================================================
# 1. the monotonicity asymmetry, symbolically
# ===========================================================================
@variables g t u c d₀

# degree channel: the factor Picard freezes is A(u) = c/(d₀ + c·u)
let
    A  = c/(d₀ + c*u)
    dA = Symbolics.derivative(A, u)
    # dA = -c²/(d₀+cu)² : prove dA·(d₀+cu)² + c² ≡ 0, hence dA < 0 for c,d₀,u > 0
    verify("UECM/DECM: the degree Picard factor A(x_i) has ∂A/∂x_i = -c²/(d₀+cx_i)² < 0 (⇒ the step UNDERSHOOTS)",
           dA*(d₀ + c*u)^2 + c^2, [c => (1, 4), d₀ => (1, 4), u => (1, 4)])
end

# strength channel: ⟨s⟩-term = g·t/D(t) with D(t) = (1-t)(1-t+g·t)
let
    D = (1-t)*(1-t+g*t)
    # (a) strict monotonicity of ⟨s⟩ in t:  d/dt[g t/D] = g(D - t·D')/D²  and  D - t·D' = 1-(1-g)t²
    verify("UECM/DECM: D(t) - t·D'(t) ≡ 1 - (1-g)t² > 0 on t∈(0,1) (⇒ ⟨s_i⟩ strictly INCREASING, unique root)",
           (D - t*Symbolics.derivative(D, t)) - (1 - (1-g)*t^2), [g => (1, 4), t => (1//8, 7//8)])
    # (b) the factor Picard freezes is B ∝ g·y_j/D(t); D is DEcreasing at the origin iff D'(0) = g-2 < 0
    verify("UECM/DECM: D'(0) ≡ g - 2, so for x_i x_j < 2 the strength Picard factor B(y_i) INCREASES (⇒ the step OVERSHOOTS)",
           Symbolics.substitute(Symbolics.derivative(D, t), Dict(t => 0)) - (g - 2), [g => (1, 4)])
    # (c) D(t) → 0 as t → 1 ⇒ ⟨s_i⟩ → ∞ at the domain wall, so the root is always bracketed
    boolcheck("UECM/DECM: D(1) = 0, so ⟨s_i⟩ → ∞ at the wall and the strength root is always bracketed",
              iszero(Symbolics.value(Symbolics.substitute(D, Dict(t => 1)))))
end

# ===========================================================================
# 2. the measured consequence: the first strength step leaves the domain
# ===========================================================================
# Reference implementations of the reduced UECM pair quantities, independent of src/.
p_(xi, xj, yi, yj) = (xi*xj*yi*yj)/(1 - yi*yj + xi*xj*yi*yj)
kexp(i, xi, x, y, F) = sum((F[j]-(i==j)) * (j==i ? p_(xi,xi,y[i],y[j]) : p_(xi,x[j],y[i],y[j]))
                           for j in eachindex(x) if F[j]-(i==j) != 0; init=0.0)
sexp(i, yi, x, y, F) = sum((F[j]-(i==j)) * (j==i ? p_(x[i],x[i],yi,yi)/(1-yi*yi) : p_(x[i],x[j],yi,y[j])/(1-yi*y[j]))
                           for j in eachindex(x) if F[j]-(i==j) != 0; init=0.0)
ybar_(i, y, F) = minimum(j==i ? 1.0 : 1/y[j] for j in eachindex(y) if F[j]-(i==j) != 0)
function bisect_(φ, lo, hi, target)
    for _ in 1:300
        mid = (lo+hi)/2; (mid == lo || mid == hi) && break
        v = φ(mid); (isfinite(v) && v < target) ? (lo = mid) : (hi = mid)
    end
    (lo+hi)/2
end

let
    m = UECM(SWG.SimpleWeightedGraph(MEG.rhesus_macaques()))
    n = length(m.dᵣ); F = m.f; d = m.dᵣ; s = m.sᵣ
    θ₀ = MEG.initial_guess(m); θ₀[isinf.(θ₀)] .= 0.0
    x = exp.(-θ₀[1:n]); y = exp.(-θ₀[n+1:end])

    # the start is comfortably feasible
    boolcheck("UECM rhesus: the :strengths cold start is inside the domain (max y_i y_j << 1)",
              maximum(y[i]*y[j] for i in 1:n, j in 1:n if F[j]-(i==j) != 0) < 0.1)

    overshoot = Float64[]; undershoot_ok = true; left_domain = 0
    for i in 1:n
        d[i] == 0 && continue
        A = 0.0; B = 0.0
        for j in 1:n
            w = F[j]-(i==j); w == 0 && continue
            c1 = x[i]*x[j]; c2 = y[i]*y[j]; den = 1 - c2 + c1*c2
            A += w*(x[j]*c2)/den
            B += w*(c1*y[j])/((1-c2)*den)
        end
        xnew = d[i]/A; ynew = s[i]/B
        # exact roots of the two one-dimensional problems
        hi = max(x[i], 1e-8); while kexp(i,hi,x,y,F) < d[i] && hi < 1e30; hi *= 4; end
        xstar = bisect_(uu -> kexp(i,uu,x,y,F), 0.0, hi, float(d[i]))
        yb = ybar_(i, y, F)
        ystar = bisect_(vv -> sexp(i,vv,x,y,F), yb*1e-14, yb*(1-1e-14), float(s[i]))
        undershoot_ok &= (x[i] <= xnew <= xstar*(1+1e-8))
        push!(overshoot, ynew/ystar)
        ynew >= yb && (left_domain += 1)
    end
    boolcheck("UECM rhesus: the Picard DEGREE step lands in (x_i, x_i*) on every live class (safe undershoot)",
              undershoot_ok)
    boolcheck("UECM rhesus: the Picard STRENGTH step overshoots its own root by >100x on every live class",
              minimum(overshoot) > 100)
    closecheck("UECM rhesus: worst measured strength overshoot factor ≈ 1383", maximum(overshoot), 1383.0; rtol=0.05)
    boolcheck("UECM rhesus: EVERY live class leaves the domain (y_new ≥ ȳ_i) on the FIRST iteration",
              left_domain == count(!iszero, d))
end

# the optimum itself sits right against the wall — there was never any slack
let
    m = UECM(SWG.SimpleWeightedGraph(MEG.rhesus_macaques()))
    solve_model!(m, method = :fixedpoint)
    n = length(m.dᵣ); y = m.yᵣ
    wall = maximum(y[i]*y[j] for i in 1:n, j in 1:n if m.f[j]-(i==j) != 0)
    closecheck("UECM rhesus: the ML optimum sits at max y_i y_j ≈ 0.974 — 97% of the way to the domain wall",
               wall, 0.9739; rtol=1e-3)
    boolcheck("UECM rhesus: ... and is still strictly inside it", wall < 1)
end

# ===========================================================================
# 3. the replacement map: feasible by construction, and it agrees with :BFGS
# ===========================================================================
let
    g = SWG.SimpleWeightedGraph(MEG.rhesus_macaques())
    mf = UECM(g); solve_model!(mf, method = :fixedpoint)
    mb = UECM(g); solve_model!(mb, method = :BFGS)
    A = SWG.weights(g)
    MEG.set_Ĝ!(mf); MEG.set_Ŵ!(mf)
    rk = maximum(abs, vec(sum(mf.Ĝ, dims=2)) .- G_.degree(g))
    rs = maximum(abs, vec(sum(mf.Ŵ, dims=2)) .- vec(sum(A, dims=2)))
    boolcheck("UECM rhesus: :fixedpoint reproduces the degree sequence to < 1e-7", rk < 1e-7)
    boolcheck("UECM rhesus: :fixedpoint reproduces the strength sequence to < 1e-7", rs < 1e-7)
    boolcheck("UECM rhesus: :fixedpoint and :BFGS agree on θ to < 1e-6 (the UECM has no gauge freedom)",
              maximum(abs, mf.θᵣ .- mb.θᵣ) < 1e-6)
end
let
    g = MEG.rhesus_macaques()
    mf = DECM(g); solve_model!(mf, method = :fixedpoint)
    mb = DECM(g); solve_model!(mb, method = :BFGS)
    A = SWG.weights(g)
    MEG.set_Ĝ!(mf); MEG.set_Ŵ!(mf); MEG.set_Ĝ!(mb); MEG.set_Ŵ!(mb)
    r = max(maximum(abs, vec(sum(mf.Ĝ,dims=2)) .- G_.outdegree(g)),
            maximum(abs, vec(sum(mf.Ĝ,dims=1)) .- G_.indegree(g)),
            maximum(abs, vec(sum(mf.Ŵ,dims=2)) .- vec(sum(A,dims=2))),
            maximum(abs, vec(sum(mf.Ŵ,dims=1)) .- vec(sum(A,dims=1))))
    boolcheck("DECM rhesus: :fixedpoint reproduces all four constrained sequences to < 1e-7", r < 1e-7)
    # the DECM HAS a two-fold gauge freedom, so compare gauge-INVARIANT quantities only
    boolcheck("DECM rhesus: :fixedpoint and :BFGS agree on Ĝ to < 1e-6 (gauge-invariant)",
              maximum(abs, mf.Ĝ .- mb.Ĝ) < 1e-6)
    boolcheck("DECM rhesus: :fixedpoint and :BFGS agree on Ŵ to < 1e-5 (gauge-invariant)",
              maximum(abs, mf.Ŵ .- mb.Ŵ) < 1e-5)
end

# ===========================================================================
# 4. the runaway ridge, and why the 2x2 polish is needed
# ===========================================================================
# rhesus has an out-class with k = s = 1: every one of its links carries weight exactly 1, so the
# optimum needs y_out → 0 AND x_out → ∞ together. Alternating 1-D solves can only approach such a
# joint limit geometrically (measured rate 0.999823/sweep ⇒ ~7e4 sweeps to 1e-9); the 2x2 Newton
# step follows it directly. This is a REGRESSION GUARD on that: the shipped solver must converge.
let
    m = DECM(MEG.rhesus_macaques())
    has_runaway = any(i -> m.sᵣ_out[i] == m.dᵣ_out[i], eachindex(m.dᵣ_out)) ||
                  any(i -> m.sᵣ_in[i]  == m.dᵣ_in[i],  eachindex(m.dᵣ_in))
    boolcheck("DECM rhesus: the network really does carry a runaway (s = k) constraint", has_runaway)
    ok = try (solve_model!(m, method = :fixedpoint); true) catch; false end
    boolcheck("DECM rhesus: :fixedpoint converges DESPITE the runaway (the 2x2 polish follows the ridge)", ok)
end

# ===========================================================================
# 5. the domain is pairwise, and a singleton class has no per-coordinate bound
# ===========================================================================
# L_UECM_reduced evaluates the same-class term for every class and weights it by F_i(F_i-1)/2.
# For a SINGLETON class that weight is zero — but the out-of-domain branch returns NaN, and
# `0 * NaN = NaN` poisoned the whole sum, making L non-finite on part of its own domain. That in
# turn is why the first-order box floored every β_i: the optimiser could not see past the NaN.
let
    d = [1, 2, 2]; s = [3, 5, 6]; F = [1, 1, 1]; n = 3
    α = [0.5, 0.4, 0.3]
    inside  = [-0.20, 0.5, 0.5]     # every EXISTING pair has β_i + β_j > 0; only β₁ itself is negative
    outside = [-0.51, 0.5, 0.5]     # β₁ + β₂ = -0.01 < 0 — genuinely out of the domain
    boolcheck("UECM: all pairs of the `inside` point satisfy β_i + β_j > 0 (it IS in the domain)",
              all(inside[i]+inside[j] > 0 for i in 1:n for j in 1:i-1))
    boolcheck("UECM: L is FINITE at a domain point with a negative β on a singleton class (0·NaN guarded)",
              isfinite(MEG.L_UECM_reduced(vcat(α, inside), d, s, F, n)))
    boolcheck("UECM: L is still NaN genuinely outside the domain (β_i + β_j < 0 on a real pair)",
              isnan(MEG.L_UECM_reduced(vcat(α, outside), d, s, F, n)))
    # a class of multiplicity F ≥ 2 DOES have a same-class pair, so its own β must stay positive
    F2 = [2, 1, 1]
    boolcheck("UECM: a class with F ≥ 2 has a same-class pair, so β_i < 0 IS out of the domain for it",
              isnan(MEG.L_UECM_reduced(vcat(α, inside), d, s, F2, n)))
end

report("UECM & DECM fixed point")
