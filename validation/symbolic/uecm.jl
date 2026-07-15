# UECM — undirected enhanced configuration model (discrete Bernoulli–geometric dyad weights).
#
# Derivation (Squartini & Garlaschelli, NJP 13 (2011) 083001, App. B):
# the Hamiltonian H = Σᵢ αᵢkᵢ + βᵢsᵢ factorizes per dyad. With x = xᵢxⱼ = exp(-αᵢ-αⱼ)
# and y = yᵢyⱼ = exp(-βᵢ-βⱼ), the unnormalized dyad weight for integer w ≥ 0 is
#     w = 0 → 1,      w ≥ 1 → x·yʷ,
# so the dyadic partition function is Z = 1 + x·y/(1-y) and the probability
# generating function (rational in t, hence safe for Symbolics)
#     G(t) = ⟨tʷ⟩ = (1 + x·t·y/(1 - t·y)) / Z.
# All per-dyad moments follow from G:
#     p      = P(w>0)  = 1 - G(0)          == x·y/(1 - y + x·y)      [f_UECM / Ĝ code form]
#     ⟨w⟩    = G'(1)                       == p/(1-y)                [Ŵ code form]
#     ⟨w²⟩   = G''(1) + G'(1)              == p(1+y)/(1-y)²
#     Var[w] = ⟨w²⟩ - ⟨w⟩²                 == p(1+y-p)/(1-y)²        [PROPOSED σʷ form]
#     ⟨a·w⟩  = ⟨w⟩   (w>0 ⟺ a=1)  ⇒  Cov(a,w) = ⟨w⟩(1-p)             [cross-layer]
#     Var[a] = p - p²                      == p(1-p)                 [σˣ code form, squared]
#
# Code forms compared against (READ, not modified):
#   src/Models/UECM.jl:429      f_UECM(xixj,yiyj) = (x·y)/(1 - y + x·y)
#   src/Models/UECM.jl:453-476  Ĝ    → pᵢⱼ  (same kernel, inlined in the loop)
#   src/Models/UECM.jl:496-520  Ŵ    → wᵢⱼ = pᵢⱼ/(1 - yᵢyⱼ)
#   src/Models/UECM.jl:532-555  σˣ   → sqrt(pᵢⱼ(1 - pᵢⱼ))  (binary layer; compared squared)
# There is no σʷ (weight-layer variance) in src yet — the PROPOSED form above is
# validated purely against the PGF derivation and the truncated-sum oracle.
#
# All identities are sqrt-free rational functions; `verify` settles them with exact
# Rational{BigInt} multi-point substitution. Numeric closure pins the symbolic forms to
# the actual package kernels (f_UECM / Ĝ / Ŵ / σˣ) on a solved model, and to a
# truncated-sum oracle Σ_{w=0}^{5000} wᵏ·P(w).

include(joinpath(@__DIR__, "common.jl"))

using MaxEntropyGraphs

# ---------------------------------------------------------------------------
# Symbolic derivation via the probability generating function
# ---------------------------------------------------------------------------

@variables x y t

doms = [x => (1//10, 5), y => (1//100, 9//10)]

# dyadic partition function and PGF
Z  = 1 + x*y/(1 - y)
Gt = (1 + x*t*y/(1 - t*y)) / Z

# (1) connection probability p = P(w>0) = 1 - G(0) vs the f_UECM code form
p_pgf  = 1 - Symbolics.substitute(Gt, Dict(t => 0))
p_code = (x*y) / (1 - y + x*y)                       # f_UECM, src/Models/UECM.jl:429
verify("(1) p = 1 - G(0) == x·y/(1-y+x·y)   [f_UECM / Ĝ code form]", p_pgf - p_code, doms)

# (2) expected weight ⟨w⟩ = G'(1) vs the Ŵ code form p/(1-y)
dG  = Symbolics.derivative(Gt, t)
w1  = Symbolics.substitute(dG, Dict(t => 1))
w1_code = p_code / (1 - y)                           # Ŵ, src/Models/UECM.jl:496-520
verify("(2) ⟨w⟩ = G'(1) == p/(1-y)          [Ŵ code form]", w1 - w1_code, doms)

# (3) second moment ⟨w²⟩ = G''(1) + G'(1)
d2G = Symbolics.derivative(dG, t)
w2  = Symbolics.substitute(d2G, Dict(t => 1)) + w1
w2_closed = p_code*(1 + y) / (1 - y)^2
verify("(3) ⟨w²⟩ = G''(1)+G'(1) == p(1+y)/(1-y)²", w2 - w2_closed, doms)

# (4) PROPOSED weight-layer variance form (planned for implementation; no σʷ in src yet)
varw      = w2 - w1^2
varw_prop = p_code*(1 + y - p_code) / (1 - y)^2
verify("(4) PROPOSED Var[w] = ⟨w²⟩-⟨w⟩² == p(1+y-p)/(1-y)²", varw - varw_prop, doms)

# (5) cross-layer covariance: ⟨a·w⟩ = Σ_{w≥1} w·x·yʷ/Z = x·y/((1-y)²·Z) equals ⟨w⟩
#     (w>0 ⟺ a=1), hence Cov(a,w) = ⟨a·w⟩ - p·⟨w⟩ = ⟨w⟩(1-p)
aw = x*y / ((1 - y)^2 * Z)
verify("(5a) ⟨a·w⟩ = x·y/((1-y)²Z) == ⟨w⟩   (w>0 ⟺ a=1)", aw - w1, doms)
verify("(5b) Cov(a,w) = ⟨a·w⟩ - p⟨w⟩ == ⟨w⟩(1-p)", (aw - p_code*w1) - w1_code*(1 - p_code), doms)

# (6) binary-layer variance: Var[a] = p - p² vs the σˣ code form squared
vara = p_pgf - p_pgf^2
verify("(6) Var[a] = p - p² == p(1-p)       [σˣ code form²]", vara - p_code*(1 - p_code), doms)

# ---------------------------------------------------------------------------
# Numeric closure 1 — truncated-sum oracle (Float64)
# ---------------------------------------------------------------------------
# P(0) = 1/Z, P(w) = x·yʷ/Z; sum wᵏ·P(w) up to w = 5000 (truncation error ≪ rtol for y ≤ 0.7).

p_cl(xv, yv)  = xv*yv / (1 - yv + xv*yv)
w1_cl(xv, yv) = p_cl(xv, yv) / (1 - yv)
w2_cl(xv, yv) = p_cl(xv, yv) * (1 + yv) / (1 - yv)^2
var_cl(xv, yv) = (p = p_cl(xv, yv); p * (1 + yv - p) / (1 - yv)^2)

function oracle_moments(xv::Float64, yv::Float64; wmax::Int=5000)
    Zv = 1 + xv*yv/(1 - yv)
    s0, s1, s2 = 0.0, 0.0, 0.0
    for w in 1:wmax
        P   = xv * yv^w / Zv
        s0 += P
        s1 += w * P
        s2 += w^2 * P
    end
    return s0, s1, s2          # P(w>0), ⟨w⟩, ⟨w²⟩ (truncated)
end

for (xv, yv) in ((0.5, 0.3), (2.0, 0.7), (1.2, 0.05))
    s0, s1, s2 = oracle_moments(xv, yv)
    closecheck("oracle p      @ (x=$xv, y=$yv)", s0,          p_cl(xv, yv))
    closecheck("oracle ⟨w⟩    @ (x=$xv, y=$yv)", s1,          w1_cl(xv, yv))
    closecheck("oracle ⟨w²⟩   @ (x=$xv, y=$yv)", s2,          w2_cl(xv, yv))
    closecheck("oracle Var[w] @ (x=$xv, y=$yv)", s2 - s1^2,   var_cl(xv, yv))
end

# ---------------------------------------------------------------------------
# Numeric closure 2 — pin the closed forms to the actual package kernels
# ---------------------------------------------------------------------------
# Deterministic: fixed dataset, deterministic :strengths initial guess, BFGS solve.

closecheck("pkg f_UECM(0.7,0.4) == x·y/(1-y+x·y)", MaxEntropyGraphs.f_UECM(0.7, 0.4), p_cl(0.7, 0.4))

model = UECM(MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(MaxEntropyGraphs.rhesus_macaques()))
solve_model!(model, method=:BFGS, g_tol=1e-5)

Ge = MaxEntropyGraphs.Ĝ(model)
We = MaxEntropyGraphs.Ŵ(model)
Se = MaxEntropyGraphs.σˣ(model)
xv = model.xᵣ[model.dᵣ_ind]     # parameter access pattern from src/Models/UECM.jl:462-463
yv = model.yᵣ[model.dᵣ_ind]

for (i, j) in ((1, 2), (3, 7), (5, 11))
    xij, yij = xv[i]*xv[j], yv[i]*yv[j]
    p = p_cl(xij, yij)
    closecheck("pkg Ĝ[$i,$j]  == p", Ge[i, j], p)
    closecheck("pkg Ŵ[$i,$j]  == p/(1-y)", We[i, j], w1_cl(xij, yij))
    closecheck("pkg σˣ[$i,$j]² == p(1-p)", Se[i, j]^2, p*(1 - p))
end

# ---------------------------------------------------------------------------
report("UECM (Bernoulli–geometric, PGF)")
