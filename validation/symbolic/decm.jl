# DECM — directed enhanced configuration model (discrete Bernoulli–geometric channel weights).
#
# Derivation (Vallarano et al., Sci. Rep. 11 (2021) 15227, arXiv:2101.12625; per-channel
# distribution as in Parisi, Squartini & Garlaschelli, NJP 22 (2020) 053053):
# the Hamiltonian H = Σᵢ α_out,ᵢ·k_out,ᵢ + α_in,ᵢ·k_in,ᵢ + β_out,ᵢ·s_out,ᵢ + β_in,ᵢ·s_in,ᵢ
# factorizes per ORDERED pair (i→j). With the composite parameters
#     x = xᵢ_out·xⱼ_in = exp(-α_out,ᵢ - α_in,ⱼ) and y = yᵢ_out·yⱼ_in = exp(-β_out,ᵢ - β_in,ⱼ),
# the unnormalized channel weight for integer w ≥ 0 is
#     w = 0 → 1,      w ≥ 1 → x·yʷ,
# identical in form to the UECM dyad, so the channel partition function is Z = 1 + x·y/(1-y)
# and the probability generating function (rational in t, hence safe for Symbolics)
#     G(t) = ⟨tʷ⟩ = (1 + x·t·y/(1 - t·y)) / Z.
# All per-channel moments follow from G:
#     p      = P(w>0)  = 1 - G(0)          == x·y/(1 - y + x·y)      [f_DECM / Ĝ code form]
#     ⟨w⟩    = G'(1)                       == p/(1-y)                [Ŵ code form]
#     ⟨w²⟩   = G''(1) + G'(1)              == p(1+y)/(1-y)²
#     Var[w] = ⟨w²⟩ - ⟨w⟩²                 == p(1+y-p)/(1-y)²        [σʷ code form, squared]
#     ⟨a·w⟩  = ⟨w⟩   (w>0 ⟺ a=1)  ⇒  Cov(a,w) = ⟨w⟩(1-p)             [cross-layer]
#     Var[a] = p - p²                      == p(1-p)                 [σˣ code form, squared]
# The DIRECTED specificity: the two channels (i→j) and (j→i) of a dyad carry distinct
# composite parameters and the joint distribution factorizes, so Cov(w_ij, w_ji) = 0
# (unlike the UECM, where w_ij ≡ w_ji). This is what makes σₓ(m::DECM, ...) covariance-free.
#
# Code forms compared against (READ, not modified):
#   src/Models/DECM.jl:567      f_DECM(xixj,yiyj) = (x·y)/(1 - y + x·y)
#   src/Models/DECM.jl:592-627  Ĝ    → pᵢⱼ  (same kernel, inlined in the ordered-pair loop)
#   src/Models/DECM.jl:638-673  Ŵ    → wᵢⱼ = pᵢⱼ/(1 - yᵢ_out·yⱼ_in)
#   src/Models/DECM.jl:687-722  σˣ   → sqrt(pᵢⱼ(1 - pᵢⱼ))            (binary layer; compared squared)
#   src/Models/DECM.jl:738-773  σʷ   → sqrt(pᵢⱼ(1+y-pᵢⱼ))/(1-y)      (weight layer; compared squared)
#
# All identities are sqrt-free rational functions; `verify` settles them with exact
# Rational{BigInt} multi-point substitution. Numeric closure pins the symbolic forms to
# the actual package kernels (f_DECM / Ĝ / Ŵ / σˣ / σʷ) on a solved model, and to a
# truncated-sum oracle Σ_{w=0}^{5000} wᵏ·P(w).

include(joinpath(@__DIR__, "common.jl"))

using MaxEntropyGraphs

# ---------------------------------------------------------------------------
# Symbolic derivation via the probability generating function
# ---------------------------------------------------------------------------

@variables x y t x1 y1 x2 y2 s

doms = [x => (1//10, 5), y => (1//100, 9//10)]

# channel partition function and PGF
Z  = 1 + x*y/(1 - y)
Gt = (1 + x*t*y/(1 - t*y)) / Z

# (1) connection probability p = P(w>0) = 1 - G(0) vs the f_DECM code form
p_pgf  = 1 - Symbolics.substitute(Gt, Dict(t => 0))
p_code = (x*y) / (1 - y + x*y)                       # f_DECM, src/Models/DECM.jl:567
verify("(1) p = 1 - G(0) == x·y/(1-y+x·y)   [f_DECM / Ĝ code form]", p_pgf - p_code, doms)

# (2) expected weight ⟨w⟩ = G'(1) vs the Ŵ code form p/(1-y)
dG  = Symbolics.derivative(Gt, t)
w1  = Symbolics.substitute(dG, Dict(t => 1))
w1_code = p_code / (1 - y)                           # Ŵ, src/Models/DECM.jl:638-673
verify("(2) ⟨w⟩ = G'(1) == p/(1-y)          [Ŵ code form]", w1 - w1_code, doms)

# (3) second moment ⟨w²⟩ = G''(1) + G'(1)
d2G = Symbolics.derivative(dG, t)
w2  = Symbolics.substitute(d2G, Dict(t => 1)) + w1
w2_closed = p_code*(1 + y) / (1 - y)^2
verify("(3) ⟨w²⟩ = G''(1)+G'(1) == p(1+y)/(1-y)²", w2 - w2_closed, doms)

# (4) weight-layer variance vs the σʷ code form squared
varw      = w2 - w1^2
varw_code = p_code*(1 + y - p_code) / (1 - y)^2      # σʷ², src/Models/DECM.jl:738-773
verify("(4) Var[w] = ⟨w²⟩-⟨w⟩² == p(1+y-p)/(1-y)²   [σʷ code form²]", varw - varw_code, doms)

# (5) cross-layer covariance: ⟨a·w⟩ = Σ_{w≥1} w·x·yʷ/Z = x·y/((1-y)²·Z) equals ⟨w⟩
#     (w>0 ⟺ a=1), hence Cov(a,w) = ⟨a·w⟩ - p·⟨w⟩ = ⟨w⟩(1-p)
aw = x*y / ((1 - y)^2 * Z)
verify("(5a) ⟨a·w⟩ = x·y/((1-y)²Z) == ⟨w⟩   (w>0 ⟺ a=1)", aw - w1, doms)
verify("(5b) Cov(a,w) = ⟨a·w⟩ - p⟨w⟩ == ⟨w⟩(1-p)", (aw - p_code*w1) - w1_code*(1 - p_code), doms)

# (6) binary-layer variance: Var[a] = p - p² vs the σˣ code form squared
vara = p_pgf - p_pgf^2
verify("(6) Var[a] = p - p² == p(1-p)       [σˣ code form²]", vara - p_code*(1 - p_code), doms)

# (7) DIRECTED independence across the two channels of a dyad: the joint PGF factorizes
#     over distinct composite parameters (x1,y1) for i→j and (x2,y2) for j→i, so
#     ⟨w_ij·w_ji⟩ = ⟨w_ij⟩⟨w_ji⟩, i.e. Cov(w_ij, w_ji) = 0. This is the code-level basis
#     for the covariance-free delta method in σₓ(m::DECM, ...) — the defining difference
#     with the UECM (where w_ij ≡ w_ji and the cross-term doubles the aggregate variance).
doms2 = [x1 => (1//10, 5), y1 => (1//100, 9//10),
         x2 => (1//10, 5), y2 => (1//100, 9//10)]
G1(u) = (1 + x1*u*y1/(1 - u*y1)) / (1 + x1*y1/(1 - y1))
G2(u) = (1 + x2*u*y2/(1 - u*y2)) / (1 + x2*y2/(1 - y2))
Gjoint = G1(s) * G2(t)
E_prod = Symbolics.substitute(Symbolics.derivative(Symbolics.derivative(Gjoint, s), t), Dict(s => 1, t => 1))
E_wij  = Symbolics.substitute(Symbolics.derivative(G1(s), s), Dict(s => 1))
E_wji  = Symbolics.substitute(Symbolics.derivative(G2(t), t), Dict(t => 1))
verify("(7) Cov(w_ij, w_ji) = 0             [directed channel independence]", E_prod - E_wij*E_wji, doms2)

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

closecheck("pkg f_DECM(0.7,0.4) == x·y/(1-y+x·y)", MaxEntropyGraphs.f_DECM(0.7, 0.4), p_cl(0.7, 0.4))

model = DECM(MaxEntropyGraphs.rhesus_macaques())   # directed weighted anchor, used unsymmetrised
solve_model!(model, method=:BFGS, g_tol=1e-5)

Ge = MaxEntropyGraphs.Ĝ(model)
We = MaxEntropyGraphs.Ŵ(model)
Se = MaxEntropyGraphs.σˣ(model)
Sw = MaxEntropyGraphs.σʷ(model)
xov = model.xᵣ_out[model.dᵣ_ind]   # parameter access pattern from src/Models/DECM.jl:600-604
xiv = model.xᵣ_in[model.dᵣ_ind]
yov = model.yᵣ_out[model.dᵣ_ind]
yiv = model.yᵣ_in[model.dᵣ_ind]

for (i, j) in ((1, 2), (3, 7), (5, 11), (11, 5)) # includes both orientations of a dyad
    xij, yij = xov[i]*xiv[j], yov[i]*yiv[j]
    p = p_cl(xij, yij)
    closecheck("pkg Ĝ[$i,$j]  == p", Ge[i, j], p)
    closecheck("pkg Ŵ[$i,$j]  == p/(1-y)", We[i, j], w1_cl(xij, yij))
    closecheck("pkg σˣ[$i,$j]² == p(1-p)", Se[i, j]^2, p*(1 - p))
    closecheck("pkg σʷ[$i,$j]² == p(1+y-p)/(1-y)²", Sw[i, j]^2, var_cl(xij, yij))
end

# ---------------------------------------------------------------------------
report("DECM (directed Bernoulli–geometric, PGF)")
