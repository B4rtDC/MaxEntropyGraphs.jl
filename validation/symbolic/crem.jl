# CReM — symbolic validation of the per-dyad moments (undirected conditional reconstruction
# model, Bernoulli topology layer + conditionally exponential weights).
#
# Derivation route: moment generating function of the *unconditional* weight
#       w = 0                     with probability (1 - f)
#       w ~ Exponential(rate ts)  with probability f,        ts = θᵢ + θⱼ
#   =>  M(t) = (1 - f) + f · ts / (ts - t)          (rational in t, valid for t < ts)
# so every moment is a derivative of a rational function and all identities below stay
# sqrt-free (variances / squared code forms only, per validation/symbolic/common.jl).
#
# Code forms validated against:
#   ⟨w⟩  = fᵢⱼ/(θᵢ+θⱼ)                        src/Models/CReM.jl:471-489   (Ŵ)
#   σˣ   = sqrt(fᵢⱼ(1-fᵢⱼ))                   src/Models/CReM.jl:501-518   (binary layer)
#   σʷ   = sqrt(fᵢⱼ(2-fᵢⱼ))/(θᵒᵢ+θⁱⱼ)         src/Models/DCReM.jl:632-652  (directed twin;
#         the undirected CReM does not ship a σʷ yet — docstring gap, same algebra)
# with fᵢⱼ = xᵢxⱼ/(1+xᵢxⱼ) (undirected UBCM prior).

include(joinpath(@__DIR__, "common.jl"))

using MaxEntropyGraphs
using Distributions: Exponential, mean, var

# ---------------------------------------------------------------------------------------
# Symbolic derivation via the MGF
# ---------------------------------------------------------------------------------------
@variables f ti tj t xi xj

ts = ti + tj
M  = (1 - f) + f * ts / (ts - t)                       # MGF of the zero-inflated exponential

Mp  = Symbolics.derivative(M, t)                       # M'(t)
Mpp = Symbolics.derivative(Mp, t)                      # M''(t)
meanw = Symbolics.substitute(Mp,  Dict(t => 0))        # ⟨w⟩  = M'(0)
m2w   = Symbolics.substitute(Mpp, Dict(t => 0))        # ⟨w²⟩ = M''(0)
Varw  = m2w - meanw^2                                  # Var[w]

dom_f  = f  => (1//100, 99//100)
dom_ti = ti => (1//10, 10)
dom_tj = tj => (1//10, 10)
dom_xi = xi => (1//100, 20)
dom_xj = xj => (1//100, 20)

fij = xi * xj / (1 + xi * xj)                          # binary-layer probability (UBCM prior)

# (1) mean: M'(0) == f/ts, and — with f = xᵢxⱼ/(1+xᵢxⱼ) — the exact Ŵ kernel of the code
verify("⟨w⟩ = M'(0) == f/(θi+θj)",
       meanw - f / ts, [dom_f, dom_ti, dom_tj])
W_code = (xi * xj / (1 + xi * xj)) / (ti + tj)         # src/Models/CReM.jl:480-482
verify("⟨w⟩ at f=xixj/(1+xixj) == Ŵ code form fij/(θi+θj)",
       Symbolics.substitute(meanw, Dict(f => fij)) - W_code,
       [dom_xi, dom_xj, dom_ti, dom_tj])

# (2) second moment: M''(0) == 2f/ts²
verify("⟨w²⟩ = M''(0) == 2f/(θi+θj)²",
       m2w - 2 * f / ts^2, [dom_f, dom_ti, dom_tj])

# (3) proposed weight variance: Var[w] == f(2-f)/ts², i.e. exactly the (squared) directed
#     code form sqrt(fij(2-fij))/(θᵒi+θⁱj) shipped in src/Models/DCReM.jl:646
verify("Var[w] = M''(0) - M'(0)² == f(2-f)/(θi+θj)²",
       Varw - f * (2 - f) / ts^2, [dom_f, dom_ti, dom_tj])
σw²_code = fij * (2 - fij) / (ti + tj)^2               # (sqrt(fij(2-fij))/(θi+θj))², DCReM.jl:646
verify("Var[w] at f=xixj/(1+xixj) == squared DCReM σʷ code form",
       Symbolics.substitute(Varw, Dict(f => fij)) - σw²_code,
       [dom_xi, dom_xj, dom_ti, dom_tj])

# (4) cross-layer covariance: a·w = w (weights vanish off-edge) so ⟨aw⟩ = ⟨w⟩ and
#     Cov(a,w) = ⟨w⟩ - ⟨a⟩⟨w⟩ = ⟨w⟩(1-f) = f(1-f)/ts
Cov_aw = meanw - f * meanw
verify("Cov(a,w) = ⟨w⟩(1-f) == f(1-f)/(θi+θj)",
       Cov_aw - f * (1 - f) / ts, [dom_f, dom_ti, dom_tj])

# (5) binary layer: Var[a] = f(1-f) == xixj/(1+xixj)², the squared σˣ code form
#     (src/Models/CReM.jl:511, sqrt(pij(1-pij)) with pij = xixj/(1+xixj))
verify("Var[a] = fij(1-fij) == xixj/(1+xixj)² (squared σˣ code form)",
       fij * (1 - fij) - xi * xj / (1 + xi * xj)^2, [dom_xi, dom_xj])

# (6) undirected convention: wᵢⱼ and wⱼᵢ are the *same* random variable, hence
#     Cov(wᵢⱼ, wⱼᵢ) = Var[w] by definition (recorded as a convention check)
verify("convention: undirected ⇒ Cov(w_ij, w_ji) = Var[w] (same variable)",
       Varw - Varw, [dom_f, dom_ti, dom_tj])

# ---------------------------------------------------------------------------------------
# Numeric closure 1 — Distributions.jl / quadrature oracle for the zero-inflated exponential
# ---------------------------------------------------------------------------------------
# Composite Simpson quadrature of g on [0, hi] (n even), used as an oracle independent of
# both the MGF algebra above and Distributions' moment formulas.
function simpson(g, hi, n=20_000)
    h = hi / n
    s = g(0.0) + g(hi)
    for k in 1:n-1
        s += (isodd(k) ? 4 : 2) * g(k * h)
    end
    return s * h / 3
end

for (fv, tsv) in ((0.2, 0.7), (0.5, 1.3), (0.9, 2.5))
    # Distributions.jl moments of the exponential component (scale parameterisation: 1/rate)
    E = Exponential(1 / tsv)
    m1_dist = fv * mean(E)                             # zero-inflated mean
    m2_dist = fv * (var(E) + mean(E)^2)                # zero-inflated second moment
    closecheck("oracle(Distributions) ⟨w⟩  == f/ts        (f=$fv, ts=$tsv)", m1_dist, fv / tsv)
    closecheck("oracle(Distributions) ⟨w²⟩ == 2f/ts²      (f=$fv, ts=$tsv)", m2_dist, 2 * fv / tsv^2)
    closecheck("oracle(Distributions) Var  == f(2-f)/ts²  (f=$fv, ts=$tsv)",
               m2_dist - m1_dist^2, fv * (2 - fv) / tsv^2)
    # independent quadrature of the density fv · ts·exp(-ts·w) plus the atom at 0
    hi = 60 / tsv
    m1_quad = fv * simpson(w -> w * tsv * exp(-tsv * w), hi)
    m2_quad = fv * simpson(w -> w^2 * tsv * exp(-tsv * w), hi)
    closecheck("oracle(quadrature)    ⟨w⟩  == f/ts        (f=$fv, ts=$tsv)", m1_quad, fv / tsv; rtol=1e-6)
    closecheck("oracle(quadrature)    ⟨w²⟩ == 2f/ts²      (f=$fv, ts=$tsv)", m2_quad, 2 * fv / tsv^2; rtol=1e-6)
end

# ---------------------------------------------------------------------------------------
# Numeric closure 2 — pin the symbolic forms to the actual package kernels
# ---------------------------------------------------------------------------------------
# Two-step solve exactly as in test/models.jl ("CReM - parameter computation"): the binary
# UBCM layer on the degrees, then the weighted layer on the strengths (default fixedpoint,
# deterministic :degrees/:strengths initial guesses — no RNG involved).
Gw = MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(MaxEntropyGraphs.rhesus_macaques())
model = MaxEntropyGraphs.CReM(Gw)
MaxEntropyGraphs.solve_model!(model)

x = model.xᵣ[model.dᵣ_ind]                             # per-node binary fitness (code accessor)
θ = model.θ                                            # per-node weighted-layer parameter
W = MaxEntropyGraphs.Ŵ(model)
G = MaxEntropyGraphs.Ĝ(model)
σ = MaxEntropyGraphs.σˣ(model)

for (i, j) in ((1, 2), (3, 7), (5, 16))
    fnum = x[i] * x[j] / (1 + x[i] * x[j])
    tsn  = θ[i] + θ[j]
    closecheck("package Ĝ[$i,$j]   == fij", G[i, j], fnum)
    closecheck("package Ŵ[$i,$j]   == fij/(θi+θj)", W[i, j], fnum / tsn)
    closecheck("package σˣ[$i,$j]² == fij(1-fij)", σ[i, j]^2, fnum * (1 - fnum))
end

report("CReM")
