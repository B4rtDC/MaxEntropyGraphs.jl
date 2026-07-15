# DCReM — directed conditional reconstruction model: per-dyad moment derivation via the MGF.
#
# Per ORDERED pair (i,j) the (unconditional) weight is a zero-inflated exponential:
#     w_ij = 0                              with probability 1 - f_ij,
#     w_ij ~ Exponential(rate θᵒᵢ + θⁱⱼ)    with probability f_ij,
# where f_ij = x_i y_j / (1 + x_i y_j) is the marginal edge probability of the conditional
# (binary) DBCM layer. Writing ts = θᵒᵢ + θⁱⱼ (the rate sum as it appears in Ŵ,
# src/Models/DCReM.jl:546-565, `W[i,j] = (xiyj/(1+xiyj)) / (θᵒ[i] + θⁱ[j])`), the MGF is
#     M(s) = ⟨e^{s w}⟩ = (1 - f) + f · ts/(ts - s),          (s < ts)
# from which the moments follow by differentiation at s = 0:
#     ⟨w⟩   = M'(0)  = f/ts,
#     ⟨w²⟩  = M''(0) = 2f/ts²,
#     Var[w] = ⟨w²⟩ - ⟨w⟩² = f(2 - f)/ts²,
# matching the σʷ code form  σ[i,j] = sqrt(fij(2 - fij))/(θᵒᵢ + θⁱⱼ)  [src/Models/DCReM.jl:632-652]
# (compared squared to stay sqrt-free).
#
# The two directions of a dyad are INDEPENDENT: the conditional likelihood L_DCReM factorizes
# over ordered pairs (each term couples only θᵒᵢ + θⁱⱼ with f_ij; see L_DCReM and the σʷ
# docstring "the weights of distinct ordered pairs are independent (Cov(wᵢⱼ, wⱼᵢ) = 0)").
# Accordingly the joint MGF factorizes, M(s,t) = M_ij(s)·M_ji(t), and
#     ∂²M/∂s∂t |₀ = ⟨w_ij w_ji⟩ = ⟨w_ij⟩⟨w_ji⟩   ⟹   Cov(w_ij, w_ji) = 0.
#
# The binary layer entries are Bernoulli(f): Var[a_ij] = f(1 - f), matching the σˣ code form
# σ[i,j] = sqrt(pij(1 - pij)) [src/Models/DCReM.jl:589-607].

include(joinpath(@__DIR__, "common.jl"))

@variables xi xj yi yj      # binary-layer fitnesses (f-generating)
@variables toi tii toj tij  # rates: toi = θᵒᵢ, tij = θⁱⱼ (i→j); toj = θᵒⱼ, tii = θⁱᵢ (j→i)
@variables s t              # MGF dummy variables

# domains: f-generating fitnesses in (1/100, 20), rates in (1/10, 10)
fitdom  = [xi => (1//100, 20), yj => (1//100, 20)]
ratedom = [toi => (1//10, 10), tij => (1//10, 10)]
domains = vcat(fitdom, ratedom)
domains_dyad = vcat(domains,
                    [xj => (1//100, 20), yi => (1//100, 20)],
                    [toj => (1//10, 10), tii => (1//10, 10)])

at0(expr) = Symbolics.substitute(expr, Dict(s => 0, t => 0))

# ---------------------------------------------------------------------------
# MGF of the zero-inflated exponential for the ordered pair i→j
# ---------------------------------------------------------------------------
f_ij = xi * yj / (1 + xi * yj)   # marginal edge probability (conditional DBCM layer)
ts   = toi + tij                 # rate sum θᵒᵢ + θⁱⱼ (naming as in Ŵ / σʷ)

M(f, r, u) = (1 - f) + f * r / (r - u)   # mixture MGF: (1-f)·e^{u·0} + f·(rate MGF)
M_ij = M(f_ij, ts, s)

# sanity: MGF normalisation M(0) = 1
verify("MGF normalisation M(0) == 1", at0(M_ij) - 1, domains)

E_w  = at0(Symbolics.derivative(M_ij, s))                            # ⟨w_ij⟩  = M'(0)
E_w2 = at0(Symbolics.derivative(Symbolics.derivative(M_ij, s), s))   # ⟨w_ij²⟩ = M''(0)

# (1) first moment matches the Ŵ code form fᵢⱼ/(θᵒᵢ + θⁱⱼ) [src/Models/DCReM.jl:546-565]
verify("⟨w⟩ == f/ts (Ŵ code form)", E_w - f_ij / ts, domains)

# (2) second moment ⟨w²⟩ = 2f/ts²
verify("⟨w²⟩ == 2f/ts²", E_w2 - 2 * f_ij / ts^2, domains)

# (3) variance matches the σʷ code form squared: fij(2-fij)/(θᵒᵢ+θⁱⱼ)² [src/Models/DCReM.jl:632-652]
Var_w   = E_w2 - E_w^2
code_sq = f_ij * (2 - f_ij) / ts^2
verify("Var[w] == f(2-f)/ts² (σʷ code form²)", Var_w - code_sq, domains)

# ---------------------------------------------------------------------------
# (4) joint MGF of the two (independent) directions of a dyad ⟹ Cov = 0
# ---------------------------------------------------------------------------
f_ji = xj * yi / (1 + xj * yi)   # marginal edge probability j→i
tsr  = toj + tii                 # rate sum θᵒⱼ + θⁱᵢ
M_ji = M(f_ji, tsr, t)

# independence across directions (likelihood factorizes over ordered pairs): M(s,t) = M_ij(s)·M_ji(t)
M_joint = M_ij * M_ji

E_prod = at0(Symbolics.derivative(Symbolics.derivative(M_joint, s), t))  # ⟨w_ij w_ji⟩
E_wr   = at0(Symbolics.derivative(M_ji, t))                              # ⟨w_ji⟩

verify("⟨w_ij w_ji⟩ == ⟨w_ij⟩⟨w_ji⟩ (Cov == 0)", E_prod - E_w * E_wr, domains_dyad)

# ---------------------------------------------------------------------------
# (5) binary layer: Bernoulli(f) variance vs the σˣ code form [src/Models/DCReM.jl:589-607]
# ---------------------------------------------------------------------------
# two-state sum: W(a) = (xi·yj)^a, Z = 1 + xi·yj
Z_bin  = 1 + xi * yj
E_a    = (0 * (xi * yj)^0 + 1 * (xi * yj)^1) / Z_bin   # ⟨a⟩
E_a2   = (0^2 * (xi * yj)^0 + 1^2 * (xi * yj)^1) / Z_bin
Var_a  = E_a2 - E_a^2
code_sq_bin = f_ij * (1 - f_ij)                        # (sqrt(pij(1-pij)))²
verify("binary Var[a] == f(1-f) (σˣ code form²)", Var_a - code_sq_bin, fitdom)

# ---------------------------------------------------------------------------
# Numeric closure against the package implementation (rhesus_macaques, as in test/models.jl)
# ---------------------------------------------------------------------------
using MaxEntropyGraphs

model = DCReM(MaxEntropyGraphs.rhesus_macaques())
solve_model!(model)                       # deterministic (fixed-point, :strengths initial guess)
MaxEntropyGraphs.set_Ĝ!(model)
MaxEntropyGraphs.set_σ!(model)
MaxEntropyGraphs.set_Ŵ!(model)
MaxEntropyGraphs.set_σʷ!(model)

n  = model.status[:N]
x  = model.xᵣ[model.dᵣ_ind]
y  = model.yᵣ[model.dᵣ_ind]
θᵒ = model.θ[1:n]
θⁱ = model.θ[n+1:end]

for (i, j) in ((1, 2), (3, 7))
    u   = x[i] * y[j]
    f   = u / (1 + u)
    rts = θᵒ[i] + θⁱ[j]
    closecheck("closure Ŵ[$i,$j] == f/ts", model.Ŵ[i, j], f / rts)
    closecheck("closure σʷ[$i,$j]² == f(2-f)/ts²", model.σʷ[i, j]^2, f * (2 - f) / rts^2)
    closecheck("closure σ[$i,$j]² == f(1-f)", model.σ[i, j]^2, f * (1 - f))
end

report("DCReM")
