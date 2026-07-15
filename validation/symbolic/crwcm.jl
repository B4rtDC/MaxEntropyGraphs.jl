# CRWCM — reciprocal conditional weighted configuration model: per-dyad moment derivation
# via the joint moment-generating function of (w_ij, w_ji).
#
# Binary layer = RBCM (Squartini & Garlaschelli 2011, App. B): the dyad (a_ij, a_ji) has four
# states {none, →, ←, ↔} with probabilities
#   p0 = 1/D,  p→ = x_i*y_j/D,  p← = x_j*y_i/D,  p↔ = z_i*z_j/D,   D = 1 + x_i*y_j + x_j*y_i + z_i*z_j.
# Conditional weights (rates read off src/Models/CRWCM.jl:626-629, 758-761, 812-827):
#   state →: w_ij ~ Exp(r1), w_ji = 0,      r1 = θ→_i + θ←_j
#   state ←: w_ji ~ Exp(r2), w_ij = 0,      r2 = θ→_j + θ←_i
#   state ↔: w_ij ~ Exp(r3) ⟂ w_ji ~ Exp(r4), r3 = θ↔o_i + θ↔i_j,  r4 = θ↔o_j + θ↔i_i
# (θ→ = θ[1:n], θ← = θ[n+1:2n], θ↔o = θ[2n+1:3n], θ↔i = θ[3n+1:4n].)
# Joint MGF of the three-channel mixture:
#   M(s,t) = p0 + p→ r1/(r1-s) + p← r2/(r2-t) + p↔ (r3/(r3-s)) (r4/(r4-t)),
# and all moments follow by differentiation at (s,t) = (0,0). Checked against the closed forms
# implemented in src/Models/CRWCM.jl (Ĝ: 576-597, Ŵ: 618-647, σˣ: 670-692, _cov_dyads: 712-735,
# σʷ: 750-779, _covʷ: 805-835).

include(joinpath(@__DIR__, "common.jl"))

using Symbolics

@variables xi xj yi yj zi zj r1 r2 r3 r4 s t

# ---------------------------------------------------------------------------
# Symbolic derivation: dyad state probabilities and joint MGF
# ---------------------------------------------------------------------------
D  = 1 + xi * yj + xj * yi + zi * zj
p0 = 1 / D
pr = xi * yj / D          # p→ : single link i → j
pl = xj * yi / D          # p← : single link j → i
pb = zi * zj / D          # p↔ : reciprocated pair

M = p0 + pr * r1 / (r1 - s) + pl * r2 / (r2 - t) + pb * (r3 / (r3 - s)) * (r4 / (r4 - t))

at0(expr) = Symbolics.substitute(expr, Dict(s => 0, t => 0))

dMds   = Symbolics.derivative(M, s)
dMdt   = Symbolics.derivative(M, t)
d2Mds2 = Symbolics.derivative(dMds, s)
d2Mdsdt = Symbolics.derivative(dMds, t)

mean_wij  = at0(dMds)      # ⟨w_ij⟩
mean_wji  = at0(dMdt)      # ⟨w_ji⟩
m2_wij    = at0(d2Mds2)    # ⟨w_ij²⟩
cross_w   = at0(d2Mdsdt)   # ⟨w_ij w_ji⟩
var_wij   = m2_wij - mean_wij^2
cov_w     = cross_w - mean_wij * mean_wji

# ---------------------------------------------------------------------------
# Code forms restated symbolically from src/Models/CRWCM.jl
# ---------------------------------------------------------------------------
# Ŵ (639-640):     W[i,j]  = (xiyj/D)/(θ→_i+θ←_j) + (zizj/D)/(θ↔o_i+θ↔i_j)          = pr/r1 + pb/r3
# σʷ (769-773):    m2      = 2(xiyj/D)/r1² + 2(zizj/D)/r3²;  σ² = m2 - (w1+w2)²
# _covʷ (823-828): joint   = (zizj/D)/(r3 r4);  C = joint - wij*wji
# Ĝ (590-591):     G[i,j]  = (xiyj + zizj)/D
# σˣ (685-686):    a = (xiyj + zizj)/D;  σ² = a(1-a)
# _cov_dyads (726-728): C = zizj/D - aij*aji
wij_code   = (xi * yj / D) / r1 + (zi * zj / D) / r3
wji_code   = (xj * yi / D) / r2 + (zi * zj / D) / r4
varw_code  = 2 * (xi * yj / D) / r1^2 + 2 * (zi * zj / D) / r3^2 - wij_code^2
covw_code  = (zi * zj / D) / (r3 * r4) - wij_code * wji_code
a_code     = (xi * yj + zi * zj) / D
aji_code   = (xj * yi + zi * zj) / D
varx_code  = a_code * (1 - a_code)
covx_code  = zi * zj / D - a_code * aji_code

domains = [xi => (1//100, 5), xj => (1//100, 5),
           yi => (1//100, 5), yj => (1//100, 5),
           zi => (1//100, 5), zj => (1//100, 5),
           r1 => (1//10, 10), r2 => (1//10, 10),
           r3 => (1//10, 10), r4 => (1//10, 10)]

# (0) sanity: the MGF is normalised and the four state probabilities sum to 1
verify("MGF sanity: M(0,0) = 1", at0(M) - 1, domains)
verify("state probabilities sum to 1", p0 + pr + pl + pb - 1, domains)

# (1) first moments: ∂M/∂s|₀ = p→/r1 + p↔/r3 and the Ŵ code form
verify("⟨w_ij⟩: ∂M/∂s|₀ vs p→/r1 + p↔/r3", mean_wij - (pr / r1 + pb / r3), domains)
verify("⟨w_ij⟩: MGF vs Ŵ code form", mean_wij - wij_code, domains)
verify("⟨w_ji⟩: ∂M/∂t|₀ vs p←/r2 + p↔/r4", mean_wji - (pl / r2 + pb / r4), domains)
verify("⟨w_ji⟩: MGF vs _covʷ wji code form", mean_wji - wji_code, domains)

# (2) second moment and variance: ∂²M/∂s²|₀ = 2p→/r1² + 2p↔/r3² and the σʷ code form (squared)
verify("⟨w_ij²⟩: ∂²M/∂s²|₀ vs 2p→/r1² + 2p↔/r3²", m2_wij - (2pr / r1^2 + 2pb / r3^2), domains)
verify("Var[w_ij]: MGF vs σʷ code form (squared)", var_wij - varw_code, domains)

# (3) cross moment and within-dyad covariance: ∂²M/∂s∂t|₀ = p↔/(r3 r4) and the _covʷ code form
verify("⟨w_ij w_ji⟩: ∂²M/∂s∂t|₀ vs p↔/(r3 r4)", cross_w - pb / (r3 * r4), domains)
verify("Cov(w_ij,w_ji): MGF vs _covʷ code form", cov_w - covw_code, domains)

# (4) binary layer (state sum: a_ij = 1 in states → and ↔, a_ij a_ji = 1 only in ↔)
a_ij = pr + pb
a_ji = pl + pb
verify("⟨a_ij⟩: state sum vs Ĝ code form", a_ij - a_code, domains)
verify("Var[a_ij]: state sum vs σˣ code form a(1-a)", (a_ij - a_ij^2) - varx_code, domains)
verify("Cov(a_ij,a_ji): state sum vs _cov_dyads code form", (pb - a_ij * a_ji) - covx_code, domains)

# ---------------------------------------------------------------------------
# Numeric closure: pin the symbolic forms to the actual package kernels
# ---------------------------------------------------------------------------
using MaxEntropyGraphs

Gw = MaxEntropyGraphs.rhesus_macaques()
model = MaxEntropyGraphs.CRWCM(Gw)
MaxEntropyGraphs.solve_model!(model)     # deterministic (:fixedpoint from the :strengths initial guess)

n  = model.status[:N]
Wm = MaxEntropyGraphs.Ŵ(model)
σw = MaxEntropyGraphs.σʷ(model)
Cw = MaxEntropyGraphs._covʷ(model)
Gm = MaxEntropyGraphs.Ĝ(model)
Cx = MaxEntropyGraphs._cov_dyads(model)

# fitted parameters, expanded to node level (accessor pattern from src/Models/CRWCM.jl:623-629)
x  = model.xᵣ[model.dᵣ_ind]
y  = model.yᵣ[model.dᵣ_ind]
z  = model.zᵣ[model.dᵣ_ind]
θr = model.θ[1:n]           # θ→
θl = model.θ[n+1:2n]        # θ←
θbo = model.θ[2n+1:3n]      # θ↔,out
θbi = model.θ[3n+1:4n]      # θ↔,in

# pick nodes with all four channels alive (rhesus_macaques has dead s→/s← channels; the closed
# forms below assume finite rates, so restrict to fully live pairs — deterministic selection)
live = [k for k in 1:n if model.s_out[k] > 0 && model.s_in[k] > 0 &&
                          model.s_rec_out[k] > 0 && model.s_rec_in[k] > 0]
@assert length(live) >= 4 "need at least 4 fully live nodes for the closure checks"
pairs = [(live[1], live[2]), (live[3], live[4])]

rate1(i, j) = θr[i] + θl[j]        # r1 (right arc i→j); rate1(j,i) = r2
rate3(i, j) = θbo[i] + θbi[j]      # r3 (recip, w_ij);   rate3(j,i) = r4
Dn(i, j)  = 1 + x[i] * y[j] + x[j] * y[i] + z[i] * z[j]
prn(i, j) = x[i] * y[j] / Dn(i, j)
pbn(i, j) = z[i] * z[j] / Dn(i, j)

mean_closed(i, j) = prn(i, j) / rate1(i, j) + pbn(i, j) / rate3(i, j)
var_closed(i, j)  = 2 * prn(i, j) / rate1(i, j)^2 + 2 * pbn(i, j) / rate3(i, j)^2 - mean_closed(i, j)^2
covw_closed(i, j) = pbn(i, j) / (rate3(i, j) * rate3(j, i)) - mean_closed(i, j) * mean_closed(j, i)
a_closed(i, j)    = prn(i, j) + pbn(i, j)
covx_closed(i, j) = pbn(i, j) - a_closed(i, j) * a_closed(j, i)

for (i, j) in pairs
    closecheck("closure Ŵ[$i,$j] vs symbolic ⟨w_ij⟩", Wm[i, j], mean_closed(i, j))
end
for (i, j) in pairs
    closecheck("closure σʷ[$i,$j]² vs symbolic Var[w_ij]", σw[i, j]^2, var_closed(i, j))
end
for (i, j) in pairs
    closecheck("closure _covʷ[$i,$j] vs symbolic Cov(w_ij,w_ji)", Cw[i, j], covw_closed(i, j))
end
# binary layer closure (bonus: pins check (4) to the Ĝ/_cov_dyads kernels)
for (i, j) in pairs
    closecheck("closure Ĝ[$i,$j] vs symbolic ⟨a_ij⟩", Gm[i, j], a_closed(i, j))
    closecheck("closure _cov_dyads[$i,$j] vs symbolic Cov(a_ij,a_ji)", Cx[i, j], covx_closed(i, j))
end

report("CRWCM")
