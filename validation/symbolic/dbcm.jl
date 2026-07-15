# DBCM — directed binary configuration model: per-dyad moment derivation.
#
# Dyad (a_ij, a_ji) ∈ {0,1}² with factorized graph weight
#     W(a_ij, a_ji) = (x_i y_j)^(a_ij) * (x_j y_i)^(a_ji),
# so the dyadic partition function is Z = Σ_states W = (1 + x_i y_j)(1 + x_j y_i).
# From the 4-state sum we derive
#     p_ij  = ⟨a_ij⟩          = x_i y_j / (1 + x_i y_j),
#     Var   = ⟨a_ij²⟩ - p_ij² = p_ij (1 - p_ij),
#     Cov(a_ij, a_ji)         = ⟨a_ij a_ji⟩ - p_ij p_ji = 0   (dyad channels independent),
# and check them against the code form σ[i,j] = sqrt(x_i y_j)/(1 + x_i y_j)
# implemented in src/Models/DBCM.jl (σˣ, lines 614-635), compared squared to stay sqrt-free.

include(joinpath(@__DIR__, "common.jl"))

@variables xi xj yi yj

domains = [xi => (1//100, 20), xj => (1//100, 20), yi => (1//100, 20), yj => (1//100, 20)]

# ---------------------------------------------------------------------------
# Symbolic derivation from the 4-state dyadic distribution
# ---------------------------------------------------------------------------
states = [(0, 0), (1, 0), (0, 1), (1, 1)]
W(a, b) = (xi * yj)^a * (xj * yi)^b

Z = sum(W(a, b) for (a, b) in states)

# sanity: the dyadic partition function factorizes over the two channels
verify("Z == (1 + xi*yj)(1 + xj*yi)", Z - (1 + xi * yj) * (1 + xj * yi), domains)

E_aij  = sum(a * W(a, b) for (a, b) in states) / Z          # ⟨a_ij⟩
E_aij2 = sum(a^2 * W(a, b) for (a, b) in states) / Z        # ⟨a_ij²⟩ (= ⟨a_ij⟩, a ∈ {0,1})
E_aji  = sum(b * W(a, b) for (a, b) in states) / Z          # ⟨a_ji⟩
E_prod = sum(a * b * W(a, b) for (a, b) in states) / Z      # ⟨a_ij a_ji⟩

p_ij = xi * yj / (1 + xi * yj)
p_ji = xj * yi / (1 + xj * yi)

# (1) first moment from the state sum matches the closed form used for Ĝ
verify("⟨a_ij⟩ == xi*yj/(1+xi*yj)", E_aij - p_ij, domains)

# (2) variance identity: Var[a_ij] = ⟨a_ij²⟩ - ⟨a_ij⟩² = p(1-p)
Var_aij = E_aij2 - E_aij^2
verify("Var[a_ij] == p(1-p)", Var_aij - p_ij * (1 - p_ij), domains)

# (3) squared code form (src/Models/DBCM.jl:629, σ[i,j] = sqrt(xi*yj)/(1+xi*yj)):
#     (sqrt(xi*yj)/(1+xi*yj))² = xi*yj/(1+xi*yj)² must equal p(1-p)
code_sq = (xi * yj) / (1 + xi * yj)^2
verify("code σ² == p(1-p)", code_sq - p_ij * (1 - p_ij), domains)

# (4) the two channels of a dyad are independent: Cov(a_ij, a_ji) = 0
Cov_dyad = E_prod - E_aij * E_aji
verify("Cov(a_ij,a_ji) == 0", Cov_dyad, domains)

# ---------------------------------------------------------------------------
# Numeric closure against the package implementation
# ---------------------------------------------------------------------------
using MaxEntropyGraphs

model = DBCM(MaxEntropyGraphs.maspalomas())
solve_model!(model)
MaxEntropyGraphs.set_Ĝ!(model)
MaxEntropyGraphs.set_σ!(model)

x = model.xᵣ[model.dᵣ_ind]
y = model.yᵣ[model.dᵣ_ind]

for (i, j) in ((1, 2), (3, 7))
    u = x[i] * y[j]
    p = u / (1 + u)
    closecheck("closure Ĝ[$i,$j] == p_ij", model.Ĝ[i, j], p)
    closecheck("closure σ[$i,$j]² == p(1-p)", model.σ[i, j]^2, p * (1 - p))
end

report("DBCM")
