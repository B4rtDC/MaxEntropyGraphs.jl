# RBCM — reciprocal binary configuration model: per-dyad moment derivation.
#
# Dyadic distribution (Squartini & Garlaschelli 2011, App. B): the dyad (a_ij, a_ji) has four
# states {(0,0), (1,0), (0,1), (1,1)} with unnormalized weights {1, x_i*y_j, x_j*y_i, z_i*z_j}
# and partition function Z = 1 + x_i*y_j + x_j*y_i + z_i*z_j. From the state sum we derive
#   ⟨a_ij⟩          = (x_i*y_j + z_i*z_j)/Z            (= p→ + p↔)
#   Var[a_ij]       = ⟨a_ij⟩(1 - ⟨a_ij⟩)               (Bernoulli marginal)
#   ⟨a_ij a_ji⟩     = z_i*z_j/Z                        (= p↔)
#   Cov(a_ij, a_ji) = z_i*z_j/Z - ⟨a_ij⟩⟨a_ji⟩
# and check these against the closed forms implemented in src/Models/RBCM.jl
# (Ĝ: lines 854-880, σˣ: lines 904-930, _cov_dyads: lines 951-974).

include(joinpath(@__DIR__, "common.jl"))

using Symbolics

@variables xi xj yi yj zi zj

# ---------------------------------------------------------------------------
# Symbolic derivation from the four-state dyad distribution
# ---------------------------------------------------------------------------
w00 = 1               # state (a_ij, a_ji) = (0, 0)
w10 = xi * yj         # state (1, 0): single link i → j
w01 = xj * yi         # state (0, 1): single link j → i
w11 = zi * zj         # state (1, 1): reciprocated pair
Z   = w00 + w10 + w01 + w11

p00 = w00 / Z
p10 = w10 / Z
p01 = w01 / Z
p11 = w11 / Z

# First moments via the state sum (a_ij = 1 in states (1,0) and (1,1); a_ji in (0,1) and (1,1))
a_ij = p10 + p11
a_ji = p01 + p11
# Second moments via the state sum (a ∈ {0,1} so a² = a; cross moment only from state (1,1))
a_ij_sq  = p10 + p11
cross    = p11
var_stat = a_ij_sq - a_ij^2            # Var[a_ij]      = ⟨a_ij²⟩ - ⟨a_ij⟩²
cov_stat = cross - a_ij * a_ji         # Cov(a_ij,a_ji) = ⟨a_ij a_ji⟩ - ⟨a_ij⟩⟨a_ji⟩

# ---------------------------------------------------------------------------
# Code forms restated symbolically from src/Models/RBCM.jl
# ---------------------------------------------------------------------------
# Ĝ (lines 871-874):        D = 1 + x[i]y[j] + x[j]y[i] + z[i]z[j];  G[i,j] = (x[i]y[j] + z[i]z[j])/D
# σˣ (lines 920-924):       a = (x[i]y[j] + z[i]z[j])/D;             σ[i,j] = sqrt(a(1 - a))
# _cov_dyads (lines 961-967): aij = (x[i]y[j] + z[i]z[j])/D; aji = (x[j]y[i] + z[i]z[j])/D
#                             C[i,j] = z[i]z[j]/D - aij*aji
D        = 1 + xi * yj + xj * yi + zi * zj
a_code   = (xi * yj + zi * zj) / D
aji_code = (xj * yi + zi * zj) / D
var_code = a_code * (1 - a_code)                 # squared σˣ code form (sqrt-free comparison)
cov_code = zi * zj / D - a_code * aji_code

domains = [xi => (1//100, 10), xj => (1//100, 10),
           yi => (1//100, 10), yj => (1//100, 10),
           zi => (1//100, 10), zj => (1//100, 10)]

# (1) first moment: state sum vs closed form (xi*yj + zi*zj)/Z (identical to the Ĝ code form)
verify("⟨a_ij⟩: state sum vs (xi*yj + zi*zj)/Z", a_ij - a_code, domains)
# (2) marginal variance: state sum vs the σˣ code form a(1-a)
verify("Var[a_ij]: state sum vs code form a(1-a)", var_stat - var_code, domains)
# (3) within-dyad covariance: state sum vs the _cov_dyads code form
verify("⟨a_ij a_ji⟩: state sum vs zi*zj/Z", cross - zi * zj / Z, domains)
verify("Cov(a_ij,a_ji): state sum vs _cov_dyads code form", cov_stat - cov_code, domains)
# (4) sanity: the four state probabilities sum to 1
verify("state probabilities sum to 1", p00 + p10 + p01 + p11 - 1, domains)

# ---------------------------------------------------------------------------
# Numeric closure: pin the symbolic forms to the actual package kernels
# ---------------------------------------------------------------------------
using MaxEntropyGraphs

G = MaxEntropyGraphs.Graphs.SimpleDiGraph(MaxEntropyGraphs.rhesus_macaques())
model = MaxEntropyGraphs.RBCM(G)
MaxEntropyGraphs.solve_model!(model)     # deterministic (:fixedpoint from the :degrees initial guess)
MaxEntropyGraphs.set_σ!(model)
C = MaxEntropyGraphs._cov_dyads(model)

# fitted parameters, expanded to node level (accessor pattern from src/Models/RBCM.jl:955-957)
x = model.xᵣ[model.dᵣ_ind]
y = model.yᵣ[model.dᵣ_ind]
z = model.zᵣ[model.dᵣ_ind]

function var_closed(i, j)
    Dij = 1 + x[i] * y[j] + x[j] * y[i] + z[i] * z[j]
    a = (x[i] * y[j] + z[i] * z[j]) / Dij
    return a * (1 - a)
end

function cov_closed(i, j)
    Dij = 1 + x[i] * y[j] + x[j] * y[i] + z[i] * z[j]
    aij = (x[i] * y[j] + z[i] * z[j]) / Dij
    aji = (x[j] * y[i] + z[i] * z[j]) / Dij
    return z[i] * z[j] / Dij - aij * aji
end

for (i, j) in [(1, 2), (5, 12)]
    closecheck("closure σ[$i,$j]² vs symbolic Var", model.σ[i, j]^2, var_closed(i, j))
end
for (i, j) in [(3, 7), (10, 4)]
    closecheck("closure _cov_dyads[$i,$j] vs symbolic Cov", C[i, j], cov_closed(i, j))
end

report("RBCM")
