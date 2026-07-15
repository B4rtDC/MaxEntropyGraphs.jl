# UBCM — undirected binary configuration model: per-dyad moment derivation.
#
# Derivation from scratch (Squartini & Garlaschelli 2011, NJP 13 083001):
#   Hamiltonian H(A) = Σ_i θ_i k_i(A); with x_i = exp(-θ_i) the graph probability
#   factorizes over the dyads (i<j). A single dyad has one binary state a ∈ {0,1}
#   with unnormalized weight (x_i x_j)^a. Writing v = x_i x_j:
#       Z      = Σ_{a∈{0,1}} v^a          = 1 + v
#       ⟨a⟩    = Σ a v^a / Z              = v/(1+v)              =: p
#       ⟨a²⟩   = Σ a² v^a / Z             = v/(1+v)   (a² = a)
#       Var[a] = ⟨a²⟩ − ⟨a⟩²              = p(1−p)
#   Undirected convention: a_ij and a_ji are the SAME random variable, so
#       Cov(a_ij, a_ji) = ⟨a·a⟩ − ⟨a⟩⟨a⟩ = Var[a].
#
# Code form under test: src/Models/UBCM.jl:539  σ[i,j] = sqrt(xij)/(1 + xij).
# All identities are kept sqrt-free: we compare the SQUARED code form,
# (sqrt(v)/(1+v))² = v/(1+v)² (exact for v > 0), against p(1−p).

include(joinpath(@__DIR__, "common.jl"))

using Symbolics

@variables xi xj
v = xi * xj

domains = [xi => (1//100, 20), xj => (1//100, 20)]

# ---------------------------------------------------------------------------
# State-sum derivation (explicit enumeration over a ∈ {0,1})
# ---------------------------------------------------------------------------
states  = (0, 1)
Z       = sum(v^a for a in states)                 # 1 + v
Ea      = sum(a   * v^a for a in states) / Z       # ⟨a⟩
Ea2     = sum(a^2 * v^a for a in states) / Z       # ⟨a²⟩
Var_ss  = Ea2 - Ea^2                               # Var from the state sum

p = v / (1 + v)

# (1) first moment from the state sum equals v/(1+v)
verify("⟨a⟩ state sum == v/(1+v)", Ea - p, domains)

# (2) variance from the state sum equals p(1−p)
verify("Var[a] state sum == p(1-p)", Var_ss - p * (1 - p), domains)

# (3) squared code form (src/Models/UBCM.jl:539): (sqrt(v)/(1+v))² = v/(1+v)²
#     must equal the Bernoulli variance p(1−p). Squared algebraically to stay
#     sqrt-free (exact since v = x_i x_j > 0 on the domain).
code_sigma_sq = v / (1 + v)^2
verify("code σ² == p(1-p)  [UBCM.jl:539]", code_sigma_sq - p * (1 - p), domains)

# (4) undirected dyad convention: a_ij ≡ a_ji, hence
#     Cov(a_ij, a_ji) = ⟨a·a⟩ − ⟨a⟩⟨a⟩ with ⟨a·a⟩ = ⟨a²⟩ from the state sum,
#     and this covariance equals the variance p(1−p).
Eaa    = sum(a * a * v^a for a in states) / Z      # ⟨a_ij a_ji⟩ = ⟨a²⟩
Cov_ss = Eaa - Ea * Ea
verify("Cov(a_ij,a_ji) == Var  (undirected: same variable)", Cov_ss - p * (1 - p), domains)

# ---------------------------------------------------------------------------
# Numeric closure: pin the symbolic form to the actual package kernel
# ---------------------------------------------------------------------------
using MaxEntropyGraphs

G = MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate)
m = UBCM(G)
solve_model!(m)
set_σ!(m)

x = m.xᵣ[m.dᵣ_ind]                                 # per-node x, expanded to full size
var_closed(i, j) = (vij = x[i] * x[j]; vij / (1 + vij)^2)

closecheck("closure: m.σ[1,2]² == v/(1+v)² (karate)", m.σ[1, 2]^2, var_closed(1, 2))
closecheck("closure: m.σ[3,4]² == v/(1+v)² (karate)", m.σ[3, 4]^2, var_closed(3, 4))

report("UBCM")
