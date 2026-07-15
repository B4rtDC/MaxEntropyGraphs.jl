# BiCM (bipartite configuration model) — symbolic validation.
#
# Part 1 — dyadic distribution:
#   Each biadjacency entry m_cp ∈ {0,1} carries Boltzmann weight (x_c y_p)^m_cp, so the
#   per-dyad partition function is Z = 1 + x y. From this we derive
#       p = ⟨m⟩ = x y / (1 + x y),   Var[m] = ⟨m²⟩ - ⟨m⟩² = p (1 - p),
#   and independence of distinct entries (the joint weight factorises), stated through the
#   explicit two-entry product sum ⟨m₁ m₂⟩ == ⟨m₁⟩⟨m₂⟩ — both for fully distinct dyads and
#   for dyads sharing a ⊥ node. The scalar kernel f_BiCM (src/Models/BiCM.jl:843) is called
#   on a symbolic argument and pinned to the derived p.
#
# Part 2 — Saracco et al. 2015 (SI) motif-family identities in u (a degree) and s2 (its
#   variance), which justify the Vn / Λn machinery:
#   (a) SI III.7:  d/du C(u,n) == C(u,n) · Σ_{i=0}^{n-1} 1/(u-i)  for n = 2,3,4,
#       with C(u,n) = Π_{i=0}^{n-1}(u-i)/n! written as an explicit polynomial;
#   (b) SI III.10 vs III.6 (n=2 delta form): [dC(u,2)/du]² == ((2u-1)/2)²;
#   (c) Gaussian expectation shifts: expand Π_{i=0}^{n-1}(u+δ-i) in powers of δ, apply
#       ⟨δ⟩=0, ⟨δ²⟩=s2, ⟨δ³⟩=0, ⟨δ⁴⟩=3 s2², divide by n!; the shift ⟨C(u+δ,n)⟩ - C(u,n) is
#       s2/2 (n=2, SI III.9), s2(u-1)/2 (n=3), (3 s2² + s2(6u²-18u+11))/24 (n=4, SI III.13).
#
# Numeric closure: BiCM on corporateclub(); the expected biadjacency matrix Ĝ
# (src/Models/BiCM.jl:619, rows = ⊥ / bottom layer, columns = ⊤ / top layer) must match
# p evaluated at the fitted parameters x = m.xᵣ[m.d⊥ᵣ_ind], y = m.yᵣ[m.d⊤ᵣ_ind].

include(joinpath(@__DIR__, "common.jl"))

using MaxEntropyGraphs

@variables x y x1 y1 x2 y2 u d s2

const DOM_XY   = [x => (1//100, 20), y => (1//100, 20)]
const DOM_XY4  = [x1 => (1//100, 20), y1 => (1//100, 20), x2 => (1//100, 20), y2 => (1//100, 20)]
const DOM_U    = [u => (5, 50)]
const DOM_US2  = [u => (5, 50), s2 => (1//10, 10)]

# ------------------------------------------------------------------
# Part 1 — dyadic distribution of a biadjacency entry
# ------------------------------------------------------------------
# P(m) = (x y)^m / Z with m ∈ {0,1}
Z   = sum((x*y)^m for m in 0:1)                 # = 1 + x y
Em  = sum(m   * (x*y)^m for m in 0:1) / Z       # ⟨m⟩
Em2 = sum(m^2 * (x*y)^m for m in 0:1) / Z       # ⟨m²⟩
p   = x*y / (1 + x*y)

verify("dyad: <m> == x*y/(1 + x*y)", Em - p, DOM_XY)
verify("dyad: Var[m] == p*(1-p)", (Em2 - Em^2) - p*(1 - p), DOM_XY)

# scalar kernel used by the package (src/Models/BiCM.jl:843) evaluated symbolically
verify("kernel: f_BiCM(x*y) == p", MaxEntropyGraphs.f_BiCM(x*y) - p, DOM_XY)

# independence of distinct entries — fully distinct dyads (c,p) ≠ (c',p'):
# joint weight (x1 y1)^m1 (x2 y2)^m2 factorises
W2(m1, m2) = (x1*y1)^m1 * (x2*y2)^m2
Z2    = sum(W2(m1, m2) for m1 in 0:1, m2 in 0:1)
Em1m2 = sum(m1*m2*W2(m1, m2) for m1 in 0:1, m2 in 0:1) / Z2
p1    = x1*y1 / (1 + x1*y1)
p2    = x2*y2 / (1 + x2*y2)
verify("independence: <m1*m2> == <m1>*<m2> (distinct dyads)", Em1m2 - p1*p2, DOM_XY4)

# independence with a shared ⊥ node c: entries (c,p1) and (c,p2), weight x^(m1+m2) y1^m1 y2^m2
Ws(m1, m2) = x^(m1 + m2) * y1^m1 * y2^m2
Zs    = sum(Ws(m1, m2) for m1 in 0:1, m2 in 0:1)
Es12  = sum(m1*m2*Ws(m1, m2) for m1 in 0:1, m2 in 0:1) / Zs
ps1   = x*y1 / (1 + x*y1)
ps2   = x*y2 / (1 + x*y2)
verify("independence: <m1*m2> == <m1>*<m2> (shared ⊥ node)", Es12 - ps1*ps2,
       [x => (1//100, 20), y1 => (1//100, 20), y2 => (1//100, 20)])

# ------------------------------------------------------------------
# Part 2 — Saracco et al. 2015 SI motif-family identities
# ------------------------------------------------------------------
"C(u,n) = Π_{i=0}^{n-1}(u-i)/n! as an explicit polynomial in u"
binomexpr(n::Int) = Symbolics.expand(prod(u - i for i in 0:n-1) / factorial(n))

# (a) SI III.7: logarithmic-derivative identity, n = 2, 3, 4
for n in 2:4
    B  = binomexpr(n)
    dB = Symbolics.derivative(B, u)
    harm = sum(1 / (u - i) for i in 0:n-1)
    verify("SI III.7 (n=$n): d/du C(u,$n) == C(u,$n)*Sum 1/(u-i)", dB - B*harm, DOM_U)
end

# (b) SI III.10 vs III.6, n=2 delta form (sqrt-free: compare squares)
dB2 = Symbolics.derivative(binomexpr(2), u)
verify("SI III.10 (n=2): (dC(u,2)/du)^2 == ((2u-1)/2)^2", dB2^2 - ((2u - 1) / 2)^2, DOM_U)

# (c) Gaussian expectation shifts ⟨C(u+δ,n)⟩ - C(u,n)
"""
Expand Π_{i=0}^{n-1}(u+δ-i) in powers of δ, apply the Gaussian moments
⟨δ⟩=0, ⟨δ²⟩=s2, ⟨δ³⟩=0, ⟨δ⁴⟩=3 s2², divide by n!, subtract C(u,n).
"""
function gaussian_shift(n::Int)
    P = Symbolics.expand(prod(u + d - i for i in 0:n-1))
    moments = Dict(1 => 0, 2 => s2, 3 => 0, 4 => 3*s2^2)
    EP = Symbolics.substitute(P, Dict(d => 0))            # δ⁰ coefficient
    for k in 1:n
        # wrap: Symbolics.coeff may return an unwrapped BasicSymbolic
        EP += Symbolics.wrap(Symbolics.coeff(P, d^k)) * moments[k]
    end
    return EP / factorial(n) - binomexpr(n)
end

verify("SI III.9  (n=2): <C(u+d,2)> - C(u,2) == s2/2",
       gaussian_shift(2) - s2/2, DOM_US2)
verify("SI shift  (n=3): <C(u+d,3)> - C(u,3) == s2*(u-1)/2",
       gaussian_shift(3) - s2*(u - 1)/2, DOM_US2)
verify("SI III.13 (n=4): <C(u+d,4)> - C(u,4) == (3*s2^2 + s2*(6u^2-18u+11))/24",
       gaussian_shift(4) - (3*s2^2 + s2*(6u^2 - 18u + 11))/24, DOM_US2)

# ------------------------------------------------------------------
# Numeric closure — Ĝ entries on corporateclub() vs the derived p
# ------------------------------------------------------------------
model = BiCM(MaxEntropyGraphs.corporateclub())
solve_model!(model)                      # deterministic fixed-point solve
G = Ĝ(model)                             # 25 (⊥, rows) × 15 (⊤, columns)

# full-length parameter vectors, exactly as in Ĝ (src/Models/BiCM.jl:630-631)
xf = model.xᵣ[model.d⊥ᵣ_ind]
yf = model.yᵣ[model.d⊤ᵣ_ind]

for (i, j) in ((1, 1), (7, 3), (25, 15))
    pij = Symbolics.value(Symbolics.substitute(p, Dict(x => xf[i], y => yf[j])))
    closecheck("closure: Ĝ[$i,$j] == p(x⊥[$i], y⊤[$j]) on corporateclub", Float64(pij), G[i, j])
end

report("BiCM")
