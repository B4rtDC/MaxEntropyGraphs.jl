# =====================================================================================
# undirected_dyad_factor.jl — GATE for the within-dyad covariance factor (sqrt(2)) in
# the undirected delta-method σₓ.
#
# Framework (Squartini & Garlaschelli 2011, NJP 13 083001, App. B): the graph
# probability factorizes per dyad and the delta-method variance of a metric X is
#
#   (σ*[X])² = Σᵢⱼ [ (σ[gᵢⱼ] ∂X/∂gᵢⱼ)² + Cov(gᵢⱼ,gⱼᵢ) (∂X/∂gᵢⱼ)(∂X/∂gⱼᵢ) ]   at G = ⟨G⟩.
#
# For UNDIRECTED models aᵢⱼ and aⱼᵢ are the SAME Bernoulli variable, hence
# Cov(aᵢⱼ,aⱼᵢ) = Var(aᵢⱼ) = σᵢⱼ². Grouping per dyad {i,j} (i<j):
#
#   Var[X] ≈ Σ_{i<j} σᵢⱼ² (gᵢⱼ + gⱼᵢ)²           with  g = ∇X at Ĝ
#          = Σ_{i≠j} σᵢⱼ² gᵢⱼ²  +  Σ_{i≠j} σᵢⱼ² gᵢⱼ gⱼᵢ
#
# i.e. the standard deviation including the covariance cross-term is
#
#   σ*[X] = sqrt( sum((s .* g).^2) + sum((s.^2) .* g .* transpose(g)) ),   s = m.σ.
#
# The PACKAGE implementations of σₓ for the undirected models
#   src/Models/UBCM.jl, src/Models/UECM.jl (binary layer), src/Models/CReM.jl (binary layer)
# include this cross-term since v0.6.0.
#
# HISTORICAL BUG (pre-0.6.0, guarded against regression here): σₓ used to return only
# sqrt(sum((m.σ .* ∇X).^2)), omitting the cross-term. For a metric whose gradient is
# symmetric (gᵢⱼ = gⱼᵢ, e.g. X = sum(A)) both terms are equal, so the old value
# underestimated the true σ by exactly a factor 1/sqrt(2) (~29% low — far outside the
# 12% Monte-Carlo band gated below). For a metric that touches each dyad only once
# (upper-triangle mask) the cross-term vanishes and old/new coincide.
#
# This script gates the package σₓ directly against Monte-Carlo sampling from the real
# package models (UBCM on karate; the binary layers of UECM and CReM on the symmetrised
# rhesus macaques network — the graphs used by their testsets in test/models.jl), and
# additionally pins σₓ to an independent in-script evaluation of the cross-term formula
# to machine precision.
#
# Deterministic (fixed Xoshiro seeds). Prints a pass/fail table; exits non-zero on failure.
# =====================================================================================

using MaxEntropyGraphs
const Graphs = MaxEntropyGraphs.Graphs
const SWG = MaxEntropyGraphs.SimpleWeightedGraphs
import MaxEntropyGraphs: ReverseDiff, Xoshiro

# ---------------------------------------------------------------- helpers
_mean(xs) = sum(xs) / length(xs)
_var(xs) = (m = _mean(xs); sum(abs2, xs .- m) / (length(xs) - 1))
_std(xs) = sqrt(_var(xs))

const RESULTS = Tuple{String,Bool,String}[]
ok(name::String, cond::Bool; detail::String="") = (push!(RESULTS, (name, cond, detail)); cond)

"Independent in-script reference: delta-method σ with the within-dyad covariance cross-term (see header)."
function manual_sigma(m, X::Function)
    g = ReverseDiff.gradient(X, m.Ĝ)
    s = m.σ
    return sqrt(sum((s .* g) .^ 2) + sum((s .^ 2) .* g .* transpose(g)))
end

"""
Run the three σ comparisons for one model/metric:
returns (sampled, package, manual).
`samples_stat` = vector with the metric evaluated on every sampled graph.
"""
function compare(m, X::Function, samples_stat::Vector{Float64})
    sampled = _std(samples_stat)
    package = MaxEntropyGraphs.σₓ(m, X)    # what src/ implements (cross-term included, v0.6.0)
    manual = manual_sigma(m, X)            # independent evaluation of the same formula
    return sampled, package, manual
end

fmt(x) = string(round(x, sigdigits=5))

"Register the standard set of checks for a SYMMETRIC-gradient metric (cross-term active)."
function check_symmetric!(label::String, sampled, package, manual)
    detail = "sampled=$(fmt(sampled)) package=$(fmt(package)) manual=$(fmt(manual)) package/sampled=$(fmt(package / sampled))"
    # Direct gate on the shipped σₓ: a regression to the pre-0.6.0 formula (cross-term
    # dropped) would come in low by exactly 1/sqrt(2) ≈ 0.707 and fail the 12% band.
    ok("$label: package σₓ matches sampled std (rtol 0.12)",
       isapprox(package, sampled, rtol=0.12); detail=detail)
    ok("$label: package σₓ == manual cross-term form (machine precision)",
       isapprox(package, manual, rtol=1e-12);
       detail="package=$(fmt(package)) manual=$(fmt(manual))")
    return nothing
end

"Register the checks for the upper-triangle metric (each dyad touched once → no cross-term)."
function check_uppertriangle!(label::String, sampled, package, manual)
    detail = "sampled=$(fmt(sampled)) package=$(fmt(package)) manual=$(fmt(manual))"
    ok("$label: package σₓ matches sampled std (rtol 0.12) [cross-term exactly zero]",
       isapprox(package, sampled, rtol=0.12); detail=detail)
    ok("$label: package σₓ == manual cross-term form (machine precision)",
       isapprox(package, manual, rtol=1e-12); detail=detail)
    return nothing
end

"0/1 upper-triangle mask (constant) for the dyad-once metric X_U(A) = Σ_{i<j} A[i,j]."
upper_mask(n::Int) = Float64[i < j ? 1.0 : 0.0 for i in 1:n, j in 1:n]

# =====================================================================================
# 1. UBCM — Zachary karate club (the graph used by its testsets / ensemble_validation.jl)
# =====================================================================================
println("[1/3] UBCM (karate) ...")
G = Graphs.SimpleGraphs.smallgraph(:karate)
m_ubcm = UBCM(G)
solve_model!(m_ubcm)
set_Ĝ!(m_ubcm)
set_σ!(m_ubcm)
n_ubcm = size(m_ubcm.Ĝ, 1)

N_UBCM = 20_000
S_ubcm = rand(m_ubcm, N_UBCM; rng=Xoshiro(2026))

# deterministic mid-degree node: degree closest to the midpoint of the degree range
d_obs = Graphs.degree(G)
node = argmin(abs.(d_obs .- (minimum(d_obs) + maximum(d_obs)) / 2))
println("      ANND node = $node (degree $(d_obs[node]))")

# per-sample statistics on the 0/1 adjacency matrix
sum_samples  = Vector{Float64}(undef, N_UBCM)   # full-matrix sum = 2 × #edges
ut_samples   = Vector{Float64}(undef, N_UBCM)   # upper-triangle sum = #edges
annd_samples = Vector{Float64}(undef, N_UBCM)   # package ANND of the fixed node
for (k, g) in enumerate(S_ubcm)
    A = Matrix(Graphs.adjacency_matrix(g))
    sum_samples[k]  = Float64(sum(A))
    ut_samples[k]   = Float64(Graphs.ne(g))
    annd_samples[k] = MaxEntropyGraphs.ANND(A, node)
end

# metrics as functions of the (full) adjacency matrix
X_sum  = A -> sum(A)
maskU  = upper_mask(n_ubcm)
X_ut   = A -> sum(maskU .* A)
X_annd = A -> MaxEntropyGraphs.ANND(A, node; check_dimensions=false, check_directed=false)

s1 = compare(m_ubcm, X_sum, sum_samples)
check_symmetric!("UBCM sum(A)", s1...)

s2 = compare(m_ubcm, X_annd, annd_samples)
ok("UBCM ANND(node $node): package σₓ matches sampled std (rtol 0.12)",
   isapprox(s2[2], s2[1], rtol=0.12);
   detail="sampled=$(fmt(s2[1])) package=$(fmt(s2[2])) manual=$(fmt(s2[3]))")
ok("UBCM ANND(node $node): package σₓ == manual cross-term form (machine precision)",
   isapprox(s2[2], s2[3], rtol=1e-12);
   detail="package=$(fmt(s2[2])) manual=$(fmt(s2[3]))")

s3 = compare(m_ubcm, X_ut, ut_samples)
check_uppertriangle!("UBCM upper-triangle Σ_{i<j} aᵢⱼ", s3...)

# =====================================================================================
# 2. UECM binary layer — symmetrised rhesus macaques (graph of its testset, test/models.jl:810)
# =====================================================================================
println("[2/3] UECM (symmetrised rhesus macaques, binary layer) ...")
Gw = SWG.SimpleWeightedGraph(MaxEntropyGraphs.rhesus_macaques())
m_uecm = UECM(Gw)
solve_model!(m_uecm, method=:BFGS)  # fixed point is unstable for the UECM (cf. test/models.jl)
set_Ĝ!(m_uecm)
set_σ!(m_uecm)
n_uecm = size(m_uecm.Ĝ, 1)

N_UECM = 10_000
S_uecm = rand(m_uecm, N_UECM; rng=Xoshiro(2026))
# binary-layer statistics: sum of the 0/1 adjacency = 2 × #edges
uecm_sum = Float64[2.0 * Graphs.ne(g) for g in S_uecm]
uecm_ut  = Float64[Graphs.ne(g) for g in S_uecm]

maskU_uecm = upper_mask(n_uecm)
X_ut_uecm  = A -> sum(maskU_uecm .* A)

s4 = compare(m_uecm, X_sum, uecm_sum)
check_symmetric!("UECM sum(A) (binary layer)", s4...)

s5 = compare(m_uecm, X_ut_uecm, uecm_ut)
check_uppertriangle!("UECM upper-triangle Σ_{i<j} aᵢⱼ", s5...)

# =====================================================================================
# 3. CReM binary layer — symmetrised rhesus macaques (graph of its testset, test/models.jl:1023)
# =====================================================================================
println("[3/3] CReM (symmetrised rhesus macaques, binary layer) ...")
m_crem = CReM(Gw)
solve_model!(m_crem, method=:BFGS)  # solves the conditional (binary) layer + weighted layer
set_Ĝ!(m_crem)
set_σ!(m_crem)
n_crem = size(m_crem.Ĝ, 1)

N_CREM = 10_000
S_crem = rand(m_crem, N_CREM; rng=Xoshiro(2026))
crem_sum = Float64[2.0 * Graphs.ne(g) for g in S_crem]
crem_ut  = Float64[Graphs.ne(g) for g in S_crem]

maskU_crem = upper_mask(n_crem)
X_ut_crem  = A -> sum(maskU_crem .* A)

s6 = compare(m_crem, X_sum, crem_sum)
check_symmetric!("CReM sum(A) (binary layer)", s6...)

s7 = compare(m_crem, X_ut_crem, crem_ut)
check_uppertriangle!("CReM upper-triangle Σ_{i<j} aᵢⱼ", s7...)

# =====================================================================================
# report
# =====================================================================================
println()
println("="^132)
println("GATE: undirected within-dyad covariance factor (sqrt(2)) — package σₓ vs sampled std vs manual cross-term form")
println("  samples: UBCM=$(N_UBCM), UECM=$(N_UECM), CReM=$(N_CREM)   seeds: Xoshiro(2026)")
println("="^132)
npass = 0
for (name, cond, detail) in RESULTS
    global npass += cond
    println(rpad(cond ? "PASS" : "FAIL", 6), rpad(name, 78), detail)
end
println("-"^132)
if npass == length(RESULTS)
    println("ALL PASS ($npass/$(length(RESULTS)))")
    println("=> The package σₓ (within-dyad covariance cross-term included, v0.6.0) reproduces the sampled std;")
    println("   a regression to the pre-0.6.0 formula would fail the symmetric-metric gates by a factor 1/sqrt(2).")
    exit(0)
else
    println("FAILURES: $(length(RESULTS) - npass)/$(length(RESULTS))")
    exit(1)
end
