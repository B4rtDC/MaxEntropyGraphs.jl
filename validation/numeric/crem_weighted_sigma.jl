# CReM — Monte-Carlo gate for the PROPOSED weighted-layer standard deviation σʷ
# (NOT yet shipped in src/Models/CReM.jl — the model only implements the binary σˣ).
#
# Model (two-step, undirected, continuous weights):
#   binary layer  : aᵢⱼ ~ Bernoulli(fᵢⱼ),   fᵢⱼ = xᵢxⱼ/(1+xᵢxⱼ)      (UBCM prior)
#   weight layer  : wᵢⱼ | aᵢⱼ=1 ~ Exponential(rate tsᵢⱼ), tsᵢⱼ = θᵢ+θⱼ
# Unconditional per-dyad moments (zero-inflated exponential, cf. validation/symbolic/crem.jl):
#   ⟨w⟩    = f/ts                                  == Ŵ code form (src/Models/CReM.jl:471-489)
#   Var[w] = 2f/ts² - (f/ts)² = f(2-f)/ts²         == PROPOSED σʷ² (directed twin: DCReM.jl σʷ)
#   Cov(a,w) = ⟨aw⟩-⟨a⟩⟨w⟩ = ⟨w⟩(1-f) = f(1-f)/ts  (a·w = w since w vanishes off-edge)
# Undirected convention: wᵢⱼ and wⱼᵢ are the SAME variable, so the delta-method identity for
# the total weight is Var[sum(W)/2] = Σ_{i<j} Var[wᵢⱼ] = sum(σʷ.^2)/2 (dyads independent).
#
# The script gates the proposal against the REAL package sampler (rand(m::CReM)):
#   (1) build & solve the CReM on the weighted graph the testsets use (rhesus macaques, N=16)
#   (2) in-script What_proto == package Ŵ(m)
#   (3) sample ≥ 3000 weighted graphs (continuous exponential weights) with a fixed seed
#   (4) entrywise Var[w] on the 5 largest-⟨w⟩ dyads within 5·SE
#   (5) aggregate: std(sum(W)/2) vs sqrt(sum(sigw_proto.^2)/2) within rtol 0.1
#   (6) empirical Cov(a,w) vs ⟨w⟩(1-f) on those dyads within 5·SE
# Deterministic (fixed Xoshiro seeds); prints a pass/fail table; exits non-zero on failure.

using MaxEntropyGraphs
const Graphs = MaxEntropyGraphs.Graphs

_mean(xs) = sum(xs) / length(xs)
_var(xs) = (m = _mean(xs); sum(abs2, xs .- m) / (length(xs) - 1))

# ---------------------------------------------------------------------------------------
# pass/fail bookkeeping
# ---------------------------------------------------------------------------------------
const RESULTS = Vector{Tuple{String,Bool,String}}()
function ok(name::AbstractString, cond::Bool; detail::AbstractString="")
    push!(RESULTS, (String(name), cond, String(detail)))
    return cond
end

# ---------------------------------------------------------------------------------------
# (1) build & solve the CReM exactly as the testsets do (test/models.jl "CReM")
# ---------------------------------------------------------------------------------------
Gw = MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(MaxEntropyGraphs.rhesus_macaques())
model = CReM(Gw)
solve_model!(model)   # two-step: binary UBCM layer (degrees) then weighted layer (strengths)

n = model.status[:N]
x = model.xᵣ[model.dᵣ_ind]   # per-node binary fitness (full, un-reduced)
θ = model.θ                  # per-node weighted-layer rate parameters

# solve sanity: both constraint sequences are reproduced
ok("solve: expected degrees == observed degrees (rtol 1e-4)",
   isapprox(MaxEntropyGraphs.degree(model), Float64.(model.d), rtol=1e-4))
ok("solve: expected strengths == observed strengths (rtol 1e-4)",
   isapprox(MaxEntropyGraphs.strength(model), model.s, rtol=1e-4))

# ---------------------------------------------------------------------------------------
# (2) proposed per-dyad moments in-script; ⟨w⟩ must match the package Ŵ(m)
# ---------------------------------------------------------------------------------------
F          = zeros(Float64, n, n)   # fᵢⱼ  (binary marginal probability)
What_proto = zeros(Float64, n, n)   # ⟨w⟩  = f/ts
sigw_proto = zeros(Float64, n, n)   # σʷ   = sqrt(f(2-f))/ts   (PROPOSED)
for i in 1:n, j in 1:n
    i == j && continue
    fij = x[i] * x[j] / (1 + x[i] * x[j])
    ts  = θ[i] + θ[j]
    F[i, j]          = fij
    What_proto[i, j] = fij / ts
    sigw_proto[i, j] = sqrt(fij * (2 - fij)) / ts
end

W_pkg = MaxEntropyGraphs.Ŵ(model)
dev_W = maximum(abs.(What_proto .- W_pkg))
ok("proto: What_proto == package Ŵ(m) (max |Δ| ≤ 1e-12)", dev_W <= 1e-12,
   detail="max|Δ| = $dev_W")

# ---------------------------------------------------------------------------------------
# (3) sample the ensemble with the real package sampler (fixed seed, continuous weights)
# ---------------------------------------------------------------------------------------
N = 10_000   # ≥ 3000 required; 16-node graph, well within the runtime budget
S = rand(model, N; rng = MaxEntropyGraphs.Xoshiro(161))
ok("sampling: drew N = $N graphs", length(S) == N)

# weights must be CONTINUOUS exponential draws: essentially none should be integer-valued
wpos = Float64[]
for g in S[1:100], e in Graphs.edges(g)
    push!(wpos, Graphs.weights(g)[Graphs.src(e), Graphs.dst(e)])
end
frac_noninteger = count(w -> !isinteger(w), wpos) / length(wpos)
ok("sampling: weights are continuous (non-integer fraction == 1)", frac_noninteger == 1.0,
   detail="non-integer fraction = $frac_noninteger over $(length(wpos)) sampled weights")

# 5 largest-⟨w⟩ node pairs (upper triangle, wᵢⱼ ≡ wⱼᵢ)
pairs = [(i, j) for i in 1:n for j in i+1:n]
sort!(pairs, by = p -> -What_proto[p...])
top = pairs[1:5]

# per-pair weight samples (SimpleWeightedGraphs: symmetric weight matrix for undirected graphs)
wsamples = Dict((i, j) => [Float64(g.weights[j, i]) for g in S] for (i, j) in top)

# entrywise mean gate (prerequisite for the variance gate): ⟨w⟩ within 5·SE
for (i, j) in top
    w = wsamples[(i, j)]
    se = sqrt(_var(w) / N)
    dev = abs(_mean(w) - What_proto[i, j])
    ok("mean  ⟨w⟩[$i,$j]: |emp - f/ts| ≤ 5·SE", dev <= 5 * se,
       detail="emp = $(round(_mean(w), sigdigits=5)), theory = $(round(What_proto[i,j], sigdigits=5)), dev/SE = $(round(dev/se, digits=2))")
end

# ---------------------------------------------------------------------------------------
# (4) entrywise variance gate: Var[w] == f(2-f)/ts² on the 5 largest-⟨w⟩ dyads (5·SE)
# ---------------------------------------------------------------------------------------
for (i, j) in top
    w = wsamples[(i, j)]
    v_emp = _var(w)
    # SE of the sample variance from the empirical 4th central moment: SE ≈ std((w-w̄)²)/√N
    centered_sq = (w .- _mean(w)) .^ 2
    se = sqrt(_var(centered_sq) / N)
    v_theory = sigw_proto[i, j]^2
    dev = abs(v_emp - v_theory)
    ok("var   Var[w][$i,$j]: |emp - f(2-f)/ts²| ≤ 5·SE", dev <= 5 * se,
       detail="emp = $(round(v_emp, sigdigits=5)), theory = $(round(v_theory, sigdigits=5)), dev/SE = $(round(dev/se, digits=2))")
end

# ---------------------------------------------------------------------------------------
# (5) aggregate delta-method gate: std(total weight) == sqrt(sum(σʷ.^2)/2)  (rtol 0.1)
#     sum(W) counts both triangles of the symmetric weight matrix, so sum(W)/2 is the total
#     edge weight; undirected identity covariance (wᵢⱼ ≡ wⱼᵢ) gives Var = Σ_{i<j} σʷᵢⱼ².
# ---------------------------------------------------------------------------------------
Wtot_samples = [sum(g.weights) / 2 for g in S]
σW_sampled  = sqrt(_var(Wtot_samples))
σW_analytic = sqrt(sum(sigw_proto .^ 2) / 2)
ok("aggregate: std(sum(W)/2) == sqrt(sum(σʷ²)/2) (rtol 0.1)",
   isapprox(σW_sampled, σW_analytic, rtol=0.1),
   detail="sampled = $(round(σW_sampled, sigdigits=5)), analytic = $(round(σW_analytic, sigdigits=5)), ratio = $(round(σW_sampled/σW_analytic, digits=4))")

# ---------------------------------------------------------------------------------------
# (6) cross-layer covariance gate: Cov(a,w) == ⟨w⟩(1-f) on the same dyads (5·SE)
# ---------------------------------------------------------------------------------------
for (i, j) in top
    w = wsamples[(i, j)]
    a = [Float64(Graphs.has_edge(g, i, j)) for g in S]
    cov_emp = _mean(a .* w) - _mean(a) * _mean(w)
    # SE from the spread of the centred products (delta-method SE of the covariance estimator)
    u = (a .- _mean(a)) .* (w .- _mean(w))
    se = sqrt(_var(u) / N)
    cov_theory = What_proto[i, j] * (1 - F[i, j])
    dev = abs(cov_emp - cov_theory)
    ok("cov   Cov(a,w)[$i,$j]: |emp - ⟨w⟩(1-f)| ≤ 5·SE", dev <= 5 * se,
       detail="emp = $(round(cov_emp, sigdigits=5)), theory = $(round(cov_theory, sigdigits=5)), dev/SE = $(round(dev/se, digits=2))")
end

# ---------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------
println()
println("="^100)
println(" CReM weighted-layer σʷ — Monte-Carlo gate (N = $N samples, seed 161, rhesus macaques n = $n)")
println("="^100)
npass = 0
for (name, cond, detail) in RESULTS
    global npass += cond
    status = cond ? "PASS" : "FAIL"
    println(rpad(" [$status] $name", 62), isempty(detail) ? "" : "  | $detail")
end
println("-"^100)
if npass == length(RESULTS)
    println(" ALL PASS ($npass/$(length(RESULTS)))")
    exit(0)
else
    println(" FAILURES: $(length(RESULTS) - npass)/$(length(RESULTS))")
    exit(1)
end
