###############################################################################
# uecm_weighted_sigma.jl
#
# Monte-Carlo gate for the PROPOSED UECM weighted-layer standard deviation σʷ
# (NOT yet implemented in src/Models/UECM.jl — the current σˣ covers the binary
# layer only; see the note in its docstring).
#
# Dyadic distribution (Squartini & Garlaschelli 2011, NJP 13 083001, App. B;
# enhanced CM with non-negative integer weights). Per dyad (i,j), writing
# x = xᵢxⱼ and y = yᵢyⱼ from the fitted reduced parameters expanded to node
# level (the access pattern of src/Models/UECM.jl:453-530), the sampler
# rand(m::UECM) (src/Models/UECM.jl:790-830) draws
#   an edge with probability  p = x·y / (1 - y + x·y)          (= f_UECM, src:429)
#   given an edge:            w = 1 + Geometric(1 - y)
# so ⟨w|edge⟩ = 1/(1-y) and Var[w|edge] = y/(1-y)². Unconditionally:
#   ⟨w⟩    = p/(1-y)                                (matches Ŵ in src:496-520)
#   ⟨w²⟩   = p·(Var[w|edge] + ⟨w|edge⟩²) = p(1+y)/(1-y)²
#   Var[w] = ⟨w²⟩ - ⟨w⟩² = p(1+y-p)/(1-y)²          (PROPOSED σʷ² — the gate)
#   Cov(a,w) = ⟨a·w⟩ - ⟨a⟩⟨w⟩ = ⟨w⟩(1-p)            (a·w = w since w=0 ⇔ a=0)
# Distinct dyads are independent and w_ij ≡ w_ji (undirected), so for the
# total weight X(W) = sum(W)/2 (gradient 1/2 on every off-diagonal entry):
#   σ²[X] = Σ_{i<j} Var[w_ij] = sum(σʷ.^2)/2
# (the naive independent-entry delta sqrt(sum((σʷ/2).^2)) would undercount by
# √2 because Cov(w_ij, w_ji) = Var[w_ij], not 0).
#
# The script validates all of this against the REAL package on the weighted
# anchor graph of the UECM testsets in test/models.jl (symmetrised rhesus
# macaques, N = 16), following the Monte-Carlo patterns of
# test/ensemble_validation.jl. Deterministic: fixed Xoshiro seeds throughout.
###############################################################################

using MaxEntropyGraphs
const Graphs = MaxEntropyGraphs.Graphs
const SWG = MaxEntropyGraphs.SimpleWeightedGraphs

_mean(xs) = sum(xs) / length(xs)
_var(xs) = (m = _mean(xs); sum(abs2, xs .- m) / (length(xs) - 1))

const RESULTS = Tuple{String,Bool}[]
"Record a named check; warn on failure so the offending numbers are visible in the log."
function ok(name::String, cond::Bool)
    push!(RESULTS, (name, cond))
    cond || @warn "FAILED: $name"
    return cond
end

# ---------------------------------------------------------------------------
# (1) build & solve the UECM on the test/models.jl anchor graph
# ---------------------------------------------------------------------------
Gw = SWG.SimpleWeightedGraph(MaxEntropyGraphs.rhesus_macaques())
model = UECM(Gw)
# BFGS with g_tol=1e-5, exactly as the test/models.jl parameter-computation testset
solve_model!(model, method=:BFGS, g_tol=1e-5)

n = model.status[:N]
Apkg = MaxEntropyGraphs.Ĝ(model)
Wpkg = MaxEntropyGraphs.Ŵ(model)
ok("solve: degree constraint reproduced (rtol 1e-5)",
   isapprox(vec(sum(Apkg, dims=2)), model.d, rtol=1e-5))
ok("solve: strength constraint reproduced (rtol 1e-5)",
   isapprox(vec(sum(Wpkg, dims=2)), model.s, rtol=1e-5))

# ---------------------------------------------------------------------------
# (2) proposed per-edge moments, computed in-script from the fitted parameters
# ---------------------------------------------------------------------------
# access pattern of Ĝ/Ŵ/σˣ (src/Models/UECM.jl:453-530): reduced params → node level
x = model.xᵣ[model.dᵣ_ind]
y = model.yᵣ[model.dᵣ_ind]
# zero-degree guard: src keeps the non-zero constraint index set in m.nz
# (nz = findall(!iszero, dᵣ), src:134); a zero-degree class has no meaningful
# (x,y). On this anchor graph every reduced degree is non-zero, so the guard is
# moot — assert that so the proto matrices below are comparable to Ŵ everywhere.
ok("nz guard: all reduced degrees non-zero on the anchor graph (guard moot)",
   length(model.nz) == length(model.dᵣ))

P_proto    = zeros(n, n)   # pᵢⱼ = x·y/(1-y+x·y)
What_proto = zeros(n, n)   # ⟨wᵢⱼ⟩ = p/(1-y)
sigw_proto = zeros(n, n)   # σʷᵢⱼ = sqrt(p(1+y-p)/(1-y)²)
for i in 1:n
    for j in i+1:n
        xixj = x[i] * x[j]
        yiyj = y[i] * y[j]
        p = (xixj * yiyj) / (1 - yiyj + xixj * yiyj)
        w = p / (1 - yiyj)
        v = p * (1 + yiyj - p) / (1 - yiyj)^2
        P_proto[i, j] = p;          P_proto[j, i] = p
        What_proto[i, j] = w;       What_proto[j, i] = w
        sigw_proto[i, j] = sqrt(v); sigw_proto[j, i] = sqrt(v)
    end
end

# the in-script expected-weight matrix must match the package's Ŵ(m) everywhere
maxdev_W = maximum(abs.(What_proto .- Wpkg))
maxdev_A = maximum(abs.(P_proto .- Apkg))
println("max |Ŵ_proto - Ŵ(m)| = $maxdev_W ; max |P_proto - Ĝ(m)| = $maxdev_A")
ok("proto: What_proto == Ŵ(m) everywhere (atol 1e-12)", maxdev_W < 1e-12)
ok("proto: P_proto == Ĝ(m) everywhere (atol 1e-12)", maxdev_A < 1e-12)

# ---------------------------------------------------------------------------
# (3) sample the ensemble (≥ 3000 graphs; rand(m, n; rng) is seed-reproducible)
# ---------------------------------------------------------------------------
N = 10_000
S = rand(model, N; rng = MaxEntropyGraphs.Xoshiro(161))
ok("sampling: got $N graphs of $n vertices",
   length(S) == N && all(g -> Graphs.nv(g) == n, S))

# the 5 largest-⟨w⟩ node pairs (upper triangle)
pairs = [(i, j) for i in 1:n for j in i+1:n]
sort!(pairs, by = p -> -What_proto[p...])
top5 = pairs[1:5]

# ---------------------------------------------------------------------------
# (4) entrywise: sampled Var(w_ij) vs proposed σʷ² on the 5 heaviest pairs
#     band: 5·SE with SE(sample variance) ≈ Var·sqrt(2/(N-1))
#     (generous, in the spirit of test/ensemble_validation.jl)
# ---------------------------------------------------------------------------
println("\nentrywise checks on the 5 largest-⟨w⟩ pairs (N = $N samples):")
println(rpad("pair", 10), rpad("⟨w⟩ proto", 12), rpad("w̄ sampled", 12),
        rpad("σʷ² proto", 12), rpad("Var sampled", 13), rpad("Cov(a,w) proto", 16), "Cov(a,w) sampled")
for (i, j) in top5
    # SimpleWeightedGraphs stores weights[dst, src]; symmetric here, so [i,j] is fine
    ws = [Float64(g.weights[i, j]) for g in S]
    as = Float64.(ws .> 0)
    w̄, v̂ = _mean(ws), _var(ws)
    # cross-layer empirical covariance and its per-sample SE (std of the centered product)
    zc = (as .- _mean(as)) .* (ws .- w̄)
    emp_cov = _mean(as .* ws) - _mean(as) * w̄
    cov_target = What_proto[i, j] * (1 - P_proto[i, j])
    println(rpad("($i,$j)", 10), rpad(string(round(What_proto[i, j], digits=4)), 12),
            rpad(string(round(w̄, digits=4)), 12),
            rpad(string(round(sigw_proto[i, j]^2, digits=4)), 12),
            rpad(string(round(v̂, digits=4)), 13),
            rpad(string(round(cov_target, digits=4)), 16),
            string(round(emp_cov, digits=4)))

    # mean weight: 5·SE(w̄) band
    ok("mean ⟨w⟩ pair ($i,$j): |w̄ - ⟨w⟩| ≤ 5·SE",
       abs(w̄ - What_proto[i, j]) <= 5 * sqrt(v̂ / N))
    # variance: 5·SE(sample variance) band, SE ≈ Var·sqrt(2/(N-1))
    ok("variance pair ($i,$j): |Var̂(w) - σʷ²| ≤ 5·SE",
       abs(v̂ - sigw_proto[i, j]^2) <= 5 * v̂ * sqrt(2 / (N - 1)))
    # (6) cross-layer: Cov(a,w) vs ⟨w⟩(1-p), 5·SE band (SE from the centered product,
    #     floored by the conservative sqrt(Var(a)Var(w)/N) of test/ensemble_validation.jl)
    se_cov = max(sqrt(_var(zc) / N), sqrt(_var(as) * v̂ / N))
    ok("cross-layer pair ($i,$j): |Côv(a,w) - ⟨w⟩(1-p)| ≤ 5·SE",
       abs(emp_cov - cov_target) <= 5 * se_cov)
end

# ---------------------------------------------------------------------------
# (5) aggregate: X(W) = sum(W)/2 (total weight). Gradient is 1/2 on every
#     off-diagonal entry; with Cov(w_ij, w_ji) = Var[w_ij] (same variable) the
#     corrected delta-method sigma is sqrt(sum(σʷ.^2)/2).
# ---------------------------------------------------------------------------
Wtot_samples = [sum(g.weights) / 2 for g in S]     # g.weights is symmetric → /2
σ_delta = sqrt(sum(sigw_proto .^ 2) / 2)
σ_naive = sqrt(sum((sigw_proto ./ 2) .^ 2))        # covariance-blind delta (should undercount by √2)
σ_sampled = sqrt(_var(Wtot_samples))
W̄tot_expected = sum(What_proto) / 2
W̄tot_sampled = _mean(Wtot_samples)
println("\naggregate total weight: ⟨X⟩ = $W̄tot_expected vs sampled $W̄tot_sampled")
println("σ[X]: corrected delta = $σ_delta ; naive (no within-dyad cov) = $σ_naive ; sampled = $σ_sampled")
ok("aggregate: ⟨sum(W)/2⟩ matches sampled mean (rtol 0.05)",
   isapprox(W̄tot_expected, W̄tot_sampled, rtol=0.05))
ok("aggregate: corrected delta σ = sqrt(sum(σʷ²)/2) matches sampled σ (rtol 0.1)",
   isapprox(σ_delta, σ_sampled, rtol=0.1))
ok("aggregate: covariance-blind delta undercounts (σ_naive < 0.9·σ_sampled)",
   σ_naive < 0.9 * σ_sampled)

# ---------------------------------------------------------------------------
# pass/fail table
# ---------------------------------------------------------------------------
println("\n=== UECM weighted-layer σʷ — Monte-Carlo validation ===")
for (name, pass) in RESULTS
    println(rpad(pass ? "PASS" : "FAIL", 6), name)
end
nfail = count(r -> !r[2], RESULTS)
println(nfail == 0 ? "ALL PASS ($(length(RESULTS)) checks)" : "$nfail FAILURE(S)")
exit(nfail == 0 ? 0 : 1)
