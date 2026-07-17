###############################################################################
# decm_weighted_sigma.jl
#
# Monte-Carlo gate for the DECM weighted-layer standard deviation σʷ and the
# covariance-FREE directed delta method of σₓ(m::DECM, ...).
#
# Channel distribution (Vallarano et al. 2021, arXiv:2101.12625; enhanced CM
# with non-negative integer weights). Per ORDERED pair (i→j), writing
# x = xᵢ_out·xⱼ_in and y = yᵢ_out·yⱼ_in from the fitted reduced parameters
# expanded to node level (the access pattern of src/Models/DECM.jl:592-773),
# the sampler rand(m::DECM) (src/Models/DECM.jl:1176-1224) draws
#   an edge with probability  p = x·y / (1 - y + x·y)          (= f_DECM, src:567)
#   given an edge:            w = 1 + Geometric(1 - y)
# so ⟨w|edge⟩ = 1/(1-y) and Var[w|edge] = y/(1-y)². Unconditionally:
#   ⟨w⟩    = p/(1-y)                                (matches Ŵ in src:638-673)
#   ⟨w²⟩   = p·(Var[w|edge] + ⟨w|edge⟩²) = p(1+y)/(1-y)²
#   Var[w] = ⟨w²⟩ - ⟨w⟩² = p(1+y-p)/(1-y)²          (σʷ² — the gate, src:738-773)
#   Cov(a,w) = ⟨a·w⟩ - ⟨a⟩⟨w⟩ = ⟨w⟩(1-p)            (a·w = w since w=0 ⇔ a=0)
# Distinct ORDERED pairs are independent — in particular w_ij and w_ji are
# DISTINCT independent variables (the defining difference with the UECM, whose
# w_ij ≡ w_ji). So for the total weight X(W) = sum(W) (gradient 1 on every
# off-diagonal entry):
#   σ²[X] = Σ_{i≠j} Var[w_ij] = sum(σʷ.^2)
# with NO within-dyad covariance correction — the UECM-style corrected delta
# sqrt(2·sum(σʷ.^2)) would OVERcount by √2 here.
#
# The script validates all of this against the REAL package on the directed
# anchor graph of the DECM testsets in test/models.jl (rhesus macaques, N = 16,
# used unsymmetrised), following the Monte-Carlo patterns of
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
# (1) build & solve the DECM on the test/models.jl anchor graph
# ---------------------------------------------------------------------------
Gw = MaxEntropyGraphs.rhesus_macaques()
model = DECM(Gw)
# BFGS with g_tol=1e-5, exactly as the test/models.jl parameter-computation testset
solve_model!(model, method=:BFGS, g_tol=1e-5)
# store the expected matrices and standard deviations (required by the package σₓ used in (6))
set_Ĝ!(model); set_σ!(model); set_Ŵ!(model); set_σʷ!(model)

n = model.status[:N]
Apkg = MaxEntropyGraphs.Ĝ(model)
Wpkg = MaxEntropyGraphs.Ŵ(model)
ok("solve: out-degree constraint reproduced (rtol 1e-5)",
   isapprox(vec(sum(Apkg, dims=2)), model.d_out, rtol=1e-5))
ok("solve: in-degree constraint reproduced (rtol 1e-5)",
   isapprox(vec(sum(Apkg, dims=1)), model.d_in, rtol=1e-5))
ok("solve: out-strength constraint reproduced (rtol 1e-5)",
   isapprox(vec(sum(Wpkg, dims=2)), model.s_out, rtol=1e-5))
ok("solve: in-strength constraint reproduced (rtol 1e-5)",
   isapprox(vec(sum(Wpkg, dims=1)), model.s_in, rtol=1e-5))

# ---------------------------------------------------------------------------
# (2) per-channel moments, computed in-script from the fitted parameters
# ---------------------------------------------------------------------------
# access pattern of Ĝ/Ŵ/σˣ/σʷ (src/Models/DECM.jl:592-773): reduced params → node level
x_out = model.xᵣ_out[model.dᵣ_ind]
x_in  = model.xᵣ_in[model.dᵣ_ind]
y_out = model.yᵣ_out[model.dᵣ_ind]
y_in  = model.yᵣ_in[model.dᵣ_ind]
# dead-channel guard: src keeps the live channels in m.dᵣ_out_nz / m.dᵣ_in_nz.
# On this anchor graph every channel is live, so the guard is moot — assert that
# so the proto matrices below are comparable to Ŵ everywhere.
ok("nz guard: all out/in channels live on the anchor graph (guard moot)",
   length(model.dᵣ_out_nz) == length(model.dᵣ_out) && length(model.dᵣ_in_nz) == length(model.dᵣ_in))

P_proto    = zeros(n, n)   # pᵢⱼ = x·y/(1-y+x·y)
What_proto = zeros(n, n)   # ⟨wᵢⱼ⟩ = p/(1-y)
sigw_proto = zeros(n, n)   # σʷᵢⱼ = sqrt(p(1+y-p)/(1-y)²)
for i in 1:n
    for j in 1:n
        i == j && continue
        xixj = x_out[i] * x_in[j]
        yiyj = y_out[i] * y_in[j]
        p = (xixj * yiyj) / (1 - yiyj + xixj * yiyj)
        w = p / (1 - yiyj)
        v = p * (1 + yiyj - p) / (1 - yiyj)^2
        P_proto[i, j] = p
        What_proto[i, j] = w
        sigw_proto[i, j] = sqrt(v)
    end
end

# the in-script matrices must match the package's Ĝ(m)/Ŵ(m)/σʷ(m) everywhere
sigw_pkg = MaxEntropyGraphs.σʷ(model)
maxdev_W = maximum(abs.(What_proto .- Wpkg))
maxdev_A = maximum(abs.(P_proto .- Apkg))
maxdev_S = maximum(abs.(sigw_proto .- sigw_pkg))
println("max |Ŵ_proto - Ŵ(m)| = $maxdev_W ; max |P_proto - Ĝ(m)| = $maxdev_A ; max |σʷ_proto - σʷ(m)| = $maxdev_S")
ok("proto: What_proto == Ŵ(m) everywhere (atol 1e-12)", maxdev_W < 1e-12)
ok("proto: P_proto == Ĝ(m) everywhere (atol 1e-12)", maxdev_A < 1e-12)
ok("proto: sigw_proto == σʷ(m) everywhere (atol 1e-12)", maxdev_S < 1e-12)

# ---------------------------------------------------------------------------
# (3) sample the ensemble (rand(m, n; rng) is seed-reproducible)
# ---------------------------------------------------------------------------
N = 10_000
S = rand(model, N; rng = MaxEntropyGraphs.Xoshiro(161))
ok("sampling: got $N graphs of $n vertices",
   length(S) == N && all(g -> Graphs.nv(g) == n, S))

# the 5 largest-⟨w⟩ ORDERED pairs
pairs = [(i, j) for i in 1:n for j in 1:n if i ≠ j]
sort!(pairs, by = p -> -What_proto[p...])
top5 = pairs[1:5]

# ---------------------------------------------------------------------------
# (4) entrywise: sampled Var(w_ij) vs σʷ² on the 5 heaviest ordered pairs
#     band: 5·SE with SE(sample variance) ≈ Var·sqrt(2/(N-1))
#     (generous, in the spirit of test/ensemble_validation.jl)
# ---------------------------------------------------------------------------
println("\nentrywise checks on the 5 largest-⟨w⟩ ordered pairs (N = $N samples):")
println(rpad("pair", 10), rpad("⟨w⟩ proto", 12), rpad("w̄ sampled", 12),
        rpad("σʷ² proto", 12), rpad("Var sampled", 13), rpad("Cov(a,w) proto", 16), "Cov(a,w) sampled")
for (i, j) in top5
    # SimpleWeightedGraphs stores weights[dst, src]: entry [j, i] is the weight of the edge i→j
    ws = [Float64(g.weights[j, i]) for g in S]
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
    # cross-layer: Cov(a,w) vs ⟨w⟩(1-p), 5·SE band (SE from the centered product,
    # floored by the conservative sqrt(Var(a)Var(w)/N) of test/ensemble_validation.jl)
    se_cov = max(sqrt(_var(zc) / N), sqrt(_var(as) * v̂ / N))
    ok("cross-layer pair ($i,$j): |Côv(a,w) - ⟨w⟩(1-p)| ≤ 5·SE",
       abs(emp_cov - cov_target) <= 5 * se_cov)
end

# ---------------------------------------------------------------------------
# (5) directed independence: empirical Cov(w_ij, w_ji) compatible with zero on
#     the 5 heaviest dyads (this is what licenses the covariance-free σₓ)
# ---------------------------------------------------------------------------
ud_pairs = [(i, j) for i in 1:n for j in i+1:n]
sort!(ud_pairs, by = p -> -(What_proto[p...] + What_proto[p[2], p[1]]))
for (i, j) in ud_pairs[1:5]
    w_ij = [Float64(g.weights[j, i]) for g in S] # weights[dst, src]
    w_ji = [Float64(g.weights[i, j]) for g in S]
    emp_cov = _mean(w_ij .* w_ji) - _mean(w_ij) * _mean(w_ji)
    se = sqrt(_var(w_ij) * _var(w_ji) / N)
    ok("directed independence dyad ($i,$j): |Côv(w_ij, w_ji)| ≤ 5·SE",
       abs(emp_cov) <= 5 * se)
end

# ---------------------------------------------------------------------------
# (6) aggregate: X(W) = sum(W) (total weight). Gradient is 1 on every
#     off-diagonal entry; all ordered pairs are independent so the delta-method
#     sigma is sqrt(sum(σʷ.^2)) — the UECM-style within-dyad correction
#     sqrt(2·sum(σʷ.^2)) would OVERcount here.
# ---------------------------------------------------------------------------
Wtot_samples = [sum(g.weights) for g in S]
σ_delta = sqrt(sum(sigw_proto .^ 2))
σ_pkg = MaxEntropyGraphs.σₓ(model, sum, layer=:weighted)
σ_overcorrected = sqrt(2 * sum(sigw_proto .^ 2)) # the UECM-style cross-term, wrong for a directed model
σ_sampled = sqrt(_var(Wtot_samples))
W̄tot_expected = sum(What_proto)
W̄tot_sampled = _mean(Wtot_samples)
println("\naggregate total weight: ⟨X⟩ = $W̄tot_expected vs sampled $W̄tot_sampled")
println("σ[X]: independent delta = $σ_delta ; package σₓ = $σ_pkg ; UECM-style overcorrection = $σ_overcorrected ; sampled = $σ_sampled")
ok("aggregate: ⟨sum(W)⟩ matches sampled mean (rtol 0.05)",
   isapprox(W̄tot_expected, W̄tot_sampled, rtol=0.05))
ok("aggregate: package σₓ == sqrt(sum(σʷ²)) (the covariance-free delta)",
   isapprox(σ_pkg, σ_delta, rtol=1e-10))
ok("aggregate: independent delta σ = sqrt(sum(σʷ²)) matches sampled σ (rtol 0.1)",
   isapprox(σ_delta, σ_sampled, rtol=0.1))
ok("aggregate: the UECM-style within-dyad correction overcounts (σ_overcorrected > 1.2·σ_sampled)",
   σ_overcorrected > 1.2 * σ_sampled)

# ---------------------------------------------------------------------------
# pass/fail table
# ---------------------------------------------------------------------------
println("\n=== DECM weighted-layer σʷ — Monte-Carlo validation ===")
for (name, pass) in RESULTS
    println(rpad(pass ? "PASS" : "FAIL", 6), name)
end
nfail = count(r -> !r[2], RESULTS)
println(nfail == 0 ? "ALL PASS ($(length(RESULTS)) checks)" : "$nfail FAILURE(S)")
exit(nfail == 0 ? 0 : 1)
