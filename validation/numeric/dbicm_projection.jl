# DBiCM directed projection — numeric (Monte-Carlo) validation.
#
# A directed bipartite pair (i,j) can share a neighbour α in three ways, because either link may point
# either way:
#     :out    Σ_α B⁺_iα B⁺_jα     i → α ← j      symmetric
#     :in     Σ_α B⁻_iα B⁻_jα     i ← α → j      symmetric
#     :path   Σ_α B⁺_iα B⁻_jα     i → α → j      asymmetric
#
# The claim this script tests is that ALL THREE are exactly Poisson-binomial under the DBiCM, with
# success probabilities q_α equal to the corresponding product of two entry probabilities. That holds
# because every factor is a product of two DISTINCT Bernoulli entries, and the entries are independent
# both within a dyad and across α — the algebraic half is proved in validation/symbolic/dbicm.jl; here
# the law is checked against sampling, and not only through its first two moments: the full pmf is
# compared with `Distributions.PoissonBinomial`, which is the statement that the law is exact rather
# than merely correctly centred.
#
# Also checked: the `:path` diagonal is the reciprocated-partner count; V^path_ij and V^path_ji are
# independent (they draw on disjoint entries); the three totals count over the index sets the
# docstrings claim (unordered pairs for the symmetric kinds, ordered for `:path`); and the analytical
# `reciprocity` matches sampling.
#
# Sampling is done by drawing the two channels' Bernoulli entries directly rather than through
# `rand(m)`, so the check is independent of the sampler under test.

using MaxEntropyGraphs
using Distributions
using Random

const M = MaxEntropyGraphs
const G_ = MaxEntropyGraphs.Graphs

const RESULTS = Vector{Tuple{String,Bool,String}}()
function ok(name::AbstractString, cond::Bool; detail::AbstractString="")
    push!(RESULTS, (String(name), cond, String(detail)))
    return cond
end

function planted_dibipartite(seed, Nb, Nt, p⁺, p⁻)
    rng = Xoshiro(seed)
    g = G_.SimpleDiGraph(Nb + Nt)
    for i in 1:Nb, j in 1:Nt
        rand(rng) < p⁺ && G_.add_edge!(g, i, Nb + j)
        rand(rng) < p⁻ && G_.add_edge!(g, Nb + j, i)
    end
    for v in G_.vertices(g)
        iszero(G_.degree(g, v)) && (v <= Nb ? G_.add_edge!(g, v, Nb + 1 + (v % Nt)) :
                                              G_.add_edge!(g, 1 + (v % Nb), v))
    end
    return g
end

const NSAMPLES = 60_000

G = planted_dibipartite(5, 9, 7, 0.40, 0.30)
model = DBiCM(G)
solve_model!(model, method = :BFGS)
ok("model converged", constraint_residual(model) < 1e-7,
   detail = "residual $(round(constraint_residual(model), sigdigits=3))")

N⊥, N⊤ = model.status[:N⊥], model.status[:N⊤]
P⁺ = M.Ĝ(model, channel = :to_top)
P⁻ = M.Ĝ(model, channel = :to_bottom)

rng = Xoshiro(20260918)
samples = [(rand(rng, N⊥, N⊤) .< P⁺, rand(rng, N⊥, N⊤) .< P⁻) for _ in 1:NSAMPLES]
mean_(v) = sum(v) / length(v)
var_(v)  = (mu = mean_(v); sum(abs2, v .- mu) / (length(v) - 1))

# --- the three kernels, both layers: moments and the FULL law -------------------------------------
for layer in (:bottom, :top), kind in (:out, :in, :path)
    n = layer === :bottom ? N⊥ : N⊤
    # pick the pair with the largest expectation, so the check is not trivially satisfied by a pair
    # whose expected count happens to be zero
    i, j = argmax(((a, b) -> M.V_motifs(model, a, b; layer = layer, kind = kind)).(
                   [a for a in 1:n, b in 1:n], [b for a in 1:n, b in 1:n]) .*
                  [a == b ? 0 : 1 for a in 1:n, b in 1:n]).I
    obs = [M.V_motifs(Int.(a), Int.(b), i, j; layer = layer, kind = kind) for (a, b) in samples]
    q   = M.V_PB_parameters(model, i, j; layer = layer, kind = kind)
    μ   = M.V_motifs(model, i, j; layer = layer, kind = kind)
    ok("⟨V⟩ analytic == sampled ($layer/$kind)", isapprox(μ, mean_(obs), atol = 0.05),
       detail = "$(round(μ, digits=4)) vs $(round(mean_(obs), digits=4))")
    ok("Var[V] == Σ q(1-q) == sampled ($layer/$kind)", isapprox(sum(q .* (1 .- q)), var_(obs), rtol = 0.06),
       detail = "$(round(sum(q .* (1 .- q)), digits=4)) vs $(round(var_(obs), digits=4))")
    pb  = PoissonBinomial(q)
    emp = [count(==(k), obs) / length(obs) for k in 0:length(q)]
    dev = maximum(abs.(emp .- pdf.(pb, 0:length(q))))
    ok("pmf == PoissonBinomial, not just its moments ($layer/$kind)", dev < 0.01,
       detail = "max |Δpmf| = $(round(dev, sigdigits=3))")
end

# --- the :path diagonal, and the independence of the two directed tests of a pair -----------------
i = 2
recip = [sum(a[i, α] * b[i, α] for α in 1:N⊤) for (a, b) in samples]
ok("reciprocated_degree == sampled", isapprox(M.reciprocated_degree(model, i), mean_(recip), atol = 0.05),
   detail = "$(round(M.reciprocated_degree(model, i), digits=4)) vs $(round(mean_(recip), digits=4))")
ok("reciprocated_degree == the :path diagonal",
   isapprox(M.reciprocated_degree(model, i), M.V_motifs(model, i, i; layer = :bottom, kind = :path), rtol = 1e-12))

vij = [M.V_motifs(Int.(a), Int.(b), 1, 3; layer = :bottom, kind = :path) for (a, b) in samples]
vji = [M.V_motifs(Int.(a), Int.(b), 3, 1; layer = :bottom, kind = :path) for (a, b) in samples]
cov = mean_(vij .* vji) - mean_(vij) * mean_(vji)
ok("Cov(V^path_ij, V^path_ji) == 0 (disjoint entry sets)", abs(cov) < 0.03,
   detail = "measured $(round(cov, sigdigits=3))")

# --- totals count over the index set the docstrings claim ----------------------------------------
for layer in (:bottom, :top), kind in (:out, :in, :path)
    tot = [M.V_motifs(Int.(a), Int.(b); layer = layer, kind = kind) for (a, b) in samples[1:5000]]
    ok("total ⟨V⟩ == sampled ($layer/$kind)",
       isapprox(M.V_motifs(model; layer = layer, kind = kind), mean_(tot), rtol = 0.05),
       detail = "$(round(M.V_motifs(model; layer=layer, kind=kind), digits=3)) vs $(round(mean_(tot), digits=3))")
end

# --- reciprocity is a ratio OF expectations, so it is compared to the same ratio of sampled means --
m⁺ = sum(first.(samples)) ./ NSAMPLES
m⁻ = sum(last.(samples)) ./ NSAMPLES
r_s = 2 * sum(m⁺ .* m⁻) / sum(m⁺ .+ m⁻)
ok("reciprocity == sampled", isapprox(M.reciprocity(model), r_s, rtol = 0.05),
   detail = "$(round(M.reciprocity(model), digits=4)) vs $(round(r_s, digits=4))")

# --- report ---------------------------------------------------------------------------------------
println("\n=== DBiCM projection — numeric validation ($(NSAMPLES) samples) ===")
for (name, cond, detail) in RESULTS
    println(rpad(cond ? "PASS" : "FAIL", 6), rpad(name, 58), detail)
end
npass = count(r -> r[2], RESULTS)
println(npass == length(RESULTS) ? " ALL PASS ($npass/$(length(RESULTS)))" :
                                   " $(length(RESULTS) - npass) FAILURE(S) of $(length(RESULTS))")
exit(npass == length(RESULTS) ? 0 : 1)
