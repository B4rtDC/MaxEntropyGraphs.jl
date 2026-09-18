###############################################################################
# bicm_variance.jl — Monte-Carlo gate for the BiCM variance machinery SHIPPED
# in src/ since v0.6.0: the per-entry σ layer (σˣ/set_σ!/σₓ, src/Models/BiCM.jl)
# and the Vn/Λn motif family Vn_motifs / Vn_sigma / Vn_zscore (src/metrics.jl).
#
# Model (Squartini & Garlaschelli 2011, NJP 13 083001, App. B — bipartite CM):
# the graph probability factorises per biadjacency entry; each entry (i,α) of
# the bottom×top biadjacency is an INDEPENDENT Bernoulli with
#     p_iα  = f_BiCM(x_i y_α) = x_i y_α / (1 + x_i y_α)     (src/Models/BiCM.jl:843, Ĝ:619)
# so Var[g_iα] = p_iα (1 - p_iα) and there is NO within-dyad cross term (each
# rectangular entry appears exactly once; no (i,α)/(α,i) mirror as in DBCM/RBCM).
#
# (1) Delta-method σ of a linear metric is exact:
#     σ[#edges] = sqrt(Σ_{iα} p_iα (1 - p_iα))              == sqrt(sum(sigx_proto.^2))
#
# (2) V-motif (Λ-motif) counts, Saracco et al. SI III.9–III.13. For n co-occurring
#     nodes of one layer, N_Vn = Σ_p binom(u_p, n) with u_p the OPPOSITE-layer
#     degrees. The package ships two routes, gated/reported here:
#     * method=:exact (DEFAULT, gated): each random opposite-layer degree
#       U_p ~ PoissonBinomial(p_col) (independent Bernoulli entries; distinct
#       columns/rows use disjoint entry sets ⇒ independent), so
#         ⟨N_Vn⟩_exact  = Σ_p E[binom(U_p,n)]   (= Σ_p e_n(p_col))
#         Var[N_Vn]_exact = Σ_p Var[binom(U_p,n)]
#       Vn_motifs/Vn_sigma/Vn_zscore(method=:exact) are gated against the
#       sampled mean/std/z AND cross-checked (~1e-10) against this script's own
#       PB convolution reference — same math, independent implementations
#       (reduced degree classes in src/ vs raw per-node columns here).
#     * method=:delta (INFORMATIONAL, not gated): the Saracco closed forms.
#       Degrees are conserved in expectation (⟨u_p⟩ = u_p), so with
#       s2_up = Σ_c p_cp (1 - p_cp) = Var[u_p] a Taylor expansion of binom(U_p, n)
#       around u_p gives ⟨N_Vn⟩ = N_Vn(observed) + shift_n with
#         shift_2 = Σ_p s2_up / 2               (exact: equals e₂ of the column
#                   probs, i.e. V_motifs(Ĝ) from src/metrics.jl — this identity
#                   IS gated below)
#         shift_3 = Σ_p s2_up (u_p - 1) / 2
#         shift_4 = Σ_p (3 s2_up² + s2_up (6 u_p² - 18 u_p + 11)) / 24
#       and the first-order delta std with H(u,n) = Σ_{i=u-n+1}^{u} 1/i:
#         σ[N_Vn] = sqrt(Σ_p (binom(u_p,n) H(u_p,n))² s2_up)  (term 0 if u_p < n)
#       These forms are asymptotic in the opposite-layer degrees: the n ≥ 3
#       shift misses the skewness term f'''·μ₃/6 and the delta σ UNDERESTIMATES
#       where the degrees are O(n) (small/sparse layers) — see the documented
#       validity regime in the Vn_motifs/Vn_sigma docstrings. Only the
#       delta/sampled ratios are printed (INFO lines), no gating.
#
# (3) Per-pair V-motif variance nuance (goes into the docs later): for a pair (i,j)
#     of one layer, V_ij = Σ_p g_ip g_jp is a Poisson-Binomial with q_p = p_ip p_jp
#     (V_PB_parameters, src/metrics.jl:2731), hence EXACTLY
#         Var[V_ij]_PB    = Σ_p q_p (1 - q_p)
#     while the first-order delta method gives
#         Var[V_ij]_delta = Σ_p (p_jp² s2_ip + p_ip² s2_jp),  s2_ip = p_ip(1-p_ip)
#     which UNDERESTIMATES by exactly the product-variance correction
#         Var_PB - Var_delta = Σ_p p_ip p_jp (1 - p_ip)(1 - p_jp).
#
# The script gates all of the above against the REAL package sampler
# (rand(m::BiCM, n; rng), src/Models/BiCM.jl:701) on two graphs:
#   * MaxEntropyGraphs.corporateclub()  (25 ⊥ + 15 ⊤)
#   * the planted bipartite graph of test/ensemble_validation.jl:25-37
#     (3 bottom hubs sharing 10 top neighbours — the 3 hub pairs drive part 3)
# Deterministic (fixed Xoshiro seeds); prints a pass/fail table; exits non-zero
# on failure.
###############################################################################

using MaxEntropyGraphs
const Graphs = MaxEntropyGraphs.Graphs

_mean(xs) = sum(xs) / length(xs)
_var(xs) = (m = _mean(xs); sum(abs2, xs .- m) / (length(xs) - 1))

# ---------------------------------------------------------------------------
# pass/fail bookkeeping
# ---------------------------------------------------------------------------
const RESULTS = Vector{Tuple{String,Bool,String}}()
function ok(name::AbstractString, cond::Bool; detail::AbstractString="")
    push!(RESULTS, (String(name), cond, String(detail)))
    return cond
end

r5(x) = round(x, sigdigits=5)

# planted bipartite helper — copied verbatim from test/ensemble_validation.jl:25-37
function _planted_bipartite()
    Nb, Nt, hubs, hubdeg = 24, 100, 3, 10
    g = Graphs.SimpleGraph(Nb + Nt)
    topnode(j) = Nb + j
    for b in 1:hubs, j in 1:hubdeg
        Graphs.add_edge!(g, b, topnode(j))
    end
    for b in (hubs+1):Nb, k in 0:2
        j = ((b * 7 + k * 13) % Nt) + 1
        Graphs.add_edge!(g, b, topnode(j))
    end
    # The patterns above leave part of the top layer untouched, and an isolated vertex has no
    # determinable layer: `bipartite_map` would put every one of them in ⊥, turning this 24x100 graph
    # into a 57x67 model. `BiCM` refuses such input, so attach the leftovers deterministically — the
    # same repair `test/ensemble_validation.jl`'s copy of this fixture already carries. The planted hub
    # co-occurrence the checks below rely on is untouched.
    for j in 1:Nt
        Graphs.degree(g, topnode(j)) == 0 && Graphs.add_edge!(g, hubs + 1 + (j % (Nb - hubs)), topnode(j))
    end
    return g
end

"Poisson-Binomial pmf of Σ Bernoulli(ps) via convolution; d[k] = P(U = k-1)"
function _pb_dist(ps::AbstractVector{<:Real})
    d = zeros(Float64, length(ps) + 1)
    d[1] = 1.0
    for p in ps
        for k in length(d):-1:2
            d[k] = d[k] * (1 - p) + d[k-1] * p
        end
        d[1] *= (1 - p)
    end
    return d
end

"exact (E[binom(U,n)], Var[binom(U,n)]) for U ~ PoissonBinomial(ps)"
function _pb_binom_moments(ps::AbstractVector{<:Real}, n::Int)
    d = _pb_dist(ps)
    Ef  = sum(d[k] * binomial(k - 1, n) for k in eachindex(d))
    Ef2 = sum(d[k] * binomial(k - 1, n)^2 for k in eachindex(d))
    return Ef, Ef2 - Ef^2
end

# ---------------------------------------------------------------------------
# per-graph analysis
# ---------------------------------------------------------------------------
function analyze!(label::String, G; nsamples::Int, seed::Int, hubpairs=nothing)
    model = BiCM(G)
    solve_model!(model)     # default :fixedpoint, ftol 1e-8
    set_Ĝ!(model)

    n⊥, n⊤ = model.status[:N⊥], model.status[:N⊤]
    x = model.xᵣ[model.d⊥ᵣ_ind]     # per-node fitted fitnesses, un-reduced
    y = model.yᵣ[model.d⊤ᵣ_ind]

    # per-entry Bernoulli p from f_BiCM at the fitted parameters (bottom rows × top cols);
    # must reproduce the package's expected biadjacency Ĝ(m)
    P = [MaxEntropyGraphs.f_BiCM(x[i] * y[a]) for i in 1:n⊥, a in 1:n⊤]
    ok("[$label] p_proto = f_BiCM(x·y) == Ĝ(m) (src)",
       maximum(abs.(P .- model.Ĝ)) < 1e-12,
       detail="max|Δ| = $(r5(maximum(abs.(P .- model.Ĝ))))")
    S2 = P .* (1 .- P)              # per-entry Bernoulli variance p(1-p)

    # observed biadjacency in the MODEL's membership convention (⊥nodes rows, ⊤nodes cols)
    B = [Graphs.has_edge(G, nb, nt) ? 1 : 0 for nb in model.⊥nodes, nt in model.⊤nodes]
    u⊤ = vec(sum(B, dims=1))        # observed degrees of the top-layer nodes (columns)
    u⊥ = vec(sum(B, dims=2))        # observed degrees of the bottom-layer nodes (rows)

    # degree conservation ⟨u_p⟩ = u_p — the shift formulas rely on it
    dev⊤ = maximum(abs.(vec(sum(P, dims=1)) .- u⊤))
    dev⊥ = maximum(abs.(vec(sum(P, dims=2)) .- u⊥))
    ok("[$label] degree conservation ⟨u⟩ == u (both layers, atol 1e-4)",
       dev⊤ < 1e-4 && dev⊥ < 1e-4,
       detail="max|Δ| top = $(r5(dev⊤)), bottom = $(r5(dev⊥))")

    # -- sampling (REAL package sampler, per-sample seeds from the passed rng) --
    S = rand(model, nsamples; rng=MaxEntropyGraphs.Xoshiro(seed))

    # =======================================================================
    # (1) σ of the biadjacency sum (= #edges): delta method with NO cross term
    # =======================================================================
    sigx_proto = sqrt.(S2)                       # per-entry σ[g_iα]
    σL_delta = sqrt(sum(sigx_proto .^ 2))        # entries independent, each counted once
    nes = [Float64(Graphs.ne(g)) for g in S]
    σL_samp = sqrt(_var(nes))
    ok("[$label] (1) ⟨#edges⟩: sum(P) vs sampled mean (rtol 0.02)",
       isapprox(sum(P), _mean(nes), rtol=0.02),
       detail="ana = $(r5(sum(P))), samp = $(r5(_mean(nes)))")
    ok("[$label] (1) σ[#edges]: delta vs sampled std (rtol 0.1)",
       isapprox(σL_delta, σL_samp, rtol=0.1),
       detail="delta = $(r5(σL_delta)), samp = $(r5(σL_samp)), ratio = $(round(σL_delta/σL_samp, digits=4))")

    # =======================================================================
    # (2) V-/Λ-motif totals N_Vn, n = 2,3,4, both layers
    # =======================================================================
    # sampled N_Vn = Σ_p binom(u_p, n) needs only the opposite-layer degrees of each sample
    NVsamp = Dict{Tuple{Symbol,Int},Vector{Float64}}()
    for layer in (:bottom, :top), n in 2:4
        NVsamp[(layer, n)] = Vector{Float64}(undef, nsamples)
    end
    for (si, g) in enumerate(S)
        dtop = Graphs.degree(g, model.⊤nodes)
        dbot = Graphs.degree(g, model.⊥nodes)
        for n in 2:4
            NVsamp[(:bottom, n)][si] = sum(u -> binomial(u, n), dtop)  # bottom pairs share top nodes
            NVsamp[(:top, n)][si]    = sum(u -> binomial(u, n), dbot)  # top pairs share bottom nodes
        end
    end

    for layer in (:bottom, :top)
        u = layer === :bottom ? u⊤ : u⊥                                  # opposite-layer observed degrees

        # observed N_V2 must tie to the package's V_motifs on the observed biadjacency
        ok("[$label/$layer] (2) N_V2(obs) == V_motifs(B) (src, exact)",
           sum(v -> binomial(v, 2), u) == V_motifs(B; layer=layer, skipchecks=true),
           detail="N_V2(obs) = $(sum(v -> binomial(v, 2), u))")

        for n in 2:4
            Nobs = sum(v -> binomial(v, n), u)
            xs = NVsamp[(layer, n)]
            μ_s, σ_s = _mean(xs), sqrt(_var(xs))

            # the package's observed count (matrix method) must tie to the script's Nobs
            ok("[$label/$layer] (2) Vn_motifs(B, $n) == N_V$n(obs) (src)",
               isapprox(Vn_motifs(B, n; layer=layer, skipchecks=true), Nobs, rtol=1e-12),
               detail="pkg = $(r5(Vn_motifs(B, n; layer=layer, skipchecks=true))), obs = $Nobs")

            # in-script exact reference: per opposite-layer node, U_p ~ PoissonBinomial(column/row of P)
            μ_ex, var_ex = 0.0, 0.0
            for p in eachindex(u)
                ps = layer === :bottom ? view(P, :, p) : view(P, p, :)
                Ef, Vf = _pb_binom_moments(ps, n)
                μ_ex += Ef
                var_ex += Vf
            end
            σ_ex = sqrt(var_ex)

            # package values, :exact (default) — the gated route
            μ_pkg = Vn_motifs(model, n; layer=layer, method=:exact)
            σ_pkg = Vn_sigma(model, n; layer=layer, method=:exact)
            z_pkg = Vn_zscore(model, n; layer=layer, method=:exact)

            # package :exact must reproduce the script's own PB convolution reference:
            # same math, independent implementations (reduced degree classes in src/
            # vs raw per-node columns/rows here)
            ok("[$label/$layer] (2) pkg ⟨N_V$n⟩ (:exact) == script PB reference (rtol 1e-10)",
               isapprox(μ_pkg, μ_ex, rtol=1e-10),
               detail="pkg = $(r5(μ_pkg)), script = $(r5(μ_ex))")
            ok("[$label/$layer] (2) pkg σ[N_V$n] (:exact) == script PB reference (rtol 1e-10)",
               isapprox(σ_pkg, σ_ex, rtol=1e-10),
               detail="pkg = $(r5(σ_pkg)), script = $(r5(σ_ex))")

            if n == 2
                # the n = 2 closed form is exact: pkg :delta mean == e₂(P) == V_motifs(Ĝ) from src
                vĜ = V_motifs(model.Ĝ; layer=layer, skipchecks=true)
                μ_d2 = Vn_motifs(model, 2; layer=layer, method=:delta)
                ok("[$label/$layer] (2) pkg ⟨N_V2⟩ (:delta) == V_motifs(Ĝ) (src, rtol 1e-6)",
                   isapprox(μ_d2, vĜ, rtol=1e-6),
                   detail="delta = $(r5(μ_d2)), V_motifs(Ĝ) = $(r5(vĜ))")
            end

            # exact PB reference vs sampling — validates the sampler & accumulation (5·SE)
            se_mean = σ_s / sqrt(nsamples)
            m4 = _mean((xs .- μ_s) .^ 4)
            s² = σ_s^2
            se_var = sqrt(max(m4 - s²^2 * (nsamples - 3) / (nsamples - 1), 0.0) / nsamples)
            ok("[$label/$layer] (2) exact ⟨N_V$n⟩ (PB) vs sampled mean (5·SE)",
               abs(μ_ex - μ_s) <= 5 * se_mean,
               detail="exact = $(r5(μ_ex)), samp = $(r5(μ_s)), dev/SE = $(round(abs(μ_ex - μ_s)/se_mean, digits=2))")
            ok("[$label/$layer] (2) exact Var[N_V$n] (PB) vs sampled var (5·SE)",
               abs(var_ex - s²) <= 5 * se_var,
               detail="exact = $(r5(var_ex)), samp = $(r5(s²)), dev/SE = $(round(abs(var_ex - s²)/se_var, digits=2))")

            # GATES on the shipped :exact route, original bands
            ok("[$label/$layer] (2) pkg ⟨N_V$n⟩ (:exact) vs sampled mean (rtol 0.05)",
               isapprox(μ_pkg, μ_s, rtol=0.05),
               detail="pkg = $(r5(μ_pkg)), samp = $(r5(μ_s)), Nobs = $Nobs")
            ok("[$label/$layer] (2) pkg σ[N_V$n] (:exact) vs sampled std (rtol 0.15)",
               isapprox(σ_pkg, σ_s, rtol=0.15),
               detail="pkg = $(r5(σ_pkg)), samp = $(r5(σ_s)), pkg/samp = $(round(σ_pkg/σ_s, digits=4))")

            if σ_pkg > 0 && σ_s > 0
                z_s = (Nobs - μ_s) / σ_s
                zdiff = abs(z_pkg - z_s)
                ok("[$label/$layer] (2) pkg z_$n (:exact): |Δz| < 0.35 or rtol 0.2",
                   zdiff < 0.35 || zdiff <= 0.2 * abs(z_s),
                   detail="z_pkg = $(r5(z_pkg)), z_samp = $(r5(z_s)), |Δz| = $(r5(zdiff))")
            else
                ok("[$label/$layer] (2) pkg z_$n (:exact): |Δz| < 0.35 or rtol 0.2", false,
                   detail="degenerate σ (pkg = $(r5(σ_pkg)), samp = $(r5(σ_s)))")
            end

            # INFORMATIONAL only (not gated): the :delta closed forms. Asymptotic in the
            # opposite-layer degrees; documented to underestimate σ for sparse layers
            # (see the validity regime in the Vn_motifs/Vn_sigma docstrings), e.g. the
            # planted graph's low-degree layers. Report the delta/sampled ratios.
            μ_d = Vn_motifs(model, n; layer=layer, method=:delta)
            σ_d = Vn_sigma(model, n; layer=layer, method=:delta)
            z_d = σ_d > 0 ? Vn_zscore(model, n; layer=layer, method=:delta) : NaN
            z_s = (Nobs - μ_s) / σ_s
            println("  INFO [$label/$layer] :delta n=$n  ⟨N⟩ ana/samp = $(round(μ_d/μ_s, digits=4)), " *
                    "σ ana/samp = $(round(σ_d/σ_s, digits=4)), z_delta = $(r5(z_d)) vs z_samp = $(r5(z_s))")
        end
    end

    # =======================================================================
    # (3) per-pair Var(V_ij): Poisson-Binomial (exact) vs first-order delta
    # =======================================================================
    if hubpairs !== nothing
        mem = zeros(Int, Graphs.nv(G))
        mem[model.⊥nodes] .= 1
        mem[model.⊤nodes] .= 2
        for (ni, nj) in hubpairs
            i, j = model.⊥map[ni], model.⊥map[nj]        # graph node -> biadjacency row
            q   = MaxEntropyGraphs.V_PB_parameters(model, i, j; layer=:bottom, precomputed=false)
            qpc = MaxEntropyGraphs.V_PB_parameters(model, i, j; layer=:bottom, precomputed=true)
            ok("[$label] (3) pair($ni,$nj): V_PB_parameters reduced == precomputed == pᵢ∘pⱼ (src)",
               maximum(abs.(q .- qpc)) < 1e-12 && maximum(abs.(q .- P[i, :] .* P[j, :])) < 1e-12,
               detail="max|Δ| = $(r5(max(maximum(abs.(q .- qpc)), maximum(abs.(q .- P[i, :] .* P[j, :])))))")

            var_pb    = sum(q .* (1 .- q))                                        # (a) exact
            var_delta = sum(P[j, :] .^ 2 .* S2[i, :] .+ P[i, :] .^ 2 .* S2[j, :]) # (b) 1st-order delta
            corr      = sum(P[i, :] .* P[j, :] .* (1 .- P[i, :]) .* (1 .- P[j, :]))
            ok("[$label] (3) pair($ni,$nj): (a) - (b) == Σ pᵢpⱼ(1-pᵢ)(1-pⱼ) (identity)",
               abs((var_pb - var_delta) - corr) <= 1e-12 * max(1.0, var_pb),
               detail="(a) = $(r5(var_pb)), (b) = $(r5(var_delta)), correction = $(r5(corr))")
            ok("[$label] (3) pair($ni,$nj): delta underestimates PB (b) < (a)",
               var_delta < var_pb,
               detail="(b)/(a) = $(round(var_delta/var_pb, digits=4))")

            # sampled Var(V_ij) — PB variance must be exact up to MC error (5·SE of s²)
            vs = [Float64(V_motifs(g, ni, nj; membership=mem)) for g in S]
            v̄  = _mean(vs)
            s² = _var(vs)
            m4 = _mean((vs .- v̄) .^ 4)
            se = sqrt(max(m4 - s²^2 * (nsamples - 3) / (nsamples - 1), 0.0) / nsamples)
            dev = abs(s² - var_pb)
            ok("[$label] (3) pair($ni,$nj): sampled Var(V) vs PB var (5·SE)",
               dev <= 5 * se,
               detail="samp = $(r5(s²)), PB = $(r5(var_pb)), delta = $(r5(var_delta)), dev/SE = $(round(dev/se, digits=2))")
        end
    end

    return nothing
end

# ---------------------------------------------------------------------------
# run on both graphs (fixed seeds, deterministic)
# ---------------------------------------------------------------------------
analyze!("club", MaxEntropyGraphs.corporateclub(); nsamples=20000, seed=161)
analyze!("planted", _planted_bipartite(); nsamples=20000, seed=163,
         hubpairs=[(1, 2), (1, 3), (2, 3)])

# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
println()
println("="^118)
println(" BiCM variance machinery — Monte-Carlo gate (corporateclub & planted bipartite, 20000 samples each)")
println("="^118)
npass = 0
for (name, cond, detail) in RESULTS
    global npass += cond
    status = cond ? "PASS" : "FAIL"
    println(rpad(" [$status] $name", 72), isempty(detail) ? "" : "  | $detail")
end
println("-"^118)
if npass == length(RESULTS)
    println(" ALL PASS ($npass/$(length(RESULTS)))")
    exit(0)
else
    println(" FAILURES: $(length(RESULTS) - npass)/$(length(RESULTS))")
    exit(1)
end
