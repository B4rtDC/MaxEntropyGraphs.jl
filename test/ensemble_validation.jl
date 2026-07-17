using Test
using MaxEntropyGraphs

# End-to-end validation of the package's headline scientific claims, complementing the
# unit tests in models.jl / metrics.jl:
#   1. Solver parity   - the different solution methods converge to the SAME maximum-likelihood
#                        point (not just each satisfying the constraints in isolation).
#   2. Ensemble vs.    - the analytically computed ensemble averages (degrees, edges, triangles,
#      sampling          strengths, bipartite motifs) agree with a Monte-Carlo estimate from sampled
#                        graphs, and the sampler reproduces the models' analytical second moments
#                        (edge-count / weight variances and the delta-method σₓ, including the
#                        within-dyad covariance terms of the undirected and reciprocal models).
#   3. Projection      - the statistical validation of bipartite projections actually filters:
#      significance      a stricter significance level yields a subset of the looser one.
# All stochastic checks use a fixed RNG seed (Xoshiro), so they are deterministic.

const Graphs = MaxEntropyGraphs.Graphs

_mean(xs) = sum(xs) / length(xs)
_var(xs) = (m = _mean(xs); sum(abs2, xs .- m) / (length(xs) - 1))

# A deterministic bipartite graph with a *planted* significant co-occurrence: three "hub"
# bottom nodes each share the same 10 top neighbours (out of 100), so their observed
# co-occurrence (~10) far exceeds the BiCM expectation (~10^2/100 = 1) and must survive the
# projection's significance filter. The remaining bottom nodes get a sparse deterministic
# background so the degree sequence is non-degenerate.
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
    return g
end

@testset "ensemble validation" begin

    @testset "solver parity (UBCM)" begin
        G = Graphs.SimpleGraphs.smallgraph(:karate)
        d = Graphs.degree(G)
        # (method, analytical_gradient) combinations that should all reach the MLE
        configs = [(:fixedpoint, false), (:BFGS, true), (:BFGS, false), (:LBFGS, true), (:Newton, false)]
        expected = Vector{Vector{Float64}}()
        for (method, ag) in configs
            m = UBCM(G)
            solve_model!(m, method = method, analytical_gradient = ag)
            push!(expected, Float64.(degree(m, method = :reduced)))
        end
        # every method reproduces the observed degree sequence (the constraint / MLE)
        for e in expected
            @test isapprox(e, d, rtol = 1e-3)
        end
        # every method converges to the same point as the fixed-point reference
        for e in expected[2:end]
            @test isapprox(expected[1], e, rtol = 1e-3)
        end
    end

    @testset "solver parity (DBCM)" begin
        G = MaxEntropyGraphs.maspalomas()
        dout = Graphs.outdegree(G)
        din = Graphs.indegree(G)
        mats = Matrix{Float64}[]
        for (method, ag) in [(:fixedpoint, false), (:BFGS, true), (:BFGS, false), (:Newton, false)]
            m = DBCM(G)
            solve_model!(m, method = method, analytical_gradient = ag)
            push!(mats, Float64.(MaxEntropyGraphs.Ĝ(m)))
        end
        # each solution reproduces the directed degree constraints
        for A in mats
            @test isapprox(vec(sum(A, dims = 2)), dout, rtol = 1e-3)
            @test isapprox(vec(sum(A, dims = 1)), din, rtol = 1e-3)
        end
        # all methods agree on the expected adjacency matrix
        for A in mats[2:end]
            @test isapprox(mats[1], A, rtol = 1e-3)
        end
    end

    @testset "solver parity (RBCM)" begin
        G = Graphs.SimpleDiGraph(MaxEntropyGraphs.rhesus_macaques())
        dout = MaxEntropyGraphs.nonreciprocated_outdegree(G)
        din = MaxEntropyGraphs.nonreciprocated_indegree(G)
        drec = MaxEntropyGraphs.reciprocated_degree(G)
        mats = Matrix{Float64}[]
        for (method, ag) in [(:fixedpoint, false), (:BFGS, true), (:BFGS, false), (:Newton, false)]
            m = RBCM(G)
            solve_model!(m, method = method, analytical_gradient = ag)
            # each solution reproduces the three reciprocal degree constraints
            @test isapprox(MaxEntropyGraphs.nonreciprocated_outdegree(m), dout, rtol = 1e-3)
            @test isapprox(MaxEntropyGraphs.nonreciprocated_indegree(m), din, rtol = 1e-3)
            @test isapprox(MaxEntropyGraphs.reciprocated_degree(m), drec, rtol = 1e-3)
            push!(mats, Float64.(MaxEntropyGraphs.Ĝ(m)))
        end
        # all methods agree on the expected adjacency matrix
        for A in mats[2:end]
            @test isapprox(mats[1], A, rtol = 1e-3)
        end
    end

    @testset "analytical ensemble averages vs sampling (RBCM)" begin
        G = Graphs.SimpleDiGraph(MaxEntropyGraphs.rhesus_macaques())
        nv = Graphs.nv(G)
        model = RBCM(G)
        solve_model!(model)
        set_Ĝ!(model)
        set_σ!(model)
        N = 2000
        S = rand(model, N; rng = MaxEntropyGraphs.Xoshiro(161))
        @test length(S) == N

        # (a) sampled mean reciprocal degree sequences == the constraints (exact in expectation)
        sampled_rec = zeros(Float64, nv)
        sampled_nro = zeros(Float64, nv)
        for g in S
            sampled_rec .+= MaxEntropyGraphs.reciprocated_degree(g)
            sampled_nro .+= MaxEntropyGraphs.nonreciprocated_outdegree(g)
        end
        sampled_rec ./= N
        sampled_nro ./= N
        @test isapprox(sampled_rec, model.d_rec, atol = 0.4)
        @test isapprox(sampled_nro, model.d_out, atol = 0.4)

        # (b) the sampler reproduces the analytical within-dyad covariance Cov(aᵢⱼ, aⱼᵢ) = p⭤ - ⟨aᵢⱼ⟩⟨aⱼᵢ⟩
        #     (the defining difference with the DBCM). Checked on the 5 largest-|C| dyads.
        C = MaxEntropyGraphs._cov_dyads(model)
        pairs = [(i, j) for i in 1:nv for j in i+1:nv]
        sort!(pairs, by = p -> -abs(C[p...]))
        for (i, j) in pairs[1:5]
            a_ij = [Float64(Graphs.has_edge(g, i, j)) for g in S]
            a_ji = [Float64(Graphs.has_edge(g, j, i)) for g in S]
            emp_cov = _mean(a_ij .* a_ji) - _mean(a_ij) * _mean(a_ji)
            # standard error of a covariance of Bernoullis is ≲ 1/(2√N); use a 4·SE guard band
            @test isapprox(emp_cov, C[i, j], atol = 4 / (2 * sqrt(N)))
        end

        # (c) exact analytical motif expectations == sampled means (Squartini eq. C.16)
        expected_motifs = motifs(model)
        sampled_motifs = _mean([motifs(Matrix(Graphs.adjacency_matrix(g))) for g in S])
        for k in [1, 6, 8, 13] # well-populated motifs on this network
            @test isapprox(expected_motifs[k], sampled_motifs[k], rtol = 0.1)
        end

        # (d) the covariance-aware delta method is exact for linear metrics:
        #     Var(#links) = Σ σ²ᵢⱼ + Σ Cov(aᵢⱼ,aⱼᵢ) reproduced by the sampler
        analytical_σL = MaxEntropyGraphs.σₓ(model, sum)
        ne_samples = [Graphs.ne(g) for g in S]
        @test isapprox(analytical_σL, sqrt(_var(ne_samples)), rtol = 0.1)
        # without the covariance term the variance would be underestimated on this
        # high-reciprocity network (this is what distinguishes the RBCM from the DBCM)
        @test sqrt(sum(model.σ .^ 2)) < 0.9 * sqrt(_var(ne_samples))
    end

    @testset "analytical ensemble averages vs sampling (DCReM)" begin
        G = MaxEntropyGraphs.rhesus_macaques()
        nv = Graphs.nv(G)
        model = DCReM(G)
        solve_model!(model)
        set_Ĝ!(model); set_σ!(model); set_Ŵ!(model); set_σʷ!(model)
        N = 2000
        S = rand(model, N; rng = MaxEntropyGraphs.Xoshiro(161))

        # (a) sampled mean out/in-strengths == the constraints (exact in expectation);
        #     per-node tolerance from the weight variances: 5·SE of the row sums
        sampled_sout = zeros(Float64, nv)
        for g in S
            sampled_sout .+= MaxEntropyGraphs.outstrength(g)
        end
        sampled_sout ./= N
        se_sout = sqrt.(vec(sum(model.σʷ .^ 2, dims = 2)) ./ N)
        @test all(abs.(sampled_sout .- model.s_out) .<= 5 .* se_sout)

        # (b) under the DBCM binary layer the two directions of a dyad are independent:
        #     empirical Cov(wᵢⱼ, wⱼᵢ) is statistically compatible with zero on the heaviest dyads
        Wm = model.Ŵ
        pairs = [(i, j) for i in 1:nv for j in i+1:nv]
        sort!(pairs, by = p -> -(Wm[p...] + Wm[p[2], p[1]]))
        for (i, j) in pairs[1:5]
            w_ij = [Float64(g.weights[j, i]) for g in S] # SimpleWeightedGraphs stores weights[dst, src]
            w_ji = [Float64(g.weights[i, j]) for g in S]
            emp_cov = _mean(w_ij .* w_ji) - _mean(w_ij) * _mean(w_ji)
            se = sqrt(_var(w_ij) * _var(w_ji) / N)
            @test abs(emp_cov) <= 5 * se
        end

        # (c) the weighted delta method is exact for linear metrics: σ[W_tot] reproduced by the sampler
        analytical_σW = MaxEntropyGraphs.σₓ(model, sum, layer = :weighted)
        Wtot_samples = [sum(g.weights) for g in S]
        @test isapprox(analytical_σW, sqrt(_var(Wtot_samples)), rtol = 0.1)
    end

    @testset "analytical ensemble averages vs sampling (CRWCM)" begin
        G = MaxEntropyGraphs.rhesus_macaques()
        nv = Graphs.nv(G)
        model = CRWCM(G)
        solve_model!(model)
        set_Ĝ!(model); set_σ!(model); set_Ŵ!(model); set_σʷ!(model)
        N = 2000
        S = rand(model, N; rng = MaxEntropyGraphs.Xoshiro(161))

        # (a) sampled mean reciprocal strength sequences == the constraints (exact in expectation);
        #     per-node tolerance from the weight variances: 5·SE of the row sums (loose upper bound,
        #     using the total-weight variance as a proxy for each channel)
        sampled_srec = zeros(Float64, nv)
        for g in S
            sampled_srec .+= MaxEntropyGraphs.reciprocated_outstrength(g)
        end
        sampled_srec ./= N
        se_row = sqrt.(vec(sum(model.σʷ .^ 2, dims = 2)) ./ N)
        @test all(abs.(sampled_srec .- model.s_rec_out) .<= 5 .* se_row)

        # (b) the sampler reproduces the analytical within-dyad WEIGHT covariance
        #     Cov(wᵢⱼ, wⱼᵢ) (the defining difference with the DCReM). Checked on the 3 largest-|Cʷ| dyads.
        Cw = MaxEntropyGraphs._covʷ(model)
        pairs = [(i, j) for i in 1:nv for j in i+1:nv]
        sort!(pairs, by = p -> -abs(Cw[p...]))
        for (i, j) in pairs[1:3]
            w_ij = [Float64(g.weights[j, i]) for g in S] # SimpleWeightedGraphs stores weights[dst, src]
            w_ji = [Float64(g.weights[i, j]) for g in S]
            emp_cov = _mean(w_ij .* w_ji) - _mean(w_ij) * _mean(w_ji)
            se = sqrt(_var(w_ij) * _var(w_ji) / N) # conservative SE for the covariance estimator
            @test abs(emp_cov - Cw[i, j]) <= 5 * se
            @test Cw[i, j] > 0 # reciprocated-heavy dyads have positively correlated weights
        end

        # (c) the covariance-aware weighted delta method is exact for linear metrics:
        #     σ[W_tot] reproduced by the sampler, and underestimated without the covariance term
        analytical_σW = MaxEntropyGraphs.σₓ(model, sum, layer = :weighted)
        Wtot_samples = [sum(g.weights) for g in S]
        @test isapprox(analytical_σW, sqrt(_var(Wtot_samples)), rtol = 0.15)
        @test sqrt(sum(model.σʷ .^ 2)) < analytical_σW # the covariance term is positive on this network

        # (d) the exact expected triadic fluxes match the ensemble means (well-populated motifs)
        expected_F = motif_fluxes(model)
        sampled_F = _mean([motif_fluxes(Matrix(transpose(g.weights))) for g in S])
        for k in [1, 6, 8, 13]
            @test isapprox(expected_F[k], sampled_F[k], rtol = 0.1)
        end
    end

    @testset "exact expected fluxes vs sampling (DCReM)" begin
        model = DCReM(MaxEntropyGraphs.rhesus_macaques())
        solve_model!(model)
        S = rand(model, 2000; rng = MaxEntropyGraphs.Xoshiro(161))
        expected_F = motif_fluxes(model)
        sampled_F = _mean([motif_fluxes(Matrix(transpose(g.weights))) for g in S])
        for k in [1, 6, 8, 13]
            @test isapprox(expected_F[k], sampled_F[k], rtol = 0.1)
        end
    end

    @testset "analytical ensemble averages vs sampling (UBCM)" begin
        G = Graphs.SimpleGraphs.smallgraph(:karate)
        nv = Graphs.nv(G)
        model = UBCM(G)
        solve_model!(model)
        set_Ĝ!(model)
        set_σ!(model)
        N = 4000
        S = rand(model, N; rng = MaxEntropyGraphs.Xoshiro(161))
        @test length(S) == N

        # (a) expected degree sequence == sampled mean degree (exact in expectation)
        expected_deg = Float64.(degree(model, method = :reduced))
        sampled_deg = zeros(Float64, nv)
        for g in S
            sampled_deg .+= Graphs.degree(g)
        end
        sampled_deg ./= N
        @test isapprox(expected_deg, sampled_deg, atol = 0.4)

        # (b) expected number of edges == sampled mean (exact in expectation)
        Ghat = MaxEntropyGraphs.Ĝ(model)
        expected_edges = sum(Ghat) / 2
        ne_samples = [Graphs.ne(g) for g in S]
        @test isapprox(expected_edges, _mean(ne_samples), rtol = 0.05)

        # (c) expected triangle count == sampled mean (analytical motif over the ensemble)
        expected_tri = triangles(model)
        sampled_tri = _mean([triangles(g) for g in S])
        @test isapprox(expected_tri, sampled_tri, rtol = 0.1)

        # (d) the sampler reproduces the model's analytical edge-count variance.
        #     For undirected entries the upper triangle is independent, so
        #     Var(#edges) = sum_{i<j} p_ij (1 - p_ij) exactly.
        analytical_var = sum(Ghat[i, j] * (1 - Ghat[i, j]) for i in 1:nv-1 for j in i+1:nv)
        @test isapprox(analytical_var, _var(ne_samples), rtol = 0.15)

        # (e) regression test for the undirected delta method: the entries (i,j) and (j,i) of the
        #     adjacency matrix are the SAME random variable, so σₓ must include the within-dyad
        #     cross-term Cov(a_ij, a_ji) = σ²[a_ij]. With X = half the full-matrix sum (= #edges),
        #     σₓ then reproduces both the exact edge-count variance of (d) and the sampler.
        analytical_σL = MaxEntropyGraphs.σₓ(model, A -> sum(A) / 2)
        @test isapprox(analytical_σL, sqrt(analytical_var), rtol = 1e-6)
        @test isapprox(analytical_σL, sqrt(_var(ne_samples)), rtol = 0.1)
    end

    @testset "analytical ensemble averages vs sampling (UECM)" begin
        # symmetrised rhesus network (the UECM requires integer weights)
        G = MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(MaxEntropyGraphs.rhesus_macaques())
        nv = Graphs.nv(G)
        model = UECM(G)
        solve_model!(model, method = :BFGS) # the fixed point recipe is unstable for the UECM
        set_Ĝ!(model); set_σ!(model); set_Ŵ!(model); set_σʷ!(model)
        N = 2000
        S = rand(model, N; rng = MaxEntropyGraphs.Xoshiro(161))
        @test length(S) == N

        # (a) sampled mean total weight == half the sum of the Ŵ-based strengths (exact in expectation)
        Wtot_samples = [sum(g.weights) / 2 for g in S] # symmetric weight matrix -> halve the full sum
        expected_Wtot = sum(vec(sum(model.Ŵ, dims = 2))) / 2
        @test isapprox(_mean(Wtot_samples), expected_Wtot, rtol = 0.05)

        # (b) the sampler reproduces the analytical weight variance Var(wᵢⱼ) = σʷᵢⱼ² of the
        #     Bernoulli-geometric mixture. Checked on the 3 largest-⟨w⟩ dyads with a 5·SE guard band
        #     (SE of a sample variance from the empirical fourth moment).
        pairs = [(i, j) for i in 1:nv for j in i+1:nv]
        sort!(pairs, by = p -> -model.Ŵ[p...])
        for (i, j) in pairs[1:3]
            w = [Float64(g.weights[i, j]) for g in S]
            emp_var = _var(w)
            m4 = _mean((w .- _mean(w)) .^ 4)
            se = sqrt((m4 - emp_var^2) / N)
            @test abs(emp_var - model.σʷ[i, j]^2) <= 5 * se
        end

        # (c) the weighted delta method (with the undirected within-dyad cross-term) is exact for
        #     linear metrics: σ[W_tot] reproduced by the sampler
        analytical_σW = MaxEntropyGraphs.σₓ(model, W -> sum(W) / 2, layer = :weighted)
        @test isapprox(analytical_σW, sqrt(_var(Wtot_samples)), rtol = 0.1)

        # (d) same for the binary layer: σ[#edges] reproduced by the sampler
        analytical_σL = MaxEntropyGraphs.σₓ(model, A -> sum(A) / 2)
        ne_samples = [sum(g.weights .> 0) / 2 for g in S] # binary adjacency = weights .> 0
        @test isapprox(analytical_σL, sqrt(_var(ne_samples)), rtol = 0.1)
    end

    @testset "analytical ensemble averages vs sampling (DECM)" begin
        # directed rhesus network, used unsymmetrised (the DECM requires integer weights)
        G = MaxEntropyGraphs.rhesus_macaques()
        nv = Graphs.nv(G)
        model = DECM(G)
        solve_model!(model, method = :BFGS) # the fixed point recipe is unstable for the DECM
        set_Ĝ!(model); set_σ!(model); set_Ŵ!(model); set_σʷ!(model)
        N = 2000
        S = rand(model, N; rng = MaxEntropyGraphs.Xoshiro(161))
        @test length(S) == N

        # (a) sampled mean total weight == the sum of Ŵ (exact in expectation; the pairs are
        #     ordered for a directed model, so no halving as in the UECM)
        Wtot_samples = [sum(g.weights) for g in S]
        expected_Wtot = sum(model.Ŵ)
        @test isapprox(_mean(Wtot_samples), expected_Wtot, rtol = 0.05)

        # (b) the sampler reproduces the analytical weight variance Var(wᵢⱼ) = σʷᵢⱼ² of the
        #     Bernoulli-geometric mixture. Checked on the 3 largest-⟨w⟩ ordered pairs with a 5·SE
        #     guard band (SE of a sample variance from the empirical fourth moment).
        pairs = [(i, j) for i in 1:nv for j in 1:nv if i ≠ j]
        sort!(pairs, by = p -> -model.Ŵ[p...])
        for (i, j) in pairs[1:3]
            w = [Float64(g.weights[j, i]) for g in S] # SimpleWeightedGraphs stores weights[dst, src]
            emp_var = _var(w)
            m4 = _mean((w .- _mean(w)) .^ 4)
            se = sqrt((m4 - emp_var^2) / N)
            @test abs(emp_var - model.σʷ[i, j]^2) <= 5 * se
        end

        # (c) the two directions of a dyad are independent under the DECM: the empirical
        #     Cov(wᵢⱼ, wⱼᵢ) is statistically compatible with zero on the heaviest dyads
        ud_pairs = [(i, j) for i in 1:nv for j in i+1:nv]
        sort!(ud_pairs, by = p -> -(model.Ŵ[p...] + model.Ŵ[p[2], p[1]]))
        for (i, j) in ud_pairs[1:5]
            w_ij = [Float64(g.weights[j, i]) for g in S] # SimpleWeightedGraphs stores weights[dst, src]
            w_ji = [Float64(g.weights[i, j]) for g in S]
            emp_cov = _mean(w_ij .* w_ji) - _mean(w_ij) * _mean(w_ji)
            se = sqrt(_var(w_ij) * _var(w_ji) / N)
            @test abs(emp_cov) <= 5 * se
        end

        # (d) the weighted delta method (no within-dyad cross-term for a directed model) is exact
        #     for linear metrics: σ[W_tot] reproduced by the sampler
        analytical_σW = MaxEntropyGraphs.σₓ(model, sum, layer = :weighted)
        @test isapprox(analytical_σW, sqrt(_var(Wtot_samples)), rtol = 0.1)

        # (e) same for the binary layer: σ[#links] reproduced by the sampler
        analytical_σL = MaxEntropyGraphs.σₓ(model, sum)
        ne_samples = [Graphs.ne(g) for g in S]
        @test isapprox(analytical_σL, sqrt(_var(ne_samples)), rtol = 0.1)
    end

    @testset "analytical ensemble averages vs sampling (CReM)" begin
        # symmetrised rhesus network; the CReM models CONTINUOUS weights conditional on a UBCM
        # binary layer (solve_model! performs the two-step solve: binary layer, then weighted layer)
        G = MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(MaxEntropyGraphs.rhesus_macaques())
        nv = Graphs.nv(G)
        model = CReM(G)
        solve_model!(model) # the CReM fixed-point recipe is stable (unlike the UECM's)
        set_Ĝ!(model); set_σ!(model); set_Ŵ!(model); set_σʷ!(model)
        N = 2000
        S = rand(model, N; rng = MaxEntropyGraphs.Xoshiro(161))
        # continuous weights (exponential conditional on the edge being present)
        @test all(w -> w > 0, MaxEntropyGraphs.SimpleWeightedGraphs.weight.(collect(Graphs.edges(S[1]))))

        # (a) sampled mean total weight == half the sum of the Ŵ-based strengths (exact in expectation)
        Wtot_samples = [sum(g.weights) / 2 for g in S]
        expected_Wtot = sum(vec(sum(model.Ŵ, dims = 2))) / 2
        @test isapprox(_mean(Wtot_samples), expected_Wtot, rtol = 0.05)

        # (b) the sampler reproduces the analytical weight variance Var(wᵢⱼ) = fᵢⱼ(2-fᵢⱼ)/(θᵢ+θⱼ)²
        #     of the Bernoulli-exponential mixture. Checked on the 3 largest-⟨w⟩ dyads (5·SE band).
        pairs = [(i, j) for i in 1:nv for j in i+1:nv]
        sort!(pairs, by = p -> -model.Ŵ[p...])
        for (i, j) in pairs[1:3]
            w = [Float64(g.weights[i, j]) for g in S]
            emp_var = _var(w)
            m4 = _mean((w .- _mean(w)) .^ 4)
            se = sqrt((m4 - emp_var^2) / N)
            @test abs(emp_var - model.σʷ[i, j]^2) <= 5 * se
        end

        # (c) the weighted delta method (with the undirected within-dyad cross-term) is exact for
        #     linear metrics: σ[W_tot] reproduced by the sampler
        analytical_σW = MaxEntropyGraphs.σₓ(model, W -> sum(W) / 2, layer = :weighted)
        @test isapprox(analytical_σW, sqrt(_var(Wtot_samples)), rtol = 0.1)

        # (d) same for the binary layer: σ[#edges] reproduced by the sampler
        analytical_σL = MaxEntropyGraphs.σₓ(model, A -> sum(A) / 2)
        ne_samples = [sum(g.weights .> 0) / 2 for g in S] # binary adjacency = weights .> 0
        @test isapprox(analytical_σL, sqrt(_var(ne_samples)), rtol = 0.1)
    end

    @testset "analytical ensemble averages vs sampling (BiCM)" begin
        Gb = MaxEntropyGraphs.corporateclub()
        model = BiCM(Gb)
        solve_model!(model)
        set_Ĝ!(model); set_σ!(model)
        N = 2000
        S = rand(model, N; rng = MaxEntropyGraphs.Xoshiro(161))
        # the sampled graphs keep the original vertex labels, so the model's layer memberships give
        # the (n⊥ × n⊤) biadjacency matrix of any sample with a consistent orientation
        biadjacency(g) = Graphs.adjacency_matrix(g)[model.⊥nodes, model.⊤nodes]

        # (a) the biadjacency entries are independent Bernoulli variables (no cross-term): the delta
        #     method reproduces the sampled edge-count standard deviation. The sum over the
        #     biadjacency matrix counts each edge ONCE (unlike an adjacency matrix).
        ne_samples = [Graphs.ne(g) for g in S]
        @test isapprox(MaxEntropyGraphs.σₓ(model, sum), sqrt(_var(ne_samples)), rtol = 0.1)

        # (b) Λ3 motifs (triple co-occurrences between top-layer nodes): the exact Poisson-binomial
        #     expectation and standard deviation are reproduced by the sampler
        v3_samples = [Vn_motifs(biadjacency(g), 3, layer = :top, skipchecks = true) for g in S]
        @test isapprox(_mean(v3_samples), Vn_motifs(model, 3, layer = :top), rtol = 0.05)
        @test isapprox(sqrt(_var(v3_samples)), Vn_sigma(model, 3, layer = :top), rtol = 0.15)

        # (c) the analytical z-score matches the Monte-Carlo z-score of the observed count
        observed_v3 = Vn_motifs(biadjacency(Gb), 3, layer = :top, skipchecks = true)
        z_sampled = (observed_v3 - _mean(v3_samples)) / sqrt(_var(v3_samples))
        @test isapprox(z_sampled, Vn_zscore(model, 3, layer = :top), atol = 0.25)
        @test Vn_zscore(model, 3, layer = :top) <= 0 # sign-definite under the BiCM (one-sided test)
    end

    @testset "projection significance filters (BiCM)" begin
        model = BiCM(_planted_bipartite())
        solve_model!(model)
        set_Ĝ!(model)
        total_strict = 0
        total_loose = 0
        for layer in (:bottom, :top)
            for distribution in (:Poisson, :PoissonBinomial)
                strict = project(model, layer = layer, distribution = distribution, α = 0.001)
                loose  = project(model, layer = layer, distribution = distribution, α = 0.5)
                @test strict isa Graphs.SimpleGraph
                @test loose isa Graphs.SimpleGraph
                @test Graphs.nv(strict) == Graphs.nv(loose)
                # a stricter significance level can only remove edges -> subset (FDR monotonicity)
                strict_edges = Set((Graphs.src(e), Graphs.dst(e)) for e in Graphs.edges(strict))
                loose_edges  = Set((Graphs.src(e), Graphs.dst(e)) for e in Graphs.edges(loose))
                @test issubset(strict_edges, loose_edges)
                total_strict += length(strict_edges)
                total_loose += length(loose_edges)
            end
        end
        # the planted co-occurrence makes filtering non-trivial: the looser level admits
        # strictly more significant links than the stricter one (the filter actually filters).
        @test total_loose > total_strict
    end

end
