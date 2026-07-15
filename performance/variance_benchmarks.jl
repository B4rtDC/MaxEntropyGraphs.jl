##################################################################################################
# Delta-method variance (σₓ) benchmarks — materialized vs matrix-free strategies
#
# The delta-method variance σₓ(m, X) requires the expected matrix (Ĝ/Ŵ), the per-edge standard
# deviations (σ/σʷ) and the gradient ∂X/∂g_ij at ⟨G⟩. This script measures, per model and size:
#
#   setup    : set_Ĝ! + set_σ! (dense N² materialization) — time and bytes, amortizable
#   stored   : σₓ(m, X; gradient_method) on the stored matrices (the shipped path), per AD backend
#   fresh    : one-shot cost — rebuild Ĝ(m)/σˣ(m) on every call + AD gradient (no struct storage)
#   analytic : matrix-free streaming with the hand-derived ∂X/∂g_ij and the element accessor
#              A(m, i, j) — O(1)/O(N) live memory, no AD
#
# Every strategy is cross-checked against the stored-path value at every size (isapprox).
#
# Usage (from the `performance/` project):
#   julia --project=. variance_benchmarks.jl                 # full sweep
#   BENCH_QUICK=1 julia --project=. variance_benchmarks.jl   # smaller sizes, faster
#   BENCH_MAX_N=3162 ...                                     # cap the sweep
#
# Per-strategy size caps (memory/time cliffs, see the recommendations at the end):
#   ForwardDiff : N ≤ 316   (N² dual inputs; chunked passes over an O(N²)+ metric)
#   ReverseDiff : N ≤ 3162  (tape holds O(N²) tracked nodes; ≥ 10 GB at N = 10⁴)
#   Zygote      : N ≤ 10⁴ for linear metrics (pullback of `sum` is O(1)); N ≤ 3162 for
#                 matmul-based metrics (BLAS pullback temporaries)
#   analytic O(N³) metrics (triangles): N ≤ 3162
##################################################################################################

using MaxEntropyGraphs
const MEG = MaxEntropyGraphs
using MaxEntropyGraphs: Graphs, SimpleWeightedGraphs
using BenchmarkTools, LinearAlgebra, Random, Printf, JSON, Dates

const BLAS_THREADS = parse(Int, get(ENV, "BENCH_BLAS_THREADS", "1"))
BLAS_THREADS > 0 && BLAS.set_num_threads(BLAS_THREADS)
# Test the value, not mere presence: benchmarks.sh always exports BENCH_QUICK (defaulting it to 0),
# so `haskey` would silently force QUICK mode even when it is explicitly disabled.
const QUICK = get(ENV, "BENCH_QUICK", "0") == "1"
const MAXN  = parse(Int, get(ENV, "BENCH_MAX_N", QUICK ? "316" : "10000"))
const SIZES = filter(<=(MAXN), QUICK ? [50, 100, 316] : [100, 316, 1000, 3162, 10000])

const CAP_FD    = 100    # ForwardDiff (the ~N⁴ chunked-pass cliff is already fully visible at N=100)
const CAP_RD    = 3162   # ReverseDiff
const CAP_ZY_NL = 3162   # Zygote on matmul-based metrics
const CAP_CUBIC = 3162   # O(N³) analytic streaming (triangles)

const RESULTS = Dict{String,Any}()
measure(f) = ((@belapsed $f() samples=5 seconds=2 evals=1), @allocated(f()))

function report_row!(rows, N, strategy, backend, t, b, val)
    @printf("%8d | %-9s %-12s | %12.3e %12s | %.6g\n", N, strategy, backend, t, Base.format_bytes(b), val)
    flush(stdout)
    push!(rows, Dict("N"=>N, "strategy"=>strategy, "backend"=>backend, "time_s"=>t, "bytes"=>b, "value"=>val))
end

header(title) = (println("\n### $title"); @printf("%8s | %-9s %-12s | %12s %12s | %s\n", "N", "strategy", "backend", "time (s)", "alloc", "value"); flush(stdout))

# ---- synthetic model builders ---------------------------------------------------------------------
ubcm_model(N)  = (m = UBCM(Graphs.barabasi_albert(N, 6, seed=161)); solve_model!(m); m)
function dbcm_model(N)
    G = Graphs.SimpleDiGraph(Graphs.barabasi_albert(N, 6, seed=162))   # each edge → two arcs
    m = DBCM(G); solve_model!(m); m
end
function bicm_model(N)   # N bottom nodes, N÷2 top nodes, ~6 links per bottom node
    rng = Xoshiro(163)
    nb, nt = N, max(N ÷ 2, 8)
    g = Graphs.SimpleGraph(nb + nt)
    for b in 1:nb, _ in 1:6
        Graphs.add_edge!(g, b, nb + rand(rng, 1:nt))
    end
    m = BiCM(g); solve_model!(m); m
end
function dcrem_model(N)  # directed weighted, continuous exponential weights
    rng = Xoshiro(164)
    Gd = Graphs.SimpleDiGraph(Graphs.barabasi_albert(N, 6, seed=165))
    srcs = Int[]; dsts = Int[]; ws = Float64[]
    for e in Graphs.edges(Gd)
        push!(srcs, Graphs.src(e)); push!(dsts, Graphs.dst(e)); push!(ws, -log(rand(rng)))
    end
    G = SimpleWeightedGraphs.SimpleWeightedDiGraph(srcs, dsts, ws)
    m = DCReM(G); solve_model!(m); m
end

# ---- UBCM -----------------------------------------------------------------------------------------
# X₁ = sum(A) (linear, dyad-symmetric); X₂ = triangles(A) (cubic).
# Analytic forms (with the undirected within-dyad covariance): the dyadic derivative D_ij = ∂_ij + ∂_ji
# gives σ²[X] = Σ_{i<j} σ²_ij D_ij².  For sum: D = 2 everywhere.  For triangles: ∂_ij = Σ_k p_jk p_ki.
function ubcm_analytic_sum(m::UBCM)
    n = m.status[:d]::Int   # type-assert: status is Dict{Symbol,Any}
    acc = 0.0
    for i in 1:n, j in i+1:n
        p = MEG.A(m, i, j)
        acc += p * (1 - p) * 4
    end
    return sqrt(acc)
end
function ubcm_analytic_triangles(m::UBCM)   # O(N³) time, O(N) live memory
    # X = (1/6)Σ_{i≠j≠k} a_ij a_jk a_ki ⇒ ∂X/∂a_pq = (1/2)Σ_k a_qk a_kp per matrix slot;
    # the dyadic derivative D_ij = ∂_ij + ∂_ji = Σ_k p_jk p_ki, and σ²[X] = Σ_{i<j} σ²_ij D_ij².
    n = m.status[:d]::Int   # type-assert: status is Dict{Symbol,Any}
    x = m.xᵣ[m.dᵣ_ind]
    acc = 0.0
    @inbounds for i in 1:n
        xi = x[i]
        for j in i+1:n
            xj = x[j]
            D = 0.0
            @simd for k in 1:n
                vjk = xj * x[k]; vki = x[k] * xi
                D += ifelse(k == i || k == j, 0.0, (vjk / (1 + vjk)) * (vki / (1 + vki)))
            end
            vij = xi * xj
            pij = vij / (1 + vij)
            acc += pij * (1 - pij) * D^2
        end
    end
    return sqrt(acc)
end

# ---- DBCM -----------------------------------------------------------------------------------------
# X₁ = sum(A); entries independent: σ²[X] = Σ_{i≠j} σ²_ij (∂_ij)².
function dbcm_analytic_sum(m::DBCM)
    n = Int(m.status[:d])
    acc = 0.0
    for i in 1:n, j in 1:n
        i == j && continue
        pij = MEG.A(m, i, j)
        acc += pij * (1 - pij)
    end
    return sqrt(acc)
end

# ---- BiCM -----------------------------------------------------------------------------------------
function bicm_analytic_sum(m::BiCM)
    acc = 0.0
    n⊥ = m.status[:N⊥]::Int; n⊤ = m.status[:N⊤]::Int
    for i in 1:n⊥, j in 1:n⊤
        p = MEG.A(m, i, j)
        acc += p * (1 - p)
    end
    return sqrt(acc)
end

# ---- DCReM (weighted layer) -----------------------------------------------------------------------
# X = sum(W); independent entries: σ²[X] = Σ_{i≠j} Var(w_ij) with Var = f(2-f)/(θᵒᵢ+θⁱⱼ)².
function dcrem_analytic_sumW(m::DCReM)
    n = m.status[:N]::Int
    x = m.xᵣ[m.dᵣ_ind]; y = m.yᵣ[m.dᵣ_ind]
    θᵒ = @view m.θ[1:n]; θⁱ = @view m.θ[n+1:end]
    acc = 0.0
    for i in 1:n, j in 1:n
        i == j && continue
        xiyj = x[i]*y[j]
        f = xiyj/(1+xiyj)
        acc += f*(2-f)/(θᵒ[i]+θⁱ[j])^2
    end
    return sqrt(acc)
end

# ---- generic sweep driver --------------------------------------------------------------------------
# `undirected_cov`: whether the model's σₓ includes the identity-covariance cross-term (g_ij ≡ g_ji);
# must match the model's own convention so the `fresh` re-implementation cross-checks exactly.
function sweep!(name, build, X, Xname; layer=nothing, analytic=nothing, cap_zy=MAXN, cap_analytic=MAXN,
                cap_rd=CAP_RD, weighted_setup=false, undirected_cov=false)
    header("$name — X = $Xname")
    rows = []
    for N in SIZES
        t_solve = @elapsed m = build(N)
        # setup (materialization) cost
        setfun = weighted_setup ? (() -> (set_Ŵ!(m); set_σʷ!(m))) : (() -> (set_Ĝ!(m); set_σ!(m)))
        b_setup = @allocated setfun()
        t_setup = @elapsed setfun()
        push!(rows, Dict("N"=>N, "strategy"=>"setup", "backend"=>"-", "time_s"=>t_setup, "bytes"=>b_setup, "solve_s"=>t_solve))
        @printf("%8d | %-9s %-12s | %12.3e %12s | (solve: %.2fs)\n", N, "setup", "-", t_setup, Base.format_bytes(b_setup), t_solve)

        kw = isnothing(layer) ? (;) : (; layer=layer)

        # collect the strategies that fit at this size
        candidates = Tuple{String,String,Function}[]
        for (backend, cap) in ((:ForwardDiff, CAP_FD), (:ReverseDiff, cap_rd), (:Zygote, cap_zy))
            N ≤ cap && push!(candidates, ("stored", String(backend), () -> σₓ(m, X; gradient_method=backend, kw...)))
        end
        if N ≤ cap_rd
            mf = m
            freshf = if weighted_setup
                () -> (W = Ŵ(mf); S = σʷ(mf); ∇ = MEG.ReverseDiff.gradient(X, W);
                       undirected_cov ? sqrt(sum((S .* ∇).^2) + sum(S.^2 .* ∇ .* transpose(∇))) : sqrt(sum((S .* ∇).^2)))
            else
                () -> (G = Ĝ(mf); S = σˣ(mf); ∇ = MEG.ReverseDiff.gradient(X, G);
                       undirected_cov ? sqrt(sum((S .* ∇).^2) + sum(S.^2 .* ∇ .* transpose(∇))) : sqrt(sum((S .* ∇).^2)))
            end
            push!(candidates, ("fresh", "ReverseDiff", freshf))
        end
        if !isnothing(analytic) && N ≤ cap_analytic
            push!(candidates, ("analytic", "closed-form", () -> analytic(m)))
        end
        isempty(candidates) && continue

        # correctness anchor = first runnable strategy; every other strategy must agree
        ref = candidates[1][3]()
        for (strategy, backend, f) in candidates
            v = f()
            isapprox(v, ref, rtol=1e-6) || @warn "$name/$strategy/$backend value mismatch at N=$N: $v vs $ref"
            t, b = measure(f)
            report_row!(rows, N, strategy, backend, t, b, v)
        end
    end
    RESULTS[name * " | " * Xname] = rows
end

# ---- run ------------------------------------------------------------------------------------------
println("Delta-method variance benchmarks — sizes: $SIZES  (BLAS threads: $BLAS_THREADS)")

sweep!("UBCM", ubcm_model, sum, "sum(A) [edge count ×2]"; analytic=ubcm_analytic_sum, undirected_cov=true)
# check_directed=false: ForwardDiff's chunked dual perturbations break exact symmetry of the input.
# cap_rd=316: ReverseDiff records the triangles kernel as an O(N³) SCALAR tape (dispatch-heavy
# TrackedReal ops) — at N=1000 that is ~10⁹ tape nodes and runs for hours (measured by sampling).
tri_nocheck(A) = triangles(A; check_directed=false)
sweep!("UBCM", ubcm_model, tri_nocheck, "triangles(A)"; analytic=ubcm_analytic_triangles,
       cap_zy=1000, cap_analytic=CAP_CUBIC, cap_rd=316, undirected_cov=true)
sweep!("DBCM", dbcm_model, sum, "sum(A) [arc count]"; analytic=dbcm_analytic_sum)
sweep!("BiCM", bicm_model, sum, "sum(B) [edge count]"; analytic=bicm_analytic_sum)
sweep!("DCReM", dcrem_model, sum, "sum(W) [total weight]"; layer=:weighted,
       analytic=dcrem_analytic_sumW, weighted_setup=true)

# ---- persist --------------------------------------------------------------------------------------
outdir = joinpath(@__DIR__, "benchmarks", "Julia-$(VERSION)")
mkpath(outdir)
open(joinpath(outdir, "variance_benchmarks.json"), "w") do f
    write(f, JSON.json(Dict("timestamp" => Dates.format(now(), "YYYY_mm_dd_HH_MM"),
                            "sizes" => SIZES, "blas_threads" => BLAS_THREADS, "results" => RESULTS)))
end
println("\nResults written to $(joinpath(outdir, "variance_benchmarks.json"))")
