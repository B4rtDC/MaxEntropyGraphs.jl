###############################################################################
# bicm_uecm_geometry.jl
#
# Geometry of the BiCM and UECM log-likelihoods — the companion to decm_gauge.jl,
# covering the other two models whose `:Newton` path needed work.
#
# The sibling scripts `bicm.jl` / `uecm.jl` prove the per-dyad MOMENTS. This one
# proves the shape of the OBJECTIVE, and the contrast between the two models is the
# point: they sit at opposite ends of the difficulty scale, for reasons that follow
# from their Hamiltonians.
#
# ---------------------------------------------------------------------------
# BiCM — one exact gauge mode, and no runaway is possible
# ---------------------------------------------------------------------------
# L_BiCM_reduced depends on θ only through the sums α_i + β_j (one layer index each),
# so exactly as for the DECM it is invariant under
#
#     (α, β) → (α + c, β - c),
#
# the linear part shifting by -c(Σ f⊥·k⊥ - Σ f⊤·k⊤) = 0 because both sums count the
# same edges, once from each layer. ONE flat direction (the DECM has two: it has an
# α and a β block).
#
# Unlike the DECM, nothing else is degenerate:
#   * a saturated degree is REJECTED AT CONSTRUCTION (BiCM.jl:178-179 throws a
#     DomainError when max(d⊥) ≥ |d⊤| or max(d⊤) ≥ |d⊥|), so `α → -∞` cannot arise;
#   * the model is unweighted, so there is no strength constraint to pin.
# Measured condition number with the gauge mode removed: 9-23. That is why `:Newton`
# never needed the gauge-fixing the DECM required — a rank-1 deficiency in an
# otherwise well-conditioned Hessian is handled by Optim's positive-definite
# modification without trouble.
#
# ---------------------------------------------------------------------------
# UECM — NO gauge, but two runaways
# ---------------------------------------------------------------------------
# The UECM is undirected: its pair terms depend on α_i + α_j and β_i + β_j WITHIN one
# block, so a shift α → α + c sends α_i + α_j → α_i + α_j + 2c. Nothing cancels, and
# there is no flat direction at all (measured null dim 0; the DECM-style shift changes
# L by ~289).
#
# What it does have is the same runaway taxonomy as the DECM, minus the direction
# split:
#
#   condition          meaning                       limit            handled
#   -----------------  ----------------------------  ---------------  ----------------
#   k = 0              isolated node                 α → +∞           yes — `ind_inf`
#   k = n-1            adjacent to everyone          α → -∞           no — runs away
#   s = k              every link carries weight 1    β → +∞           no — runs away
#
# Measured: a saturated class fits at α ≈ -40.5; and on a graph where EVERY weight is 1
# (so s = k for every node) the whole β block runs to 184-356, far enough that the
# Hessian itself overflows to Inf. That last case is not a bug but a statement about
# the model: with all weights equal the strength sequence carries no information beyond
# the degree sequence, so β is unidentifiable and the UECM degenerates to a UBCM.
#
# Run: julia --project=validation validation/symbolic/bicm_uecm_geometry.jl
###############################################################################

using MaxEntropyGraphs
using LinearAlgebra
using Random
include("common.jl")

const MEG = MaxEntropyGraphs
const G_  = MEG.Graphs
const SWG = MEG.SimpleWeightedGraphs

# ---------------------------------------------------------------------------
# deterministic test networks
# ---------------------------------------------------------------------------
"Bipartite graph with no isolated node in either layer."
function bip(seed::Int, Nb::Int, Nt::Int, p::Real)
    rng = Xoshiro(seed); g = G_.SimpleGraph(Nb + Nt)
    for b in 1:Nb, t in 1:Nt
        rand(rng) < p && G_.add_edge!(g, b, Nb + t)
    end
    for b in 1:Nb; G_.degree(g, b)      == 0 && G_.add_edge!(g, b, Nb + rand(rng, 1:Nt)); end
    for t in 1:Nt; G_.degree(g, Nb + t) == 0 && G_.add_edge!(g, rand(rng, 1:Nb), Nb + t); end
    g
end

"Weighted undirected graph; `minw` makes every weight 1 (⇒ s = k), `sat` saturates node 1."
function wund(seed::Int, n::Int, p::Real, wmax::Int; minw::Bool=false, sat::Bool=false)
    rng = Xoshiro(seed); A = zeros(Int, n, n)
    for i in 1:n, j in i+1:n
        if rand(rng) < p
            w = minw ? 1 : rand(rng, 1:wmax); A[i, j] = w; A[j, i] = w
        end
    end
    sat && for j in 2:n; A[1, j] = rand(rng, 1:wmax); A[j, 1] = A[1, j]; end
    for i in 1:n
        if all(iszero, @view A[i, :]); j = mod1(i + 1, n); A[i, j] = 1; A[j, i] = 1; end
    end
    SWG.SimpleWeightedGraph(A)
end

Lb(m) = θ -> MEG.L_BiCM_reduced(θ, m.d⊥ᵣ, m.d⊤ᵣ, m.f⊥, m.f⊤, m.d⊥ᵣ_nz, m.d⊤ᵣ_nz, m.status[:d⊥_unique])
Lu(m) = θ -> MEG.L_UECM_reduced(θ, m.dᵣ, m.sᵣ, m.f, length(m.dᵣ))

# ===========================================================================
# 1. symbolic — why the BiCM has a gauge and the UECM does not
# ===========================================================================
@variables a b c

# BiCM: the pair term sees α_i + β_j, one index from each layer, so the shift cancels.
verify("BiCM: pair argument invariant  (a+c)+(b-c) ≡ a+b",
       ((a + c) + (b - c)) - (a + b), [a => (0, 2), b => (0, 2), c => (-2, 2)])
# UECM: the pair term sees α_i + α_j, BOTH from the same block, so a shift adds 2c.
verify("UECM: same-block shift does NOT cancel — (a+c)+(b+c) - (a+b) ≡ 2c",
       ((a + c) + (b + c)) - (a + b) - 2c, [a => (0, 2), b => (0, 2), c => (-2, 2)])

# ===========================================================================
# 2. BiCM — one gauge mode, nothing else degenerate
# ===========================================================================
for (tag, g) in [("corporate", MEG.corporateclub()), ("bip20x40", bip(1, 20, 40, 0.15))]
    m = BiCM(g); n⊥ = m.status[:d⊥_unique]; nθ = length(m.θᵣ); L = Lb(m)
    gvec = vcat(ones(n⊥), -ones(nθ - n⊥))

    closecheck("BiCM/$tag: Σ f⊥·k⊥ == Σ f⊤·k⊤ (exact; this is what makes the gauge exact)",
               sum(m.f⊥ .* m.d⊥ᵣ) - sum(m.f⊤ .* m.d⊤ᵣ), 0; rtol = 0)

    θ = MEG.initial_guess(m); θ[isinf.(θ)] .= 0.0
    L0 = L(θ)
    for cval in (0.25, 1.0, -2.0)
        closecheck("BiCM/$tag: L invariant under gauge shift c=$cval", L(θ .+ cval .* gvec), L0; rtol = 1e-13)
    end

    H = MEG.ForwardDiff.hessian(L, θ)
    closecheck("BiCM/$tag: ‖H·g‖/‖H‖ ≈ 0", norm(H * gvec) / maximum(abs, H), 0; rtol = 0, atol = 1e-12)

    solve_model!(m; method = :BFGS, maxiters = 50_000)
    Hs = MEG.ForwardDiff.hessian(L, m.θᵣ); aev = sort(abs.(eigvals(Symmetric(Hs))))
    nullish = count(<(1e-10 * maximum(aev)), aev)
    boolcheck("BiCM/$tag: Hessian null dimension is exactly 1 (the gauge, nothing else) — got $nullish",
              nullish == 1)
    κ = maximum(aev) / aev[nullish + 1]
    boolcheck("BiCM/$tag: well conditioned off the gauge (κ = $(round(κ, sigdigits=3)) < 1e3)", κ < 1e3)
end

# a saturated degree cannot even be constructed: the model rejects it up front
let Nb = 6, Nt = 5
    g = G_.SimpleGraph(Nb + Nt)
    for b in 1:Nb, t in 1:Nt; G_.add_edge!(g, b, Nb + t); end   # every ⊥ node has k = Nt
    threw = try (BiCM(g); false) catch e; e isa DomainError end
    boolcheck("BiCM: a saturated degree (k⊥ = |⊤|) is rejected at construction, so α → -∞ cannot arise",
              threw)
end

# ===========================================================================
# 3. UECM — no gauge, but the two runaways
# ===========================================================================
let m = UECM(SWG.SimpleWeightedGraph(MEG.rhesus_macaques()))
    n = length(m.dᵣ); L = Lu(m)
    θ = MEG.initial_guess(m); θ[isinf.(θ)] .= 0.0; θ[n+1:end] .= abs.(θ[n+1:end]) .+ 0.5
    L0 = L(θ)
    boolcheck("UECM: the DECM-style shift (α+c, β-c) is NOT a symmetry (ΔL ≠ 0)",
              abs(L(θ .+ 0.25 .* vcat(ones(n), -ones(n))) - L0) > 1.0)
    boolcheck("UECM: a uniform α-shift is NOT a symmetry (ΔL ≠ 0)",
              abs(L(θ .+ 0.25 .* vcat(ones(n), zeros(n))) - L0) > 1.0)
    H = MEG.ForwardDiff.hessian(L, θ); aev = sort(abs.(eigvals(Symmetric(H))))
    boolcheck("UECM: Hessian has NO null directions (no gauge freedom to fix)",
              count(<(1e-10 * maximum(aev)), aev) == 0)
end

# runaway 1: a saturated degree drives α → -∞
let g = wund(9, 12, 0.30, 5; sat = true)
    m = UECM(g); n = length(m.dᵣ); nv = G_.nv(g)
    solve_model!(m; method = :BFGS, maxiters = 50_000)
    sat = findall(==(nv - 1), m.dᵣ)
    boolcheck("UECM: the saturated-degree network really has a k = n-1 class", !isempty(sat))
    boolcheck("UECM: its α runs away strongly negative (α ≈ $(round(m.θᵣ[sat[1]], sigdigits=3)))",
              m.θᵣ[sat[1]] < -20)
    H = MEG.ForwardDiff.hessian(Lu(m), m.θᵣ); aev = sort(abs.(eigvals(Symmetric(H))))
    boolcheck("UECM: that runaway shows up as a near-null Hessian direction",
              count(<(1e-10 * maximum(aev)), aev) >= length(sat))
end

# runaway 2: all weights equal 1 ⇒ s = k for every node ⇒ the whole β block is unidentifiable
let g = wund(3, 16, 0.35, 1; minw = true)
    m = UECM(g); n = length(m.dᵣ)
    boolcheck("UECM: with every weight 1, s = k for every class (β carries no information)",
              all(i -> m.sᵣ[i] == m.dᵣ[i], 1:n))
    solve_model!(m; method = :BFGS, maxiters = 50_000)
    β = m.θᵣ[n+1:end]
    boolcheck("UECM: the entire β block runs away (min β = $(round(minimum(β), sigdigits=3)) ≫ 0)",
              minimum(β) > 20)
end

# ===========================================================================
# 4. dead channels must come from the DATA, not from the initial guess
# ===========================================================================
# `ind_inf` used to be `findall(isinf, θ₀)`. Only the `:degrees`/`:strengths` family puts
# an Inf there, so `:uniform`/`:random` left every dead channel finite and the fit came
# back silently wrong (degree residual 9.85, reported as Success).
#
# NOTE a BiCM built from a GRAPH can no longer have a dead channel at all: a zero-degree
# vertex is an isolated vertex, whose layer membership is undetermined, and the constructor
# refuses it (checked below). Dead channels reach this model only through explicit degree
# sequences, where the caller has stated the partition — so that is the probe used here.
let d⊥ = [0, 2, 2, 1, 3, 0], d⊤ = [2, 3, 1, 2, 0]
    m0 = BiCM(nothing; d⊥ = d⊥, d⊤ = d⊤)
    nzero = count(iszero, m0.d⊥ᵣ) + count(iszero, m0.d⊤ᵣ)
    boolcheck("BiCM: the probe really has dead channels (zero-degree classes: $nzero)", nzero > 0)
    for init in (:degrees, :uniform, :random)      # `:chung_lu` needs the graph
        m = BiCM(nothing; d⊥ = d⊥, d⊤ = d⊤)
        solve_model!(m; method = :BFGS, initial = init)
        MEG.set_Ĝ!(m)
        resid = maximum(abs, vcat(vec(sum(m.Ĝ, dims = 2)) .- d⊥, vec(sum(m.Ĝ, dims = 1)) .- d⊤))
        closecheck("BiCM: degrees reproduced from initial=$init (dead channels honoured)",
                   resid, 0; rtol = 0, atol = 1e-6)
    end
end

# ===========================================================================
# 5. an isolated vertex has no layer, so the graph constructor refuses it
# ===========================================================================
# `Graphs.bipartite_map` colours each component from 1, so every isolated vertex silently lands
# in ⊥: a graph built as 18x40 came back as a 44x14 model. The live-vertex fit stayed correct,
# but |⊥| and |⊤| — and hence the ensemble `rand(m)` draws from — did not.
let g = G_.SimpleGraph(8)
    for b in 1:3; G_.add_edge!(g, b, 3 + b); end                 # vertices 7, 8 isolated
    threw = try (BiCM(g); false) catch e; e isa ArgumentError end
    boolcheck("BiCM: a graph with isolated vertices is refused (their layer is undetermined)", threw)
    g2 = G_.SimpleGraph(6)
    for b in 1:3; G_.add_edge!(g2, b, 3 + b); end
    ok = try (BiCM(g2); true) catch; false end
    boolcheck("BiCM: the same graph without the isolated vertices is accepted", ok)
end

# ===========================================================================
# 6. the BiCM FIXED-POINT map inherits the gauge — and that is what broke Anderson
# ===========================================================================
# The same gauge that leaves `L` flat makes the fixed-point map G EQUIVARIANT:
#
#     G(θ + c·g) = G(θ) + c·g
#
# (the shift leaves every product xᵢ·yⱼ alone and multiplies the inner sum by e^c, which the
# outer -log turns back into +c). Three consequences, each checked below: g is an eigenvector
# of the Jacobian with eigenvalue EXACTLY 1; the residual G(θ) - θ is completely blind to the
# gauge component; and the residual Jacobian is therefore singular along g BY CONSTRUCTION.
#
# Anderson acceleration solves a least-squares problem built from residual DIFFERENCES, every
# one of which lies in gᗮ. The more history it keeps, the sooner that system is rank-deficient
# and its internal solve emits NaN. That is the whole story of the divergence — see the failure
# counts against Anderson memory below, and ../bicm_uecm_solver_geometry.md §3b.
let m = BiCM(bip(11, 10, 14, 0.3))
    n⊥ = m.status[:d⊥_unique]; nθ = length(m.θᵣ)
    function Gmap(θ::AbstractVector{T}) where {T}
        xb = zeros(T, length(m.d⊥ᵣ)); yb = zeros(T, length(m.d⊤ᵣ)); Gb = zeros(T, nθ)
        MEG.BiCM_reduced_iter!(collect(θ), m.d⊥ᵣ, m.d⊤ᵣ, m.f⊥, m.f⊤, m.d⊥ᵣ_nz, m.d⊤ᵣ_nz, xb, yb, Gb, n⊥)
        Gb
    end
    g = vcat(ones(n⊥), -ones(nθ - n⊥))
    θ = MEG.initial_guess(m); θ[isinf.(θ)] .= 0.0

    for cval in (0.3, 1.0, -2.0)
        closecheck("BiCM fixed point: G(θ+cg) = G(θ)+cg  (c=$cval)",
                   maximum(abs, Gmap(θ .+ cval .* g) .- Gmap(θ) .- cval .* g), 0; rtol = 0, atol = 1e-12)
    end
    J = MEG.ForwardDiff.jacobian(Gmap, θ)
    closecheck("BiCM fixed point: g is an eigenvector of the Jacobian with eigenvalue exactly 1",
               maximum(abs, J * g .- g), 0; rtol = 0, atol = 1e-12)
    resid(θ) = Gmap(θ) .- θ
    closecheck("BiCM fixed point: the residual is blind to the gauge component",
               maximum(abs, resid(θ .+ 0.7 .* g) .- resid(θ)), 0; rtol = 0, atol = 1e-12)
    Jf = MEG.ForwardDiff.jacobian(resid, θ)
    boolcheck("BiCM fixed point: the residual Jacobian is singular along g (rank $(rank(Jf)) of $nθ)",
              rank(Jf) == nθ - 1)
end

# The practical consequence, and the reason the fix is a shorter MEMORY rather than damping:
# failures rise monotonically with the Anderson memory, and vanish at m = 0 (plain Picard, which
# has no least-squares at all). Damping (`beta = 0.5`) makes it WORSE, so the UBCM's remedy for
# its own overflow problem is the wrong tool here.
let
    function count_nonfinite(mem; ngraph = 40)
        rng = Xoshiro(20260917); bad = 0; tot = 0
        for _ in 1:ngraph
            Nb = rand(rng, 4:24); Nt = rand(rng, 4:24); p = rand(rng) * 0.6 + 0.05
            g = bip(rand(rng, 1:10^6), Nb, Nt, p)
            mm = try BiCM(g) catch; continue end
            tot += 1
            n⊥ = mm.status[:d⊥_unique]
            xb = zeros(length(mm.d⊥ᵣ)); yb = zeros(length(mm.d⊤ᵣ)); Gb = zeros(length(mm.θᵣ))
            θ₀ = MEG.initial_guess(mm); θ₀[isinf.(θ₀)] .= 0.0
            FP! = (θ::Vector) -> MEG.BiCM_reduced_iter!(θ, mm.d⊥ᵣ, mm.d⊤ᵣ, mm.f⊥, mm.f⊤, mm.d⊥ᵣ_nz, mm.d⊤ᵣ_nz, xb, yb, Gb, n⊥)
            kw = mem === nothing ? NamedTuple() : (m = mem,)
            try
                MEG.NLsolve.fixedpoint(FP!, copy(θ₀); method = :anderson, ftol = 1e-8, iterations = 1000, kw...)
            catch e
                e isa MEG.NLsolve.IsFiniteException && (bad += 1)
            end
        end
        bad, tot
    end
    b0, t0 = count_nonfinite(0)
    bd, _  = count_nonfinite(nothing)
    b20, _ = count_nonfinite(20)
    boolcheck("BiCM fixed point: Picard (m=0) never hits the singular least-squares — $b0 of $t0 failed",
              b0 == 0)
    boolcheck("BiCM fixed point: failures grow with Anderson memory (m=0: $b0, default: $bd, m=20: $b20)",
              b0 <= bd <= b20 && b20 > 0)
end

# and the shipped solver, which walks that ladder, converges where the plain accelerator did not
let nok = 0, ntot = 0
    rng = Xoshiro(987)
    for _ in 1:25
        Nb = rand(rng, 4:24); Nt = rand(rng, 4:24); p = rand(rng) * 0.6 + 0.05
        g = bip(rand(rng, 1:10^6), Nb, Nt, p)
        mm = try BiCM(g) catch; continue end
        ntot += 1
        try
            solve_model!(mm; method = :fixedpoint)
            MEG.set_Ĝ!(mm)
            r = max(maximum(abs, vec(sum(mm.Ĝ, dims = 2)) .- mm.d⊥),
                    maximum(abs, vec(sum(mm.Ĝ, dims = 1)) .- mm.d⊤))
            r < 1e-6 && (nok += 1)
        catch; end
    end
    boolcheck("BiCM `:fixedpoint` (shipped, with the memory ladder) solves all $ntot probes — got $nok", nok == ntot)
end

# ===========================================================================
# 7. the saturation ceiling counts LIVE vertices, not layer size
# ===========================================================================
# A vertex adjacent to every *available* counterpart forces p = 1 and its fitness diverges. Dead
# vertices cannot be connected to, so they do not raise the ceiling — comparing against the layer
# size let such inputs through, and they then burned the iteration cap instead of being refused.
let
    # 4 live ⊤ vertices (plus one dead); a ⊥ vertex of degree 4 is saturated among the live ones
    d⊥ = [4, 2, 2, 1, 0]; d⊤ = [3, 3, 2, 1, 0]
    threw = try (BiCM(nothing; d⊥ = d⊥, d⊤ = d⊤); false) catch e; e isa DomainError end
    boolcheck("BiCM: a vertex adjacent to every LIVE counterpart is refused (ceiling counts live, not layer size)",
              threw)
    # the same shape with one more live ⊤ vertex is fine
    d⊥b = [4, 2, 2, 1, 0]; d⊤b = [2, 2, 2, 2, 1]
    ok = try (BiCM(nothing; d⊥ = d⊥b, d⊤ = d⊤b); true) catch; false end
    boolcheck("BiCM: one more live counterpart and the same degree is accepted", ok)
end

report("BiCM & UECM geometry")
