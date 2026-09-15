###############################################################################
# decm_gauge.jl
#
# Geometry of the DECM log-likelihood: the exact gauge freedom, the degeneracy
# taxonomy of its Hessian, and what both imply for the solvers.
#
# The sibling script `decm.jl` proves the per-channel MOMENTS (p, ⟨w⟩, Var[w], …).
# This one proves the shape of the OBJECTIVE that the solver walks on.
#
# ---------------------------------------------------------------------------
# 1. Exact gauge freedom (two flat directions, always present)
# ---------------------------------------------------------------------------
# L_DECM_reduced (src/Models/DECM.jl) is
#
#   L = -Σᵢ Fᵢ(k_out,ᵢ α_out,ᵢ + k_in,ᵢ α_in,ᵢ + s_out,ᵢ β_out,ᵢ + s_in,ᵢ β_in,ᵢ)
#       -Σᵢ Fᵢ Σⱼ wᵢⱼ log(1 + x·y/(1-y)),   x = e^(-α_out,ᵢ-α_in,ⱼ), y = e^(-β_out,ᵢ-β_in,ⱼ)
#
# and is EXACTLY invariant under the two shifts
#
#   (α_out, α_in) → (α_out + c, α_in - c)      (β_out, β_in) → (β_out + c, β_in - c)
#
# because (a) every pair term depends on α_out,ᵢ + α_in,ⱼ and β_out,ᵢ + β_in,ⱼ only, which the
# shift leaves alone, and (b) the linear part changes by -c(Σ Fᵢk_out,ᵢ - Σ Fᵢk_in,ᵢ) resp.
# -c(Σ Fᵢs_out,ᵢ - Σ Fᵢs_in,ᵢ), both of which vanish identically: the first counts the edges
# twice over, the second the total weight twice over.
#
# Consequence: the Hessian is singular, with the two gauge vectors
#   g_α = (1…1, -1…-1, 0…0, 0…0)      g_β = (0…0, 0…0, 1…1, -1…-1)
# in its kernel. `Newton` factorises that Hessian, so the degeneracy hands it a meaningless
# step: from a `:uniform` start it returns -L ≈ 1e4 against a true optimum of 384.49 on the
# rhesus macaques network. Perturbing the start does NOT help (the degeneracy is structural,
# not an artifact of a symmetric starting point) — removing the flat directions does, which is
# what `_decm_gauge` exists for. `BFGS`/`LBFGS` keep a positive definite APPROXIMATION and never
# invert the true Hessian, and since ∇L ⊥ g exactly they never travel along the gauge either,
# so they are unaffected and the gauge term is applied to `Newton` only.
#
# ---------------------------------------------------------------------------
# 2. Degeneracy taxonomy (extra near-null directions, data dependent)
# ---------------------------------------------------------------------------
# Beyond the two gauge modes, a DECM Hessian acquires one near-null direction per constraint
# that is pinned at the edge of its feasible range, because the conjugate parameter runs away:
#
#   mechanism            condition        limit          handled in solve_model!
#   -------------------  ---------------  -------------  ------------------------------------
#   dead channel         k = 0            α → +∞         yes — `ind_inf`, pinned to Inf
#   saturated degree     k = N-1          α → -∞         no  — runs away (fitted α ≈ -30…-55)
#   minimum strength     s = k            β → +∞         no  — runs away (fitted β ≈ +31)
#
# ("minimum strength" means every incident link carries weight exactly 1, the smallest a present
# link may carry, so the geometric excess-weight parameter is driven to zero.)
#
# These runaways, not the gauge, are what make the DECM ill-conditioned: condition numbers of
# 1e15–1e17 on the affected networks versus ~1e3 on clean ones. They are a property of the DATA
# (the constraint is at the boundary of what any ensemble can realise, so its fitness is not
# identifiable), NOT a defect — every such solve still reproduces the constraints to ~1e-9. They
# are why first-order methods need many iterations on those networks.
#
# NOTE the rhesus macaques network shipped with the package has one `s = k` node, so it exhibits
# this too.
#
# The full derivation — the invariance proof, the stationarity conditions behind each runaway, the
# feasibility argument against a box constraint, and the conditioning measurements — is written up in
# ../decm_solver_geometry.md. This script is the executable half of that note.
#
# Run: julia --project=validation validation/symbolic/decm_gauge.jl
###############################################################################

using MaxEntropyGraphs
using LinearAlgebra
using Random
include("common.jl")

const MEG = MaxEntropyGraphs
const G_  = MEG.Graphs
const SWG = MEG.SimpleWeightedGraphs
const Opt = MEG.Optimization

# ---------------------------------------------------------------------------
# deterministic test networks
# ---------------------------------------------------------------------------
"Weighted digraph with no empty row/column; `hub` makes node 1 adjacent to everyone (⇒ k = N-1)."
function wdigraph(seed::Int, nv::Int, p::Real, wmax::Int; hub::Bool=false)
    rng = Xoshiro(seed)
    A = zeros(Int, nv, nv)
    for i in 1:nv, j in 1:nv
        i == j && continue
        rand(rng) < p && (A[i, j] = rand(rng, 1:wmax))
    end
    if hub
        for j in 2:nv
            A[1, j] = rand(rng, wmax:4wmax)
            A[j, 1] = rand(rng, wmax:4wmax)
        end
    end
    for i in 1:nv
        all(iszero, @view A[i, :]) && (A[i, mod1(i + 1, nv)] = 1)
        all(iszero, @view A[:, i]) && (A[mod1(i + 1, nv), i] = 1)
    end
    SWG.SimpleWeightedDiGraph(A)
end

Lfun(m) = θ -> MEG.L_DECM_reduced(θ, m.dᵣ_out, m.dᵣ_in, m.sᵣ_out, m.sᵣ_in, m.f, length(m.dᵣ_out))

"Gauge basis vectors of a 4n-parameter DECM."
gauge_vectors(n) = (vcat(ones(n), -ones(n), zeros(n), zeros(n)),
                    vcat(zeros(n), zeros(n), ones(n), -ones(n)))

"Interior point: finite, comfortably inside β_out,i + β_in,j > 0."
function interior(m)
    θ = MEG.initial_guess(m)
    θ[isinf.(θ)] .= 0.0
    n = length(m.dᵣ_out)
    θ[2n+1:end] .= abs.(θ[2n+1:end]) .+ 0.5
    θ
end

"Solve and return the gauge-invariant predictions."
function solved(g; method=:BFGS, initial=:strengths, maxiters=50_000)
    m = DECM(g)
    solve_model!(m; method = method, initial = initial, maxiters = maxiters)
    MEG.set_Ĝ!(m); MEG.set_Ŵ!(m)
    m
end

# ===========================================================================
# 1. symbolic — the two facts the invariance rests on
# ===========================================================================
@variables αo αi βo βi c

# a gauge shift leaves every pair EXPONENT untouched; the log arguments are functions of these
# exponents alone, so every pair term is invariant. (Stated on the exponents, which are polynomial,
# so the identity is settled exactly rather than through `exp`.)
verify("gauge: α pair exponent invariant  -(αo+c)-(αi-c) ≡ -αo-αi",
       (-(αo + c) - (αi - c)) - (-αo - αi), [αo => (0, 2), αi => (0, 2), c => (-2, 2)])
verify("gauge: β pair exponent invariant  -(βo+c)-(βi-c) ≡ -βo-βi",
       (-(βo + c) - (βi - c)) - (-βo - βi), [βo => (1, 3), βi => (1, 3), c => (-2, 2)])

# the linear part shifts by -c·(Σ F k_out - Σ F k_in) resp. -c·(Σ F s_out - Σ F s_in)
@variables So Si
verify("gauge: linear part shifts by -c·(Σ F s_out - Σ F s_in), which the balance kills",
       (-c * (So - Si)) - (-c * So + c * Si), [So => (1, 5), Si => (1, 5), c => (-2, 2)])

# ===========================================================================
# 2. numeric closure — on the shipped network and on synthetic ones
# ===========================================================================
nets = [("rhesus",  MEG.rhesus_macaques()),
        ("clean12", wdigraph(42, 12, 0.40, 5)),            # no degenerate constraint expected
        ("hub12",   wdigraph(7, 12, 0.35, 6; hub = true))] # saturated degree expected

for (tag, g) in nets
    m = DECM(g); n = length(m.dᵣ_out); L = Lfun(m)
    gα, gβ = gauge_vectors(n)

    # --- the two balance identities, in exact integer arithmetic -----------
    closecheck("$tag: Σ F·k_out == Σ F·k_in (exact)",
               sum(m.f .* m.dᵣ_out) - sum(m.f .* m.dᵣ_in), 0; rtol = 0)
    closecheck("$tag: Σ F·s_out == Σ F·s_in (exact)",
               sum(m.f .* m.sᵣ_out) - sum(m.f .* m.sᵣ_in), 0; rtol = 0)

    # --- L is invariant along both gauge directions ------------------------
    θ = interior(m); L0 = L(θ)
    for cval in (0.25, 1.0, -0.75, 3.0)
        closecheck("$tag: L invariant under α-gauge shift c=$cval",
                   L(θ .+ cval .* gα), L0; rtol = 1e-14)
        closecheck("$tag: L invariant under β-gauge shift c=$cval",
                   L(θ .+ cval .* gβ), L0; rtol = 1e-14)
    end

    # --- the Hessian annihilates both gauge vectors ------------------------
    H = MEG.ForwardDiff.hessian(L, θ)
    scale = maximum(abs, H)
    closecheck("$tag: ‖H·g_α‖/‖H‖ ≈ 0", norm(H * gα) / scale, 0; rtol = 0, atol = 1e-12)
    closecheck("$tag: ‖H·g_β‖/‖H‖ ≈ 0", norm(H * gβ) / scale, 0; rtol = 0, atol = 1e-12)
end

# --- the taxonomy: a clean network has EXACTLY the two gauge modes, a degenerate one has more
for (tag, g, degenerate) in [("clean12", wdigraph(42, 12, 0.40, 5), false),
                             ("hub12",   wdigraph(7, 12, 0.35, 6; hub = true), true)]
    nv = G_.nv(g); m = DECM(g); n = length(m.dᵣ_out)
    nsat = count(==(nv - 1), m.dᵣ_out) + count(==(nv - 1), m.dᵣ_in)
    nmin = count(i -> m.sᵣ_out[i] == m.dᵣ_out[i], 1:n) + count(i -> m.sᵣ_in[i] == m.dᵣ_in[i], 1:n)
    ms = solved(g)
    H  = MEG.ForwardDiff.hessian(Lfun(ms), ms.θᵣ)
    aev = sort(abs.(eigvals(Symmetric(H))))
    nullish = count(<(1e-10 * max(1.0, maximum(aev))), aev)
    boolcheck("$tag: degenerate constraints present == $degenerate  (saturated=$nsat, min-strength=$nmin)",
              (nsat + nmin > 0) == degenerate)
    boolcheck("$tag: Hessian null dimension ($nullish) == 2 gauge modes + degenerate ($(nsat + nmin))",
              degenerate ? nullish > 2 : nullish == 2)
end

# ===========================================================================
# 3. the gauge term changes no gauge-invariant quantity
# ===========================================================================
# `Newton` carries `_decm_gauge`, `BFGS` does not; both must land on the same physics.
for (tag, g) in [("rhesus", MEG.rhesus_macaques()), ("hub12", wdigraph(7, 12, 0.35, 6; hub = true))]
    mb = solved(g; method = :BFGS)
    mn = solved(g; method = :Newton)
    closecheck("$tag: Ĝ identical with (Newton) and without (BFGS) the gauge term",
               maximum(abs, mb.Ĝ .- mn.Ĝ), 0; rtol = 0, atol = 1e-6)
    closecheck("$tag: Ŵ identical with (Newton) and without (BFGS) the gauge term",
               maximum(abs, mb.Ŵ .- mn.Ŵ), 0; rtol = 0, atol = 1e-4)
end

# ===========================================================================
# 4. regression guard — Newton from a far start needs the gauge term
# ===========================================================================
# Without `_decm_gauge` this returns a retcode Failure at a garbage point; with it, it converges.
let g = MEG.rhesus_macaques()
    m = DECM(g); n = length(m.dᵣ_out); L = Lfun(m)
    ref = -L(solved(g; method = :BFGS).θᵣ)
    θ0 = MEG.initial_guess(m, method = :uniform)
    plain = Opt.OptimizationFunction((θ, p) -> -L(θ), MEG.AD_methods[:AutoZygote])
    sp = Opt.solve(Opt.OptimizationProblem(plain, copy(θ0)),
                   MEG.backtracking_optimization_methods[:Newton]; maxiters = 1000)
    boolcheck("Newton/:uniform WITHOUT the gauge term fails (retcode $(sp.retcode), -L=$(round(-L(sp.u), digits=2)) vs $(round(ref, digits=2)))",
              !Opt.SciMLBase.successful_retcode(sp.retcode) || (-L(sp.u) - ref) > 1.0)
    mg = DECM(g)
    okg = try
        solve_model!(mg; method = :Newton, initial = :uniform); true
    catch; false end
    boolcheck("Newton/:uniform WITH the gauge term converges to the same optimum",
              okg && abs(-L(mg.θᵣ) - ref) < 1e-6)
end

report("DECM gauge & degeneracy")
