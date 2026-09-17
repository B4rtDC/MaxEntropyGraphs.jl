# DBiCM (directed bipartite configuration model) — symbolic validation.
#
# The DBiCM's single structural claim is that its Hamiltonian SEPARATES: constraining the four degree
# sequences of a directed bipartite network gives, per (⊥ᵢ, ⊤α) pair,
#     H = (αᵢ + δ_α)·B⁺ + (βᵢ + γ_α)·B⁻,        B⁺, B⁻ ∈ {0,1}
# so the four-state dyad factorises into two independent Bernoulli variables and the model is exactly
# two BiCMs — one on (d⊥_out, d⊤_in), one on (d⊥_in, d⊤_out). Everything the implementation does rests
# on that, so it is what this script proves.
#
# Part 1 — the four-state dyad:
#   Z = (1 + a)(1 + b) with a = x⊥out·x⊤in, b = x⊥in·x⊤out; hence ⟨B⁺⟩ = a/(1+a) free of the ⁻
#   parameters (the separation itself), Var[B±] = p±(1-p±), and Cov(B⁺,B⁻) = 0 WITHIN a dyad — the
#   property that makes this the bipartite analogue of the DBCM rather than of the RBCM.
#
# Part 2 — independence across dyads, for all three pairings (⁺⁺, ⁻⁻, ⁺⁻) and for dyads that share a
#   ⊥ node, a ⊤ node, or nothing. This is what makes all three directed V-motif projection kernels
#   exactly Poisson-binomial, including the ⁺⁻ diagonal (the reciprocated-partner count).
#
# Part 3 — the likelihood: L_DBiCM == L_BiCM(⁺ block) + L_BiCM(⁻ block); exactly two gauge directions,
#   one per channel; and zero cross-channel curvature, which is what licenses solving the two channels
#   separately rather than as one four-block system.
#
# Numeric closure: a DBiCM fitted on a deterministic synthetic directed bipartite graph; Ĝ⁺/Ĝ⁻
# (src/Models/DBiCM.jl) must match p± evaluated at the fitted parameters, and each channel must
# reproduce a standalone BiCM fitted on the same pair of sequences.

include(joinpath(@__DIR__, "common.jl"))

using MaxEntropyGraphs
using LinearAlgebra

@variables xo wi yi zo xo2 wi2 yi2 zo2

const DOM4 = [xo => (1//100, 20), wi => (1//100, 20), yi => (1//100, 20), zo => (1//100, 20)]
const DOM8 = vcat(DOM4, [xo2 => (1//100, 20), wi2 => (1//100, 20), yi2 => (1//100, 20), zo2 => (1//100, 20)])

# ------------------------------------------------------------------
# Part 1 — the four-state dyad (B⁺, B⁻) ∈ {0,1}²
# ------------------------------------------------------------------
a = xo * wi      # ⊥ out-fitness × ⊤ in-fitness   -> governs B⁺ (⊥ → ⊤)
b = yi * zo      # ⊥ in-fitness  × ⊤ out-fitness  -> governs B⁻ (⊤ → ⊥)

Z    = sum(a^mp * b^mm for mp in 0:1, mm in 0:1)
Ep   = sum(mp      * a^mp * b^mm for mp in 0:1, mm in 0:1) / Z
Em   = sum(mm      * a^mp * b^mm for mp in 0:1, mm in 0:1) / Z
Ep2  = sum(mp^2    * a^mp * b^mm for mp in 0:1, mm in 0:1) / Z
Epm  = sum(mp * mm * a^mp * b^mm for mp in 0:1, mm in 0:1) / Z

pp = a / (1 + a)
pm = b / (1 + b)

verify("dyad: Z factorises as (1+a)(1+b)", Z - (1 + a)*(1 + b), DOM4)
# THE SEPARATION: ⟨B⁺⟩ depends on the ⁻ parameters not at all
verify("dyad: <B+> == a/(1+a), free of the ⁻ parameters", Ep - pp, DOM4)
verify("dyad: <B-> == b/(1+b), free of the ⁺ parameters", Em - pm, DOM4)
verify("dyad: Var[B+] == p+(1-p+)", (Ep2 - Ep^2) - pp*(1 - pp), DOM4)
# NO RECIPROCITY COUPLING: the defining contrast with the RBCM/DECM
verify("dyad: Cov(B+, B-) == 0 within a dyad", Epm - Ep*Em, DOM4)
# the shipped kernel is the BiCM's, called on each channel's own fitness product
verify("kernel: f_BiCM(a) == p+", MaxEntropyGraphs.f_BiCM(a) - pp, DOM4)
verify("kernel: f_BiCM(b) == p-", MaxEntropyGraphs.f_BiCM(b) - pm, DOM4)

# ------------------------------------------------------------------
# Part 2 — independence across dyads, for every pairing the projection uses
# ------------------------------------------------------------------
# Two dyads with independent weights; the joint sum factorises, which is what the three directed
# V-motif kernels need in order to be exactly Poisson-binomial.
function joint(f, wa, wb)
    Zj = sum(wa(m1) * wb(m2) for m1 in 0:1, m2 in 0:1)
    return sum(f(m1, m2) * wa(m1) * wb(m2) for m1 in 0:1, m2 in 0:1) / Zj
end
w(t) = m -> t^m

for (name, wa, wb, ea, eb) in (
        ("V-out  (B+_iα, B+_jα): distinct ⊥, shared ⊤", w(xo*wi),  w(xo2*wi),  xo*wi/(1+xo*wi),  xo2*wi/(1+xo2*wi)),
        ("V-in   (B-_iα, B-_jα): distinct ⊥, shared ⊤", w(yi*zo),  w(yi2*zo),  yi*zo/(1+yi*zo),  yi2*zo/(1+yi2*zo)),
        ("V-path (B+_iα, B-_jα): distinct ⊥, shared ⊤", w(xo*wi),  w(yi2*zo),  xo*wi/(1+xo*wi),  yi2*zo/(1+yi2*zo)),
        ("V-path diagonal (B+_iα, B-_iα): same ⊥",      w(xo*wi),  w(yi*zo),   xo*wi/(1+xo*wi),  yi*zo/(1+yi*zo)),
        ("fully distinct dyads",                         w(xo*wi),  w(xo2*wi2), xo*wi/(1+xo*wi),  xo2*wi2/(1+xo2*wi2)))
    verify("independence: $name", joint((m1,m2) -> m1*m2, wa, wb) - ea*eb, DOM8)
end

# ------------------------------------------------------------------
# Part 3 — likelihood, gauge and cross-channel curvature (numeric, on a fitted model)
# ------------------------------------------------------------------
G_ = MaxEntropyGraphs.Graphs
function planted_dibipartite(seed=17, Nb=13, Nt=8, pp=0.32, pm=0.25)
    rng = MaxEntropyGraphs.Xoshiro(seed)
    g = G_.SimpleDiGraph(Nb + Nt)
    for i in 1:Nb, j in 1:Nt
        rand(rng) < pp && G_.add_edge!(g, i, Nb + j)
        rand(rng) < pm && G_.add_edge!(g, Nb + j, i)
    end
    for v in G_.vertices(g)
        if iszero(G_.degree(g, v))
            v <= Nb ? G_.add_edge!(g, v, Nb + 1 + (v % Nt)) : G_.add_edge!(g, 1 + (v % Nb), v)
        end
    end
    return g
end

G = planted_dibipartite()
model = DBiCM(G)
solve_model!(model, method = :BFGS)

# L_DBiCM is the two channels' BiCM likelihoods summed — no new mathematics
b⁺ = BiCM(nothing; d⊥ = model.d⊥_out, d⊤ = model.d⊤_in)
b⁻ = BiCM(nothing; d⊥ = model.d⊥_in,  d⊤ = model.d⊤_out)
solve_model!(b⁺, method = :BFGS); solve_model!(b⁻, method = :BFGS)
closecheck("L_DBiCM == L_BiCM(⁺) + L_BiCM(⁻)",
           MaxEntropyGraphs.L_DBiCM_reduced(model),
           MaxEntropyGraphs.L_BiCM_reduced(b⁺) + MaxEntropyGraphs.L_BiCM_reduced(b⁻); rtol = 1e-8)

# each channel reproduces a standalone BiCM: validates the four independent reductions and all four
# class-index maps against an implementation that builds its own reduction and θ layout
boolcheck("channel ⁺ Ĝ == standalone BiCM Ĝ",
          maximum(abs.(MaxEntropyGraphs.Ĝ(model, channel = :to_top) .- MaxEntropyGraphs.Ĝ(b⁺))) < 1e-6)
boolcheck("channel ⁻ Ĝ == standalone BiCM Ĝ",
          maximum(abs.(MaxEntropyGraphs.Ĝ(model, channel = :to_bottom) .- MaxEntropyGraphs.Ĝ(b⁻))) < 1e-6)

# four independent reductions are never coarser than a DBCM-style joint (out, in) reduction
joint_classes = 2 * (length(unique(collect(zip(model.d⊥_out, model.d⊥_in)))) +
                     length(unique(collect(zip(model.d⊤_out, model.d⊤_in)))))
boolcheck("independent reduction ($(length(model.θᵣ)) classes) ≤ joint-pair reduction ($joint_classes)",
          length(model.θᵣ) <= joint_classes)

# gauge: exactly two flat directions, one per channel, and no curvature between the channels
θ = copy(model.θᵣ); live = findall(isfinite, θ)
posmap = Dict(i => k for (k, i) in enumerate(live))
Lfun = t -> MaxEntropyGraphs.L_DBiCM_reduced(map(i -> haskey(posmap, i) ? t[posmap[i]] : zero(eltype(t)), eachindex(θ)), model)
H = MaxEntropyGraphs.ForwardDiff.hessian(Lfun, θ[live])
ev = sort(abs.(eigvals(Symmetric(H))))
boolcheck("Hessian null dimension is exactly 2 (one gauge per channel)", ev[1] < 1e-8 && ev[2] < 1e-8 && ev[3] > 1e-4)
n⁺live = count(<=(model.status[:n⁺]), live)
boolcheck("cross-channel Hessian block is identically zero", maximum(abs.(H[1:n⁺live, n⁺live+1:end])) < 1e-10)
boolcheck("condition number off the gauge is O(10), as for the BiCM", ev[end]/ev[3] < 1e3)

# numeric closure of Ĝ against p± at the fitted parameters
x⊥o = model.x⊥ᵣ_out[model.d⊥ᵣ_out_ind]; x⊤i = model.x⊤ᵣ_in[model.d⊤ᵣ_in_ind]
x⊥i = model.x⊥ᵣ_in[model.d⊥ᵣ_in_ind];   x⊤o = model.x⊤ᵣ_out[model.d⊤ᵣ_out_ind]
P⁺ = MaxEntropyGraphs.Ĝ(model, channel = :to_top); P⁻ = MaxEntropyGraphs.Ĝ(model, channel = :to_bottom)
for (i, α) in ((1, 1), (2, 3), (model.status[:N⊥], model.status[:N⊤]))
    closecheck("Ĝ⁺[$i,$α] == x⊥out·x⊤in/(1+…)", P⁺[i,α], x⊥o[i]*x⊤i[α]/(1 + x⊥o[i]*x⊤i[α]))
    closecheck("Ĝ⁻[$i,$α] == x⊥in·x⊤out/(1+…)", P⁻[i,α], x⊥i[i]*x⊤o[α]/(1 + x⊥i[i]*x⊤o[α]))
end

# reciprocity: exact, because ⟨B⁺B⁻⟩ = p⁺p⁻, and it must agree with the matrix method on the full
# expected adjacency matrix
N⊥, N⊤ = model.status[:N⊥], model.status[:N⊤]
Â = zeros(N⊥ + N⊤, N⊥ + N⊤)
Â[1:N⊥, N⊥+1:end] = P⁺
Â[N⊥+1:end, 1:N⊥] = permutedims(P⁻)
closecheck("reciprocity(m) == reciprocity(Â)", MaxEntropyGraphs.reciprocity(model), MaxEntropyGraphs.reciprocity(Â); rtol = 1e-10)

report("DBiCM")
