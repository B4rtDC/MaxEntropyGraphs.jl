##############################################################################################
#  Accuracy comparison: MaxEntropyGraphs.jl vs. NEMtropy and NuMeTriS
#
#  NEMtropy is the comparator for UBCM, DBCM, BiCM, UECM and CReM; NuMeTriS is the comparator
#  for the reciprocity-aware models (RBCM, DCReM and CRWCM).
#
#  The timing benchmarks only show how *fast* each implementation is; this script validates
#  that they converge to the *same, correct* solution. The relevant quantity is how well the
#  maximum-likelihood solution reproduces the imposed constraints, i.e. the maximum absolute
#  difference between the expected and the observed degree sequence (this is zero exactly at
#  the MLE, so it is a direct measure of solution quality / gradient norm at the optimum).
#
#  For each reference graph we report MaxEntropyGraphs.jl's constraint violation, and -- when a
#  NEMtropy solution dump produced by the generated Python scripts is available in ./accuracy/ --
#  NEMtropy's constraint violation alongside it. Each implementation is compared against its own
#  observed degree sequence, so the comparison does not depend on matching node orderings.
##############################################################################################

cd(@__DIR__)
using Pkg
Pkg.activate(pwd())
using MaxEntropyGraphs
using JSON
using Dates

const Graphs = MaxEntropyGraphs.Graphs
const accuracypath = joinpath(@__DIR__, "accuracy")
isdir(accuracypath) || mkdir(accuracypath)

"""
    constraint_violation(model) -> (max, mean)

Maximum and mean absolute difference between the expected degree sequence of a solved model and
the observed degree sequence it was fit to. Zero at the maximum-likelihood solution.
"""
function constraint_violation(model::UBCM)
    Δ = abs.(degree(model, method = :reduced) .- model.d)
    return (maximum(Δ), sum(Δ) / length(Δ))
end

function constraint_violation(model::DBCM)
    A = MaxEntropyGraphs.Ĝ(model)
    Δout = abs.(vec(sum(A, dims = 2)) .- Graphs.outdegree(model.G))
    Δin = abs.(vec(sum(A, dims = 1)) .- Graphs.indegree(model.G))
    Δ = vcat(Δout, Δin)
    return (maximum(Δ), sum(Δ) / length(Δ))
end

function constraint_violation(model::BiCM)
    A = MaxEntropyGraphs.Ĝ(model)        # biadjacency
    d⊥ = vec(sum(A, dims = 2))
    d⊤ = vec(sum(A, dims = 1))
    Δ = vcat(abs.(d⊥ .- model.d⊥), abs.(d⊤ .- model.d⊤))
    return (maximum(Δ), sum(Δ) / length(Δ))
end

# The UECM constrains both the degree and the (integer) strength sequence, so the violation combines
# the expected-adjacency row sums (degree) and the expected-weight row sums (strength).
function constraint_violation(model::UECM)
    A = MaxEntropyGraphs.Ĝ(model)        # expected adjacency
    W = MaxEntropyGraphs.Ŵ(model)        # expected (unconditional) weights
    Δd = abs.(vec(sum(A, dims = 2)) .- model.d)
    Δs = abs.(vec(sum(W, dims = 2)) .- model.s)
    Δ = vcat(Δd, Δs)
    return (maximum(Δ), sum(Δ) / length(Δ))
end

# The CReM (two-step) reproduces the degree sequence through its binary (UBCM) layer and the (continuous)
# strength sequence through its weighted layer, so the violation combines both, exactly like the UECM.
function constraint_violation(model::CReM)
    A = MaxEntropyGraphs.Ĝ(model)        # expected (binary) adjacency
    W = MaxEntropyGraphs.Ŵ(model)        # expected (unconditional) weights
    Δd = abs.(vec(sum(A, dims = 2)) .- model.d)
    Δs = abs.(vec(sum(W, dims = 2)) .- model.s)
    Δ = vcat(Δd, Δs)
    return (maximum(Δ), sum(Δ) / length(Δ))
end

# The RBCM constrains the three reciprocal degree sequences (k→, k←, k↔).
function constraint_violation(model::RBCM)
    Δ = vcat(abs.(MaxEntropyGraphs.nonreciprocated_outdegree(model) .- model.d_out),
             abs.(MaxEntropyGraphs.nonreciprocated_indegree(model)  .- model.d_in),
             abs.(MaxEntropyGraphs.reciprocated_degree(model)       .- model.d_rec))
    return (maximum(Δ), sum(Δ) / length(Δ))
end

# The DCReM (two-step) reproduces the out/in-degrees through its binary (DBCM) layer and the
# continuous out/in-strengths through its weighted layer.
function constraint_violation(model::DCReM)
    A = MaxEntropyGraphs.Ĝ(model)
    W = MaxEntropyGraphs.Ŵ(model)
    Δ = vcat(abs.(vec(sum(A, dims = 2)) .- model.d_out),
             abs.(vec(sum(A, dims = 1)) .- model.d_in),
             abs.(vec(sum(W, dims = 2)) .- model.s_out),
             abs.(vec(sum(W, dims = 1)) .- model.s_in))
    return (maximum(Δ), sum(Δ) / length(Δ))
end

# The CRWCM (two-step) reproduces the reciprocal degrees through its binary (RBCM) layer and the
# four reciprocal strength sequences through its weighted layer.
function constraint_violation(model::CRWCM)
    Δ = vcat(abs.(MaxEntropyGraphs.nonreciprocated_outdegree(model)   .- model.d_out),
             abs.(MaxEntropyGraphs.nonreciprocated_indegree(model)    .- model.d_in),
             abs.(MaxEntropyGraphs.reciprocated_degree(model)         .- model.d_rec),
             abs.(MaxEntropyGraphs.nonreciprocated_outstrength(model) .- model.s_out),
             abs.(MaxEntropyGraphs.nonreciprocated_instrength(model)  .- model.s_in),
             abs.(MaxEntropyGraphs.reciprocated_outstrength(model)    .- model.s_rec_out),
             abs.(MaxEntropyGraphs.reciprocated_instrength(model)     .- model.s_rec_in))
    return (maximum(Δ), sum(Δ) / length(Δ))
end

"""
    dyadic_probabilities(model) -> Dict{String,Matrix{Float64}}

The model's dyadic connection probability matrices, keyed exactly as the NuMeTriS dumps key them.

These are the right object for a cross-package comparison. The fitted Lagrange multipliers are **not**:
the binary sector carries an exact one-parameter gauge freedom (`αᵢ → αᵢ + c`, `βⱼ → βⱼ - c` rescales
the fitnesses by a reciprocal constant and leaves `xᵢyⱼ`, `xⱼyᵢ` and `zᵢzⱼ` all unchanged), so it leaves
every dyadic probability, and the whole likelihood, invariant. The two packages therefore settle in
different gauges and their raw parameters differ by an arbitrary offset. The dyadic probabilities are
what the models actually predict, and they are gauge-invariant.
"""
function dyadic_probabilities(m::RBCM)
    P, R, _ = MaxEntropyGraphs._dyadic_probability_matrices(m)
    return Dict("p_nonrec" => Matrix{Float64}(P), "p_rec" => Matrix{Float64}(R))
end

function dyadic_probabilities(m::DCReM)
    return Dict("p_link" => Matrix{Float64}(MaxEntropyGraphs.Ĝ(m)))
end

# The CRWCM's binary layer is an internally solved RBCM, but it stores its fitnesses rather than a
# nested model, so the dyadic matrices are rebuilt here from the same expression the model uses
# (`_CRWCM_f⭢` / `_CRWCM_f⭤`), which is also NuMeTriS's.
function dyadic_probabilities(m::CRWCM)
    x = m.xᵣ[m.dᵣ_ind]; y = m.yᵣ[m.dᵣ_ind]; z = m.zᵣ[m.dᵣ_ind]
    n = length(x)
    P = zeros(Float64, n, n)
    R = zeros(Float64, n, n)
    for i = 1:n, j = 1:n
        i == j && continue
        D = 1 + x[i]*y[j] + x[j]*y[i] + z[i]*z[j]
        P[i, j] = x[i]*y[j] / D
        R[i, j] = z[i]*z[j] / D
    end
    return Dict("p_nonrec" => P, "p_rec" => R)
end

"""
    dyadic_agreement(model, dump) -> Union{Nothing, Float64}

Maximum absolute difference between this model's dyadic probabilities and the NuMeTriS dump's, over
every shared matrix. Returns `nothing` when the dump carries no probability matrices (they are only
written for the small networks). The dump stores rows as nested lists, so `d[key][i][j]` is entry
`(i, j)`; this avoids relying on a flattening order matching between NumPy and Julia.
"""
function dyadic_agreement(model, dump)
    dump === nothing && return nothing
    ours = dyadic_probabilities(model)
    Δmax = nothing
    for (key, P) in ours
        haskey(dump, key) || continue
        rows = dump[key]
        (length(rows) == size(P, 1)) || continue
        for i in axes(P, 1), j in axes(P, 2)
            δ = abs(P[i, j] - Float64(rows[i][j]))
            Δmax = Δmax === nothing ? δ : max(Δmax, δ)
        end
    end
    return Δmax
end

"""
    nemtropy_violation(name) -> Union{Nothing, NamedTuple}

Load a NEMtropy solution dump (`accuracy/<name>_nemtropy.json`, written by the generated Python
scripts) and return its own max/mean constraint violation, or `nothing` if no dump exists.
The dump is expected to contain the keys "expected_dseq" and "dseq" (and, for bipartite graphs,
the "_rows"/"_cols" variants).
"""
function nemtropy_violation(name::AbstractString; variant::AbstractString = "")
    f = joinpath(accuracypath, "$(name)_nemtropy.json")
    isfile(f) || return nothing
    d = JSON.parsefile(f)
    Δall = Float64[]
    for (ekey, okey) in (("expected_dseq", "dseq"),
                         ("expected_sseq", "sseq"),               # UECM strength sequence
                         ("expected_dseq_rows", "rows_deg"),
                         ("expected_dseq_cols", "cols_deg"))
        # `variant` selects an alternative solver's dump (e.g. "_quasinewton"); the observed
        # sequence it is compared against is the same either way.
        ek = ekey * variant
        if haskey(d, ek) && haskey(d, okey)
            append!(Δall, abs.(Float64.(d[ek]) .- Float64.(d[okey])))
        end
    end
    isempty(Δall) && return nothing
    return (max = maximum(Δall), mean = sum(Δall) / length(Δall))
end

"""
    nemtropy_motifs(name) -> Union{Nothing, Vector{Float64}}

Load NEMtropy's expected directed 3-node motif spectrum (`accuracy/<name>_nemtropy.json`, key
`"expected_3motifs"`, written by the generated DBCM Python script), or `nothing` if absent. Motif counts are
permutation-invariant, so this can be compared to `motifs(model)` without matching node orderings.
"""
function nemtropy_motifs(name::AbstractString)
    f = joinpath(accuracypath, "$(name)_nemtropy.json")
    isfile(f) || return nothing
    d = JSON.parsefile(f)
    haskey(d, "expected_3motifs") || return nothing
    return Float64.(d["expected_3motifs"])
end

"""
    numetris_dump(name) -> Union{Nothing, Dict}

Load a NuMeTriS solution dump (`accuracy/<name>_numetris.json`, written by the generated Python
scripts), or `nothing` if no dump exists. The dump contains NuMeTriS's own (gauge-invariant)
constraint violations, reconstructed from its fitted parameters against its own observed sequences
(node-ordering independent), plus its solver norm and the empirical triadic statistics
(`Nm_emp`, and `Fm_emp` for the weighted models).
"""
function numetris_dump(name::AbstractString)
    f = joinpath(accuracypath, "$(name)_numetris.json")
    isfile(f) || return nothing
    return JSON.parsefile(f)
end

## Representative reference graphs (the small benchmark problems). These are fast to solve and
## exercise all three model types; the harness can be extended to the larger graphs if desired.
const reference_models = [
    ("UBCM_small", () -> UBCM(Graphs.SimpleGraphs.smallgraph(:karate))),
    ("DBCM_small", () -> DBCM(MaxEntropyGraphs.maspalomas())),
    ("BiCM_small", () -> BiCM(MaxEntropyGraphs.corporateclub())),
    ("UECM_small", () -> UECM(MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(MaxEntropyGraphs.rhesus_macaques()))),
    ("CReM_small", () -> CReM(MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(MaxEntropyGraphs.rhesus_macaques()))),
    ("RBCM_small", () -> RBCM(Graphs.SimpleDiGraph(MaxEntropyGraphs.rhesus_macaques()))),
    ("DCReM_small", () -> DCReM(MaxEntropyGraphs.rhesus_macaques())),
    ("CRWCM_small", () -> CRWCM(MaxEntropyGraphs.rhesus_macaques())),
]

results = Dict{String,Any}("generated" => string(now()), "models" => Dict{String,Any}())

for (name, builder) in reference_models
    model = builder()
    solve_model!(model)                         # each model's own default (fixed-point; BFGS for the UECM)
    jmax, jmean = constraint_violation(model)
    entry = Dict{String,Any}(
        "julia_max_violation" => jmax,
        "julia_mean_violation" => jmean,
    )
    nv = nemtropy_violation(name)
    if nv !== nothing
        entry["nemtropy_max_violation"] = nv.max
        entry["nemtropy_mean_violation"] = nv.mean
    end

    # Bipartite quasi-Newton pairing. The default solved above is the fixed point on both sides, but
    # the reported bipartite accuracy result is scoped to the gradient-based solver, so solve that
    # pairing explicitly here rather than leaving the quoted numbers unreproducible from this script.
    if model isa BiCM
        qmodel = builder()
        solve_model!(qmodel, method = :BFGS, initial = :degrees, maxiters = 1000, g_tol = 1e-8)
        qmax, qmean = constraint_violation(qmodel)
        entry["julia_quasinewton_max_violation"] = qmax
        entry["julia_quasinewton_mean_violation"] = qmean
        nvq = nemtropy_violation(name; variant = "_quasinewton")
        if nvq !== nothing
            entry["nemtropy_quasinewton_max_violation"] = nvq.max
            entry["nemtropy_quasinewton_mean_violation"] = nvq.mean
            @info "$(name) [quasi-Newton]: MaxEntropyGraphs max = $(qmax) | NEMtropy max = $(nvq.max)"
        end
    end

    # Motif cross-check (DBCM only): the accelerated expected 3-node motif spectrum must reproduce
    # NEMtropy's analytical ensemble means. Both are computed from each side's own solved model, so the
    # residual is dominated by the (matched, 1e-8) solver tolerance rather than any algorithmic difference.
    motif_reldiff = nothing
    if model isa DBCM
        set_Ĝ!(model)
        meg_m = motifs(model)
        entry["julia_motifs"] = meg_m
        nem_m = nemtropy_motifs(name)
        if nem_m !== nothing && length(nem_m) == length(meg_m)
            motif_reldiff = maximum(abs.(meg_m .- nem_m) ./ max.(abs.(meg_m), abs.(nem_m), 1e-300))
            entry["nemtropy_motifs"] = nem_m
            entry["motif_max_reldiff"] = motif_reldiff
        end
    end

    # NuMeTriS cross-check (reciprocity-aware models): NuMeTriS's own gauge-invariant constraint
    # violation (computed on the Python side from its fitted parameters), plus a cross-package
    # check of the EMPIRICAL triadic statistics — the observed motif counts (and fluxes, for the
    # weighted models) are convention-sensitive, so agreement validates both implementations.
    nmt = numetris_dump(name)
    if nmt !== nothing && model isa Union{RBCM,DCReM,CRWCM}
        # The two "1e-8" tolerances are NOT the same test, so the raw violations above are not a like
        # for like accuracy comparison. NuMeTriS's `tol` bounds the jacobian infinity norm, which for
        # these models IS the constraint residual, so its violation cannot exceed 1e-8 by construction.
        # `ftol` here bounds the Anderson fixed-point increment ‖FP(θ) - θ‖∞ in PARAMETER space, and the
        # jacobian factor carrying that into constraint space is ~5e3 on the weighted layers of these
        # graphs. Re-solving at a tolerance tight enough to make the constraint residuals comparable
        # shows that the two packages do agree on the optimum, so record that here rather than leaving
        # the default-tolerance numbers to be misread as an accuracy deficit.
        tight = builder()
        solve_model!(tight, ftol = 1e-12)
        tmax, tmean = constraint_violation(tight)
        entry["julia_tight_ftol"] = 1e-12
        entry["julia_tight_max_violation"] = tmax
        entry["julia_tight_mean_violation"] = tmean

        # Gauge-invariant cross-package agreement on what the models actually predict.
        dagree = dyadic_agreement(tight, nmt)
        if dagree !== nothing
            entry["numetris_dyadic_max_absdiff"] = dagree
            @info "$(name): dyadic probability agreement vs NuMeTriS = $(dagree) | MaxEntropyGraphs max violation at ftol=1e-12 = $(tmax)"
        end
    end
    if nmt !== nothing
        entry["numetris_max_violation"] = nmt["max_violation"]
        entry["numetris_mean_violation"] = nmt["mean_violation"]
        entry["numetris_norm"] = nmt["norm"]
        # Convention alignment (verified empirically on rhesus_macaques): this package (like NEMtropy
        # and the Squartini analytical formulas) counts LABELED ordered triples, i.e. each subgraph
        # occurrence |Aut(m)| times, and its flux is the TOTAL weight on the occurrence; NuMeTriS counts
        # each occurrence once and normalises the flux by the motif's link count. The per-motif
        # conversion factors below are exact, so the comparison (and any z-score) is convention-free.
        Nm_factor = [2, 1, 1, 2, 1, 2, 1, 2, 3, 1, 2, 1, 6]                     # |Aut(m)|
        Fm_factor = Nm_factor .* [2, 2, 3, 2, 3, 4, 3, 4, 3, 4, 4, 5, 6]        # |Aut(m)| · #links(m)
        if model isa RBCM && haskey(nmt, "Nm_emp")
            meg_Nm = Float64.(motifs(Matrix(Graphs.adjacency_matrix(model.G)))) ./ Nm_factor
            entry["Nm_emp_max_reldiff"] = maximum(abs.(meg_Nm .- Float64.(nmt["Nm_emp"])) ./ max.(meg_Nm, 1.0))
        end
        if model isa CRWCM && haskey(nmt, "Fm_emp")
            meg_Fm = motif_fluxes(model.G) ./ Fm_factor
            entry["Fm_emp_max_reldiff"] = maximum(abs.(meg_Fm .- Float64.(nmt["Fm_emp"])) ./ max.(meg_Fm, 1.0))
        end
    end

    results["models"][name] = entry
    @info """$(name): MaxEntropyGraphs max constraint violation = $(jmax)""" *
          (nv === nothing ? "" : " | NEMtropy max = $(nv.max)") *
          (nmt === nothing ? "" : " | NuMeTriS max = $(nmt["max_violation"])") *
          (motif_reldiff === nothing ? "" : " | motif max rel.diff vs NEMtropy = $(motif_reldiff)")
end

open(joinpath(accuracypath, "accuracy_summary.json"), "w") do f
    write(f, JSON.json(results))
end

@info "Accuracy comparison written to $(joinpath(accuracypath, "accuracy_summary.json"))"
