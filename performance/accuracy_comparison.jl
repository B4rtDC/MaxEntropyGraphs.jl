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
    nemtropy_violation(name) -> Union{Nothing, NamedTuple}

Load a NEMtropy solution dump (`accuracy/<name>_nemtropy.json`, written by the generated Python
scripts) and return its own max/mean constraint violation, or `nothing` if no dump exists.
The dump is expected to contain the keys "expected_dseq" and "dseq" (and, for bipartite graphs,
the "_rows"/"_cols" variants).
"""
function nemtropy_violation(name::AbstractString)
    f = joinpath(accuracypath, "$(name)_nemtropy.json")
    isfile(f) || return nothing
    d = JSON.parsefile(f)
    Δall = Float64[]
    for (ekey, okey) in (("expected_dseq", "dseq"),
                         ("expected_sseq", "sseq"),               # UECM strength sequence
                         ("expected_dseq_rows", "rows_deg"),
                         ("expected_dseq_cols", "cols_deg"))
        if haskey(d, ekey) && haskey(d, okey)
            append!(Δall, abs.(Float64.(d[ekey]) .- Float64.(d[okey])))
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
