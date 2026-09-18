# Seeded graph corpora for the solver-robustness sweep.
#
# Provenance. These generators are the ones that produced the published robustness tables,
# lifted here so that those tables become reproducible rather than remaining as prose:
#
#   * `bipartite`                          -> validation/bicm_uecm_solver_geometry.md (the 183-graph
#                                             Anderson-ladder tables), seed SEED_BIPARTITE
#   * `weighted_undirected` / `_directed`  -> validation/uecm_decm_fixedpoint.md (the 150-graph
#                                             cold-start tables), seed SEED_WEIGHTED
#
# Every generator repairs isolated vertices deterministically. That is not cosmetic: the package
# only models connected inputs, the BiCM rejects isolated vertices outright since v0.8.0, and
# CReM/DCReM raise `NLsolve.IsFiniteException` on a zero-strength node. A corpus that leaked one
# would be measuring input rejection rather than solver robustness.

using Random
using MaxEntropyGraphs
const MEG = MaxEntropyGraphs
const G_  = MEG.Graphs
const SWG = MEG.SimpleWeightedGraphs

"Seed behind the published BiCM Anderson-ladder tables (183 constructible graphs of 200 draws)."
const SEED_BIPARTITE = 20260917

"Seed behind the published UECM/DECM cold-start tables (150 draws each)."
const SEED_WEIGHTED = 2026

"Seed for the corpora that have no published predecessor (binary and directed bipartite)."
const SEED_BINARY = 20260918

"""
    binary_undirected(rng, nv, p)

Erdos-Renyi `SimpleGraph` on `nv` vertices, every isolated vertex repaired by linking it to
`mod1(i+1, nv)`.
"""
function binary_undirected(rng, nv, p)
    g = G_.SimpleGraph(nv)
    for i in 1:nv, j in i+1:nv
        rand(rng) < p && G_.add_edge!(g, i, j)
    end
    for i in 1:nv
        G_.degree(g, i) == 0 && G_.add_edge!(g, i, mod1(i + 1, nv))
    end
    g
end

"""
    binary_directed(rng, nv, p)

Erdos-Renyi `SimpleDiGraph`. Both a zero out-degree and a zero in-degree are repaired, since the
DBCM and RBCM carry a dead channel for either one.
"""
function binary_directed(rng, nv, p)
    g = G_.SimpleDiGraph(nv)
    for i in 1:nv, j in 1:nv
        i == j && continue
        rand(rng) < p && G_.add_edge!(g, i, j)
    end
    for i in 1:nv
        G_.outdegree(g, i) == 0 && G_.add_edge!(g, i, mod1(i + 1, nv))
        G_.indegree(g, i)  == 0 && G_.add_edge!(g, mod1(i + 1, nv), i)
    end
    g
end

"""
    bipartite(rng, Nb, Nt, p)

Bipartite `SimpleGraph` with the bottom layer `1:Nb` and the top layer `Nb+1:Nb+Nt`. Isolated
vertices on either side are attached to a random counterpart.
"""
function bipartite(rng, Nb, Nt, p)
    g = G_.SimpleGraph(Nb + Nt)
    for b in 1:Nb, t in 1:Nt
        rand(rng) < p && G_.add_edge!(g, b, Nb + t)
    end
    for b in 1:Nb
        G_.degree(g, b) == 0 && G_.add_edge!(g, b, Nb + rand(rng, 1:Nt))
    end
    for t in 1:Nt
        G_.degree(g, Nb + t) == 0 && G_.add_edge!(g, rand(rng, 1:Nb), Nb + t)
    end
    g
end

"""
    dibipartite(rng, Nb, Nt, p⁺, p⁻)

Directed bipartite `SimpleDiGraph` with two independent Bernoulli channels, `p⁺` for the
bottom-to-top links and `p⁻` for top-to-bottom. The repair pass uses total degree, matching the
DBiCM's own isolated-vertex guard: a vertex that only receives is layer-assignable and legitimate.
"""
function dibipartite(rng, Nb, Nt, p⁺, p⁻)
    g = G_.SimpleDiGraph(Nb + Nt)
    for b in 1:Nb, t in 1:Nt
        rand(rng) < p⁺ && G_.add_edge!(g, b, Nb + t)
        rand(rng) < p⁻ && G_.add_edge!(g, Nb + t, b)
    end
    for b in 1:Nb
        G_.degree(g, b) == 0 && G_.add_edge!(g, b, Nb + rand(rng, 1:Nt))
    end
    for t in 1:Nt
        G_.degree(g, Nb + t) == 0 && G_.add_edge!(g, rand(rng, 1:Nb), Nb + t)
    end
    g
end

"""
    weighted_undirected(rng, nv, p, wmax)

`SimpleWeightedGraph` with integer weights drawn from `1:wmax`.
"""
function weighted_undirected(rng, nv, p, wmax)
    A = zeros(Int, nv, nv)
    for i in 1:nv, j in i+1:nv
        rand(rng) < p && (A[i, j] = A[j, i] = rand(rng, 1:wmax))
    end
    for i in 1:nv
        if all(iszero, @view A[i, :])
            j = mod1(i + 1, nv)
            A[i, j] = A[j, i] = rand(rng, 1:wmax)
        end
    end
    SWG.SimpleWeightedGraph(A)
end

"""
    weighted_directed(rng, nv, p, wmax)

`SimpleWeightedDiGraph` with integer weights drawn from `1:wmax`, repaired on both out- and
in-strength.
"""
function weighted_directed(rng, nv, p, wmax)
    A = zeros(Int, nv, nv)
    for i in 1:nv, j in 1:nv
        i == j && continue
        rand(rng) < p && (A[i, j] = rand(rng, 1:wmax))
    end
    for i in 1:nv
        all(iszero, @view A[i, :]) && (A[i, mod1(i + 1, nv)] = rand(rng, 1:wmax))
        all(iszero, @view A[:, i]) && (A[mod1(i + 1, nv), i] = rand(rng, 1:wmax))
    end
    SWG.SimpleWeightedDiGraph(A)
end

# ---------------------------------------------------------------------------------------------
# Corpora. Each returns a `Vector` of graphs built from one seed, so a corpus is fully described
# by its name and the number of draws.
# ---------------------------------------------------------------------------------------------

"""
    corpus(kind::Symbol, n::Int)

Build `n` graphs of the given kind. The draw ranges reproduce the published studies where one
exists; `:binary_undirected`, `:binary_directed` and `:dibipartite` are new and sized to match.
"""
function corpus(kind::Symbol, n::Int)
    if kind === :bipartite
        rng = Xoshiro(SEED_BIPARTITE)
        return [bipartite(rng, rand(rng, 4:30), rand(rng, 4:30), rand(rng) * 0.6 + 0.05) for _ in 1:n]
    elseif kind === :dibipartite
        rng = Xoshiro(SEED_BINARY)
        return [dibipartite(rng, rand(rng, 4:20), rand(rng, 4:20),
                            rand(rng) * 0.4 + 0.1, rand(rng) * 0.4 + 0.1) for _ in 1:n]
    elseif kind === :weighted_undirected
        rng = Xoshiro(SEED_WEIGHTED)
        return [weighted_undirected(rng, rand(rng, 6:18), 0.15 + 0.45rand(rng), rand(rng, 2:8)) for _ in 1:n]
    elseif kind === :weighted_directed
        rng = Xoshiro(SEED_WEIGHTED)
        return [weighted_directed(rng, rand(rng, 6:16), 0.15 + 0.45rand(rng), rand(rng, 2:8)) for _ in 1:n]
    elseif kind === :binary_undirected
        rng = Xoshiro(SEED_BINARY)
        return [binary_undirected(rng, rand(rng, 8:30), 0.15 + 0.45rand(rng)) for _ in 1:n]
    elseif kind === :binary_directed
        rng = Xoshiro(SEED_BINARY)
        return [binary_directed(rng, rand(rng, 8:24), 0.15 + 0.45rand(rng)) for _ in 1:n]
    else
        throw(ArgumentError("unknown corpus $(kind)"))
    end
end
