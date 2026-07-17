# Which null model should I use?

The package implements nine maximum-entropy null models. They differ in the type of network they apply
to (undirected/directed/bipartite, binary/weighted) and in the structural information they preserve
(degrees, strengths, reciprocity). This page helps you pick the right one; the individual model pages
document the mathematics and the API.

## Decision tree

1. **Is the network bipartite?** → [`BiCM`](@ref MaxEntropyGraphs.BiCM) (binary; project afterwards with
   [`project`](@ref MaxEntropyGraphs.project)).
2. **Is the network undirected?**
   - binary → [`UBCM`](@ref MaxEntropyGraphs.UBCM) (degree sequence);
   - weighted, **integer** weights, degrees *and* strengths matter → [`UECM`](@ref MaxEntropyGraphs.UECM);
   - weighted, **continuous** weights, strengths conditional on the topology → [`CReM`](@ref MaxEntropyGraphs.CReM).
3. **Is the network directed and binary?**
   - reciprocity is *not* a feature you need to preserve → [`DBCM`](@ref MaxEntropyGraphs.DBCM)
     (out-/in-degree sequences);
   - reciprocity *is* structurally important → [`RBCM`](@ref MaxEntropyGraphs.RBCM) (non-reciprocated
     out-/in-degrees **and** reciprocated degrees).
4. **Is the network directed and weighted?**
   - **integer** weights, degrees *and* strengths matter → [`DECM`](@ref MaxEntropyGraphs.DECM)
     (out-/in-degrees and out-/in-strengths, jointly);
   - **continuous** weights, reciprocity is *not* a feature you need to preserve →
     [`DCReM`](@ref MaxEntropyGraphs.DCReM) (out-/in-strengths conditional on a DBCM topology);
   - **continuous** weights, reciprocity *is* structurally important → [`CRWCM`](@ref MaxEntropyGraphs.CRWCM)
     (the four reciprocal strength sequences conditional on an RBCM topology).

## Does reciprocity matter for my network?

Directed networks range from fully anti-reciprocal (e.g. food webs, ``r_t \approx 0.03``–``0.1`` for the
bundled demo food webs) to fully reciprocal (e.g. the `taro_exchange()` gift network, ``r_t = 1``).
Two quick diagnostics:

```julia
using MaxEntropyGraphs
G = MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques())

# 1. compare the observed reciprocity with the DBCM baseline
r_obs = reciprocity(G)                     # 0.757
model = DBCM(G); solve_model!(model)
r_dbcm = reciprocity(model)                # 0.561 -> the DBCM *underestimates* reciprocity

# 2. compare information criteria (same n = N(N-1) observations, directly comparable)
rmodel = RBCM(G); solve_model!(rmodel)
AICc(rmodel) < AICc(model)                 # true -> the RBCM wins despite its extra N parameters
```

If the observed reciprocity sits far from the DBCM expectation (or the RBCM wins on `AICc`), dyadic
patterns are a genuine feature of your network and higher-order analyses (e.g. triadic motifs) should use
the reciprocal models — Squartini & Garlaschelli (2011) and Di Vece et al. (2023) show that motif profiles
change *qualitatively* between the DBCM and RBCM benchmarks. The same reasoning carries over to the
weighted pair DCReM/CRWCM through the weighted reciprocity ``r_w`` ([`weighted_reciprocity`](@ref)).

## Model overview

| Model | Network | Constraints | # parameters | ``a_{ij} \perp a_{ji}``? | Weights |
| ----- | ------- | ----------- | :----------: | :----------------------: | ------- |
| [`UBCM`](@ref MaxEntropyGraphs.UBCM)   | undirected, binary   | ``k_i``                                       | ``N``  | (symmetric) | — |
| [`DBCM`](@ref MaxEntropyGraphs.DBCM)   | directed, binary     | ``k^{out}_i, k^{in}_i``                       | ``2N`` | yes | — |
| [`RBCM`](@ref MaxEntropyGraphs.RBCM)   | directed, binary     | ``k^{→}_i, k^{←}_i, k^{↔}_i``                 | ``3N`` | **no** | — |
| [`BiCM`](@ref MaxEntropyGraphs.BiCM)   | bipartite, binary    | ``k_i`` (both layers)                         | ``N_⊥ + N_⊤`` | yes | — |
| [`UECM`](@ref MaxEntropyGraphs.UECM)   | undirected, weighted | ``k_i, s_i``                                  | ``2N`` | (symmetric) | integer |
| [`DECM`](@ref MaxEntropyGraphs.DECM)   | directed, weighted   | ``k^{out}_i, k^{in}_i, s^{out}_i, s^{in}_i``  | ``4N`` | yes | integer |
| [`CReM`](@ref MaxEntropyGraphs.CReM)   | undirected, weighted | ``s_i`` (conditional on UBCM topology)        | ``N``  | (symmetric) | continuous |
| [`DCReM`](@ref MaxEntropyGraphs.DCReM) | directed, weighted   | ``s^{out}_i, s^{in}_i`` (cond. on DBCM)       | ``2N`` | yes | continuous |
| [`CRWCM`](@ref MaxEntropyGraphs.CRWCM) | directed, weighted   | ``s^{→}_i, s^{←}_i, s^{↔,out}_i, s^{↔,in}_i`` (cond. on RBCM) | ``4N`` | **no** | continuous |

!!! note "Conditional (two-step) models"
    The CReM, DCReM and CRWCM are *conditional* models: a binary layer (UBCM/DBCM/RBCM respectively) fixes
    the topology, and the weighted layer places weights on the realised links. Their likelihoods — and
    therefore their `AIC`/`AICc`/`BIC` values — are conditional on the binary layer, so compare information
    criteria only *within* the conditional family (e.g. DCReM vs CRWCM), not against the fully joint
    UECM/DECM.

!!! note "Dyadic dependence"
    Under the RBCM and CRWCM the two directions of a dyad are **correlated**
    (``\mathrm{Cov}(a_{ij}, a_{ji}) = p^{↔}_{ij} - \langle a_{ij}\rangle\langle a_{ji}\rangle \neq 0``, and
    similarly for the weights). The package handles this everywhere it matters: motif expectations are
    evaluated from the dyadic probabilities (not from `Ĝ`), the delta-method `σₓ` includes the covariance
    cross-terms, and sampling draws whole dyads. `Ĝ`-based shortcuts that are exact for the independent
    models would silently be wrong here — which is why e.g. `rand(m::RBCM, precomputed=true)` is not
    supported.

## Higher-order analysis

Once a model is chosen and solved, the typical pattern-detection workflow is:

- **Binary triadic motifs**: observed [`motifs`](@ref)`(G)` vs the model expectation `motifs(model)`
  (exact for DBCM and RBCM) with analytical z-scores via [`σₓ`](@ref MaxEntropyGraphs.σₓ(::RBCM, ::Function)),
  or sampling-based z-scores via [`motif_zscores`](@ref).
- **Triadic fluxes** (weight circulating on motifs): observed [`motif_fluxes`](@ref)`(G)` vs the exact
  expectation `motif_fluxes(model)` (DCReM/CRWCM), with sampling z-scores via [`flux_zscores`](@ref).
- **Triadic intensities** ([`motif_intensities`](@ref)): observed values plus sampling z-scores via
  [`ensemble_zscores`](@ref) (no closed-form expectation exists).

See the [Metrics](metrics.md) pages for the full analytical/simulation toolkit.

_References_

```@raw html
<ul>
<li>
<a id="1">[1]</a>
Squartini, Tiziano and Garlaschelli, Diego.
<em>"Analytical maximum-likelihood method to detect patterns in real networks"</em>
2011 New J. Phys. 13 083001.
<a href="https://iopscience.iop.org/article/10.1088/1367-2630/13/8/083001">https://iopscience.iop.org/article/10.1088/1367-2630/13/8/083001</a>
</li>
<li>
<a id="2">[2]</a>
Di Vece, Marzio; Pijpers, Frank P. and Garlaschelli, Diego.
<em>"Commodity-specific triads in the Dutch inter-industry production network"</em>
Sci Rep 14, 3625 (2024) / arXiv:2305.12179.
<a href="https://arxiv.org/abs/2305.12179">https://arxiv.org/abs/2305.12179</a>
</li>
<li>
<a id="3">[3]</a>
Parisi, Federica; Squartini, Tiziano and Garlaschelli, Diego.
<em>"A faster horse on a safer trail: generalized inference for the efficient reconstruction of weighted networks"</em>
2020 New J. Phys. 22 053053.
<a href="https://iopscience.iop.org/article/10.1088/1367-2630/ab74a7">https://iopscience.iop.org/article/10.1088/1367-2630/ab74a7</a>
</li>
</ul>
```
