# CRWCM
## Model description
The Conditionally Reciprocal Weighted Configuration Model (CRWCM) is the **reciprocity-aware** counterpart of the [`DCReM`](@ref MaxEntropyGraphs.DCReM): a maximum-entropy null model for **weighted, directed** networks with **continuous, positive** weights that accounts for the different nature of the links weights sit on. It was introduced in [[1]](#1) (`RBCM+CRWCM` in the NuMeTriS package [[2]](#2)) and is a **two-step** model:

1. a **binary layer** fixes the topology *including its reciprocity structure*: an internally-solved [`RBCM`](@ref MaxEntropyGraphs.RBCM) on the reciprocal degree sequences supplies the dyadic probabilities ``f^{→}_{ij} = p^{→}_{ij}`` (single link `i→j`) and ``f^{↔}_{ij} = p^{↔}_{ij}`` (reciprocated dyad);
2. a **weighted layer** fixes the four *reciprocal strength* sequences. Conditional on the dyad state, the weights are exponential: a non-reciprocated link `i→j` has rate ``\theta^{→}_i + \theta^{←}_j``, while a reciprocated pair carries two (conditionally independent) weights with rates ``\theta^{↔,o}_i + \theta^{↔,i}_j`` and ``\theta^{↔,o}_j + \theta^{↔,i}_i``.

The ``4N`` parameters ``\theta = [\theta^{→}; \theta^{←}; \theta^{↔,o}; \theta^{↔,i}]`` are obtained by maximising the generalised (conditional) log-likelihood, which is **block-separable** into the non-reciprocated system ``\{\theta^{→}, \theta^{←}\}`` and the reciprocated system ``\{\theta^{↔,o}, \theta^{↔,i}\}`` (solved jointly for API uniformity; the Hessian is block-diagonal).

| Description                        | Formula |
| --------------------------         | :-------------------------------------------------------------------------------- |
| Constraints                        | `` \forall i: \begin{cases} s^{→}_{i}(W^{*}) = \sum_{j \ne i} a^{*→}_{ij} w^{*}_{ij} \\ s^{←}_{i}(W^{*}) = \sum_{j \ne i} a^{*←}_{ij} w^{*}_{ji} \\ s^{↔,out}_{i}(W^{*}) = \sum_{j \ne i} a^{*↔}_{ij} w^{*}_{ij} \\ s^{↔,in}_{i}(W^{*}) = \sum_{j \ne i} a^{*↔}_{ij} w^{*}_{ji} \end{cases} ``|
| Hamiltonian                        | `` H(W, \theta) = \sum_{i=1}^{N} \left[ \theta^{→}_i s^{→}_{i} + \theta^{←}_i s^{←}_{i} + \theta^{↔,o}_i s^{↔,out}_{i} + \theta^{↔,i}_i s^{↔,in}_{i} \right]`` |
| $q(w_{ij} \mid \text{dyad state})$ | `` \begin{cases} (\theta^{→}_i + \theta^{←}_j)\, e^{-(\theta^{→}_i + \theta^{←}_j) w} & \text{single link } i→j \\ (\theta^{↔,o}_i + \theta^{↔,i}_j)\, e^{-(\theta^{↔,o}_i + \theta^{↔,i}_j) w} & \text{reciprocated dyad} \end{cases}`` |
| Log-likelihood                     | `` \mathcal{G}(\theta) = -\sum_{i} \left[ \theta^{→}_i s^{→}_{i} + \theta^{←}_i s^{←}_{i} + \theta^{↔,o}_i s^{↔,out}_{i} + \theta^{↔,i}_i s^{↔,in}_{i} \right] + \sum_{i \ne j} \left[ f^{→}_{ij} \ln (\theta^{→}_i + \theta^{←}_j) + f^{↔}_{ij} \ln (\theta^{↔,o}_i + \theta^{↔,i}_j) \right]``|
| $\langle w_{ij} \rangle$           | `` \frac{f^{→}_{ij}}{\theta^{→}_i + \theta^{←}_j} + \frac{f^{↔}_{ij}}{\theta^{↔,o}_i + \theta^{↔,i}_j}`` |
| $\mathrm{Var}(w_{ij})$             | `` \frac{2f^{→}_{ij}}{(\theta^{→}_i + \theta^{←}_j)^{2}} + \frac{2f^{↔}_{ij}}{(\theta^{↔,o}_i + \theta^{↔,i}_j)^{2}} - \langle w_{ij} \rangle^{2}`` |
| $\mathrm{Cov}(w_{ij}, w_{ji})$     | `` \frac{f^{↔}_{ij}}{(\theta^{↔,o}_i + \theta^{↔,i}_j)(\theta^{↔,o}_j + \theta^{↔,i}_i)} - \langle w_{ij} \rangle \langle w_{ji} \rangle \ne 0`` |

!!! note "When to use the CRWCM instead of the DCReM"

    Within a dyad the two weights are **correlated** under the CRWCM (they are simultaneously non-zero exactly when the dyad is reciprocated). This is the defining difference with the [`DCReM`](@ref MaxEntropyGraphs.DCReM), where ``\mathrm{Cov}(w_{ij}, w_{ji}) = 0``. If the observed (weighted) reciprocity of the network deviates from the DBCM/DCReM baseline (compare `weighted_reciprocity(G)` with `weighted_reciprocity(::DCReM)`), weights sit on reciprocated links in a structured way and the CRWCM is the appropriate benchmark. Also see the [Which model when?](../model_selection.md) page and [[1]](#1).

## Creation
```julia
using MaxEntropyGraphs

# a weighted, directed network with substantial (weighted) reciprocity (r_w ≈ 0.9)
G = rhesus_macaques()

# instantiate a CRWCM model
model = CRWCM(G)
```

Because the weights are strictly positive, a node has a zero strength in a channel *iff* its degree in that channel is zero; the constructor enforces this consistency.

## Obtaining the parameters
```julia
# solve using the default settings (two-step: internal RBCM, then the fixed-point weighted layer)
solve_model!(model)
```

!!! note

    The weighted parameters ``\theta`` are the **direct** exponential rates, so the feasible region requires positive rate sums; every initial guess is strictly positive and the gradient methods use a `BackTracking` line search. *Dead channels* (zero strength ⟺ zero dyadic probability everywhere, e.g. nodes without any non-reciprocated out-link) have an undetermined parameter: they are excluded from the optimisation and pinned to ``+\infty`` (an infinite rate, i.e. an exactly zero weight) after the solve. The default `fixedpoint` solver is stable; like the DCReM, the weighted layer has a per-block gauge freedom, so only rate sums are identified.

## Expected adjacency and weights
```julia
# expected (binary) adjacency matrix from the RBCM layer; and its dyadic probabilities
set_Ĝ!(model)

# expected weighted adjacency matrix; row/column sums reproduce the TOTAL out-/in-strengths (s→ + s↔out etc.)
set_Ŵ!(model)
```

## Expectation and variance
Under the CRWCM every dyad carries a **two-layer** random variable: the pair ``(a_{ij}, a_{ji})`` follows the four-state dyadic distribution of the RBCM layer, while the weights follow the three-channel exponential mixture of the weighted layer (a weight is present on exactly the channels the dyad state activates). Writing ``r^{→}_{ij} = \theta^{→}_i + \theta^{←}_j`` and ``r^{↔}_{ij} = \theta^{↔,o}_i + \theta^{↔,i}_j`` for the exponential rates, the first two moments are:

| Layer                | ``\langle g_{ij} \rangle`` | ``\text{Var}(g_{ij})`` | ``\text{Cov}(g_{ij}, g_{ji})`` |
| -------------------- | :--- | :--- | :--- |
| binary ``(g = a)``   | ``f^{→}_{ij} + f^{↔}_{ij}`` | ``\langle a_{ij} \rangle (1 - \langle a_{ij} \rangle)`` | ``f^{↔}_{ij} - \langle a_{ij} \rangle \langle a_{ji} \rangle`` |
| weighted ``(g = w)`` | ``\frac{f^{→}_{ij}}{r^{→}_{ij}} + \frac{f^{↔}_{ij}}{r^{↔}_{ij}}`` | ``\frac{2f^{→}_{ij}}{(r^{→}_{ij})^{2}} + \frac{2f^{↔}_{ij}}{(r^{↔}_{ij})^{2}} - \langle w_{ij} \rangle^{2}`` | ``\frac{f^{↔}_{ij}}{r^{↔}_{ij} r^{↔}_{ji}} - \langle w_{ij} \rangle \langle w_{ji} \rangle`` |

The two layers of the same pair are correlated as well: ``\text{Cov}(a_{ij}, w_{ij}) = \langle w_{ij} \rangle (1 - \langle a_{ij} \rangle)``.

!!! note "Variance propagation is per-layer, dyadic covariance included"

    `σₓ` propagates the uncertainty of **one layer at a time** (`layer=:binary`, the default, or `layer=:weighted`), and within each layer it **includes** the within-dyad covariance cross-terms of the table above (the binary layer inherits the RBCM dyadic covariance; the weighted layer the reciprocated-channel correlation). The cross-layer covariance is documented for reference but not propagated: for a metric that mixes both layers, estimate its variance by sampling the ensemble (`rand(model, n)`).

```julia
# precompute the expected values and standard deviations of both layers
set_Ĝ!(model); set_σ!(model)     # binary (RBCM) layer
set_Ŵ!(model); set_σʷ!(model)    # weighted layer

# delta-method standard deviation of a weighted metric (use layer=:binary for adjacency-based metrics)
X = W -> sum(W .^ 2)             # sum of the squared weights
X_std = σₓ(model, X, layer=:weighted)

# z-score of the observed value
W_obs = MaxEntropyGraphs.Graphs.weights(G)
z_X = (X(W_obs) - X(model.Ŵ)) / X_std
```

!!! warning "Memory footprint"

    `Ĝ`/`σˣ` and `Ŵ`/`σʷ` (with their `set_Ĝ!`/`set_σ!`/`set_Ŵ!`/`set_σʷ!` variants) materialize dense ``N \times N`` matrices, and `σₓ` requires them. This is ``O(N^2)`` memory, intended for small networks; for large networks, prefer sampling to estimate variances (see [Performance and scalability](../performance.md)).

## Sampling the ensemble
```julia
# generate 10 random weighted directed instances (per-dyad four-state draw + exponential weights)
rand(model, 10)
```

## Model comparison
```julia
# compute the AIC (the conditional CRWCM has 4N parameters; compare within the conditional family,
# e.g. against the DCReM with 2N parameters, both with n = N(N-1) observations)
AIC(model)
```

## Triadic fluxes
The CRWCM is the reciprocity-aware benchmark for weighted triadic analysis [[1]](#1): `motif_fluxes(model)` returns the **exact** expected weight circulating on each of the 13 directed 3-node motifs (the within-dyad correlation is handled through the dyadic expectations), and `flux_zscores` provides sampling-based significance:

```julia
motif_fluxes(model)            # exact ⟨F₁⟩, …, ⟨F₁₃⟩
flux_zscores(model, n=500)     # sampling-based z-scores of the observed fluxes (NuMeTriS-style)
motif_zscores(model, n=500)    # binary motif z-scores under the RBCM layer
```

_References_

```@raw html
<ul>
<li>
<a id="1">[1]</a> 
Di Vece, Marzio and Pijpers, Frank P. and Garlaschelli, Diego. <!--  author(s) --> 
<em>"Commodity-specific triads in the Dutch inter-industry production network"</em> <!--  title --> 
Sci Rep 14, 3625 (2024) / arXiv:2305.12179. <!--  publisher(s) --> 
<a href="https://arxiv.org/abs/2305.12179">https://arxiv.org/abs/2305.12179</a>
</li>
<li>
<a id="2">[2]</a> 
Di Vece, Marzio. <!--  author(s) --> 
<em>"NuMeTriS: Null Models for Triadic Structures"</em> <!--  title --> 
<a href="https://github.com/MarsMDK/NuMeTriS">https://github.com/MarsMDK/NuMeTriS</a>
</li>
<li>
<a id="3">[3]</a> 
Parisi, Federica and Squartini, Tiziano and Garlaschelli, Diego. <!--  author(s) --> 
<em>"A faster horse on a safer trail: generalized inference for the efficient reconstruction of weighted networks"</em> <!--  title --> 
New Journal of Physics 22, 2020. <!--  publisher(s) --> 
<a href="https://doi.org/10.1088/1367-2630/ab74a7">https://doi.org/10.1088/1367-2630/ab74a7</a>
</li>
</ul>
```
