# DCReM
## Model description
The directed Conditional Reconstruction Method (DCReM) is the **directed** counterpart of the [`CReM`](@ref MaxEntropyGraphs.CReM): a maximum-entropy null model for **weighted, directed** networks with **continuous, positive** weights. In the literature the model is known as CReM``_A`` [[1]](#1); the NuMeTriS package [[3]](#3) calls it `DBCM+CReMa`. It is a **two-step** model:

1. a **binary layer** fixes the topology. The marginal probability of a link `i→j`, ``f_{ij} = \langle a_{ij} \rangle``, is supplied by an internally-solved [`DBCM`](@ref MaxEntropyGraphs.DBCM) on the out- and in-degree sequences, so ``f_{ij} = \frac{x_i y_j}{1 + x_i y_j}`` with ``x_i = e^{-\alpha_i}``, ``y_i = e^{-\beta_i}``;
2. a **weighted layer** fixes the out- and in-strengths. Conditional on a link `i→j` existing, its weight is exponentially distributed with rate ``\theta^{o}_i + \theta^{i}_j``. The ``2N`` parameters ``\theta = [\theta^{o}; \theta^{i}]`` are obtained by maximising the generalised (conditional) log-likelihood [[1](#1),[2](#2)].

| Description                    | Formula |
| --------------------------     | :-------------------------------------------------------------------------------- |
| Constraints                    | `` \forall i: \begin{cases} s^{out}_{i}(W^{*}) = \sum_{j \ne i} w^{*}_{ij} \\ s^{in}_{i}(W^{*}) = \sum_{j \ne i} w^{*}_{ji} \end{cases} ``|
| Hamiltonian                    | `` H(W, \theta) = \sum_{i=1}^{N} \left[ \theta^{o}_i s^{out}_{i}(W) + \theta^{i}_i s^{in}_{i}(W) \right]`` |
| $\langle a_{ij} \rangle$       | `` f_{ij} = \frac{x_i y_j}{1 + x_i y_j} \quad (\text{binary/DBCM layer})`` |
| $q(w_{ij} \mid a_{ij}=1)$      | `` (\theta^{o}_i + \theta^{i}_j)\, e^{-(\theta^{o}_i + \theta^{i}_j) w} , \quad w > 0`` |
| Log-likelihood                 | `` \mathcal{L}(\theta) = -\sum_{i=1}^{N} \left[ \theta^{o}_i s^{out}_{i} + \theta^{i}_i s^{in}_{i} \right] + \sum_{i \ne j} f_{ij} \ln (\theta^{o}_i + \theta^{i}_j) ``|
| $\langle w_{ij} \rangle$       | `` \frac{f_{ij}}{\theta^{o}_i + \theta^{i}_j}`` |
| $\mathrm{Var}(w_{ij})$         | `` \frac{f_{ij}(2 - f_{ij})}{(\theta^{o}_i + \theta^{i}_j)^{2}}`` |
| $\mathrm{Cov}(w_{ij}, w_{ji})$ | `` 0 \quad (\text{directions independent under the DBCM layer})`` |

## Creation
```julia
using MaxEntropyGraphs

# a weighted, directed network (continuous positive weights)
G = rhesus_macaques()

# instantiate a DCReM model
model = DCReM(G)
```

## Obtaining the parameters
```julia
# solve using the default settings (two-step: internal DBCM, then the fixed-point weighted layer)
solve_model!(model)
```

!!! note

    The weighted parameters ``\theta`` are the **direct** exponential rates (they appear as ``\ln(\theta^{o}_i + \theta^{i}_j)`` in the log-likelihood), so a solution requires ``\theta^{o}_i + \theta^{i}_j > 0``. Every initial guess is therefore strictly positive. The default solver is the (stable) `fixedpoint`; `BFGS` and `Newton` are also available and use a `BackTracking` line search to stay inside the feasible region. Note that the directed weighted layer has a *gauge freedom* (``\theta^{o} + c``, ``\theta^{i} - c`` leaves all rates invariant): individual parameter values are not unique, only the rate sums are.

## Expected adjacency and weights
```julia
# expected (binary) adjacency matrix; row/column sums reproduce the out-/in-degree sequences
Ĝ(model)

# expected weighted adjacency matrix; row/column sums reproduce the out-/in-strength sequences
set_Ŵ!(model)
```

## Expectation and variance
Under the DCReM every ordered node pair carries a **two-layer** random variable: the adjacency entry ``a_{ij}`` follows a Bernoulli distribution (the DBCM layer), while the weight ``w_{ij}`` follows a Bernoulli–exponential mixture (no weight without a link, an exponentially distributed weight conditional on a link). The two directions of a dyad are **independent**, so `σₓ` contains no within-dyad cross-terms:

| Layer                | ``\langle g_{ij} \rangle`` | ``\text{Var}(g_{ij})`` | ``\text{Cov}(g_{ij}, g_{ji})`` |
| -------------------- | :--- | :--- | :--- |
| binary ``(g = a)``   | ``f_{ij} = \frac{x_i y_j}{1 + x_i y_j}`` | ``f_{ij}(1 - f_{ij})`` | ``0`` (independent) |
| weighted ``(g = w)`` | ``\frac{f_{ij}}{\theta^{o}_i + \theta^{i}_j}`` | ``\frac{f_{ij}(2 - f_{ij})}{(\theta^{o}_i + \theta^{i}_j)^{2}}`` | ``0`` (independent) |

The two layers of the same pair are correlated, however: ``\text{Cov}(a_{ij}, w_{ij}) = \langle w_{ij} \rangle (1 - f_{ij})``.

!!! note "Variance propagation is per-layer"

    `σₓ` propagates the uncertainty of **one layer at a time** (`layer=:binary`, the default, or `layer=:weighted`); the cross-layer covariance above is documented for reference but not propagated. For a metric that mixes both layers, estimate its variance by sampling the ensemble (`rand(model, n)`).

```julia
# precompute the expected values and standard deviations of both layers
set_Ĝ!(model); set_σ!(model)     # binary layer
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
# generate 10 random weighted directed instances of the ensemble (continuous, exponential weights)
rand(model, 10)
```

## Model comparison
```julia
# compute the AIC (the conditional DCReM has 2N parameters; its likelihood is conditional on the
# binary layer, so compare only within the conditional family, e.g. against the CRWCM)
AIC(model)
```

## Triadic fluxes
The DCReM is the reciprocity-agnostic benchmark for weighted triadic analysis [[2]](#2): `motif_fluxes(model)` returns the **exact** expected weight circulating on each of the 13 directed 3-node motifs, and `flux_zscores` provides sampling-based significance:

```julia
motif_fluxes(model)            # exact ⟨F₁⟩, …, ⟨F₁₃⟩
flux_zscores(model, n=500)     # sampling-based z-scores of the observed fluxes
```

_References_

```@raw html
<ul>
<li>
<a id="1">[1]</a> 
Parisi, Federica and Squartini, Tiziano and Garlaschelli, Diego. <!--  author(s) --> 
<em>"A faster horse on a safer trail: generalized inference for the efficient reconstruction of weighted networks"</em> <!--  title --> 
New Journal of Physics 22, 2020. <!--  publisher(s) --> 
<a href="https://doi.org/10.1088/1367-2630/ab74a7">https://doi.org/10.1088/1367-2630/ab74a7</a>
</li>
<li>
<a id="2">[2]</a> 
Di Vece, Marzio and Pijpers, Frank P. and Garlaschelli, Diego. <!--  author(s) --> 
<em>"Commodity-specific triads in the Dutch inter-industry production network"</em> <!--  title --> 
Sci Rep 14, 3625 (2024) / arXiv:2305.12179. <!--  publisher(s) --> 
<a href="https://arxiv.org/abs/2305.12179">https://arxiv.org/abs/2305.12179</a>
</li>
<li>
<a id="3">[3]</a> 
Di Vece, Marzio. <!--  author(s) --> 
<em>"NuMeTriS: Null Models for Triadic Structures"</em> <!--  title --> 
<a href="https://github.com/MarsMDK/NuMeTriS">https://github.com/MarsMDK/NuMeTriS</a>
</li>
</ul>
```
