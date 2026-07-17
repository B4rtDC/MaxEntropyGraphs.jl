# DECM
## Model description
The Directed Enhanced Configuration Model (DECM) is a maximum-entropy null model for **weighted, directed** networks. It simultaneously constrains, per node, the out- and in-degree sequences (the number of outgoing and incoming neighbours) and the out- and in-strength sequences (the total outgoing and incoming weight). It is the directed counterpart of the [UECM](UECM.md). The weights are required to take non-negative **integer** values; each ordered pair of nodes then carries a Bernoulli–geometric weight distribution [[1](#1),[2](#2)].

We define the parameter vector as ``\theta = [\alpha^{out} ; \alpha^{in} ; \beta^{out} ; \beta^{in}]``, where ``\alpha`` and ``\beta`` denote the parameters associated with the degree and the strength sequences respectively. In the code the exponentiated parameters ``x^{out}_i = e^{-\alpha^{out}_i}``, ``x^{in}_i = e^{-\alpha^{in}_i}``, ``y^{out}_i = e^{-\beta^{out}_i}`` and ``y^{in}_i = e^{-\beta^{in}_i}`` are used.

| Description                    | Formula |
| --------------------------     | :-------------------------------------------------------------------------------- |
| Constraints                    | `` \forall i: \begin{cases} k^{out}_{i}(A^{*}) = \sum_{j \ne i} a^{*}_{ij} \\ k^{in}_{i}(A^{*}) = \sum_{j \ne i} a^{*}_{ji} \\ s^{out}_{i}(W^{*}) = \sum_{j \ne i} w^{*}_{ij} \\ s^{in}_{i}(W^{*}) = \sum_{j \ne i} w^{*}_{ji} \end{cases} ``|
| Hamiltonian                    | `` H(W, \theta) = \sum_{i=1}^{N} \left[ \alpha^{out}_i k^{out}_{i}(A) + \alpha^{in}_i k^{in}_{i}(A) + \beta^{out}_i s^{out}_{i}(W) + \beta^{in}_i s^{in}_{i}(W) \right]`` |
| $\langle a_{ij} \rangle$       | `` p_{ij} = \frac{e^{-\alpha^{out}_i - \alpha^{in}_j - \beta^{out}_i - \beta^{in}_j}}{1 - e^{-\beta^{out}_i - \beta^{in}_j} + e^{-\alpha^{out}_i - \alpha^{in}_j - \beta^{out}_i - \beta^{in}_j}}`` |
| Log-likelihood                 | `` \mathcal{L}(\theta) = -\sum_{i=1}^{N} \left[ \alpha^{out}_i k^{out}_{i} + \alpha^{in}_i k^{in}_{i} + \beta^{out}_i s^{out}_{i} + \beta^{in}_i s^{in}_{i} \right] - \sum_{i=1}^{N} \sum_{j \ne i} \ln \left[ 1 + e^{-\alpha^{out}_i - \alpha^{in}_j} \left( \frac{e^{-\beta^{out}_i - \beta^{in}_j}}{1 - e^{-\beta^{out}_i - \beta^{in}_j}} \right) \right] ``|
| $\langle w_{ij} \rangle$       | `` \frac{p_{ij}}{1 - e^{-\beta^{out}_i - \beta^{in}_j}}`` |
| $\sigma^{*}(X)$                | ``\sqrt{\sum_{i \ne j} \left( \sigma^{*}[a_{ij}] \frac{\partial X}{\partial a_{ij}}  \right)^{2}_{A = \langle A^{*} \rangle}}`` |
| $\sigma^{*}[a_{ij}]$           | ``\sqrt{p_{ij} (1 - p_{ij})} ``   |

Note that the sums in the log-likelihood run over the **ordered** pairs ``(i,j)``: a directed dyad carries two distinct, independent random variables ``g_{ij}`` and ``g_{ji}``.

## Creation
```julia
using Graphs, SimpleWeightedGraphs
using MaxEntropyGraphs

# a weighted, directed network with integer weights
# (here: the rhesus macaques grooming network)
G = rhesus_macaques()

# instantiate a DECM model
model = DECM(G)
```

## Obtaining the parameters
```julia
# solve using the default settings (BFGS)
solve_model!(model)
```

!!! note

    Contrary to the purely binary models, the fixed-point recipe is very unstable for the DECM and should not be used [[1]](#1). The default solver is therefore `BFGS`; `Newton` is also available (and typically the fastest, cf. [[1]](#1)) but is more sensitive to the initial guess. Because the likelihood is only defined on the feasible region ``e^{-\beta^{out}_i - \beta^{in}_j} < 1``, the DECM uses a `BackTracking` line search that keeps the iterates inside that region (as does the UECM).

## Expected adjacency and weights
```julia
# expected (binary) adjacency matrix;
# row sums reproduce the out-degrees, column sums the in-degrees
Ĝ(model)

# expected weighted adjacency matrix;
# row sums reproduce the out-strengths, column sums the in-strengths
Ŵ(model)
```

## Expectation and variance
Under the DECM each **ordered** pair of nodes carries a **two-layer** random variable: the adjacency entry ``a_{ij}`` follows a Bernoulli distribution, while the weight ``w_{ij}`` follows a Bernoulli–geometric mixture (no weight without a link, a geometrically distributed weight conditional on a link). Writing ``x = x^{out}_i x^{in}_j`` and ``y = y^{out}_i y^{in}_j``, the first two moments are:

| Layer                | ``\langle g_{ij} \rangle`` | ``\text{Var}(g_{ij})`` | ``\text{Cov}(g_{ij}, g_{ji})`` |
| -------------------- | :--- | :--- | :--- |
| binary ``(g = a)``   | ``p_{ij} = \frac{x y}{1 - y + x y}`` | ``p_{ij}(1 - p_{ij})`` | ``= 0`` (independent variables) |
| weighted ``(g = w)`` | ``\frac{p_{ij}}{1 - y}`` | ``\frac{p_{ij}(1 + y - p_{ij})}{(1 - y)^{2}}`` | ``= 0`` (independent variables) |

The two layers of a channel are correlated: ``\text{Cov}(a_{ij}, w_{ij}) = \langle w_{ij} \rangle (1 - p_{ij})``.

!!! note "Variance propagation is per-layer"

    `σₓ` propagates the uncertainty of **one layer at a time** (`layer=:binary`, the default, or `layer=:weighted`); the cross-layer covariance above is documented for reference but not propagated. For a metric that mixes both layers, estimate its variance by sampling the ensemble (`rand(model, n)`).

The workflow mirrors that of the binary models: precompute the expected matrices and the entry-wise standard deviations, then propagate them through a metric `X` with `σₓ` (the delta method):

```jldoctest DECM_variance; output = false
using Graphs, SimpleWeightedGraphs
using MaxEntropyGraphs

G = rhesus_macaques()
model = DECM(G)
solve_model!(model)

# precompute the expected values and standard deviations of both layers
set_Ĝ!(model); set_σ!(model)     # binary layer
set_Ŵ!(model); set_σʷ!(model)    # weighted layer
nothing

# output


```

```jldoctest DECM_variance; output = false
# metric: the total weight of the network (a function of the weighted adjacency matrix)
X = W -> sum(W)
# delta-method standard deviation under the null model
σₓ(model, X, layer=:weighted)

# output

89.30268766693133
```

```jldoctest DECM_variance; output = false
# metric: the sum of the squared weights (not a constrained quantity)
X = W -> sum(W .^ 2)
# expected value, standard deviation, observed value and z-score
X_expected = X(model.Ŵ)
X_std = σₓ(model, X, layer=:weighted)
X_observed = X(Graphs.weights(G))
z_X = (X_observed - X_expected) / X_std

# output

1.4593199696859283
```

!!! note "No within-dyad covariance"

    The network is directed, so ``g_{ij}`` and ``g_{ji}`` are **distinct, independent** random variables: `σₓ` carries no within-dyad covariance term (unlike the undirected [UECM](UECM.md), where ``w_{ij} \equiv w_{ji}`` and the cross-term doubles the variance of a full-matrix metric).

!!! warning "Memory footprint"

    `Ĝ`/`σˣ` and `Ŵ`/`σʷ` (with their `set_Ĝ!`/`set_σ!`/`set_Ŵ!`/`set_σʷ!` variants) materialize dense ``N \times N`` matrices, and `σₓ` requires them. This is ``O(N^2)`` memory, intended for small networks; for large networks, prefer sampling to estimate variances (see [Performance and scalability](../performance.md)).

## Sampling the ensemble
```julia
# generate 10 random directed weighted instances of the ensemble
rand(model, 10)
```

## Model comparison
```julia
# compute the AIC
AIC(model)
```

_References_

```@raw html
<ul>
<li>
<a id="1">[1]</a> 
Vallarano, Nicolò and Bruno, Matteo and Marchese, Emiliano and Trapani, Giuseppe and Saracco, Fabio and Cimini, Giulio and Zanon, Mario and Squartini, Tiziano. <!--  author(s) --> 
<em>"Fast and scalable likelihood maximization for Exponential Random Graph Models with local constraints"</em> <!--  title --> 
Scientific Reports 11, 2021. <!--  publisher(s) --> 
<a href="https://doi.org/10.1038/s41598-021-93830-4">https://doi.org/10.1038/s41598-021-93830-4</a>
</li>
<li>
<a id="2">[2]</a> 
Parisi, Federica and Squartini, Tiziano and Garlaschelli, Diego. <!--  author(s) --> 
<em>"A faster horse on a safer trail: generalized inference for the efficient reconstruction of weighted networks"</em> <!--  title --> 
New Journal of Physics 22, 2020. <!--  publisher(s) --> 
<a href="https://doi.org/10.1088/1367-2630/ab74a7">https://doi.org/10.1088/1367-2630/ab74a7</a>
</li>
</ul>
```
