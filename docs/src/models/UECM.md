# UECM
## Model description
The Undirected Enhanced Configuration Model (UECM) is a maximum-entropy null model for **weighted, undirected** networks. It simultaneously constrains the degree sequence (the number of neighbours of each node) and the strength sequence (the total weight incident to each node). The weights are required to take non-negative **integer** values; the model then sits "halfway" between a Bernoulli and a geometric distribution for each pair of nodes [[1](#1),[2](#2)].

We define the parameter vector as ``\theta = [\alpha ; \beta]``, where ``\alpha`` and ``\beta`` denote the parameters associated with the degree and the strength sequence respectively. In the code the exponentiated parameters ``x_i = e^{-\alpha_i}`` and ``y_i = e^{-\beta_i}`` are used.

| Description                    | Formula |
| --------------------------     | :-------------------------------------------------------------------------------- |
| Constraints                    | `` \forall i: \begin{cases} k_{i}(A^{*}) = \sum_{j \ne i} a^{*}_{ij} \\ s_{i}(W^{*}) = \sum_{j \ne i} w^{*}_{ij} \end{cases} ``|
| Hamiltonian                    | `` H(W, \alpha, \beta) = \sum_{i=1}^{N} \left[ \alpha_i k_{i}(A) +  \beta_i s_{i}(W) \right]`` |
| $\langle a_{ij} \rangle$       | `` p_{ij} = \frac{e^{-\alpha_i - \alpha_j - \beta_i - \beta_j}}{1 - e^{-\beta_i - \beta_j} + e^{-\alpha_i - \alpha_j - \beta_i - \beta_j}}`` |
| Log-likelihood                 | `` \mathcal{L}(\alpha, \beta) = -\sum_{i=1}^{N} \left[ \alpha_i k_{i}(A^{*}) +  \beta_i s_{i}(W^{*}) \right] - \sum_{i=1}^{N} \sum_{j=1, j < i}^{N} \ln \left[ 1 + e^{-\alpha_i - \alpha_j} \left( \frac{e^{-\beta_i - \beta_j}}{1 - e^{-\beta_i - \beta_j}} \right) \right] ``|
| $\langle w_{ij} \rangle$       | `` \frac{p_{ij}}{1 - e^{-\beta_i - \beta_j}}`` |
| $\sigma^{*}(X)$                | ``\sqrt{\sum_{i,j} \left( \sigma^{*}[a_{ij}] \frac{\partial X}{\partial a_{ij}}  \right)^{2}_{A = \langle A^{*} \rangle} + \dots }`` |
| $\sigma^{*}[a_{ij}]$           | ``\sqrt{p_{ij} (1 - p_{ij})} ``   |

## Creation
```julia
using Graphs, SimpleWeightedGraphs
using MaxEntropyGraphs

# a weighted, undirected network with integer weights
# (here: the symmetrised rhesus macaques grooming network)
G = SimpleWeightedGraph(rhesus_macaques())

# instantiate a UECM model
model = UECM(G)
```

## Obtaining the parameters
```julia
# solve using the default settings (BFGS)
solve_model!(model)
```

!!! note

    The classical Picard fixed-point recipe for the UECM is unusable [[1]](#1), and measurably so: from a cold start it converged on **none** of 150 random weighted networks. Since v0.8.0 `:fixedpoint` is not that recipe. It is block coordinate ascent, which solves each block exactly rather than freezing a factor, and is unconditionally well posed because both constraint functions are monotone in their own parameter. On 100 well-posed networks it converges 98 times against `BFGS`'s 86, with a median constraint residual of ``3\cdot10^{-9}`` against ``2.5\cdot10^{-7}``, and it is far cheaper: on a 250-vertex weighted network, ``9\cdot10^{-9}`` in 0.28 s against ``1.4\cdot10^{-5}`` in 57 s.

    The default is still `BFGS`, because it degrades more gracefully on the inputs the fixed point cannot solve. When a **runaway** constraint is present (``k = 0``, ``k = N-1`` or ``s_i = k_i``) the maximum-likelihood optimum sits at an infinite parameter and no solver can settle at a finite tolerance; there `BFGS` reaches 38 of 50 against the fixed point's 8. On a well-posed network, prefer `method = :fixedpoint`.

    `Newton` is also available (and typically the fastest, cf. [[1]](#1)) but it is the one method that gets an unbounded problem rather than a box, so it needs the default `initial = :strengths`. Because the likelihood is only defined on the feasible region ``e^{-\beta_i - \beta_j} < 1``, the UECM uses a `BackTracking` line search that keeps the iterates inside that region.

## Expected adjacency and weights
```julia
# expected (binary) adjacency matrix; row sums reproduce the degree sequence
Ĝ(model)

# expected weighted adjacency matrix; row sums reproduce the strength sequence
Ŵ(model)
```

## Expectation and variance
Under the UECM each pair of nodes carries a **two-layer** random variable: the adjacency entry ``a_{ij}`` follows a Bernoulli distribution, while the weight ``w_{ij}`` follows a Bernoulli–geometric mixture (no weight without a link, a geometrically distributed weight conditional on a link). Writing ``x = x_i x_j`` and ``y = y_i y_j``, the first two moments are:

| Layer                | ``\langle g_{ij} \rangle`` | ``\text{Var}(g_{ij})`` | ``\text{Cov}(g_{ij}, g_{ji})`` |
| -------------------- | :--- | :--- | :--- |
| binary ``(g = a)``   | ``p_{ij} = \frac{x y}{1 - y + x y}`` | ``p_{ij}(1 - p_{ij})`` | ``= \text{Var}(a_{ij})`` (same variable) |
| weighted ``(g = w)`` | ``\frac{p_{ij}}{1 - y}`` | ``\frac{p_{ij}(1 + y - p_{ij})}{(1 - y)^{2}}`` | ``= \text{Var}(w_{ij})`` (same variable) |

The two layers of a pair are correlated as well: ``\text{Cov}(a_{ij}, w_{ij}) = \langle w_{ij} \rangle (1 - p_{ij})``.

!!! note "Variance propagation is per-layer"

    `σₓ` propagates the uncertainty of **one layer at a time** (`layer=:binary`, the default, or `layer=:weighted`); the cross-layer covariance above is documented for reference but not propagated. For a metric that mixes both layers, estimate its variance by sampling the ensemble (`rand(model, n)`).

The workflow mirrors that of the binary models: precompute the expected matrices and the entry-wise standard deviations, then propagate them through a metric `X` with `σₓ` (the delta method):

```jldoctest UECM_variance; output = false
using Graphs, SimpleWeightedGraphs
using MaxEntropyGraphs

G = SimpleWeightedGraph(rhesus_macaques())
model = UECM(G)
solve_model!(model)

# precompute the expected values and standard deviations of both layers
set_Ĝ!(model); set_σ!(model)     # binary layer
set_Ŵ!(model); set_σʷ!(model)    # weighted layer
nothing

# output


```

```jldoctest UECM_variance; output = false
# metric: the total weight of the network (a function of the weighted adjacency matrix)
X = W -> sum(W) / 2
# delta-method standard deviation under the null model
# rounded: the fitted parameters are converged to ~1e-8, so later digits are not determined
round(σₓ(model, X, layer=:weighted), digits=4)

# output

102.7652
```

```jldoctest UECM_variance; output = false
# metric: the sum of the squared weights (not a constrained quantity)
X = W -> sum(W .^ 2) / 2
# expected value, standard deviation, observed value and z-score
X_expected = X(model.Ŵ)
X_std = σₓ(model, X, layer=:weighted)
X_observed = X(Graphs.weights(G))
# rounded: the fitted parameters are converged to ~1e-8, so later digits are not determined
z_X = round((X_observed - X_expected) / X_std, digits=4)

# output

1.6936
```

!!! note "Within-dyad covariance"

    The network is undirected, so ``g_{ij}`` and ``g_{ji}`` denote the **same** random variable: `σₓ` includes the corresponding within-dyad covariance term. The result is therefore consistent between the one-triangle and full-matrix conventions for writing a metric, e.g. ``\sigma_X\left(W \mapsto \sum_{i<j} w_{ij}\right) = \sigma_X\left(W \mapsto \sum_{i,j} w_{ij}\right)/2``.

!!! warning "Memory footprint"

    `Ĝ`/`σˣ` and `Ŵ`/`σʷ` (with their `set_Ĝ!`/`set_σ!`/`set_Ŵ!`/`set_σʷ!` variants) materialize dense ``N \times N`` matrices, and `σₓ` requires them. This is ``O(N^2)`` memory, intended for small networks; for large networks, prefer sampling to estimate variances (see [Performance and scalability](../performance.md)).

## Sampling the ensemble
```julia
# generate 10 random weighted instances of the ensemble
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
Squartini, Tiziano and Garlaschelli, Diego. <!--  author(s) --> 
<em>"Maximum-Entropy Networks: Pattern Detection, Network Reconstruction and Graph Combinatorics"</em> <!--  title --> 
Springer-Verlag GmbH; 1st ed. 2017 edition (25 Dec. 2017). <!--  publisher(s) --> 
<a href="https://link.springer.com/book/10.1007/978-3-319-69438-2">https://link.springer.com/book/10.1007/978-3-319-69438-2</a>
</li>
</ul>
```
