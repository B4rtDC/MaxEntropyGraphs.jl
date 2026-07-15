# CReM
## Model description
The Conditional Reconstruction Method (CReM) is a maximum-entropy null model for **weighted, undirected** networks with **continuous, positive** weights. It is a **two-step** model [[1](#1),[2](#2)]:

1. a **binary layer** fixes the topology. The marginal probability of an edge, ``f_{ij} = \langle a_{ij} \rangle``, is supplied by a prior binary model. Here an internally-solved [`UBCM`](@ref) on the degree sequence, so ``f_{ij} = \frac{x_i x_j}{1 + x_i x_j}`` with ``x_i = e^{-\alpha_i}``;
2. a **weighted layer** fixes the strengths. Conditional on an edge existing, the weight is exponentially distributed with rate ``\theta_i + \theta_j``. The parameters ``\theta`` (one per node) are obtained by maximising the conditional log-likelihood.

Because the binary structure is taken as given, the CReM is easier to solve than the joint [`UECM`](@ref): both the fixed-point and Newton recipes converge well [[1]](#1), and the fixed point is the default.

| Description                    | Formula |
| --------------------------     | :-------------------------------------------------------------------------------- |
| Constraints                    | `` \forall i: \begin{cases} k_{i}(A^{*}) = \sum_{j \ne i} a^{*}_{ij} \\ s_{i}(W^{*}) = \sum_{j \ne i} w^{*}_{ij} \end{cases} ``|
| Hamiltonian                    | `` H(W, \theta) = \sum_{i=1}^{N} \theta_i s_{i}(W)`` |
| $\langle a_{ij} \rangle$       | `` f_{ij} = \frac{x_i x_j}{1 + x_i x_j} \quad (\text{binary/UBCM layer})`` |
| $q(w_{ij} \mid a_{ij}=1)$      | `` (\theta_i + \theta_j)\, e^{-(\theta_i + \theta_j) w} , \quad w > 0`` |
| Log-likelihood                 | `` \mathcal{L}(\theta) = -\sum_{i=1}^{N} \theta_i s_{i}(W^{*}) + \sum_{i=1}^{N} \sum_{j=1, j < i}^{N} f_{ij} \ln (\theta_i + \theta_j) ``|
| $\langle w_{ij} \rangle$       | `` \frac{f_{ij}}{\theta_i + \theta_j}`` |
| $\sigma^{*}(X)$                | ``\sqrt{\sum_{i,j} \left( \sigma^{*}[a_{ij}] \frac{\partial X}{\partial a_{ij}}  \right)^{2}_{A = \langle A^{*} \rangle} + \dots }`` |
| $\sigma^{*}[a_{ij}]$           | ``\sqrt{f_{ij} (1 - f_{ij})} ``   |

## Creation
```julia
using Graphs, SimpleWeightedGraphs
using MaxEntropyGraphs

# a weighted, undirected network (here: the symmetrised rhesus macaques grooming network; the CReM
# treats the weights as continuous positive quantities)
G = SimpleWeightedGraph(rhesus_macaques())

# instantiate a CReM model
model = CReM(G)
```

## Obtaining the parameters
```julia
# solve using the default settings (two-step: internal UBCM, then the fixed-point weighted layer)
solve_model!(model)
```

!!! note

    The weighted parameters ``\theta`` are the **direct** exponential rates (they appear as ``\ln(\theta_i + \theta_j)`` in the log-likelihood), so a solution requires ``\theta_i + \theta_j > 0``. Every initial guess is therefore strictly positive. The default solver is the (stable) `fixedpoint`; `BFGS` and `Newton` are also available and use a `BackTracking` line search to stay inside the feasible region.

## Expected adjacency and weights
```julia
# expected (binary) adjacency matrix; row sums reproduce the degree sequence
Ĝ(model)

# expected weighted adjacency matrix; row sums reproduce the strength sequence
Ŵ(model)
```

## Expectation and variance
Under the CReM each pair of nodes carries a **two-layer** random variable: the adjacency entry ``a_{ij}`` follows a Bernoulli distribution (the UBCM layer), while the weight ``w_{ij}`` follows a Bernoulli–exponential mixture (no weight without a link, an exponentially distributed weight conditional on a link). The first two moments are:

| Layer                | ``\langle g_{ij} \rangle`` | ``\text{Var}(g_{ij})`` | ``\text{Cov}(g_{ij}, g_{ji})`` |
| -------------------- | :--- | :--- | :--- |
| binary ``(g = a)``   | ``f_{ij} = \frac{x_i x_j}{1 + x_i x_j}`` | ``f_{ij}(1 - f_{ij})`` | ``= \text{Var}(a_{ij})`` (same variable) |
| weighted ``(g = w)`` | ``\frac{f_{ij}}{\theta_i + \theta_j}`` | ``\frac{f_{ij}(2 - f_{ij})}{(\theta_i + \theta_j)^{2}}`` | ``= \text{Var}(w_{ij})`` (same variable) |

The two layers of a pair are correlated as well: ``\text{Cov}(a_{ij}, w_{ij}) = \langle w_{ij} \rangle (1 - f_{ij})``.

!!! note "Variance propagation is per-layer"

    `σₓ` propagates the uncertainty of **one layer at a time** (`layer=:binary`, the default, or `layer=:weighted`); the cross-layer covariance above is documented for reference but not propagated. For a metric that mixes both layers, estimate its variance by sampling the ensemble (`rand(model, n)`). Because the weighted layer is conditional on the binary one, `σʷ`/`set_σʷ!` require both the binary and the conditional parameters; `solve_model!` computes both.

The workflow mirrors that of the binary models: precompute the expected matrices and the entry-wise standard deviations, then propagate them through a metric `X` with `σₓ` (the delta method):

```jldoctest CReM_variance; output = false
using Graphs, SimpleWeightedGraphs
using MaxEntropyGraphs

G = SimpleWeightedGraph(rhesus_macaques())
model = CReM(G)
solve_model!(model)

# precompute the expected values and standard deviations of both layers
set_Ĝ!(model); set_σ!(model)     # binary layer
set_Ŵ!(model); set_σʷ!(model)    # weighted layer
nothing

# output


```

```jldoctest CReM_variance; output = false
# metric: the total weight of the network (a function of the weighted adjacency matrix)
X = W -> sum(W) / 2
# delta-method standard deviation under the null model
σₓ(model, X, layer=:weighted)

# output

108.64282821918191
```

```jldoctest CReM_variance; output = false
# metric: the sum of the squared weights (not a constrained quantity)
X = W -> sum(W .^ 2) / 2
# expected value, standard deviation, observed value and z-score
X_expected = X(model.Ŵ)
X_std = σₓ(model, X, layer=:weighted)
X_observed = X(Graphs.weights(G))
z_X = (X_observed - X_expected) / X_std

# output

1.6671230437867095
```

!!! note "Within-dyad covariance"

    The network is undirected, so ``g_{ij}`` and ``g_{ji}`` denote the **same** random variable: `σₓ` includes the corresponding within-dyad covariance term. The result is therefore consistent between the one-triangle and full-matrix conventions for writing a metric, e.g. ``\sigma_X\left(W \mapsto \sum_{i<j} w_{ij}\right) = \sigma_X\left(W \mapsto \sum_{i,j} w_{ij}\right)/2``.

!!! warning "Memory footprint"

    `Ĝ`/`σˣ` and `Ŵ`/`σʷ` (with their `set_Ĝ!`/`set_σ!`/`set_Ŵ!`/`set_σʷ!` variants) materialize dense ``N \times N`` matrices, and `σₓ` requires them. This is ``O(N^2)`` memory, intended for small networks; for large networks, prefer sampling to estimate variances (see [Performance and scalability](../performance.md)).

## Sampling the ensemble
```julia
# generate 10 random weighted instances of the ensemble (continuous, exponential weights)
rand(model, 10)
```

## Model comparison
```julia
# compute the AIC (the conditional CReM has N parameters, one θ per node)
AIC(model)
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
Vallarano, Nicolò and Bruno, Matteo and Marchese, Emiliano and Trapani, Giuseppe and Saracco, Fabio and Cimini, Giulio and Zanon, Mario and Squartini, Tiziano. <!--  author(s) --> 
<em>"Fast and scalable likelihood maximization for Exponential Random Graph Models with local constraints"</em> <!--  title --> 
Scientific Reports 11, 2021. <!--  publisher(s) --> 
<a href="https://doi.org/10.1038/s41598-021-93830-4">https://doi.org/10.1038/s41598-021-93830-4</a>
</li>
</ul>
```
