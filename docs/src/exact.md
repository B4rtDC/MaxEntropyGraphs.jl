# Analytical
The maximum likelihood method can be used to compute the expected value and the standard deviation of any metric that is a (differentiable) function of the adjacency matrix — analytically, without sampling the ensemble. Depending on the underlying model, some details change, but the principle remains. This formalism allows us to compute z-scores and assess which topological properties are consistent with their randomized value within a statistical error, and which deviate significantly from the null model expectation. In the latter case, the observed property cannot be traced back to the constraints, and therefore requires additional explanations or generating mechanisms besides those required in order to explain the constraints themselves [[1](@ref exact_references), [2](@ref exact_references)].

Some topological properties are available in the package by default (e.g. [`ANND`](@ref) and different network motifs), but you can define additional metrics as well. This allows you to obtain both the expected value and the standard deviation of any metric in a standardized way. In the expression of the variance of a topological property ``X``, we find the partial derivatives ``\frac{\partial X}{\partial g_{ij}}``. We leverage Julia's autodiff capabilities to compute these terms: the `gradient_method` keyword argument of [`σₓ`](@ref MaxEntropyGraphs.σₓ(::UBCM, ::Function)) selects the backend (`:ReverseDiff` (default), `:ForwardDiff` or `:Zygote`). Depending on the size of the problem, different autodiff techniques can give different performance results, so you might want to experiment a bit for your own use case. If desired, you can always compute the gradient of a specific metric by hand and implement it yourself as well.

!!! warning "Memory usage"
    This approach requires the complete expected adjacency matrix, its entry-wise standard deviations and the (dense) gradient of the metric, i.e. ``O(N^2)`` memory for a graph with ``N`` nodes. It is therefore not suited for the analysis of very large graphs. For those cases, consider the [sampling-based approach](simulated.md) instead.

## Expected value
Let ``\langle G \rangle`` denote the matrix of expected entries ``\langle g_{ij} \rangle``, which is known in closed form for every model in the package (cf. [the moments table](@ref exact_moments) below). Truncating the multidimensional Taylor expansion of ``X`` around ``\langle G \rangle`` after the first-order term gives the *plug-in* estimate (Eq. (A.3) in [[1](@ref exact_references)]):

```math
\langle X \rangle  \approx  X \left( \left< G \right> \right)
```

This approximation ignores the second- and higher-order terms of the expansion. In the package, `set_Ĝ!(model)` computes and stores ``\langle G \rangle``, after which `X(model.Ĝ)` evaluates the expected value of the metric. For the weighted layer of the two-layer models, `set_Ŵ!(model)` and `model.Ŵ` play the same role.

## Variance
Propagating the fluctuations of the matrix entries through the same first-order expansion (the *delta method*, Eq. (B.16) in [[1](@ref exact_references)]) yields the variance of a topological property ``X``:

```math
\sigma ^{2} \left[ X \right] = \sum_{i,j} \sum_{t,s} \sigma \left[g_{ij}, g_{ts} \right] \left(  \frac{\partial X}{\partial g_{ij}} \frac{\partial X}{\partial g_{ts}}  \right)_{G = \left< G \right>}
```

where

```math
\sigma \left[ g_{ij}, g_{ts} \right] = \left< g_{ij}g_{ts}\right> - \left< g_{ij}\right>\left< g_{ts}\right>
```

For all models in this package, entries belonging to *different* node pairs are statistically independent, so every covariance between distinct dyads vanishes. Only the within-dyad terms survive, and the double sum collapses to a single sum over ordered pairs ``(i,j)``:

```math
\sigma ^{2} \left[ X \right] = \sum_{i,j} \left[ \left( \sigma \left[ g_{ij} \right] \frac{\partial X}{\partial g_{ij}} \right)^{2} + \sigma \left[ g_{ij}, g_{ji} \right] \frac{\partial X}{\partial g_{ij}} \frac{\partial X}{\partial g_{ji}} \right]_{G = \left< G \right>}
```

This is exactly the quantity computed by `σₓ(model, X)`: `set_σ!(model)` stores the entry-wise standard deviations ``\sigma[g_{ij}]``, autodiff supplies the gradient of ``X`` evaluated at ``\langle G \rangle``, and the model supplies the within-dyad covariance ``\sigma[g_{ij}, g_{ji}]``. Using the appropriate model-specific expressions (cf. [the moments table](@ref exact_moments) below), a highly reliable estimate for the variance of the metric can be obtained.

### Undirected networks and the within-dyad term
For an undirected model the matrix entries ``g_{ij}`` and ``g_{ji}`` denote the *same* random variable, so the within-dyad covariance equals the variance: ``\sigma[g_{ij}, g_{ji}] = \sigma^{2}[g_{ij}]``. The package includes this cross-term in the error propagation, which makes the result independent of the matrix convention used to write the metric — whether it reads one triangle of the matrix or the full symmetric matrix, the fluctuations are propagated consistently. For example, counting every dyad twice simply doubles both the expected value and the standard deviation:

```math
\sigma_{X}\left( A \mapsto \sum_{i,j} a_{ij} \right) = 2\, \sigma_{X}\left( A \mapsto \sum_{i<j} a_{ij} \right)
```

For the other model families the within-dyad term takes a different form:
* the directed models with independent entries ([`DBCM`](@ref MaxEntropyGraphs.DBCM), [`DCReM`](@ref MaxEntropyGraphs.DCReM)) and the bipartite [`BiCM`](@ref MaxEntropyGraphs.BiCM) have ``\sigma[g_{ij}, g_{ji}] = 0`` (for the BiCM the biadjacency matrix is rectangular and every entry is a distinct, independent variable), so the formula reduces to the familiar sum of squares;
* the reciprocal models ([`RBCM`](@ref MaxEntropyGraphs.RBCM), [`CRWCM`](@ref MaxEntropyGraphs.CRWCM)) draw both directions of a dyad *jointly*, so ``g_{ij}`` and ``g_{ji}`` are correlated through the reciprocity constraint; `σₓ` includes the corresponding dyadic covariance.

## [Model-specific moments](@id exact_moments)
The table below lists, for each model (and each layer of the two-layer models), the distribution of a matrix entry ``g_{ij}``, its expected value, its variance and the within-dyad covariance used by `σₓ`. For the undirected models "``= \sigma^2[g_{ij}]``" indicates that the covariance equals the variance because ``g_{ij}`` and ``g_{ji}`` are the same variable.

| Model (layer) | distribution of ``g_{ij}`` | ``\left< g_{ij} \right>`` | ``\sigma^{2}[g_{ij}]`` | ``\sigma[g_{ij}, g_{ji}]`` |
|:--- |:--- |:--- |:--- |:--- |
| UBCM | Bernoulli | ``p_{ij}=\frac{x_ix_j}{1+x_ix_j}`` | ``p_{ij}(1-p_{ij})`` | ``= \sigma^2[g_{ij}]`` |
| DBCM | Bernoulli | ``p_{ij}=\frac{x_iy_j}{1+x_iy_j}`` | ``p_{ij}(1-p_{ij})`` | ``0`` |
| RBCM | 4-state dyad | ``\left< a_{ij} \right>=\frac{x_iy_j+z_iz_j}{D_{ij}}`` | ``\left< a_{ij} \right>(1-\left< a_{ij} \right>)`` | ``p^{\leftrightarrow}_{ij}-\left< a_{ij} \right>\left< a_{ji} \right>`` |
| BiCM (biadjacency) | Bernoulli | ``p_{i\alpha}=\frac{x_iy_\alpha}{1+x_iy_\alpha}`` | ``p_{i\alpha}(1-p_{i\alpha})`` | n/a (independent entries) |
| UECM (binary) | Bernoulli | ``p_{ij}=\frac{xy}{1-y+xy}`` | ``p_{ij}(1-p_{ij})`` | ``= \sigma^2[g_{ij}]`` |
| UECM (weighted) | Bernoulli–geometric | ``\left< w_{ij} \right>=\frac{p_{ij}}{1-y}`` | ``\frac{p_{ij}(1+y-p_{ij})}{(1-y)^{2}}`` | ``= \sigma^2[g_{ij}]`` |
| CReM (binary) | Bernoulli | ``f_{ij}=\frac{x_ix_j}{1+x_ix_j}`` | ``f_{ij}(1-f_{ij})`` | ``= \sigma^2[g_{ij}]`` |
| CReM (weighted) | Bernoulli–exponential | ``\left< w_{ij} \right>=\frac{f_{ij}}{\theta_i+\theta_j}`` | ``\frac{f_{ij}(2-f_{ij})}{(\theta_i+\theta_j)^{2}}`` | ``= \sigma^2[g_{ij}]`` |
| DCReM (binary) | Bernoulli | ``f_{ij}=\frac{x_iy_j}{1+x_iy_j}`` | ``f_{ij}(1-f_{ij})`` | ``0`` |
| DCReM (weighted) | Bernoulli–exponential | ``\left< w_{ij} \right>=\frac{f_{ij}}{\theta^{o}_i+\theta^{i}_j}`` | ``\frac{f_{ij}(2-f_{ij})}{(\theta^{o}_i+\theta^{i}_j)^{2}}`` | ``0`` |
| CRWCM (binary) | 4-state dyad (as RBCM) | ``\left< a_{ij} \right>=\frac{x_iy_j+z_iz_j}{D_{ij}}`` | ``\left< a_{ij} \right>(1-\left< a_{ij} \right>)`` | ``p^{\leftrightarrow}_{ij}-\left< a_{ij} \right>\left< a_{ji} \right>`` |
| CRWCM (weighted) | 3-channel exponential mixture | ``\left< w_{ij} \right>=\frac{f^{\rightarrow}_{ij}}{r^{\rightarrow}_{ij}}+\frac{f^{\leftrightarrow}_{ij}}{r^{\leftrightarrow}_{ij}}`` | ``\frac{2f^{\rightarrow}_{ij}}{(r^{\rightarrow}_{ij})^{2}}+\frac{2f^{\leftrightarrow}_{ij}}{(r^{\leftrightarrow}_{ij})^{2}}-\left< w_{ij} \right>^{2}`` | ``\frac{f^{\leftrightarrow}_{ij}}{r^{\leftrightarrow}_{ij}\, r^{\leftrightarrow}_{ji}}-\left< w_{ij} \right>\left< w_{ji} \right>`` |

Notation (see the individual model pages for the full definitions):
* ``x_i``, ``y_i``, ``z_i`` denote the exponentiated Lagrange multipliers of the respective models; for the UECM ``x = x_ix_j`` and ``y = y_iy_j``;
* for the reciprocal models ``D_{ij} = 1 + x_iy_j + x_jy_i + z_iz_j`` and ``p^{\leftrightarrow}_{ij} = \frac{z_iz_j}{D_{ij}}`` is the probability of a reciprocated dyad;
* for the CReM family ``\theta`` denotes the conditional exponential rates (``\theta^{o}``/``\theta^{i}`` for the out/in rates of the DCReM);
* for the CRWCM ``f^{\rightarrow}_{ij}`` (``f^{\leftrightarrow}_{ij}``) is the probability of a single (reciprocated) link and ``r^{\rightarrow}_{ij} = \theta^{\rightarrow}_i + \theta^{\leftarrow}_j``, ``r^{\leftrightarrow}_{ij} = \theta^{\leftrightarrow,o}_i + \theta^{\leftrightarrow,i}_j`` are the associated exponential rates.

### Layers of the two-layer models
For the two-layer models (UECM, CReM, DCReM, CRWCM), `σₓ` accepts a `layer` keyword argument:
* `σₓ(model, X)` or `σₓ(model, X, layer=:binary)` treats `X` as a function of the *adjacency* matrix and requires `set_Ĝ!(model)` and `set_σ!(model)`;
* `σₓ(model, X, layer=:weighted)` treats `X` as a function of the *weight* matrix and requires `set_Ŵ!(model)` and `set_σʷ!(model)`.

!!! note "Cross-layer covariance"
    Within a node pair the binary entry and the weight are correlated: ``\sigma[a_{ij}, w_{ij}] = \left< w_{ij} \right>(1 - p_{ij})`` (there is no weight without a link). This cross-layer covariance is documented here for completeness, but it is *not* propagated by `σₓ`, which works per layer. For a metric that mixes binary and weighted quantities, use the [sampling-based approach](simulated.md) instead.

## Examples
Some examples using built-in functions of the package are listed below:
* [Assortativity in the UBCM](@ref Assortativity_analytical)
* [Motif significance in the DBCM](@ref Motif_analytical)
* [Significance of V-motifs and projection in the BiCM](@ref BiCM_Vmotifs_analytical)

### [Assortativity in the UBCM](@id Assortativity_analytical)
Let us consider the UBCM applied to the Zachary Karate Club network. We want to analyze if the assortativity of each node (measured by its ANND) is statistically significant from what one would expect under the null model.

First, we define the network and the associated UBCM model.
```jldoctest UBCM_z_demo
using Graphs
using MaxEntropyGraphs

# define the network
G = MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate)
# generate a UBCM model from the karate club network
model = UBCM(G);
# compute the maximum likelihood parameters
solve_model!(model);
# compute and set the expected adjacency matrix
set_Ĝ!(model);
# compute and set the standard deviation of the adjacency matrix
set_σ!(model);
nothing

# output


```

Now we can define our specific metric. Before computing the z-score for all nodes, we illustrate the process for a single node. We use `X` as variable name for our metric. Defining methods for `X` in such a way that it can work
with both an `AbstractArray` and an `AbstractGraph` is recommended, but not necessary.
```jldoctest UBCM_z_demo
# We consider the ANND of node 1 as our metric
node_id = 1
X = A -> ANND(A, node_id, check_dimensions=false, check_directed=false);
# Expected value under the null model
X_expected = X(model.Ĝ)
# Expected standard deviation under the null model
X_std = σₓ(model, X)
# Observed value (using the underlying network)
X_observed = X(model.G)
# compute z-score
z_X = (X_observed - X_expected) / X_std

# output

-1.6289646436441225
```

In the same way, we can compute the z-score for every node:
```jldoctest UBCM_z_demo
# Observed value
ANND_obs = [ANND(G, i) for i in vertices(G)]
# Expected values
ANND_exp = [ANND(model.Ĝ, i) for i in vertices(G)]
# Standard deviation
ANND_std = [σₓ(model, A -> ANND(A, i, check_dimensions=false, check_directed=false)) for i in vertices(G)]
# Z-score
Z_ANND = (ANND_obs - ANND_exp) ./ ANND_std

# output

34-element Vector{Float64}:
 -1.6289646436441225
 -1.053660062962129
 -0.1729140902358404
 -0.2745008132148558
 -0.5514938959670105
 -1.0938255625612319
 -1.0938255625612325
  0.5584124741043061
  1.5745559084120984
  1.0315044121643198
  ⋮
 -1.5807614460636297
  0.224928239079853
 -0.06117678964527059
  0.5921367152514553
  0.04208808764632542
  0.7649422286874977
  0.4598931201169171
 -1.2585480308297787
 -2.2346220475416696
```

Because the UBCM is undirected, `σₓ` includes the within-dyad covariance term (cf. [Undirected networks and the within-dyad term](@ref)), so the result does not depend on whether a metric reads one triangle of the matrix or the full symmetric matrix. We can verify the factor-two relation from above explicitly:
```jldoctest UBCM_z_demo
# metric written on the full (symmetric) matrix: every dyad counted twice
X_full = A -> sum(A)
# the same metric written on the strict upper triangle: every dyad counted once
X_triangle = A -> sum(A[i,j] for j in axes(A,2) for i in 1:j-1)
# the standard deviations scale with the same double-counting factor as the metric itself
σₓ(model, X_full) / σₓ(model, X_triangle)

# output

2.0
```

### [Motif significance in the DBCM](@id Motif_analytical)
Let us consider the DBCM applied to the Chesapeake Bay foodweb. We want to analyze if any of the different network motifs is statistically significant of what one would expect under the null model.

First, we define the network and the associated DBCM model.
```jldoctest DBCM_z_demo
using Graphs
using MaxEntropyGraphs

# define the network
G = chesapeakebay()
# extract its adjacency matrix
A = adjacency_matrix(G)
# generate a DBCM model from the Chesapeake Bay foodweb
model = DBCM(G);
# compute the maximum likelihood parameters
solve_model!(model);
# compute and set the expected adjacency matrix
set_Ĝ!(model);
# compute and set the standard deviation of the adjacency matrix
set_σ!(model);
nothing

# output


```

We want to know the values of `M1`, ..., `M13`. These are network-wide measures.
```jldoctest DBCM_z_demo
# compute the observed motif counts
motifs_observed = [@eval begin $(f)(A) end for f in MaxEntropyGraphs.directed_graph_motif_function_names];
# Expected value under the null model
motifs_expected = [@eval begin $(f)(model) end for f in MaxEntropyGraphs.directed_graph_motif_function_names];
# Expected standard deviation under the null model
motifs_std = [@eval begin  σₓ(model, $(f), gradient_method=:ForwardDiff) end for f in MaxEntropyGraphs.directed_graph_motif_function_names];
# compute the z-score
motifs_z = (motifs_observed .- motifs_expected) ./ motifs_std

# output

13-element Vector{Float64}:
 -0.0777982129523697
  0.4517818796564166
 -0.5539006800257769
 -0.3602659406627635
 -2.0695338110813677
  0.3690013899689748
  1.0292421736597916
 -1.6028568152284417
 -3.168481282851789
  1.3101076515056782
  1.6115264819280963
  4.862910567915925
 13.309574362333343
```

!!! note "Rare motifs and the delta method"
    The delta method is a first-order approximation: it works well for metrics whose expected value is far from the boundary of its domain, but it *underestimates* the standard deviation of rare, sign-definite counts. In this example the expected number of fully reciprocated triangles (`M13`) under the DBCM is only ``\approx 0.23``, so its distribution is strongly skewed and the analytical z-score (``\approx 13.3``) vastly overstates the significance: a sampling-based z-score (cf. [the simulation page](simulated.md)) yields ``\approx 3.2`` for the same motif. Whenever the expected count of a motif is of order one or smaller, prefer the sampling-based approach.

### [Significance of V-motifs and projection in the BiCM](@id BiCM_Vmotifs_analytical)
Let us consider the BiCM applied to the [corporate club membership network](http://konect.cc/networks/brunson_club-membership/). This bipartite network is composed of 25 persons and 15 social organizations. An edge between a person and a social organization shows that the person is a member. We want to obtain the projection of this network on both the person and the social organization layer. We also want to determine if any of these connections in the projected networks are statistically significant under the BiCM null model. This is done by evaluating if the number of organizations that two users have in common is more than what one would expect under the null model.

First, we define the network and the associated BiCM model.
```jldoctest BiCM_projection_demo
using MaxEntropyGraphs
using Graphs

# define the network
G = corporateclub();
# project the model on its layers
G_persons, G_organizations = project(G, layer=:bottom), project(G, layer=:top);
# generate a BiCM from the corporate club network
model = BiCM(G);
# compute the maximum likelihood parameters
solve_model!(model);
# determine the validated projected networks
G_persons_validated, G_organizations_validated = project(model, layer=:bottom), project(model, layer=:top);
# check the number of (validated) edges in both networks
(ne(G_persons),ne(G_persons_validated)),(ne(G_organizations), ne(G_organizations_validated))

# output

((259, 0), (66, 0))
```
We can observe that no links were found to be statistically significant, using the default settings, i.e. a significance threshold of ``\alpha=0.05`` and the [Benjamini Hochberg](https://en.wikipedia.org/wiki/False_discovery_rate) procedure from MultipleTesting to correct for multiple testing. These can be modified if required.

```jldoctest BiCM_projection_demo
# using a very high threshold value for significance
project(model, layer=:bottom, α=0.9)

# output

{25, 255} undirected simple Int64 graph
```

Beyond the pairwise V-motifs underlying the projection, the analytical machinery also covers the higher-order ``V_n``-motifs (``n`` nodes of one layer sharing a common neighbor in the other layer): [`Vn_motifs`](@ref), [`Vn_sigma`](@ref) and [`Vn_zscore`](@ref) provide the observed counts, expected values, standard deviations and z-scores under the BiCM, with both an exact and a delta-method backend (see [the BiCM model page](models/BiCM.md) for an example).

## [References](@id exact_references)

```@raw html
<ul>
<li>
<a id="1">[1]</a> 
Squartini, Tiziano and Garlaschelli, Diego. <!--  author(s) --> 
<em>"Maximum-Entropy Networks: Pattern Detection, Network Reconstruction and Graph Combinatorics"</em> <!--  title --> 
Springer-Verlag GmbH; 1st ed. 2017 edition (25 Dec. 2017). <!--  publisher(s) --> 
<a href="https://link.springer.com/book/10.1007/978-3-319-69438-2">https://link.springer.com/book/10.1007/978-3-319-69438-2</a>
</li>
</ul>

<ul>
<li>
<a id="2">[2]</a> 
Squartini, Tiziano and Garlaschelli, Diego. <!--  author(s) --> 
<em>"Analytical maximum-likelihood method to detect patterns in real networks"</em> <!--  title --> 
New Journal of Physics 13, 083001, 2011. <!--  publisher(s) --> 
<a href="https://doi.org/10.1088/1367-2630/13/8/083001">https://doi.org/10.1088/1367-2630/13/8/083001</a>
</li>
</ul>
```
