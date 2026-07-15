# UBCM
## Model description
The Undirected Binary Configuration Model is a maximum-entropy null model for undirected networks. It is based on the idea of fixing the degree sequence of the network, i.e., the number of edges incident to each node, and then randomly rewiring the edges while preserving the degree sequence. The model assumes that the edges are unweighted and that the network is simple, i.e., it has no self-loops or multiple edges between the same pair of nodes [[1](#1),[2](#2)]. 

| Description                   | Formula |
| --------------------------    | :-------------------------------------------------------------------------------- |
| Constraints                   | `` k_i(A^{*}) = \sum_{j=1}^{N} a^{*}_{ij}  \text{  } (\forall i)`` |
| Hamiltonian                   | `` H(A, \Theta) = \sum_{i=1}^{N} \Theta_i k_{i}(A) `` |
| Factorized graph probability  | `` P(A \| \Theta) = \prod_{i=1}^{N}\prod_{j=1, j<i}^{N} p_{ij}^{a_{ij}} (1 - p_{ij})^{1-a_{ij}}``  |
| $\langle a_{ij} \rangle$      | `` p_{ij} = \frac{e^{-\theta_i - \theta_j}}{1+e^{-\theta_i - \theta_j}}`` |
| Log-likelihood                | `` \mathcal{L}(\Theta) = -\sum_{i=1}^{N}\theta_i k_i(A^{*}) - \sum_{i=1}^{N} \sum_{j=1, j<i}^{N} \ln \left( 1+e^{-\theta_i - \theta_j} \right) ``|
| $\langle a_{ij}^{2} \rangle$  | `` \langle a_{ij} \rangle`` |
| $\langle a_{ij}a_{ts} \rangle$| `` \langle a_{ij} \rangle \langle a_{ts} \rangle`` |
| $\sigma^{*}(X)$               | `` \sqrt{\sum_{i,j} \left( \sigma^{*}[a_{ij}] \frac{\partial X}{\partial a_{ij}}  \right)^{2}_{A = \langle A^{*} \rangle} + \dots } ``  |
| $\sigma^{*}[a_{ij}]$          | ``\frac{\sqrt{e^{-\theta_i - \theta_j}}}{1+e^{-\theta_i - \theta_j}} ``   |



## Creation
```julia
using Graphs
using MaxEntropyGraphs

# define the network
G = smallgraph(:karate)

# instantiate a UBCM model
model = UBCM(G)
```

## Obtaining the parameters
```julia
# solve using default settings
solve_model!(model)
```

## Expectation and variance
Under the UBCM every unordered node pair ``(i,j)`` carries an **independent Bernoulli** random variable with success probability ``p_{ij}``. Because the network is undirected, ``a_{ij}`` and ``a_{ji}`` denote the *same* random variable.

| Quantity                         | Formula |
| ------------------------------   | :------------------------------------------------ |
| ``\langle a_{ij} \rangle``       | ``p_{ij} = \frac{x_i x_j}{1 + x_i x_j}, \quad x_i = e^{-\theta_i}`` |
| ``\mathrm{Var}(a_{ij})``         | ``p_{ij}(1 - p_{ij})`` |
| ``\mathrm{Cov}(a_{ij}, a_{ji})`` | ``p_{ij}(1 - p_{ij})`` (identical variables: ``a_{ij} \equiv a_{ji}``) |

The delta-method standard deviation of any metric ``X(A)`` follows the standard workflow (cf. [the analytical metrics page](../exact.md)): store the expected adjacency matrix and the entry-wise standard deviations, then propagate the gradient of ``X`` through `σₓ` (the autodiff backend is selectable via `gradient_method ∈ {:ReverseDiff, :ForwardDiff, :Zygote}`):

```jldoctest UBCM_variance
using MaxEntropyGraphs

# define the network and solve the model
G = MaxEntropyGraphs.Graphs.smallgraph(:karate)
model = UBCM(G)
solve_model!(model)
# expected adjacency matrix ⟨A⟩ and entry-wise standard deviations σ[a_ij]
set_Ĝ!(model)
set_σ!(model)
nothing

# output


```
```jldoctest UBCM_variance
# metric: the number of triangles, as a function of the adjacency matrix
X = A -> triangles(A, check_dimensions=false, check_directed=false)
# z-score: (observed - expected) / σ
z = (triangles(G) - X(model.Ĝ)) / σₓ(model, X)

# output

-0.6160580827583924
```

!!! note "Within-dyad covariance"
    In an undirected model the matrix entries ``a_{ij}`` and ``a_{ji}`` are perfectly correlated (they are the same variable), so `σₓ` includes the within-dyad covariance cross-term ``\mathrm{Cov}(a_{ij}, a_{ji}) \frac{\partial X}{\partial a_{ij}}\frac{\partial X}{\partial a_{ji}}`` in the error propagation. As a result the value of `σₓ` does not depend on whether your metric reads one triangle of the matrix or the full symmetric matrix:

```jldoctest UBCM_variance
# the number of links, in the full-matrix and in the upper-triangle convention
L_full = A -> sum(A) / 2
L_triu = A -> sum(A[i,j] for i in 1:34 for j in i+1:34)
σₓ(model, L_full) ≈ σₓ(model, L_triu)

# output

true
```

!!! warning "Memory footprint"
    `Ĝ`/`set_Ĝ!` and `σˣ`/`set_σ!` materialize dense ``N \times N`` matrices, and `σₓ` requires both, so this analysis uses ``O(N^2)`` memory. For large networks, prefer the sampling route to estimate variances (see [Performance, scalability & GPU](../GPU.md)).

## Sampling the ensemble
```julia
# generate 10 random instance of the ensemble
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
Squartini, Tiziano and Garlaschelli, Diego. <!--  author(s) --> 
<em>"Maximum-Entropy Networks: Pattern Detection, Network Reconstruction and Graph Combinatorics"</em> <!--  title --> 
Springer-Verlag GmbH; 1st ed. 2017 edition (25 Dec. 2017). <!--  publisher(s) --> 
<a href="https://link.springer.com/book/10.1007/978-3-319-69438-2">https://link.springer.com/book/10.1007/978-3-319-69438-2</a>
</li>
<li>
<a id="2">[2]</a> 
Squartini, Tiziano and Garlaschelli, Diego. <!--  author(s) --> 
<em>"Analytical maximum-likelihood method to detect patterns in real networks"</em> <!--  title --> 
2011 New J. Phys. 13 083001. <!--  publisher(s) --> 
<a href="https://iopscience.iop.org/article/10.1088/1367-2630/13/8/083001">https://iopscience.iop.org/article/10.1088/1367-2630/13/8/083001</a>
</li>
</ul>
```

