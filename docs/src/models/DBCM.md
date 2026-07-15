# DBCM
## Model description
The Directed Binary Configuration Model (DBCM) is a maximum-entropy null model for directed networks. It is based on the idea of fixing the in- and out-degree sequence of the network, i.e., the number of incoming and outgoing edges incident to each node, and then randomly rewiring the edges while preserving the degree sequence. The model assumes that the edges are unweighted and that the network is simple, i.e., it has no self-loops or multiple edges between the same pair of nodes [[1](#1),[2](#2)]. 

We define the parameter vector as ``\theta = [\alpha ; \beta]``, where ``\alpha`` and ``\beta`` denote the parameters associated with the out- and in-degree respectively.

| Description                   | Formula |
| --------------------------    | :-------------------------------------------------------------------------------- |
| Constraints                   | `` \forall i: \begin{cases} k_{i, out}(A^{*}) = \sum_{j=1}^{N} a^{*}_{ij} \\ k_{i, in}(A^{*}) = \sum_{j=1}^{N} a^{*}_{ji} \end{cases} ``|
| Hamiltonian                   | `` H(A, \Theta) = H(A, \alpha, \beta) = \sum_{i=1}^{N} \alpha_i k_{i,out}(A) +  \beta_i k_{i,in}(A)`` |
| Factorized graph probability  | `` P(A \| \Theta) = \prod_{i=1}^{N}\prod_{j=1, j \ne i}^{N} p_{ij}^{a_{ij}} (1 - p_{ij})^{1-a_{ij}}``  |
| $\langle a_{ij} \rangle$      | `` p_{ij} = \frac{e^{-\alpha_i - \beta_j}}{1+e^{-\alpha_i - \beta_j}}`` |
| Log-likelihood                | `` \mathcal{L}(\Theta) = -\sum_{i=1}^{N} \left[ \alpha_i k_{i,out}(A^{*}) +  \beta_i k_{i,in}(A^{*}) \right] - \sum_{i=1}^{N} \sum_{j=1, j\ne i}^{N} \ln \left( 1+e^{-\alpha_i - \beta_j} \right) ``|
| $\langle a_{ij}^{2} \rangle$  | `` \langle a_{ij} \rangle`` |
| $\langle a_{ij}a_{ts} \rangle$| `` \langle a_{ij} \rangle \langle a_{ts} \rangle`` |
| $\sigma^{*}(X)$               | ``\sqrt{\sum_{i,j} \left( \sigma^{*}[a_{ij}] \frac{\partial X}{\partial a_{ij}}  \right)^{2}_{A = \langle A^{*} \rangle} + \dots }`` |
| $\sigma^{*}[a_{ij}]$          | ``\frac{\sqrt{e^{-\alpha_i - \beta_j}}}{1+e^{-\alpha_i - \beta_j}} ``   |





## Creation
```julia
using Graphs
using MaxEntropyGraphs

# define the network
G = SimpleDiGraph(rhesus_macaques())

# instantiate a DBCM model
model = DBCM(G)
```

## Obtaining the parameters
```julia
# solve using default settings
solve_model!(model)
```

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

## Counting network motifs
The count of a specific network motif can be computed by using `M{motif_number}`. The motif numbers match the patterns shown on the image below. So for example if you want to compute the number of reciprocated triangles, you would use `M13(model)`.

```@example
HTML("""<object type="image/png" data=$(joinpath(Main.buildpath, "../assets/directed_motifs.png"))  alt="Motif image not found" style="width: 100%;" ></object> """) # hide
```
[source](https://snap-stanford.github.io/cs224w-notes/preliminaries/motifs-and-structral-roles_lecture)


```julia
# Compute the number of occurences of M13
M13(model)
```

## Expectation and variance
Under the DBCM every ordered node pair ``(i,j)``, ``i \ne j``, carries an **independent Bernoulli** random variable with success probability ``p_{ij}``. In particular the two directions of a dyad are independent: ``\mathrm{Cov}(a_{ij}, a_{ji}) = 0`` (contrast this with the [`RBCM`](@ref MaxEntropyGraphs.RBCM), which constrains the reciprocity structure).

| Quantity                         | Formula |
| ------------------------------   | :------------------------------------------------ |
| ``\langle a_{ij} \rangle``       | ``p_{ij} = \frac{x_i y_j}{1 + x_i y_j}, \quad x_i = e^{-\alpha_i}, \; y_i = e^{-\beta_i}`` |
| ``\mathrm{Var}(a_{ij})``         | ``p_{ij}(1 - p_{ij})`` |
| ``\mathrm{Cov}(a_{ij}, a_{ji})`` | ``0`` (independent directions) |

Because all matrix entries are independent, the delta-method error propagation contains no covariance cross-terms: ``σ^{2}[X] = \sum_{i \ne j} \left( σ[a_{ij}] \frac{\partial X}{\partial a_{ij}} \right)^{2}``. The workflow is the standard one (cf. [the analytical metrics page](../exact.md)): store the expected adjacency matrix and the entry-wise standard deviations, then propagate the gradient of ``X`` through `σₓ` (the autodiff backend is selectable via `gradient_method ∈ {:ReverseDiff, :ForwardDiff, :Zygote}`):

```jldoctest DBCM_variance
using MaxEntropyGraphs

# define the network and solve the model
G = MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques())
model = DBCM(G)
solve_model!(model)
# expected adjacency matrix ⟨A⟩ and entry-wise standard deviations σ[a_ij]
set_Ĝ!(model)
set_σ!(model)
nothing

# output


```
```jldoctest DBCM_variance
# metric: the number of fully reciprocated triangles (motif 13)
A = Matrix(MaxEntropyGraphs.Graphs.adjacency_matrix(G))
# z-score: (observed - expected) / σ
z_M13 = (M13(A) - M13(model)) / σₓ(model, M13)

# output

2.0015141124711335
```

The reciprocated triangle count sits about two standard deviations above the DBCM expectation on this highly reciprocal network (``r \approx 0.76``). The DBCM does not constrain reciprocity, so reciprocity-driven patterns show up as deviations. The [`RBCM`](@ref MaxEntropyGraphs.RBCM) absorbs them (same metric, same network: ``z_{M13} \approx -0.28``).

!!! warning "Memory footprint"
    `Ĝ`/`set_Ĝ!` and `σˣ`/`set_σ!` materialize dense ``N \times N`` matrices, and `σₓ` requires both, so this analysis uses ``O(N^2)`` memory. For large networks, prefer the sampling route to estimate variances (see [Performance and scalability](../performance.md)).

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

