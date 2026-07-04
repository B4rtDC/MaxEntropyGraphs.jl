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

!!! note

    The variance-based quantities (`σˣ`, `set_σ!`, `σₓ`) currently describe the **binary** (adjacency) layer only, for which ``a_{ij}`` follows a Bernoulli distribution with success probability ``p_{ij}``. The variance of the weights is not implemented.

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

    Contrary to the purely binary models, the fixed-point recipe is very unstable for the UECM and should not be used [[1]](#1). The default solver is therefore `BFGS`; `Newton` is also available (and typically the fastest, cf. [[1]](#1)) but is more sensitive to the initial guess. Because the likelihood is only defined on the feasible region ``e^{-\beta_i - \beta_j} < 1``, the UECM uses a `BackTracking` line search that keeps the iterates inside that region.

## Expected adjacency and weights
```julia
# expected (binary) adjacency matrix; row sums reproduce the degree sequence
Ĝ(model)

# expected weighted adjacency matrix; row sums reproduce the strength sequence
MaxEntropyGraphs.Ŵ(model)
```

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
