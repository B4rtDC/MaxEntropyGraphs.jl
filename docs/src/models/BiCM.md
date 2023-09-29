# BiCM
## Model description
An undirected bipartite network can be described by its biadjacency matrix ``B = \left{ b_{i\alpha} \right}_{i,\alpha}`` of size ``N \times M`` whose generic entry ``b_{iα}`` is 1 if node ``i`` belonging to layer ⊥ is linked to node α belonging to layer ⊤ and 0 otherwise.
The two sets of nodes (sometimes referred to a layers) are defined as as ⊥ and ⊤. 
The Bipartite Configuration Model (BiCM) is a maximum-entropy null model for undirected bipartite networks. 
It is based on the idea of fixing the degree sequences for each set of nodes (layers) of the network. 
The model assumes that the edges are unweighted and that the network is simple, i.e., it has no self-loops or multiple edges between the same pair of nodes [[1](#1)]. 


[!NOTE]  
For the computation we use the bi-adjacency matrix, whereas the current implementation of the BiCM uses a `::Graphs.SimpleGraph` to construct the models and assesses its bipartiteness using the functionality available in the `Graphs.jl` package.

We define the parameter vector as ``\theta = [\gamma ; \beta]``, where ``\gamma`` and ``\beta`` denote the parameters associated with the ⊥ and ⊤ layer respectively. To speed up the computation of the likelihood maximising parameters, 
we use the reduced version of the model where we consider the unique values the degrees in each layer [[2](#2)].

| Description                   | Formula |
| --------------------------    | :-------------------------------------------------------------------------------- |
| Constraints                   | `` \begin{cases} \forall i \in \bot:  k_{i}(A^{*}) = \sum_{\alpha \in \top} b^{*}_{i\alpha} \\  \forall \alpha \in \top:  d_{\alpha}(A^{*}) = \sum_{i \in \bot} b^{*}_{i\alpha} \end{cases} ``|
| Hamiltonian                   | `` H(A, \Theta) = H(A, \gamma, \beta) = \sum_{i \in \bot} \gamma_i k_{i}(A) +  \sum_{\alpha \in \top} \beta_\alpha d_{\alpha}(A)`` |
| Factorized graph probability  | `` P(A \| \Theta) = \prod_{i=1}^{N}\prod_{j=1}^{M} p_{i\alpha}^{b_{i\alpha}} (1 - p_{\alpha})^{1-b_{i\alpha}}``  |
| $\langle p_{i\alpha} \rangle$ | `` p_{i\alpha} = \frac{e^{-\gamma_i - \beta_{\alpha}}}{1+e^{-\gamma_i - \beta_{\alpha}}}`` |
| Log-likelihood                | `` \mathcal{L}(\Theta) = -\sum_{i \in \bot} \gamma_i k_{i}(A) -  \sum_{\alpha \in \top} \beta_{\alpha} d_{\alpha}(A) - \sum_{i \in \bot}  \sum_{\alpha \in \top} \ln \left( 1 + e^{-\gamma_i - \beta_{\alpha}} \right) ``|
| $\langle p_{i\alpha}^{2} \rangle$  | `` `` |
| $\langle p_{i\alpha}a_{t\kappa} \rangle$| `` `` |
| $\sigma^{*}(X)$               | `` `` |
| $\sigma^{*}[p_{i\alpha}]$          | `` ``   |



## Creation
```julia
using Graphs
using MaxEntropyGraphs

# define the network
G =

# instantiate a UBCM model
model = BiCM(G)
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

```julia
# Compute the number of occurences of M13
M13(model)
```


_References_

```@raw html
<ul>
<li>
<a id="1">[1]</a> 
M. Baltakiene, K. Baltakys, D. Cardamone, F. Parisi, T. Radicioni, M. Torricelli, J. A. van Lidth de Jeude, F. Saracco <!--  author(s) --> 
<em>"Maximum entropy approach to link prediction in bipartite networks"</em> <!--  title --> 
 arXiv preprint arXiv:1805.04307 (2018). <!--  publisher(s) --> 
<a href="https://arxiv.org/abs/1805.04307">https://arxiv.org/abs/1805.04307</a>
</li>
<li>
<a id="2">[2]</a> 
Vallarano, N., Bruno, M., Marchese, E. et al. <!--  author(s) --> 
<em>"Fast and scalable likelihood maximization for Exponential Random Graph Models with local constraints"</em> <!--  title --> 
Sci Rep 11, 15227 (2021) <!--  publisher(s) --> 
<a href="https://doi.org/10.1038/s41598-021-93830-4">https://doi.org/10.1038/s41598-021-93830-4</a>
</li>
</ul>
```

Fast and scalable likelihood maximization for Exponential Random Graph Models with local constraints

Vallarano, N., Bruno, M., Marchese, E. et al. Fast and scalable likelihood maximization for Exponential Random Graph Models with local constraints. Sci Rep 11, 15227 (2021). https://doi.org/10.1038/s41598-021-93830-4

