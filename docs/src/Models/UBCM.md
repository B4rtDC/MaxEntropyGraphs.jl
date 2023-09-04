## Description
The Undirected Binary Configuration Model is a maximum-entropy null model for undirected networks. It is based on the idea of fixing the degree sequence of the network, i.e., the number of edges incident to each node, and then randomly rewiring the edges while preserving the degree sequence. The model assumes that the edges are unweighted and that the network is simple, i.e., it has no self-loops or multiple edges between the same pair of nodes. 

| Description                   | Formula |
| --------------------------    | --------------------------------------------------------------------------------- |
| Constraints                   | ```math k_i(A^{*}) = \sum_{j=1}^{N} a^{*}_{ij}  \text{  } (\forall i) ``` |
| Hamiltonian                   | ```math H(A, \Theta) = \sum_{i=1}^{N} \Theta_i k_{i}(A) ``` |
| Factorized graph probability  | ```math P(A \| \Theta) = \prod_{i=1}^{N}\prod_{j=1, j<i}^{N} p_{ij}^{a_{ij}} (1 - p_{ij})^{1-a_{ij}}  \text{ where } p_{ij} = \frac{e^{-\theta_i - \theta_j}}{1+e^{-\theta_i - \theta_j}}``` |
| Log-likelihood                | ```math \mathcal{L}(\Theta) = -\sum_{i=1}^{N}\theta_i k_i(A^{*}) - \sum_{i=1}^{N} \sum_{j=1, j<i}^{N} \ln \left( 1+e^{-\theta_i - \theta_j} \right) ```|
| $\langle a_{ij} \rangle$      | ```math p_{ij} = \frac{e^{-\theta_i - \theta_j}}{1+e^{-\theta_i - \theta_j}}``` |
| $\langle a_{ij}^{2} \rangle$  | ```math \langle a_{ij} \rangle``` |
| $\langle a_{ij}a_{ts} \rangle$| ```math \langle a_{ij} \rangle \langle a_{ts} \rangle``` |

## Creation

## Obtaining the parameters

## Sampling the ensemble



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
</ul>
```

