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


```@raw latex
\begin{table}[h]
\centering
\begin{tabular}{|l|l|}
\hline
\textbf{Description} & \textbf{Formula} \\ \hline
Constraints & $k_i(A^{*}) = \sum_{j=1}^{N} a^{*}_{ij}  \text{  } (\forall i)$ \\ \hline
Hamiltonian & $H(A, \Theta) = \sum_{i=1}^{N} \Theta_i k_{i}(A)$ \\ \hline
Factorized graph probability & $P(A \| \Theta) = \prod_{i=1}^{N}\prod_{j=1, j<i}^{N} p_{ij}^{a_{ij}} (1 - p_{ij})^{1-a_{ij}}  \text{ where } p_{ij} = \frac{e^{-\theta_i - \theta_j}}{1+e^{-\theta_i - \theta_j}}$ \\ \hline
Log-likelihood & $\mathcal{L}(\Theta) = -\sum_{i=1}^{N}\theta_i k_i(A^{*}) - \sum_{i=1}^{N} \sum_{j=1, j<i}^{N} \ln \left( 1+e^{-\theta_i - \theta_j} \right)$ \\ \hline
$\langle a_{ij} \rangle$ & $p_{ij} = \frac{e^{-\theta_i - \theta_j}}{1+e^{-\theta_i - \theta_j}}$ \\ \hline
$\langle a_{ij}^{2} \rangle$ & $\langle a_{ij} \rangle$ \\ \hline
$\langle a_{ij}a_{ts} \rangle$ & $\langle a_{ij} \rangle \langle a_{ts} \rangle$ \\ \hline
\end{tabular}
\caption{My table caption}
\label{my-label}
\end{table}

```
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

