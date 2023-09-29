# BiCM
## Model description
An undirected bipartite network can be described by its biadjacency matrix ``B = \left{ b_{i\alpha} \right}_{i,\alpha}`` of size ``N \times M`` whose generic entry ``b_{iα}`` is 1 if node ``i`` belonging to layer ⊥ is linked to node α belonging to layer ⊤ and 0 otherwise.
The two sets of nodes (sometimes referred to a layers) are defined as as ⊥ and ⊤. 
The Bipartite Configuration Model (BiCM) is a maximum-entropy null model for undirected bipartite networks. 
It is based on the idea of fixing the degree sequences for each set of nodes (layers) of the network. 
The model assumes that the edges are unweighted and that the network is simple, i.e., it has no self-loops or multiple edges between the same pair of nodes [[1](#1),[2](#2)]. 


[!NOTE]  
For the computation we use the bi-adjacency matrix, whereas the current implementation of the BiCM uses a `::Graphs.SimpleGraph` to construct the models and assesses its bipartiteness using the functionality available in the `Graphs.jl` package.

We define the parameter vector as ``\theta = [\gamma ; \beta]``, where ``\gamma`` and ``\beta`` denote the parameters associated with the ⊥ and ⊤ layer respectively. To speed up the computation of the likelihood maximising parameters, we use the reduced version of the model where we consider the unique values the degrees in each layer [3](#3).

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


# complete this

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

![logo](./assets/logo.jpeg)
![directed network motifs naming convention](./docs/src/assets/directed_motifs.png)
![directed network motifs naming convention](./src/assets/directed_motifs.png)
![directed network motifs naming convention](./assets/directed_motifs.png)
![directed network motifs naming convention](./directed_motifs.png)
![logo](/assets/logo.jpeg)
![directed network motifs naming convention](/docs/src/assets/directed_motifs.png)
![directed network motifs naming convention](/src/assets/directed_motifs.png)
![directed network motifs naming convention](/assets/directed_motifs.png)
![directed network motifs naming convention](/directed_motifs.png)


![logo](./assets/logo.jpeg)
![directed network motifs naming convention](./docs/src/assets/directed_motifs.jpg)
![directed network motifs naming convention](./src/assets/directed_motifs.jpg)
![directed network motifs naming convention](./assets/directed_motifs.jpg)
![directed network motifs naming convention](./directed_motifs.jpg)
![logo](/assets/logo.jpeg)
![directed network motifs naming convention](/docs/src/assets/directed_motifs.jpg)
![directed network motifs naming convention](/src/assets/directed_motifs.jpg)
![directed network motifs naming convention](/assets/directed_motifs.jpg)
![directed network motifs naming convention](/directed_motifs.jpg)

![the logo](./assets/logo.png)
![the logo](./../assets/logo.png)
![the logo](./../../assets/logo.png)
![the logo](/assets/logo.png)
![the logo](/../assets/logo.png)
![the logo](/../../assets/logo.png)

```@example
HTML("""<object type="image/png+xml" data=$(joinpath(Main.buildpath, "..","assets","directed_motifs.png"))></object>""") # hide
```
```@example
HTML("""<object type="image/png+xml" data=$(joinpath(Main.buildpath, "..", "src", "assets","directed_motifs.png"))></object>""") # hide
```
```@example
HTML("""<object type="image/png+xml" data=$(joinpath(Main.buildpath, "..","docs","src", "assets","directed_motifs.png"))></object>""") # hide
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
<li>
<a id="3">[3]</a> 
Vallarano, N., Bruno, M., Marchese, E. et al. <!--  author(s) --> 
<em>"Fast and scalable likelihood maximization for Exponential Random Graph Models with local constraints"</em> <!--  title --> 
Sci Rep 11, 15227 (2021) <!--  publisher(s) --> 
<a href="https://doi.org/10.1038/s41598-021-93830-4">https://doi.org/10.1038/s41598-021-93830-4</a>
</li>
</ul>
```

Fast and scalable likelihood maximization for Exponential Random Graph Models with local constraints

Vallarano, N., Bruno, M., Marchese, E. et al. Fast and scalable likelihood maximization for Exponential Random Graph Models with local constraints. Sci Rep 11, 15227 (2021). https://doi.org/10.1038/s41598-021-93830-4