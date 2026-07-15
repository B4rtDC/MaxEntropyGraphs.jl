# BiCM
## Model description
An undirected bipartite network can be described by its biadjacency matrix ``B`` of size ``N \times M`` whose generic entry ``b_{iα}`` is 1 if node ``i`` belonging to layer ⊥ is linked to node α belonging to layer ⊤ and 0 otherwise.
The two sets of nodes (sometimes referred to a layers) are defined as as ⊥ and ⊤. 

The Bipartite Configuration Model (BiCM) is a maximum-entropy null model for undirected bipartite networks. 
It is based on the idea of fixing the degree sequences for each set of nodes (layers) of the network. 
The model assumes that the edges are unweighted and that the network is simple, i.e., it has no self-loops or multiple edges between the same pair of nodes [[1](#1)]. 


!!! note
    For the computation we use the bi-adjacency matrix, whereas the current implementation of the BiCM uses a `::Graphs.SimpleGraph` to construct the models and assesses its bipartiteness using the functionality available in the `Graphs.jl` package.

The parameter vector is defined as ``\theta = [\gamma ; \beta]``, where ``\gamma`` and ``\beta`` denote the parameters associated with the ⊥ and ⊤ layer respectively. To speed up the computation of the likelihood maximizing parameters, 
we use the reduced version of the model where we consider the unique values the degrees in each layer [[2](#2)].

| Description                   | Formula |
| --------------------------    | :-------------------------------------------------------------------------------- |
| Constraints                   | `` \begin{cases} \forall i \in \bot:  k_{i}(A^{*}) = \sum_{\alpha \in \top} b^{*}_{i\alpha} \\  \forall \alpha \in \top:  d_{\alpha}(A^{*}) = \sum_{i \in \bot} b^{*}_{i\alpha} \end{cases} ``|
| Hamiltonian                   | `` H(A, \Theta) = H(A, \gamma, \beta) = \sum_{i \in \bot} \gamma_i k_{i}(A) +  \sum_{\alpha \in \top} \beta_\alpha d_{\alpha}(A)`` |
| Factorized graph probability  | `` P(A \| \Theta) = \prod_{i=1}^{N}\prod_{j=1}^{M} p_{i\alpha}^{b_{i\alpha}} (1 - p_{\alpha})^{1-b_{i\alpha}}``  |
| $\langle p_{i\alpha} \rangle$ | `` p_{i\alpha} = \frac{e^{-\gamma_i - \beta_{\alpha}}}{1+e^{-\gamma_i - \beta_{\alpha}}}`` |
| Log-likelihood                | `` \mathcal{L}(\Theta) = -\sum_{i \in \bot} \gamma_i k_{i}(A) -  \sum_{\alpha \in \top} \beta_{\alpha} d_{\alpha}(A) - \sum_{i \in \bot}  \sum_{\alpha \in \top} \ln \left( 1 + e^{-\gamma_i - \beta_{\alpha}} \right) ``|




## Creation
```julia
using Graphs
using MaxEntropyGraphs

# define the network
G = corporateclub()

# instantiate a BiCM model
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

## Expectation and variance
Under the BiCM every entry ``b_{i\alpha}`` of the biadjacency matrix is an independent Bernoulli random variable with success probability ``p_{i\alpha}``. There is no within-dyad covariance to account for (the matrix is rectangular; every entry is a distinct, independent variable):

| Quantity                          | Formula |
| ------------------------------    | :------------------------------------------------ |
| ``\langle b_{i\alpha} \rangle``   | ``p_{i\alpha} = \frac{x_i y_\alpha}{1 + x_i y_\alpha}, \quad x_i = e^{-\gamma_i}, \; y_\alpha = e^{-\beta_\alpha}`` |
| ``\mathrm{Var}(b_{i\alpha})``     | ``p_{i\alpha}(1 - p_{i\alpha})`` |
| ``\mathrm{Cov}(b_{i\alpha}, b_{j\beta})`` | ``0`` for ``(i,\alpha) \ne (j,\beta)`` (independent entries) |

The σ machinery of the BiCM operates on the ``n_{\bot} \times n_{\top}`` biadjacency matrix: `Ĝ(model)`/`set_Ĝ!(model)` and `σˣ(model)`/`set_σ!(model)` return ``n_{\bot} \times n_{\top}`` matrices, and any metric passed to `σₓ` must be a function of the biadjacency matrix (not of the square adjacency matrix of the underlying graph). Since all entries are independent, the delta-method error propagation contains no covariance cross-terms: ``σ^{2}[X] = \sum_{i,\alpha} \left( σ[b_{i\alpha}] \frac{\partial X}{\partial b_{i\alpha}} \right)^{2}`` (the autodiff backend is selectable via `gradient_method ∈ {:ReverseDiff, :ForwardDiff, :Zygote}`):

```jldoctest BiCM_variance
using MaxEntropyGraphs

# define the network and solve the model
G = corporateclub()
model = BiCM(G)
solve_model!(model)
# expected biadjacency matrix ⟨B⟩ and entry-wise standard deviations σ[b_iα]
set_Ĝ!(model)
set_σ!(model)
size(model.Ĝ), size(model.σ)

# output

((25, 15), (25, 15))
```
```jldoctest BiCM_variance
# metric: the total V-motif count among the ⊥ nodes, as a function of the biadjacency matrix
N_V = B -> sum(u -> u * (u - 1) / 2, sum(B, dims=1))
# the delta-method σₓ coincides with the closed-form delta σ of the Vn machinery (next section)
σₓ(model, N_V) ≈ Vn_sigma(model, 2, layer=:bottom, method=:delta)

# output

true
```

!!! warning "Memory footprint"
    `Ĝ`/`set_Ĝ!` and `σˣ`/`set_σ!` materialize dense ``n_{\bot} \times n_{\top}`` matrices, and `σₓ` requires both, so this analysis uses ``O(n_{\bot} n_{\top})`` memory. For large networks, prefer the sampling route to estimate variances (see [Performance and scalability](../performance.md)).

## Counting network motifs

```julia
# Compute the number of expected occurrences of a V-motif between two users
V_motifs(model, 1, 2, layer=:bottom)
# Compute the number of occurrences of a V-motif in the original graph
V_motifs(model.G, model.⊥nodes[1], model.⊥nodes[2])
```

## Motif-family significance (Vn / Λn)
The pairwise V-motif generalizes to co-occurrences of arbitrary order: the total number of `Vn`-motifs between the nodes of a layer is

```math
N_{Vn} = \sum_{\alpha} \binom{u_\alpha}{n}
```

where the sum runs over the nodes ``\alpha`` of the *opposite* layer and ``u_\alpha`` is their degree (for `layer=:bottom` these are the column sums of the biadjacency matrix). Each term counts the groups of ``n`` layer nodes sharing the common neighbour ``\alpha``. Following Saracco et al. (2015) [[4]](#4), the family for the bottom (top) layer is also known as the `Vn` (`Λn`) family; ``n = 2`` recovers the total V-motif count.

[`Vn_motifs`](@ref), [`Vn_sigma`](@ref) and [`Vn_zscore`](@ref) provide the observed count, the expected value, the standard deviation and the z-score under the BiCM, with two methods:

- `method=:exact` (default): under the BiCM the random degree ``U_\alpha`` follows a Poisson-binomial distribution, and the ``U_\alpha`` of distinct nodes are independent, so the exact mean and variance of ``N_{Vn}`` follow by direct convolution. Works for any ``n \ge 2``.
- `method=:delta`: the closed forms of [[4]](#4), i.e., a Taylor expansion of the expectation around the observed degrees (``n \in \{2, 3, 4\}``, exact for ``n = 2``) and a first-order delta-method σ (any ``n``). Accurate when the opposite-layer degrees are large compared to ``n``; for sparse layers the delta σ underestimates the exact one (measured ratios down to ``\approx 0.15`` when the opposite-layer degrees are close to ``n``), which inflates ``|z|`` — prefer `:exact` there.

```jldoctest BiCM_Vn
using MaxEntropyGraphs

model = BiCM(corporateclub())
solve_model!(model)

# z-score of the Λ₃ count (triples of organizations sharing a member), exact Poisson-binomial
Vn_zscore(model, 3, layer=:top)

# output

-1.2505607656327316
```
```jldoctest BiCM_Vn
# the closed-form delta method underestimates σ, hence overestimates |z|
Vn_zscore(model, 3, layer=:top, method=:delta)

# output

-1.6219521450601682
```

!!! note "Sign-definite z-scores"
    The observed count ``N_{Vn}^{obs}`` is fully determined by the constrained degree sequences (it is computed from the degrees stored in the model; no graph needed), and under the BiCM ``\langle N_{Vn} \rangle \ge N_{Vn}^{obs}`` [[4]](#4). The z-scores are therefore always ``\le 0`` and the associated significance tests are one-sided: they can only detect a *lack* of aggregate co-occurrence, never an excess.

These aggregate statistics complement the per-pair validation used by the projection machinery (see below): for a single pair ``(i,j)``, ``V_{ij} = \sum_p b_{ip} b_{jp}`` is a sum of independent Bernoulli variables with success probabilities ``q_p = p_{ip} p_{jp}``, and `project(model; distribution=:PoissonBinomial)` evaluates its p-value from that exact Poisson-binomial law. Note the same delta-method caveat at the per-pair level: the first-order delta variance of ``V_{ij}`` underestimates the exact ``\sum_p q_p(1 - q_p)`` by exactly ``\sum_p p_{ip} p_{jp} (1 - p_{ip})(1 - p_{jp})``.

## Projecting the bipartite network
```julia
# Raw projection of the network
project(model.G, layer=:bottom)
# Obtaining the statistically significant links
project(model, layer=:bottom)
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
<li>
<a id="3">[3]</a> 
F. Saracco, M. J. Straka, R. Di Clemente, A. Gabrielli, G. Caldarelli, and T. Squartini <!--  author(s) --> 
<em>"Inferring monopartite projections of bipartite networks: an entropy-based approach"</em> <!--  title --> 
 New J. Phys. 19, 053022 (2017) <!--  publisher(s) --> 
<a href="http://stacks.iop.org/1367-2630/19/i=5/a=053022">http://stacks.iop.org/1367-2630/19/i=5/a=053022</a>
</li>
<li>
<a id="4">[4]</a> 
F. Saracco, R. Di Clemente, A. Gabrielli, and T. Squartini <!--  author(s) --> 
<em>"Randomizing bipartite networks: the case of the World Trade Web"</em> <!--  title --> 
 Sci Rep 5, 10595 (2015) <!--  publisher(s) --> 
<a href="https://doi.org/10.1038/srep10595">https://doi.org/10.1038/srep10595</a>
</li>
</ul>

```
