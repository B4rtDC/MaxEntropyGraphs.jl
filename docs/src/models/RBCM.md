# RBCM
## Model description
The Reciprocal Binary Configuration Model (RBCM) is a maximum-entropy null model for **directed, binary** networks that preserves the **reciprocity structure** of the network. Instead of the plain out- and in-degrees of the [`DBCM`](@ref MaxEntropyGraphs.DBCM), it constrains three quantities per node [[1](#1),[2](#2),[3](#3)]: the *non-reciprocated out-degree* ``k^{→}_i`` (links `i→j` without a link back), the *non-reciprocated in-degree* ``k^{←}_i`` and the *reciprocated degree* ``k^{↔}_i`` (links in both directions). The model was introduced as the reciprocal configuration model (RCM) in [[1]](#1) and is called RBCM in [[3]](#3) and in the NuMeTriS package [[4]](#4).

We define the parameter vector as ``\theta = [\alpha; \beta; \gamma]`` with fitnesses ``x_i = e^{-\alpha_i}``, ``y_i = e^{-\beta_i}``, ``z_i = e^{-\gamma_i}``. Every unordered dyad ``(i,j)`` is an independent categorical variable over four mutually exclusive states, with normaliser ``D_{ij} = 1 + x_iy_j + x_jy_i + z_iz_j``.

| Description                    | Formula |
| --------------------------     | :-------------------------------------------------------------------------------- |
| Constraints                    | `` \forall i: \begin{cases} k^{→}_{i}(A^{*}) = \sum_{j \ne i} a^{*}_{ij}(1-a^{*}_{ji}) \\ k^{←}_{i}(A^{*}) = \sum_{j \ne i} a^{*}_{ji}(1-a^{*}_{ij}) \\ k^{↔}_{i}(A^{*}) = \sum_{j \ne i} a^{*}_{ij}a^{*}_{ji} \end{cases} ``|
| Hamiltonian                    | `` H(A, \Theta) = \sum_{i=1}^{N} \left[ \alpha_i k^{→}_{i}(A) + \beta_i k^{←}_{i}(A) + \gamma_i k^{↔}_{i}(A) \right]`` |
| Dyadic state probabilities     | `` p^{→}_{ij} = \frac{x_iy_j}{D_{ij}}, \quad p^{←}_{ij} = \frac{x_jy_i}{D_{ij}}, \quad p^{↔}_{ij} = \frac{z_iz_j}{D_{ij}}, \quad p^{∅}_{ij} = \frac{1}{D_{ij}}`` |
| $\langle a_{ij} \rangle$       | `` p^{→}_{ij} + p^{↔}_{ij}`` |
| Log-likelihood                 | `` \mathcal{L}(\Theta) = -\sum_{i=1}^{N} \left[ \alpha_i k^{→}_{i} + \beta_i k^{←}_{i} + \gamma_i k^{↔}_{i} \right] - \sum_{i<j} \ln D_{ij} ``|
| $\mathrm{Cov}(a_{ij}, a_{ji})$ | `` p^{↔}_{ij} - \langle a_{ij} \rangle \langle a_{ji} \rangle \ne 0`` |
| $\sigma^{*}(X)$                | ``\sqrt{\sum_{i,j} \left[ \left( \sigma^{*}[a_{ij}] \frac{\partial X}{\partial a_{ij}} \right)^{2} + \mathrm{Cov}(a_{ij},a_{ji}) \frac{\partial X}{\partial a_{ij}}\frac{\partial X}{\partial a_{ji}} \right]_{A = \langle A^{*} \rangle} + \dots }`` |
| $\sigma^{*}[a_{ij}]$           | ``\sqrt{\langle a_{ij} \rangle (1 - \langle a_{ij} \rangle)} ``   |

!!! note "Dyadic dependence"

    Unlike under the DBCM, within a dyad the entries ``a_{ij}`` and ``a_{ji}`` are **not** independent: ``\langle a_{ij}a_{ji} \rangle = p^{↔}_{ij} \ne \langle a_{ij} \rangle\langle a_{ji} \rangle``. The package handles this everywhere it matters: motif expectations are evaluated from the dyadic probability matrices (not from `Ĝ`) and are **exact**, the delta-method `σₓ` includes the covariance cross-term, and sampling draws whole dyad states. For the same reason `Ĝ`-based sampling (`rand(model, precomputed=true)`) is deliberately unsupported.

## Creation
```julia
using Graphs
using MaxEntropyGraphs

# a directed network with substantial reciprocity (r_t ≈ 0.76)
G = SimpleDiGraph(rhesus_macaques())

# instantiate an RBCM model
model = RBCM(G)
```

## Obtaining the parameters
```julia
# solve using default settings (anderson-accelerated fixed point)
solve_model!(model)
```

!!! note

    The fixed-point method (default) is stable on typical networks. On *degenerate* inputs, in particular fully reciprocal networks (``k^{→} = k^{←} = 0`` everywhere, e.g. `taro_exchange()`), where only the ``\gamma``-channel is identified. The accelerated fixed point can overshoot to non-finite values; use a gradient-based method (e.g. `method=:BFGS`) in that case. Channels with a zero-valued constraint (e.g. nodes without reciprocated links) are handled automatically (their parameter is pinned at its analytical ``+\infty`` optimum).

## Sampling the ensemble
```julia
# generate 10 random instances of the ensemble (per-dyad four-state draws)
rand(model, 10)
```

## Model comparison
The RBCM uses ``k = 3N`` parameters and ``n = N(N-1)`` observations. The *same* observation count as the DBCM, so their information criteria are directly comparable on the same network:

```julia
dmodel = DBCM(G); solve_model!(dmodel)

# does reciprocity warrant the extra N parameters?
reciprocity(G), reciprocity(dmodel)   # observed vs DBCM baseline
AICc(model) < AICc(dmodel)            # true on this network
```

## Counting network motifs
`motifs(model)` and the individual `M1(model)`, …, `M13(model)` return the **exact** expected occurrences of the 13 directed 3-node motifs under the RBCM, evaluated from the dyadic probabilities (Squartini & Garlaschelli (2011), Eq. C.16). Analytical z-scores combine these with the covariance-aware delta method (see the next section); sampling-based z-scores are available as well:

```julia
motifs(model)                  # exact ⟨M₁⟩, …, ⟨M₁₃⟩
motif_zscores(model, n=500)    # sampling-based z-scores of the whole spectrum (NuMeTriS-style)
```

## Expectation and variance
Under the RBCM every unordered dyad ``(i,j)`` is an **independent four-state categorical** variable (states ``→, ←, ↔, ∅``), so within a dyad the two entries ``a_{ij}`` and ``a_{ji}`` are *correlated*:

| Quantity                         | Formula |
| ------------------------------   | :------------------------------------------------ |
| ``\langle a_{ij} \rangle``       | ``p^{→}_{ij} + p^{↔}_{ij} = \frac{x_i y_j + z_i z_j}{D_{ij}}`` |
| ``\mathrm{Var}(a_{ij})``         | ``\langle a_{ij} \rangle (1 - \langle a_{ij} \rangle)`` |
| ``\mathrm{Cov}(a_{ij}, a_{ji})`` | ``p^{↔}_{ij} - \langle a_{ij} \rangle \langle a_{ji} \rangle \ne 0`` |

The workflow is the standard one (cf. [the analytical metrics page](../exact.md)): store the expected adjacency matrix and the entry-wise standard deviations, then propagate the gradient of ``X`` through `σₓ` (the autodiff backend is selectable via `gradient_method ∈ {:ReverseDiff, :ForwardDiff, :Zygote}`):

```jldoctest RBCM_variance
using MaxEntropyGraphs

# define the network and solve the model
G = MaxEntropyGraphs.Graphs.SimpleDiGraph(rhesus_macaques())
model = RBCM(G)
solve_model!(model)
# expected adjacency matrix ⟨A⟩ and entry-wise standard deviations σ[a_ij]
set_Ĝ!(model)
set_σ!(model)
nothing

# output


```
```jldoctest RBCM_variance
# analytical z-score of the fully reciprocated triangle (motif 13)
A = Matrix(MaxEntropyGraphs.Graphs.adjacency_matrix(G))
z_M13 = (M13(A) - M13(model)) / σₓ(model, M13)

# output

-0.2810404657808886
```

The reciprocated triangle count is fully consistent with the RBCM on this network, whereas the same metric deviates by about two standard deviations from the [`DBCM`](@ref MaxEntropyGraphs.DBCM) baseline (``z_{M13} \approx 2.0``), which does not constrain reciprocity.

!!! note "Dyadic covariance in σₓ"
    The delta-method `σₓ` includes the within-dyad covariance cross-term ``\mathrm{Cov}(a_{ij}, a_{ji}) \frac{\partial X}{\partial a_{ij}} \frac{\partial X}{\partial a_{ji}}`` in the error propagation (cf. the "Dyadic dependence" note above); distinct dyads remain independent.

!!! warning "Memory footprint"
    `Ĝ`/`set_Ĝ!` and `σˣ`/`set_σ!` materialize dense ``N \times N`` matrices, and `σₓ` requires both, so this analysis uses ``O(N^2)`` memory. For large networks, prefer the sampling route to estimate variances (see [Performance and scalability](../performance.md)).

_References_

```@raw html
<ul>
<li>
<a id="1">[1]</a> 
Squartini, Tiziano and Garlaschelli, Diego. <!--  author(s) --> 
<em>"Analytical maximum-likelihood method to detect patterns in real networks"</em> <!--  title --> 
2011 New J. Phys. 13 083001. <!--  publisher(s) --> 
<a href="https://iopscience.iop.org/article/10.1088/1367-2630/13/8/083001">https://iopscience.iop.org/article/10.1088/1367-2630/13/8/083001</a>
</li>
<li>
<a id="2">[2]</a> 
Squartini, Tiziano and Garlaschelli, Diego. <!--  author(s) --> 
<em>"Maximum-Entropy Networks: Pattern Detection, Network Reconstruction and Graph Combinatorics"</em> <!--  title --> 
Springer-Verlag GmbH; 1st ed. 2017 edition (25 Dec. 2017). <!--  publisher(s) --> 
<a href="https://link.springer.com/book/10.1007/978-3-319-69438-2">https://link.springer.com/book/10.1007/978-3-319-69438-2</a>
</li>
<li>
<a id="3">[3]</a> 
Di Vece, Marzio and Pijpers, Frank P. and Garlaschelli, Diego. <!--  author(s) --> 
<em>"Commodity-specific triads in the Dutch inter-industry production network"</em> <!--  title --> 
Sci Rep 14, 3625 (2024) / arXiv:2305.12179. <!--  publisher(s) --> 
<a href="https://arxiv.org/abs/2305.12179">https://arxiv.org/abs/2305.12179</a>
</li>
<li>
<a id="4">[4]</a> 
Di Vece, Marzio. <!--  author(s) --> 
<em>"NuMeTriS: Null Models for Triadic Structures"</em> <!--  title --> 
<a href="https://github.com/MarsMDK/NuMeTriS">https://github.com/MarsMDK/NuMeTriS</a>
</li>
</ul>
```
