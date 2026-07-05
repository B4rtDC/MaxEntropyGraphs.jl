# CReM
## Model description
The Conditional Reconstruction Method (CReM) is a maximum-entropy null model for **weighted, undirected** networks with **continuous, positive** weights. It is a **two-step** model [[1](#1),[2](#2)]:

1. a **binary layer** fixes the topology. The marginal probability of an edge, ``f_{ij} = \langle a_{ij} \rangle``, is supplied by a prior binary model — here an internally-solved [`UBCM`](@ref) on the degree sequence, so ``f_{ij} = \frac{x_i x_j}{1 + x_i x_j}`` with ``x_i = e^{-\alpha_i}``;
2. a **weighted layer** fixes the strengths. Conditional on an edge existing, the weight is exponentially distributed with rate ``\theta_i + \theta_j``. The parameters ``\theta`` (one per node) are obtained by maximising the conditional log-likelihood.

Because the binary structure is taken as given, the CReM is easier to solve than the joint [`UECM`](@ref): both the fixed-point and Newton recipes converge well [[1]](#1), and the fixed point is the default.

| Description                    | Formula |
| --------------------------     | :-------------------------------------------------------------------------------- |
| Constraints                    | `` \forall i: \begin{cases} k_{i}(A^{*}) = \sum_{j \ne i} a^{*}_{ij} \\ s_{i}(W^{*}) = \sum_{j \ne i} w^{*}_{ij} \end{cases} ``|
| Hamiltonian                    | `` H(W, \theta) = \sum_{i=1}^{N} \theta_i s_{i}(W)`` |
| $\langle a_{ij} \rangle$       | `` f_{ij} = \frac{x_i x_j}{1 + x_i x_j} \quad (\text{binary/UBCM layer})`` |
| $q(w_{ij} \mid a_{ij}=1)$      | `` (\theta_i + \theta_j)\, e^{-(\theta_i + \theta_j) w} , \quad w > 0`` |
| Log-likelihood                 | `` \mathcal{L}(\theta) = -\sum_{i=1}^{N} \theta_i s_{i}(W^{*}) + \sum_{i=1}^{N} \sum_{j=1, j < i}^{N} f_{ij} \ln (\theta_i + \theta_j) ``|
| $\langle w_{ij} \rangle$       | `` \frac{f_{ij}}{\theta_i + \theta_j}`` |
| $\sigma^{*}(X)$                | ``\sqrt{\sum_{i,j} \left( \sigma^{*}[a_{ij}] \frac{\partial X}{\partial a_{ij}}  \right)^{2}_{A = \langle A^{*} \rangle} + \dots }`` |
| $\sigma^{*}[a_{ij}]$           | ``\sqrt{f_{ij} (1 - f_{ij})} ``   |

!!! note

    The variance-based quantities (`σˣ`, `set_σ!`, `σₓ`) currently describe the **binary** (adjacency) layer only, for which ``a_{ij}`` follows a Bernoulli distribution with success probability ``f_{ij}``. The variance of the (continuous, exponential) weights is not implemented.

## Creation
```julia
using Graphs, SimpleWeightedGraphs
using MaxEntropyGraphs

# a weighted, undirected network (here: the symmetrised rhesus macaques grooming network; the CReM
# treats the weights as continuous positive quantities)
G = SimpleWeightedGraph(rhesus_macaques())

# instantiate a CReM model
model = CReM(G)
```

## Obtaining the parameters
```julia
# solve using the default settings (two-step: internal UBCM, then the fixed-point weighted layer)
solve_model!(model)
```

!!! note

    The weighted parameters ``\theta`` are the **direct** exponential rates (they appear as ``\ln(\theta_i + \theta_j)`` in the log-likelihood), so a solution requires ``\theta_i + \theta_j > 0``. Every initial guess is therefore strictly positive. The default solver is the (stable) `fixedpoint`; `BFGS` and `Newton` are also available and use a `BackTracking` line search to stay inside the feasible region.

## Expected adjacency and weights
```julia
# expected (binary) adjacency matrix; row sums reproduce the degree sequence
Ĝ(model)

# expected weighted adjacency matrix; row sums reproduce the strength sequence
MaxEntropyGraphs.Ŵ(model)
```

## Sampling the ensemble
```julia
# generate 10 random weighted instances of the ensemble (continuous, exponential weights)
rand(model, 10)
```

## Model comparison
```julia
# compute the AIC (the conditional CReM has N parameters, one θ per node)
AIC(model)
```

_References_

```@raw html
<ul>
<li>
<a id="1">[1]</a> 
Parisi, Federica and Squartini, Tiziano and Garlaschelli, Diego. <!--  author(s) --> 
<em>"A faster horse on a safer trail: generalized inference for the efficient reconstruction of weighted networks"</em> <!--  title --> 
New Journal of Physics 22, 2020. <!--  publisher(s) --> 
<a href="https://doi.org/10.1088/1367-2630/ab74a7">https://doi.org/10.1088/1367-2630/ab74a7</a>
</li>
<li>
<a id="2">[2]</a> 
Vallarano, Nicolò and Bruno, Matteo and Marchese, Emiliano and Trapani, Giuseppe and Saracco, Fabio and Cimini, Giulio and Zanon, Mario and Squartini, Tiziano. <!--  author(s) --> 
<em>"Fast and scalable likelihood maximization for Exponential Random Graph Models with local constraints"</em> <!--  title --> 
Scientific Reports 11, 2021. <!--  publisher(s) --> 
<a href="https://doi.org/10.1038/s41598-021-93830-4">https://doi.org/10.1038/s41598-021-93830-4</a>
</li>
</ul>
```
