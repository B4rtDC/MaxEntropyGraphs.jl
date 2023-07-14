[![Build status (Github Actions)](https://github.com/B4rtDC/MaxEntropyGraphs.jl/workflows/CI/badge.svg)](https://github.com/B4rtDC/MaxEntropyGraphs.jl/actions)
[![codecov.io](http://codecov.io/github/B4rtDC/MaxEntropyGraphs.jl/coverage.svg?branch=main)](http://codecov.io/github/B4rtDC/MaxEntropyGraphs.jl?branch=2023rework)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://B4rtDC.github.io/MaxEntropyGraphs.jl/dev/)

# MaxEntropyGraphs.jl
A package to work with maximum entropy networks in the Julia ecosystem. The functionality of this package is heavily inspired by the [NEMtropy](https://github.com/nicoloval/NEMtropy) package for Python, but has been adapted to the Julia ecosystem.

### Table of contents
* [Installation](#installation)
* [Differences with the NEMtropy package](#differences-with-the-nemtropy-package)
* [Available models](#available-models)
* [Example usage](#example-usage)


## Installation
Assuming that you already have Julia correctly installed, it suffices to import MaxEntropyGraphs.jl in the standard way:
```julia
using Pkg
Pkg.add("MaxEntropyGraphs")
```

## Differences and similarities with the NEMtropy package
* We use the [JuliaGraphs](https://juliagraphs.org/) ecosystem for everything network related, whereas NEMtropy requires you to work with degree sequences, adjacency matrices or edge lists. These option are also available, either directly (in the case of degree sequences) or indirectly (by passing through the JuliaGraphs ecosystem).
* To obtain the maximum likelihood parameters of a model we use either:
    - the well-established [Optimization.jl](https://github.com/SciML/Optimization.jl) package (this maximises the loglikelihood of the networks ensemble). Working this way uses automated differentiation, but explicit non-allocating gradient functions are also provided for the different models.
    - [NLsolve.jl](https://github.com/JuliaNLSolvers/NLsolve.jl#anderson-acceleration)'s Anderson acceleration for the fixed point methods proposed in [[4]](#4) (cf. documentation/examples). The iterative methods are non-allocating, so they are orders of magnitude faster than the Python implementation.
* We have also maintained the different options for the initial values for each method (cf. `initial_guess(::AbstractMaxEntropyModel)`).
* By making use of the automatic differentiation capabilities of Julia, we can:
    - approximation the gradient of the likelihood function of the graphs
    - compute the gradient of any metric with respect to its adjacency matrix without having to compute these by hand and implement the partial derivative for each possible metric (cf. examples for more details on this).
* For both the models and the computing functions we make a clear distinction between the likelihood maximising paramers and the variable substitution (e.g. $\theta_i \leftrightarrow x_i = e^{-\theta_i}$ for the UBCM). Doing so makes the code more readable because it is closer to the mathematical formulation.
* Sampling from an `AbstractMaxEntropyModel` will generate a corresponding subtype of  `AbstractGraph` from the [JuliaGraphs](https://juliagraphs.org/) ecosystem. The methods that are available in `Graphs.jl`, `SimpleWeightedGraphs` etc. have been extended for the different maximum entropy models wherever applicable. 

## Available models
`MaxEntropyGraphs.jl` is intended to do *network randomization*. The specific kind of network to randomize and the properties to preserve defines the model you need. The checklists below indicated which models are 
currently available.

* [ ] **UBCM** *Undirected Binary Configuration Model* [[1]](#1)
    - [x] Types
    - [x] Likelihood related functions
    - [x] Likelihood maximisation in pure Julia (FP + optimisation)
    - [ ] Check for:
        - [ ] behavior of zero degree nodes
        - [ ] networks with self links (in principle not allowed in SimpleGraphs, TBC)
    - [ ] $\dots$
* [ ] **DBCM** *Directed Binary Configuration Model* [[1]](#1)
    - [x] Types
    - [x] Likelihood related functions
    - [ ] Likelihood maximisation in pure Julia (FP ok + optimisation to do)
    - [ ] Check for:
        - [X] behavior of zero degree nodes
        - [ ] networks with self links (in principle not allowed in SimpleGraphs, TBC)
    - [ ] $\dots$
* [ ] **BiCM** *Bipartite Configuration Model* [[3]](#3)
* [ ] **UECM** *Undirected Enhanced Configuration Model* [[1]](#1)
* [ ] **DECM** *Directed Enhanced Configuration Model* [[1]](#1)
* [ ] **CReMa** [[2]](#2)



## Example usage
More detailed information is available in the documentation and some demos are available in the examples subfolder.
```julia
using MaxEntropyGraphs
# generate a simple graph
G = NEMTROPY.Graphs.smallgraph(:karate)
# generate a model
model = UBCM(G)
# solve the model
solve_model!(model)
# generate 100 random instances from the ensemble
G_sample = rand(model, 100)
```



## To do list:
- [ ] add additional models
- [ ] optimize the code for speed after benchmarking 
- [ ] update documentation
- [ ] add GPU support 
    - [ ] maximum liklihood parametercomputation
    - [ ] metric computation


_References_

* <a id="1">[1]</a>
    Squartini, Tiziano, Rossana Mastrandrea, and Diego Garlaschelli.
    "Unbiased sampling of network ensembles."
    New Journal of Physics 17.2 (2015): 023052.
    https://arxiv.org/abs/1406.1197
* <a id="2">[2]</a>
    Parisi, Federica, Tiziano Squartini, and Diego Garlaschelli.
    "A faster horse on a safer trail: generalized inference for the efficient reconstruction of weighted networks."
    New Journal of Physics 22.5 (2020): 053053.
    https://arxiv.org/abs/1811.09829
* <a id="3">[3]</a>
    Saracco, Fabio, Riccardo Di Clemente, Andrea Gabrielli, and Tiziano Squartini.
	"Randomizing bipartite networks: the case of the World Trade Web." 
	Scientific reports 5, no. 1 (2015): 1-18.
    https://doi.org/10.1038/srep10595
* <a id="4">[4]</a>
    Nicolò Vallarano, Matteo Bruno, Emiliano Marchese, Giuseppe Trapani, Fabio Saracco, Giulio Cimini, Mario Zanon & Tiziano Squartini 
    Saracco, Fabio, Riccardo Di Clemente, Andrea Gabrielli, and Tiziano Squartini.
	"Fast and scalable likelihood maximization for Exponential Random Graph Models with local constraints." 
	Scientific reports 11 (2021).
    https://doi.org/10.1038/s41598-021-93830-4