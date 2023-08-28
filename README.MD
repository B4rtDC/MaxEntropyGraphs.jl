[![Build status (Github Actions)](https://github.com/B4rtDC/MaxEntropyGraphs.jl/workflows/CI/badge.svg)](https://github.com/B4rtDC/MaxEntropyGraphs.jl/actions)
[![codecov.io](http://codecov.io/github/B4rtDC/MaxEntropyGraphs.jl/coverage.svg?branch=main)](http://codecov.io/github/B4rtDC/MaxEntropyGraphs.jl?branch=2023rework)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://B4rtDC.github.io/MaxEntropyGraphs.jl/dev/)


# MaxEntropyGraphs
The goal of the *MaxEntropyGraphs.jl* package is to group the various maximum-entropy null models for network randomization and make them available to the Julia community in a single package. This work was in part inspired by the [Maximum Entropy Hub](https://meh.imtlucca.it), but unlike the latter, this package works in an integrated way with the exisiting Julia ecosystem for handling graphs, optimization tools and numerical solvers and groups all models in a single framework.

The package provides the following functionalities:
* Computing the likelihood maximizing parameters for a broad set of network models (cf. `Models` section of the documentation).
* Sampling of networks from a network ensemble once the parameters have been computed.
* Analytically computing ensemble averages and their standard deviations (for a subset of models).
* Running motif based analysis (for a subset of models).
* Bipartite network projections with statistical significance analysis (for a subset of models).

Each network models can be solved in different ways, with a fixed-point method typically being the fastest and a Newton-based method being the slowest. Depending on the complexity of the network model, some solvers might not always converge.


### Table of contents
* [Installation](#installation)
* [Example usage](#example-usage)
* [Available models](#available-models)
* 

## Installation
Assuming that you already have Julia correctly installed, installation is straightforward. 
It suffices to import MaxEntropyGraphs.jl in the standard way:
```julia
using Pkg
Pkg.add("MaxEntropyGraphs")
```
or enter the Pkg mode by hitting ```]```, and then run the following command:
```Julia
pkg> add MaxEntropyGraphs
```

## Example usage
More detailed information is available in the documentation and some demos are available in the examples subfolder.
```julia
using Graphs
using MaxEntropyGraphs
# generate a simple graph
G = smallgraph(:karate)
# generate a model
model = UBCM(G)
# solve the model
solve_model!(model)
# generate 100 random instances from the ensemble
G_sample = rand(model, 100)
```

## Citing


## Available models
`MaxEntropyGraphs.jl` is intended to do *network randomization*. The specific kind of network to randomize and the properties to preserve defines the model you need. 

The checklists below indicate the status of each model that is currently available.

* [ ] **UBCM** *Undirected Binary Configuration Model* [[1]](#1)
    - [x] Types
    - [x] Likelihood related functions
    - [x] Likelihood maximisation in pure Julia (FP + optimisation)
    - [x] Sampling
    - [ ] Unit tests
    - [ ] Check for:
        - [ ] behavior of zero degree nodes
        - [ ] networks with self links (in principle not allowed in SimpleGraphs, TBC)
    - [ ] $\dots$
* [ ] **DBCM** *Directed Binary Configuration Model* [[1]](#1)
    - [x] Types
    - [x] Likelihood related functions
    - [x] Likelihood maximisation in pure Julia (FP ok + optimisation)
    - [x] Sampling
    - [ ] Unit tests
    - [ ] Check for:
        - [X] behavior of zero degree nodes
        - [ ] networks with self links (in principle not allowed in SimpleGraphs, TBC)
    - [ ] $\dots$
* [ ] **BiCM** *Bipartite Configuration Model* [[3]](#3)
    - [x] Types
    - [x] Likelihood related functions
    - [x] Likelihood maximisation in pure Julia (FP ok + optimisation). Note: possible problem in Nemtropy gradient implementation (difference when compared to autodiff and method does not converge)
    - [x] Sampling
    - [ ] Unit tests
    - [ ] Monopartite projection
    - [x] Check for:
        - [x] behavior of zero degree nodes
        - [ ] networks with self links (in principle not allowed in SimpleGraphs, TBC)
    - [ ] $\dots$
* [ ] **UECM** *Undirected Enhanced Configuration Model* [[1]](#1)
    - [x] Types
    - [x] Likelihood related functions
    - [x] Likelihood maximisation in pure Julia (FP ok + optimisation). Note: FP method needs very good initial conditions to work propoerly (expected)
    - [x] Sampling
    - [ ] Unit tests
    - [x] Check for:
        - [x] behavior of zero degree nodes
        - [ ] networks with self links (in principle not allowed in SimpleGraphs, TBC)
    - [ ] $\dots$
* [ ] **CReM** *Conditional Reconstruction Method*[[2]](#2)
* [ ] **DBiCM** *Directed Bipartite Configuration Model*  **add ref**
* [ ] **DECM** *Directed Enhanced Configuration Model* [[1]](#1)








## To do list:
- [ ] convergence metrics to check if expected adjacency matrix matches the constraints
- [ ] add additional models
- [ ] optimize the code for speed after benchmarking 
- [ ] update documentation
- [ ] add information criterion for model selection (AIC)
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