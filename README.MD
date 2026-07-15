[![CI](https://github.com/B4rtDC/MaxEntropyGraphs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/B4rtDC/MaxEntropyGraphs.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/B4rtDC/MaxEntropyGraphs.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/B4rtDC/MaxEntropyGraphs.jl)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://B4rtDC.github.io/MaxEntropyGraphs.jl/dev/)

[![DOI](https://zenodo.org/badge/437887003.svg)](https://zenodo.org/badge/latestdoi/437887003)

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
* [Running the tests](#running-the-tests)
* [Getting help](#getting-help)
* [Citing](#citing)

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

## Available models
`MaxEntropyGraphs.jl` is intended to do *network randomization*. The specific kind of network to randomize and the properties to preserve defines the model you need. 
Details for each of the available models can be found in the [Models](./models.html) section. 
Sometimes multiple models might be applicable. In that case, you can use information theory-based metrics to help make an informed decision. 
The Bayesian Information Criterion (BIC) and Akaike Information Criterion (AIC) are available to quantify this.
Lower BIC or AIC values indicate a better model fit, considering the trade-off between complexity and fit.

The package currently implements the following models (see the [Models](https://B4rtDC.github.io/MaxEntropyGraphs.jl/dev/models/) section of the documentation for details):

* **UBCM** — Undirected Binary Configuration Model
* **DBCM** — Directed Binary Configuration Model
* **BiCM** — Bipartite Configuration Model

## Running the tests
To run the test suite, activate the package environment and call `Pkg.test`:
```julia
using Pkg
Pkg.test("MaxEntropyGraphs")
```
The tests also run automatically on every push and pull request via GitHub Actions (Julia 1.10–1.12 on Linux, macOS, and Windows) and include [Aqua.jl](https://github.com/JuliaTesting/Aqua.jl) quality checks.

## Getting help
* Documentation: <https://B4rtDC.github.io/MaxEntropyGraphs.jl/dev/>
* Questions and discussion: [GitHub Discussions](https://github.com/B4rtDC/MaxEntropyGraphs.jl/discussions)
* Bug reports and feature requests: [GitHub Issues](https://github.com/B4rtDC/MaxEntropyGraphs.jl/issues)
* Contributing: see [CONTRIBUTING.md](CONTRIBUTING.md) and our [Code of Conduct](CODE_OF_CONDUCT.md)

## Citing
When using this package in your scientific research, please cite it. The canonical citation metadata lives in [`CITATION.cff`](CITATION.cff). A BibTeX entry for the software release:
```
@software{de_clerck_maxentropygraphs,
  author       = {De Clerck, Bart},
  title        = {{MaxEntropyGraphs.jl}},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v0.6.0},
  doi          = {10.5281/zenodo.8314609},
  url          = {https://github.com/B4rtDC/MaxEntropyGraphs.jl}
}
```

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
    Vallarano, Nicolò, Matteo Bruno, Emiliano Marchese, Giuseppe Trapani, Fabio Saracco, Giulio Cimini, Mario Zanon, and Tiziano Squartini.
    "Fast and scalable likelihood maximization for Exponential Random Graph Models with local constraints."
    Scientific Reports 11 (2021): 15227.
    https://doi.org/10.1038/s41598-021-93830-4