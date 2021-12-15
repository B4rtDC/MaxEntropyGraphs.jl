[![Build status (Github Actions)](https://github.com/B4rtDC/MaxEntropyGraphs.jl/workflows/CI/badge.svg)](https://github.com/B4rtDC/MaxEntropyGraphs.jl/actions)
[![codecov.io](http://codecov.io/github/B4rtDC/MaxEntropyGraphs.jl/coverage.svg?branch=main)](http://codecov.io/github/B4rtDC/MaxEntropyGraphs.jl?branch=main)


# MaxEntropyGraphs


A Julia package to use maximum entropy networks using the JuliaGraphs ecosystem (LightGraphs & SimpleWeightedGraphs).


It provides the following functionalities:
* Base and accelerated implementation to find the Lagrangian multipliers that maximize the entropy of the random graph models (UBCM, DBCM, UWCM, DWCM)
* Implements efficient estimates of some graph metrics
* Implements efficient metric variance for anomaly detection (using both automated differentiation as well as exact expressions)

## Examples
```Julia


```

## Documenation
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://B4rtDC.github.io/MaxEntropyGraphs.jl/dev/)

## To do list:
- [x] add direct mapping to compact model for usability
- [ ] compare performance test for optimized exact expression with automated differentiation (it is already faster than working with the exact expressions). Possible downside: requires tailored expression for each metric, which can be complicated.
- [ ] compare performance with some existing methods (NEMtropy hub)
- [ ] add GPU support
