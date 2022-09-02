[![Build status (Github Actions)](https://github.com/B4rtDC/MaxEntropyGraphs.jl/workflows/CI/badge.svg)](https://github.com/B4rtDC/MaxEntropyGraphs.jl/actions)
[![codecov.io](http://codecov.io/github/B4rtDC/MaxEntropyGraphs.jl/coverage.svg?branch=main)](http://codecov.io/github/B4rtDC/MaxEntropyGraphs.jl?branch=main)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://B4rtDC.github.io/MaxEntropyGraphs.jl/dev/)

# MaxEntropyGraphs.jl


A Julia package to use maximum entropy networks using the JuliaGraphs ecosystem.


It provides the following functionalities (currently only for the UBCM and DBCM model):
* sampling from a networks ensemble
* computing expected value and variance for any metric that can be defined using the adjacence matrix of the network. This is done by using the automatic differentiation capabilities of Julia. In doing so, there is no need to compute the gradient of the metric explicitly and its also tends to be faster than doing it manually. 

The computation of the parameters that maximize the likelihood is done using the [NEMtropy package](https://nemtropy.readthedocs.io/en/master/index.html) (making use of PyCall).

More detailed information is available in the documentation and some demo notebooks are available in the examples subfolder.



## To do list:
- [ ] add additional models
- [ ] use [NLsolve.jl](https://github.com/JuliaNLSolvers/NLsolve.jl) to obtain the model parameters instead of using PyCall 
- [ ] add GPU support (both for parameter and metric computation)

