# Fastmaxent


A Julia package to use maximum entropy networks based on the JuliaGraphs ecosystem (LightGraphs & SimpleWeightedGraphs).


It provides the following functionalities:
* Base and accelerated implementation to find the Lagrangian multipliers that maximize the entropy of the random graph models (UBCM, DBCM, UWCM, DWCM)
* Implements efficient estimates of some graph metrics
* Implements efficient metric variance for anomaly detection (using both automated differentiation as well as exact expressions)

## Examples
```Julia
using fastmaxent

```

## To do list:
- [ ] add direct mapping to compact model for usability
- [ ] compare performance test for optimized exact expression with automated differentiation (it is already faster than working with the exact expressions). Possible downside: requires tailored expression for each metric, which can be complicated.
- [ ] compare performance with some existing methods (e.g. Max&Sam package - uses likelihood function  - in Matlab and some other in Python)
- [ ] implement indexing on from the small model directly to the large one (macro magic)

