# Models
The different models that are available are described below. Each of these models is a subtype of an `AbstractMaxEntropyModel`

```@docs
MaxEntropyGraphs.AbstractMaxEntropyModel
```
## UBCM
```@docs
MaxEntropyGraphs.UBCM
```
This type can be instantiated with the following methods:

```@docs
MaxEntropyGraphs.UBCM(G::Graphs.SimpleGraph; method="fixed-point", initial_guess="degrees", max_steps=5000, tol=1e-12, kwargs...)
```
## DBCM
```@docs
MaxEntropyGraphs.DBCM
```