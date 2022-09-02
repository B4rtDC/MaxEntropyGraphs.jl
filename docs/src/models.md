# Models
The different models that are available are described below. Each of these models is a subtype of an `AbstractMaxEntropyModel`

```@docs
MaxEntropyGraphs.::AbstractMaxEntropyModel
```
For each model, the following methods should be implemented:
```@docs
σ(::AbstractMaxEntropyModel)
Ĝ(::AbstractMaxEntropyModel)
σˣ(X::Function, M::T) where T <: MaxEntropyGraphs.AbstractMaxEntropyModel
Base.rand(::AbstractMaxEntropyModel)
```


## UBCM
Generate a Undirected Binary Configuration Model.
```@docs
MaxEntropyGraphs.UBCM
σ(::MaxEntropyGraphs.UBCM)
Ĝ(::MaxEntropyGraphs.UBCM)
σˣ(X::Function, M::MaxEntropyGraphs.UBCM)
Base.rand(::MaxEntropyGraphs.UBCM)
```

## DBCM
Generate a Directed Binary Configuration Model.

```@docs
MaxEntropyGraphs.DBCM
σ(::MaxEntropyGraphs.DBCM)
Ĝ(::MaxEntropyGraphs.DBCM)
σˣ(X::Function, M::MaxEntropyGraphs.DBCM)
Base.rand(::MaxEntropyGraphs.DBCM)
```