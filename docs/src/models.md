# Models
The different models that are available are described below. Each of these models is a subtype of an `AbstractMaxEntropyModel`

```@docs
MaxEntropyGraphs.AbstractMaxEntropyModel
```

 For each model there are multiple methods available to obtain the parameters that generate the maximum likelihood. Each of these models can be solved by using the `solve!` function.

```Julia
solve!(model)
```

```@autodocs
Modules = [MaxEntropyGraphs]
Order = [:function]
Filter = x -> occursin("UBCM,"$(x)"))
```
## UBCM
Generate a Undirected Binary Configuration Model.


## DBCM
Generate a Directed Binary Configuration Model.

