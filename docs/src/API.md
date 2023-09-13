# API
## Index

```@index
Pages = ["API.md"]
```

## Global
```@docs
MaxEntropyGraphs
AbstractMaxEntropyModel
MaxEntropyGraphs.ConvergenceError

```

## Utility fuctions
Some specific utility functions are made available:

```@docs
MaxEntropyGraphs.log_nan
```

```@docs
MaxEntropyGraphs.np_unique_clone
```

## Graph constructors
```@docs
MaxEntropyGraphs.taro_exchange
MaxEntropyGraphs.rhesus_macaques
```



## Graph metrics
```@docs
MaxEntropyGraphs.degree
MaxEntropyGraphs.outdegree
MaxEntropyGraphs.indegree
MaxEntropyGraphs.strength
MaxEntropyGraphs.ANND
```

## UBCM
```@docs 
MaxEntropyGraphs.UBCM
UBCM(::T) where {T}
MaxEntropyGraphs.solve_model!(::UBCM)
MaxEntropyGraphs.initial_guess(::UBCM)
Base.rand(::UBCM)
Base.rand(::UBCM,::Int)
MaxEntropyGraphs.AIC(::UBCM)
MaxEntropyGraphs.AICc(::UBCM)
MaxEntropyGraphs.BIC(::UBCM)
Base.length(::UBCM)
MaxEntropyGraphs.L_UBCM_reduced
MaxEntropyGraphs.∇L_UBCM_reduced!
MaxEntropyGraphs.∇L_UBCM_reduced_minus!
MaxEntropyGraphs.UBCM_reduced_iter!
MaxEntropyGraphs.set_xᵣ!(::UBCM)
MaxEntropyGraphs.Ĝ(::UBCM)
MaxEntropyGraphs.set_Ĝ!(::UBCM)
MaxEntropyGraphs.σˣ(::UBCM)
MaxEntropyGraphs.set_σ!(::UBCM)
MaxEntropyGraphs.precision(::UBCM)
MaxEntropyGraphs.A(::UBCM,::Int64,::Int64)
MaxEntropyGraphs.f_UBCM(::UBCM)
```

## DBCM
```@docs 
MaxEntropyGraphs.DBCM
DBCM(::T) where {T}
MaxEntropyGraphs.solve_model!(::DBCM)
MaxEntropyGraphs.initial_guess(::DBCM)
Base.rand(::DBCM)
Base.rand(::DBCM,::Int)
MaxEntropyGraphs.AIC(::DBCM)
MaxEntropyGraphs.AICc(::DBCM)
MaxEntropyGraphs.BIC(::DBCM)
Base.length(::DBCM)
MaxEntropyGraphs.L_DBCM_reduced
MaxEntropyGraphs.∇L_DBCM_reduced!
MaxEntropyGraphs.∇L_DBCM_reduced_minus!
MaxEntropyGraphs.DBCM_reduced_iter!
MaxEntropyGraphs.set_xᵣ!(::DBCM)
MaxEntropyGraphs.set_yᵣ!(::DBCM)
MaxEntropyGraphs.Ĝ(::DBCM)
MaxEntropyGraphs.set_Ĝ!(::DBCM)
MaxEntropyGraphs.σˣ(::DBCM)
MaxEntropyGraphs.set_σ!(::DBCM)
MaxEntropyGraphs.precision(::DBCM)
MaxEntropyGraphs.A(::DBCM,::Int64,::Int64)
MaxEntropyGraphs.f_DBCM(::DBCM)
```

