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
MaxEntropyGraphs.np_unique_clone
```


## Small graph constructors
```@docs
MaxEntropyGraphs.taro_exchange
MaxEntropyGraphs.rhesus_macaques
MaxEntropyGraphs.chesapeakebay
MaxEntropyGraphs.everglades
MaxEntropyGraphs.florida
MaxEntropyGraphs.littlerock
MaxEntropyGraphs.maspalomas
MaxEntropyGraphs.stmarks
MaxEntropyGraphs.parse_konect
MaxEntropyGraphs.readpajek
```



## Graph metrics
```@docs
MaxEntropyGraphs.degree
MaxEntropyGraphs.outdegree
MaxEntropyGraphs.indegree
MaxEntropyGraphs.strength
MaxEntropyGraphs.outstrength
MaxEntropyGraphs.instrength
MaxEntropyGraphs.ANND
MaxEntropyGraphs.ANND_out
MaxEntropyGraphs.ANND_in
MaxEntropyGraphs.wedges
MaxEntropyGraphs.triangles
MaxEntropyGraphs.squares
MaxEntropyGraphs.a⭢
MaxEntropyGraphs.a⭠
MaxEntropyGraphs.a⭤
MaxEntropyGraphs.a̲
MaxEntropyGraphs.M1
MaxEntropyGraphs.M2
MaxEntropyGraphs.M3
MaxEntropyGraphs.M4
MaxEntropyGraphs.M4
MaxEntropyGraphs.M5
MaxEntropyGraphs.M6
MaxEntropyGraphs.M8
MaxEntropyGraphs.M9
MaxEntropyGraphs.M10
MaxEntropyGraphs.M11
MaxEntropyGraphs.M12
MaxEntropyGraphs.M13
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
MaxEntropyGraphs.σₓ(::UBCM, ::Function)
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
MaxEntropyGraphs.σₓ(::DBCM, ::Function)
```

