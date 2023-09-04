# API
## Global
```@docs
MaxEntropyGraphs
AbstractMaxEntropyModel
MaxEntropyGraphs.ConvergenceError
MaxEntropyGraphs.strength
```

## Utility fuctions
Some specific utility functions are made available:

```@docs
MaxEntropyGraphs.log_nan
```

```@docs
MaxEntropyGraphs.np_unique_clone
```

## UBCM
```@docs 
MaxEntropyGraphs.UBCM
UBCM(::T) where {T}
Base.length(::UBCM)
MaxEntropyGraphs.L_UBCM_reduced
MaxEntropyGraphs.∇L_UBCM_reduced!
MaxEntropyGraphs.∇L_UBCM_reduced_minus!
MaxEntropyGraphs.UBCM_reduced_iter!
MaxEntropyGraphs.solve_model!(::UBCM)
MaxEntropyGraphs.initial_guess(::UBCM)
MaxEntropyGraphs.set_xᵣ!(::UBCM)
MaxEntropyGraphs.Ĝ(::UBCM)
MaxEntropyGraphs.set_Ĝ!(::UBCM)
MaxEntropyGraphs.σˣ(::UBCM)
MaxEntropyGraphs.set_σ!(::UBCM)

Base.rand(::UBCM)
Base.rand(::UBCM,::Int)
```

