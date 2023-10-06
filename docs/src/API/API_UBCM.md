## Index

```@index
Pages = ["API_UBCM.md"]
```

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