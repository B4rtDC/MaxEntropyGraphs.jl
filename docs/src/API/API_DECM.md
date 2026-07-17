## Index

```@index
Pages = ["API_DECM.md"]
```

```@docs 
MaxEntropyGraphs.DECM
DECM(::T) where {T}
MaxEntropyGraphs.solve_model!(::DECM)
MaxEntropyGraphs.initial_guess(::DECM)
Base.rand(::DECM)
Base.rand(::DECM,::Int)
MaxEntropyGraphs.AIC(::DECM)
MaxEntropyGraphs.AICc(::DECM)
MaxEntropyGraphs.BIC(::DECM)
Base.length(::DECM)
MaxEntropyGraphs.L_DECM_reduced
MaxEntropyGraphs.∇L_DECM_reduced!
MaxEntropyGraphs.∇L_DECM_reduced_minus!
MaxEntropyGraphs.DECM_reduced_iter!
MaxEntropyGraphs.set_xᵣ!(::DECM)
MaxEntropyGraphs.set_yᵣ!(::DECM)
MaxEntropyGraphs.Ĝ(::DECM)
MaxEntropyGraphs.set_Ĝ!(::DECM)
MaxEntropyGraphs.Ŵ(::DECM)
MaxEntropyGraphs.set_Ŵ!(::DECM)
MaxEntropyGraphs.σˣ(::DECM)
MaxEntropyGraphs.set_σ!(::DECM)
MaxEntropyGraphs.σʷ(::DECM)
MaxEntropyGraphs.set_σʷ!(::DECM)
MaxEntropyGraphs.precision(::DECM)
MaxEntropyGraphs.A(::DECM,::Int64,::Int64)
MaxEntropyGraphs.f_DECM
MaxEntropyGraphs.σₓ(::DECM, ::Function)
```
