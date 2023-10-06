## Index

```@index
Pages = ["API_DBCM.md"]
```

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