## Index

```@index
Pages = ["API_DCReM.md"]
```

```@docs 
MaxEntropyGraphs.DCReM
DCReM(::T) where {T}
MaxEntropyGraphs.solve_model!(::DCReM)
MaxEntropyGraphs.initial_guess(::DCReM)
Base.rand(::DCReM)
Base.rand(::DCReM,::Int)
MaxEntropyGraphs.AIC(::DCReM)
MaxEntropyGraphs.AICc(::DCReM)
MaxEntropyGraphs.BIC(::DCReM)
Base.length(::DCReM)
MaxEntropyGraphs.L_DCReM
MaxEntropyGraphs.∇L_DCReM!
MaxEntropyGraphs.∇L_DCReM_minus!
MaxEntropyGraphs.DCReM_iter!
MaxEntropyGraphs.set_xᵣ!(::DCReM)
MaxEntropyGraphs.set_yᵣ!(::DCReM)
MaxEntropyGraphs.Ĝ(::DCReM)
MaxEntropyGraphs.set_Ĝ!(::DCReM)
MaxEntropyGraphs.Ŵ(::DCReM)
MaxEntropyGraphs.set_Ŵ!(::DCReM)
MaxEntropyGraphs.σˣ(::DCReM)
MaxEntropyGraphs.set_σ!(::DCReM)
MaxEntropyGraphs.σʷ(::DCReM)
MaxEntropyGraphs.set_σʷ!(::DCReM)
MaxEntropyGraphs.precision(::DCReM)
MaxEntropyGraphs.A(::DCReM,::Int64,::Int64)
MaxEntropyGraphs.σₓ(::DCReM, ::Function)
```

The strength accessors and the model-expected `reciprocity`/`weighted_reciprocity` are documented with the [shared graph metrics](API.md).

