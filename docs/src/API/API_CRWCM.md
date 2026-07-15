## Index

```@index
Pages = ["API_CRWCM.md"]
```

```@docs 
MaxEntropyGraphs.CRWCM
CRWCM(::T) where {T}
MaxEntropyGraphs.solve_model!(::CRWCM)
MaxEntropyGraphs.initial_guess(::CRWCM)
Base.rand(::CRWCM)
Base.rand(::CRWCM,::Int)
MaxEntropyGraphs.AIC(::CRWCM)
MaxEntropyGraphs.AICc(::CRWCM)
MaxEntropyGraphs.BIC(::CRWCM)
Base.length(::CRWCM)
MaxEntropyGraphs.L_CRWCM
MaxEntropyGraphs.∇L_CRWCM!
MaxEntropyGraphs.∇L_CRWCM_minus!
MaxEntropyGraphs.CRWCM_iter!
MaxEntropyGraphs.set_xᵣ!(::CRWCM)
MaxEntropyGraphs.set_yᵣ!(::CRWCM)
MaxEntropyGraphs.set_zᵣ!(::CRWCM)
MaxEntropyGraphs.Ĝ(::CRWCM)
MaxEntropyGraphs.set_Ĝ!(::CRWCM)
MaxEntropyGraphs.Ŵ(::CRWCM)
MaxEntropyGraphs.set_Ŵ!(::CRWCM)
MaxEntropyGraphs.σˣ(::CRWCM)
MaxEntropyGraphs.set_σ!(::CRWCM)
MaxEntropyGraphs.σʷ(::CRWCM)
MaxEntropyGraphs.set_σʷ!(::CRWCM)
MaxEntropyGraphs._cov_dyads(::CRWCM)
MaxEntropyGraphs._covʷ
MaxEntropyGraphs.precision(::CRWCM)
MaxEntropyGraphs.A(::CRWCM,::Int64,::Int64)
MaxEntropyGraphs.p⭢(::CRWCM,::Int64,::Int64)
MaxEntropyGraphs.p⭠(::CRWCM,::Int64,::Int64)
MaxEntropyGraphs.p⭤(::CRWCM,::Int64,::Int64)
MaxEntropyGraphs.σₓ(::CRWCM, ::Function)
```

The reciprocal degree/strength accessors and the model-expected `reciprocity`/`weighted_reciprocity` are documented with the [shared graph metrics](API.md).

