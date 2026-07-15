## Index

```@index
Pages = ["API_RBCM.md"]
```

```@docs 
MaxEntropyGraphs.RBCM
RBCM(::T) where {T}
MaxEntropyGraphs.solve_model!(::RBCM)
MaxEntropyGraphs.initial_guess(::RBCM)
Base.rand(::RBCM)
Base.rand(::RBCM,::Int)
MaxEntropyGraphs.AIC(::RBCM)
MaxEntropyGraphs.AICc(::RBCM)
MaxEntropyGraphs.BIC(::RBCM)
Base.length(::RBCM)
MaxEntropyGraphs.L_RBCM_reduced
MaxEntropyGraphs.∇L_RBCM_reduced!
MaxEntropyGraphs.∇L_RBCM_reduced_minus!
MaxEntropyGraphs.RBCM_reduced_iter!
MaxEntropyGraphs.log1pexpsum
MaxEntropyGraphs.set_xᵣ!(::RBCM)
MaxEntropyGraphs.set_yᵣ!(::RBCM)
MaxEntropyGraphs.set_zᵣ!(::RBCM)
MaxEntropyGraphs.Ĝ(::RBCM)
MaxEntropyGraphs.set_Ĝ!(::RBCM)
MaxEntropyGraphs.σˣ(::RBCM)
MaxEntropyGraphs.set_σ!(::RBCM)
MaxEntropyGraphs._dyadic_probability_matrices
MaxEntropyGraphs._cov_dyads(::RBCM)
MaxEntropyGraphs.precision(::RBCM)
MaxEntropyGraphs.A(::RBCM,::Int64,::Int64)
MaxEntropyGraphs.p⭢(::RBCM,::Int64,::Int64)
MaxEntropyGraphs.p⭠(::RBCM,::Int64,::Int64)
MaxEntropyGraphs.p⭤(::RBCM,::Int64,::Int64)
MaxEntropyGraphs.p∅(::RBCM,::Int64,::Int64)
MaxEntropyGraphs.σₓ(::RBCM, ::Function)
```

The reciprocal degree accessors (`nonreciprocated_outdegree`, `nonreciprocated_indegree`, `reciprocated_degree`) and the model-expected `reciprocity` are documented with the [shared graph metrics](API.md).

