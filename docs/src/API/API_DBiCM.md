## Index

```@index
Pages = ["API_DBiCM.md"]
```

```@docs 
MaxEntropyGraphs.DBiCM
DBiCM(::T) where {T}
MaxEntropyGraphs.solve_model!(::DBiCM)
MaxEntropyGraphs.initial_guess(::DBiCM)
Base.rand(::DBiCM)
Base.rand(::DBiCM,::Int)
MaxEntropyGraphs.AIC(::DBiCM)
MaxEntropyGraphs.AICc(::DBiCM)
MaxEntropyGraphs.BIC(::DBiCM)
Base.length(::DBiCM)
MaxEntropyGraphs.L_DBiCM_reduced
MaxEntropyGraphs.∇L_DBiCM_reduced!
MaxEntropyGraphs.∇L_DBiCM_reduced_minus!
MaxEntropyGraphs.DBiCM_reduced_iter!
MaxEntropyGraphs._channel_id
MaxEntropyGraphs._channel
MaxEntropyGraphs._check_vertex
MaxEntropyGraphs._dbicm_expected_degree
MaxEntropyGraphs._solve_dbicm_channel
MaxEntropyGraphs.set_xᵣ!(::DBiCM)
MaxEntropyGraphs.Ĝ(::DBiCM)
MaxEntropyGraphs.set_Ĝ!(::DBiCM)
MaxEntropyGraphs.σˣ(::DBiCM)
MaxEntropyGraphs.set_σ!(::DBiCM)
MaxEntropyGraphs.precision(::DBiCM)
MaxEntropyGraphs.A(::DBiCM,::Int64,::Int64)
MaxEntropyGraphs.p⁺(::DBiCM,::Int64,::Int64)
MaxEntropyGraphs.p⁻(::DBiCM,::Int64,::Int64)
MaxEntropyGraphs.σₓ(::DBiCM, ::Function)
MaxEntropyGraphs.reciprocity(::DBiCM)
MaxEntropyGraphs.outdegree(::DBiCM, ::Int)
MaxEntropyGraphs.indegree(::DBiCM, ::Int)
MaxEntropyGraphs.degree(::DBiCM, ::Int)
```
