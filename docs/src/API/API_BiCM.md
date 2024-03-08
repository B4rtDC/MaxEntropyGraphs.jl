## Index

```@index
Pages = ["API_BiCM.md"]
```

```@docs
MaxEntropyGraphs.BiCM
BiCM(::T) where {T}
MaxEntropyGraphs.solve_model!(::BiCM)
MaxEntropyGraphs.initial_guess(::BiCM)
Base.rand(::BiCM)
Base.rand(::BiCM, ::Int)
MaxEntropyGraphs.AIC(::BiCM)
MaxEntropyGraphs.AICc(::BiCM)
MaxEntropyGraphs.BIC(::BiCM)
Base.length(::BiCM)
MaxEntropyGraphs.L_BiCM_reduced
MaxEntropyGraphs.∇L_BiCM_reduced!
MaxEntropyGraphs.∇L_BiCM_reduced_minus!
MaxEntropyGraphs.BiCM_reduced_iter!
MaxEntropyGraphs.set_xᵣ!(::BiCM)
MaxEntropyGraphs.set_yᵣ!(::BiCM)
MaxEntropyGraphs.Ĝ(::BiCM)
MaxEntropyGraphs.set_Ĝ!(::BiCM)
MaxEntropyGraphs.precision(::BiCM)
MaxEntropyGraphs.A(::BiCM,::Int64,::Int64)
MaxEntropyGraphs.f_BiCM(::BiCM)
MaxEntropyGraphs.biadjacency_matrix
MaxEntropyGraphs.project
```