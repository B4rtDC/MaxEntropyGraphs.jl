var documenterSearchIndex = {"docs":
[{"location":"derivedquantities/#Derived-Quantities","page":"Higher order metrics","title":"Derived Quantities","text":"","category":"section"},{"location":"GPU/#GPU","page":"GPU acceleration","title":"GPU","text":"","category":"section"},{"location":"GPU/","page":"GPU acceleration","title":"GPU acceleration","text":"Most methods can be translated to GPU computation directly thanks to the CUDA.jl environment.","category":"page"},{"location":"models/#Models","page":"Models","title":"Models","text":"","category":"section"},{"location":"models/","page":"Models","title":"Models","text":"The different models that are available are described below. Each of these models is a subtype of an AbstractMaxEntropyModel","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"MaxEntropyGraphs.AbstractMaxEntropyModel","category":"page"},{"location":"models/#MaxEntropyGraphs.AbstractMaxEntropyModel","page":"Models","title":"MaxEntropyGraphs.AbstractMaxEntropyModel","text":"AbstractMaxEntropyModel\n\nAn abstract type for a MaxEntropyModel. Each model has one or more structural constraints   that are fixed while the rest of the network is completely random.\n\n\n\n\n\n","category":"type"},{"location":"models/","page":"Models","title":"Models","text":"For each model there are multiple methods available to obtain the parameters that generate the maximum likelihood. Each of these models can be solved by using the solve! function.","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"solve!(model)","category":"page"},{"location":"models/","page":"Models","title":"Models","text":"Modules = [MaxEntropyGraphs]\nOrder = [:function]\nFilter = x -> occursin(\"UBCM,\"$(x)\"))","category":"page"},{"location":"models/#MaxEntropyGraphs.UBCM_∂ℒ_∂x!-NTuple{4, Vector}","page":"Models","title":"MaxEntropyGraphs.UBCM_∂ℒ_∂x!","text":"UBCM_∂ℒ_∂x!(F::Vector, x::Vector, κ::Vector, f::Vector)\n\nGradient of the likelihood function of the UBCM model using the x_i formulation. \n\nF value of gradients x value of parameters κ value of (reduced) degree vector f frequency associated with each value in κ \n\nSee also:\n\nUBCM\n\n\n\n\n\n","category":"method"},{"location":"models/#MaxEntropyGraphs.UBCM_∂ℒ_∂x_it!-Union{Tuple{T}, NTuple{4, Vector{T}}} where T","page":"Models","title":"MaxEntropyGraphs.UBCM_∂ℒ_∂x_it!","text":"UBCM_∂ℒ_∂x_it!(F::Vector, x::Vector, κ::Vector, f::Vector)\n\nIterative gradient of the likelihood function of the UBCM model using the x_i formulation. \n\nF value of function x value of parameters κ value of (reduce) degree vector f frequency associated with each value in κ \n\nSee also:\n\nUBCM\n\n\n\n\n\n","category":"method"},{"location":"models/#MaxEntropyGraphs.UBCM_∂ℒ_∂θ!-NTuple{4, Vector}","page":"Models","title":"MaxEntropyGraphs.UBCM_∂ℒ_∂θ!","text":"UBCM_∂ℒ_∂x!(F::Vector, x::Vector, κ::Vector, f::Vector)\n\nGradient of the likelihood function of the UBCM model using the xi formulation. Used to generate a function (F, x) -> UBCM_∂ℒ_∂x!(F, x, κ, f) by a UBCM instance.\n\nF value of gradients x value of parameters κ value of (reduced degree vector) f frequency associated with each value in κ \n\nsee also:\n\nUBCM\n\n\n\n\n\n","category":"method"},{"location":"models/#MaxEntropyGraphs.fffff-Tuple{Any}","page":"Models","title":"MaxEntropyGraphs.fffff","text":"fffff(x)\n\nhelper function for UBCM gradient computation\n\n\n\n\n\n","category":"method"},{"location":"models/#MaxEntropyGraphs.solve!-Tuple{UBCM}","page":"Models","title":"MaxEntropyGraphs.solve!","text":"solve!model::UBCM; , kwargs...)\n\nSolve the equations of a UBCM model to obtain the maximum likelihood parameters.\n\nSee also UBCM\n\n\n\n\n\n","category":"method"},{"location":"models/#UBCM","page":"Models","title":"UBCM","text":"","category":"section"},{"location":"models/","page":"Models","title":"Models","text":"Generate a Undirected Binary Configuration Model.","category":"page"},{"location":"models/#DBCM","page":"Models","title":"DBCM","text":"","category":"section"},{"location":"models/","page":"Models","title":"Models","text":"Generate a Directed Binary Configuration Model.","category":"page"},{"location":"#MaxEntropyGraphs.jl","page":"Home","title":"MaxEntropyGraphs.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for MaxEntropyGraphs.jl","category":"page"}]
}
