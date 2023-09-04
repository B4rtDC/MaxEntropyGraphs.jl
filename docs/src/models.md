# Overview

## Common interface
All models share a common interface:
* A model (```<:AbstractMaxEntropyModel```) can be instantiated either from a graph (```<:AbstractGraph```) or by the model's constraint vector(s). 
* The parameters of a model can be computed with ```solve_model!```.
* The expected adjacency matrix can be obtained with ```Ĝ```, where applicable the weights can be obtained with ```Ŵ```.
* A random network can be sampled from the ensemble with ```rand```
Please refer to the page of each specific model for more details.


## Solution methods
Computing the parameters of a model can be done with different approaches. Either by running an optimisation algorithm on the Loglikelihood of the model (and thus implicitely solving a system of equations for the gradient of the loglikelihood of the model) [[1]](#1) or by using a fixed point approach [[2]](#2). In both cases, we have also included the acceleration method that was proposed in [[2]](#2) for nodes sharing the same (pair of) constraints.


## Sampling
We have extend ```Base.rand``` to accept substypes of `::AbstractMaxEntropyModel`. When working with larger network, it might not always be desirable to keep the entire expected adjacency matrix in memory, so by default this option is not used. Specifying a number of samples will return a vector of the appropriate substype of  `::AbstractGraph`. Multithreading will be used if available to generate multiple graphs in parallel.



## Differences and similarities with the NEMtropy package
* We use the [JuliaGraphs](https://juliagraphs.org/) ecosystem for everything network related, whereas NEMtropy requires you to work with degree sequences, adjacency matrices or edge lists. These option are also available, either directly (in the case of degree sequences) or indirectly (by passing through the JuliaGraphs ecosystem).
* To obtain the maximum likelihood parameters of a model we use either:
    - the well-established [Optimization.jl](https://github.com/SciML/Optimization.jl) package (this maximises the loglikelihood of the networks ensemble). Working this way uses automated differentiation, but explicit non-allocating gradient functions are also provided for the different models.
    - [NLsolve.jl](https://github.com/JuliaNLSolvers/NLsolve.jl#anderson-acceleration)'s Anderson acceleration for the fixed point methods proposed in [[4]](#4) (cf. documentation/examples). The iterative methods are non-allocating, so they are orders of magnitude faster than the Python implementation.
* We have also maintained the different options for the initial values for each method (cf. `initial_guess(::AbstractMaxEntropyModel)`).
* By making use of the automatic differentiation capabilities of Julia, we can:
    - approximation the gradient of the likelihood function of the graphs
    - compute the gradient of any metric with respect to its adjacency matrix without having to compute these by hand and implement the partial derivative for each possible metric (cf. examples for more details on this).
* For both the models and the computing functions we make a clear distinction between the likelihood maximising paramers and the variable substitution (e.g. $\theta_i \leftrightarrow x_i = e^{-\theta_i}$ for the UBCM). Doing so makes the code more readable because it is closer to the mathematical formulation.
* Sampling from an `<:AbstractMaxEntropyModel` will generate a corresponding subtype of  `<:AbstractGraph` from the [JuliaGraphs](https://juliagraphs.org/) ecosystem. The methods that are available in `Graphs.jl`, `SimpleWeightedGraphs` etc. have been extended for the different maximum entropy models wherever applicable. 



_References_

```@raw html
<ul>
<li>
<a id="1">[1]</a> 
Squartini, Tiziano and Garlaschelli, Diego. <!--  author(s) --> 
<em>"Maximum-Entropy Networks: Pattern Detection, Network Reconstruction and Graph Combinatorics"</em> <!--  title --> 
Springer-Verlag GmbH; 1st ed. 2017 edition (25 Dec. 2017). <!--  publisher(s) --> 
<a href="https://link.springer.com/book/10.1007/978-3-319-69438-2">https://link.springer.com/book/10.1007/978-3-319-69438-2</a>
</li>
</ul>

<ul>
<li>
<a id="2">[2]</a> 
Vallarano, Nicolò and Bruno, Matteo and Marchese, Emiliano and Trapani, Giuseppe and Saracco, Fabio and Cimini, Giulio and Zanon, Mario and Squartini, Tiziano. <!--  author(s) --> 
<em>"Fast and scalable likelihood maximization for Exponential Random Graph Models with local constraints"</em> <!--  title --> 
Scientific Reports 11, 2021. <!--  publisher(s) --> 
<a href="https://doi.org/10.1038/s41598-021-93830-4">https://doi.org/10.1038/s41598-021-93830-4</a>
</li>
</ul>
```

