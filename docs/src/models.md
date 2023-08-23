# Overview
## Common interface
All models share a common interface:
* A model can be instantiated either from a substype of an ```AbstractGraph``` or by the model's constraint vector(s). 
* The parameters of a model can be computed with ```solve_model!```.
* The expected adjacency matrix can be obtained with ```Ĝ```, where applicable the weights can be obtained with ```Ŵ```.
* A random network can be sampled from the ensemble with ```rand```
Please refer to the page of each specific model for more details. Each of these models is a subtype of an `AbstractMaxEntropyModel`


```@docs
AbstractMaxEntropyModel
```


# Solution methods
Computing the parameters of a model can be done with different approaches. Either by running an optimisation algorithm on the Loglikelihood of the model (and thus implicitely solving a system of equations for the gradient of the loglikelihood of the model) [[1]](#1) or by using a fixed point approach [[2]](#2). In both cases, we have also included the acceleration method that was proposed in [[2]](#2) for nodes sharing the same (pair of) constraints.


# Sampling
We have extend ```Base.rand``` to accept substypes of `::AbstractMaxEntropyModel`. When working with larger network, it might not always be desirable to keep the entire expected adjacency matrix in memory, so by default this option is not used. Specifying a number of samples will return a vector of the appropriate substype of  `::AbstractGraph`. Multithreading will be used if available to generate multiple graphs in parallel.

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
<a href="https://doi.org/10.1038/s41598-021-93830-4</a>
</li>
</ul>

```

