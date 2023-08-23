# MaxEntropyGraphs.jl


# Overview
The goal of the *MaxEntropyGraphs.jl* package is to group the various maximum-entropy null models for network randomization and make them available to the Julia community in a single package. This work was in part inspired by the [Maximum Entropy Hub](https://meh.imtlucca.it), but unlike the latter, this package works in an integrated way with the exisiting Julia ecosystem for handling graphs, optimization tools and numerical solvers and groups all models in a single framework.

The package provides the following functionalities:
* Computing the likelihood maximizing parameters for a broad set of network models (cf. `Models` section of the documentation).
* Sampling of networks from a network ensemble once the parameters have been computed.
* Analytically computing ensemble averages and their standard deviations (for a subset of models).
* Running motif based analysis (for a subset of models).
* Bipartite network projections with statistical significance analysis (for a subset of models).

Each network models can be solved in different ways, with a fixed-point method typically being the fastest and a Newton-based method being the slowest. Depending on the complexity of the network model, some solvers might not always converge.

# Installation
Assuming that you already have Julia correctly installed, installation is straightforward. 
It suffices to import MaxEntropyGraphs.jl in the standard way:
```julia
using Pkg
Pkg.add("MaxEntropyGraphs")
```
or enter the Pkg mode by hitting ```]```, and then run the following command:
```Julia
pkg> add MaxEntropyGraphs
```

# The maximum entropy principle for networks
The maximum entropy principle is a general principle in statistical mechanics that states that, given a set of constraints on a system, the probability distribution that maximizes the entropy subject to those constraints is the one that should be used to describe the system. The maximum entropy principle provides a way of constructing null models that are as unbiased or as uncertain as possible, subject to the available information, and that can be used to make statistical inferences about the underlying processes that generate the observed networks.

In the context of networks, the idea is to specify a set of constraints that capture some of the structural features of the network, such as the degree sequence, the clustering coefficient, or the degree-degree correlations, and then to generate random networks that satisfy those constraints while being as unbiased or as uncertain as possible with respect to other structural features. The resulting null models can be used to test whether the observed structural features of a real-world network are statistically significant or whether they can be explained by chance alone. The principle and its applications are explained in detail in [[1]](#1).



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
```

