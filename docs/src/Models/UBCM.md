## Description
The Undirected Binary Configuration Model is a maximum-entropy null model for undirected networks. It is based on the idea of fixing the degree sequence of the network, i.e., the number of edges incident to each node, and then randomly rewiring the edges while preserving the degree sequence. The model assumes that the edges are unweighted and that the network is simple, i.e., it has no self-loops or multiple edges between the same pair of nodes. 

```@docs
UBCM
```

## Creation
```@docs
UBCM(::T) where {T}
```
## Obtaining the parameters

## Sampling the ensemble
```@docs
rand(::UBCM)
rand(::UBCM, ::Int)
```


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

