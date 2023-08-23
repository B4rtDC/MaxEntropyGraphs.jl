## UBCM
The Undirected Binary Configuration Model is a maximum-entropy null model for undirected networks. It is based on the idea of fixing the degree sequence of the network, i.e., the number of edges incident to each node, and then randomly rewiring the edges while preserving the degree sequence. The model assumes that the edges are unweighted and that the network is simple, i.e., it has no self-loops or multiple edges between the same pair of nodes. 

```@docs
MaxEntropyGraphs.UBCM
```
This type can be instantiated with the following methods: