# Metrics
Metrics are a crucial part of network analysis. They provide a way to quantify and understand the structure and behavior of a network. In the context of graph theory, metrics can be used to analyze various properties of a network, such as its connectivity, centrality, community structure and the presence of motifs. 

Within the package, there are two main approaches to compute metrics in network analysis: analytical and simulation. 

## Analytical
The analytical approach involves using mathematical methods to compute the expected value and the standard deviation of any metric that is based on the adjacency matrix. This approach allows us to compute z-scores and assess which topological properties are consistent with their randomized value within a statistical error, and which deviate significantly from the null model expectation. 

The analytical method can be computationally expensive for large graphs due to the need for:
- the complete adjacency matrix
- the gradient of the metric with respect to the adjacency matrix. 

However, it provides a highly reliable estimate for the variance of the metric.  For more details, see the [Analytical](./exact.html) section.

## Simulation
The simulation approach involves generating a large number of random graphs from the ensemble and computing the value of the metric(s) for each graph. This method can be more efficient for large graphs where the analytical method becomes computationally expensive. For more details, see the [Simulation](./simulated.html) section.

## Examples
The same examples are provided for both approaches.