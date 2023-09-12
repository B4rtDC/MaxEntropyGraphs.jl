# Simulation
The analytical method can quickly become expensive to compute since each element in the adjacency matrix has a non-zero value. It is also possible to obtain an estimate of both the metric and its standard deviation by generating a large number of random graphs from the ensemble and by computing the value of the metric(s) for each graph. This can also allow you to compute the z-scores of the metric, but some caution is advised. Using the analytical method, we do consider the actual underlying distribution of the metric to combine the expected value and standard deviation. This contrasts with the sample, where we simply apply the mean and standard deviation to the vector of computed metrics. Also, considering the z-score and interpreting statistical significance if ``|z| \ge 3`` implies that the underlying distribution of the metric follows a standard normal distribution, which is not always the case.

## Examples
Some examples using built-in function of package are listed below:
* [Assortativity in the UBCM](@ref Assortativity_simulation)
* [Motif significance in the UBCM](@ref Motif_simulation)

### [Assortativity in the UBCM](@id Assortativity_simulation)
Let us consider the UBCM applied to the Zachary Karate Club network. We want to analyse if the assortativity of each node (measured by its ANND) is statistically significant from what one would expect under the null model.

First, we define the network and the associated UBCM model.
```jldoctest UBCM_z_demo; output = false
using Graphs
using MaxEntropyGraphs

# define the network
G = MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate)
# generate a UBCM model from the karate club network
model = UBCM(G); 
# compute the maximum likelihood parameters
solve_model!(model); 
# compute and set the expected adjacency matrix
set_Ĝ!(model); 
# compute and set the standard deviation of the adjacency matrix
set_σ!(model); 
nothing

# output


```

### [Motif significance in the UBCM](@id Motif_simulation)
TO DO