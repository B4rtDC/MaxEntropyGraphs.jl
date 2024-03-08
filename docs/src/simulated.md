# Simulation
The analytical method can quickly become expensive to compute since each element in the adjacency matrix has a non-zero value. It is also possible to obtain an estimate of both the metric and its standard deviation by generating a large number of random graphs from the ensemble and by computing the value of the metric(s) for each graph. This can also allow you to compute the z-scores of the metric, but some caution is advised. Using the analytical method, we do consider the actual underlying distribution of the metric to combine the expected value and standard deviation. This contrasts with the sample, where we simply apply the mean and standard deviation to the vector of computed metrics. Also, considering the z-score and interpreting statistical significance if ``|z| \ge 3`` implies that the underlying distribution of the metric follows a standard normal distribution, which is not always the case.

## Examples
Some examples using built-in function of package are listed below:
* [Assortativity in the UBCM](@ref Assortativity_simulation)
* [Motif significance in the DBCM](@ref Motif_simulation)
* [Significance of V-motifs and projection in the BiCM](@ref BiCM_Vmotifs_simulation)

### [Assortativity in the UBCM](@id Assortativity_simulation)
Let us consider the UBCM applied to the Zachary Karate Club network. We want to analyze if the assortativity of each node (measured by its ANND) is statistically significant from what one would expect under the null model.

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

### [Motif significance in the DBCM](@id Motif_simulation)
Let us consider the DBCM applied to the Chesapeake Bay foodweb. We want to analyze if any of the different network motifs is statistically significant of what one would expect under the null model. We want to know the values of `M1`, ..., `M13`. These are network-wide measures. 

First, we define the network and the associated UBCM model.
```jldoctest DBCM_simulation_demo; output = false
using Graphs
using MaxEntropyGraphs
import Statistics: mean, std

# define the network
G = chesapeakebay();
# extract its adjacency matrix
A = adjacency_matrix(G);
# generate a UBCM model from the karate club network
model = DBCM(G); 
# compute the maximum likelihood parameters
solve_model!(model); 
# compute and set the expected adjacency matrix
set_Ĝ!(model); 
# compute and set the standard deviation of the adjacency matrix
set_σ!(model); 
# compute the observed motif counts
motifs_observed = [@eval begin $(f)(A) end for f in MaxEntropyGraphs.directed_graph_motif_function_names];
nothing

# output


```

Now we need to generate a sample from the network ensemble so that we can compute the sample mean and standard deviation for each motif.


```jldoctest DBCM_simulation_demo; output = false
# Get sample adjacency matrix
S = adjacency_matrix.(rand(model, 100));
# compute the motifs
motif_counts_S = hcat(map(s -> [@eval begin $(f)($s) end for f in MaxEntropyGraphs.directed_graph_motif_function_names], S)...);
# compute the sample mean and standard deviation
motifs_mean_S = reshape(mean(motif_counts_S, dims=2),:);
motifs_std_S = reshape(std(motif_counts_S, dims=2),:);
# compute the z-score
motifs_z_S = (motifs_observed .- motifs_mean_S) ./ motifs_std_S;
nothing

# output


```


### [Significance of V-motifs and projection in the BiCM](@id BiCM_Vmotifs_simulation)
Let us consider the BiCM applied to the [corporate club membership network](http://konect.cc/networks/brunson_club-membership/). This bipartite network is composed of 25 persons and 15 social organizations. An edge between a person and a social organization shows that the person is a member. We want to obtain the projection of this network on both the person and the social organization layer. We also want to determine if any of these connections in the projected networks are statistically significant under the BiCM null model. This is done by evaluating if the number of organizations that two users have in common is more than what one would expect under the null model.

First, we define the network and obtain its weighted projection. This tells us how often a specific connection was observed the real network.
```jldoctest BiCM_projection_demo_sampling; output = false
using MaxEntropyGraphs
using Graphs
using MultipleTesting

# define the network
G = corporateclub();
# project the model on its layers
G_persons, G_organizations = project(G, layer=:bottom, method=:weighted), project(G, layer=:top, method=:weighted);

# output

```

We then define a BiCM, compute its likelihood maximizing parameters, generate a sample and the projections of the sample
```jldoctest BiCM_projection_demo_sampling; output = false
# generate a BiCM from the corporate club network
model = BiCM(G); 
# compute the maximum likelihood parameters
solve_model!(model); 
# sample
S = rand(model, 100);
# projected samples
S_persons, S_organizations = project.(S, layer=:bottom, method=:weighted), project.(S, layer=:bottom, method=:weighted)

# output

```

For each edge in the original network, we can now compare how its weight compares to the distribution of weights in the random sample and determine their empirical p-values.
```jldoctest BiCM_projection_demo_sampling; output = true
# determine the empirical pvalues
p = zeros(length(edges(G_persons)))
for (i,e) in enumerate(edges(G_persons))
    e_w   = e.weight
    e_w_s = map(s -> s.weights[e.src, e.dst], S_persons)
    p[i] = (sum(e_w_s .> e_w) + 1) / (length(e_w_s) + 1)
end
# correct p-values for multiple testing
p_corrected = adjust(p, BenjaminiHochberg())
# get significant edges
filter(x -> x[1] < α, collect(zip(p_corrected, edges(G_persons))))

# output
Tuple{Float64, SimpleWeightedGraphs.SimpleWeightedEdge{Int64, Int64}}[]
```

We come to the conclusion that there are no statistically significant edges in the projected network. This is the same conclusions that was obtained by using `project(model)` directly.