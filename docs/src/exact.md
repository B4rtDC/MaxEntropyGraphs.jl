# Analytical
The maximum likelihood method can be used to compute the expected value and the standard deviation of any metric that is based on the adjacency matrix. Depending on the underlying model, some details change, but the principle remains. This formalism allows us to computed z-scores and assess which topological properties are consistent with their randomized value within a statistical error, and which deviate significantly from the null model expectation. In the latter case, the observed property cannot be traced back to the constraints, and therefore requires additional explanations or generating mechanisms besides those required in order to explain the constraints themselves. 

Some topological properties are available in the package default (e.g. `ANND` and different network motifs), but you can define an additional metrics as well. This allows you to obtain both the expected value and the standard deviation of any matrix in a standardised way. In the expression of the variance of a topological property ``X``, we find ``\frac{\partial X}{\partial a_{ij}}``. We use leverage Julia's autodiff capabilities to compute these terms. If desired, you can always compute the gradient of a specific metric by hand and implement it yourself as well. The downside of using this approach is that you need the complete adjacency matrix, so this is not suited for the analysis of very large graphs due to memory constraints. Depending on the size of the problem, different autodiff techniques can give different performance results. You might want to experiment a bit with this for your own use case (some examples are provided as well).  


## Expected value
Using the maximum likelihood method method the expected value for any topological property `X` can be computed from the expected values in the adjacency matrix of the graph `G` (this approximation ignores the second and higher order terms in the multidimensional Taylor expansion of `X`).

```math
X \left( G \right)  =  X \left( \left< G \right> \right)
```

## Variance
The variance of a topological property `S` can be written as follows

```math
\sigma ^{2} \left[ X \right] = \sum_{i,j} \sum_{t,s} \sigma \left[g_{ij}, g_{ts} \right] \left(  \frac{\partial X}{\partial g_{ij}} \frac{\partial X}{\partial g_{ts}}  \right)_{G = \left< G \right>}
```

where

```math
\sigma \left[ g_{ij}, g_{ts} \right] = \left< g_{ij}g_{ts}\right> - \left< g_{ij}\right>\left< g_{ts}\right>
```

Using the appropriate expressions for 
``\left< g_{ij} \right>`` and ``\left< g_{ij}\right>``
(depending on the model considered, cf. examples), a highly reliable estimate for the variance of the metric can be obtained.

## Examples
Some examples using built-in function of package are listed below:
* [Assortativity in the UBCM](@ref Assortativity_analytical)
* [Motif significance in the UBCM](@ref Motif_analytical)

### [Assortativity in the UBCM](@id Assortativity_analytical)
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

Now we can define our specific metric. Before computing the z-score for all nodes, we illustrate the process for a single node. We use `X` as variable name for our metric. Defining methods for `X` in such a way that it can  
with both an `AbstractArray` and an `AbstractGraph` is recommended, but not necessary.
```jldoctest UBCM_z_demo; output = false
# We consider the ANND of node 1 as our metric
node_id = 1
X = A -> ANND(A, node_id, check_dimensions=false, check_directed=false);    
# Expected value under the null model
X_expected = X(model.Ĝ)
# Expected standard deviation under the null model
X_std = σₓ(model, X)
# Observed value (using the underlying network)
X_observed = X(model.G)
# compute z-score
z_X = (X_observed - X_expected) / X_std

# output

-0.8694874720776825
```

In the same way, we can compute the z-score for every node:
```jldoctest UBCM_z_demo; output = false
# Observed value
ANND_obs = [ANND(G, i) for i in vertices(G)]
# Expected values
ANND_exp = [ANND(model.Ĝ, i) for i in vertices(G)]
# Standard deviation
ANND_std = [σₓ(model, A -> ANND(A, i, check_dimensions=false, check_directed=false)) for i in vertices(G)]
# Z-score
Z_ANND = (ANND_obs - ANND_exp) ./ ANND_std;

# output

34-element Vector{Float64}:
 -0.8694874720776825
 -0.5292199109039629
 -0.08739363417954547
 -0.12970352719726422
 -0.2319316593015868
 -0.4799299169446237
 -0.4799299169446234
  0.24501059537335462
  0.718367288006965
  0.41374043075904055
  ⋮
 -0.6647918097128745
  0.09021959133602485
 -0.026842096745887254
  0.24902406337762192
  0.018466685273986393
  0.33562815941310226
  0.2173026706708909
 -0.640873675884825
 -1.2188220468342164
```

### [Motif significance in the UBCM](@id Motif_analytical)
Let us consider the DBCM applied to the Chesapeake Bay foodweb. We want to analyse if any of the different network motifs is statistically significant of what one would expect uner the null model.

First, we define the network and the associated UBCM model.
```jldoctest DBCM_z_demo; output = false
using Graphs
using MaxEntropyGraphs
import Statistics: mean, std

# define the network
G = chesapeakebay()
# extract its adjacency matrix
A = adjacency_matrix(G)
# generate a UBCM model from the karate club network
model = DBCM(G); 
# compute the maximum likelihood parameters
solve_model!(model); 
# compute and set the expected adjacency matrix
set_Ĝ!(model); 
# compute and set the standard deviation of the adjacency matrix
set_σ!(model); 
nothing

# output


```

We want to know the values of `M1`, ..., `M13`. These are network-wide measures. 
```jldoctest DBCM_z_demo; output = false
# compute the observed motif counts
motifs_observed = [@eval begin $(f)(A) end for f in MaxEntropyGraphs.directed_graph_motif_function_names];
# Expected value under the null model
motifs_expected = [@eval begin $(f)(model) end for f in MaxEntropyGraphs.directed_graph_motif_function_names];
# Expected standard deviation under the null model
motifs_std = [@eval begin  σₓ(model, $(f), gradient_method=:ForwardDiff) end for f in MaxEntropyGraphs.directed_graph_motif_function_names];
# compute the z-score
motifs_z = (motifs_observed .- motifs_expected) ./ motifs_std

# output

13-element Vector{Float64}:
 -0.006078466706358218
  0.4517818796564363
 -0.5539006800258
 -0.3424785093439874
 -2.069533811081406
 -0.01782802382498426
  1.0292421736598207
 -0.859327218816445
 -3.1684812828517512
  1.3101076515056505
  0.8379127965150848
  4.862910567915945
 -0.8148586665737436
```
