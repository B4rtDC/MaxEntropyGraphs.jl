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