# Analytical
## Expected value
Using the maximum likelihood method method the expected value for any topological property `X` can be computed from the expected values in the adjacency matrix of the graph `G` (this approximation ignores the second and higher order terms in the multidimensional Taylor expansion of `X`).

``math
X \left( G \right)  =  X \left( \left< G \right> \right)
``

## Variance
The variance of a topological property `S` can be written as follows

``math
\sigma ^{2} \left[ X \right] = \sum_{i,j} \sum_{t,s} \sigma \left[g_{ij}, g_{ts} \right] \left(  \frac{\partial X}{\partial g_{ij}} \frac{\partial X}{\partial g_{ts}}  \right)_{G = \left< G \right>}
``

where

``
\sigma \left[ g_{ij}, g_{ts} \right] = \left< g_{ij}g_{ts}\right> - \left< g_{ij}\right>\left< g_{ts}\right>
``

Using the appropriate expressions for 
``\left< g_{ij} \right>``
(depending on the model considered, cf. examples), a highly reliable estimate for the variance of the metric can be obtained.