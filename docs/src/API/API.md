## Index

```@index
Pages = ["API.md"]
```

## Global
```@docs
MaxEntropyGraphs
AbstractMaxEntropyModel
MaxEntropyGraphs.ConvergenceError

```

## Convergence diagnostics
A solved model reproduces its imposed constraints only up to the solver's stopping rule, and the
solver tolerances are *not* expressed in constraint units. `ftol` bounds the fixed-point increment in
parameter space, and `g_tol` is a stopping criterion for the gradient-based methods rather than a
guarantee. `constraint_residual` reports what a solve actually achieved, in the units of the
constraints themselves.

This matters most for the two-step weighted models (`CReM`, `DCReM`, `CRWCM`), whose parameters carry
units of one over weight: there the constraint residual is bounded by roughly `ftol * max(sᵢ/θᵢ)`,
which grows as the *square* of the weight scale.

```@docs
constraint_residual
```

## Utility fuctions
Some specific utility functions are made available:

```@docs
MaxEntropyGraphs.log_nan
MaxEntropyGraphs.np_unique_clone
```


## Small graph constructors
```@docs
MaxEntropyGraphs.taro_exchange
MaxEntropyGraphs.rhesus_macaques
MaxEntropyGraphs.chesapeakebay
MaxEntropyGraphs.everglades
MaxEntropyGraphs.florida
MaxEntropyGraphs.littlerock
MaxEntropyGraphs.maspalomas
MaxEntropyGraphs.stmarks
MaxEntropyGraphs.parse_konect
MaxEntropyGraphs.readpajek
MaxEntropyGraphs.corporateclub
```



## Graph metrics
```@docs
MaxEntropyGraphs.degree
MaxEntropyGraphs.outdegree
MaxEntropyGraphs.indegree
MaxEntropyGraphs.strength
MaxEntropyGraphs.outstrength
MaxEntropyGraphs.instrength
MaxEntropyGraphs.ANND
MaxEntropyGraphs.ANND_out
MaxEntropyGraphs.ANND_in
MaxEntropyGraphs.wedges
MaxEntropyGraphs.triangles
MaxEntropyGraphs.squares
MaxEntropyGraphs.a⭢
MaxEntropyGraphs.a⭠
MaxEntropyGraphs.a⭤
MaxEntropyGraphs.a̲
MaxEntropyGraphs.M1
MaxEntropyGraphs.M2
MaxEntropyGraphs.M3
MaxEntropyGraphs.M4
MaxEntropyGraphs.M5
MaxEntropyGraphs.M6
MaxEntropyGraphs.M7
MaxEntropyGraphs.M8
MaxEntropyGraphs.M9
MaxEntropyGraphs.M10
MaxEntropyGraphs.M11
MaxEntropyGraphs.M12
MaxEntropyGraphs.M13
MaxEntropyGraphs.motifs
MaxEntropyGraphs.V_motifs
MaxEntropyGraphs.V_PB_parameters
```

## Reciprocity metrics
```@docs
MaxEntropyGraphs.reciprocity
MaxEntropyGraphs.weighted_reciprocity
MaxEntropyGraphs.nonreciprocated_outdegree
MaxEntropyGraphs.nonreciprocated_indegree
MaxEntropyGraphs.reciprocated_degree
MaxEntropyGraphs.nonreciprocated_outstrength
MaxEntropyGraphs.nonreciprocated_instrength
MaxEntropyGraphs.reciprocated_outstrength
MaxEntropyGraphs.reciprocated_instrength
```

## Triadic fluxes, intensities and sampling-based significance
```@docs
MaxEntropyGraphs.motif_fluxes
MaxEntropyGraphs.motif_flux
MaxEntropyGraphs.motif_intensities
MaxEntropyGraphs.ensemble_zscores
MaxEntropyGraphs.motif_zscores
MaxEntropyGraphs.flux_zscores
```


