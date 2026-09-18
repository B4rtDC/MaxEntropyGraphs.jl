```@meta
DocTestSetup = quote
    using MaxEntropyGraphs
end
```

# DBiCM

## Model description
The Directed Bipartite Configuration Model is the maximum-entropy null model for a **bipartite network
whose edges carry a direction**, constraining all four degree sequences: the out- and in-degrees of the
⊥ layer and those of the ⊤ layer.

A directed bipartite graph has two disjoint link channels between the layers, each an ``N_⊥ × N_⊤``
binary matrix:

* channel ``+`` (`:to_top`, ``⊥ → ⊤``): ``B^+_{iα} = 1`` iff ``⊥_i → ⊤_α``
* channel ``-`` (`:to_bottom`, ``⊤ → ⊥``): ``B^-_{iα} = 1`` iff ``⊤_α → ⊥_i``

Collected per entry, the Hamiltonian reads
``\sum_{i,α} (α_i + δ_α) B^+_{iα} + \sum_{i,α} (β_i + γ_α) B^-_{iα}``. **No term couples the two
channels**, so the partition function factorises over all ``2 N_⊥ N_⊤`` entries and the model *is*
two independent [`BiCM`](@ref MaxEntropyGraphs.BiCM)s — one on the sequence pair
``(d_⊥^{out}, d_⊤^{in})``, one on ``(d_⊥^{in}, d_⊤^{out})``.

| Description | Formula |
| ----------- | ------- |
| Constraints | ``k^{out}_i, k^{in}_i`` (⊥ layer), ``h^{out}_α, h^{in}_α`` (⊤ layer) |
| Hamiltonian | ``H(B) = \sum_i (α_i k^{out}_i + β_i k^{in}_i) + \sum_α (γ_α h^{out}_α + δ_α h^{in}_α)`` |
| ``\langle B^+_{iα}\rangle`` | ``\dfrac{x^{out}_i w^{in}_α}{1 + x^{out}_i w^{in}_α}`` |
| ``\langle B^-_{iα}\rangle`` | ``\dfrac{y^{in}_i z^{out}_α}{1 + y^{in}_i z^{out}_α}`` |
| ``\sigma^2[B^{\pm}_{iα}]`` | ``p^{\pm}_{iα}(1 - p^{\pm}_{iα})`` |
| ``\sigma[B^+_{iα}, B^-_{iα}]`` | ``0`` — the two channels are exactly independent |
| ``\sigma^{*}(X)`` | ``\sqrt{\sum (\sigma_{iα} \partial X/\partial B_{iα})^2}`` |

Because ``B^+_{iα}`` and ``B^-_{iα}`` are independent, the DBiCM carries **no reciprocity coupling**:
it is the bipartite analogue of the [`DBCM`](@ref MaxEntropyGraphs.DBCM), not of the
[`RBCM`](@ref MaxEntropyGraphs.RBCM). It will generally *under*-predict the observed reciprocity, and
that gap is informative — it is precisely what a reciprocity-aware model would absorb.

### Class reduction

Each of the four sequences reduces to unique-value classes **independently**. This is forced rather
than merely convenient: the [`DBCM`](@ref MaxEntropyGraphs.DBCM) needs joint ``(out, in)`` classes only
because its sums carry the self-exclusion ``i \neq j``, so a node's out-equation depends on its own
class. Bipartite sums range over two disjoint vertex sets and have no such term, and the cross-channel
second derivatives of the log-likelihood vanish identically, so the joint pair never enters at all.

### Directions and layers

Two selectors appear throughout, and they mean different things:

* `channel` is **absolute** — it names the direction links travel. `:to_top` is ``⊥ → ⊤``,
  `:to_bottom` is ``⊤ → ⊥``. Aliases `:⁺`/`:⁻` are accepted.
* `layer` (in the projection functions) is **relative** — it selects which side a projection lands on.

`:out`/`:in` are deliberately *not* accepted as a `channel`, because they are the vocabulary of the
projection API, where they are layer-relative and would select the opposite matrix under `layer=:top`.

## Creation
```jldoctest DBiCM_docs
julia> using Graphs

julia> G = SimpleDiGraph(8);

julia> for (a,b) in ((1,5),(1,6),(2,6),(2,7),(3,7),(4,8),(5,2),(6,3),(7,4),(8,1)); add_edge!(G, a, b); end

julia> model = DBiCM(G)
DBiCM{Graphs.SimpleGraphs.SimpleDiGraph{Int64}, Float64} (4 + 4 vertices, 6 parameters, 0.38 compression ratio)

```

The layer membership is read off the **undirected skeleton**: `Graphs.bipartite_map` traverses
`outneighbors` only, so on a directed graph it would explore just the out-reachable set of its seed
while `Graphs.is_bipartite` still reported `true`.

A model can equally be built from the four sequences alone, in which case the partition is stated by
the caller and zero entries (dead channels) are allowed:

```jldoctest DBiCM_docs
julia> DBiCM(d⊥_out=[2,2,1,1], d⊥_in=[1,1,1,1], d⊤_out=[1,1,1,1], d⊤_in=[1,2,2,1])
DBiCM{Nothing, Float64} (4 + 4 vertices, 6 parameters, 0.38 compression ratio)

```

## Obtaining the parameters
```jldoctest DBiCM_docs
julia> solve_model!(model);

julia> model.status[:params_computed]
true

```

The two channels are solved **separately**. They share no parameter and their cross derivatives vanish,
so this is the exact decomposition rather than an approximation — and it keeps each solve at a single
gauge direction, which is the regime the fixed-point accelerator is measured on. A channel carrying no
links at all (a purely one-directional network) is not iterated: its optimum is ``θ ≡ ∞``, ``p ≡ 0``
exactly.

## Sampling the ensemble
```jldoctest DBiCM_docs
julia> sample = rand(model, 10);

julia> length(sample)
10

```

Samples are `SimpleDiGraph`s in the original vertex numbering of the source graph.

## Expected quantities
```jldoctest DBiCM_docs
julia> Pplus = Ĝ(model, channel=:to_top);     # ⊥ → ⊤ link probabilities

julia> Pminus = Ĝ(model, channel=:to_bottom); # ⊤ → ⊥ link probabilities

julia> size(Pplus) == size(Pminus) == (4, 4)
true

```

`Ĝ(model, channel=:both)` stacks the two vertically into a ``2N_⊥ × N_⊤`` matrix, which is the form
[`σₓ`](@ref MaxEntropyGraphs.σₓ) expects for a metric that reads both channels at once — the total
link count, the reciprocity, or a directed path motif.

## Model comparison
```jldoctest DBiCM_docs
julia> isfinite(AICc(model))
true

```
