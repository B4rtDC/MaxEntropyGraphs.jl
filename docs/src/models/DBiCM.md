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

## Projection

An undirected bipartite pair has **one** way of sharing a neighbour. A directed pair has three, because
either link may point either way:

| `kind` | ⊥-pair kernel | shape | symmetry |
| --- | --- | --- | --- |
| `:out` | ``\sum_α B^+_{iα} B^+_{jα}`` | ``i → α ← j`` | symmetric |
| `:in` | ``\sum_α B^-_{iα} B^-_{jα}`` | ``i ← α → j`` | symmetric |
| `:path` | ``\sum_α B^+_{iα} B^-_{jα}`` | ``i → α → j`` | **asymmetric** |

`kind` is relative to the projected layer: `:out` always means "both members of the pair *send* to the
shared node". This is forced rather than chosen — `:path` is defined by the pair ordering, so one of the
three is layer-relative whatever one does — and it is why the model's `channel` selector is spelled
`:to_top`/`:to_bottom` instead of `:out`/`:in`.

Under the DBiCM all three counts are **exactly** Poisson-binomial, with ``q_α`` the corresponding
product of two entry probabilities. Every factor is a product of two *distinct* Bernoulli entries, and
the entries are independent both within a dyad and across ``α`` — including on the diagonal, where
``V^{path}_{ii}`` counts the ⊤ nodes reciprocally linked to ``i`` and is available as
[`reciprocated_degree`](@ref MaxEntropyGraphs.reciprocated_degree).

```jldoctest DBiCM_docs
julia> V = V_motifs(model, 1, 2, layer=:bottom, kind=:path);   # expected i → α → j count

julia> q = MaxEntropyGraphs.V_PB_parameters(model, 1, 2, layer=:bottom, kind=:path);

julia> V ≈ sum(q)          # the expectation is the sum of the Poisson-binomial parameters
true

```

[`project`](@ref MaxEntropyGraphs.project) turns this into a statistically validated monopartite
network: observed counts are compared with their distribution under the model, and the upper-tail
p-values are corrected for multiple testing before thresholding.

```jldoctest DBiCM_docs
julia> P = project(model, α=0.05, layer=:bottom, kind=:out);

julia> P isa MaxEntropyGraphs.Graphs.SimpleGraph      # symmetric kinds give an undirected projection
true

julia> project(model, layer=:bottom, kind=:path) isa MaxEntropyGraphs.Graphs.SimpleDiGraph
true

```

Two things to keep in mind when comparing across kinds:

- **The totals count different index sets.** `:out` and `:in` are symmetric, so they count each
  *unordered* pair once; `:path` counts every *ordered* pair.
- **The multiple-testing pools differ in size.** A `:path` projection tests roughly twice as many
  hypotheses as an `:out` one on the same data, so any correction is correspondingly more conservative.

The higher-order [`Vn_motifs`](@ref MaxEntropyGraphs.Vn_motifs) family (`n` nodes sharing a partner)
is single-channel, so it takes `kind ∈ (:out, :in)` and delegates to the matching channel. There is no
`:path` analogue: a mixed-direction higher-order motif needs the joint law of a shared node's in- and
out-degree, which is a two-dimensional Poisson-multinomial over the four dyad states rather than the
one-dimensional convolution used here.

## Model comparison
```jldoctest DBiCM_docs
julia> isfinite(AICc(model))
true

```
