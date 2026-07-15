# Performance

This folder holds the written record of the performance work behind the graph-metric kernels in
[`src/metrics.jl`](../src/metrics.jl). For user-facing guidance on performance and scalability
(which backend to pick, how far the dense representation scales), see the
[Performance and scalability](https://B4rtDC.github.io/MaxEntropyGraphs.jl/stable/performance/)
page of the documentation instead.

## `metrics_acceleration.tex`

A LaTeX note documenting how the metric kernels were accelerated, and *proving* that the
accelerated forms compute the same thing as the naive loops they replace. It covers:

* the average nearest-neighbour degree (ANND), reformulated from `O(N³)` to `O(N²)`;
* the triangle count as `tr(A³)/6`, with `O(N)` peak memory;
* the thirteen directed three-node motifs, in an inclusion–exclusion matrix form;
* the count of "pure" squares, with an exact 8× dihedral symmetry reduction, why no
  sub-`O(N⁴)` closed form is available, and a sparse fast path;
* the bipartite V-motif count, as a marginal-sum closed form.

For each kernel it gives the exact identity behind the reformulation, a proof of computational
equivalence to the original loop, the resulting complexity and peak-memory behaviour, and the two
cross-cutting constraints that shape the implementations:

1. the kernels sit on the reverse-/forward-mode **automatic-differentiation** path used to compute
   metric variances, and
2. they are mapped over many sampled graphs inside an already thread-parallel outer loop, so
   per-call memory must not scale with the thread count.

The equivalence proofs are stated for **real-valued** matrices rather than merely `0/1` ones, which
is what makes them applicable here: the metrics are evaluated not only on sampled graphs but also
analytically on the dense expected matrix `Ĝ`, whose entries are probabilities.

The note closes with the validation methodology (bit-exact integer counts, `isapprox` float
expectations, finite-difference gradient checks) and empirical timing/memory results. The
corresponding regression tests are the `metrics acceleration` testset in
[`test/metrics.jl`](../test/metrics.jl).

### Building

```bash
pdflatex metrics_acceleration.tex
```
