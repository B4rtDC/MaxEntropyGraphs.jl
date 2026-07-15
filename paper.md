---
title: 'MaxEntropyGraphs.jl: Maximum-entropy null models for network randomization in Julia'
tags:
  - Julia
  - network science
  - complex networks
  - maximum entropy
  - null models
  - graph randomization
  - statistical physics
authors:
  - name: Bart De Clerck
    orcid: 0000-0002-9718-260X
    corresponding: true
    affiliation: "1"
affiliations:
  - name: Royal Military Academy, Brussels, Belgium
    index: 1
date: 15 July 2026
bibliography: paper.bib
---

# Summary

`MaxEntropyGraphs.jl` is a Julia [@bezanson2017julia] package that brings
maximum-entropy null models for networks together in a single,
ecosystem-integrated framework. Given an
observed network, a maximum-entropy null model defines an ensemble of randomized
graphs that preserves a chosen set of structural constraints (for example the
degree sequence) on average, while being maximally random in every other
respect. Such ensembles are the standard tool for deciding whether a measured
network property is significant or merely a by-product of those constraints
[@park2004statistical; @squartini2011analytical; @cimini2019statistical].

The package computes the likelihood-maximizing parameters of a model and samples
randomized graphs from the resulting ensemble. For every model it also computes
ensemble averages and the standard deviations of network metrics analytically,
using automatic differentiation, so that significance can be assessed without
sampling. It further performs motif-based analysis and extracts statistically
validated projections of bipartite networks. All graphs
are standard objects from the Julia graph ecosystem [@graphs2021], so models
integrate directly with existing tooling. The currently supported models are the
Undirected, Directed and Bipartite Binary Configuration Models (UBCM, DBCM,
BiCM), the Reciprocal Binary Configuration Model (RBCM)
[@squartini2011analytical], the Undirected Enhanced Configuration Model (UECM),
the undirected and directed Conditional Reconstruction Methods (CReM, DCReM)
[@parisi2020faster], and the Conditionally Reciprocal Weighted Configuration
Model (CRWCM) [@divece2024commodity]. The reciprocity-aware models (RBCM, CRWCM)
preserve the dyadic structure of directed networks, that is the numbers of
reciprocated and non-reciprocated links and the weights they carry, which is
essential for higher-order (motif-based) analyses of directed networks.

# Statement of need

Graph randomization is a cornerstone of network science: by comparing a
real-world network to an ensemble of graphs that share some of its features, one
can test null hypotheses such as whether a network is more clustered, or contains
more of a given motif, than expected by chance. The maximum-entropy method makes
this rigorous. Given an observed network $G^*$ and a set of constraints
$\mathbf{C}$ measured on it, one constructs the ensemble $\mathcal{G}$ that
maximizes the Shannon entropy $S = -\sum_{G \in \mathcal{G}} P(G) \ln P(G)$
subject to $\mathbf{C}$. This yields the least-biased ensemble consistent with
the constraints, and finding it is equivalent to maximizing the (log-)likelihood
of the observed network [@squartini2015unbiased].

Implementations of these methods have so far lived primarily in Python, in
packages such as `NEMtropy` [@vallarano2021fast; @nemtropy] and `NuMeTriS`
[@numetris]. While capable, these tools require converting graphs into bespoke
formats, ship custom numerical solvers, and do not allow users to differentiate
arbitrary, user-defined metrics over the ensemble. No comparable package existed
in Julia. `MaxEntropyGraphs.jl` fills this gap. It is built on `Graphs.jl`
[@graphs2021] and on the well-established Julia optimization stack
(`Optimization.jl` [@dixit2023optimization] and `NLsolve.jl` [@nlsolve]), so the
sampled objects are ordinary `Graphs.jl` graphs that interoperate with the rest
of the ecosystem. Crucially, it leverages the JuliaDiff automatic-differentiation
tools (`ForwardDiff.jl` [@revels2016forward], `ReverseDiff.jl` [@reversediff] and
`Zygote.jl` [@innes2019zygote]), which lets users define their own network
metrics and obtain ensemble expectations and variances without deriving gradients
by hand. The intended audience is network scientists, computational social
scientists, and any researcher needing principled null models for network data.

# State of the field

Compared with the existing Python tools, `MaxEntropyGraphs.jl` offers three main
advantages. First, native integration with the Julia graph ecosystem removes the
format-conversion step and makes models composable with existing analysis code.
Second, automatic differentiation enables analytical ensemble statistics for
*user-supplied* metrics, rather than only the handful hard-coded by the library.
Third, for the validated projection of bipartite networks the package defaults to
the exact Poisson-binomial distribution (from `Distributions.jl`
[@besancon2021distributions]), instead of the Poisson approximation used by
default elsewhere, and supports flexible control of the false discovery rate
through `MultipleTesting.jl` [@multipletesting], with the Benjamini-Hochberg
procedure [@benjamini1995controlling] as the default [@saracco2015randomizing;
@saracco2017inferring].

# Functionality and design

Users interact with the package through a common `AbstractMaxEntropyModel`
interface (\autoref{fig:schematic}). A model is constructed either from a
`Graphs.jl` graph or directly from a constraint sequence. The
likelihood-maximizing parameters are then obtained with `solve_model!`, which
exposes both a fixed-point iteration (using Anderson acceleration via `NLsolve.jl`)
and gradient-based optimization (BFGS, L-BFGS and Newton, via
`Optimization.jl`). To keep problems tractable, all computations are performed in
a reduced parameter space spanned by the distinct constraints. After solving, the
extended `rand` function samples randomized graphs from the ensemble (in
parallel across threads), while `X(M)` and `sigma_x(M, X)` return the ensemble
mean and standard deviation of a metric `X`. Additional functionality includes
degree/strength and average-nearest-neighbour-degree metrics, the topological and
weighted reciprocity together with the reciprocal degree and strength sequences,
three- and four-node subgraph (motif) counts with their weighted fluxes and
intensities, the bipartite motif families, and statistically validated bipartite
projections. Performance-oriented choices (preallocated buffers, `@simd` and
`@inbounds` inner loops, multithreaded sampling and projection, and a
`PrecompileTools` workload that accelerates first use) are documented in the
package manual, which also contains installation instructions, per-model guides,
and a complete API reference.

## Design decisions and trade-offs

Two choices define the package and are worth stating explicitly. The first is to
obtain ensemble standard deviations by differentiating the metric a user
supplies, rather than by shipping a fixed catalogue of closed forms. This is the
main functional difference from the Python tools, and its price is that the
metric kernels must stay on the automatic-differentiation path, which constrains
how they may be written and is why their reformulations are derived, proved
equivalent, and tested as kernels in their own right. The second is that the
expectation and variance machinery works per layer, binary or weighted, rather
than jointly over the pair. That is a deliberate limitation, documented in the
manual, and it keeps the delta method exact within a layer at the cost of not
propagating the cross-layer covariance.

Elsewhere the defaults follow from measurement rather than from preference.
Solving is performed in a reduced parameter space spanned by the distinct
constraints; the default solver is chosen per model, because the fixed-point
recipe is stable for some models and not for others; and the default
differentiation backend for `sigma_x` is reverse mode, because the dense
alternatives exhaust memory well below the problem sizes of interest. The
benchmark harness that produced these conclusions, together with the symbolic and
Monte-Carlo suite that validates every per-dyad moment used by the delta method,
is part of the repository.

# Performance

The package is benchmarked against `NEMtropy` for model construction, parameter
computation, ensemble sampling, bipartite-projection validation, and directed
three-node motif computation across a
range of problem scales, using `BenchmarkTools.jl` [@chen2016benchmarktools] on
the Julia side and `pytest-benchmark` on the Python side, with both
implementations restricted to the same number of cores. `MaxEntropyGraphs.jl` is
consistently and often substantially faster at model construction and at
fixed-point parameter computation, with the gap widening as the number of
distinct constraints grows (\autoref{fig:ubcm}, \autoref{fig:bicm}). For the
gradient-based (quasi-Newton) solver it is faster on the binary configuration
models; for the bipartite model it is somewhat slower, but converges to a
markedly tighter solution, recovering the imposed degree sequence to within
$\sim\!10^{-9}$ against `NEMtropy`'s $\sim\!10^{-7}$ (both under matched solver
tolerances). Ensemble
sampling, which `MaxEntropyGraphs.jl` performs in memory as native graph objects,
is orders of magnitude faster than `NEMtropy`'s disk-based sampler. For the
analytical directed three-node motif spectrum, `MaxEntropyGraphs.jl` evaluates
the expected counts of all thirteen motifs through a compact linear-algebra
reformulation (matrix products on the expected adjacency matrix) rather than
per-motif triple loops, and the two implementations agree to
$\sim\!10^{-8}$ on the *maspalomas* food web. That agreement is a direct
cross-package validation of correctness, with both solved to a matched $10^{-8}$
tolerance. The weighted models
(UECM, CReM) are benchmarked against `NEMtropy`'s `ecm` and `crema` solvers in
the same harness, with both implementations reproducing their imposed
degree/strength sequences at comparable accuracy.

The reciprocity-aware models are benchmarked against `NuMeTriS` [@numetris], the
reference implementation accompanying @divece2024commodity, under the same
protocol (the `NuMeTriS` model names `RBCM`, `DBCM+CReMa` and `RBCM+CRWCM`
correspond to the RBCM, DCReM and CRWCM; the mixture models solve both the
binary and the weighted layer, matching the two-step `solve_model!`).
`MaxEntropyGraphs.jl` is one to two orders of magnitude faster at model
construction, and its default fixed-point solver computes the parameters one to
nearly three orders of magnitude faster across the benchmarked scales, again
profiting from the reduced parameter space; for the larger weighted problems the
gradient-based alternatives are slower than `NuMeTriS`'s compiled solver, but
the fixed point remains the fastest option overall (\autoref{fig:crwcm}). The
comparison again doubles as a cross-package validation: both implementations
converge to the same maximum-likelihood solution (identical binary Lagrange
multipliers to $\sim\!10^{-5}$, with each implementation reproducing its
constraint sequences at its own $10^{-8}$ solver tolerance), and, after
aligning the deterministic counting conventions of the two packages, the
empirical triadic motif counts agree exactly and the triadic fluxes to machine
precision. Beyond parity, `MaxEntropyGraphs.jl` evaluates the *exact* expected
motif and flux spectra under these models from the dyadic probabilities (within
a dyad the two link directions are correlated, so these expectations cannot be
formed from the expected adjacency matrix alone) and propagates the
corresponding within-dyad covariances through the delta method for analytical
z-scores, whereas `NuMeTriS` estimates triadic z-scores by ensemble sampling,
an approach that `MaxEntropyGraphs.jl` also provides and mirrors. The fully
scripted, reproducible benchmark harness is provided in the `performance/`
directory of the repository.

![Schematic overview of `MaxEntropyGraphs.jl`. Black indicates the core
functionality; blue indicates interaction with external packages. Arrows show
possible directions of the workflow.\label{fig:schematic}](figures/schematic.pdf)

![Performance comparison between `NEMtropy` and `MaxEntropyGraphs.jl` for the
UBCM model at different problem scales: model creation time (left) and median
parameter-computation time (right) as a function of the number of unique
constraints.\label{fig:ubcm}](figures/ubcm_benchmark.pdf)

![Performance comparison between `NEMtropy` and `MaxEntropyGraphs.jl` for the
BiCM model at different problem scales, including the validated bipartite
projection.\label{fig:bicm}](figures/bicm_benchmark.pdf)

![Performance comparison between `NuMeTriS` and `MaxEntropyGraphs.jl` for the
CRWCM model (binary and weighted layers solved jointly) at different problem
scales: model creation time (left) and median parameter-computation time
(right).\label{fig:crwcm}](figures/crwcm_benchmark.pdf)

# AI usage disclosure

During the preparation of this submission, the author used generative AI
assistance, specifically Anthropic's Claude (Opus 4.8), in three places: to help
draft and edit the manuscript text, to refactor and modernize parts of the
codebase, and to prepare benchmarking, validation and testing scaffolding. The
research problem, the choice of models to implement, the design of the package
and its interfaces, and the interpretation of all results are the author's own.

Every AI-assisted contribution was validated for correctness and efficiency
before being accepted, and the means of validation are part of the repository
rather than a claim made only here. Code is covered by a test suite of over 1700
tests that runs on every supported Julia version across Linux, macOS and Windows.
The per-dyad moments underlying the analytical variances are derived symbolically
and checked against Monte-Carlo sampling in a dedicated validation suite, a fast
subset of which runs in continuous integration. The metric kernels are proved
equivalent to the naive implementations they replace, and are tested for that
equivalence. Independent confirmation comes from outside the project as well: the
solutions, motif spectra and fluxes are cross-validated against the established
`NEMtropy` and `NuMeTriS` packages, as reported above. Efficiency claims rest on
the scripted benchmark harness in `performance/`, not on assertion. The author
reviewed and edited all such output and takes full responsibility for the content
of the software and of this paper.

# Acknowledgements

The author thanks Ben Lauwens for sharing his experience and knowledge of Julia
and Julia package development. This work was initiated in the context of the
author's PhD and funded by the DAP/19-03 project of the Belgian Defence. 
Additional features and performance improvements were added later, without further funding, 
as part of the author's ongoing research in network science and statistical physics.

# References

