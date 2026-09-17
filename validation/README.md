# Validation suite — per-dyad moments & delta-method variance

Standalone derivation and Monte-Carlo scripts backing the expectation/variance machinery
(Squartini & Garlaschelli 2011, NJP 13 083001, App. A.3/A.4/B; Saracco et al. 2015,
Sci. Rep. 5 10595, SI). Run any script with:

```
julia --project=validation validation/symbolic/<model>.jl
julia --project=validation validation/numeric/<name>.jl
```

Every script is deterministic (fixed seeds / exact `Rational{BigInt}` substitution), prints a
pass/fail table, and exits non-zero on failure. A fast distilled subset runs in CI
(`test/symbolics.jl`).

## Derivation notes

Longer write-ups that do not fit in a script header:

| Note | Subject |
|---|---|
| [`bicm_uecm_solver_geometry.md`](bicm_uecm_solver_geometry.md) | Companion for the other two models whose `:Newton` path needed work. The BiCM has **one** exact gauge mode and *nothing else* degenerate (saturation is rejected at construction, there are no weights), κ = 9–23 off the gauge — which is why it never needed the DECM's gauge-fixing. The UECM has **no** gauge at all (same-block shifts give `2c`) but both runaways, including the limit where all weights are 1 and the whole β block becomes unidentifiable (the UECM collapses to a UBCM). Also records why `IPNewton` was rejected for the UECM barrier (it turns honest failures into silent wrong answers) and the cross-model dead-channel bug. Backed by `symbolic/bicm_uecm_geometry.jl`. |
| [`uecm_decm_fixedpoint.md`](uecm_decm_fixedpoint.md) | Why the `UECM`/`DECM` `:fixedpoint` recipe could not be started from a cold guess, and what replaced it in `v0.8.0`. The Picard degree step provably **undershoots** (its factor is decreasing in its own `x`) while the strength step provably **overshoots without bound** (its factor is *increasing* in its own `y` whenever `xᵢxⱼ < 2`), so the iterate leaves the domain `yᵢyⱼ < 1` on the **first** step — measured overshoot 208×–1383× on every class of the rhesus UECM, against an optimum that sits at 97 % of the domain wall. The replacement solves each block exactly (both constraint functions are monotone, and the strength root is feasible *by construction*), plus a safeguarded 2×2 Newton polish for runaway ridges. Also records two traps: a per-node polish accepted on `‖residual‖` **cycles**, and a fixed-point-*increment* stopping test reports success on such a cycle. Plus the `L_UECM_reduced` `0·NaN` defect and the over-tight `β` box it caused. Backed by `symbolic/uecm_decm_fixedpoint.jl`. |
| [`decm_solver_geometry.md`](decm_solver_geometry.md) | Geometry of the DECM log-likelihood: the exact two-fold gauge freedom and why only `Newton` is hurt by it; the feasible polyhedron and why the `UECM`'s box constraint must **not** be transplanted; the runaway-constraint degeneracy taxonomy (`k = 0`, `k = N-1`, `s = k`) and the `10¹⁵`–`10¹⁷` condition numbers it produces; what that costs each solver; and the `AutoZygote` second-order crash. Backed by `symbolic/decm_gauge.jl`. |

## symbolic/ — dyad-level derivations vs the shipped closed forms

`common.jl` provides the equality oracles: structural `simplify∘expand` first, then exact
multi-point `Rational{BigInt}` substitution (authoritative; never stalls, no floating-point
doubt). All identities are kept sqrt-free. Techniques: direct state-sum enumeration for the
binary models (2-state and 4-state dyads), probability generating function for the UECM
(discrete geometric weights), moment generating functions for CReM/DCReM/CRWCM (mixture of
exponentials, joint MGF for the reciprocal coupling).

| Script | Verdict | What is proven |
|---|---|---|
| `ubcm.jl` | ALL PASS (6) | p = v/(1+v), Var = p(1−p) = code form (UBCM.jl:539); undirected convention Cov(a_ij,a_ji)=Var |
| `dbcm.jl` | ALL PASS (9) | p, Var, and Cov(a_ij,a_ji)=0 from the factorized 4-state sum |
| `rbcm.jl` | ALL PASS (9) | ⟨a⟩=(x_iy_j+z_iz_j)/Z, Var=a(1−a), Cov=z_iz_j/Z−⟨a_ij⟩⟨a_ji⟩ ≡ `_cov_dyads` |
| `bicm.jl` | ALL PASS (15) | p=xy/(1+xy) ≡ `f_BiCM`; entry independence; Saracco III.7 binomial-derivative identity (n=2,3,4); III.10≡III.6; Gaussian shifts for ⟨N_Vn⟩ (n=2,3,4) |
| `uecm.jl` | ALL PASS (29) | via PGF: p ≡ `f_UECM`, ⟨w⟩=p/(1−y) ≡ `Ŵ`, **Var[w]=p(1+y−p)/(1−y)²** (proposed σʷ), Cov(a,w)=⟨w⟩(1−p) |
| `decm.jl` | ALL PASS (37) | directed twin of `uecm.jl` via the per-channel PGF with composite params `x=xᵢ_out·xⱼ_in`, `y=yᵢ_out·yⱼ_in`: p ≡ `f_DECM`, ⟨w⟩ ≡ `Ŵ`, Var[w] ≡ `σʷ`², Cov(a,w)=⟨w⟩(1−p); joint PGF factorizes ⇒ Cov(w_ij,w_ji)=0 |
| `crem.jl` | ALL PASS (32) | via MGF: ⟨w⟩=f/(θ_i+θ_j) ≡ `Ŵ`, **Var[w]=f(2−f)/(θ_i+θ_j)²** (proposed σʷ, = DCReM code form), Cov(a,w)=⟨w⟩(1−f) |
| `dcrem.jl` | ALL PASS (12) | MGF moments ≡ `Ŵ`/`σʷ` code; joint MGF factorizes ⇒ Cov(w_ij,w_ji)=0 |
| `bicm_uecm_geometry.jl` | ALL PASS (30) | BiCM: gauge invariance `(α+c, β−c)` from `Σ f⊥·k⊥ = Σ f⊤·k⊤`, `‖H·g‖ ≈ 7e-17`, null dim exactly 1, κ < 10³, and that a saturated degree is rejected at construction. UECM: **no** gauge (both candidate shifts move `L`, null dim 0), the `k = n−1` runaway (`α ≈ −40.5`) and the all-weights-1 limit (`min β = 184`). Plus: dead channels honoured from **every** initial guess (`:degrees`, `:uniform`, `:random`, `:chung_lu`) |
| `uecm_decm_fixedpoint.jl` | ALL PASS (23) | the *solver* rather than the moments: the monotonicity asymmetry that makes the Picard degree step safe and the strength step unbounded (`∂A/∂x = -c²/(d₀+cx)²`, `D'(0) = g-2`), that `D - tD' = 1-(1-g)t² > 0` so `⟨sᵢ⟩` is strictly increasing and its root always exists and is feasible, the measured first-step domain exit and 1383× overshoot, the optimum at `max yᵢyⱼ = 0.974`, agreement of `:fixedpoint` with `:BFGS` (θ for the UECM, gauge-invariant `Ĝ`/`Ŵ` for the DECM), and that `L_UECM_reduced` is finite inside its own domain for a singleton class with `βᵢ < 0` but still `NaN` genuinely outside it |
| `decm_gauge.jl` | ALL PASS (49) | the *geometry* of the DECM objective rather than its moments: the exact two-fold gauge freedom `(α_out,α_in)→(α_out+c,α_in−c)`, `(β_out,β_in)→(β_out+c,β_in−c)` (invariance follows from `Σ F·k_out = Σ F·k_in` and `Σ F·s_out = Σ F·s_in`), the resulting singular Hessian (`‖H·g‖ ≈ 2e-19`), the degeneracy taxonomy (dead channel `k=0` → `α→+∞`, **saturated** degree `k=N−1` → `α→−∞`, **minimum** strength `s=k` → `β→+∞`; null dim = 2 gauge modes + one per degenerate constraint), that the gauge term changes no gauge-invariant quantity (`Ĝ`/`Ŵ` identical with and without), and a regression guard that `Newton` from `:uniform` fails without it (`-L = 10472` vs `384.49`) |
| `crwcm.jl` | ALL PASS (23) | joint MGF: ⟨w⟩, Var ≡ `Ŵ`/`σʷ`; ⟨w_ij w_ji⟩=π↔/(r₃r₄) ⇒ Cov ≡ `_covʷ`; binary layer ≡ RBCM |

**No discrepancies between the derivations and the shipped formulas.**

## numeric/ — Monte-Carlo gates for the changes introduced in v0.6.0

### `undirected_dyad_factor.jl` — the within-dyad covariance bug (FIXED in v0.6.0)
Confirmed (UBCM karate 20k samples; UECM/CReM binary layers on symmetrised rhesus, 10k):
the pre-0.6.0 `σₓ` omitted the cross-term `sum((σ.^2).*∇X.*∇X')` required because a_ij ≡ a_ji.
Effect: symmetric-gradient metrics (X=sum) low by exactly 1/√2 (measured current/sampled
0.710/0.714/0.716 for UBCM/UECM/CReM); asymmetric-gradient metrics can go EITHER way — ANND of
karate node 2: old form 2.52 vs sampled 1.415 (78 % OVERestimate, the omitted cross-term is
negative there), corrected 1.265 (within the delta-method linearization error). Upper-triangle
(one-slot) metrics unaffected. **Note: the script was written against the pre-fix package and
pins the buggy ratio; after the Stage-B fix it asserts current == corrected == sampled.**

### `uecm_weighted_sigma.jl` — ALL PASS (24)
Proposed UECM σʷ validated: sampler semantics w|edge = 1 + Geom(1−y); Ŵ matches bit-exactly;
Var[w]=p(1+y−p)/(1−y)² within 5·SE on the 5 heaviest dyads; total-weight σ via
sqrt(sum(σʷ²)/2) matches sampling to 0.18 % (covariance-blind form undercounts by √2);
Cov(a,w)=⟨w⟩(1−p) confirmed.

### `decm_weighted_sigma.jl` — ALL PASS (33)
DECM σʷ and the covariance-FREE directed delta method validated (rhesus, unsymmetrised, 10k samples):
Ĝ/Ŵ bit-exact, σʷ within 1e-15; entrywise Var[w] within 5·SE on the 5 heaviest ordered pairs;
Cov(w_ij,w_ji) compatible with zero on the 5 heaviest dyads (directed independence); total-weight σ via
sqrt(sum(σʷ²)) matches sampling (and equals the package σₓ), while the UECM-style within-dyad
correction sqrt(2·sum(σʷ²)) overcounts by √2; Cov(a,w)=⟨w⟩(1−p) confirmed.

### `crem_weighted_sigma.jl` — ALL PASS (21)
Proposed CReM σʷ validated on the same anchor: Ŵ bit-exact; entrywise variances within
1.9·SE; total-weight σ ratio 0.9984; Cov(a,w)=⟨w⟩(1−f) within 1.3·SE.

### `bicm_variance.jl` — 73/88 PASS; the 15 failures are the FINDING
- Part 1 (biadjacency σ layer): delta σ of the edge count matches sampling (ratios
  0.9953/0.9987) with NO cross-term (entries independent) → safe to implement.
- Part 2 (Vn/Λn families): the Saracco closed forms are **asymptotic in the opposite-layer
  degrees**: the Taylor mean shift is exact for n=2 but misses the skewness term for n≥3
  (17 % low in the worst case), and the first-order delta σ systematically underestimates
  (σ_ana/σ_sampled from 0.98 at degrees ≫ n down to 0.15 at degrees ≈ n), overstating |z|.
- The **exact route matches sampling everywhere** (all 12 graph/layer/n cases within 1.9·SE):
  per opposite-layer node p, U_p ~ PoissonBinomial(column p of Ĝ) and the U_p are independent,
  so ⟨N_Vn⟩ = Σ_p E[binom(U_p,n)] and Var[N_Vn] = Σ_p Var[binom(U_p,n)], both computable by an
  O(deg²) pmf convolution per node. ⇒ **v0.6.0 implements the exact PB route as the default
  (`method=:exact`) and ships the Saracco closed forms as `method=:delta` with the
  large-degree validity regime documented.**
- Part 3 (per-pair V_ij variance): PoissonBinomial Σq(1−q) is exact (matches sampling within
  0.5·SE); the first-order delta variance underestimates by exactly Σ_p p_ip p_jp(1−p_ip)(1−p_jp)
  (identity verified to machine precision) — documented in the BiCM docs.
