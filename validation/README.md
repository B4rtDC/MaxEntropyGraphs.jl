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
