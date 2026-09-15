# Geometry of the DECM log-likelihood

Why the DECM is the hardest model in this package to solve, what is degenerate about it, and which of
those degeneracies a solver has to be told about.

The sibling derivations under `symbolic/` establish the per-dyad **moments** of each model (what the
fitted parameters *predict*). This note establishes the shape of the **objective** those parameters are
found by minimising. Every claim below is checked numerically by
[`symbolic/decm_gauge.jl`](symbolic/decm_gauge.jl) (49 checks); the section headings match the check
names there.

References: Vallarano et al., *Sci. Rep.* **11** (2021) 15227 (arXiv:2101.12625); Parisi, Squartini &
Garlaschelli, *NJP* **22** (2020) 053053.

---

## 1. The objective

The DECM constrains four sequences: out- and in-degree `k_out`, `k_in`, and out- and in-strength
`s_out`, `s_in`. Weights are non-negative integers and a link exists iff its weight is positive. The
Hamiltonian is linear in the constraints,

```
H(G) = Σᵢ [ α_out,ᵢ k_out,ᵢ(G) + α_in,ᵢ k_in,ᵢ(G) + β_out,ᵢ s_out,ᵢ(G) + β_in,ᵢ s_in,ᵢ(G) ],
```

and because every ordered pair `(i→j)`, `i ≠ j`, contributes `1` to `k_out,ᵢ` and `k_in,ⱼ` when
`w_ij > 0` and `w_ij` to `s_out,ᵢ` and `s_in,ⱼ`, it factorises over ordered pairs:

```
H(G) = Σ_{i≠j} [ (α_out,ᵢ + α_in,ⱼ)·1(w_ij > 0) + (β_out,ᵢ + β_in,ⱼ)·w_ij ].
```

**Every dependence on the parameters is through the two sums**

```
    x_ij = exp(-(α_out,ᵢ + α_in,ⱼ))          y_ij = exp(-(β_out,ᵢ + β_in,ⱼ)).
```

That single observation drives everything in §2. The channel partition function is geometric,

```
Z_ij = Σ_{w≥0} e^{-H_ij(w)} = 1 + x_ij·Σ_{w≥1} y_ij^w = 1 + x_ij·y_ij/(1 - y_ij),     valid iff y_ij < 1,
```

so with `Fᵢ` the multiplicity of reduced class `i` (a class pairs with itself `Fᵢ(Fᵢ-1)` times, hence
the weight `Fⱼ - δ_ij` in the code),

```
L(θ) = -Σᵢ Fᵢ (k_out,ᵢ α_out,ᵢ + k_in,ᵢ α_in,ᵢ + s_out,ᵢ β_out,ᵢ + s_in,ᵢ β_in,ᵢ)
       -Σᵢ Fᵢ Σⱼ (Fⱼ - δ_ij) log(1 + x_ij·y_ij/(1 - y_ij)),
```

which is `L_DECM_reduced` in `src/Models/DECM.jl`. The parameter vector is
`θ = (α_out, α_in, β_out, β_in) ∈ ℝ^{4n}`.

---

## 2. An exact two-fold gauge freedom

**Claim.** `L` is *exactly* invariant under each of

```
g_α :  (α_out, α_in) → (α_out + c, α_in - c)        g_β :  (β_out, β_in) → (β_out + c, β_in - c)
```

for any `c ∈ ℝ`, i.e. `g_α = (1ₙ, -1ₙ, 0ₙ, 0ₙ)` and `g_β = (0ₙ, 0ₙ, 1ₙ, -1ₙ)` are flat directions.

**Proof.** Split `L` into its pair part and its linear part.

*Pair part.* It depends on θ only through `x_ij` and `y_ij`, i.e. only through the sums
`α_out,ᵢ + α_in,ⱼ` and `β_out,ᵢ + β_in,ⱼ`. Under `g_α`,

```
(α_out,ᵢ + c) + (α_in,ⱼ - c) = α_out,ᵢ + α_in,ⱼ,
```

so every `x_ij` — and hence every pair term — is unchanged. Identically, not approximately.

*Linear part.* Under `g_α` it changes by

```
-c · Σᵢ Fᵢ (k_out,ᵢ - k_in,ᵢ) = -c · (Σᵢ Fᵢ k_out,ᵢ - Σᵢ Fᵢ k_in,ᵢ) = -c · (E - E) = 0,
```

because both sums count the same `E` edges — once by their tail, once by their head. The `g_β` case is
the same argument with the total weight `W` in place of `E`:
`Σᵢ Fᵢ s_out,ᵢ = Σᵢ Fᵢ s_in,ᵢ = W`. ∎

So the balance identities `Σ F·k_out = Σ F·k_in` and `Σ F·s_out = Σ F·s_in` — which hold for *any*
directed graph, and are checked in exact integer arithmetic — are precisely what makes the gauge exact.

### 2.1 Consequences for the derivatives

Differentiating `L(θ + c·g) = L(θ)` with respect to `c` at `c = 0`, and then again:

```
∇L(θ)·g = 0          and          H(θ)·g = 0          for g ∈ {g_α, g_β}, at every θ.
```

Measured: `‖H·g‖/‖H‖ ≈ 2·10⁻¹⁹` for both vectors, at arbitrary θ and at the optimum.

Two things follow, and they pull in opposite directions:

- **`θᵣ` is only determined up to the orbit.** Two runs may legitimately return parameters differing by
  a gauge shift. Every *gauge-invariant* quantity — `Ĝ`, `Ŵ`, all dyadic probabilities, all metrics —
  is identical. (This is the same caveat already recorded for the `DCReM`/`CRWCM` in v0.7.0.)
- **The Hessian is singular, with rank deficiency ≥ 2, everywhere.**

### 2.2 Why only `Newton` is hurt

- **Quasi-Newton (`BFGS`, `LBFGS`).** They never form or invert the true Hessian; they build a
  *positive definite approximation* from gradient differences. And since `∇L ⊥ g` exactly, the gradient
  never has a component along the gauge, so the iterates never travel along it either. The degeneracy
  is invisible to them.
- **`Newton`.** It solves `H·d = -∇L`. With `H` singular that system does not determine `d` along the
  null space, and the positive-definite *modification* Optim applies (via `PositiveFactorizations`) is
  essentially arbitrary there. The result is a step with a large, meaningless gauge component: it
  changes `L` not at all (the direction is flat) while moving `θ` far, frequently out of the domain of
  §3. Measured: from a `:uniform` start, `Newton` returned `-L ≈ 1.0·10⁴` against a true optimum of
  `384.49` on the rhesus macaques network — and did so on **8 of 8** networks tested.

  Perturbing the starting point does not help (checked): the degeneracy is a property of the objective,
  not an artifact of starting at a symmetric point.

### 2.3 The fix, and why it is exact

`solve_model!` adds, on the `:Newton` path only,

```
P(θ) = (λ/2)·[ (Σα_out - Σα_in)² + (Σβ_out - Σβ_in)² ] = (λ/2)·[ (g_α·θ)² + (g_β·θ)² ].
```

Its Hessian is `λ·(g_α g_αᵀ + g_β g_βᵀ)` — **exactly rank 2, with range exactly the null space of `H`**.
So `H + ∇²P` is non-singular in those two directions (eigenvalue `λ‖g‖² = 2nλ`) and unchanged in every
other direction.

`P ≥ 0`, with `P = 0` exactly on the slice `Σα_out = Σα_in`, `Σβ_out = Σβ_in`. Since `L` is *constant*
along each orbit, minimising `L + P` selects the unique representative of the optimal orbit lying in
that slice. **No gauge-invariant quantity moves.** Verified directly: `Ĝ` and `Ŵ` from `Newton` (with
the term) and from `BFGS` (without it) agree to `1e-6`/`1e-4`.

Any `λ > 0` is mathematically equivalent; it is purely a conditioning knob. `λ = 10⁻²` is used.

**It is applied to `Newton` only** — deliberately. For `BFGS`/`LBFGS` the extra curvature is a pure
trajectory perturbation, and measurement shows it is a coin flip: `λ = 10⁻²` breaks a `BFGS` case that
both `λ = 0` and `λ = 1` solve, while `λ = 1` breaks a different one. The effect is non-monotone in `λ`,
i.e. luck, so it is not imposed where it buys nothing.

---

## 3. The domain, and why a box constraint is wrong here

`Z_ij` converges iff `y_ij < 1`, i.e.

```
β_out,ᵢ + β_in,ⱼ > 0        for every ordered pair that actually occurs.
```

In the reduced parametrisation pair `(i,j)` occurs with multiplicity `Fᵢ(Fⱼ - δ_ij)`, so **the diagonal
of a singleton class is exempt**: if `Fᵢ = 1` there is no `i→i` pair and no constraint. The feasible set

```
D = { θ : β_out,ᵢ + β_in,ⱼ > 0  for all (i,j) with Fⱼ - δ_ij > 0 }
```

is an open convex polyhedron. As `β_out,ᵢ + β_in,ⱼ → 0⁺` we get `y → 1⁻` and
`log(1 + x y/(1-y)) → +∞`, so `-L → +∞`: the boundary is a natural barrier and the minimiser is
repelled from it. (Outside `D` the code returns `NaN`, which is what the line search must cope with.)

### 3.1 The tempting wrong move

The `UECM` (v0.7.1) has the analogous problem and *is* solved with a box: there the diagonal self-pair
term is evaluated for every class, so its domain is exactly the box `βᵢ > 0 ∀i`, and `Fminbox` keeps
every iterate feasible.

**That must not be transplanted to the DECM.** Using the gauge of §2, a feasible θ can be shifted into
the box `{β_out > 0, β_in > 0}` if and only if

```
minᵢ β_out,ᵢ + minⱼ β_in,ⱼ > 0,
```

(take `c = (min β_in - min β_out)/2`). If the minimising indices `i*`, `j*` form a *constrained* pair
this is guaranteed by `D`. But when `i* = j*` and `F_{i*} = 1`, that pair is exactly the exempt
diagonal, and nothing forces the sum positive.

This is not hypothetical. On a hub network (one node adjacent to all others) the fitted optimum has

```
β_out,12 + β_in,12 = -0.204        (the exempt diagonal — node 12 is the hub)
min over constrained i≠j pairs     = +0.038        (strictly feasible)
```

The optimum is feasible and lies **outside every box gauge**. A box would cut it off.

---

## 4. Runaway constraints: the rest of the degeneracy

Beyond the two gauge modes, the Hessian picks up one *near*-null direction for every constraint sitting
at the edge of what an ensemble can realise. In each case the conjugate parameter has no finite
maximiser and the optimiser simply drifts.

| condition | meaning | stationarity forces | limit | handled in `solve_model!` |
|---|---|---|---|---|
| `k = 0` | isolated in that direction | `p_ij = 0 ∀j` | `x → 0`, `α → +∞` | **yes** — `ind_inf`, pinned to `Inf` |
| `k = N-1` | links to everyone | `p_ij = 1 ∀j≠i` | `x → ∞`, `α → -∞` | no — runs away |
| `s = k` | every link carries weight 1 | `⟨w\|link⟩ = 1` | `y → 0`, `β → +∞` | no — runs away |

Derivations, from the stationarity conditions `⟨k⟩ = k*` and `⟨s⟩ = s*`:

- **Saturated degree.** `⟨k_out,ᵢ⟩ = Σ_{j≠i} p_ij = N-1` with every `p_ij ≤ 1` forces `p_ij = 1` for all
  `j ≠ i`. Since `p = x y/(1 - y + x y)`, `p → 1` requires `x → ∞`, i.e. `α_out,ᵢ → -∞`.
- **Minimum strength.** Conditional on a link, `⟨w_ij | w_ij>0⟩ = 1/(1 - y_ij)`. The observed
  `s_out,ᵢ = k_out,ᵢ` means every present out-link of `i` carries weight exactly `1`, so
  `1/(1-y_ij) → 1`, i.e. `y_ij → 0` and `β_out,ᵢ → +∞`.

Measured fitted values confirm both: `α ≈ -30 … -55` on saturated coordinates (so `p = 1 - O(e^{-30})`,
numerically indistinguishable from 1) and `β ≈ +31` on minimum-strength ones (`y ≈ 3·10⁻¹⁴`).

**These are properties of the data, not defects.** Such a fitness is simply not identifiable — the
constraint is at the boundary of what any ensemble can produce — and the fit is still right: the
constraints are reproduced to `~10⁻⁹` regardless. Note the `rhesus_macaques` network shipped with this
package has one `s = k` node, so it exhibits this.

Counting: generically

```
dim ker H  =  2 (gauge)  +  #saturated degrees  +  #minimum strengths.
```

Checked in `decm_gauge.jl` on a clean network (null dim exactly 2) and a hub network (null dim 4 = 2 + 2
saturated). A caution when reproducing this: a runaway direction is *near*-null, with curvature decaying
like `e^{-|θ|}`, so an absolute eigenvalue cutoff will miscount whichever coordinates have not drifted
far — at `α = -23`, `e^{-23} ≈ 10⁻¹⁰` sits exactly on a `1e-10` threshold. Use a relative cutoff.

(Disconnection of the observed graph is *not* a source of any of this, and was checked: the model's
ensemble ranges over all pairs regardless of observed connectivity, and every network used above is both
weakly and strongly connected.)

---

## 5. Conditioning, and what it costs each solver

The runaway directions of §4 — not the gauge of §2 — are what make the DECM numerically hard. Their
curvature is `~e^{-|θ|}` with `|θ| ≈ 30–55`, against a largest eigenvalue of order `10⁴–10⁵`:

| network | extra null directions | condition number (gauge modes removed) |
|---|---|---|
| clean random | 0 | `6·10²` – `2·10³` |
| rhesus macaques | 1 (`s = k`) | `6·10¹⁵` |
| hub / dense / saturated | 1–5 | `2·10¹⁵` – `2·10¹⁷` |

Measured iterations to convergence, from the default `:strengths` guess and from the deliberately far
`:uniform` one:

| method | uses | from `:strengths` | from `:uniform` |
|---|---|---|---|
| `:Newton` | exact Hessian | ≤ 40 | ≤ 160 |
| `:BFGS` | full dense inverse-Hessian approximation | ≤ 200 | ≤ 265 |
| `:LBFGS` | last `m = 10` curvature pairs | up to ~7 300 | up to ~16 000 |

The ordering is the expected one: a limited-memory approximation cannot represent a spectrum spanning
fifteen orders of magnitude, whereas full `BFGS` accumulates it over `O(4n)` iterations and `Newton` has
it exactly. Hence `:BFGS` (default) and `:Newton` are the recommended methods, `:LBFGS` is not, and the
`maxiters` default is `10_000` — a cap, so it costs the fast methods nothing.

`:LBFGS` was previously reported as *unstable* here. That is the wrong word and has been corrected: it
converges to the same optimum, it is merely slow. What used to happen is that it hit the old
`maxiters = 1000` while already within `7·10⁻⁴` of the optimum, and `solve_model!` turned that
`MaxIters` return code into a hard `ConvergenceError`, discarding a good fit.

### 5.1 Line searches at the barrier

A counter-intuitive but measured point. `BackTracking` enforces only the Armijo (sufficient-decrease)
condition, not the curvature condition that keeps an L-BFGS Hessian approximation positive definite, so
one would expect a Wolfe line search to do better. **It does worse**, because the `NaN` barrier of §3
defeats it:

| line search | result at `maxiters = 1000` |
|---|---|
| `BackTracking` (used) | `MaxIters`, close to the optimum |
| `HagerZhang`, `MoreThuente` | `MaxIters`, further away |
| `StrongWolfe` | **`Failure`**, `-L` off by `10³`–`10⁴` |

`BackTracking` halves the step until the objective is finite, which is exactly the behaviour a barrier
demands. This is the same reasoning already recorded for the `UECM`.

---

## 6. A note on automatic differentiation for `Newton`

Not geometry, but it belongs with the record. `:Newton` needs second derivatives. Given a *first-order*
ADtype, `OptimizationBase` wraps it as `SecondOrder(inner, AutoForwardDiff)` — it emits a warning saying
so — and with the package default `:AutoZygote` as the inner backend, that nested HVP path **aborts the
Julia process** (`signal 4: illegal instruction`, inside `DifferentiationInterface`'s `hvp!`) whenever
`Symbolics` is loaded into the same session.

| model | `:AutoZygote` | `:AutoForwardDiff` | `:AutoReverseDiff` |
|---|---|---|---|
| `UBCM`, `DBCM` | ok | ok | ok |
| `BiCM`, `UECM`, `DECM` | **crash** | ok | ok |

The DECM therefore builds its `Newton` Hessian with `ForwardDiff` directly: no nesting, what the
upstream warning recommends, and substantially more accurate — the degree residual on rhesus drops from
`~4·10⁻⁹` to `3.6·10⁻¹⁵`. An explicitly requested non-Zygote backend is honoured as given.

⚠️ **`BiCM` and `UECM` are still exposed**; `AD_method = :AutoForwardDiff` works around it for both.

⚠️ The crash is state- and order-sensitive: it reproduces reliably under `validation/` but **not** in the
package's own test sandbox, on byte-identical dependency versions (DifferentiationInterface 0.7.21,
OptimizationBase 5.6.1, Symbolics 7.39.2, Zygote 0.7.13). A green test suite is not evidence of its
absence.

---

## 7. What is checked where

| claim | checked by |
|---|---|
| balance identities `Σ F·k_out = Σ F·k_in`, `Σ F·s_out = Σ F·s_in` | `decm_gauge.jl`, exact integer |
| pair exponents invariant under both shifts | `decm_gauge.jl`, symbolic (`ratzero`) |
| `L` invariant along `g_α`, `g_β` | `decm_gauge.jl`, 3 networks × 4 shifts |
| `H·g_α = H·g_β = 0` | `decm_gauge.jl`, 3 networks |
| `dim ker H` = 2 + degenerate constraints | `decm_gauge.jl`, clean vs hub network |
| gauge term changes no gauge-invariant quantity | `decm_gauge.jl`, `Ĝ`/`Ŵ` from `Newton` vs `BFGS` |
| `Newton` from `:uniform` fails *without* the gauge term | `decm_gauge.jl`, regression guard |
| every recommended method × initial guess reproduces the constraints | `test/ensemble_validation.jl` |
| per-channel moments (`p`, `⟨w⟩`, `Var[w]`, covariances) | `symbolic/decm.jl`, `numeric/decm_weighted_sigma.jl` |
