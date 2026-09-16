# Geometry of the BiCM and UECM log-likelihoods

Companion to [`decm_solver_geometry.md`](decm_solver_geometry.md), covering the other two models whose
`:Newton` path needed work. Shared material (the `SecondOrder` AD crash, the general shape of the
runaway argument) is derived there and only referenced here.

The two models sit at opposite ends of the difficulty scale, and that contrast is the useful part: it
shows which features of a maximum-entropy likelihood actually cause solver trouble, and which do not.

Checked by [`symbolic/bicm_uecm_geometry.jl`](symbolic/bicm_uecm_geometry.jl) (30 checks).

---

## 0. Summary

| | gauge modes | runaways possible | κ (off the gauge) | `:Newton` |
|---|---|---|---|---|
| **BiCM** | **1** | **none** | **9 – 23** | fine everywhere |
| **UECM** | **0** | `k = n-1`, `s = k` | 249 – 7 250 (clean data) | fine from `:strengths`; fails from far starts |
| *DECM* (for comparison) | *2* | *`k = N-1`, `s = k`* | *10¹⁵ – 10¹⁷* | *needs gauge-fixing* |

The headline: **a gauge freedom on its own is harmless.** The BiCM has one and never needed the
gauge-fixing the DECM required, because nothing else about its Hessian is badly conditioned. It is the
*combination* of a singular direction with a `10¹⁵` condition number that breaks a Newton solve.

---

## 1. BiCM — one exact gauge mode, and no runaway is possible

### 1.1 The gauge

`L_BiCM_reduced` is

```
L = -Σᵢ f⊥ᵢ k⊥ᵢ αᵢ  -  Σⱼ f⊤ⱼ k⊤ⱼ βⱼ  -  Σᵢ Σⱼ f⊥ᵢ f⊤ⱼ softplus(-αᵢ - βⱼ),
```

and the pair term depends on θ only through `αᵢ + βⱼ` — one index from each layer. So under

```
(α, β) → (α + c, β - c)
```

every pair term is unchanged, and the linear part shifts by

```
-c·(Σᵢ f⊥ᵢ k⊥ᵢ - Σⱼ f⊤ⱼ k⊤ⱼ) = -c·(E - E) = 0,
```

because both sums count the same edges, once from each side of the bipartition. **One** flat direction
`g = (1…1, -1…-1)` — the DECM has two because it carries an α *and* a β block; the BiCM is unweighted
and has only one of each kind.

Measured: `ΔL = 0` to machine precision, `‖H·g‖/‖H‖ ≈ 7·10⁻¹⁷`, Hessian null dimension **exactly 1**.

As always the consequence is that `θᵣ` is determined only up to the orbit, while every gauge-invariant
quantity (`Ĝ`, all dyadic probabilities, all metrics) is unique.

### 1.2 Why nothing else is degenerate

The DECM's trouble came from constraints pinned at the edge of their feasible range. The BiCM has
neither kind:

- **A saturated degree cannot occur.** The constructor rejects it outright
  ([`BiCM.jl:178-179`](../src/Models/BiCM.jl#L178)):

  ```julia
  maximum(d⊥) >= length(d⊤) && throw(DomainError(...))
  maximum(d⊤) >= length(d⊥) && throw(DomainError(...))
  ```

  A ⊥-node adjacent to every ⊤-node would force `p_ij = 1 ∀j` and hence `α → -∞`; the model refuses the
  input instead. (This guard predates the present work — it is noted here because it is exactly the
  degeneracy the DECM has no protection against.)
- **There is no strength constraint**, so the `s = k` runaway has no analogue.

Only the dead channel `k = 0` remains, and that is handled (§3).

### 1.3 Consequence for `:Newton`

Measured condition number with the gauge mode removed: **9.07** (corporate club), **14.9** (20×40
random bipartite), **23.0** (dense). That is about as well-conditioned as a fitting problem gets.

A rank-1 deficiency in an otherwise benign Hessian is handled without fuss by the positive-definite
modification Optim applies (via `PositiveFactorizations`): the modification is arbitrary along the null
space, but since `L` is flat there it costs nothing, and the remaining directions are well scaled enough
that the step is still good. Verified: `:Newton` succeeds on every network × initial guess tested, with
residuals of `10⁻¹³`–`10⁻¹⁵` — the *best* of the three methods.

**So the BiCM needs no gauge-fixing, and none was added.** The contrast with the DECM is the lesson: it
is not the singularity that breaks Newton, it is the singularity *plus* a `10¹⁵` condition number.

---

## 2. UECM — no gauge at all, but two runaways

### 2.1 There is no gauge

The UECM is undirected: a pair term couples `αᵢ + αⱼ` and `βᵢ + βⱼ`, **both indices from the same
block**. A shift `α → α + c` sends

```
αᵢ + αⱼ  →  αᵢ + αⱼ + 2c,
```

which does not cancel against anything — there is no second block with the opposite sign, as the
directed and bipartite models have. The DECM-style shift `(α + c, β - c)` is likewise not a symmetry:
it changes the α pair terms by `+2c` and the β pair terms by `-2c`, and those enter the likelihood
differently.

Measured: both candidate shifts change `L` by `O(10²)` (289 and 34.5 on the rhesus macaques network),
and the Hessian has **null dimension 0**. There is simply nothing to gauge-fix, which is why the DECM's
`_decm_gauge` has no UECM counterpart.

On clean data the UECM is also decently conditioned: κ between **249** and **7 250**.

### 2.2 The two runaways

The taxonomy is the DECM's, minus the out/in split (derivation in
[`decm_solver_geometry.md` §4](decm_solver_geometry.md)):

| condition | meaning | stationarity forces | limit | handled |
|---|---|---|---|---|
| `k = 0` | isolated node | `p_ij = 0 ∀j` | `α → +∞` | **yes** — `ind_inf` (§3) |
| `k = n-1` | adjacent to everyone | `p_ij = 1 ∀j≠i` | `α → -∞` | no — runs away |
| `s = k` | every incident link has weight 1 | `⟨w\|link⟩ = 1` | `β → +∞` | no — runs away |

Measured on a graph with one saturated node: that class fits at **α ≈ -40.5**, and the Hessian picks up
a matching near-null direction.

### 2.3 The interesting limit: all weights equal 1

If *every* weight in the graph is 1 then `sᵢ = kᵢ` for **every** node, and the entire β block runs away
at once. Measured: `min β = 184`, `max β = 356` — far enough that the Hessian itself overflows to `Inf`
in Float64.

This is not a numerical accident, it is the model being honest. With uniform weights the strength
sequence carries **no information beyond the degree sequence**: the two constraints are the same
constraint written twice. The weighted layer is unidentifiable, and the UECM has degenerated into a
UBCM. A user who sees a UECM fit with all-huge β should read it as "your weights told me nothing".

(Partly as a consequence, `:LBFGS` on such a network returns a loose fit — degree/strength residual
`6·10⁻²` where the other methods reach `10⁻⁵`–`10⁻⁹`.)

### 2.4 `:Newton` and the feasibility barrier

The UECM likelihood is defined only on the open box `βᵢ > 0` (see
[`decm_solver_geometry.md` §3](decm_solver_geometry.md) for why the diagonal self-pair term makes the
domain exactly a box here, unlike the DECM). Since v0.7.1 the **first-order** methods carry that box via
`Fminbox`, which keeps every iterate feasible.

`Fminbox` does not accept `Newton`, so `:Newton` is the one UECM method solving the box-constrained
problem **without** box protection. From the default `:strengths` guess this is fine — measured success
on every network tested, residuals `10⁻⁹`–`10⁻¹⁴`. From a far start it is not:

| start | outcome | diagnosis |
|---|---|---|
| `:strengths` | Success, `-L` = optimum | — |
| `:strengths_minor` | **Failure** | pinned against the barrier, `min β = 1.6·10⁻¹¹` |
| `:uniform` | **Failure** | never moves at all (`min β` unchanged at 6.908) |

The `:uniform` case is the clearer one. It starts at `β = -log(0.001) = 6.9`, i.e. `y ≈ 10⁻³` — almost
no excess weight — while the optimum needs `β ≈ 8·10⁻³`. The Newton step across that gap is enormous
and lands at `β < 0`, outside the domain, where the objective is `NaN`; `BackTracking` halves ~52 times,
never reaches a finite point, and the solve aborts without having moved.

#### What was tried and rejected

**`IPNewton` with the box.** The obvious fix is the interior-point Newton, which *does* accept box
constraints. It converts every failure into a success — and that is precisely the problem:

| network / start | unconstrained `Newton` | `IPNewton` (box `β ≥ 10⁻¹⁰`) |
|---|---|---|
| rhesus / `:strengths_minor` | Failure | Success, Δ = -6·10⁻¹⁴ ✔ |
| rhesus / `:uniform` | Failure | **Success, Δ = 5 080** ✘ |
| w14 / `:strengths` | Success, Δ = 3·10⁻¹⁴ | **Success, Δ = 317** ✘ |
| w22 / `:strengths` | Success, Δ = 0 | **Success, Δ = 1 060** ✘ |
| dense10 / `:strengths` | Success, Δ = 0 | **Success, Δ = 241** ✘ |

`IPNewton` meets its own interior-point stopping criterion far from the optimum and reports success.
It turns honest failures into **silent wrong answers**, and even breaks cases the unconstrained solver
gets right. Rejected.

**Line-search tuning.** `BackTracking(order=2)` and small forced initial steps
(`InitialStatic(α = 0.05 / 0.01)`) were tried across three networks × two far starts. None fixed the
failures, and the most aggressive setting introduced a *new* silent wrong answer (dense10/`:uniform`:
Success at Δ = 120). Rejected.

**Conclusion.** An honest `ConvergenceError` is the correct behaviour here, and it is what ships.
`:Newton` is documented as requiring the default `:strengths` guess for this model. This mirrors the
DECM conclusion about `:LBFGS`: when a method cannot be made reliable, say so rather than paper over it.

---

## 3. Dead channels must come from the data, not from the initial guess

This was a live bug across four models, found while stress-testing the above.

A class with `k = 0` is a **dead channel**: its parameter belongs at `+∞`, so that `x = e^{-θ} = 0` and
the channel can never carry a link. `solve_model!` handles this by collecting those indices in
`ind_inf`, neutralising them for the solve, and restoring `Inf` afterwards.

The indices were collected as

```julia
ind_inf = findall(isinf, θ₀)        # ← from the INITIAL GUESS
```

But only the `:degrees` / `:strengths` family puts an `Inf` there, via `-log(0)`. For `:uniform` and
`:random` the initial guess is finite everywhere, so **`ind_inf` came back empty**, every dead channel
kept a finite parameter, and the corresponding rows of `Ĝ` acquired spurious edges. The solve then
reported `Success` at a wrong answer — the worst possible failure mode.

Measured on a planted bipartite graph with one zero-degree class: degree residual **9.85**, reported as
`Success`, for all three optimisation methods.

The fix is to derive the set from the **data**, which is what it always meant:

```julia
ind_inf = vcat(findall(iszero, m.d⊥ᵣ), length(m.d⊥ᵣ) .+ findall(iszero, m.d⊤ᵣ))   # BiCM
```

and analogously for `DBCM` (`dᵣ_out`, `dᵣ_in`), `UECM` (`dᵣ`, `sᵣ`) and `DECM` (all four blocks). On the
`:degrees`/`:strengths` guesses this is *exactly* the old set — `-log(x) = Inf ⟺ x = 0` — so nothing
changes on the default path; the other guesses are simply no longer wrong.

The `RBCM` already did it this way ([`RBCM.jl:646`](../src/Models/RBCM.jl#L646)), which is what made the
discrepancy visible.

---

## 3b. Isolated vertices have no layer, so the BiCM now refuses them

A `k = 0` constraint is **not** infeasible — it is met exactly (`α → +∞`, `p_ij = 0 ∀j`), and every other
model in the package fits isolated vertices without trouble. The BiCM is different for a reason that has
nothing to do with feasibility: a vertex must also be **assigned to a layer**, and an isolated vertex
gives no evidence for either side.

`Graphs.bipartite_map` colours each connected component starting from 1, so every isolated vertex lands
in ⊥. Measured on an 18×40 graph with 29 isolated vertices (3 from ⊥, 26 from ⊤ by construction):

| | intended | what `BiCM` built |
|---|---|---|
| layer sizes | 18 × 40 | **44 × 14** |

The *fit* survives — live-vertex residual `2.7e-12`, isolated rows of `Ĝ` exactly zero — but `|⊥|` and
`|⊤|` are wrong, so `rand(m)` samples a 44×14 ensemble and anything keyed on layer sizes is off. Since
the input genuinely does not determine the model, the constructor now throws an `ArgumentError` naming
the offending vertices, rather than silently picking a side.

The package's own test fixture was an instance of this: `_planted_bipartite()` (24 ⊥ × 100 ⊤) left 33
vertices isolated and was being built as a **57 × 67** model. It has been de-isolated.

Dead channels still reach the BiCM, just not through a graph — via explicit degree sequences, where the
caller has stated the partition:

```julia
BiCM(nothing; d⊥ = [0, 2, 2, 1], d⊤ = [1, 2, 2])   # zeros are fine here
```

That is the form the §3 dead-channel checks now use.

### An earlier claim, corrected

This section previously said the BiCM `:fixedpoint` "diverges on any bipartite graph containing isolated
nodes". **That was wrong** — generalised from a single probe. Adding 1, 3 or 5 isolated vertices to
otherwise healthy graphs converges fine; and among single-live-class systems, `deg 2` diverges while
`deg 3` and `deg 4` converge. There is no clean structural rule: it is data-dependent instability of the
Anderson acceleration, of the same kind already documented for the `UECM` and `DECM` fixed points.

The isolated-vertex guard removes the *observed* failures, because the graphs that triggered them are now
refused at construction — but it is not a fix for the underlying accelerator, and a degenerate reduced
system reached some other way could still diverge. `:fixedpoint` remains the BiCM default; `:BFGS` and
`:Newton` were solid on every case tested.

---

## 4. Automatic differentiation for `:Newton`

Both models were subject to the crash derived in
[`decm_solver_geometry.md` §6](decm_solver_geometry.md): `:Newton` needs second derivatives, so
`OptimizationBase` wraps a first-order ADtype as `SecondOrder(inner, AutoForwardDiff)`, and with the
package default `:AutoZygote` as inner backend that nested HVP path **aborts the Julia process**
(`signal 4`) once `Symbolics` is loaded in the same session.

`BiCM`, `UECM` and `DECM` were affected; `UBCM` and `DBCM` were not. All three now build their `Newton`
Hessian with `ForwardDiff` directly. An explicitly requested non-Zygote backend is honoured as given.

⚠️ As noted there, the crash is state- and order-sensitive — it reproduces under `validation/` but not
in the package's test sandbox on identical dependency versions — so a green test suite is not evidence
of its absence.

---

## 5. What is checked where

| claim | checked by |
|---|---|
| BiCM pair argument invariant; UECM same-block shift gives `2c` | `bicm_uecm_geometry.jl`, symbolic |
| `Σ f⊥·k⊥ = Σ f⊤·k⊤` (exact) | `bicm_uecm_geometry.jl`, exact integer |
| BiCM `L` invariant along `g`; `H·g = 0`; null dim exactly 1; κ < 10³ | `bicm_uecm_geometry.jl`, 2 networks |
| BiCM rejects a saturated degree at construction | `bicm_uecm_geometry.jl` |
| UECM has no gauge (both candidate shifts change `L`; null dim 0) | `bicm_uecm_geometry.jl` |
| UECM saturated degree → `α ≈ -40.5` + near-null direction | `bicm_uecm_geometry.jl` |
| UECM all-weights-1 → whole β block runs away (`min β = 184`) | `bicm_uecm_geometry.jl` |
| dead channels honoured from **every** initial guess | `bicm_uecm_geometry.jl`, 4 guesses |
| every method × initial guess reproduces the constraints | `test/ensemble_validation.jl` |
| per-dyad moments | `symbolic/bicm.jl`, `symbolic/uecm.jl`, `numeric/uecm_weighted_sigma.jl` |
