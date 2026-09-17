# The UECM/DECM fixed point: why the Picard recipe could not start, and what replaced it

Companion to [`symbolic/uecm_decm_fixedpoint.jl`](symbolic/uecm_decm_fixedpoint.jl), which executes every
claim below. Sibling notes: [`decm_solver_geometry.md`](decm_solver_geometry.md) (the shape of the DECM
objective) and [`bicm_uecm_solver_geometry.md`](bicm_uecm_solver_geometry.md) (the BiCM/UECM `:Newton`
work). This one is about the **`:fixedpoint`** path of the two weighted-with-degrees models.

Up to `v0.7.0` both models shipped a `:fixedpoint` method that carried the warning *"very unstable for
this model and should not be used"*. That was an understatement: measured over 150 random weighted
networks and 150 random weighted digraphs, from the package's own default initial guess, it converged on
**0 of each**. This note derives why — the reason is exact, not empirical — and what the replacement does.

---

## 1. The feasible set

Both models give an (ordered) pair the Bernoulli–geometric weight law

$$P(w) \propto (x_i x_j)^{\mathbb 1[w>0]}\,(y_i y_j)^{w},\qquad w = 0,1,2,\dots$$

With $g = x_i x_j$ and $t = y_i y_j$ the normaliser is $1 + g\,t/(1-t)$, which converges **only for
$t < 1$**. So the model is defined on

| model | condition | in $\theta$ |
|---|---|---|
| UECM | $y_i y_j < 1$ | $\beta_i + \beta_j > 0$ |
| DECM | $y_{\text{out},i}\,y_{\text{in},j} < 1$ | $\beta_{\text{out},i} + \beta_{\text{in},j} > 0$ |

over the pairs that **exist**. Two things follow immediately.

*It is a pairwise condition, not a box.* The only per-coordinate case is a UECM class of multiplicity
$F_i \ge 2$, whose same-class pair needs $2\beta_i > 0$. A **singleton** class contains one vertex and has
no same-class pair, so nothing in the model bounds the sign of its own $\beta_i$. §6 is about what
assuming otherwise cost.

*There is very little slack.* On the symmetrised rhesus network the ML optimum sits at
$\max_{i\neq j} y_i y_j = 0.974$ — 97% of the way to the wall. A method that overshoots by even a few
percent is outside.

---

## 2. Why the Picard recipe leaves the domain on step 1

The classical recipe (NEMtropy's `iterative_ecm`; shipped here as `UECM_reduced_iter!` /
`DECM_reduced_iter!`) factors one parameter out of each constraint and inverts with everything else
frozen at the previous iterate:

$$\langle k_i\rangle = x_i A_i \;\Rightarrow\; x_i \leftarrow d_i / A_i(\theta_{\text{old}}),
\qquad
\langle s_i\rangle = y_i B_i \;\Rightarrow\; y_i \leftarrow s_i / B_i(\theta_{\text{old}}).$$

The two steps behave in **opposite** ways, and that asymmetry is the whole story. Write
$D(t) = (1-t)(1-t+g t)$ for the pair denominator.

**The degree step undershoots, and is safe.** Each term of $A_i$ has the form $c/(d_0 + c\,x_i)$ with
$d_0 = 1-t > 0$, so

$$\frac{\partial}{\partial x_i}\frac{c}{d_0 + c x_i} = -\frac{c^2}{(d_0+c x_i)^2} < 0 .$$

$A_i$ is strictly **decreasing** in $x_i$. Hence if $x_i < x_i^\*$ then $A_i(x_i) > A_i(x_i^\*)$ and the
update lands *below* the root: $x_i < x_i^{\text{new}} < x_i^\*$. It is a monotone contraction towards the
root, and $x_i$ has no upper constraint, so it can never be infeasible.

**The strength step overshoots, without bound.** $B_i = \sum_j w_j\,g_j y_j / D(y_i y_j)$, and

$$D'(0) = g - 2 < 0 \quad\text{whenever } x_i x_j < 2,$$

which is the entire sparse regime. So $D$ decreases, $B_i$ strictly **increases** in $y_i$, and if
$y_i < y_i^\*$ then $B_i(y_i) < B_i(y_i^\*)$ and the update lands *above* the root. Worse, the overshoot
is unbounded: as $y_i \to 0$, $B_i \to x_i\sum_j w_j x_j y_j$, so

$$y_i^{\text{new}} \longrightarrow \frac{s_i}{x_i \sum_j w_j x_j y_j} = O\!\left(\frac{1}{x^2 y}\right).$$

Any cold start has small $x$ and $y$. **Measured**, on the symmetrised rhesus UECM from the default
`:strengths` guess: the degree step lands inside $(x_i, x_i^\*)$ on *every* class, while the strength step
overshoots its own root by a factor **208–1383**, on every class, and every class lands beyond its own
feasibility ceiling $\bar y_i$. The map is outside the domain after **one** iteration, at which point
$1-t < 0$ and the next evaluation is meaningless.

Nothing in the recipe enforces $t<1$, and no amount of Anderson acceleration can repair a map whose first
image is outside the domain of the function being accelerated.

---

## 3. The replacement: solve each block exactly

Do not freeze anything — the one-dimensional problems are **unconditionally well posed**, because both
constraint functions are monotone in their own parameter:

* $\langle k_i\rangle(x_i)$ rises strictly from $0$ to $\sum_j w_j$. A root exists iff $d_i$ is below that
  ceiling — it is not for a *saturated* degree, which is a genuine runaway — and is then unique.
* $\langle s_i\rangle(y_i)$ rises strictly from $0$ to $\infty$ on $(0,\bar y_i)$, $\bar y_i = \min_j 1/y_j$.
  Strictness is

  $$\frac{d}{dt}\frac{g t}{D(t)} = \frac{g\bigl(1 - (1-g)t^2\bigr)}{D(t)^2} > 0
    \qquad\text{for all } t\in(0,1),\ g>0,$$

  (the bracket is $>0$ for $g\le 1$ because $(1-g)t^2 < 1$, and for $g>1$ because it is $1+(g-1)t^2$), and
  $D(t)\to 0$ as $t\to 1$ sends $\langle s_i\rangle \to \infty$. So the strength root **always** exists, is
  unique, and is **feasible by construction** — exactly the property the Picard step lacked.

Each solve is the exact maximisation of a concave objective in one coordinate, so a sweep of them
increases $L$ monotonically; $L$ is bounded above, so the iteration converges. That is the guarantee the
old map never had.

---

## 4. Runaway ridges, and the two safeguards the polish needs

Alternating one-dimensional solves converge, but they *crawl* whenever the optimum is at infinity in a
direction that no single coordinate spans. A class with $s_i = d_i$ — every incident link of weight
exactly 1 — has its optimum at $x_i\to\infty$ **and** $y_i\to 0$ *jointly*. Alternating moves approach
such a limit geometrically: measured on the rhesus DECM, the residual stalls at $2\times10^{-4}$ with a
per-sweep rate of $0.999823$, i.e. some $7\times10^{4}$ sweeps to reach $10^{-9}$.

Solving each node's **two** equations together in $(\log x_i, \log y_i)$ follows that ridge directly, and
helps the well-posed cases too (DECM median 43 sweeps → 9). Getting it to behave took two goes.

**Trap 1: a per-node Newton step accepted on its own residual cycles.** Near a runaway the node Jacobian
is nearly singular, so the step is enormous; it reduces that node's residual while wrecking the others,
and the next node undoes it. Measured: the global residual bounces around $4\times10^{-1}$ indefinitely.

**Trap 2: a fixed-point-*increment* stopping test hides that.** An orbit that cycles occasionally passes
near a repeat, the increment momentarily dips below `ftol`, and the solver reports success at whatever
residual it happened to be at. An earlier version of this work reported 30/50 and 71/79 on the runaway
sets that way. Those numbers were not real — both the merit function and the stopping test had to change
before the measurement meant anything. The driver now converges on the **constraint residual** (which is
also what `ftol` already means on the `CReM`/`DCReM`/`CRWCM` layers, so it is not a new convention here).

The shipped design keeps the per-node residual merit — it is what makes the step travel far along a ridge
— and safeguards it **at the sweep level**: the polish pass is applied to a copy and kept only if it
lowers the *global* constraint residual. That removes the cycling without giving up the ridge-following.

A fully monotone alternative was tried and rejected: line-searching each node on its own **block
likelihood** (whose gradient is exactly that node's residual, and which is concave, so the Newton
direction is a genuine ascent direction). It is provably convergent, but it takes vanishingly small steps
along a ridge, and it measured **worse**: UECM runaway 2/50 and DECM 11/79, against 8/50 and 22/79 for the
shipped safeguard, with worse residuals on the well-posed sets too.

---

## 5. What is fixed, and what is not

Measured over 150 random weighted networks (UECM) and 150 random weighted digraphs (DECM), from the
package's default `:strengths` cold start, through `solve_model!`. "Converged" means the realised degree
**and** strength sequences match the data to better than $10^{-6}$. The corpus is split by whether the
network carries a **runaway constraint** — $k = 0$, $k = N-1$ or $s = k$ — because those have their
optimum at an infinite parameter and are a different problem.

| model | subset | `:fixedpoint` (v0.7.0, Picard/Anderson) | `:fixedpoint` (now) | `:BFGS` |
|---|---|---|---|---|
| UECM | well-posed (100) | **0** | **98** | 86 |
| UECM | runaway present (50) | **0** | 8 | 38 |
| DECM | well-posed (71) | **0** | **71** | 71 |
| DECM | runaway present (79) | **0** | 22 | 78 |

Median residual on the well-posed sets: $3\times10^{-9}$ (UECM) and $1.5\times10^{-9}$ (DECM), against
$2.5\times10^{-7}$ and $3.1\times10^{-8}$ for `:BFGS`.

And it is far cheaper. Wall time for one cold-start solve through `solve_model!`, JIT warmed:

| network | `:fixedpoint` | `:BFGS` | residual (fp / BFGS) |
|---|---|---|---|
| UECM, 40 vertices | 0.008 s | 0.44 s | 5.3e-9 / 2.1e-6 |
| UECM, 80 vertices | 0.025 s | 3.6 s | 7.6e-9 / 1.1e-5 |
| UECM, 150 vertices | 0.124 s | 25.4 s | 8.0e-9 / 1.1e-5 |
| UECM, 250 vertices | 0.281 s | 57.1 s | 9.2e-9 / 1.4e-5 |
| DECM, 40 vertices | 0.013 s | 4.6 s | 7.2e-9 / 5.1e-9 |
| DECM, 150 vertices | 0.188 s | 593 s | 9.9e-9 / 1.7e-6 |

So on a well-posed network the fixed point is now the **most accurate** path as well as the cheapest, and
the gap widens with size: at 150 vertices the DECM solve goes from just under ten minutes to a fifth of a
second, with a residual two orders of magnitude smaller.

A stagnation guard — give up after N sweeps without a new best residual — was tried and **rejected**: it
makes failures fast but costs real convergences, because a runaway network that does eventually reach
`ftol` improves only intermittently. At a 300-sweep window the DECM runaway set fell from 22 of 79 to 10,
and even at 1000 it only recovered to 12. `maxiters` is already the knob for bounding the wait.

**What is not fixed.** With a runaway constraint present the map still usually fails to settle at `ftol`.
It no longer fails *silently* — it either reaches a genuinely small residual or throws, and the failure
now carries a diagnosis naming the offending constraint, reporting the best residual it did reach, and
pointing at `:BFGS`. Nothing here converges to a finite parameter on those networks because there is no
finite optimum to converge to; `:BFGS` succeeds because it can travel a long way in $\theta$ per step.

---

## 6. A defect found on the way: the UECM box was too tight

Chasing the domain question turned up a separate, shipped bug in the **first-order** path.

`L_UECM_reduced` evaluates its same-class term for every class and weights it by `Fᵢ(Fᵢ-1)/2`. For a
singleton class that weight is zero — but the out-of-domain branch returns `NaN`, and **`0 * NaN = NaN`**.
So `L` was non-finite on part of its own domain: any point with `βᵢ ≤ 0` on a singleton class, however
comfortably all the *real* pairs satisfied `βᵢ + βⱼ > 0`. (`L_DECM_reduced` already guarded this with
`iszero(w) && continue`; only the UECM's separately-written diagonal term was missing it.)

That is why `_UECM_β_FLOOR` was applied to the whole `β` block: the optimiser could not see past the
`NaN`, so the box was drawn at `βᵢ > 0 ∀i`. But as §1 shows, only classes with `Fᵢ ≥ 2` need that.

The cost was not hypothetical. On **5 of 128** random weighted networks the ML optimum has a `βᵢ < 0` on a
singleton class. There, `:BFGS` stopped against the floor and reported `Success` with a degree/strength
residual between **0.17 and 2.31** — a silently wrong fit, the same failure mode as the dead-channel bug
fixed in `v0.8.0`. Evaluated with the corrected likelihood, the true optimum is strictly better every
time (`ΔL` from `+3·10⁻⁴` to `+0.20`).

Both are fixed: the same-class term is skipped when `Fᵢ = 1`, and the box floors only the classes that
have a same-class pair. On the 6-vertex example `d = [2,1,3,2,2,2]`, `s = [5,5,9,5,5,3]`, `:BFGS` now
reaches `min β = -0.0163` with a residual of `1.1·10⁻⁷`, where before it reported success at `β = 0`
with a residual of `0.22`.

---

## 7. Summary

| | before | after |
|---|---|---|
| `:fixedpoint` on a well-posed UECM | 0/100 | 98/100, and more accurate than `:BFGS` |
| `:fixedpoint` on a well-posed DECM | 0/71 | 71/71, and more accurate than `:BFGS` |
| `:fixedpoint` with a runaway constraint | 0 | 8/50 (UECM), 22/79 (DECM); the rest fail loudly, with a diagnosis |
| `L_UECM_reduced` inside its own domain | `NaN` for `βᵢ ≤ 0` on a singleton class | finite |
| `:BFGS` when the optimum needs `βᵢ < 0` | `Success` at residual 0.17–2.31 | reaches the optimum |
