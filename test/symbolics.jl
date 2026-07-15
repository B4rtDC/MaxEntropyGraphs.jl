# Fast symbolic CI subset — per-dyad moment identities for every model.
#
# Distilled from validation/symbolic/*.jl (see validation/README.md for the full
# derivations, code-form cross-references and numeric closures). This file keeps only
# the pure per-dyad moment identities: no model solving, no matrices, no Monte-Carlo.
#
# Equality oracle: `ratzero` substitutes exact Rational{BigInt} points drawn from each
# variable's open domain box. A rational function that vanishes at many random points is
# identically zero (up to vanishing probability), and exact arithmetic avoids both
# floating-point doubt and `simplify()` stalls. All identities are kept sqrt-free
# (variances and squared code forms only).
#
# Skip switch: set the environment variable MEG_SKIP_SYMBOLIC=1 to skip this whole file
# (e.g. on CI runners where loading Symbolics is undesirable). The guard is split over
# two top-level `if` blocks (the first loads Symbolics only when not skipping), and the
# test code deliberately avoids Symbolics *macros* (`@variables`): a guarded block is
# still macro-expanded even when its condition is false, so symbolic variables are made
# with the runtime function form `Symbolics.variable` (see `svars` below) instead.

if get(ENV, "MEG_SKIP_SYMBOLIC", "0") != "1"
    using Symbolics
    using Random: MersenneTwister
end

if get(ENV, "MEG_SKIP_SYMBOLIC", "0") != "1"

@testset "symbolic dyad-level moment checks" begin

    # Exact multi-point zero-test for rational expressions.
    # `domains` is a vector of `var => (lo, hi)` pairs (lo/hi rationals or integers).
    function ratzero(expr, domains::AbstractVector; npoints::Int=8, seed::Int=161)
        rng = MersenneTwister(seed)
        vars = [first(d) for d in domains]
        for _ in 1:npoints
            vals = map(domains) do d
                lo = Rational{BigInt}(d.second[1])
                hi = Rational{BigInt}(d.second[2])
                k = rand(rng, 1:(2^20 - 1))
                lo + (hi - lo) * (BigInt(k) // BigInt(2^20))
            end
            raw = Symbolics.value(Symbolics.substitute(expr, Dict(zip(vars, vals))))
            raw isa Number || return false
            iszero(raw) || return false
        end
        return true
    end

    # runtime replacement for `@variables ...` (macro-free so the file lowers when skipped)
    svars(names::Symbol...) = Tuple(Symbolics.variable(n) for n in names)

    @testset "UBCM" begin
        # single Bernoulli dyad: weight v^a with v = x_i x_j, a ∈ {0,1}
        xi, xj = svars(:xi, :xj)
        v = xi * xj
        doms = [xi => (1//100, 20), xj => (1//100, 20)]
        states = (0, 1)
        Z  = sum(v^a for a in states)
        Ea = sum(a * v^a for a in states) / Z
        Ea2 = sum(a^2 * v^a for a in states) / Z
        p  = v / (1 + v)
        # ⟨a⟩ from the state sum equals v/(1+v)
        @test ratzero(Ea - p, doms)
        # Var[a] = ⟨a²⟩ − ⟨a⟩² equals p(1−p)
        @test ratzero((Ea2 - Ea^2) - p * (1 - p), doms)
    end

    @testset "DBCM" begin
        # dyad (a_ij, a_ji) ∈ {0,1}² with factorized weight (x_i y_j)^a (x_j y_i)^b
        xi, xj, yi, yj = svars(:xi, :xj, :yi, :yj)
        doms = [xi => (1//100, 20), xj => (1//100, 20),
                yi => (1//100, 20), yj => (1//100, 20)]
        states = ((0, 0), (1, 0), (0, 1), (1, 1))
        W(a, b) = (xi * yj)^a * (xj * yi)^b
        Z      = sum(W(a, b) for (a, b) in states)
        E_aij  = sum(a * W(a, b) for (a, b) in states) / Z
        E_aji  = sum(b * W(a, b) for (a, b) in states) / Z
        E_prod = sum(a * b * W(a, b) for (a, b) in states) / Z
        p_ij = xi * yj / (1 + xi * yj)
        # ⟨a_ij⟩ from the 4-state sum equals x_i y_j/(1 + x_i y_j)
        @test ratzero(E_aij - p_ij, doms)
        # Var[a_ij] equals p(1−p) (a² = a for a ∈ {0,1})
        @test ratzero((E_aij - E_aij^2) - p_ij * (1 - p_ij), doms)
        # the two channels of a dyad are independent: Cov(a_ij, a_ji) = 0
        @test ratzero(E_prod - E_aij * E_aji, doms)
    end

    @testset "RBCM" begin
        # 4-state reciprocal dyad: weights {1, x_i y_j, x_j y_i, z_i z_j}
        xi, xj, yi, yj, zi, zj = svars(:xi, :xj, :yi, :yj, :zi, :zj)
        doms = [xi => (1//100, 10), xj => (1//100, 10),
                yi => (1//100, 10), yj => (1//100, 10),
                zi => (1//100, 10), zj => (1//100, 10)]
        Z    = 1 + xi * yj + xj * yi + zi * zj
        a_ij = (xi * yj + zi * zj) / Z          # state-sum ⟨a_ij⟩: states (1,0) and (1,1)
        a_ji = (xj * yi + zi * zj) / Z
        # ⟨a_ij⟩ equals the Ĝ code form (x_i y_j + z_i z_j)/Z
        @test ratzero((xi * yj / Z + zi * zj / Z) - a_ij, doms)
        # Var[a_ij] = ⟨a_ij⟩(1 − ⟨a_ij⟩) (Bernoulli marginal)
        @test ratzero((a_ij - a_ij^2) - a_ij * (1 - a_ij), doms)
        # within-dyad covariance: ⟨a_ij a_ji⟩ = p↔ = z_i z_j/Z (only state (1,1)) hence
        # Cov(a_ij, a_ji) = z_i z_j/Z − ⟨a_ij⟩⟨a_ji⟩ (the _cov_dyads code form)
        cov_stat = zi * zj / Z - a_ij * a_ji
        cov_code = zi * zj / Z - ((xi * yj + zi * zj) / Z) * ((xj * yi + zi * zj) / Z)
        @test ratzero(cov_stat - cov_code, doms)
    end

    @testset "BiCM" begin
        # biadjacency entry m ∈ {0,1} with weight (x y)^m; distinct entries independent
        x, y, x1, y1, x2, y2, u = svars(:x, :y, :x1, :y1, :x2, :y2, :u)
        doms   = [x => (1//100, 20), y => (1//100, 20)]
        doms4  = [x1 => (1//100, 20), y1 => (1//100, 20),
                  x2 => (1//100, 20), y2 => (1//100, 20)]
        Z   = sum((x*y)^m for m in 0:1)
        Em  = sum(m * (x*y)^m for m in 0:1) / Z
        Em2 = sum(m^2 * (x*y)^m for m in 0:1) / Z
        p   = x * y / (1 + x * y)
        # ⟨m⟩ from the state sum equals x y/(1 + x y)
        @test ratzero(Em - p, doms)
        # Var[m] equals p(1−p)
        @test ratzero((Em2 - Em^2) - p * (1 - p), doms)
        # independence of distinct entries (joint weight factorizes): Cov = 0
        W2(m1, m2) = (x1*y1)^m1 * (x2*y2)^m2
        Z2    = sum(W2(m1, m2) for m1 in 0:1, m2 in 0:1)
        Em1m2 = sum(m1 * m2 * W2(m1, m2) for m1 in 0:1, m2 in 0:1) / Z2
        p1 = x1 * y1 / (1 + x1 * y1)
        p2 = x2 * y2 / (1 + x2 * y2)
        @test ratzero(Em1m2 - p1 * p2, doms4)
        # Saracco et al. 2015 SI III.7 (n = 2), used by the Vn/Λn machinery:
        # d/du C(u,2) == C(u,2)·(1/u + 1/(u−1)) with C(u,2) = u(u−1)/2
        B2  = u * (u - 1) / 2
        dB2 = Symbolics.derivative(B2, u)
        @test ratzero(dB2 - B2 * (1/u + 1/(u - 1)), [u => (5, 50)])
    end

    @testset "UECM" begin
        # Bernoulli–geometric dyad weight via the probability generating function
        # G(t) = ⟨tʷ⟩ = (1 + x t y/(1 − t y))/Z with Z = 1 + x y/(1 − y)
        x, y, t = svars(:x, :y, :t)
        doms = [x => (1//10, 5), y => (1//100, 9//10)]
        Z  = 1 + x * y / (1 - y)
        Gt = (1 + x * t * y / (1 - t * y)) / Z
        p_code = (x * y) / (1 - y + x * y)                  # f_UECM / Ĝ code form
        # p = P(w>0) = 1 − G(0)
        p_pgf = 1 - Symbolics.substitute(Gt, Dict(t => 0))
        @test ratzero(p_pgf - p_code, doms)
        # ⟨w⟩ = G'(1) equals the Ŵ code form p/(1−y)
        dG = Symbolics.derivative(Gt, t)
        w1 = Symbolics.substitute(dG, Dict(t => 1))
        w1_code = p_code / (1 - y)
        @test ratzero(w1 - w1_code, doms)
        # Var[w] = G''(1) + G'(1) − G'(1)² equals the σʷ form p(1+y−p)/(1−y)²
        w2 = Symbolics.substitute(Symbolics.derivative(dG, t), Dict(t => 1)) + w1
        @test ratzero((w2 - w1^2) - p_code * (1 + y - p_code) / (1 - y)^2, doms)
        # cross-layer covariance: ⟨a·w⟩ = ⟨w⟩ (w>0 ⟺ a=1) ⇒ Cov(a,w) = ⟨w⟩(1−p)
        aw = x * y / ((1 - y)^2 * Z)
        @test ratzero((aw - p_code * w1) - w1_code * (1 - p_code), doms)
    end

    @testset "CReM" begin
        # zero-inflated exponential weight via the MGF M(t) = (1−f) + f·ts/(ts−t), ts = θ_i+θ_j
        f, ti, tj, t = svars(:f, :ti, :tj, :t)
        doms = [f => (1//100, 99//100), ti => (1//10, 10), tj => (1//10, 10)]
        ts = ti + tj
        M  = (1 - f) + f * ts / (ts - t)
        Mp  = Symbolics.derivative(M, t)
        Mpp = Symbolics.derivative(Mp, t)
        meanw = Symbolics.substitute(Mp,  Dict(t => 0))     # ⟨w⟩  = M'(0)
        m2w   = Symbolics.substitute(Mpp, Dict(t => 0))     # ⟨w²⟩ = M''(0)
        # ⟨w⟩ equals the Ŵ code form f/(θ_i+θ_j)
        @test ratzero(meanw - f / ts, doms)
        # Var[w] equals the σʷ form f(2−f)/(θ_i+θ_j)²
        @test ratzero((m2w - meanw^2) - f * (2 - f) / ts^2, doms)
    end

    @testset "DCReM" begin
        # directed twin: f_ij = x_i y_j/(1+x_i y_j), rate sum ts = θᵒ_i + θⁱ_j;
        # the joint MGF factorizes over the two directions of a dyad
        xi, xj, yi, yj, toi, tii, toj, tij, s, t =
            svars(:xi, :xj, :yi, :yj, :toi, :tii, :toj, :tij, :s, :t)
        doms = [xi => (1//100, 20), yj => (1//100, 20),
                toi => (1//10, 10), tij => (1//10, 10)]
        doms_dyad = vcat(doms, [xj => (1//100, 20), yi => (1//100, 20),
                                toj => (1//10, 10), tii => (1//10, 10)])
        at0(expr) = Symbolics.substitute(expr, Dict(s => 0, t => 0))
        M(f, r, u) = (1 - f) + f * r / (r - u)
        f_ij = xi * yj / (1 + xi * yj)
        ts   = toi + tij
        M_ij = M(f_ij, ts, s)
        E_w  = at0(Symbolics.derivative(M_ij, s))
        E_w2 = at0(Symbolics.derivative(Symbolics.derivative(M_ij, s), s))
        # ⟨w⟩ equals the Ŵ code form f/(θᵒ_i + θⁱ_j)
        @test ratzero(E_w - f_ij / ts, doms)
        # Var[w] equals the σʷ form f(2−f)/(θᵒ_i + θⁱ_j)²
        @test ratzero((E_w2 - E_w^2) - f_ij * (2 - f_ij) / ts^2, doms)
        # independence across directions: M(s,t) = M_ij(s)·M_ji(t) ⇒ Cov(w_ij, w_ji) = 0
        f_ji = xj * yi / (1 + xj * yi)
        M_ji = M(f_ji, toj + tii, t)
        M_joint = M_ij * M_ji
        E_prod = at0(Symbolics.derivative(Symbolics.derivative(M_joint, s), t))
        E_wr   = at0(Symbolics.derivative(M_ji, t))
        @test ratzero(E_prod - E_w * E_wr, doms_dyad)
    end

    @testset "CRWCM" begin
        # RBCM binary layer + conditionally exponential weights; joint MGF of (w_ij, w_ji):
        # M(s,t) = p0 + p→·r1/(r1−s) + p←·r2/(r2−t) + p↔·(r3/(r3−s))·(r4/(r4−t))
        xi, xj, yi, yj, zi, zj, r1, r2, r3, r4, s, t =
            svars(:xi, :xj, :yi, :yj, :zi, :zj, :r1, :r2, :r3, :r4, :s, :t)
        doms = [xi => (1//100, 5), xj => (1//100, 5),
                yi => (1//100, 5), yj => (1//100, 5),
                zi => (1//100, 5), zj => (1//100, 5),
                r1 => (1//10, 10), r2 => (1//10, 10),
                r3 => (1//10, 10), r4 => (1//10, 10)]
        D  = 1 + xi * yj + xj * yi + zi * zj
        p0 = 1 / D
        pr = xi * yj / D          # p→
        pl = xj * yi / D          # p←
        pb = zi * zj / D          # p↔
        M = p0 + pr * r1 / (r1 - s) + pl * r2 / (r2 - t) + pb * (r3 / (r3 - s)) * (r4 / (r4 - t))
        at0(expr) = Symbolics.substitute(expr, Dict(s => 0, t => 0))
        # MGF normalisation: M(0,0) = 1
        @test ratzero(at0(M) - 1, doms)
        # ⟨w_ij⟩ = ∂M/∂s|₀ equals the Ŵ code form p→/r1 + p↔/r3
        mean_wij = at0(Symbolics.derivative(M, s))
        @test ratzero(mean_wij - (pr / r1 + pb / r3), doms)
        # cross moment: ⟨w_ij w_ji⟩ = ∂²M/∂s∂t|₀ equals p↔/(r3 r4) (the _covʷ code form)
        cross_w = at0(Symbolics.derivative(Symbolics.derivative(M, s), t))
        @test ratzero(cross_w - pb / (r3 * r4), doms)
    end

end

end # MEG_SKIP_SYMBOLIC
