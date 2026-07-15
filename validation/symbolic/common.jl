# Shared helpers for the symbolic validation scripts.
#
# Each script derives the per-dyad moments (⟨g⟩, ⟨g²⟩, Var[g], Cov[g_ij, g_ji]) of one model
# from its dyadic distribution and checks them against the closed forms implemented in src/.
# All checks are stated as "expression must be identically zero".
#
# Equality oracle strategy (see validation/README.md):
#   1. `symzero` attempts a structural proof via expand/simplify_fractions/simplify.
#   2. `ratzero` is the authoritative fallback: substitute exact Rational{BigInt} points drawn
#      from each variable's domain. A rational function that vanishes at many random points of
#      an open box is identically zero (up to vanishing probability), and exact arithmetic
#      avoids both floating-point doubt and simplification stalls.
#
# NOTE: keep every checked identity sqrt-free (compare variances, not standard deviations;
# compare squared code forms), otherwise rational substitution leaves the rationals.

using Symbolics
using Random: MersenneTwister

const RESULTS = Tuple{String,Bool}[]

"Structural zero-proof attempt. Returns true only on success; false means 'unknown'."
function symzero(expr)
    try
        s = Symbolics.simplify(Symbolics.expand(Symbolics.simplify_fractions(expr)))
        return iszero(s)
    catch
        return false
    end
end

"""
    ratzero(expr, domains; npoints=20, seed=161)

Authoritative zero-test for rational expressions: substitute `npoints` exact
`Rational{BigInt}` values drawn from the open interval of each variable's domain.
`domains` is a vector of `var => (lo, hi)` pairs (lo/hi rationals or integers).
"""
function ratzero(expr, domains::AbstractVector; npoints::Int=20, seed::Int=161)
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

"Record a zero-identity check: structural proof first, exact rational substitution as authority."
function verify(name::String, expr, domains; kwargs...)
    pass = symzero(expr) || ratzero(expr, domains; kwargs...)
    push!(RESULTS, (name, pass))
    pass || @warn "FAILED: $name"
    return pass
end

"Record a numeric closure check (symbolic value vs the actual package function's output)."
function closecheck(name::String, a::Real, b::Real; rtol::Real=1e-8)
    pass = isapprox(a, b; rtol=rtol)
    push!(RESULTS, (name, pass))
    pass || @warn "FAILED: $name ($a vs $b)"
    return pass
end

"Print the pass/fail table and exit non-zero on any failure."
function report(modelname::String)
    println("\n=== $modelname — symbolic validation ===")
    for (n, p) in RESULTS
        println(rpad(p ? "PASS" : "FAIL", 6), n)
    end
    nfail = count(r -> !r[2], RESULTS)
    println(nfail == 0 ? "ALL PASS ($(length(RESULTS)) checks)" : "$nfail FAILURE(S)")
    exit(nfail == 0 ? 0 : 1)
end
