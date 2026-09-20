# The BiCM Anderson-ladder study.
#
# This is the experiment behind `_anderson_memory_ladder` and behind the tables in
# validation/bicm_uecm_solver_geometry.md. It is separate from `sweep.jl` because it compares
# *internal* fixed-point strategies rather than the public API: the point is which accelerator
# setting to ship, not how the shipped solver behaves.
#
# The result it must reproduce (200 draws, 183 constructible):
#
#   plain (default m)  158/183      every failure a non-finite abort, never a timeout
#   m = 2              181/183
#   m = 20             108/183      failures rise monotonically with accelerator memory
#   Picard (m = 0)     183/183      but at 10 413 total iterations
#   ladder             183/183      at 4 551 total iterations
#   damped retry       169/183      damping is worse than doing nothing here
#   gauge projection   127/183      and projecting the gauge out is worse still
#
# The mechanism: the fixed-point map is gauge-equivariant, so the gauge direction `g` is an
# eigenvector of the Jacobian with eigenvalue exactly 1 and the residual Jacobian is singular
# along `g` by construction. Anderson solves a least-squares built from residual differences,
# every one of which therefore lies in the orthogonal complement of `g`; more memory means a more
# rank-deficient least-squares, hence the monotone rise in failures.
#
# Usage:  julia --project=. robustness/ladder.jl
#
# Environment variables:
#   ROB_NDRAWS   graphs drawn (default 200, which is what the published table used)
#   ROB_OUT      output JSON path

using Dates, JSON, LinearAlgebra, Printf, Random
include(joinpath(@__DIR__, "corpus.jl"))

const NL = MEG.NLsolve

"Strategies compared. `:ladder` is what the package actually ships."
const STRATEGIES = (:plain, :m2, :m20, :picard, :damped, :gauge, :ladder)

"""
    _pieces(m)

The raw fixed-point map, its starting point, the dead-class indices and the gauge vector, all as
`solve_model!` would build them. The gauge vector is restricted to the live entries: dead classes
are inert and stay at zero.
"""
function _pieces(m)
    n⊥ = m.status[:d⊥_unique]::Int
    nθ = length(m.θᵣ)
    xb = zeros(length(m.d⊥ᵣ)); yb = zeros(length(m.d⊤ᵣ)); Gb = zeros(nθ)
    ind = vcat(findall(iszero, m.d⊥ᵣ), length(m.d⊥ᵣ) .+ findall(iszero, m.d⊤ᵣ))
    θ₀ = MEG.initial_guess(m); θ₀[ind] .= 0.0
    gv = zeros(nθ)
    for i in m.d⊥ᵣ_nz; gv[i] = 1.0; end
    for j in m.d⊤ᵣ_nz; gv[n⊥ + j] = -1.0; end
    FP! = (θ::Vector) -> MEG.BiCM_reduced_iter!(θ, m.d⊥ᵣ, m.d⊤ᵣ, m.f⊥, m.f⊤,
                                                m.d⊥ᵣ_nz, m.d⊤ᵣ_nz, xb, yb, Gb, n⊥)
    FP!, θ₀, ind, gv
end

"Constraint reproduction of a fitted BiCM."
function _resid(m)
    MEG.set_Ĝ!(m)
    max(maximum(abs, vec(sum(m.Ĝ, dims = 2)) .- m.d⊥),
        maximum(abs, vec(sum(m.Ĝ, dims = 1)) .- m.d⊤))
end

"""
    run_strategy(m, strategy; ftol, maxiters) -> (status, residual, iterations)

`status` is one of `:ok`, `:maxiters`, `:nonfinite`, `:other`. The distinction matters: the whole
finding is that every failure of the accelerated path is a non-finite abort rather than a failure
to converge in time.
"""
function run_strategy(m, strategy; ftol = 1e-8, maxiters = 1000)
    FP!, θ₀, ind, gv = _pieces(m)
    gn = dot(gv, gv)
    proj! = (θ::Vector) -> (G = FP!(θ); G .-= (dot(gv, G) / gn) .* gv; G)
    fp(f, θ; kw...) = NL.fixedpoint(f, copy(θ); method = :anderson, ftol = ftol,
                                    iterations = maxiters, kw...)
    sol = try
        if strategy === :plain
            fp(FP!, θ₀)
        elseif strategy === :m2
            fp(FP!, θ₀; m = 2)
        elseif strategy === :m20
            fp(FP!, θ₀; m = 20)
        elseif strategy === :picard
            fp(FP!, θ₀; m = 0)
        elseif strategy === :damped
            try fp(FP!, θ₀)
            catch e
                e isa NL.IsFiniteException || rethrow()
                fp(FP!, θ₀; m = 5, beta = 0.5)
            end
        elseif strategy === :gauge
            θp = copy(θ₀); θp .-= (dot(gv, θp) / gn) .* gv
            fp(proj!, θp)
        elseif strategy === :ladder
            MEG._anderson_memory_ladder(FP!, copy(θ₀); ftol = ftol, maxiters = maxiters)
        else
            throw(ArgumentError("unknown strategy $(strategy)"))
        end
    catch e
        return (e isa NL.IsFiniteException ? :nonfinite : :other, NaN, 0)
    end
    NL.converged(sol) || return (:maxiters, NaN, sol.iterations)
    m.θᵣ .= sol.zero
    m.θᵣ[ind] .= Inf
    m.status[:params_computed] = true
    MEG.set_xᵣ!(m); MEG.set_yᵣ!(m)
    (:ok, _resid(m), sol.iterations)
end

function main()
    ndraws = parse(Int, get(ENV, "ROB_NDRAWS", "200"))
    rng = Xoshiro(SEED_BIPARTITE)
    graphs = Any[]
    for _ in 1:ndraws
        g = bipartite(rng, rand(rng, 4:30), rand(rng, 4:30), rand(rng) * 0.6 + 0.05)
        (try MEG.BiCM(g) catch; nothing end) === nothing && continue
        push!(graphs, g)
    end
    @info "BiCM Anderson ladder study" draws=ndraws constructible=length(graphs)

    rows = Dict{String,Any}[]
    for (gi, g) in enumerate(graphs), s in STRATEGIES
        st, r, it = run_strategy(MEG.BiCM(g), s)
        push!(rows, Dict{String,Any}("graph" => gi, "strategy" => string(s),
                                     "status" => string(st),
                                     "residual" => isnan(r) ? nothing : r,
                                     "iterations" => it))
    end

    println()
    @printf("%-10s %-10s %-10s %-11s %-8s %-10s %s\n",
            "strategy", "accurate", "maxiters", "nonfinite", "other", "total it", "worst resid")
    for s in STRATEGIES
        sub = filter(r -> r["strategy"] == string(s), rows)
        acc = filter(r -> r["status"] == "ok" && r["residual"] !== nothing && r["residual"] < 1e-6, sub)
        cnt(x) = count(r -> r["status"] == x, sub)
        worst = isempty(acc) ? NaN : maximum(r -> r["residual"], acc)
        @printf("%-10s %-10s %-10d %-11d %-8d %-10d %.2e\n", s,
                string(length(acc), "/", length(graphs)), cnt("maxiters"), cnt("nonfinite"),
                cnt("other"), sum(r -> r["iterations"], acc; init = 0), worst)
    end

    outdir = joinpath(@__DIR__, "results"); mkpath(outdir)
    stamp = Dates.format(now(), "yyyymmdd-HHMMSS")
    outfile = get(ENV, "ROB_OUT", joinpath(outdir, "ladder_julia-$(VERSION)_$(stamp).json"))
    open(outfile, "w") do io
        JSON.print(io, Dict("meta" => Dict("julia" => string(VERSION),
                                           "timestamp" => string(now()),
                                           "draws" => ndraws,
                                           "constructible" => length(graphs),
                                           "seed" => SEED_BIPARTITE,
                                           "package_version" => string(pkgversion(MaxEntropyGraphs))),
                            "rows" => rows))
    end
    @info "wrote" outfile
    outfile
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
