##################################################################################
# constraints.jl
#
# This file contains the constraint-residual diagnostic for the models of the
# MaxEntropyGraphs.jl package
##################################################################################


"""
    _constraint_residual(r::AbstractVector, obs::AbstractVector, relative::Bool)

Reduce the per-constraint residuals `r` (already undone of any multiplicity weighting) against the
observed sequence `obs` to a single maximum.

When `relative` is `false` this is `maximum(abs, r)`. When `relative` is `true` each residual is divided
by its observed value and the zero-valued constraints are masked out: dead channels (a zero degree or a
zero strength) occur throughout this model family and would otherwise divide by zero. If every constraint
is zero, the relative residual is defined to be zero.
"""
function _constraint_residual(r::AbstractVector, obs::AbstractVector, relative::Bool)
    T = float(eltype(r))
    isempty(r) && return zero(T)
    relative || return T(maximum(abs, r))
    res = zero(T)
    @inbounds for i in eachindex(r, obs)
        iszero(obs[i]) && continue # dead channel: no relative residual is defined
        res = max(res, abs(T(r[i]) / obs[i]))
    end
    return res
end


"""
    constraint_residual(m::AbstractMaxEntropyModel; relative::Bool=false)

Return the largest absolute constraint residual ``\\max_i |\\langle x_i \\rangle - x_i|`` of the solved
model `m`, i.e. how far the model's expected constrained sequence sits from the observed one. This is the
quantity that says whether a solve is actually good: it is measured in the units of the constraint itself
(degrees, or strengths/weights), so a residual of `1e-8` on a degree sequence means the expected degrees
reproduce the observed ones to eight decimals.

The maximum runs over **every** constrained sequence of the model, so for a model constraining several
sequences the answer is the worst case across all of them:

| model   | constrained sequences                                                     |
|:--------|:--------------------------------------------------------------------------|
| `UBCM`  | degree                                                                    |
| `DBCM`  | out-degree, in-degree                                                     |
| `BiCM`  | ⊥-layer degree, ⊤-layer degree                                            |
| `RBCM`  | non-reciprocated out-degree, non-reciprocated in-degree, reciprocated degree |
| `UECM`  | degree, strength                                                          |
| `DECM`  | out-degree, in-degree, out-strength, in-strength                          |
| `CReM`  | strength                                                                  |
| `DCReM` | out-strength, in-strength                                                 |
| `CRWCM` | non-reciprocated out/in-strength, reciprocated out/in-strength            |

Set `relative=true` to divide each residual by its observed value and obtain the largest **relative**
residual instead. Constraints whose observed value is zero (dead channels, which this model family
admits) are masked out of the relative form, where they are not defined.

!!! note "This is not `ftol`"
    The `ftol` keyword of [`solve_model!`](@ref) bounds a fixed-point *increment*, not the constraint
    residual, so it is a proxy rather than the measurement. On the binary models the increment lives in
    **parameter** space and tracks the degree residual up to a modest factor. On the weighted
    `CReM`/`DCReM`/`CRWCM` layers, which are solved in log-parameter space, it is the **relative**
    strength residual, so it is comparable to `constraint_residual(m, relative=true)` but says nothing
    directly about the absolute one (that carries the units of the weights). Either way
    `constraint_residual` measures the thing you actually care about, and is the recommended way to check
    a solve. For the gradient-based methods, `g_tol` bounds the gradient, which for these models *is* the
    constraint residual.

The residual is read off the model's analytical gradient (which, by the stationarity of these
exponential-family models, equals the expected-minus-observed sequence), so it costs a single gradient
evaluation, a fraction of the cost of a solve.

# Examples
```jldoctest constraint_residual
julia> model = UBCM(MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate));

julia> solve_model!(model);

julia> constraint_residual(model) < 1e-6 # expected degrees match the observed ones
true

julia> constraint_residual(model, relative=true) < 1e-6
true

```

For the two-step weighted models `ftol` is the *relative* tolerance, while the absolute residual
carries the units of the weights (the *rhesus macaques* strengths are of order `10²`, so it is
correspondingly larger):

```jldoctest constraint_residual
julia> wmodel = CReM(MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(rhesus_macaques()));

julia> solve_model!(wmodel, ftol=1e-8);

julia> constraint_residual(wmodel, relative=true) < 1e-8 # `ftol` bounds this one
true

julia> constraint_residual(wmodel) > 1e-8 # in strength units, so larger
true

```

See also: [`solve_model!`](@ref).
"""
function constraint_residual end


function constraint_residual(m::UBCM; relative::Bool=false)
    m.status[:params_computed] || throw(ArgumentError("The parameters have not been computed yet"))
    N = precision(m)
    n = length(m.dᵣ)
    ∇L = zeros(N, n)
    x  = zeros(N, n)
    ∇L_UBCM_reduced!(∇L, m.θᵣ, m.dᵣ, m.f, x)
    # the reduced gradient is multiplicity-weighted (∇L[r] = f[r]·(⟨k_r⟩ - k_r)): undo it
    return _constraint_residual(∇L ./ m.f, m.dᵣ, relative)
end


function constraint_residual(m::DBCM; relative::Bool=false)
    m.status[:params_computed] || throw(ArgumentError("The parameters have not been computed yet"))
    N = precision(m)
    n = m.status[:d_unique]::Int
    ∇L = zeros(N, length(m.θᵣ))
    x  = zeros(N, n)
    y  = zeros(N, n)
    ∇L_DBCM_reduced!(∇L, m.θᵣ, m.dᵣ_out, m.dᵣ_in, m.f, m.dᵣ_out_nz, m.dᵣ_in_nz, x, y, n)
    # undo the multiplicity weighting on both the out- and the in-block
    r = vcat(∇L[1:n] ./ m.f, ∇L[n+1:2*n] ./ m.f)
    return _constraint_residual(r, vcat(m.dᵣ_out, m.dᵣ_in), relative)
end


function constraint_residual(m::BiCM; relative::Bool=false)
    m.status[:params_computed] || throw(ArgumentError("The parameters have not been computed yet"))
    N = precision(m)
    n⊥ = m.status[:d⊥_unique]::Int
    ∇L = zeros(N, length(m.θᵣ))
    x  = zeros(N, length(m.d⊥ᵣ))
    y  = zeros(N, length(m.d⊤ᵣ))
    ∇L_BiCM_reduced!(∇L, m.θᵣ, m.d⊥ᵣ, m.d⊤ᵣ, m.f⊥, m.f⊤, m.d⊥ᵣ_nz, m.d⊤ᵣ_nz, x, y, n⊥)
    # each layer carries its own multiplicities
    r = vcat(∇L[1:n⊥] ./ m.f⊥, ∇L[n⊥+1:end] ./ m.f⊤)
    return _constraint_residual(r, vcat(m.d⊥ᵣ, m.d⊤ᵣ), relative)
end


function constraint_residual(m::RBCM; relative::Bool=false)
    m.status[:params_computed] || throw(ArgumentError("The parameters have not been computed yet"))
    N = precision(m)
    n = m.status[:d_unique]::Int
    ∇L = zeros(N, length(m.θᵣ))
    x  = zeros(N, n)
    y  = zeros(N, n)
    z  = zeros(N, n)
    ∇L_RBCM_reduced!(∇L, m.θᵣ, m.dᵣ_out, m.dᵣ_in, m.dᵣ_rec, m.f,
                     m.dᵣ_out_nz, m.dᵣ_in_nz, m.dᵣ_rec_nz, x, y, z, n)
    # undo the multiplicity weighting on the out-, in- and reciprocated block
    r = vcat(∇L[1:n] ./ m.f, ∇L[n+1:2*n] ./ m.f, ∇L[2*n+1:3*n] ./ m.f)
    return _constraint_residual(r, vcat(m.dᵣ_out, m.dᵣ_in, m.dᵣ_rec), relative)
end


function constraint_residual(m::UECM; relative::Bool=false)
    m.status[:params_computed] || throw(ArgumentError("The parameters have not been computed yet"))
    N = precision(m)
    n = length(m.dᵣ)
    ∇L = zeros(N, length(m.θᵣ))
    x  = zeros(N, n)
    y  = zeros(N, n)
    ∇L_UECM_reduced!(∇L, m.θᵣ, m.dᵣ, m.sᵣ, m.f, x, y, n)
    # undo the multiplicity weighting on the degree- and the strength-block
    r = vcat(∇L[1:n] ./ m.f, ∇L[n+1:2*n] ./ m.f)
    return _constraint_residual(r, vcat(m.dᵣ, m.sᵣ), relative)
end


function constraint_residual(m::DECM; relative::Bool=false)
    m.status[:params_computed] || throw(ArgumentError("The parameters have not been computed yet"))
    N = precision(m)
    n = length(m.dᵣ_out)
    ∇L    = zeros(N, length(m.θᵣ))
    x_out = zeros(N, n)
    x_in  = zeros(N, n)
    y_out = zeros(N, n)
    y_in  = zeros(N, n)
    ∇L_DECM_reduced!(∇L, m.θᵣ, m.dᵣ_out, m.dᵣ_in, m.sᵣ_out, m.sᵣ_in, m.f, x_out, x_in, y_out, y_in, n)
    # undo the multiplicity weighting on the out/in degree- and strength-blocks
    r = vcat(∇L[1:n] ./ m.f, ∇L[n+1:2*n] ./ m.f, ∇L[2*n+1:3*n] ./ m.f, ∇L[3*n+1:4*n] ./ m.f)
    return _constraint_residual(r, vcat(m.dᵣ_out, m.dᵣ_in, m.sᵣ_out, m.sᵣ_in), relative)
end


function constraint_residual(m::CReM; relative::Bool=false)
    m.status[:params_computed] || throw(ArgumentError("The parameters have not been computed yet"))
    N = precision(m)
    ∇L = zeros(N, length(m.θ))
    # the CReM gradient is a full (per-node) gradient: no multiplicity weighting to undo
    ∇L_CReM!(∇L, m.θ, m.s, m.xᵣ[m.dᵣ_ind])
    return _constraint_residual(∇L, m.s, relative)
end


function constraint_residual(m::DCReM; relative::Bool=false)
    m.status[:params_computed] || throw(ArgumentError("The parameters have not been computed yet"))
    N = precision(m)
    ∇L = zeros(N, length(m.θ))
    # full (per-node) gradient: no multiplicity weighting to undo
    ∇L_DCReM!(∇L, m.θ, m.s_out, m.s_in, m.xᵣ[m.dᵣ_ind], m.yᵣ[m.dᵣ_ind])
    return _constraint_residual(∇L, vcat(m.s_out, m.s_in), relative)
end


function constraint_residual(m::CRWCM; relative::Bool=false)
    m.status[:params_computed] || throw(ArgumentError("The parameters have not been computed yet"))
    N = precision(m)
    ∇L = zeros(N, length(m.θ))
    # full (per-node) gradient: no multiplicity weighting to undo
    ∇L_CRWCM!(∇L, m.θ, m.s_out, m.s_in, m.s_rec_out, m.s_rec_in,
              m.s_out_nz, m.s_in_nz, m.s_rec_nz,
              m.xᵣ[m.dᵣ_ind], m.yᵣ[m.dᵣ_ind], m.zᵣ[m.dᵣ_ind])
    return _constraint_residual(∇L, vcat(m.s_out, m.s_in, m.s_rec_out, m.s_rec_in), relative)
end
