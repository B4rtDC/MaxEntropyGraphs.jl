##################################################################################################
# Metric-acceleration benchmarks
#
# Head-to-head timing AND memory of the accelerated metric kernels in `src/metrics.jl` against
# reference (pre-acceleration) implementations, across a per-metric-adaptive size sweep. Confirms
# that the reformulations are an actual gain and reports the scaling / peak-memory behaviour.
#
# Usage (from the `performance/` project):
#   julia --project=. metrics_benchmarks.jl                 # full sweep
#   BENCH_QUICK=1 julia --project=. metrics_benchmarks.jl   # smaller sizes, faster
#
# Notes:
#   * BLAS threads are pinned to 1 by default (`BENCH_BLAS_THREADS`) because these metrics are typically
#     mapped over MANY graphs inside a `Threads.@threads` outer loop (rand(m, n) / significance sweeps);
#     the outer pool should own the parallelism so per-call BLAS does not oversubscribe. A second value
#     (e.g. `BENCH_BLAS_THREADS=0` -> library default) shows the single-graph best case.
#   * "alloc" is total bytes allocated per call. For the BLAS-3 paths (motifs, and triangles' generic
#     path) this is an O(N^2) temporary; triangles' concrete path streams column-by-column so its PEAK
#     live memory is O(N) even though the churn is O(N^2). squares/ANND/V_motifs keep allocation small.
##################################################################################################

using MaxEntropyGraphs
const MEG = MaxEntropyGraphs
using MaxEntropyGraphs: Graphs
using BenchmarkTools, LinearAlgebra, Random, Printf, JSON, Dates

# ---- BLAS thread configuration -------------------------------------------------------------------
const BLAS_THREADS = parse(Int, get(ENV, "BENCH_BLAS_THREADS", "1"))
BLAS_THREADS > 0 && BLAS.set_num_threads(BLAS_THREADS)
# Test the value, not mere presence: benchmarks.sh always exports BENCH_QUICK (defaulting it to 0),
# so `haskey` would silently force QUICK mode even when it is explicitly disabled.
const QUICK = get(ENV, "BENCH_QUICK", "0") == "1"

# ---- reference (pre-acceleration) implementations ------------------------------------------------
function ref_triangles(A)
    res = zero(eltype(A))
    for i in axes(A,1), j in axes(A,1), k in axes(A,1)
        (i != j && j != k && k != i) && (res += A[i,j]*A[j,k]*A[k,i])
    end
    return res/6
end
function ref_squares(A)
    res = zero(eltype(A)); o = one(eltype(A))
    for i in axes(A,1), j in axes(A,1)
        j == i && continue
        for k in axes(A,1)
            (k==i||k==j) && continue
            for l in axes(A,1)
                (l==i||l==j||l==k) && continue
                res += A[i,j]*A[j,k]*A[k,l]*A[l,i]*(o-A[i,k])*(o-A[l,j])
            end
        end
    end
    return res/8
end
function ref_ANND(A)
    N = size(A,1); out = zeros(Float64, N)
    for i in 1:N
        di = sum(@view A[:,i])
        out[i] = iszero(di) ? 0.0 : mapreduce(x -> A[i,x]*sum(@view A[:,x]), +, 1:N)/di
    end
    return out
end
r_arr(A,i,j)=A[i,j]*(one(eltype(A))-A[j,i]); r_bak(A,i,j)=(one(eltype(A))-A[i,j])*A[j,i]
r_rec(A,i,j)=A[i,j]*A[j,i];                  r_abs(A,i,j)=(one(eltype(A))-A[i,j])*(one(eltype(A))-A[j,i])
const REFT = [ (r_bak,r_arr,r_abs),(r_bak,r_bak,r_abs),(r_bak,r_rec,r_abs),(r_bak,r_abs,r_arr),
    (r_bak,r_arr,r_arr),(r_bak,r_rec,r_arr),(r_arr,r_rec,r_abs),(r_rec,r_rec,r_abs),(r_arr,r_arr,r_arr),
    (r_rec,r_arr,r_arr),(r_rec,r_bak,r_arr),(r_rec,r_rec,r_arr),(r_rec,r_rec,r_rec) ]
function ref_motif(A,f1,f2,f3)
    res = zero(eltype(A))
    for i in axes(A,1), j in axes(A,1), k in axes(A,1)
        (i != j && j != k && k != i) && (res += f1(A,i,j)*f2(A,j,k)*f3(A,k,i))
    end
    return res
end
ref_motifs(A) = [ref_motif(A, t...) for t in REFT]
function ref_fluxes(W)
    n = size(W,1)
    A = (!iszero).(W) .* 1.0
    a(l,i,j) = l === :P ? A[i,j]*(1-A[j,i]) : l === :Q ? (1-A[i,j])*A[j,i] : l === :R ? A[i,j]*A[j,i] : (1-A[i,j])*(1-A[j,i])
    w(l,i,j) = l === :P ? W[i,j] : l === :Q ? W[j,i] : l === :R ? W[i,j]+W[j,i] : 0.0
    res = zeros(13)
    for (k, sp) in enumerate(MEG._motif_specs)
        for i in 1:n, j in 1:n, l in 1:n
            (i == j || j == l || i == l) && continue
            ind = a(sp[1],i,j)*a(sp[2],j,l)*a(sp[3],l,i)
            iszero(ind) && continue
            res[k] += ind * (w(sp[1],i,j) + w(sp[2],j,l) + w(sp[3],l,i))
        end
    end
    return res
end
# pre-acceleration intensities: per-triple Dict lookup + symbol branching (the original implementation)
function ref_intensities(W)
    n = size(W,1)
    res = zeros(float(eltype(W)), 13)
    state_index = Dict{NTuple{3,Symbol},Int}((MEG._motif_specs[k][1], MEG._motif_specs[k][2], MEG._motif_specs[k][3]) => k for k in 1:13)
    state(i, j) = @inbounds !iszero(W[i,j]) ? (!iszero(W[j,i]) ? :R : :P) : (!iszero(W[j,i]) ? :Q : :Z)
    @inbounds for i in 1:n, j in 1:n, k in 1:n
        (i == j || j == k || i == k) && continue
        sp = (state(i,j), state(j,k), state(k,i))
        idx = get(state_index, sp, 0)
        iszero(idx) && continue
        prod_w = one(float(eltype(W))); nlinks = 0
        for (a, b, st) in ((i,j,sp[1]), (j,k,sp[2]), (k,i,sp[3]))
            if st === :P
                prod_w *= W[a,b]; nlinks += 1
            elseif st === :Q
                prod_w *= W[b,a]; nlinks += 1
            elseif st === :R
                prod_w *= W[a,b] * W[b,a]; nlinks += 2
            end
        end
        res[idx] += prod_w^(1/nlinks)
    end
    return res
end
function ref_Vmotifs(A, layer)
    res = zero(eltype(A))
    if layer == :bottom
        for i in axes(A,1), j in axes(A,1); j>i && (res += dot(@view(A[i,:]), @view(A[j,:]))); end
    else
        for i in axes(A,2), j in axes(A,2); j>i && (res += dot(@view(A[:,i]), @view(A[:,j]))); end
    end
    return res
end

# ---- inputs --------------------------------------------------------------------------------------
sym01(N; k=6, seed=161)  = Matrix(Graphs.adjacency_matrix(Graphs.barabasi_albert(N, k, seed=seed)))      # dense 0/1 symmetric
symreal(N; seed=1) = (rng=Xoshiro(seed); M=rand(rng,N,N); S=(M.+M')./2; S[diagind(S)].=0.0; S)          # dense real in [0,1]
dirreal(N; seed=2) = (rng=Xoshiro(seed); D=rand(rng,N,N); D[diagind(D)].=0.0; D)                         # dense directed real
sparse01(N; k=6, seed=161) = Graphs.adjacency_matrix(Graphs.barabasi_albert(N, k, seed=seed))            # sparse 0/1
birand(N,M; seed=3) = (rng=Xoshiro(seed); rand(rng, N, M))                                               # dense biadjacency
# directed weighted with ~35% link density: all four dyad states (→, ←, ↔, absent) are populated
dirsparse(N; seed=4) = (rng=Xoshiro(seed); D=rand(rng,N,N).*(rand(rng,N,N).<0.35); D[diagind(D)].=0.0; D)

# ---- measurement helper --------------------------------------------------------------------------
# returns (time_seconds, bytes_allocated); short budget to keep the whole sweep tractable
function measure(f, A)
    t = @belapsed $f($A) samples=5 seconds=2 evals=1
    b = @allocated f(A)
    return t, b
end

const RESULTS = Dict{String,Any}()

function head2head(name, newf, reff, inputs; ref_max=typemax(Int))
    println("\n### $name  (new vs reference)")
    @printf("%8s | %12s %12s %9s | %12s %12s\n", "N", "new (s)", "old (s)", "speedup", "new alloc", "old alloc")
    rows = []
    for (N, A) in inputs
        tn, bn = measure(newf, A)
        if N <= ref_max
            to, bo = measure(reff, A)
            @printf("%8d | %12.3e %12.3e %8.1fx | %12s %12s\n", N, tn, to, to/tn, Base.format_bytes(bn), Base.format_bytes(bo))
            push!(rows, Dict("N"=>N, "new_s"=>tn, "old_s"=>to, "speedup"=>to/tn, "new_bytes"=>bn, "old_bytes"=>bo))
        else
            @printf("%8d | %12.3e %12s %9s | %12s %12s\n", N, tn, "(skipped)", "-", Base.format_bytes(bn), "-")
            push!(rows, Dict("N"=>N, "new_s"=>tn, "new_bytes"=>bn))
        end
    end
    RESULTS[name] = rows
end

# ---- sizes ---------------------------------------------------------------------------------------
S(small, big) = QUICK ? small : big

@info "metric benchmarks" julia=string(VERSION) blas_threads=BLAS.get_num_threads() julia_threads=Threads.nthreads() quick=QUICK

# ANND — O(N^3) -> O(N^2): head-to-head where the reference is feasible, then new-only at scale
head2head("ANND", ANND, ref_ANND,
    [(N, symreal(N)) for N in S([100,300], [100,400,1000,3000,10000])]; ref_max = QUICK ? 300 : 1000)

# triangles — branchy O(N^3) loop vs BLAS
head2head("triangles", triangles, ref_triangles,
    [(N, symreal(N)) for N in S([100,300], [100,400,1000,3000])]; ref_max = QUICK ? 300 : 800)

# directed motifs (full spectrum) — 13 O(N^3) loops vs shared-base matrix form
head2head("motifs (all 13)", motifs, ref_motifs,
    [(N, dirreal(N)) for N in S([50,150], [50,150,400,1000])]; ref_max = QUICK ? 150 : 300)

# squares — dense: 8-fold-symmetric kernel vs naive quadruple loop (O(N^4), small N only)
head2head("squares (dense)", squares, ref_squares,
    [(N, symreal(N)) for N in S([25,50], [25,50,100,150])]; ref_max = QUICK ? 50 : 150)

# squares — sparse 0/1 fast path (neighbour enumeration): tractable at large N
println("\n### squares (sparse 0/1 fast-path)  (new only)")
@printf("%8s | %12s %12s\n", "N", "new (s)", "new alloc")
let rows = []
    for N in S([1000,10000], [1000,10000,50000])
        A = sparse01(N); t, b = measure(squares, A)
        @printf("%8d | %12.3e %12s\n", N, t, Base.format_bytes(b))
        push!(rows, Dict("N"=>N, "new_s"=>t, "new_bytes"=>b))
    end
    RESULTS["squares (sparse)"] = rows
end

# V_motifs — pairwise-dot loop vs closed form; head-to-head then new-only at scale
head2head("V_motifs", A->V_motifs(A, layer=:bottom, skipchecks=true), A->ref_Vmotifs(A, :bottom),
    [(N, birand(N, N)) for N in S([100,400], [100,400,1000,3000,10000])]; ref_max = QUICK ? 400 : 1000)

# triadic fluxes — per-term matrix products vs the shared cyclic-trace form (and both vs the naive
# O(N^3) triple loop as reference)
head2head("motif_fluxes", motif_fluxes, ref_fluxes,
    [(N, dirsparse(N)) for N in S([50,150], [50,150,400,1000])]; ref_max = QUICK ? 150 : 300)

# triadic intensities — precomputed state/weight/link tables vs the per-triple Dict/symbol version
# (both are O(N^3); the gain is the constant factor)
head2head("motif_intensities", motif_intensities, ref_intensities,
    [(N, dirsparse(N)) for N in S([50,150], [50,150,300,600])]; ref_max = QUICK ? 150 : 300)

# ---- persist -------------------------------------------------------------------------------------
outdir = joinpath(@__DIR__, "benchmarks", "Julia-$(VERSION)")
isdir(outdir) || mkpath(outdir)
meta = Dict("julia"=>string(VERSION), "blas_threads"=>BLAS.get_num_threads(),
            "julia_threads"=>Threads.nthreads(), "quick"=>QUICK, "results"=>RESULTS)
open(joinpath(outdir, "metrics_acceleration.json"), "w") do io; JSON.print(io, meta, 2); end
println("\nsaved: ", joinpath(outdir, "metrics_acceleration.json"))
