"""
Idea: starting from models with known parameters:
- obtain expected values and variances for adjacency/weight matrix elements
- sample networks, returning 
    1. Adjacency matrix (dense/ sparse (ULT)) 
    2. Graph 
    3. Adjecency List & node number
- compute z-scores of different metrics by 
    1. "exact" method 
    2. sampling method
"""

import Graphs                   # network interface for graphs
import PyCall                   # for calling NEMtropy package in Python
import ReverseDiff              # for gradient
import Statistics: std, mean    # for statistics
import Printf: @sprintf         # for specific printing
using Plots                     # for plotting
using Measures                  # for margin settings

# ----------------------------------------------------------------------------------------------------------------------
#
#                                               General model
#
# ----------------------------------------------------------------------------------------------------------------------

"""
    AbstractMaxEntropyModel

An abstract type for a MaxEntropyModel. Each model has one or more structural constraints  
that are fixed while the rest of the network is completely random. 

The different functions below should be implemented in the subtypes.
"""
abstract type AbstractMaxEntropyModel end

"""
    σ(::AbstractMaxEntropyModel)

Compute variance for elements of the adjacency matrix for the specific `AbstractMaxEntropyModel` based on the ML parameters.
"""
σ(::AbstractMaxEntropyModel) = nothing


"""
    Ĝ(::AbstractMaxEntropyModel)

Compute expected adjacency and/or weight matrix for a given `AbstractMaxEntropyModel`
"""
Ĝ(::AbstractMaxEntropyModel) = nothing


"""
    rand(::AbstractMaxEntropyModel)

Sample a random network from the `AbstractMaxEntropyModel`
"""
Base.rand(::AbstractMaxEntropyModel) = nothing

# ----------------------------------------------------------------------------------------------------------------------
#
#                                               UBCM model
#
# ----------------------------------------------------------------------------------------------------------------------

"""
    UBCM

Maximum entropy model for the Undirected Binary Configuration Model (UBCM). 
    
The object holds the maximum likelihood parameters of the model (x), the expected adjacency matrix (G), 
and the variance for the elements of the adjacency matrix (σ).

"""
struct UBCM{T} <: AbstractMaxEntropyModel where {T<:Real}
    x::Vector{T}
    G::Matrix{T}
    σ::Matrix{T}
end

"""
    UBCM(x::Vector{T}; compute::Bool=true) where {T<:Real}

Constructor for the `UBCM` type. If `compute` is true, the expected adjacency matrix and variance are computed. 
Otherwise the memory is allocated but not initialized.
"""
function UBCM(x::Vector{T}; compute::Bool=true) where {T<:Real}
    G = compute ? Ĝ(::UBCM, x) : Matrix{T}(undef, length(x), length(x))
    σ = compute ? σ(::UBCM, x) : Matrix{T}(undef, length(x), length(x))
    return UBCM(x, G, σ)
end

"""Return the number of nodes in the UBCM network"""
Base.length(m::UBCM) = length(m.x)

"""
    Ĝ(::UBCM, x::Vector{T}) where {T<:Real}

Compute the expected adjacency matrix for the UBCM model with maximum likelihood parameters `x`.
"""
function Ĝ(::UBCM, x::Vector{T}) where T
    n = length(x)
    G = zeros(T, n, n)
    for i = 1:n
        @simd for j = i+1:n
            @inbounds xij = x[i]*x[j]
            @inbounds G[i,j] = xij/(1 + xij)
            @inbounds G[j,i] = xij/(1 + xij)
        end
    end
    
    return G
end


"""
    σ(::UBCM, x::Vector{T})

Compute the variance for the elements of the adjacency matrix for the UBCM model with maximum likelihood parameters `x`.
"""
function σ(::UBCM, x::Vector{T}) where T
    n = length(x)
    res = zeros(T, n, n)
    for i = 1:n
        @simd for j = i+1:n
            @inbounds xij =  x[i]*x[j]
            @inbounds res[i,j] = sqrt(xij)/(1 + xij)
            @inbounds res[j,i] = sqrt(xij)/(1 + xij)
        end
    end

    return res
end

"""
    rand(m::UBCM)

Generate a random graph from the UBCM model. The function returns a `LightGraph` object.
"""
function Base.rand(m::UBCM)
    n = length(m)
    g = Graphs.SimpleGraph(n)
    for i = 1:n
        for j = i+1:n
            if rand() < m.G[i,j]
                add_edge!(g, i, j)
            end
        end
    end

    return g
end



# ----------------------------------------------------------------------------------------------------------------------
#
#                                               DBCM model
#
# ----------------------------------------------------------------------------------------------------------------------



"""
∇X(X::Function, G::Matrix{T})

Compute the gradient of a metric X given an adjacency/weight matrix G
"""
∇X(::Function, ::Matrix)

function ∇X(X::Function, G::Matrix{T}) where T<:Real
    return ReverseDiff.gradient(X, G)
end

function ∇X(X::Function, m::UBCM{T}) where T<:Real
    G = Ĝ(m)
    return ReverseDiff.gradient(X, G)
end

"""
σₓ(::Function)

Compute variance of a metric X given the expected adjacency/weight matrix
"""
σₓ(::Function, ::Matrix)

function σX(X::Function, Ĝ::Matrix{T}, σ̂::Matrix{T}) where T<:Real
    return sqrt( sum((σ̂ .* ∇X(X, Ĝ)) .^ 2) )
end




## some metrics 
d(A,i) = sum(@view A[:,i]) # degree node i
ANND(G::Graphs.SimpleGraph,i) where T = iszero(Graphs.degree(G,i)) ? zero(Float64) : sum(map( n -> Graphs.degree(G,n), Graphs.neighbors(G,i))) / Graphs.degree(G,i)
ANND(G::Graphs.SimpleGraph) where T = map( i -> ANND(G,i), 1:Graphs.nv(G))
ANND(A,i) =  sum(A[i,j] * d(A,j) for j=1:size(A,1) if j≠i) / d(A,i)

## quick demo - Zachary karate club
NP = PyCall.pyimport("NEMtropy")
mygraph = Graphs.smallgraph(:karate)
mygraph_nem = NP.UndirectedGraph(degree_sequence=Graphs.degree(mygraph))
mygraph_nem.solve_tool(model="cm", method="fixed-point", initial_guess="random");

# model
M = UBCM(mygraph_nem.x)
G_exp = Ĝ(M)
σ_exp = σ(M)

# check expected value
d_star = Graphs.degree(mygraph)
inds = sortperm(d_star)



# sampling the model
S = [rand(M) for _ in 1:10000];
dS = hcat(map( g -> Graphs.degree(g), S)...);
d_sample_mean_estimate(V,n) = mean(V[:,1:n], dims=2);
d_sample_var_estimate(V,n) =  std(V[:,1:n], dims=2);

# illustration for degree - acceptable domain
p1 = scatter(d_star[inds], d_star[inds], label="observed", legend=:topleft)
plot!(d_star[inds], map(n -> d(G_exp,n), 1:34)[inds], label="expected", linestyle=:dash, color=:steelblue)
plot!(d_star[inds], map(n -> d(G_exp,n), 1:34)[inds] .+  2* map(n-> σX(A->d(A,n), G_exp, σ_exp), 1:34)[inds],linestyle=:dash,color=:black,label="acceptance domain (analytical)")
plot!(d_star[inds], map(n -> d(G_exp,n), 1:34)[inds] .-  2* map(n-> σX(A->d(A,n), G_exp, σ_exp), 1:34)[inds],linestyle=:dash,color=:black,label="")
for ns in [10;100;1000;10000]
    plot!(d_star[inds], d_sample_mean_estimate(dS,ns)[inds] .+ 2* d_sample_var_estimate(dS,ns)[inds],linestyle=:dot,label="acceptance domain, sample (n=$(ns))")
    plot!(d_star[inds], d_sample_mean_estimate(dS,ns)[inds] .- 2* d_sample_var_estimate(dS,ns)[inds],linestyle=:dot,label="")
end
xlabel!("observed degree")
ylabel!("expected degree")
title!("UBCM - Zachary karate club")

# illustration for degree - z-scores per node
p2 = scatter(abs.((d_star - map(n -> d(G_exp,n), 1:34)) ./ map(n-> σX(A->d(A,n), G_exp, σ_exp), 1:34)), label="analytical", yscale=:log10)
for ns in [10;100;1000;10000]
    z = abs.((d_star - d_sample_mean_estimate(dS,ns)) ./ d_sample_var_estimate(dS,ns))
    lab = @sprintf "sample (n= %1.0e)" ns
    scatter!(z[z.>0], label=lab, yscale=:log10)
end
plot!([1; Graphs.nv(mygraph)], [3;3], linestyle=:dash,color=:red,label="")
xlabel!("node id")
ylabel!("|z-score degree|")
title!("UBCM - Zachary karate club")
ylims!(1e-10,1)

plot!(p1,p2, size=(1200,600), bottom_margin=5mm)
savefig("ZKC - degree.pdf")
nothing



##
## ANND part ##
##
X_star = ANND(mygraph)  # observed value (real network)
X_exp  = map(n->ANND(G_exp,n), 1:Graphs.nv(mygraph))    # expected value (analytical)
X_std  = map(n-> σX(A->ANND(A,n), G_exp, σ_exp), 1:34)  # standard deviation (analytical)
z_anal = (X_star - X_exp) ./ X_std

xS = hcat(map( g -> ANND(g), S)...);  #segmentation fault!!! => solved, tpye unstable


AP1 = scatter(d_star[inds], X_star[inds], label="observed", legend=:topright)
plot!(d_star[inds], X_exp[inds], label="expected", linestyle=:dash, color=:steelblue)
plot!(d_star[inds], X_exp[inds] .+ 2*X_std[inds], linestyle=:dash,color=:black,label="acceptance domain (analytical)")
plot!(d_star[inds], X_exp[inds] .- 2*X_std[inds], linestyle=:dash,color=:black,label="")
for ns in [10;100;1000;10000]
    plot!(d_star[inds], d_sample_mean_estimate(xS,ns)[inds] .+ 2* d_sample_var_estimate(xS,ns)[inds],linestyle=:dot,label="acceptance domain, sample (n=$(ns))")
    plot!(d_star[inds], d_sample_mean_estimate(xS,ns)[inds] .- 2* d_sample_var_estimate(xS,ns)[inds],linestyle=:dot,label="")
end
xlabel!("observed degree")
ylabel!("ANND")
title!("UBCM - Zachary karate club")

# illustration for degree - z-scores per node
AP2 = scatter(abs.(z_anal), label="analytical", yscale=:log10)
for ns in [10;100;1000;10000]
    z = abs.((X_star - d_sample_mean_estimate(xS,ns)) ./ d_sample_var_estimate(xS,ns))
    lab = @sprintf "sample (n= %1.0e)" ns
    scatter!(z[z.>0], label=lab, yscale=:log10)
end
plot!([1; Graphs.nv(mygraph)], [3;3], linestyle=:dash,color=:red,label="", legend=:bottomleft)
xlabel!("node id")
ylabel!("|z-score ANND|")
title!("UBCM - Zachary karate club")
ylims!(1e-3,1e1)

plot(AP1,AP2, size=(1200,600), bottom_margin=5mm)
savefig("ZKC - ANND.pdf")



# counting motifs
"""
    M₂(A)

Count number of occurences of motif number 2 (triangles)
"""
M₂(A) = sum(A[i,j]*A[j,k]*A[k,i] for i = 1:size(A,1) for j=i+1:size(A,1) for k=j+1:size(A,1))

"""
    M₁(A)

Count number of occurences of motif number 1 (v-shapes)
"""
M₁(A) = sum(A[i,j]*A[j,k]*(1 - A[k,i]) for i = 1:size(A,1) for j=i+1:size(A,1) for k=j+1:size(A,1))


N₂_obs = M₂(Graphs.adjacency_matrix(mygraph))
N₂_exp = M₂(G_exp)
N₂_σ   = σX(M₂, G_exp, σ_exp) # standard deviation (analytical)
z_N₂ = (N₂_obs- N₂_exp) / N₂_σ
# from the sample
N₂_sampled = map(g -> M₂(Graphs.adjacency_matrix(g)), S)
# computed values:
N₂_hat = [mean(N₂_sampled[1:n]) for n ∈ [10;100;1000;10000]]
N₂_std_hat = [std(N₂_sampled[1:n]) for n ∈ [10;100;1000;10000]]
N₂_z_hat = (N₂_obs .- N₂_hat) ./ N₂_std_hat

scatter([10;100;1000;10000],abs.(N₂_z_hat), xscale=:log10, label="sampled")
plot!([10;100;1000;10000], abs.(z_N₂ .* ones(4)), linestyle=:dash, label="analytical")
ylims!(0,2)
xticks!([10;100;1000;10000])
xlabel!("sample size")
ylabel!("|z-score triangle count|")
title!("UBCM - Zachary karate club")
Plots.savefig("ZKC - Triangles.pdf")

N₁_obs = M₁(Graphs.adjacency_matrix(mygraph))
N₁_exp = M₁(G_exp)
N₁_σ   = σX(M₁, G_exp, σ_exp) # standard deviation (analytical)
z_N₁ = (N₁_obs- N₁_exp) / N₁_σ
# from the sample
N₁_sampled = map(g -> M₁(Graphs.adjacency_matrix(g)), S)
# computed values:
N₁_hat = [mean(N₁_sampled[1:n]) for n ∈ [10;100;1000;10000]]
N₁_std_hat = [std(N₁_sampled[1:n]) for n ∈ [10;100;1000;10000]]
N₁_z_hat = (N₁_obs .- N₁_hat) ./ N₁_std_hat

scatter([10;100;1000;10000],abs.(N₁_z_hat), xscale=:log10, label="sampled")
plot!([10;100;1000;10000], abs.(z_N₁ .* ones(4)), linestyle=:dash, label="analytical")
ylims!(0,2)
xticks!([10;100;1000;10000])
xlabel!("sample size")
ylabel!("|z-score v-motif count|")
title!("UBCM - Zachary karate club")
savefig("ZKC - v-motifs.pdf")


# scaffolding for the directed network motif elements
"""
    a⭢(A::Matrix{T}, i, j)

directed link from i to j and not from j to i
"""
a⭢(A::Matrix{T}, i, j) where T = A[i,j] * (one(T) - A[j,i]) 

"""
    a⭠(A::Matrix{T}, i, j)

directed link from j to i and not from i to j
"""
a⭠(A::Matrix{T}, i, j) where T = (one(T) - A[i,j]) * A[j,i]

"""
    a⭤(A::Matrix{T}, i, j)

recipocrated link between i and j
"""
a⭤(A,i,j) = A[i,j]*A[j,i]

"""
    a_̸(A::Matrix{T}, i, j)

no link between i and j
"""
a_̸(A::Matrix{T},i,j) where T = (one(T) - A[j,i]) * (one(T) - A[i,j])

# motifs itself

M_13(A) = sum(a⭤(A,i,j)*a⭤(A,j,k)*a⭤(A,k,i) for i = 1:size(A,1) for j=1:size(A,1) for k=1:size(A,1) if i≠j && j≠k && i≠k)