using Calculus
using Symbolics
using SparseArrays
using ForwardDiff
using ReverseDiff
using LinearAlgebra
using LightGraphs
using fastmaxent
using BenchmarkTools
using Latexify 

# supporting functions - define the metric in terms of the adjacency matrix
"""using the adjacency matrix"""
function degree(i::Int,A::AbstractArray{T,2}) where T<:Real
    return sum(@view A[i,:]) - A[i,i]
end

"""using the symbolic representation"""
function degree(i::Int, A::Symbolics.Arr{Num, 2}) 
    return sum(@view A[i,:]) - A[i,i]
end

"""using the computed parameters"""
function degree(i, x::AbstractVector)
    res = - a(i,i,x)
    for j in eachindex(x)
        res += a(i,j,x)
    end
    return res
end

"""using the adjacency matrix"""
function ANND(i::Int, A::AbstractArray{T,2}) where T<:Real
    return sum(A[i,j] * degree(j,A) for j=1:size(A,1) if j≠i) / degree(i,A)
end

"""using the symbolic representation"""
function ANND(i::Int, A::Symbolics.Arr{Num, 2}) 
    return sum(A[i,j] * degree(j,A) for j=1:size(A,1) if j≠i) / degree(i,A)
end

"""using the computed parameters"""
function ANND(i::Int, x::Symbolics.Arr{Num, 1})
    @warn "ANND(i::Int, x::Symbolics.Arr{Num, 1}) is used!"
    res = zero(Num)
    for j in eachindex(x)
        @info "j: $(j), $(typeof(j)) - i: $(i), $(typeof(i))"
        if j[1] ≠ i 
            res += a(i,j,x) * degree(j, x)
        end
    end
    return res / degree(i, x)
end

"""using the computed parameters"""
function ANND(i::Int, x::AbstractVector{T}) where T
    @warn "ANND(i::Int, x::AbstractVector{T}) where T is used!"
    res = zero(T)
    for j in eachindex(x)
        @info "j: $(j), $(typeof(j)) - i: $(i), $(typeof(i))"
        if j ≠ i 
            res += a(i,j,x) * degree(j, x)
        end
    end
    return res / degree(i, x)
end





# second method: make use of Symbolics.jl (symbolic computations => sparsity pattern)
n = 7
@variables A[1:n, 1:n]
@variables x[1:n]
A_num = Array{Float64}(Symmetric(rand(Bool, n,n)))

i = 2

X = ANND(i, A) # X is the symbolic expression of our topological property 
X_num = ANND(i,A_num) # X_num is the numerical version of our topological property
X_sym_to_num = eval(Symbolics.build_function(X, A))(A_num) # X_sym_to_num is the numerical evaluation of the symbolic expression

# compute each contribution ∂X/∂a_{t,s} and store it in an array 
∂X∂a = zeros(Num,n,n)
for t = 1:n
    for s = 1:n
        ∂X∂a[t,s] = Symbolics.derivative(X, A[t,s])
    end
end

# you can eavluate the numerical value of each entry using
ΔxΔA = Symbolics.build_function(∂X∂a, A) # leads to two function definitions (with or without output arg)
ΔxΔA_exact = eval(ΔxΔA[1]) # this is the function
ΔxΔA_exact(A_num) # this is the results based on the numerical values

# third method: make use of ForwardDiff.jl (automated differentiation => numerical result)


# appropriate function that takes only one input argument
ANND_w(A::AbstractArray{R,2}) where {R<:Real} = ANND(i, A)
ANND_w(x::AbstractVector{R}) where {R<:Real} = ANND(i, x)
ForwardDiff.gradient(ANND_w, A_num)



# Challenges: 
# 1. translate this to a closure for a single contribution, i.e. one ∂X/∂a_{ij}(a) at a time instead of ∇f(a) at once
# 2. inlude vector of parameters that allows to avoid computing matrix a as a whole


#@btime ΔxΔA_exact($A_num)
#@btime ForwardDiff.gradient($ANND_w, $A_num)
#@btime ReverseDiff.gradient($ANND_w, $A_num)

a_slow(x,i,j) = x[i]*x[j] / (1 + x[i]*x[j])
a_fast(x,i,j) = @inbounds f(x[i]*x[j])
f(x::T) where {T} = x / (one(T) + x)


# Include the entire method for an actual network

# supporting functions for computations
f(x::T) where {T} = x / (one(T) + x) # helper function UBCM]–
a(i,j,x::AbstractVector) = @inbounds f(x[i]*x[j]) # UBCM

# generate random graph
N = n
G = barabasi_albert(N, 2, 2, seed=-5)

model = UBCM(G)
model_c = UBCMCompact(G)
solve!(model)
solve!(model_c)
@info model.x
A_star = adjacency_matrix(G)                # observed adjacency matrix
d_star = LightGraphs.degree(G)              # observed degree sequence
ANND_star = map(i -> ANND(i, A_star), 1:N)  # observed ANND

A_exp = reshape([a(i,j,model.x) for j = 1:N for i in 1:N],N,:)  # expected adjacency_matrix
d_exp = map(x->degree(x, A_exp), 1:N)                           # expected degree sequence
ANND_exp = map(i -> ANND(i, A_exp), 1:N)                        # expected ANND

# variance for specific entry e.g. ANND (node number 2)
ΔA⁺Δa_exact = ΔxΔA_exact(A_exp)                                 # ∂ANND_i∂a_ij using analytical expression
ΔA⁺Δa_num = ForwardDiff.gradient(ANND_w, A_exp)                 # ∂ANND_i∂a_ij using autodiff (entire matrix A required)

# Trying vector based method
ΔMΔx = ForwardDiff.gradient(ANND_w, model.x)                    # gradient of M wrt computed parameters x
∂xi∂aij(i::Int, j::Int, x::Vector{T}) where T<: Real = (1 + x[i]*x[j])^2 / x[j]        # supposed form of ∂xi∂aij
∂M∂aij(ΔM::Vector, x::Vector, i::Int,j::Int) = ΔM[i] * ∂xi∂aij(i,j,x) + ΔM[j] * ∂xi∂aij(j,i,x) # unique contribution
# first problem stems from symmetrical case i = j (∂xi∂aij changes with factor 2, not main cause)
# 
ΔA⁺Δa_piecewise = map(t -> t[1]==t[2] ? 0. : ∂M∂aij(ΔMΔx, model.x, t[1], t[2]), Base.product(1:N, 1:N));


## Run and compare the numerical method with the expected analytical result

# 1. Write M in function of aij & xi: OK
M_A = ANND(i, A) # M_A is the symbolic expression of our topological property using the matrix A
M_x = ANND(i, x) # M_x is the symbolic expression of our topological property using the vector x

#latexify(M_A)
# this is wrong, because there are other values that exist as well, it is better to solve the problem in the 
# compact space in order to obtain the desired result. This reduces the matrix from N² to Ŋ² elements that
# can be reused :-)
#Symbolics.derivative(M_x, x[1]) / Symbolics.derivative(a(1,2,x), x[1]) + Symbolics.derivative(M_x, x[2]) / Symbolics.derivative(a(1,2,x), x[2])




##### Compact Vs extended model

model = UBCM(G)
model_c = UBCMCompact(G)
solve!(model)
solve!(model_c)
N_c = length(model_c.x0)

A_exp = reshape([i ≠ j ? a(i,j,model.x) : 0.0 for j = 1:N for i in 1:N],N,:)
A_exp_c = reshape([ a(i,j,model_c.x)  for j = 1:N_c for i in 1:N_c],N_c,:)

# Lets expand the matrix
nodemap = similar(model.x, Int)
for (key, val) in model_c.revmap
    for node in val
        nodemap[node] = key
    end
end

exp_ind(i,nodemap) = nodemap[i]
exp_ind_f(i,nodemap) = @inbounds nodemap[i]

A_exp_CL = reshape([A_exp_c[exp_ind(i, nodemap), exp_ind(j, nodemap)] for j in 1:N for i in 1:N], N,:)
ΔA⁺Δa_num_c = ForwardDiff.gradient(ANND_w, A_exp_c)


σ²ₐ(i,j,x) = (x[i]*x[j]) / (1 + x[i]*x[j])^2

σ²ₓ(A,x) = sum( σ²ₐ(i,j,x) * A[i,j]^2 for i in eachindex(x) for j in eachindex(x) if i≠j)


σ²ₓ(ΔA⁺Δa_num, model.x)

σ²ₓ_mat = zeros(length(model.x),length(model.x))
for i in eachindex(model.x)
    for j in eachindex(model.x)
        if i ≠ j
            σ²ₓ_mat[i,j] = σ²ₐ(i,j,model.x) * ΔA⁺Δa_num[i,j]^2
        end
    end
end
σ²ₓ_mat_c = zeros(length(model_c.x),length(model_c.x))
for i in eachindex(model_c.x)
    for j in eachindex(model_c.x)
        if i ≠ j
            σ²ₓ_mat_c[i,j] = σ²ₐ(i,j,model_c.x) * ΔA⁺Δa_num_c[i,j]^2
        end
    end
end



σ²ₓ(ΔA⁺Δa_num_c, model_c.x)

# first method: make use of Calculus.jl (expression) combined with symbolic differentiation

# - not trivial
# - text/symbol-based, so should work rather fast => hard to generate expressions

#=

function degree(i::Int,A::Array{String,2})
    res = IOBuffer()
    for j in 1:size(A,1)
        if j≠i
            write(res, " + $(A[i,j])")
        end
    end

    return String(take!(res))
end

function degree!(i::Int,A::Array{String,2}, res::IOBuffer)
    for j in 1:size(A,1)
        if j≠i
            write(res, " + $(A[i,j])")
        end
    end

    return String(take!(res))
end

n = 20
bin = IOBuffer()
A = rand(Bool,n,n)
As = reshape(["a_$(i)_$(j)" for j in 1:n for i in 1:n],n,:)
using BenchmarkTools

@btime degree(1,$As)
@btime degree!(1,$As,$bin)


d_i = degree(1, As)

@btime differentiate($d_i, :a_1_7)
=#

#=
struct CompactArray{T,N,A<:AbstractArray{T,N},J<:Integer,I<:AbstractArray{J,N} } <: AbstractArray{T,N}
    values::A
    index::I
end
Base.IndexStyle(::Type{CompactArray{T,N,A,J,I}}) where {T,N,A,J,I} = IndexStyle(A)
Base.show(io::IO,A::CompactArray{T,N}) where {T,N} = print(io, "CompactArray{$(T)}\n$(Base.print_matrix(io,[1 2;4 5]))")
Base.display(A::CompactArray{T,N}) where {T,N} = show(A)
Base.size(A::CompactArray{T,N}) where {T,N} = length(A.index) .* size(A.values)
Base.getindex(A::CompactArray, i::Int) = getindex(A.values,i)
Base.getindex(A::CompactArray{T,N}, I::Vararg{Int,N}) where {T,N} = getindex(A.values, I)
Base.setindex(A::CompactArray{T,N}, v::T, i::Int) where {T,N} = setindex(A.values, v, i)
Base.setindex!(A::CompactArray{T,N}, v, I::Vararg{Int,N}) where {T,N}  = setindex(A.values, v, I)
val = [1 2;3 4]
inds = [1;2;3;4;5]
CompactArray(val, inds)

Base.size(A::CompactArray{T,N}) where {T,N} = size(A.index)
Base.getindex(A::CompactArray, i::Int) = getindex(A.values,i)
Base.getindex(A::CompactArray{T,N}, I::Vararg{Int,N}) where {T,N} = getindex(A.values, I)
Base.setindex(A::CompactArray{T,N}, v::T, i::Int) where {T,N} = setindex(A.values, v, i)
Base.setindex!(A::CompactArray{T,N}, v, I::Vararg{Int,N}) where {T,N}  = setindex(A.values, v, I)
#=


struct CompactArray{T,N,I, A<:AbstractArray{T,N}, V<:AbstractVector{I}} <: AbstractArray{T,N}
    values::A
    index::V
    function CompactArray{T,N,I,A,V}(values::AbstractArray{T,N}, index::AbstractVector{I}) where {T,N,I,A,V}
        I <: Integer || error("index must have integer eltype")
        new{T,N,I,A,V}(values, index)
    end
end

const CompactVector{T,N,I,A<:AbstractVector,V} = IndirectArray{T,N,I,A,V}


#Base.size(A::CompactArray) = length(A.index) .* size(A.values)
Base.size(A::CompactArray{T,N,I,A,V}) where {T,N,I,A,V} = length(A.index) .* size(A.values)
Base.size(A::IndirectArray) = size(A.index)
Base.getindex(A::CompactArray, inds...) = 
x = CompactArray(model_c.x, nodemap)

=#
=#







# Large model

