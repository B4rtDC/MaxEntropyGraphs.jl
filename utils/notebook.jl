### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ 059f8d36-1aa3-11ec-3f43-d93d1a3c4bfc
begin
	using Pkg
	cd(joinpath(dirname(@__FILE__),".."))
    Pkg.activate(pwd())
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
end

# ╔═╡ 0317538a-5e03-4759-b77a-68da095d8d9e
using ResumableFunctions

# ╔═╡ 12ca747c-bb31-4b73-bd1a-8c152cbb969e
begin
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
		res = zero(eltype(x))
		for j in eachindex(x)
			@info "j: $(j), $(typeof(j)) - i: $(i), $(typeof(i))"
			if j ≠ i 
				res += a(i,j,x) * degree(j, x)
			end
		end
		return res / degree(i, x)
	end
	
	f(x::T) where {T} = x / (one(T) + x)
	a(i,j,x::AbstractVector) = @inbounds f(x[i]*x[j]) # UBCM
	
	i = 2
	ANND_w(A::AbstractArray{R,2}) where {R<:Real} = ANND(i, A)
	AAND_w_x(A::AbstractArray{R,1}) where {R<:Real} = ANND(i, A)

end

# ╔═╡ 8b01b4cb-21f1-4701-90f0-25d7617c9517


# ╔═╡ 0908aed7-580c-44e4-8861-475976172634
begin
	N = 500
	G = barabasi_albert(N, 2, 2, seed=-5)
	
	# generate models
	model = UBCM(G)
	model_c = UBCMCompact(G)
	solve!(model)
	solve!(model_c)

	# observed (real) values
	A_star = LightGraphs.adjacency_matrix(G)    # observed adjacency matrix
	d_star = LightGraphs.degree(G)              # observed degree sequence
	ANND_star = map(i -> ANND(i, A_star), 1:N)  # observed ANND

	# expected values
	A_exp = fastmaxent.adjacency_matrix(model)
	d_exp = map(x->degree(x, A_exp), 1:N)                           
	ANND_exp = map(i -> ANND(i, A_exp), 1:N)                        
	
	
	# expected values inderect storage
	A_ind = fastmaxent.adjacency_matrix(model_c)
end

# ╔═╡ cf80ab40-c4b2-4b74-8b75-b797bacd9a10
ForwardDiff.hessian(AAND_w_x, model_c.x)

# ╔═╡ 946f1e55-dc0f-47b3-b927-265925ff69b8
"""
# symbolic method 
	let
	@variables A[1:N, 1:N]
	X = ANND(i, A) # X is the symbolic expression of our topological property 
	X_num = ANND(i,A_star) # X_num is the numerical version of our topological property
	X_sym_to_num = eval(Symbolics.build_function(X, A))(A_star) # X_sym_to_num is the numerical evaluation of the symbolic expression

	# compute each contribution ∂X/∂a_{t,s} and store it in an array 
	∂X∂a = zeros(Num,N,N)
	for t = 1:N
		for s = 1:N
			∂X∂a[t,s] = Symbolics.derivative(X, A[t,s])
		end
	end

	# you can eavluate the numerical value of each entry using
	ΔxΔA = Symbolics.build_function(∂X∂a, A) # leads to two function definitions (with or without output arg)
	ΔxΔA_exact = eval(ΔxΔA[1]) # this is the function
	ΔxΔA_exact(A_star) # this is the results based on the numerical values

	ΔA⁺Δa_exact = ΔxΔA_exact(A_exp)
	end
"""

# ╔═╡ 0114bb4a-0b49-4749-86f3-0316b25ec02c
md"""
### Complete gradient
=> costly and non feasible for large matrices
"""

# ╔═╡ 3b8e6d04-729c-400a-a7cc-878f339e42d6
∇k_nn_i = ForwardDiff.gradient(ANND_w, A_ind)

# ╔═╡ 2bb7586a-1e9a-484d-8bde-b61aece41c27
ANND_w_x(model_c.x)

# ╔═╡ d4fbe854-891e-4645-823f-09e98a212df1
∇k_nn_i[3,1] # actual value

# ╔═╡ 2b1616e0-f140-46fe-b5a7-ff72e93380c9
ForwardDiff.Dual.([1;2],[4;3])

# ╔═╡ ec679289-615d-4b0e-b9d9-2a30990791f5
d_exp[i]

# ╔═╡ f84c8a27-3eec-4537-a24f-f3279d235198
A_ind[i,1] / d_exp[i]

# ╔═╡ b472a947-6d4e-4b02-b87b-ae5742a3d863
(d_exp[1] + ANND_exp[i]) / d_exp[1]

# ╔═╡ 8300663c-c2ac-4918-8783-d3bdb2815492
md"""
## Single element of ∇X
"""

# ╔═╡ 73489838-3be0-4408-8f6e-b0100239941b
let
	inds = LinearIndices((N,N))
	(t,s) = (3,1) # value for which we are searching partial derivative
	vv = inds[t,s] # associate linear index
	ANND_w(A::AbstractArray{R,2}) where {R<:Real} = ANND(i, A)
	# try to accelerate this ↓
	foo = x -> ANND(i, reshape(vcat(A_exp[1:vv-1], x, A_exp[vv+1:end]), N,:))
	ForwardDiff.derivative(foo, A_exp[t,s])
	
	#@btime reshape(vcat($A_exp[1:$vv-1], 100, $A_exp[$vv+1:end]), $N,:)
	#@btime	ForwardDiff.derivative($foo, $A_exp[t,s])
	# this is a value that can be used :) in the matrix
end

# ╔═╡ 38894f23-f2b8-497a-adc6-1920d6d287a0
length(unique(ForwardDiff.gradient(ANND_w, A_exp)))

# ╔═╡ 503071ee-1a55-44f2-8a8c-1040209e0c53
∇knni = ForwardDiff.gradient(ANND_w, A_ind)

# ╔═╡ d241642d-a9d1-4b28-a9ee-01ced55ff8db


# ╔═╡ 5ec93a7d-5658-44e8-a63a-74525c90e05c
let
	ki = sum(@view A_exp[i,:])
#	length(unique(sum(A_ind, dims=2)))
end

# ╔═╡ 37c7f2f3-f94a-4184-bc3c-8f4e5ac00a04
A_exp

# ╔═╡ d33936a4-ee98-43a6-bb83-33c1ceca851e
i

# ╔═╡ 957f47f1-4a04-4ac8-8dd3-22e928c43197
macro round(A)
	return :(round.($A, digits=8))
end

# ╔═╡ 2fb3a638-57b5-4bc2-b803-59d730c77b16
md"""
## Building the adjacency matrix A
Starting from the mapped vector
"""

# ╔═╡ 3c71a2d6-bd69-4d9b-ae49-ed63e403d3e6
MMM =IndirectArray([1;2;3;4;4;4])

# ╔═╡ 7d9fbefc-60f6-425e-a20b-ff6470834cd1
eltype(typeof(MMM.index))

# ╔═╡ 45117ba2-a5d7-4fcf-b77d-5b080114a3fc
@btime degree(i,A_exp)

# ╔═╡ 144a7569-6d4c-458c-a789-3193764f954a
@btime degree(i,A_ind)

# ╔═╡ f48079da-2efe-4ec2-8721-07324dc96bd7
@btime degree(i, A_star)

# ╔═╡ 5d56c756-5d43-4b38-aeda-78cc8502ecec
md"""
## Obtaining ∂X\∂a
makeing use of:
- pattern in ∂X\∂a
- Indirect representation for A
"""

# ╔═╡ fc2a78e8-5cb3-42dc-abe4-2675470c084c
# exact solution 
ANND_w(A_exp)

# ╔═╡ 8998af15-dd88-403f-8739-0b38f2348895
function bigfun(i,j, X::Function, A::AbstractArray{T}, args...) where T
	backup = copy(A[i,j]) # copy original value of A[i,j]
	#setindex!(A,x,i,j) # replace by x
	# compute the magnitude
	foo = (x::Number,) -> X(setindex!(A,x,i,j))
	@info foo(A[i,j])
	res = ForwardDiff.derivative(foo, A[i,j])
	#setindex!(A,backup,i,j)
	return res
end

# ╔═╡ f13664a8-eb86-4ec6-8e9a-27202fcbe26a
let 
	N = 100
	# testfunction with result initiated within function
	function F(x::AbstractArray{T}) where {T}
		res = 0.#zero(eltype(x))
		for i in 1:size(x,1)
			for j in 1:size(x,2)
				res += @inbounds x[i,j]#^(i/j)
			end
		end
		return res
	end

	# location where we want the gradient
	A = rand(N,N)
	
	# computing the gradient (for all variables) > OK
	∇foo = ForwardDiff.gradient(F, A)
	
	# computing a specific component of the gradient > OK
	inds = LinearIndices((N,N))
	(t,s) = (1,3)  # value for which we are searching partial derivative: ∂foo/∂a_{t,s}
	vv = inds[t,s] # associated linear index
	subfoo = x -> F(reshape(  vcat( A[1:vv-1], x,  A[vv+1:end]), N,:))
	res = ForwardDiff.derivative(subfoo, A[t,s])
	
	@assert res == ∇foo[t,s]
	
	
	function bigfoo(t,s,foo::Function, A::AbstractArray, args...)
		backup = copy(A[t,s]) # copy original value of A[t,s]
		
		# closure - no new matrix (not working)
		#localfoo = (x::Number,) -> foo(setindex!(A,x,t,s), args...)
		
		# closure - new matrix (working, but build a lot of matrices)
		localfoo = (x::Number,) -> foo(reshape(vcat(A[1:vv-1], x, A[vv+1:end]), N,:), args...) 
	
		res = ForwardDiff.derivative(localfoo, A[t,s])
		# restore original value
		A[t,s] = backup
		return res
	end
	
	@assert	bigfoo(t,s, F, A) == ∇foo[t,s]
	
	#@btime $subfoo($A[$t,$s])
	@time F(A)
	@time F(A)
	
	@time res
	@info "single component forward diifferentiation"
	for _ in 1:2
		@time ForwardDiff.gradient(F, A)
		@time ForwardDiff.derivative(subfoo, A[t,s])
	end
end

# ╔═╡ 2f00837e-b93c-4413-9fbe-71978235d492
function adjacency_matrix(model::UBCM)
    A = zeros(length(model.x), length(model.x))
    for i in eachindex(model.x)
        for j in eachindex(model.x)
            if i≠j
                A[i,j] = f(model.x[i]*model.x[j])
            end
        end
    end
	return A
end

# ╔═╡ 2b64db3c-a2d9-4a2e-ae87-94d092115855
let
	d = [1;1;1;2;3;4;5;6;6;6]
	freqs = fastmaxent.countmap(d)
	κ = collect(keys(freqs))
	revmap = similar(d)
	for i in eachindex(κ)
		revmap[findall(x->isequal(x, κ[i]), d)] .= i
	end
	
	
	
	revmap, length(d), length(κ)
end


# ╔═╡ 5882c2e8-42bf-4cb1-b9dd-55b170004a93
Int(typemax(UInt16))

# ╔═╡ 9d09dac3-7a74-423f-916e-7cac87b6d874
A = [1 ;2 ;3; 4]

# ╔═╡ 14458710-4e31-4318-a80d-af056e8f49b3
IndirectArrays.IndirectArray

# ╔═╡ 33906cba-be7c-427d-85e7-48cebbf972c2
let
	d = ones(1000000000)
	indtype = length(d) < typemax(UInt8) ? UInt8 : length(d) < typemax(UInt16) ? UInt16 : length(d) < typemax(UInt32) ? UInt32 : UInt64
end

# ╔═╡ 2b0910f4-77a8-452d-90ee-f4174393de1d
begin
	M = reshape([i for i in 1:9], 3,:)
	@resumable function matmat!(X,x,t::Int,s::Int)
		# get original values
		oldval = X[t,s]
		while true
			# replace value
			X[t,s] = x
			@yield X
			# return previous one
			X[t,s] = oldval
		end
		
	end
end

# ╔═╡ e75b9420-a7a4-42a9-b2e4-647428d5f149
begin
	@resumable function foo(M)
		input = @yield 
		(x,t,s) = input[1], input[2], input[3]
		println("received x:$(x), t: $(t), s:(s)")
	end
	
	Foo = foo(M)
	Foo("4")
	Foo("5")
end

# ╔═╡ 9d4f890d-91c3-4117-ab69-05856fc894e4
ff =matmat!(M,10,5,6)

# ╔═╡ 46d419ab-5c23-4527-a561-148f7c7bf807
function adjacency_matrix(model::UBCMCompact)
    # infer index typemax (?)
    # TO DO
    # generate all possible unique values in the adjacency matrix (Tuple: ((ı,ȷ), Value))
    compact = map(t -> (t,f(model.xs[t[1]]*model.xs[t[2]])),
                    Iterators.filter(t->t[2]>=t[1], Iterators.product(1:length(model.xs),1:length(model.xs)))); 
    # sort the values
    sort!(compact, by = x -> x[2]) # required?
    # obtain the values
    values = [x[2] for x in compact] # infer indices from here
    # obtain the map of compact tuple (i,j) ↦ value
    mapper = Dict(compact[i][1] => i for i in eachindex(compact))
    # obtain the indices (matrix referring to vector holding the actual values)
    indices = Int8[mapper[model.revmap[i]<model.revmap[j] ? (model.revmap[i],model.revmap[j]) : (model.revmap[j],model.revmap[i])] for i in 1:N, j in 1:N]

    return IndirectArray(indices, values)
end

# ╔═╡ 1d30f081-92cf-4c57-861c-db52bac156e1
@time adjacency_matrix(model)

# ╔═╡ 54ca3d21-199f-424c-a2b9-eeb1f9520789
begin
	function indmap(model::UBCMCompact,T::DataType,i,j)
		return model.revmap[i] <= model.revmap[j] ? T(model.revmap[i]+model.revmap[j]*(model.revmap[j]-1)÷2) : T(model.revmap[j]+model.revmap[i]*(model.revmap[i]-1)÷2)
	end
	
	function adjacency_matrix_f(model::UBCMCompact)
    # generate vector of all possible unique values in the adjacency matrix
    values = @inbounds [f(model.xs[i]*model.xs[j]) for j = 1:length(model.xs) for i = 1:j]
	push!(values,zero(eltype(values))) # include the zero for later
	# infer storage index type
	T = length(values) < typemax(UInt8) ? UInt8 : length(values) < typemax(UInt16) ? UInt16 : UInt32
	# generate indices in matrix from xs
	indices = [indmap(model,T,i,j)  for i in 1:length(model.revmap), j in 1:length(model.revmap)]
	# set diagonal to zero
	indices[diagind(indices)] .= T(length(values))
    # return indirectarray
	return IndirectArray(indices, values)
end
end

# ╔═╡ 75530ba3-6a9b-425f-9db9-f99db9546a33
sizeof(adjacency_matrix_f(model_c))

# ╔═╡ c225cea5-7a11-4f58-85b5-890d7220d0ce
@time adjacency_matrix(model)

# ╔═╡ 7b108c14-14e3-42e4-b86f-5e73a0830f5c
isapprox(adjacency_matrix_f(model_c), adjacency_matrix(model))

# ╔═╡ ad38c142-e83c-4b79-9966-55754da8a2fe
function adjacency_matrix_f_x(model::UBCMCompact)
	A = zeros(length(model.x), length(model.x))
    for i in eachindex(model.x)
        for j in eachindex(model.x)
            if i≠j
                A[i,j] = f(model.x[i]*model.x[j])
            end
        end
    end
    return A
end

# ╔═╡ d6dd8195-6842-4884-81d3-f2e0e92c8295
@time adjacency_matrix_f_x(model_c)

# ╔═╡ 2fd1846d-35b7-492b-89c3-540e41f6997a
let 
	f(x) = [x[2] + cos(x[1]); x[1]]
	df(x) = [-sin(x[1]; 0]
	x = ForwardDiff.Dual.([1.; 3.])
	ans = f(x)
	ans.
end

# ╔═╡ Cell order:
# ╠═059f8d36-1aa3-11ec-3f43-d93d1a3c4bfc
# ╠═12ca747c-bb31-4b73-bd1a-8c152cbb969e
# ╠═cf80ab40-c4b2-4b74-8b75-b797bacd9a10
# ╠═8b01b4cb-21f1-4701-90f0-25d7617c9517
# ╠═0908aed7-580c-44e4-8861-475976172634
# ╠═946f1e55-dc0f-47b3-b927-265925ff69b8
# ╟─0114bb4a-0b49-4749-86f3-0316b25ec02c
# ╠═3b8e6d04-729c-400a-a7cc-878f339e42d6
# ╠═2bb7586a-1e9a-484d-8bde-b61aece41c27
# ╠═d4fbe854-891e-4645-823f-09e98a212df1
# ╠═2b1616e0-f140-46fe-b5a7-ff72e93380c9
# ╠═ec679289-615d-4b0e-b9d9-2a30990791f5
# ╠═f84c8a27-3eec-4537-a24f-f3279d235198
# ╠═b472a947-6d4e-4b02-b87b-ae5742a3d863
# ╟─8300663c-c2ac-4918-8783-d3bdb2815492
# ╠═73489838-3be0-4408-8f6e-b0100239941b
# ╠═38894f23-f2b8-497a-adc6-1920d6d287a0
# ╠═503071ee-1a55-44f2-8a8c-1040209e0c53
# ╠═d241642d-a9d1-4b28-a9ee-01ced55ff8db
# ╠═5ec93a7d-5658-44e8-a63a-74525c90e05c
# ╠═37c7f2f3-f94a-4184-bc3c-8f4e5ac00a04
# ╠═d33936a4-ee98-43a6-bb83-33c1ceca851e
# ╠═957f47f1-4a04-4ac8-8dd3-22e928c43197
# ╟─2fb3a638-57b5-4bc2-b803-59d730c77b16
# ╠═3c71a2d6-bd69-4d9b-ae49-ed63e403d3e6
# ╠═7d9fbefc-60f6-425e-a20b-ff6470834cd1
# ╠═45117ba2-a5d7-4fcf-b77d-5b080114a3fc
# ╠═144a7569-6d4c-458c-a789-3193764f954a
# ╠═f48079da-2efe-4ec2-8721-07324dc96bd7
# ╟─5d56c756-5d43-4b38-aeda-78cc8502ecec
# ╠═fc2a78e8-5cb3-42dc-abe4-2675470c084c
# ╠═8998af15-dd88-403f-8739-0b38f2348895
# ╠═f13664a8-eb86-4ec6-8e9a-27202fcbe26a
# ╠═2f00837e-b93c-4413-9fbe-71978235d492
# ╠═1d30f081-92cf-4c57-861c-db52bac156e1
# ╠═2b64db3c-a2d9-4a2e-ae87-94d092115855
# ╠═5882c2e8-42bf-4cb1-b9dd-55b170004a93
# ╠═9d09dac3-7a74-423f-916e-7cac87b6d874
# ╠═14458710-4e31-4318-a80d-af056e8f49b3
# ╠═33906cba-be7c-427d-85e7-48cebbf972c2
# ╠═0317538a-5e03-4759-b77a-68da095d8d9e
# ╠═2b0910f4-77a8-452d-90ee-f4174393de1d
# ╠═e75b9420-a7a4-42a9-b2e4-647428d5f149
# ╠═9d4f890d-91c3-4117-ab69-05856fc894e4
# ╠═46d419ab-5c23-4527-a561-148f7c7bf807
# ╠═54ca3d21-199f-424c-a2b9-eeb1f9520789
# ╠═75530ba3-6a9b-425f-9db9-f99db9546a33
# ╠═c225cea5-7a11-4f58-85b5-890d7220d0ce
# ╠═7b108c14-14e3-42e4-b86f-5e73a0830f5c
# ╠═ad38c142-e83c-4b79-9966-55754da8a2fe
# ╠═d6dd8195-6842-4884-81d3-f2e0e92c8295
# ╠═2fd1846d-35b7-492b-89c3-540e41f6997a
