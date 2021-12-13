### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ 3025ee60-4c30-11ec-1512-2ba1bd45a489
begin
	using Pkg
	cd(joinpath(dirname(@__FILE__),".."))
	Pkg.activate(pwd())
	
	using BenchmarkTools, LoopVectorization, Graphs, PyCall, fastmaxent, Plots, NLsolve
end

# ╔═╡ f2aa794f-1c5b-42c2-8413-f4547d62ce2f
# cell with modification code (cf. https://discourse.julialang.org/t/modify-right-margin-in-pluto-notebook/61452/3?u=bdc)
html"""<style>
/*              screen size more than:                     and  less than:                     */
@media screen and (max-width: 699px) { /* Tablet */ 
  /* Nest everything into here */
    main { /* Same as before */
        max-width: 1000px !important; /* Same as before */
        margin-right: 100px !important; /* Same as before */
    } /* Same as before*/

}

@media screen and (min-width: 700px) and (max-width: 1199px) { /* Laptop*/ 
  /* Nest everything into here */
    main { /* Same as before */
        max-width: 1000px !important; /* Same as before */
        margin-right: 100px !important; /* Same as before */
    } /* Same as before*/
}

@media screen and (min-width:1200px) and (max-width: 1920px) { /* Desktop */ 
  /* Nest everything into here */
    main { /* Same as before */
        max-width: 1000px !important; /* Same as before */
        margin-right: 100px !important; /* Same as before */
    } /* Same as before*/
}

@media screen and (min-width:1921px) { /* Stadium */ 
  /* Nest everything into here */
    main { /* Same as before */
        max-width: 1000px !important; /* Same as before */
        margin-right: 200px !important; /* Same as before */
    } /* Same as before*/
}


</style>
"""

# ╔═╡ 53533ead-5eb2-4bd4-b273-c50548683bec
#=
TO DO:
- convert fastmaxent to Graphs instead of LightGraphs (!) => OK
- compare indirect (x) with direct domain (θ) +- OK
-
-
-
=#

# ╔═╡ 15240395-9674-4ab9-90fa-b60fc7bbb584
begin
	
	f(x) = x / (1 + x) # helper function 
	
	"""
		UBCM_ref!(F::Vector, x::Vector, k::Vector)
	
	reference version for UBCM benchmarking
	"""
	function UBCM_ref!(F::Vector, x::Vector, k::Vector)
    	for i in eachindex(x)
			F[i] = -f(x[i]*x[i]) - k[i]
			for j in eachindex(x)
				F[i] += f(x[i]*x[j])
			end
		end
		
    	return F
	end
	
	"""
		UBCM_own!(F::Vector, x::Vector, k::Vector)
	
	cached threaded version of UBCM (own build)
	"""
	function UBCM_own!(F::Vector, x::Vector, k::Vector)
		Threads.@threads for i in eachindex(x)
			@inbounds F[i] = -f(x[i]*x[i]) - k[i]
			@simd for j in eachindex(x)
				@inbounds F[i] += f(x[i]*x[j])
			end
		end

		return F
	end
	
	"""
		UBCM_loopvec!(F::Vector, x::Vector, k::Vector)
	
	cached threaded version of UBCM (own build)
	"""
	function UBCM_loopvec!(F::Vector, x::Vector, k::Vector)
		@turbo for i in eachindex(x)
			F[i] = -f(x[i]*x[i]) - k[i]
			for j in eachindex(x)
				F[i] += f(x[i]*x[j])
			end
		end

		return F
	end
end

# ╔═╡ 9357d218-ce59-42b4-bbc4-b40806722e0a
begin
	funs = [UBCM_ref!, UBCM_own!, UBCM_loopvec!]
	N = [10;100]#;1000;10000;100000]

	# generate tags
	suite = BenchmarkGroup(["$(String(Symbol(foo)))" for foo in funs])
	for foo in funs
		suite["$(String(Symbol(foo)))"] = BenchmarkGroup(["$n" for n in N])
	end
	
	# prepare benchmark
	for n in N
		x = rand(n)
		k = rand(n)
		F = zeros(n)
		for foo in funs
			localfun = (F::Vector, x::Vector) -> foo(F, x, k)
			suite["$(String(Symbol(foo)))"]["$(n)"] = @benchmarkable $(localfun)($F, $x)
		end
	end
	
	if false
		# tune it
		tune!(suite)

		# run it
		results = run(suite, verbose = true)
	end
end

# ╔═╡ 4c586a3c-4578-4b36-bf62-4f15d2d49c87
begin
	unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))
	# illustrate it
	yticks = 10. .^ collect(-9:2:2)
	p = plot(scale=:log10, legend=:topleft, 
		title="UBCM result on $(Threads.nthreads()) threads",
	yticks= yticks, ylims=[minimum(yticks), maximum(yticks)], xticks=N,
	xlabel="Number of unique parameters", ylabel="computation time [s]")

	for foo in funs
		# get values
		_, times = unzip([(key,val.time) for (key, val) in median(results["$(String(Symbol(foo)))"])])
		plot!(N, sort(times)./1e9, marker=:circle, label="$(String(Symbol(foo)))")
	end

	p
end

# ╔═╡ b92633f2-65bd-41ee-882c-12d7dc0013a7
md"""
# Define the testcase
we consider the standard method (without reduction) for the following UBCM cases:
* nlsolve using x representation
* nlsolve directly for θ
* nlsolve fixed point anderson acceleration
* iterative method for θ based on the Squartini paper

We use a small toy network that is common for all problems. They all get the same starting conditions ``x_0`` or ``\theta_0 = \ln \left(x_0 \right)``.
"""

# ╔═╡ 7ca4264c-79b8-4a00-aacd-f434a880410c
begin
	# Toy network definition
	G = barabasi_albert!(cycle_graph(2), 20, 2);
	degree(G)
	K = degree(G)
end

# ╔═╡ 85406732-dd6b-450e-847d-a0da3a5b5aec
fastmaxent.UBCMCompact(G)

# ╔═╡ cb3cbb8c-34bb-4410-9f77-b0cd9b6dd1a7
K

# ╔═╡ 0e2a1e4b-73bf-4d81-ad09-fd88d7d7de9d
begin
	# nlsolve with theta
	"""
		UBCM_loopvec_θ!(F::Vector, x::Vector, k::Vector)
	
	cached threaded version of UBCM exponential form
	"""
	function UBCM_loopvec_θ!(F::Vector, Θ::Vector, k::Vector)
		@turbo for i in eachindex(x)
			F[i] = -f(exp(-Θ[i] - Θ[i])) - k[i]
			for j in eachindex(Θ)
				F[i] += f(exp(-Θ[i] - Θ[j]))
			end
		end

		return F
	end

	# iterative with Θ
	
	f_it(θᵢ, θⱼ) = exp(-θⱼ) / (1 + exp(-θᵢ-θⱼ)) # helper function iterative ubcm
	
	"""
		UBCM_FP_θ!(F::Vector, x::Vector, k::Vector)
	
	cached threaded version of UBCM fixed point version
	"""
	function UBCM_FP_θ!(F::Vector, Θ::Vector, k::Vector)
		#@turbo 
		for i in eachindex(Θ)
			r = -f_it(Θ[i], Θ[i])
			for j in eachindex(Θ)
				r += f_it(Θ[i], Θ[j])
			end
		
			F[i] = r
		end
		

		return -log.(k ./ F)
	end

	
	Θ₀ = -log.(degree(G) ./ sqrt(nv(G)))
	N_it = 1000
	Θ = zeros(length(Θ₀), N_it)
	Θ[:,1] = Θ₀
	F = similar(Θ₀)
	UBCM_FP_θ!(F, Θ₀, Float64.(K))

	function iterative_cm_exp(θ::Vector{T}, k) where T
		x1 = exp.(-θ)
		f = zeros(T, length(θ))
		for i in eachindex(θ)
			fx =  zero(T)
			for j in eachindex(θ)
				if i≠j
					fx += x1[j] / (1 + x1[j] * x1[i])
				end
			end
			f[i] = -log(k[i] / fx)
		end
		
		return f
	end
	UBCM_FP_θ!(F, Θ₀, Float64.(K))
	

end

# ╔═╡ 0685d90d-271a-4712-866a-de01a651aa39
collect(zip(K))

# ╔═╡ 06437f79-2978-4c51-a5dd-e56b0a0978c0
let
	a = 3
	exp(-a), -log(a)
end

# ╔═╡ 7e761b5a-fb8d-4783-a26a-75336d72b993


# ╔═╡ c186db46-2423-4315-8d9a-d5502ebc4c68
let
	import StatsBase: countmap
	k_in = indegree(G)
	k_out = outdegree(G)
	f = countmap(zip(k_in, k_out))
	@benchmark $f[(2,2)]
end

# ╔═╡ f3b8f4ee-77fe-4c2f-82dd-b4ae58de1c19
typeof(Float64)

# ╔═╡ fb1d1a69-8354-481a-a507-a3cff019aa8f
begin
	function testfun(n::Int=10000)
		f = rand(n)
		x = similar(f)
		
		for _ in 1:100
			x .= f
		end
		
		return f
	end
	
	@benchmark testfun()
end

# ╔═╡ 70ca796f-e449-48f7-8d2d-5d782ae4dc0b
bresult

# ╔═╡ d54f6183-6170-4e0c-ad3f-3305df269f7b
let
	
end

# ╔═╡ 96571ea3-5eb5-4926-8951-0c4e4902b2ac
begin
	# solution in the x-domain 
	model = fastmaxent.UBCMCompact(G)
	model.x0
	fastmaxent.solve!(model)
	sort(model.xs)
end
	

# ╔═╡ 48f5edb2-1449-4264-9a9b-09f45bb40cbb
begin
	struct DBCM{T} <: fastmaxent.AbstractMaxEntropyModel where {T<:Real}
		method::Symbol
		precision::DataType
		GPU::Bool
		f!::Function
		compressed::Bool
		compression_ratio::Float64
		x0::Vector{T}
		F::Vector{T}
		x_sol::Vector{T}
		f::Dict
		#n_out::Int
		#n_in::Int
	end
	
	Base.show(io::IO, model::DBCM{T}) where {T} = print(io, """$(model.compressed ? "Compressed " : "")DBCM model ($(T) precision, GPU $(model.GPU ? "enabled" : "disabled"))""")
	
	
	function DBCM(G::T; method::Symbol=:fixedpoint, GPU::Bool=false, compressed::Bool=true, precision::DataType=Float64) where T<:AbstractGraph
		# TO DO: add more input validation
		
		if !is_directed(G)
			@warn "Graph is not directed, consider using the UBCM model for speed"
		end
		
		# assert validity of the method TO DO: set disconnected or fully connect to 0/1 values
    	!is_connected(G) && throw(DomainError("Graph is not connected"))
    	has_self_loops(G) && throw(DomainError("Graph has self loops"))
		# get degrees
		k_out = precision.(outdegree(G))
		k_in = precision.(indegree(G))
		
		# check compression if required
		if compressed
			f = countmap(zip(k_out, k_in))
			κ_out = [k[1] for k in keys(f)]
			κ_in  = [k[2] for k in keys(f)]
		else
			κ_out = k_out
			κ_in  = k_in
			f = Dict(v => one(precision) for v in zip(k_out, k_in))
		end
		
		#n_out = length(κ_out)
		#n_in  = length(κ_in)
		compression_ratio = length(κ_out) / (nv(G))
		K = vcat(κ_out, κ_in)
		@info "K: $(length(K)) - $K"
		# datatype continuity
		α₀ = κ_out / precision.(sqrt(nv(G)))
		β₀ = κ_in  / precision.(sqrt(nv(G)))
		
		x₀ = vcat(α₀, β₀)
		F = similar(x₀)
		x_sol = similar(x₀)
		
		# generate function
		if isequal(model, :fixedpoint)
			f! = (F::Vector, x::Vector) -> (F,x)
		else
			f! = (F::Vector, x::Vector) -> DBCM_∇ℒ!(F, x, K)#, f, n_out, n_in)
		end
		
		return DBCM(method, precision, GPU,	f!, compressed, compression_ratio, x₀, F, x_sol,f)#, n_out, n_in)
	end
	
	
	function DBCM_∇ℒ!(F::Vector, X::Vector, K::Vector)#, f::Dict, n_out::Int, n_in::Int)# where {T}
		n = round(Int,length(X)/2)

		x = @view X[1:n] # linked to outdegree
		y = @view X[n+1:end] # linked to indegree
		@info n, x, y
		k_out = @view K[1:n]
		k_in  = @view K[n+1:end]
		
		for i in eachindex(x)
			F[i] = κ_out[i] + foo(x[i]*y[i])
			for j in eachindex(y)
				F[i] -= foo(x[i]*y[j])
			end
		end
		
		for i in eachindex(y)
			F[i+n] = κ_in[i] + foo(x[i]*y[i])
			for j in eachindex(x)
				F[i+n] -= foo(x[j]*y[i])
			end
		end
		
		return F
	end
	
	foo(x) = x / (1+x)
		
	

	DBCMmodel = DBCM(G, precision=Float16, method=:newton, compressed=false)

end

# ╔═╡ 174fc3be-5f2a-4802-942f-e75cf7e0ec9f
DBCMmodel.x0

# ╔═╡ ed073c0e-85b9-4608-91c3-be3cc9432408
let
	DBCMmodel.x0
end

# ╔═╡ b4eaed6d-3005-4718-8615-8c8e6072c773
DBCMmodel.f!(DBCMmodel.F, DBCMmodel.x0)

# ╔═╡ c25d79c9-ec7a-4479-802d-1c23c67f0ac8
resdbcm = nlsolve(DBCMmodel.f!,  DBCMmodel.x0)

# ╔═╡ 1fa1f3c4-d124-4d2e-9270-4a6a16002274
resdbcm.zero

# ╔═╡ 4c1f6502-f506-4b7d-afd9-611c327f2bcc
unique(

# ╔═╡ b5370f57-5665-49f0-b2eb-6dfded6b516f
@tturbo

# ╔═╡ c1e99863-e481-4fc9-8189-282b7a864e79


# ╔═╡ 9842f90f-1a0f-4de6-9cae-f5b2215c8a75
md"""
# Evaluating an example in Python
"""

# ╔═╡ d64258c7-a0f1-4b95-8507-421ba42a4491
begin
	# import python module
	nemtropy = pyimport("NEMtropy")
	np = pyimport("numpy")
	# generate model from existing graph
	graph = nemtropy.UndirectedGraph(np.array(adjacency_matrix(G)))
	# solve the problem using fixed point method -> solve in x-domain
	#@info "running new computation"
	graph.solve_tool(model="cm",method="fixed-point", initial_guess="degrees_minor",verbose=true) 
	# show solution (currently in the x domain, use cm_exp for θ domain)
	#sort(graph.solution_array)
end


# ╔═╡ 01b3511f-6228-4b91-bbda-a57d58c90ed2
graph.x0

# ╔═╡ e83efc05-0ef7-499b-b4d2-9183a4236f1f
graph.solution_array

# ╔═╡ 3beeb09b-b7f3-4392-ad12-cd7c8705715a
begin
	# this is OK
	function iterative_cm!(F::Vector{T}, x::Vector{T}, κ::Vector{T}, f::Vector{T}) where {T}
		@tturbo for i in eachindex(x)
			fx = -x[i] / (1 + x[i] * x[i])
			for j in eachindex(x)
					fx += f[j] * x[j] / (1 + x[j] * x[i])
			end
			
			F[i] = κ[i] / fx
		end
		
		return F
	end
	
	# this is OK
	function iterative_cm_own!(F::Vector{T}, x::Vector{T}, κ::Vector{T}) where {T}
		for i in eachindex(x)
			fx = -x[i] / (1 + x[i] * x[i])
			for j in eachindex(x)
					fx += x[j] / (1 + x[j] * x[i])
			end
			
			F[i] = κ[i] / fx
		end
		
		return F
	end
	
	# this works, but not perfectly OK for fixedpoint, see below
	function iterative_cm_exp!(F::Vector{T}, x::Vector{T}, κ::Vector{T}, f::Vector{T}) where {T}
		x .= exp.(-x)
		@tturbo for i in eachindex(x)
			fx = -x[i] / (1 + x[i] * x[i])
			for j in eachindex(x)
					fx += f[j] * x[j] / (1 + x[j] * x[i])
			end
			
			F[i] = -log(κ[i] / fx)
		end
		
		return F
	end
	
	
	using LinearAlgebra: norm
	function myfp(x₀, k, f)
		# initialise
		𝔉 = similar(χ₀)
		x = copy(x₀)
		# first computation
		iterative_cm!(𝔉, x, k, f)
		# next computations
		while norm(𝔉 - x) > 1e-8
			x .= 𝔉
			iterative_cm!(𝔉, x, k, f)
		end
		
		return 𝔉
	end
	
	κ = Float64.(sort(unique(degree(G))))
	freq = Float64.([count(x->x==κ[i], degree(G)) for i in eachindex(κ)])
	χ₀ = graph.x0
	fval = similar(χ₀)
	
	#iterative_cm!(𝔉, χ₀, κ, freq)
	
	#bresult = @benchmark iterative_cm!($(fval), $(χ₀), $(κ), $(freq))
	#@benchmark myfp($χ₀, $κ, $freq)
	myfp(χ₀, κ, freq)
	
	myfun! = (F::Vector, x::Vector) -> iterative_cm!(F, x, κ, freq)
	#myfun_exp! = (F::Vector, θ::Vector) -> iterative_cm!(F, θ, κ, freq)
	#@benchmark fixedpoint($(myfun!), $(χ₀))
	myfun_bis! = (F::Vector, x::Vector) -> iterative_cm_own!(F, x, Float64.(degree(G)))
	
	res_nonscaled = fixedpoint(myfun_bis!, degree(G)/nv(G))
	sort(graph.solution_array), sort(res_nonscaled.zero)
end

# ╔═╡ 9e552f4d-375d-4e56-91eb-e30698e90b08
norm(DBCMmodel.f!(DBCMmodel.F, resdbcm.zero))

# ╔═╡ ef36cc13-8863-4c0a-a39e-2692c65d2833
let
	# exponential version - gives correct answer, but fixedpoint appears to bug :-(
	graph_exp = nemtropy.UndirectedGraph(np.array(adjacency_matrix(G)))
	# solve the problem using fixed point method -> solve in θ-domain
	graph_exp.solve_tool(model="cm_exp",method="fixed-point", initial_guess="degrees_minor",verbose=true) 
	
	Θ₀ = graph_exp.x0
	myfun_exp! = (F::Vector, θ::Vector) -> iterative_cm_exp!(F, θ, κ, freq)
	
	FΘ = similar(Θ₀)
	xΘ = copy(Θ₀)
	
	@warn graph_exp.solution_array
	for _ in 1:100
		xΘ .= myfun_exp!(FΘ, xΘ)
		@info xΘ
	end
	
	isapprox(graph_exp.solution_array, xΘ)
	
	res_exp = fixedpoint(myfun_exp!, Θ₀)
	
	#@info graph_exp.solution_array, res_exp.zero
end

# ╔═╡ 4de4681e-297e-461b-b3ec-0dafa24ce434
let
	function newdbcm!(F::Vector, X::Vector, k_out::Vector, k_in::Vector,n::Int)
		x = @view X[1:n]
		y = @view X[n+1:end]
		for i in eachindex(x)
			F[i] = -k_out[i]
			for j in eachindex(y)
				if i ≠ j
					F[i] += foo(x[i]*y[j])
				end
			end
		end
		
		for i in eachindex(y)
			F[i+n] =  -k_in[i]
			for j in eachindex(x)
				if i ≠ j
					F[i+n] += foo(x[j]*y[i])
				end
			end
		end
		
		return F
	end
	
	function DBCMbis!(F::Vector, v::Vector, k::Vector)
    n = Int(length(v)/2)
    for i in 1:n
        # outdegree
        @inbounds F[i]   = -f((v[i]*v[i+n])^2) - k[i]
        # indegree
        @inbounds F[i+n] = -f((v[i]*v[i+n])^2) - k[i+n]
        @simd for j in 1:n
            # outdegree
            @inbounds F[i]   += f((v[i]*v[j+n])^2)
            # indegree
            @inbounds F[i+n] += f((v[j]*v[i+n])^2)
        end
    end
end
	
	A = [0 0 0 1 1 0 1;
	     1 0 1 0 1 0 1;
		 1 1 0 0 0 1 1;
		 1 0 0 0 1 0 0;
		 0 2 1 1 0 1 0;
		 0 1 1 0 0 0 1;
		 1 0 1 0 1 1 0]
	dgraph = DiGraph(A)
	ff = (F::Vector, X::Vector) -> newdbcm!(F::Vector, X::Vector, outdegree(dgraph), indegree(dgraph),7)
	ffbis = (F::Vector, X::Vector) -> DBCMbis!(F, X, vcat(outdegree(dgraph), indegree(dgraph)))
	X0 = [outdegree(dgraph) ; indegree(dgraph) ]./ sqrt(ne(dgraph))
	
	res = nlsolve(ff, X0)
	resbis = nlsolve(ffbis, X0)
	
	dpgraph = nemtropy.DirectedGraph(np.array(A))
	# solve the problem using fixed point method -> solve in x-domain
	#@info "running new computation"
	dpgraph.solve_tool(model="dcm",method="newton", initial_guess="degrees_minor",verbose=true) 
	
	X0, dpgraph.x0, dpgraph.dseq, hcat(outdegree(dgraph), indegree(dgraph))
	dpgraph.solution_array, res.zero, sqrt.(resbis.zero)
end

# ╔═╡ 188e8179-a00a-4994-a2cf-783f4ef35c37
let
	# let's try iterative DBCM
	A = [0 0 0 1 1 0 1;
	     1 0 1 0 1 0 1;
		 1 1 0 0 0 1 1;
		 1 0 0 0 1 0 0;
		 0 2 1 1 0 1 0;
		 0 1 1 0 0 0 1;
		 1 0 1 0 1 1 0]
	dgraph = DiGraph(A) # digraph in julia
	α₀ = -log.(outdegree(dgraph)/sqrt(ne(dgraph)))
	β₀ = -log.(indegree(dgraph)/sqrt(ne(dgraph)))
	κ = vcat(outdegree(dgraph), indegree(dgraph))
	# digraph in python
	dpgraph = nemtropy.DirectedGraph(np.array(A)) 
	dpgraph.solve_tool(model="dcm",method="fixed-point", initial_guess="degrees_minor",verbose=true) 
	#[α₀;β₀], dpgraph.x0, dpgraph.solution_array

	function iterative_dcm!(F::Vector, x::Vector, κ::Vector)
		# vector length
		n = round(Int,length(x)/2)
			
		# compute
		for i in 1:n
			fx =  - x[i+n] / (1 + x[i] * x[i+n])
			fy =  - x[i] / (1 + x[i] * x[i])
			for j in 1:n
				fx += x[j+n] / (1 + x[j+n] * x[i])
				fy += x[j]   / (1 + x[j+n] * x[i])
			end

			F[i] = κ[i] / fx
			F[i+n] = κ[i+n] / fy
		end
	
		return F
	
	end
	
	FOO! = (F::Vector, x::Vector) -> iterative_dcm!(F, x, κ)
	
	res = fixedpoint(FOO!, vcat(α₀,β₀))
	res.zero, dpgraph.solution_array
	
	
	#norm(FOO!(similar(res.zero),res.zero) - res.zero)
	#@benchmark $(FOO!)($(similar(res.zero)), $(res.zero))
end

# ╔═╡ cc067c76-989a-4790-987d-e27d42d1174d
graph.dseq

# ╔═╡ 21360c05-f3a6-4175-b4ec-61a763bc218f
isapprox(sort(model.xs), sort(graph.solution_array))

# ╔═╡ 8adde1b3-5329-418f-abb0-595e6192ad38
# location of source code of nemtropy module
nemtropy.__file__

# ╔═╡ 40845e56-cf47-4fb9-9adc-e4e80b237da1
PyCall.python

# ╔═╡ cbe474ed-4c46-40fe-9a1d-f075fdd579ec
graph.dseq

# ╔═╡ a59a741b-7cdd-40ef-ae94-801cfd207331
graph.adjacency

# ╔═╡ aaf66b29-4d6d-49f6-aa28-e522218d41c8
graph.x0

# ╔═╡ 1ef8244a-4045-48b9-9a3f-b7ce7d8ef871
sort(graph.solution_array)

# ╔═╡ 60998308-44a1-4a38-9306-f8bbb5bc8e92
graph.dseq

# ╔═╡ b3609c03-2f1d-42c8-9f29-786501ffb249
graph.r_dseq

# ╔═╡ 6b2da739-16d4-4c8d-82e9-2cc31de1d48b
# associated multiplities
graph.r_multiplicity

# ╔═╡ 31ebaed9-5b6f-40e7-b101-655fae055617
# first occurence of the degree in the initial vector
graph.r_index_dseq

# ╔═╡ fb20a3b9-96b7-44c5-bc8e-d1af8283b4af
graph.r_invert_dseq

# ╔═╡ 395bcbea-d388-41a1-ad02-2f618d2dbcad
# arguments used in addition to the unknown vector (k_i, f(k_i))
graph.args

# ╔═╡ 5bf10086-88a6-415f-88cf-96e544706e03
# blup
graph.fun(graph.x0), graph.fun(graph.fun(graph.x0)), graph.fun(graph.fun(graph.fun(graph.x0)))

# ╔═╡ 53a98c0f-5892-4aa1-813a-0161f8117f46
begin
	function mycopy(θ::Vector{T}, c, k) where {T}
    x1 = exp.(-θ)
    f = zeros(T, length(x1))
    for i in eachindex(θ)
        fx = zero(T)
        for j in eachindex(θ)
            if i == j
                fx += (c[j] - 1) * (x1[j] / (1 + x1[j] * x1[i]))
            else
                fx += (c[j]) * (x1[j] / (1 + x1[j] * x1[i]))
			end
		end

		f[i] = k[i] / fx
	end

    return f
end

function optfun(x0, c, k)
		res = mycopy(x0, c, k)
		for _ = 1:1000
			res = mycopy(res, c, k)
		end
		return res
	end
	
	myx0 = graph.x0
	myc = graph.args[2]
	myk = graph.args[1]
	res = optfun(myx0, myc, myk)
	
end

# ╔═╡ 8bc26970-40fd-49dc-9a04-be7863856937
let
	A = [0 0 0 1 1 0 1;
	     1 0 1 0 1 0 1;
		 1 1 0 0 0 1 1;
		 1 0 0 0 1 0 0;
		 0 2 1 1 0 1 0;
		 0 1 1 0 0 0 1;
		 1 0 1 0 1 1 0]
	# generate model from existing graph
	graph = nemtropy.DirectedGraph(np.array(A))
	# solve the problem using fixed point method -> solve in x-domain
	#@info "running new computation"
	graph.solve_tool(model="dcm",method="fixed-point", initial_guess="degrees_minor",verbose=true) 
	
	size(A),
graph.solution_array,length(countmap(zip(sum(A,dims=2)', sum(A,dims=1)))),vcat(sum(A,dims=2)', sum(A,dims=1)), countmap(zip(sum(A,dims=2)', sum(A,dims=1)))
	#
	dseq = np.array(collect(zip(Vector(sum(A,dims=2)[:,1]), sum(A,dims=1))))
	np.unique(dseq, return_index=true, return_inverse=true, return_counts=true,axis=0) , countmap(zip(sum(A,dims=2)', sum(A,dims=1))), collect(zip(sum(A,dims=2)', sum(A,dims=1)))
	graph.solution_array
end

# ╔═╡ 6f1b96fd-7a24-489d-8e74-dc5d9bb67e4b
graph.args

# ╔═╡ 3951b7e2-2ff2-4a9d-9123-a73253a05c82
md"""
# Overview of DBCM computation functions
"""

# ╔═╡ 8a34e528-8666-4c4f-9825-c2a1072b4019
let
	"""
	Reduced version for DBCM
	"""
	function DBCM_fp_r!(F::Vector{T}, X::Vector{T}, K::Vector{T}, f, n::Int) where {T}
		x = @view X[1:n]
		y = @view X[n+1:end]
		κ_out = @view K[1:n]
		κ_in  = @view K[n+1:end]
		
		#@info x, y, κ_out, κ_in
		
		@tturbo for i in 1:n
			fx = - y[i] / (1 + x[i]*y[i])
			fy = - x[i] / (1 + x[i]*y[i])
			for j in 1:n
				fx += f[j] * y[j] / (1 + x[i]*y[j])
				fy += f[j] * x[j] / (1 + x[j]*y[i])
			end
			
			F[i]   = κ_out[i] / fx
			F[i+n] = κ_in[i]  / fy
		end
		
		return F
	end
	
	"""
	Faster reduced version for DBCM
	"""
	function DBCM_fp_r2!(F::Vector{T}, x::Vector{T}, K::Vector{T}, f, n::Int) where {T}
		@tturbo for i in 1:n
			fx = - x[i+n] / (1 + x[i]*x[i+n])
			fy = - x[i] / (1 + x[i]*x[i+n])
			for j in 1:n
				fx += f[j] * x[j+n] / (1 + x[i]*x[j+n])
				fy += f[j] * x[j] / (1 + x[j]*x[i+n])
			end
			
			F[i]   = K[i] / fx
			F[i+n] = K[i+n]  / fy
		end
		
		return F
	end
	
	"""
	Non-reduced version for DBCM
	"""
	function iterative_dcm!(F::Vector, x::Vector, κ::Vector)
		# vector length
		n = round(Int,length(x)/2)
			
		# compute
		@tturbo for i in 1:n
			fx =  - x[i+n] / (1 + x[i] * x[i+n])
			fy =  - x[i] / (1 + x[i] * x[i+n])
			for j in 1:n
				fx += x[j+n] / (1 + x[j+n] * x[i])
				fy += x[j]   / (1 + x[j] * x[i+n])
			end

			F[i] = κ[i] / fx
			F[i+n] = κ[i+n] / fy
		end
	
		return F
	end
	
	
	minifoo(x::T) where T = exp(-x) / (one(T) + exp(-x))
	
	"""
	Non-reduced Newton method for DBCM - OK
	"""
	function DBCM_newton!(F::Vector, x::Vector, κ::Vector)	
		# vector length
		n = round(Int,length(x)/2)
		
		@tturbo for i in 1:n
			fx =  - x[i+n] / (1 + x[i] * x[i+n])
			fy =  - x[i] / (1 + x[i] * x[i+n])
			for j in 1:n
				fx += x[j+n] / (1 + x[j+n] * x[i])
				fy += x[j]   / (1 + x[j] * x[i+n])
			end

			F[i]   = -fx + κ[i] / x[i]
			F[i+n] = -fy + κ[i+n] / x[i+n]
		end
		
		return F
	end
	
	"""
	Reduced Newton method for DBCM - OK
	"""
	function DBCM_newton_r!(F::Vector, x::Vector, κ::Vector, f, n::Int)		
		@tturbo for i in 1:n
			fx =  - x[i+n] / (1 + x[i] * x[i+n])
			fy =  - x[i]   / (1 + x[i] * x[i+n])
			for j in 1:n
				fx += f[j] * x[j+n] / (1 + x[j+n] * x[i])
				fy += f[j] * x[j]   / (1 + x[j] * x[i+n])
			end

			F[i]   = -fx + κ[i] / x[i]
			F[i+n] = -fy + κ[i+n] / x[i+n]
		end

		return F
	end
	
	# APPLICATION
	
	# large scale version
	k_out = outdegree(G)
	k_in  = indegree(G)
	K = vcat(k_out, k_in)
	
	# compact version
	f = countmap(zip(Float64.(k_out), Float64.(k_in)))
	ff = Float64.(values(f))
	κ_out = [v[1] for v in keys(f)]
	κ_in  = [v[2] for v in keys(f)]
	KK = vcat(κ_out, κ_in)
	
	# function to solve
	foo_large!(F::Vector, X::Vector) = iterative_dcm!(F, X, K)
	foo_compact!(F::Vector, X::Vector) = DBCM_fp_r!(F, X, KK, ff, length(κ_out))
	foo_compact2!(F::Vector, X::Vector) = DBCM_fp_r2!(F, X, KK, ff, length(κ_out))
	foo_newt!(F::Vector, X::Vector)  = DBCM_newton!(F, X, Float64.(K))
	foo_newt_c!(F::Vector, X::Vector)  = DBCM_newton_r!(F, X, KK, ff, length(κ_out))
	
	# intial values
	X0 = K/sqrt(nv(G))
	X0_c = KK/sqrt(nv(G))
	
	# python solution
	GGG = nemtropy.DirectedGraph(np.array(adjacency_matrix(G)))
	# solve the problem using fixed point method -> solve in x-domain
	#@info "running new computation"
	GGG.solve_tool(model="dcm",method="newton", initial_guess="degrees_minor",verbose=false) 
	GGG_newton = copy(GGG.solution_array)
	GGG.solve_tool(model="dcm",method="fixed-point", initial_guess="degrees_minor",verbose=false)
	GGG_fp = copy(GGG.solution_array)
	@assert isapprox(GGG_newton, GGG_fp, rtol=1e-7) # => shuold be the same value, so something is wrong in the 
	
	# solve
	# - normal methods
	sol = fixedpoint(foo_large!, X0)
	sol_n =  nlsolve(foo_newt!,  X0)
	# - compact methods
	sol_c = fixedpoint(foo_compact!, X0_c)
	sol_c2 = fixedpoint(foo_compact2!, X0_c)
	sol_nc = nlsolve(foo_newt_c!, X0_c)
	
	if true
		b = @benchmark fixedpoint($(foo_large!), $(X0))
		b_c = @benchmark fixedpoint($(foo_compact!), $(X0_c))
		b_c2 = @benchmark fixedpoint($(foo_compact2!), $(X0_c))
		b_n = @benchmark nlsolve($(foo_newt!), $(X0))
		b_nc = @benchmark nlsolve($(foo_newt_c!), $(X0_c))

		@info "\n - fixed point large: $(b)"
		@info "\n - fixed point compact: $(b_c)"
		@info "\n - fixed point compact (bis): $(b_c2)"
		@info "\n - newton large $(b_n)"
		@info "\n - newton compact $(b_nc)"

	end
	
	sort(unique(sol.zero)), sort(unique(sol_c.zero)), sort(unique(sol_c2.zero)),  sort(unique(round.(sol_n.zero, digits=6))) , sort(unique(round.(sol_nc.zero, digits=6))), sort(unique(GGG.solution_array))
	
	
end

# ╔═╡ 6e7ddc8a-5c39-4ff6-9d9a-fdc3a0c580df


# ╔═╡ 08eca895-acb5-419c-9aa1-148b97902d79
md"""
TO DO:
* check behaviour with zero degrees
* press out more performance
* TEST FOR ACTUAL DIRECTED NETWORK!!
"""

# ╔═╡ Cell order:
# ╟─f2aa794f-1c5b-42c2-8413-f4547d62ce2f
# ╠═3025ee60-4c30-11ec-1512-2ba1bd45a489
# ╠═53533ead-5eb2-4bd4-b273-c50548683bec
# ╠═15240395-9674-4ab9-90fa-b60fc7bbb584
# ╠═9357d218-ce59-42b4-bbc4-b40806722e0a
# ╠═4c586a3c-4578-4b36-bf62-4f15d2d49c87
# ╠═85406732-dd6b-450e-847d-a0da3a5b5aec
# ╟─b92633f2-65bd-41ee-882c-12d7dc0013a7
# ╠═7ca4264c-79b8-4a00-aacd-f434a880410c
# ╠═cb3cbb8c-34bb-4410-9f77-b0cd9b6dd1a7
# ╠═0e2a1e4b-73bf-4d81-ad09-fd88d7d7de9d
# ╠═01b3511f-6228-4b91-bbda-a57d58c90ed2
# ╠═e83efc05-0ef7-499b-b4d2-9183a4236f1f
# ╠═0685d90d-271a-4712-866a-de01a651aa39
# ╠═3beeb09b-b7f3-4392-ad12-cd7c8705715a
# ╠═ef36cc13-8863-4c0a-a39e-2692c65d2833
# ╠═48f5edb2-1449-4264-9a9b-09f45bb40cbb
# ╠═4de4681e-297e-461b-b3ec-0dafa24ce434
# ╠═188e8179-a00a-4994-a2cf-783f4ef35c37
# ╠═06437f79-2978-4c51-a5dd-e56b0a0978c0
# ╠═cc067c76-989a-4790-987d-e27d42d1174d
# ╠═174fc3be-5f2a-4802-942f-e75cf7e0ec9f
# ╠═ed073c0e-85b9-4608-91c3-be3cc9432408
# ╠═b4eaed6d-3005-4718-8615-8c8e6072c773
# ╠═c25d79c9-ec7a-4479-802d-1c23c67f0ac8
# ╠═7e761b5a-fb8d-4783-a26a-75336d72b993
# ╠═1fa1f3c4-d124-4d2e-9270-4a6a16002274
# ╠═9e552f4d-375d-4e56-91eb-e30698e90b08
# ╠═c186db46-2423-4315-8d9a-d5502ebc4c68
# ╠═f3b8f4ee-77fe-4c2f-82dd-b4ae58de1c19
# ╠═fb1d1a69-8354-481a-a507-a3cff019aa8f
# ╠═70ca796f-e449-48f7-8d2d-5d782ae4dc0b
# ╠═d54f6183-6170-4e0c-ad3f-3305df269f7b
# ╠═96571ea3-5eb5-4926-8951-0c4e4902b2ac
# ╠═4c1f6502-f506-4b7d-afd9-611c327f2bcc
# ╠═21360c05-f3a6-4175-b4ec-61a763bc218f
# ╠═b5370f57-5665-49f0-b2eb-6dfded6b516f
# ╠═c1e99863-e481-4fc9-8189-282b7a864e79
# ╟─9842f90f-1a0f-4de6-9cae-f5b2215c8a75
# ╠═d64258c7-a0f1-4b95-8507-421ba42a4491
# ╠═8adde1b3-5329-418f-abb0-595e6192ad38
# ╠═40845e56-cf47-4fb9-9adc-e4e80b237da1
# ╠═cbe474ed-4c46-40fe-9a1d-f075fdd579ec
# ╠═a59a741b-7cdd-40ef-ae94-801cfd207331
# ╠═aaf66b29-4d6d-49f6-aa28-e522218d41c8
# ╠═1ef8244a-4045-48b9-9a3f-b7ce7d8ef871
# ╠═60998308-44a1-4a38-9306-f8bbb5bc8e92
# ╠═b3609c03-2f1d-42c8-9f29-786501ffb249
# ╠═6b2da739-16d4-4c8d-82e9-2cc31de1d48b
# ╠═31ebaed9-5b6f-40e7-b101-655fae055617
# ╠═fb20a3b9-96b7-44c5-bc8e-d1af8283b4af
# ╠═395bcbea-d388-41a1-ad02-2f618d2dbcad
# ╠═5bf10086-88a6-415f-88cf-96e544706e03
# ╠═53a98c0f-5892-4aa1-813a-0161f8117f46
# ╠═8bc26970-40fd-49dc-9a04-be7863856937
# ╠═6f1b96fd-7a24-489d-8e74-dc5d9bb67e4b
# ╟─3951b7e2-2ff2-4a9d-9123-a73253a05c82
# ╠═8a34e528-8666-4c4f-9825-c2a1072b4019
# ╠═6e7ddc8a-5c39-4ff6-9d9a-fdc3a0c580df
# ╠═08eca895-acb5-419c-9aa1-148b97902d79
