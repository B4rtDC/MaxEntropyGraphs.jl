### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ b4cd6f40-2d17-11ec-0cbe-e512796ade0b
begin
	using Pkg
	cd(joinpath(dirname(@__FILE__)))
    Pkg.activate(pwd())
    using Optim, LightGraphs, Distributions, ForwardDiff, Zygote
	using BenchmarkTools
	using fastmaxent
	using Plots
	using LinearAlgebra
	using SparseArrays
	using DataStructures
	ENV["GKSwstype"]="nul"
	nothing
end

# ╔═╡ 8074a705-62e7-4c89-afef-dc78640ba4ac
begin
	using Symbolics
	
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
		res = - fastmaxent.a(i,i,x)
		for j in eachindex(x)
			res += fastmaxent.a(i,j,x)
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
				res += fastmaxent.a(i,j,x) * degree(j, x)
			end
		end
		return res / degree(i, x)
	end

	"""using the computed parameters"""
	function ANND(i::Int, x::AbstractVector{T}) where T
		#@warn "ANND(i::Int, x::AbstractVector{T}) where T is used!"
		res = zero(T)
		for j in eachindex(x)
			#@info "j: $(j), $(typeof(j)) - i: $(i), $(typeof(i))"
			if j ≠ i 
				res += fastmaxent.a(i,j,x) * degree(j, x)
			end
		end
		return res / degree(i, x)
	end
	
	# appropriate function that takes only one input argument
	ANND_w(A::AbstractArray{R,2};node::Int=2) where {R<:Real} = ANND(node, A)
	ANND_w(x::AbstractVector{R};node::Int=2) where {R<:Real} = ANND(node, x)
end

# ╔═╡ 91638ea8-6dbd-443e-b3f8-df9bf58a1bc3
let
	using NLsolve
	
	"function"
	function f!(F, x)
		F[1] = (x[1]+3)*(x[2]^3-7)+18
		F[2] = sin(x[2]*exp(x[1])-1)
		return F
	end
	
	"exact jacobian"
	function J!(J, x)
		J[1, 1] = x[2]^3-7
		J[1, 2] = 3*x[2]^2*(x[1]+3)
		u = exp(x[1])*cos(x[2]*exp(x[1])-1)
		J[2, 1] = x[2]*u
		J[2, 2] = u
		#return J
	end

	# shared starting vector
	x_start = [0.1; 1.2]
	for dtype in [Float64; Float32; Float16]
		x0 = dtype.(x_start)
		@info "x0: $(typeof(x0))"
		
		# using finite differencing
		res_fd1 = @benchmark nlsolve($f!, $x0)
		@info rpad("    Finite differencing (1/2):",35," ") * "$(res_fd1)"
		# using finite differencing, OnceDifferentiable
		F0 = similar(x0)
		df = OnceDifferentiable(f!, x0, F0)
		res_df2 = @benchmark nlsolve($df, $x0)
		@info rpad("    Finite differencing (2/2):",35," ") * "$(res_df2)"
		# using exact jacobian
		res_j = @benchmark nlsolve($f!, $J!, $x0)
		@info rpad("    Jacobian:",35," ") * "$(res_j)"
		# using autodiff
		res_AD = @benchmark nlsolve($f!, $x0, autodiff=:forward)
		@info rpad("    Autodiff:",35," ") * "$(res_AD)"
		 
	end
end

# ╔═╡ cf4c1f55-47ab-49ca-b434-4c8c1938bbc0
# Make cells wider
html"""<style>
/*              screen size more than:                     and  less than:                     */
@media screen and (max-width: 699px) { /* Tablet */ 
  /* Nest everything into here */
    main { /* Same as before */
        max-width: 1200px !important; /* Same as before */
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
        margin-right: 100px !important; /* Same as before */
    } /* Same as before*/
}
</style>
"""

# ╔═╡ 9267a30e-2690-4d75-9235-a94a5c3dd703
md"""
# Performance comparison
We compare the performance of the solution of the entropy maximisation problem by :
1. actually maximising the entropy:
    - with Autodiff (forward & backward gradient)
    - with interior point methods
    - with L-BFGS (less memory intensive)
2. by solving the system of coupled non-lineair equations:
    - using a Newton method with autodiff (forward)
    - using Newton method with backward autodiff
    - using a reduced model (performance gains)
3. investigate the performance on different scales of networks
4. include comparison of performance using threading or multiprocessing/distributed

The compairison should include both computation time as well as memory footprint and control of solution validity.
"""

# ╔═╡ 3d8cb51e-9ff5-4b7e-9603-e73e701ae7e9
md"""
## General tools
"""

# ╔═╡ f871609e-4a84-42bd-8b03-874ec8a35367
# Network generation
begin
	"""
		netgenerator(N::Int)
	
	Network generator for benchmarking. Uses power law degree distributions to approximate real world interaction networks. Also generates a sparse graph in order to  mimic real world behavior
	"""
	function netgenerator(n::Int; m::Int=round(Int,sqrt(n^2*0.1)), α::Float64=rand(Uniform(2.,3.)))
		@debug "Generating graph G($(n),$(m)) with α = $(round(α, digits=2))"
		# do initial graph generation
		G = LightGraphs.static_scale_free(n, m, α)
		# check requirements
		while any(degree(G) .< 1)
			@debug "Re-generating graph G($(n),$(m)) with α = $(round(α, digits=2))"
			# rerun the generation process
			G = LightGraphs.static_scale_free(n, m, α)
		end
		
		return G
	end
	
	
	"""
		loglikelihoodUBCM(x;k)
	
	Evaluate loglikelihood function for UBCM network model with known degree vector k.
	"""
	function loglikelihoodUBCM(x::Vector{T};k=Vector{T}) where T<:Real
		n = length(x)
		res = 0.
		for i in 1:n
			res += k[i]*log(x[i])
			for j in i+1:n
				res -= log( 1 + x[i]* x[j])
			end
		end
		return res
	end
	
	UBCM(k=Vector{T}) where T<:Real = x -> - loglikelihoodUBCM(x;k=k)
	
	
end


# ╔═╡ 881a7ea7-c121-4818-87a4-ba9eeb629be2
#=
begin
	G = netgenerator(100;m=900)
	k = degree(G)
	foo = UBCM(Float64.(k))

	# starting point for optimisation
	x0 = k ./ maximum(k)
	# actual optimisation
	
	lower = zeros(size(k))
	upper = Inf .* ones(size(k))
	@warn "Starting with backwards mode automated differentation\n"
	# define function gradient
	function g!(G, x)
   		G .=  foo'(x)
	end
	odb = OnceDifferentiable(foo, g!, x0)
	resb = optimize(odb, lower, upper, x0, Optim.Fminbox(LBFGS()))
	@info "Result with LBFGS:"
	#@btime optimize($odb, $lower, $upper, $x0, $Optim.Fminbox(LBFGS()))
	@info "Result with gradient descent:"
	#@btime optimize($odb, $lower, $upper, $x0, $Optim.Fminbox(GradientDescent()))
	@info "Result with conjugate gradient:"
	#@btime optimize($odb, $lower, $upper, $x0, $Optim.Fminbox(ConjugateGradient()))

	
	@warn "Starting with forward mode automated differentation\n"
	# define function gradient
	function dfoo!(G, x)
		G .= ForwardDiff.gradient(foo, x);
	end
	
	odf = OnceDifferentiable(foo, dfoo!, x0)
	resf = optimize(odf, lower, upper, x0, Optim.Fminbox(LBFGS()))
	@info "Result with LBFGS:"
	@btime optimize($odf, $lower, $upper, $x0, $Optim.Fminbox(LBFGS()))
	@info "Result with gradient descent:"
	@btime optimize($odf, $lower, $upper, $x0, $Optim.Fminbox(GradientDescent()))
	@info "Result with conjugate gradient:"
	@btime optimize($odf, $lower, $upper, $x0, $Optim.Fminbox(ConjugateGradient()))
	
	@warn "gradient free method (Nelder-Mead)"
	res_nm = optimize(Optim.NonDifferentiable(foo, x0), lower, upper, x0, Optim.Fminbox(NelderMead()))
	@btime optimize($Optim.NonDifferentiable(foo, x0), $lower, $upper, $x0, $Optim.Fminbox(NelderMead()))
	
	
	@warn "Solving system of constraints with forward mode autodiff"
	@info "complete model:"
	model = fastmaxent.UBCM(G)
	res_s = fastmaxent.NLsolve.nlsolve(model.f!, model.x0, autodiff=:forward)
	@btime fastmaxent.NLsolve.nlsolve($model.f!, $model.x0, autodiff=:forward)
	
	@info "compact model:"
	model_c = fastmaxent.UBCMCompact(G)
	res_sc = fastmaxent.NLsolve.nlsolve(model_c.f!, model_c.x0, autodiff=:forward)
	@btime fastmaxent.NLsolve.nlsolve(model_c.f!, model_c.x0, autodiff=:forward)
	
	@warn "Solving system of constraints with backward mode autodiff"
	@info "complete model:"
	#model = fastmaxent.UBCM(G)
	#res_s = fastmaxent.NLsolve.nlsolve(model.f!, model.x0, autodiff=:backward)
	#@btime fastmaxent.NLsolve.nlsolve($model.f!, $model.x0, autodiff=:backward)
	
	#@info "compact model:"
	#model_c = fastmaxent.UBCMCompact(G)
	#res_sc = fastmaxent.NLsolve.nlsolve(model_c.f!, model_c.x0, autodiff=:backward)
	#@btime fastmaxent.NLsolve.nlsolve(model_c.f!, model_c.x0, autodiff=:backward)
end
=#

# ╔═╡ 362a8368-6a2f-4e27-9bca-b67b36e06dfc
md"""
## helper functions to obtain the maximul likelihood parameters.
"""

# ╔═╡ 1b34c22c-fb15-4ffd-931f-570d889d1994
begin
	"""
		loglikelihoodUBCMthreaded(x;k)
	
	Evaluate loglikelihood function for UBCM network model with known degree vector k.
	
	not faster :-(
	"""
	function loglikelihoodUBCMthreaded(x::Vector{T};k=Vector{T}) where T<:Real
		n = length(x)
		
		tempres = zeros(Threads.nthreads())
		out = zeros(Threads.nthreads())
		Threads.@threads for i in 1:n
			tempres[Threads.threadid()] = @inbounds k[i]*log(x[i])
			for j in i+1:n
				tempres[Threads.threadid()] -= @inbounds log( 1 + x[i]* x[j])
			end
			out[Threads.threadid()] += tempres[Threads.threadid()]
		end
		return sum(out)
	end
	
	#UBCM_threaded(k=Vector{T}) where T<:Real = x -> - loglikelihoodUBCMthreaded(x;k=k)
	
	#foo_thread = UBCM_threaded(Float64.(k))
end


# ╔═╡ 28ac5e15-2f04-42a3-b3b6-3d997cc8e836
begin
		# define function gradient for backwards mode automated differentiation
	function g_b!(G, x; f::Function)
		G .=  f'(x)
	end

	# define function gradient for forward mode automated differentiation
	function g_f!(G, x; f::Function)
		G .= ForwardDiff.gradient(f, x);
	end
	
	"""
		bencher()
	
	Function that build benchmarkgroup
	"""
	function bencher()		
		suite = BenchmarkGroup()
		
		# Consider different network sizes
		for N in [10; 100; 1000]
			# generate network
			G = netgenerator(N;m=Int(N^2*0.1))
			k = degree(G)
			f = UBCM(Float64.(k))
			@info f
			x0 = k ./ maximum(k)
			lower = zeros(size(k))
			upper = Inf .* ones(size(k))
			
			suite[N] = BenchmarkGroup()
			# Gradient free optimisation
			#suite[N]["Nelder-Mead"] = @benchmarkable optimize($Optim.NonDifferentiable($f, $x0), $lower, $upper, $x0, $Optim.Fminbox(NelderMead()))
			
			# Gradient based optimisation, forward mode autodiff
			g! = (G, x) -> g_f!(G, x, f=f)
			df = OnceDifferentiable(f, g!, x0)
			suite[N]["LBFGS-forward"] = @benchmarkable optimize($df, $lower, $upper, $x0, $Optim.Fminbox(LBFGS()))
			suite[N]["CG-forward"] =    @benchmarkable optimize($df, $lower, $upper, $x0, $Optim.Fminbox(ConjugateGradient()))
		
			# Gradient based optimisation, backward mode autodiff
			g! = (G, x) -> g_b!(G, x, f=f)
			df = OnceDifferentiable(f, g!, x0)
			suite[N]["LBFGS-backward"] = @benchmarkable optimize($df, $lower, $upper, $x0, $Optim.Fminbox(LBFGS()))
			suite[N]["CG-backward"] =    @benchmarkable optimize($df, $lower, $upper, $x0, $Optim.Fminbox(ConjugateGradient()))
			
			# NL-systems solving, forward mode autodiff
			model = fastmaxent.UBCM(G)
			model_c = fastmaxent.UBCMCompact(G)
			suite[N]["system-forward"] =        @benchmarkable fastmaxent.NLsolve.nlsolve($model.f!, $model.x0, autodiff=:forward)
			suite[N]["system-compact-forward"]= @benchmarkable fastmaxent.NLsolve.nlsolve($model_c.f!, $model_c.x0, autodiff=:forward)

			# NL-systems solving, backward mode autodiff (not available yet in NLsolve)
			#model = fastmaxent.UBCM(G)
			#model_c = fastmaxent.UBCMCompact(G)
			#suite[N]["system-backward"] =        @benchmarkable fastmaxent.NLsolve.nlsolve($model.f!, $model.x0, autodiff=:backward)
			#suite[N]["system-compact-backward"]= @benchmarkable fastmaxent.NLsolve.nlsolve($model_c.f!, $model_c.x0, autodiff=:backward)
			
		end
		
		return suite
	end
	@warn "Tuning started"
	S = bencher()
	tune!(S)
end

# ╔═╡ 9804815e-92e4-4ed4-a185-88754a737b7e
md"""
## Benchmarking solution methods
"""

# ╔═╡ c4aa5f69-a961-4ede-bd32-73e8151e804f
results = run(S, verbose=true)

# ╔═╡ 40a300bf-95b8-4b95-8835-acdc2c5673f8
# median times
begin
	times = reshape(map(x->(median(x[2].times / 1e9)), sort(leaves(results), by = x -> (x[1][1], x[1][2]))),:,length(results)) # sorted by number & method 
	memory = reshape(map(x->(x[2].memory), sort(leaves(results), by = x -> (x[1][1], x[1][2]))),:,length(results))
	method = reshape(map(x->(x[1][2]), sort(leaves(results), by = x -> (x[1][1], x[1][2]))),:,length(results)) # sorted by number & method 	
	
	reshape(map(x->(x[1][2], median(x[2].times)), sort(leaves(results), by = x -> (x[1][1], x[1][2]))),:,length(results)) # sorted by number & method 


end

# ╔═╡ 05b6c5d9-6e30-4061-a7d8-c8870cedfabe
begin
	# duration plot
	ctime = plot(times', marker=:circle, yscale=:log10, label=permutedims(method[:,1]), legend=:topleft)
	title!("UBCM comparison - time")
	xticks!(1:length(results), ["1e$(i)" for i in 1:length(results)])
	xlims!(-1,length(results)+1)
	xlabel!("N")
	ylabel!("median computation time [s]")
	
	# memory plot
	cmem = plot(memory', marker=:circle, yscale=:log10, label=permutedims(method[:,1]), legend=:topleft)
	title!("UBCM comparison - memory")
	xticks!(1:length(results), ["1e$(i)" for i in 1:length(results)])
	xlims!(-1,length(results)+1)
	xlabel!("N")
	ylabel!("memory [bytes]")
	
	p = plot(ctime, cmem, size=(800, 300))
	savefig(joinpath(homedir(),"numericalmethodperformance.pdf"))
	p
end

# ╔═╡ 45bcafcd-5954-42a6-b7ee-45429d3415a1
ff(x) = x / (1+ x)

# ╔═╡ ced286d5-24bb-4fa0-a21f-d1089c1734a4
md"""
# Obtaining derived quantities
We have now determined that our own method is the fastest and also requires the least amount of memory. Now an efficient method for determining derivative quantities is determined. For small-scale problems, it is perfectly possible to obtain these values directly by using automated differentiation applied to the quantity in question, using the full adjacency matrix. For larger problems, it quickly becomes more difficult to keep a full adjacency matrix in memory. We try to write the problem as a linear system by using the generalized chain rule.
"""

# ╔═╡ 0c9c7184-6384-4715-bb63-af1b9a693be8
md"""
Steps followed below
1. Generate network
2. Compute ML parameters
3. Generate expected adjacency matrix
4. Select metric
5. compute variability metric wrt matrix
6. scaling tests
"""

# ╔═╡ e1606670-5c7b-4eb0-99cb-adf076339453
begin
	# generate the network:
	#N = 10; G = barabasi_albert(N, 2, 2, seed=-5)
	N = 15; G = barabasi_albert(N, 5, 5, seed=-5)
	model = fastmaxent.UBCM(G)
	model_c = UBCMCompact(G)
	solve!(model)
	solve!(model_c)
	@info model.x
	A_star = adjacency_matrix(G)                # observed adjacency matrix
	d_star = LightGraphs.degree(G)              # observed degree sequence
	ANND_star = map(i -> ANND(i, A_star), 1:N)  # observed ANND

	A_exp = reshape([fastmaxent.a(i,j,model.x) for j = 1:N for i in 1:N],N,:)  # expected adjacency_matrix
	# rectify adjacency matrix
	
	d_exp = map(x->degree(x, A_exp), 1:N)                           # expected degree sequence
	ANND_exp = map(i -> ANND(i, A_exp), 1:N)                        # expected ANND
end

# ╔═╡ 4c627052-ff77-4e6d-a280-08469b666cee
model.x

# ╔═╡ 9d28215a-3a88-4458-9389-b3bb58b2ace7
function my_adjacency_matrix(x::Vector{T}) where T <: Real
	A = zeros(T, length(x), length(x))
    for i in eachindex(x)
        for j in eachindex(x)
            if i≠j
                A[i,j] = fastmaxent.f(x[i]*x[j])
            end
        end
    end
    return A
end

# ╔═╡ 22b14cfc-4cf6-454c-b528-20fb05e4200c
my_adjacency_matrix(model.x) # expected adjecency matrix

# ╔═╡ 052f6d1f-cfbc-4745-9826-852e7fb01b17
∂annd_w∂ats = ForwardDiff.gradient(ANND_w, A_exp) # partial derivates with respect to 

# ╔═╡ 600a214a-26c7-462f-9d90-e4a72cb4e341
∂annd_w∂x = ForwardDiff.gradient(ANND_w, model.x)

# ╔═╡ 6824c19a-8a0d-4011-912d-e1dd3c115557
@btime ForwardDiff.gradient($ANND_w, $A_exp)

# ╔═╡ 1c05a592-ff93-474c-ab56-8b2f82e34e91
@btime Zygote.gradient($ANND_w, $A_exp)

# ╔═╡ 01e839f8-02b0-4d37-af5a-40dc23e871d6
begin 
	"""
		∂ats∂xi_UBCM(X, t, s, i)
	
	Computes the partial derivative for the UBCM model link probabilities with respect to the likelihood maximisers: 
	```math 
	\\frac{\\partial a_{t,s}}{\\partial x_{i}}
	```
	
	This will be zero most of the time with some noteable exceptions:
	* `i==t`
	* `i==s`
	
	# Example
	```julia
	X = [0.5; 0.2; 0.1; 0.3]
	t = 1; s = 2; i=2;
	∂ats∂xi_UBCM(X, t, s, i)
	```
	
	# Notes
	If this function is called to compute `` \\frac{\\partial a_{i,i}}{\\partial x_{i}} `` it will not provide the proper value as the network model do not allow self-links. The underlying function Δa_UBCM only works for ``t \\ne s``.
	
	# See also: [Δa_UBCM](@ref)
	
	"""
	function ∂ats∂xi_UBCM(X::Vector{T}, t,s,i) where T
		if i ≠ t && i ≠ s 
			return zero(T)
		elseif i == t
			return Δa_UBCM(X, i, s)
		elseif i == s
			return Δa_UBCM(X, i, t)
		else
			return zero(T)
		end
	end
	
	"""
		Δa_UBCM(X::Vector, i, m)
	
	return numerical value for
	```math 
	\\frac{\\partial a_{i,m}}{\\partial x_{i}} = \\frac{\\partial}{\\partial x_{i}} \\left(\\frac{x_{i}x_{m}}{1+x_{i}x_{m}} \\right)= \\frac{x_{m}}{(1+x_{i}x_{m})^{2}}
	```

	"""
	function Δa_UBCM(X::Vector, i, m)
		return X[m]/(1+X[i]*X[m])^2
	end
	
	# Some tests related to this:
	
	# 1. Test function - OK, appears to work
	∂ats∂xi_UBCM(model.x, 1, 2, 3)
	∂ats∂xi_UBCM(model.x, 1, 2, 1)
	∂ats∂xi_UBCM(model.x, 1, 2, 2)
	∂ats∂xi_UBCM(model.x, 1, 1, 1) # issue, this does noet compute the correct value
	
	# 2. check numerical values with autodiff ones => PROBLEM!
	@warn "Starting checks"
	for t in eachindex(model.x)
		# generate partial wrt t
		V = model.x .+ [ForwardDiff.Dual(0, j==t ? 1 : 0) for j in eachindex(model.x)]
		# go over all values a_{t,s}
		for s in eachindex(model.x)
			if s ≠ t 
				@info abs( fastmaxent.a(t,s,V).partials[1] -  ∂ats∂xi_UBCM(model.x, t, s, t))
				@assert isapprox(fastmaxent.a(t,s,V).partials[1], ∂ats∂xi_UBCM(model.x, t, s, t))
			end
		end
	end
	
	
	# 3. some performance checks
	function partialperftest_1(X, t, s, i)
		V = X .+ [ForwardDiff.Dual(0, j==t ? 1 : 0) for j in eachindex(model.x)]
		return fastmaxent.a(t,s,V).partials[1]
	end
	function partialperftest_2(X, t, s, i)
		return ∂ats∂xi_UBCM(X, t, s, i)
	end
	
	@btime partialperftest_1(model.x, 1, 2, 2)
	@btime partialperftest_2(model.x, 1, 2, 2)
end

# ╔═╡ cadcd632-1995-4781-92f6-9d2e48cc1958
md"""
We now have dispose of the different functions that allow us to compose our system and we can compute the unknowns ``\frac{\partial M}{\partial a_{i,j}}`` by solving the system shown below. Note that the diagonal components ``a_{i,i}`` are not present, as we exclude self-links within the models.
```math
\underbrace{
\left[ \begin{matrix} 
\frac{\partial a_{1,2}}{\partial x_{1}} & \dots &  \frac{\partial a_{1,N}}{\partial x_{1}} & \frac{\partial a_{2,1}}{\partial x_{1}} & \frac{\partial a_{2,3}}{\partial x_{1}} & \dots &  \frac{\partial a_{2,N}}{\partial x_{1}} & \dots & \frac{\partial a_{N,N-1}}{\partial x_{1}} \\ 
\frac{\partial a_{1,2}}{\partial x_{2}} & \dots &  \frac{\partial a_{1,N}}{\partial x_{2}} & \frac{\partial a_{2,1}}{\partial x_{2}} & \frac{\partial a_{2,3}}{\partial x_{2}} & \dots &  \frac{\partial a_{2,N}}{\partial x_{2}} & \dots & \frac{\partial a_{N,N-1}}{\partial x_{2}} \\  
& \vdots &   &  &  & \vdots &   & \\
\frac{\partial a_{1,2}}{\partial x_{N}} & \dots &  \frac{\partial a_{1,N}}{\partial x_{N}} & \frac{\partial a_{2,1}}{\partial x_{N}} & \frac{\partial a_{2,3}}{\partial x_{N}} & \dots &  \frac{\partial a_{2,N}}{\partial x_{N}} & \dots & \frac{\partial a_{N,N-1}}{\partial x_{N}} \\ 
\end{matrix} \right] }_{\mathbb{R}^{N \times N(N-1)}}
\underbrace{
\left[ \begin{matrix} 
\frac{\partial M}{\partial a_{1,2}} \\ 
\frac{\partial M}{\partial a_{1,3}} \\ 
\vdots \\ 
\frac{\partial M}{\partial a_{N,N-1}} 
\end{matrix} \right] }_{\mathbb{R}^{N(N-1) \times 1}}
= 
\underbrace{
\left[ \begin{matrix} 
\frac{\partial M}{\partial x_{1}} \\ 
\frac{\partial M}{\partial x_{2}} \\ 
\vdots \\ 
\frac{\partial M}{\partial x_{N}}

\end{matrix} \right] }_{\mathbb{R}^{N \times 1}}
```
or in short:
```math
\nabla_{X} A \times \nabla_{a} M = \nabla_{X}M 
```

This is a sparse system of linear equations. Notice that the matrix on the left is common for all operations and even is independent of the metric ``M``.

"""

# ╔═╡ 1e2864a6-8637-49e2-8de3-8dadcd322d57
begin
	"""
		Cit
	
	Custom iterator used for building the system of unknowns
	"""
	struct Cit
		N::Int
	end
	
	# Required iteration methods
	function Base.iterate(I::Cit)
		return (1,2), (1,2)
	end
	function Base.iterate(I::Cit, state::Tuple{Int64, Int64})
		t,s = state
		if t == I.N && s == I.N - 1 # out of bounds
			return nothing
		elseif s + 1 > I.N          # over the top
			return (t+1, 1), (t+1, 1)
		else
			if s + 1 == t 			# skip identical
				return (t,s+2), (t, s+2)
			else 					# normal loop
				return (t,s+1), (t, s+1)
			end
		end
	end
	
	# optional methods
	Base.IteratorSize(m::Cit) = Base.HasLength()
	Base.IteratorEltype(m::Cit) = Base.HasEltype()
	Base.eltype(m::Cit) = Tuple{Int, Int}
	Base.length(m::Cit) = *(m.N, (m.N - 1))

	# helper function
	Base.show(io::IO, m::Cit) = print(io, "(1,2):($(m.N),$(m.N - 1))")
	# quick test
	m = Cit(10)
end

# ╔═╡ 259c686d-fcac-44b3-b1e3-ec72e541f7f5
md"""
Below we build the sparse matrix that holds the different elements ``\frac{\partial a_{i,j}}{\partial x_i}``:
"""

# ╔═╡ 21bd5ba0-0684-4c68-b434-9d06e242af87
begin
	In = Vector{Int}(undef,0)
	J = Vector{Int}(undef,0)
	V = Vector{Float64}(undef, 0)
	for (t,s) in Cit(N)
		for i in 1:N
			val = ∂ats∂xi_UBCM(model.x, t, s, i)
			if !iszero(val)
				push!(In, i)
				push!(J, (t-1)*(N-1) + (s > t ? s - 1 : s ))
				push!(V, val)
			end
		end
	end

	# A * sol = B => sol = A\B
	A = sparse(In, J, V, N, N*(N-1))
	B = ForwardDiff.gradient(ANND_w, model.x)
	sol = qr(A)\B
	nothing
end

# ╔═╡ 7d9f7269-56fe-48eb-a30c-8d82122e9833
A

# ╔═╡ b05dfebb-09f1-484e-891f-c604741c5726
heatmap(Matrix(A), yflip=true, title="illustration of sparsity pattern", size=(600, 200))

# ╔═╡ fa6e4c23-8829-4412-9616-4b27686eb11a
md"""
### Actual variance computation
the variance of M is given by:
```math
\sigma_{M}^{2} = \sum_{i,j} \left(\sigma_{a_{i,j}}  \frac{\partial M}{\partial a_{i,j}} \right) ^2 _{A = \left< A \right>^{*}}
```

We first compute the variance using the exact solution (analytical expression). We compare this with the value obtained using automated differentiation. Finally we verify the result with the analysis from the systems-based approach.
"""

# ╔═╡ ee1958df-7a22-405e-b453-f7adcd4a621c
begin
	# analytical variance:
	
	"""
		σ2_UBCM(i,j,x)
	
	Compute ``\\sigma_{a_{i,j}} = \\frac{\\sqrt{x_ix_j}}{1+x_ix_j}``

	"""
	σ2_UBCM(i::Int,j::Int,x::Vector{Float64}) = ( x[i]*x[j] ) / (1 + x[i]*x[j] )^2
	
	function δknn_iδa_ts(i::Integer, t::Integer, s::Integer, x::Vector{U}) where {U}
		if t==s
			return zero(U)
		else
			h = sum( fastmaxent.a(i, j, x) for j in eachindex(x) if j≠i)
			if i ≠ t
				return fastmaxent.a(i,t, x) / h
			else
				v = sum( fastmaxent.a(s, j, x) for j in eachindex(x) if j≠s)
				return  v/h - ANND(i, x) / h 
			end
		end
	end
	
	∂annd_w∂ats_analytical = map(T -> δknn_iδa_ts(2, T[1], T[2], model.x), Iterators.product(1:N,1:N))
	nothing
end

# ╔═╡ e34eab9b-cc67-4f76-a98b-7aea62844e72


# ╔═╡ cd6f3589-8c39-495d-a252-be2ec3a25b8a


# ╔═╡ f5355532-01c7-4520-9342-119025d63470
begin
	σ_A = zeros(typeof(model.x[1]), N,N)
	for i = 1:N
		for j = 1:N
			σ_A[i,j] = σ2_UBCM(i,j, model.x)
		end
	end
	σ_A[diagind(σ_A)] .= zero(eltype(model.x[1]))
	σ_A
end

# ╔═╡ 9dc7959f-a298-4209-84f2-7ff53b5168bb
md"""
Observe the pattern that occurs for the partial derivative. The number of unique values is similar to the number of unique degrees. So in the worst case we only need (at worst) to compute ``2(N-1)`` values instead of``N^2``. This can be reduced even more if the some degrees occur more than once.

$(heatmap(∂annd_w∂ats_analytical, yflip=true, title="partial derivatives: ∂k^nn_2  ∂ a_(ij)\nNote the values in the lines match the degree sequence\n=> less values to compute O(2N)", titlefontsize=10))

We should now exploit this observation in an efficient way to compute the partial derivatives, taking into account the layout of the variance values:

$(heatmap(σ_A, yflip=true, title="σa_{i,j}\nis_symmetric: $(issymmetric(σ_A))\n# unique elements: $(length(unique(σ_A))) / $(*(size(σ_A)...))", titlefontsize=10))

"""

# ╔═╡ 75a23746-421c-4252-bffd-14957bda7147
md"""
We find that the analytical and autodiff matrix for the variance of each model are equal (up to numerical rounding errors):

`∂annd_w∂ats_analytical ≈ ∂annd_w∂ats`: $(isapprox(∂annd_w∂ats_analytical, ∂annd_w∂ats))

Given that both use the same definition for the variance, we can conclude that the computed result will be the same (as expected).
"""

# ╔═╡ 45a02d3d-f8b3-4ed5-8818-2ee01aad3a1d
σ2_A = map(T -> σ2_UBCM(T[1], T[2], model.x), Iterators.product(1:N,1:N)); nothing

# ╔═╡ 40bb999c-ec06-44ce-8a1b-33e6087ea18b
md"""
This allow us to compute ``\sigma_{M}``:
"""

# ╔═╡ b269fdc1-9420-42d9-ad45-1fedf3b97872
σ_M = sqrt(sum(σ2_A .* (∂annd_w∂ats_analytical .^2)))

# ╔═╡ 2318d879-20b5-4afb-929b-abcd94d28338
md"""
We now return to the values that we found using the linear system:

"""

# ╔═╡ db27ccdd-c542-4551-a706-1fbad96a1fb1
begin
	# insert diagonal (zero) elements
	DD = deepcopy(sol)
	for i = 1:N
	insert!(DD, (i-1)*N + i, 0.)
	end
	# reshape into matrix
	DDD = reshape(DD,N,:)
	# compute σ
	σ_M_s = sqrt(sum(σ2_A .* DDD .^2))
end

# ╔═╡ b8e1dc68-fd4c-4d60-bab7-c3b291779fbf
md"""
It appears that we do not obtain the expected result for the variance when working with the system of linear equations:

`σ_M_s ≈ σ_M`: $(isapprox(σ_M_s, σ_M))

#sadface

#### verifying the methodology
as we know the actual values for the partial derivatives, it is perfectly possible to verify the system the other way around:

```math
\nabla_{X} A \times \nabla_{a} M \overset{?}{=} \nabla_{X}M 
```

"""

# ╔═╡ 88e819fc-acba-46a0-a5b8-da36c9aed4ad
let
	# reshape the actual value into a proper column vector:
	ctrl = Vector{Float64}(undef, N*(N-1))
	for i = 1:N
		ctrl[(i-1)*(N-1)+1 : (i)*(N-1)] = vcat(∂annd_w∂ats_analytical[i,1:i-1], ∂annd_w∂ats_analytical[i,i+1:end])
	end
	∂annd_w∂ats_analytical
	A * ctrl ≈ B
end

# ╔═╡ 9b45d824-d1c7-4c85-8025-316010fdaf75
md"""
We actually find this holds, so the "problem" is located in the solution that is found for the system. This gives us an indication that we should find a different method to compute the partial derivatives of the observed metric.
"""

# ╔═╡ 0f5e0e4a-d67a-4337-beb2-2591d28011be


# ╔═╡ 5357ff8b-110a-4577-b3f6-34f47356c385
md"""
#### Using autodiff for a combined function:
- ``f : \mathbb{R}^{m} \mapsto \mathbb{R} = f(u_1,\dots, u_m)`` where ``u_i = g_i(x_1, \dots, x_m)``
- ``g_i : \mathbb{R}^{n} \mapsto \mathbb{R}^m = g_i(x_1, \dots, x_n)``

consider the testcase where:
- ``\bar{x} = [1;2;3;4]``
- ``g_i(\bar{x}) = x_1 x_i``
- ``f(g_1, \dots, g_m) = g_1g_2^2 + \frac{g_2g_3}{g_4}``

So we find: 
- ``\frac{\partial f}{\partial g_1} = g_2^2 = (x_1x_2)^2 ``
- ``\frac{\partial f}{\partial g_2} = 2g_1g_2 + \frac{g_3}{g_4} ``
- ``\frac{\partial f}{\partial g_3} = \frac{g_2}{g_4} ``
- ``\frac{\partial f}{\partial g_4} = -\frac{g_2g_3}{g_4^2} ``
"""

# ╔═╡ ccb9af13-5b3c-4a6b-ab73-416bed7a8d7a
let
	x = [5;2;3;4]
	g(i::Int, x::Vector) = x[1]*x[i]
	f(G::Vector) = G[1]*G[2]^2 + G[2]*G[3]/G[4]
	# détail de G
	G = [g(i,x) for i in eachindex(x)]
	# dérive
	df = ForwardDiff.gradient(f, G)

	# numerical trial
	function ∂f(i, x::Vector{T}) where {T}
		# helper function
		g(i::Int,x::Vector) = x[1]*x[i]
		# helper values
		#∇ = ForwardDiff.Dual(0, one(T))
		# computation
		#(i==1 ? g(1,x) + ∇ : g(1,x)) * (i==2 ? g(2,x) + ∇ : g(2,x))^2 + (i==2 ? g(2,x) + ∇ : g(2,x))*(i==3 ? g(3,x) + ∇ : g(3,x))/(i==4 ? g(4,x) + ∇ : g(4,x))
		ForwardDiff.Dual(g(1,x), i==1 ? 1. : 0.) * ForwardDiff.Dual(g(2,x), i==2 ? 1. : 0.)^2 + ForwardDiff.Dual(g(2,x), i==2 ? 1. : 0.)*ForwardDiff.Dual(g(3,x), i==3 ? 1. : 0.)/ForwardDiff.Dual(g(4,x), i==4 ? 1. : 0.)
	end
	
	wruf = ForwardDiff.Dual(g(1,x),1.,0.,0.,0.) * ForwardDiff.Dual(g(2,x), 0.,1.,0.,0.) ^2 + ForwardDiff.Dual(g(2,x), 0.,1.,0.,0.) * ForwardDiff.Dual(g(3,x), 0.,0.,1.,0.) / ForwardDiff.Dual(g(4,x), 0.,0.,0.,1.)
	# make a single dual number
	bigx = [ForwardDiff.Dual(g(1,x),1.,0.,0.,0.);ForwardDiff.Dual(g(2,x), 0.,1.,0.,0.); ForwardDiff.Dual(g(3,x), 0.,0.,1.,0.); ForwardDiff.Dual(g(4,x), 0.,0.,0.,1.)]
	# evaluate the function
	res = f(bigx)
	
	#=
	@btime ForwardDiff.gradient($f, $G)
	@btime [$∂f(i, $x) for i in eachindex($x)] # twice as slow :-(, but works :-)
	@btime $f($bigx)
	@btime [x for x in $f($x).partials] # very fast!
	=#
	
	# results
	x,G,f(G), df, [∂f(i,x) for i in eachindex(x)], res.partials.values
	
	#[x for x in res.partials]
	#@btime [x for x in $f($bigx).partials] # very fast!
end

# ╔═╡ 0a3adfc8-646c-4462-be3e-321fb65d3178
md"""
## General method for computing ``\frac{\partial M}{\partial a_{ts}}``:
As long as the adjacenncy matrix ``A`` fits in main memory, one can simply use the gradient by using forward or backward automated differentiation. However, a lot of times, holding A in memory can be too costly. In this case, working with vectors (and accounting for some additional properties of the graphs/metrics) can be more memory-friendly.

Below we make use of `ForwardDiff.jl`'s `Dual` numbers to compute partial derivatives on the fly. We can compute a set of values by working with chunks (similar to what is done for the actual gradient computation). In function of the size of the `Dual`s, we will need to to a number of evaluations of the metric.

Each entry ``a_{i,j}``, can be computed from the vector ``x``. This allows us to compute any variant on the fly by setting the appropriate values: if `i==t && j==s`, we will set the `Dual` to 1 and 0 otherwise. In pratice we can store this in a Dict for a specific subset of values ``\{t,s\}``.
```math
\frac{\partial a_{i,j}}{\partial a_{t,s}} = \delta_{it}\delta_{js}
```

The example below illustrates how one can implement the computation for any metric, using the ``x`` vector. The appropriate `Dual` is used when required. One must find a trade-off the number of values that will be computed simultaneously. If you computed everything at once, we are back to square one, where we hold the dual value for each element of the adjacency matrix `A`.


An additional advantage of working like this, is that we can used paired couples required for the covariation when using non-local constraints.
"""

# ╔═╡ 587a5c1b-071e-4f4a-9b2d-52744517b483
let
	
	# entry point: Array(Tuple(t,s))
	TS = collect(Iterators.product(1:N, 1:N))
	# generate partials matching for each tuple => store in Dict
	V = Float64
	d = Dict{NTuple{2, Int}, ForwardDiff.Partials{length(TS), Float64}}()
	for i in eachindex(TS)
		e = Expr(:tuple, [ifelse(i === j, :(one(Float64)), :(zero(Float64))) for j in 1:length(TS)]...)
		d[TS[i]] = eval(:(ForwardDiff.Partials($(e))))
	end

	d = DataStructures.DefaultDict(zero(d[TS[1]]), d)
	d[(1,1)]
	
	"""
	
	Naive implementation (without using a buffered degree vector, possible extra gains to be made)
	"""
	function ∂annd_i∂a_ts(x::Vector{T}, d::DefaultDict, i) where {T}
		res = zero(T)
		divider = zero(T)
		for j in eachindex(x)
			if j ≠ i
				for k in eachindex(x)
					if k ≠ j
						res += ForwardDiff.Dual(fastmaxent.a(i,j,x), d[(i,j)]) * ForwardDiff.Dual(fastmaxent.a(j,k,x), d[(j,k)])
					end
				end
				
				divider += ForwardDiff.Dual(fastmaxent.a(i,j,x), d[(i,j)])
			end
		end
		
		return res / divider
	end
	
	"""
	
	faster implementation (using a buffered degree vector, possible extra gains to be made)
	"""
	function ∂annd_i∂a_ts_fast(x::Vector{T}, d::DefaultDict, i) where {T}
		function degree(x::Vector{T}, d::DefaultDict, i) where {T}
			res = zero(T)
			for j  in eachindex(x)
				if j≠i
					res += ForwardDiff.Dual(fastmaxent.a(i,j,x), d[(i,j)])
				end
			end
			return res
		end
		
		res = zero(T)
		k = [degree(x, d, i) for i in eachindex(x)] # cache degree vector
		for j in eachindex(x)
			if j ≠ i
				res += ForwardDiff.Dual(fastmaxent.a(i,j,x), d[(i,j)]) * k[j]
			end
		end
				
		
		return res / k[i]
	end
	@info "STARTING COMPAIRISON"
	@info ""
	@info "Benching entire matrix based on model.x"
	@btime $∂annd_i∂a_ts($model.x, $d, $2).partials.values
	@info ""
	@info "Benching entire matrix based on model.x (fast method)"
	@btime $∂annd_i∂a_ts_fast($model.x, $d, $2).partials.values
	@info ""
	@info "benching entire matrix based on gradient from matrix"
	@btime ForwardDiff.gradient($ANND_w, $A_exp)
	@info ""
	@info "benching analytical solution"
	@btime map(T -> δknn_iδa_ts(2, T[1], T[2], model.x), Iterators.product(1:N,1:N))
	@info ""
	@info "benching only required values"
	
	TS_2 = vcat(collect(Iterators.product(2, 1:N)), collect(Iterators.product(1:N, 1)))
	d_2 = Dict{NTuple{2, Int}, ForwardDiff.Partials{length(TS_2), Float64}}()
	for i in eachindex(TS_2)
		e = Expr(:tuple, [ifelse(i === j, :(one(Float64)), :(zero(Float64))) for j in 1:length(TS_2)]...)
		d_2[TS_2[i]] = eval(:(ForwardDiff.Partials($(e))))
	end
	d_2 = DataStructures.DefaultDict(zero(d_2[TS_2[1]]), d_2)
	@btime $∂annd_i∂a_ts($model.x, $d_2, $2).partials.values # 3x faster than full analytical, 13x faster than full odel.x
	@info "DONE"
	
	# identify if the obtained results are the same => they are !
	reshape(collect(∂annd_i∂a_ts(model.x, d, 2).partials.values), N,:) ≈ ∂annd_w∂ats_analytical
	
end

# ╔═╡ 5b6d8142-5289-4821-a640-4d1e1ec78ee2
md"""
### Optimising it more by using a `DualCache`
We check if more gains can be made by using a `DualCache`.
"""

# ╔═╡ 685dcd19-5f3f-4bd2-acf2-390acabdbe06
let
	# quick sanity check to see what is actually happening when calling upon a gradient:
	function ANND_w_check(A::AbstractArray{R,2};node::Int=2) where {R}
		@warn "running the function <ANND_w_check> with argument"#:\n$(A.dulas)"
		ANND(node, A)
	end
	
	# function runs 19 times
	ForwardDiff.gradient(ANND_w_check, A_exp)
end

# ╔═╡ 7c36b41b-314d-4c46-887c-efe9f31c0126
ForwardDiff.Dual.(A_exp, A_exp)[1,1]

# ╔═╡ 81d35d59-cd0b-473d-a03d-a3b1fb1b321c
let
	# settings
	i = 2
	t = 1
	s = 3
	# given values of the problem
	T = eltype(model.x[1])
	A = spzeros(T, N, N) 
	cache  = dualcache(A) # => why is chunck size 12?
	k = LightGraphs.degree(G) # cached degree vector
	eltype(cache.dual_du)    # check elements
	#cache.dual_du[1,1]
	#Val{ForwardDiff.Chunk(1)}
	#Val{ForwardDiff.pickchunksize(1)}
#	Val{ForwardDiff.default_cache_size(length(A))}
#PreallocationTools.DiffCache(A, size(A), 1)
	
	# Note: we impose a length of one for the chunksize (planning on doing one at a time (?)
	#x = PreallocationTools.ArrayInterface.restructure(A,zeros(ForwardDiff.Dual{nothing,T,1}, size(A)...))
	#cache = PreallocationTools.DiffCache(A, x)
	"""
	We now dispose of our cache, let's put it to use...
	"""
	
	
end

# ╔═╡ 7309f9d0-320e-4320-b19c-0131a039c5cf


# ╔═╡ Cell order:
# ╟─cf4c1f55-47ab-49ca-b434-4c8c1938bbc0
# ╟─9267a30e-2690-4d75-9235-a94a5c3dd703
# ╠═b4cd6f40-2d17-11ec-0cbe-e512796ade0b
# ╟─3d8cb51e-9ff5-4b7e-9603-e73e701ae7e9
# ╠═f871609e-4a84-42bd-8b03-874ec8a35367
# ╠═881a7ea7-c121-4818-87a4-ba9eeb629be2
# ╟─362a8368-6a2f-4e27-9bca-b67b36e06dfc
# ╠═1b34c22c-fb15-4ffd-931f-570d889d1994
# ╠═28ac5e15-2f04-42a3-b3b6-3d997cc8e836
# ╟─9804815e-92e4-4ed4-a185-88754a737b7e
# ╠═c4aa5f69-a961-4ede-bd32-73e8151e804f
# ╠═40a300bf-95b8-4b95-8835-acdc2c5673f8
# ╠═05b6c5d9-6e30-4061-a7d8-c8870cedfabe
# ╠═45bcafcd-5954-42a6-b7ee-45429d3415a1
# ╟─ced286d5-24bb-4fa0-a21f-d1089c1734a4
# ╠═8074a705-62e7-4c89-afef-dc78640ba4ac
# ╟─0c9c7184-6384-4715-bb63-af1b9a693be8
# ╠═e1606670-5c7b-4eb0-99cb-adf076339453
# ╠═4c627052-ff77-4e6d-a280-08469b666cee
# ╠═9d28215a-3a88-4458-9389-b3bb58b2ace7
# ╠═22b14cfc-4cf6-454c-b528-20fb05e4200c
# ╠═052f6d1f-cfbc-4745-9826-852e7fb01b17
# ╠═600a214a-26c7-462f-9d90-e4a72cb4e341
# ╠═6824c19a-8a0d-4011-912d-e1dd3c115557
# ╠═1c05a592-ff93-474c-ab56-8b2f82e34e91
# ╠═01e839f8-02b0-4d37-af5a-40dc23e871d6
# ╟─cadcd632-1995-4781-92f6-9d2e48cc1958
# ╠═1e2864a6-8637-49e2-8de3-8dadcd322d57
# ╟─259c686d-fcac-44b3-b1e3-ec72e541f7f5
# ╠═21bd5ba0-0684-4c68-b434-9d06e242af87
# ╠═7d9f7269-56fe-48eb-a30c-8d82122e9833
# ╠═b05dfebb-09f1-484e-891f-c604741c5726
# ╟─fa6e4c23-8829-4412-9616-4b27686eb11a
# ╠═ee1958df-7a22-405e-b453-f7adcd4a621c
# ╟─9dc7959f-a298-4209-84f2-7ff53b5168bb
# ╠═e34eab9b-cc67-4f76-a98b-7aea62844e72
# ╠═cd6f3589-8c39-495d-a252-be2ec3a25b8a
# ╠═f5355532-01c7-4520-9342-119025d63470
# ╟─75a23746-421c-4252-bffd-14957bda7147
# ╠═45a02d3d-f8b3-4ed5-8818-2ee01aad3a1d
# ╟─40bb999c-ec06-44ce-8a1b-33e6087ea18b
# ╠═b269fdc1-9420-42d9-ad45-1fedf3b97872
# ╟─2318d879-20b5-4afb-929b-abcd94d28338
# ╠═db27ccdd-c542-4551-a706-1fbad96a1fb1
# ╟─b8e1dc68-fd4c-4d60-bab7-c3b291779fbf
# ╠═88e819fc-acba-46a0-a5b8-da36c9aed4ad
# ╟─9b45d824-d1c7-4c85-8025-316010fdaf75
# ╠═0f5e0e4a-d67a-4337-beb2-2591d28011be
# ╟─5357ff8b-110a-4577-b3f6-34f47356c385
# ╠═ccb9af13-5b3c-4a6b-ab73-416bed7a8d7a
# ╟─0a3adfc8-646c-4462-be3e-321fb65d3178
# ╠═587a5c1b-071e-4f4a-9b2d-52744517b483
# ╟─5b6d8142-5289-4821-a640-4d1e1ec78ee2
# ╠═685dcd19-5f3f-4bd2-acf2-390acabdbe06
# ╠═7c36b41b-314d-4c46-887c-efe9f31c0126
# ╠═81d35d59-cd0b-473d-a03d-a3b1fb1b321c
# ╠═91638ea8-6dbd-443e-b3f8-df9bf58a1bc3
# ╠═7309f9d0-320e-4320-b19c-0131a039c5cf
