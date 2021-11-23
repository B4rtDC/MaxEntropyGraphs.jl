### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ 3025ee60-4c30-11ec-1512-2ba1bd45a489
begin
	using Pkg
	cd(joinpath(dirname(@__FILE__),".."))
	Pkg.activate(pwd())
	
	using BenchmarkTools, LoopVectorization
end

# ╔═╡ 3c1fd9c7-4b98-4252-8537-ead955b0f19b
using Plots

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
	N = [10;100;1000;10000;100000]

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
	
	# tune it
	tune!(suite)
	
	# run it
	results = run(suite, verbose = true)
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

# ╔═╡ Cell order:
# ╟─f2aa794f-1c5b-42c2-8413-f4547d62ce2f
# ╠═3025ee60-4c30-11ec-1512-2ba1bd45a489
# ╠═3c1fd9c7-4b98-4252-8537-ead955b0f19b
# ╠═15240395-9674-4ab9-90fa-b60fc7bbb584
# ╠═9357d218-ce59-42b4-bbc4-b40806722e0a
# ╠═4c586a3c-4578-4b36-bf62-4f15d2d49c87
