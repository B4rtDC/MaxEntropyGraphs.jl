### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 029004c2-2a9c-11ed-293f-3961817e3186
begin
	using Pkg
	Pkg.activate(dirname(@__FILE__))
	using Printf
	
	using Graphs
	using GraphIO
	using MaxEntropyGraphs
	using BenchmarkTools
	using Statistics
	using PlutoUI
	using Plots
	using Measures
	using LaTeXStrings
	using Distributions
end

# ╔═╡ 74545a15-78cf-4bfc-837d-fc4395ac5626
html"""
 <! -- this adapts the width of the cells to display its being used on -->
<style>
	main {
		margin: 0 auto;
		max-width: 2000px;
    	padding-left: max(160px, 10%);
    	padding-right: max(160px, 10%);
	}
</style>
"""

# ╔═╡ b99ba474-374b-4210-be2b-3bdbdb7fad1c
TableOfContents()

# ╔═╡ 2cf7416d-904f-4d13-9b57-c39a70287239
md"""
# UBCM demo

Let's start with the classic Zachary karate club network.

## Model generation
"""

# ╔═╡ bf60aef3-e074-4017-be63-d6d1763cc6b1
G = Graphs.smallgraph(:karate)

# ╔═╡ 4b3d5c2d-3847-4237-9df8-370a9dab4ee2
md"""
You can build the model directly from the network, or from your own vector of coefficients
"""

# ╔═╡ 648f4b22-c5ec-402c-9d22-e5ee7e636266
methods(UBCM)

# ╔═╡ 1b6cc7b0-8059-413e-9caa-8a0c2d718382
model = MaxEntropyGraphs.UBCM(G)

# ╔═╡ f0b345d2-a941-4f41-b321-22692f386aff
@assert degree(model) ≈ degree(G) # this should be respected (UBCM fixes the degree sequence)

# ╔═╡ de3c7d9e-ed66-4427-94af-6707c5694a51
md"""
## Degree metric
Let's have a look at the degree using the analytical method and the sampling/simulation method.
"""

# ╔═╡ 3fa1470d-1d16-482d-a4b3-234f454d6aaf
function estimate_degree_params_sim(N::Int=100)
	# generate the sample
	sample = [rand(model) for _ in 1:100];
	# compute the estimates
	μ_d = reshape(mean(hcat(degree.(sample)...),dims=2),:)
	σ_d = reshape(std(hcat(degree.(sample)...),dims=2),:)

	return μ_d, σ_d
end

# ╔═╡ b7fc1147-17d2-4e54-a6bc-3caf12810b91
function estimate_degree_params_ME(m::UBCM)
	μ_d = degree(m)
	σ_d = map(n -> MaxEntropyGraphs.σˣ(G -> degree(G, n), m), 1:length(m))

	return μ_d, σ_d
end

# ╔═╡ 1ebc60be-c265-48f5-aba4-3e2eda110508
md"""
### Performance (computation time)
"""

# ╔═╡ b9245598-ced5-42d7-8f72-5a4789dc44c1
@btime estimate_degree_params_sim(100);

# ╔═╡ 2b072ed7-4e3a-4809-9a66-1ffbfc12c651
@btime estimate_degree_params_ME(model);

# ╔═╡ 567c2d26-893a-400f-b026-a6f2c8742a90
md"""
### Performance (precision)
Let's have a look at the difference between the expected degree for both methods with repect to the observed degree. This should be zero, as this is the way the UBCM model is defined.
"""

# ╔═╡ c953ccac-3677-456f-841d-4b03b197a3fc
begin
	# computation of the different estimates
	d_obs = degree(G)
	d̂, σ̂ = estimate_degree_params_ME(model)
	d̂ₛ, σ̂ₛ = Vector{Float64}[], Vector{Float64}[]
	samplesizes = [100;1000; 10000]
	for N = samplesizes
		res = estimate_degree_params_sim(N)
		push!(d̂ₛ, res[1])
		push!(σ̂ₛ, res[2])
	end
end

# ╔═╡ 3fd44870-f7e1-4484-9862-82f23289f244
md"""
The result obtained by `MaxEntropyGraphs` is much closer to the expected result.
"""

# ╔═╡ 6dbd1571-85fb-465f-a101-a9488aa99ba4
let
	plot(size=(600,400), bottom_margin=0mm, left_margin=5mm, thickness_scaling=1.0, legendposition=:topleft)
	x = 1:length(model)
	scatter!(x, abs.(d_obs - d̂)./d_obs, label="MaxEntropyGraphs.jl")
	for i in eachindex(samplesizes)
		y = abs.(d_obs - d̂ₛ[i])./d_obs
		filt =  .!iszero.(y)
		scatter!(x[filt], y[filt], label="sampled (n = $(samplesizes[i]))", yscale=:log10)
	end
	
	xlabel!("node id")
	ylabel!(L"\left| \frac{d_{obs} - d_{s}}{d_{obs}} \right|")
	ylims!(1e-11, 1e3)
	yticks!(10. .^ (-10:2:0))
	title!("difference in expected degree")
end

# ╔═╡ bacd35c3-87d8-4193-8692-752bc508079b
md"""
Under the `UBCM`, the degree of a node follows a Poisson-Binomial distrbution
```math
k_i\sim P_b(\bar{p}) \text{ where } \bar{p} = \{a_{i,j} \}_{j \ne i}
```
so ``\mu_{k_i} = \sum \bar{p}`` and ``\sigma_{k_i} = \sum_t p[t](1 - p[t])``

In the following plot we show that the sample variance is not a good estimator for the variance.
"""

# ╔═╡ 2c0c1b0b-d02f-4715-ad8f-91ed63834db1
let
	# compute actual distribution of the degrees
	x = 1:length(model)
	Pb = MaxEntropyGraphs.degree_dist(model)
	σ_th = std.(Pb) .^ 2

	# illustration (sometimes the difference is actually zero, these values won't be shown)
	p=plot(size=(600,400), bottom_margin=0mm, left_margin=5mm, thickness_scaling=1.0, legendposition=:bottomleft)
	scatter!(x, abs.(σ_th - σ̂ .^ 2)./σ_th, label="MaxEntropyGraphs.jl")
	for i in eachindex(samplesizes)
		y = abs.(σ_th - σ̂ₛ[i] .^2)./σ_th
		filt =  .!iszero.(y)
		scatter!(x[filt], y[filt], label="sampled (n = $(samplesizes[i]))", yscale=:log10)
	end
	xlabel!("node id")
	ylabel!(L"\left| \frac{\hat{\sigma}^2 - \sigma^2_{th}}{\sigma^2_{th}} \right|")
	ylims!(1e-22, 1e1)
	yticks!(10. .^ (-18:2:0))
	title!("Difference in expected variance")
	savefig(p,"variance.png")
	p
end

# ╔═╡ 04eedbbd-fbf5-4515-b081-3739f550b261
md"""
The estimators that are typically used, i.e. sample mean and variance are the optimal estimators when the underlying distribution of the data is normal. In the illustration below, we show that is is not the case. We look at the sample distribution of the node with the lowest degree.
"""

# ╔═╡ f437603e-dd0d-4e3d-adcb-e521f7b1337e
begin
	Node = 12
	Ns = 10000
	S_d_max = map( x -> degree(x, Node), [rand(model) for _ in  1:Ns])
	μ̂_d_max, σ̂_d_max = mean(S_d_max), std(S_d_max)
	N_d_max = Normal(μ̂_d_max, σ̂_d_max)
	pvecinds = [i for i in range(1,length(model)) if i≠Node]
	Pb_d_max = PoissonBinomial(model.G[Node, pvecinds])
	hist = histogram(S_d_max, label="sample")
	
	x = unique(S_d_max)
	y = map(x -> count(i-> isequal(i,x), S_d_max), x)
	myplot = bar(x, y ./ sum(y), bar_width=1, label=LaTeXString("sample disttribution (n = $(Ns))"), fillalpha=0.5)

	xvals = range(-5, 10, step=0.2)
	k = 10
	mylabel = LaTeXString(@sprintf("\$\\mathcal{N}(\\hat{\\mu}, \\hat{\\sigma}) \\;\\;\\;  (σ = %1.3e)\$", std(N_d_max)))
	plot!(myplot,xvals, pdf.(N_d_max, xvals), label=mylabel, linecolor=:black, linestyle=:dash)
	myotherlabel = LaTeXString(@sprintf("\$\\mathcal{P}_b(\\bar{p}) \\;\\;\\;\\;\\;\\;\\;  (σ = %1.3e)\$", std(Pb_d_max)))
	plot!(0:10, pdf.(Pb_d_max, 0:10), label=myotherlabel, linecolor=:red, linestyle=:dot, marker=:circle, markercolor=:red)
	
	xticks!(-4:2:8)
	xlabel!(L"d")
	ylabel!(L"f_X(x)")
	title!("Experimental vs theoretical distribution")
end

# ╔═╡ 7ddac9cb-78ed-49c6-933e-6c119d37fb5b
md"""
## Average nearest neighbor degree
We have seen that there are differences for the structural constraints that control the model. When looking at higher order metrics, the difference between the values obtained from the sample and those computed using the analytical method can be even more important.
"""

# ╔═╡ 8027ee24-9c17-452c-ae49-0e25ccf370e8
begin
	N = 1000
	S = [rand(model) for _ in 1:N]
	ANND(model,1)
end

# ╔═╡ 3c4f9957-1f70-4a49-a710-aeb145163fe1
methods(ANND)

# ╔═╡ 7412c9c1-8303-407b-a988-2b12bbcdb6fe
methods(degree)

# ╔═╡ 42c7e783-f2ea-411a-a0ca-724bf327a4cf
degree(model)[34]

# ╔═╡ Cell order:
# ╟─74545a15-78cf-4bfc-837d-fc4395ac5626
# ╠═029004c2-2a9c-11ed-293f-3961817e3186
# ╠═b99ba474-374b-4210-be2b-3bdbdb7fad1c
# ╟─2cf7416d-904f-4d13-9b57-c39a70287239
# ╠═bf60aef3-e074-4017-be63-d6d1763cc6b1
# ╟─4b3d5c2d-3847-4237-9df8-370a9dab4ee2
# ╠═648f4b22-c5ec-402c-9d22-e5ee7e636266
# ╠═1b6cc7b0-8059-413e-9caa-8a0c2d718382
# ╠═f0b345d2-a941-4f41-b321-22692f386aff
# ╟─de3c7d9e-ed66-4427-94af-6707c5694a51
# ╠═3fa1470d-1d16-482d-a4b3-234f454d6aaf
# ╠═b7fc1147-17d2-4e54-a6bc-3caf12810b91
# ╟─1ebc60be-c265-48f5-aba4-3e2eda110508
# ╠═b9245598-ced5-42d7-8f72-5a4789dc44c1
# ╠═2b072ed7-4e3a-4809-9a66-1ffbfc12c651
# ╟─567c2d26-893a-400f-b026-a6f2c8742a90
# ╠═c953ccac-3677-456f-841d-4b03b197a3fc
# ╟─3fd44870-f7e1-4484-9862-82f23289f244
# ╟─6dbd1571-85fb-465f-a101-a9488aa99ba4
# ╟─bacd35c3-87d8-4193-8692-752bc508079b
# ╟─2c0c1b0b-d02f-4715-ad8f-91ed63834db1
# ╟─04eedbbd-fbf5-4515-b081-3739f550b261
# ╟─f437603e-dd0d-4e3d-adcb-e521f7b1337e
# ╟─7ddac9cb-78ed-49c6-933e-6c119d37fb5b
# ╠═8027ee24-9c17-452c-ae49-0e25ccf370e8
# ╠═3c4f9957-1f70-4a49-a710-aeb145163fe1
# ╠═7412c9c1-8303-407b-a988-2b12bbcdb6fe
# ╠═42c7e783-f2ea-411a-a0ca-724bf327a4cf
