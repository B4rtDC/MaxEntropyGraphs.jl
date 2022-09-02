### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 029004c2-2a9c-11ed-293f-3961817e3186
begin
	using Pkg
	Pkg.activate(dirname(@__FILE__))
	
	using Graphs
	using GraphIO
	using MaxEntropyGraphs
	using BenchmarkTools
	using Statistics
	using PlutoUI
	using Plots
	using Measures
	using LaTeXStrings
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
@btime estimate_degree_params_sim();

# ╔═╡ 2b072ed7-4e3a-4809-9a66-1ffbfc12c651
@btime estimate_degree_params_ME(model);

# ╔═╡ 567c2d26-893a-400f-b026-a6f2c8742a90
md"""
### Perfomance (precision)
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
	plot(size=(800,600), bottom_margin=0mm, left_margin=0mm, thickness_scaling=1.2, legendposition=:topleft)
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
When looking at the variance, and additional advantage becomes clear: under the `UBCM`, the degree of a node follows a Poisson-Binomial distrbution
```math
k_i\sim P_b(\bar{p}) \text{ where } \bar{p} = \{a_{i,j} \}_{j \ne i}
```
so ``\mu_{k_i} = \sum \bar{p}`` and ``\sigma_{k_i} = \sum_t p[t](1 - p[t])``

In the following plot we show that the sample variance is not a good estimator for the variance.
"""

# ╔═╡ 2c0c1b0b-d02f-4715-ad8f-91ed63834db1
let
	# compute actual distribution of the degrees
	
	plot(size=(800,600), bottom_margin=0mm, left_margin=0mm, thickness_scaling=1.2, legendposition=:topleft)
	x = 1:length(model)
	Pb = MaxEntropyGraphs.degree_dist(model)
	#=
	scatter!(x, abs.(σ_obs - σ̂)./d_obs, label="MaxEntropyGraphs.jl")
	for i in eachindex(samplesizes)
		y = abs.(σ_obs - σ̂ₛ[i])./d_obs
		filt =  .!iszero.(y)
		scatter!(x[filt], y[filt], label="sampled (n = $(samplesizes[i]))", yscale=:log10)
	end
	xlabel!("node id")
	ylabel!(L"\left| \frac{d_{obs} - d_{s}}{d_{obs}} \right|")
	ylims!(1e-11, 1e3)
	yticks!(10. .^ (-10:2:0))
	title!("difference in expected variance")
	=#
end

# ╔═╡ cf449299-4ea0-4e2a-a264-4f49a9b94382
abs.(d_obs - d̂ₛ[2])

# ╔═╡ 2895a789-e0cb-4096-897c-8f79f2935f95
d̂ₛ

# ╔═╡ 670d7147-75d6-4ce7-9dff-4c29cb06d5a6
abs.(d_obs - d̂)

# ╔═╡ 1c4dcf30-8801-4159-a729-f8feb3380eb9
estimate_degree_params_sim()

# ╔═╡ 2de07a0e-8058-42ee-9f92-533956e8cbea
estimate_degree_params_ME(model)

# ╔═╡ c4d218db-e976-483b-9e98-39d2de498c50
hcat(degree.(sample)...)

# ╔═╡ 74a633fc-15fb-43d3-8a95-66dff892542a
degree(G)

# ╔═╡ 78b050f4-c5d2-4fcb-b3e9-e26bf071f993
degree(model)

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
# ╠═2c0c1b0b-d02f-4715-ad8f-91ed63834db1
# ╠═cf449299-4ea0-4e2a-a264-4f49a9b94382
# ╠═2895a789-e0cb-4096-897c-8f79f2935f95
# ╠═670d7147-75d6-4ce7-9dff-4c29cb06d5a6
# ╠═1c4dcf30-8801-4159-a729-f8feb3380eb9
# ╠═2de07a0e-8058-42ee-9f92-533956e8cbea
# ╠═c4d218db-e976-483b-9e98-39d2de498c50
# ╠═74a633fc-15fb-43d3-8a95-66dff892542a
# ╠═78b050f4-c5d2-4fcb-b3e9-e26bf071f993
