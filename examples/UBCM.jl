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

# ╔═╡ 2cf7416d-904f-4d13-9b57-c39a70287239
md"""
# UBCM demo

Let's start with the classic Zachary karate club network.
"""

# ╔═╡ bf60aef3-e074-4017-be63-d6d1763cc6b1
G = Graphs.smallgraph(:karate)

# ╔═╡ 648f4b22-c5ec-402c-9d22-e5ee7e636266
methods(UBCM)

# ╔═╡ 1b6cc7b0-8059-413e-9caa-8a0c2d718382
model = MaxEntropyGraphs.UBCM(G)

# ╔═╡ Cell order:
# ╟─74545a15-78cf-4bfc-837d-fc4395ac5626
# ╠═029004c2-2a9c-11ed-293f-3961817e3186
# ╟─2cf7416d-904f-4d13-9b57-c39a70287239
# ╠═bf60aef3-e074-4017-be63-d6d1763cc6b1
# ╠═648f4b22-c5ec-402c-9d22-e5ee7e636266
# ╠═1b6cc7b0-8059-413e-9caa-8a0c2d718382
