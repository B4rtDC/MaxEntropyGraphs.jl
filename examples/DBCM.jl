### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 34ee4b6a-232a-4154-ab42-050efab16424
begin
	using Pkg
	Pkg.activate(dirname(@__FILE__))
	using Printf
	using Dates
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

# ╔═╡ ce77d578-328b-11ed-34ef-7b8e9f47f3c5
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

# ╔═╡ 81e2adc6-75dd-4d48-9259-9a87902e7e8e
md"""
# DBCM demo

For the Directed Binary Configuration Model (DBCM), we will be using a set of networks that has also been used in “*Analytical maximum-likelihood method to detect patterns in real networks*” by T. Squartini and D. Garlaschelli. The different networks are available as a `.lg` file in the `/data/networks` folder.
"""

# ╔═╡ 85ac94a3-160f-43ca-9a2a-bbf59eb6aede
G = Graphs.loadgraph("../data/networks/floridabay_directed.lg")

# ╔═╡ e096d53b-64f2-469e-b05b-006565e0b06c
md"""
The function below is a the helper function. It generates the model and a sample from the network ensemble. It also computes the degree sequences and the motifs. This is done for the observed network, according to the Squartini method and based on the sample.

You can use this function to compute the different values and a new sample. Running everything might take some time (depending on your computer).
The analysis has been done for each network and is stored in `/data/computed_results/DBCM_complete.jld`. This file can be loaded in 
memory using `JLD`
```Julia
path = "./data/computed_results/DBCM_result_more.jld"
data = jldopen(path)
````
"""

# ╔═╡ c13394e0-7a86-4ab1-9504-25d60aa1b2bb
DBCM_analysis(G)

# ╔═╡ 5cdf0f9e-f65d-48bb-9da7-73963b1ba2b7


# ╔═╡ f1e37b3a-e3ae-42e6-85d4-5c065d23e89a


# ╔═╡ Cell order:
# ╟─ce77d578-328b-11ed-34ef-7b8e9f47f3c5
# ╠═34ee4b6a-232a-4154-ab42-050efab16424
# ╟─81e2adc6-75dd-4d48-9259-9a87902e7e8e
# ╠═85ac94a3-160f-43ca-9a2a-bbf59eb6aede
# ╠═e096d53b-64f2-469e-b05b-006565e0b06c
# ╠═f1e37b3a-e3ae-42e6-85d4-5c065d23e89a
# ╠═c13394e0-7a86-4ab1-9504-25d60aa1b2bb
# ╟─5cdf0f9e-f65d-48bb-9da7-73963b1ba2b7
