### A Pluto.jl notebook ###
# v0.19.12

using Markdown
using InteractiveUtils

# ╔═╡ c33689f4-ab9d-44c8-a528-f25f99b48478
begin
	using Pkg
	Pkg.activate(dirname(@__FILE__))
end

# ╔═╡ 139ece8c-4557-11ed-091d-9ba155b70bfd
begin
	# The essentials
	using Graphs
	using MaxEntropyGraphs
	using PyCall

	# For the plots
	using Plots
	using Measures
	using LaTeXStrings

	# For the analysis
	using Statistics
	using StatsBase
	using Distributions
	using SparseArrays
	using LinearAlgebra

	# For the performance
	using BenchmarkTools

	# For I/O and logs
    using JLD2
	using Printf
	using Dates
	using GraphIO

	# for analysis
	using Distances
end

# ╔═╡ 445de09a-c066-4121-bac5-844ec125622a
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

# ╔═╡ f85b116d-1ef0-4b5c-b147-690701694b91
begin
	# load up python modules
	nemtropy = PyCall.pyimport("NEMtropy")
	# load up numpy
	np = PyCall.pyimport("numpy")
	# load up networkx
	nx = PyCall.pyimport("networkx")
	sp = PyCall.pyimport("scipy")
	ig = PyCall.pyimport("igraph")
	# show versions that are used
	println("Working with the following dependencies:\n- python-igraph $(ig.version.__version__)\n- networkx $(nx.__version__)\n- scipy $(sp.__version__)\n- igraph $(ig.__version__)")
end

# ╔═╡ 291576ff-912e-43fb-ac7d-0fdaba9f8d82
begin
	"""
		scipyCSC_to_julia(A)

	Convert a scipy CSC matrix to a Julia matrix. 
	"""
function scipyCSC_to_julia(A; dense=false) # Compressed Sparse Column
    m, n = A.shape
    colPtr = Int[i+1 for i in PyArray(A."indptr")]
    rowVal = Int[i+1 for i in PyArray(A."indices")]
    nzVal = Vector(PyArray(A."data"))
    B = SparseMatrixCSC{eltype(nzVal),Int}(m, n, colPtr, rowVal, nzVal)
    return dense ? Array(B) : B
end

	"""
		scipyCSR_to_julia(A)

	Convert a scipy CSR matrix to a Julia matrix. 
	"""
function scipyCSR_to_julia(A; dense=false) #  Compressed Sparse Row
	# dimension
	m,n = A.shape
	I,J,V = sp.sparse.find(A)
	B = sparse(I .+ 1,J .+ 1, V, m, n)
    
    return dense ? Array(B) : B
end
	
println("Conversion functions defined")
end

# ╔═╡ 974133be-c0d0-40cb-ae4b-48558ed72b13
md"""
# Impact of the sampling method
Some methods can be subject to the following limitations:
- generate duplicate edges and loops. These self-loops and parallel edges can be removed (see below), but this will likely result in a graph that doesn’t have the exact degree sequence specified. This “finite-size effect” decreases as the size of the graph increases.
- the sampling can be non-uniform

Below we provide the tools to work with the samplers from `igraph` and `networkx`. Sampling from the grand canonical ensemble can be done with `MaxEntropyGraphs.jl`
"""

# ╔═╡ 0fcd72ac-bc46-46d1-8532-eddd8c1f8fcf
begin
	"""
		igraph_sample(G; method="no_multiple")
	
	sample the microcanonical ensemble using the method implemented in igraph. Method can be either one of:
	* "configuration" -- simple generator that implements the stub-matching configuration model. It may generate self-loops and multiple edges. **This method does not sample multigraphs uniformly**, but it can be used to implement uniform sampling for simple graphs by rejecting any result that is non-simple (i.e. contains loops or multi-edges).
	* "configuration_simple" -- similar to "configuration" but rejects generated graphs if they are not simple. **This method samples simple graphs uniformly**
	* "fast_heur_simple" -- similar to "configuration" but avoids the generation of multiple and loop edges at the expense of increased time complexity. The method will re-start the generation every time it gets stuck in a configuration where it is not possible to insert any more edges without creating loops or multiple edges, and there is no upper bound on the number of iterations, but it will succeed eventually if the input degree sequence is graphical and throw an exception if the input degree sequence is not graphical. **This method does not sample simple graphs uniformly**.
	* "edge_switching_simple" -- an MCMC sampler based on degree-preserving edge switches. It generates simple undirected or directed graphs. The algorithm uses Graph.Realize_Degree_Sequence() to construct an initial graph, then rewires it using Graph.rewire(). => Note that the default settings here resort to (from the underlying C code) generating a simple graph with the same degree sequence follow by rewiring in sutch a way that the graph remains simple. The number of rewirings equals 10* the number of edges counts. In the original Newman paper on random sampling they used Q=100 instead of 10.
	
	Returns an adjacency matrix for further analysis
	"""
	function igraph_sample(G; method="configuration")
		if is_directed(G)
			ig.Graph.Degree_Sequence(PyCall.array2py(outdegree(G)), 		 
							 				  in_=PyCall.array2py(indegree(G)),
											  method=method).get_adjacency().data
		else
			ig.Graph.Degree_Sequence(PyCall.array2py(degree(G)), 		 
											   method=method).get_adjacency().data
		end
	end

	"""
		networkx_sample(G; method="simple")

	sample the microcanonical ensemble using the method implemented in networkx

	Returns an adjacency matrix for further analysis
	"""
	function networkx_sample(G; method="simple", dense=false)
		if is_directed(G)
			net = nx.directed_configuration_model(indegree(G), outdegree(G))
			# remove multi-edge
			net = nx.DiGraph(net)
			# remove self-edges
			net.remove_edges_from(nx.selfloop_edges(net))
		else
			nx.configuration_model(degree(G))
			# remove multi-edge
			net = nx.Graph(net)
			# remove self-edges
			net.remove_edges_from(nx.selfloop_edges(net))
		end
		
		# return the adjacency matrix
		mat = nx.adjacency_matrix(net)
			
		if sp.sparse.isspmatrix_csr(mat)
			return scipyCSR_to_julia(mat;dense)
		elseif sp.sparse.isspmatrix_csc
			return scipyCSC_to_julia(mat; dense)
		else
			error("spare matrixtype $(mat) not parseable")
			return
		end
		
	end
	
	nothing

end

# ╔═╡ d6e00c15-6cf8-42ac-b898-7ed814441f68
md"""
# Impact of the significance evaluation method
This can be done in multiple ways:
- z-score
- SP
- frequentist p-value. For the frequentist p-value, you could also obtain an equivalent z-score to allow for a compairison in profiles.
"""

# ╔═╡ fc9fb1a5-293c-4faa-a37d-b3ead498c941
begin
	"""
		p_val_frequentist(x, X; tail=:right)
	
	compute the probability of observing a value `x` larger (right sided) or smaller (left sided) than the randomly generated values in vector `X`.

	It is recommended to have at least 1000 observations.
	"""
	function p_val_frequentist(x::T, X::Vector; tail=:right) where T<: Number 
		if tail == :right
			return sum(X .>= x) / length(X)
		elseif tail == :left
			return sum(X .<= x) / length(X)
		end
	end
end

# ╔═╡ 9a2b21df-a0bd-4757-8091-8a26777c9157
begin
	# quick sanity test
	@assert abs( p_val_frequentist(0.9, rand(100000), tail=:right) - 0.1) < 1e-3
	@assert abs( p_val_frequentist(0.9, rand(100000), tail=:left)  - 0.9) < 1e-3
end

# ╔═╡ d36a948f-21c1-4b16-8601-62d8fd29ed1b
md"""
# Some actual results
We load up a selection of datasets used in the original Squartini paper. For each sampling method 1000 random instances are generated. The z-scores and frequentist p-values are computed. For the grand canonical ensemble, the z-scores are computed analytically and from a sample (there are some differences between these) => why?

the SP for the different methods is shown.
"""

# ╔═╡ e854f6ed-35ab-4f97-ad65-590f572b57a7
# use existing datasets
data = JLD2.jldopen("../data/computed_results/DBCM_result_NS22.jld")

# ╔═╡ 126420c8-ef84-4744-8336-35349dedbe96
begin
	function motifs(A::Matrix, n::Int...)
		eval.(map(f -> :($(f)($A)), [MaxEntropyGraphs.DBCM_motif_functions[i] for i in n])) 
	end
	
	motifs(A::Matrix) = motifs(A,1:13...)
end


# ╔═╡ c4a938fb-7a9b-4bef-87f2-45d6ed8c3400
data

# ╔═╡ 00d38637-a674-4e3e-b076-13b9811002ed
begin
	# SAMPLE GENERATION
	networkname = "littlerock"
	G = data[networkname][:network]
	G_ME = MaxEntropyGraphs.DBCM(G)
	N = 1000
end

# ╔═╡ 61120fcb-7038-4c08-9e83-67e087e93370
begin
	# networkx sample
	S_networkx     = [networkx_sample(G,dense=true) for _ in 1:N]
	println("finished networkx")
	S = Dict("networkx" => S_networkx)
	nothing
end

# ╔═╡ 251daacd-febc-4701-9264-b36d8fb4d2fb
begin
	# igraph sample
	S_igraph_conf  = [igraph_sample(G, method="configuration") for _ in 1:N]
	println("finished igraph (conf)")
	S["igraph (configuration)"] = S_igraph_conf
	nothing
end

# ╔═╡ 812fe1c3-c8aa-40a4-88ea-d6687e2c9eab
begin
	S_igraph_MCMC  = [igraph_sample(G, method="edge_switching_simple") for _ in 1:N]
	println("finished igraph (mcmc)")
	S["igraph (MCMC)"] = S_igraph_MCMC
	nothing
end

# ╔═╡ 60b5ab75-d629-485a-ac8e-1ed52be6703c
begin
	#S_igraph_nunif = [igraph_sample(G, method="fast_heur_simple") for _ in 1:N]
	#println("finished igraph (nunif)")
	#S["igraph (non-uniform)"] => S_igraph_nunif
end

# ╔═╡ a49bbc34-dc0a-4929-9877-9036e156665d
begin
	# maximum entropygraphs sample
	S_nemtropy = [Matrix(adjacency_matrix(rand(G_ME))) for _ in 1:N]
	println("finished nemtropy")
	S["nemtropy (sample)"] = S_nemtropy
	nothing
end

# ╔═╡ d036cca7-1179-4240-9fcc-73f51de741c1
begin
	# degree distribution
	d_out = outdegree(G)
	d_in = indegree(G)
	p_out = plot([minimum(d_out); maximum(d_out)], 
				 [minimum(d_out); maximum(d_out)], 
				 label="", linestyle=:dash, color=:black, xlabel="outdegree", ylabel="mean outdegree")
	p_in = plot([minimum(d_in); maximum(d_in)], 
				 [minimum(d_in); maximum(d_in)], 
				 label="", linestyle=:dash, color=:black, xlabel="indegree", ylabel="mean indegree")
	for (method, sample) in S
		d_mean_out = mean(hcat(map(x -> sum(x, dims=2), sample)...), dims=2)
		d_mean_in = reshape(mean(vcat(map(x -> sum(x, dims=1), sample)...), dims=1),:)  

		scatter!(p_in,   indegree(G), d_mean_in,  label=method, markeralpha=0.8)
		scatter!(p_out, outdegree(G), d_mean_out, label=method, markeralpha=0.8)
	end
	scatter!(p_in,  indegree(G), indegree(G_ME),  label="nemtropy", markeralpha=0.8)
	scatter!(p_out, outdegree(G),outdegree(G_ME), label="nemtropy", markeralpha=0.8)

	# illustration assembly
	global_title = plot(title = "Mean node degree ($(N) generations per model)", grid=false, showaxis=false, ticks=false, bottom_margin = -10Plots.px)
	subplots = plot(p_out, p_in, legendposition=:topleft)
	degree_plot = plot(global_title, subplots, layout=@layout([A{0.01h}; B]) )
	savefig(degree_plot, "$(networkname) - degree_plot.pdf")
	degree_plot
end

# ╔═╡ 36501914-a352-4899-aea6-fc55f4b6070f
begin
	# MOTIF analysis (SP)
	Mˣ = MaxEntropyGraphs.motifs(G) # observed value
	SP_p = plot(xlabel="motif id", ylabel="significance profile")
	z_p = plot(xlabel="motif id", ylabel="z-score")
	for (method, sample) in S
		# observed motifs
		M_s = hcat(motifs.(sample)...)
		# central values
		μ = reshape(mean(M_s, dims=2),:)
		σ = reshape( std(M_s, dims=2),:)
		z = (Mˣ - μ) ./ σ
		SP = z ./ norm(z)
		# frequentist p value
		p_obs = map(i -> p_val_frequentist(Mˣ[i], M_s[i,:],tail=:left), 1:13)
		z_p_obs = quantile.(Normal(), p_obs)
		SP_p_obs = z_p_obs ./ norm(z_p_obs)

		# original SP
		plot!(SP_p, 1:13, SP, label=method, marker=:circle, markeralpha=0.8)
		# SP based on frequentist approach
		plot!(SP_p, 1:13, SP_p_obs, label="$(method) - frequentist", marker=:cross, markeralpha=0.8, 			color=SP_p.series_list[end][:linecolor])

		# original z
		plot!(z_p, 1:13, z, label=method, marker=:circle, markeralpha=0.8)
		# z based on frequentist approach
		#plot!(z_p, 1:13, z_p_obs, label="$(method) - frequentist", marker=:cross, markeralpha=0.8,
		#color=z_p.series_list[end][:linecolor])
	end

	m̂  = MaxEntropyGraphs.motifs(G_ME)
	σ̂_m̂  = Vector{Float64}(undef, length(m̂))
    for i = 1:length(m̂)
        σ̂_m̂[i] = MaxEntropyGraphs.σˣ(MaxEntropyGraphs.DBCM_motif_functions[i], G_ME)
    end
	z_m_a = (Mˣ - m̂) ./ σ̂_m̂  
	plot!(SP_p, 1:13, z_m_a ./ norm(z_m_a), label="nemtropy", marker=:circle, markeralpha=0.8)
	plot!(z_p, 1:13, z_m_a, label="nemtropy", marker=:circle, markeralpha=0.8)
	
	# illustration assembly
	SP_title = plot(title = "Significance profile ($(N) generations per model)", grid=false, showaxis=false, ticks=false, bottom_margin = -10Plots.px)
	SP_subplots = plot(SP_p, legendposition=:bottomright,xticks=collect(1:13),ylims=(-3,1), xlims=(0,14))
	SP_plot = plot(SP_title, SP_subplots, layout=@layout([A{0.01h}; B]) )
	savefig(SP_plot, "$(networkname) - SP_plot.pdf")
	SP_plot
end

# ╔═╡ f5cb4e5a-29cd-4006-8cff-2fa0917b62d2
begin
	z_title = plot(title = "z-scores ($(N) generations per model)", grid=false, showaxis=false, ticks=false, bottom_margin = -10Plots.px)
	plot!(z_p,[0; 14], [-2; -2], label="", line=:dash, color=:black)
	plot!(z_p,[0; 14], [2; 2], label="", line=:dash, color=:black)
	z_subplots = plot(z_p, legendposition=:bottomright,xticks=collect(1:13),ylims=(-15,7), xlims=(0,14))
	z_plot = plot(z_title, z_subplots, layout=@layout([A{0.01h}; B]) )
	savefig(z_plot, "$(networkname) - z_plot.pdf")
	z_plot
end

# ╔═╡ 195e6a21-4acb-462d-a449-0376868fc1f0
let
	# assortativity check
	# => for each network plot mean k[i]_nn in function of its actual degree (these metrics are defined in MaxEntrtopyNetworks)
	# we strat from the existing samples and we will use the approach followed by squartini for representation (i.e. if an observed degree is associated with multiple knn, we only represent the mean).
	
	"""
		groupreduce(x,y; group=unique, reduce=mean)

	
	"""
	function groupreduce(x,y; group::Function=unique,
							  reducer::Function=mean)
		@assert length(x) == length(y)
		red = Dict{eltype(x),typeof(y)}()
		for i in eachindex(x)
			push!(get!(red, x[i], Vector{typeof(y)}()), y[i])
		end

		return collect(keys(red)), map(reducer, values(red))
	end

	# Observed data
	indegassosplot = scatter(groupreduce(indegree(G), MaxEntropyGraphs.ANND_in(G))..., label="observed", marker=:cross, color=:black )
	
	outdegassosplot = scatter(groupreduce(outdegree(G), MaxEntropyGraphs.ANND_out(G))..., label="observed", marker=:cross, color=:black )
	# different samples
	for (method, sample) in S
		# computation
		s_annd_in = reshape(mean(hcat(map(MaxEntropyGraphs.ANND_in, sample)...), dims=2),:)
		s_annd_out = reshape(mean(hcat(map(MaxEntropyGraphs.ANND_out, 
		sample)...), dims=2),:)
		# plotting
		
		scatter!(indegassosplot, 
				groupreduce(indegree(G), s_annd_in)..., 
							label=method)
		scatter!(outdegassosplot, 
				groupreduce(outdegree(G), s_annd_out)..., 
							label=method  )
		
	end
	# nemtropy method
	scatter!(indegassosplot, 
				groupreduce(indegree(G), MaxEntropyGraphs.ANND_in(G_ME))..., 
							label="nemtropy")
	scatter!(outdegassosplot, 
				groupreduce(outdegree(G), MaxEntropyGraphs.ANND_out(G_ME))..., 
							label="nemtropy")

	
	xlabel!(indegassosplot, "indegree")
	ylabel!(indegassosplot, "ANND_in")
	xlabel!(outdegassosplot, "outdegree")
	ylabel!(outdegassosplot, "ANND_out")
	


	# illustration assembly
	annd_in_title = plot(title = "ANND ($(N) generations per model)", grid=false, showaxis=false, ticks=false, bottom_margin = -10Plots.px)
	annd_in_subplot = plot(indegassosplot, legendposition=:bottomright)
	annd_in_plot = plot(annd_in_title, annd_in_subplot, layout=@layout([A{0.01h}; B]) )
	savefig(annd_in_plot, "$(networkname) - annd_in_plot.pdf")
	
	annd_out_title = plot(title = "ANND ($(N) generations per model)", grid=false, showaxis=false, ticks=false, bottom_margin = -10Plots.px)
	annd_out_subplot = plot(outdegassosplot, legendposition=:bottomright)
	annd_out_plot = plot(annd_out_title, annd_out_subplot, layout=@layout([A{0.01h}; B]) )
	savefig(annd_out_plot, "$(networkname) - annd_out_plot.pdf")
	
	
	
	annd_in_plot, annd_out_plot

	
	#plot(indegassosplot, outdegassosplot)
	#scatter(indegree(G), MaxEntropyGraphs.ANND_out(G_ME),label="")
	#xred, yred = groupreduce(indegree(G), MaxEntropyGraphs.ANND_out(G_ME))
	#@info redres
	#@info mean(redres[1]) 
	#xred = collect(keys(redres))
	#yred = map(mean, values(redres))
	#scatter(xred, yred, xlabel="indegree", ylabel="ANND", label="observed", marker=:cross, color=:black)
	
	#savefig(indegassosplot, "assorativity_in_plot.pdf")
end

# ╔═╡ 8c7fc4d9-1a21-413e-8fd7-26ab9fe7478c
md"""
## Quick check Luis' paper on migration patterns in sex workers.
"""

# ╔═╡ c786be85-d8e8-4b82-bff9-772d6142dc48
begin 
	# Load up and parse the raw data
	edges = split.(readlines("countries_weight.txt"),',')
	countries = sort(union(unique([x[1] for x in edges]), unique([x[2] for x in edges])))
	nodeids = Dict(country => num for (country, num) in zip(countries,1:length(countries)))
	# generate the actual network
	migration_net = DiGraph(length(countries))
	for edge in edges
		add_edge!(migration_net, nodeids[edge[1]], nodeids[edge[2]])
	end
	# generate sample storage
	S_luis = Dict{String, Vector}()
	# generate sample result
	res_luis = Dict{String, Matrix}()
	res_luis["observed_motifs"] = reshape(MaxEntropyGraphs.motifs(migration_net),:,1)
	# show the network
	migration_net
end

# ╔═╡ 0e79f8f8-37f2-4863-b685-395069d4441d
"$(summary(migration_net))($(nv(migration_net)))"

# ╔═╡ add2df54-0a04-42c6-8003-d2521cfed531
# Sample generation
let
	# sample size
	N = 1000
	
	# igraph samples (adjacency matrices)
	S_luis["igraph (configuration)"] = [igraph_sample(migration_net, method="configuration") for _ in 1:N]
	S_luis["igraph (MCMC)"] = [igraph_sample(migration_net, method="edge_switching_simple") for _ in 1:N]
	nothing
	
	# networkx samples (adjacency matrices)
	S_luis["networkx"] = [networkx_sample(migration_net,dense=true) for _ in 1:N]

	# nemtropy model (BDCM model)
	model = DBCM(migration_net)
	S_luis["nemtropy"] = [model]
	
	# nemtropy samples (DiGraphs)
	S_luis["nemtropy (samples)"] = [rand(model) for _ in 1:N]	
	
end

# ╔═╡ ab004d5f-fe03-42b1-8474-9f2b8aa213af
# result computation
let
	for (method, sample) in S_luis
		if eltype(sample) <: Matrix
			println("$(method)")
			res_luis[method] = hcat(motifs.(sample)...)
		elseif eltype(sample) <: DiGraph
			println("$(method)")
			res_luis[method] = hcat(MaxEntropyGraphs.motifs.(sample)...)
		end
	end
end

# ╔═╡ 88dd9d31-c057-47c6-8199-974b2e446e37
res_luis

# ╔═╡ 2d91aead-4ef8-4bc7-b4c0-3d539d1d6f88
let
	z_p = plot(xlabel="motif id", ylabel="z-score")
	for (method, res) in res_luis
		if method ≠ "observed_motifs"
			μ = reshape(mean(res, dims=2),:)
			σ = reshape( std(res, dims=2),:)
			z = (res_luis["observed_motifs"] - μ) ./ σ
				# original SP
			plot!(z_p, 1:13, z, label=method, marker=:circle, markeralpha=0.8)
		end
	end
	plot!(z_p, 0:14, -2*ones(15), label="", linestyle=:dash, color=:black)
	plot!(z_p, 0:14, 2*ones(15), label="", linestyle=:dash, color=:black)

	z_title = plot(title = """z-scores ($(size(res_luis["networkx"],2)) generations per model)""", grid=false, showaxis=false, ticks=false, bottom_margin = -10Plots.px)
	z_subplot = plot(z_p, legendposition=:topleft,xticks=collect(1:13),ylims=(-6,6), xlims=(0,14))
	z_plot = plot(z_title, z_subplot, layout=@layout([A{0.01h}; B]) )
	savefig(z_plot, "migrationnet - z_plot.pdf")
	z_plot
end

# ╔═╡ 81a3fe4a-7b22-4553-8c35-03d6594ef48a
eltype(S_luis["networkx"]) <: Array

# ╔═╡ Cell order:
# ╠═445de09a-c066-4121-bac5-844ec125622a
# ╠═c33689f4-ab9d-44c8-a528-f25f99b48478
# ╠═139ece8c-4557-11ed-091d-9ba155b70bfd
# ╠═f85b116d-1ef0-4b5c-b147-690701694b91
# ╠═291576ff-912e-43fb-ac7d-0fdaba9f8d82
# ╟─974133be-c0d0-40cb-ae4b-48558ed72b13
# ╠═0fcd72ac-bc46-46d1-8532-eddd8c1f8fcf
# ╟─d6e00c15-6cf8-42ac-b898-7ed814441f68
# ╠═fc9fb1a5-293c-4faa-a37d-b3ead498c941
# ╠═9a2b21df-a0bd-4757-8091-8a26777c9157
# ╟─d36a948f-21c1-4b16-8601-62d8fd29ed1b
# ╠═e854f6ed-35ab-4f97-ad65-590f572b57a7
# ╠═126420c8-ef84-4744-8336-35349dedbe96
# ╠═c4a938fb-7a9b-4bef-87f2-45d6ed8c3400
# ╠═00d38637-a674-4e3e-b076-13b9811002ed
# ╠═61120fcb-7038-4c08-9e83-67e087e93370
# ╠═251daacd-febc-4701-9264-b36d8fb4d2fb
# ╠═812fe1c3-c8aa-40a4-88ea-d6687e2c9eab
# ╠═60b5ab75-d629-485a-ac8e-1ed52be6703c
# ╠═a49bbc34-dc0a-4929-9877-9036e156665d
# ╠═d036cca7-1179-4240-9fcc-73f51de741c1
# ╠═36501914-a352-4899-aea6-fc55f4b6070f
# ╠═f5cb4e5a-29cd-4006-8cff-2fa0917b62d2
# ╠═195e6a21-4acb-462d-a449-0376868fc1f0
# ╠═8c7fc4d9-1a21-413e-8fd7-26ab9fe7478c
# ╠═0e79f8f8-37f2-4863-b685-395069d4441d
# ╠═c786be85-d8e8-4b82-bff9-772d6142dc48
# ╠═add2df54-0a04-42c6-8003-d2521cfed531
# ╠═ab004d5f-fe03-42b1-8474-9f2b8aa213af
# ╠═88dd9d31-c057-47c6-8199-974b2e446e37
# ╠═2d91aead-4ef8-4bc7-b4c0-3d539d1d6f88
# ╠═81a3fe4a-7b22-4553-8c35-03d6594ef48a
