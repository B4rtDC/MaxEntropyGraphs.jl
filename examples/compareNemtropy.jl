### A Pluto.jl notebook ###
# v0.19.12

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
    using JLD2
	using StatsBase
	using PyCall
end

# ╔═╡ 26ea1b33-8baf-405d-b962-9fe84f37ca5b
Pkg.add("PyCall")

# ╔═╡ 20cb867f-54b2-477a-a696-1550eb19606f
using ReverseDiff, ForwardDiff

# ╔═╡ 6275e711-9651-4f66-ba94-c1ad9b6354fd
using SparseArrays

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

# ╔═╡ a3bc7cac-3128-46e1-a67f-f3557c2e757d
md"""
# Comparing results with NEMtropy
"""



# ╔═╡ c2fa3030-6177-401e-ae89-ad163e7b762d
nemtropy = PyCall.pyimport("NEMtropy")

# ╔═╡ aca6f625-623b-45fb-be5e-19e6e513fc05
#σˣ(M₁₃, data["littlerock"][:model])

# ╔═╡ f2c806b7-7d28-411b-9d7f-e464be4b592e
@btime 1+1

# ╔═╡ 37e78aeb-ec2f-44f8-a5c8-5d1130835ec2
#ForwardDiff.gradient(M₁₃, data["littlerock"][:model].G)

# ╔═╡ 81e2adc6-75dd-4d48-9259-9a87902e7e8e
md"""
# DBCM demo

For the Directed Binary Configuration Model (DBCM), we will be using a set of networks that has also been used in “*Analytical maximum-likelihood method to detect patterns in real networks*” by T. Squartini and D. Garlaschelli. The different networks are available as a `.lg` file in the `/data/networks` folder.
"""

# ╔═╡ 85ac94a3-160f-43ca-9a2a-bbf59eb6aede
G = Graphs.loadgraph("../data/networks/maspalomas_directed.lg")

# ╔═╡ e096d53b-64f2-469e-b05b-006565e0b06c
md"""
The function below is a the helper function. It generates the model and a sample from the network ensemble. It also computes the degree sequences and the motifs. This is done for the observed network, according to the Squartini method and based on the sample.

You can use this function to compute the different values and a new sample. Running everything might take some time (depending on your computer).
The analysis has been done for each network and is stored in `/data/computed_results/DBCM_complete.jld`. This file can be loaded in 
memory using `JLD2`.
"""

# ╔═╡ c13394e0-7a86-4ab1-9504-25d60aa1b2bb
let
    res = DBCM_analysis(G, N_max = 120)
    outpath="../data/computed_results/DBCM_result_demo.jld"
	isfile(outpath) && Sys.rm(outpath)
	MaxEntropyGraphs.write_result(outpath, :maspalomas, res)
	recovered_data = JLD2.jldopen(outpath)["maspalomas"]
end

# ╔═╡ 5cdf0f9e-f65d-48bb-9da7-73963b1ba2b7
# continuing with precomputed results
data = JLD2.jldopen("../data/computed_results/DBCM_result_NS22.jld")

# ╔═╡ a7b87189-0438-40ca-b7ee-9f1115f6dfb2
mygraph = data["littlerock"][:network]

# ╔═╡ 17eea405-8a40-4bd3-926c-6fd5774ecf3b
NEM_model = nemtropy.DirectedGraph(degree_sequence=vcat(Graphs.outdegree(mygraph), Graphs.indegree(mygraph)))

# ╔═╡ a9670da9-7acf-430a-889f-4989426ece0e
NEM_model.solve_tool(model="dcm_exp"; method="fixed-point", initial_guess="degrees", max_steps=5000, tol=1e-12)

# ╔═╡ 7df8ba8e-f7a2-41b8-a97a-07f697241380
sol = vcat(NEM_model.x, NEM_model.y)

# ╔═╡ 54565e0a-1a78-4abe-9f00-89d235b7763e
m̂_nem = [ 
	nemtropy.ensemble_functions.expected_dcm_3motif_1(sol);
	nemtropy.ensemble_functions.expected_dcm_3motif_2(sol);
	nemtropy.ensemble_functions.expected_dcm_3motif_3(sol);
	nemtropy.ensemble_functions.expected_dcm_3motif_4(sol);
	nemtropy.ensemble_functions.expected_dcm_3motif_5(sol);
	nemtropy.ensemble_functions.expected_dcm_3motif_6(sol);
	nemtropy.ensemble_functions.expected_dcm_3motif_7(sol);
	nemtropy.ensemble_functions.expected_dcm_3motif_8(sol);
	nemtropy.ensemble_functions.expected_dcm_3motif_9(sol);
	nemtropy.ensemble_functions.expected_dcm_3motif_10(sol);
	nemtropy.ensemble_functions.expected_dcm_3motif_11(sol);
	nemtropy.ensemble_functions.expected_dcm_3motif_12(sol);
	nemtropy.ensemble_functions.expected_dcm_3motif_13(sol);
]

# ╔═╡ 21e933ee-8492-4963-b715-53df84e259e6
σ̂_nem = [
	nemtropy.ensemble_functions.std_dcm_3motif_1(sol);
	nemtropy.ensemble_functions.std_dcm_3motif_2(sol);
	nemtropy.ensemble_functions.std_dcm_3motif_3(sol);
	nemtropy.ensemble_functions.std_dcm_3motif_4(sol);
	nemtropy.ensemble_functions.std_dcm_3motif_5(sol);
	nemtropy.ensemble_functions.std_dcm_3motif_6(sol);
	nemtropy.ensemble_functions.std_dcm_3motif_7(sol);
	nemtropy.ensemble_functions.std_dcm_3motif_8(sol);
	nemtropy.ensemble_functions.std_dcm_3motif_9(sol);
	nemtropy.ensemble_functions.std_dcm_3motif_10(sol);
	nemtropy.ensemble_functions.std_dcm_3motif_11(sol);
	nemtropy.ensemble_functions.std_dcm_3motif_12(sol);
	nemtropy.ensemble_functions.std_dcm_3motif_13(sol);
]

# ╔═╡ 03c2cda0-083d-4a2f-8370-080f3f0529f1
nemtropy.ensemble_functions.std_dcm_3motif_12(sol), nemtropy.ensemble_functions.std_dcm_3motif_13(sol)

# ╔═╡ 38be7b27-2190-4903-8a9a-03d7f1adc5e4
m̂_mod = motifs(data["littlerock"][:model])

# ╔═╡ eb17d996-1ff2-4bc2-b61f-cd4468aeceb7
# 2-norm of computed results
sqrt(sum( (m̂_nem .- m̂_mod).^2))

# ╔═╡ 4bf64fd4-774f-4780-a5e3-248ae45f9478
data["littlerock"][:m̂]

# ╔═╡ 3f187e60-14f3-463c-b519-7828cc82d689
σ̂_mod = data["littlerock"][:σ̂_m̂]

# ╔═╡ dfd00eea-7466-4de1-b420-d80655cd17a1
sqrt(sum( (σ̂_nem .- σ̂_mod).^2))

# ╔═╡ 2a0a550a-42b6-4fcb-821f-4de8bbb63de5
ReverseDiff.gradient(M₁₃, data["littlerock"][:model].G)

# ╔═╡ e0a47426-0f27-41e3-bc77-9b66e0634d8d
data

# ╔═╡ a1d7eaba-03a7-4e55-a546-535732b4f3c5
md"""
## Degree distributions
As for the BDCM, we know the degree should follow a Poisson-Binomial distribution. This is shown again on the illustration
"""

# ╔═╡ 3a5ebf8c-40af-44db-8f96-0d3ba5e892a7
function DBCM_degree_dist_plot(data, dataset, nodeid)
	# data to show
	d_in, d_out = data[dataset][:d_in_S][nodeid, :], data[dataset][:d_out_S][nodeid, :]
	# custom histogram
	dx_in, dx_out = proportionmap(d_in), proportionmap(d_out)
	x_in, x_out = range(minimum(d_in), maximum(d_in)), range(minimum(d_out), maximum(d_out))
	# actual plot
	inplot = bar(dx_in, normalize=:pdf, bar_width=1, label="Sample")
	plot!(inplot, x_in, pdf.(data[dataset][:d_in_dist][nodeid], x_in), label="Poisson-Binomial", marker=:circle)
	title!(inplot, "indegree")
	outplot = bar(dx_out, normalize=:pdf, bar_width=1, label="Sample")
	try
		plot!(outplot, x_out, pdf.(data[dataset][:d_out_dist][nodeid], x_out), label="Poisson-Binomial", marker=:circle)
	catch
		nothing
	end
	title!(outplot, "outdegree")

	plot(inplot, outplot, size=(1200, 600))
end

# ╔═╡ 7274e006-bd84-4702-b07e-b5669234359e
DBCM_degree_dist_plot(data, "stmarkseagrass", 21)

# ╔═╡ b7077a41-9e9f-4584-bd9e-fdb3a1cbb666
md"""
## Motifs
"""

# ╔═╡ 73fee4ae-551f-42f9-8783-177c5bbbf96b
md"""
We can reproduce the top panel of Figure 5 from “Analytical maximum-likelihood method to detect patterns in real networks” by T. Squartini and D. Garlaschelli using this package.

Additionally, we can also show the sampled values for the different motifs
"""

# ╔═╡ 141b21f5-6373-4205-9ea5-822b6b446137
"""
	DBCMmotifplot(data::JLDFile)

from a `JLDFile` holding results from multiple DCBM, show an overview of the z-scores for the different motifs.
"""
function DBCMmotifplot(data)
	data_labels = keys(data)
    plotnames = ["Chesapeake Bay";"Everglades Marshes"; "Florida Bay"; "Little Rock Lake"; "Maspalomas Lagoon"; "St Marks Seagrass" ]
    mycolors = [:crimson;        :saddlebrown;          :indigo;     :blue;           :lime;  :goldenrod ]; 
    markershapes = [:circle;        :star5 ;            :dtriangle;  :rect;               :utriangle;  :star6 ]
    linestyles = [:dot; :dash; :solid]

	motifplot = plot(size=(1600, 1200), bottom_ofset=5mm, left_ofset=5mm, thickness_scaling = 2)
    for i = eachindex(data_labels)
        plot!(motifplot, 1:13, data[data_labels[i]][:z_m_a], label=plotnames[i], color=mycolors[i], marker=markershapes[i], 
        markerstrokewidth=0, markerstrokecolor=mycolors[i])
        y = data[data_labels[i]][:z_m_S]
        for j = 1:size(y,2)
            plot!(motifplot, 1:13, y[:, j], label="", color=mycolors[i], marker=markershapes[i], 
                                        markerstrokewidth=0, markerstrokecolor=mycolors[i], line=linestyles[j], linealpha=0.5, markeralpha=0.5)
        end
    end
    plot!(motifplot,[0;14], [-2 2;-2 2], label="", color=:black, legend_position=:bottomright)
    xlabel!(motifplot, "motif")
    ylabel!(motifplot, "z-score")
    xlims!(motifplot, 0,14)
    xticks!(motifplot, 1:13)
    ylims!(motifplot, -15,15)
    yticks!(motifplot ,-15:5:15)
    title!(motifplot ,"Analytical vs Simulation results for motifs (DBCM model)\n-: analytical, .: n=100, - -:, n=1000, -: n=10000")
end

# ╔═╡ de98df21-9662-4647-9c16-9b46a7c208c1
DBCMmotifplot(data)

# ╔═╡ 627b455c-b196-47f8-843e-7806dee098a2
md"""
We can have a look at the sample distribution for motifs where there appears to be a large relative difference between z-score of the motif based on a sample or based on the analytical formulation.
"""

# ╔═╡ ba8ae8d1-95ab-4dab-bef8-954e70d5fb57
"""
	motifdistribution(data, motifnumber)

show the distribution of the motifs sampled from the dataset
"""
function motifdistribution(data, subset, motifnumber)
	motif_Pb_plot = plot(size=(1200, 800), bottom_ofset=2mm, left_ofset=10mm, thickness_scaling = 1.5,legendposition=:topleft)
	y = @view data[subset][:S_m][motifnumber,:]

	counts = countmap(y)
	
	histogram!(motif_Pb_plot, y, normalize=true, label="PDF sample")
	d_normal = fit(Normal{Float64},y)
	xrange = minimum(motif_Pb_plot.series_list[end][:x]):maximum(motif_Pb_plot.series_list[end][:x])
	plot!(motif_Pb_plot, xrange, pdf.(d_normal, xrange), label="normal fit")
	xlabel!(motif_Pb_plot, "motif value")
    ylabel!(motif_Pb_plot, L"f_X(x)")
	title!(motif_Pb_plot, "motif $(motifnumber) occurence \n($(subset), n = $(length(y)))")
	annotate!((maximum(xrange)-minimum(xrange))*3/4, maximum(motif_Pb_plot.series_list[end][:y])*3/4, "σ_analytical:  $(@sprintf("%1.2e", data[subset][:σ̂_m̂][motifnumber]))\nσ_sample: $(@sprintf("%1.2e", data[subset][:σ̂_m̂_S][motifnumber,end]))")
end

# ╔═╡ 52ccfad6-eba6-44d7-8dfe-7d5fde65da46
motifdistribution(data, "chesapeakebay", 9)

# ╔═╡ 8c74de6e-21f7-4f72-b03c-1206f430ec39
md"""
## comparing the microcanonical ensemble (hard constraints) with the canonical ensemble (soft constraints)

concept: for one of the networks, especially with large differences between sample z scores and analytical z-scores, have a look at the microcanonical ensemble and how it behaves.
"""

# ╔═╡ e550f8ed-0bdd-42ce-9196-2fe35bb33c72
nx = PyCall.pyimport("networkx")

# ╔═╡ 24c2322a-8f55-4311-a807-00c8a98ec854
function scipyCSC_to_julia(A)
    m, n = A.shape
    colPtr = Int[i+1 for i in PyArray(A."indptr")]
    rowVal = Int[i+1 for i in PyArray(A."indices")]
    nzVal = Vector{Float64}(PyArray(A."data"))
    B = SparseMatrixCSC{Float64,Int}(m, n, colPtr, rowVal, nzVal)
    return Array(B)#PyCall.pyjlwrap_new(B)
end

# ╔═╡ 70f953dc-592b-46a1-bd36-3c56e2153d52
"""
	nx_configuration_model_sample(g)

use the networkx configuration model to obtain a julia adjacency matrix
"""
function nx_configuration_model_sample(g)
	# generation
	net = nx.directed_configuration_model(indegree(g), outdegree(g))
	# remove multi-edge
	net = nx.DiGraph(net)
	# remove self-edges
	net.remove_edges_from(nx.selfloop_edges(net))
	# return the adjacency matrix
	return scipyCSC_to_julia(nx.adjacency_matrix(net))
end

# ╔═╡ c664c215-e095-4fc7-b0ae-606e5854a16c
length(nx.directed_configuration_model(indegree(G), outdegree(G)).edges())

# ╔═╡ 14aedfe8-7901-4254-8836-400bce1b3ed2
length(unique(nx.directed_configuration_model(indegree(G), outdegree(G)).edges()))

# ╔═╡ bf025884-c1b2-4327-9c27-a057b80c681c
nx.degree_distribution(nx.directed_configuration_model(indegree(G), outdegree(G)))

# ╔═╡ 5405acb5-3ff2-40ad-ba6e-c388b46f5a81
length(nx.directed_configuration_model(indegree(G), outdegree(G)))

# ╔═╡ 975768bd-aa16-49a5-8b49-3894a8f319c5
print(nx.directed_configuration_model(indegree(G), outdegree(G)).in_degree())

# ╔═╡ df198107-6066-4220-8c40-1b92c3c1cdd8
print(nx.directed_configuration_model(indegree(G), outdegree(G)).out_degree())

# ╔═╡ a61a1c73-784b-4268-bb4a-3b31ab72d1f1
histogram([nx.number_of_selfloops(nx.directed_configuration_model(indegree(G), outdegree(G))) for _ in 1:1000],nbars=10)

# ╔═╡ 629afdfa-8e68-4ff6-ae44-9a115445d79d
md"""
lets check the motif counts for 1000 samples
"""

# ╔═╡ 72b9f000-832b-485d-a10d-35339c7e9cbb
begin
	# initialise
	N_sim = 1000
	motifcount = zeros(Int, 13, N_sim)
	for i in 1:N_sim
		net = nx_configuration_model_sample(data["floridabay"][:network])
		# compute motifs
		motifcount[:,i] = map(f -> f(net), MaxEntropyGraphs.DBCM_motif_functions)
	end
	
end

# ╔═╡ 1bebace5-6317-4c9d-a1f8-f785113fb252
z_m_cm = (data["floridabay"][:mˣ] - mean(motifcount, dims=2)) ./ std(motifcount, dims=2)

# ╔═╡ dcfbc87f-8d38-4c61-8ae6-a779562d8e51
z_m_a = data["floridabay"][:z_m_a]

# ╔═╡ 990a160a-90ca-43b8-bb94-52173840e5eb
z_m_S = data["floridabay"][:z_m_S][:,end]

# ╔═╡ 78e1f680-7a12-456c-bf28-cbb1adfcd190
begin 
	plot(1:13, z_m_a, label="analytical method", legendposition=:bottomleft)
	plot!(1:13, z_m_S, label="Sample, grand canonical ensemble")
	plot!(1:13, z_m_cm, label="Sample, microcanonical ensemble (configuration model)", color=:black)
	title!("method compairison, 1000 samples")
	plot!(1:13, -2*ones(13), label="", line=:dash, color=:red)
	plot!(1:13, 2*ones(13), label="", line=:dash, color=:red)
	xticks!(1:13)
	xlabel!("motif number")
	xlims!(0,14)
	ylabel!("z-score")
	ylims!(-20,35)
end

# ╔═╡ c5286810-320f-4a17-8f6c-0e7b0a8b4983
begin
	motnumber = 12
plot(histogram(motifcount[motnumber,:], normalize=:pdf, label="microcanonical"),
	histogram(data["floridabay"][:S_m][motnumber,:], normalize=:pdf, label="canonical"))
end

# ╔═╡ 5b5cc112-9fff-4f48-8ffe-3b187ec94fb8


# ╔═╡ ebb93cd3-3c58-423b-bbfa-9bfaa87055f7
net = nx_configuration_model_sample(data["floridabay"][:network])

# ╔═╡ cef40891-674d-4786-92d6-be102646fb16


# ╔═╡ e73ae349-93b4-404c-862d-671862f10342
nx_configuration_model_sample(data["floridabay"][:network])

# ╔═╡ 77a18880-1287-4a8e-9f03-5aff00438aff
begin
	# generate the network as such
	gg = nx.directed_configuration_model(indegree(data["floridabay"][:network]),
								outdegree(data["floridabay"][:network]))
	# remove parallel edges
	gg = nx.DiGraph(gg)
	# remove self-edges
	gg.remove_edges_from(nx.selfloop_edges(gg))
	gg.out_degree()
	#res = scipyCSC_to_julia(nx.adjacency_matrix(gg))
	#Array(nx.adjacency_matrix(gg))
	#gg.number_of_selfloops()
	
end

# ╔═╡ 486c44b1-2614-45ed-a858-674e706ad92e
@btime scipyCSC_to_julia($(nx.adjacency_matrix(gg)))

# ╔═╡ 7a1fe1dd-d9b8-487f-bfcc-e182f12248b7
function nx_motifcounts(net, )
	
end

# ╔═╡ 2aaaf008-6dda-46c9-9a0e-a55bf67b7426
Graphs.outdegree(data["floridabay"][:network])

# ╔═╡ 6c118bc1-8e18-4b41-bbb0-8783f8ba1784
data["floridabay"]

# ╔═╡ Cell order:
# ╟─ce77d578-328b-11ed-34ef-7b8e9f47f3c5
# ╟─a3bc7cac-3128-46e1-a67f-f3557c2e757d
# ╠═34ee4b6a-232a-4154-ab42-050efab16424
# ╠═26ea1b33-8baf-405d-b962-9fe84f37ca5b
# ╠═c2fa3030-6177-401e-ae89-ad163e7b762d
# ╠═a7b87189-0438-40ca-b7ee-9f1115f6dfb2
# ╠═17eea405-8a40-4bd3-926c-6fd5774ecf3b
# ╠═a9670da9-7acf-430a-889f-4989426ece0e
# ╠═7df8ba8e-f7a2-41b8-a97a-07f697241380
# ╠═54565e0a-1a78-4abe-9f00-89d235b7763e
# ╠═38be7b27-2190-4903-8a9a-03d7f1adc5e4
# ╠═eb17d996-1ff2-4bc2-b61f-cd4468aeceb7
# ╠═21e933ee-8492-4963-b715-53df84e259e6
# ╠═4bf64fd4-774f-4780-a5e3-248ae45f9478
# ╠═3f187e60-14f3-463c-b519-7828cc82d689
# ╠═dfd00eea-7466-4de1-b420-d80655cd17a1
# ╠═03c2cda0-083d-4a2f-8370-080f3f0529f1
# ╠═aca6f625-623b-45fb-be5e-19e6e513fc05
# ╠═f2c806b7-7d28-411b-9d7f-e464be4b592e
# ╠═20cb867f-54b2-477a-a696-1550eb19606f
# ╠═2a0a550a-42b6-4fcb-821f-4de8bbb63de5
# ╠═37e78aeb-ec2f-44f8-a5c8-5d1130835ec2
# ╟─81e2adc6-75dd-4d48-9259-9a87902e7e8e
# ╠═85ac94a3-160f-43ca-9a2a-bbf59eb6aede
# ╟─e096d53b-64f2-469e-b05b-006565e0b06c
# ╠═c13394e0-7a86-4ab1-9504-25d60aa1b2bb
# ╠═5cdf0f9e-f65d-48bb-9da7-73963b1ba2b7
# ╠═e0a47426-0f27-41e3-bc77-9b66e0634d8d
# ╟─a1d7eaba-03a7-4e55-a546-535732b4f3c5
# ╟─3a5ebf8c-40af-44db-8f96-0d3ba5e892a7
# ╠═7274e006-bd84-4702-b07e-b5669234359e
# ╟─b7077a41-9e9f-4584-bd9e-fdb3a1cbb666
# ╟─73fee4ae-551f-42f9-8783-177c5bbbf96b
# ╟─141b21f5-6373-4205-9ea5-822b6b446137
# ╠═de98df21-9662-4647-9c16-9b46a7c208c1
# ╟─627b455c-b196-47f8-843e-7806dee098a2
# ╠═ba8ae8d1-95ab-4dab-bef8-954e70d5fb57
# ╠═52ccfad6-eba6-44d7-8dfe-7d5fde65da46
# ╠═8c74de6e-21f7-4f72-b03c-1206f430ec39
# ╠═e550f8ed-0bdd-42ce-9196-2fe35bb33c72
# ╠═6275e711-9651-4f66-ba94-c1ad9b6354fd
# ╠═24c2322a-8f55-4311-a807-00c8a98ec854
# ╠═486c44b1-2614-45ed-a858-674e706ad92e
# ╠═70f953dc-592b-46a1-bd36-3c56e2153d52
# ╠═c664c215-e095-4fc7-b0ae-606e5854a16c
# ╠═14aedfe8-7901-4254-8836-400bce1b3ed2
# ╠═bf025884-c1b2-4327-9c27-a057b80c681c
# ╠═5405acb5-3ff2-40ad-ba6e-c388b46f5a81
# ╠═975768bd-aa16-49a5-8b49-3894a8f319c5
# ╠═df198107-6066-4220-8c40-1b92c3c1cdd8
# ╠═a61a1c73-784b-4268-bb4a-3b31ab72d1f1
# ╟─629afdfa-8e68-4ff6-ae44-9a115445d79d
# ╠═72b9f000-832b-485d-a10d-35339c7e9cbb
# ╠═1bebace5-6317-4c9d-a1f8-f785113fb252
# ╠═dcfbc87f-8d38-4c61-8ae6-a779562d8e51
# ╠═990a160a-90ca-43b8-bb94-52173840e5eb
# ╠═78e1f680-7a12-456c-bf28-cbb1adfcd190
# ╠═c5286810-320f-4a17-8f6c-0e7b0a8b4983
# ╠═5b5cc112-9fff-4f48-8ffe-3b187ec94fb8
# ╠═ebb93cd3-3c58-423b-bbfa-9bfaa87055f7
# ╠═cef40891-674d-4786-92d6-be102646fb16
# ╠═e73ae349-93b4-404c-862d-671862f10342
# ╠═77a18880-1287-4a8e-9f03-5aff00438aff
# ╠═7a1fe1dd-d9b8-487f-bfcc-e182f12248b7
# ╠═2aaaf008-6dda-46c9-9a0e-a55bf67b7426
# ╠═6c118bc1-8e18-4b41-bbb0-8783f8ba1784
