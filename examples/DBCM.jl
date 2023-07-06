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
    using JLD2
	using StatsBase
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

# ╔═╡ Cell order:
# ╟─ce77d578-328b-11ed-34ef-7b8e9f47f3c5
# ╠═34ee4b6a-232a-4154-ab42-050efab16424
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
