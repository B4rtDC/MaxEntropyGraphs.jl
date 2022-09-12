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
The function below is a the helper function. It generates the model, computes the degree sequences and the motifs. This is done for the observed network, according to the Squartini method and based on the sample.

"""

# ╔═╡ c13394e0-7a86-4ab1-9504-25d60aa1b2bb


# ╔═╡ 5cdf0f9e-f65d-48bb-9da7-73963b1ba2b7
"""
    DBCM_analysis

Compute the z-scores etc. for all motifs and the degrees for a `SimpleDiGraph`. Returns a Dict for storage of the computed results

* G: the network
* N_min: minimum sample length used for computing metrics
* N_max: maximum sample length used for computing metrics
* n_sample_lengths: number of values in the domain [N_min, N_max]
"""
function DBCM_analysis(  G::T;
                                N_min::Int=100, 
                                N_max::Int=10000, 
                                n_sample_lengths::Int=3,
                                subsamples::Vector{Int64}=round.(Int,exp10.(range(log10(N_min),log10(N_max), length=n_sample_lengths))), kwargs...) where T<:Graphs.SimpleDiGraph
    @info "$(round(now(), Minute)) - Started DBCM motif analysis with the following setting:\n$(kwargs)"
    NP = PyCall.pyimport("NEMtropy")
    G_nem =  NP.DirectedGraph(degree_sequence=vcat(Graphs.outdegree(G), Graphs.indegree(G)))
    G_nem.solve_tool(model="dcm_exp", method="fixed-point", initial_guess="degrees", max_steps=3000)
    if abs(G_nem.error) < 1e-6
        @warn "Method did not converge"
    end
    # generate the model
    model = DBCM(G_nem.x, G_nem.y)
    @assert indegree(model)  ≈ Graphs.indegree(G)
    @assert outdegree(model) ≈ Graphs.outdegree(G)
    # generate the sample
    @info "$(round(now(), Minute)) - Generating sample"
    S = [rand(model) for _ in 1:N_max]

    #################################
    # motif part
    #################################

    # compute motif data
    @info "$(round(now(), Minute)) - Computing observed motifs in the observed network"
    mˣ = motifs(G)
    @info "$(round(now(), Minute)) - Computing expected motifs for the DBCM model"
    m̂  = motifs(model)
    σ̂_m̂  = Vector{eltype(model.G)}(undef, length(m̂))
    for i = 1:length(m̂)
        @info "$(round(now(), Minute)) - Computing standard deviation for motif $(i)"
        σ̂_m̂[i] = σˣ(DBCM_motif_functions[i], model)
    end
    # compute z-score (Squartini)
    z_m_a = (mˣ - m̂) ./ σ̂_m̂  
    @info "$(round(now(), Minute)) - Computing expected motifs for the sample"
    #S_m = hcat(motifs.(S,full=true)...); # computed values from sample
    S_m = zeros(13, length(S))
    Threads.@threads for i in eachindex(S)
        S_m[:,i] .= motifs(S[i], full=true)
    end
    m̂_S =   hcat(map(n -> reshape(mean(S_m[:,1:n], dims=2),:), subsamples)...)
    σ̂_m̂_S = hcat(map(n -> reshape( std(S_m[:,1:n], dims=2),:), subsamples)...)
    z_m_S = (mˣ .- m̂_S) ./ σ̂_m̂_S

    #################################       
    # degree part
    #################################
    # compute degree sequence
    @info "$(round(now(), Minute)) - Computing degrees in the observed network"
    d_inˣ, d_outˣ = Graphs.indegree(G), Graphs.outdegree(G)
    @info "$(round(now(), Minute)) - Computing expected degrees for the DBCM model"
    d̂_in, d̂_out = indegree(model), outdegree(model)
    @info "$(round(now(), Minute)) - Computing standard deviations for the degrees for the DBCM model"
    σ̂_d̂_in, σ̂_d̂_out = map(j -> σˣ(m -> indegree(m, j), model), 1:length(model)), map(j -> σˣ(m -> outdegree(m, j), model), 1:length(model))
    # compute degree z-score (Squartini)
    z_d_in_sq, z_d_out_sq = (d_inˣ - d̂_in) ./ σ̂_d̂_in, (d_outˣ - d̂_out) ./ σ̂_d̂_out
    @info "$(round(now(), Minute)) - Computing distributions for degree sequences"
    d_in_dist, d_out_dist = indegree_dist(model), outdegree(model)
    z_d_in_dist, z_d_out_dist = (d_inˣ - mean.(d_in_dist)) ./ std.(d_in_dist), (d_outˣ - mean.(d_out_dist)) ./ std.(d_out_dist)

    # compute data for the sample
    @info "$(round(now(), Minute)) - Computing degree sequences for the sample"
    d_in_S, d_out_S = hcat(Graphs.indegree.(S)...), hcat(Graphs.outdegree.(S)...)
    d̂_in_S, d̂_out_S = hcat(map(n -> reshape(mean(d_in_S[:,1:n], dims=2),:), subsamples)...), hcat(map(n -> reshape(mean(d_out_S[:,1:n], dims=2),:), subsamples)...)
    σ̂_d_in_S, σ̂_d_out_S = hcat(map(n -> reshape(std( d_in_S[:,1:n], dims=2),:), subsamples)...), hcat(map(n -> reshape( std(d_out_S[:,1:n], dims=2),:), subsamples)...)
    # compute degree z-score (sample)
    z_d_in_S, z_d_out_S = (d_inˣ .- d̂_in_S) ./ σ̂_d_in_S, (d_outˣ .- d̂_out_S) ./ σ̂_d_out_S
    

    @info "$(round(now(), Minute)) - Finished"
    return Dict(:network => G,
                :model => model,
                :error => G_nem.error,
                # motif information
                :mˣ => mˣ,          # observed
                :m̂ => m̂,            # expected squartini
                :σ̂_m̂ => σ̂_m̂,        # standard deviation squartini
                :z_m_a => z_m_a,    # z_motif squartini
                :S_m => S_m,        # sample data
                :m̂_S => m̂_S,        # expected sample
                :σ̂_m̂_S => σ̂_m̂_S,    # standard deviation sample
                :z_m_S => z_m_S,    # z_motif sample
                # in/outdegree information
                :d_inˣ => d_inˣ,                # observed
                :d_outˣ => d_outˣ,              # observed
                :d̂_in => d̂_in,                  # expected squartini
                :d̂_out => d̂_out,                # expected squartini
                :σ̂_d̂_in => σ̂_d̂_in,              # standard deviation squartini
                :σ̂_d̂_out => σ̂_d̂_out,            # standard deviation squartini
                :z_d_in_sq => z_d_in_sq,        # z_degree squartini
                :z_d_out_sq => z_d_out_sq,      # z_degree squartini
                :d̂_in_S => d̂_in_S,              # expected sample
                :d̂_out_S => d̂_out_S,            # expected sample
                :σ̂_d_in_S => σ̂_d_in_S,          # standard deviation sample
                :σ̂_d_out_S => σ̂_d_out_S,        # standard deviation sample
                :z_d_in_S => z_d_in_S,          # z_degree sample
                :z_d_out_S => z_d_out_S,        # z_degree sample
                :d_in_dist => d_in_dist,        # distribution (analytical PoissonBinomial)
                :d_out_dist => d_out_dist,      # distribution (analytical PoissonBinomial)
                :z_d_in_dist => z_d_in_dist,    # z_degree distribution (analytical PoissonBinomial)
                :z_d_out_dist => z_d_out_dist,  # z_degree distribution (analytical PoissonBinomial)
                :d_in_S => d_in_S,              # indegree sample
                :d_out_S => d_out_S             # indegree sample
                ) 
                
end

# ╔═╡ f1e37b3a-e3ae-42e6-85d4-5c065d23e89a
DBCM_analysis(G)

# ╔═╡ Cell order:
# ╟─ce77d578-328b-11ed-34ef-7b8e9f47f3c5
# ╠═34ee4b6a-232a-4154-ab42-050efab16424
# ╟─81e2adc6-75dd-4d48-9259-9a87902e7e8e
# ╠═85ac94a3-160f-43ca-9a2a-bbf59eb6aede
# ╠═e096d53b-64f2-469e-b05b-006565e0b06c
# ╠═f1e37b3a-e3ae-42e6-85d4-5c065d23e89a
# ╠═c13394e0-7a86-4ab1-9504-25d60aa1b2bb
# ╟─5cdf0f9e-f65d-48bb-9da7-73963b1ba2b7
