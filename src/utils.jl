#############################################################################
# utils.jl
#
# This file contains utility functions for the MaxEntropyGraphs.jl package
#############################################################################


"""
    np_unique_clone(x::Vector; sorted::Bool=false)

Julia replication of the numpy.unique(a, return_counts=True, return_index=True, return_inverse=True) function from Python.

Returns a tuple of:
 - vector of unique values in x
 - vector of indices of the first occurence of each unique value in x. Follows the same order as the unique values.
 - vector of inverse indices of the original data in the unique values
 - vector of the counts of the unique values in x. Follows the same order as the unique values.

 If sorted is true, the unique values are sorted by size and the other vectors are sorted accordingly.

# Examples     
```jldoctest
julia> x = [1;2;2;4;1];

julia> np_unique_clone(x)
([1, 2, 4], [1, 2, 4], [1, 2, 2, 3, 1], [2, 2, 1])

julia> x = [10;9;9;8];

julia> np_unique_clone(x, sorted=true)
([8, 9, 10], [4, 2, 1], [3, 2, 2, 1], [1, 2, 1])
```
"""
function np_unique_clone(x::Vector; sorted::Bool=true)
    T = eltype(x)
    unique_x =  Vector{T}()         # unique values
    seen =      Set{T}()            # unique values set
    index =     Dict{T,Int}()       # first occurence of unique values (index)
    reverse_index = Dict{T,Int}()   # position of value in unique values
    counts =    Dict{T, Int}()      # unique values counts
    
    for i in eachindex(x)
        if !in(x[i], seen)
            push!(seen, x[i])
            push!(unique_x, x[i])
            index[x[i]] = i
            reverse_index[x[i]] = length(unique_x) #
            counts[x[i]] = 1
        else
            counts[x[i]] += 1
        end

    end

    first_occ = [index[v] for v in unique_x]
    inverse_index = [reverse_index[v] for v in x]
    freqs = [counts[v] for v in unique_x]
    
    if sorted
        # sorted indices
        inds = sortperm(unique_x)
        # sort vectors
        unique_x = unique_x[inds]
        first_occ = first_occ[inds]
        freqs = freqs[inds]
        
        # Create a map from old indices to new indices
        new_indices_map = Dict(inds[i] => i for i in 1:length(inds))
        
        # Update inverse_index using the map
        inverse_index = [new_indices_map[i] for i in inverse_index]
    end

    return unique_x, first_occ, inverse_index, freqs
end


"""
    log_nan(x::T)

Same as `log(x)`, but returns `NaN` if `x <= 0`. Inspired by `NaNMath.jl` and https://github.com/JuliaMath/NaNMath.jl/issues/63. This methods is prefered
over the ones from `NaNMath.jl` version because it does not require a foreign call expression to be evaluated, hence autodiff methods can be used with this.

# Examples     
```jldoctest
julia> MaxEntropyGraphs.log_nan(10.)
2.302585092994046

julia> MaxEntropyGraphs.log_nan(-10.)
NaN
```
"""
function log_nan(x::T)::T where {T<:Real}
    x <= T(0) && return T(NaN)
    return log(x)
end





#     DBCM_analysis

# Compute the z-scores etc. for all motifs and the degrees for a `SimpleDiGraph`. Returns a Dict for storage of the computed results

# G: the network
# N_min: minimum sample length used for computing metrics
# N_max: maximum sample length used for computing metrics
# n_sample_lengths: number of values in the domain [N_min, N_max]
# """
# function DBCM_analysis(  G::T;
#                                 N_min::Int=100, 
#                                 N_max::Int=10000, 
#                                 n_sample_lengths::Int=3,
#                                 subsamples::Vector{Int64}=round.(Int,exp10.(range(log10(N_min),log10(N_max), length=n_sample_lengths))), kwargs...) where T<:Graphs.SimpleDiGraph
#     @info "$(round(now(), Minute)) - Started DBCM motif analysis"
#     NP = PyCall.pyimport("NEMtropy")
#     G_nem =  NP.DirectedGraph(degree_sequence=vcat(Graphs.outdegree(G), Graphs.indegree(G)))
#     G_nem.solve_tool(model="dcm_exp", method="fixed-point", initial_guess="degrees", max_steps=3000)
#     if abs(G_nem.error) > 1e-6
#         @warn "Method did not converge"
#     end
#     # generate the model
#     model = DBCM(G_nem.x, G_nem.y)
#     @assert Graphs.indegree(model)  ≈ Graphs.indegree(G)
#     @assert Graphs.outdegree(model) ≈ Graphs.outdegree(G)
#     # generate the sample
#     @info "$(round(now(), Minute)) - Generating sample"
#     S = [rand(model) for _ in 1:N_max]

#     #################################
#     # motif part
#     #################################

#     # compute motif data
#     @info "$(round(now(), Minute)) - Computing observed motifs in the observed network"
#     mˣ = motifs(G)
#     @info "$(round(now(), Minute)) - Computing expected motifs for the DBCM model"
#     m̂  = motifs(model)
#     σ̂_m̂  = Vector{eltype(model.G)}(undef, length(m̂))
#     for i = 1:length(m̂)
#         @info "$(round(now(), Minute)) - Computing standard deviation for motif $(i)"
#         σ̂_m̂[i] = σˣ(DBCM_motif_functions[i], model)
#     end
#     # compute z-score (Squartini)
#     z_m_a = (mˣ - m̂) ./ σ̂_m̂  
#     @info "$(round(now(), Minute)) - Computing expected motifs for the sample"
#     #S_m = hcat(motifs.(S,full=true)...); # computed values from sample
#     S_m = zeros(13, length(S))
#     Threads.@threads for i in eachindex(S)
#         S_m[:,i] .= motifs(S[i], full=true)
#     end
#     m̂_S =   hcat(map(n -> reshape(mean(S_m[:,1:n], dims=2),:), subsamples)...)
#     σ̂_m̂_S = hcat(map(n -> reshape( std(S_m[:,1:n], dims=2),:), subsamples)...)
#     z_m_S = (mˣ .- m̂_S) ./ σ̂_m̂_S

#     #################################       
#     # degree part
#     #################################
#     # compute degree sequence
#     @info "$(round(now(), Minute)) - Computing degrees in the observed network"
#     d_inˣ, d_outˣ = Graphs.indegree(G), Graphs.outdegree(G)
#     @info "$(round(now(), Minute)) - Computing expected degrees for the DBCM model"
#     d̂_in, d̂_out = Graphs.indegree(model), Graphs.outdegree(model)
#     @info "$(round(now(), Minute)) - Computing standard deviations for the degrees for the DBCM model"
#     σ̂_d̂_in, σ̂_d̂_out = map(j -> σˣ(m -> Graphs.indegree(m, j), model), 1:length(model)), map(j -> σˣ(m -> Graphs.outdegree(m, j), model), 1:length(model))
#     # compute degree z-score (Squartini)
#     z_d_in_sq, z_d_out_sq = (d_inˣ - d̂_in) ./ σ̂_d̂_in, (d_outˣ - d̂_out) ./ σ̂_d̂_out
#     @info "$(round(now(), Minute)) - Computing distributions for degree sequences"
#     d_in_dist, d_out_dist = indegree_dist(model), outdegree_dist(model)
#     z_d_in_dist, z_d_out_dist = (d_inˣ - mean.(d_in_dist)) ./ std.(d_in_dist), (d_outˣ - mean.(d_out_dist)) ./ std.(d_out_dist)

#     # compute data for the sample
#     @info "$(round(now(), Minute)) - Computing degree sequences for the sample"
#     d_in_S, d_out_S = hcat(Graphs.indegree.(S)...), hcat(Graphs.outdegree.(S)...)
#     d̂_in_S, d̂_out_S = hcat(map(n -> reshape(mean(d_in_S[:,1:n], dims=2),:), subsamples)...), hcat(map(n -> reshape(mean(d_out_S[:,1:n], dims=2),:), subsamples)...)
#     σ̂_d_in_S, σ̂_d_out_S = hcat(map(n -> reshape(std( d_in_S[:,1:n], dims=2),:), subsamples)...), hcat(map(n -> reshape( std(d_out_S[:,1:n], dims=2),:), subsamples)...)
#     # compute degree z-score (sample)
#     z_d_in_S, z_d_out_S = (d_inˣ .- d̂_in_S) ./ σ̂_d_in_S, (d_outˣ .- d̂_out_S) ./ σ̂_d_out_S
    

#     @info "$(round(now(), Minute)) - Finished"
#     return Dict(:network => G,
#                 :model => model,
#                 :error => G_nem.error,
#                 # motif information
#                 :mˣ => mˣ,          # observed
#                 :m̂ => m̂,            # expected squartini
#                 :σ̂_m̂ => σ̂_m̂,        # standard deviation squartini
#                 :z_m_a => z_m_a,    # z_motif squartini
#                 :S_m => S_m,        # sample data
#                 :m̂_S => m̂_S,        # expected motifs in sample
#                 :σ̂_m̂_S => σ̂_m̂_S,    # standard deviation motifs sample
#                 :z_m_S => z_m_S,    # z_motif sample
#                 # in/outdegree information
#                 :d_inˣ => d_inˣ,                # observed
#                 :d_outˣ => d_outˣ,              # observed
#                 :d̂_in => d̂_in,                  # expected squartini
#                 :d̂_out => d̂_out,                # expected squartini
#                 :σ̂_d̂_in => σ̂_d̂_in,              # standard deviation squartini
#                 :σ̂_d̂_out => σ̂_d̂_out,            # standard deviation squartini
#                 :z_d_in_sq => z_d_in_sq,        # z_degree squartini
#                 :z_d_out_sq => z_d_out_sq,      # z_degree squartini
#                 :d̂_in_S => d̂_in_S,              # expected sample
#                 :d̂_out_S => d̂_out_S,            # expected sample
#                 :σ̂_d_in_S => σ̂_d_in_S,          # standard deviation sample
#                 :σ̂_d_out_S => σ̂_d_out_S,        # standard deviation sample
#                 :z_d_in_S => z_d_in_S,          # z_degree sample
#                 :z_d_out_S => z_d_out_S,        # z_degree sample
#                 :z_d_in_dist => z_d_in_dist,    # z_degree distribution (analytical PoissonBinomial)
#                 :z_d_out_dist => z_d_out_dist,  # z_degree distribution (analytical PoissonBinomial)
#                 :d_in_dist => d_in_dist,        # distribution (analytical PoissonBinomial)
#                 :d_out_dist => d_out_dist,      # distribution (analytical PoissonBinomial)
#                 :d_in_S => d_in_S,              # indegree sample
#                 :d_out_S => d_out_S             # indegree sample
#                 ) 
                
# end


# """
#     write_result(outfile::String, label::Union{String, Symbol}, data)

# write out data to jld file. Checks for existance of the file and appends if it exists.

# outfile::String - path to output file
# label - label for the data in the file
# data - actual data to write to the file
# """
# function write_result(outfile::String, label::Union{String, SubString{String}, Symbol}, data)
#     outfile = endswith(outfile, ".jld") ? outfile : outfile * ".jld"
#     # append or create file
#     JLD2.jldopen(outfile, isfile(outfile) ? "r+" : "w") do file
#         write(file, String(label), data)
#     end
# end

# """
#     produce_squartini_dbcm_data

# utility function to reproduce the data from the original 2011 Squartini paper (https://arxiv.org/abs/1103.0701)
# """
# function produce_squartini_dbcm_data(output = "./data/computed_results/DBCM_result_more.jld",
#                                      netdata = "./data/networks")
#     for network in filter(x-> occursin("_directed", x), joinpath.(netdata, readdir(netdata)))
#         @info "working on $(network)"
#         # reference name for storage
#         refname = split(split(network,"/")[end],"_")[1]
#         @info refname
#         # load network
#         G = Graphs.loadgraph(network)
#         # compute motifs
#         res = DBCM_analysis(G)
#         # write out results
#         write_result(output, refname, res)
#     end
# end

# function readpajek(f::String; is_directed::Bool=true)
#     G = is_directed ? Graphs.SimpleDiGraph() : Graphs.SimpleGraph()
#     nv = r"\*vertices\s(\d+)"
#     arcs = r"\*arcs"
#     arc = r"(\d+)\s+(\d+)"
#     arcing, finished = false, false
#     for line in readlines(f)
#         if !isnothing(match(nv, line))
#             N = parse(Int,(match(nv, line).captures[1]))
#             Graphs.nv(G) == 0 && Graphs.add_vertices!(G, N) 
#         end
#         if !isnothing(match(arcs, line))
#             arcing = true
#             continue
#         end

#         if arcing 
#             # check empty line
#             if isempty(line)
#                 arcing = false
#                 return G
#             end

#             src, dst = parse.(Int, match(arc, line).captures)
#             Graphs.add_edge!(G, src, dst)
#         end

#     end
    
# end