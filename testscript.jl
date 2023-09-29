
using MaxEntropyGraphs
using Graphs
using BenchmarkTools

#MaxEntropyGraphs.produce_squartini_dbcm_data("./data/computed_results/DBCM_result_NS22.jld", "./data/networks")
begin
    G = MaxEntropyGraphs.Graphs.smallgraph(:karate)
    model = UBCM(G)
    solve_model!(model)
    set_Ĝ!(model)
    wedges(G), wedges(model), wedges(MaxEntropyGraphs.Graphs.adjacency_matrix(G))
    squares(G), squares(MaxEntropyGraphs.Graphs.adjacency_matrix(G)), squares(model.Ĝ)

    @btime squares(G)
    @btime squares(model.Ĝ)
end

MaxEntropyGraphs.Graphs.grif
begin
    # testing the experimental algorithms for counting subgraph isomorphisms on the karate graph
    G = MaxEntropyGraphs.Graphs.smallgraph(:karate)
    # wedges
    G_V = MaxEntropyGraphs.Graphs.complete_graph(3); MaxEntropyGraphs.Graphs.rem_edge!(G_V, 1, 3) 
    # triangles
    G_Δ = MaxEntropyGraphs.Graphs.complete_graph(3)
    # squares
    G_□ = MaxEntropyGraphs.Graphs.complete_graph(4)
    MaxEntropyGraphs.Graphs.rem_edge!(G_□, 1, 3)
    MaxEntropyGraphs.Graphs.rem_edge!(G_□, 2, 4)

    V_comp = MaxEntropyGraphs.Graphs.Experimental.count_subgraphisomorph(G, G_V)
    @info "Number of subgraph V isomorphisms: $(V_comp) <> expected: 528 (ratio: $(V_comp / 22))"

    Δ_comp = MaxEntropyGraphs.Graphs.Experimental.count_subgraphisomorph(G, G_Δ)
    @info "Number of subgraph triangle isomorphisms: $(Δ_comp) <> expected: 45 (ratio: $(Δ_comp / 45))"
    Δ_comp = MaxEntropyGraphs.Graphs.Experimental.count_isomorph(G, G_Δ)
    @info "Number of subgraph triangle isomorphisms: $(Δ_comp) <> expected: 45 (ratio: $(Δ_comp / 45))"
    Δ_comp = MaxEntropyGraphs.Graphs.Experimental.has_induced_subgraphisomorph(G, G_Δ)
    @info "Number of subgraph triangle isomorphisms: $(Δ_comp) <> expected: 45 (ratio: $(Δ_comp / 45))"

    □_comp =  MaxEntropyGraphs.Graphs.Experimental.count_subgraphisomorph(G, G_□)
    @info "Number of subgraph square isomorphisms: $(□_comp) <> expected: 154 (ratio: $(□_comp / 154))"
    □_comp =  MaxEntropyGraphs.Graphs.Experimental.count_isomorph(G, G_□)
    @info "Number of subgraph square isomorphisms: $(□_comp) <> expected: 154 (ratio: $(□_comp / 154))"
    
end

begin 
    # South African companies graph (wedges: 13, triangles:0 (bipartite), squares: 4)
    G_af = MaxEntropyGraphs.Graphs.SimpleGraph(MaxEntropyGraphs.Graphs.SimpleEdge.([(1 ,1 );
    (1, 2 );
    (1 ,3 );
    (2 ,1 );
    (2 ,3 );
    (3 ,4 );
    (3 ,3 );
    (4 ,1 );
    (4 ,3 );
    (5 ,5 );
    (5 ,2 );
    (6 ,1 );
    (6 ,2 )]))
    wedges(G_af)
    squares(MaxEntropyGraphs.Graphs.adjacency_matrix(G_af)), squares(G_af)

end

set_σ!(model)

E_squares = squares(model)
@btime σ_squares = σₓ(model, squares)
@btime σ_squares = σₓ(model, squares, gradient_method=:ForwardDiff)
z_squares = (squares(G) - E_squares) / σ_squares
begin 
    # South African companies graph (wedges: 13, triangles:0 (bipartite), squares: 4)
    G_grid = MaxEntropyGraphs.Graphs.grid([2,3])
    wedges(G_grid)
    squares(MaxEntropyGraphs.Graphs.adjacency_matrix(G_grid))
    MaxEntropyGraphs.Graphs.Experimental.count_subgraphisomorph(G_grid, G_□) / 8

    MaxEntropyGraphs.squares(MaxEntropyGraphs.Graphs.adjacency_matrix(G_grid)), MaxEntropyGraphs.squares(G_grid)

end
squares(MaxEntropyGraphs.Graphs.adjacency_matrix(G_□))/24
triangles(MaxEntropyGraphs.Graphs.adjacency_matrix(G_grid))

@btime Graphs.Experimental.count_subgraphisomorph(G, G_Δ)


# Example vector
v = [1, 2, 3]

# Generate all pairwise combinations
pairs = Iterators.filter(pair -> pair[1] != pair[2], Iterators.product(v, v))
collect(pairs)
sum(Graphs.degree(G) .* ( Graphs.degree(G) .- 1) ./ 2)

# number of triangle in a graph G
sum(triangles(G)) / 3


join(["M$(i)" for i in 1:13], ", ")

# read in an exiting network
pwd()
import Graphs

path = joinpath(pwd(), "data/networks/stmarkseagrass_directed.lg")
print(join(["($(e.src), $(e.dst))" for e in collect(Graphs.edges(Graphs.loadgraph(path)))], ", "))
Graphs.loadgraph(path) == MaxEntropyGraphs.stmarks()



begin
    # exact approach
    G = chesapeakebay()
    A = MaxEntropyGraphs.Graphs.adjacency_matrix(G)
    motifs_observed = [@eval begin $(f)(A) end for f in MaxEntropyGraphs.directed_graph_motif_function_names]
    model = DBCM(G)
    solve_model!(model)
    set_Ĝ!(model)
    set_σ!(model)
    motifs_expected = [@eval begin $(f)(model) end for f in MaxEntropyGraphs.directed_graph_motif_function_names]
    motifs_std = [@eval begin  σₓ(model, $(f), gradient_method=:ForwardDiff) end for f in MaxEntropyGraphs.directed_graph_motif_function_names]
    motifs_z = (motifs_observed .- motifs_expected) ./ motifs_std
    @info "Motifs observed: $(motifs_observed)"
    @info "Motifs expected: $(motifs_expected)"
    @info "Motifs std: $(motifs_std)"
    @info "Motifs z: $(motifs_z)"
end

begin
    import Statistics: mean, std
    # sampling approach (100 samples, converted to adjacency matrix)
    S = MaxEntropyGraphs.Graphs.adjacency_matrix.(rand(model, 1000))
    # compute the motifs
    motif_counts = hcat(map(s -> [@eval begin $(f)($s) end for f in MaxEntropyGraphs.directed_graph_motif_function_names], S)...)
    # compute the mean and standard deviation
    motifs_mean = reshape(mean(motif_counts, dims=2),:)
    motifs_std = reshape(std(motif_counts, dims=2),:)
    motifs_z_s = (motifs_observed .- motifs_mean) ./ motifs_std
    @info "Motifs observed: $(motifs_observed)"
    @info "Motifs mean: $(motifs_mean)"
    @info "Motifs std: $(motifs_std)"
    @info "Motifs z: $(motifs_z_s)"

    @info "Differences: $(motifs_z .- motifs_z_s)"
    @info "Differences relative: $((motifs_z .- motifs_z_s) ./ motifs_z)"
end

begin
    # exact approach
    G = littlerock()
    A = MaxEntropyGraphs.Graphs.adjacency_matrix(G)
    motifs_observed = [@eval begin $(f)(A) end for f in MaxEntropyGraphs.directed_graph_motif_function_names]
    model = DBCM(G)
    solve_model!(model)
    set_Ĝ!(model)
    set_σ!(model)
    motifs_expected = [@eval begin $(f)(model) end for f in MaxEntropyGraphs.directed_graph_motif_function_names]
    motifs_std = [@eval begin  σₓ(model, $(f), gradient_method=:ForwardDiff) end for f in MaxEntropyGraphs.directed_graph_motif_function_names]
    motifs_z = (motifs_observed .- motifs_expected) ./ motifs_std
    @info "Motifs observed: $(motifs_observed)"
    @info "Motifs expected: $(motifs_expected)"
    @info "Motifs std: $(motifs_std)"
    @info "Motifs z: $(motifs_z)"
end



