##################################################################################
# CReM_demo.jl
#
# This file contains some demos for the CReM model
# CReM: Conditional Reconstruction Model
##################################################################################

# setup for the CReM model
begin
    using Revise
    using BenchmarkTools
    # load up the module
    using MaxEntropyGraphs
    using Graphs
    using SimpleWeightedGraphs

    ## testing graph (single repreated pair)
    ## ______________________________________
    sources =       [1,1,1,2,3,3,4,4,5,6];
    destinations =  [2,2,3,3,4,5,6,7,7,5];
    weights =       [1,2,3,4,5,1,2,3,4,5];
    G = SimpleWeightedGraph(sources, destinations, float.(weights))
    # pythonsolutions
    α_python = [0.504265218660552072549307922599837183952331542968750000000000,0.504265218660552072549307922599837183952331542968750000000000,3.003999684368159339697967880056239664554595947265625000000000,1.293245515067872775105684013396967202425003051757812500000000,1.293245515067872775105684013396967202425003051757812500000000,0.504265218660552072549307922599837183952331542968750000000000,0.504265218660552072549307922599837183952331542968750000000000]
    θ_python = [0.185537669534768495660514986411726567894220352172851562500000,0.132939214624878454529266491590533405542373657226562500000000,0.160958338216350360649897766052163206040859222412109375000000,0.150129605521063469453224570315796881914138793945312500000000,0.150129605521063469453224570315796881914138793945312500000000,0.132939214624878454529266491590533405542373657226562500000000,0.132939214624878454529266491590533405542373657226562500000000]

    # obtain the degree and strength sequences
    d = degree(G)
    s = MaxEntropyGraphs.strength(G)
    @info "N: $(Graphs.nv(G)), E: $(Graphs.ne(G))"
    @info "d: $(d)"
    @info "s: $(s)"
    nothing
end

# Model generation
begin
    #
end

