###########################################################################################
# utils.jl
#
# This file contains the tests for the utility functions of the MaxEntropyGraphs.jl package
###########################################################################################


@testset "Utils" begin
    @testset "np_unique_clone_sorted_integer" begin
        # 1-D degree sequence from Zachary's karate club
        d = [16, 9, 10, 6, 3, 4, 4, 4, 5, 2, 3, 1, 2, 5, 2, 2, 2, 2, 2, 3, 2, 2, 2, 5, 3, 3, 2, 4, 3, 4, 4, 6, 12, 17]
        # Numpy values
        np_unique = [ 1,  2,  3,  4,  5,  6,  9, 10, 12, 16, 17]
        np_first = [11,  9,  4,  5,  8,  3,  1,  2, 32,  0, 33] .+ 1 # offset by 1 because Julia is 1-indexed
        np_inverse = [ 9,  6,  7,  5,  2,  3,  3,  3,  4,  1,  2,  0,  1,  4,  1,  1,  1, 1,  1,  2,  1,  1,  1,  4,  2,  2,  1,  3,  2,  3,  3,  5,  8, 10] .+ 1 # offset by 1 because Julia is 1-indexed
        np_counts = [ 1, 11,  6,  6,  3,  2,  1,  1,  1,  1,  1]
        # compute equivalent values in Julia
        j_unique, j_first, j_inverse, j_counts = MaxEntropyGraphs.np_unique_clone(d, sorted=true)
        # compare
        @test j_unique == np_unique
        @test j_first == np_first
        @test j_counts == np_counts
        @test np_unique[np_inverse] == d
        @test j_unique[j_inverse] == d
    end

    @testset "np_unique_clone_integer" begin
        # example from the docs
        x = [1;2;2;4;1];
        j_unique, j_first, j_inverse, j_counts = MaxEntropyGraphs.np_unique_clone(x, sorted=false)
        @test j_unique == [1;2;4]
        @test j_first == [1, 2, 4]
        @test j_inverse == [1;2;2;3;1]
        @test j_counts == [2;2;1]
        @test j_unique[j_inverse] == x
    end

    # to be extended for lists of tuples

    @testset "Graph metrics" begin
        @testset "ANND" begin
            # empty graph
            G = MaxEntropyGraphs.Graphs.SimpleGraph()
            @test ANND(G) == Float64[]
            # single node, no edges
            MaxEntropyGraphs.Graphs.add_vertex!(G)
            @test ANND(G,1) == zero(Float64)
            # standard graph
            G = MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate)
            Gd = MaxEntropyGraphs.Graphs.SimpleDiGraph(G)
            @test length(ANND(G)) == MaxEntropyGraphs.Graphs.nv(G)
            @test_logs (:warn,"The graph is directed. The degree function returns the incoming plus outgoing edges for node `i`. Consider using ANND_in or ANND_out instead.") ANND(Gd,1)
            # should give the same 
            @test ANND(G) == ANND(Gd)
        end
    end
end
