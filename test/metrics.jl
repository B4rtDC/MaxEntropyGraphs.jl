###########################################################################################
# utmetricsls.jl
#
# This file contains the tests for the metrics of the MaxEntropyGraphs.jl package
###########################################################################################

@testset "Graph metrics" begin
    @testset "Strength" begin
        @testset "undirected" begin
            for weighttype in [Float64; Float32; Float16; Int64; Int32; Int16]
                # undirected setup
                G = MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(Int, weighttype) # empty graph
                for _ in 1:4
                    MaxEntropyGraphs.Graphs.add_vertex!(G)
                end
                MaxEntropyGraphs.Graphs.add_edge!(G, 1, 2, 1.0)
                MaxEntropyGraphs.Graphs.add_edge!(G, 2, 3, 2.0)
                MaxEntropyGraphs.Graphs.add_edge!(G, 3, 4, 3.0)
                MaxEntropyGraphs.Graphs.add_edge!(G, 4, 1, 4.0)
                # testing
                @test eltype(MaxEntropyGraphs.strength(G)) == weighttype
                @test MaxEntropyGraphs.strength(G) == weighttype.([5, 3, 5, 7])
            end
        end

        @testset "directed" begin
            for weighttype in [Float64; Float32; Float16; Int64; Int32; Int16]
                # undirected setup
                G = MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedDiGraph(Int, weighttype) # empty graph
                for _ in 1:4
                    MaxEntropyGraphs.Graphs.add_vertex!(G)
                end
                MaxEntropyGraphs.Graphs.add_edge!(G, 1, 2, 1.0)
                MaxEntropyGraphs.Graphs.add_edge!(G, 2, 3, 2.0)
                MaxEntropyGraphs.Graphs.add_edge!(G, 3, 4, 3.0)
                MaxEntropyGraphs.Graphs.add_edge!(G, 4, 1, 4.0)
                # testing
                @test eltype(MaxEntropyGraphs.strength(G)) == weighttype
                @test_throws DomainError MaxEntropyGraphs.strength(G, dir=:invalid_dir)
                @test MaxEntropyGraphs.outstrength(G) == weighttype.([1, 2, 3, 4])
                @test MaxEntropyGraphs.instrength(G) == weighttype.([4, 1, 2, 3])
                @test MaxEntropyGraphs.strength(G, dir=:both) == weighttype.([5, 3, 5, 7])
            end
        end
    end

    @testset "ANND" begin
        ## working with a graph
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
        @test_throws ArgumentError ANND(Gd,1)
        @test_throws ArgumentError ANND(Gd)
        # should give the same 
        @test ANND(G) == ANND_in(Gd)
        @test ANND(G) == ANND_out(Gd)

        ## working with an adjacency matrix
        A = MaxEntropyGraphs.Graphs.adjacency_matrix(G)
        Ad = MaxEntropyGraphs.Graphs.adjacency_matrix(Gd)
        @test_throws DimensionMismatch ANND(A[:,1:end-1])
        @test_throws DimensionMismatch ANND(A[:,1:end-1],1)
        @test_throws DimensionMismatch ANND_in(A[:,1:end-1])
        @test_throws DimensionMismatch ANND_in(A[:,1:end-1], 1)
        @test_throws DimensionMismatch ANND_out(A[:,1:end-1])
        @test_throws DimensionMismatch ANND_out(A[:,1:end-1], 1)
        B = copy(A); B[1,2] = 1; B[2,1] = 0
        @test_throws ArgumentError ANND(B)
        @test_throws ArgumentError ANND(B,1)
        # test equality with graph
        @test ANND(A) == ANND(G)
        @test ANND(A) == ANND_in(Ad)
        @test ANND(A) == ANND_out(Ad)

        ## working with a model
        # UBCM
        model = MaxEntropyGraphs.UBCM(MaxEntropyGraphs.Graphs.smallgraph(:karate))
        solve_model!(model)
        @test_throws ArgumentError ANND(model)
        set_Ĝ!(model)
        @test ANND(model) == ANND(model.Ĝ)
        # DBCM
        model = MaxEntropyGraphs.DBCM(MaxEntropyGraphs.maspalomas())
        solve_model!(model)
        @test_throws ArgumentError ANND_in(model)
        @test_throws ArgumentError ANND_out(model)
        set_Ĝ!(model)
        @test ANND_in(model) == ANND_in(model.Ĝ)
        @test ANND_out(model) == ANND_out(model.Ĝ)
    end

    @testset "wedges" begin
        # setup
        G = MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate)
        A = MaxEntropyGraphs.Graphs.adjacency_matrix(G)
        # coherence checks
        @test_throws DimensionMismatch wedges(A[:,1:end-1])

        # equality
        @test wedges(G) == wedges(A)
        
        model = MaxEntropyGraphs.UBCM(G)
        solve_model!(model)
        @test_throws ArgumentError wedges(model)
        set_Ĝ!(model)
        @test wedges(model) == wedges(model.Ĝ)
        
        
    end

    @testset "triangles" begin
        # setup undirected
        G = MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate)
        A = MaxEntropyGraphs.Graphs.adjacency_matrix(G)
        # coherence checks
        @test_throws DimensionMismatch triangles(A[:,1:end-1])
        # equality/equivalence
        @test MaxEntropyGraphs.triangles(G) == MaxEntropyGraphs.triangles(A)
        
        # combination with model
        model = MaxEntropyGraphs.UBCM(G)
        @test_throws ArgumentError MaxEntropyGraphs.triangles(model)
        MaxEntropyGraphs.solve_model!(model)
        MaxEntropyGraphs.set_Ĝ!(model)
        @test triangles(model) == triangles(model.Ĝ)

        ## setup directed
        A = zeros(Bool, 5, 5); A[1,2] = true;
        @test_throws ArgumentError MaxEntropyGraphs.triangles(A) 
    end

    @testset "squares" begin
        # setup
        G = MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate)
        Gd = MaxEntropyGraphs.maspalomas()
        A = MaxEntropyGraphs.Graphs.adjacency_matrix(G)
        # coherence checks
        @test_throws DimensionMismatch squares(A[:,1:end-1])
        @test_throws ArgumentError squares(MaxEntropyGraphs.Graphs.adjacency_matrix(Gd))

        # equality
        @test squares(G) == squares(A)
        
        model = MaxEntropyGraphs.UBCM(G)
        solve_model!(model)
        @test_throws ArgumentError squares(model)
        set_Ĝ!(model)
        @test squares(model) == squares(model.Ĝ)
    end

    @testset "3-node directed subgraphs" begin
        # setup
        G = MaxEntropyGraphs.maspalomas()
        GA = MaxEntropyGraphs.Graphs.adjacency_matrix(G)
        Gu = MaxEntropyGraphs.Graphs.SimpleGraph(G)
        model = MaxEntropyGraphs.DBCM(G)
        # solve model
        solve_model!(model)
        set_Ĝ!(model)
        A = MaxEntropyGraphs.Ĝ(model)
        for motif_name in MaxEntropyGraphs.directed_graph_motif_function_names
            # on an adjacency_matrix
            motif_count_matrix = @eval begin $(motif_name)($(A)) end
            # on a DBCM model
            motif_count_model = @eval begin $(motif_name)($(model)) end
            @test motif_count_matrix == motif_count_model
            # on a graph
            gcount = @eval begin $(motif_name)($(G)) end
            acount = @eval begin $(motif_name)($(GA)) end
            @test isa(gcount, Int)
            @test_throws ArgumentError @eval begin $(motif_name)($(Gu)) end
            @test gcount == acount
        end
    end    
end

@testset "bipartite graphs" begin
    
end