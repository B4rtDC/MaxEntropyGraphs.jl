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

@testset "Bipartite graph metrics" begin
    @testset "biadjacency matrix" begin
        # error from graph
        G = MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate)
        @test_throws ArgumentError MaxEntropyGraphs.biadjacency_matrix(G)
        # testgraph (undirected)
        A = [0 0 0 0 1 0 0;
             0 0 0 0 1 1 0;
             0 0 0 0 0 0 1;
             0 0 0 0 0 1 1;
             1 1 0 0 0 0 0;
             0 1 0 1 0 0 0;
             0 0 1 1 0 0 0]
        G = MaxEntropyGraphs.Graphs.SimpleGraph(A)
        @test MaxEntropyGraphs.Graphs.is_bipartite(G)
        B = Array(MaxEntropyGraphs.biadjacency_matrix(G))
        @test B == A[1:4,5:7]
    end

    @testset "project (ignoring significance)" begin
        ## setup
        A = [0 0 0 0 1 0 0;
             0 0 0 0 1 1 0;
             0 0 0 0 0 0 1;
             0 0 0 0 0 1 1;
             1 1 0 0 0 0 0;
             0 1 0 1 0 0 0;
             0 0 1 1 0 0 0];
        ## graphs
        G = MaxEntropyGraphs.Graphs.SimpleGraph(A)
        Ad = copy(A); Ad[1,2] = 1; Ad[2,1] = 1;
        Gd = MaxEntropyGraphs.Graphs.SimpleGraph(Ad) # not bipartite

        # test
        @test_throws ArgumentError MaxEntropyGraphs.project(Gd) # not bipartite
        @test_throws ArgumentError MaxEntropyGraphs.project(G, method=:obscure_methode)
        @test_throws ArgumentError MaxEntropyGraphs.project(G, layer=:invalid_layer)
        
        ## bipartite matrix
        B = Array(MaxEntropyGraphs.biadjacency_matrix(G))
        @test_logs (:warn,"The matrix `B` is square, make sure it is a biadjacency matrix.") MaxEntropyGraphs.project(A)
        @test_throws ArgumentError MaxEntropyGraphs.project(A, method=:obscure_methode)
        @test_throws ArgumentError MaxEntropyGraphs.project(A, layer=:invalid_layer)

        for layer in [:top, :bottom]
            @test MaxEntropyGraphs.Graphs.adjacency_matrix(MaxEntropyGraphs.project(G, layer=layer)) == MaxEntropyGraphs.project(A[1:4,5:7], layer=layer)
        end
    end

    @testset "V_motifs (global)" begin
        # setup
        G = MaxEntropyGraphs.corporateclub()
        B = MaxEntropyGraphs.biadjacency_matrix(G)
        # test
        @test_throws ArgumentError MaxEntropyGraphs.V_motifs(G, layer=:invalid_layer)
        @test_throws ArgumentError MaxEntropyGraphs.V_motifs(B, layer=:invalid_layer)
        for layer in [:top, :bottom]
            @test MaxEntropyGraphs.V_motifs(G, layer=layer) == MaxEntropyGraphs.V_motifs(B, layer=layer)
        end

        # test with model
        model = MaxEntropyGraphs.BiCM(G)
        @test_throws ArgumentError MaxEntropyGraphs.V_motifs(model)
        MaxEntropyGraphs.solve_model!(model)
        @test_throws ArgumentError MaxEntropyGraphs.V_motifs(model, precomputed=true)
        MaxEntropyGraphs.set_Ĝ!(model)

        @test isa(MaxEntropyGraphs.V_motifs(model), precision(model))
        for layer in [:top, :bottom]
            @test MaxEntropyGraphs.V_motifs(model, layer=layer, precomputed=true) ≈ MaxEntropyGraphs.V_motifs(model, layer=layer, precomputed=false)
        end
    end

    @testset "V_motifs (local)" begin
        # setup
        G = MaxEntropyGraphs.corporateclub()
        B = MaxEntropyGraphs.biadjacency_matrix(G)
        membership = MaxEntropyGraphs.Graphs.bipartite_map(G)

        ## tests
        # graph based
        @test_throws ArgumentError MaxEntropyGraphs.V_motifs(G, findfirst(membership .== 1), findfirst(membership .== 2))

        # matrix based
        @test_logs (:warn,"The matrix `B` is square, make sure it is a biadjacency matrix.") MaxEntropyGraphs.V_motifs(B*B',1,2)
        @test_throws ArgumentError MaxEntropyGraphs.V_motifs(B,1,2, layer=:invalid_layer)
        @test_throws BoundsError MaxEntropyGraphs.V_motifs(B,1,size(B,1)+1, layer=:bottom)
        @test_throws BoundsError MaxEntropyGraphs.V_motifs(B,size(B,2)+1,1, layer=:top)

        # model based
        model = MaxEntropyGraphs.BiCM(G)
        @test_throws ArgumentError MaxEntropyGraphs.V_motifs(model, 1, 2) # parameters not computed
        MaxEntropyGraphs.solve_model!(model)
        @test_throws ArgumentError MaxEntropyGraphs.V_motifs(model, 1, 2, precomputed=true) # biadjacency matrix not computed
        MaxEntropyGraphs.set_Ĝ!(model)
        for layer in [:top, :bottom]
            @test isa(MaxEntropyGraphs.V_motifs(model, 1, 2,layer=layer), precision(model))
            @test MaxEntropyGraphs.V_motifs(model, 1, 2, layer=layer, precomputed=true) ≈ MaxEntropyGraphs.V_motifs(model, 1, 2, layer=layer, precomputed=false)
        end
    end

    @testset "V_motif_PB_parameters" begin
        # setup
        G = MaxEntropyGraphs.corporateclub()
        model = MaxEntropyGraphs.BiCM(G)

        ## tests
        @test_throws ArgumentError MaxEntropyGraphs.V_PB_parameters(model, 1, 2)
        MaxEntropyGraphs.solve_model!(model)
        @test_throws ArgumentError MaxEntropyGraphs.V_PB_parameters(model, 1, 2, precomputed=true)
        @test_throws ArgumentError MaxEntropyGraphs.V_PB_parameters(model, 1, 2, layer=:invalid_layer)
        for layer in [:bottom, :top]
            @test eltype(MaxEntropyGraphs.V_PB_parameters(model, 1, 2, layer=layer, precomputed=false)) == precision(model)
        end
        MaxEntropyGraphs.set_Ĝ!(model)
        @test_throws ArgumentError MaxEntropyGraphs.V_PB_parameters(model, 1, 2, precomputed=true, layer=:invalid_layer)
        for layer in [:bottom, :top]
            @test MaxEntropyGraphs.V_PB_parameters(model, 1, 2, layer=layer, precomputed=true) ≈ MaxEntropyGraphs.V_PB_parameters(model, 1, 2, layer=layer, precomputed=false)
        end
    end

    @testset "V_motif_projection" begin
        # setup
        G = MaxEntropyGraphs.corporateclub()
        model = MaxEntropyGraphs.BiCM(G)
        @test_throws ArgumentError MaxEntropyGraphs.project(model)
        MaxEntropyGraphs.solve_model!(model)
        @test_throws ArgumentError MaxEntropyGraphs.project(model, precomputed=true)
        MaxEntropyGraphs.set_Ĝ!(model)
        @test_throws ArgumentError MaxEntropyGraphs.project(model, α=-0.1)
        @test_throws ArgumentError MaxEntropyGraphs.project(model, layer=:invalid_layer)
        @test_throws ArgumentError MaxEntropyGraphs.project(model, distribution=:invalid_distribution)
        MaxEntropyGraphs.solve_model!(model)
        MaxEntropyGraphs.set_Ĝ!(model)

        for layer in [:bottom, :top]
            for distribution in [:Poisson, :PoissonBinomial]
                @test isa(MaxEntropyGraphs.project(model, layer=layer, precomputed=false, distribution=distribution), MaxEntropyGraphs.Graphs.SimpleGraph)
                @test MaxEntropyGraphs.project(model, layer=layer, precomputed=true, distribution=distribution) == MaxEntropyGraphs.project(model, layer=layer, precomputed=false, distribution=distribution)
                @test MaxEntropyGraphs.Graphs.nv(MaxEntropyGraphs.project(model, layer=layer, precomputed=false, distribution=distribution)) == (layer == :bottom ? model.status[:N⊥] : model.status[:N⊤])
                @test  MaxEntropyGraphs.project(model, layer=layer, precomputed=true, distribution=distribution, multithreaded=true) == MaxEntropyGraphs.project(model, layer=layer, precomputed=false, distribution=distribution, multithreaded=false)
            end
        end
    end

end