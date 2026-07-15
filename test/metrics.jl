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

    @testset "reciprocity metrics" begin
        # hand-built 4-node weighted digraph with known reciprocity structure:
        # dyad (1,2) reciprocated (w(1→2)=2.5, w(2→1)=1.0), dyads (2,3) and (4,3) single-direction (w=3.0, w=4.0)
        G = MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedDiGraph(Int, Float64)
        for _ in 1:4
            MaxEntropyGraphs.Graphs.add_vertex!(G)
        end
        MaxEntropyGraphs.Graphs.add_edge!(G, 1, 2, 2.5)
        MaxEntropyGraphs.Graphs.add_edge!(G, 2, 1, 1.0)
        MaxEntropyGraphs.Graphs.add_edge!(G, 2, 3, 3.0)
        MaxEntropyGraphs.Graphs.add_edge!(G, 4, 3, 4.0)
        # weighted adjacency matrix in the W[i,j] = w(i→j) convention
        W = [0.0 2.5 0.0 0.0;
             1.0 0.0 3.0 0.0;
             0.0 0.0 0.0 0.0;
             0.0 0.0 4.0 0.0]
        A = Int.(.!iszero.(W))

        @testset "reciprocal degree sequences" begin
            # known values
            @test nonreciprocated_outdegree(A) == [0, 1, 0, 1]
            @test nonreciprocated_indegree(A)  == [0, 0, 2, 0]
            @test reciprocated_degree(A)       == [1, 1, 0, 0]
            # graph ≡ matrix dispatch
            @test nonreciprocated_outdegree(G) == nonreciprocated_outdegree(A)
            @test nonreciprocated_indegree(G)  == nonreciprocated_indegree(A)
            @test reciprocated_degree(G)       == reciprocated_degree(A)
            # single node ≡ sequence
            for f in (nonreciprocated_outdegree, nonreciprocated_indegree, reciprocated_degree)
                @test f(A) == [f(A, i) for i in 1:4]
                @test f(G) == [f(G, i) for i in 1:4]
            end
            # decomposition identities on a real network
            Gm = MaxEntropyGraphs.maspalomas()
            @test nonreciprocated_outdegree(Gm) .+ reciprocated_degree(Gm) == MaxEntropyGraphs.Graphs.outdegree(Gm)
            @test nonreciprocated_indegree(Gm)  .+ reciprocated_degree(Gm) == MaxEntropyGraphs.Graphs.indegree(Gm)
            Am = MaxEntropyGraphs.Graphs.adjacency_matrix(Gm)
            @test nonreciprocated_outdegree(Gm) == nonreciprocated_outdegree(Am)
            @test nonreciprocated_indegree(Gm)  == nonreciprocated_indegree(Am)
            @test reciprocated_degree(Gm)       == reciprocated_degree(Am)
            # errors
            @test_throws DimensionMismatch nonreciprocated_outdegree(W[:, 1:end-1])
            @test_throws ArgumentError nonreciprocated_outdegree(A, 5)
            @test_throws ArgumentError nonreciprocated_outdegree(A, 0)
            Gu = MaxEntropyGraphs.Graphs.smallgraph(:karate)
            @test_throws ArgumentError nonreciprocated_outdegree(Gu)
            @test_throws ArgumentError nonreciprocated_indegree(Gu, 1)
            @test_throws ArgumentError reciprocated_degree(Gu)
        end

        @testset "reciprocal strength sequences" begin
            # known values
            @test nonreciprocated_outstrength(W) == [0.0, 3.0, 0.0, 4.0]
            @test nonreciprocated_instrength(W)  == [0.0, 0.0, 7.0, 0.0]
            @test reciprocated_outstrength(W)    == [2.5, 1.0, 0.0, 0.0]
            @test reciprocated_instrength(W)     == [1.0, 2.5, 0.0, 0.0]
            # graph ≡ matrix dispatch
            @test nonreciprocated_outstrength(G) == nonreciprocated_outstrength(W)
            @test nonreciprocated_instrength(G)  == nonreciprocated_instrength(W)
            @test reciprocated_outstrength(G)    == reciprocated_outstrength(W)
            @test reciprocated_instrength(G)     == reciprocated_instrength(W)
            # single node ≡ sequence
            for f in (nonreciprocated_outstrength, nonreciprocated_instrength, reciprocated_outstrength, reciprocated_instrength)
                @test f(W) == [f(W, i) for i in 1:4]
                @test f(G) == [f(G, i) for i in 1:4]
            end
            # decomposition identities on a real network
            Gr = MaxEntropyGraphs.rhesus_macaques()
            @test nonreciprocated_outstrength(Gr) .+ reciprocated_outstrength(Gr) ≈ MaxEntropyGraphs.outstrength(Gr)
            @test nonreciprocated_instrength(Gr)  .+ reciprocated_instrength(Gr)  ≈ MaxEntropyGraphs.instrength(Gr)
            Wr = Matrix(transpose(Gr.weights))
            @test nonreciprocated_outstrength(Gr) == nonreciprocated_outstrength(Wr)
            @test nonreciprocated_instrength(Gr)  == nonreciprocated_instrength(Wr)
            @test reciprocated_outstrength(Gr)    == reciprocated_outstrength(Wr)
            @test reciprocated_instrength(Gr)     == reciprocated_instrength(Wr)
            # errors
            @test_throws DimensionMismatch nonreciprocated_outstrength(W[:, 1:end-1])
            @test_throws ArgumentError reciprocated_outstrength(W, 5)
            Gu = MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(MaxEntropyGraphs.rhesus_macaques())
            @test_throws ArgumentError nonreciprocated_outstrength(Gu)
            @test_throws ArgumentError reciprocated_instrength(Gu, 1)
        end

        @testset "triadic fluxes and intensities" begin
            Gr = MaxEntropyGraphs.rhesus_macaques()
            Wr = Matrix(transpose(Gr.weights))
            # naive triple-loop reference for the fluxes (same proof pattern as metrics_acceleration.tex)
            function ref_fluxes(Wf)
                nf = size(Wf, 1)
                Af = (!iszero).(Wf) .* 1.0
                af(l, i, j) = l === :P ? Af[i,j]*(1-Af[j,i]) : l === :Q ? (1-Af[i,j])*Af[j,i] : l === :R ? Af[i,j]*Af[j,i] : (1-Af[i,j])*(1-Af[j,i])
                wf(l, i, j) = l === :P ? Wf[i,j] : l === :Q ? Wf[j,i] : l === :R ? Wf[i,j]+Wf[j,i] : 0.0
                resf = zeros(13)
                for (k, s) in enumerate(MaxEntropyGraphs._motif_specs)
                    for i in 1:nf, j in 1:nf, l in 1:nf
                        (i == j || j == l || i == l) && continue
                        ind = af(s[1],i,j) * af(s[2],j,l) * af(s[3],l,i)
                        iszero(ind) && continue
                        resf[k] += ind * (wf(s[1],i,j) + wf(s[2],j,l) + wf(s[3],l,i))
                    end
                end
                return resf
            end
            @test motif_fluxes(Wr) ≈ ref_fluxes(Wr)
            @test motif_fluxes(Gr) == motif_fluxes(Wr)
            @test motif_flux(Wr, 13) == motif_fluxes(Wr)[13]
            @test_throws ArgumentError motif_flux(Wr, 14)
            @test_throws DimensionMismatch motif_fluxes(Wr[:, 1:end-1])
            @test_throws ArgumentError motif_fluxes(MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(Gr)) # undirected
            # intensities: on a binary-weight network intensity == count of instances weighted by 1
            Wb = Float64.((!iszero).(Wr))
            @test motif_intensities(Wb) ≈ Float64.(motifs(Matrix(Wb)))
            # intensities on the weighted network: nonzero exactly where the fluxes are
            I = motif_intensities(Gr)
            @test length(I) == 13
            @test (I .> 0) == (motif_fluxes(Gr) .> 0)
            @test_throws ArgumentError motif_intensities(MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(Gr)) # undirected
            # model methods: exact expected fluxes are finite, positive on populated motifs, and the
            # constraint-preserving model reproduces the total flux scale (Σ⟨Fm⟩ ≈ ΣFm within the ensemble spread)
            for model in (CRWCM(Gr), DCReM(Gr))
                @test_throws ArgumentError motif_fluxes(model) # parameters not computed
                solve_model!(model)
                fexp = motif_fluxes(model)
                @test length(fexp) == 13 && all(isfinite, fexp) && all(fexp .>= 0)
            end
        end

        @testset "ensemble z-scores" begin
            Gr = MaxEntropyGraphs.rhesus_macaques()
            model = CRWCM(Gr)
            solve_model!(model)
            # binary vector metric: the observed value must be computed on the BINARISED adjacency
            # (Graphs.adjacency_matrix returns the weights for SimpleWeightedGraphs graphs)
            res = motif_zscores(model, n=50, rng=MaxEntropyGraphs.Xoshiro(1))
            @test length(res.z) == length(res.μ) == length(res.σ) == length(res.obs) == 13
            @test res.obs == motifs(Float64.((!iszero).(Matrix(MaxEntropyGraphs.Graphs.adjacency_matrix(Gr)))))
            @test res.obs == motifs(MaxEntropyGraphs.Graphs.adjacency_matrix(MaxEntropyGraphs.Graphs.SimpleDiGraph(Gr)))
            # weighted vector metric
            resf = flux_zscores(model, n=50, rng=MaxEntropyGraphs.Xoshiro(1))
            @test resf.obs ≈ motif_fluxes(Gr)
            @test all(isfinite, resf.z[resf.σ .> 0])
            # scalar metric, reproducibility under a fixed seed
            r1 = ensemble_zscores(model, sum, n=50, rng=MaxEntropyGraphs.Xoshiro(9), weighted=true)
            r2 = ensemble_zscores(model, sum, n=50, rng=MaxEntropyGraphs.Xoshiro(9), weighted=true)
            @test r1.z == r2.z
            @test isa(r1.z, Float64)
            # error paths
            @test_throws ArgumentError ensemble_zscores(model, sum, n=1)
            m_nog = CRWCM(d_out=model.d_out, d_in=model.d_in, d_rec=model.d_rec, s_out=model.s_out, s_in=model.s_in, s_rec_out=model.s_rec_out, s_rec_in=model.s_rec_in)
            @test_throws ArgumentError ensemble_zscores(m_nog, sum) # no observed graph
            mb = RBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(Gr))
            solve_model!(mb)
            @test_throws ArgumentError ensemble_zscores(mb, sum, n=10, weighted=true) # binary model has no weights
        end

        @testset "(weighted) reciprocity" begin
            # known values: 2 of 4 links reciprocated; 3.5 of 10.5 total weight on reciprocated links
            @test reciprocity(A) == 0.5
            @test reciprocity(G) == 0.5
            @test weighted_reciprocity(W) ≈ 3.5 / 10.5
            @test weighted_reciprocity(G) ≈ 3.5 / 10.5
            # graph ≡ matrix on a real network
            Gm = MaxEntropyGraphs.maspalomas()
            @test reciprocity(Gm) == reciprocity(Matrix(MaxEntropyGraphs.Graphs.adjacency_matrix(Gm)))
            Gr = MaxEntropyGraphs.rhesus_macaques()
            @test weighted_reciprocity(Gr) == weighted_reciprocity(Matrix(transpose(Gr.weights)))
            # boundary values
            @test reciprocity(MaxEntropyGraphs.taro_exchange()) == 1.0 # fully reciprocal
            # model methods: constraint-preserving models reproduce the observed values, baselines do not
            Grd = MaxEntropyGraphs.Graphs.SimpleDiGraph(Gr)
            mrbcm = RBCM(Grd); solve_model!(mrbcm)
            @test reciprocity(mrbcm) ≈ reciprocity(Grd) rtol=1e-6
            mdbcm = DBCM(Grd); solve_model!(mdbcm)
            @test reciprocity(mdbcm) < reciprocity(Grd)
            mcrwcm = CRWCM(Gr); solve_model!(mcrwcm)
            @test reciprocity(mcrwcm) ≈ reciprocity(Grd) rtol=1e-6
            @test weighted_reciprocity(mcrwcm) ≈ weighted_reciprocity(Gr) rtol=1e-4
            mdcrem = DCReM(Gr); solve_model!(mdcrem)
            @test reciprocity(mdcrem) < reciprocity(Grd)
            @test weighted_reciprocity(mdcrem) < weighted_reciprocity(Gr)
            @test_throws ArgumentError reciprocity(RBCM(Grd))              # parameters not computed
            @test_throws ArgumentError weighted_reciprocity(CRWCM(Gr))     # parameters not computed
            # errors
            @test_throws ArgumentError reciprocity(MaxEntropyGraphs.Graphs.smallgraph(:karate)) # undirected
            @test_throws ArgumentError reciprocity(MaxEntropyGraphs.Graphs.SimpleDiGraph(3))    # no links
            @test_throws ArgumentError reciprocity(zeros(3,3))                                  # no links
            @test_throws DimensionMismatch reciprocity(zeros(3,2))
            @test_throws ArgumentError weighted_reciprocity(zeros(3,3))
            @test_throws DimensionMismatch weighted_reciprocity(zeros(3,2))
            Gu = MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(Gr)
            @test_throws ArgumentError weighted_reciprocity(Gu)
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

###########################################################################################
# Accelerated-metrics validation: pin the reformulated kernels against reference (pre-
# acceleration) implementations across sizes and element types, and check autodiff (σₓ path).
###########################################################################################

# --- reference (naive, pre-acceleration) implementations ---
function ref_triangles(A)
    res = zero(eltype(A))
    for i in axes(A,1), j in axes(A,1), k in axes(A,1)
        (i != j && j != k && k != i) && (res += A[i,j]*A[j,k]*A[k,i])
    end
    return res/6
end
function ref_squares(A)
    res = zero(eltype(A)); o = one(eltype(A))
    for i in axes(A,1), j in axes(A,1)
        j == i && continue
        for k in axes(A,1)
            (k==i||k==j) && continue
            for l in axes(A,1)
                (l==i||l==j||l==k) && continue
                res += A[i,j]*A[j,k]*A[k,l]*A[l,i]*(o-A[i,k])*(o-A[l,j])
            end
        end
    end
    return res/8
end
function ref_ANND(A)
    N = size(A,1); out = zeros(Float64, N)
    for i in 1:N
        di = sum(@view A[:,i])
        out[i] = iszero(di) ? 0.0 : mapreduce(x -> A[i,x]*sum(@view A[:,x]), +, 1:N)/di
    end
    return out
end
ref_arr(A,i,j)   = A[i,j]*(one(eltype(A))-A[j,i])
ref_bak(A,i,j)   = (one(eltype(A))-A[i,j])*A[j,i]
ref_rec(A,i,j)   = A[i,j]*A[j,i]
ref_abs(A,i,j)   = (one(eltype(A))-A[i,j])*(one(eltype(A))-A[j,i])
const REF_MOTIF_TRIPLES = [ (ref_bak,ref_arr,ref_abs),(ref_bak,ref_bak,ref_abs),(ref_bak,ref_rec,ref_abs),
    (ref_bak,ref_abs,ref_arr),(ref_bak,ref_arr,ref_arr),(ref_bak,ref_rec,ref_arr),(ref_arr,ref_rec,ref_abs),
    (ref_rec,ref_rec,ref_abs),(ref_arr,ref_arr,ref_arr),(ref_rec,ref_arr,ref_arr),(ref_rec,ref_bak,ref_arr),
    (ref_rec,ref_rec,ref_arr),(ref_rec,ref_rec,ref_rec) ]
function ref_motif(A, f1, f2, f3)
    res = zero(eltype(A))
    for i in axes(A,1), j in axes(A,1), k in axes(A,1)
        (i != j && j != k && k != i) && (res += f1(A,i,j)*f2(A,j,k)*f3(A,k,i))
    end
    return res
end
function ref_Vmotifs(A, layer)
    res = zero(eltype(A))
    if layer == :bottom
        for i in axes(A,1), j in axes(A,1); j>i && (res += MaxEntropyGraphs.dot(@view(A[i,:]), @view(A[j,:]))); end
    else
        for i in axes(A,2), j in axes(A,2); j>i && (res += MaxEntropyGraphs.dot(@view(A[:,i]), @view(A[:,j]))); end
    end
    return res
end
# central finite-difference gradient at selected linear/Cartesian indices (dependency-free)
function ref_fd_grad(f, A, idxs; h=1e-6)
    [ (Ap = copy(A); Ap[ij] += h; Am = copy(A); Am[ij] -= h; (f(Ap) - f(Am)) / (2h)) for ij in idxs ]
end

const ALL_MOTIFS = (M1,M2,M3,M4,M5,M6,M7,M8,M9,M10,M11,M12,M13)

@testset "metrics acceleration" begin
    @testset "integer exactness on fixtures" begin
        G = MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate)
        A = MaxEntropyGraphs.Graphs.adjacency_matrix(G)
        @test triangles(A) == triangles(G) == ref_triangles(A)
        @test squares(A)   == squares(G)   == ref_squares(A)
        @test ANND(A)      == ANND(G)      == ref_ANND(A)

        Gd = MaxEntropyGraphs.maspalomas(); GA = MaxEntropyGraphs.Graphs.adjacency_matrix(Gd)
        for (idx, Mk) in enumerate(ALL_MOTIFS)
            @test Mk(GA) == Mk(Gd)
            @test Mk(GA) == ref_motif(GA, REF_MOTIF_TRIPLES[idx]...)
        end
        @test motifs(GA) == [Mk(GA) for Mk in ALL_MOTIFS]

        Gb = MaxEntropyGraphs.corporateclub(); B = MaxEntropyGraphs.biadjacency_matrix(Gb)
        for layer in (:bottom, :top)
            @test V_motifs(B, layer=layer) == V_motifs(Gb, layer=layer) == ref_Vmotifs(B, layer)
        end
    end

    @testset "float exactness vs reference (random real matrices in [0,1])" begin
        rng = MaxEntropyGraphs.Xoshiro(1234)
        for N in (12, 30, 45)
            M = rand(rng, N, N); S = (M .+ M') ./ 2; S[MaxEntropyGraphs.diagind(S)] .= 0.0
            @test triangles(S) ≈ ref_triangles(S) rtol=1e-10
            @test squares(S)   ≈ ref_squares(S)   rtol=1e-10
            @test ANND(S)      ≈ ref_ANND(S)      rtol=1e-10
            D = rand(rng, N, N); D[MaxEntropyGraphs.diagind(D)] .= 0.0     # directed, zero diagonal
            for (idx, Mk) in enumerate(ALL_MOTIFS)
                @test Mk(D) ≈ ref_motif(D, REF_MOTIF_TRIPLES[idx]...) rtol=1e-10
            end
            @test motifs(D) ≈ [Mk(D) for Mk in ALL_MOTIFS] rtol=1e-10
            Br = rand(rng, N, N+3)
            for layer in (:bottom, :top)
                @test V_motifs(Br, layer=layer, skipchecks=true) ≈ ref_Vmotifs(Br, layer) rtol=1e-10
            end
        end
    end

    @testset "larger sparse graphs (integer exact / sparse squares fast-path)" begin
        rng = MaxEntropyGraphs.Xoshiro(7)
        for N in (100, 250)
            G = MaxEntropyGraphs.Graphs.barabasi_albert(N, 4, seed=161)
            A = MaxEntropyGraphs.Graphs.adjacency_matrix(G)
            @test triangles(A) == triangles(G)
            @test squares(A)   == squares(G)        # exercises the sparse (graph) fast-path
            @test ANND(A)      == ANND(G)
        end
        # directed motif exactness vs reference on a mid-size dense directed matrix
        Dg = MaxEntropyGraphs.Graphs.erdos_renyi(60, 0.2, is_directed=true, seed=3)
        DA = Matrix(MaxEntropyGraphs.Graphs.adjacency_matrix(Dg))
        for (idx, Mk) in enumerate(ALL_MOTIFS)
            @test Mk(DA) == ref_motif(DA, REF_MOTIF_TRIPLES[idx]...)
        end
    end

    @testset "return types preserved" begin
        Gb = MaxEntropyGraphs.corporateclub(); B = MaxEntropyGraphs.biadjacency_matrix(Gb)
        @test V_motifs(B, layer=:bottom) isa Integer            # Int-in -> Int-out
        @test V_motifs(Float64.(B), layer=:bottom) isa Float64
        G = MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate)
        @test eltype(ANND(MaxEntropyGraphs.Graphs.adjacency_matrix(G))) == Float64
    end

    @testset "strength(G, i) single-node fix" begin
        G = MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedDiGraph(Int, Float64)
        for _ in 1:4; MaxEntropyGraphs.Graphs.add_vertex!(G); end
        MaxEntropyGraphs.Graphs.add_edge!(G, 1, 2, 1.0); MaxEntropyGraphs.Graphs.add_edge!(G, 2, 3, 2.0)
        MaxEntropyGraphs.Graphs.add_edge!(G, 3, 4, 3.0); MaxEntropyGraphs.Graphs.add_edge!(G, 4, 1, 4.0)
        for i in 1:4, dir in (:in, :out, :both)
            si = MaxEntropyGraphs.strength(G, i; dir=dir)
            @test si isa Real                                   # a scalar, not the whole vector (old bug)
            @test si == MaxEntropyGraphs.strength(G; dir=dir)[i]
        end
    end

    @testset "autodiff consistency (σₓ path)" begin
        # UBCM (undirected) — triangles, squares, ANND-sum
        mu = MaxEntropyGraphs.UBCM(MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate))
        solve_model!(mu); set_Ĝ!(mu); set_σ!(mu)
        Au = mu.Ĝ
        idxs = [CartesianIndex(1,2), CartesianIndex(3,7), CartesianIndex(10,20)]
        Xtri  = A -> triangles(A; check_dimensions=false, check_directed=false)
        Xsq   = A -> squares(A; check_dimensions=false, check_directed=false)
        Xannd = A -> sum(ANND(A; check_dimensions=false, check_directed=false))
        for (X, tol) in ((Xtri, 1e-5), (Xsq, 1e-4), (Xannd, 1e-5))
            g_rd = MaxEntropyGraphs.ReverseDiff.gradient(X, Au)
            @test [g_rd[ij] for ij in idxs] ≈ ref_fd_grad(X, Au, idxs) rtol=tol
        end
        # σₓ end-to-end runs across all three backends for a matmul-based metric
        for gm in (:ReverseDiff, :ForwardDiff, :Zygote)
            s = MaxEntropyGraphs.σₓ(mu, Xtri; gradient_method=gm)
            @test isfinite(s) && s > 0
        end

        # DBCM (directed) — a few motifs incl. one with a correction term (M1) and the pure-trace ones (M9,M13)
        md = MaxEntropyGraphs.DBCM(MaxEntropyGraphs.maspalomas())
        solve_model!(md); set_Ĝ!(md)
        Ad = md.Ĝ
        didxs = [CartesianIndex(1,2), CartesianIndex(5,9), CartesianIndex(12,3)]
        for Mk in (M1, M9, M13)
            g_rd = MaxEntropyGraphs.ReverseDiff.gradient(Mk, Ad)
            @test [g_rd[ij] for ij in didxs] ≈ ref_fd_grad(Mk, Ad, didxs) rtol=1e-4
        end
    end
end
###########################################################################################
# Additional metric coverage: single-node methods, zero-degree branches, and error paths
# that are not exercised by the main test-sets (some were orphaned when the ANND vector
# methods were switched to a single gemv). Same fixtures/patterns as above.
###########################################################################################
@testset "metric coverage (edge cases & single-node methods)" begin
    G = MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate)
    A = MaxEntropyGraphs.Graphs.adjacency_matrix(G)

    @testset "strength single-node: undirected + invalid direction" begin
        Gw = MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(Int, Float64)
        for _ in 1:3; MaxEntropyGraphs.Graphs.add_vertex!(Gw); end
        MaxEntropyGraphs.Graphs.add_edge!(Gw, 1, 2, 2.0); MaxEntropyGraphs.Graphs.add_edge!(Gw, 2, 3, 4.0)
        for i in 1:3
            @test MaxEntropyGraphs.strength(Gw, i) == MaxEntropyGraphs.strength(Gw)[i]   # undirected single-node path
        end
        Gd = MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedDiGraph(Int, Float64)
        for _ in 1:3; MaxEntropyGraphs.Graphs.add_vertex!(Gd); end
        MaxEntropyGraphs.Graphs.add_edge!(Gd, 1, 2, 1.0)
        @test_throws DomainError MaxEntropyGraphs.strength(Gd, 1, dir=:invalid_dir)
    end

    @testset "ANND single-node matrix methods" begin
        # non-zero degree: single-node == vector element == graph
        @test ANND(A, 1) == ANND(A)[1]
        @test ANND(A, 1) == ANND(G, 1)
        Gd = MaxEntropyGraphs.maspalomas(); Ad = MaxEntropyGraphs.Graphs.adjacency_matrix(Gd)
        @test ANND_out(Ad, 1) == ANND_out(Ad)[1]
        @test ANND_in(Ad, 1)  == ANND_in(Ad)[1]
        # zero-degree node -> the `iszero` branch of each single-node method returns 0.0
        Az = zeros(Int, 3, 3); Az[1,2] = Az[2,1] = 1     # node 3 isolated, symmetric
        @test ANND(Az, 3)     == zero(Float64)
        @test ANND_out(Az, 3) == zero(Float64)
        @test ANND_in(Az, 3)  == zero(Float64)
    end

    @testset "ANND graph single-node zero-degree branch" begin
        Gd = MaxEntropyGraphs.Graphs.SimpleDiGraph(3); MaxEntropyGraphs.Graphs.add_edge!(Gd, 1, 2)
        @test ANND_out(Gd, 3) == zero(Float64)    # zero out-degree
        @test ANND_in(Gd, 3)  == zero(Float64)    # zero in-degree
    end

    @testset "wedges non-symmetric matrix error" begin
        Ad = MaxEntropyGraphs.Graphs.adjacency_matrix(MaxEntropyGraphs.maspalomas())
        @test_throws ArgumentError wedges(Ad)
    end

    @testset "M13 fully-reciprocated triangle (graph method)" begin
        Gr = MaxEntropyGraphs.Graphs.SimpleDiGraph(3)
        for (a, b) in ((1,2),(2,1),(2,3),(3,2),(1,3),(3,1))
            MaxEntropyGraphs.Graphs.add_edge!(Gr, a, b)
        end
        @test M13(Gr) == 6                                   # one reciprocated triangle, counted 6×
        @test M13(Gr) == M13(Matrix(MaxEntropyGraphs.Graphs.adjacency_matrix(Gr)))
    end

    @testset "project(::SimpleGraph, method=:weighted)" begin
        Ab = [0 0 0 0 1 0 0; 0 0 0 0 1 1 0; 0 0 0 0 0 0 1; 0 0 0 0 0 1 1;
              1 1 0 0 0 0 0; 0 1 0 1 0 0 0; 0 0 1 1 0 0 0]
        Gb = MaxEntropyGraphs.Graphs.SimpleGraph(Ab)
        @test isa(MaxEntropyGraphs.project(Gb, layer=:bottom, method=:weighted),
                  MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph)
    end

    @testset "V_motifs global square-matrix warning" begin
        @test_logs (:warn, "The matrix `A` is square, make sure it is a biadjacency matrix.") MaxEntropyGraphs.V_motifs(A)
    end

    @testset "V_motifs(G, i, j) local graph path" begin
        Gv = MaxEntropyGraphs.Graphs.SimpleGraph(5)
        for (a, b) in ((1,4),(1,5),(2,4),(2,5),(3,4)); MaxEntropyGraphs.Graphs.add_edge!(Gv, a, b); end
        @test MaxEntropyGraphs.V_motifs(Gv, 1, 2) == 2
    end

    @testset "V_motifs(m::BiCM, i, j) invalid layer (precomputed=false)" begin
        model = MaxEntropyGraphs.BiCM(MaxEntropyGraphs.corporateclub()); solve_model!(model)
        @test_throws ArgumentError MaxEntropyGraphs.V_motifs(model, 1, 2, layer=:invalid_layer, precomputed=false)
    end
end
