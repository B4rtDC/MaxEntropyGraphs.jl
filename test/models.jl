###########################################################################################
# models.jl
#
# This file contains the tests for the models functions of the MaxEntropyGraphs.jl package
###########################################################################################


allowedDataTypes =[Float16, Float32, Float64, BigFloat]


@testset "Models" begin
    @testset "UBCM" begin
        @testset "UBCM - generation" begin
            G = MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate)
            d = MaxEntropyGraphs.Graphs.degree(G)
            # simple model, directly from graph, different precisions
            for precision in allowedDataTypes
                model = UBCM(G, precision=precision)
                @test isa(model, UBCM)
                @test typeof(model).parameters[2] == precision
                @test typeof(model).parameters[1] == typeof(G)
                @test all([eltype(model.Θᵣ) == precision, eltype(model.xᵣ) == precision])
            end
            # simple model, directly from degree sequence, different precisions
            for precision in allowedDataTypes
                model = UBCM(d, precision=precision)
                @test isa(model, UBCM)
                @test typeof(model).parameters[2] == precision
                @test typeof(model).parameters[1] == Nothing
                @test all([eltype(model.Θᵣ) == precision, eltype(model.xᵣ) == precision])
            end
            # testing breaking conditions
            @test_throws MethodError UBCM(1) # wrong input type
            # directed graph info loss
            Gd = MaxEntropyGraphs.Graphs.SimpleDiGraph(G)
            @test_logs (:warn,"The graph is directed, the UBCM model is undirected, the directional information will be lost") UBCM(Gd, MaxEntropyGraphs.Graphs.indegree(Gd))
            # weighted graph info loss
            Gw = MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(G)
            @test_logs (:warn,"The graph is weighted, the UBCM model is unweighted, the weight information will be lost") UBCM(Gw, MaxEntropyGraphs.Graphs.degree(Gw))
            # zero degree node
            dd = copy(d); dd[1] = 0
            @test_logs (:warn,"The graph has vertices with zero degree, this may lead to convergence issues.") UBCM(G, dd)
            # invalid combination of parameters
            @test_throws ArgumentError UBCM(MaxEntropyGraphs.Graphs.SimpleGraph(0)) # zero node graph
            @test_throws ArgumentError UBCM(MaxEntropyGraphs.Graphs.SimpleGraph(1)) # single node graph
            dt = d[1:end-1]
            @test_throws DimensionMismatch UBCM(G, dt) # different lengths
            
            @test_throws ArgumentError UBCM(Int[]) # zero length degree
            @test_throws ArgumentError UBCM(Int[1]) # single length degree
            
            dd = copy(d); dd[1] = length(d) + 1
            @test_throws DomainError UBCM(G, dd) # degree out of range
        end
    end
end
    # @testset "UBCM" begin
    #     # generation from graph 
    #     @test_throws MethodError UBCM(Graphs.erdos_renyi(10, 0.5, is_directed=true))
    #     model = UBCM(Graphs.erdos_renyi(10, 0.5, is_directed=false))
    #     @test isa(model, UBCM)
    #     # generation from maximum likelihood parameters
    #     x = rand(20)
    #     for T in [Float64; Float32; Float16]
    #         model = UBCM(T.(x))
    #         @test isa(model, UBCM{T})
    #         @test eltype(model.G) == T
    #         @test eltype(model.σ) == T
    #         @test size(model.G) == (length(x), length(x))
    #         @test issymmetric(model.G)
    #         @test size(model.σ) == (length(x), length(x))
    #     end
    #     @test_throws InexactError UBCM([1;2;3;4]) # ML parameters cannot be integers
    # end

    # @testset "DBCM" begin
    #    # generation from graph
    #    @test_throws MethodError DBCM(Graphs.erdos_renyi(10, 0.5, is_directed=false))
    #    model = DBCM(Graphs.erdos_renyi(10, 0.5, is_directed=true))
    #    @test isa(model, DBCM)
    #    # generation from maximum likelihood parameters
    #    x,y = rand(20), rand(20)
    #    for T in [Float64; Float32; Float16]
    #     model = DBCM(T.(x), T.(y))
    #     @test isa(model, DBCM{T})
    #     @test eltype(model.G) == T
    #     @test eltype(model.σ) == T
    #     @test !issymmetric(model.G)
    #     @test size(model.G) == (length(x), length(y))
    #     @test size(model.σ) == (length(x), length(y))
    # end
    # @test_throws InexactError DBCM([1;2;3;4], [1;2;3;4]) # ML parameters cannot be integers
    # end
#end