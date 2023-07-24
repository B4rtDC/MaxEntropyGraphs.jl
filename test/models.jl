###########################################################################################
# models.jl
#
# This file contains the tests for the models functions of the MaxEntropyGraphs.jl package
###########################################################################################


const allowedDataTypes =[Float16, Float32, Float64, BigFloat]


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
                model = UBCM(d=d, precision=precision)
                @test isa(model, UBCM)
                @test typeof(model).parameters[2] == precision
                @test typeof(model).parameters[1] == Nothing
                @test all([eltype(model.Θᵣ) == precision, eltype(model.xᵣ) == precision])
            end
            # testing breaking conditions
            @test_throws MethodError UBCM(1) # wrong input type
            # directed graph info loss warning message
            Gd = MaxEntropyGraphs.Graphs.SimpleDiGraph(G)
            @test_logs (:warn,"The graph is directed, the UBCM model is undirected, the directional information will be lost") UBCM(Gd, d=MaxEntropyGraphs.Graphs.indegree(Gd))
            # weighted graph info loss warning message
            Gw = MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(G)
            @test_logs (:warn,"The graph is weighted, the UBCM model is unweighted, the weight information will be lost") UBCM(Gw, d=MaxEntropyGraphs.Graphs.degree(Gw))
            # zero degree node
            dd = copy(d); dd[1] = 0
            @test_logs (:warn,"The graph has vertices with zero degree, this may lead to convergence issues.") UBCM(G, d=dd)
            # invalid combination of parameters
            @test_throws ArgumentError UBCM(MaxEntropyGraphs.Graphs.SimpleGraph(0)) # zero node graph
            @test_throws ArgumentError UBCM(MaxEntropyGraphs.Graphs.SimpleGraph(1)) # single node graph
            dt = d[1:end-1]
            @test_throws DimensionMismatch UBCM(G, d=dt) # different lengths
            
            @test_throws ArgumentError UBCM(d=Int[]) # zero length degree
            @test_throws ArgumentError UBCM(d=Int[1]) # single length degree
            
            dd = copy(d); dd[1] = length(d) + 1
            @test_throws DomainError UBCM(G, d=dd) # degree out of range
        end
        
        @testset "UBCM - parameter computation" begin
            
        end

        @testset "UBCM - sampling" begin

        end
    end

    @testset "DBCM" begin
        @testset "DBCM - generation" begin
            
        end
        @testset "DBCM - parameter computation" begin
            
        end
        @testset "DBCM - sampling" begin
            
        end
    end

    @testset "BiCM" begin
        @testset "BiCM - generation" begin
            
        end
        @testset "BiCM - parameter computation" begin
            
        end
        @testset "BiCM - sampling" begin
            
        end
    end
end

