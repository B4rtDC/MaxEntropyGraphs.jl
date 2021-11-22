module ModelsTest

using Test
using LightGraphs
#using Random
using fastmaxent

@testset "Models" begin

#rng = MersenneTwister(1)
@testset "UBCM" begin
    @testset "UBCM Model" begin
    @testset "UBCM - Error catching" begin
        G = LightGraphs.static_scale_free(100,1000,2.1)
        add_vertex!(G)
        # unconnected nodes
        @test_throws DomainError UBCM(G)
        rem_vertex!(G, nv(G))
        # self loops
        add_edge!(G,1,1)
        @test_throws DomainError UBCM(G)
    end
    @testset "UBCM - Instance generation" begin
        G = LightGraphs.static_scale_free(100,1000,2.3)
        model = UBCM(G)
        @test isa(model.x0, Vector)
        @test isa(model.x, Vector)
        @test isa(model.F, Vector)
        @test isa(model.f!, Function)
        @test isa(model, UBCM)
    end
    @testset "UBCM - Solving" begin
        G = LightGraphs.static_scale_free(100,2000,2.4)
        model = UBCM(G)
        @info model
        res = solve!(model)
        @test all(res.zero .> 0) # non-negative solution
        @test all(model.x .> 0) # non-negative solution
        @test isapprox(degree(G), degree(model)) # check constraints match
    end
    end

    @testset "UBCM Compact Model" begin
    @testset "UBCM Compact - Error catching" begin
        G = LightGraphs.static_scale_free(100,1000,2.1)
        add_vertex!(G)
        # unconnected nodes
        @test_throws DomainError UBCM(G)
        rem_vertex!(G, nv(G))
        # self loops
        add_edge!(G,1,1)
        @test_throws DomainError UBCM(G)
    end
    @testset "UBCM - Instance generation" begin
        G = LightGraphs.static_scale_free(100,1000,2.3)
        d = degree(G)
        model = UBCMCompact(G)
        @test isa(model.x0, Vector)
        @test isa(model.xs, Vector)
        @test isa(model.F, Vector)
        @test isa(model.f!, Function)
        @test isa(model, UBCMCompact)
        @test length(model.x0) == length(unique(d))
        @test length(model.F) == length(unique(d))
        @test length(model.xs) == length(unique(d))
        @test isnothing(model.x)
    end
    @testset "UBCM - Solving" begin
        G = LightGraphs.static_scale_free(100,2000,2.4)
        model = UBCMCompact(G)
        @info model
        res = solve!(model)
        @test all(res.zero .> 0) # non-negative solution
        @test all(model.xs .> 0) # non-negative solution
        @test isapprox(degree(G), degree(model)) # check constraints match
    end
    end

    @testset "UBCM Compare" begin
        G = LightGraphs.static_scale_free(300,10000,2.4)
        large_model = UBCM(G)
        compact_model = UBCMCompact(G)
        res_large = solve!(large_model)
        res_compact = solve!(compact_model)
        @test all(isapprox(large_model.x, compact_model.x))
    end
end


@testset "DBCM" begin

end

end
end