

@testset "Models" begin
@testset "UBCM" begin
@testset "UBCM - generation, DataTypes" begin
    for maxdegree in [10;100;1000]
        for N in [10;1000;10000]
            for P in [Float16, Float32, Float64]
                k = rand(1:maxdegree, N)
                model = UBCM(k, P=P)
                # proper use of reduction
                @test map(x-> getindex(model.κ,x), model.idx) == k
                # element types
                @test eltype(model.κ) === P
            end
        end
    end 
end
@testset "UBCM - configuration setting" begin
    k = collect(1:10000)
    # test solution methods
    @test_throws ArgumentError UBCM(k, compact=false, method=:somethingillegal)
    # testing starting condition
    @test_throws DimensionMismatch UBCM(k, compact=false, initial=collect(1:length(k)÷2))
    @test_throws ArgumentError UBCM(k, compact=false, initial=:somethingillegal)
    @test UBCM(k, compact=false, initial=:nodes).x0 == k ./sqrt(length(k))
    @test UBCM(k, compact=false, initial=:links).x0 == k ./sqrt(2*length(k))
    L = 20.
    @test UBCM(k, compact=false, initial=:links, L=L).x0 == k ./sqrt(2*L)
end
@testset "UBCM - solutions" begin
    #=using PyCall
    using Graphs
    nemtropy = pyimport("NEMtropy")
	np = pyimport("numpy")
    # generate Graph to work on 
    G = barabasi_albert!(cycle_graph(2), 20, 2);
	K = degree(G)
    =#
end

end # end UBCM
end # end MODELS

#=
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
=#