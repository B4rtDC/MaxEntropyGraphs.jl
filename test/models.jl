using Graphs
using LinearAlgebra

@testset "Models" begin
    @testset "UBCM" begin
        # generation from graph 
        @test_throws MethodError UBCM(Graphs.erdos_renyi(10, 0.5, is_directed=true))
        model = UBCM(Graphs.erdos_renyi(10, 0.5, is_directed=false))
        @test isa(model, UBCM)
        # generation from maximum likelihood parameters
        x = rand(20)
        for T in [Float64; Float32; Float16]
            model = UBCM(T.(x))
            @test isa(model, UBCM{T})
            @test eltype(model.G) == T
            @test eltype(model.σ) == T
            @test size(model.G) == (length(x), length(x))
            @test issymmetric(model.G)
            @test size(model.σ) == (length(x), length(x))
        end
        @test_throws InexactError UBCM([1;2;3;4]) # ML parameters cannot be integers
    end

    @testset "DBCM" begin
       # generation from graph
       @test_throws MethodError DBCM(Graphs.erdos_renyi(10, 0.5, is_directed=false))
       model = DBCM(Graphs.erdos_renyi(10, 0.5, is_directed=true))
       @test isa(model, DBCM)
       # generation from maximum likelihood parameters
       x,y = rand(20), rand(20)
       for T in [Float64; Float32; Float16]
        model = DBCM(T.(x), T.(y))
        @test isa(model, DBCM{T})
        @test eltype(model.G) == T
        @test eltype(model.σ) == T
        @test !issymmetric(model.G)
        @test size(model.G) == (length(x), length(y))
        @test size(model.σ) == (length(x), length(y))
    end
    @test_throws InexactError DBCM([1;2;3;4], [1;2;3;4]) # ML parameters cannot be integers
    end
end