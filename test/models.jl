###########################################################################################
# models.jl
#
# This file contains the tests for the models functions of the MaxEntropyGraphs.jl package
###########################################################################################


const allowedDataTypes =[Float64]


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
        
        @testset "UBCM - Likelihood gradient test" begin
            G = MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate)
            model = MaxEntropyGraphs.UBCM(G)
            θ₀ = MaxEntropyGraphs.initial_guess(model)
            ∇L_buf = zeros(length(θ₀))
            ∇L_buf_min = zeros(length(θ₀))
            x_buff = zeros(length(θ₀))
            MaxEntropyGraphs.∇L_UBCM_reduced!(∇L_buf, θ₀, model.dᵣ, model.f, x_buff)
            MaxEntropyGraphs.∇L_UBCM_reduced_minus!(∇L_buf_min, θ₀, model.dᵣ, model.f, x_buff)
            @info ∇L_buf, ∇L_buf_min
            @test ∇L_buf ≈ -∇L_buf_min
            ∇L_zyg = MaxEntropyGraphs.Zygote.gradient(θ -> MaxEntropyGraphs.L_UBCM_reduced(θ, model.dᵣ, model.f), θ₀)[1]
            @test ∇L_zyg ≈ ∇L_buf
            @test ∇L_zyg ≈ -∇L_buf_min
        end

        @testset "UBCM - parameter computation" begin
            #for (method, analytical_gradient) in [(:BFGS, true), (:Newton, true))]
            G = MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate)
            d = MaxEntropyGraphs.Graphs.degree(G)
            # simple model, directly from graph, different precisions
            for precision in allowedDataTypes
                @testset "$(precision) precision" begin
                    model = UBCM(G, precision=precision)
                    @test_throws ArgumentError Ĝ(model)
                    @test_throws ArgumentError σˣ(model)
                    @test_throws ArgumentError MaxEntropyGraphs.initial_guess(model, method=:strange)
                    for initial in [:degrees, :random, :degrees_minor, :chung_lu, :uniform] # degrees_minor fails
                        @testset "initial guess: $initial" begin
                            @test length(MaxEntropyGraphs.initial_guess(model, method=initial)) == model.status[:d_unique]
                            MaxEntropyGraphs.solve_model!(model, initial=initial)
                            A = MaxEntropyGraphs.Ĝ(model)
                            # check that constraints are respected
                            @test sum(A, dims=2) ≈ d
                        end
                    end
                    @test all([eltype(model.Θᵣ) == precision, eltype(model.xᵣ) == precision])
                end
            end
        end

        @testset "UBCM - sampling" begin
            model = UBCM(MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate))
            # parameters unknown
            @test_throws ArgumentError MaxEntropyGraphs.rand(model, precomputed=false)
            @test_throws ArgumentError MaxEntropyGraphs.Ĝ(model)
            @test_throws ArgumentError MaxEntropyGraphs.σˣ(model)
            # solve model
            MaxEntropyGraphs.solve_model!(model)
            # parameters known, but G not set
            @test_throws ArgumentError MaxEntropyGraphs.rand(model, precomputed=true)
            @test MaxEntropyGraphs.Graphs.nv(rand(model)) == MaxEntropyGraphs.Graphs.nv(model.G)
            MaxEntropyGraphs.set_Ĝ!(model)
            @test eltype(MaxEntropyGraphs.σˣ(model)) == MaxEntropyGraphs.precision(model)
            @test size(MaxEntropyGraphs.σˣ(model)) == (MaxEntropyGraphs.Graphs.nv(model.G),MaxEntropyGraphs.Graphs.nv(model.G))
            @test eltype(MaxEntropyGraphs.Ĝ(model)) == MaxEntropyGraphs.precision(model)
            @test size(MaxEntropyGraphs.Ĝ(model)) == (MaxEntropyGraphs.Graphs.nv(model.G),MaxEntropyGraphs.Graphs.nv(model.G))
            # sampling
            S = rand(model, 100)
            @test length(S) == 100
            @test all(MaxEntropyGraphs.Graphs.nv.(S) .== MaxEntropyGraphs.Graphs.nv(model.G))

            
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

