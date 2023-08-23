###########################################################################################
# models.jl
#
# This file contains the tests for the models functions of the MaxEntropyGraphs.jl package
###########################################################################################


const allowedDataTypes =[Float64]


@testset "Models" begin
    #=
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
                    @test_throws ArgumentError MaxEntropyGraphs.set_xᵣ!(model)
                    @test_throws ArgumentError MaxEntropyGraphs.initial_guess(model, method=:strange)
                    for initial in [:degrees, :random, :degrees_minor, :chung_lu, :uniform] # degrees_minor fails
                        @testset "initial guess: $initial" begin
                            @test length(MaxEntropyGraphs.initial_guess(model, method=initial)) == model.status[:d_unique]
                            for (method, analytical_gradient) in [(:BFGS, true), (:BFGS, false), (:fixedpoint, false)]
                                @testset "method: $method, analytical_gradient: $analytical_gradient" begin
                                    MaxEntropyGraphs.solve_model!(model, initial=initial, method=method, analytical_gradient=analytical_gradient)
                                    A = MaxEntropyGraphs.Ĝ(model)
                                    # check that constraints are respected
                                    @test sum(A, dims=2) ≈ d
                                end
                                MaxEntropyGraphs.solve_model!(model, initial=initial)
                                A = MaxEntropyGraphs.Ĝ(model)
                                # check that constraints are respected
                                @test sum(A, dims=2) ≈ d
                            end
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
            MaxEntropyGraphs.set_σ!(model)
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
    =#
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

    @testset "UECM" begin
        @testset "UECM - generation" begin
            
        end
        @testset "UECM - parameter computation" begin
            
        end
        @testset "UECM - sampling" begin
            
        end
    end

    @testset "CReM" begin
        sources =       [1,1,1,2,3,3,4,4,5,6];
        destinations =  [2,2,3,3,4,5,6,7,7,5];
        weights =       [1,2,3,4,5,1,2,3,4,5];
        G = MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(sources, destinations, float.(weights))
        d = MaxEntropyGraphs.Graphs.degree(G)
        s = MaxEntropyGraphs.strength(G)
        @testset "CReM - generation" begin
            @testset "$(precision) precisions" for precision in allowedDataTypes
            # simple model, directly from graph, different precisions
                model = MaxEntropyGraphs.CReM(G, precision=precision)
                @test isa(model, MaxEntropyGraphs.CReM)
                @test typeof(model).parameters[2] == precision
                @test typeof(model).parameters[1] == typeof(G)
                @test all([ eltype(model.Θ) == precision, 
                            eltype(model.x) == precision,
                            eltype(model.α) == precision,
                            eltype(model.αᵣ) == precision,
                            eltype(model.x) == precision,
                            eltype(model.s) == precision])
            
            # simple model, directly from degree and strength sequences, different precisions
                model = MaxEntropyGraphs.CReM(d=d, s=s, precision=precision)
                @test isa(model, MaxEntropyGraphs.CReM)
                @test typeof(model).parameters[2] == precision
                @test typeof(model).parameters[1] == Nothing
                @test all([ eltype(model.Θ) == precision, 
                            eltype(model.x) == precision,
                            eltype(model.α) == precision,
                            eltype(model.αᵣ) == precision,
                            eltype(model.x) == precision,
                            eltype(model.s) == precision])
            end
            @testset "Errors and warnings" begin
                @test_throws MethodError MaxEntropyGraphs.CReM(1) # wrong input type
                # directed graph info loss warning message
                Gd = MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedDiGraph(G)
                @test_logs (:warn,"The graph is directed, the CReM model is undirected, the directional information will be lost") MaxEntropyGraphs.CReM(Gd,d=d)
                # zero degree nodes
                @test_logs (:warn, "The graph has vertices with zero degree, this may lead to convergence issues.") MaxEntropyGraphs.CReM(d=[d...;0], s=[s...;2.])
                @test_logs (:warn, "The graph has vertices with zero strength, this may lead to convergence issues.") MaxEntropyGraphs.CReM(d=[d...;1], s=[s...;0.])
                # empty/single node graphs
                @test_throws ArgumentError MaxEntropyGraphs.CReM(MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(0))
                @test_throws ArgumentError MaxEntropyGraphs.CReM(MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(1))
                # different lengths
                @test_throws DimensionMismatch MaxEntropyGraphs.CReM(G, d=d[1:end-1], s=s)
                @test_throws DimensionMismatch MaxEntropyGraphs.CReM(G, d=d, s=s[1:end-1])
                # specific check for the degree/strength sequences
                @test_throws ArgumentError MaxEntropyGraphs.CReM(d=Int[], s=Float64[])
                @test_throws ArgumentError MaxEntropyGraphs.CReM(d=Int[1], s=Float64[1])
                @test_throws DomainError MaxEntropyGraphs.CReM(d=d.*100, s=s)
                @test_throws DomainError MaxEntropyGraphs.CReM(d=d .*-1, s=s)
                @test_throws DomainError MaxEntropyGraphs.CReM(d=d, s=s.*-1)
            end
        end

        @testset "CReM - parameter computation" begin
            model = MaxEntropyGraphs.CReM(G)

            @testset "Initial conditions" begin
                for initial ∈ [:strengths, :strengths_minor, :random]
                    θ₀ = MaxEntropyGraphs.initial_guess(model, method=initial)
                    @test length(θ₀) == model.status[:d]
                    @test eltype(θ₀) == MaxEntropyGraphs.precision(model)
                end
                @test_throws ArgumentError MaxEntropyGraphs.initial_guess(model, method=:strange)
            end 

            @testset "Likelihood" begin
                @test_throws ArgumentError MaxEntropyGraphs.L_CReM(model)
                model.α .= [0.504265218660552072549307922599837183952331542968750000000000,0.504265218660552072549307922599837183952331542968750000000000,3.003999684368159339697967880056239664554595947265625000000000,1.293245515067872775105684013396967202425003051757812500000000,1.293245515067872775105684013396967202425003051757812500000000,0.504265218660552072549307922599837183952331542968750000000000,0.504265218660552072549307922599837183952331542968750000000000]
                model.status[:conditional_params_computed] = true
                @test_throws ArgumentError MaxEntropyGraphs.L_CReM(model)
                model.θ .= [0.185537669534768495660514986411726567894220352172851562500000,0.132939214624878454529266491590533405542373657226562500000000,0.160958338216350360649897766052163206040859222412109375000000,0.150129605521063469453224570315796881914138793945312500000000,0.150129605521063469453224570315796881914138793945312500000000,0.132939214624878454529266491590533405542373657226562500000000,0.132939214624878454529266491590533405542373657226562500000000]
            end

            @testset "Likelihood gradient" begin
                model = MaxEntropyGraphs.CReM(G)
                @test true
            end

            

            
        end

        @testset "CReM - sampling" begin
            
        end
    end
end

