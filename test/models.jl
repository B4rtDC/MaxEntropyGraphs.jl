###########################################################################################
# models.jl
#
# This file contains the tests for the models functions of the MaxEntropyGraphs.jl package
###########################################################################################


#const


@testset "Models" begin
    @testset "UBCM" begin
        @testset "UBCM - generation" begin
            allowedDataTypes = [Float64; Float32; Float16]
            G = MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate)
            d = MaxEntropyGraphs.Graphs.degree(G)
            # simple model, directly from graph, different precisions
            for precision in allowedDataTypes
                model = UBCM(G, precision=precision)
                @test isa(model, UBCM)
                @test typeof(model).parameters[2] == precision
                @test MaxEntropyGraphs.precision(model) == precision
                @test typeof(model).parameters[1] == typeof(G)
                @test all([eltype(model.θᵣ) == precision, eltype(model.xᵣ) == precision])
            end
            # simple model, directly from degree sequence, different precisions
            for precision in allowedDataTypes
                model = UBCM(d=d, precision=precision)
                @test isa(model, UBCM)
                @test typeof(model).parameters[2] == precision
                @test MaxEntropyGraphs.precision(model) == precision
                @test typeof(model).parameters[1] == Nothing
                @test all([eltype(model.θᵣ) == precision, eltype(model.xᵣ) == precision])
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
            allowedDataTypes = [Float64] # ; Float32; Float16 kept out for ocasional convergence issues
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
                    @test all([eltype(model.θᵣ) == precision, eltype(model.xᵣ) == precision])
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

        @testset "UBCM - degree metrics" begin
            model = UBCM(MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate))
            # adjacency matric
            @test iszero(MaxEntropyGraphs.A(model,1,1))
            @test isa(MaxEntropyGraphs.A(model,1,2), precision(model))
            # parameters not computed yet
            @test_throws ArgumentError degree(model, 1)
            solve_model!(model)
            # check out of bounds
            @test_throws ArgumentError degree(model, length(model.d) + 1)
            # check methods
            @test_throws ArgumentError degree(model, method=:unknown_method)
            # check that the degree is correct
            for method in [:reduced, :full]
                @test isapprox(degree(model, method=method), model.d)
            end
            # check precompute
            @test_throws ArgumentError degree(model, method=:adjacency)
            set_Ĝ!(model)
            @test isapprox(degree(model, method=:adjacency), model.d)
        end

        @testset "UBCM - (B/A)IC(c)" begin
            model = UBCM(MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate))
            # parameters not computed yet
            @test_throws ArgumentError MaxEntropyGraphs.AIC(model)
            @test_throws ArgumentError MaxEntropyGraphs.AICc(model)
            @test_throws ArgumentError MaxEntropyGraphs.BIC(model)
            solve_model!(model)
            # check warning
            @test_logs (:warn, "The number of observations is small with respect to the number of parameters (n/k < 40). Consider using the corrected AIC (AICc) instead.") MaxEntropyGraphs.AIC(model)
            # test types
            @test isa(MaxEntropyGraphs.AIC(model), precision(model))
            @test isa(MaxEntropyGraphs.AICc(model), precision(model))
            @test isa(MaxEntropyGraphs.BIC(model), precision(model))
        end

        @testset "UBCM - adjacancy matrix variance" begin
            model = UBCM(MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate))
            MaxEntropyGraphs.solve_model!(model)
            # parameters not computed yet
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum)
            MaxEntropyGraphs.set_Ĝ!(model)
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum)
            MaxEntropyGraphs.set_σ!(model)
            # unknown autodiff method
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum, gradient_method=:unknown_method)
            # normal functioning
            for method in [:ForwardDiff; :ReverseDiff; :Zygote]
                @testset "gradient_method: $(method)" begin
                    @assert MaxEntropyGraphs.σₓ(model, sum, gradient_method=method) ≈ sqrt(sum(model.σ .^ 2))
                end
            end
        end
    end
    
    
    @testset "DBCM" begin
        @testset "DBCM - generation" begin
            allowedDataTypes = [Float64; Float32; Float16]
            G = MaxEntropyGraphs.taro_exchange()
            d_in = MaxEntropyGraphs.Graphs.indegree(G)
            d_out = MaxEntropyGraphs.Graphs.outdegree(G)
            # simple model, directly from graph, different precisions
            for precision in allowedDataTypes
                model = DBCM(G, precision=precision)
                @test isa(model, DBCM)
                @test typeof(model).parameters[2] == precision
                @test MaxEntropyGraphs.precision(model) == precision
                @test typeof(model).parameters[1] == typeof(G)
                @test all([eltype(model.θᵣ) == precision, eltype(model.xᵣ) == precision])
            end
            # simple model, directly from degree sequence, different precisions
            for precision in allowedDataTypes
                model = DBCM(d_in=d_in, d_out=d_out, precision=precision)
                @test isa(model, DBCM)
                @test typeof(model).parameters[2] == precision
                @test MaxEntropyGraphs.precision(model) == precision
                @test typeof(model).parameters[1] == Nothing
                @test all([eltype(model.θᵣ) == precision, eltype(model.xᵣ) == precision])
            end
            ## testing breaking conditions
            @test_throws MethodError DBCM(1) # wrong input type
            # directed graph info loss warning message
            G_und = MaxEntropyGraphs.Graphs.SimpleGraph(G)
            @test_logs (:warn,"The graph is undirected, while the DBCM model is directed, the in- and out-degree will be the same") DBCM(G_und)
            # weighted graph info loss warning message
            Gw = MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedDiGraph(G)
            @test_logs (:warn,"The graph is weighted, while DBCM model is unweighted, the weight information will be lost") DBCM(Gw)
            # graph problems
            @test_throws ArgumentError DBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(0)) # zero node graph
            @test_throws ArgumentError DBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(1)) # single node graph
            @test_throws DimensionMismatch DBCM(G, d_in=d_in[1:end-1], d_out=d_out)
            @test_throws DimensionMismatch DBCM(G, d_in=d_in, d_out=d_out[1:end-1])
            # degree sequences problems
            @test_throws ArgumentError DBCM(d_in=Int[], d_out=Int[])
            @test_throws ArgumentError DBCM(d_in=Int[1], d_out=Int[1])
            d_inb = copy(d_in); d_inb[end] = length(d_in) + 1
            @test_throws DomainError DBCM(d_in=d_inb, d_out=d_out) # degree out of range
            d_outb = copy(d_out); d_outb[end] = length(d_out) + 1
            @test_throws DomainError DBCM(d_in=d_in, d_out=d_outb) # degree out of range
        end

        @testset "DBCM - Likelihood gradient test" begin
            G = MaxEntropyGraphs.taro_exchange()
            model = MaxEntropyGraphs.DBCM(G)
            θ₀ = MaxEntropyGraphs.initial_guess(model)
            ∇L_buf = zeros(length(θ₀))
            ∇L_buf_min = zeros(length(θ₀))
            x_buf = zeros(length(model.xᵣ))
            y_buf = zeros(length(model.yᵣ))
            MaxEntropyGraphs.∇L_DBCM_reduced!(∇L_buf, θ₀, model.dᵣ_out, model.dᵣ_in,  model.f, model.dᵣ_in_nz, model.dᵣ_out_nz, x_buf, y_buf, model.status[:d_unique])
            MaxEntropyGraphs.∇L_DBCM_reduced_minus!(∇L_buf_min, θ₀, model.dᵣ_out, model.dᵣ_in,  model.f, model.dᵣ_in_nz, model.dᵣ_out_nz, x_buf, y_buf, model.status[:d_unique])
            @test ∇L_buf ≈ -∇L_buf_min
        end
        
        @testset "DBCM - parameter computation" begin
            #G = MaxEntropyGraphs.taro_exchange()
            G = MaxEntropyGraphs.maspalomas()
            d_in = MaxEntropyGraphs.Graphs.indegree(G)
            d_out = MaxEntropyGraphs.Graphs.outdegree(G)
            # simple model, directly from graph, different precisions
            allowedDataTypes = [Float64]
            for precision in allowedDataTypes
                @testset "$(precision) precision" begin
                    model = MaxEntropyGraphs.DBCM(G, precision=precision)
                    @test_throws ArgumentError Ĝ(model)
                    @test_throws ArgumentError σˣ(model)
                    @test_throws ArgumentError MaxEntropyGraphs.set_xᵣ!(model)
                    @test_throws ArgumentError MaxEntropyGraphs.set_yᵣ!(model)
                    @test_throws ArgumentError MaxEntropyGraphs.initial_guess(model, method=:strange)
                    for initial in [:degrees, :degrees_minor, :chung_lu] # :random, :uniform have convergence issues for this graph
                        @test length(MaxEntropyGraphs.initial_guess(model, method=initial)) == model.status[:d_unique] * 2
                        for (method, analytical_gradient) in [(:BFGS, true), (:BFGS, false), (:fixedpoint, false), (:fixedpoint, true)]
                            @testset "initials: $initial, method: $method, analytical_gradient: $analytical_gradient" begin
                                MaxEntropyGraphs.solve_model!(model, initial=initial, method=method, analytical_gradient=analytical_gradient)
                                A = MaxEntropyGraphs.Ĝ(model)
                                # check that constraints are respected
                                @test reshape(sum(A, dims=2),:,1) ≈ d_out
                                @test reshape(sum(A, dims=1),:,1) ≈ d_in
                            end
                        end
                    end
                end
            end
        end

        @testset "DBCM - sampling" begin
            model = DBCM(MaxEntropyGraphs.maspalomas())
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

        @testset "DBCM - degree metrics" begin
            model = DBCM(MaxEntropyGraphs.maspalomas())
            # adjacency matric
            @test iszero(MaxEntropyGraphs.A(model,1,1))
            @test isa(MaxEntropyGraphs.A(model,1,2), precision(model))
            # parameters not computed yet
            @test_throws ArgumentError degree(model, 1)
            solve_model!(model)
            # check out of bounds
            @test_throws ArgumentError degree(model, length(model.d_out) + 1)
            # check methods
            @test_throws ArgumentError degree(model, method=:unknown_method)
            # check that the degree is correct
            for method in [:reduced, :full]
                @test isapprox(outdegree(model, method=method), model.d_out)
                @test isapprox(indegree( model, method=method), model.d_in)
            end
            # check precompute
            @test_throws ArgumentError degree(model, method=:adjacency)
            set_Ĝ!(model)
            @test isapprox(outdegree(model, method=:adjacency), model.d_out)
            @test isapprox(indegree( model, method=:adjacency), model.d_in)
        end

        @testset "DBCM - (B/A)IC(c)" begin
            model = DBCM(MaxEntropyGraphs.maspalomas())
            # parameters not computed yet
            @test_throws ArgumentError MaxEntropyGraphs.AIC(model)
            @test_throws ArgumentError MaxEntropyGraphs.AICc(model)
            @test_throws ArgumentError MaxEntropyGraphs.BIC(model)
            solve_model!(model)
            # check warning
            @test_logs (:warn, "The number of observations is small with respect to the number of parameters (n/k < 40). Consider using the corrected AIC (AICc) instead.") MaxEntropyGraphs.AIC(model)
            # test types
            @test isa(MaxEntropyGraphs.AIC(model), precision(model))
            @test isa(MaxEntropyGraphs.AICc(model), precision(model))
            @test isa(MaxEntropyGraphs.BIC(model), precision(model))
        end

        @testset "DBCM - adjacency matrix variance" begin
            model = DBCM(MaxEntropyGraphs.maspalomas())
            solve_model!(model)
            # parameters not computed yet
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum)
            MaxEntropyGraphs.set_Ĝ!(model)
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum)
            MaxEntropyGraphs.set_σ!(model)
            # unknown autodiff method
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum, gradient_method=:unknown_method)
            # normal functioning
            for method in [:ForwardDiff, :ReverseDiff, :Zygote]
                @testset "gradient_method: $(method)" begin
                    @assert MaxEntropyGraphs.σₓ(model, sum, gradient_method=method) ≈ sqrt(sum(model.σ .^ 2))
                end
            end
        end
    end

    @testset "BiCM" begin
        @testset "BiCM - generation" begin
            allowedDataTypes = [Float64; Float32; Float16]
            G = MaxEntropyGraphs.corporateclub()
            membership = MaxEntropyGraphs.Graphs.bipartite_map(G)
            ⊥nodes, ⊤nodes = findall(membership .== 1), findall(membership .== 2)
            d⊥ = MaxEntropyGraphs.Graphs.degree(G, ⊥nodes)
            d⊤ = MaxEntropyGraphs.Graphs.degree(G, ⊤nodes)
            # simple model, directly from graph, different precisions
            for precision in allowedDataTypes
                model = BiCM(G, precision=precision)
                @test isa(model, BiCM)
                @test typeof(model).parameters[2] == precision
                @test MaxEntropyGraphs.precision(model) == precision
                @test typeof(model).parameters[1] == typeof(G)
                @test all([eltype(model.θᵣ) == precision, eltype(model.xᵣ) == precision, eltype(model.yᵣ) == precision])
            end
            # simple model, directly from degree sequences, different precisions
            for precision in allowedDataTypes
                model = BiCM(d⊥=d⊥, d⊤=d⊤, precision=precision)
                @test isa(model, BiCM)
                @test typeof(model).parameters[2] == precision
                @test MaxEntropyGraphs.precision(model) == precision
                @test typeof(model).parameters[1] == Nothing
                @test all([eltype(model.θᵣ) == precision, eltype(model.xᵣ) == precision, eltype(model.yᵣ) == precision])
            end
            # testing breaking conditions
            @test_throws MethodError BiCM(1) # wrong input type
            @test_throws ArgumentError BiCM(MaxEntropyGraphs.Graphs.SimpleGraph(0)) # zero node graph
            @test_throws ArgumentError BiCM(MaxEntropyGraphs.Graphs.SimpleGraph(1)) # single node graph
            # directed graph info loss warning message
            Gd = MaxEntropyGraphs.Graphs.SimpleDiGraph(G)
            @test_logs (:warn,"The graph is directed, while the BiCM model is undirected, the directional information will be lost") BiCM(Gd, d⊥=MaxEntropyGraphs.Graphs.outdegree(Gd,⊥nodes), d⊤=MaxEntropyGraphs.Graphs.outdegree(Gd,⊤nodes))
            # weighted graph info loss warning message
            Gw = MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(G)
            @test_logs (:warn,"The graph is weighted, while the BiCM model is unweighted, the weight information will be lost") BiCM(Gw)
            
            # coherence degrees
            @test_throws ArgumentError BiCM(d⊥=Int[], d⊤=d⊤)
            @test_throws ArgumentError BiCM(d⊥=d⊥, d⊤=Int[])
            @test_throws ArgumentError BiCM(d⊥=d⊥[1:1], d⊤=d⊤)
            @test_throws ArgumentError BiCM(d⊥=d⊥, d⊤=d⊤[1:1])
            
            # different lengths
            @test_throws DimensionMismatch BiCM(G; d⊥=d⊥[1:end-1], d⊤=d⊤)
            
        end
        
        @testset "BiCM - log-Likelihood" begin
            G = MaxEntropyGraphs.corporateclub()
            model = MaxEntropyGraphs.BiCM(G)
            θ₀ = ones(size(MaxEntropyGraphs.initial_guess(model))) .* Inf
            @assert MaxEntropyGraphs.L_BiCM_reduced(θ₀, model.d⊥ᵣ, model.d⊤ᵣ, model.f⊥, model.f⊤, model.d⊥ᵣ_nz, model.d⊤ᵣ_nz, model.status[:d⊥_unique] ) ≈ -Inf
        end

        @testset "BiCM - log-likelihood gradient" begin
            G = MaxEntropyGraphs.corporateclub()
            model = MaxEntropyGraphs.BiCM(G)
            θ₀ = ones(size(MaxEntropyGraphs.initial_guess(model)))
            # buffers
            ∇L_buf = zeros(length(θ₀))
            ∇L_buf_min = zeros(length(θ₀))
            x_buf = zeros(length(model.xᵣ))
            y_buf = zeros(length(model.yᵣ))
            # gradient
            MaxEntropyGraphs.∇L_BiCM_reduced!(∇L_buf, θ₀, model.d⊥ᵣ, model.d⊤ᵣ, model.f⊥, model.f⊤, model.d⊥ᵣ_nz, model.d⊤ᵣ_nz, x_buf, y_buf, model.status[:d⊥_unique])
            MaxEntropyGraphs.∇L_BiCM_reduced_minus!(∇L_buf_min, θ₀, model.d⊥ᵣ, model.d⊤ᵣ, model.f⊥, model.f⊤, model.d⊥ᵣ_nz, model.d⊤ᵣ_nz, x_buf, y_buf, model.status[:d⊥_unique])
            @test ∇L_buf ≈ -∇L_buf_min
            # AD gradient equivalence
            ∇L_zyg = MaxEntropyGraphs.Zygote.gradient(θ -> MaxEntropyGraphs.L_BiCM_reduced(θ, model.d⊥ᵣ, model.d⊤ᵣ, model.f⊥, model.f⊤, model.d⊥ᵣ_nz, model.d⊤ᵣ_nz, model.status[:d⊥_unique]), θ₀)[1]
            @test ∇L_zyg ≈ ∇L_buf
            @test ∇L_zyg ≈ -∇L_buf_min
        end

        @testset "BiCM - parameter computation" begin
            allowedDataTypes = [Float64] # ; Float32; Float16 kept out for ocasional convergence issues
            G = MaxEntropyGraphs.corporateclub()
            membership = MaxEntropyGraphs.Graphs.bipartite_map(G)
            ⊥nodes, ⊤nodes = findall(membership .== 1), findall(membership .== 2)
            d⊥ = MaxEntropyGraphs.Graphs.degree(G, ⊥nodes)
            d⊤ = MaxEntropyGraphs.Graphs.degree(G, ⊤nodes)
            # simple model, directly from graph, different precisions
            for precision in allowedDataTypes
                @testset "$(precision) precision" begin
                    model = BiCM(G, precision=precision)
                    @test_throws ArgumentError Ĝ(model)
                    #@test_throws ArgumentError σˣ(model)
                    @test_throws ArgumentError MaxEntropyGraphs.set_xᵣ!(model)
                    @test_throws ArgumentError MaxEntropyGraphs.set_yᵣ!(model)
                    @test_throws ArgumentError MaxEntropyGraphs.initial_guess(model, method=:strange)
                    for initial in [:degrees, :uniform, :random, :chung_lu]
                        @testset "initial guess: $initial" begin
                            @test length(MaxEntropyGraphs.initial_guess(model, method=initial)) == model.status[:d⊥_unique] + model.status[:d⊤_unique]
                            for (method, analytical_gradient) in [(:BFGS, true), (:BFGS, false), (:fixedpoint, false), (:fixedpoint, true)]
                                @testset "initials: $initial, method: $method, analytical_gradient: $analytical_gradient" begin
                                    MaxEntropyGraphs.solve_model!(model, initial=initial, method=method, analytical_gradient=analytical_gradient)
                                    A = MaxEntropyGraphs.Ĝ(model)
                                    # check that constraints are respected
                                    @test reshape(sum(A, dims=2),:,1) ≈ d⊥
                                    @test reshape(sum(A, dims=1),:,1) ≈ d⊤
                                end
                            end
                        end
                    end
                end
            end
        end

        @testset "BiCM - sampling" begin
            model = BiCM(MaxEntropyGraphs.corporateclub())
            # parameters unknown
            @test_throws ArgumentError MaxEntropyGraphs.rand(model, precomputed=false)
            @test_throws ArgumentError MaxEntropyGraphs.Ĝ(model)
            #@test_throws ArgumentError MaxEntropyGraphs.σˣ(model)
            # solve model
            MaxEntropyGraphs.solve_model!(model)
            # parameters known, but G not set
            @test_throws ArgumentError MaxEntropyGraphs.rand(model, precomputed=true)
            @test MaxEntropyGraphs.Graphs.nv(rand(model)) == MaxEntropyGraphs.Graphs.nv(model.G)
            MaxEntropyGraphs.set_Ĝ!(model)
            #MaxEntropyGraphs.set_σ!(model)
            #@test eltype(MaxEntropyGraphs.σˣ(model)) == MaxEntropyGraphs.precision(model)
            #@test size(MaxEntropyGraphs.σˣ(model)) == (MaxEntropyGraphs.Graphs.nv(model.G),MaxEntropyGraphs.Graphs.nv(model.G))
            @test eltype(MaxEntropyGraphs.Ĝ(model)) == MaxEntropyGraphs.precision(model)
            @test size(MaxEntropyGraphs.Ĝ(model)) == (length(model.d⊥),length(model.d⊤ ))
            # sampling
            S = rand(model, 100)
            @test length(S) == 100
            @test all(MaxEntropyGraphs.Graphs.nv.(S) .== MaxEntropyGraphs.Graphs.nv(model.G))
        end

        @testset "BiCM - degree metrics" begin
            G = MaxEntropyGraphs.corporateclub()
            model = BiCM(G)
            # bi-adjacency matric
            @test isa(MaxEntropyGraphs.A(model,1,1), precision(model))
            # parameters not computed yet
            @test_throws ArgumentError degree(model, 1)
            solve_model!(model)
            # check out of bounds
            @test_throws ArgumentError degree(model, length(model.d⊥) + length(model.d⊤) + 1)
            # check methods
            @test_throws ArgumentError degree(model, method=:unknown_method)
            # check that the degree is correct
            for method in [:reduced, :full]
                @test isapprox(degree(model, method=method), MaxEntropyGraphs.Graphs.degree(G))
            end
            # check precompute
            @test_throws ArgumentError degree(model, method=:adjacency)
            set_Ĝ!(model)
            @test isapprox(degree(model, method=:adjacency), MaxEntropyGraphs.Graphs.degree(G))
        end

        @testset "BiCM - (B/A)IC(c)" begin
            model = BiCM(MaxEntropyGraphs.corporateclub())
            # parameters not computed yet
            @test_throws ArgumentError MaxEntropyGraphs.AIC(model)
            @test_throws ArgumentError MaxEntropyGraphs.AICc(model)
            @test_throws ArgumentError MaxEntropyGraphs.BIC(model)
            solve_model!(model)
            # check warning
            @test_logs (:warn, "The number of observations is small with respect to the number of parameters (n/k < 40). Consider using the corrected AIC (AICc) instead.") MaxEntropyGraphs.AIC(model)
            # test types
            @test isa(MaxEntropyGraphs.AIC(model), precision(model))
            @test isa(MaxEntropyGraphs.AICc(model), precision(model))
            @test isa(MaxEntropyGraphs.BIC(model), precision(model))

        end

    end

    @testset "UECM" begin

    end
end

    #=
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
    =#
    #=
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
    =#



