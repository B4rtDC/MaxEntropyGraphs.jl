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
            # a_ij ≡ a_ji: the identity-covariance cross-term doubles the variance of a full-matrix metric
            n = length(model.dᵣ_ind)
            U = [j > i ? 1.0 : 0.0 for i in 1:n, j in 1:n]  # strict upper-triangle mask (one slot per dyad)
            for method in [:ForwardDiff; :ReverseDiff; :Zygote]
                @testset "gradient_method: $(method)" begin
                    @test MaxEntropyGraphs.σₓ(model, sum, gradient_method=method) ≈ sqrt(2 * sum(model.σ .^ 2))
                    # one-slot-per-dyad metric: no cross-term contribution
                    @test MaxEntropyGraphs.σₓ(model, A -> sum(A .* U), gradient_method=method) ≈ sqrt(sum((model.σ .* U) .^ 2))
                    # convention independence: X(A) = sum(A) counts each dyad twice
                    @test MaxEntropyGraphs.σₓ(model, sum, gradient_method=method) ≈ 2 * MaxEntropyGraphs.σₓ(model, A -> sum(A .* U), gradient_method=method)
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

    @testset "RBCM" begin
        @testset "RBCM - generation" begin
            allowedDataTypes = [Float64; Float32; Float16]
            G = MaxEntropyGraphs.Graphs.SimpleDiGraph(MaxEntropyGraphs.rhesus_macaques())
            d_out = MaxEntropyGraphs.nonreciprocated_outdegree(G)
            d_in  = MaxEntropyGraphs.nonreciprocated_indegree(G)
            d_rec = MaxEntropyGraphs.reciprocated_degree(G)
            # simple model, directly from graph, different precisions
            for precision in allowedDataTypes
                model = RBCM(G, precision=precision)
                @test isa(model, RBCM)
                @test typeof(model).parameters[2] == precision
                @test MaxEntropyGraphs.precision(model) == precision
                @test typeof(model).parameters[1] == typeof(G)
                @test all([eltype(model.θᵣ) == precision, eltype(model.xᵣ) == precision, eltype(model.zᵣ) == precision])
            end
            # simple model, directly from degree sequences, different precisions
            for precision in allowedDataTypes
                model = RBCM(d_out=d_out, d_in=d_in, d_rec=d_rec, precision=precision)
                @test isa(model, RBCM)
                @test typeof(model).parameters[2] == precision
                @test MaxEntropyGraphs.precision(model) == precision
                @test typeof(model).parameters[1] == Nothing
                @test all([eltype(model.θᵣ) == precision, eltype(model.xᵣ) == precision, eltype(model.zᵣ) == precision])
            end
            # the model stores the reciprocal (NOT the ordinary) degree sequences
            model = RBCM(G)
            @test model.d_out == d_out
            @test model.d_in == d_in
            @test model.d_rec == d_rec
            @test model.d_out .+ model.d_rec == MaxEntropyGraphs.Graphs.outdegree(G)
            ## testing breaking conditions
            @test_throws MaxEntropyGraphs.Graphs.NotImplementedError RBCM(1) # wrong input type (fails in the kwarg defaults)
            # undirected graph -> fully reciprocated warning
            G_und = MaxEntropyGraphs.Graphs.SimpleGraph(G)
            @test_logs (:warn,"The graph is undirected, while the RBCM model is directed; every edge will be considered reciprocated (k→ = k← = 0)") (:warn, "The non-reciprocated degree sequences are all zeros (fully reciprocal network): only the reciprocated parameters are identified.") RBCM(G_und)
            # weighted graph info loss warning message
            Gw = MaxEntropyGraphs.rhesus_macaques()
            @test_logs (:warn,"The graph is weighted, while the RBCM model is unweighted, the weight information will be lost") RBCM(Gw)
            # fully reciprocal directed graph warning
            @test_logs (:warn, "The non-reciprocated degree sequences are all zeros (fully reciprocal network): only the reciprocated parameters are identified.") RBCM(MaxEntropyGraphs.taro_exchange())
            # zero-reciprocity warning (star digraph has no reciprocated links)
            G_star = MaxEntropyGraphs.Graphs.SimpleDiGraph(5)
            for j in 2:5; MaxEntropyGraphs.Graphs.add_edge!(G_star, 1, j); end
            @test_logs (:warn, "The reciprocated degree sequence is all zeros: the RBCM degenerates to a DBCM with an additional (unidentified) parameter set. Consider using the DBCM instead.") RBCM(G_star)
            # graph problems
            @test_throws ArgumentError RBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(0)) # zero node graph
            @test_throws ArgumentError RBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(1)) # single node graph
            @test_throws DimensionMismatch RBCM(G, d_out=d_out[1:end-1], d_in=d_in, d_rec=d_rec)
            @test_throws DimensionMismatch RBCM(G, d_out=d_out, d_in=d_in[1:end-1], d_rec=d_rec)
            @test_throws DimensionMismatch RBCM(G, d_out=d_out, d_in=d_in, d_rec=d_rec[1:end-1])
            # degree sequences problems
            @test_throws ArgumentError RBCM(d_out=Int[], d_in=Int[], d_rec=Int[])
            @test_throws ArgumentError RBCM(d_out=Int[1], d_in=Int[1], d_rec=Int[1])
            @test_throws DomainError RBCM(d_out=[1, 0], d_in=[0, 0], d_rec=[0, 0]) # unbalanced non-reciprocated stubs
            @test_throws DomainError RBCM(d_out=[0, 0, 0], d_in=[0, 0, 0], d_rec=[1, 1, 1]) # odd total reciprocated stubs
            @test_throws DomainError RBCM(d_out=[-1, 1], d_in=[1, -1], d_rec=[0, 0]) # negative degrees
            d_recb = copy(d_rec); d_recb[1] = length(d_rec) # k→ + k← + k↔ exceeds N-1
            @test_throws DomainError RBCM(d_out=d_out, d_in=d_in, d_rec=d_recb)
        end

        @testset "RBCM - Likelihood gradient test" begin
            G = MaxEntropyGraphs.Graphs.SimpleDiGraph(MaxEntropyGraphs.rhesus_macaques())
            model = MaxEntropyGraphs.RBCM(G)
            n = model.status[:d_unique]
            # analytical gradient == minus the minus-gradient
            θ₀ = MaxEntropyGraphs.initial_guess(model, method=:uniform)
            ∇L_buf = zeros(length(θ₀))
            ∇L_buf_min = zeros(length(θ₀))
            x_buf, y_buf, z_buf = zeros(n), zeros(n), zeros(n)
            MaxEntropyGraphs.∇L_RBCM_reduced!(∇L_buf, θ₀, model.dᵣ_out, model.dᵣ_in, model.dᵣ_rec, model.f, model.dᵣ_out_nz, model.dᵣ_in_nz, model.dᵣ_rec_nz, x_buf, y_buf, z_buf, n)
            MaxEntropyGraphs.∇L_RBCM_reduced_minus!(∇L_buf_min, θ₀, model.dᵣ_out, model.dᵣ_in, model.dᵣ_rec, model.f, model.dᵣ_out_nz, model.dᵣ_in_nz, model.dᵣ_rec_nz, x_buf, y_buf, z_buf, n)
            @test ∇L_buf ≈ -∇L_buf_min
            # analytical gradient == autodiff gradient of the likelihood at a feasible point
            ∇L_AD = MaxEntropyGraphs.ForwardDiff.gradient(θ -> MaxEntropyGraphs.L_RBCM_reduced(θ, model.dᵣ_out, model.dᵣ_in, model.dᵣ_rec, model.f, model.dᵣ_out_nz, model.dᵣ_in_nz, model.dᵣ_rec_nz, n), θ₀)
            @test isapprox(∇L_buf, ∇L_AD, rtol=1e-10)
            # log1pexpsum: numerical stability of the dyadic normaliser
            @test MaxEntropyGraphs.log1pexpsum(0.0, 0.0, 0.0) ≈ log(4)
            @test MaxEntropyGraphs.log1pexpsum(-1.2, 0.3, 2.1) ≈ log(1 + exp(-1.2) + exp(0.3) + exp(2.1))
            @test MaxEntropyGraphs.log1pexpsum(700.0, -700.0, 0.0) ≈ 700.0 # no overflow for large positive arguments
            @test MaxEntropyGraphs.log1pexpsum(-Inf, -Inf, -Inf) == 0.0   # pinned channels contribute exact zeros
            @test MaxEntropyGraphs.log1pexpsum(-Inf, 1.0, -Inf) ≈ log(1 + exp(1.0))
        end

        @testset "RBCM - parameter computation" begin
            G = MaxEntropyGraphs.Graphs.SimpleDiGraph(MaxEntropyGraphs.rhesus_macaques())
            for precision in [Float64]
                @testset "$(precision) precision" begin
                    model = MaxEntropyGraphs.RBCM(G, precision=precision)
                    @test_throws ArgumentError Ĝ(model)
                    @test_throws ArgumentError σˣ(model)
                    @test_throws ArgumentError MaxEntropyGraphs.set_xᵣ!(model)
                    @test_throws ArgumentError MaxEntropyGraphs.set_yᵣ!(model)
                    @test_throws ArgumentError MaxEntropyGraphs.set_zᵣ!(model)
                    @test_throws ArgumentError MaxEntropyGraphs.initial_guess(model, method=:strange)
                    @test_throws ArgumentError MaxEntropyGraphs.solve_model!(model, method=:strange)
                    for initial in [:degrees, :degrees_minor, :chung_lu]
                        @test length(MaxEntropyGraphs.initial_guess(model, method=initial)) == model.status[:d_unique] * 3
                        # analytical-gradient runs are scoped to the :degrees initial guess (poorer guesses
                        # can stall the line search on emulated x86_64 CI runners)
                        methods = initial == :degrees ? [(:BFGS, true), (:BFGS, false), (:fixedpoint, false), (:fixedpoint, true)] :
                                                        [(:BFGS, false), (:fixedpoint, false)]
                        for (method, analytical_gradient) in methods
                            @testset "initials: $initial, method: $method, analytical_gradient: $analytical_gradient" begin
                                MaxEntropyGraphs.solve_model!(model, initial=initial, method=method, analytical_gradient=analytical_gradient, g_tol=1e-8)
                                # check that the three reciprocal degree constraints are respected
                                @test isapprox(MaxEntropyGraphs.nonreciprocated_outdegree(model), model.d_out, rtol=1e-6)
                                @test isapprox(MaxEntropyGraphs.nonreciprocated_indegree(model),  model.d_in,  rtol=1e-6)
                                @test isapprox(MaxEntropyGraphs.reciprocated_degree(model),       model.d_rec, rtol=1e-6)
                            end
                        end
                    end
                end
            end
            # zero-valued constraints (maspalomas: most nodes have k↔ = 0 -> pinned γ parameters)
            @testset "zero-valued constraints" begin
                model = MaxEntropyGraphs.RBCM(MaxEntropyGraphs.maspalomas())
                MaxEntropyGraphs.solve_model!(model)
                @test any(isinf, model.θᵣ)                # pinned parameters at +Inf
                @test any(iszero, model.zᵣ)               # corresponding fitnesses exactly zero
                @test isapprox(MaxEntropyGraphs.nonreciprocated_outdegree(model), model.d_out, rtol=1e-6)
                @test isapprox(MaxEntropyGraphs.nonreciprocated_indegree(model),  model.d_in,  rtol=1e-6)
                @test isapprox(MaxEntropyGraphs.reciprocated_degree(model),       model.d_rec, rtol=1e-6)
                # regression: the pinned set is derived from the CONSTRAINTS, not from the initial
                # guess — a finite initial guess (:uniform/:random) must not leave junk in the
                # (deliberately flat) dead coordinates
                model = MaxEntropyGraphs.RBCM(MaxEntropyGraphs.maspalomas())
                MaxEntropyGraphs.solve_model!(model, initial=:uniform, method=:BFGS)
                @test all(iszero, model.zᵣ[model.dᵣ_rec .== 0])
                @test isapprox(MaxEntropyGraphs.reciprocated_degree(model), model.d_rec, rtol=1e-6)
            end
            # degenerate fully reciprocal network: accelerated fixed point can overshoot, gradient methods work
            @testset "fully reciprocal network" begin
                model = MaxEntropyGraphs.RBCM(MaxEntropyGraphs.taro_exchange())
                @test_throws Exception MaxEntropyGraphs.solve_model!(model) # anderson overshoots to non-finite values
                model = MaxEntropyGraphs.RBCM(MaxEntropyGraphs.taro_exchange())
                MaxEntropyGraphs.solve_model!(model, method=:BFGS)
                @test isapprox(MaxEntropyGraphs.reciprocated_degree(model), model.d_rec, rtol=1e-6)
            end
        end

        @testset "RBCM - sampling" begin
            model = RBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(MaxEntropyGraphs.rhesus_macaques()))
            # parameters unknown
            @test_throws ArgumentError MaxEntropyGraphs.rand(model, precomputed=false)
            @test_throws ArgumentError MaxEntropyGraphs.Ĝ(model)
            @test_throws ArgumentError MaxEntropyGraphs.σˣ(model)
            # solve model
            MaxEntropyGraphs.solve_model!(model)
            # precomputed sampling is deliberately unsupported (Ĝ cannot capture the dyadic joint distribution)
            @test_throws ArgumentError MaxEntropyGraphs.rand(model, precomputed=true)
            @test MaxEntropyGraphs.Graphs.nv(rand(model)) == MaxEntropyGraphs.Graphs.nv(model.G)
            MaxEntropyGraphs.set_Ĝ!(model)
            MaxEntropyGraphs.set_σ!(model)
            @test eltype(MaxEntropyGraphs.σˣ(model)) == MaxEntropyGraphs.precision(model)
            @test size(MaxEntropyGraphs.σˣ(model)) == (MaxEntropyGraphs.Graphs.nv(model.G),MaxEntropyGraphs.Graphs.nv(model.G))
            @test eltype(MaxEntropyGraphs.Ĝ(model)) == MaxEntropyGraphs.precision(model)
            @test size(MaxEntropyGraphs.Ĝ(model)) == (MaxEntropyGraphs.Graphs.nv(model.G),MaxEntropyGraphs.Graphs.nv(model.G))
            # sampling
            S = rand(model, 100)
            @test length(S) == 100
            @test all(MaxEntropyGraphs.Graphs.nv.(S) .== MaxEntropyGraphs.Graphs.nv(model.G))
            @test all(MaxEntropyGraphs.Graphs.is_directed.(S))
        end

        @testset "RBCM - degree metrics" begin
            model = RBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(MaxEntropyGraphs.rhesus_macaques()))
            # parameters not computed yet
            @test_throws ArgumentError degree(model, 1)
            @test_throws ArgumentError MaxEntropyGraphs.nonreciprocated_outdegree(model, 1)
            solve_model!(model)
            # adjacency and dyadic probability accessors
            @test iszero(MaxEntropyGraphs.A(model,1,1))
            @test isa(MaxEntropyGraphs.A(model,1,2), precision(model))
            @test MaxEntropyGraphs.A(model,1,2) ≈ MaxEntropyGraphs.p⭢(model,1,2) + MaxEntropyGraphs.p⭤(model,1,2)
            @test MaxEntropyGraphs.p⭠(model,1,2) == MaxEntropyGraphs.p⭢(model,2,1)
            @test MaxEntropyGraphs.p⭢(model,1,2) + MaxEntropyGraphs.p⭠(model,1,2) + MaxEntropyGraphs.p⭤(model,1,2) + MaxEntropyGraphs.p∅(model,1,2) ≈ 1.0
            # check out of bounds
            @test_throws ArgumentError degree(model, length(model.d_out) + 1)
            @test_throws ArgumentError MaxEntropyGraphs.reciprocated_degree(model, length(model.d_out) + 1)
            # check methods
            @test_throws ArgumentError degree(model, method=:unknown_method)
            @test_throws ArgumentError MaxEntropyGraphs.reciprocated_degree(model, method=:unknown_method)
            # check that the reciprocal degrees are correct
            for method in [:reduced, :full]
                @test isapprox(MaxEntropyGraphs.nonreciprocated_outdegree(model, method=method), model.d_out, rtol=1e-6)
                @test isapprox(MaxEntropyGraphs.nonreciprocated_indegree(model,  method=method), model.d_in,  rtol=1e-6)
                @test isapprox(MaxEntropyGraphs.reciprocated_degree(model,       method=method), model.d_rec, rtol=1e-6)
                # totals align with the Graphs.jl definitions
                @test isapprox(outdegree(model, method=method), model.d_out .+ model.d_rec, rtol=1e-6)
                @test isapprox(indegree(model,  method=method), model.d_in  .+ model.d_rec, rtol=1e-6)
            end
            # the reciprocal degree split cannot be recovered from Ĝ
            @test_throws ArgumentError MaxEntropyGraphs.nonreciprocated_outdegree(model, method=:adjacency)
            @test_throws ArgumentError MaxEntropyGraphs.nonreciprocated_indegree(model, method=:adjacency)
            @test_throws ArgumentError MaxEntropyGraphs.reciprocated_degree(model, method=:adjacency)
            # the totals can (once Ĝ is set)
            @test_throws ArgumentError degree(model, method=:adjacency)
            set_Ĝ!(model)
            @test isapprox(outdegree(model, method=:adjacency), model.d_out .+ model.d_rec, rtol=1e-6)
            @test isapprox(indegree( model, method=:adjacency), model.d_in  .+ model.d_rec, rtol=1e-6)
            # model reciprocity reproduces the observed value; the DBCM baseline does not
            G = MaxEntropyGraphs.Graphs.SimpleDiGraph(MaxEntropyGraphs.rhesus_macaques())
            @test isapprox(MaxEntropyGraphs.reciprocity(model), MaxEntropyGraphs.reciprocity(G), rtol=1e-6)
            dmodel = DBCM(G); solve_model!(dmodel)
            @test MaxEntropyGraphs.reciprocity(dmodel) < MaxEntropyGraphs.reciprocity(G)
        end

        @testset "RBCM - motifs" begin
            model = RBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(MaxEntropyGraphs.rhesus_macaques()))
            # parameters not computed yet
            @test_throws ArgumentError motifs(model)
            @test_throws ArgumentError M13(model)
            solve_model!(model)
            # batched == individual
            @test motifs(model) == [Mᵢ(model) for Mᵢ in (M1,M2,M3,M4,M5,M6,M7,M8,M9,M10,M11,M12,M13)]
            # naive reference over the dyadic probabilities
            P̂, R̂, Ẑ = MaxEntropyGraphs._dyadic_probability_matrices(model)
            @test isapprox(model.Ĝ === nothing ? MaxEntropyGraphs.Ĝ(model) : model.Ĝ, P̂ .+ R̂, rtol=1e-12)
            n = model.status[:d]
            pick = Dict(:P => (i,j) -> P̂[i,j], :Q => (i,j) -> P̂[j,i], :R => (i,j) -> R̂[i,j], :Z => (i,j) -> Ẑ[i,j])
            naive = [sum(pick[s[1]](i,j) * pick[s[2]](j,k) * pick[s[3]](k,i)
                         for i in 1:n for j in 1:n for k in 1:n
                         if (i != j && j != k && i != k)) for s in MaxEntropyGraphs._motif_specs]
            @test isapprox(motifs(model), naive, rtol=1e-10)
            # the Ĝ shortcut (valid for the DBCM) would be wrong under the RBCM
            set_Ĝ!(model)
            @test !isapprox(M13(model), M13(model.Ĝ), rtol=0.1)
        end

        @testset "RBCM - (B/A)IC(c)" begin
            model = RBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(MaxEntropyGraphs.rhesus_macaques()))
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
            # on this high-reciprocity network the RBCM beats the DBCM
            dmodel = DBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(MaxEntropyGraphs.rhesus_macaques()))
            solve_model!(dmodel)
            @test MaxEntropyGraphs.AICc(model) < MaxEntropyGraphs.AICc(dmodel)
        end

        @testset "RBCM - adjacency matrix variance" begin
            model = RBCM(MaxEntropyGraphs.Graphs.SimpleDiGraph(MaxEntropyGraphs.rhesus_macaques()))
            solve_model!(model)
            # expected values / variances not computed yet
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum)
            MaxEntropyGraphs.set_Ĝ!(model)
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum)
            MaxEntropyGraphs.set_σ!(model)
            # unknown autodiff method
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum, gradient_method=:unknown_method)
            # normal functioning: identity including the within-dyad covariance term
            C = MaxEntropyGraphs._cov_dyads(model)
            @test isapprox(C, transpose(C))
            @test all(iszero, C[MaxEntropyGraphs.diagind(C)])
            for method in [:ForwardDiff, :ReverseDiff, :Zygote]
                @testset "gradient_method: $(method)" begin
                    @test MaxEntropyGraphs.σₓ(model, sum, gradient_method=method) ≈ sqrt(sum(model.σ .^ 2) + sum(C))
                    # direction-selective metric: only row 1 -> the covariance cross-term vanishes
                    @test MaxEntropyGraphs.σₓ(model, A -> sum(A[1,:]), gradient_method=method) ≈ sqrt(sum(model.σ[1,:] .^ 2))
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

        @testset "BiCM - biadjacency matrix variance" begin
            model = BiCM(MaxEntropyGraphs.corporateclub())
            solve_model!(model)
            # expected values not computed yet
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum)
            MaxEntropyGraphs.set_Ĝ!(model)
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum)
            MaxEntropyGraphs.set_σ!(model)
            # the standard deviations have the biadjacency shape (n⊥ × n⊤)
            @test size(model.σ) == (25, 15)
            # unknown autodiff method
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum, gradient_method=:unknown_method)
            # normal functioning: the (rectangular) biadjacency entries are independent,
            # so no covariance cross-terms occur (no factor 2)
            for method in [:ForwardDiff; :ReverseDiff; :Zygote]
                @testset "gradient_method: $(method)" begin
                    @test MaxEntropyGraphs.σₓ(model, sum, gradient_method=method) ≈ sqrt(sum(model.σ .^ 2))
                end
            end

            ## Vn/Λn motif family
            # n = 2 recovers the V-motif machinery, and the delta expectation is exact for n = 2
            @test Vn_motifs(model, 2, layer=:bottom) ≈ MaxEntropyGraphs.V_motifs(model, layer=:bottom, precomputed=false)
            @test Vn_motifs(model, 2, layer=:top) ≈ MaxEntropyGraphs.V_motifs(model, layer=:top, precomputed=false)
            @test Vn_motifs(model, 2, layer=:top, method=:delta) ≈ Vn_motifs(model, 2, layer=:top, method=:exact)
            # the delta σ for n = 2 reproduces the Saracco et al. (2015) SI Eq. III.10 closed form,
            # recomputed by hand: sqrt(Σ_α ((2u_α - 1)/2)² σ²[U_α]) over the classes with u_α ≥ n.
            # CAREFUL with the layer semantics: for layer=:top the aggregation runs over the ⊥ degrees
            # (u = d⊥, σ²[U_α] = Σ_β p_αβ(1 - p_αβ) over the ⊤ nodes) and vice versa for layer=:bottom;
            # with model.σ the biadjacency (n⊥ × n⊤) entry-wise standard deviations, σ²[U] follows
            # from its row (dims=2, :top) or column (dims=1, :bottom) sums of squares.
            for (layer, u, s2) in [(:top, model.d⊥, vec(sum(model.σ .^ 2, dims=2))),
                                   (:bottom, model.d⊤, vec(sum(model.σ .^ 2, dims=1)))]
                @test Vn_sigma(model, 2, layer=layer, method=:delta) ≈ sqrt(sum(((2 * u[α] - 1) / 2)^2 * s2[α] for α in eachindex(u) if u[α] ≥ 2))
            end
            # value pins (exact Poisson-binomial machinery)
            @test isapprox(Vn_sigma(model, 3, layer=:top), 70.311, rtol=1e-2)
            @test isapprox(Vn_zscore(model, 3, layer=:top), -1.2506, rtol=1e-2)
            # the z-scores are sign-definite (one-sided tests): ⟨N_Vn⟩ ≥ N_Vn^obs under the BiCM
            for n in 2:4, layer in (:bottom, :top)
                @test Vn_zscore(model, n, layer=layer) ≤ 0
            end
            # breaking conditions: the closed forms only exist for n ∈ {2, 3, 4}, the motif order must be ≥ 2
            @test_throws ArgumentError Vn_motifs(model, 5, method=:delta)
            @test_throws ArgumentError Vn_motifs(model, 1)
            # the exact method works for any n ≥ 2
            @test Vn_sigma(model, 5, layer=:top) isa Real
            # the first-order delta method underestimates the variance for sparse layers
            @test Vn_sigma(model, 4, layer=:top, method=:delta) < Vn_sigma(model, 4, layer=:top)
        end

    end

    @testset "UECM" begin
        # small, self-contained integer-weighted undirected graph for the constructor tests
        Usrc = [1, 1, 2, 3]; Udst = [2, 3, 3, 4]; Uw = [2, 1, 3, 4]
        Gsmall = MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(Usrc, Udst, Uw)
        dsmall = MaxEntropyGraphs.Graphs.degree(Gsmall)
        ssmall = MaxEntropyGraphs.strength(Gsmall)
        # real integer-weighted undirected anchor (symmetrised rhesus macaques network, N=16)
        Gw = MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(MaxEntropyGraphs.rhesus_macaques())

        @testset "UECM - generation" begin
            allowedDataTypes = [Float64; Float32; Float16]
            # simple model, directly from graph, different precisions
            for precision in allowedDataTypes
                model = UECM(Gsmall, precision=precision)
                @test isa(model, UECM)
                @test typeof(model).parameters[2] == precision
                @test MaxEntropyGraphs.precision(model) == precision
                @test typeof(model).parameters[1] == typeof(Gsmall)
                @test all([eltype(model.θᵣ) == precision, eltype(model.xᵣ) == precision, eltype(model.yᵣ) == precision])
            end
            # simple model, directly from degree and strength sequences, different precisions
            for precision in allowedDataTypes
                model = UECM(d=dsmall, s=ssmall, precision=precision)
                @test isa(model, UECM)
                @test typeof(model).parameters[2] == precision
                @test MaxEntropyGraphs.precision(model) == precision
                @test typeof(model).parameters[1] == Nothing
                @test all([eltype(model.θᵣ) == precision, eltype(model.xᵣ) == precision, eltype(model.yᵣ) == precision])
            end
            # testing breaking conditions
            @test_throws MethodError UECM(1) # wrong input type
            # directed graph info loss warning message
            Gd = MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedDiGraph(Usrc, Udst, Uw)
            @test_logs (:warn,"The graph is directed, the UECM model is undirected, the directional information will be lost") match_mode=:any UECM(Gd, d=MaxEntropyGraphs.Graphs.degree(Gd), s=MaxEntropyGraphs.strength(Gd))
            # zero degree / zero strength node warnings
            @test_logs (:warn,"The graph has vertices with zero degree, this may lead to convergence issues.") UECM(d=[0, 1, 1], s=[1, 1, 1])
            @test_logs (:warn,"The graph has vertices with zero strength, this may lead to convergence issues.") UECM(d=[1, 1, 1], s=[0, 1, 1])
            # invalid combination of parameters
            @test_throws ArgumentError UECM(MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(0)) # zero node graph
            @test_throws ArgumentError UECM(MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(1)) # single node graph
            @test_throws DimensionMismatch UECM(Gsmall, d=dsmall[1:end-1], s=ssmall) # different lengths (graph)
            @test_throws DimensionMismatch UECM(d=[1, 2, 3], s=[1, 2]) # different lengths (sequence)
            @test_throws ArgumentError UECM(d=Int[], s=Int[]) # zero length
            @test_throws ArgumentError UECM(d=Int[1], s=Int[1]) # single length
            @test_throws DomainError UECM(d=[1.5, 2.0], s=[1, 2]) # non-integer degree
            @test_throws DomainError UECM(d=[1, 2], s=[1.5, 2.0]) # non-integer strength
            @test_throws DomainError UECM(d=[3, 1, 1], s=[1, 1, 1]) # max degree >= number of nodes
        end

        @testset "UECM - Likelihood gradient test" begin
            model = UECM(Gw)
            n = model.status[:d_unique]
            θ₀ = MaxEntropyGraphs.initial_guess(model)
            ∇L_buf = zeros(length(θ₀))
            ∇L_buf_min = zeros(length(θ₀))
            x_buff = zeros(n)
            y_buff = zeros(n)
            MaxEntropyGraphs.∇L_UECM_reduced!(∇L_buf, θ₀, model.dᵣ, model.sᵣ, model.f, x_buff, y_buff, n)
            MaxEntropyGraphs.∇L_UECM_reduced_minus!(∇L_buf_min, θ₀, model.dᵣ, model.sᵣ, model.f, x_buff, y_buff, n)
            @test ∇L_buf ≈ -∇L_buf_min
            ∇L_zyg = MaxEntropyGraphs.Zygote.gradient(θ -> MaxEntropyGraphs.L_UECM_reduced(θ, model.dᵣ, model.sᵣ, model.f, n), θ₀)[1]
            @test ∇L_zyg ≈ ∇L_buf
            @test ∇L_zyg ≈ -∇L_buf_min
            # the accessor requires computed parameters
            @test_throws ArgumentError MaxEntropyGraphs.L_UECM_reduced(model)
        end

        @testset "UECM - parameter computation" begin
            allowedDataTypes = [Float64] # low precision kept out for occasional convergence issues
            for precision in allowedDataTypes
                @testset "$(precision) precision" begin
                    model = UECM(Gw, precision=precision)
                    @test_throws ArgumentError Ĝ(model)
                    @test_throws ArgumentError MaxEntropyGraphs.Ŵ(model)
                    @test_throws ArgumentError σˣ(model)
                    @test_throws ArgumentError MaxEntropyGraphs.set_xᵣ!(model)
                    @test_throws ArgumentError MaxEntropyGraphs.set_yᵣ!(model)
                    @test_throws ArgumentError MaxEntropyGraphs.initial_guess(model, method=:strange)
                    # every initial-guess method returns a guess of length 2·d_unique ( θ = [α; β] )
                    for initial in [:strengths, :strengths_minor, :random, :uniform]
                        @test length(MaxEntropyGraphs.initial_guess(model, method=initial)) == 2 * model.status[:d_unique]
                    end
                    # convergence + constraint reproduction (fixed point is unstable for the UECM, so BFGS/Newton).
                    # The AD-gradient BFGS is robust from either initial guess. The analytical gradient is a
                    # fast path but numerically sensitive from the poorer `:strengths_minor` start: on some
                    # emulated x86_64 platforms its BackTracking line search stalls far from the optimum, so it
                    # is exercised only from the robust `:strengths` guess. (`g_tol=1e-5` stops the solve just
                    # short of the feasibility barrier — the default 1e-8 over-converges into a fragile region —
                    # while still reproducing both constraints to ~1e-5, well inside the tolerance below.)
                    for (initial, gradient_modes) in [(:strengths, [true, false]), (:strengths_minor, [false])]
                        @testset "BFGS - initial guess: $initial" begin
                            for analytical_gradient in gradient_modes
                                @testset "analytical_gradient: $analytical_gradient" begin
                                    MaxEntropyGraphs.solve_model!(model, initial=initial, method=:BFGS, analytical_gradient=analytical_gradient, g_tol=1e-5)
                                    A = MaxEntropyGraphs.Ĝ(model)
                                    W = MaxEntropyGraphs.Ŵ(model)
                                    # both the degree and the strength constraints must be reproduced
                                    @test isapprox(vec(sum(A, dims=2)), model.d, rtol=1e-6)
                                    @test isapprox(vec(sum(W, dims=2)), model.s, rtol=1e-6)
                                end
                            end
                        end
                    end
                    # Newton is more sensitive to the initial guess; test it from the default strengths guess.
                    @testset "Newton - analytical_gradient: $analytical_gradient" for analytical_gradient in [false, true]
                        MaxEntropyGraphs.solve_model!(model, initial=:strengths, method=:Newton, analytical_gradient=analytical_gradient)
                        A = MaxEntropyGraphs.Ĝ(model)
                        W = MaxEntropyGraphs.Ŵ(model)
                        @test isapprox(vec(sum(A, dims=2)), model.d, rtol=1e-6)
                        @test isapprox(vec(sum(W, dims=2)), model.s, rtol=1e-6)
                    end
                    @test all([eltype(model.θᵣ) == precision, eltype(model.xᵣ) == precision, eltype(model.yᵣ) == precision])
                    # the fixed point method emits its instability warning
                    @test_logs (:warn, "The fixed point method is very unstable for this model and should not be used. `BFGS` is prefered for quasinewton methods.") match_mode=:any try
                        MaxEntropyGraphs.solve_model!(model, method=:fixedpoint)
                    catch
                    end
                end
            end
        end

        @testset "UECM - sampling" begin
            model = UECM(Gw)
            # parameters unknown
            @test_throws ArgumentError MaxEntropyGraphs.rand(model, precomputed=false)
            @test_throws ArgumentError MaxEntropyGraphs.Ĝ(model)
            @test_throws ArgumentError MaxEntropyGraphs.σˣ(model)
            # solve model
            MaxEntropyGraphs.solve_model!(model, method=:BFGS)
            # precomputed sampling is not implemented for the UECM
            @test_throws ArgumentError MaxEntropyGraphs.rand(model, precomputed=true)
            # single sample: correct number of vertices and integer weights ≥ 1
            sample = rand(model)
            @test isa(sample, MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph)
            @test MaxEntropyGraphs.Graphs.nv(sample) == model.status[:N]
            @test all(w -> w ≥ 1, MaxEntropyGraphs.SimpleWeightedGraphs.weight.(collect(MaxEntropyGraphs.Graphs.edges(sample))))
            # expected adjacency / variance sizes and eltypes
            MaxEntropyGraphs.set_Ĝ!(model)
            MaxEntropyGraphs.set_σ!(model)
            @test eltype(MaxEntropyGraphs.σˣ(model)) == MaxEntropyGraphs.precision(model)
            @test size(MaxEntropyGraphs.σˣ(model)) == (model.status[:N], model.status[:N])
            @test eltype(MaxEntropyGraphs.Ĝ(model)) == MaxEntropyGraphs.precision(model)
            @test size(MaxEntropyGraphs.Ĝ(model)) == (model.status[:N], model.status[:N])
            # batch sampling
            S = rand(model, 100)
            @test length(S) == 100
            @test all(MaxEntropyGraphs.Graphs.nv.(S) .== model.status[:N])
            # reproducibility with a fixed rng
            @test MaxEntropyGraphs.SimpleWeightedGraphs.weights(rand(model, rng=MaxEntropyGraphs.Xoshiro(42))) == MaxEntropyGraphs.SimpleWeightedGraphs.weights(rand(model, rng=MaxEntropyGraphs.Xoshiro(42)))
        end

        @testset "UECM - degree/strength metrics" begin
            model = UECM(Gw)
            # adjacency matrix accessor
            @test iszero(MaxEntropyGraphs.A(model, 1, 1))
            @test isa(MaxEntropyGraphs.A(model, 1, 2), precision(model))
            # parameters not computed yet
            @test_throws ArgumentError degree(model, 1)
            @test_throws ArgumentError MaxEntropyGraphs.strength(model, 1)
            solve_model!(model, method=:BFGS, g_tol=1e-5)
            # check out of bounds
            @test_throws ArgumentError degree(model, model.status[:N] + 1)
            @test_throws ArgumentError MaxEntropyGraphs.strength(model, model.status[:N] + 1)
            # unknown method
            @test_throws ArgumentError degree(model, method=:unknown_method)
            @test_throws ArgumentError MaxEntropyGraphs.strength(model, method=:unknown_method)
            # reduced/full reproduce the observed degree and strength sequences
            for method in [:reduced, :full]
                @test isapprox(degree(model, method=method), model.d, rtol=1e-6)
                @test isapprox(MaxEntropyGraphs.strength(model, method=method), model.s, rtol=1e-6)
            end
            # adjacency-based methods require set_Ĝ!
            @test_throws ArgumentError degree(model, method=:adjacency)
            @test_throws ArgumentError MaxEntropyGraphs.strength(model, method=:adjacency)
            set_Ĝ!(model)
            @test isapprox(degree(model, method=:adjacency), model.d, rtol=1e-6)
            @test isapprox(MaxEntropyGraphs.strength(model, method=:adjacency), model.s, rtol=1e-6)
        end

        @testset "UECM - (B/A)IC(c)" begin
            model = UECM(Gw)
            # parameters not computed yet
            @test_throws ArgumentError MaxEntropyGraphs.AIC(model)
            @test_throws ArgumentError MaxEntropyGraphs.AICc(model)
            @test_throws ArgumentError MaxEntropyGraphs.BIC(model)
            solve_model!(model, method=:BFGS)
            # small-sample warning (n/k < 40)
            @test_logs (:warn, "The number of observations is small with respect to the number of parameters (n/k < 40). Consider using the corrected AIC (AICc) instead.") MaxEntropyGraphs.AIC(model)
            # test types
            @test isa(MaxEntropyGraphs.AIC(model), precision(model))
            @test isa(MaxEntropyGraphs.AICc(model), precision(model))
            @test isa(MaxEntropyGraphs.BIC(model), precision(model))
        end

        @testset "UECM - adjacency matrix variance" begin
            model = UECM(Gw)
            MaxEntropyGraphs.solve_model!(model, method=:BFGS)
            # binary layer gating
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum)
            MaxEntropyGraphs.set_Ĝ!(model)
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum)
            MaxEntropyGraphs.set_σ!(model)
            # weighted layer gating
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum, layer=:weighted)
            MaxEntropyGraphs.set_Ŵ!(model)
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum, layer=:weighted)
            MaxEntropyGraphs.set_σʷ!(model)
            # unknown layer / autodiff method
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum, layer=:unknown_layer)
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum, gradient_method=:unknown_method)
            # sanity: the expected weights reproduce the strength sequence, the σʷ are non-negative and finite
            @test isapprox(vec(sum(MaxEntropyGraphs.Ŵ(model), dims=2)), model.s, rtol=1e-4)
            @test all(x -> isfinite(x) && x ≥ 0, model.σʷ)
            # normal functioning (both layers): w_ij ≡ w_ji, so the identity-covariance
            # cross-term doubles the variance of a full-matrix metric
            for method in [:ForwardDiff; :ReverseDiff; :Zygote]
                @testset "gradient_method: $(method)" begin
                    @test MaxEntropyGraphs.σₓ(model, sum, gradient_method=method) ≈ sqrt(2 * sum(model.σ .^ 2))
                    @test MaxEntropyGraphs.σₓ(model, sum, layer=:weighted, gradient_method=method) ≈ sqrt(2 * sum(model.σʷ .^ 2))
                end
            end
        end
    end

    @testset "CReM" begin
        # small, self-contained (continuously) weighted undirected graph for the constructor tests
        Csrc = [1, 1, 2, 3]; Cdst = [2, 3, 3, 4]; Cw = [2.0, 1.0, 3.0, 4.0]
        Gsmall = MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(Csrc, Cdst, Cw)
        dsmall = MaxEntropyGraphs.Graphs.degree(Gsmall)
        ssmall = MaxEntropyGraphs.strength(Gsmall)
        # real weighted undirected anchor (symmetrised rhesus macaques network, N=16)
        Gw = MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(MaxEntropyGraphs.rhesus_macaques())

        @testset "CReM - generation" begin
            allowedDataTypes = [Float64; Float32; Float16]
            # simple model, directly from graph, different precisions
            for precision in allowedDataTypes
                model = CReM(Gsmall, precision=precision)
                @test isa(model, CReM)
                @test typeof(model).parameters[2] == precision
                @test MaxEntropyGraphs.precision(model) == precision
                @test typeof(model).parameters[1] == typeof(Gsmall)
                @test all([eltype(model.θ) == precision, eltype(model.αᵣ) == precision, eltype(model.xᵣ) == precision, eltype(model.s) == precision])
            end
            # simple model, directly from degree and strength sequences, different precisions
            for precision in allowedDataTypes
                model = CReM(d=dsmall, s=ssmall, precision=precision)
                @test isa(model, CReM)
                @test typeof(model).parameters[2] == precision
                @test MaxEntropyGraphs.precision(model) == precision
                @test typeof(model).parameters[1] == Nothing
                @test all([eltype(model.θ) == precision, eltype(model.αᵣ) == precision, eltype(model.xᵣ) == precision, eltype(model.s) == precision])
            end
            # the CReM allows continuous (non-integer) strengths
            @test isa(CReM(d=[1, 2, 1], s=[1.5, 2.5, 1.0]), CReM)
            # testing breaking conditions
            @test_throws MethodError CReM(1) # wrong input type
            # directed graph info loss warning message
            Gd = MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedDiGraph(Csrc, Cdst, Cw)
            @test_logs (:warn, "The graph is directed, the CReM model is undirected, the directional information will be lost") match_mode=:any CReM(Gd, d=MaxEntropyGraphs.Graphs.degree(Gd), s=MaxEntropyGraphs.strength(Gd))
            # zero degree / zero strength node warnings
            @test_logs (:warn, "The graph has vertices with zero degree, this may lead to convergence issues.") CReM(d=[0, 1, 1], s=[1.0, 1.0, 1.0])
            @test_logs (:warn, "The graph has vertices with zero strength, this may lead to convergence issues.") CReM(d=[1, 1, 1], s=[0.0, 1.0, 1.0])
            # invalid combination of parameters
            @test_throws ArgumentError CReM(MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(0)) # zero node graph
            @test_throws ArgumentError CReM(MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(1)) # single node graph
            @test_throws DimensionMismatch CReM(Gsmall, d=dsmall[1:end-1], s=ssmall) # different lengths (graph)
            @test_throws DimensionMismatch CReM(d=[1, 2, 3], s=[1.0, 2.0]) # different lengths (sequence)
            @test_throws ArgumentError CReM(d=Int[], s=Float64[]) # zero length
            @test_throws ArgumentError CReM(d=Int[1], s=Float64[1.0]) # single length
            @test_throws DomainError CReM(d=[1.5, 2.0], s=[1.0, 2.0]) # non-integer degree
            @test_throws DomainError CReM(d=[3, 1, 1], s=[1.0, 1.0, 1.0]) # max degree >= number of nodes
            @test_throws DomainError CReM(d=[-1, 2, 1], s=[1.0, 1.0, 1.0]) # negative degree
            @test_throws DomainError CReM(d=[1, 2, 1], s=[-1.0, 1.0, 1.0]) # negative strength
        end

        @testset "CReM - Likelihood gradient test" begin
            model = CReM(Gw)
            # the accessor requires computed parameters
            @test_throws ArgumentError L_CReM(model)
            # solve to populate the binary fitness xᵣ (and θ)
            solve_model!(model)
            n = model.status[:N]
            x = model.xᵣ[model.dᵣ_ind]
            # evaluate the gradient at a feasible, non-optimal point
            θtest = MaxEntropyGraphs.initial_guess(model, method=:strengths_minor)
            ∇L_buf = zeros(n); ∇L_buf_min = zeros(n)
            MaxEntropyGraphs.∇L_CReM!(∇L_buf, θtest, model.s, x)
            MaxEntropyGraphs.∇L_CReM_minus!(∇L_buf_min, θtest, model.s, x)
            @test ∇L_buf ≈ -∇L_buf_min
            ∇L_zyg = MaxEntropyGraphs.Zygote.gradient(θ -> L_CReM(θ, model.s, x), θtest)[1]
            @test ∇L_zyg ≈ ∇L_buf
            @test ∇L_zyg ≈ -∇L_buf_min
            # the same must hold for the precomputed (matrix) path
            MaxEntropyGraphs.set_Ĝ!(model)
            ∇L_mat = zeros(n)
            MaxEntropyGraphs.∇L_CReM!(∇L_mat, θtest, model.s, model.Ĝ)
            ∇L_zyg_mat = MaxEntropyGraphs.Zygote.gradient(θ -> L_CReM(θ, model.s, model.Ĝ), θtest)[1]
            @test ∇L_mat ≈ ∇L_zyg_mat
            # both likelihood paths agree
            @test L_CReM(θtest, model.s, x) ≈ L_CReM(θtest, model.s, model.Ĝ)
        end

        @testset "CReM - parameter computation" begin
            allowedDataTypes = [Float64] # low precision kept out for occasional convergence issues
            for precision in allowedDataTypes
                @testset "$(precision) precision" begin
                    model = CReM(Gw, precision=precision)
                    @test_throws ArgumentError Ĝ(model)
                    @test_throws ArgumentError MaxEntropyGraphs.Ŵ(model)
                    @test_throws ArgumentError σˣ(model)
                    @test_throws ArgumentError MaxEntropyGraphs.set_xᵣ!(model)
                    @test_throws ArgumentError MaxEntropyGraphs.initial_guess(model, method=:strange)
                    # every initial-guess method returns a strictly positive guess of length N
                    for initial in [:strengths, :strengths_minor, :random]
                        g = MaxEntropyGraphs.initial_guess(model, method=initial)
                        @test length(g) == model.status[:N]
                        @test all(g .> 0)
                    end
                    # the fixed point recipe is stable for the CReM (unlike the UECM); reproduces both constraints
                    @testset "fixedpoint - initial guess: $initial" for initial in [:strengths, :strengths_minor]
                        MaxEntropyGraphs.solve_model!(model, initial=initial, method=:fixedpoint)
                        @test isapprox(vec(sum(Ĝ(model), dims=2)), model.d, rtol=1e-4)
                        @test isapprox(vec(sum(MaxEntropyGraphs.Ŵ(model), dims=2)), model.s, rtol=1e-4)
                    end
                    # BFGS is robust across initial guesses (both analytical and AD gradient)
                    for initial in [:strengths, :strengths_minor]
                        @testset "BFGS - initial guess: $initial" begin
                            for analytical_gradient in [true, false]
                                @testset "analytical_gradient: $analytical_gradient" begin
                                    MaxEntropyGraphs.solve_model!(model, initial=initial, method=:BFGS, analytical_gradient=analytical_gradient)
                                    @test isapprox(vec(sum(Ĝ(model), dims=2)), model.d, rtol=1e-4)
                                    @test isapprox(vec(sum(MaxEntropyGraphs.Ŵ(model), dims=2)), model.s, rtol=1e-6)
                                end
                            end
                        end
                    end
                    # Newton from the default strengths guess
                    @testset "Newton - analytical_gradient: $analytical_gradient" for analytical_gradient in [false, true]
                        MaxEntropyGraphs.solve_model!(model, initial=:strengths, method=:Newton, analytical_gradient=analytical_gradient)
                        @test isapprox(vec(sum(Ĝ(model), dims=2)), model.d, rtol=1e-4)
                        @test isapprox(vec(sum(MaxEntropyGraphs.Ŵ(model), dims=2)), model.s, rtol=1e-6)
                    end
                    @test all([eltype(model.θ) == precision, eltype(model.αᵣ) == precision, eltype(model.xᵣ) == precision])
                end
            end
        end

        @testset "CReM - sampling" begin
            model = CReM(Gw)
            # parameters unknown
            @test_throws ArgumentError MaxEntropyGraphs.rand(model, precomputed=false)
            @test_throws ArgumentError MaxEntropyGraphs.Ĝ(model)
            @test_throws ArgumentError MaxEntropyGraphs.σˣ(model)
            # solve model
            MaxEntropyGraphs.solve_model!(model, method=:BFGS)
            # precomputed sampling is not implemented for the CReM
            @test_throws ArgumentError MaxEntropyGraphs.rand(model, precomputed=true)
            # single sample: correct number of vertices and continuous positive weights
            sample = rand(model)
            @test isa(sample, MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph)
            @test MaxEntropyGraphs.Graphs.nv(sample) == model.status[:N]
            @test all(w -> w > 0, MaxEntropyGraphs.SimpleWeightedGraphs.weight.(collect(MaxEntropyGraphs.Graphs.edges(sample))))
            # expected adjacency / variance sizes and eltypes
            MaxEntropyGraphs.set_Ĝ!(model)
            MaxEntropyGraphs.set_σ!(model)
            @test eltype(MaxEntropyGraphs.σˣ(model)) == MaxEntropyGraphs.precision(model)
            @test size(MaxEntropyGraphs.σˣ(model)) == (model.status[:N], model.status[:N])
            @test eltype(MaxEntropyGraphs.Ĝ(model)) == MaxEntropyGraphs.precision(model)
            @test size(MaxEntropyGraphs.Ĝ(model)) == (model.status[:N], model.status[:N])
            # batch sampling
            S = rand(model, 100)
            @test length(S) == 100
            @test all(MaxEntropyGraphs.Graphs.nv.(S) .== model.status[:N])
            # reproducibility with a fixed rng
            @test MaxEntropyGraphs.SimpleWeightedGraphs.weights(rand(model, rng=MaxEntropyGraphs.Xoshiro(42))) == MaxEntropyGraphs.SimpleWeightedGraphs.weights(rand(model, rng=MaxEntropyGraphs.Xoshiro(42)))
        end

        @testset "CReM - degree/strength metrics" begin
            model = CReM(Gw)
            # adjacency matrix accessor
            @test iszero(MaxEntropyGraphs.A(model, 1, 1))
            # parameters not computed yet
            @test_throws ArgumentError degree(model, 1)
            @test_throws ArgumentError MaxEntropyGraphs.strength(model, 1)
            solve_model!(model, method=:BFGS)
            @test isa(MaxEntropyGraphs.A(model, 1, 2), precision(model))
            # check out of bounds
            @test_throws ArgumentError degree(model, model.status[:N] + 1)
            @test_throws ArgumentError MaxEntropyGraphs.strength(model, model.status[:N] + 1)
            # unknown method
            @test_throws ArgumentError degree(model, method=:unknown_method)
            @test_throws ArgumentError MaxEntropyGraphs.strength(model, method=:unknown_method)
            # reduced/full reproduce the observed degree and strength sequences
            for method in [:reduced, :full]
                @test isapprox(degree(model, method=method), model.d, rtol=1e-4)
                @test isapprox(MaxEntropyGraphs.strength(model, method=method), model.s, rtol=1e-6)
            end
            # adjacency-based methods require set_Ĝ!
            @test_throws ArgumentError degree(model, method=:adjacency)
            @test_throws ArgumentError MaxEntropyGraphs.strength(model, method=:adjacency)
            set_Ĝ!(model)
            @test isapprox(degree(model, method=:adjacency), model.d, rtol=1e-4)
            @test isapprox(MaxEntropyGraphs.strength(model, method=:adjacency), model.s, rtol=1e-6)
        end

        @testset "CReM - (B/A)IC(c)" begin
            model = CReM(Gw)
            # parameters not computed yet
            @test_throws ArgumentError MaxEntropyGraphs.AIC(model)
            @test_throws ArgumentError MaxEntropyGraphs.AICc(model)
            @test_throws ArgumentError MaxEntropyGraphs.BIC(model)
            solve_model!(model, method=:BFGS)
            # small-sample warning (n/k < 40)
            @test_logs (:warn, "The number of observations is small with respect to the number of parameters (n/k < 40). Consider using the corrected AIC (AICc) instead.") MaxEntropyGraphs.AIC(model)
            # test types
            @test isa(MaxEntropyGraphs.AIC(model), precision(model))
            @test isa(MaxEntropyGraphs.AICc(model), precision(model))
            @test isa(MaxEntropyGraphs.BIC(model), precision(model))
        end

        @testset "CReM - adjacency matrix variance" begin
            model = CReM(Gw)
            # the two-step solve computes both the binary (conditional UBCM) and the weighted layer parameters
            MaxEntropyGraphs.solve_model!(model, method=:BFGS)
            # binary layer gating
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum)
            MaxEntropyGraphs.set_Ĝ!(model)
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum)
            MaxEntropyGraphs.set_σ!(model)
            # weighted layer gating
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum, layer=:weighted)
            MaxEntropyGraphs.set_Ŵ!(model)
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum, layer=:weighted)
            MaxEntropyGraphs.set_σʷ!(model)
            # unknown layer / autodiff method
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum, layer=:unknown_layer)
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum, gradient_method=:unknown_method)
            # sanity: the expected weights reproduce the strength sequence, the σʷ are non-negative and finite
            @test isapprox(vec(sum(MaxEntropyGraphs.Ŵ(model), dims=2)), model.s, rtol=1e-4)
            @test all(x -> isfinite(x) && x ≥ 0, model.σʷ)
            # normal functioning (both layers): w_ij ≡ w_ji, so the identity-covariance
            # cross-term doubles the variance of a full-matrix metric
            for method in [:ForwardDiff; :ReverseDiff; :Zygote]
                @testset "gradient_method: $(method)" begin
                    @test MaxEntropyGraphs.σₓ(model, sum, gradient_method=method) ≈ sqrt(2 * sum(model.σ .^ 2))
                    @test MaxEntropyGraphs.σₓ(model, sum, layer=:weighted, gradient_method=method) ≈ sqrt(2 * sum(model.σʷ .^ 2))
                end
            end
        end
    end

    @testset "DCReM" begin
        # small, self-contained (continuously) weighted directed graph for the constructor tests
        Dsrc = [1, 2, 2, 3, 4]; Ddst = [2, 1, 3, 4, 3]; Dw = [2.0, 1.0, 3.0, 4.0, 1.5]
        Gsmall = MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedDiGraph(Dsrc, Ddst, Dw)
        dout_small = MaxEntropyGraphs.Graphs.outdegree(Gsmall)
        din_small  = MaxEntropyGraphs.Graphs.indegree(Gsmall)
        sout_small = MaxEntropyGraphs.strength(Gsmall, dir=:out)
        sin_small  = MaxEntropyGraphs.strength(Gsmall, dir=:in)
        # real weighted directed anchor (rhesus macaques network, N=16)
        Gw = MaxEntropyGraphs.rhesus_macaques()

        @testset "DCReM - generation" begin
            allowedDataTypes = [Float64; Float32; Float16]
            # simple model, directly from graph, different precisions
            for precision in allowedDataTypes
                model = DCReM(Gsmall, precision=precision)
                @test isa(model, DCReM)
                @test typeof(model).parameters[2] == precision
                @test MaxEntropyGraphs.precision(model) == precision
                @test typeof(model).parameters[1] == typeof(Gsmall)
                @test all([eltype(model.θ) == precision, eltype(model.αᵣ) == precision, eltype(model.xᵣ) == precision, eltype(model.s_out) == precision])
            end
            # simple model, directly from degree and strength sequences, different precisions
            for precision in allowedDataTypes
                model = DCReM(d_out=dout_small, d_in=din_small, s_out=sout_small, s_in=sin_small, precision=precision)
                @test isa(model, DCReM)
                @test typeof(model).parameters[2] == precision
                @test MaxEntropyGraphs.precision(model) == precision
                @test typeof(model).parameters[1] == Nothing
                @test all([eltype(model.θ) == precision, eltype(model.αᵣ) == precision, eltype(model.yᵣ) == precision, eltype(model.s_in) == precision])
            end
            # the DCReM allows continuous (non-integer) strengths
            @test isa(DCReM(d_out=[1, 1, 1], d_in=[1, 1, 1], s_out=[1.5, 2.5, 1.0], s_in=[2.0, 1.5, 1.5]), DCReM)
            # testing breaking conditions
            @test_throws MethodError DCReM(1) # wrong input type
            # undirected graph info loss warning message
            Gu = MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(MaxEntropyGraphs.rhesus_macaques())
            @test_logs (:warn, "The graph is undirected, while the DCReM model is directed, the out- and in-quantities will be the same") match_mode=:any DCReM(Gu)
            # zero degree / zero strength node warnings
            @test_logs (:warn, "The graph has vertices with zero out- or in-degree, this may lead to convergence issues.") match_mode=:any DCReM(d_out=[0, 1, 1], d_in=[1, 0, 1], s_out=[0.0, 1.0, 1.0], s_in=[1.0, 0.0, 1.0])
            @test_logs (:warn, "The graph has vertices with zero out- or in-strength, this may lead to convergence issues.") match_mode=:any DCReM(d_out=[0, 1, 1], d_in=[1, 0, 1], s_out=[0.0, 1.0, 1.0], s_in=[1.0, 0.0, 1.0])
            # invalid combination of parameters
            @test_throws ArgumentError DCReM(MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedDiGraph(0)) # zero node graph
            @test_throws ArgumentError DCReM(MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedDiGraph(1)) # single node graph
            @test_throws DimensionMismatch DCReM(Gsmall, d_out=dout_small[1:end-1], d_in=din_small, s_out=sout_small, s_in=sin_small)
            @test_throws DimensionMismatch DCReM(d_out=[1, 2, 1], d_in=[1, 2], s_out=[1.0, 2.0, 1.0], s_in=[1.0, 2.0, 1.0])
            @test_throws ArgumentError DCReM(d_out=Int[], d_in=Int[], s_out=Float64[], s_in=Float64[]) # zero length
            @test_throws ArgumentError DCReM(d_out=Int[1], d_in=Int[1], s_out=Float64[1.0], s_in=Float64[1.0]) # single length
            @test_throws DomainError DCReM(d_out=[1.5, 2.0], d_in=[1.0, 1.0], s_out=[1.0, 2.0], s_in=[1.0, 2.0]) # non-integer degree
            @test_throws DomainError DCReM(d_out=[3, 1, 1], d_in=[1, 2, 2], s_out=[1.0, 1.0, 1.0], s_in=[1.0, 1.0, 1.0]) # max degree >= number of nodes
            @test_throws DomainError DCReM(d_out=[-1, 2, 1], d_in=[1, 1, 0], s_out=[1.0, 1.0, 1.0], s_in=[1.0, 1.0, 1.0]) # negative degree
            @test_throws DomainError DCReM(d_out=[1, 2, 1], d_in=[1, 2, 1], s_out=[-1.0, 1.0, 1.0], s_in=[1.0, 1.0, 1.0]) # negative strength
        end

        @testset "DCReM - Likelihood gradient test" begin
            model = DCReM(Gw)
            MaxEntropyGraphs.solve_model!(model) # need the binary layer for the fitnesses
            n = model.status[:N]
            x = model.xᵣ[model.dᵣ_ind]
            y = model.yᵣ[model.dᵣ_ind]
            θt = MaxEntropyGraphs.initial_guess(model, method=:strengths_minor)
            # analytical gradient == minus the minus-gradient (both dispatches)
            ∇L_buf = zeros(2*n); ∇L_buf_min = zeros(2*n)
            MaxEntropyGraphs.∇L_DCReM!(∇L_buf, θt, model.s_out, model.s_in, x, y)
            MaxEntropyGraphs.∇L_DCReM_minus!(∇L_buf_min, θt, model.s_out, model.s_in, x, y)
            @test ∇L_buf ≈ -∇L_buf_min
            # analytical gradient == autodiff gradient of the likelihood at a feasible point
            ∇L_AD = MaxEntropyGraphs.ForwardDiff.gradient(θ -> MaxEntropyGraphs.L_DCReM(θ, model.s_out, model.s_in, x, y), θt)
            @test isapprox(∇L_buf, ∇L_AD, rtol=1e-10)
            # on-the-fly and precomputed-matrix dispatches agree
            MaxEntropyGraphs.set_Ĝ!(model)
            @test MaxEntropyGraphs.L_DCReM(θt, model.s_out, model.s_in, x, y) ≈ MaxEntropyGraphs.L_DCReM(θt, model.s_out, model.s_in, model.Ĝ)
            ∇L_mat = zeros(2*n)
            MaxEntropyGraphs.∇L_DCReM!(∇L_mat, θt, model.s_out, model.s_in, model.Ĝ)
            @test isapprox(∇L_buf, ∇L_mat, rtol=1e-12)
            # domain guard: infeasible rates give NaN (rejected by the line search)
            θbad = copy(θt); θbad[1] = -1.0
            @test isnan(MaxEntropyGraphs.L_DCReM(θbad, model.s_out, model.s_in, x, y))
        end

        @testset "DCReM - parameter computation" begin
            model = DCReM(Gw)
            @test_throws ArgumentError Ĝ(model)
            @test_throws ArgumentError σˣ(model)
            @test_throws ArgumentError Ŵ(model)
            @test_throws ArgumentError σʷ(model)
            @test_throws ArgumentError MaxEntropyGraphs.set_xᵣ!(model)
            @test_throws ArgumentError MaxEntropyGraphs.set_yᵣ!(model)
            @test_throws ArgumentError MaxEntropyGraphs.initial_guess(model, method=:strange)
            for initial in [:strengths, :strengths_minor]
                @test length(MaxEntropyGraphs.initial_guess(model, method=initial)) == 2 * model.status[:N]
                @test all(MaxEntropyGraphs.initial_guess(model, method=initial) .>= 0) # feasible direct rates
                methods = initial == :strengths ? [(:fixedpoint, false), (:BFGS, false), (:BFGS, true), (:Newton, false)] :
                                                  [(:fixedpoint, false), (:BFGS, false)]
                for (method, analytical_gradient) in methods
                    @testset "initials: $initial, method: $method, analytical_gradient: $analytical_gradient" begin
                        MaxEntropyGraphs.solve_model!(model, initial=initial, method=method, analytical_gradient=analytical_gradient)
                        # binary layer reproduces the degrees, weighted layer the strengths
                        @test isapprox(outdegree(model), model.d_out, rtol=1e-6)
                        @test isapprox(indegree(model),  model.d_in,  rtol=1e-6)
                        W = MaxEntropyGraphs.Ŵ(model)
                        @test isapprox(vec(sum(W, dims=2)), model.s_out, rtol=1e-4)
                        @test isapprox(vec(sum(W, dims=1)), model.s_in,  rtol=1e-4)
                    end
                end
            end
            # two-step kwarg plumbing (conditional layer settings) + cached-adjacency solve
            model = DCReM(Gw)
            MaxEntropyGraphs.solve_model!(model, method_conditional=:BFGS, analytical_gradient_conditional=true, store_adjacency=true)
            @test model.status[:G_computed]
            @test isapprox(vec(sum(MaxEntropyGraphs.Ŵ(model), dims=2)), model.s_out, rtol=1e-4)
        end

        @testset "DCReM - sampling" begin
            model = DCReM(Gw)
            # parameters unknown
            @test_throws ArgumentError MaxEntropyGraphs.rand(model, precomputed=false)
            MaxEntropyGraphs.solve_model!(model)
            # precomputed sampling not implemented
            @test_throws ArgumentError MaxEntropyGraphs.rand(model, precomputed=true)
            s = rand(model)
            @test isa(s, MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedDiGraph)
            @test MaxEntropyGraphs.Graphs.nv(s) == MaxEntropyGraphs.Graphs.nv(model.G)
            # weights are continuous and positive
            @test all(w -> w > 0, [MaxEntropyGraphs.SimpleWeightedGraphs.get_weight(s, MaxEntropyGraphs.Graphs.src(e), MaxEntropyGraphs.Graphs.dst(e)) for e in MaxEntropyGraphs.Graphs.edges(s)])
            # batch sampling
            S = rand(model, 100)
            @test length(S) == 100
            @test all(MaxEntropyGraphs.Graphs.nv.(S) .== MaxEntropyGraphs.Graphs.nv(model.G))
        end

        @testset "DCReM - degree/strength metrics" begin
            model = DCReM(Gw)
            @test_throws ArgumentError degree(model, 1)
            @test_throws ArgumentError outstrength(model, 1)
            MaxEntropyGraphs.solve_model!(model)
            # adjacency accessor
            @test iszero(MaxEntropyGraphs.A(model, 1, 1))
            @test isa(MaxEntropyGraphs.A(model, 1, 2), precision(model))
            # out of bounds / unknown methods
            @test_throws ArgumentError degree(model, model.status[:N] + 1)
            @test_throws ArgumentError outstrength(model, model.status[:N] + 1)
            @test_throws ArgumentError degree(model, method=:unknown_method)
            @test_throws ArgumentError instrength(model, method=:unknown_method)
            # binary layer degrees, weighted layer strengths
            for method in [:reduced, :full]
                @test isapprox(outdegree(model, method=method), model.d_out, rtol=1e-6)
                @test isapprox(indegree(model,  method=method), model.d_in,  rtol=1e-6)
                @test isapprox(outstrength(model, method=method), model.s_out, rtol=1e-4)
                @test isapprox(instrength(model,  method=method), model.s_in,  rtol=1e-4)
            end
            # :adjacency requires the precomputed matrix
            @test_throws ArgumentError degree(model, method=:adjacency)
            @test_throws ArgumentError outstrength(model, method=:adjacency)
            MaxEntropyGraphs.set_Ĝ!(model)
            @test isapprox(outdegree(model, method=:adjacency), model.d_out, rtol=1e-6)
            @test isapprox(outstrength(model, method=:adjacency), model.s_out, rtol=1e-4)
            @test isapprox(instrength(model,  method=:adjacency), model.s_in,  rtol=1e-4)
            # expected weight matrix
            W = MaxEntropyGraphs.Ŵ(model)
            @test size(W) == (model.status[:N], model.status[:N])
            @test all(iszero, W[MaxEntropyGraphs.diagind(W)])
        end

        @testset "DCReM - (B/A)IC(c)" begin
            model = DCReM(Gw)
            @test_throws ArgumentError MaxEntropyGraphs.AIC(model)
            @test_throws ArgumentError MaxEntropyGraphs.AICc(model)
            @test_throws ArgumentError MaxEntropyGraphs.BIC(model)
            MaxEntropyGraphs.solve_model!(model)
            @test_logs (:warn, "The number of observations is small with respect to the number of parameters (n/k < 40). Consider using the corrected AIC (AICc) instead.") MaxEntropyGraphs.AIC(model)
            @test isa(MaxEntropyGraphs.AIC(model), precision(model))
            @test isa(MaxEntropyGraphs.AICc(model), precision(model))
            @test isa(MaxEntropyGraphs.BIC(model), precision(model))
        end

        @testset "DCReM - adjacency matrix variance" begin
            model = DCReM(Gw)
            MaxEntropyGraphs.solve_model!(model)
            # binary layer gating
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum)
            MaxEntropyGraphs.set_Ĝ!(model)
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum)
            MaxEntropyGraphs.set_σ!(model)
            # weighted layer gating
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum, layer=:weighted)
            MaxEntropyGraphs.set_Ŵ!(model)
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum, layer=:weighted)
            MaxEntropyGraphs.set_σʷ!(model)
            # unknown layer / autodiff method
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum, layer=:unknown_layer)
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum, gradient_method=:unknown_method)
            # normal functioning (both layers; the DBCM binary layer has independent entries)
            for method in [:ForwardDiff; :ReverseDiff; :Zygote]
                @testset "gradient_method: $(method)" begin
                    @test MaxEntropyGraphs.σₓ(model, sum, gradient_method=method) ≈ sqrt(sum(model.σ .^ 2))
                    @test MaxEntropyGraphs.σₓ(model, sum, layer=:weighted, gradient_method=method) ≈ sqrt(sum(model.σʷ .^ 2))
                end
            end
        end
    end

    @testset "CRWCM" begin
        # real weighted directed anchor (rhesus macaques network, N=16, r_t=0.76,
        # with 5 nodes of zero s→ and 4 of zero s← -> exercises the dead-channel machinery)
        Gw = MaxEntropyGraphs.rhesus_macaques()

        @testset "CRWCM - generation" begin
            allowedDataTypes = [Float64; Float32; Float16]
            d_out = MaxEntropyGraphs.nonreciprocated_outdegree(Gw)
            d_in  = MaxEntropyGraphs.nonreciprocated_indegree(Gw)
            d_rec = MaxEntropyGraphs.reciprocated_degree(Gw)
            s_out = MaxEntropyGraphs.nonreciprocated_outstrength(Gw)
            s_in  = MaxEntropyGraphs.nonreciprocated_instrength(Gw)
            s_ro  = MaxEntropyGraphs.reciprocated_outstrength(Gw)
            s_ri  = MaxEntropyGraphs.reciprocated_instrength(Gw)
            # simple model, directly from graph, different precisions
            for precision in allowedDataTypes
                model = CRWCM(Gw, precision=precision)
                @test isa(model, CRWCM)
                @test MaxEntropyGraphs.precision(model) == precision
                @test typeof(model).parameters[1] == typeof(Gw)
                @test all([eltype(model.θ) == precision, eltype(model.αᵣ) == precision, eltype(model.zᵣ) == precision, eltype(model.s_rec_out) == precision])
            end
            # simple model, directly from the seven sequences
            for precision in allowedDataTypes
                model = CRWCM(d_out=d_out, d_in=d_in, d_rec=d_rec, s_out=s_out, s_in=s_in, s_rec_out=s_ro, s_rec_in=s_ri, precision=precision)
                @test isa(model, CRWCM)
                @test MaxEntropyGraphs.precision(model) == precision
                @test typeof(model).parameters[1] == Nothing
            end
            # the model stores the reciprocal sequences
            model = CRWCM(Gw)
            @test model.s_out == s_out
            @test model.s_rec_in == s_ri
            @test model.s_out .+ model.s_rec_out ≈ MaxEntropyGraphs.outstrength(Gw)
            ## testing breaking conditions
            @test_throws MaxEntropyGraphs.Graphs.NotImplementedError CRWCM(1) # wrong input type (fails in the kwarg defaults)
            # undirected graph -> fully reciprocated warning (the BARE constructor must reach this
            # path: the strength keyword defaults must branch on directedness like the degree ones)
            Gu = MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(Gw)
            @test_logs (:warn, "The graph is undirected, while the CRWCM model is directed; every edge will be considered reciprocated (k→ = k← = 0)") match_mode=:any CRWCM(Gu)
            # zero-reciprocity warning
            @test_logs (:warn, "The reciprocated degree sequence is all zeros: the CRWCM degenerates to a DCReM. Consider using the DCReM instead.") CRWCM(d_out=[1,1,0], d_in=[0,1,1], d_rec=[0,0,0], s_out=[1.0,1.0,0.0], s_in=[0.0,1.0,1.0], s_rec_out=[0.0,0.0,0.0], s_rec_in=[0.0,0.0,0.0])
            # graph problems
            @test_throws ArgumentError CRWCM(MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedDiGraph(0))
            @test_throws ArgumentError CRWCM(MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedDiGraph(1))
            @test_throws DimensionMismatch CRWCM(Gw, d_out=d_out[1:end-1], d_in=d_in, d_rec=d_rec, s_out=s_out, s_in=s_in, s_rec_out=s_ro, s_rec_in=s_ri)
            # degree/strength sequence problems
            @test_throws ArgumentError CRWCM(d_out=Int[], d_in=Int[], d_rec=Int[], s_out=Float64[], s_in=Float64[], s_rec_out=Float64[], s_rec_in=Float64[])
            @test_throws DomainError CRWCM(d_out=[1,0], d_in=[0,0], d_rec=[0,0], s_out=[1.0,0.0], s_in=[0.0,0.0], s_rec_out=[0.0,0.0], s_rec_in=[0.0,0.0]) # unbalanced non-reciprocated stubs
            @test_throws DomainError CRWCM(d_out=[0,0,0], d_in=[0,0,0], d_rec=[1,1,1], s_out=zeros(3), s_in=zeros(3), s_rec_out=ones(3), s_rec_in=ones(3)) # odd reciprocated stubs
            @test_throws DomainError CRWCM(d_out=[-1,1], d_in=[1,-1], d_rec=[0,0], s_out=[1.0,1.0], s_in=[1.0,1.0], s_rec_out=[0.0,0.0], s_rec_in=[0.0,0.0]) # negative degrees
            @test_throws DomainError CRWCM(d_out=[1,1,0], d_in=[0,1,1], d_rec=[0,0,0], s_out=[-1.0,1.0,0.0], s_in=[0.0,1.0,1.0], s_rec_out=zeros(3), s_rec_in=zeros(3)) # negative strength
            # positive weights: zero strength in a channel iff zero degree in that channel
            @test_throws DomainError CRWCM(d_out=[1,1,0], d_in=[0,1,1], d_rec=[0,0,0], s_out=[0.0,1.0,0.0], s_in=[0.0,1.0,1.0], s_rec_out=zeros(3), s_rec_in=zeros(3)) # s→=0 but k→>0
            @test_throws DomainError CRWCM(d_out=[1,1,0], d_in=[0,1,1], d_rec=[2,2,2], s_out=[1.0,1.0,0.0], s_in=[0.0,1.0,1.0], s_rec_out=[1.0,1.0,0.0], s_rec_in=[1.0,1.0,1.0]) # s↔out=0 but k↔>0
        end

        @testset "CRWCM - Likelihood gradient test" begin
            model = CRWCM(Gw)
            MaxEntropyGraphs.solve_model!(model) # need the binary layer for the fitnesses
            n = model.status[:N]
            x = model.xᵣ[model.dᵣ_ind]
            y = model.yᵣ[model.dᵣ_ind]
            z = model.zᵣ[model.dᵣ_ind]
            θt = MaxEntropyGraphs.initial_guess(model, method=:strengths_minor)
            args = (model.s_out, model.s_in, model.s_rec_out, model.s_rec_in, model.s_out_nz, model.s_in_nz, model.s_rec_nz)
            # analytical gradient == minus the minus-gradient
            ∇L_buf = zeros(4*n); ∇L_buf_min = zeros(4*n)
            MaxEntropyGraphs.∇L_CRWCM!(∇L_buf, θt, args..., x, y, z)
            MaxEntropyGraphs.∇L_CRWCM_minus!(∇L_buf_min, θt, args..., x, y, z)
            @test ∇L_buf ≈ -∇L_buf_min
            # analytical gradient == autodiff gradient of the likelihood at a feasible point
            ∇L_AD = MaxEntropyGraphs.ForwardDiff.gradient(θ -> MaxEntropyGraphs.L_CRWCM(θ, args..., x, y, z), θt)
            @test isapprox(∇L_buf, ∇L_AD, rtol=1e-10)
            # on-the-fly and precomputed-matrix dispatches agree
            P̂ = [MaxEntropyGraphs.p⭢(model, i, j) for i in 1:n, j in 1:n]
            R̂ = [MaxEntropyGraphs.p⭤(model, i, j) for i in 1:n, j in 1:n]
            @test MaxEntropyGraphs.L_CRWCM(θt, args..., x, y, z) ≈ MaxEntropyGraphs.L_CRWCM(θt, args..., P̂, R̂)
            # block separability: perturbing the reciprocated block leaves the non-reciprocated gradient unchanged
            θp = copy(θt); θp[3*n+1:4*n] .*= 2
            ∇L_pert = zeros(4*n)
            MaxEntropyGraphs.∇L_CRWCM!(∇L_pert, θp, args..., x, y, z)
            @test ∇L_buf[1:2*n] == ∇L_pert[1:2*n]
            # domain guard: infeasible rates give NaN (rejected by the line search)
            θbad = copy(θt); θbad[model.s_out_nz[1]] = -10.0
            @test isnan(MaxEntropyGraphs.L_CRWCM(θbad, args..., x, y, z))
        end

        @testset "CRWCM - parameter computation" begin
            model = CRWCM(Gw)
            @test_throws ArgumentError Ĝ(model)
            @test_throws ArgumentError σˣ(model)
            @test_throws ArgumentError Ŵ(model)
            @test_throws ArgumentError σʷ(model)
            @test_throws ArgumentError MaxEntropyGraphs.set_xᵣ!(model)
            @test_throws ArgumentError MaxEntropyGraphs.set_yᵣ!(model)
            @test_throws ArgumentError MaxEntropyGraphs.set_zᵣ!(model)
            @test_throws ArgumentError MaxEntropyGraphs.initial_guess(model, method=:strange)
            for initial in [:strengths, :strengths_minor]
                θ₀ = MaxEntropyGraphs.initial_guess(model, method=initial)
                @test length(θ₀) == 4 * model.status[:N]
                @test all(θ₀ .>= 0) # feasible direct rates (dead channels at zero)
                methods = initial == :strengths ? [(:fixedpoint, false), (:BFGS, false), (:BFGS, true), (:Newton, false)] :
                                                  [(:fixedpoint, false), (:BFGS, false)]
                for (method, analytical_gradient) in methods
                    @testset "initials: $initial, method: $method, analytical_gradient: $analytical_gradient" begin
                        MaxEntropyGraphs.solve_model!(model, initial=initial, method=method, analytical_gradient=analytical_gradient)
                        # binary layer reproduces the reciprocal degrees
                        @test isapprox(MaxEntropyGraphs.nonreciprocated_outdegree(model), model.d_out, rtol=1e-6)
                        @test isapprox(MaxEntropyGraphs.reciprocated_degree(model), model.d_rec, rtol=1e-6)
                        # weighted layer reproduces the four reciprocal strengths
                        @test isapprox(MaxEntropyGraphs.nonreciprocated_outstrength(model), model.s_out, rtol=1e-4, atol=1e-4)
                        @test isapprox(MaxEntropyGraphs.nonreciprocated_instrength(model),  model.s_in,  rtol=1e-4, atol=1e-4)
                        @test isapprox(MaxEntropyGraphs.reciprocated_outstrength(model), model.s_rec_out, rtol=1e-4)
                        @test isapprox(MaxEntropyGraphs.reciprocated_instrength(model),  model.s_rec_in,  rtol=1e-4)
                        # dead channels are pinned to +Inf (5 zero-s→ and 4 zero-s← nodes on rhesus)
                        @test count(isinf, model.θ) == count(iszero, model.s_out) + count(iszero, model.s_in) + 2*count(iszero, model.s_rec_out)
                        # the expected weight matrix is finite and NaN-free despite the pinned parameters
                        W = MaxEntropyGraphs.Ŵ(model)
                        @test all(isfinite, W)
                        @test isapprox(vec(sum(W, dims=2)), model.s_out .+ model.s_rec_out, rtol=1e-4)
                    end
                end
            end
            # two-step kwarg plumbing (conditional layer settings)
            model = CRWCM(Gw)
            MaxEntropyGraphs.solve_model!(model, method_conditional=:BFGS, analytical_gradient_conditional=true)
            @test isapprox(MaxEntropyGraphs.reciprocated_outstrength(model), model.s_rec_out, rtol=1e-4)
        end

        @testset "CRWCM - sampling" begin
            model = CRWCM(Gw)
            # parameters unknown
            @test_throws ArgumentError MaxEntropyGraphs.rand(model, precomputed=false)
            MaxEntropyGraphs.solve_model!(model)
            # precomputed sampling is deliberately unsupported
            @test_throws ArgumentError MaxEntropyGraphs.rand(model, precomputed=true)
            s = rand(model)
            @test isa(s, MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedDiGraph)
            @test MaxEntropyGraphs.Graphs.nv(s) == MaxEntropyGraphs.Graphs.nv(model.G)
            # batch sampling
            S = rand(model, 100)
            @test length(S) == 100
            @test all(MaxEntropyGraphs.Graphs.nv.(S) .== MaxEntropyGraphs.Graphs.nv(model.G))
        end

        @testset "CRWCM - degree/strength metrics" begin
            model = CRWCM(Gw)
            @test_throws ArgumentError MaxEntropyGraphs.reciprocated_degree(model, 1)
            @test_throws ArgumentError MaxEntropyGraphs.reciprocated_outstrength(model, 1)
            MaxEntropyGraphs.solve_model!(model)
            # dyadic probability accessors
            @test iszero(MaxEntropyGraphs.A(model, 1, 1))
            @test MaxEntropyGraphs.A(model, 1, 2) ≈ MaxEntropyGraphs.p⭢(model, 1, 2) + MaxEntropyGraphs.p⭤(model, 1, 2)
            @test MaxEntropyGraphs.p⭠(model, 1, 2) == MaxEntropyGraphs.p⭢(model, 2, 1)
            # out of bounds / unknown methods
            @test_throws ArgumentError MaxEntropyGraphs.reciprocated_degree(model, model.status[:N] + 1)
            @test_throws ArgumentError MaxEntropyGraphs.nonreciprocated_outstrength(model, model.status[:N] + 1)
            @test_throws ArgumentError MaxEntropyGraphs.reciprocated_outstrength(model, 1, method=:unknown_method)
            # binary layer: reciprocal degrees; weighted layer: reciprocal strengths; totals
            @test isapprox(MaxEntropyGraphs.nonreciprocated_indegree(model), model.d_in, rtol=1e-6)
            @test isapprox(MaxEntropyGraphs.nonreciprocated_instrength(model), model.s_in, rtol=1e-4, atol=1e-4)
            @test isapprox(MaxEntropyGraphs.reciprocated_instrength(model), model.s_rec_in, rtol=1e-4)
            @test isapprox(MaxEntropyGraphs.outstrength(model), model.s_out .+ model.s_rec_out, rtol=1e-4)
            @test isapprox(MaxEntropyGraphs.instrength(model),  model.s_in  .+ model.s_rec_in,  rtol=1e-4)
            # the expected weight matrix row/column sums equal the total strengths
            W = MaxEntropyGraphs.Ŵ(model)
            @test isapprox(vec(sum(W, dims=2)), MaxEntropyGraphs.outstrength(model), rtol=1e-8)
            @test isapprox(vec(sum(W, dims=1)), MaxEntropyGraphs.instrength(model),  rtol=1e-8)
        end

        @testset "CRWCM - (B/A)IC(c)" begin
            model = CRWCM(Gw)
            @test_throws ArgumentError MaxEntropyGraphs.AIC(model)
            @test_throws ArgumentError MaxEntropyGraphs.AICc(model)
            @test_throws ArgumentError MaxEntropyGraphs.BIC(model)
            MaxEntropyGraphs.solve_model!(model)
            @test_logs (:warn, "The number of observations is small with respect to the number of parameters (n/k < 40). Consider using the corrected AIC (AICc) instead.") MaxEntropyGraphs.AIC(model)
            @test isa(MaxEntropyGraphs.AIC(model), precision(model))
            @test isa(MaxEntropyGraphs.AICc(model), precision(model))
            @test isa(MaxEntropyGraphs.BIC(model), precision(model))
            @test isfinite(MaxEntropyGraphs.AICc(model)) # the pinned +Inf parameters must not leak into the likelihood
        end

        @testset "CRWCM - adjacency matrix variance" begin
            model = CRWCM(Gw)
            MaxEntropyGraphs.solve_model!(model)
            # binary layer gating
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum)
            MaxEntropyGraphs.set_Ĝ!(model)
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum)
            MaxEntropyGraphs.set_σ!(model)
            # weighted layer gating
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum, layer=:weighted)
            MaxEntropyGraphs.set_Ŵ!(model)
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum, layer=:weighted)
            MaxEntropyGraphs.set_σʷ!(model)
            # unknown layer / autodiff method
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum, layer=:unknown_layer)
            @test_throws ArgumentError MaxEntropyGraphs.σₓ(model, sum, gradient_method=:unknown_method)
            # the covariance matrices are symmetric with zero diagonal, and everything is finite
            C  = MaxEntropyGraphs._cov_dyads(model)
            Cw = MaxEntropyGraphs._covʷ(model)
            @test isapprox(C, transpose(C)) && isapprox(Cw, transpose(Cw))
            @test all(iszero, C[MaxEntropyGraphs.diagind(C)]) && all(iszero, Cw[MaxEntropyGraphs.diagind(Cw)])
            @test all(isfinite, model.σʷ) && all(isfinite, Cw)
            # normal functioning: identities including the within-dyad covariance cross-terms
            for method in [:ForwardDiff; :ReverseDiff; :Zygote]
                @testset "gradient_method: $(method)" begin
                    @test MaxEntropyGraphs.σₓ(model, sum, gradient_method=method) ≈ sqrt(sum(model.σ .^ 2) + sum(C))
                    @test MaxEntropyGraphs.σₓ(model, sum, layer=:weighted, gradient_method=method) ≈ sqrt(sum(model.σʷ .^ 2) + sum(Cw))
                    # direction-selective metric: only row 1 -> the covariance cross-terms vanish
                    @test MaxEntropyGraphs.σₓ(model, W -> sum(W[1,:]), layer=:weighted, gradient_method=method) ≈ sqrt(sum(model.σʷ[1,:] .^ 2))
                end
            end
        end
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



