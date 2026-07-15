"""
    MaxEntropyGraphs

Julia module for working with maximum entropy graphs
"""
module MaxEntropyGraphs
    import Base: show, rand, showerror, precision
    # reproducible / thread-safe sampling
    import Random: AbstractRNG, default_rng, Xoshiro
    # for logmessages
    import Printf: @sprintf
    #import Dates: now, Day, Minute 
    

    # to work with all sorts of graphs
    import Graphs
    import Graphs: degree
    import SimpleWeightedGraphs

    # to solve the optimization problem(s)
    import Optimization
    import OptimizationOptimJL
    import ForwardDiff, ReverseDiff, Zygote
    import NLsolve
    import LinearAlgebra: issymmetric, diagind, dot, triu!, mul!, BLAS
    import SparseArrays: dropzeros!, issparse, findnz, SparseMatrixCSC
    #import NaNMath # returns a NaN instead of a DomainError for some functions. The solver(s) will use the NaN within the error control routines to reject the out of bounds step.

    import Distributions: cdf, Poisson, PoissonBinomial, Geometric, Exponential
    import Combinatorics: combinations
    import MultipleTesting
    #import LoopVectorization: @tturbo, @turbo  # not for now

    # actual source code
    include("Models/models.jl")
    include("Models/UBCM.jl")
    include("Models/DBCM.jl")
    include("Models/RBCM.jl")
    include("Models/BiCM.jl")
    include("Models/UECM.jl")
    include("Models/CReM.jl")
    include("Models/DCReM.jl")
    include("Models/CRWCM.jl")
    include("constraints.jl")
    include("utils.jl")
    include("metrics.jl")
    include("smallnetworks.jl")
    
    ### exports
    ## common types
    export AbstractMaxEntropyModel
    ## utils 
    # compressing
    # np_unique_clone, 
    # graph metrics
    export ANND, ANND_in, ANND_out
    export wedges, triangles, squares
    # reciprocity metrics
    export reciprocity, weighted_reciprocity
    export nonreciprocated_outdegree, nonreciprocated_indegree, reciprocated_degree
    export nonreciprocated_outstrength, nonreciprocated_instrength, reciprocated_outstrength, reciprocated_instrength
    for motif_name in directed_graph_motif_function_names
        @eval begin
        export $(motif_name)
        end
    end
    export V_motifs
    export Vn_motifs, Vn_sigma, Vn_zscore # bipartite Vn/Λn motif families (co-occurrence significance)
    export motifs # batched directed 3-node motif spectrum
    export motif_fluxes, motif_flux, motif_intensities # weighted triadic statistics
    export ensemble_zscores, motif_zscores, flux_zscores # sampling-based significance
    export project # bipartite graph or BiCM projection

    ## common model functions
    export initial_guess, solve_model!, Ĝ, set_Ĝ!, σˣ, set_σ!, set_xᵣ!, set_yᵣ!, set_zᵣ!, precision, σₓ
    export degree, outdegree, indegree
    export strength, outstrength, instrength
    export Ŵ, set_Ŵ!, σʷ, set_σʷ!
    export constraint_residual
    export AIC, AICc, BIC
    
    ## model specific types and functions
    export UBCM, L_UBCM_reduced, ∇L_UBCM_reduced!, UBCM_reduced_iter!
    export DBCM, L_DBCM_reduced, ∇L_DBCM_reduced!, DBCM_reduced_iter!
    export RBCM, L_RBCM_reduced, ∇L_RBCM_reduced!, RBCM_reduced_iter!
    export BiCM, L_BiCM_reduced, ∇L_BiCM_reduced!, BiCM_reduced_iter!
    export UECM, L_UECM_reduced, ∇L_UECM_reduced!, UECM_reduced_iter!
    export CReM, L_CReM, ∇L_CReM!, CReM_iter!
    export DCReM, L_DCReM, ∇L_DCReM!, DCReM_iter!
    export CRWCM, L_CRWCM, ∇L_CRWCM!, CRWCM_iter!

    ## demo networks
    export rhesus_macaques, taro_exchange, chesapeakebay, everglades, florida, littlerock, maspalomas, stmarks, corporateclub


    ## ------------------------------------------------ ##
    ## Precompiletools workload to accelate first usage ##
    ## ________________________________________________ ##
    ### Has a **substantial** impact on precompiletime of the package, but massive performance improvements for first usage 
    ### => 30x to >100x speedup for parameter computation
    using PrecompileTools
    using Preferences
    # The (expensive) precompile workload below is gated on a preference so developers can
    # disable it locally (set `precompile_workload = false` in LocalPreferences.toml) WITHOUT
    # mutating preferences at load time. Defaults to `true` for fast first-use in production.
    if @load_preference("precompile_workload", true)
        # UBCM workload
        @setup_workload begin
            G = MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate)
            @compile_workload begin
                # model building and solving
                model = UBCM(G)
                solve_model!(model)
                solve_model!(model, method=:BFGS)
                solve_model!(model, method=:BFGS, analytical_gradient=true)
                solve_model!(model, method=:LBFGS)
                solve_model!(model, method=:LBFGS, analytical_gradient=true)
                solve_model!(model, method=:Newton)
                solve_model!(model, method=:Newton, analytical_gradient=true)
                # sampling
                rand(model,10)
                # metrics
                # [TO DO]
            end
        end

        # DBCM workload
        @setup_workload begin
            G = MaxEntropyGraphs.maspalomas()
            @compile_workload begin
                # model building and solving
                model = DBCM(G)
                solve_model!(model)
                solve_model!(model, method=:BFGS)
                solve_model!(model, method=:BFGS, analytical_gradient=true)
                solve_model!(model, method=:LBFGS)
                solve_model!(model, method=:LBFGS, analytical_gradient=true)
                solve_model!(model, method=:Newton)
                solve_model!(model, method=:Newton, analytical_gradient=true)
                # sampling
                rand(model,10)
                # metrics
                # [TO DO]
            end
        end

        # RBCM workload (fixed point is stable for the RBCM on non-degenerate networks)
        @setup_workload begin
            G = MaxEntropyGraphs.Graphs.SimpleDiGraph(MaxEntropyGraphs.rhesus_macaques())
            @compile_workload begin
                # model building and solving
                model = RBCM(G)
                solve_model!(model)
                solve_model!(model, method=:BFGS)
                solve_model!(model, method=:BFGS, analytical_gradient=true)
                # sampling
                rand(model, 10)
                # metrics
                set_Ĝ!(model)
                set_σ!(model)
            end
        end

        # BiCM workload
        @setup_workload begin
            G = MaxEntropyGraphs.corporateclub()
            @compile_workload begin
                # model building and solving
                model = BiCM(G)
                solve_model!(model)
                solve_model!(model, method=:BFGS)
                solve_model!(model, method=:BFGS, analytical_gradient=true)
                solve_model!(model, method=:LBFGS)
                solve_model!(model, method=:LBFGS, analytical_gradient=true)
                solve_model!(model, method=:Newton)
                solve_model!(model, method=:Newton, analytical_gradient=true)
                # sampling
                rand(model,10)
                # metrics
                set_Ĝ!(model) # required before the precomputed=true projections below
                set_σ!(model)
                for layer in [:bottom, :top]
                    for precomputed in [true, false]
                        for distribution in [:Poisson, :PoissonBinomial]
                            project(model, layer=layer, precomputed=precomputed, distribution=distribution)
                        end
                    end
                    Vn_motifs(model, 2, layer=layer)
                    Vn_sigma(model, 2, layer=layer)
                    Vn_zscore(model, 2, layer=layer, method=:delta)
                end
            end
        end

        # UECM workload (fixed point is unstable for the UECM, so only BFGS/Newton are exercised)
        @setup_workload begin
            G = MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(MaxEntropyGraphs.rhesus_macaques())
            @compile_workload begin
                # model building and solving
                model = UECM(G)
                solve_model!(model, method=:BFGS)
                solve_model!(model, method=:BFGS, analytical_gradient=true)
                solve_model!(model, method=:Newton)
                solve_model!(model, method=:Newton, analytical_gradient=true)
                # sampling
                rand(model, 10)
                # metrics
                set_Ĝ!(model)
                set_σ!(model)
                set_Ŵ!(model)
                set_σʷ!(model)
            end
        end

        # DCReM workload (two-step: internal DBCM + weighted layer; fixed point is stable here)
        @setup_workload begin
            G = MaxEntropyGraphs.rhesus_macaques()
            @compile_workload begin
                # model building and solving
                model = DCReM(G)
                solve_model!(model)
                solve_model!(model, method=:BFGS)
                solve_model!(model, method=:BFGS, analytical_gradient=true)
                # sampling
                rand(model, 10)
                # metrics
                set_Ĝ!(model)
                set_σ!(model)
                set_Ŵ!(model)
                set_σʷ!(model)
            end
        end

        # CRWCM workload (two-step: internal RBCM + weighted layer over the four reciprocal strengths)
        @setup_workload begin
            G = MaxEntropyGraphs.rhesus_macaques()
            @compile_workload begin
                # model building and solving
                model = CRWCM(G)
                solve_model!(model)
                solve_model!(model, method=:BFGS)
                solve_model!(model, method=:BFGS, analytical_gradient=true)
                # sampling
                rand(model, 10)
                # metrics
                set_Ĝ!(model)
                set_σ!(model)
                set_Ŵ!(model)
                set_σʷ!(model)
            end
        end

        # CReM workload (two-step: internal UBCM + weighted layer; fixed point is stable here)
        @setup_workload begin
            G = MaxEntropyGraphs.SimpleWeightedGraphs.SimpleWeightedGraph(MaxEntropyGraphs.rhesus_macaques())
            @compile_workload begin
                # model building and solving
                model = CReM(G)
                solve_model!(model)
                solve_model!(model, method=:BFGS)
                solve_model!(model, method=:BFGS, analytical_gradient=true)
                solve_model!(model, method=:Newton)
                solve_model!(model, method=:Newton, analytical_gradient=true)
                # sampling
                rand(model, 10)
                # metrics
                set_Ĝ!(model)
                set_σ!(model)
                set_Ŵ!(model)
                set_σʷ!(model)
            end
        end
    end
end

