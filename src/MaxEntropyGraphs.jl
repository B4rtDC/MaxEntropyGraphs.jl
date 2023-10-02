"""
    MaxEntropyGraphs

Julia module for working with maximum entropy graphs
"""
module MaxEntropyGraphs
    import Base: show, rand, showerror, precision
    # for logmessages
    import Printf: @sprintf
    #import Dates: now, Day, Minute 
    

    # to work with all sorts of graphs
    import Graphs
    import SimpleWeightedGraphs

    # to solve the optimization problem
    import Optimization
    import OptimizationOptimJL
    import ForwardDiff, ReverseDiff, Zygote
    import NLsolve
    import LinearAlgebra: issymmetric
    #import NaNMath # returns a NaN instead of a DomainError for some functions. The solver(s) will use the NaN within the error control routines to reject the out of bounds step.

    import Distributions
    import Combinatorics: combinations
    #import LoopVectorization: @tturbo, @turbo  # not for now

    # actual source code
    include("Models/models.jl")
    include("Models/UBCM.jl")
    include("Models/DBCM.jl")
    include("Models/BiCM.jl")
    #include("Models/UECM.jl")
    #include("Models/CReM.jl")
    include("utils.jl")
    include("metrics.jl")
    include("smallnetworks.jl")
    
    ### exports
    ## common types
    export AbstractMaxEntropyModel
    ## utils 
    # compressing
    export np_unique_clone
    # graph metrics
    export ANND, ANND_in, ANND_out
    export wedges, triangles, squares
    for motif_name in directed_graph_motif_function_names
        @eval begin
        export $(motif_name)
        end
    end

    ## common model functions
    export initial_guess, solve_model!, Ĝ, set_Ĝ!, σˣ, set_σ!, set_xᵣ!, set_yᵣ!, precision, σₓ
    export degree, outdegree, indegree
    export AIC, AICc, BIC
    
    ## model specific types and functions
    export UBCM, L_UBCM_reduced, ∇L_UBCM_reduced!, UBCM_reduced_iter!
    export DBCM, L_DBCM_reduced, ∇L_DBCM_reduced!, DBCM_reduced_iter!
    export BiCM, L_BiCM_reduced, ∇L_BiCM_reduced!, BiCM_reduced_iter!
    
    ## demo networks
    export rhesus_macaques, taro_exchange, chesapeakebay, everglades, florida, littlerock, maspalomas, stmarks, corporateclub


    ## ------------------------------------------------ ##
    ## Precompiletools workload to accelate first usage ##
    ## ________________________________________________ ##
    ### Has a **substantial** impact on precompiletime of the package, but massive performance improvements for first usage 
    ### => 30x to >100x speedup for parameter computation
    using PrecompileTools
    using Preferences
    # disable during development
    set_preferences!(MaxEntropyGraphs, "precompile_workload" => false; force=true)
    let
        @setup_workload begin
            G = MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate)
            @compile_workload begin
                # UBCM workload
                model = UBCM(G)
                solve_model!(model)
                solve_model!(model, method=:BFGS)
                solve_model!(model, method=:BFGS, analytical_gradient=true)
                # DBCM workload
            end
        end
    end
end

