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
    #include("Models/BiCM.jl")
    #include("Models/UECM.jl")
    #include("Models/CReM.jl")
    include("utils.jl")
    ## exports
    # common types
    export AbstractMaxEntropyModel
    # utils 
    export np_unique_clone, ANND, AAND_in, ANND_out, wedges, triangles, squares
    export rhesus_macaques, taro_exchange # demo networks
    # common model functions
    export initial_guess, solve_model!, Ĝ, set_Ĝ!, σˣ, set_σ!, set_xᵣ!, precision, σₓ
    export degree, outdegree, indegree
    export AIC, AICc, BIC #, set_yᵣ!
    # model specific types and functions
    export UBCM, L_UBCM_reduced, ∇L_UBCM_reduced!, UBCM_reduced_iter!
    export DBCM, L_DBCM_reduced, ∇L_DBCM_reduced!, DBCM_reduced_iter!
    #export BiCM, L_BiCM_reduced, ∇L_BiCM_reduced!, BiCM_reduced_iter!


    #import Distributions
    
    #import PyCall                   # for calling NEMtropy package in Python, dependency should be removed in a future version
    #import ReverseDiff              # for gradient
    #import StatsBase: mean, std     # for mean and standard deviation
    #import JLD2                     # for saving and loading models
    #
    #
    #include("metrics.jl")
    
    
    # models
    #export AbstractMaxEntropyModel, UBCM, DBCM
    #export rand
    #export σˣ

    # metrics
    #export degree, indegree, outdegree, indegree_dist, outdegree_dist
    #export ANND, ANND_in, ANND_out
    #for foo in DBCM_motif_function_names
    #    @eval begin
    #    export $(foo)
    #    end
    #end
    #export motifs

    # utils
    #export DBCM_analysis

end


    #import Printf: @sprintf         # for specific printing
#using Plots                     # for plotting
#using Measures                  # for margin settings
#using LaTeXStrings              # for LaTeX printing

#using GraphIO                   # to read and write external graphs


# using SimpleWeightedGraphs

#using NLsolve
#using IndirectArrays
#using LoopVectorization
# MODELS

# METRICS
#export degree



#include("metrics.jl")
#end # module
