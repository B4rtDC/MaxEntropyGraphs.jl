##################################################################################
# UBCM_demo.jl
#
# This file contains some demos for the UBCM model
##################################################################################

using MaxEntropyGraphs
using Zygote
using OptimizationNLopt
using BenchmarkTools
# load the karate club graph
G = MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate)
# generate the model
model = UBCM(G)
## solve the model using different solvers
# 1. define the function to be maximized
fun = (θ, p) ->  - MaxEntropyGraphs.L_UBCM_reduced(θ, model.dᵣ, model.f)
θ₀ = -log.( model.dᵣ ./ maximum(model.dᵣ))
prob = MaxEntropyGraphs.Optimization.OptimizationProblem(fun, θ₀)

# 2. go the optimization
# using a gradient-free method (Nelder-Mead) with Optim.jl
# _________________________________________________________
@btime sol = MaxEntropyGraphs.Optimization.solve($prob, $(MaxEntropyGraphs.OptimizationOptimJL.NelderMead()))


# using a gradient based method with zygote
# __________________________________________
f = MaxEntropyGraphs.Optimization.OptimizationFunction(fun, MaxEntropyGraphs.Optimization.AutoZygote())
prob = MaxEntropyGraphs.Optimization.OptimizationProblem(f, θ₀)
@btime sol = MaxEntropyGraphs.Optimization.solve($prob, $(NLopt.LD_LBFGS()))
sol = MaxEntropyGraphs.Optimization.solve(prob, NLopt.LD_LBFGS())
sol.solve_time
model.xᵣ .= exp.(-sol.u)
pysol = [
2.851659905721819932011840137420
2.053008374478822783970599630265
1.543263951429110658608578887652
1.152360118330124638674760717549
0.827126749244577630371111354179
0.544504527280376793285654457577
-0.139872682463162062438399857456
-0.329325227165018830088882850760
-0.670620745297166087617313223745
-1.268557558407136909295331861358
-1.410096540785121144168101636751
]
model.xᵣ .= exp.(-pysol)
fun(sol.u, [])
fun(pysol, [])
G = getG(model)
sum(G, dims=2) ≈ model.dᵣ[model.dᵣ_ind] 


function getG(m)
        # check if possible
        #m.status[:params_computed] ? nothing : throw(UndefRefError("The parameters have not been computed yet"))
        
        # check network size
        n = m.status[:d]
        # initiate G
        G = zeros(Float64, n, n)
        # initiate x
        x = m.xᵣ[m.dᵣ_ind]
        # compute G
        for i = 1:n
            @simd for j = i+1:n
                @inbounds xij = x[i]*x[j]
                @inbounds G[i,j] = xij/(1 + xij)
                @inbounds G[j,i] = xij/(1 + xij)
            end
        end
    
        return G    
    end

MaxEntropyGraphs.Ĝ(model)


suite = BenchmarkGroup()
# gradient free methods
gradient_free_methods = [   "Nelder-Mead"=> MaxEntropyGraphs.OptimizationOptimJL.NelderMead(), 
                            "Particle swarm"  =>  MaxEntropyGraphs.OptimizationOptimJL.ParticleSwarm()]
suite["gradient-free"] = BenchmarkGroup([x[1] for x in gradient_free_methods])
for (method_name, method) in gradient_free_methods
    suite["gradient-free"][method_name] = @benchmarkable MaxEntropyGraphs.Optimization.solve($prob, $(method))
end
# gradient based methods
gradient_based_methods = ["L-BFGS" =>  NLopt.LD_LBFGS()] 
using ForwardDiff
using ReverseDiff
using Tracker
using Zygote
using FiniteDiff
using ModelingToolkit
suite["gradient-based"] = BenchmarkGroup([x[1] for x in gradient_based_methods])
autodiff_methods = [    "AutoForwardDiff" => AutoForwardDiff(), 
                        "AutoReverseDiff" => AutoReverseDiff(compile=false), 
                        "AutoTracker" => AutoTracker(), 
                        "AutoZygote" => AutoZygote(), 
                        "AutoFiniteDiff" => AutoFiniteDiff(), 
                        "AutoModelingToolkit" => AutoModelingToolkit()]
for (method_name, numerical_method) in gradient_based_methods
    for (autodiff_name, autodiff_method) in autodiff_methods
        f = MaxEntropyGraphs.Optimization.OptimizationFunction(fun, autodiff_method)
        prob = MaxEntropyGraphs.Optimization.OptimizationProblem(f, θ₀)
        suite["gradient-based"]["$(method_name)-$(autodiff_name)"] = @benchmarkable sol = MaxEntropyGraphs.Optimization.solve($prob, $(numerical_method))
    end
end
# c

tune!(suite)
suite  


results = run(suite, verbose = true)

