##################################################################################
# UBCM_demo.jl
#
# This file contains some demos for the UBCM model
##################################################################################

begin
using Revise
using BenchmarkTools
# Load up the module
using MaxEntropyGraphs

# python NEMtropy solution(s)
const UBCM_pysol_L = 168.683251363028119840237195603549480438232421875000000000000000
const UBCM_pysol = [
2.851659905681046325298666488379240036010742187500000000000000
2.053008374129531521390390480519272387027740478515625000000000
1.543263951525769339667704116436652839183807373046875000000000
1.152360116147663315899762892513535916805267333984375000000000
0.827126749819360962412417848099721595644950866699218750000000
0.544504527715587993696999546955339610576629638671875000000000
-0.139872682088905353481322890729643404483795166015625000000000
-0.329325226769753010014341043643071316182613372802734375000000
-0.670620744875011332020164900313830003142356872558593750000000
-1.268557557961735193785557385126594454050064086914062500000000
-1.410096540337653747698709594260435551404953002929687500000000
]
const ∇UBCM_pysol = [
-0.000000000000000083266726846886740531772375106811523437500000
0.000000006753889891797371092252433300018310546875000000000000
-0.000000000162921676150062921806238591670989990234375000000000
0.000000039616037117440328074735589325428009033203125000000000
-0.000000003675222126631183527933899313211441040039062500000000
-0.000000001316600828360492414503823965787887573242187500000000
0.000000000000001998401444325281772762537002563476562500000000
0.000000000000002109423746787797426804900169372558593750000000
-0.000000000000000111022302462515654042363166809082031250000000
-0.000000000000000444089209850062616169452667236328125000000000
-0.000000000000001443289932012703502550721168518066406250000000
]
const UBCM_pysol_iter = [
2.851659905681046325298666488379240036010742187500000000000000
2.053008374436526395356850116513669490814208984375000000000000
1.543263951516718135437145065225195139646530151367187500000000
1.152360117798331806682199385249987244606018066406250000000000
0.827126749574346065152496976224938407540321350097656250000000
0.544504527605871424533745539520168676972389221191406250000000
-0.139872682088905325725747275100729893893003463745117187500000
-0.329325226769752898992038581127417273819446563720703125000000
-0.670620744875011109975559975282521918416023254394531250000000
-1.268557557961735193785557385126594454050064086914062500000000
-1.410096540337653747698709594260435551404953002929687500000000
]
const UBCM_newton_time = 0.013710975646972656
const UBCM_quasinewton_time = 0.020680904388427734
const UBCM_fixedpoint_time = 0.005337715148925781
end



##################################################################################
# UBCM_demo.jl
##################################################################################
# Julia version
begin
G = MaxEntropyGraphs.Graphs.SimpleGraphs.smallgraph(:karate);
model = UBCM(G);
θ₀ = initial_guess(model);
# Python version
python_model = deepcopy(model);
python_model.xᵣ .= exp.(-UBCM_pysol);
python_model.Θᵣ .= UBCM_pysol;
python_model.status[:params_computed] = true;
# pure likelihood function
model_fun = θ -> - L_UBCM_reduced(θ, model.dᵣ, model.f);
# likelihood function for `Optimization.jl`
obj = (θ, p) ->  - L_UBCM_reduced(θ, model.dᵣ, model.f);
end

# using a gradient-free method (Nelder-Mead) with Optim.jl
# ________________________________________________________
begin
prob = MaxEntropyGraphs.Optimization.OptimizationProblem(obj, θ₀);
# actual optimisation (gradient free)
sol = MaxEntropyGraphs.Optimization.solve(prob, (MaxEntropyGraphs.OptimizationOptimJL.NelderMead()));
model.Θᵣ .= sol.u; 
model.status[:params_computed] = true;
set_xᵣ!(model);
# precision check on imposed constraints
@info "Maximum degree error for NelderMead: $(MaxEntropyGraphs.@sprintf("%.2e", maximum(abs.(sum(Ĝ(model), dims=2) .- model.dᵣ[model.dᵣ_ind]))))"
@info "Maximum degree error for NEMtropy:   $(MaxEntropyGraphs.@sprintf("%.2e", maximum(abs.(sum(Ĝ(python_model), dims=2) .- python_model.dᵣ[python_model.dᵣ_ind]))))"
@info "Difference between entropy values:   $(MaxEntropyGraphs.@sprintf("%.2e", model_fun(model.Θᵣ) - model_fun(UBCM_pysol)))"
UBCM_GF_perf = @benchmark MaxEntropyGraphs.Optimization.solve($(prob), $(MaxEntropyGraphs.OptimizationOptimJL.NelderMead()));
@info "Median compute time for NelderMead:  $(MaxEntropyGraphs.@sprintf("%2.2es", median(UBCM_GF_perf).time/1e9)) - speedup vs. NEMtropy (quasinewton): x$(MaxEntropyGraphs.@sprintf("%1.2f", UBCM_quasinewton_time/(median(UBCM_GF_perf).time/1e9))))";
end


# using a gradient based method (LBDFGS) with Optim.jl using Zygote for automatic differentiation
# _______________________________________________________________________________________________
begin
using Zygote
f = MaxEntropyGraphs.Optimization.OptimizationFunction(obj, MaxEntropyGraphs.Optimization.AutoZygote());
prob = MaxEntropyGraphs.Optimization.OptimizationProblem(f, θ₀);
sol = MaxEntropyGraphs.Optimization.solve(prob, MaxEntropyGraphs.OptimizationOptimJL.LBFGS(), );
model.Θᵣ .= sol.u;
model.status[:params_computed] = true;
set_xᵣ!(model);
# precision check on imposed constraints
@info "Maximum degree error for L-FBGS with AutoZygote: $(MaxEntropyGraphs.@sprintf("%.2e", maximum(abs.(sum(Ĝ(model), dims=2) .- model.dᵣ[model.dᵣ_ind]))))"
@info "Maximum degree error for NEMtropy:               $(MaxEntropyGraphs.@sprintf("%.2e", maximum(abs.(sum(Ĝ(python_model), dims=2) .- python_model.dᵣ[python_model.dᵣ_ind]))))"
@info "Difference between entropy values:               $(MaxEntropyGraphs.@sprintf("%.2e", model_fun(model.Θᵣ) - model_fun(UBCM_pysol)))"
UBCM_BG_perf = @benchmark MaxEntropyGraphs.Optimization.solve($(prob), $(MaxEntropyGraphs.OptimizationOptimJL.LBFGS()));
@info "Median compute time for L-FBGS with AutoZygote:  $(MaxEntropyGraphs.@sprintf("%2.2es", median(UBCM_BG_perf).time/1e9)) - speedup vs. NEMtropy (quasinewton): x$(MaxEntropyGraphs.@sprintf("%1.2f", UBCM_quasinewton_time/(median(UBCM_BG_perf).time/1e9))))";
# checking the quality of the gradient function
@assert ∇L_UBCM_reduced!(UBCM_pysol, model.dᵣ, model.f, zeros(length( model.dᵣ)), zeros(length( model.dᵣ))) == ∇UBCM_pysol
end


# using a fixed point approach leveraging NLsolve.jl through anderson acceleration
# ________________________________________________________________________________
begin
using NLsolve
x_buffer = zeros(length(model.dᵣ)); # buffer for x = exp(-θ)
G_buffer = zeros(length(model.dᵣ)); # buffer for G(x)
FP_model! = (θ::Vector) -> UBCM_reduced_iter!(θ, model.dᵣ, model.f, x_buffer, G_buffer);
sol = fixedpoint(FP_model!, θ₀, method=:anderson, ftol=1e-12, iterations=1000);
model.Θᵣ .= sol.zero;
model.status[:params_computed] = true;
set_xᵣ!(model);
# precision check for FP iteration
@assert UBCM_reduced_iter!(UBCM_pysol, model.dᵣ, model.f, x_buffer, G_buffer) == UBCM_pysol_iter
@info "Maximum degree error for FP iteration: $(MaxEntropyGraphs.@sprintf("%.2e", maximum(abs.(sum(Ĝ(model), dims=2) .- model.dᵣ[model.dᵣ_ind]))))"
@info "Maximum degree error for NEMtropy:     $(MaxEntropyGraphs.@sprintf("%.2e", maximum(abs.(sum(Ĝ(python_model), dims=2) .- python_model.dᵣ[python_model.dᵣ_ind]))))"
@info "Difference between entropy values:     $(MaxEntropyGraphs.@sprintf("%.2e", model_fun(model.Θᵣ) - model_fun(UBCM_pysol)))"
UBCM_FP_perf = @benchmark fixedpoint($(FP_model!), $(θ₀), method=:anderson, ftol=1e-12, iterations=1000);
@info "Median compute time for FP iteration:  $(MaxEntropyGraphs.@sprintf("%2.2es", median(UBCM_FP_perf).time/1e9)) - speedup vs. NEMtropy (fixedpoint): x$(MaxEntropyGraphs.@sprintf("%1.2f", UBCM_fixedpoint_time/(median(UBCM_FP_perf).time/1e9))))";
end

UBCM_fixedpoint_time / 3.85e-05
# obtain the expected adjacency matrix of the UBCM model
# ______________________________________________________
begin
    Ĝ_UBCM = Ĝ(model);
    # store the expected adjacency matrix in the model object
    set_Ĝ!(model);
end

# obtain the standard deviation of the expected adjacency matrix of the UBCM model
# ________________________________________________________________________________
begin
    # compute the standard deviation of the expected adjacency matrix
    σ̂_UBCM = σˣ(model);
    # store the standard deviation of the expected adjacency matrix in the model object
    set_σ!(model);
end

# sampling a network from the UBCM model
# ______________________________________
begin
    rand(model)
end










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

