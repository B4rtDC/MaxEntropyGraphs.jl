##################################################################################
# DBCM_demo.jl
#
# This file contains some demos for the DBCM model
##################################################################################

begin
    using Revise
    using BenchmarkTools
    # Load up the module
    using MaxEntropyGraphs
    
    # python settings 
    const edgelist_py = [(0, 2),
    (1, 8),
    (1, 13),
    (2, 3),
    (2, 5),
    (2, 10),
    (2, 14),
    (2, 17),
    (3, 0),
    (3, 6),
    (3, 7),
    (3, 8),
    (4, 7),
    (4, 19),
    (5, 3),
    (5, 10),
    (5, 12),
    (6, 10),
    (7, 4),
    (8, 5),
    (8, 14),
    (9, 4),
    (9, 5),
    (9, 18),
    (10, 4),
    (10, 12),
    (11, 0),
    (11, 3),
    (11, 7),
    (11, 15),
    (11, 17),
    (12, 6),
    (12, 8),
    (12, 18),
    (13, 2),
    (13, 3),
    (13, 4),
    (13, 5),
    (14, 4),
    (14, 8),
    (14, 19),
    (15, 2),
    (15, 8),
    (15, 16),
    (15, 17),
    (16, 2),
    (17, 1),
    (17, 7),
    (17, 9),
    (17, 19),
    (18, 9),
    (18, 13),
    (18, 16),
    (18, 17),
    (19, 16)] 
    const edgelist_jul = [edge .+ (1,1) for edge in edgelist_py]
    const d_out_py = [1., 2., 5., 4., 2., 3., 1., 1., 2., 3., 2., 5., 3., 4., 3., 4., 1., 4., 4., 1.]
    const d_in_py  = [2., 1., 4., 4., 5., 4., 2., 4., 5., 2., 3., 0., 2., 2., 2., 1., 3., 4., 2., 3.]
    const d_r_out_py = [1., 1., 1., 2., 2., 2., 3., 3., 4., 4., 4., 5., 5.]
    const d_r_in_py  = [2., 3., 4., 1., 3., 5., 2., 4., 1., 2., 4., 0., 4.]
    const d_r_out_nz_py = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]
    const d_r_in_nz_py  = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 12]
    const F_r_py =  [2, 2, 1, 1, 1, 2, 3, 1, 1, 2, 2, 1, 1]
    # python NEMtropy solution(s)
    const DBCM_pysol_L = -140.024610793795318386401049792766571044921875000000000000000000
    const DBCM_pysol_alpha = [1.981160141317101830438218712515663355588912963867187500000000,1.959941522570228578103979089064523577690124511718750000000000,1.935725469173093182817524393612984567880630493164062500000000,1.232759259644305549485920892038848251104354858398437500000000,1.189022937587318695662474965502042323350906372070312500000000,1.137073941778148800096914783352985978126525878906250000000000,0.725859771090746064281518101779511198401451110839843750000000,0.676064144843745440383031564124394208192825317382812500000000,0.378570991970373582624631580983987078070640563964843750000000,0.352551660693394663947941580772749148309230804443359375000000,0.301374016447932324691549865747219882905483245849609375000000,0.097186852896756686925883172989415470510721206665039062500000,-0.013834421741768944402739549559555598534643650054931640625000    ]
    const DBCM_pysol_beta  = [1.388761003744900524026206767302937805652618408203125000000000,0.905502169119993927104417252849088981747627258300781250000000,0.535350182042531819170960716292029246687889099121093750000000,2.136921769368254953036512233666144311428070068359375000000000,0.882204908579033109106148913269862532615661621093750000000000,0.195609393060857422730336452332267072051763534545898437500000,1.345022275861593952228645321156363934278488159179687500000000,0.483530796363261350379048053582664579153060913085937500000000,2.091426793023562158424510926124639809131622314453125000000000,1.320035452208659032535820188059005886316299438476562500000000,0.458258322810216167653152297134511172771453857421875000000000,Inf,0.433991454737065240898630236188182607293128967285156250000000]
    const DCBM_pysol_x     = [0.137909150363763999269650639689643867313861846923828125000000,0.140866658200307220960567633483151439577341079711914062500000,0.144319531465120026725301727310579735785722732543945312500000,0.291487178231027543873210561287123709917068481445312500000000,0.304518652488828145408206182764843106269836425781250000000000,0.320756201346866109958000379265286028385162353515625000000000,0.483908339649038898855337720306124538183212280273437500000000,0.508614892611114699505492353637237101793289184570312500000000,0.684839351234354576369867118046386167407035827636718750000000,0.702892256971376139240703651012154296040534973144531250000000,0.739801023244610278517541246401378884911537170410156250000000,0.907386442506778467809169796964852139353752136230468750000000,1.013930560182561890769648016430437564849853515625000000000000]
    const DBCM_pyol_y      = [0.249384099259535502168816378798510413616895675659179687500000,0.404338787659476173175221447309013456106185913085937500000000,0.585464235181901049287489513517357409000396728515625000000000,0.118017569755395013153531635907711461186408996582031250000000,0.413869360799378638304801825142931193113327026367187500000000,0.822333381078034886257910329732112586498260498046875000000000,0.260533904178144815055873095843708142638206481933593750000000,0.616602446159916883772211804171092808246612548828125000000000,0.123510785712432749616773719480988802388310432434082031250000,0.267125831597258611704859276869683526456356048583984375000000,0.632384095869056972460953147674445062875747680664062500000000,0.000000000000000000000000000000000000000000000000000000000000,0.647917792089588218118478835094720125198364257812500000000000]
    const ∇DBCM_pysol = [
        -0.000000000000000444089209850062616169452667236328125000000000
        -0.000000000000000222044604925031308084726333618164062500000000
        -0.000000000000000111022302462515654042363166809082031250000000
        0.000000000000000444089209850062616169452667236328125000000000
        0.000000000000000444089209850062616169452667236328125000000000
        0.000000000000000888178419700125232338905334472656250000000000
        0.000000000000000000000000000000000000000000000000000000000000
        0.000000000000000888178419700125232338905334472656250000000000
        0.000000000000000888178419700125232338905334472656250000000000
        0.000000000000000000000000000000000000000000000000000000000000
        0.000000000000000000000000000000000000000000000000000000000000
        0.000000000000000000000000000000000000000000000000000000000000
        0.000000000000000000000000000000000000000000000000000000000000
        -0.000000000000000444089209850062616169452667236328125000000000
        0.000000000000000888178419700125232338905334472656250000000000
        0.000000000000000000000000000000000000000000000000000000000000
        0.000000000000000444089209850062616169452667236328125000000000
        -0.000000000000000444089209850062616169452667236328125000000000
        0.000000000000000000000000000000000000000000000000000000000000
        0.000000000000000000000000000000000000000000000000000000000000
        0.000000000000000000000000000000000000000000000000000000000000
        0.000000000000000000000000000000000000000000000000000000000000
        0.000000000000000888178419700125232338905334472656250000000000
        0.000000000000000000000000000000000000000000000000000000000000
        0.000000000000000000000000000000000000000000000000000000000000
        0.000000000000000000000000000000000000000000000000000000000000
    ]
    const DBCM_pysol_iter = [ # these are the alpha/beta values
        1.981160141317101608393613787484355270862579345703125000000000
        1.959941522570228578103979089064523577690124511718750000000000
        1.935725469173092960772919468581676483154296875000000000000000
        1.232759259644305993575130742101464420557022094726562500000000
        1.189022937587318695662474965502042323350906372070312500000000
        1.137073941778148800096914783352985978126525878906250000000000
        0.725859771090746064281518101779511198401451110839843750000000
        0.676064144843745884472241414187010377645492553710937500000000
        0.378570991970373749158085274757468141615390777587890625000000
        0.352551660693394663947941580772749148309230804443359375000000
        0.301374016447932324691549865747219882905483245849609375000000
        0.097186852896756686925883172989415470510721206665039062500000
        -0.013834421741768944402739549559555598534643650054931640625000
        1.388761003744900301981601842271629720926284790039062500000000
        0.905502169119994038126719715364743024110794067382812500000000
        0.535350182042531819170960716292029246687889099121093750000000
        2.136921769368255397125722083728760480880737304687500000000000
        0.882204908579032887061543988238554447889328002929687500000000
        0.195609393060857422730336452332267072051763534545898437500000
        1.345022275861593952228645321156363934278488159179687500000000
        0.483530796363261350379048053582664579153060913085937500000000
        2.091426793023562158424510926124639809131622314453125000000000
        1.320035452208659254580425113090313971042633056640625000000000
        0.458258322810216167653152297134511172771453857421875000000000
        Inf
        0.433991454737065240898630236188182607293128967285156250000000
    ]
    const DBCM_newton_time = 0.12884306907653809
    const DBCM_quasinewton_time = 0.0054857730865478516
    const DBCM_fixedpoint_time = 0.0038568973541259766

    nothing
end

# testing of the dimension reduction method => works out of the box :-)
begin
    dᵣ, d_ind , dᵣ_ind, f = MaxEntropyGraphs.np_unique_clone(collect(zip(d_out_py, d_in_py)), sorted=true)
    dᵣ_out = [t[1] for t in dᵣ]
    dᵣ_in = [t[2] for t in dᵣ]
    @assert dᵣ_out == d_r_out_py
    @assert dᵣ_in == d_r_in_py
    @assert f == F_r_py
    @assert dᵣ_out[dᵣ_ind] == d_out_py
    @assert dᵣ_in[dᵣ_ind] == d_in_py
end

# defining the BDCM model from our edgelist
begin
    G = MaxEntropyGraphs.Graphs.SimpleDiGraphFromIterator(MaxEntropyGraphs.Graphs.Edge(e) for e  in edgelist_jul)
    # check coherence
    @assert MaxEntropyGraphs.Graphs.outdegree(G) == d_out_py 
    @assert MaxEntropyGraphs.Graphs.indegree(G)  == d_in_py
    # make the model
    model = MaxEntropyGraphs.DBCM(G)
    # check coherence
    @assert model.d_out == MaxEntropyGraphs.Graphs.outdegree(G)
    @assert model.d_in  == MaxEntropyGraphs.Graphs.indegree(G)  
    @assert model.d_out == d_out_py
    @assert model.d_in  == d_in_py
    @assert model.dᵣ_out == d_r_out_py
    @assert model.dᵣ_in  == d_r_in_py
    @assert model.f == F_r_py
    @assert model.dᵣ_out_nz == d_r_out_nz_py .+ 1 # julia is 1-based
    @assert model.dᵣ_in_nz  == d_r_in_nz_py .+ 1  # julia is 1-based
    @assert model.dᵣ_out[model.dᵣ_ind] == d_out_py
    @assert model.dᵣ_in[model.dᵣ_ind]  == d_in_py
end

# defining the BDCM model from our degree sequences
begin
    # make the model
    model = MaxEntropyGraphs.DBCM(d_out=Int32.(d_out_py), d_in=Int32.(d_in_py))
    # check coherence
    @assert model.d_out == d_out_py
    @assert model.d_in  == d_in_py
    @assert model.dᵣ_out == d_r_out_py
    @assert model.dᵣ_in  == d_r_in_py
    @assert model.f == F_r_py
    @assert model.dᵣ_out_nz == d_r_out_nz_py .+ 1 # julia is 1-based
    @assert model.dᵣ_in_nz  == d_r_in_nz_py .+ 1  # julia is 1-based
end

# testing of the entropy function using the solution parameters
begin
    model = MaxEntropyGraphs.DBCM(d_out=Int32.(d_out_py), d_in=Int32.(d_in_py))
    # setting the solution parameters
    θ = vcat(DBCM_pysol_alpha, DBCM_pysol_beta) 
    model.θᵣ .= θ
    model.xᵣ .= exp.(-DBCM_pysol_alpha)
    model.yᵣ .= exp.(-DBCM_pysol_beta)
    # testing the likelihood function using the solution parameters
    @assert MaxEntropyGraphs.L_DBCM_reduced(model) ≈ DBCM_pysol_L
    @assert MaxEntropyGraphs.L_DBCM_reduced(vcat(DBCM_pysol_alpha, DBCM_pysol_beta), d_r_out_py, d_r_in_py, F_r_py, d_r_out_nz_py .+1 , d_r_in_nz_py .+1) ≈ DBCM_pysol_L
    @btime MaxEntropyGraphs.L_DBCM_reduced(model)
    nothing
end

# testing the iterative method
begin
    model = MaxEntropyGraphs.DBCM(d_out=Int32.(d_out_py), d_in=Int32.(d_in_py))
    # setting the solution parameters
    θ = vcat(DBCM_pysol_alpha, DBCM_pysol_beta) 
    model.θᵣ .= θ
    model.xᵣ .= exp.(-DBCM_pysol_alpha)
    model.yᵣ .= exp.(-DBCM_pysol_beta)
    
    # Initiating the buffers
    #θbuffer = copy(model.θᵣ)
    x = similar(model.xᵣ)
    y = similar(model.yᵣ)
    G = similar(model.xᵣ)
    H = similar(model.yᵣ)
    
    res = MaxEntropyGraphs.DBCM_reduced_iter!(θ, model.dᵣ_out, model.dᵣ_in, model.f, model.dᵣ_out_nz, model.dᵣ_in_nz, x, y, G, H, model.status[:d_unique])

    @assert res ≈ DBCM_pysol_iter
    @btime MaxEntropyGraphs.DBCM_reduced_iter!(θ, model.dᵣ_out, model.dᵣ_in, model.f, model.dᵣ_out_nz, model.dᵣ_in_nz, x, y, G, H, model.status[:d_unique]);
    nothing

end

# testing the gradient function
begin
    model = MaxEntropyGraphs.DBCM(d_out=Int32.(d_out_py), d_in=Int32.(d_in_py))
    # setting the solution parameters
    θ = vcat(DBCM_pysol_alpha, DBCM_pysol_beta) 
    model.θᵣ .= θ
    model.xᵣ .= exp.(-DBCM_pysol_alpha)
    model.yᵣ .= exp.(-DBCM_pysol_beta)
    
    # Initiating the buffers
    ∇L = similar(model.θᵣ) 
    x =  similar(model.xᵣ)
    y =  similar(model.yᵣ)

    # run compute => gradients are non-allocating :-)
    ∇L_model = θ -> MaxEntropyGraphs.∇L_DBCM_reduced!(∇L, θ, model.dᵣ_out, model.dᵣ_in, model.f, model.dᵣ_out_nz, model.dᵣ_in_nz, x, y, model.status[:d_unique])
    @assert ∇L_model(θ) ≈ ∇DBCM_pysol
    @btime ∇L_model(θ);

    ∇L_model_minus = θ -> MaxEntropyGraphs.∇L_DBCM_reduced_minus!(∇L, θ, model.dᵣ_out, model.dᵣ_in, model.f, model.dᵣ_out_nz, model.dᵣ_in_nz, x, y, model.status[:d_unique])
    @assert ∇L_model_minus(θ) ≈ -∇DBCM_pysol
    @btime ∇L_model_minus(θ);
    nothing
end







## Evaluating allocations for other function to make them faster
# ______________________________________________________________
begin
    using Profile
    using PProf
    # clear current
    Profile.Allocs.clear()
    # sample function (looking at all allocations)
    @time Profile.Allocs.@profile sample_rate=1 MaxEntropyGraphs.L_DBCM_reduced(model);
    # serve up the result in the browser for analysis
    PProf.Allocs.pprof(from_c=false)
end



## Testing an optimisation problem with a pre-defined gradient
# ______________________________________________________________
begin
    ## Directly
    import OptimizationOptimJL
    # function
    foo(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2 
    # gradient
    function g!(G, x)
        G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
        G[2] = 200.0 * (x[2] - x[1]^2)
    end
    # initial_guess 
    x0 = [0.0, 0.0]
    # optimise
    res = OptimizationOptimJL.optimize(foo, g!, x0, OptimizationOptimJL.BFGS())
    res.minimizer

    ## Through Optimization.jl
    import Optimization
    # function
    Foo = (x,p) -> foo(x)
    # gradient
    G! = (G,x,p) -> g!(G,x)
    # initial_guess
    x0 = [0.0, 0.0]

    # optimise with own gradient
    fun = Optimization.OptimizationFunction(Foo, grad=G!)
    prob = Optimization.OptimizationProblem(fun, x0)
    sol = Optimization.solve(prob, OptimizationOptimJL.BFGS())
    #@btime Optimization.solve(prob, $(OptimizationOptimJL.BFGS()))
    
    # optimise with atuodiff gradient
    fun_2 = Optimization.OptimizationFunction(Foo, Optimization.AutoForwardDiff())
    prob_2 = Optimization.OptimizationProblem(fun_2, x0)
    sol_2 = Optimization.solve(prob_2, OptimizationOptimJL.BFGS())

    @info sol.solve_time, sol_2.solve_time
    @assert sol.u ≈ sol_2.u
end



