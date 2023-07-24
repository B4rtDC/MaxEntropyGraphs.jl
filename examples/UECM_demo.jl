##################################################################################
# UECM_demo.jl
#
# This file contains some demos for the UECM model
# UECM: Undirected Enhanced Configuration Model
##################################################################################

# setup for the UECM model
begin
    using Revise
    using BenchmarkTools
    # load up the module
    using MaxEntropyGraphs
    using Graphs
    using SimpleWeightedGraphs

    ## testing graph (single repreated pair)
    ## ______________________________________
    sources =       [1, 1, 1, 2, 3, 4, 6];
    destinations =  [2, 3, 7, 3, 4, 5, 7];
    weights =       [3, 3, 1, 1, 1, 3, 2];
    G = SimpleWeightedGraph(sources, destinations, float.(weights))
    # pythonsolutions
    L_pysol_newton = -20.553911287074285496601078193634748458862304687500000000000000
    L_θ₀_pysol_xy = -71.290312591266868480488483328372240066528320312500000000000000
    θ₀_pysol_xy = [0.980829253011726187594376824563369154930114746093750000000000,1.386294361119890572453527965990360826253890991210937500000000,0.980829253011726187594376824563369154930114746093750000000000,1.386294361119890572453527965990360826253890991210937500000000,2.079441541679835747657989486469887197017669677734375000000000,2.079441541679835747657989486469887197017669677734375000000000,1.386294361119890572453527965990360826253890991210937500000000,0.000000000000000000000000000000000000000000000000000000000000,1.386294361119890572453527965990360826253890991210937500000000,1.945910149055313453914095589425414800643920898437500000000000,1.722766597741103522523076208017300814390182495117187500000000,1.945910149055313453914095589425414800643920898437500000000000,2.233592221507094244259405968477949500083923339843750000000000,2.639057329615258851163162034936249256134033203125000000000000,2.233592221507094244259405968477949500083923339843750000000000,0.000000000000000000000000000000000000000000000000000000000000    ]
    θ₀_pysol_θ = exp.(-θ₀_pysol_xy)
    θ_pysol_newton = [-0.250106650402073915628875511174555867910385131835937500000000,0.397714139968617041986931326391641050577163696289062500000000,-1.066543425419856250258021646004635840654373168945312500000000,0.397714139968616930964628863875987008213996887207031250000000,2.157195583506671621165651231422089040279388427734375000000000,1.540972643411323561579706620250362902879714965820312500000000,-0.348568193952031479820874437791644595563411712646484375000000,11.512781380047256263310373469721525907516479492187500000000000,0.190090956210798467385103549531777389347553253173828125000000,0.342015805976093378859559379634447395801544189453125000000000,0.597270100969257367040654571610502898693084716796875000000000,0.342015805976093600904164304665755480527877807617187500000000,0.077533009627115956341469882318051531910896301269531250000000,0.340844844607292474236714951985049992799758911132812500000000,0.764361259104477608339323069230886176228523254394531250000000,9.448671652883239957532168773468583822250366210937500000000000]
    ∇L_θ₀ = [-2.870726454732364985034109849948436021804809570312500000000000,-1.951013085747823616600271634524688124656677246093750000000000,-2.909854302286123406418028025655075907707214355468750000000000,-1.951013085747823616600271634524688124656677246093750000000000,-0.981565408767152258739940862142248079180717468261718750000000,-0.988013836245684284875778757850639522075653076171875000000000,-1.963929980748802872625446980237029492855072021484375000000000,0.319916343772416844615236186655238270759582519531250000000000,-6.833006263096861765404810284962877631187438964843750000000000,-3.944101129785460457810586376581341028213500976562500000000000,-4.892973539274390049058638396672904491424560546875000000000000,-3.944101129785460457810586376581341028213500976562500000000000,-2.979718492723671285915543194278143346309661865234375000000000,-1.987248505202185100415590568445622920989990234375000000000000,-2.960292152557594480555280824773944914340972900390625000000000,0.392674138743432266096533567178994417190551757812500000000000    ]
    ∇L_θ_pysol = [0.000000000835696236545915570214756637988651188919675405486487,0.000000000375587439510019060999102695258457934590623494841566,0.000000001258222161933441964989439958893038817855369870812865,0.000000000375587383998867829741275674076874530049607869841566,0.000000000084229911847684271081387054422323244426418220598407,0.000000000119869621673100829736452579681190398763757087863269,0.000000000519240844744320642324755030844966180447741521675198,0.000000003573478814499730413495310167928953326565988390939310,0.000000000835866560394419370755690679636101467209208237818530,0.000000000375653622546135013515733083367298568433501060326307,0.000000001258390820496165231243699906204473043525027264877281,0.000000000375651846189295613265268405556629623121001060326307,0.000000000084250523018349429588672062190060152789272684970001,0.000000000119890920163105489672221515496308361092836491934577,0.000000000519300418795454489479456009230415314270601356838597,0.000000003573662041680076442013177549848583491876041762225213    ]

    add_vertex!(G) # for zero valued degree
    α_psyol = @view θ_pysol_newton[1:nv(G)]
    β_psyol = @view θ_pysol_newton[nv(G)+1:end]
    α₀_pysol = @view θ₀_pysol_θ[1:nv(G)]
    β₀_pysol = @view θ₀_pysol_θ[nv(G)+1:end]
    # obtain the degree and strength sequences
    d = degree(G)
    s = MaxEntropyGraphs.strength(G)
    @info "d: $(d)"
    @info "s: $(s)"
    nothing
end


begin
    # # mutiple repeated pairs > check python solution for this
    # sources =              [1, 1, 1, 2, 3, 4, 6,2,8,3];
    # destinations =         [2, 3, 7, 3, 4, 5, 7,3,6,4];
    # weights =       Float32[3, 3, 1, 1, 1, 3, 2,1,1,1];
    # L_pysol_newton = -25.256540564038221674536544014699757099151611328125000000000000
    # θ_pysol_newton = [-0.101035112479234454041510105071211000904440879821777343750000,0.801453857700332172875334890704834833741188049316406250000000,-0.101035112479234231996905180039902916178107261657714843750000,0.801453857700332061853032428189180791378021240234375000000000,2.057892277077995935030685359379276633262634277343750000000000,-0.251973447212675394357717095772386528551578521728515625000000,-0.251973447212675671913473252061521634459495544433593750000000,-19.015190306016556576196308014914393424987792968750000000000000,19.460258117195607496796583291143178939819335937500000000000000,0.233022146579048516912280319957062602043151855468750000000000,0.201185540666585532809662595354893710464239120483398437500000,0.233022146579048405889977857441408559679985046386718750000000,0.201185540666585505054086979725980199873447418212890625000000,0.121642819433363760794897245887113967910408973693847656250000,0.768809159535196684487345919478684663772583007812500000000000,0.768809159535196795509648381994338706135749816894531250000000,20.158987779432568032689232495613396167755126953125000000000000,1.464638125396839463121523294830694794654846191406250000000000    ]
    # G = SimpleWeightedGraph(sources, destinations, weights)
    # add_vertex!(G) # for zero valued degree
    # α_psyol = @view θ_pysol_newton[1:nv(G)]
    # β_psyol = @view θ_pysol_newton[nv(G)+1:end]
    # # obtain the degree and strength sequences
    # d = degree(G)
    # s = MaxEntropyGraphs.strength(G)
    # @info "d: $(d)"
    # @info "s: $(s)"

    
    #@info MaxEntropyGraphs.np_unique_clone(collect(zip(degree(G), MaxEntropyGraphs.strength(G))), sorted=true)
    


end



# custom precision, initiated from graph
begin
    prec = Float16
    model = MaxEntropyGraphs.UECM(G, precision=prec)
    # small checks
    @assert model.status[:N] == nv(G)
    @assert model.d == d
    @assert model.s == s
    @assert eltype(model.Θᵣ) == prec
    @assert eltype(model.xᵣ) == prec
    @assert eltype(model.yᵣ) == prec
    @assert model.dᵣ[model.dᵣ_ind] == model.d
    @assert model.sᵣ[model.dᵣ_ind] == model.s
end

# custom precision, initiated from degree and strength sequence
begin
    prec = Float32
    model = MaxEntropyGraphs.UECM(d=d,s=s, precision=prec)
    # small checks
    @assert model.status[:N] == length(d)
    @assert model.d == d
    @assert model.s == s
    @assert eltype(model.Θᵣ) == prec
    @assert eltype(model.xᵣ) == prec
    @assert eltype(model.yᵣ) == prec
    @assert model.dᵣ[model.dᵣ_ind] == model.d
    @assert model.sᵣ[model.dᵣ_ind] == model.s
end

# Likelihood computation
begin
    # generate a model
    model = MaxEntropyGraphs.UECM(G)
    
    ## using the solution values
    # compute the likelihood
    L = MaxEntropyGraphs.L_UECM_reduced(θ_pysol_newton, model.d, model.s, ones(length(model.d)), length(model.d))
    # compute the reduced likelihood
    α_psyolᵣ = α_psyol[model.d_ind]
    β_psyolᵣ = β_psyol[model.d_ind]
    Lᵣ = MaxEntropyGraphs.L_UECM_reduced(vcat(α_psyolᵣ, β_psyolᵣ), model.dᵣ, model.sᵣ, model.f, model.status[:d_unique])
    @info """with the solution values:
    L:          $(L)
    Lᵣ:         $(Lᵣ)
    Difference: $(L - Lᵣ)"""
    # check likelihood equivalence
    @assert L ≈ Lᵣ
    @assert L ≈ L_pysol_newton
    @assert Lᵣ ≈ L_pysol_newton

    ## using the initial guess
    # compute the likelihood
    L = MaxEntropyGraphs.L_UECM_reduced(θ₀_pysol_θ, model.d, model.s, ones(length(model.d)), length(model.d))
    # compute the reduced likelihood 
    α_psyolᵣ = α₀_pysol[model.d_ind]
    β_psyolᵣ = β₀_pysol[model.d_ind]
    Lᵣ = MaxEntropyGraphs.L_UECM_reduced(vcat(α_psyolᵣ, β_psyolᵣ), model.dᵣ, model.sᵣ, model.f, model.status[:d_unique])
    @info """with the solution values:
    L:          $(L)
    Lᵣ:         $(Lᵣ)
    Difference: $(L - Lᵣ)"""
    # check likelihood equivalence
    @assert L ≈ Lᵣ
end

# Gradient computation without compression
begin
    # generate a model
    model = MaxEntropyGraphs.UECM(G)
    # gradient buffers 
    ∇L = Vector{eltype(θ_pysol_newton)}(undef, length(θ_pysol_newton))
    x = Vector{eltype(θ_pysol_newton)}(undef, length(model.d))
    y = Vector{eltype(θ_pysol_newton)}(undef, length(model.s))
    # compute the gradient
    ∇L_anal = MaxEntropyGraphs.∇L_UECM_reduced!(∇L, θ_pysol_newton, model.d, model.s, ones(length(θ_pysol_newton)), x, y, length(model.d))
    @assert maximum(abs.(∇L_anal)) < 1e-8
    @assert isapprox(∇L_anal, ∇L_θ_pysol, rtol=1e-6)
end

# Gradient computation with compression
begin
    # generate a model
    model = MaxEntropyGraphs.UECM(G)
    # reduced values
    α_psyolᵣ = α_psyol[model.d_ind]
    β_psyolᵣ = β_psyol[model.d_ind]
    θ_pysolᵣ = vcat(α_psyolᵣ, β_psyolᵣ)
    # gradient buffers 
    ∇Lᵣ = Vector{eltype(θ_pysol_newton)}(undef, length(θ_pysolᵣ))
    xᵣ = Vector{eltype(θ_pysol_newton)}(undef, length(model.dᵣ))
    yᵣ = Vector{eltype(θ_pysol_newton)}(undef, length(model.sᵣ))
    # compute the gradient
    MaxEntropyGraphs.∇L_UECM_reduced!(∇Lᵣ, θ_pysolᵣ, model.dᵣ, model.sᵣ, model.f, xᵣ, yᵣ, length(model.dᵣ))
    @assert maximum(abs.(∇Lᵣ)) < 1e-8
    ∇Lᵣ_expand = vcat(∇Lᵣ[1:model.status[:d_unique]][model.dᵣ_ind], ∇Lᵣ[model.status[:d_unique]+1:end][model.dᵣ_ind])

end

# Gradient computation with autodiff and compression in the solution
begin
    using Zygote
    # generate a model
    model = MaxEntropyGraphs.UECM(G)
    
    ## gradient compute
    # compute the reduced likelihood parameters with autodiff
    α_psyolᵣ = α_psyol[model.d_ind]
    β_psyolᵣ = β_psyol[model.d_ind]
    ∇Lᵣ = Zygote.gradient(θ -> MaxEntropyGraphs.L_UECM_reduced(θ, model.dᵣ, model.sᵣ, model.f, model.status[:d_unique]), vcat(α_psyolᵣ, β_psyolᵣ))[1] # gradient is near zero
    @assert maximum(abs.(∇Lᵣ)) < 1e-8

    # compute the reduced likelihood gradient with analytical expression
    Gbuff = similar(∇Lᵣ)
    xᵣ = similar(α_psyolᵣ)
    yᵣ = similar(β_psyolᵣ)
    ∇Lᵣ_anal = MaxEntropyGraphs.∇L_UECM_reduced!(Gbuff, vcat(α_psyolᵣ, β_psyolᵣ), model.dᵣ, model.sᵣ, model.f, xᵣ, yᵣ, length(model.dᵣ))
    
    # reconstruct the gradient
    ∇L      = vcat(∇Lᵣ[1:length(α_psyolᵣ)][model.dᵣ_ind],      ∇Lᵣ[length(α_psyolᵣ)+1:end][model.dᵣ_ind])
    ∇L_anal = vcat(∇Lᵣ_anal[1:length(α_psyolᵣ)][model.dᵣ_ind], ∇Lᵣ_anal[length(α_psyolᵣ)+1:end][model.dᵣ_ind])
    
    # run some tests
    @assert L ≈ Lᵣ
    @assert isapprox(∇L,  ∇L_anal, rtol=1e-6)   # This is acceptable
    @info """maximum relative error between zygote and analytical gradient: $(MaxEntropyGraphs.@sprintf("%1.2e",maximum(abs.(∇L - ∇L_anal)) / maximum(abs.(∇L))))"""
    @error """maximum relative error between zygote and python:             $(MaxEntropyGraphs.@sprintf("%1.2e",maximum(abs.(∇L - ∇L_θ_pysol)) / maximum(abs.(∇L))))"""
    @assert !isapprox(∇L, ∇L_θ_pysol, rtol=1e-6)
end



# Optimization with Optim.jl and AutoZygote
begin
    let 
        # generate a model
        model = MaxEntropyGraphs.UECM(G)
        using Zygote
        # likelihood function for `Optimization.jl`
        obj = (θ, p) ->  - MaxEntropyGraphs.L_UECM_reduced(θ, model.d, model.s, ones(length(model.d)), length(model.d));
        f = MaxEntropyGraphs.Optimization.OptimizationFunction(obj, MaxEntropyGraphs.Optimization.AutoZygote());
        θ₀ = -log.([0.98082925, 1.38629436, 0.98082925, 1.38629436, 2.07944154,
        2.07944154, 1.38629436, 0.        , 1.38629436, 1.94591015,
        1.7227666 , 1.94591015, 2.23359222, 2.63905733, 2.23359222,
        0.        ]) #[model.d ./ maximum(model.d); model.s ./ maximum(model.s)]
        prob = MaxEntropyGraphs.Optimization.OptimizationProblem(f, θ₀);
        sol = MaxEntropyGraphs.Optimization.solve(prob, MaxEntropyGraphs.OptimizationOptimJL.BFGS() );
        @info length(sol.u), sol.u
        @info obj(sol.u, nothing)
    end

    let 
        # generate a model
        model = MaxEntropyGraphs.UECM(G)
        using Zygote
        # likelihood function for `Optimization.jl`
        obj = (θ, p) ->  - MaxEntropyGraphs.L_UECM_reduced(θ, model.dᵣ, model.sᵣ, model.f, length(model.dᵣ));
        f = MaxEntropyGraphs.Optimization.OptimizationFunction(obj, MaxEntropyGraphs.Optimization.AutoZygote());
        θ₀ = rand(length(model.dᵣ)*2)
        prob = MaxEntropyGraphs.Optimization.OptimizationProblem(f, θ₀);
        sol = MaxEntropyGraphs.Optimization.solve(prob, MaxEntropyGraphs.OptimizationOptimJL.BFGS() );
        @info length(sol.u), sort(sol.u)
        @info obj(sol.u, nothing)
        @assert -sol.objective ≈ L_pysol_newton
    end

end


# Solution with own gradient:
begin
    # generate a model
    model = MaxEntropyGraphs.UECM(G)
    # objective 
    obj = (θ, p) ->  - MaxEntropyGraphs.L_UECM_reduced(θ, model.dᵣ, model.sᵣ, model.f, length(model.dᵣ));
    # buffers
    x = Vector{eltype(θ_pysol_newton)}(undef, length(model.dᵣ))
    y = Vector{eltype(θ_pysol_newton)}(undef, length(model.sᵣ))
    #G = Vector{eltype(θ_pysol_newton)}(undef, length(model.θᵣ))
    # gradient
    grad! = (G, θ, p) -> MaxEntropyGraphs.∇L_UECM_reduced_minus!(G, θ,  model.dᵣ, model.sᵣ, model.f, x, y, length(model.dᵣ));
    @btime grad!($(zeros(length(model.θᵣ))), $(θ₀_pysol_θ), nothing) # Non-allocating :-)

    f = MaxEntropyGraphs.Optimization.OptimizationFunction(obj, grad=grad!);
    θ₀ = rand(length(model.θᵣ))
    prob = MaxEntropyGraphs.Optimization.OptimizationProblem(f, θ₀);
    sol = MaxEntropyGraphs.Optimization.solve(prob, MaxEntropyGraphs.OptimizationOptimJL.BFGS());
    sol.original
    @assert isapprox(-sol.objective, L_pysol_newton)
    @info sort(sol.u)
    sort(unique(round.(θ_pysol_newton, digits=12))) - sort(sol.u) # the values with non-zero indices are within tolerance
    
end

# Fixed point iteration
begin
    using NLsolve
    # generate a model
    model = MaxEntropyGraphs.UECM(G)
    x_buffer = zeros(length(model.dᵣ)); # buffer for x = exp(-α)
    y_buffer = zeros(length(model.sᵣ)); # buffer for y = exp(-β)
    G_buffer = zeros(length(model.θᵣ)); # buffer for G(x) θ
    FP_model! = (θ::Vector) -> MaxEntropyGraphs.UECM_reduced_iter!(θ, model.dᵣ, model.sᵣ, model.f, x_buffer, y_buffer, G_buffer, model.nz, length(model.dᵣ));

    # initial guess
    θ₀ = vcat(θ₀_pysol_θ[1:7], θ₀_pysol_θ[9:end-1])#rand(length(model.θᵣ))
    θ_sorted_pysol = vcat(α_psyol[model.d_ind], β_psyol[model.d_ind])
    θ₀ = vcat(α_psyol[model.d_ind], β_psyol[model.d_ind]) + (rand(14) .- 0.5) / 1#rand(length(model.θᵣ)) 
    θ₀[1] = 1e8; θ₀[8] = 1e8; 
    
    @btime FP_model!(θ₀) # Non-allocating :-)
    # solve
    sol = fixedpoint(FP_model!, θ₀, method=:anderson, ftol=1e-12, iterations=1000);
    hcat( sol.zero ,θ_sorted_pysol, sol.zero - θ_sorted_pysol)
end

# work without zero values
begin
    
    # work without zero values
    sources =       [1, 1, 1, 2, 3, 4, 6];
    destinations =  [2, 3, 7, 3, 4, 5, 7];
    weights =       [3, 3, 1, 1, 1, 3, 2];
    # graph
    G = SimpleWeightedGraph(sources, destinations, weights)
    d = degree(G)
    s = MaxEntropyGraphs.strength(G)
    θ_sol = vcat(θ_pysol_newton[1:nv(G)-1], θ_pysol_newton[nv(G)+1:end-1])
    # buffers
    x = Vector{eltype(θ_pysol_newton)}(undef, length(d))
    y = Vector{eltype(θ_pysol_newton)}(undef, length(s))
    G = Vector{eltype(θ_pysol_newton)}(undef, length(θ_pysol_newton))



    # try to optimise with own gradient
    obj = (θ, p) -> - LEUCM!(θ, d, s, x, y, length(d));
    grad! = (G, θ, p) -> ∇LEUCM_minus!(G, θ, d, s, x, y, length(d));

    f = MaxEntropyGraphs.Optimization.OptimizationFunction(obj, grad=grad!);
    prob = MaxEntropyGraphs.Optimization.OptimizationProblem(f, θ₀_pysol_xy);
    sol = MaxEntropyGraphs.Optimization.solve(prob, MaxEntropyGraphs.OptimizationOptimJL.LBFGS());
    @assert -sol.objective ≈ L_pysol_newton
end

# Generating expected adjacency Matrix
begin
    # generate a model
    model = MaxEntropyGraphs.UECM(G)
    # objective 
    obj = (θ, p) ->  - MaxEntropyGraphs.L_UECM_reduced(θ, model.dᵣ, model.sᵣ, model.f, length(model.dᵣ));
    # buffers
    x = Vector{eltype(θ_pysol_newton)}(undef, length(model.dᵣ))
    y = Vector{eltype(θ_pysol_newton)}(undef, length(model.sᵣ))
    # gradient
    grad! = (G, θ, p) -> MaxEntropyGraphs.∇L_UECM_reduced_minus!(G, θ,  model.dᵣ, model.sᵣ, model.f, x, y, length(model.dᵣ));
    # target and initial guess
    f = MaxEntropyGraphs.Optimization.OptimizationFunction(obj, grad=grad!);
    θ₀ = MaxEntropyGraphs.initial_guess(model)
    @info "f(θ₀): $(f(θ₀, nothing))"
    prob = MaxEntropyGraphs.Optimization.OptimizationProblem(f, θ₀);
    sol = MaxEntropyGraphs.Optimization.solve(prob, MaxEntropyGraphs.OptimizationOptimJL.BFGS());
    @info sol.original
    # store result
    model.θᵣ .= sol.u
    model.status[:params_computed] = true
    MaxEntropyGraphs.set_xᵣ!(model)
    MaxEntropyGraphs.set_yᵣ!(model)
    # compute the adjacency matrix
    A = MaxEntropyGraphs.Ĝ(model)
    @assert sum(A, dims=2) ≈ model.d # OK degrees
    W = MaxEntropyGraphs.Ŵ(model)
    @assert sum(A.*W, dims=2) ≈ model.s # OK strengths
    Q = A.*W
    y = model.yᵣ[model.dᵣ_ind]
    #P = [iszero(y[i]*y[j]) ? 1e-6 : MaxEntropyGraphs.Distributions.Geometric(y[i]*y[j]) for i in 1:length(y), j in 1:length(y)]

    sol
    nothing
end

# sampling the family of networks and checking coherence
begin
    fam = rand(model, 100)
    avg_d_diff = reshape(mean(hcat(MaxEntropyGraphs.Graphs.degree.(fam)...), dims=2) - model.d,:,1)
    avg_s_diff = reshape(mean(hcat(MaxEntropyGraphs.strength.(fam)...), dims=2) - model.s,:,1)
    avg_diff =  vcat(avg_d_diff,avg_s_diff)
    @assert maximum(abs.(avg_diff)) < 1e-1
end


# using the solve_model! funtion
begin
    sources =       [1, 1, 1, 2, 3, 4, 6];
    destinations =  [2, 3, 7, 3, 4, 5, 7];
    weights =       [3, 3, 1, 1, 1, 3, 2];
    G = SimpleWeightedGraph(sources, destinations, float.(weights))
    model = MaxEntropyGraphs.UECM(G)
    # solve model using the solve_model! function
    model, sol = MaxEntropyGraphs.solve_model!(model, verbose=true, analytical_gradient=true, method=:BFGS)
    model, sol = MaxEntropyGraphs.solve_model!(model, verbose=true, analytical_gradient=false, method=:BFGS)
    # get adjacency matrix
    A = MaxEntropyGraphs.Ĝ(model)
    @assert sum(A, dims=2) ≈ model.d#≈ model.d # OK degrees
    #
    #α_c = model.θᵣ[1:length(model.dᵣ)][model.dᵣ_ind] 
    #β_c = model.θᵣ[length(model.dᵣ)+1:end][model.dᵣ_ind]

end

#############################################################################################
#####################  Symbolic math computation to verify the gradient #####################
#############################################################################################
begin
    # definitions of different functions
    begin
        using Symbolics # for compute
        using Latexify  # for latex printing

        """
            myL(α, β, d, s,n)

        loglikelihood for UECM model, without compression
        """
        function myL(α, β, d, s,n)
            @info """MyL argument:
            α: $(α)
            β: $(β)
            d: $(d)
            s: $(s)
            n: $(n)
            """
            res = 0.
            for i in 1:n
                res -= d[i] * α[i] + s[i] * β[i]
                for j in 1:i-1
                    #res -= log( 1 + aux1 * aux2 / (1 - aux2) )
                    res -= log(1 + exp(-α[i] - α[j]) * exp(-β[i] - β[j]) / (1 - exp(-β[i] - β[j])))
                end
            end

            return res
        end

        """
            myL(α, β, d, s,n)

        gradient of the loglikelihood for UECM model, without compression
        """
        function ∇myL(α, β, d, s, n)
            @info """∇MyL argument:
            α: $(α)
            β: $(β)
            d: $(d)
            s: $(s)
            n: $(n)
            """
            res = Vector{Num}(undef, length(α)+length(β))
            # reset res
            res .= 0
            for i in 1:n
                res[i]   -= d[i]
                res[i+n] -= s[i]
                for j in 1:n
                    if i ≠ j
                        aux1 = exp(-α[i] - α[j])
                        aux2 = exp(-β[i] - β[j])
                        res[i]     += (aux1 * aux2) / (1 - aux2 + aux1 * aux2) # this appear to be OK
                        res[i + n] += (aux1 * aux2) / ((1 - aux2) * (1 - aux2 + aux1 * aux2))
                    end
                end
            end

            return res
        end


        """
            myL_r(α, β, d, s, f, n)

        loglikelihood for the reduced UECM model, without compression
        """
        function myL_r(α, β, d, s, f, n)
            @info """MyL_r argument:
            α: $(α)
            β: $(β)
            d: $(d)
            s: $(s)
            f: $(f)
            n: $(n)
            """
            res = 0.
            for i in 1:n
                res -= (d[i] * α[i] + s[i] * β[i]) * f[i]
                for j in 1:i
                    contrib = log(1 + exp(-α[i] - α[j]) * exp(-β[i] - β[j]) / (1 - exp(-β[i] - β[j])))
                    #contrib = log_nan( 1 + exp(-α[i] - α[j]) * exp(-β[i] - β[j]) / (1 - exp(-β[i] - β[j])) ) # for optimisation
                if i == j
                    res -=  f[i] * (f[j] - 1) * contrib * 0.5 # to avoid double counting
                else
                    res -=  f[i]*f[j]*contrib
                end
                end
            end

            return res
        end
    end

    # test cases
    begin
        # define symbolic variables
        Symbolics.@variables α[1:5]
        Symbolics.@variables β[1:5]
        Symbolics.@variables d[1:5]
        Symbolics.@variables s[1:5]

        # define which ones are unique for global model
        identical_values = [1;1;1;2;3;3]
        a = [α[i] for i in identical_values]
        b = [β[i] for i in identical_values]
        D = [d[i] for i in identical_values]
        S = [s[i] for i in identical_values]

        # define the reduced model with their frequencies
        ident_subset = [1;2;3]
        f = [3;1;2]
        Symbolics.@variables F[1:3]
        #F = [F[1];F[3]]
        a_sub = [α[i] for i in ident_subset]
        b_sub = [β[i] for i in ident_subset]
        D_sub = [d[i] for i in ident_subset]
        S_sub = [s[i] for i in ident_subset]

        ## compute the loglikelihoods
        # likelihood for global model with repeated values
        L_f = myL(a, b, D, S,length(a))
        # likelihood reduced model with numerical values for the frequencies (`f`)
        L_r = myL_r(a_sub, b_sub, D_sub, S_sub, f, length(a_sub))
        # likelihood reduced model with symbolic values for the frequencies (`F`)
        L_rF = myL_r(a_sub, b_sub, D_sub, S_sub, F, length(a_sub))
        # check differences
        Ldiff = simplify(L_f - L_r,expand=true)
        @info """"full:\n"$(L_f)"""
        @info """"reduced:\n"$(L_r)"""
        @info """Difference: $(Ldiff)"""

        ## Compute the gradient symbolicallly
        ∇L_f_a = Symbolics.derivative(L_f, α[1])
        ∇L_f_b = Symbolics.derivative(L_f, β[1])
        ∇L_r_a = Symbolics.derivative(L_r, α[1])
        ∇L_r_b = Symbolics.derivative(L_r, β[1])

        # compute the gradient with the repeated values
        gradcom = ∇myL(a, b, D, S,length(a)) 
        # compute the gradient for the reduced likelihood with numerical values for the frequencies (`f`)
        gradc = vcat([Symbolics.derivative(L_r, α[i]) for i in 1:length(F)], [Symbolics.derivative(L_r, β[i]) for i in 1:length(F)]) 
        # compute the gradient for the reduced likelihood with symbolic values for the frequencies (`F`)
        gradcF = vcat([Symbolics.derivative(L_rF, α[i]) for i in 1:length(F)], [Symbolics.derivative(L_rF, β[i]) for i in 1:length(F)]) # "ground truth"

        ## check if gradient is working properly
        # ∂L/∂αᵢ
        @assert iszero(simplify(sum(gradcom[1:3]) - gradc[1])) # OK
        @assert iszero(simplify(sum(gradcom[4]  ) - gradc[2])) # OK
        @assert iszero(simplify(sum(gradcom[5:6]) - gradc[3])) # OK
        # ∂L/∂βᵢ
        #@assert iszero(simplify(sum(gradcom[7:9]) - gradc[4])) # OK

    end

    # writing results to latex for the paper
    begin    
        println("∂Lc/∂α₂:\n$(latexify(gradcF[2]))")
        println("∂Lc/∂β₂:\n$(latexify(gradcF[5]))")
    end

    # simplifying some observations:
    begin
        begin
            # partial deritivative of the log-likelihood wrt α
            @variables aux[1:2]
            myexpre = aux[1]*aux[2] / ((1 + aux[1]*aux[2]/(1-aux[2])) * (1 - aux[2]))
            println("before simplification:\n$(latexify(myexpre))")
            println("after simplification:\n$(latexify(Symbolics.simplify(myexpre, expand=true)))")
        end

        begin
            # partial deritivative of the log-likelihood wrt β
            @variables aux[1:2]
            myexpre = (aux[1]*aux[2]/(1 - aux[2]) + aux[1]*aux[2]/(1 - aux[2])^2 * aux[2]) / (1 + aux[1]*aux[2]/(1 - aux[2]))
            println("before simplification:\n$(latexify(myexpre))")
            println("after simplification:\n$(latexify(Symbolics.simplify(myexpre, expand=true)))")
        end

        begin 
            # loglikelihood function
            println("L_r:\n$(latexify(L_rF))")
        end
    end
end