##################################################################################
# BiCM_demo.jl
#
# This file contains some demos for the BiCM model
# BiCM: Bipartite Configuration Model
##################################################################################

# setup for the BiCM model
begin
    using Revise
    using BenchmarkTools
    # load up the module
    using MaxEntropyGraphs
    using Graphs
    #using MultilayerGraphs

    # setup for the test-network
    N = 11
    G = MaxEntropyGraphs.Graphs.SimpleGraph(N)
    for (src,dst) in [(1,2); (1, 4); (5,4); (5,6);(5,8);(7,4);(9,8);(11,8);(11,10)]
        add_edge!(G, src, dst)
    end
    # check bipirtiteness
    @assert MaxEntropyGraphs.Graphs.is_bipartite(G)
    # get membership
    membership = MaxEntropyGraphs.Graphs.bipartite_map(G)
    ⊥nodes, ⊤nodes = findall(membership .== 1), findall(membership .== 2)
    # degree sequences
    d⊥, d⊤ = MaxEntropyGraphs.Graphs.degree(G, ⊥nodes), MaxEntropyGraphs.Graphs.degree(G, ⊤nodes)

    # Python results
    pysol_r_x = [0.282967243090962550233768979524029418826103210449218750000000
    0.953554248902308665414295774098718538880348205566406250000000
    2.661954894848762798176267097005620598793029785156250000000000]
    pysol_r_α = -log.(pysol_r_x)
    pysol_r_y = [0.281832732500096383443377590083400718867778778076171875000000
    2.251946638178301096644418066716752946376800537109375000000000
    ]
    pysol_r_β = -log.(pysol_r_y)
    pysol_r_x0 = [0.333333333333333314829616256247390992939472198486328125000000
    0.666666666666666629659232512494781985878944396972656250000000
    1.000000000000000000000000000000000000000000000000000000000000
    0.333333333333333314829616256247390992939472198486328125000000
    1.000000000000000000000000000000000000000000000000000000000000]
    pysol_L = - 12.722758944946432890787946234922856092453002929687500000000000
    pysol_∇L = [0.000000000000021316282072803005576133728027343750000000000000
    0.000000000000006661338147750939242541790008544921875000000000
    0.000000000000029087843245179101359099149703979492187500000000
    0.000000000000014210854715202003717422485351562500000000000000
    0.000000000000019984014443252817727625370025634765625000000000]
    pysol_time_newton = 0.005248591899871826
    pysol_time_FP = 0.0037072968482971193
    # initiate pysolution
    m = MaxEntropyGraphs.BiCM(G)
    θ_r_pysol = fill(Inf, length(m.θᵣ))
    θ_r_pysol[m.d⊥ᵣ_nz] = pysol_r_α
    θ_r_pysol[m.d⊤ᵣ_nz .+ last(m.d⊥ᵣ_nz)] = pysol_r_β
    # initiate initial guess
    x0_pysol = fill(Inf, length(m.θᵣ))
    x0_pysol[m.d⊥ᵣ_nz] = pysol_r_x0[1:length(m.d⊥ᵣ_nz)]
    x0_pysol[m.d⊤ᵣ_nz .+ last(m.d⊥ᵣ_nz)] = pysol_r_x0[length(m.d⊥ᵣ_nz)+1:end]
    # initiate gradients
    ∇L_pysol = fill(0.0, length(m.θᵣ))
    ∇L_pysol[m.d⊥ᵣ_nz] = pysol_∇L[1:length(m.d⊥ᵣ_nz)]
    ∇L_pysol[m.d⊤ᵣ_nz .+ last(m.d⊥ᵣ_nz)] = pysol_∇L[length(m.d⊥ᵣ_nz)+1:end]
    θ_r_pysol, x0_pysol, ∇L_pysol
end

begin
    # adjacency matrix (global)
    A = MaxEntropyGraphs.Graphs.adjacency_matrix(G)
    # bi-adjacency matrix (local)
    Abi = A[⊥nodes, ⊤nodes]
    Abi*Abi'

    Abi' * Abi
end

# defining the BiCM model from our graph
begin
    model = MaxEntropyGraphs.BiCM(G)
    @assert model.d⊥ == d⊥
    @assert model.d⊥ᵣ[model.d⊥ᵣ_ind] == d⊥
    @assert model.d⊤ == d⊤
    @assert model.d⊤ᵣ[model.d⊤ᵣ_ind] == d⊤
    @assert model.status[:N] == N
    @info model.d⊥ᵣ, model.d⊤ᵣ
    @info model.d⊥ᵣ_nz, model.d⊤ᵣ_nz
end


# defining the BiCM model from our degree sequences
begin
    model = MaxEntropyGraphs.BiCM(d⊥=d⊥, d⊤=d⊤)
end

# testing the likelihood function
begin 
    m = MaxEntropyGraphs.BiCM(G)
    m.θᵣ .= θ_r_pysol
    @info m.θᵣ
    @info m.status[:d⊥_unique]
    @info MaxEntropyGraphs.L_BiCM_reduced(m.θᵣ, m.d⊥ᵣ, m.d⊤ᵣ, m.f⊥, m.f⊤, m.d⊥ᵣ_nz, m.d⊤ᵣ_nz, m.status[:d⊥_unique])
    @assert MaxEntropyGraphs.L_BiCM_reduced(θ_r_pysol, m.d⊥ᵣ, m.d⊤ᵣ, m.f⊥, m.f⊤, m.d⊥ᵣ_nz, m.d⊤ᵣ_nz, m.status[:d⊥_unique]) ≈ pysol_L
    @assert MaxEntropyGraphs.L_BiCM_reduced(m) ≈ pysol_L
end

# testing the gradient function
begin
    m = MaxEntropyGraphs.BiCM(G)
    m.θᵣ .= θ_r_pysol
    # buffers
    grad_buff = zeros(length(m.θᵣ))
    x_buff = zeros(length(m.d⊥ᵣ))
    y_buff = zeros(length(m.d⊤ᵣ))
    # actual function
    n = m.status[:d⊥_unique]
    gradL! = (∇L,θ) -> MaxEntropyGraphs.∇L_BiCM_reduced!(   ∇L, θ, 
                                                            m.d⊥ᵣ, m.d⊤ᵣ, 
                                                            m.f⊥, m.f⊤, 
                                                            m.d⊥ᵣ_nz, m.d⊤ᵣ_nz,
                                                            x_buff, y_buff,
                                                            n)

    @btime gradL!(grad_buff, x0_pysol) # two allocations remain: for UnitRange (,why?)
 # two allocations remain: for UnitRange (,why?)
end

# Evaluating allocations for other function to make them faster
begin
    using Profile
    using PProf
    # clear current
    Profile.Allocs.clear()
    # sample function (looking at all allocations)
    @time Profile.Allocs.@profile sample_rate=1 gradL!(grad_buff, x0_pysol)
    # serve up the result in the browser for analysis
    PProf.Allocs.pprof(from_c=false)
end



# auto gradient to check manually coded gradient => OK
begin
    using Zygote
    m = MaxEntropyGraphs.BiCM(G)
    m.θᵣ .= θ_r_pysol
    
    for point in [rand(length(θ_r_pysol)), rand(length(θ_r_pysol)), rand(length(θ_r_pysol)), ones(length(θ_r_pysol)), x0_pysol, θ_r_pysol ]
        autograd = first(Zygote.gradient(θ -> MaxEntropyGraphs.L_BiCM_reduced(θ, m.d⊥ᵣ, m.d⊤ᵣ, m.f⊥, m.f⊤, m.d⊥ᵣ_nz, m.d⊤ᵣ_nz, m.status[:d⊥_unique]), point)) 
        analgrad = gradL!(grad_buff, point) 
        @info """Point: $(point)\n
    - autodiff :  $(autograd)
    - analgrad:   $(analgrad)
    - difference: $(autograd-analgrad)
    - ≈:          $(autograd ≈ analgrad)"""
    end

end

# testing the iterative method
begin
    # generate model
    m = MaxEntropyGraphs.BiCM(G)
    m.θᵣ .= θ_r_pysol
    # buffers
    x_buff = zeros(length(m.d⊥ᵣ))
    y_buff = zeros(length(m.d⊤ᵣ))
    buf = fill(Inf, length(m.θᵣ))
    # testing
    MaxEntropyGraphs.BiCM_reduced_iter!(θ_r_pysol, m.d⊥ᵣ, m.d⊤ᵣ, m.f⊥, m.f⊤, m.d⊥ᵣ_nz, m.d⊤ᵣ_nz, x_buff, y_buff, buf, m.status[:d⊥_unique])
    @assert buf ≈ θ_r_pysol
end


# using a gradient based method (LBDFGS) with Optim.jl using Zygote for automatic differentiation => OK (for small scale problems, i.e. less than 100 parameters, you should you ForwardDiff.jl instead of Zygote.jl)
begin
    using Zygote
    # make the model
    m = MaxEntropyGraphs.BiCM(G)
    # set the initial guess
    θ₀ = initial_guess(m)
    # define the objective function
    obj = (θ, p) -> - MaxEntropyGraphs.L_BiCM_reduced(θ,  m.d⊥ᵣ, m.d⊤ᵣ, m.f⊥, m.f⊤, m.d⊥ᵣ_nz, m.d⊤ᵣ_nz, m.status[:d⊥_unique])
    f = MaxEntropyGraphs.Optimization.OptimizationFunction(obj, MaxEntropyGraphs.Optimization.AutoZygote());
    # defin the optimisation problem
    prob = MaxEntropyGraphs.Optimization.OptimizationProblem(f, θ₀);
    # # actual optimisation (gradient free)
    sol = MaxEntropyGraphs.Optimization.solve(prob, MaxEntropyGraphs.OptimizationOptimJL.Newton());
    @info "Likelihood value: $(obj(sol.u, nothing))"
    @info """Difference in likelihood with NEMtropy: $(MaxEntropyGraphs.@sprintf("%1.2e",-obj(sol.u, nothing) - pysol_L)) (i.e. we get a slightly better likelihood)"""
    nanind = .!isinf.(sol.u)
    @info """Largest difference in parameter value: $(MaxEntropyGraphs.@sprintf("%1.2e",maximum(abs.(sol.u[nanind] - θ_r_pysol[nanind]))))"""
    # store solution and compute adjacency matrix
    m.θᵣ .= sol.u
    m.status[:params_computed] = true;
    set_xᵣ!(m) 
    set_yᵣ!(m)
    A = set_Ĝ!(m) 
    @info "Maximum degree error ⊥ layer (autodiff + Newton): $(MaxEntropyGraphs.@sprintf("%.2e", maximum(abs.(sum(Ĝ(m), dims=2) .- m.d⊥))))"
    @info "Maximum degree error ⊤ layer (autodiff + Newton): $(MaxEntropyGraphs.@sprintf("%.2e", maximum(abs.(reshape(sum(Ĝ(m), dims=1),:,1) .- d⊤))))"
    BiCM_autodiff_perf = @benchmark MaxEntropyGraphs.Optimization.solve($(prob), $(MaxEntropyGraphs.OptimizationOptimJL.Newton()));
    @info "Median compute time for for newton iteration (autodiff): $(MaxEntropyGraphs.@sprintf("%2.2e", median(BiCM_autodiff_perf).time/1e9)) - speedup vs. NEMtropy: x$(MaxEntropyGraphs.@sprintf("%1.2f", pysol_time_newton/(median(BiCM_autodiff_perf).time/1e9))))";
end 


# using the analytical gradient with Optim.jl => OK
begin
    # make the model
    m = MaxEntropyGraphs.BiCM(G)
    # set the initial guess
    θ₀ = initial_guess(m)
    #infind = isinf.(θ₀)
    #θ₀[infind] .= 1e8
    @info θ₀
    # define the objective function
    obj = (θ, p) -> - MaxEntropyGraphs.L_BiCM_reduced(θ,  m.d⊥ᵣ, m.d⊤ᵣ, m.f⊥, m.f⊤, m.d⊥ᵣ_nz, m.d⊤ᵣ_nz, m.status[:d⊥_unique])
    # define the gradient and the required buffers
    # buffers
    grad_buff = zeros(length(m.θᵣ))
    x_buff = zeros(length(m.d⊥ᵣ))
    y_buff = zeros(length(m.d⊤ᵣ))
    grad! = (G, θ, p) -> MaxEntropyGraphs.∇L_BiCM_reduced_minus!(G, θ, m.d⊥ᵣ, m.d⊤ᵣ, m.f⊥, m.f⊤, m.d⊥ᵣ_nz, m.d⊤ᵣ_nz, x_buff, y_buff, m.status[:d⊥_unique])
    f = MaxEntropyGraphs.Optimization.OptimizationFunction(obj, grad=grad!);
    # defin the optimisation problem
    prob = MaxEntropyGraphs.Optimization.OptimizationProblem(f, θ₀);
    # actual optimisation
    sol = MaxEntropyGraphs.Optimization.solve(prob, MaxEntropyGraphs.OptimizationOptimJL.LBFGS());
    @info "Likelihood value: $(obj(sol.u, nothing))"
    @info """Difference in likelihood with NEMtropy: $(MaxEntropyGraphs.@sprintf("%1.2e",-obj(sol.u, nothing) - pysol_L)) (i.e. we get a slightly $(-obj(sol.u, nothing) > pysol_L ? "better" : "worse" ) likelihood)"""
    nanind = .!isinf.(sol.u)
    @info """Largest difference in parameter value: $(MaxEntropyGraphs.@sprintf("%1.2e",maximum(abs.(sol.u[nanind] - θ_r_pysol[nanind]))))"""
    # store solution and compute adjacency matrix
    m.θᵣ .= sol.u
    m.status[:params_computed] = true;
    set_xᵣ!(m) 
    set_yᵣ!(m)
    A = set_Ĝ!(m) 
    @info "Maximum degree error ⊥ layer (analytical gradient + LBFGS): $(MaxEntropyGraphs.@sprintf("%.2e", maximum(abs.(sum(Ĝ(m), dims=2) .- m.d⊥))))"
    @info "Maximum degree error ⊤ layer (analytical gradient + LBFGS): $(MaxEntropyGraphs.@sprintf("%.2e", maximum(abs.(reshape(sum(Ĝ(m), dims=1),:,1) .- d⊤))))"
    BiCM_analgrad_perf = @benchmark MaxEntropyGraphs.Optimization.solve($(prob), $(MaxEntropyGraphs.OptimizationOptimJL.LBFGS()));
    @info "Median compute time (analytical gradient + LBFGS): $(MaxEntropyGraphs.@sprintf("%2.2e", median(BiCM_analgrad_perf).time/1e9)) - speedup vs. NEMtropy: x$(MaxEntropyGraphs.@sprintf("%1.2f", pysol_time_newton/(median(BiCM_analgrad_perf).time/1e9))))";
end 

# using the fixed point iteration method => OK!
begin
    using NLsolve
    # make the model
    m = MaxEntropyGraphs.BiCM(G)
    # set the initial guess
    θ₀ = initial_guess(m);
    # avoid Inf values for compute
    ind_inf = θ₀ .== Inf
    θ₀[ind_inf] .= 1e8
    # set buffers
    x_buffer = zeros(length(m.d⊥ᵣ)); # buffer for x = exp(-θ)
    y_buffer = zeros(length(m.d⊤ᵣ));  # buffer for y = exp(-θ)
    G_buffer = zeros(length(m.θᵣ)); # buffer for G(x)
    BiCM_FP! = (θ::Vector) -> MaxEntropyGraphs.BiCM_reduced_iter!(θ, m.d⊥ᵣ, m.d⊤ᵣ, m.f⊥, m.f⊤, m.d⊥ᵣ_nz, m.d⊤ᵣ_nz, x_buffer, y_buffer, G_buffer, m.status[:d⊥_unique])
    sol = fixedpoint(BiCM_FP!, θ₀, method=:anderson, ftol=1e-12, iterations=1000);
    m.θᵣ .= sol.zero;
    m.θᵣ[ind_inf] .= Inf # rectify to Inf
    m.status[:params_computed] = true;

    @info "Likelihood value: $(MaxEntropyGraphs.L_BiCM_reduced(m))"
    @info """Difference in likelihood with NEMtropy: $(MaxEntropyGraphs.@sprintf("%1.2e",MaxEntropyGraphs.L_BiCM_reduced(m) - pysol_L)) (i.e. we get a slightly $(MaxEntropyGraphs.L_BiCM_reduced(m) > pysol_L ? "better" : "worse" ) likelihood)"""
    # store solution and compute adjacency matrix
    set_xᵣ!(m) 
    set_yᵣ!(m)
    A = set_Ĝ!(m) 
    @info "Maximum degree error ⊥ layer (FP): $(MaxEntropyGraphs.@sprintf("%.2e", maximum(abs.(sum(Ĝ(m), dims=2) .- m.d⊥))))"
    @info "Maximum degree error ⊤ layer (FP): $(MaxEntropyGraphs.@sprintf("%.2e", maximum(abs.(reshape(sum(Ĝ(m), dims=1),:,1) .- d⊤))))"
    BiCM_FP_perf = @benchmark fixedpoint(BiCM_FP!, θ₀, method=:anderson, ftol=1e-12, iterations=1000);
    @info "Median compute time for for FP iteration: $(MaxEntropyGraphs.@sprintf("%2.2e", median(BiCM_FP_perf).time/1e9)) - speedup vs. NEMtropy: x$(MaxEntropyGraphs.@sprintf("%1.2f", pysol_time_FP/(median(BiCM_FP_perf).time/1e9))))";
end

# sampling the BiCM 
begin
    membership = Graphs.bipartite_map(m.G)
    ⊥nodes, ⊤nodes = findall(membership .== 1), findall(membership .== 2)
    ⊥map, ⊤map = Dict(enumerate(⊥nodes)), Dict(enumerate(⊤nodes))
    for (i, or⊥) in enumerate(⊥nodes)
        for (j, or⊤) in enumerate(⊤nodes)
            @info "⊥ ↦ ⊤ linke ($(i),$(j)) is originally ($(or⊥), $(or⊤))"
        end
    end
end

begin
    edges = Vector{Tuple{Int,Int}}()
    for (i, i_or) in enumerate(m.⊥nodes)
        for (j, j_or) in enumerate(m.⊤nodes)
            if rand() <= m.Ĝ[i,j] 
                push!(edges, (i_or, j_or))
            end
        end
    end
    degree(Graphs.SimpleDiGraphFromIterator( Graphs.Edge.(edges))) - degree(m.G)
end

"""
function rand(m::DBCM; precomputed::Bool=false)
    if precomputed
        # check if possible to use precomputed Ĝ
        m.status[:G_computed] ? nothing : throw(ArgumentError("The expected adjacency matrix has not been computed yet"))
        # generate random graph
        G = Graphs.SimpleDiGraphFromIterator( Graphs.Edge.([(or⊥,or⊤) for (i,or⊥) in enumerate(m.⊥nodes) for (j,or⊤) in enumerate(m.⊤nodes) if rand()<m.Ĝ[i,j]]))
    else
        # check if possible to use parameters
        m.status[:params_computed] ? nothing : throw(ArgumentError("The parameters have not been computed yet"))
        # initiate x and y
        x = m.xᵣ[m.dᵣ_ind]
        y = m.yᵣ[m.dᵣ_ind]
        # generate random graph
        # G = Graphs.SimpleGraphFromIterator(Graphs.Edge.([(i,j) for i = 1:m.status[:d] for j in i+1:m.status[:d] if rand()< (x[i]*x[j])/(1 + x[i]*x[j]) ]))
        G = Graphs.SimpleDiGraphFromIterator(Graphs.Edge.([(i,j) for i = 1:m.status[:d] for j in   1:m.status[:d] if (rand() < (x[i]*y[j])/(1 + x[i]*y[j]) && i≠j) ]))
    end

    # deal with edge case where no edges are generated for the last node(s) in the graph
    while Graphs.nv(G) < m.status[:d]
        Graphs.add_vertex!(G)
    end

    return G
end

using Zygote
grad!(grad_buff, θ₀, nothing) 

#- first(gradient(θ -> obj(θ, nothing), θ₀))

# using the fixed point iteration method 
begin
    # make the model
    m = MaxEntropyGraphs.BiCM(G)
    # set the initial guess
    θ₀ = initial_guess(m)
end

# Crafting a bipartite network in the MultilayerGraphs environment (PM)begin
begin

    nA = 4
    nB = 3
    # node list
    nodes_list = [Node("node_$i") for i in 1:nA +nB + 1]
    # multilayervertices
    multilayervertices = MV.(nodes_list)
    # layer creation
    # Layer A
    layer_A = Layer(:layer_A, multilayervertices[[collect(1:nA); end]],     0, SimpleGraph{Int64}(), Int64)
    @info layer_A, nodes(layer_A)
    # Layer B
    layer_B = Layer(:layer_B, multilayervertices[nA+1:end], 0, SimpleGraph{Int64}(), Int64)
    @info layer_B, nodes(layer_B)
    # generate the interlayer and its edges
    interlayer = Interlayer(layer_A, layer_B, SimpleGraph{Int64}(), MultilayerEdge{Nothing}[])
    internodes = nodes(interlayer)
    add_edge!(interlayer, ME(intervertices[1], intervertices[5]))
    add_edge!(interlayer, ME(intervertices[2], intervertices[5]))
    add_edge!(interlayer, ME(intervertices[2], intervertices[6]))
    add_edge!(interlayer, ME(intervertices[3], intervertices[6]))
    add_edge!(interlayer, ME(intervertices[6], intervertices[4]))
    add_edge!(interlayer, ME(intervertices[4], intervertices[7]))
    add_edge!(interlayer, ME(intervertices[8], intervertices[5]))
    @info interlayer, nodes(interlayer)
    intervertices = mv_vertices(interlayer)
    
    # generate the multilayer graph
    GML = MultilayerGraph([layer_A, layer_B], [interlayer])
    @info GML
    @info "GML nodes: $(nodes(GML))"
    @info "GML is bipartite: $(is_bipartite(GML))"
    @info """GML edges $(eltype(edges(GML))): $(prod(["\n - $(e)" for e in edges(GML)]))"""

    @info degree(GML, vertices(layer_A)), degree(GML, vertices(layer_B)), degree(GML)
    #add_edge!(Gml, multilayervertices[1], multilayervertices[5])
   # LA = Layer(:LA, Node[],MultilayerEdge[], MultilayerGraphs.Graphs.SimpleGraphs.SimpleGraph, Int64)
    #MultilayerGraph(nA, nB)
    #degree(Gml)
    #is_weighted(Gml)
    #is_bipartite(Gml)
   # , degree(layer_A)
end     
