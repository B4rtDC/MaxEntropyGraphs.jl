

# ----------------------------------------------------------------------------------------------------------------------
#
#                                               model testing
#
# ----------------------------------------------------------------------------------------------------------------------

function UBCM_test()
    NP = PyCall.pyimport("NEMtropy")

    mygraph = Graphs.smallgraph(:karate) # quick demo - Zachary karate club
    mygraph_nem = NP.UndirectedGraph(degree_sequence=Graphs.degree(mygraph))
    mygraph_nem.solve_tool(model="cm", method="fixed-point", initial_guess="random");

    M = UBCM(mygraph_nem.x)         # generate the model
    S = [rand(M) for _ in 1:10000]; # sample 10000 graphs from the model

    ## degree metric testing
    dˣ = Graphs.degree(mygraph)                         # observed degree sequence ("d star")
    inds = sortperm(dˣ)
    # Squartini method
    d̂  = degree(M)                                      # expected degree sequence from the model
    σ_d = map(n -> σˣ(G -> degree(G, n), M), 1:length(M))      # standard deviation for each degree in the sequence
    z_d = (dˣ - d̂) ./ σ_d
    # Sampling method
    S_d = hcat(map( g -> Graphs.degree(g), S)...);       # degree sequences from the sample
    # illustration 
    # - acceptable domain
    p1 = scatter(dˣ[inds], dˣ[inds], label="observed", color=:black, marker=:xcross, 
                xlabel="observed node degree", ylabel="node degree", legend_position=:topleft,
                bottom_margin=5mm, left_margin=5mm)
    scatter!(p1, dˣ[inds], d̂[inds], label="expected (analytical)", linestyle=:dash, markerstrokecolor=:steelblue,markercolor=:white)
    plot!(p1, dˣ[inds], d̂[inds] .+  2* σ_d[inds], linestyle=:dash,color=:steelblue,label="acceptance domain (analytical)")
    plot!(p1, dˣ[inds], d̂[inds] .-  2* σ_d[inds], linestyle=:dash,color=:steelblue,label="")
    # - z-scores (NOTE: manual log computation to be able to deal with zero values)
    p2 = scatter(log10.(abs.(z_d)), label="analytical", xlabel="node ID", ylabel="|degree z-score|", color=:steelblue,legend=:bottom)
    for ns in [100;1000;10000]
        μ̂ = reshape(mean( S_d[:,1:ns], dims=2), :)
        ŝ = reshape(std(  S_d[:,1:ns], dims=2), :)
        z_s = (dˣ - μ̂) ./ ŝ
        scatter!(p1, dˣ[inds], μ̂[inds], label="sampled (n = $(ns))", color=Int(log10(ns)), marker=:cross)
        plot!(p1, dˣ[inds], μ̂[inds] .+ 2*ŝ[inds], linestyle=:dash, color=Int(log10(ns)),  label="acceptance domain, sampled (n=$(ns))")
        plot!(p1, dˣ[inds], μ̂[inds] .- 2*ŝ[inds], linestyle=:dash, color=Int(log10(ns)), label="")
        scatter!(p2, log10.(abs.(z_s)), label="sampled (n = $(ns))", color=Int(log10(ns)))
    end
    plot!(p2, [1; length(M)], log10.([3;3]), color=:red, label="statistical significance threshold", linestyle=:dash)
    yrange = -12:2:0
    yticks!(p2,yrange, ["1e$(x)" for x in yrange])
    plot(p1,p2,layout=(1,2), size=(1200,600), title="Zachary karate club")
    savefig("""ZKC_degree_$("$(round(now(), Day))"[1:10]).pdf""")

    ## ANND metric testing
    # Squartini method
    ANNDˣ = ANND(mygraph)                                   # observed ANND sequence ("ANND star")
    ANND_hat = ANND(M)                                      # expected ANND sequence
    σ_ANND = map(n -> σˣ(G -> ANND(G, n), M), 1:length(M))  # standard deviation for each ANND in the sequence
    z_ANND = (ANNDˣ - ANND_hat) ./ σ_ANND
    # Sampling method
    S_ANND = hcat(map( g -> ANND(g), S)...);                # ANND sequences from the sample
    # illustration 
    # - acceptable domain
    p1 = scatter(dˣ[inds], ANNDˣ[inds], label="observed", color=:black, marker=:xcross, 
                xlabel="observed node degree", ylabel="ANND", legend_position=:topleft,
                bottom_margin=5mm, left_margin=5mm)
    scatter!(p1, dˣ[inds], ANND_hat[inds], label="expected (analytical)", linestyle=:dash, markerstrokecolor=:steelblue,markercolor=:white)
    plot!(p1, dˣ[inds], ANND_hat[inds] .+  2* σ_ANND[inds], linestyle=:dash,color=:steelblue,label="acceptance domain (analytical)")
    plot!(p1, dˣ[inds], ANND_hat[inds] .-  2* σ_ANND[inds], linestyle=:dash,color=:steelblue,label="")
    # - z-scores (NOTE: manual log computation to be able to deal with zero values)
    p2 = scatter(log10.(abs.(z_ANND)), label="analytical", xlabel="node ID", ylabel="|ANND z-score|", color=:steelblue,legend=:bottom)
    for ns in [100;1000;10000]
        μ̂ = reshape(mean( S_ANND[:,1:ns], dims=2), :)
        ŝ = reshape(std(  S_ANND[:,1:ns], dims=2), :)
        z_s = (ANNDˣ - μ̂) ./ ŝ
        scatter!(p1, dˣ[inds], μ̂[inds], label="sampled (n = $(ns))", color=Int(log10(ns)), marker=:cross)
        plot!(p1, dˣ[inds], μ̂[inds] .+ 2*ŝ[inds], linestyle=:dash, color=Int(log10(ns)),  label="acceptance domain, sampled (n=$(ns))")
        plot!(p1, dˣ[inds], μ̂[inds] .- 2*ŝ[inds], linestyle=:dash, color=Int(log10(ns)), label="")
        scatter!(p2, log10.(abs.(z_s)), label="sampled (n = $(ns))", color=Int(log10(ns)))
    end
    plot!(p2, [1; length(M)], log10.([3;3]), color=:red, label="statistical significance threshold", linestyle=:dash)
    yrange = -3:1:1
    ylims!(p2,-3,1)
    yticks!(p2,yrange, [L"$10^{ %$(x)}$" for x in yrange])
    plot(p1,p2,layout=(1,2), size=(1200,600), title="Zachary karate club")
    savefig("""ZKC_ANND_$("$(round(now(), Day))"[1:10]).pdf""")

    ## motifs testing
    # Squartini method
    motifs = [M₁; M₂]                  # different motifs that will be computed
    motifnames = Dict(M₁ => "M₁, v-motif", M₂ => "M₂, triangle")
    Mˣ  = map(f -> f(mygraph), motifs) # observed motifs sequence ("M star")
    M̂   = map(f -> f(M), motifs)       # expected motifs sequence
    σ_M = map(f -> σˣ(f, M), motifs)   # standard deviation for each motif in the sequence
    z_M̂ = (Mˣ - M̂) ./ σ_M
    # Sampling method
    Ns = [100;1000;10000]
    z_M = zeros(length(motifs), length(Ns))
    for i in eachindex(Ns)
        res = hcat(map(s -> map(f -> f(s), motifs),S[1:Ns[i]])...)
        μ̂ = mean(res, dims=2)
        ŝ = std(res, dims=2)
        z_M[:,i] = (Mˣ - μ̂) ./ ŝ
    end
    # illustration
    plot(Ns, abs.(permutedims(repeat(z_M̂, 1,3))), color=[:steelblue :sienna], label=reshape(["$(motifnames[f]) (analytical)" for f in motifs],1,2), linestyle=[:dash :dot], 
        xlabel="Sample size", ylabel="|motif z-score|", xscale=:log10)
    plot!(Ns, abs.(permutedims(z_M)), color=[:steelblue :sienna], label=reshape(["$(motifnames[f]) (sampled)" for f in motifs],1,2), marker=:circle, linestyle=[:dash :dot])
    savefig("""ZKC_motifs_$("$(round(now(), Day))"[1:10]).pdf""")
end

function DBCM_test()
    NP = PyCall.pyimport("NEMtropy")
    mygraph = Graphs.erdos_renyi(183,2494, is_directed=true, seed=161)
    mygraph_nem = NP.DirectedGraph(degree_sequence=vcat(Graphs.outdegree(mygraph), Graphs.indegree(mygraph)))
    mygraph_nem.solve_tool(model="dcm", method="fixed-point", initial_guess="random");
    
    # quick checks functionality
    @assert outdegree(Graphs.adjacency_matrix(mygraph)) == Graphs.outdegree(mygraph)
    @assert indegree( Graphs.adjacency_matrix(mygraph)) == Graphs.indegree(mygraph)
    @assert mygraph_nem.dseq_out == Graphs.outdegree(mygraph)
    @assert mygraph_nem.dseq_in  == Graphs.indegree(mygraph)
    
    M = UBCM(mygraph_nem.x, mygraph_nem.y) # generate the model
    # quick checks convergence
    @assert Graphs.outdegree(mygraph) ≈ outdegree(M)
    @assert Graphs.indegree(mygraph)  ≈ indegree(M)
         
    S = [rand(M) for _ in 1:10000]; # sample 10000 graphs from the model
    
    ## degree metric testing
    dˣ_in, dˣ_out = Graphs.indegree(mygraph), Graphs.outdegree(mygraph) # observed degree sequence ("d star")
    inds_in, inds_out = sortperm(dˣ_in), sortperm(dˣ_out)
    # Squartini method
    d̂_in, d̂_out  = indegree(M), outdegree(M)                            # expected degree sequence from the model
    σ_d_in =  map(n -> σˣ(G -> indegree(G, n), M),  1:length(M))        # standard deviation for each degree in the sequence
    σ_d_out = map(n -> σˣ(G -> outdegree(G, n), M), 1:length(M))      
    z_d_in, z_d_out = (dˣ_in - d̂_in) ./ σ_d_in, (dˣ_out - d̂_out) ./ σ_d_out
    # Sampling method
    S_d_in, S_d_out = hcat(map(g -> Graphs.indegree(g), S)...), hcat(map(g -> Graphs.outdegree(g), S)...);       # degree sequences from the sample
    
    # illustration
    # - acceptable domain
    # --indegree
    p1 = scatter(dˣ_in[inds_in], dˣ_in[inds_in], label="observed", color=:black, marker=:xcross, 
                    xlabel="observed node indegree", ylabel="node indegree", legend_position=:topleft,
                    bottom_margin=5mm, left_margin=5mm)
    scatter!(p1, dˣ_in[inds_in], d̂_in[inds_in], label="expected (analytical)", linestyle=:dash, markerstrokecolor=:steelblue,markercolor=:white)                
    plot!(p1, dˣ_in[inds_in], d̂_in[inds_in] .+  2* σ_d_in[inds_in], linestyle=:dash,color=:steelblue,label="acceptance domain (analytical)")
    plot!(p1, dˣ_in[inds_in], d̂_in[inds_in] .-  2* σ_d_in[inds_in], linestyle=:dash,color=:steelblue,label="")
    # --outdegree
    p3 = scatter(dˣ_out[inds_out], dˣ_out[inds_out], label="observed", color=:black, marker=:xcross, 
                    xlabel="observed node outdegree", ylabel="node outdegree", legend_position=:topleft,
                    bottom_margin=5mm, left_margin=5mm)
    scatter!(p3, dˣ_out[inds_out], d̂_out[inds_out], label="expected (analytical)", linestyle=:dash, markerstrokecolor=:steelblue,markercolor=:white)                
    plot!(p3, dˣ_out[inds_out], d̂_out[inds_out] .+  2* σ_d_out[inds_out], linestyle=:dash,color=:steelblue,label="acceptance domain (analytical)")
    plot!(p3, dˣ_out[inds_out], d̂_out[inds_out] .-  2* σ_d_out[inds_out], linestyle=:dash,color=:steelblue,label="")
    # - z-scores (NOTE: manual log computation to be able to deal with zero values)
    p2 = scatter(log10.(abs.(z_d_in)),  label="analytical", xlabel="node ID", ylabel="|indegree z-score|",  color=:steelblue,legend=:bottom)
    p4 = scatter(log10.(abs.(z_d_out)), label="analytical", xlabel="node ID", ylabel="|outdegree z-score|", color=:steelblue,legend=:bottom)
    for ns in [100;1000;10000]
        # --indegree
        μ̂_in = reshape(mean( S_d_in[:,1:ns], dims=2), :)
        ŝ_in = reshape(std(  S_d_in[:,1:ns], dims=2), :)
        z_s_in = (dˣ_in - μ̂_in) ./ ŝ_in
        scatter!(p1, dˣ_in[inds_in], μ̂_in[inds_in], label="sampled (n = $(ns))", color=Int(log10(ns)), marker=:cross)
        plot!(p1, dˣ_in[inds_in], μ̂_in[inds_in] .+ 2*ŝ_in[inds_in], linestyle=:dash, color=Int(log10(ns)),  label="acceptance domain, sampled (n=$(ns))")
        plot!(p1, dˣ_in[inds_in], μ̂_in[inds_in] .- 2*ŝ_in[inds_in], linestyle=:dash, color=Int(log10(ns)), label="")
        scatter!(p2, log10.(abs.(z_s_in)), label="sampled (n = $(ns))", color=Int(log10(ns)))
        # --outdegree
        μ̂_out = reshape(mean( S_d_out[:,1:ns], dims=2), :)
        ŝ_out = reshape(std(  S_d_out[:,1:ns], dims=2), :)
        z_s_out = (dˣ_out - μ̂_out) ./ ŝ_out
        scatter!(p3, dˣ_out[inds_out], μ̂_out[inds_out], label="sampled (n = $(ns))", color=Int(log10(ns)), marker=:cross)
        plot!(p3, dˣ_out[inds_out], μ̂_out[inds_out] .+ 2*ŝ_out[inds_out], linestyle=:dash, color=Int(log10(ns)),  label="acceptance domain, sampled (n=$(ns))")
        plot!(p3, dˣ_out[inds_out], μ̂_out[inds_out] .- 2*ŝ_out[inds_out], linestyle=:dash, color=Int(log10(ns)), label="")
        scatter!(p4, log10.(abs.(z_s_out)), label="sampled (n = $(ns))", color=Int(log10(ns)))
    end
    yrange = -10:2:0
    ylims!(p2,-10,0); ylims!(p4,-10,0)
    yticks!(p2,yrange, ["1e$(x)" for x in yrange])
    yticks!(p4,yrange, ["1e$(x)" for x in yrange])
    p_in  = plot(p1,p2,layout=(1,2), size=(1200,600), title="random Erdos-Renyi graph")
    p_out = plot(p3,p4,layout=(1,2), size=(1200,600), title="random Erdos-Renyi graph")
    savefig(p_in,  """ER_indegree_$("$(round(now(), Day))"[1:10]).pdf""")
    savefig(p_out, """ER_outdegree_$("$(round(now(), Day))"[1:10]).pdf""")
    
    ## motif testing
    # Squartini method
    mˣ = motifs(mygraph)
    m̂  = motifs(M)
    mfuns = [Symbol('M' * prod(map(x -> Char(x+48+8272),map(v -> reverse(digits(v)), i)))) for i = 1:13]
    σ_m = Dict{Int, Float64}()
    for i = 1:13
        @info "working on $(i)"
        σ_m[i] = σˣ(eval(mfuns[i]), M)
    end
    z_m = [(mˣ[i] - m̂[i]) ./ σ_m[i] for i = 1:13]
    # simulation method
    S_m = motifs.(S, full=true);
    S_M = hcat(S_m...)
    
    # illustration
    p_mot = scatter(z_m, xlabel="Motif ID", ylabel="z-score", color=:steelblue, label="analytical", 
                    size=(1200,600), bottom_margin=5mm, left_margin=5mm, legend=:bottomleft, legendfontsize=6)
    xticks!(collect(1:13))
    for ns in [100;1000;10000]
        m̂_sim = mean(S_M[:,1:ns], dims=2)
        σ̂_sim = std( S_M[:,1:ns], dims=2) 
        z_sim = (mˣ - m̂_sim) ./ σ̂_sim
        scatter!(p_mot, z_sim, color=Int(log10(ns)), label="sampled (n = $(ns))", markeralpha=0.5)
    end
    plot!(p_mot, collect(1:13), -2*ones(13), color=:red, label="statistical significance threshold", linestyle=:dash)
    plot!(p_mot, collect(1:13), 2*ones(13), color=:red, label="", linestyle=:dash)
    
    title!("random Erdos-Renyi graph")
    xlims!(0,14)
    savefig(p_mot, """ER_motifs_$("$(round(now(), Day))"[1:10]).pdf""")
end


#ylims!(-10,0)

#σˣ($(f), $(M)
#=

"""
∇X(X::Function, G::Matrix{T})

Compute the gradient of a metric X given an adjacency/weight matrix G
"""
∇X(::Function, ::Matrix)

function ∇X(X::Function, G::Matrix{T}) where T<:Real
    return ReverseDiff.gradient(X, G)
end

function ∇X(X::Function, m::UBCM{T}) where T<:Real
    G = Ĝ(m)
    return ReverseDiff.gradient(X, G)
end

"""
σₓ(::Function)

Compute variance of a metric X given the expected adjacency/weight matrix
"""
σₓ(::Function, ::Matrix)

function σX(X::Function, Ĝ::Matrix{T}, σ̂::Matrix{T}) where T<:Real
    return sqrt( sum((σ̂ .* ∇X(X, Ĝ)) .^ 2) )
end




## some metrics 
d(A,i) = sum(@view A[:,i]) # degree node i
ANND(G::Graphs.SimpleGraph,i) where T = iszero(Graphs.degree(G,i)) ? zero(Float64) : sum(map( n -> Graphs.degree(G,n), Graphs.neighbors(G,i))) / Graphs.degree(G,i)
ANND(G::Graphs.SimpleGraph) where T = map( i -> ANND(G,i), 1:Graphs.nv(G))
ANND(A,i) =  sum(A[i,j] * d(A,j) for j=1:size(A,1) if j≠i) / d(A,i)






# sampling the model
S = [rand(M) for _ in 1:10000];
dS = hcat(map( g -> Graphs.degree(g), S)...);
d_sample_mean_estimate(V,n) = mean(V[:,1:n], dims=2);
d_sample_var_estimate(V,n) =  std(V[:,1:n], dims=2);

# illustration for degree - acceptable domain
p1 = scatter(d_star[inds], d_star[inds], label="observed", legend=:topleft)
plot!(d_star[inds], map(n -> d(G_exp,n), 1:34)[inds], label="expected", linestyle=:dash, color=:steelblue)
plot!(d_star[inds], map(n -> d(G_exp,n), 1:34)[inds] .+  2* map(n-> σX(A->d(A,n), G_exp, σ_exp), 1:34)[inds],linestyle=:dash,color=:black,label="acceptance domain (analytical)")
plot!(d_star[inds], map(n -> d(G_exp,n), 1:34)[inds] .-  2* map(n-> σX(A->d(A,n), G_exp, σ_exp), 1:34)[inds],linestyle=:dash,color=:black,label="")
for ns in [10;100;1000;10000]
    plot!(d_star[inds], d_sample_mean_estimate(dS,ns)[inds] .+ 2* d_sample_var_estimate(dS,ns)[inds],linestyle=:dot,label="acceptance domain, sample (n=$(ns))")
    plot!(d_star[inds], d_sample_mean_estimate(dS,ns)[inds] .- 2* d_sample_var_estimate(dS,ns)[inds],linestyle=:dot,label="")
end
xlabel!("observed degree")
ylabel!("expected degree")
title!("UBCM - Zachary karate club")

# illustration for degree - z-scores per node
p2 = scatter(abs.((d_star - map(n -> d(G_exp,n), 1:34)) ./ map(n-> σX(A->d(A,n), G_exp, σ_exp), 1:34)), label="analytical", yscale=:log10)
for ns in [10;100;1000;10000]
    z = abs.((d_star - d_sample_mean_estimate(dS,ns)) ./ d_sample_var_estimate(dS,ns))
    lab = @sprintf "sample (n= %1.0e)" ns
    scatter!(z[z.>0], label=lab, yscale=:log10)
end
plot!([1; Graphs.nv(mygraph)], [3;3], linestyle=:dash,color=:red,label="")
xlabel!("node id")
ylabel!("|z-score degree|")
title!("UBCM - Zachary karate club")
ylims!(1e-10,1)

plot!(p1,p2, size=(1200,600), bottom_margin=5mm)
savefig("ZKC - degree.pdf")
nothing



##
## ANND part ##
##
X_star = ANND(mygraph)  # observed value (real network)
X_exp  = map(n->ANND(G_exp,n), 1:Graphs.nv(mygraph))    # expected value (analytical)
X_std  = map(n-> σX(A->ANND(A,n), G_exp, σ_exp), 1:34)  # standard deviation (analytical)
z_anal = (X_star - X_exp) ./ X_std

xS = hcat(map( g -> ANND(g), S)...);  #segmentation fault!!! => solved, tpye unstable


AP1 = scatter(d_star[inds], X_star[inds], label="observed", legend=:topright)
plot!(d_star[inds], X_exp[inds], label="expected", linestyle=:dash, color=:steelblue)
plot!(d_star[inds], X_exp[inds] .+ 2*X_std[inds], linestyle=:dash,color=:black,label="acceptance domain (analytical)")
plot!(d_star[inds], X_exp[inds] .- 2*X_std[inds], linestyle=:dash,color=:black,label="")
for ns in [10;100;1000;10000]
    plot!(d_star[inds], d_sample_mean_estimate(xS,ns)[inds] .+ 2* d_sample_var_estimate(xS,ns)[inds],linestyle=:dot,label="acceptance domain, sample (n=$(ns))")
    plot!(d_star[inds], d_sample_mean_estimate(xS,ns)[inds] .- 2* d_sample_var_estimate(xS,ns)[inds],linestyle=:dot,label="")
end
xlabel!("observed degree")
ylabel!("ANND")
title!("UBCM - Zachary karate club")

# illustration for degree - z-scores per node
AP2 = scatter(abs.(z_anal), label="analytical", yscale=:log10)
for ns in [10;100;1000;10000]
    z = abs.((X_star - d_sample_mean_estimate(xS,ns)) ./ d_sample_var_estimate(xS,ns))
    lab = @sprintf "sample (n= %1.0e)" ns
    scatter!(z[z.>0], label=lab, yscale=:log10)
end
plot!([1; Graphs.nv(mygraph)], [3;3], linestyle=:dash,color=:red,label="", legend=:bottomleft)
xlabel!("node id")
ylabel!("|z-score ANND|")
title!("UBCM - Zachary karate club")
ylims!(1e-3,1e1)

plot(AP1,AP2, size=(1200,600), bottom_margin=5mm)
savefig("ZKC - ANND.pdf")






N₂_obs = M₂(Graphs.adjacency_matrix(mygraph))
N₂_exp = M₂(G_exp)
N₂_σ   = σX(M₂, G_exp, σ_exp) # standard deviation (analytical)
z_N₂ = (N₂_obs- N₂_exp) / N₂_σ
# from the sample
N₂_sampled = map(g -> M₂(Graphs.adjacency_matrix(g)), S)
# computed values:
N₂_hat = [mean(N₂_sampled[1:n]) for n ∈ [10;100;1000;10000]]
N₂_std_hat = [std(N₂_sampled[1:n]) for n ∈ [10;100;1000;10000]]
N₂_z_hat = (N₂_obs .- N₂_hat) ./ N₂_std_hat

scatter([10;100;1000;10000],abs.(N₂_z_hat), xscale=:log10, label="sampled")
plot!([10;100;1000;10000], abs.(z_N₂ .* ones(4)), linestyle=:dash, label="analytical")
ylims!(0,2)
xticks!([10;100;1000;10000])
xlabel!("sample size")
ylabel!("|z-score triangle count|")
title!("UBCM - Zachary karate club")
Plots.savefig("ZKC - Triangles.pdf")

N₁_obs = M₁(Graphs.adjacency_matrix(mygraph))
N₁_exp = M₁(G_exp)
N₁_σ   = σX(M₁, G_exp, σ_exp) # standard deviation (analytical)
z_N₁ = (N₁_obs- N₁_exp) / N₁_σ
# from the sample
N₁_sampled = map(g -> M₁(Graphs.adjacency_matrix(g)), S)
# computed values:
N₁_hat = [mean(N₁_sampled[1:n]) for n ∈ [10;100;1000;10000]]
N₁_std_hat = [std(N₁_sampled[1:n]) for n ∈ [10;100;1000;10000]]
N₁_z_hat = (N₁_obs .- N₁_hat) ./ N₁_std_hat

scatter([10;100;1000;10000],abs.(N₁_z_hat), xscale=:log10, label="sampled")
plot!([10;100;1000;10000], abs.(z_N₁ .* ones(4)), linestyle=:dash, label="analytical")
ylims!(0,2)
xticks!([10;100;1000;10000])
xlabel!("sample size")
ylabel!("|z-score v-motif count|")
title!("UBCM - Zachary karate club")
savefig("ZKC - v-motifs.pdf")


# scaffolding for the directed network motif elements
"""
    a⭢(A::Matrix{T}, i, j)

directed link from i to j and not from j to i
"""
a⭢(A::Matrix{T}, i, j) where T = A[i,j] * (one(T) - A[j,i]) 

"""
    a⭠(A::Matrix{T}, i, j)

directed link from j to i and not from i to j
"""
a⭠(A::Matrix{T}, i, j) where T = (one(T) - A[i,j]) * A[j,i]

"""
    a⭤(A::Matrix{T}, i, j)

recipocrated link between i and j
"""
a⭤(A,i,j) = A[i,j]*A[j,i]

"""
    a_̸(A::Matrix{T}, i, j)

no link between i and j
"""
a_̸(A::Matrix{T},i,j) where T = (one(T) - A[j,i]) * (one(T) - A[i,j])

# motifs itself

M_13(A) = sum(a⭤(A,i,j)*a⭤(A,j,k)*a⭤(A,k,i) for i = 1:size(A,1) for j=1:size(A,1) for k=1:size(A,1) if i≠j && j≠k && i≠k)

=#


# helper function to load up a network from a file
load_network_from_edges(f::String) = Graphs.loadgraph(f, EdgeListFormat())


"""
    run a complete analysis (with plots and storage (?))

"""
function DBCM_analysis(G::Graphs.SimpleDiGraph, name::String; 
                        N::Int=10000, subsamples::Vector{Int64}=[100;1000;10000],
                        nodefunctions= [(indegree, Graphs.indegree, Graphs.indegree); 
                                        (outdegree, Graphs.outdegree, Graphs.outdegree);
                                        (ANND_in, ANND_in, Graphs.indegree);
                                        (ANND_out, ANND_out, Graphs.outdegree)])
    NP = PyCall.pyimport("NEMtropy")
    
    # for plotting
    sample_labels = permutedims(map(l -> "sampled (n = $(l))", subsamples))
    sample_colors = permutedims(Int.(log10.(round.(Int,subsamples))))

    @info "$(round(now(), Minute)) - Starting analysis for $(name)"
    @info "$(round(now(), Minute)) - Computing ML parameters for $(name)"
    G_nem =  NP.DirectedGraph(degree_sequence=vcat(Graphs.outdegree(G), Graphs.indegree(G)))
    G_nem.solve_tool(model="dcm", method="fixed-point", initial_guess="random")
    model = DBCM(G_nem.x, G_nem.y)
    @info "$(round(now(), Minute)) - Generating sample (n = $(N)) for $(name)"
    S = [rand(model) for _ in 1:N]

    ### Different METRICS ###
    ## Node metrics ##
    #=
    @info "$(round(now(), Minute)) - Computing node metrics for $(name)"
    for (f, f_graphs, f_ref) in nodefunctions
        @info "$(round(now(), Minute)) - Computing $(f) for $(name)"
        Xˣ, X̂, σ̂_X̂, X_S = node_metric(f, G, model, S; graph_fun=f_graphs)
        ref_val = f_ref(G)
        inds = sortperm(ref_val) # sorted indices based on oberved value (not based on degree)
        # compute z-score (Squartini)
        z_X_a = (Xˣ .- X̂) ./ σ̂_X̂
        # compute z-score (Simulation)
        X̂_S = hcat(map(n -> reshape(mean(X_S[:,1:n], dims=2),:), subsamples)...)
        σ̂_X̂_S = hcat(map(n -> reshape( std(X_S[:,1:n], dims=2),:), subsamples)...)
        z_X_S = (Xˣ .- X̂_S) ./ σ̂_X̂_S

        # illustration
        p1 = scatter(ref_val[inds], Xˣ[inds], label="observed", color=:red, marker=:circle, 
                     xlabel="observed node $(f_ref)", ylabel="node $(f)", legend_position=:topleft,
                     bottom_margin=5mm, left_margin=5mm)
        scatter!(p1, ref_val[inds], X̂[inds], label="expected (analytical - DBCM)", markerstrokecolor=:steelblue, markercolor=:white)  
        plot!(p1,ref_val[inds],X̂[inds] .+ 2* σ̂_X̂[inds], label="expected ± 2σ (analytical - DBCM)", linestyle=:dash, color=:steelblue)              
        plot!(p1,ref_val[inds],X̂[inds] .- 2* σ̂_X̂[inds], label="", linestyle=:dash, color=:steelblue)
        
        # maybe re-evaluate this
        p2 = scatter(log10.(abs.(z_X_a)),  label="analytical - BDCM)", xlabel="node ID", ylabel="|$(f) z-score|",
                     color=:steelblue,legend=:bottom)
        scatter!(p2, log10.(abs.(z_X_S)), label=sample_labels, color=sample_colors)
        plot!(p2, )
        p = plot(p1,p2,layout=(1,2), size=(1200,600))
        savefig(p, """$(name)_$("$(round(now(), Day))"[1:10])_$(f)_z-score.pdf""")
    end
=#
    ## Graph metrics (e.g. motifs) ##
    @info "$(round(now(), Minute)) - Computing graph metrics for $(name)"
    @info "$(round(now(), Minute)) - Computing triadic motifs for $(name)"
    mˣ = motifs(G)
    m̂  = motifs(model)
    σ̂_m̂  = Vector{eltype(model.G)}(undef, length(m̂))
    for i = 1:length(m̂)
        @info "$(round(now(), Minute)) - Computing standard deviation for motif $(i)"
        σ̂_m̂[i] = σˣ(DBCM_motif_functions[i], model)
    end
    # compute z-score (Squartini)
    z_m_a = [(mˣ[i] - m̂[i]) ./ σ̂_m̂[i] for i = 1:length(m̂)]
    # compute z-score (Simulation)
    @info "$(round(now(), Minute)) - Computing motifs for the sample"
    S_m = hcat(motifs.(S,full=true)...); # computed values from sample
    m̂_S =   hcat(map(n -> reshape(mean(S_m[:,1:n], dims=2),:), subsamples)...)
    σ̂_m̂_S = hcat(map(n -> reshape( std(S_m[:,1:n], dims=2),:), subsamples)...)
    z_m_S = (mˣ .- m̂_S) ./ σ̂_m̂_S
    #=
    # illustration
    p_mot = scatter(z_m_a, xlabel="Motif ID", ylabel="z-score", color=:steelblue, label="analytical (BDCM)", 
                    size=(1200,600), bottom_margin=5mm, left_margin=5mm, legend=:topleft, legendfontsize=6, xticks=collect(1:13))


    scatter!(p_mot, z_m_S, label=sample_labels, color=sample_colors, markeralpha=0.5)
    plot!(p_mot, collect(1:13), -2*ones(13), color=:red, label="statistical significance threshold", linestyle=:dash)
    plot!(p_mot, collect(1:13), 2*ones(13), color=:red, label="", linestyle=:dash)
    title!("$(name)")
    xlims!(0,14)
    savefig(p_mot, """$(name)_$("$(round(now(), Day))"[1:10])_motifs_z-score.pdf""")
    =#

    return hcat(z_m_a, z_m_S), mˣ, m̂_S, σ̂_m̂_S
end
   

"""
node_metric(X::Function, G::Graphs.SimpleDiGraph, M::DBCM, S::Vector{T}; graph_fun::Function=X) where T<: Graphs.SimpleDiGraph

For the metric `X`, compute the observed value from the graph `G`, the expected value under the DBCM `M` and the values for the 
elements of the sample `S`. The function `graph_fun` is used to indicate the function that will be used to compute the metric `X`
on the `::Graphs.SimpleDiGraph` object (some metrics such as degree are available in the `Graphs` package and are prefered for speed). 
The default is to use the function `X` itself.

By default, this function assumes that the metric `X` needs to be computed for each node in the graph.

The return tuple is (Xˣ, X̂, σ̂_X̂, X_S) where
- `Xˣ`: the observed value of the metric `X`
- `X̂`: the expected value of the metrix `X` under the null model
- `σ̂_X̂`: the standard deviation of the metric `X` under the null model
- `X_S`: the values of the metric `X` for the elements of the sample

Can be used in combination with: (indegree, Graphs.indegree), (outdegree, Graphs.outdegree), ANND

"""
function node_metric(X::Function, G::Graphs.SimpleDiGraph, M::DBCM, S::Vector{T}; graph_fun::Function=X) where T<: Graphs.SimpleDiGraph
    Xˣ =  graph_fun(G)                        # compute observed value
    X̂  =  X(M)                                # compute value for the DBCM
    σ̂_X̂  =  map(n -> σˣ(g -> indegree(g, n), M),  1:length(M)) # compute variance for the DBCM
    X_S = hcat(map(g -> graph_fun(g), S)...) # compute value for the sample

    return Xˣ, X̂, σ̂_X̂, X_S
end

#=
G_nem =  NP.DirectedGraph(degree_sequence=vcat(Graphs.outdegree(G), Graphs.indegree(G)))
    G_nem.solve_tool(model="dcm", method="fixed-point", initial_guess="random")
    model = UBCM(mygraph_nem.x, mygraph_nem.y)

    model = UBCM(mygraph_nem.x, mygraph_nem.y)


LRFW_nem = NP.DirectedGraph(degree_sequence=vcat(Graphs.outdegree(LRFW), Graphs.indegree(LRFW)))
LRFW_nem.solve_tool(model="dcm", method="fixed-point", initial_guess="random")
LRFW_M = DBCM(LRFW_nem.x, LRFW_nem.y)
LRFB_sample = [rand(LRFW_M) for i = 1:100]
res = computevals(ANND_in, LRFB, LRFB_M, LRFB_sample)
inds = sortperm(Graphs.indegree(LRFB))


LRFW = load_network_from_edges("./data/foodweb_little_rock/edges.csv")
LRFW_nem =  NP.DirectedGraph(degree_sequence=vcat(Graphs.outdegree(LRFW), Graphs.indegree(LRFW)))
LRFW_nem.solve_tool(model="dcm", method="fixed-point", initial_guess="random")
LRFW_model = DBCM(mygraph_nem.x, mygraph_nem.y)
LRFW_sample = [rand(LRFW_model) for i = 1:100]
res = computevals(ANND_in, LRFW, LRFW_model, LRFW_sample)

inds = sortperm(Graphs.indegree(LRFW))
x = Graphs.indegree(LRFW)
scatter(x[inds], res[1][inds], label="analytical", color=:red)
plot!(x[inds], res[2][inds], color=:black)
plot!(x[inds], res[2][inds] .+ 2*res[3][inds], color=:blue,line=:dash)
plot!(x[inds], res[2][inds] .- 2*res[3][inds], color=:blue,line=:dash)

sss = reshape(std(res[4], dims=2),:)
#(res[1] .- reshape(mean(res[4], dims=2),:)) ./ reshape(std(res[4], dims=2),:)

plot!(x[inds], res[2][inds] .+ 2*sss[inds], label="sample", color=:green)
xticks!(0:20:100)

=#

#mijnetwerk = Graphs.SimpleDiGraph( Graphs.smallgraph(:karate))
#DBCM_analysis(mijnetwerk, "karate")
#Gbis = Graphs.erdos_renyi(70, 2500, is_directed=true)
#DBCM_analysis(Gbis, "randomerdosrenyi")

"""
    DBCM_analysis

Compute the z-scores etc. for all motifs and the degrees for a `SimpleDiGraph`. Returns a Dict for storage of the computed results

G: the network
N_min: minimum sample length used for computing metrics
N_max: maximum sample length used for computing metrics
n_sample_lengths: number of values in the domain [N_min, N_max]
"""
function DBCM_analysis(  G::T;
                                N_min::Int=100, 
                                N_max::Int=10000, 
                                n_sample_lengths::Int=3,
                                subsamples::Vector{Int64}=round.(Int,exp10.(range(log10(N_min),log10(N_max), length=n_sample_lengths))), kwargs...) where T<:Graphs.SimpleDiGraph
    @info "$(round(now(), Minute)) - Started DBCM motif analysis with the following setting:\n$(kwargs)"
    NP = PyCall.pyimport("NEMtropy")
    G_nem =  NP.DirectedGraph(degree_sequence=vcat(Graphs.outdegree(G), Graphs.indegree(G)))
    G_nem.solve_tool(model="dcm_exp", method="fixed-point", initial_guess="degrees", max_steps=3000)
    if abs(G_nem.error) < 1e-6
        @warn "Method did not converge"
    end
    # generate the model
    model = DBCM(G_nem.x, G_nem.y)
    @assert indegree(model)  ≈ Graphs.indegree(G)
    @assert outdegree(model) ≈ Graphs.outdegree(G)
    # generate the sample
    @info "$(round(now(), Minute)) - Generating sample"
    S = [rand(model) for _ in 1:N_max]

    #################################
    # motif part
    #################################

    # compute motif data
    @info "$(round(now(), Minute)) - Computing observed motifs in the observed network"
    mˣ = motifs(G)
    @info "$(round(now(), Minute)) - Computing expected motifs for the DBCM model"
    m̂  = motifs(model)
    σ̂_m̂  = Vector{eltype(model.G)}(undef, length(m̂))
    for i = 1:length(m̂)
        @info "$(round(now(), Minute)) - Computing standard deviation for motif $(i)"
        σ̂_m̂[i] = σˣ(DBCM_motif_functions[i], model)
    end
    # compute z-score (Squartini)
    z_m_a = (mˣ - m̂) ./ σ̂_m̂  
    @info "$(round(now(), Minute)) - Computing expected motifs for the sample"
    #S_m = hcat(motifs.(S,full=true)...); # computed values from sample
    S_m = zeros(13, length(S))
    Threads.@threads for i in eachindex(S)
        S_m[:,i] .= motifs(S[i], full=true)
    end
    m̂_S =   hcat(map(n -> reshape(mean(S_m[:,1:n], dims=2),:), subsamples)...)
    σ̂_m̂_S = hcat(map(n -> reshape( std(S_m[:,1:n], dims=2),:), subsamples)...)
    z_m_S = (mˣ .- m̂_S) ./ σ̂_m̂_S

    #################################       
    # degree part
    #################################
    # compute degree sequence
    @info "$(round(now(), Minute)) - Computing degrees in the observed network"
    d_inˣ, d_outˣ = Graphs.indegree(G), Graphs.outdegree(G)
    @info "$(round(now(), Minute)) - Computing expected degrees for the DBCM model"
    d̂_in, d̂_out = indegree(model), outdegree(model)
    @info "$(round(now(), Minute)) - Computing standard deviations for the degrees for the DBCM model"
    σ̂_d̂_in, σ̂_d̂_out = map(j -> σˣ(m -> indegree(m, j), model), 1:length(model)), map(j -> σˣ(m -> outdegree(m, j), model), 1:length(model))
    # compute degree z-score (Squartini)
    z_d_in_sq, z_d_out_sq = (d_inˣ - d̂_in) ./ σ̂_d̂_in, (d_outˣ - d̂_out) ./ σ̂_d̂_out
    @info "$(round(now(), Minute)) - Computing distributions for degree sequences"
    d_in_dist, d_out_dist = indegree_dist(model), outdegree(model)
    z_d_in_dist, z_d_out_dist = (d_inˣ - mean.(d_in_dist)) ./ std.(d_in_dist), (d_outˣ - mean.(d_out_dist)) ./ std.(d_out_dist)

    # compute data for the sample
    @info "$(round(now(), Minute)) - Computing degree sequences for the sample"
    d_in_S, d_out_S = hcat(Graphs.indegree.(S)...), hcat(Graphs.outdegree.(S)...)
    d̂_in_S, d̂_out_S = hcat(map(n -> reshape(mean(d_in_S[:,1:n], dims=2),:), subsamples)...), hcat(map(n -> reshape(mean(d_out_S[:,1:n], dims=2),:), subsamples)...)
    σ̂_d_in_S, σ̂_d_out_S = hcat(map(n -> reshape(std( d_in_S[:,1:n], dims=2),:), subsamples)...), hcat(map(n -> reshape( std(d_out_S[:,1:n], dims=2),:), subsamples)...)
    # compute degree z-score (sample)
    z_d_in_S, z_d_out_S = (d_inˣ .- d̂_in_S) ./ σ̂_d_in_S, (d_outˣ .- d̂_out_S) ./ σ̂_d_out_S
    

    @info "$(round(now(), Minute)) - Finished"
    return Dict(:network => G,
                :model => model,
                :error => G_nem.error,
                # motif information
                :mˣ => mˣ,          # observed
                :m̂ => m̂,            # expected squartini
                :σ̂_m̂ => σ̂_m̂,        # standard deviation squartini
                :z_m_a => z_m_a,    # z_motif squartini
                :S_m => S_m,        # sample data
                :m̂_S => m̂_S,        # expected sample
                :σ̂_m̂_S => σ̂_m̂_S,    # standard deviation sample
                :z_m_S => z_m_S,    # z_motif sample
                # in/outdegree information
                :d_inˣ => d_inˣ,                # observed
                :d_outˣ => d_outˣ,              # observed
                :d̂_in => d̂_in,                  # expected squartini
                :d̂_out => d̂_out,                # expected squartini
                :σ̂_d̂_in => σ̂_d̂_in,              # standard deviation squartini
                :σ̂_d̂_out => σ̂_d̂_out,            # standard deviation squartini
                :z_d_in_sq => z_d_in_sq,        # z_degree squartini
                :z_d_out_sq => z_d_out_sq,      # z_degree squartini
                :d̂_in_S => d̂_in_S,              # expected sample
                :d̂_out_S => d̂_out_S,            # expected sample
                :σ̂_d_in_S => σ̂_d_in_S,          # standard deviation sample
                :σ̂_d_out_S => σ̂_d_out_S,        # standard deviation sample
                :z_d_in_S => z_d_in_S,          # z_degree sample
                :z_d_out_S => z_d_out_S,        # z_degree sample
                :d_in_dist => d_in_dist,        # distribution (analytical PoissonBinomial)
                :d_out_dist => d_out_dist,      # distribution (analytical PoissonBinomial)
                :z_d_in_dist => z_d_in_dist,    # z_degree distribution (analytical PoissonBinomial)
                :z_d_out_dist => z_d_out_dist,  # z_degree distribution (analytical PoissonBinomial)
                :d_in_S => d_in_S,              # indegree sample
                :d_out_S => d_out_S             # indegree sample
                ) 
                
end




### Different METRICS ###
## Node metrics ##
#=
@info "$(round(now(), Minute)) - Computing node metrics for $(name)"
for (f, f_graphs, f_ref) in nodefunctions
@info "$(round(now(), Minute)) - Computing $(f) for $(name)"
Xˣ, X̂, σ̂_X̂, X_S = node_metric(f, G, model, S; graph_fun=f_graphs)
ref_val = f_ref(G)
inds = sortperm(ref_val) # sorted indices based on oberved value (not based on degree)
# compute z-score (Squartini)
z_X_a = (Xˣ .- X̂) ./ σ̂_X̂
# compute z-score (Simulation)
X̂_S = hcat(map(n -> reshape(mean(X_S[:,1:n], dims=2),:), subsamples)...)
σ̂_X̂_S = hcat(map(n -> reshape( std(X_S[:,1:n], dims=2),:), subsamples)...)
z_X_S = (Xˣ .- X̂_S) ./ σ̂_X̂_S

# illustration
p1 = scatter(ref_val[inds], Xˣ[inds], label="observed", color=:red, marker=:circle, 
 xlabel="observed node $(f_ref)", ylabel="node $(f)", legend_position=:topleft,
 bottom_margin=5mm, left_margin=5mm)
scatter!(p1, ref_val[inds], X̂[inds], label="expected (analytical - DBCM)", markerstrokecolor=:steelblue, markercolor=:white)  
plot!(p1,ref_val[inds],X̂[inds] .+ 2* σ̂_X̂[inds], label="expected ± 2σ (analytical - DBCM)", linestyle=:dash, color=:steelblue)              
plot!(p1,ref_val[inds],X̂[inds] .- 2* σ̂_X̂[inds], label="", linestyle=:dash, color=:steelblue)

# maybe re-evaluate this
p2 = scatter(log10.(abs.(z_X_a)),  label="analytical - BDCM)", xlabel="node ID", ylabel="|$(f) z-score|",
 color=:steelblue,legend=:bottom)
scatter!(p2, log10.(abs.(z_X_S)), label=sample_labels, color=sample_colors)
plot!(p2, )
p = plot(p1,p2,layout=(1,2), size=(1200,600))
savefig(p, """$(name)_$("$(round(now(), Day))"[1:10])_$(f)_z-score.pdf""")
end
=#
## Graph metrics (e.g. motifs) ##
#@info "$(round(now(), Minute)) - Computing graph metrics for $(name)"
#@info "$(round(now(), Minute)) - Computing triadic motifs for $(name)"

#=
# illustration
p_mot = scatter(z_m_a, xlabel="Motif ID", ylabel="z-score", color=:steelblue, label="analytical (BDCM)", 
size=(1200,600), bottom_margin=5mm, left_margin=5mm, legend=:topleft, legendfontsize=6, xticks=collect(1:13))


scatter!(p_mot, z_m_S, label=sample_labels, color=sample_colors, markeralpha=0.5)
plot!(p_mot, collect(1:13), -2*ones(13), color=:red, label="statistical significance threshold", linestyle=:dash)
plot!(p_mot, collect(1:13), 2*ones(13), color=:red, label="", linestyle=:dash)
title!("$(name)")
xlims!(0,14)
savefig(p_mot, """$(name)_$("$(round(now(), Day))"[1:10])_motifs_z-score.pdf""")
=#




###############################################################################################
# checking the motif results
###############################################################################################
function makeplots()
    ## general settings ##
    output = "./data/computed_results/DBCM_result_more.jld"
    data = jldopen(output)
    data_labels = keys(data)
    plotnames = ["Chesapeake Bay";"Everglades Marshes"; "Florida Bay"; "Little Rock Lake"; "Maspalomas Lagoon"; "St Marks Seagrass" ]
    mycolors = [:crimson;        :saddlebrown;          :indigo;     :blue;           :lime;  :goldenrod ]; 
    markershapes = [:circle;        :star5 ;            :dtriangle;  :rect;               :utriangle;  :star6 ]
    linestyles = [:dot; :dash; :solid]


    #####################################################
    # plot for the z-scores (analytical + simulation)
    #####################################################
    @info "plotting z-scores for the motifs"
    motifplot = plot(size=(1600, 1200), bottom_ofset=5mm, left_ofset=5mm, thickness_scaling = 2)
    for i = eachindex(data_labels)
        plot!(motifplot, 1:13, data[data_labels[i]][:z_m_a], label=plotnames[i], color=mycolors[i], marker=markershapes[i], 
        markerstrokewidth=0, markerstrokecolor=mycolors[i])
        y = data[data_labels[i]][:z_m_S]
        for j = 1:size(y,2)
            plot!(motifplot, 1:13, y[:, j], label="", color=mycolors[i], marker=markershapes[i], 
                                        markerstrokewidth=0, markerstrokecolor=mycolors[i], line=linestyles[j], linealpha=0.5, markeralpha=0.5)
        end
    end
    plot!(motifplot,[0;14], [-2 2;-2 2], label="", color=:black, legend_position=:bottomright)
    xlabel!(motifplot, "motif")
    ylabel!(motifplot, "z-score")
    xlims!(motifplot, 0,14)
    xticks!(motifplot, 1:13)
    ylims!(motifplot, -15,15)
    yticks!(motifplot ,-15:5:15)
    title!(motifplot ,"Analytical vs Simulation results for motifs (DBCM model)\n-: analytical, .: n=100, - -:, n=1000, -: n=10000")
    savefig(motifplot, "./data/computed_results/DBCM_motifs_z-score.pdf")

    ############################################################################
    # plot for the difference in expected motif counts (analytical - simulation)
    ############################################################################
    @info "plotting difference in expected motif counts"
    motifcountplot = plot(size=(1600, 1200), bottom_ofset=5mm, left_ofset=10mm, thickness_scaling = 2,legendposition=:topleft)
    for i = eachindex(data_labels)
        y = abs.((data[data_labels[i]][:m̂] - data[data_labels[i]][:m̂_S][:,end]) ./ data[data_labels[i]][:m̂])
        plot!(motifcountplot, 1:13, y, label=plotnames[i], color=mycolors[i], marker=markershapes[i], 
        markerstrokewidth=0, markerstrokecolor=mycolors[i], yscale=:log10)
    end
    xlabel!(motifcountplot, "motif")
    ylabel!(motifcountplot, L"\left|\frac{\hat{m}_a - \hat{m}_s}{\hat{m}_a}\right|")
    title!(motifcountplot ,"Analytical vs Simulation difference in expected value estimate")
    xlims!(motifcountplot, 0,14)
    xticks!(motifcountplot, 1:13)
    ylims!(motifcountplot, 1e-6,1e-1)
    savefig(motifcountplot, "./data/computed_results/DBCM_motifs_count.pdf")

    ############################################################################
    # plot for the difference in expected motif standard deviavtion (analytical - simulation)
    ############################################################################
    @info "plotting difference in expected motif count standard deviation"
    motifstdplot = plot(size=(1600, 1200), bottom_ofset=5mm, left_ofset=10mm, thickness_scaling = 2,legendposition=:topleft)
    for i = eachindex(data_labels)
        y = abs.((data[data_labels[i]][:σ̂_m̂] - data[data_labels[i]][:σ̂_m̂_S][:,end]) ./ data[data_labels[i]][:σ̂_m̂])
        plot!(motifstdplot, 1:13, y, label=plotnames[i], color=mycolors[i], marker=markershapes[i], 
        markerstrokewidth=0, markerstrokecolor=mycolors[i], yscale=:log10)
    end
    xlabel!(motifstdplot, "motif")
    ylabel!(motifstdplot, L"\left|\frac{\hat{\sigma_m}_a - \hat{\sigma_m}_s}{\hat{\sigma_m}_a}\right|")
    title!(motifstdplot ,"Analytical vs Simulation difference in motif count standard deviation")
    xlims!(motifstdplot, 0,14)
    xticks!(motifstdplot, 1:13)
    #ylims!(motifcountplot, 1e-6,1e-1)
    savefig(motifstdplot, "./data/computed_results/DBCM_motifs_std.pdf")

    #=
    ############################################################################
    # plot some histograms of the sample
    ############################################################################
    
    for j = 1:13
        global motif_hist_plot = plot(size=(1600, 1200), bottom_ofset=5mm, left_ofset=10mm, thickness_scaling = 2,legendposition=:topleft)
        y = @view data[data_labels[4]][:S_m][j,:]

        # check data fitting
        d = fit(Normal{Float64}, y)
        dΓ = fit(Gamma, y)
        # hypothesis test for normality
        p = ExactOneSampleKSTest(y, d)
        d_F = fit( Fdistribution{Float64}, y)
        if pvalue(p) < 1e-2
            histogram!(motif_hist_plot, y, normalize=true, label="PDF")
            d_χ = Chisq(mean(y))
            x_χ = motif_hist_plot.series_list[end].plotattributes[:x]
            y_χ = pdf.(d_χ, x_χ)
            yΓ = pdf.(dΓ, x_χ)
            xΓ = motif_hist_plot.series_list[end].plotattributes[:x]
            
            p_χ = ExactOneSampleKSTest(y, d_χ)
            @info pvalue(p_χ)
            # add markers indicating the estimated of the standard deviation
            ymax = maximum(motif_hist_plot.series_list[end].plotattributes[:y])
            x_sq = data[data_labels[4]][:σ̂_m̂][j]
            x_S  = data[data_labels[4]][:σ̂_m̂_S][j,end]
            plot!(motif_hist_plot, [x_sq; x_sq], [0; ymax], label="σ_analytical", color=:red, line=:dot)
            plot!(motif_hist_plot, [x_S; x_S], [0; ymax], label="σ_sample", color=:red, line=:dash)
            #plot!(motif_hist_plot, [std(d_χ); std(d_χ)], [0; ymax], label="σ_χ (ν = $(@sprintf("%1.2f", d_χ.ν )))", color=:black, line=:dash)
            plot!(motif_hist_plot, [std(dΓ); std(dΓ)], [0; ymax], label="σ_Γ", color=:black, line=:dash)
            # add χ² pdf to plot
            #plot!(motif_hist_plot, x_χ, y_χ, label="X ~ χ²(ν=$(@sprintf("%1.2f", d_χ.ν ))) ", fillrange=0, fillalpha=0.5) 
            plot!(motif_hist_plot, xΓ, yΓ, label="X ~ Γ", fillrange=0, fillalpha=0.5) 
   
            # fix other plot details
            xlabel!(motif_hist_plot, "motif value")
            ylabel!(motif_hist_plot, L"f_X(x)")
            title!(motif_hist_plot ,"$(plotnames[4])\n motif 9 PDF (n = $(size(data["littlerock"][:S_m], 2)))\n p-value normality: $(@sprintf("%1.2e", pvalue(p)))\n skewness: $(skewness(y))\n curtosis: $(kurtosis(y)))")
            savefig(motif_hist_plot, "./data/computed_results/DBCM_motif_$(j)_histogram (non normal).pdf")
        else
            histogram!(motif_hist_plot, y, normalize=true, label="PDF")
            # add markers indicating the estimated of the standard deviation
            ymax = maximum(motif_hist_plot.series_list[end].plotattributes[:y])
            x_sq = data[data_labels[4]][:σ̂_m̂][j]
            x_S  = data[data_labels[4]][:σ̂_m̂_S][j,end]
            plot!(motif_hist_plot, [x_sq; x_sq], [0; ymax], label="σ_analytical", color=:red, line=:dot)
            plot!(motif_hist_plot, [x_S; x_S], [0; ymax], label="σ_sample", color=:red, line=:dash)

            xlabel!(motif_hist_plot, "motif value")
            ylabel!(motif_hist_plot, L"f_X(x)")
            title!(motif_hist_plot ,"$(plotnames[4])\n motif 9 PDF (n = $(size(data["littlerock"][:S_m], 2)))\n p-value normality: $(@sprintf("%1.2e", pvalue(p)))\n skewness: $(skewness(y))\n curtosis: $(kurtosis(y)))")
            savefig(motif_hist_plot, "./data/computed_results/DBCM_motif_$(j)_histogram (normal).pdf")
        end
    end
    =#

    ############################################################################################################################
    # plot some histograms, sample standard deviation, sample mean and theoretical values using the PoissonBinomial distribution
    ############################################################################################################################
    @info "plotting histograms of sample standard deviation, sample mean and theoretical values using the PoissonBinomial distribution for dataset $(data_labels[4])"
    α_lim = 0.01 # limit p-value for normality test
    for dataname in data_labels
    for j = 1:13
        motif_Pb_plot = plot(size=(1600, 1200), bottom_ofset=5mm, left_ofset=10mm, thickness_scaling = 2,legendposition=:topleft)
        # get sample counts for the motif
        y = @view data[dataname][:S_m][j,:]
        # check theoretical data fitting
        d_N = fit(Normal{Float64}, y)                           # Normal distribution
        d_Pb = eval(:($(Symbol("M_dist_$(j)"))(data[$(dataname)][:model].G)))       # Poissin-Binomial distribution
        # hypothesis test for normality
        normalitytest = ExactOneSampleKSTest(y, d_N)
        # plot the histogram
        histogram!(motif_Pb_plot, y, normalize=true, label="PDF")
        # add markers indicating the estimated of the standard deviation
        ymax = maximum(motif_Pb_plot.series_list[end].plotattributes[:y])
        σ_sq = data[dataname][:σ̂_m̂][j]            # Squartini standard deviation
        σ_S  = data[dataname][:σ̂_m̂_S][j,end]      # Sample standard deviation
        σ_th = std(d_Pb)                                # Theoretical standard deviation according to the Poisson-Binomial distribution
        plot!(motif_Pb_plot, [σ_sq; σ_sq], [0; ymax], label="σ_analytical ($(@sprintf("%1.2f", σ_sq)))", color=:black, line=:dot)
        plot!(motif_Pb_plot, [σ_S; σ_S], [0; ymax], label="σ_sample ($(@sprintf("%1.2f", σ_S)))", color=:red, line=:dot)
        plot!(motif_Pb_plot, [σ_th; σ_th], [0; ymax], label="σ_{Pb} ($(@sprintf("%1.2f", σ_th)))", color=:blue, line=:dot)
        # add markers indication the estimate of the mean
        E_sq = data[dataname][:m̂][j]            # Squartini mean
        E_s  = data[dataname][:m̂_S][j,end]      # Sample mean
        E_th = mean(d_Pb)                             # Theoretical mean according to the Poisson-Binomial distribution
        plot!(motif_Pb_plot, [E_sq; E_sq], [0; ymax], label="E_analytical ($(@sprintf("%1.12f", E_sq)))", color=:black, line=:dash)
        plot!(motif_Pb_plot, [E_s; E_s], [0; ymax], label="E_sample ($(@sprintf("%1.12f", E_s)))", color=:black, line=:dash)
        plot!(motif_Pb_plot, [E_th; E_th], [0; ymax], label="E_{Pb} ($(@sprintf("%1.12f", E_th)))", color=:black, line=:dash)
        @info abs(E_th - E_sq)
        # final layout options
        xlabel!(motif_Pb_plot, "motif value")
        ylabel!(motif_Pb_plot, L"f_X(x)")
        z_th = (data[dataname][:mˣ][j] - E_th) / σ_th
        title!(motif_Pb_plot ,"$(plotnames[4])\n motif $(j) PDF (n = $(size(data[dataname][:S_m], 2)))\n p-value normality: $(@sprintf("%1.2e", pvalue(normalitytest)))\n skewness: $(skewness(y))\n curtosis: $(kurtosis(y)))\n theoretical z-score: $(@sprintf("%1.2f", z_th))")
        figoutpath = "./data/computed_results/DBCM_$(dataname)_Pb_motif_$(j)_histogram ($(pvalue(normalitytest) < α_lim ? "non " : "")normal).pdf"
        savefig(motif_Pb_plot, figoutpath)
    end
end

    ##########################################################################################
    # illustrate the difference in σ values in function of the magnitude of the expected value
    ##########################################################################################
    @info "plotting the difference in σ values in function of the magnitude of the expected value"
    motifsigmadiffplot = plot(size=(1600, 1200), bottom_ofset=1mm, left_ofset=5mm, thickness_scaling = 2,legendposition=:bottomleft)
    motifsigmadiffplot_2 = plot(size=(1600, 1200), bottom_ofset=1mm, left_ofset=5mm, thickness_scaling = 2,legendposition=:bottomleft)
    for i = eachindex(data_labels)
        y = abs.((data[data_labels[i]][:σ̂_m̂] - data[data_labels[i]][:σ̂_m̂_S][:,end]) ./ data[data_labels[i]][:σ̂_m̂])
        x = data[data_labels[i]][:m̂]
        scatter!(motifsigmadiffplot, x, y, label=plotnames[i], color=mycolors[i], marker=markershapes[i], 
        markerstrokewidth=0, markerstrokecolor=mycolors[i], scale=:log10)

        scatter!(motifsigmadiffplot_2, 1:13, data[data_labels[i]][:m̂], label=plotnames[i], color=mycolors[i], marker=markershapes[i], 
        markerstrokewidth=0, markerstrokecolor=mycolors[i], yscale=:log10)
    end
    xlabel!(motifsigmadiffplot, L"$E[m]$")
    ylabel!(motifsigmadiffplot, L"\left|\frac{\hat{\sigma_m}_a - \hat{\sigma_m}_s}{\hat{\sigma_m}_a}\right|")
    title!(motifsigmadiffplot ,"Relative difference in σ vs. the expected value")
    xlims!(motifsigmadiffplot, 0.1,1e6)
    xticks!(motifsigmadiffplot, 10. .^(0:5))
    ylims!(motifsigmadiffplot, 1e-2,10)
    yticks!(motifsigmadiffplot, 10. .^(-2:1))

    xlabel!(motifsigmadiffplot_2, "motif")
    ylabel!(motifsigmadiffplot_2, L"$E[m]$")
    title!(motifsigmadiffplot_2 ,"expected motif count")
    xlims!(motifsigmadiffplot_2, 0, 14)
    xticks!(motifsigmadiffplot_2, collect(1:13))
    ylims!(motifsigmadiffplot_2, 1e-1,1e5)
    yticks!(motifsigmadiffplot_2, 10. .^(-1:2:5))

    savefig(plot(motifsigmadiffplot,motifsigmadiffplot_2, size=(3200, 1200), bottom_ofset=1mm, left_ofset=5mm, thickness_scaling = 2,legendposition=:bottomleft), "./data/computed_results/DBCM_sigma_diff.pdf")

end

#=

###############################################################################################
# determine maximum likelihood estimator of χ²distribution
###############################################################################################
ν_true = 10.;
x = rand(Chisq(ν_true), 100)
fig = histogram(x,normalize=:pdf, label="sample histogram")
xpoints = range(0,30,length=100)
ypoints = pdf.(Chisq(ν_true), xpoints)
plot!(xpoints, ypoints, label="true distribution")
plot!(xpoints, pdf.(Chisq(mean(x)), xpoints), label="estimated distribution")
savefig("./data/chisqauredata.pdf")

function chisqLSfit(x, ν)


end
L(x, ν) = -sum(x.^(ν[1]/2-1) .* exp.(-x ./ 2)) ./ (2^(ν[1]/2) * gamma(ν[1]/2))
fooo(ν) =  L(x, ν)
res = optimize(fooo, [mean(x)])
res

=#

#xlabel!(motifstdplot, "motif")
#ylabel!(motifstdplot, L"\left|\frac{\hat{\sigma_m}_a - \hat{\sigma_m}_s}{\hat{\sigma_m}_a}\right|")
#end
    #=
    ####################################################################
    # plot for the expacted values, normalized (analytical + simulation)
    ####################################################################
    expvalplot = plot(size=(1600, 1200), bottom_ofset=5mm, left_ofset=5mm, thickness_scaling = 2)
    for i = eachindex(data_labels)
        
        mˣ = data[data_labels[i]][:mˣ]
        m̂ = data[data_labels[i]][:m̂]
        m̂_S = data[data_labels[i]][:m̂_S][:,3]
       #  @info data_labels[i], plotnames[i], ( mˣ - m̂) ./  m̂
        plot!(expvalplot, 1:13, (m̂ -  mˣ) ./  m̂, 
            label=plotnames[i], color=mycolors[i], marker=markershapes[i], 
            markerstrokewidth=0, markerstrokecolor=mycolors[i], line=:dash)
        plot!(expvalplot, 1:13, (m̂_S - mˣ) ./ m̂_S , 
            label=plotnames[i], color=mycolors[i], marker=markershapes[i], 
            markerstrokewidth=0, markerstrokecolor=mycolors[i], line=:dot)
        #plot!(expvalplot, 1:13, log10.(data[data_labels[i]][:m̂]), label="", color=mycolors[i], marker=markershapes[i], 
        #markerstrokewidth=0, markerstrokecolor=mycolors[i], line=:dash)
        #plot!(expvalplot, 1:13, log10.(data[data_labels[i]][:m̂_S][:,3]), label="", color=mycolors[i], marker=markershapes[i], 
        #markerstrokewidth=0, markerstrokecolor=mycolors[i], line=:dot)
        #=
        plot!(expvalplot, 1:13, log10.(data[data_labels[i]][:mˣ] ), label=plotnames[i], color=mycolors[i], marker=markershapes[i], 
        markerstrokewidth=0, markerstrokecolor=mycolors[i])
        plot!(expvalplot, 1:13, log10.(data[data_labels[i]][:m̂]), label="", color=mycolors[i], marker=markershapes[i], 
        markerstrokewidth=0, markerstrokecolor=mycolors[i], line=:dash)
        plot!(expvalplot, 1:13, log10.(data[data_labels[i]][:m̂_S][:,3]), label="", color=mycolors[i], marker=markershapes[i], 
        markerstrokewidth=0, markerstrokecolor=mycolors[i], line=:dot) =#
    end
   # plot!(expvalplot, [-1], [-1], line=:solid, label="observed", color=:black)
    #plot!(expvalplot, [-1], [-1], line=:dash, label="expected", color=:black)
    #plot!(expvalplot, [-1], [-1], line=:dot, label="sample", color=:black)
    xlabel!(expvalplot, "motif")
    ylabel!(expvalplot, "value")
    xlims!(expvalplot, 0,14)
    ylims!(expvalplot, -5,2)
    xticks!(expvalplot, 1:13)
    title!(expvalplot ,"Analytical results for motifs (DBCM model)")
    savefig(expvalplot, "./data/computed_results/DBCM_motifs_counts.pdf")

    # plot for the relation between difference in analytical and simulation results and the motif actual motif count 
    # for different networks
    differenceplot = plot(size=(1600, 1200), bottom_ofset=5mm, left_ofset=5mm, thickness_scaling = 2)
    for i = eachindex(data_labels)
        @info data_labels[i], plotnames[i]
        Δ_z_score = data[data_labels[i]][:z_m_a] .- data[data_labels[i]][:z_m_S][:,2]
        x = data[data_labels[i]][:mˣ] ./ Graphs.nv(data[data_labels[i]][:network])
        @info x, Δ_z_score
        scatter!(differenceplot, x .+ 0.1  , Δ_z_score, label=plotnames[i], color=mycolors[i], marker=markershapes[i], 
        markerstrokewidth=0, markerstrokecolor=mycolors[i], xscale=:log10)
    end
    ylims!(differenceplot, -5, 5)
    xlabel!(differenceplot, "normalised motif count")
    ylabel!(differenceplot, "difference in z-score (simulation/analytical)")
    title!(differenceplot ,"Results for motifs (DBCM model)")
    savefig(differenceplot, "./data/computed_results/DBCM_motifs_delta_z-score-normalised_count.pdf")
end
=#
    #=
    motifplot = plot()

    plotnames = ["Chesapeake Bay";"Little Rock Lake"; "Maspalomas Lagoon";"Florida Bay"; "St Marks Seagrass"; "Everglades Marshes"]
    mycolors = [:crimson; :steelblue; :seagreen; :indigo; :goldenrod; :saddlebrown]; 
    markershapes = [:circle; :utriangle; :rect; :star5]

    myplot = plot(size=(800,600), bottom_margin=5mm)
    x = collect(1:13)
    for i in 1:length(plotnames)
        #@info Int((i-1)*13+1):Int((i-1)*13+13)
        #data[Int((i-1)*1+1):Int((i-1)+13),1]
        plot!(myplot, x, data[i][:,1] , color=mycolors[i], marker=markershapes[1], label="$(plotnames[i])", markeralpha=0.5, markersize=2, markerstrokewidth=0)
    # scatter!(myplot, x, data[i][:,2:end], color=mycolors[i], marker=permutedims(markershapes[2:end]), label="", markeralpha=0.5, markersize=2,markerstrokewidth=0)
    end
    plot!(myplot,[0;14], [-2 2;-2 2], label="", color=:black, legend_position=:bottomright)
    xlabel!(myplot, "motif \n∘: analytical\n △: n=100, □: n=1000, ⋆:n=10000")
    ylabel!(myplot, "z-score")
    xlims!(myplot, 0,14)
    xticks!(myplot, x)
    ylims!(myplot, -17,15)
    yticks!(myplot ,-15:5:15)
    title!(myplot ,"Analytical vs simulation results for motifs (DBCM model)")
    savefig(myplot, "comparison_motifs_dbcm_$("$(round(now(), Minute))").pdf")
    =#

#=
output = "./data/computed_results/DBCM_results.jld"
netdata = "./data/networks"
for network in filter(x-> occursin("_directed", x), joinpath.(netdata, readdir(netdata)))
    @info "working on $(network)"
    # reference name for storage
    refname = split(split(network,"/")[end],"_")[1]
    @info refname
    # load network
    G = Graphs.loadgraph(network)
    # compute motifs
    res = DBCM_motifs_analysis(G)
    # write out results
    #@info "storage result= $(Symbol($(write_result(output, refname, res))))"
    write_result(output, refname, res)
end
=#
# exploit the results




#=
# Figure 5 reproduction from the paper
G_chesapeake = readpajek("./data/Chesapeake_bay.txt");
z_chesapeake, r_chesapeake... = DBCM_analysis(G_chesapeake, "Chesapeake_bay")


Graphs.savegraph("./data/networks/everglades.lg", G_everglade)

G_little_rock = load_network_from_edges("./data/foodweb_little_rock/edges.csv")
z_little_rock, r_little_rock... = DBCM_analysis(G_little_rock, "Little_rock")

G_maspalomas = readpajek("./data/Maspalomas_lagoon.txt");
z_maspalomas, r_maspalomas... = DBCM_analysis(G_maspalomas, "Maspalomas")

G_florida_bay = readpajek("./data/Florida_bay.txt");
z_florida_bay, r_florida_bay... = DBCM_analysis(G_florida_bay, "Floriday_bay")

G_stmark = readpajek("./data/StMarks_seagrass.txt");
z_stmark, r_stmark... = DBCM_analysis(G_stmark, "St_mark")

G_everglade = readpajek("./data/raw_network_data/Everglades.txt")
z_everglade, r_everglade... = DBCM_analysis(G_everglade, "Everglades")

data = (z_chesapeake, z_little_rock, z_maspalomas, z_florida_bay, z_stmark, z_everglade)
plotnames = ["Chesapeake Bay";"Little Rock Lake"; "Maspalomas Lagoon";"Florida Bay"; "St Marks Seagrass"; "Everglades Marshes"]
mycolors = [:crimson; :steelblue; :seagreen; :indigo; :goldenrod; :saddlebrown]; 
markershapes = [:circle; :utriangle; :rect; :star5]

myplot = plot(size=(800,600), bottom_margin=5mm)
x = collect(1:13)
for i in 1:length(plotnames)
    #@info Int((i-1)*13+1):Int((i-1)*13+13)
    #data[Int((i-1)*1+1):Int((i-1)+13),1]
    plot!(myplot, x, data[i][:,1] , color=mycolors[i], marker=markershapes[1], label="$(plotnames[i])", markeralpha=0.5, markersize=2, markerstrokewidth=0)
   # scatter!(myplot, x, data[i][:,2:end], color=mycolors[i], marker=permutedims(markershapes[2:end]), label="", markeralpha=0.5, markersize=2,markerstrokewidth=0)
end
plot!(myplot,[0;14], [-2 2;-2 2], label="", color=:black, legend_position=:bottomright)
xlabel!(myplot, "motif \n∘: analytical\n △: n=100, □: n=1000, ⋆:n=10000")
ylabel!(myplot, "z-score")
xlims!(myplot, 0,14)
xticks!(myplot, x)
ylims!(myplot, -17,15)
yticks!(myplot ,-15:5:15)
title!(myplot ,"Analytical vs simulation results for motifs (DBCM model)")
savefig(myplot, "comparison_motifs_dbcm_$("$(round(now(), Minute))").pdf")

using Plots; gr()
plot(rand(10))
png("test_gr_saving")



#=
DBCMpath = "./data/computed_results/DBCM.jld"
jldopen(DBCMpath, isfile(DBCMpath) ? "r+" : "w") do file
    #=g = JLD.create_group(file, :DBCM)=#
    to_store = Dict(:zscores => z_maspalomas,
                    :m_observed => r_maspalomas[1],                        
                    :m_sim_hat => r_maspalomas[2],
                    :m_sim_std => r_maspalomas[3]
)
    write(file, String(:Maspalomas), to_store)
    #g[:Chesapeake] = 
end
jldopen(DBCMpath, "r+") do file
    to_store = Dict(:zscores => z_chesapeake,
                    :m_observed => r_chesapeake[1],                        
                    :m_sim_hat => r_chesapeake[2],
                    :m_sim_std => r_chesapeake[3]
)
    write(file, String(:Chesapeake), to_store)
    #g[:Chesapeake] = 
end

memmem =  jldopen("./data/computed_results/DBCM.jld", "r") do file
    read(file)#, "DBCM")
end

memmem =  jldopen("./data/computed_results/minitest.jld", "r") do file
    read(file)#, "DBCM")
end

=#
=#

#=

using BenchmarkTools
M = [rand(100,100) for _ in 1:100];
motifs(A::T) where T<:AbstractArray = eval.(map(f -> :($(f)($A)), DBCM_motif_functions))
@btime motifs.(M);                      # 1.946 s (47807 allocations: 2.27 MiB)
result = zeros(13, length(M))
@btime begin                            # 346.120 ms (48462 allocations: 2.29 MiB) x6 faster than above
    Threads.@threads for i in 1:length(M)
        result[:,i] .= motifs(M[i]);                      
    end
end

@btime asyncmap(motifs, M; ntasks=4);   # 1.902 s (48346 allocations: 2.28 MiB)
@btime asyncmap(motifs, M; ntasks=8);   # 1.990 s (48370 allocations: 2.28 MiB)
=#

"""
based on the wikipedia article on person type distributions
return 
- the type of pearson family (0), I → VII
- the distribution (if possible)
- pdf value associated with x
- the coefficients (TBC)
"""
function pearsonfamily(x::T) where T <: AbstractVector
    @warn "pearsonfamily"
    γ₁ = skewness(x) # symmetry measure
    γ₂ = kurtosis(x) # "tailedness" measure Note: Julia uses excess kurtosis
    β₁ = γ₁^2
    β₂ = γ₂ + 3
    
    μ₁ = mean(x)
    μ₂ = var(x)
    @info "μ₁:  $(μ₁) +  μ₂: $(μ₂)"
    @info "β₁:  $(β₁) +  β₂: $(β₂)"
    b₀ = (4*β₂ - 3*β₁) / (10*β₂ - 12*β₁ - 18) * μ₂
    b₁ = sqrt(μ₂) * sqrt(β₁) * (β₂ + 3) / (10*β₂ - 12*β₁ - 18)
    b₂ = (2*β₂ - 3*β₁ - 6)/(10*β₂ - 12*β₁ - 18)
    a = b₁
    @info "b₀: $(b₀), b₁: $(b₁), b₂: $(b₂), a: $(a)"
    # case symmetric distributions
    type = 0
    if iszero(b₁)
        if isequal(β₂, 3)
            type = 0
        elseif β₂ < 3
            type = 2
        elseif β₂ > 3
            type = 7
        end
    # case single root
    elseif isequal(b₂, 3)
        type = 3
    # case assymetrical distributions
    else
        κ = b₁^2 - 4*b₂*b₀
        if κ ≥ 0
            a₁ = (-b₁ - sqrt(κ))/(2*b₂)
            a₂ = (-b₁ + sqrt(κ))/(2*b₂)
            if isequal(sign(a₁ * a₂), -1)
                type = 1
            else
                type = 6
            end
        
        elseif κ < 0
            type = 4
        end
    end

    @info "picked type $(type) distribution"

    if isequal(type, 6)
        m₁ = (a + a₁) / (b₂*(a₂ - a₁))
        m₂ = - (a + a₂) / (b₂*(a₂ - a₁))

        @info "a₁: $(a₁) and a₂: $(a₂)"
        @info "m₁: $(m₁) and m₂: $(m₂)"
        λ = μ₁ + (a₂ - a₁) * (m₂ + 1)/(m₂ + m₁ + 2) - a₂
        @info "λ = $(λ)"
        @info "λ + a₂: $(λ + a₂)"
        @info "(a₂ - a₁): $(a₂ - a₁)"
        dist = BetaPrime(m₂ + 1, -m₂ - m₁ - 1)
        trf = (x .- (λ + a₂)) ./ (a₂ - a₁)
        #@info unique(trf), trf .< 0
        @info "difference in a: $((a₁- a₂  )) $((a₂ - a₁ )), λ = $(λ)"
        return pdf.(dist, trf) ./ (a₂ - a₁), dist,  (λ + a₂), (a₂ - a₁), std(dist) * (a₂ - a₁)
    end
end

function myfitpearson(mu1::T, mu2::T, beta1::T, beta2::T, atol::T=sqrt(eps())) where T <: Real
    @warn "working method"
    @info "mu1: $(mu1), mu2: $(mu2)"
    @info "beta1: $(beta1), beta2: $(beta2)"
    if beta1 > (beta2 - 1)
      error("There are no probability distributions with these moments")
    elseif isapprox(beta1, (beta2 -1))
      p = (1 + sqrt(beta1)/sqrt(4+beta1))/2
      a = mu1 - mu2 * sqrt((1-p)/p)
      b = mu1 + mu2 * sqrt(p/(1-p))
      error("Not a pearson distribution. Try discrete distribution with mass \n $p on $a and mass $(1-p) on $b")
    else
      b0 = mu2 * (4*beta2 - 3*beta1) / (10*beta2 - 12*beta1 - 18)
      b1 = sqrt(mu2) * sqrt(beta1) * (beta2 + 3) / (10*beta2 - 12*beta1 - 18)
      b2 = (2*beta2 - 3*beta1 - 6) / (10*beta2 - 12*beta1 - 18)
    @info "b0 = $(b0) b1 = $(b1) b2 = $(b2)"
      if isapprox(beta1, 0.0, rtol=0.0, atol=atol)
        if isapprox(beta2, 3.0, rtol=0.0, atol=atol) #Type 0
          return Normal(mu1, sqrt(mu2))
        elseif beta2 < 3 #Type II
          a1 = sqrt(mu2) / 2 * (-sqrt(-16 * beta2 * (2 * beta2 - 6)) / (2 * beta2 - 6))    
          m1 = - 1.0 / (2 * b2)
          return PearsonII(1+m1, mu1 - a1, 2*a1)
        else #beta2 > 3 #Type VII
          r = 6 * (beta2 - 1) / (2 * beta2 - 6)
          a = sqrt(mu2 * (r-1))
          return PearsonVII(1 + r, mu1, a / sqrt(1 + r))
        end
      elseif !isapprox((2*beta2 - 3*beta1 - 6), 0.0, rtol=0.0, atol=atol)
        k = 0.25 * beta1 * (beta2 + 3)^2 / ((4*beta2 - 3*beta1) * (2*beta2 - 3*beta1 - 6))
        if k < 0 #Type I
          a1 = sqrt(mu2) / 2 * ((-sqrt(beta1) * (beta2 + 3) - sqrt(beta1 * (beta2 + 3) ^ 2 - 4 * (4*beta2 - 3*beta1) * (2*beta2-3*beta1-6))) / (2*beta2-3*beta1-6))
          a2 = sqrt(mu2) / 2 * ((-sqrt(beta1) * (beta2 + 3) + sqrt(beta1 * (beta2 + 3) ^ 2 - 4 * (4*beta2 - 3*beta1) * (2*beta2-3*beta1-6))) / (2*beta2-3*beta1-6))
          if a1 > a2
            tmp = a1
            a1 = a2
            a2 = tmp
          end
          m1 =  -(sqrt(beta1) * (beta2+3) + a1*(10*beta2 - 12*beta1 - 18) / sqrt(mu2)) / 
          (sqrt(beta1 * (beta2 + 3)^2 -4*(4*beta2 - 3*beta1)*(2*beta2 - 3*beta1-6)))
          m2 = -(-sqrt(beta1) * (beta2+3) - a2*(10*beta2 - 12*beta1 - 18) / sqrt(mu2)) /
          (sqrt(beta1 * (beta2 + 3)^2 -4*(4*beta2 - 3*beta1)*(2*beta2 - 3*beta1-6)))
          lambda = mu1 - (a2-a1) * (1+m1) / (m1+m2+2)
          
          return PearsonI(1+m1, 1+m2, lambda, a2-a1)
        elseif isapprox(k, 1.0, rtol=0.0, atol=atol) #Type V
          return PearsonV(1/b2-1, mu1 - b1/(2*b2), -(b1-b1/(2*b2)) / b2) 
        elseif k > 1 #Type VI

        #@info "TYPE VI"
          a1 = sqrt(mu2)/2*((-sqrt(beta1)*(beta2+3)-sqrt(beta1*(beta2+3)^2-4*(4*beta2-3*beta1)*
                                                         (2*beta2-3*beta1-6)))/(2*beta2-3*beta1-6))
          a2 = sqrt(mu2)/2*((-sqrt(beta1)*(beta2+3)+sqrt(beta1*(beta2+3)^2-4*(4*beta2-3*beta1)*
                                                         (2*beta2-3*beta1-6)))/(2*beta2-3*beta1-6))
        @info "a1: $(a1), a2: $(a2)"
          if a1 > a2
            tmp = a1
            a1 = a2
            a2 = tmp
          end
          m1 = (b1 + a1)/(b2*(a2-a1))
          m2 = - (b1 + a2)/(b2*(a2-a1))
          @info "m1: $(m1), m2: $(m2)"
          alpha = (a2-a1)
          lambda = mu1 + alpha * (m2+1)/(m2+m1+2)
          @info "λ = $(lambda)"
          @info "α = $(alpha)"
          return
          return PearsonVI(1+m2, -m2-m1-1, lambda, alpha)
        else # (k > 0) && (k < 1) #Type IV
          r = 6*(beta2-beta1-1)/(2*beta2-3*beta1-6)
          nu = -r*(r-2)*sqrt(beta1)/sqrt(16*(r-1)-beta1*(r-2)^2)
          alpha = sqrt(mu2*(16*(r-1)-beta1*(r-2)^2))/4
          lambda = mu1 - ((r-2)*sqrt(beta1)*sqrt(mu2))/4
          m = 1+r/2
          return PearsonIV(m, nu, lambda, alpha)
        end
      else # Type III
        m = b0 / (b1^2) - 1
        return PearsonIII(1+m, mu1 - b0/b1, b1)
      end
    end
  end






  #=
output = "./data/computed_results/DBCM_result_more.jld"
motif = 11;
data = jldopen(output);
x = data["littlerock"][:S_m][motif,:];
res = pearsonfamily(x);
y = res[1]
myfig = scatter(x, y, left_margin=10mm, label="Pearson Type VI fit", size=(1200,800))
histogram!(myfig, x, label="histogram",normalize=:pdf, bins=:fd)
σ_th = data["littlerock"][:σ̂_m̂][motif]

X = (x .- res[end-2]) ./ res[end-1]
p = ExactOneSampleKSTest(X, res[2])

@info "σ_th: $(σ_th), σ_s: $(std(x)), σ_f: $(res[end])"
plot!(myfig, [σ_th; σ_th],[0.; maximum(myfig.series_list[end][:y])], line=:dot, color=:black, label="theoretical")
plot!(myfig, [std(x); std(x)],[0.; maximum(myfig.series_list[end][:y])], line=:dot, color=:red, label="from sample")
plot!(myfig, [res[end]; res[end]],[0.; maximum(myfig.series_list[end][:y])], line=:dash, color=:blue, label="from dist")
savefig(myfig, "./data/computed_results/pearsontest.pdf")


mf = MatFile("./data/computed_results/DBCM_result_probs.mat", "w")
put_variable(mf, "jprobs", y)
close(mf)

myfitpearson(mean(x), var(x), skewness(x)^2, kurtosis(x)+3)

=#

#=
    d = b₁^2 - 4*b₂*b₀
    # Type 0
    if iszero(μ₁) && iszero(μ₂)
        dist = Normal()
        return 0, dist, pdf.(dist)
    # Type I
    if β
    # Type II



    if d < 0
        @info "negative discriminant"
        # Type IV

        # Type VII
    else
        @info "positive discriminant"
        a₁ = (-b₁ - sqrt(d))/(2*b₂)
        a₂ = (-b₁ + sqrt(d))/(2*b₂)
        ν = 1/(b₂*(a₁ - a₂))
        # Type I distribution
        if sign(a₁) ≠ sign(a₂)
            @info "Type I distribution"
            
            return dist, y
        else
            @info "other type"
        end
    end
    return [b₀; b₁; b₂]
    =#
#end

#=
# sample standard deviation as estimator for the standard deviation
a, b = 100., 40.
testdist = BetaPrime(a, b)
std(testdist) == sqrt(a*(a+b-1)/((b-2)*(b-1)^2))

myS = rand(testdist,100);
sigma = std(myS);
@info std(myS), std(testdist), abs(std(myS) - std(testdist))/std(testdist)


function too_opt_mle(x, p1, p2)
    dist = BetaPrime(p1, p2)
    return -sum(log.(pdf.(dist, x)))

end

res = optimize(p -> too_opt_mle(X, p...), [63.; 7780.])




function chisqLSfit(x, ν)


end
L(x, ν) = -sum(x.^(ν[1]/2-1) .* exp.(-x ./ 2)) ./ (2^(ν[1]/2) * gamma(ν[1]/2))
fooo(ν) =  L(x, ν)
res = optimize(fooo, [mean(x)])
res

=#

#outpath = "./data/computed_results/DBCM_complete.jld"

path = "./data/computed_results/DBCM_result_more.jld"
data = jldopen(path)["littlerock"];
data[:mˣ]
Graphs.adjacency_matrix(data[:network])
#produce_squartini_dbcm_data(outpath)

#output = "./data/computed_results/DBCM_result_more.jld"
#motif = 12;
#data = jldopen(output);
#model = data["littlerock"][:model]
#count(iszero,Graphs.indegree(data["littlerock"][:network])), count(iszero,Graphs.outdegree(data["littlerock"][:network]))
#x = data["littlerock"][:S_m][motif,:];

#DBCM_analysis(data["littlerock"][:network])

#d = M_dist_(model.G)
#=
d = eval(:($(Symbol("M_dist_$(motif)"))(model.G)))
sqrt(sum(d.p .* (1 .- d.p)))
#=
@eval begin
    dd = :($(Symbol("M_dist_$(motif)"))(model.G))
end
=#



################################################################################
# Let's check the variance computation - degree: OK
################################################################################
@info "Checking the variance of the in/out-degrees"
# outdegree
σ_outdegree_theory(model,i) = sqrt(sum(model.G[i,j]*(1-model.G[i,j]) for j = 1:length(model) if j≠i))
σ_out_theory = map(j -> σ_outdegree_theory(model, j), 1:length(model))
σ_outdegree_squart =  map(j-> σˣ(m -> outdegree(m, j), model), 1:length(model))
@assert isapprox(σ_out_theory, σ_outdegree_squart) # => perfect match rounding error only
# indegree
σ_indegree_theoryf(model,i) = sqrt(sum(model.G[j,i]*(1-model.G[j,i]) for j = 1:length(model) if j≠i))
σ_in_theory = map(j -> σ_indegree_theoryf(model, j), 1:length(model))
σ_indegree_squart =  map(j-> σˣ(m -> indegree(m, j), model), 1:length(model))
@assert isapprox(σ_in_theory, σ_indegree_squart) # => perfect match rounding error only
# note: this follows a 

################################################################################
# Let's compute the variance for a motif
################################################################################
@info "Checking the variance of the motif M_13"
# e.g. motif 13 (trianle in all directions)
σ_m13_squartini = σˣ(M₁₃, model) # single core, memory intensive: 11.731 s (180851614 allocations: 7.32 GiB)
∇M_13 = ReverseDiff.gradient(M₁₃, model.G) # gradient of M_13

function ∂M_13∂a_ts(M, t,s)
    A = M.G
    res = zero(eltype(A))
    for i = axes(A,1)
        for j = axes(A,1)
            @simd for k = axes(A,1)
                if i ≠ j && j ≠ k && k ≠ i
                    if     (t,s) == (i,j)
                        res += A[j,i] * A[j,k] * A[k,j] * A[k,i] * A[i,k]
                    elseif (t,s) == (j,i)
                        res += A[i,j] * A[j,k] * A[k,j] * A[k,i] * A[i,k]
                    elseif (t,s) == (j,k)
                        res += A[i,j] * A[j,i] * A[k,j] * A[k,i] * A[i,k]
                    elseif (t,s) == (k,j)
                        res += A[i,j] * A[j,i] * A[j,k] * A[k,i] * A[i,k]
                    elseif (t,s) == (k,i)
                        res += A[i,j] * A[j,i] * A[j,k] * A[k,j] * A[i,k]
                    elseif (t,s) == (i,k)
                        res += A[i,j] * A[j,i] * A[j,k] * A[k,j] * A[k,i] 
                    end
                end
            end
        end
    end
    return res
end

∇M_13_anal = zeros(size(∇M_13))
# multi core, less memory intensive: 99.600 s (60943 allocations: 2.33 MiB)
@btime begin
    for i = 1:size(∇M_13_anal, 1)
        @info "working on $(i)/$(size(∇M_13_anal, 1))"
        Threads.@threads for j = 1:size(∇M_13_anal, 2)
            ∇M_13_anal[i,j] = ∂M_13∂a_ts(model, i,j)
        end
    end
end

@assert isapprox(∇M_13, ∇M_13_anal) # => perfect match rounding error only

σ_m13_anal = sqrt(sum((model.σ .* ∇M_13_anal) .^2) + sum())

@assert isapprox(σ_m13_squartini, σ_m13_anal) # => perfect match rounding error only

k_1_out_S = [Graphs.outdegree(rand(model),1) for _ in 1:10000];
k_1_out_S_mean = mean(k_1_out_S)
k_1_out_S_std = std(k_1_out_S)



S = [rand(model) for _ in 1:10]
d_in_S = hcat(Graphs.indegree.(S)...)
subsamples = [1;4;10]
hcat(map(n -> reshape(mean(d_in_S[:,1:n], dims=2),:), subsamples)...)
=#