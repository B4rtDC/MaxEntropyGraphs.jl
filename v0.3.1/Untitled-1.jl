


# figure for z-score in function of the degree
overview = plot(size=(1200, 800), bottom_ofset=5mm, left_ofset=10mm, thickness_scaling = 2,legendposition=:bottomright)
scatter!(res[:d_inˣ], abs.(res[:z_d_in_S][:,end-1]), label="Sample (n=10000)",markerstrokewidth=0, markeralpha=0.5);
scatter!(res[:d_inˣ], abs.(res[:z_d_in_sq][:,end]), label="Squartini method",markerstrokewidth=0, markeralpha=0.5);
scatter!(res[:d_inˣ], abs.(res[:z_d_in_dist]), label="PoissonBinomial", yscale=:log10, markerstrokewidth=0, markeralpha=0.5);
#xlims!(0,100)
ylims!(1e-20, 1)
xlabel!("observed indegree")
ylabel!("z-score indegre")

nodeid = 68
specific = plot(size=(1200, 800), bottom_ofset=5mm, left_ofset=10mm, thickness_scaling = 2,legendposition=:bottomright)
b =histogram(res[:d_in_S][nodeid,:], label="Sample (n=10000)", normalize=:pdf, bin_width=1);
ymax = maximum(b.series_list[end].plotattributes[:y])

σ_sq = res[:σ̂_d̂_in][nodeid]            # Squartini standard deviation
σ_S  = res[:σ̂_in_S ][nodeid,end]        # Sample standard deviation
σ_th = std(res[:d_in_dist][nodeid])                                # Theoretical standard deviation according to the Poisson-Binomial distribution

x = range(1,100,step=1)
plot!(x, pdf.(res[:d_in_dist][nodeid], x))

plot!(specific, [σ_sq; σ_sq], [0; ymax], label="σ_analytical ($(@sprintf("%1.2f", σ_sq)))", color=:black, line=:dot)
plot!(specific, [σ_S; σ_S], [0; ymax], label="σ_sample ($(@sprintf("%1.2f", σ_S)))", color=:red, line=:dot)
plot!(specific, [σ_th; σ_th], [0; ymax], label="σ_{Pb} ($(@sprintf("%1.2f", σ_th)))", color=:blue, line=:dash, xscale=:log10)



#scatter!(res[:d_inˣ], abs.(res[:z_d_in_S][:,end]), label="Sample (n=10000)",markerstrokewidth=0, markeralpha=0.5);
#scatter!(res[:d_inˣ], abs.(res[:z_d_in_sq][:,end]), label="Squartini method",markerstrokewidth=0, markeralpha=0.5);
#scatter!(res[:d_inˣ], abs.(res[:z_d_in_dist]), label="PoissonBinomial", yscale=:log10, markerstrokewidth=0, markeralpha=0.5);
#xlims!(0,100)
#ylims!(1e-20, 1)
xlabel!("indegree")
ylabel!("PDF")

plot(overview, specific)
savefig("./data/degree_scatter.pdf")


# fix de motieven (e.g. moetief 13

function M_13_dist_bis(A::T)  where T<:AbstractArray
    res = eltype(A)[] 
    buffer = Set{Vector{Tuple{Int64, Int64}}}()
    for i = axes(A,1)
        for j = axes(A,1)
            @simd for k = axes(A,1)
                if i ≠ j && j ≠ k && k ≠ i
                    combo = sort!([(i,j),(j,i),(j,k),(k,j),(k,i),(i,k)])
                    if combo ∉ buffer
                        push!(buffer, combo)
                        push!(res, motif_functions[13][1](A,i,j) * motif_functions[13][2](A,j,k) * motif_functions[13][3](A,k,i))
                    end
                end
            end
        end
    end
    return PoissonBinomial(res)
end 

function M_13_dist_bis_bis(m::DBCM)
    A = m.G
    res = eltype(A)[] 
    σ²_v  = eltype(A)[] # will hold the different sigma values 
    buffer = Set{Vector{Tuple{Int64, Int64}}}()
    for i = axes(A,1)
        for j = axes(A,1)
            @simd for k = axes(A,1)
                if i ≠ j && j ≠ k && k ≠ i
                    combo = sort!([(i,j),(j,i),(j,k),(k,j),(k,i),(i,k)])
                    if combo ∉ buffer
                        push!(buffer, combo) # add tracked result
                        v = prod([m.σ[coord...]^2 - m.G[coord...]^2 for coord in combo]) - prod([ m.G[coord...] for coord in combo])^2
                        push!(σ²_v, v)
                    end
                end
            end
        end
    end
    return σ²_v
end 


Pd=M_dist_13(data[:model].G)
Pd_bis=M_13_dist_bis(data[:model].G)