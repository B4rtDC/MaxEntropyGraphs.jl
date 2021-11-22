# degree

degree(model::Union{UBCM, UBCMCompact}, i::Int) = sum(a(i,j,model.x) for j in eachindex(model.x) if j≠i)
degree(model::Union{UBCM, UBCMCompact}, v::Vector{Int}=collect(1:length(model.x))) = [degree(model,i) for i in v]

indegree(model::Union{UBCM, UBCMCompact}, i::Int) = degree(model, i)
outdegree(model::Union{UBCM, UBCMCompact}, v::Vector{Int}=collect(1:length(model.x))) = degree(model, v)

#⇽(x) = x + oneunit(x)


#f⟨a⟩(x) = x + oneunit(x)