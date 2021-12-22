"""
    AbstractMaxEntropyModel

An abstract type for a MaxEntropyModel. Each model has one or more structural constraints  
that are fixed while the rest of the network is completely random.
"""
abstract type AbstractMaxEntropyModel end

"""
    UBCM{T,N} <: AbstractMaxEntropyModel where {T<:Real, N<:UInt}

# Undirected Binary Configuration Model (UBCM)
Maximum entropy model with a fixed degree sequence. Uses the model where ``x_i = e^{-\\theta_i}``

See also: [AbstractMaxEntropyModel](@ref)
"""
mutable struct UBCM{T,N} <: AbstractMaxEntropyModel where {T<:Real, N<:UInt}
    idx::Vector{N}
    κ::Vector{T}
    f::Vector{T} 
    method::Symbol
    x0::Vector{T} # initial guess
    F::Vector{T}  # buffer for solution => used?
    f!::Function  # function that will need to be solved
    xs::Vector{T} # solution vector
    x::IndirectArray{T,1, N, Vector{N}, Vector{T}}  # solution vector expanded
end

Base.show(io::IO, model::UBCM{T,N}) where {T,N} = print(io, """UBCM{$(T)} model ($(length(model.κ) < length(model.idx)  ? "$(round((1- length(model.κ)/length(model.idx)) * 100,digits=2))% compression in $(N)" : "uncompressed"))""")
	
"""
	UBCM(k::Vector) where {T<:Int}

Generate a UBCM model from degree vector of a graph.
### Input
- `k` degree vector of the graph
- keyword arguments:
    - `compact` (optional, default: `true`) flag to indicate compression usage (as the same constraint leads to the same value)
	- `P` (optional, default: `Float64`) datatype used for storage
	- `method` (optional, default: `:newton`) method to uses to solve the system of equations; should be one of `:newton` or `fixedpoint`
	- `initial` (optional, default: `:degree`) vector of initial values or symbol indication what method tu use for initial guess

### Output
`UBCM` object

### Examples
```Julia
UBCM([1;2;2;3])
````

### See also
[`UBCM`](@ref)
"""
function UBCM(k::Vector{T}; compact::Bool=true, 
							P::DataType=Float64,
							method::Symbol=:newton,
							initial::Union{Symbol, Vector}=:degree, kwargs...) where {T<:Int}
	# check input
	method ∈ [:newton, :fixedpoint] ? nothing : throw(ArgumentError("'$(method)' is not a valid solution method"))
	
	# convert datatypes
	k = convert.(P, k) 
	
	# compress or not
	if compact
		idxtype = length(unique(k)) < typemax(UInt8) ? UInt8 : length(unique(k)) < typemax(UInt16) ? UInt16 : UInt32 # limited to 4.29e9 nodes
		idx, κ = IndirectArray{idxtype}(k).index, IndirectArray{idxtype}(k).values
		f = countmap(k, ones(P, size(k))) # using weight vector for proper type inference
		f = [f[v] for v in κ] # addition
	else
		κ = k
		idxtype = length(κ) < typemax(UInt8) ? UInt8 : length(κ) < typemax(UInt16) ? UInt16 : UInt32 # limited to 4.29e9 nodes
		idx = collect(one(idxtype):idxtype(length(κ)))
		f = ones(P,length(κ)) #     Dict(v => one(P) for v in κ)
	end

	# initial vector - need more work (current conditions are unstable)
	if isa(initial, Vector)
		@warn "THIS IS NOT WORKING!\n$(initial)\n$(initial[idx])\n$(κ)"
		length(initial[idx]) == length(κ) ? nothing : throw(DimensionMismatch("Length of initial vector $(length(initial)) does not match the length of κ $(length(κ))"))
		x0 = P.(initial[idx])
	elseif isa(initial, Symbol)
		initial ∈ [:random, :uniform, :degree, :nodes, :edges] ? nothing : throw(ArgumentError("'$(initial)' is not a valid initial argument method in combination with `$(method)` as a solution method"))
		# should be evaluated more
		if isequal(initial, :random)      		# random in  [0,1]
			x0 = rand(P, length(κ))
		elseif isequal(initial, :uniform) 		# uniform unique random value in [0,1]
			x0 = rand(P) .* ones(P, length(κ))
		elseif isequal(initial, :degree) 		# proportional to degree in [0,1]
			x0 = κ ./ maximum(κ)
		elseif isequal(initial, :nodes) 		# proportional to sqrt of number or nodes
			x0 = κ ./ P.(sqrt(sum(f)))
		elseif isequal(initial, :edges)			# proportional to sqrt of number of edges
			x0 = κ ./ P.(sqrt(sum(f .* κ)/2))
		end
	else
		throw(ArgumentError("`initial` should be either a Vector or a Symbol"))
	end

	# functions to compute
	if isequal(method, :newton)
		f! = (F::Vector, x::Vector) -> UBCM_∂ℒ_∂x!(F, x, κ, f)
	elseif isequal(method, :fixedpoint)
		f! = (F::Vector, x::Vector) -> UBCM_∂ℒ_∂x_it!(F, x, κ, f)
	end

	# buffers (required?)
	F = similar(x0) # modify in place method
	xs = similar(x0) # solution value

	# outresult with indirect indexing for further use
	x = IndirectArray(idx, xs)
	
	return UBCM(idx, κ, f, method, x0, F, f!, xs, x)
end

"""
	UBCM(G::T; kwargs...) where T<:AbstractGraph

Generate UBCM model directly from a graph.

### See also: 
[`UBCM`](@ref)
"""
function UBCM(G::T; kwargs...) where T<:AbstractGraph
	if is_directed(G)
		@warn "Using a directed graph ($(G)) with an undirected network model"
	end

	return  UBCM(degree(G); kwargs...)
end


"""
	fffff(x)

helper function for UBCM gradient computation
"""
fffff(x) = x / (1 + x)

"""
	UBCM_∂ℒ_∂x!(F::Vector, x::Vector, κ::Vector, f::Vector)

Gradient of the likelihood function of the UBCM model using the ``x_{i}`` formulation. 

F value of gradients
x value of parameters
κ value of (reduced) degree vector
f frequency associated with each value in κ 

### See also: 
[`UBCM`](@ref)

"""
function UBCM_∂ℒ_∂x!(F::Vector, x::Vector, k::Vector,f::Vector)
	@tturbo for i in eachindex(x)
		F[i] = -fffff(x[i]*x[i]) - k[i]
		for j in eachindex(x)
			F[i] += f[j] * fffff(x[i]*x[j])
		end
	end

	return F
end


"""
	UBCM_∂ℒ_∂x_it!(F::Vector, x::Vector, κ::Vector, f::Vector)

Iterative gradient of the likelihood function of the UBCM model using the `x_i` formulation. 

F value of function
x value of parameters
κ value of (reduce) degree vector
f frequency associated with each value in κ 

### See also: 
[`UBCM`](@ref)

"""
function UBCM_∂ℒ_∂x_it!(F::Vector{T}, x::Vector{T}, κ::Vector{T}, f::Vector{T}) where {T}
    @tturbo for i in eachindex(x)
		fx = -x[i] / (1 + x[i] * x[i])
		for j in eachindex(x)
			fx += f[j] * x[j] / (1 + x[j] * x[i])
		end

		F[i] = κ[i] / fx
	end

	return F
end


"""
	solve!model::UBCM; , kwargs...)

Solve the equations of a UBCM model to obtain the maximum likelihood parameters.

### See also [`UBCM`](@ref)
"""
function solve!(model::UBCM; kwargs...)
	if isequal(model.method, :newton)
		res = nlsolve(model.f!, model.x0; autodiff=:forward, kwargs...)
	elseif isequal(model.method, :fixedpoint)
		res = fixedpoint(model.f!, model.x0; kwargs...)
	end
	
	if res.f_converged || res.x_converged
		model.x = res.zero[model.idx]
	else
		@warn "Method `$(model.method)` did not converge. Verify initial conditions or try another method."
	end

	return res
end









"""
    DBCM{T,N} <: AbstractMaxEntropyModel where {T<:Real, N<:UInt}

# Directed Binary Configuration Model (DBCM)
Maximum entropy model with a fixed in- and outdegree sequence. 
"""
mutable struct DBCM{T,N} <: AbstractMaxEntropyModel where {T<:Real, N<:UInt} end


"""
	UBCM_∂ℒ_∂x!(F::Vector, x::Vector, κ::Vector, f::Vector)

Gradient of the likelihood function of the UBCM model using the ``x{i}`` formulation. Used to generate a function
`(F, x) -> UBCM_∂ℒ_∂x!(F, x, κ, f)` by a `UBCM` instance.

F value of gradients
x value of parameters
κ value of (reduced degree vector)
f frequency associated with each value in κ 

## see also: 
[`UBCM`](@ref)

"""
function UBCM_∂ℒ_∂θ!(F::Vector, x::Vector, k::Vector,f::Vector)
	@tturbo for i in eachindex(x)
		F[i] = -fffff(x[i]*x[i]) - k[i]
		for j in eachindex(x)
			F[i] += f[j] * fffff(x[i]*x[j])
		end
	end
	return F
end