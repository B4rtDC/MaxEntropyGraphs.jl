### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# ╔═╡ 47ff2cd0-5c1c-11ec-2d9c-b779168d01f6
begin
	# adding revise for quick test
	using Pkg
	Pkg.activate(pwd())
end

# ╔═╡ 4d15bb3f-1bb2-4899-b74c-c95d5e3d0cec
using StatsBase: countmap

# ╔═╡ 133df588-5e95-4f15-b1bd-399c0ff55c36
using IndirectArrays

# ╔═╡ 2c0c6029-2df6-48b0-8e3c-ce6619058751
begin
	# add my own path
	modulepath = "/Users/bart/Documents/Stack/PhD/Coding (experiments)/fastmaxent/src/MaxEntropyGraphs.jl"
	
	function ingredients(path::String)
	# this is from the Julia source code (evalfile in base/loading.jl)
	# but with the modification that it returns the module instead of the last object
	name = Symbol(basename(path))
	m = Module(name)
	Core.eval(m,
        Expr(:toplevel,
             :(eval(x) = $(Expr(:core, :eval))($name, x)),
             :(include(x) = $(Expr(:top, :include))($name, x)),
             :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
             :(include($path))))
	m
end
	#ingredients(modulepath)
end

# ╔═╡ 24606b57-6502-4399-a0e8-52fecfe2075b
begin
	abstract type AbstractMaxEntropyModel end
#IndirectArrays.IndirectArray{Float64, 1, UInt8, Vector{UInt8}, Vector{Float64}}: 


	
	mutable struct UBCM{T,N} <: AbstractMaxEntropyModel where {T<:Real, N<:UInt}
    	idx::Vector{N}
		κ::Vector{T}
		f::Dict{T,T}
		method::Symbol
		x0::Vector{T} # initial guess
		F::Vector{T}  # buffer for solution
		f!::Function  # function that will need to be solved
		xs::Vector{T} # solution vector
		x::IndirectArray{T,1, N, Vector{N}, Vector{T}}  # solution vector expanded
	end

	Base.show(io::IO, model::UBCM{T,N}) where {T,N} = print(io, """UBCM{$(T)} model ($(length(model.κ) < length(model.idx)  ? "$(round((1- length(model.κ)/length(model.idx)) * 100,digits=2))% compression in $(N)" : "uncompressed"))""")
	
	function UBCM(k::Vector{T}; compact::Bool=true, 
	                            P::DataType=Float64,
								method::Symbol=:newton,
								initial::Union{Symbol, Vector{T}}=:nodes, kwargs...) where {T<:Int}
		# check input
		method ∈ [:newton, :fixedpoint] ? nothing : throw(ArgumentError("'$(method)' is not a valid solution method"))
		
		
	    # convert datatypes
		k = convert.(P, k) 
		
		# compress or not
		if compact
			idxtype = length(unique(k)) < typemax(UInt8) ? UInt8 : length(unique(k)) < typemax(UInt16) ? UInt16 : UInt32 # limited to 4.29e9 nodes
			idx, κ = IndirectArray{idxtype}(k).index, IndirectArray{idxtype}(k).values
			f = countmap(k,ones(P, size(k)))
		else
			κ = k
			idxtype = length(κ) < typemax(UInt8) ? UInt8 : length(κ) < typemax(UInt16) ? UInt16 : UInt32 # limited to 4.29e9 nodes
			idx = collect(one(idxtype):idxtype(length(κ)))
			f = Dict(v => one(P) for v in κ)
		end

		# initial vector
		if isa(initial, Vector)
			length(initial) == length(κ) ? nothing : throw(DimensionMismatch("Length of initial vector $(length(initial)) does not match the length of κ $(length(κ))"))
			x0 = P.(initial)
		
		elseif isa(initial, Symbol)
			initial ∈ [:links, :nodes,:random] ? nothing : throw(ArgumentError("'$(initial)' is not a valid initial argument method"))
			if isequal(initial, :nodes)
				x0 = κ ./ P(sqrt(length(κ)))
			elseif isequal(initial, :links)
				x0 = κ ./ P(get(kwargs, :L, length(κ)))
			elseif isequal(initial, :random)
				x0 = rand(P, length(κ))
			end
		end

		# functions to compute
		if isequal(method, :newton)
			f! = (F::Vector, x::Vector) -> ∂UBCM∂x!(F, x, κ, f)
		elseif isequal(method, :fixedpoint)
			f! = (F::Vector, x::Vector) -> ∂UBCM∂x_it!(F, x, κ, f)
		end

		# buffers (required?)
		F = similar(x0) # modify in place method
		xs = similar(x0) # solution value

		# outresult with indirect indexing for further use
		x = IndirectArray(idx, xs)
		
	
		return UBCM(idx, κ, f, method, x0, F, f!, xs, x)
	end
	
end

# ╔═╡ 506e642c-1a9b-44c3-9cc5-633ce6824461


# ╔═╡ c8128ec1-10ab-4db0-b93d-84c2ef18a60e
Int64(typemax(UInt32))/1e9

# ╔═╡ b7ff393a-b683-4aa8-8f5e-1ac76f146a33
begin
	k = rand(1:10,20)
	
	model = UBCM(k, initial=collect(1:length(unique(k))))

	model, model.idx, model.κ, model.f, model.method, model.x0, model.F, model.xs, model.f!, model.x
end

# ╔═╡ c6c8a0d8-9edf-4a08-804b-388eca227019
eltype(model.idx)

# ╔═╡ 307ca440-9f43-4477-8deb-d8a221ab228d
IndirectArray{UInt16}(rand(1:300,100000))

# ╔═╡ 094d4633-d47f-4ea6-b828-52ac2eb62ca3
Float32[2;3;4] ./ Float32(sqrt(5))

# ╔═╡ 0eba168a-e01e-489f-8a4f-6fc044487dbc
typeof(4.) == Float64

# ╔═╡ c6ab3509-0703-4fd5-9ff0-7c3bf91cbd1a
md"""
## tests 
```julia
	# testing the proper compression method
	k = rand(1:10,30)
	idx, κ, f = UBCM(k, P=Float16, compact=false)
	map(x-> getindex(κ,x), idx) == k # check proper mapping

	# testing proper method
	UBCM(k, P=Float32, compact=false, method=:mf)

	# testing proper starting condition
	UBCM(k, P=Float32, compact=false, initial=:myinitial)

```



"""

# ╔═╡ 438c10bc-253b-4c49-b713-46dcf5609bc8
v = IndirectArray([3;2;1;3;3])

# ╔═╡ be5dadb8-c7fd-4247-aeec-b27bd4a54ba7
v.index

# ╔═╡ f7fb2a80-fafd-402e-be74-8c868c332d8e
v.values

# ╔═╡ ab85b947-74f1-4ffa-af4b-ca9dc75103b8
v.values

# ╔═╡ Cell order:
# ╠═47ff2cd0-5c1c-11ec-2d9c-b779168d01f6
# ╠═2c0c6029-2df6-48b0-8e3c-ce6619058751
# ╠═4d15bb3f-1bb2-4899-b74c-c95d5e3d0cec
# ╠═24606b57-6502-4399-a0e8-52fecfe2075b
# ╠═506e642c-1a9b-44c3-9cc5-633ce6824461
# ╠═c8128ec1-10ab-4db0-b93d-84c2ef18a60e
# ╠═b7ff393a-b683-4aa8-8f5e-1ac76f146a33
# ╠═c6c8a0d8-9edf-4a08-804b-388eca227019
# ╠═307ca440-9f43-4477-8deb-d8a221ab228d
# ╠═094d4633-d47f-4ea6-b828-52ac2eb62ca3
# ╠═0eba168a-e01e-489f-8a4f-6fc044487dbc
# ╠═c6ab3509-0703-4fd5-9ff0-7c3bf91cbd1a
# ╠═133df588-5e95-4f15-b1bd-399c0ff55c36
# ╠═438c10bc-253b-4c49-b713-46dcf5609bc8
# ╠═be5dadb8-c7fd-4247-aeec-b27bd4a54ba7
# ╠═f7fb2a80-fafd-402e-be74-8c868c332d8e
# ╠═ab85b947-74f1-4ffa-af4b-ca9dc75103b8
