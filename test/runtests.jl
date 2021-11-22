using fastmaxent

println("Testing Models...")
t = @elapsed include("ModelsTest.jl")
println("done (tookt $t seconds).")