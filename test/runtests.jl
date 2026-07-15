using Test
using MaxEntropyGraphs

@testset "MaxEntropyGraphs.jl" begin
    include("./utils.jl")
    include("./metrics.jl")
    include("./models.jl")
    include("./symbolics.jl")
    include("./solver.jl")
    include("./ensemble_validation.jl")
    include("./aqua.jl")
end


