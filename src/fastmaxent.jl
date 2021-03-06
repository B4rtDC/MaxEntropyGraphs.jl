module fastmaxent

import Base: show

import Graphs: degree
import LinearAlgebra: diagind
import StatsBase: countmap
#=
import Base:
    +

import Graphs:
    degree, strength
=#
using Graphs
using SimpleWeightedGraphs
using NLsolve
using IndirectArrays

# MODELS
export UBCM, UBCMCompact, solve!

# METRICS
export degree

include("models.jl")
include("metrics.jl")
end # module
