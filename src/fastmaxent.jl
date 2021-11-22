module fastmaxent

import Base: show

import LightGraphs: degree
import LinearAlgebra: diagind
import StatsBase: countmap
#=
import Base:
    +

import LightGraphs:
    degree, strength
=#
using LightGraphs
using SimpleWeightedGraphs
using NLsolve
using LightGraphs
using IndirectArrays

# MODELS
export UBCM, UBCMCompact, solve!

# METRICS
export degree

include("models.jl")
include("metrics.jl")
end # module
