"""
    MaxEntropyGraphs

Julia module for working with maximum entropy graphs
"""
module MaxEntropyGraphs

import Base: show
#import Graphs: degree
#import LinearAlgebra: diagind
import StatsBase: countmap

using Graphs
# using SimpleWeightedGraphs

using NLsolve
using IndirectArrays
using LoopVectorization
# MODELS
export #UBCM #, UBCMCompact, solve!

# METRICS
#export degree

include("models.jl")
#include("metrics.jl")
end # module
