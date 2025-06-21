using MetaheuristicsAlgorithms
using Test
using Random
using Plots

include("benchfunctions.jl")

# include("testbenchmark.jl")
# include("testdso.jl")
# include("testaeo.jl")

# This tests are disabled due to a possible bug (? maybe) in the AHA and AFT algorithms
# include("testaha.jl")
# include("testaft.jl")

include("testalgo/testaeostruct.jl")
include("testplot.jl")
# include("testwoastruct.jl")



