using MetaheuristicsAlgorithms
# using Plots

include("../src/plotting/convergence.jl")
include("../src/compare_algorithms.jl")
include("../test/benchfunctions.jl")



# Define benchmark problem
lb = [-512.0, -512.0]
ub = [512.0, 512.0]
problem = OptimizationProblem(Ackley, lb, ub, 2)

# Select algorithms to compare (you can add more)
algos = Dict(
    "AEO" => AEO,
    "AFT" => AFT,
    "GWO" => GWO,
    "WOA" => WOA,
)

# Run comparison
# results = compare_algorithms(algos, problem; npop=30, max_iter=100)
results = compare_algorithms([AEO, GWO], problem; npop=30, max_iter=100)

# Plot all convergence curves
p = plot()
for (name, result) in results
    plot!(p, result.his_best_fit, label=name)
end

xlabel!("Iterations")
ylabel!("Objective Function Value")
title!("Algorithm Comparison on Ackley")
display(p)


# using MetaheuristicsAlgorithms
# using Test

# @testset "Compare Algorithms" begin
#     problem = OptimizationProblem(Ackley; dim=5, lower=-32.768, upper=32.768)
#     results = compare_algorithms([AEO], problem; npop=30, max_iter=100)

#     @test haskey(results, "AEO")
#     @test typeof(results["AEO"]) == OptimizationResult
# end