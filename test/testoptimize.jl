
# using MetaheuristicsAlgorithms
# using Plots

include("../src/plotting/convergence.jl")
include("../test/benchfunctions.jl")

# Define bounds and problem
lb = [-512.0, -512.0]
ub = [512.0, 512.0]
problem = OptimizationProblem(Ackley, lb, ub, 2)

# Run AEO algorithm
result = AEO(problem, 50, 200)

# Call your convergence curve plotting function if it returns a plot object
fig = convergence_curve(result)

# Optionally, you can create your own plot directly too:
p = plot(result.his_best_fit,
    label="Convergence Curve",
    xlabel="Iterations",
    ylabel="Objective Function Value",
    title="AEO on Ackley Function")

display(p)
