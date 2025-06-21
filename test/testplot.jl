using MetaheuristicsAlgorithms, Plots

include("../src/plotting/convergence.jl")
include("../test/benchfunctions.jl")

lb = [-512., -512.]
ub = [512., 512.]

problem = OptimizationProblem(eggholder, lb, ub, 2)

result = AEO(problem, 50, 200)

convergence_curve(result.his_best_fit)