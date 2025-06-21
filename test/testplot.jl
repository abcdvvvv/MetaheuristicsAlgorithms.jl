using MetaheuristicsAlgorithms, Plots

include("../src/plotting/convergence.jl")
include("../test/benchfunctions.jl")

lb = [-512., -512.]
ub = [512., 512.]

problem = OptimizationProblem(eggholder, lb, ub, 2)

result = AEO(problem, 50, 200)

convergence_curve(result)

fig = plot(result.his_best_fit,
        # bg_color=RGB(0.12),
        label="Convergence Curve",
        # linecolor=:yellow3,
        # marker=:circle,
        # markercolor=:yellow3,
        xlabel="Iterations",
        ylabel="Objective Function Value")
        # display(fig)
        plot(1:10,1:10)