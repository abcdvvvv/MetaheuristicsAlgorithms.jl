using MetaheuristicsAlgorithms
using Statistics
using StatsPlots
using Plots
using PlotlyJS

include("../src/plotting/convergence.jl")
include("../src/compare_algorithms.jl")
include("../test/benchfunctions.jl")

# Define benchmark problem
lb = [-512.0, -512.0]
ub = [512.0, 512.0]
problem = OptimizationProblem(Ackley, lb, ub, 2)

# Select algorithms to compare
algos = [AEO, ALA, BES, BKA, BOA, ChOA]

# Run comparison with multiple runs
results = compare_algorithms(algos, problem; npop=30, max_iter=100, runs=30)

mean_fitnesses = Dict{String, Float64}()
# Print statistics
for (name, sols) in results
    fitnesses = [sol.bestF for sol in sols]
    mean_val = mean(fitnesses)
    mean_fitnesses[name] = mean_val
    println("Algorithm: $name")
    println("  Mean:   ", mean_val)
    println("  Median: ", median(fitnesses))
    println("  Std:    ", std(fitnesses))
    println("  Min:    ", minimum(fitnesses))
    println("  Max:    ", maximum(fitnesses))
end
# --- Convergence Curve ---
p1 = plot()
for (name, sols) in results
    # Collect convergence curves (each is a Vector of length max_iter)
    curves = [sol.his_best_fit for sol in sols]

    # Pad shorter curves (if any) with last value to match max length
    max_len = maximum(length.(curves))
    padded_curves = [length(c) == max_len ? c : vcat(c, fill(c[end], max_len - length(c))) for c in curves]

    # Compute mean curve (mean at each iteration)
    mean_curve = [mean(getindex.(padded_curves, i)) for i in 1:max_len]

    # Plot mean curve
    plot!(p1, mean_curve, label=name)
end

xlabel!("Iterations")
ylabel!("Objective Function Value")
title!("Mean Convergence Curves")
savefig(p1, "convergence.png")
display(p1)
# gui(p1)
# gr()

# --- Box plot ---

labels = collect(keys(results))
data = [ [sol.bestF for sol in results[name]] for name in labels ]

# boxplot(data; labels=labels, notch=true, legend=false)
p2 = boxplot(data; notch=true, legend=false)
xticks!(p2, 1:length(labels), labels)
xlabel!("Algorithms")
ylabel!("Final Fitness Value")
title!("Box Plot of Final Fitness Values")
savefig(p2, "box_plot.png")
display(p2)
# gui(p2)

# --- Radar plot ---
# labels = collect(keys(mean_fitnesses))
# values = [mean_fitnesses[label] for label in labels]
# # Create radar plot
# p3 = spider(values; 
#     labels=labels,
#     linecolor=:blue, 
#     marker=:circle,
#     legend=false,
#     title="Mean Fitness Radar Plot"
# )

# display(p3)
# savefig(p3, "radar_plot.png")

labels = collect(keys(results))
mean_fitness = [mean([sol.bestF for sol in results[name]]) for name in labels]

trace = scatterpolar(
    r = mean_fitness,
    theta = labels,
    fill = "toself",
    name = "Mean Fitness"
)

layout = Layout(
    polar = attr(
        radialaxis = attr(visible=true, range=[0, maximum(mean_fitness)*1.1])
    ),
    showlegend = false,
    title = "Radar Plot of Mean Fitness"
)

p3 = Plot(trace, layout)
display(p3)
# savefig(p3, "radar_plot.png")


# trace = scatterpolar(
#     r = mean_fitnesses,
#     theta = labels,
#     fill = "toself",
#     name = "Mean Fitness"
# )

# # Define layout
# layout = Layout(
#     polar = attr(
#         radialaxis = attr(visible=true, range=[0, maximum(mean_fitnesses)*1.1])
#     ),
#     showlegend = false,
#     title = "Radar Plot of Mean Fitness"
# )

# # Create and display plot
# p3 = Plot(trace, layout)
# display(p3)

# # Save the radar plot to HTML file
# savefig(p3, "radar_plot.html")
