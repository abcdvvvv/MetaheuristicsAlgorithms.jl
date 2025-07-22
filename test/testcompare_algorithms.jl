using MetaheuristicsAlgorithms
using Statistics
using StatsPlots
using Plots
using PlotlyJS
using DataFrames
using XLSX

include("../src/plotting/convergence.jl")
include("../src/compare_algorithms.jl")
include("../test/benchfunctions.jl")

# Define benchmark problem
lb = [-512.0, -512.0]
ub = [512.0, 512.0]
problem = OptimizationProblem(Ackley, lb, ub, 2)

# Select algorithms to compare
# algos = [AEO, ALA, BES, BKA, BOA, ChOA]
algos =[AEO, ALA, LSHADE_cnEpSin]

# Run comparison with multiple runs
results = compare_algorithms(algos, problem; npop=30, max_iter=100, runs=30)
# results = compare_algorithms(algos, problem; npop=3, max_iter=10, runs=30)

mean_fitnesses = Dict{String, Float64}()

stats_df = DataFrame(
    Algorithm = String[],
    Mean = Float64[],
    Median = Float64[],
    Std = Float64[],
    Min = Float64[],
    Max = Float64[]
)

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

    push!(stats_df, (
        Algorithm = name,
        Mean = mean(fitnesses),
        Median = median(fitnesses),
        Std = std(fitnesses),
        Min = minimum(fitnesses),
        Max = maximum(fitnesses)
    ))

end
# CSV.write("results.csv", stats_df)
XLSX.writetable("results.xlsx", stats_df; sheetname="Sheet1", overwrite=true)

# --- Convergence Curve ---
p1 = Plots.plot()
for (name, sols) in results
    curves = [sol.his_best_fit for sol in sols]
    max_len = maximum(length.(curves))
    padded_curves = [length(c) == max_len ? c : vcat(c, fill(c[end], max_len - length(c))) for c in curves]
    mean_curve = [mean(getindex.(padded_curves, i)) for i in 1:max_len]
    Plots.plot!(p1, mean_curve, label=name)
end

xlabel!("Iterations")
ylabel!("Objective Function Value")
title!("Mean Convergence Curves")
Plots.savefig(p1, "convergence.png")
display(p1)


# --- Box plot ---
labels = collect(keys(results))
data = [ [sol.bestF for sol in results[name]] for name in labels ]
p2 = boxplot(data; notch=true, legend=false)
xticks!(p2, 1:length(labels), labels)
xlabel!("Algorithms")
ylabel!("Final Fitness Value")
title!("Box Plot of Final Fitness Values")
Plots.savefig(p2, "box_plot.png")
display(p2)

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

p3 = PlotlyJS.Plot(trace, layout)
display(p3)
PlotlyJS.savefig(p3, "radar_plot.png")


