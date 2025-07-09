using MetaheuristicsAlgorithms
using Statistics
using StatsPlots
using Plots
using PlotlyJS
using DataFrames
using XLSX

include("../src/plotting/convergence.jl")
include("../src/compare_algorithms.jl")
include("../src/benchmark/CEC2005.jl")  

# Select algorithms
# algos = [AEO, ALA, BES, BKA, BOA, ChOA]
algos = [ALA, BES, BKA, ChOA]

# Output directory
mkpath("engineering_results")

const fixed_dims = Dict(
    "Engineering_F1" => 3,
    "Engineering_F2" => 4,
    "Engineering_F3" => 3,
    "Engineering_F4" => 3,
    "Engineering_F5" => 3,
    "Engineering_F6" => 2,
    "Engineering_F7" => 2,
    "Engineering_F8" => 2,
    "Engineering_F9" => 10
)

for i in 1:8#9
    funcname = "Engineering_F$i"
    objfun = getfield(Main, Symbol(funcname))
    # dim = 30  
    dim = fixed_dims[funcname]
    lb = -100.0
    ub = 100.0

    println("Running $funcname...")

    # Run comparison
    # (algorithms,OptimizationProblem(objfun,lb, ub, dim), npop, max_iter,runs)
    results = compare_algorithms(algos, objfun, lb, ub, 30, 100, dim, 30)

    # Collect stats
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
        push!(stats_df, (
            Algorithm = name,
            Mean = mean(fitnesses),
            Median = median(fitnesses),
            Std = std(fitnesses),
            Min = minimum(fitnesses),
            Max = maximum(fitnesses)
        ))
    end

    # Save Excel file
    # XLSX.writetable("cec_results/results_$funcname.xlsx", stats_df; sheetname="Sheet1", overwrite=true)
    XLSX.writetable("engineering_results/results.xlsx", stats_df; sheetname="$funcname", overwrite=true)

    # Convergence Plot
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
    title!("Mean Convergence - $funcname")
    Plots.savefig(p1, "cec_results/convergence_$funcname.png")

    # Boxplot
    labels = collect(keys(results))
    data = [ [sol.bestF for sol in results[name]] for name in labels ]
    p2 = boxplot(data; notch=true, legend=false)
    xticks!(p2, 1:length(labels), labels)
    xlabel!("Algorithms")
    ylabel!("Final Fitness Value")
    title!("Box Plot - $funcname")
    Plots.savefig(p2, "cec_results/boxplot_$funcname.png")

    # Radar plot
    mean_fitness = [mean([sol.bestF for sol in results[name]]) for name in labels]
    trace = scatterpolar(
        r = mean_fitness,
        theta = labels,
        fill = "toself",
        name = "Mean Fitness"
    )
    layout = Layout(
        polar = attr(radialaxis = attr(visible=true, range=[0, maximum(mean_fitness)*1.1])),
        showlegend = false,
        title = "Radar Plot - $funcname"
    )
    p3 = PlotlyJS.Plot(trace, layout)
    PlotlyJS.savefig(p3, "cec_results/radar_$funcname.png")

    println("✅ Finished $funcname\n")
end