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
mkpath("cec_results")

# XLSX.openxlsx("cec_results/results.xlsx", mode="w") do xf
#     XLSX.addsheet!(xf, "Init")
# end

output_file = "cec_results/results.xlsx"

# Remove old file if exists
rm(output_file; force=true)

# Create new workbook (Excel needs at least one sheet)
# XLSX.openxlsx(output_file, mode="w") do xf
#     XLSX.addsheet!(xf, "Init")
# end

if !isfile(output_file)
    XLSX.openxlsx(output_file, mode="w") do xf
        # do nothing, just create an empty .xlsx file
    end
end

fixed_dims = Dict(
    "F14" => 2,
    "F15" => 4,
    "F16" => 2,
    "F17" => 2,
    "F18" => 2,
    "F19" => 3,
    "F20" => 6,
    "F21" => 4,
    "F22" => 4,
    "F23" => 4
)

for i in 1:23
    funcname = "F$i"
    objfun = getfield(Main, Symbol(funcname))
    # dim = 30  
    dim = get(fixed_dims, funcname, 30)
    lb = -100.0
    ub = 100.0

    println("Running $funcname...")
    # println("dim ... ", dim)

    # Run comparison
    # (algorithms,OptimizationProblem(objfun,lb, ub, dim), npop, max_iter,runs)
    results = compare_algorithms(algos, objfun, lb, ub, dim, 100, dim, 30)

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

    # XLSX.writetable("cec_results/results.xlsx", stats_df; sheetname="$funcname", overwrite=true)

    # XLSX.openxlsx("cec_results/results.xlsx", mode="rw") do xf
    #     sheet = XLSX.addsheet!(xf, "$funcname")
    #     XLSX.writetable(sheet, stats_df)
    # end
    # XLSX.openxlsx("cec_results/results.xlsx", mode="rw") do xf
    #     sheet = XLSX.addsheet!(xf, "$funcname")
    #     XLSX.writetable(sheet, Tables.columntable(stats_df), names(stats_df); anchor_cell="A1")
    # end

    # XLSX.openxlsx("cec_results/results.xlsx", mode="rw") do xf
    #     # Delete the dummy Init sheet once
    #     if "Init" in XLSX.sheetnames(xf)
    #         delete!(xf, "Init")
    #     end
    
    #     # Add new sheet and write data
    #     sheet = XLSX.addsheet!(xf, "$funcname")
    #     XLSX.writetable(sheet, Tables.columntable(stats_df), names(stats_df); anchor_cell="A1")
    # end

    # XLSX.openxlsx(output_file, mode="rw") do xf
    #     # If first real sheet, just overwrite the dummy "Init"
    #     if "Init" in XLSX.sheetnames(xf)
    #         sheet = xf["Init"]
    #         sheet.name = funcname
    #     else
    #         sheet = XLSX.addsheet!(xf, funcname)
    #     end

    #     # Write data into the sheet
    #     XLSX.writetable(sheet, Tables.columntable(stats_df), names(stats_df); anchor_cell="A1")
    # end

    #     XLSX.writetable(
    #     output_file,
    #     Tables.columntable(stats_df),
    #     names(stats_df);
    #     sheetname = funcname,
    #     anchor_cell = "A1",
    #     overwrite = false
    # )


    # Append sheet to file

    XLSX.openxlsx(output_file, mode="rw") do xf
        # Delete the sheet if it already exists
        if funcname in XLSX.sheetnames(xf)
            XLSX.delsheet!(xf, funcname)
        end
    
        # Add a new sheet and write the DataFrame
        sheet = XLSX.addsheet!(xf, funcname)
        XLSX.writetable!(sheet, Tables.columntable(stats_df), names(stats_df); anchor_cell=XLSX.CellRef("A1"))
    end
    # XLSX.openxlsx(output_file, mode="rw") do xf
    #     # Add new sheet
    #     sheet = XLSX.addsheet!(xf, funcname)
    #     # Write data to the new sheet
    #     # XLSX.writetable!(sheet, Tables.columntable(stats_df), names(stats_df); anchor_cell="A1")
    #     XLSX.writetable!(sheet, Tables.columntable(stats_df), names(stats_df); anchor_cell=XLSX.CellRef("A1"))

    # end
    
    
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
    # p2 = boxplot(data; notch=true, legend=false)
    p2 = boxplot(data; notch=false, legend=false)
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

    # println("✅ Finished $funcname\n")
end
