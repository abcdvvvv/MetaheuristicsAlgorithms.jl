using Plots
import ..OptimizationResult  # if in module structure

function convergence_curve(result::OptimizationResult)
    fig = plot(result.his_best_fit,
        # bg_color=RGB(0.12),
        label="Convergence Curve",
        # linecolor=:yellow3,
        # marker=:circle,
        # markercolor=:yellow3,
        xlabel="Iterations",
        ylabel="Objective Function Value")
        # display(fig)
        # plot(1:10,1:10)
end

export convergence_curve

default(fontfamily="arial",
    bg_color=:transparent,
    framestyle=:box,
    tick_direction=:none,
    linewidth=1.5,
    tickfont=(12, RGB(0)), # RGB(0)=:black
    guidefont=(12, RGB(0)),
    fg_color_axis=RGB(0),
    fg_color_border=RGB(0),
    fg_color_grid=RGB(0.82),
    markersize=3,
    markerstrokewidth=0,
    legendfontsize=11,
    legendfontcolor=RGB(0),
    legend_foreground_color=RGB(0))
# plotlyjs()