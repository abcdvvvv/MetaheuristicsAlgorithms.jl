using Plots
import ..OptimizationResult  # if in module structure

function convergence_curve(result::OptimizationResult)
    return result.convergence_curve
end


export convergence_curve
