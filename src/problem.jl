
# module ProblemDefinition



struct OptimizationProblem
    objfun::Function
    lb::Union{Real, AbstractVector{<:Real}}
    ub::Union{Real, AbstractVector{<:Real}}
    dim::Int

    # function OptimizationProblem(f::Function, lb, ub, dim::Int)
    #     return OptimizationProblem(f, lb, ub, dim)
    # end
    
end

export OptimizationProblem