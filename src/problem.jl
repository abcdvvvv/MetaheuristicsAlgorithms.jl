
# module ProblemDefinition

# export OptimizationProblem

struct OptimizationProblem{F}
    f::Function
    lb::Union{Real, AbstractVector{<:Real}}
    ub::Union{Real, AbstractVector{<:Real}}
    dim::Int
end

function OptimizationProblem(f::Function, lb, ub, dim::Int)
    return OptimizationProblem{typeof(f)}(f, lb, ub, dim)
end

# end