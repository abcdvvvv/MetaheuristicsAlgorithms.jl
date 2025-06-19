
# module ProblemDefinition

# export MyOptimization_Problem

struct MyOptimization_Problem
    objfun::Function
    lb::Union{Real, AbstractVector{<:Real}}
    ub::Union{Real, AbstractVector{<:Real}}
    dim::Int

    # function MyOptimization_Problem(f::Function, lb, ub, dim::Int)
    #     return MyOptimization_Problem(f, lb, ub, dim)
    # end
    
end



# end