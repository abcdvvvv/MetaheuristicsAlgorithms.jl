export OptimizationProblem

struct OptimizationProblem{F}
    objfun::F
    lb::Vector{Float64}
    ub::Vector{Float64}
    dim::Int
end

function OptimizationProblem(f::F, lb::Vector{Float64}, ub::Vector{Float64}, dim::Int=length(lb)) where {F}
    length(lb) == length(ub) == dim || throw(DimensionMismatch("lb, ub, and dim must have the same length"))
    OptimizationProblem{F}(f, lb, ub, dim)
end

OptimizationProblem(f::F, lb::Real, ub::Real, dim::Int=1) where {F} =
    OptimizationProblem{F}(f, fill(lb, dim), fill(ub, dim), dim)
