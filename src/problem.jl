export OptimizationProblem

struct OptimizationProblem{F}
    objfun::F
    lb::Vector{Float64}
    ub::Vector{Float64}
    dim::Int
end

function OptimizationProblem(f::F, lb::Vector{T}, ub::Vector{T}, dim::Integer=length(lb)) where {F,T<:Real}
    length(lb) == length(ub) == dim || throw(DimensionMismatch("lb, ub, and dim must have the same length"))
    OptimizationProblem{F}(f, lb, ub, dim)
end

OptimizationProblem(f::F, lb::Real, ub::Real, dim::Integer=1) where {F} =
    OptimizationProblem{F}(f, fill(lb, dim), fill(ub, dim), dim)
