# function compare_algorithms(
#     algorithms::Vector{<:Function},
#     problem::OptimizationProblem;
#     npop::Integer = 30,
#     max_iter::Integer = 1000
# )
#     results = Dict{String, OptimizationResult}()

#     for algo in algorithms
#         name = string(nameof(algo))
#         sol = optimize(algo, problem; npop, max_iter)
#         results[name] = sol
#     end

#     return results
# end

function compare_algorithms(
    algorithms::Vector{<:Function},
    problem::OptimizationProblem;
    npop::Integer = 30,
    max_iter::Integer = 1000,
    runs::Integer = 30,
)
    results = Dict{String, Vector{OptimizationResult}}()

    for algo in algorithms
        name = string(nameof(algo))
        sols = OptimizationResult[]
        for _ in 1:runs
            sol = optimize(algo, problem; npop, max_iter)
            push!(sols, sol)
        end
        results[name] = sols
    end

    return results
end

# (objfun, lb::Real, ub::Real, npop::Integer, max_iter::Integer, dim::Integer)
function compare_algorithms(
    algorithms::Vector{<:Function},
    objfun::Function,
    lb::Real, 
    ub::Real,
    # problem::OptimizationProblem;
    dim::Integer,
    npop::Integer = 30,
    max_iter::Integer = 1000,
    runs::Integer = 30,
)
    # println("dim inside compare_algorithms ", dim)
    return compare_algorithms(algorithms,OptimizationProblem(objfun,lb, ub, dim), npop=npop, max_iter=max_iter,runs=runs)
end