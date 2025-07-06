function compare_algorithms(
    algorithms::Vector{<:Function},
    problem::OptimizationProblem;
    npop::Integer = 30,
    max_iter::Integer = 1000
)
    results = Dict{String, OptimizationResult}()

    for algo in algorithms
        name = string(nameof(algo))
        sol = optimize(algo, problem; npop, max_iter)
        results[name] = sol
    end

    return results
end