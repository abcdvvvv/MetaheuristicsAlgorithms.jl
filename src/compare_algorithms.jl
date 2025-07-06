function compare_algorithms(
    algorithms::Vector{<:Function},
    problem::OptimizationProblem;
    npop::Int = 30,
    max_iter::Int = 1000
)
    results = Dict{String, OptimizationResult}()

    for algo in algorithms
        name = string(nameof(algo))
        sol = optimize(algo, problem; npop=npop, max_iter=max_iter)
        results[name] = sol
    end

    return results
end