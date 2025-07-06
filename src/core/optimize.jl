function optimize(algorithm::Function, problem::OptimizationProblem; npop=30, max_iter=100)
    return algorithm(problem, npop, max_iter)
end