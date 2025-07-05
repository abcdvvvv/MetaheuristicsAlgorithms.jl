"""

    AEO(npop, max_iter, lb, ub, objfun)
    AEO(problem::OptimizationProblem, npop, max_iter)

Run the Artificial Ecosystem-based Optimization (AEO) algorithm.

This function supports two ways to define the optimization problem:
- Pass the population size, iteration count, bounds, and objective function directly.
- Pass an `OptimizationProblem` struct along with `npop` and `max_iter`.

# Arguments

## Common
- `npop::Integer`: Number of individuals in the population.
- `max_iter::Integer`: Maximum number of iterations.

## For `AEO(npop, max_iter, lb, ub, objfun)`
- `lb::Union{Real, AbstractVector}`: Lower bounds of the search space.
- `ub::Union{Real, AbstractVector}`: Upper bounds of the search space.
- `objfun::Function`: Objective function to minimize.

## For `AEO(problem::OptimizationProblem, npop, max_iter)`
- `problem::OptimizationProblem`: A struct containing `objfun`, `lb`, `ub`, and `dim`.

# Returns

- `OptimizationResult`: A struct containing:
  - `bestX::Vector`: The best solution found.
  - `bestF::Float64`: The best fitness value.
  - `his_best_fit::Vector{Float64}`: History of best fitness value at each iteration.

# Example

```julia
# Signature 1
result = AEO(30, 100, -5.12, 5.12, Ackley)

# Signature 2
problem = OptimizationProblem(Ackley, -5.12, 5.12, 10)
result = AEO(problem, 30, 100)
```
"""
function AEO(npop::Integer, max_iter::Integer, lb::Union{Real,AbstractVector{<:Real}}, ub::Union{Real,AbstractVector{<:Real}}, objfun)::OptimizationResult
    dim = length(lb)
    PopPos = zeros(npop, dim)
    PopFit = zeros(npop)

    newPopFit = zeros(npop, dim)
    his_best_fit = zeros(max_iter + 1)
    PopPos = initialization(npop, dim, ub, lb)
    for i = 1:npop
        PopFit[i] = objfun(PopPos[i, :])
    end

    indF = sortperm(PopFit, rev=true)
    PopPos = PopPos[indF, :]
    PopFit = PopFit[indF]
    his_best_fit[1] = BestF = PopFit[end]
    BestX = PopPos[end, :]

    Matr = [1, dim]

    for it = 1:max_iter
        r1 = rand()
        a = (1 - it / max_iter) * r1
        xrand = rand(dim) .* (ub - lb) .+ lb
        newPopPos = zeros(npop, dim)
        newPopPos[1, :] = (1 - a) * PopPos[end, :] .+ a * xrand  # equation (1)

        for i = 3:npop
            u = randn(dim)
            v = randn(dim)
            C = 1 / 2 * u ./ abs.(v)  # equation (4)

            r = rand()
            if r < 1 / 3
                newPopPos[i, :] = PopPos[i, :] .+ C .* (PopPos[i, :] .- newPopPos[1, :])  # equation (6)
            elseif r < 2 / 3
                r_idx = rand(2:i-1)
                newPopPos[i, :] = PopPos[i, :] .+ C .* (PopPos[i, :] .- PopPos[r_idx, :])  # equation (7)
            else
                r2 = rand()
                r_idx = rand(2:i-1)
                newPopPos[i, :] = PopPos[i, :] .+ C .* (r2 * (PopPos[i, :] .- newPopPos[1, :]) .+ (1 - r2) .* (PopPos[i, :] .- PopPos[r_idx, :]))  # equation (8)
            end
        end

        for i = 1:npop
            newPopPos[i, :] = max.(min.(newPopPos[i, :], ub), lb)
            newPopFit[i] = objfun(newPopPos[i, :])
            if newPopFit[i] < PopFit[i]
                PopFit[i] = newPopFit[i]
                PopPos[i, :] = newPopPos[i, :]
            end
        end

        indOne = argmin(PopFit)
        for i = 1:npop
            r3 = rand()
            Ind = rand(1:2)
            newPopPos[i, :] = PopPos[indOne, :] .+ 3 * randn(Matr[Ind]) .* ((r3 * rand(1:2) .- 1) .* PopPos[indOne, :] .- (2 * r3 - 1) .* PopPos[i, :])  # equation (9)
        end

        for i = 1:npop
            newPopPos[i, :] = max.(min.(newPopPos[i, :], ub), lb)
            newPopFit[i] = objfun(newPopPos[i, :])
            if newPopFit[i] < PopFit[i]
                PopPos[i, :] = newPopPos[i, :]
                PopFit[i] = newPopFit[i]
            end
        end

        indF = sortperm(PopFit, rev=true)
        PopPos = PopPos[indF, :]
        PopFit = PopFit[indF]

        if PopFit[end] < BestF
            BestF = PopFit[end]
            BestX = PopPos[end, :]
        end

        his_best_fit[it+1] = BestF
    end

    return OptimizationResult(
        BestX,
        BestF,
        his_best_fit)
end

function AEO(problem::OptimizationProblem, npop::Integer = 30, max_iter::Integer = 1000)::OptimizationResult
    dim = problem.dim
    objfun = problem.objfun
    lb = problem.lb
    ub = problem.ub
    return AEO(npop, max_iter, lb, ub, objfun)
end