# struct AEOResult <: OptimizationResult
#     BestF::Any
#     BestX::Any
#     HisBestFit::Any
# end

"""
    AEO(npop, max_iter, lb, ub, objfun)

    Artificial Ecosystem-based Optimization (AEO) algorithm implementation in Julia.

# Arguments:

- `npop`: Number of individuals in the population.
- `max_iter`: Maximum number of iterations.
- `lb`: Lower bounds for the search space.
- `ub`: Upper bounds for the search space.
- `objfun`: Function to evaluate the fitness of individuals.

# Returns:

- `AEOResult`: A struct containing:
  - `BestF`: The best fitness value found.
  - `BestX`: The position corresponding to the best fitness.
  - `his_best_fit`: A vector of best fitness values at each iteration.

# References: 

- Zhao, Weiguo, Liying Wang, and Zhenxing Zhang. "Artificial ecosystem-based optimization: a novel nature-inspired meta-heuristic algorithm." Neural Computing and Applications 32, no. 13 (2020): 9383-9425.
"""
function AEO(npop::Integer, max_iter::Integer, lb::Union{Real,AbstractVector{<:Real}}, ub::Union{Real,AbstractVector{<:Real}}, objfun)#::AEOResult
    dim = length(lb)
    PopPos = zeros(npop, dim)
    PopFit = zeros(npop)

    newPopFit = zeros(npop, dim)
    for i = 1:npop
        PopPos[i, :] = rand(dim) .* (ub - lb) .+ lb
        PopFit[i] = objfun(PopPos[i, :])
    end

    indF = sortperm(PopFit, rev=true)
    PopPos = PopPos[indF, :]
    PopFit = PopFit[indF]
    BestF = PopFit[end]
    BestX = PopPos[end, :]

    his_best_fit = zeros(max_iter)
    Matr = [1, dim]

    for It = 1:max_iter
        r1 = rand()
        a = (1 - It / max_iter) * r1
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

        his_best_fit[It] = BestF
    end

    return OptimizationResult(
        BestX,
        BestF,
        his_best_fit)
end