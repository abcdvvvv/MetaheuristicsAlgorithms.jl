struct AEOResult <: OptimizationResult
    BestF
    BestX
    HisBestFit
end 

"""
    AEO(nPop, MaxIt, Low, Up, F_index)

    Artificial Ecosystem-based Optimization (AEO) algorithm implementation in Julia.

# Arguments:

- `nPop`: Number of individuals in the population.
- `MaxIt`: Maximum number of iterations.
- `Low`: Lower bounds for the search space.
- `Up`: Upper bounds for the search space.
- `F_index`: Function to evaluate the fitness of individuals.

# Returns:

- `AEOResult`: A struct containing:
  - `BestF`: The best fitness value found.
  - `BestX`: The position corresponding to the best fitness.
  - `HisBestFit`: A vector of best fitness values at each iteration.

# References: 

- Zhao, Weiguo, Liying Wang, and Zhenxing Zhang. "Artificial ecosystem-based optimization: a novel nature-inspired meta-heuristic algorithm." Neural Computing and Applications 32, no. 13 (2020): 9383-9425.
"""
function AEO(nPop::Int, MaxIt::Int, Low::Vector, Up::Vector, F_index)::AEOResult

    Dim = length(Low)
    PopPos = zeros(nPop, Dim)
    PopFit = zeros(nPop)

    newPopFit = zeros(nPop, Dim)
    for i in 1:nPop
        PopPos[i, :] = rand(Dim) .* (Up - Low) .+ Low
        PopFit[i] = F_index(PopPos[i, :])
    end

    indF = sortperm(PopFit, rev=true)
    PopPos = PopPos[indF, :]
    PopFit = PopFit[indF]
    BestF = PopFit[end]
    BestX = PopPos[end, :]

    HisBestFit = zeros(MaxIt)
    Matr = [1, Dim]

    for It in 1:MaxIt
        r1 = rand()
        a = (1 - It / MaxIt) * r1
        xrand = rand(Dim) .* (Up - Low) .+ Low
        newPopPos = zeros(nPop, Dim) 
        newPopPos[1, :] = (1 - a) * PopPos[end, :] .+ a * xrand  # equation (1)

        for i in 3:nPop
            u = randn(Dim)
            v = randn(Dim)
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

        for i in 1:nPop
            newPopPos[i, :] = max.(min.(newPopPos[i, :], Up), Low)
            newPopFit[i] = F_index(newPopPos[i, :])
            if newPopFit[i] < PopFit[i]
                PopFit[i] = newPopFit[i]
                PopPos[i, :] = newPopPos[i, :]
            end
        end

        indOne = argmin(PopFit)
        for i in 1:nPop
            r3 = rand()
            Ind = rand(1:2)  
            newPopPos[i, :] = PopPos[indOne, :] .+ 3 * randn(Matr[Ind]) .* ((r3 * rand(1:2) .- 1) .* PopPos[indOne, :] .- (2 * r3 - 1) .* PopPos[i, :])  # equation (9)
        end

        for i in 1:nPop
            newPopPos[i, :] = max.(min.(newPopPos[i, :], Up), Low)
            newPopFit[i] = F_index(newPopPos[i, :])
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

        HisBestFit[It] = BestF
    end

    #return BestF, BestX, HisBestFit
    return AEOResult(
        BestF, 
        BestX, 
        HisBestFit)
end