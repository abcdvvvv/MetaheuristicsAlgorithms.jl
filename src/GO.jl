"""
Zhang, Qingke, Hao Gao, Zhi-Hui Zhan, Junqing Li, and Huaxiang Zhang. 
"Growth Optimizer: A powerful metaheuristic algorithm for solving continuous and discrete global optimization problems." 
Knowledge-Based Systems 261 (2023): 110206.
"""
function GO(popsize, MaxIter, xmin, xmax, dimension, Func) #(popsize, dimension, xmax, xmin, MaxFEs, Func, FuncId)
    # Parameter setting
    FEs = 0
    P1 = 5
    P2 = 0.001
    P3 = 0.3
    MaxFEs = MaxIter * popsize
    
    # Initialization
    x = xmin .+ rand(popsize, dimension) .* (xmax - xmin)
    gbestX = zeros(dimension)
    gbestfitness = Inf
    fitness = zeros(popsize)
    gbesthistory = []

    for i in 1:popsize
        # fitness[i] = Func(x[i, :], FuncId)
        fitness[i] = Func(x[i, :])
        FEs += 1
        if gbestfitness > fitness[i]
            gbestfitness = fitness[i]
            gbestX = x[i, :]
        end
        push!(gbesthistory, gbestfitness)
        # println("FEs: $FEs, fitness error: $(gbestfitness)")
    end

    while true
        ind = sortperm(fitness)
        Best_X = x[ind[1], :]

        # Learning phase
        for i in 1:popsize
            Worst_X = x[ind[rand(popsize-P1+1:popsize)], :]
            Better_X = x[ind[rand(2:P1)], :]
            # random = setdiff(1:popsize, [i])
            random = selectID(popsize, i, 2)
            L1, L2 = random[rand(1:length(random), 2)]

            Gap1 = Best_X - Better_X
            Gap2 = Best_X - Worst_X
            Gap3 = Better_X - Worst_X
            Gap4 = x[L1, :] - x[L2, :]

            Distance1 = norm(Gap1)
            Distance2 = norm(Gap2)
            Distance3 = norm(Gap3)
            Distance4 = norm(Gap4)
            SumDistance = Distance1 + Distance2 + Distance3 + Distance4

            LF1 = Distance1 / SumDistance
            LF2 = Distance2 / SumDistance
            LF3 = Distance3 / SumDistance
            LF4 = Distance4 / SumDistance
            SF = fitness[i] / maximum(fitness)

            KA1 = LF1 * SF * Gap1
            KA2 = LF2 * SF * Gap2
            KA3 = LF3 * SF * Gap3
            KA4 = LF4 * SF * Gap4

            newx = x[i, :] + KA1 + KA2 + KA3 + KA4
            newx = clamp.(newx, xmin, xmax)

            # newfitness = Func(newx, FuncId)
            newfitness = Func(newx)
            FEs += 1

            # Update
            if fitness[i] > newfitness
                fitness[i] = newfitness
                x[i, :] = newx
            elseif rand() < P2 && ind[i] != 1
                fitness[i] = newfitness
                x[i, :] = newx
            end

            if gbestfitness > fitness[i]
                gbestfitness = fitness[i]
                gbestX = x[i, :]
            end

            push!(gbesthistory, gbestfitness)
            # println("FEs: $FEs, fitness error: $(gbestfitness)")
        end

        if FEs >= MaxFEs
            break
        end

        # Reflection phase
        for i in 1:popsize
            newx = x[i, :]

            for j in 1:dimension
                if rand() < P3
                    R = x[ind[rand(1:P1)], :]
                    newx[j] = x[i, j] + (R[j] - x[i, j]) * rand()

                    AF = 0.01 + (0.1 - 0.01) * (1 - FEs / MaxFEs)
                    if rand() < AF
                        newx[j] = xmin + (xmax - xmin) * rand()
                    end
                end
            end

            newx = clamp.(newx, xmin, xmax)
            # newfitness = Func(newx, FuncId)
            newfitness = Func(newx)
            FEs += 1

            if fitness[i] > newfitness
                fitness[i] = newfitness
                x[i, :] = newx
            elseif rand() < P2 && ind[i] != 1
                fitness[i] = newfitness
                x[i, :] = newx
            end

            if gbestfitness > fitness[i]
                gbestfitness = fitness[i]
                gbestX = x[i, :]
            end

            push!(gbesthistory, gbestfitness)
            # println("FEs: $FEs, fitness error: $(gbestfitness)")
        end

        if FEs >= MaxFEs
            break
        end
    end

    # Adjust gbesthistory size if necessary
    if FEs < MaxFEs
        append!(gbesthistory, fill(gbestfitness, MaxFEs - FEs))
    elseif FEs > MaxFEs
        gbesthistory = gbesthistory[1:MaxFEs]
    end

    return gbestfitness, gbestX, gbesthistory
end

function selectID(popsize, i, k)
    # Generate k random integer values within [1, popsize] that do not include i and do not repeat each other
    if k <= popsize
        vecc = vcat(1:i-1, i+1:popsize)  # Create vector excluding i
        r = zeros(Int, k)  # Initialize array to store results
        for kkk in 1:k
            n = popsize - kkk
            t = rand(1:n)  # Generate random index
            r[kkk] = vecc[t]
            vecc = deleteat!(vecc, t)  # Remove selected element from vecc
        end
        return r
    else
        error("k must be less than or equal to popsize")
    end
end
