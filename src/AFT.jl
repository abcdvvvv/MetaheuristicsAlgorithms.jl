"""
Braik, Malik, Mohammad Hashem Ryalat, and Hussein Al-Zoubi. 
"A novel meta-heuristic algorithm for solving numerical optimization problems: Ali Baba and the forty thieves." 
Neural Computing and Applications 34, no. 1 (2022): 409-455.
"""

function AFT(noThieves, itemax, lb, ub, dim, fobj)

    # Convergence curve
    ccurve = zeros(1, itemax)

    # Initialize positions of thieves
    # xth = zeros(noThieves, dim)
    # for i in 1:noThieves
    #     for j in 1:dim
    #         xth[i, j] = lb[j] - rand() * (lb[j] - ub[j])  # Position of the thieves in the space
    #     end
    # end
    xth = lb .- rand(noThieves, dim) .* (lb .- ub)  # Position of the thieves in the space

    # Evaluate fitness of the initial population
    fit = zeros(noThieves)
    for i in 1:noThieves
        fit[i] = fobj(xth[i, :])
    end

    # Initialization of AFT parameters
    fitness = fit
    sorted_indexes = sortperm(fit)
    sorted_thieves_fitness = fit[sorted_indexes]
    # sorted_thieves_fitness, sorted_indexes = sort(fit)

    ##println(sorted_indexes)

    Sorted_thieves = xth[sorted_indexes, :]  # Sort based on fitness
    gbest = Sorted_thieves[1, :]  # Global best position
    fit0 = sorted_thieves_fitness[1]

    best = xth  # Best positions of thieves
    xab = xth   # Position of Ali Baba

    # Start the AFT main loop
    for ite in 1:itemax
        Pp = 0.1 * log(2.75 * (ite / itemax)^0.1)  # Perception potential
        Td = 2 * exp(-2 * (ite / itemax)^2)       # Tracking distance

        # Generate random candidate followers (thieves)
        # a = ceil.((noThieves - 1) .* rand(noThieves))'
        a = Int.(ceil.((noThieves - 1) .* rand(noThieves))')
        # a = rand(1:ceil(Int, noThieves - 1), noThieves)

        # Generate new positions for the thieves
        for i in 1:noThieves
            if rand() >= 0.5  # Thieves know where to search
                if rand() > Pp
                    # println(i)
                    # println(gbest .+ (Td .* (best[i, :] .- xab[i, :]) .* rand() .+ Td .* (xab[i, :] .- best[a[i], :]) .* rand()) .* sign(rand() - 0.50))
                    xth[i, :] = gbest .+ (Td .* (best[i, :] .- xab[i, :]) .* rand() .+ Td .* (xab[i, :] .- best[a[i], :]) .* rand()) .* sign(rand() - 0.50)
                else
                    # for j in 1:dim
                    #     println(ub, "  "size(ub))
                    #     xth[i, j] = Td * ((ub[j] - lb[j]) * rand() + lb[j])  # Case 3
                    # end
                    xth[i, :] .= Td * ((ub .- lb) .* rand(dim) .+ lb)  # Case 3
                end
            else  # Thieves guess based on Marjaneh's astute plans
                for j in 1:dim
                    xth[i, j] = gbest[j] .- (Td .* (best[i, j] .- xab[i, j]) .* rand() .+ Td .* (xab[i, j] .- best[a[i], j]) .* rand()) .* sign(rand() - 0.50)
                end
            end
        end

        # Update global and best positions of the thieves and Ali Baba
        for i in 1:noThieves
            fit[i] = fobj(xth[i, :])

            # Boundary conditions
            if all(xth[i, :] .>= lb) && all(xth[i, :] .<= ub)
                xab[i, :] = xth[i, :]  # Update Ali Baba's position
                
                if fit[i] < fitness[i]
                    best[i, :] = xth[i, :]  # Update best position
                    fitness[i] = fit[i]     # Update fitness
                end

                # Update global best position
                if fitness[i] < fit0
                    fit0 = fitness[i]
                    gbest = best[i, :]
                end
            end
        end

        # Store the best found value in the convergence curve
        ccurve[ite] = fit0
    end

    # Final solution
    bestThieves = findall(x -> x == minimum(fitness), fitness)
    gbestSol = best[bestThieves[1], :]
    fitness = fobj(gbestSol)

    return fitness, gbest, ccurve
end