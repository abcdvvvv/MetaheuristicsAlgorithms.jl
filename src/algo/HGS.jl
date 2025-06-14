"""
Yang, Yutao, Huiling Chen, Ali Asghar Heidari, and Amir H. Gandomi. 
"Hunger games search: Visions, conception, implementation, deep analysis, perspectives, and towards performance shifts." 
Expert Systems with Applications 177 (2021): 114864.
"""
function HGS(N::Int, Max_iter::Int, lb, Ub, dim::Int, fobj)

    bestPositions = zeros(Float64, dim)
    tempPosition = zeros(Float64, N, dim)
    Destination_fitness = Inf
    Worstest_fitness = -Inf
    AllFitness = fill(Inf, N)
    VC1 = ones(N)

    weight3 = ones(Float64, N, dim)
    weight4 = ones(Float64, N, dim)

    X = initialization(N, dim, ub, lb)
    Convergence_curve = zeros(Float64, Max_iter)
    it = 1 
    hungry = zeros(Float64, N)
    count = 0

    # Main loop
    while it <= Max_iter
        VC2 = 0.03  
        sumHungry = 0  

        for i in 1:N
            X[i, :] = clamp.(X[i, :], lb, ub)
            AllFitness[i] = fobj(X[i, :])
        end

        IndexSorted = sortperm(AllFitness) # sortperm(AllFitness, rev=false)  # Get the indices that would sort the array
        AllFitnessSorted = AllFitness[IndexSorted]
        bestFitness = AllFitnessSorted[1]
        worstFitness = AllFitnessSorted[end]

        if bestFitness < Destination_fitness
            bestPositions = X[IndexSorted[1], :]
            Destination_fitness = bestFitness
            count = 0
        end

        if worstFitness > Worstest_fitness
            Worstest_fitness = worstFitness
        end

        for i in 1:N
            VC1[i] = sech(abs(AllFitness[i] - Destination_fitness))

            if Destination_fitness == AllFitness[i]
                hungry[i] = 0
                count += 1
                tempPosition[count, :] = X[i, :]
            else
                temprand = rand()
                c = (AllFitness[i] - Destination_fitness) / (Worstest_fitness - Destination_fitness) * temprand * 2 * (ub - lb)
                b = ifelse(c < 100, 100 * (1 + temprand), c)
                hungry[i] += maximum(b)
                sumHungry += hungry[i]
            end
        end

        for i in 1:N
            for j in 2:dim
                weight3[i, j] = (1 - exp(-abs(hungry[i] - sumHungry))) * rand() * 2
                if rand() < VC2
                    weight4[i, j] = hungry[i] * N / sumHungry * rand()
                else
                    weight4[i, j] = 1
                end
            end
        end

        shrink = 2 * (1 - it / Max_iter)  
        for i in 1:N
            if rand() < VC2
                X[i, :] *= 1 + randn()
            else
                A = rand(1:count)
                for j in 1:dim
                    r = rand()
                    vb = 2 * shrink * r - shrink  
                    if r > VC1[i]
                        X[i, j] = weight4[i, j] * tempPosition[A, j] + vb * weight3[i, j] * abs(tempPosition[A, j] - X[i, j])
                    else
                        X[i, j] = weight4[i, j] * tempPosition[A, j] - vb * weight3[i, j] * abs(tempPosition[A, j] - X[i, j])
                    end
                end
            end
        end

        Convergence_curve[it] = Destination_fitness
        it += 1
    end

    return Destination_fitness, bestPositions, Convergence_curve
end