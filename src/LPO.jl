"""
Ghasemi, Mojtaba, Mohsen Zare, Amir Zahedi, Pavel Trojovský, Laith Abualigah, and Eva Trojovská. 
"Optimization based on performance of lungs in body: Lungs performance-based optimization (LPO)." 
Computer Methods in Applied Mechanics and Engineering 419 (2024): 116582.
"""

function LPO(nPop, MaxIt, VarMin, VarMax, nVar, fhd)
    Positions = initialization(nPop, nVar, VarMin, VarMax)  
    Fitness = fill(Inf, nPop)                                 
    BestPosition = zeros(nVar)                                
    BestCost = Inf                                            
    Convergence = zeros(Float64, MaxIt)
    Delta = rand(nPop) .* 2π
    sigma1 = [rand(1, nVar) for _ in 1:nPop]

    # Evaluate initial population
    for i in 1:nPop
        Fitness[i] = fhd(Positions[i, :])  # Call cost function
        if Fitness[i] <= BestCost
            BestPosition = Positions[i, :]
            BestCost = Fitness[i]
        end
    end

    it = 0
    Pos1 = zeros(nVar)
    while it <= MaxIt
        BestCost = minimum(Fitness)
        WorstCost = maximum(Fitness)

        for i in 1:nPop
            for jj in 1:5
                R = Fitness[i]
                C = (R / 2) * sin(Delta[i])

                newsol = deepcopy(Positions[i, :])
                newsol2 = deepcopy(Positions[i, :])

                if jj == 1
                    newsol .= Positions[i, :] .+ ((R^2 + (1 / (2π * nVar * R * C)^2))^-0.5) * sin(2π * nVar * it) * sin((2π * nVar * it) + Delta[i]) .* Positions[i, :]
                else
                    newsol .= Positions[i, :] .+ ((R^2 + (1 / (2π * nVar * R * C)^2))^-0.5) * sin(2π * nVar * it) * sin((2π * nVar * it) + Delta[i]) .* Pos1
                end

                perm = randperm(nPop)
                a1, a2, a3 = perm[1], perm[2], perm[3]

                aa1 = (Fitness[a2] - Fitness[a3]) / abs(Fitness[a3] - Fitness[a2])
                aa1 = iszero(Fitness[a2] - Fitness[a3]) ? 1.0 : aa1

                aa2 = (Fitness[a1] - Fitness[i]) / abs(Fitness[a1] - Fitness[i])
                aa2 = iszero(Fitness[a1] - Fitness[i]) ? 1.0 : aa2

                newsol = newsol .+ aa2 * sigma1[i] .* (newsol .- Positions[a1, :]) + aa1 * sigma1[i] .* (Positions[a3, :] .- Positions[a2, :])
                newsol2 = Positions[a1, :] .+ sigma1[i] .* (Positions[a3, :] .- Positions[a2, :])

                for j in 1:nVar
                    Pos1[j] = rand() / jj > rand() ? newsol2[j] : newsol[j]
                end

                Pos1 = clamp.(Pos1, VarMin, VarMax)
                newsol = Pos1
                Delta[i] = atan(1 / (2π * nVar * R * C))

                newCost = fhd(newsol)
                if newCost < Fitness[i]
                    Positions[i, :] = newsol
                    Fitness[i] = newCost
                    if newCost <= BestCost
                        BestPosition = newsol
                        BestCost = newCost
                    end
                end
                sigma1[i] = rand(1, nVar)
            end
        end

        it += 1
    end

    Result = Convergence

    return BestCost, BestPosition, Convergence
end

              