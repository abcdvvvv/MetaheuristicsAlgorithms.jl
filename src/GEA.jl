"""
Ghasemi, Mojtaba, Mohsen Zare, Amir Zahedi, Mohammad-Amin Akbari, Seyedali Mirjalili, and Laith Abualigah. 
"Geyser inspired algorithm: a new geological-inspired meta-heuristic for real-parameter and constrained engineering optimization." 
Journal of Bionic Engineering 21, no. 1 (2024): 374-408.
"""

using Random

function GEA(nPop, MaxIt, VarMin, VarMax, nVar, CostFunction)
    # Problem Definition
    # VarMin = -100  # Decision Variables Lower Bound
    # VarMax = -VarMin  # Decision Variables Upper Bound
    # nVar = 30  # Number of Decision Variables
    VarSize = (1, nVar)  # Decision Variables Matrix Size
    # Nc = 40  # Number of Channels
    Nc = Int(floor(nPop / 3))  # Number of Channels
    # FEs = 300000  # Maximum Number of Function Evaluations
    FEs = nPop * MaxIt # Maximum Number of Function Evaluations
    # nPop = 60  # Number of Geysers (Swarm Size)

    # Positions = [rand(VarSize) * (VarMax - VarMin) .+ VarMin for _ in 1:nPop]  # Initial Positions
    Positions = rand(nPop, nVar) * (VarMax - VarMin) .+ VarMin
    # Positions = initialization(nPop, nVar, VarMin, VarMax)
    # println("size(Positions[1,:]) ", size(Positions[1,:]))
    # Costs = [CostFunction(pos[:]) for pos in Positions]  # Initial Costs
    # Costs = [CostFunction(vec(pos)) for pos in Positions]
    Costs = zeros(Float64, nPop)
    # for i in 1:length(Positions)
    for i in 1:nPop
        Costs[i] = CostFunction(Positions[i,:])  # Flatten each Position and calculate the cost
    end
    BestCost = minimum(Costs)  # Initialize Best Solution Cost
    BestPosition = Positions[argmin(Costs)]  # Initialize Best Solution Position
    BestCosts = zeros(FEs)  # Array to track best cost in each iteration

    it = nPop

    while it < FEs
        # println("size(Positions) $(size(Positions))")
        # println("it $(it)")
        for i in 1:nPop
            # Implementing Eq(3)
            D1 = [sum(Positions[i,:] .* Positions[j,:]) / sqrt(sum(Positions[i,:] .^ 2) * sum(Positions[j,:] .^ 2)) for j in 1:nPop]
            D1[i] = Inf  # Exclude self-comparison
            S1 = minimum(D1)
            j1 = argmin(D1)

            # Calculating the pressure value in Eq(6)
            CS = sort(Costs)
            dif = CS[end] - CS[1]  # fmax - fmin Eq(6)
            G = (Costs[i] - CS[1]) / dif  # f(i) - fmin Eq(6)
            G = dif == 0 ? 0 : G  # Avoid division by zero

            P_i = sqrt(((G^(2 / it)) - (G^((it + 1) / it))) * (it / (it - 1)))  # Pi Eq(6)

            # Search for channels using Roulette wheel selection
            # println("Nc $Nc")
            ImpFitness = sum(Costs[1:Nc])
            p = [Costs[ii] / ImpFitness for ii in 1:Nc]
            ImpFitness = ImpFitness == 0 ? 1e-320 : ImpFitness

            i1 = sum(p) == 0 ? 1 : RouletteWheelSelection(p)

            flag = 0
            # Implementing Eq(5)
            # newsol_Position = Positions[j1] .+ rand(VarSize) .* (Positions[i1] - Positions[j1]) .+ rand(VarSize) .* (Positions[i1] - Positions[i])
            # println("j1 = $j1, i1 = $i1, i = $i")
            # println("$size(Positions) $(size(Positions))")
            ####
            # println("Size of Positions[j1, :]: ", size(Positions[j1, :]))
            # println("Size of Positions[i1, :] - Positions[j1, :]: ", size(Positions[i1, :] - Positions[j1, :]))
            # println("Size of Positions[i1, :] - Positions[i, :]: ", size(Positions[i1, :] - Positions[i, :]))
            # println("Size of rand(nVar): ", size(rand(nVar)))
            # println("Size of rand(nVar): ", size(rand(nVar)))
            ####
            newsol_Position = Positions[j1,:] .+ rand(nVar) .* (Positions[i1,:] - Positions[j1,:]) .+ rand(nVar) .* (Positions[i1,:] - Positions[i,:])
            newsol_Position = max.(newsol_Position, VarMin)
            newsol_Position = min.(newsol_Position, VarMax)
            # #
            # println("1 - Size(newsol_Position) ", size(newsol_Position))
            # println("Size(Positions[j1,:]) ", size(Positions[j1,:]))
            newsol_Cost = CostFunction(newsol_Position)
            it += 1


            ####################
            ####################
            ####################
            if it > FEs  
                break
            end
            ####################
            ####################
            ####################

            #
            # println("It $it Positions[i,:] $(size(Positions[i,:])) newsol_Position $(size(newsol_Position))", )

            if newsol_Cost <= Costs[i]
                # println("It $it Positions[i,:] $(size(Positions[i,:])) newsol_Position $(size(newsol_Position))", )
                Positions[i,:] = newsol_Position
                Costs[i] = newsol_Cost
            end
            if newsol_Cost <= BestCost
                BestCost = newsol_Cost
                # BestPosition .= newsol_Position
                BestPosition = newsol_Position
            end
            BestCosts[it] = BestCost
            # it += 1

            if flag == 0
                i2 = RouletteWheelSelection(1 .- p)  # Implementing Roulette wheel selection by Eq(7)
                # Implementing Eq(8)
                # newsol_Position = Positions[i2] .+ rand() * (P_i - rand()) .* rand(VarSize) * (VarMax - VarMin) .+ VarMin
                newsol_Position = Positions[i2,:] .+ rand() * (P_i - rand()) .* rand(nVar) * (VarMax - VarMin) .+ VarMin
                newsol_Position = max.(newsol_Position, VarMin)
                newsol_Position = min.(newsol_Position, VarMax)
                newsol_Cost = CostFunction(newsol_Position)
                # println("2 - Size(newsol_Position) ", size(newsol_Position))
                it += 1

                if newsol_Cost <= Costs[i]
                    Positions[i,:] = newsol_Position
                    Costs[i] = newsol_Cost
                end
                if newsol_Cost <= BestCost
                    BestCost = newsol_Cost
                    # BestPosition .= newsol_Position
                    BestPosition = newsol_Position
                end
                BestCosts[it] = BestCost
                # it += 1
                # println("i $i Iteration $it: Best Cost = $BestCost")
            end
        end

        # Sorting based on Costs
        sorted_indices = sortperm(Costs)
        Positions = Positions[sorted_indices,:]
        Costs = Costs[sorted_indices]

        # println("Iteration $it: Best Cost = $BestCost")

        # if it % 100 == 0
        #     println("Iteration $it: Best Cost = $BestCost")
        # end
    end
    # println("Iteration $it: Best Cost = $BestCost")
    return BestCost, BestPosition, BestCosts
end

function RouletteWheelSelection(p)
    c = cumsum(p)           # Compute the cumulative sum of probabilities
    r = rand()              # Generate a random number between 0 and 1
    i = findfirst(x -> x >= r, c)  # Find the first index where cumulative probability is greater than or equal to r
    return i                # Return the index
end
