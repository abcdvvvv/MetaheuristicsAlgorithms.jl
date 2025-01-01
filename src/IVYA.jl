"""
Ghasemi, Mojtaba, Mohsen Zare, Pavel Trojovský, Ravipudi Venkata Rao, Eva Trojovská, and Venkatachalam Kandasamy. 
"Optimization based on the smart behavior of plants with its engineering applications: Ivy algorithm." 
Knowledge-Based Systems 295 (2024): 111850.
"""

function IVYA(N, Max_iteration, lb, ub, dim, fobj)
    CostFunction = x -> fobj(x)

    VarMin = lb                          
    VarMax = ub                          

    MaxIt = Max_iteration                 
    nPop = N                              
    VarSize = (1, dim)                   

    Position = rand(nPop, dim) .* (VarMax .- VarMin) .+ VarMin
    GV = Position ./ (VarMax .- VarMin)  
    Cost = [CostFunction(Position[i, :]) for i in 1:nPop]  

    BestCosts = zeros(Float64, MaxIt)
    Convergence_curve = zeros(Float64, MaxIt)

    for it in 1:MaxIt
        Costs = Cost
        BestCost = minimum(Costs)
        WorstCost = maximum(Costs)

        newpop = []

        for i in 1:nPop
            ii = i + 1 > nPop ? 1 : i + 1
            beta_1 = 1 + rand() / 2  

            if Cost[i] < beta_1 * Cost[1]
                new_position = Position[i, :] .+ abs.(randn(dim)) .* (Position[ii, :] .- Position[i, :]) .+ randn(dim) .* (GV[i, :])
            else
                new_position = Position[1, :] .* (rand() .+ randn(dim) .* (GV[i, :]))
            end
            
            GV[i, :] = GV[i, :] .* ((rand()^2) * randn(dim))

            new_position = clamp.(new_position, VarMin, VarMax)

            new_GV = new_position ./ (VarMax .- VarMin)

            new_cost = CostFunction(vec(new_position))

            push!(newpop, (new_position, new_cost, new_GV))
        end

        for (new_position, new_cost, new_GV) in newpop
            Position = vcat(Position, new_position') 
            push!(Cost, new_cost)
            GV = vcat(GV, new_GV')
        end

        sorted_indices = sortperm(Cost)
        Position = Position[sorted_indices, :]
        Cost = Cost[sorted_indices]

        if length(Cost) > nPop
            Position = Position[1:nPop, :]
            Cost = Cost[1:nPop]
            GV = GV[1:nPop, :]
        end

        BestCosts[it] = Cost[1]
        Convergence_curve[it] = Cost[1]

    end

    Destination_fitness = Cost[1]
    Destination_position = Position[1, :]
    
    return Destination_fitness, Destination_position, Convergence_curve
end