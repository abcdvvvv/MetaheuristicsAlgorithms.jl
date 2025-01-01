""" 
Ghasemi, M., Golalipour, K., Zare, M., Mirjalili, S., Trojovský, P., Abualigah, L. and Hemmati, R., 2024. 
Flood algorithm (FLA): an efficient inspired meta-heuristic for engineering optimization. 
The Journal of Supercomputing, 80(15), pp.22913-23017.
"""

function FLoodA(nPop, MaxIt, lb, ub, nVar, fobj)

    CostFunction(x) = fobj(x)

    
    VarMin = lb           
    VarMax = ub           
    VarSize = (1, nVar)   
    NFEs = 0              
    Ne = 5                
    Convergence_curve = []  
   
    Position = zeros(nPop, nVar)  
    println("1 - size(Position) ", size(Position))
    Cost = zeros(nPop)            
    Val = zeros(nPop, nVar)       
    
    Position =initialization(nPop, nVar, VarMin, VarMax) 
    for i in 1:nPop
        Cost[i] = CostFunction(Position[i, :])
    end


    BestSolCost = Inf
    BestSolPosition = nothing

    for i in 1:nPop
        if Cost[i] <= BestSolCost
            BestSolCost = Cost[i]
            BestSolPosition = Position[i,:]
        end
    end

    it = 0
    time1 = 0
    while NFEs < nPop * MaxIt
        it += 1
        PK = ((((MaxIt * (it^2) + 1)^0.5 + (1 / ((MaxIt / 4) * it)) * log(((MaxIt * (it^2) + 1)^0.5 + (MaxIt / 4) * it))))^(-2/3)) * (1.2 / it)

        for i in 1:nPop
            sorted_costs = sort(Cost)
            Pe_i = ((Cost[i] - sorted_costs[1]) / (sorted_costs[end] - sorted_costs[1]))^2
            A = filter(x -> x != i, collect(1:nPop))
            a = A[rand(1:end)]

            if rand() > (rand() + Pe_i)
                Val[i,:] = ((PK^randn()) / it) * (rand(nVar) .* (VarMax - VarMin) .+ VarMin)
                new_position = Position[i,:] .+ Val[i,:]
            else
                new_position = BestSolPosition .+ rand(nVar) .* (Position[a, :] .- Position[i, :])
            end

            new_position = clamp.(new_position, VarMin, VarMax)

            new_cost = CostFunction(new_position)
            NFEs += 1

            if new_cost <= Cost[i]
                Position[i,:] = new_position
                Cost[i] = new_cost
            end

            if new_cost <= BestSolCost
                BestSolCost = new_cost
                BestSolPosition = new_position
            end
        end

        Pt = abs(sin(rand() / it))
        if rand() < Pt
            sorted_indices = sortperm(Cost)
            Position = Position[sorted_indices[1:nPop - Ne],:]
            Cost = Cost[sorted_indices[1:nPop - Ne]]
            Val = Val[sorted_indices[1:nPop - Ne],:]

            for _ in 1:Ne
                new_position = BestSolPosition .+ rand() .* (rand(nVar) .* (VarMax - VarMin) .+ VarMin)
                new_cost = CostFunction(new_position)
                NFEs += 1

                new_position = vec(new_position)
                Position = vcat(Position, new_position')
                push!(Cost, new_cost)
                Val = vcat(Val, zeros(1, nVar))

                if new_cost <= BestSolCost
                    BestSolCost = new_cost
                    BestSolPosition = new_position
                end
            end
        end

        push!(Convergence_curve, BestSolCost)
    end

    Destination_fitness = BestSolCost
    Destination_position = BestSolPosition

    return Destination_fitness, Destination_position, Convergence_curve
end
