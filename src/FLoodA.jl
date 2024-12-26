""" 
Ghasemi, M., Golalipour, K., Zare, M., Mirjalili, S., Trojovský, P., Abualigah, L. and Hemmati, R., 2024. 
Flood algorithm (FLA): an efficient inspired meta-heuristic for engineering optimization. 
The Journal of Supercomputing, 80(15), pp.22913-23017.
"""

function FLoodA(nPop, MaxIt, lb, ub, nVar, fobj)

    # Define Cost Function
    CostFunction(x) = fobj(x)

    # Initialization
    VarMin = lb           # Lower Bound of Variables
    VarMax = ub           # Upper Bound of Variables
    VarSize = (1, nVar)   # Variable size for uniform distribution
    NFEs = 0              # Number of fitness evaluations
    Ne = 5                # Number of weak population that will be replaced
    Convergence_curve = []  # Convergence curve for plotting

    # Initialize Population and Values

    # Position = [rand(VarSize)[1, :] * (VarMax - VarMin) + VarMin for _ in 1:nPop]
    # Cost = [CostFunction(pos) for pos in Position]
    # Val = [zeros(VarSize) for _ in 1:nPop]
        
    Position = zeros(nPop, nVar)  # Matrix to hold positions for each individual
    println("1 - size(Position) ", size(Position))
    Cost = zeros(nPop)            # Vector to hold cost for each individual
    Val = zeros(nPop, nVar)       # Matrix to hold Val for each individual
    
    # Populate Position, Cost, and Val arrays
    Position =initialization(nPop, nVar, VarMin, VarMax) # rand(VarSize)[1, :] * (VarMax - VarMin) + VarMin
    for i in 1:nPop
        # Position[i, :] = rand(VarSize)[1, :] * (VarMax - VarMin) + VarMin
        Cost[i] = CostFunction(Position[i, :])
        # Val[i, :] = zeros(VarSize)
    end

    # Best Solution Initialization
    BestSolCost = Inf
    BestSolPosition = nothing

    # Create Initial Population
    for i in 1:nPop
        if Cost[i] <= BestSolCost
            BestSolCost = Cost[i]
            BestSolPosition = Position[i,:]
        end
    end

    # Main Loop
    it = 0
    time1 = 0
    while NFEs < nPop * MaxIt
        it += 1
        PK = ((((MaxIt * (it^2) + 1)^0.5 + (1 / ((MaxIt / 4) * it)) * log(((MaxIt * (it^2) + 1)^0.5 + (MaxIt / 4) * it))))^(-2/3)) * (1.2 / it)

        for i in 1:nPop
            # Sort Costs for Phase I
            sorted_costs = sort(Cost)
            Pe_i = ((Cost[i] - sorted_costs[1]) / (sorted_costs[end] - sorted_costs[1]))^2
            A = filter(x -> x != i, collect(1:nPop))
            a = A[rand(1:end)]

            # Compute new solution Position
            if rand() > (rand() + Pe_i)
                # Val[i] = ((PK^randn()) / it) * (rand(VarSize)[1, :] * (VarMax - VarMin) + VarMin)
                Val[i,:] = ((PK^randn()) / it) * (rand(nVar) .* (VarMax - VarMin) .+ VarMin)
                new_position = Position[i,:] .+ Val[i,:]
                # new_position = vec(new_position)
            else
                # new_position = BestSolPosition .+ rand(VarSize)[1, :] .* (Position[a] .- Position[i])
                new_position = BestSolPosition .+ rand(nVar) .* (Position[a, :] .- Position[i, :])
                # new_position = vec(new_position)
            end

            # Clip new position to bounds
            new_position = clamp.(new_position, VarMin, VarMax)

            # Evaluate New Solution
            new_cost = CostFunction(new_position)
            NFEs += 1

            # Replace if new solution is better
            if new_cost <= Cost[i]
                Position[i,:] = new_position
                Cost[i] = new_cost
            end

            # Update Best Solution if improved
            if new_cost <= BestSolCost
                BestSolCost = new_cost
                BestSolPosition = new_position
            end
        end

        # Implementing Phase II
        Pt = abs(sin(rand() / it))
        if rand() < Pt
            # Sort and Keep Best Individuals
            sorted_indices = sortperm(Cost)
            Position = Position[sorted_indices[1:nPop - Ne],:]
            Cost = Cost[sorted_indices[1:nPop - Ne]]
            Val = Val[sorted_indices[1:nPop - Ne],:]

            # Generate New Individuals
            for _ in 1:Ne
                # new_position = BestSolPosition .+ rand() .* (rand(VarSize)[1, :] * (VarMax - VarMin) + VarMin)
                new_position = BestSolPosition .+ rand() .* (rand(nVar) .* (VarMax - VarMin) .+ VarMin)
                new_cost = CostFunction(new_position)
                NFEs += 1

                # Add new individual
                new_position = vec(new_position)
                Position = vcat(Position, new_position')
                push!(Cost, new_cost)
                # push!(Val, zeros(VarSize))
                Val = vcat(Val, zeros(1, nVar))

                # Update Best Solution if improved
                if new_cost <= BestSolCost
                    BestSolCost = new_cost
                    BestSolPosition = new_position
                end
            end
        end

        # Store Best Cost for Convergence Curve
        push!(Convergence_curve, BestSolCost)

        # println("Iteration $NFEs: Best Cost = $(BestSolCost)")
    end

    # Final Results
    Destination_fitness = BestSolCost
    Destination_position = BestSolPosition

    return Destination_fitness, Destination_position, Convergence_curve
end
