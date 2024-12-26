"""
Agushaka, Jeffrey O., Absalom E. Ezugwu, and Laith Abualigah. 
"Dwarf mongoose optimization algorithm." 
Computer methods in applied mechanics and engineering 391 (2022): 114570.
"""
mutable struct Mongoose
    Position::Vector{Float64}
    Cost::Float64
end


function DMOA(nPop, MaxIt, VarMin, VarMax, nVar, F_obj)
    VarSize = nVar  # Number of Decision Variables

    nBabysitter = 3  # Number of babysitters
    nAlphaGroup = nPop - nBabysitter  # Number of Alpha group
    nScout = nAlphaGroup  # Number of Scouts

    L = round(Int, 0.6 * nVar * nBabysitter)  # Babysitter Exchange Parameter
    peep = 2.0  # Alpha female vocalization

    # Initialize Population Array
    pop = [Mongoose(zeros(VarSize), Inf) for _ in 1:nAlphaGroup]

    # Initialize Best Solution Ever Found
    BestSol = Mongoose(zeros(VarSize), Inf)
    tau = Inf
    sm = fill(Inf, nAlphaGroup)  # Sleeping mould

    # Create Initial Population
    for i in 1:nAlphaGroup
        pop[i].Position = rand(VarSize) .* (VarMax .- VarMin) .+ VarMin
        pop[i].Cost = F_obj(pop[i].Position)
        if pop[i].Cost <= BestSol.Cost
            BestSol = pop[i]
        end
    end

    # Abandonment Counter
    C = zeros(Int, nAlphaGroup)
    CF = (1 - 1 / MaxIt)^(2 * 1 / MaxIt)

    # Array to Hold Best Cost Values
    BestCost = zeros(Float64, MaxIt)

    # DMOA Main Loop
    for it in 1:MaxIt
        # Alpha group
        F = zeros(Float64, nAlphaGroup)
        MeanCost = mean([pop[i].Cost for i in 1:nAlphaGroup])
        
        for i in 1:nAlphaGroup
            # Calculate Fitness Values and Selection of Alpha
            F[i] = exp(-pop[i].Cost / MeanCost)  # Convert Cost to Fitness
        end
        
        P = F / sum(F)  # Normalized fitness
        
        # Foraging led by Alpha female
        for m in 1:nAlphaGroup
            # Select Alpha female
            i = RouletteWheelSelection(P)
            
            # Choose k randomly, not equal to Alpha
            K = setdiff(1:nAlphaGroup, i)
            k = K[rand(1:length(K))]
            
            # Define Vocalization Coeff.
            phi = (peep / 2) * rand(VarSize) .* (2 .* rand(VarSize) .- 1)
            
            # New Mongoose Position
            newpop = Mongoose(zeros(VarSize), Inf)
            newpop.Position = pop[i].Position .+ phi .* (pop[i].Position .- pop[k].Position)
            
            # Evaluation
            newpop.Cost = F_obj(newpop.Position)
            
            # Comparison
            if newpop.Cost <= pop[i].Cost
                pop[i] = newpop
            else
                C[i] += 1
            end
        end   

        # Scout group
        for i in 1:nScout
            # Choose k randomly, not equal to i
            K = setdiff(1:nAlphaGroup, i)
            k = K[rand(1:length(K))]
            
            # Define Vocalization Coeff.
            phi = (peep / 2) * rand(VarSize) .* (2 .* rand(VarSize) .- 1)
            
            # New Mongoose Position
            newpop = Mongoose(zeros(VarSize), Inf)
            newpop.Position = pop[i].Position .+ phi .* (pop[i].Position .- pop[k].Position)
            
            # Evaluation
            newpop.Cost = F_obj(newpop.Position)
            
            # Sleeping mould
            sm[i] = (newpop.Cost - pop[i].Cost) / max(newpop.Cost, pop[i].Cost)
            
            # Comparison
            if newpop.Cost <= pop[i].Cost
                pop[i] = newpop
            else
                C[i] += 1
            end
        end    

        # Babysitters
        for i in 1:nBabysitter
            if C[i] >= L
                pop[i].Position = rand(VarSize) .* (VarMax .- VarMin) .+ VarMin
                pop[i].Cost = F_obj(pop[i].Position)
                C[i] = 0
            end
        end    

        # Update Best Solution Ever Found
        for i in 1:nAlphaGroup
            if pop[i].Cost <= BestSol.Cost
                BestSol = pop[i]
            end
        end    
        
        # Next Mongoose Position
        newtau = mean(sm)
        for i in 1:nScout
            M = (pop[i].Position .* sm[i]) ./ pop[i].Position
            if newtau > tau
                newpop = Mongoose(zeros(VarSize), Inf)
                newpop.Position = pop[i].Position .- CF * rand(VarSize) .* (pop[i].Position .- M)
            else
                newpop = Mongoose(zeros(VarSize), Inf)
                newpop.Position = pop[i].Position .+ CF * rand(VarSize) .* (pop[i].Position .- M)
            end
            tau = newtau
        end
       
        # Update Best Solution Ever Found
        for i in 1:nAlphaGroup
            if pop[i].Cost <= BestSol.Cost
                BestSol = pop[i]
            end
        end
    
        # Store Best Cost Ever Found
        BestCost[it] = BestSol.Cost
        
        # Display Iteration Information
        # println("Iteration $it: Best Cost = $(BestCost[it])")
    end

    return BestSol.Cost, BestSol.Position, BestCost
end