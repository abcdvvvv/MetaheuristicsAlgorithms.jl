"""
Ghasemi, Mojtaba, Mohammad-Amin Akbari, Changhyun Jun, Sayed M. Bateni, Mohsen Zare, Amir Zahedi, Hao-Ting Pai, 
Shahab S. Band, Massoud Moslehpour, and Kwok-Wing Chau. 
"Circulatory System Based Optimization (CSBO): an expert multilevel biologically inspired meta-heuristic algorithm." 
Engineering Applications of Computational Fluid Mechanics 16, no. 1 (2022): 1483-1525.
"""

struct Plant
    Position::Vector{Float64}
    Cost::Float64
end

function CSBO(nPop, MaxIt, VarMin, VarMax, nVar, CostFunction)
    # nVar = 30                 # Number of Decision Variables
    # VarMin = -100.0           # Decision Variables Lower Bound
    # VarMax = -VarMin          # Decision Variables Upper Bound
    VarSize = (1, nVar)       # Decision Variables Matrix Size

    # CSBO Parameters
    # MaxIt = 300_000           # Maximum Number of Iterations
    MaxFE = nPop * MaxIt
    # nPop = 45                 # Population Size
    NR = div(nPop, 3)         # Number of the weakes
    BestSolPosition = zeros(nVar)
    BestSolCost = Inf

    # Generate initial population
    pop = Vector{Plant}(undef, nPop)
    sigma1 = [rand(nVar) for _ in 1:nPop]

    for i in 1:nPop
        position = rand(VarMin:VarMax, nVar)
        # cost = ShiftedRosenbrock(position)
        cost = CostFunction(position)
        pop[i] = Plant(position, cost)

        if cost <= BestSolCost
            BestSolPosition = position
            BestSolCost = cost
        end
    end

    BestCost = Float64[BestSolCost]

    # CSBO Main Loop
    it = nPop

    # while it <= MaxIt 
    while it <= MaxFE 
        # Get Best and Worst Cost Values
        Costs = [p.Cost for p in pop]
        BestCostNow = minimum(Costs)
        WorstCost = maximum(Costs)

        for i in 1:nPop
            # Movement of blood mass in veins
            A = shuffle(1:nPop)
            A = filter(x -> x != i, A)
            a1, a2, a3 = A[1:3]
            
            AA1 = (pop[a2].Cost - pop[a3].Cost) / abs(pop[a3].Cost - pop[a2].Cost)
            AA2 = (pop[a1].Cost - pop[i].Cost) / abs(pop[a1].Cost - pop[i].Cost)
            AA1 = (pop[a2].Cost == pop[a3].Cost) ? 1.0 : AA1
            AA2 = (pop[a1].Cost == pop[i].Cost) ? 1.0 : AA2

            pos = pop[i].Position .+ AA2 * sigma1[i] .* (pop[i].Position - pop[a1].Position) .+
                AA1 * sigma1[i] .* (pop[a3].Position - pop[a2].Position)
            
            pos = clamp.(pos, VarMin, VarMax)
            # cost = ShiftedRosenbrock(pos)
            cost = CostFunction(pos)
            
            if cost < pop[i].Cost
                pop[i] = Plant(pos, cost)
                if cost <= BestSolCost
                    BestSolPosition = pos
                    BestSolCost = cost
                end
            end
            
            it += 1
            push!(BestCost, BestSolCost)
        end

        # Systematic Circulation
        for i in 1:(nPop - NR)
            ratioi = (pop[i].Cost - WorstCost) / (BestCostNow - WorstCost)
            sigma1[i] = fill(ratioi, nVar)
            A = shuffle(1:nPop)
            A = filter(x -> x != i, A)
            a1, a2, a3 = A[1:3]
            
            pos = pop[a1].Position .+ sigma1[i] .* (pop[a3].Position - pop[a2].Position)
            pos = clamp.(pos, VarMin, VarMax)
            
            if rand() > 0.9
                pos .= pop[i].Position
            end

            # cost = ShiftedRosenbrock(pos)
            cost = CostFunction(pos)
            if cost < pop[i].Cost
                pop[i] = Plant(pos, cost)
                if cost <= BestSolCost
                    BestSolPosition = pos
                    BestSolCost = cost
                end
            end
            
            it += 1
            push!(BestCost, BestSolCost)
        end

        # Pulmonary Circulation
        for i in (nPop - NR + 1):nPop
            pos = pop[i].Position .+ (randn() / it) .* randn(nVar)
            pos = clamp.(pos, VarMin, VarMax)
            # cost = ShiftedRosenbrock(pos)
            cost = CostFunction(pos)
            
            if cost < pop[i].Cost
                pop[i] = Plant(pos, cost)
                if cost <= BestSolCost
                    BestSolPosition = pos
                    BestSolCost = cost
                end
            end
            
            sigma1[i] = rand(nVar)
            it += 1
            push!(BestCost, BestSolCost)
        end

        # Display Iteration Information
        # @printf("Iteration %d: Best Cost = %f\n", it, BestSolCost)
    end
    return BestSolCost, BestSolPosition, BestCost
end

# using Random, Printf

# # Problem Definition
# function ShiftedRosenbrock(x)
#     # Define your Shifted Rosenbrock function here
#     return sum(100 * (x[2:end] .- x[1:end-1].^2).^2 .+ (x[1:end-1] .- 1).^2)
# end



# Initialization




# println("Best Solution Position: ", BestSolPosition)
# using Plots
# plot(log.(BestCost), linewidth=2, label="Best Cost")
# xlabel!("Iteration")
# ylabel!("Log(Best Cost)")
