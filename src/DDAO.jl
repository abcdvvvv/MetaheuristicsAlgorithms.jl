"""
Ghafil, H. N., & Jármai, K. (2020). 
Dynamic differential annealed optimization: New metaheuristic optimization algorithm for engineering applications. 
Applied Soft Computing, 93, 106392.
"""
# using Random, Printf, Plots
using Random

function DDAO(Npop, MaxIt, L_limit, U_limit, Nvar, CostFunction)
    # CostFunction = DeJong
    # Nvar = 10              # Number of Variables
    VarLength = Nvar       # Solution vector size
    # L_limit = -10.0        # Lower limit of solution vector
    # U_limit = 10.0         # Upper limit of solution vector

    # DDAO Parameters
    # MaxIt = 500        # Maximum Number of Iterations
    MaxSubIt = 1000    # Maximum Number of Sub-iterations
    T0 = 2000.0        # Initial Temp.
    alpha = 0.995      # Temp. Reduction Rate
    # Npop = 3           # Population Size

    # Initialization
    empty_template = Dict("Phase" => [], "Cost" => Inf)
    pop = [deepcopy(empty_template) for _ in 1:Npop]

    # Initialize Best Solution
    BestSol = Dict("Phase" => [], "Cost" => Inf)


    # Initialize Population
    for i in 1:Npop
        # Initialize Position
        # pop[i]["Phase"] = rand(Limit(L_limit, U_limit), VarLength)
        pop[i]["Phase"] = L_limit .+ rand(VarLength) .* (U_limit - L_limit)
        # Evaluation
        pop[i]["Cost"] = CostFunction(pop[i]["Phase"])

        # Update Best Solution
        if pop[i]["Cost"] <= BestSol["Cost"]
            BestSol = deepcopy(pop[i])
        end
    end

    # Vector to Hold Best Costs
    BestCost = zeros(MaxIt)

    # Initialize Temp.
    T = T0

    # Main loop
    for t in 1:MaxIt
        newpop = [deepcopy(empty_template) for _ in 1:MaxSubIt]

        for subit in 1:MaxSubIt
            # Create and Evaluate New Solutions
            newpop[subit]["Phase"] = L_limit .+ rand(VarLength) .* (U_limit - L_limit)
            # newpop[subit]["Phase"] = rand(Limit(L_limit, U_limit), VarLength)
            # Set the new solution within the search space
            newpop[subit]["Phase"] = clamp.(newpop[subit]["Phase"], L_limit, U_limit)
            # Evaluate new solution
            newpop[subit]["Cost"] = CostFunction(newpop[subit]["Phase"])
        end

        # Sort Neighbors
        SortOrder = sortperm([newpop[subit]["Cost"] for subit in 1:MaxSubIt])
        newpop = newpop[SortOrder]
        bnew = newpop[1]
        kk = rand(1:Npop)
        bb = rand(1:Npop)

        # Forging parameter
        Mnew = deepcopy(empty_template)
        if isodd(t)
            Mnew["Phase"] = (pop[kk]["Phase"] - pop[bb]["Phase"]) + bnew["Phase"]
        else
            Mnew["Phase"] = (pop[kk]["Phase"] - pop[bb]["Phase"]) + bnew["Phase"] * rand()
        end

        # Set the new solution within the search space
        Mnew["Phase"] = clamp.(Mnew["Phase"], L_limit, U_limit)
        # Evaluate new solution
        Mnew["Cost"] = CostFunction(Mnew["Phase"])

        for i in 1:Npop
            if Mnew["Cost"] <= pop[i]["Cost"]
                pop[i] = deepcopy(Mnew)
            else
                DELTA = Mnew["Cost"] - pop[i]["Cost"]
                P = exp(-DELTA / T)
                if rand() <= P
                    pop[end] = deepcopy(Mnew)
                end
            end

            # Update Best Solution Ever Found
            if pop[i]["Cost"] <= BestSol["Cost"]
                BestSol = deepcopy(pop[i])
            end
        end

        # Store Best Cost Ever Found

        # Display Iteration Information
        # @printf("Iteration = %d: Best Cost = %f\n", t, BestCost[t])

        # Update Temp.
        T *= alpha
    end

    y = BestSol["Cost"]

    # Results
    # plot(1:MaxIt, BestCost, lw=2, ylabel="Best Cost", xlabel="Iteration", yscale=:log10, grid=true)
    return  BestSol["Cost"], BestSol["Phase"], BestCost
end