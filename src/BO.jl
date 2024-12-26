"""
Das, Amit Kumar, and Dilip Kumar Pratihar. 
"Bonobo optimizer (BO): an intelligent heuristic with self-adjusting parameters over continuous spaces and its applications to engineering problems." 
Applied Intelligence 52, no. 3 (2022): 2942-2974.
"""

using Distributions

function BO(N, max_it, Var_min, Var_max, d, CostFunction)
    p_xgm_initial = 0.001 # Initial probability for extra-group mating (generally 1/d for higher dimensions)
    scab = 1.25 # %Sharing cofficient for alpha bonobo (Generally 1-2)
    scsb = 1.3 # Sharing coefficient for selected bonobo(Generally 1-2)
    rcpp = 0.0035 # Rate of change in phase probability (Generally 1e-3 to 1e-2)
    tsgs_factor_max = 0.05 # Max. value of temporary sub-group size factor
    if length(Var_min) == 1
        Var_min = Var_min * ones(d)
    end

    if length(Var_max) == 1
        Var_max = Var_max * ones(d)
    end


    # Initialization
    cost = zeros(N)
    bonobo = zeros(N, d)
    convergence = zeros(max_it)
    for i in 1:N
        # bonobo[i, :] = rand(Uniform.(Var_min, Var_max), d)  # Using Uniform distribution
        bonobo[i, :] = [rand(Uniform(Var_min[j], Var_max[j])) for j in 1:d]
        cost[i] = CostFunction(bonobo[i, :])
    end

    bestcost, ID = findmin(cost)
    alphabonobo = bonobo[ID, :]  # Best solution found in the population
    pbestcost = bestcost  # Initialization of previous best cost

    # Initialization of other parameters
    npc = 0  # Negative phase count
    ppc = 0  # Positive phase count
    p_xgm = p_xgm_initial  # Probability for extra-group mating
    tsgs_factor_initial = 0.5 * tsgs_factor_max  # Initial value for temporary sub-group size factor
    tsgs_factor = tsgs_factor_initial  # Temporary sub-group size factor
    p_p = 0.5  # Phase probability
    p_d = 0.5  # Directional probability
    it = 1

    # Main Loop of BO
    while it <= max_it
        tsgs_max = max(2, ceil(Int, N * tsgs_factor))  # Maximum size of the temporary sub-group
        
        for i in 1:N
            newbonobo = zeros(1, d)
            B = collect(1:N)
            deleteat!(B, i)  # Exclude the current index
            
            # Determining the actual size of the temporary sub-group
            tsg = rand(2:tsgs_max)  # Random size of the temporary sub-group
            
            # Selection of pth Bonobo using fission-fusion social strategy & flag value determination
            q = rand(B, tsg)  # Random sample from B
            temp_cost = cost[q]
            _, ID1 = findmin(temp_cost)
            p = q[ID1]
            
            if cost[i] < cost[p]
                p = q[rand(1:tsg)]  # Randomly select p
                flag = 1
            else
                flag = -1
            end
            
            # Creation of newbonobo
            if rand() <= p_p
                r1 = rand(1, d)  # Promiscuous or restrictive mating strategy
                newbonobo = bonobo[i, :] .+ scab * r1 .* (alphabonobo .- bonobo[i, :]) .+ flag * scsb * (1 .- r1) .* (bonobo[i, :] .- bonobo[p, :])
            else
                for j in 1:d
                    if rand() <= p_xgm
                        rand_var = rand()  # Extra group mating strategy
                        
                        if alphabonobo[1, j] >= bonobo[i, j]
                            if rand() <= p_d
                                beta1 = exp(rand_var^2 + rand_var - 2 / rand_var)
                                newbonobo[1, j] = bonobo[i, j] + beta1 * (Var_max[j] - bonobo[i, j])
                            else
                                beta2 = exp(-rand_var^2 + 2 * rand_var - 2 / rand_var)
                                newbonobo[1, j] = bonobo[i, j] - beta2 * (bonobo[i, j] - Var_min[j])
                            end
                        else
                            if rand() <= p_d
                                beta1 = exp(rand_var^2 + rand_var - 2 / rand_var)
                                newbonobo[1, j] = bonobo[i, j] - beta1 * (bonobo[i, j] - Var_min[j])
                            else
                                beta2 = exp(-rand_var^2 + 2 * rand_var - 2 / rand_var)
                                newbonobo[1, j] = bonobo[i, j] + beta2 * (Var_max[j] - bonobo[i, j])
                            end
                        end
                    else
                        if (flag == 1) || (rand() <= p_d)  # Consortship mating strategy
                            newbonobo[1, j] = bonobo[i, j] + flag * exp(-rand()) * (bonobo[i, j] - bonobo[p, j])
                        else
                            newbonobo[1, j] = bonobo[p, j]
                        end
                    end
                end
            end
            
            # Clipping
            # println("size(newbonobo, 1) ", size(newbonobo,1))
            # println("size(newbonobo, 2) ", size(newbonobo,2))
            # println("size(vec(newbonobo)) ", size(vec(newbonobo')))
            for j in 1:d
                if newbonobo[1, j] > Var_max[j]
                    newbonobo[1, j] = Var_max[j]
                elseif newbonobo[1, j] < Var_min[j]
                    newbonobo[1, j] = Var_min[j]
                end
            end
            
            newcost = CostFunction(newbonobo[1,:])  # New cost evaluation
            
            # New bonobo acceptance criteria
            if (newcost < cost[i]) || (rand() <= p_xgm)
                cost[i] = newcost
                bonobo[i, :] = newbonobo[1,:]  # Update bonobo
                
                if newcost < bestcost
                    bestcost = newcost
                    alphabonobo = newbonobo  # Update best solution
                end
            end
        end
        
        # Parameters updation
        if bestcost < pbestcost
            npc = 0  # Positive phase
            ppc += 1
            cp = min(0.5, ppc * rcpp)
            pbestcost = bestcost
            p_xgm = p_xgm_initial
            p_p = 0.5 + cp
            p_d = p_p
            tsgs_factor = min(tsgs_factor_max, tsgs_factor_initial + ppc * (rcpp^2))
        else
            npc += 1  # Negative phase
            ppc = 0
            cp = -(min(0.5, npc * rcpp))
            p_xgm = min(0.5, p_xgm_initial + npc * (rcpp^2))
            tsgs_factor = max(0, tsgs_factor_initial - npc * (rcpp^2))
            p_p = 0.5 + cp
            p_d = 0.5
        end
        
        convergence[it] = bestcost
        it += 1
    end

    # println("Best Cost: ", bestcost)
    # println("Best solution: ", alphabonobo)
    return bestcost, alphabonobo, convergence

end
