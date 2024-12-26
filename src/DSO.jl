"""
Oladejo, Sunday O., Stephen O. Ekwe, Lateef A. Akinyemi, and Seyedali A. Mirjalili. 
"The deep sleep optimiser: A human-based metaheuristic approach." 
IEEE Access (2023).
"""

function DSO(search_agent, run, LB, UB, dim, ObjFun)#(ObjFun, LB, UB, dim, search_agent, run)
    # Problem Parameters
    prob = ObjFun                # objective function
    nVar = dim                   # dimension of problem
    lb = LB                      # lower bound
    ub = UB                      # upper bound
    nPop = search_agent          # number of search agents
    MaxIt = run                  # number of individual runs

    # DSO Tuning Parameters
    H0_minus = 0.17              # minimum initial homeostatic value
    H0_plus = 0.85               # maximum initial homeostatic value
    a = 0.1                      # circadian cost per unit function
    xs = 4.2                     # sleep power index
    xw = 18.2                    # wake power index
    T = 24                       # maximum sleep duration

    # Pre-allocate
    fit = zeros(nPop)                   # vector to store fitness values
    Best_iteration = zeros(MaxIt + 1)   # vector to store best fitness per iteration

    # Initialize Deep Sleep Optimization
    C = sin((2 * π) / T)                # periodic function
    H_min = H0_minus + a * C            # lower homeostatic threshold
    H_max = H0_plus + a * C             # upper homeostatic threshold

    Pop = [lb .+ (ub .- lb) .* rand(nVar) for _ in 1:nPop]  # generate initial population

    # Evaluate initial fitness
    for q in 1:nPop
        fit[q] = prob(Pop[q])
    end
    Best_iteration[1] = minimum(fit)   # store the best fitness value of 0th iteration

    # Main Loop
    for i in 1:MaxIt
        for j in 1:nPop
            Xini = Pop[j]                # current agent's position
            _, ind = findmin(fit)        # position of best agent
            Xbest = Pop[ind]             # best solution so far

            # Generate a candidate solution
            mu = rand()
            mu = clamp(mu, H_min, H_max)  # bound mu between H_min and H_max

            H0 = Pop[j] .+ rand(nVar) .* (Xbest .- mu .* Xini)

            # Apply sleep-wake cycle
            if rand() > mu
                H = H0 .* 10^(-1 / xs)   # exploit agent's position (sleep)
            else
                H = mu .+ (H0 .- mu) .* 10^(-1 / xw)  # explore new positions (wake)
            end

            H = clamp.(H, lb, ub)        # bound variables to their limits

            # Evaluate the new fitness
            fnew = prob(H)
            if fnew < fit[j]             # greedy selection
                Pop[j] = H               # update agent's position
                fit[j] = fnew            # update fitness
            end
        end
        Best_iteration[i + 1] = minimum(fit)  # store best fitness for this iteration
    end

    # DSO-TPM Solution: Global best value and position
    Best_Cost, idx = findmin(fit)
    Best_Position = Pop[idx]

    return Best_Cost, Best_Position, Best_iteration

    # return Dict(
    #     "Cost" => Best_Cost,
    #     "Position" => Best_Position,
    #     "iteration" => Best_iteration
    # )
end
