"""
Oladejo, Sunday O., Stephen O. Ekwe, and Seyedali Mirjalili. 
"The Hiking Optimization Algorithm: A novel human-based metaheuristic approach." 
Knowledge-Based Systems 296 (2024): 111880.
"""
#function MRFO(nPop, MaxIt, Low, Up, Dim, F_index)#(F_index, MaxIt, nPop)
function HikingOA(hiker, MaxIter, LB, UB, dim, ObjFun)#(, LB, UB, dim, hiker, MaxIter)
    # Problem Parameters
    prob = ObjFun             # objective function
    nVar = dim                # dimension of problem
    lb = LB                   # lower bound
    ub = UB                   # upper bound
    nPop = hiker              # number of hikers
    MaxIt = MaxIter           # max iterations

    # Pre-allocate
    fit = zeros(nPop)                  # vectorize variable to store fitness values
    Best_iteration = zeros(MaxIt + 1)  # vectorize variable to store fitness per iteration

    # Initialize population
    Pop = [lb .+ (ub .- lb) .* rand(nVar) for _ in 1:nPop]  # generate initial positions of hikers

    # Evaluate initial fitness
    for q in 1:nPop
        fit[q] = prob(Pop[q])
    end
    Best_iteration[1] = minimum(fit)  # store initial fitness value at 0th iteration

    # Main Loop
    for i in 1:MaxIt
        _, ind = findmin(fit)           # find best fitness
        Xbest = Pop[ind]                # global best position

        for j in 1:nPop
            Xini = Pop[j]               # current position of the jth hiker

            theta = rand(0:50)          # random elevation angle
            s = tan(deg2rad(theta))     # compute slope
            SF = rand([1, 2])           # sweep factor (1 or 2 randomly)

            Vel = 6 * exp(-3.5 * abs(s + 0.05))  # compute walking velocity
            newVel = Vel .+ rand(nVar) .* (Xbest .- SF .* Xini)  # determine new position

            newPop = Xini .+ newVel     # update position of hiker
            newPop = clamp.(newPop, lb, ub)  # enforce bounds

            fnew = prob(newPop)         # re-evaluate fitness
            if fnew < fit[j]            # greedy selection
                Pop[j] = newPop         # store best position
                fit[j] = fnew           # store new fitness
            end
        end

        Best_iteration[i + 1] = minimum(fit)  # store best fitness per iteration
        # println("Iteration $(i + 1): Best Cost = $(Best_iteration[i + 1])")
    end

    # THFO Solution: Store global best fitness and position
    Best_Hike, idx = findmin(fit)
    Best_Position = Pop[idx]

    return Best_Hike, Best_Position, Best_iteration

    # return Dict(
    #     "Hike" => Best_Hike,
    #     "Position" => Best_Position,
    #     "iteration" => Best_iteration
    # )
end
