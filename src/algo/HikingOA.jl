"""
Oladejo, Sunday O., Stephen O. Ekwe, and Seyedali Mirjalili. 
"The Hiking Optimization Algorithm: A novel human-based metaheuristic approach." 
Knowledge-Based Systems 296 (2024): 111880.
"""

function HikingOA(hiker, MaxIter, LB, UB, dim, ObjFun)
    # Problem Parameters
    prob = ObjFun             
    nVar = dim                
    lb = LB                   
    ub = UB                   
    nPop = hiker              
    MaxIt = MaxIter           

    # Pre-allocate
    fit = zeros(nPop)                  
    Best_iteration = zeros(MaxIt + 1)  

    # Initialize population
    Pop = [lb .+ (ub .- lb) .* rand(nVar) for _ in 1:nPop]  

    # Evaluate initial fitness
    for q in 1:nPop
        fit[q] = prob(Pop[q])
    end
    Best_iteration[1] = minimum(fit)  

    # Main Loop
    for i in 1:MaxIt
        _, ind = findmin(fit)           
        Xbest = Pop[ind]                

        for j in 1:nPop
            Xini = Pop[j]               

            theta = rand(0:50)          
            s = tan(deg2rad(theta))     
            SF = rand([1, 2])           

            Vel = 6 * exp(-3.5 * abs(s + 0.05))  
            newVel = Vel .+ rand(nVar) .* (Xbest .- SF .* Xini)  

            newPop = Xini .+ newVel     
            newPop = clamp.(newPop, lb, ub)  

            fnew = prob(newPop)         
            if fnew < fit[j]            
                Pop[j] = newPop         
                fit[j] = fnew           
            end
        end

        Best_iteration[i + 1] = minimum(fit)  
    end

    Best_Hike, idx = findmin(fit)
    Best_Position = Pop[idx]

    return Best_Hike, Best_Position, Best_iteration
end
