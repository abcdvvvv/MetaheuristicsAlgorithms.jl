"""
Oladejo, Sunday O., Stephen O. Ekwe, Lateef A. Akinyemi, and Seyedali A. Mirjalili. 
"The deep sleep optimiser: A human-based metaheuristic approach." 
IEEE Access (2023).
"""

function DSO(search_agent, run, LB, UB, dim, ObjFun)

    prob = ObjFun                
    nVar = dim                   
    lb = LB                      
    ub = UB                      
    nPop = search_agent          
    MaxIt = run                  

    H0_minus = 0.17              
    H0_plus = 0.85               
    a = 0.1                      
    xs = 4.2                     
    xw = 18.2                    
    T = 24                       

    fit = zeros(nPop)                   
    Best_iteration = zeros(MaxIt + 1)   

    C = sin((2 * π) / T)                
    H_min = H0_minus + a * C            
    H_max = H0_plus + a * C             

    Pop = [lb .+ (ub .- lb) .* rand(nVar) for _ in 1:nPop]  

    for q in 1:nPop
        fit[q] = prob(Pop[q])
    end
    Best_iteration[1] = minimum(fit)   


    for i in 1:MaxIt
        for j in 1:nPop
            Xini = Pop[j]                
            _, ind = findmin(fit)        
            Xbest = Pop[ind]             

            mu = rand()
            mu = clamp(mu, H_min, H_max)  
            H0 = Pop[j] .+ rand(nVar) .* (Xbest .- mu .* Xini)

            if rand() > mu
                H = H0 .* 10^(-1 / xs)   
            else
                H = mu .+ (H0 .- mu) .* 10^(-1 / xw)  
            end

            H = clamp.(H, lb, ub)        


            fnew = prob(H)
            if fnew < fit[j]             
                Pop[j] = H               
                fit[j] = fnew            
            end
        end
        Best_iteration[i + 1] = minimum(fit)  
    end

    Best_Cost, idx = findmin(fit)
    Best_Position = Pop[idx]

    return Best_Cost, Best_Position, Best_iteration
end
