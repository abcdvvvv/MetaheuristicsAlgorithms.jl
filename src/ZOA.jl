"""
Trojovská, Eva, Mohammad Dehghani, and Pavel Trojovský. 
"Zebra optimization algorithm: A new bio-inspired optimization algorithm for solving optimization algorithm." 
Ieee Access 10 (2022): 49445-49473.

"""

function ZOA(SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness)
    lowerbound = ones(dimension) .* lowerbound
    upperbound = ones(dimension) .* upperbound

    X = zeros(SearchAgents, dimension)  
    for i in 1:dimension
        X[:, i] = lowerbound[i] .+ rand(SearchAgents) .* (upperbound[i] - lowerbound[i])
    end

    fit = zeros(SearchAgents)
    for i in 1:SearchAgents
        L = X[i, :]
        fit[i] = fitness(L)
    end

    best_so_far = zeros(Max_iterations)
    average = zeros(Max_iterations)
    fbest = Inf  
    PZ = zeros(dimension)  

    for t in 1:Max_iterations
        best, location = findmin(fit)  
        if t == 1 || best < fbest
            fbest = best
            PZ = X[location, :]
        end

        for i in 1:SearchAgents
            I = round(Int, 1 + rand())  
            X_newP1 = X[i, :] + rand(dimension) .* (PZ - I * X[i, :])  
            X_newP1 = clamp.(X_newP1, lowerbound, upperbound)  

            f_newP1 = fitness(X_newP1)
            if f_newP1 <= fit[i]
                X[i, :] = X_newP1
                fit[i] = f_newP1
            end
        end

        Ps = rand()
        k = rand(1:SearchAgents)
        AZ = X[k, :] 

        for i in 1:SearchAgents
            if Ps < 0.5
                R = 0.1
                X_newP2 = X[i, :] + R * (2 * rand(dimension) .- 1) * (1 - t / Max_iterations) .* X[i, :]  # Eq(5) S1
                X_newP2 = clamp.(X_newP2, lowerbound, upperbound)  
            else
                I = round(Int, 1 + rand())
                X_newP2 = X[i, :] + rand(dimension) .* (AZ - I * X[i, :])  
                X_newP2 = clamp.(X_newP2, lowerbound, upperbound)  
            end

            f_newP2 = fitness(X_newP2) 
            if f_newP2 <= fit[i]
                X[i, :] = X_newP2
                fit[i] = f_newP2
            end
        end

        best_so_far[t] = fbest
        average[t] = mean(fit)
    end

    Best_score = fbest
    Best_pos = PZ
    ZOA_curve = best_so_far

    return Best_score, Best_pos, ZOA_curve
end
