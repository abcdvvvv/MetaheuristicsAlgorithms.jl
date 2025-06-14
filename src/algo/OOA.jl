"""
Dehghani, Mohammad, and Pavel Trojovský. 
"Osprey optimization algorithm: A new bio-inspired metaheuristic algorithm for solving engineering optimization problems." 
Frontiers in Mechanical Engineering 8 (2023): 1126450.
"""

function OOA(SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness)
    # Lower limit for variables
    lowerbound = ones(dimension) .* lowerbound
    # Upper limit for variables
    upperbound = ones(dimension) .* upperbound

    # INITIALIZATION
    X = zeros(SearchAgents, dimension)  # Initialize the population matrix
    for i in 1:dimension
        X[:, i] = lowerbound[i] .+ rand(SearchAgents) .* (upperbound[i] - lowerbound[i])  # Initial population
    end

    fit = zeros(SearchAgents)  # Initialize fitness array
    for i in 1:SearchAgents
        L = X[i, :]
        fit[i] = fitness(L)  # Compute fitness for each individual
    end

    best_so_far = zeros(Max_iterations)  # Array to store best scores
    average = zeros(Max_iterations)  # Array to store average scores
    fbest = Inf
    xbest = zeros(dimension)

    # Main loop for iterations
    for t in 1:Max_iterations
        # Update: BEST proposed solution
        Fbest, blocation = findmin(fit)  # Get the minimum fitness value and its index
        
        # Update best solution found
        if t == 1
            xbest = X[blocation, :]  # Optimal location
            fbest = Fbest  # The optimization objective function
        elseif Fbest < fbest
            fbest = Fbest
            xbest = X[blocation, :]
        end

        # Loop through each search agent
        for i in 1:SearchAgents
            # PHASE 1: POSITION IDENTIFICATION AND HUNTING THE FISH (EXPLORATION)
            fish_position = findall(x -> x < fit[i], fit)  # Get indices of fitter fish
            
            if isempty(fish_position)
                selected_fish = xbest
            else
                if rand() < 0.5
                    selected_fish = xbest
                else
                    k = rand(1:length(fish_position))  # Randomly select one of the fitter fish
                    selected_fish = X[fish_position[k]]
                end
            end

            I = round(Int, 1 + rand())  # Randomly choose 1 or 2
            X_new_P1 = X[i, :] + rand() * (selected_fish .- I * X[i, :])  # Eq(5)
            X_new_P1 = max.(X_new_P1, lowerbound)  # Clip to lower bound
            X_new_P1 = min.(X_new_P1, upperbound)  # Clip to upper bound

            # Update position based on Eq (6)
            L = X_new_P1
            fit_new_P1 = fitness(L)
            if fit_new_P1 < fit[i]
                X[i, :] = X_new_P1
                fit[i] = fit_new_P1
            end

            # PHASE 2: CARRYING THE FISH TO THE SUITABLE POSITION (EXPLOITATION)
            X_new_P1 = X[i, :] + (lowerbound .+ rand() * (upperbound .- lowerbound)) / t  # Eq(7)
            X_new_P1 = max.(X_new_P1, lowerbound)  # Clip to lower bound
            X_new_P1 = min.(X_new_P1, upperbound)  # Clip to upper bound

            # Update position based on Eq (8)
            L = X_new_P1
            fit_new_P1 = fitness(L)
            if fit_new_P1 < fit[i]
                X[i, :] = X_new_P1
                fit[i] = fit_new_P1
            end
        end

        best_so_far[t] = fbest
        average[t] = mean(fit)  # Calculate average fitness
        # println("It $t = $fbest")
    end

    return fbest, xbest, best_so_far
end