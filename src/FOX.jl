"""
Mohammed, Hardi, and Tarik Rashid. 
"FOX: a FOX-inspired optimization algorithm." 
Applied Intelligence 53, no. 1 (2023): 1030-1050.
"""

function FOX(SearchAgents_no, Max_iter, lb, ub, dim, fobj)
    Best_pos = zeros(dim)
    Best_score = Inf   # Change this to -Inf for maximization problems
    MinT = Inf

    # Initialize the positions of search agents
    X = initialization(SearchAgents_no, dim, ub, lb)
    Distance_Fox_Rat = zeros(SearchAgents_no, dim)
    convergence_curve = zeros(Max_iter)  # Array to store convergence data

    l = 0  # Loop counter

    # Both c1 and c2 have different range values
    c1 = 0.18   # Range of c1 is [0, 0.18]
    c2 = 0.82   # Range of c2 is [0.19, 1]

    # Main loop
    while l < Max_iter
        for i in 1:SearchAgents_no

            # Return search agents that go beyond boundaries
            Flag4ub = X[i, :] .> ub
            Flag4lb = X[i, :] .< lb
            X[i, :] = X[i, :] .* .~(Flag4ub .+ Flag4lb) .+ ub .* Flag4ub .+ lb .* Flag4lb

            # Calculate objective function for each search agent
            fitness = fobj(X[i, :])

            # Update Best position
            if fitness < Best_score
                Best_score = fitness
                Best_pos = copy(X[i, :])
            end
        end

        # Store the current best score in the convergence curve
        convergence_curve[l + 1] = Best_score
        # println("Iter : $(l + 1),  Best_score =$Best_score")

        a = 2 * (1 - (l / Max_iter))
        Jump = 0.0
        dir = rand()

        for i in 1:SearchAgents_no
            r = rand()
            p = rand()

            if r >= 0.5
                if p > 0.18
                    Time = rand(dim)
                    sps = Best_pos ./ Time
                    Distance_S_Travel = sps .* Time
                    Distance_Fox_Rat[i, :] = 0.5 .* Distance_S_Travel
                    tt = sum(Time) / dim
                    t = tt / 2
                    Jump = 0.5 * 9.81 * t^2
                    X[i, :] = Distance_Fox_Rat[i, :] .* Jump * c1

                elseif p <= 0.18
                    Time = rand(dim)
                    sps = Best_pos ./ Time
                    Distance_S_Travel = sps .* Time
                    Distance_Fox_Rat[i, :] = 0.5 .* Distance_S_Travel
                    tt = sum(Time) / dim
                    t = tt / 2
                    Jump = 0.5 * 9.81 * t^2
                    X[i, :] = Distance_Fox_Rat[i, :] .* Jump * c2
                end

                if MinT > tt
                    MinT = tt
                end

            elseif r < 0.5
                # Random walk
                X[i, :] = Best_pos .+ randn(dim) .* (MinT * a)
            end
        end

        l += 1
    end

    return Best_score, Best_pos, convergence_curve
end