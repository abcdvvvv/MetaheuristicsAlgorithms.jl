"""
Arora, Sankalap, and Satvir Singh. 
"Butterfly optimization algorithm: a novel approach for global optimization." 
Soft computing 23 (2019): 715-734.
"""


function BOA(n, N_iter, Lb, Ub, dim, fobj)
    # n is the population size
    # N_iter represents total number of iterations
    p = 0.6                  # probability switch
    power_exponent = 0.1      # initial value of a
    sensory_modality = 0.01   # initial value of c

    # Initialize the positions of search agents
    Sol = initialization(n, dim, Ub, Lb)
    Fitness = zeros(n)
    for i in 1:n
        Fitness[i] = fobj(Sol[i, :])
    end

    # Find the current best_pos
    fmin, I = findmin(Fitness)
    best_pos = Sol[I, :]

    S = copy(Sol)
    Convergence_curve = zeros(N_iter)

    # Start the iterations -- Butterfly Optimization Algorithm
    for t in 1:N_iter
        for i in 1:n  # Loop over all butterflies/solutions
            # Calculate fragrance of each butterfly which is correlated with objective function
            Fnew = fobj(S[i, :])
            FP = sensory_modality * (Fnew^power_exponent)

            # Global or local search
            if rand() < p
                dis = rand() * rand() * best_pos - Sol[i, :]  # Eq. (2) in paper
                S[i, :] = Sol[i, :] + dis * FP
            else
                # Find random butterflies in the neighborhood
                epsilon = rand()
                JK = randperm(n)
                dis = epsilon * epsilon * Sol[JK[1], :] - Sol[JK[2], :]
                S[i, :] = Sol[i, :] + dis * FP  # Eq. (3) in paper
            end

            # Check if the simple limits/bounds are OK
            Flag4Ub = S[i, :] .> Ub
            Flag4Lb = S[i, :] .< Lb
            S[i, :] = (S[i, :] .* .~(Flag4Ub .+ Flag4Lb)) .+ Ub .* Flag4Ub .+ Lb .* Flag4Lb

            # Evaluate new solutions
            Fnew = fobj(S[i, :])  # Fnew represents new fitness values

            # If fitness improves (better solutions found), update then
            if Fnew <= Fitness[i]
                Sol[i, :] = S[i, :]
                Fitness[i] = Fnew
            end

            # Update the current global best_pos
            if Fnew <= fmin
                best_pos = S[i, :]
                fmin = Fnew
            end
        end

        # Store convergence curve
        Convergence_curve[t] = fmin

        # Update sensory_modality
        sensory_modality = sensory_modality_NEW(sensory_modality, N_iter)
    end

    return fmin, best_pos, Convergence_curve
end

# Boundary constraints (for simplicity within the main loop)
function sensory_modality_NEW(x, Ngen)
    return x + (0.025 / (x * Ngen))
end

# Initialization of agents within bounds
function initialization(n, dim, Ub, Lb)
    return rand(n, dim) .* (Ub - Lb) .+ Lb
end
