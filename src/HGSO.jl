"""
Hashim, F.A., Houssein, E.H., Mabrouk, M.S., Al-Atabany, W. and Mirjalili, S., 2019. 
Henry gas solubility optimization: A novel physics-based algorithm. 
Future Generation Computer Systems, 101, pp.646-667.
"""

function HGSO(var_n_gases, var_niter, var_down, var_up, dim, objfunc)#(objfunc, dim, var_down, var_up, var_niter, var_n_gases, var_n_types)
    # Default parameters if fewer arguments are provided
    # if nargin < 5
    #     var_n_gases, var_n_types, var_niter = fun_getDefaultOptions()
    # end
    var_n_types = 5

    # Constants in eq (7)
    l1 = 5e-3
    l2 = 100
    l3 = 1e-2

    # Constants in eq (10)
    alpha = 1
    beta = 1

    # Constants in eq (11)
    M1 = 0.1
    M2 = 0.2

    # Parameters setting in eq (7)
    K = l1 * rand(var_n_types)
    P = l2 * rand(var_n_gases)
    C = l3 * rand(var_n_types)

    # Randomly initializes the position of agents in the search space
    X = var_down .+ rand(var_n_gases, dim) .* (var_up - var_down)

    # The population agents are divided into equal clusters with the same Henry's constant value
    Group = Create_Groups(var_n_gases, var_n_types, X)

    # Compute cost of each agent
    best_fit = zeros(var_n_types)
    best_pos = Vector{Any}(undef, var_n_types)

    for i in 1:var_n_types
        Group[i], best_fit[i], best_pos[i] = Evaluate(objfunc, var_n_types, var_n_gases, Group[i], 0, 1)
    end

    var_Gbest, var_gbest = findmin(best_fit)
    vec_Xbest = best_pos[var_gbest]

    vec_Gbest_iter = zeros(var_niter)

    for var_iter in 1:var_niter
        S = update_variables(var_iter, var_niter, K, P, C, var_n_types, var_n_gases)
        Groupnew = update_positions(Group, best_pos, vec_Xbest, S, var_n_gases, var_n_types, var_Gbest, alpha, beta, dim)
        # Groupnew = fun_checkpositions(dim, Groupnew, var_n_gases, var_n_types, var_down, var_up)
        # (dim, Group, var_n_gases, var_n_types, var_down, var_up)
        Groupnew = fun_checkpoisions(dim, Groupnew, var_n_gases, var_n_types, var_down, var_up)

        for i in 1:var_n_types
            Group[i], best_fit[i], best_pos[i] = Evaluate(objfunc, var_n_types, var_n_gases, Group[i], Groupnew[i], 0)
            Group[i] = worst_agents(Group[i], M1, M2, dim, var_up, var_down, var_n_gases, var_n_types)
        end

        var_Ybest, var_index = findmin(best_fit)
        vec_Gbest_iter[var_iter] = var_Ybest

        if var_Ybest < var_Gbest
            var_Gbest = var_Ybest
            vec_Xbest = best_pos[var_index]
        end
    end

    return var_Gbest, vec_Xbest, vec_Gbest_iter
end

function Create_Groups(var_n_gases, var_n_types, X)
    N = div(var_n_gases, var_n_types)  # Number of gases per group
    Group = Vector{Dict{Symbol, Any}}(undef, var_n_types)  # Initialize Group as a vector of dictionaries

    i = 1
    for j in 1:var_n_types
        Group[j] = Dict(:Position => X[i:i+N-1, :])  # Assign positions to the group
        # println("This is a message with a number: ", size(Group[j][:Position]))
        
        i = j * N + 1  # Update index
        if i + N > var_n_gases
            i = j * N  # Prevent out-of-bounds indexing
        end
    end

    return Group
end

function Evaluate(objfunc, var_n_types, var_n_gases, X, Xnew, init_flag)
    N = div(var_n_gases, var_n_types)  # Number of agents per type

    # Initialize the :fitness key if it doesn't exist
    if !haskey(X, :fitness)
        X[:fitness] = zeros(N)  # Initialize fitness as a zero vector with N elements
    end

    if init_flag == 1
        # Compute the fitness for all agents
        for j in 1:N
            X[:fitness][j] = objfunc(X[:Position][j, :])
        end
    else
        # Compare and update fitness
        for j in 1:N
            temp_fit = objfunc(Xnew[:Position][j, :])
            if temp_fit < X[:fitness][j]
                X[:fitness][j] = temp_fit
                X[:Position][j, :] = Xnew[:Position][j, :]
            end
        end
    end

    # Find the best fitness and its corresponding position
    best_fit, index_best = findmin(X[:fitness])
    best_pos = X[:Position][index_best, :]

    return X, best_fit, best_pos
end

function update_variables(var_iter, var_niter, K, P, C, var_n_types, var_n_gases)
    T = exp(-var_iter / var_niter)  # Temperature factor
    T0 = 298.15  # Reference temperature
    N = div(var_n_gases, var_n_types)  # Number of agents per type
    S = zeros(var_n_gases, size(P, 2))  # Initialize the matrix S with appropriate size

    i = 1
    for j in 1:var_n_types
        # Update K based on the temperature
        K[j] = K[j] * exp(-C[j] * (1 / T - 1 / T0))
        
        # Calculate the contribution of P and K
        S[i:i+N-1, :] .= P[i:i+N-1, :] * K[j]  # Element-wise multiplication and assignment
        
        i = j * N + 1  # Update index for the next block
        if i + N > var_n_gases
            i = j * N  # Prevent out-of-bounds indexing
        end
    end
    
    return S
end

function update_positions(Group, best_pos, vec_Xbest, S, var_n_gases, var_n_types, var_Gbest, alpha, beta, var_nvars)
    vec_flag = [1, -1]  # Flag for direction

    for i in 1:var_n_types
        for j in 1:div(var_n_gases, var_n_types)
            # Calculate gama
            gama = beta * exp(-(var_Gbest + 0.05) / (Group[i][:fitness][j] + 0.05))
            
            # Random flag index selection (1 or -1)
            # flag_index = floor(2 * rand() + 1)
            flag_index = Int(floor(2 * rand() + 1))
            var_flag = vec_flag[flag_index]

            # Update position of the agent
            for k in 1:var_nvars
                Group[i][:Position][j, k] += var_flag * rand() * gama * (best_pos[i][k] - Group[i][:Position][j, k]) + rand() * alpha * var_flag * (S[i] * vec_Xbest[k] - Group[i][:Position][j, k])
            end
        end
    end

    return Group
end

function fun_checkpoisions(dim, Group, var_n_gases, var_n_types, var_down, var_up)
    Lb = ones(dim) * var_down  # Lower bounds (1D vector)
    Ub = ones(dim) * var_up    # Upper bounds (1D vector)

    for j in 1:var_n_types
        for i in 1:div(var_n_gases, var_n_types)
            # Extract the position of agent i from the dictionary
            pos = Group[j][:Position][i, :]  # Get the position from the dictionary

            # Check if the position is below the lower bound
            isBelow1 = pos .< Lb
            # Check if the position is above the upper bound
            isAboveMax = pos .> Ub

            if any(isBelow1)
                Group[j][:Position][i, :] .= Lb  # Set position to lower bound
            elseif any(isAboveMax)
                Group[j][:Position][i, :] .= Ub  # Set position to upper bound
            end
        end
    end

    return Group
end

function worst_agents(X, M1, M2, dim, G_max, G_min, var_n_gases, var_n_types)
    # Sort the fitness in descending order and get the indices
    # X_sort, X_index = sortperm(X[:fitness], rev=true)

    X_index = sortperm(X[:fitness], rev=true)
    X_sort = X[:fitness][X_index]
    
    # Calculate M1N and M2N
    M1N = M1 * var_n_gases / var_n_types
    M2N = M2 * var_n_gases / var_n_types
    
    # Randomly choose the number of worst agents to replace
    Nw = round(Int, (M2N - M1N) * rand() + M1N)
    
    # Replace the positions of the worst agents with random positions within the bounds
    for k in 1:Nw
        idx = X_index[k]  # Get the index of the worst agent from the sorted indices
        # Update the position of the worst agent (we use idx to access X[:Position])
        X[:Position][idx, :] .= G_min .+ rand(dim) .* (G_max .- G_min)
    end
    
    # Return the updated dictionary X
    return X
end