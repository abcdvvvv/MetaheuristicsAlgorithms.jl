"""
Dhiman, Gaurav, Meenakshi Garg, Atulya Nagar, Vijay Kumar, and Mohammad Dehghani. 
"A novel algorithm for global optimization: rat swarm optimizer." 
Journal of Ambient Intelligence and Humanized Computing 12 (2021): 8457-8482.
"""
function RSO(Search_Agents, Max_iterations, Lower_bound, Upper_bound, dimension, objective)
    Position = zeros(dimension)
    Score = Inf
    Positions = initialization(Search_Agents, dimension, Upper_bound, Lower_bound)
    Convergence = zeros(Max_iterations)
    
    l = 0
    x = 1
    y = 5
    R = floor(Int, (y - x) * rand() + x)

    while l < Max_iterations
        # Flag4Upper_bound = Positions .> Upper_bound
        # Flag4Lower_bound = Positions .< Lower_bound
        Positions .= max.(Lower_bound, min.(Positions, Upper_bound))  # Clamp values between bounds

        fitness_values = [objective(Positions[i, :]) for i in axes(Positions, 1)]

        best_index = argmin(fitness_values)  
        best_fitness = fitness_values[best_index]

        if best_fitness < Score
            Score = best_fitness
            Position .= Positions[best_index, :]
        end

        A = R - l * (R / Max_iterations)

        for i in axes(Positions, 1)

            for j in axes(Positions, 2)
                C = 2 * rand()
                P_vec = A * Positions[i, j] + abs(C * (Position[j] - Positions[i, j]))
                P_final = Position[j] - P_vec
                Positions[i, j] = P_final
            end
        end

        l += 1
        Convergence[l] = Score
    end

    return Score, Position, Convergence
end

