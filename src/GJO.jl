"""
Chopra, Nitish, and Muhammad Mohsin Ansari. 
"Golden jackal optimization: A novel nature-inspired optimizer for engineering applications." 
Expert Systems with Applications 198 (2022): 116924.
"""
function GJO(SearchAgents_no, Max_iter, lb, ub, dim, fobj)
    # Initialize Golden Jackal pair
    Male_Jackal_pos = zeros(1, dim)
    Male_Jackal_score = Inf
    Female_Jackal_pos = zeros(1, dim)
    Female_Jackal_score = Inf

    # Initialize the positions of search agents
    Positions = initialization(SearchAgents_no, dim, ub, lb)

    Convergence_curve = zeros(Max_iter)

    l = 0  # Loop counter

    # Main loop
    while l < Max_iter
        # for i in 1:size(Positions, 1)
        for i in axes(Positions, 1)
            # Boundary checking
            Flag4ub = Positions[i, :] .> ub
            Flag4lb = Positions[i, :] .< lb
            # Positions[i, :] = (Positions[i, :].*~(Flag4ub .+ Flag4lb)) .+ ub.*Flag4ub .+ lb.*Flag4lb
            Positions[i, :] = min.(max.(Positions[i, :], lb), ub)

            # Calculate objective function for each search agent
            fitness = fobj(Positions[i, :])

            # Update Male Jackal
            if fitness < Male_Jackal_score
                Male_Jackal_score = fitness
                Male_Jackal_pos = Positions[i, :]
            end
            if fitness > Male_Jackal_score && fitness < Female_Jackal_score
                Female_Jackal_score = fitness
                Female_Jackal_pos = Positions[i, :]
            end
        end

        E1 = 1.5 * (1 - (l / Max_iter))
        RL = 0.05 * levy(SearchAgents_no, dim, 1.5)

        Male_Positions = zeros(SearchAgents_no, dim)
        Female_Positions = zeros(SearchAgents_no, dim)

        # for i in 1:size(Positions, 1)
        for i in axes(Positions, 1)
            # for j in 1:size(Positions, 2)
            for j in axes(Positions, 2)
                r1 = rand()  # r1 is a random number in [0, 1]
                E0 = 2 * r1 - 1
                E = E1 * E0  # Evading energy

                if abs(E) < 1
                    # EXPLOITATION
                    D_male_jackal = abs((RL[i, j] * Male_Jackal_pos[j] - Positions[i, j]))
                    Male_Positions[i, j] = Male_Jackal_pos[j] - E * D_male_jackal
                    D_female_jackal = abs((RL[i, j] * Female_Jackal_pos[j] - Positions[i, j]))
                    Female_Positions[i, j] = Female_Jackal_pos[j] - E * D_female_jackal
                else
                    # EXPLORATION
                    D_male_jackal = abs((Male_Jackal_pos[j] - RL[i, j] * Positions[i, j]))
                    Male_Positions[i, j] = Male_Jackal_pos[j] - E * D_male_jackal
                    D_female_jackal = abs((Female_Jackal_pos[j] - RL[i, j] * Positions[i, j]))
                    Female_Positions[i, j] = Female_Jackal_pos[j] - E * D_female_jackal
                end

                Positions[i, j] = (Male_Positions[i, j] + Female_Positions[i, j]) / 2
            end
        end

        l += 1
        Convergence_curve[l] = Male_Jackal_score
    end

    return Male_Jackal_score, Male_Jackal_pos, Convergence_curve
end
