"""
Chopra, Nitish, and Muhammad Mohsin Ansari. 
"Golden jackal optimization: A novel nature-inspired optimizer for engineering applications." 
Expert Systems with Applications 198 (2022): 116924.
"""
function GJO(SearchAgents_no, Max_iter, lb, ub, dim, fobj)
    Male_Jackal_pos = zeros(1, dim)
    Male_Jackal_score = Inf
    Female_Jackal_pos = zeros(1, dim)
    Female_Jackal_score = Inf

    Positions = initialization(SearchAgents_no, dim, ub, lb)

    Convergence_curve = zeros(Max_iter)

    l = 0  

    while l < Max_iter
        for i in axes(Positions, 1)
            Flag4ub = Positions[i, :] .> ub
            Flag4lb = Positions[i, :] .< lb
            Positions[i, :] = min.(max.(Positions[i, :], lb), ub)

            fitness = fobj(Positions[i, :])

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

        for i in axes(Positions, 1)
            for j in axes(Positions, 2)
                r1 = rand()  
                E0 = 2 * r1 - 1
                E = E1 * E0  

                if abs(E) < 1
                    D_male_jackal = abs((RL[i, j] * Male_Jackal_pos[j] - Positions[i, j]))
                    Male_Positions[i, j] = Male_Jackal_pos[j] - E * D_male_jackal
                    D_female_jackal = abs((RL[i, j] * Female_Jackal_pos[j] - Positions[i, j]))
                    Female_Positions[i, j] = Female_Jackal_pos[j] - E * D_female_jackal
                else
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
