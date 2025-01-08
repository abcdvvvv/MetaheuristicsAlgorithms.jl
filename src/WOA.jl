"""
Mirjalili, Seyedali, and Andrew Lewis. 
"The whale optimization algorithm." 
Advances in engineering software 95 (2016): 51-67.
"""
function WOA(SearchAgents_no, Max_iter, lb, ub, dim, fobj)
    Leader_pos = zeros(1, dim)
    Leader_score = Inf  

    Positions = initialization(SearchAgents_no, dim, ub, lb)

    Convergence_curve = zeros(Max_iter, 1)

    t = 0  

    while t < Max_iter
        for i in axes(Positions, 1)
            Flag4ub = Positions[i, :] .> ub
            Flag4lb = Positions[i, :] .< lb
            Positions[i, :] .= Positions[i, :] .* .~(Flag4ub .+ Flag4lb) .+ ub .* Flag4ub .+ lb .* Flag4lb

            fitness = fobj(Positions[i, :])

            if fitness < Leader_score  
                Leader_score = fitness  
                Leader_pos = Positions[i, :]
            end
        end

        a = 2 - t * (2 / Max_iter)  
        a2 = -1 + t * (-1 / Max_iter)  

        for i in axes(Positions, 1)
            r1 = rand()  
            r2 = rand()  

            A = 2 * a * r1 - a  
            C = 2 * r2  

            b = 1  
            l = (a2 - 1) * rand() + 1  

            p = rand()  

            for j in axes(Positions, 2)
                if p < 0.5
                    if abs(A) >= 1
                        rand_leader_index = floor(Int, SearchAgents_no * rand() + 1)
                        X_rand = Positions[rand_leader_index, :]
                        D_X_rand = abs(C * X_rand[j] - Positions[i, j])  
                        Positions[i, j] = X_rand[j] - A * D_X_rand  
                    elseif abs(A) < 1
                        D_Leader = abs(C * Leader_pos[j] - Positions[i, j])  
                        Positions[i, j] = Leader_pos[j] - A * D_Leader  
                    end
                elseif p >= 0.5
                    distance2Leader = abs(Leader_pos[j] - Positions[i, j])
                    Positions[i, j] = distance2Leader * exp(b * l) * cos(l * 2 * pi) + Leader_pos[j]
                end
            end
        end
        t += 1
        Convergence_curve[t] = Leader_score
    end
    return Leader_score, Leader_pos, Convergence_curve
end