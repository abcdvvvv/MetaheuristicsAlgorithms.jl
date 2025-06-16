"""
Dhiman, Gaurav, and Vijay Kumar. 
"Seagull optimization algorithm: Theory and its applications for large-scale industrial engineering problems." 
Knowledge-based systems 165 (2019): 169-196.
"""
function SOA(npop, max_iter, lb, ub, dim, objfun)
    Position = zeros(dim)
    Score = Inf
    Positions = initialization(npop, dim, ub, lb)
    Convergence = zeros(max_iter)

    l = 0

    while l < max_iter
        for i in axes(Positions, 1)
            Positions[i, :] = clamp.(Positions[i, :], lb, ub)

            fitness = objfun(Positions[i, :])

            if fitness < Score
                Score = fitness
                Position = copy(Positions[i, :])
            end
        end

        Fc = 2 - l * (2 / max_iter)

        for i in axes(Positions, 1)
            for j in axes(Positions, 2)
                r1 = rand()
                r2 = rand()

                A1 = 2 * Fc * r1 - Fc
                C1 = 2 * r2
                b = 1
                ll = (Fc - 1) * rand() + 1

                D_alpha = Fc * Positions[i, j] + A1 * (Position[j] - Positions[i, j])
                X1 = D_alpha * exp(b * ll) * cos(ll * 2 * π) + Position[j]
                Positions[i, j] = X1
            end
        end

        l += 1
        Convergence[l] = Score
        println("Convergence[$l] ", Score)
    end

    return Score, Position, Convergence
end
