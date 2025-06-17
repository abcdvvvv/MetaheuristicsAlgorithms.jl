"""
Dhiman, Gaurav, and Amandeep Kaur. 
"STOA: a bio-inspired based optimization algorithm for industrial engineering problems." 
Engineering Applications of Artificial Intelligence 82 (2019): 148-174.
"""
function STOA(npop::Int, max_iter::Int, lb, ub, dim::Int, objfun)
    Position = zeros(Float64, dim)
    Score = Inf

    Positions = initialization(npop, dim, ub, lb)
    Convergence = zeros(Float64, max_iter)

    l = 0

    while l < max_iter
        for i in axes(Positions, 1)
            Positions .= max.(lb, min.(Positions, ub))

            fitness = objfun(Positions[i, :])

            if fitness < Score
                Score = fitness
                Position .= Positions[i, :]
            end
        end

        Sa = 2 - l * (2 / max_iter)

        for i in axes(Positions, 1)
            for j in axes(Positions, 2)
                r1 = 0.5 * rand()
                r2 = 0.5 * rand()
                X1 = 2 * Sa * r1 - Sa
                b = 1
                ll = (Sa - 1) * rand() + 1
                D_alphs = Sa * Positions[i, j] + X1 * (Position[j] - Positions[i, j])
                res = D_alphs * exp(b * ll) * cos(ll * 2 * π) + Position[j]
                Positions[i, j] = res
            end
        end
        l += 1
        Convergence[l] = Score
    end

    return Score, Position, Convergence
end
