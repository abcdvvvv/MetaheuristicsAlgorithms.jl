"""
Kaur, Satnam, Lalit K. Awasthi, Amrit Lal Sangal, and Gaurav Dhiman. 
"Tunicate Swarm Algorithm: A new bio-inspired based metaheuristic paradigm for global optimization." 
Engineering Applications of Artificial Intelligence 90 (2020): 103541.
"""

function TSA(npop::Int, max_iter::Int, lb, ub, dim::Int, objfun)
    Score = Inf
    Position = zeros(dim)
    Positions = initialization(npop, dim, ub, lb)
    Convergence = zeros(max_iter)

    t = 0

    while t < max_iter
        for i in axes(Positions, 1)
            Flag4ub = Positions[i, :] .> ub
            Flag4lu = Positions[i, :] .< lb
            Positions[i, :] = (Positions[i, :] .* .~(Flag4ub .+ Flag4lu)) .+ ub .* Flag4ub .+ lb .* Flag4lu

            fitness = objfun(Positions[i, :])

            if fitness < Score
                Score = fitness
                Position = copy(Positions[i, :])
            end
        end

        xmin = 1
        xmax = 4
        xr = xmin + rand() * (xmax - xmin)
        xr = floor(Int, xr)

        for i in axes(Positions, 1)
            for j in axes(Positions, 2)
                A1 = ((rand() + rand()) - (2 * rand())) / xr
                c2 = rand()

                if i == 1
                    c3 = rand()
                    d_pos = abs(Position[j] - c2 * Positions[i, j])

                    if c3 >= 0
                        Positions[i, j] = Position[j] + A1 * d_pos
                    else
                        Positions[i, j] = Position[j] - A1 * d_pos
                    end
                else
                    c3 = rand()
                    d_pos = abs(Position[j] - c2 * Positions[i, j])

                    if c3 >= 0
                        Pos_i_j = Position[j] + A1 * d_pos
                    else
                        Pos_i_j = Position[j] - A1 * d_pos
                    end

                    Positions[i, j] = (Pos_i_j + Positions[i-1, j]) / 2
                end
            end
        end

        t += 1
        Convergence[t] = Score
    end

    return Score, Position, Convergence
end
