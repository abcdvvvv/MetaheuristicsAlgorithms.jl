"""
Jia, Heming, Xiaoxu Peng, and Chunbo Lang. 
"Remora optimization algorithm." 
Expert Systems with Applications 185 (2021): 115665.
"""
function ROA(npop::Int, max_iter::Int, lb::Union{Int,AbstractVector}, ub::Union{Int,AbstractVector}, dim::Int, objfun::Function)
    Best_pos = zeros(Float64, dim)
    Best_score = Inf

    Positions = initialization(npop, dim, ub, lb)
    Positions_attempt = zeros(Float64, npop, dim)
    Positions_previous = initialization(npop, dim, ub, lb)
    Convergence_curve = zeros(Float64, max_iter)

    fitness = zeros(Float64, npop)
    H = rand(1, npop) .> 0.5

    for t = 1:max_iter
        for i in axes(Positions, 1)
            Flag4ub = Positions[i, :] .> ub
            Flag4lb = Positions[i, :] .< lb
            Positions[i, :] = (Positions[i, :] .* .!(Flag4ub .| Flag4lb)) .+ ub .* Flag4ub .+ lb .* Flag4lb

            fitness[i] = objfun(Positions[i, :])

            if fitness[i] < Best_score
                Best_score = fitness[i]
                Best_pos .= Positions[i, :]
            end
        end

        for i in axes(Positions, 1)
            a1 = 2 - t * (2 / max_iter)
            r1 = rand()

            if H[i] == false
                a2 = -1 + t * (-1 / max_iter)
                l = (a2 - 1) * rand() + 1
                distance2Leader = abs.(Best_pos .- Positions[i, :])
                Positions[i, :] .= distance2Leader * exp(l) .* cos(l * 2 * π) .+ Best_pos

            elseif H[i] == true
                rand_leader_index = rand(1:npop)
                X_rand = Positions[rand_leader_index, :]
                Positions[i, :] .= Best_pos .- (rand(dim) .* (Best_pos .+ X_rand) / 2 .- X_rand)
            end

            Positions_attempt[i, :] .= Positions[i, :] .+ (Positions[i, :] .- Positions_previous[i, :]) .* randn(dim)

            if objfun(Positions_attempt[i, :]) < objfun(Positions[i, :])
                Positions[i, :] .= Positions_attempt[i, :]
                H[i] = rand() > 0.5 ? true : false

            else
                A = (2 * a1 * r1 - a1)
                C = 0.1
                Positions[i, :] .= Positions[i, :] .- A * (Positions[i, :] .- C * Best_pos)
            end
        end

        Positions_previous .= Positions
        Convergence_curve[t] = Best_score
    end

    return Best_score, Best_pos, Convergence_curve
end
