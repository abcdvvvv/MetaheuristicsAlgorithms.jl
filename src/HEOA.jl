"""
Lian, Junbo, and Guohua Hui. 
"Human evolutionary optimization algorithm." 
Expert Systems with Applications 241 (2024): 122638.

"""
function HEOA(N, Max_iter, lb, ub, dim, fobj)
    jump_factor = abs(lb[1] - ub[1]) / 1000

    A = 0.6  
    LN = 0.4 
    EN = 0.4 
    FN = 0.1 

    LNNumber = round(Int, N * LN)
    ENNumber = round(Int, N * EN)
    FNNumber = round(Int, N * FN)

    if length(ub) == 1
        ub = fill(ub[1], dim)  
        lb = fill(lb[1], dim)  
    end

    X0 = initializationLogistic(N, dim, ub, lb) 
    X = X0

    fitness_new = zeros(N)
    Best_pos = zeros(dim)
    Best_score = Inf

    fitness = [fobj(X[i, :]) for i in 1:N]
    index = sortperm(fitness)  
    fitness = fitness[index]
    BestF = fitness[1]
    WorstF = fitness[end]
    GBestF = fitness[1]  
    AveF = mean(fitness)
    X = X0[index, :]  
    curve = zeros(Float64, Max_iter)
    avg_fitness_curve = zeros(Float64, Max_iter)
    GBestX = X[1, :]  
    X_new = copy(X)

    for i in 1:Max_iter
        # if i % 100 == 0
        #     println("At iteration $i, the fitness is $(curve[i-1])")
        # end

        BestF = fitness[1]
        WorstF = fitness[end]
        avg_fitness_curve[i] = AveF
        R = rand()

        for j in 1:N
            if i <= (1 / 4) * Max_iter
                X_new[j, :] = GBestX * (1 - i / Max_iter) +
                (mean(X[j, :]) .- GBestX) * floor(rand() / jump_factor) * jump_factor +
                0.2 * (1 - i / Max_iter) .* (X[j, :] .- GBestX) .* Levy(dim)
            else
                for j in 1:LNNumber  
                    if R < A
                        X_new[j, :] = 0.2 * cos(pi / 2 * (1 - (i / Max_iter))) * X[j, :] * exp((-i * randn()) / (rand() * Max_iter))
                    else
                        X_new[j, :] = 0.2 * cos(pi / 2 * (1 - (i / Max_iter))) * X[j, :] + randn() * ones(Float64, dim)
                    end
                end

                for j in (LNNumber + 1):(LNNumber + ENNumber)  
                    X_new[j, :] = randn(dim) .* exp.((X[end, :] - X[j, :]) / j^2)
                end

                for j in (LNNumber + ENNumber + 1):(LNNumber + ENNumber + FNNumber)  
                    X_new[j, :] = X[j, :] + 0.2 * cos(pi / 2 * (1 - (i / Max_iter))) * rand(Float64, dim) .* (X[1, :] - X[j, :])
                end

                for j in (LNNumber + ENNumber + FNNumber + 1):N  
                    X_new[j, :] = GBestX + (GBestX - X[j, :]) * randn()
                end
            end

            Flag4ub = X_new[j, :] .> ub
            Flag4lb = X_new[j, :] .< lb
            X_new[j, :] .= (X_new[j, :] .* .!(Flag4ub .| Flag4lb)) .+ ub .* Flag4ub .+ lb .* Flag4lb

            fitness_new[j] = fobj(vec(X_new[j, :]))
            if fitness_new[j] < GBestF
                GBestF = fitness_new[j]
                GBestX = X_new[j, :]
            end
        end

        X .= X_new
        fitness .= fitness_new
        curve[i] = GBestF
        Best_pos = GBestX
        Best_score = curve[end]
    end

    return Best_score, Best_pos, curve
end

function initializationLogistic(pop::Int, dim::Int, ub, lb)
    Boundary_no = length(ub)  
    Positions = zeros(Float64, pop, dim)  

    for i in 1:pop
        for j in 1:dim
            x0 = rand()
            a = 4.0
            x = a * x0 * (1 - x0)  

            if Boundary_no == 1
                Positions[i, j] = (ub[1] - lb[1]) * x + lb[1]
                Positions[i, j] = clamp(Positions[i, j], lb[1], ub[1])  
            else
                Positions[i, j] = (ub[j] - lb[j]) * x + lb[j]
                Positions[i, j] = clamp(Positions[i, j], lb[j], ub[j])  
            end
            x0 = x  
        end
    end

    return Positions
end