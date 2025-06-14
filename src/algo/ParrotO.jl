"""
Lian, Junbo, Guohua Hui, Ling Ma, Ting Zhu, Xincan Wu, Ali Asghar Heidari, Yi Chen, and Huiling Chen. 
"Parrot optimizer: Algorithm and applications to medical problems." 
Computers in Biology and Medicine 172 (2024): 108064.
"""
function ParrotO(N, Max_iter, lb, ub, dim, fobj)

    if length(ub) == 1
        ub = fill(ub, dim)
        lb = fill(lb, dim)
    end

    X0 = initialization(N, dim, ub, lb) 
    X = X0

    fitness = zeros(N)
    for i in 1:N
        fitness[i] = fobj(X[i, :])
    end

    index = sortperm(fitness) 
    fitness = fitness[index]
    GBestF = fitness[1] 
    for i in 1:N
        X[i, :] .= X0[index[i], :]
    end

    curve = zeros(Max_iter)
    GBestX = X[1, :] 
    X_new = copy(X)
    

    for i in 1:Max_iter
        alpha = rand() / 5
        sita = rand() * π

        for i in axes(X, 1)
            St = rand(1:4)

            if St == 1
                X_new[j, :] .= (X[j, :] .- GBestX) .* Levy(dim) .+ rand() * mean(X) * (1 - i / Max_iter)^ (2 * i / Max_iter)

            elseif St == 2
                X_new[j, :] .= X[j, :] .+ GBestX .* Levy(dim) .+ randn() * (1 - i / Max_iter) * ones(dim)

            elseif St == 3
                H = rand()
                if H < 0.5
                    X_new[j, :] .= X[j, :] .+ alpha * (1 - i / Max_iter) * (X[j, :] .- mean(X))
                else
                    X_new[j, :] .= X[j, :] .+ alpha * (1 - i / Max_iter) * exp(-j / (rand() * Max_iter))
                end

            else
                X_new[j, :] .= X[j, :] .+ rand() * cos((π * i) / (2 * Max_iter)) * (GBestX .- X[j, :]) .- cos(sita) * (i / Max_iter)^ (2 / Max_iter) * (X[j, :] .- GBestX)
            end

            for m in 1:N
                for a in 1:dim
                    if X_new[m, a] > ub[a]
                        X_new[m, a] = ub[a]
                    end
                    if X_new[m, a] < lb[a]
                        X_new[m, a] = lb[a]
                    end
                end
            end

            if fobj(X_new[j, :]) < GBestF
                GBestF = fobj(X_new[j, :])
                GBestX .= X_new[j, :]
            end
        end

        fitness_new = zeros(N)
        for s in 1:N
            fitness_new[s] = fobj(X_new[s, :])
        end
        
        for s in 1:N
            if fitness_new[s] < GBestF
                GBestF = fitness_new[s]
                GBestX .= X_new[s, :]
            end
        end
        
        X .= X_new
        fitness .= fitness_new

        index = sortperm(fitness) 
        fitness = fitness[index]
        for s in 1:N
            X0[s, :] .= X[index[s], :]
        end
        
        X .= X0
        curve[i] = GBestF
    end

    Best_pos = GBestX
    Best_score = curve[end]
    
    return Best_score, Best_pos, curve 
end
