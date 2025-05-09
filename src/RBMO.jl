"""
Fu, Shengwei, et al. 
"Red-billed blue magpie optimizer: a novel metaheuristic algorithm for 2D/3D UAV path planning and engineering design problems." 
Artificial Intelligence Review 57.6 (2024): 134.
"""

function RBMO(N::Int, T::Int, Xmin::Union{Int, AbstractVector}, Xmax::Union{Int, AbstractVector}, D::Int, fobj)
    Xfood = zeros(D)
    BestValue = Inf
    Conv = zeros(T)
    fitness = fill(Inf, N)
    FES = 0
    Epsilon = 0.5

    X = initialization(N, D, Xmax, Xmin)
    X_old = copy(X)
    fitness_old = zeros(N)

    for i in 1:N
        fitness_old[i] = fobj(X_old[i, :])
    end

    t = 1

    while t ≤ T
        for i in 1:N
            p = rand(2:5)
            selected_index_p = randperm(N)[1:p]
            Xp = X[selected_index_p, :]
            Xpmean = mean(Xp, dims=1)

            q = rand(10:N)
            selected_index_q = randperm(N)[1:q]
            Xq = X[selected_index_q, :]
            Xpmean = vec(mean(Xp, dims=1))

            A = randperm(N)
            R1 = A[1]

            if rand() < Epsilon
                X[i, :] = X[i, :] .+ (Xpmean - X[R1, :]) .* rand(D)
            else
                X[i, :] = X[i, :] .+ (Xpmean - X[R1, :]) .* rand(D)
            end
        end

        X = max.(Xmin, min.(X, Xmax))

        for i in 1:N
            fitness[i] = fobj(X[i, :])
            if fitness[i] < BestValue
                BestValue = fitness[i]
                Xfood = copy(X[i, :])
            end
            FES += 1
        end

        fitness, X, fitness_old, X_old = Food_storage(fitness, X, fitness_old, X_old)

        CF = (1 - t / T)^(2 * t / T)

        for i in 1:N
            p = rand(2:5)
            selected_index_p = randperm(N)[1:p]
            Xp = X[selected_index_p, :]
            Xpmean = vec(mean(Xp, dims=1))

            q = rand(10:N)
            selected_index_q = randperm(N)[1:q]
            Xq = X[selected_index_q, :]
            Xqmean = vec(mean(Xq, dims=1))

            if rand() < Epsilon
                X[i, :] .= Xfood .+ CF .* (Xpmean - X[i, :]) .* randn(D)
            else
                X[i, :] .= Xfood .+ CF .* (Xqmean - X[i, :]) .* randn(D)
            end
        end

        X = max.(Xmin, min.(X, Xmax))

        for i in 1:N
            fitness[i] = fobj(X[i, :])
            if fitness[i] < BestValue
                BestValue = fitness[i]
                Xfood = copy(X[i, :])
            end
            FES += 1
        end

        fitness, X, fitness_old, X_old = Food_storage(fitness, X, fitness_old, X_old)

        Conv[t] = BestValue
        t += 1
    end

    return BestValue, Xfood, Conv, FES
end


function Food_storage(fit, X, fit_old, X_old)
    Inx = fit_old .< fit                       
    Indx = repeat(Inx, 1, size(X, 2))          

    X = Indx .* X_old .+ .!Indx .* X           
    fit = Inx .* fit_old .+ .!Inx .* fit       

    fit_old = copy(fit)
    X_old = copy(X)

    return fit, X, fit_old, X_old
end
