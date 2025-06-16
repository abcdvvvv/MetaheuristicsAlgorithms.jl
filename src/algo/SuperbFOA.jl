"""
Jia, Heming, et al. 
"Superb Fairy-wren Optimization Algorithm: a novel metaheuristic algorithm for solving feature selection problems." 
Cluster Computing 28.4 (2025): 246.
"""

function SuperbFOA(N, max_iter, lb, ub, dim, objfun)
    MaxFEs = max_iter * N
    curve = zeros(MaxFEs)
    X = initialization(N, dim, ub, lb)
    Xnew = zeros(N, dim)
    best_fitness = Inf
    best_position = zeros(dim)
    fitness = zeros(N)
    FEs = 1
    lb = fill(lb, dim)
    ub = fill(ub, dim)

    for i = 1:N
        fitness[i] = objfun(X[i, :])
        if fitness[i] < best_fitness
            best_fitness = fitness[i]
            best_position = copy(X[i, :])
        end
        FEs += 1
        if FEs <= MaxFEs
            curve[FEs] = best_fitness
        end
    end

    while FEs <= MaxFEs
        C = 0.8
        r1 = rand()
        r2 = rand()
        w = (π / 2) * (FEs / MaxFEs)
        k = 0.2 * sin(π / 2 - w)
        l = 0.5 * levy(N, dim, 1.5)
        y = rand(1:N)
        c1 = rand()
        T = 0.5
        m = FEs / MaxFEs * 2
        p = sin.(ub .- lb) .* 2 .+ (ub .- lb) .* m
        Xb = best_position
        XG = best_position .* C

        for i = 1:N
            if T < c1
                Xnew[i, :] = X[i, :] .+ lb .+ (ub .- lb) .* rand(1, dim)
            else
                s = r1 * 20 + r2 * 20
                if s > 20
                    Xnew[i, :] = Xb .+ X[i, :] .* l[y, :] .* k
                else
                    Xnew[i, :] = XG .+ (Xb .- X[i, :]) .* p
                end
            end
        end

        X .= Xnew

        for i = 1:N
            Xnew[i, :] = clamp.(Xnew[i, :], lb, ub)
            fitness[i] = objfun(Xnew[i, :])
            if fitness[i] < best_fitness
                best_fitness = fitness[i]
                best_position = copy(Xnew[i, :])
            end
            FEs += 1
            if FEs <= MaxFEs
                curve[FEs] = best_fitness
            else
                break
            end
        end

        if FEs >= MaxFEs
            break
        end
    end

    return best_fitness, best_position, curve
end
