"""
Yadav, Anupam. 
"AEFA: Artificial electric field algorithm for global optimization." 
Swarm and Evolutionary Computation 48 (2019): 93-108.
"""

using Statistics
using LinearAlgebra

function AEFA(N, max_it, lb, ub, D, benchmark)
    Rnorm = 2
    FCheck = 1
    Rpower = 1
    tag = 1


    Fbest = Inf 
    Lbest = zeros(D)  


    X = rand(N, D) .* (ub - lb) .+ lb

    BestValues = []
    MeanValues = []

    V = zeros(N, D)
    fitness = zeros(N)

    for iteration in 1:max_it

        for i in 1:N
            fitness[i] = benchmark(X[i, :])
        end

        if tag == 1
            best, best_X = findmin(fitness)  
        else
            best, best_X = findmax(fitness)  
        end

        if iteration == 1
            Fbest = best
            Lbest = X[best_X, :]
        end

        if tag == 1
            if best < Fbest  
                Fbest = best
                Lbest = X[best_X, :]
            end
        else
            if best > Fbest  
                Fbest = best
                Lbest = X[best_X, :]
            end
        end

        push!(BestValues, Fbest)
        push!(MeanValues, mean(fitness))


        Fmax = maximum(fitness)
        Fmin = minimum(fitness)
        Fmean = mean(fitness)

        if Fmax == Fmin
            Q = ones(N)
        else
            if tag == 1
                best, worst = Fmin, Fmax  
            else
                best, worst = Fmax, Fmin  
            end
            Q = exp.((fitness .- worst) ./ (best - worst))
        end

        Q = Q ./ sum(Q)

        fper = 3
        cbest = FCheck == 1 ? round(Int, N * (fper + (1 - iteration / max_it) * (100 - fper)) / 100) : N
        s = sortperm(Q, rev = true)  
        Qs = Q[s]  

        E = zeros(N, D)
        for i in 1:N
            for ii in 1:cbest
                j = s[ii]
                if j != i
                    R = norm(X[i, :] - X[j, :], Rnorm)  
                    for k in 1:D
                        E[i, k] += rand() * Q[j] * ((X[j, k] - X[i, k]) / (R^Rpower + eps()))
                    end
                end
            end
        end

        alfa = 30
        K0 = 500
        K = K0 * exp(-alfa * iteration / max_it)

        a = E * K

        V = rand(N, D) .* V .+ a
        X = X .+ V
        X = clamp.(X, lb, ub)

        swarm = zeros(N, 1, 2)
        swarm[:, 1, 1] = X[:, 1]
        swarm[:, 1, 2] = X[:, 2]
    end

    return Fbest, Lbest, BestValues
end