"""
Yadav, Anupam. 
"AEFA: Artificial electric field algorithm for global optimization." 
Swarm and Evolutionary Computation 48 (2019): 93-108.
"""

using Statistics
using LinearAlgebra

function AEFA(N, max_it, lb, ub, D, benchmark)#(func_num, N, max_it, FCheck, tag, Rpower)
    # Variables initialization
    Rnorm = 2
    FCheck = 1
    Rpower = 1
    tag = 1


    Fbest = Inf 
    Lbest = zeros(D)  

    # Dimension, lower and upper bounds
    # lb, ub, D = benchmark_range(func_num)

    # Random initialization of charge population
    X = rand(N, D) .* (ub - lb) .+ lb

    BestValues = []
    MeanValues = []

    V = zeros(N, D)
    fitness = zeros(N)

    for iteration in 1:max_it

        # Evaluate fitness of charged particles
        for i in 1:N
            # fitness[i] = benchmark(X[i, :], func_num, D)
            fitness[i] = benchmark(X[i, :])
        end

        if tag == 1
            best, best_X = findmin(fitness)  # Minimization
        else
            best, best_X = findmax(fitness)  # Maximization
        end

        if iteration == 1
            Fbest = best
            Lbest = X[best_X, :]
        end

        if tag == 1
            if best < Fbest  # Minimization
                Fbest = best
                Lbest = X[best_X, :]
            end
        else
            if best > Fbest  # Maximization
                Fbest = best
                Lbest = X[best_X, :]
            end
        end

        push!(BestValues, Fbest)
        push!(MeanValues, mean(fitness))

        # Charge calculation
        Fmax = maximum(fitness)
        Fmin = minimum(fitness)
        Fmean = mean(fitness)

        if Fmax == Fmin
            Q = ones(N)
        else
            if tag == 1
                best, worst = Fmin, Fmax  # Minimization
            else
                best, worst = Fmax, Fmin  # Maximization
            end
            Q = exp.((fitness .- worst) ./ (best - worst))
        end

        Q = Q ./ sum(Q)

        # Force application
        fper = 3
        cbest = FCheck == 1 ? round(Int, N * (fper + (1 - iteration / max_it) * (100 - fper)) / 100) : N
        # Qs, s = sort(Q, rev = true)
        s = sortperm(Q, rev = true)  
        Qs = Q[s]  

        E = zeros(N, D)
        for i in 1:N
            for ii in 1:cbest
                j = s[ii]
                if j != i
                    R = norm(X[i, :] - X[j, :], Rnorm)  # Euclidean distance
                    for k in 1:D
                        E[i, k] += rand() * Q[j] * ((X[j, k] - X[i, k]) / (R^Rpower + eps()))
                    end
                end
            end
        end

        # Coulomb constant calculation
        alfa = 30
        K0 = 500
        K = K0 * exp(-alfa * iteration / max_it)

        # Acceleration calculation
        a = E * K

        # Charge movement
        V = rand(N, D) .* V .+ a
        X = X .+ V
        X = clamp.(X, lb, ub)

        # Plotting (Optional)
        swarm = zeros(N, 1, 2)
        swarm[:, 1, 1] = X[:, 1]
        swarm[:, 1, 2] = X[:, 2]
    end

    return Fbest, Lbest, BestValues
end