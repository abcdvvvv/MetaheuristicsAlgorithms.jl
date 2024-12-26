"""
Lian, Junbo, Ting Zhu, Ling Ma, Xincan Wu, Ali Asghar Heidari, Yi Chen, Huiling Chen, and Guohua Hui. 
"The educational competition optimizer." 
International Journal of Systems Science 55, no. 15 (2024): 3185-3222.
"""

using Random
using LinearAlgebra

function ECO(N::Int, Max_iter::Int, lb::Union{Int, AbstractVector}, ub::Union{Int, AbstractVector}, dim::Int, fobj::Function)

    # Choosing the Nearest School
    function close(t::Vector{Float64}, G::Int, X::Matrix{Float64})
        m = copy(X[1, :])  # Initialize m with the first school's position
    
        if G == 1
            for s in 1:G1Number
                school = X[s, :]
                if sum(abs.(m .- t)) > sum(abs.(school .- t))
                    m = school
                end
            end
        else
            for s in 1:G2Number
                school = X[s, :]
                if sum(abs.(m .- t)) > sum(abs.(school .- t))
                    m = school
                end
            end
        end
        return m
    end

    # Logistic chaotic mapping initialization
    function initializationLogistic(pop::Int, dim::Int, ub::Union{Int, AbstractVector}, lb::Union{Int, AbstractVector})
        Boundary_no = length(ub)  # Number of boundaries
        Positions = zeros(Float64, pop, dim)  # Initialize positions matrix
    
        for i in 1:pop
            for j in 1:dim
                x0 = rand()
                a = 4.0
                x = a * x0 * (1 - x0)
                if Boundary_no == 1
                    Positions[i, j] = (ub[1] - lb[1]) * x + lb[1]
                    if Positions[i, j] > ub[1]
                        Positions[i, j] = ub[1]
                    end
                    if Positions[i, j] < lb[1]
                        Positions[i, j] = lb[1]
                    end
                else
                    Positions[i, j] = (ub[j] - lb[j]) * x + lb[j]
                    if Positions[i, j] > ub[j]
                        Positions[i, j] = ub[j]
                    end
                    if Positions[i, j] < lb[j]
                        Positions[i, j] = lb[j]
                    end
                end
                x0 = x  # Update x0 for the next iteration (this line is optional)
            end
        end
        return Positions
    end

    # Levy search strategy
    function Levy(d::Int)
        beta = 1.5
        sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta)
        u = randn(d) * sigma
        v = randn(d)
        step = u ./ abs.(v) .^ (1 / beta)
        return step
    end

    H = 0.5  # Learning habit boundary
    G1 = 0.2  # Proportion of primary school (Stage 1)
    G2 = 0.1  # Proportion of middle school (Stage 2)

    # Number of G1 and G2
    G1Number = round(Int, N * G1)
    G2Number = round(Int, N * G2)

    # Extend lb and ub if they are scalars
    if length(ub) == 1
        # ub .= fill(ub[1], dim)
        ub = fill(ub[1], dim)
        # lb .= fill(lb[1], dim)
        lb = fill(lb[1], dim)
    end

    # Initialization
    X0 = initializationLogistic(N, dim, ub, lb)  # Logistic Chaos Mapping Initialization
    X = copy(X0)

    # Compute initial fitness values
    fitness = zeros(Float64, N)
    fitness_new = zeros(Float64, N)
    for i in 1:N
        fitness[i] = fobj(X[i, :])
    end

    index = sortperm(fitness)  
    fitness = sort(fitness)     

    GBestF = fitness[1]  # Global best fitness value

    for i in 1:N
        X[i, :] = X0[index[i], :]
    end

    curve = zeros(Float64, Max_iter)
    Best_score = Inf
    Best_pos = zeros(dim)

    GBestX = X[1, :]  # Global best position
    GWorstX = X[end, :]  # Global worst position
    X_new = copy(X)


    # Start search
    for i in 1:Max_iter
        # if i % 100 == 0
        #     println("At iteration $i the fitness is $(curve[i - 1])")
        # end
        
        R1 = rand()
        R2 = rand()
        P = 4 * randn() * (1 - i / Max_iter)
        E = (π * i) / (P * Max_iter)
        w = 0.1 * log(2 - (i / Max_iter))

        for j in 1:N
            # Stage 1: Primary school competition
            if i % 3 == 1
                if j >= 1 && j <= G1Number  # Primary school site selection strategies
                    X_new[j, :] = X[j, :] .+ w * (mean(X[j, :]) .- X[j, :]) .* Levy(dim)
                else  # Competitive strategies for students (Stage 1)
                    X_new[j, :] = X[j, :] .+ w * (close(X[j, :], 1, X) .- X[j, :]) .* randn()
                end
            
            # Stage 2: Middle school competition
            elseif i % 3 == 2
                if j >= 1 && j <= G2Number  # Middle school site selection strategies
                    X_new[j, :] = X[j, :] .+ (GBestX .- mean(X, dims=1)') * exp(i / Max_iter - 1) .* Levy(dim)
                else
                    if R1 < H
                        X_new[j, :] = X[j, :] .- w * close(X[j, :], 2, X) .- P * (E * w * close(X[j, :], 2, X) .- X[j, :])
                    else
                        X_new[j, :] = X[j, :] .- w * close(X[j, :], 2, X) .- P * (w * close(X[j, :], 2, X) .- X[j, :])
                    end
                end
            
            # Stage 3: High school competition
            else
                if j >= 1 && j <= G2Number  # High school site selection strategies
                    X_new[j, :] = X[j, :] .+ (GBestX .- X[j, :]) .* randn() .- (GBestX .- X[j, :]) .* randn()
                else  # Competitive strategies for students (Stage 3)
                    if R2 < H
                        X_new[j, :] = GBestX .- P * (E * GBestX .- X[j, :])
                    else
                        X_new[j, :] = GBestX .- P * (GBestX .- X[j, :])
                    end
                end
            end

            # Boundary control
            Flag4ub = X_new[j, :] .> ub
            Flag4lb = X_new[j, :] .< lb
            X_new[j, :] = (X_new[j, :] .* .!(Flag4ub .| Flag4lb)) .+ ub .* Flag4ub .+ lb .* Flag4lb

            # Finding the best location so far
            fitness_new[j] = fobj(X_new[j, :])
            if fitness_new[j] > fitness[j]
                fitness_new[j] = fitness[j]
                X_new[j, :] = X[j, :]
            end

            if fitness_new[j] < GBestF
                GBestF = fitness_new[j]
                GBestX = X_new[j, :]
            end
        end

        X = copy(X_new)
        fitness = copy(fitness_new)
        curve[i] = GBestF
        Best_pos = GBestX
        Best_score = curve[end]

        # Sorting and updating

        index = sortperm(fitness)  
        fitness = fitness[index]    
        for j in 1:N
            X0[j, :] = X[index[j], :]
        end
        X = copy(X0)
    end
    
    # return avg_fitness_curve, Best_pos, Best_score, curve, search_history, fitness_history
    return Best_score, Best_pos, curve
end