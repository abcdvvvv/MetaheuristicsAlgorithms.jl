"""
Deng, Lingyun, and Sanyang Liu. 
"Snow ablation optimizer: A novel metaheuristic technique for numerical optimization and engineering design." 
Expert Systems with Applications 225 (2023): 120069.
"""

function SnowOA(N, max_iter, lb, ub, dim, objfun)
    if length(ub) == 1
        ub = ub * ones(dim)
        lb = lb * ones(dim)
    end

    X = initialization(N, dim, ub, lb)

    Best_pos = zeros(dim)
    Best_score = Inf
    Objective_values = zeros(N)

    Convergence_curve = []

    N1 = floor(Int, N * 0.5)
    Elite_pool = zeros(4, dim)

    for i = 1:N
        Objective_values[i] = objfun(X[i, :])
        if i == 1
            Best_pos = X[i, :]
            Best_score = Objective_values[i]
        elseif Objective_values[i] < Best_score
            Best_pos = X[i, :]
            Best_score = Objective_values[i]
        end
    end

    idx1 = sortperm(Objective_values)
    second_best = X[idx1[2], :]
    third_best = X[idx1[3], :]
    sum1 = sum(X[idx1[1:N1], :], dims=1)
    half_best_mean = sum1 / N1

    Elite_pool[1, :] = Best_pos
    Elite_pool[2, :] = second_best
    Elite_pool[3, :] = third_best
    Elite_pool[4, :] = half_best_mean

    push!(Convergence_curve, Best_score)

    index = collect(1:N)

    Na = N ÷ 2
    Nb = N ÷ 2

    l = 2
    while l <= max_iter
        RB = randn(N, dim)
        T = exp(-l / max_iter)
        k = 1
        DDF = 0.35 * (1 + (5 / 7) * (exp(l / max_iter) - 1)^k / (exp(1) - 1)^k)
        M = DDF * T

        X_centroid = sum(X, dims=1) / N

        index1 = randperm(N)[1:Na]
        index2 = setdiff(index, index1)

        for i = 1:Na
            r1 = rand()
            k1 = rand(1:4)
            for j = 1:dim
                X[index1[i], j] = Elite_pool[k1, j] + RB[index1[i], j] * (r1 * (Best_pos[j] - X[index1[i], j]) + (1 - r1) * (X_centroid[j] - X[index1[i], j]))
            end
        end

        if Na < N
            Na += 1
            Nb -= 1
        end

        if Nb >= 1
            for i = 1:Nb
                r2 = 2 * rand() - 1
                for j = 1:dim
                    X[index2[i], j] = M * Best_pos[j] + RB[index2[i], j] * (r2 * (Best_pos[j] - X[index2[i], j]) + (1 - r2) * (X_centroid[j] - X[index2[i], j]))
                end
            end
        end

        for i = 1:N
            X[i, :] = clamp.(X[i, :], lb, ub)
            Objective_values[i] = objfun(X[i, :])

            if Objective_values[i] < Best_score
                Best_pos = X[i, :]
                Best_score = Objective_values[i]
            end
        end

        idx1 = sortperm(Objective_values)
        second_best = X[idx1[2], :]
        third_best = X[idx1[3], :]
        sum1 = sum(X[idx1[1:N1], :], dims=1)
        half_best_mean = sum1 / N1

        Elite_pool[1, :] = Best_pos
        Elite_pool[2, :] = second_best
        Elite_pool[3, :] = third_best
        Elite_pool[4, :] = half_best_mean

        push!(Convergence_curve, Best_score)
        l += 1
    end

    return Best_score, Best_pos, Convergence_curve
end
