"""
Fu, Youfa, Dan Liu, Jiadui Chen, and Ling He. 
"Secretary bird optimization algorithm: a new metaheuristic for solving global optimization problems." 
Artificial Intelligence Review 57, no. 5 (2024): 1-102.
"""

function SBOA(N, T, lb, ub, dim, fitness) 
    lb = fill(lb, dim)
    ub = fill(ub, dim)

    X = initialization(N, dim, lb, ub)
    fit = zeros(N) 
    Bast_P = zeros(dim)
    for i in 1:N
        fit[i] = fitness(vec(X[i, :]))
    end
    SBOA_curve = zeros(T)
    fbest = Inf

    for t in 1:T
        CF = (1 - t / T)^(2 * t / T)
        
        best, location = findmin(fit)
        if t == 1
            Bast_P = X[location, :]     
            fbest = best                
        elseif best < fbest
            fbest = best
            Bast_P = X[location, :]
        end

        for i in 1:N
            if t < T / 3  
                X_random_1 = X[rand(1:N), :]
                X_random_2 = X[rand(1:N), :]
                R1 = rand(dim)
                X1 = X[i, :] .+ (X_random_1 .- X_random_2) .* R1
                X1 = clamp.(X1, lb, ub)
            elseif t > T / 3 && t < 2 * T / 3  
                RB = randn(dim)
                X1 = Bast_P .+ exp((t / T)^4) .* (RB .- 0.5) .* (Bast_P .- X[i, :])
                X1 = clamp.(X1, lb, ub)
            else  
                RL = 0.5 * Levy(dim)
                X1 = Bast_P .+ CF .* X[i, :] .* RL
                X1 = clamp.(X1, lb, ub)
            end

            f_newP1 = fitness(X1)
            if f_newP1 <= fit[i]
                X[i, :] = X1
                fit[i] = f_newP1
            end
        end

        r = rand()
        Xrandom = X[rand(1:N), :]
        for i in 1:N
            if r < 0.5
                RB = rand(dim)
                X2 = Bast_P .+ (1 - t / T)^2 .* (2 * RB .- 1) .* X[i, :]
                X2 = clamp.(X2, lb, ub)
            else
                K = round(Int, 1 + rand())
                R2 = rand(dim)
                X2 = X[i, :] .+ R2 .* (Xrandom .- K .* X[i, :])
                X2 = clamp.(X2, lb, ub)
            end

            f_newP2 = fitness(X2)
            if f_newP2 <= fit[i]
                X[i, :] = X2
                fit[i] = f_newP2
            end
        end

        SBOA_curve[t] = fbest
    end

    Best_score = fbest
    Best_pos = Bast_P
    return Best_score, Best_pos, SBOA_curve
end

function Levy(d)
    β = 1.5
    σ = (gamma(1 + β) * sin(π * β / 2) / (gamma((1 + β) / 2) * β * 2^((β - 1) / 2)))^(1 / β)
    u = randn(d) * σ
    v = randn(d)
    step = u ./ abs.(v).^(1 / β)
    return step
end