"""
Zhao, Shijie, Tianran Zhang, Liang Cai, and Ronghua Yang. 
"Triangulation topology aggregation optimizer: A novel mathematics-based meta-heuristic algorithm for 
continuous optimization and engineering applications." 
Expert Systems with Applications 238 (2024): 121744.
"""

using Random

function TTAO(PopSize, T, Low, Up, Dim, fobj)
    N = floor(Int, PopSize / 3)  
    X1 = rand(N, Dim) .* (Up - Low) .+ Low  
    Convergence_curve = zeros(T)
    t = 1
    Xbest = zeros(Dim)
    fbest = Inf

    while t <= T
        l = 9 * exp(-t / T)  
        X2 = zeros(N, Dim)
        X3 = zeros(N, Dim)
        
        for i in 1:N
            theta = rand(Dim) .* pi
            h1 = cos.(theta)
            h2 = cos.(theta .+ pi / 3)
            X2[i, :] = X1[i, :] .+ l .* h1
            X3[i, :] = X1[i, :] .+ l .* h2
        end

        X2 = clamp.(X2, Low, Up)
        X3 = clamp.(X3, Low, Up)
        r1, r2 = rand(), rand()
        X4 = r1 .* X1 .+ r2 .* X2 .+ (1 - r1 - r2) .* X3
        X4 = clamp.(X4, Low, Up)

        X1_fit = [fobj(X1[i, :]) for i in 1:N]
        X2_fit = [fobj(X2[i, :]) for i in 1:N]
        X3_fit = [fobj(X3[i, :]) for i in 1:N]
        X4_fit = [fobj(X4[i, :]) for i in 1:N]

        X = [X1; X2; X3; X4]
        fit = vcat(X1_fit, X2_fit, X3_fit, X4_fit)
        sorted_indices = sortperm(fit)  
        sorted_fit = fit[sorted_indices]  

        X_best_1 = zeros(N, Dim)
        X_best_2 = zeros(N, Dim)
        best_fit_1 = zeros(N)
        best_fit_2 = zeros(N)

        for i in 1:N
            X_best_1[i, :] = X[sorted_indices[i], :]
            best_fit_1[i] = sorted_fit[i]
            X_best_2[i, :] = X[sorted_indices[i + 1], :]
            best_fit_2[i] = sorted_fit[i + 1]
        end

        X_G = zeros(N, Dim)
        X_fit_G = zeros(N)
        for i in 1:N
            r = rand(1, Dim)
            X_new = X_best_1[[1:i-1; i+1:end], :]
            l1 = rand(1:N - 1)
            X_G[i, :] = r' .* X_best_1[i, :] .+ (1 .- r)' .* X_new[l1, :]
            X_G[i, :] = clamp.(X_G[i, :], Low, Up)
            X_fit_G[i] = fobj(X_G[i, :])

            if X_fit_G[i] < best_fit_1[i]
                X_best_1[i, :] = X_G[i, :]
                best_fit_1[i] = X_fit_G[i]
            elseif X_fit_G[i] < best_fit_2[i]
                X_best_2[i, :] = X_G[i, :]
            end
        end

        X_C = zeros(N, Dim)
        X_fit_C = zeros(N)
        for i in 1:N
            a = (exp(1) - exp(1)^3) / (T - 1)
            b = exp(1)^3 - a
            alpha = log(a * t + b)
            X_C[i, :] = X_best_1[i, :] .+ alpha .* (X_best_1[i, :] - X_best_2[i, :])
            X_C[i, :] = clamp.(X_C[i, :], Low, Up)
            X_fit_C[i] = fobj(X_C[i, :])

            if X_fit_C[i] < best_fit_1[i]
                X_best_1[i, :] = X_C[i, :]
                best_fit_1[i] = X_fit_C[i]
            end
        end

        N00 = PopSize - N * 3
        if N00 > 0
            X00 = rand(N00, Dim) .* (Up - Low) .+ Low
            X00_fit = [fobj(X00[i, :]) for i in 1:N00]

            X_1_0 = vcat(X_best_1, X00)
            X_1_0_fit = vcat(best_fit_1, X00_fit)
            sorted_fit_0, sorted_indices_0 = sort(X_1_0_fit), sortperm(X_1_0_fit)
            X_best_1 = X_1_0[sorted_indices_0[1:N], :]
            best_fit_1 = sorted_fit_0[1:N]
        end

        X1 = X_best_1  
        index1 = argmin(best_fit_1)
        Xbest = X1[index1, :]
        fbest = best_fit_1[index1]
        Convergence_curve[t] = fbest

        t += 1
    end

    return fbest, Xbest, Convergence_curve
end
