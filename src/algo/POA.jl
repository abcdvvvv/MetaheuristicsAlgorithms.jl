"""
Trojovský, Pavel, and Mohammad Dehghani. 
"Pelican optimization algorithm: A novel nature-inspired algorithm for engineering applications." 
Sensors 22, no. 3 (2022): 855.
"""

function POA(npop::Int, max_iter::Int, lb::Union{Real,AbstractVector}, ub::Union{Real,AbstractVector}, dim::Int, objfun)
    lb = ones(dim) * lb
    ub = ones(dim) * ub

    X = initialization(npop, dim, ub, lb)

    fit = [objfun(X[i, :]) for i = 1:npop]

    fbest = Inf
    Xbest = []

    best_so_far = zeros(max_iter)
    average = zeros(max_iter)

    for t = 1:max_iter
        best, location = findmin(fit)
        if t == 1 || best < fbest
            fbest = best
            Xbest = X[location, :]
        end

        k = randperm(npop)[1]
        X_FOOD = X[k, :]
        F_FOOD = fit[k]

        for i = 1:npop
            I = round(Int, 1 + rand())
            if fit[i] > F_FOOD
                X_new = X[i, :] .+ rand() .* (X_FOOD - I .* X[i, :])
            else
                X_new = X[i, :] .+ rand() .* (X[i, :] - X_FOOD)
            end
            X_new = clamp.(X_new, lb, ub)

            f_new = objfun(X_new)
            if f_new <= fit[i]
                X[i, :] = X_new
                fit[i] = f_new
            end

            X_new = X[i, :] .+ 0.2 * (1 - t / max_iter) .* (2 * rand(dim) .- 1) .* X[i, :]
            X_new = clamp.(X_new, lb, ub)

            f_new = objfun(X_new)
            if f_new <= fit[i]
                X[i, :] = X_new
                fit[i] = f_new
            end
        end

        best_so_far[t] = fbest
        # println("best_so_far[$t]  ", fbest)
        average[t] = mean(fit)
    end

    return fbest, Xbest, best_so_far
end