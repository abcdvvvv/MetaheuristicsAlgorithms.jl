"""
Ahmadianfar, Iman, Omid Bozorg-Haddad, and Xuefeng Chu. 
"Gradient-based optimizer: A new metaheuristic optimization algorithm." 
Information Sciences 540 (2020): 131-159.

"""
function GBO(nP, MaxIt, lb, ub, dim, fobj)
    # Initialization
    nV = dim                        # Number of Variables
    pr = 0.5                        # Probability Parameter
    lb = ones(dim) .* lb            # Lower boundary 
    ub = ones(dim) .* ub            # Upper boundary
    Cost = zeros(nP)
    X = initialization(nP, nV, ub, lb) # Initialize the set of random solutions
    Convergence_curve = zeros(MaxIt)

    for i in 1:nP
        Cost[i] = fobj(X[i, :])      # Calculate the value of objective function
    end

    sort_idx = sortperm(Cost)
    Best_Cost = Cost[sort_idx[1]]    # Best fitness value
    Best_X = X[sort_idx[1], :]       # Best position
    Worst_Cost = Cost[sort_idx[end]] # Worst fitness value
    Worst_X = X[sort_idx[end], :]    # Worst position

    # Main Loop
    for it in 1:MaxIt
        beta = 0.2 + (1.2 - 0.2) * (1 - (it / MaxIt)^3)^2   # Eq.(14.2)
        alpha = abs(beta * sin(3 * π / 2 + sin(3 * π / 2 * beta))) # Eq.(14.1)

        for i in 1:nP
            A1 = rand(1:nP, 4)      # Four positions randomly selected from population
            r1, r2, r3, r4 = A1     # Assigning the random indices

            Xm = (X[r1, :] + X[r2, :] + X[r3, :] + X[r4, :]) / 4 # Average of four random positions
            ro = alpha * (2 * rand() - 1)
            ro1 = alpha * (2 * rand() - 1)
            eps = 5e-3 * rand()      # Randomization epsilon

            DM = rand() * ro * (Best_X - X[r1, :])
            Flag = 1
            GSR = GradientSearchRule(ro1, Best_X, Worst_X, X[i, :], X[r1, :], DM, eps, Xm, Flag)
            DM = rand() * ro * (Best_X - X[r1, :])
            X1 = X[i, :] - GSR + DM  # Eq.(25)

            DM = rand() * ro * (X[r1, :] - X[r2, :])
            Flag = 2
            GSR = GradientSearchRule(ro1, Best_X, Worst_X, X[i, :], X[r1, :], DM, eps, Xm, Flag)
            DM = rand() * ro * (X[r1, :] - X[r2, :])
            X2 = Best_X - GSR + DM  # Eq.(26)

            Xnew = zeros(nV)
            for j in 1:nV
                ro = alpha * (2 * rand() - 1)
                X3 = X[i, j] - ro * (X2[j] - X1[j])
                ra, rb = rand(), rand()
                Xnew[j] = ra * (rb * X1[j] + (1 - rb) * X2[j]) + (1 - ra) * X3 # Eq.(27)
            end

            # Local Escaping Operator (LEO)
            if rand() < pr
                k = rand(1:nP)
                f1 = -1 + 2 * rand()
                f2 = -1 + 2 * rand()
                ro = alpha * (2 * rand() - 1)
                Xk = lb .+ (ub - lb) .* rand(nV) # Eq.(28.8)

                L1 = rand() < 0.5
                u1 = L1 * 2 * rand() + (1 - L1)
                u2 = L1 * rand() + (1 - L1)
                u3 = L1 * rand() + (1 - L1)
                L2 = rand() < 0.5
                Xp = (1 - L2) * X[k, :] + L2 * Xk # Eq.(28.7)

                if u1 < 0.5
                    Xnew = Xnew + f1 * (u1 * Best_X - u2 * Xp) + f2 * ro * (u3 * (X2 - X1) + u2 * (X[r1, :] - X[r2, :])) / 2
                else
                    Xnew = Best_X + f1 * (u1 * Best_X - u2 * Xp) + f2 * ro * (u3 * (X2 - X1) + u2 * (X[r1, :] - X[r2, :])) / 2
                end
            end

            # Check if solutions go outside the search space and bring them back
            Flag4ub = Xnew .> ub
            Flag4lb = Xnew .< lb
            Xnew = (Xnew .* (Flag4ub .+ Flag4lb .== 0)) .+ ub .* Flag4ub .+ lb .* Flag4lb
            Xnew_Cost = fobj(Xnew)

            # Update the best position
            if Xnew_Cost < Cost[i]
                X[i, :] = Xnew
                Cost[i] = Xnew_Cost
                if Cost[i] < Best_Cost
                    Best_X = X[i, :]
                    Best_Cost = Cost[i]
                end
            end

            # Update the worst position
            if Cost[i] > Worst_Cost
                Worst_X = X[i, :]
                Worst_Cost = Cost[i]
            end
        end

        Convergence_curve[it] = Best_Cost
        # Show iteration information
        # println("Iteration $it: Best Fitness = ", Convergence_curve[it])
    end

    return Best_Cost, Best_X, Convergence_curve
end

# Gradient Search Rule Function
function GradientSearchRule(ro1, Best_X, Worst_X, X, Xr1, DM, eps, Xm, Flag)
    nV = length(X)
    Delta = 2 .* rand(nV) .* abs.(Xm - X)           # Eq.(16.2)
    Step = ((Best_X - Xr1) + Delta) / 2            # Eq.(16.1)
    DelX = rand(nV) .* abs.(Step)                   # Eq.(16)

    GSR = randn() .* ro1 .* (2 .* DelX .* X) ./ (Best_X - Worst_X .+ eps) # Gradient Search Rule Eq.(15)
    if Flag == 1
        Xs = X - GSR + DM    # Eq.(21)
    else
        Xs = Best_X - GSR + DM
    end

    yp = rand() .* (0.5 .* (Xs + X) .+ rand() .* DelX) # Eq.(22.6)
    yq = rand() .* (0.5 .* (Xs + X) .- rand() .* DelX) # Eq.(22.7)
    GSR = randn() .* ro1 .* (2 .* DelX .* X) ./ (yp - yq .+ eps) # Eq.(23)

    return GSR
end
