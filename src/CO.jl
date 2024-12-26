"""
Akbari, Mohammad Amin, Mohsen Zare, Rasoul Azizipanah-Abarghooee, Seyedali Mirjalili, and Mohamed Deriche. 
"The cheetah optimizer: A nature-inspired metaheuristic algorithm for large-scale optimization problems." 
Scientific reports 12, no. 1 (2022): 10953.
"""
mutable struct Individual
    Position::Vector{Float64}
    Cost::Float64
end

function CO(n, MaxIt, lb, ub, D, fobj)
    empty_individual = Individual([], Inf)
    BestSol = Individual([], Inf)
    pop = [empty_individual for _ in 1:n]

    MaxEEs = n * MaxIt

    xd = zeros(D)


    if length(lb) == 1
        ub = ub .* ones(D)    # Lower Bound of Decision Variables
        lb = lb .* ones(D)    # Upper Bound of Decision Variables
    end

    # Generate initial population of cheetahs
    for i in 1:n
        pop[i].Position = lb .+ rand(D) .* (ub .- lb)
        pop[i].Cost = fobj(pop[i].Position)
        if pop[i].Cost < BestSol.Cost
            BestSol = pop[i]  # Initial leader position
        end
    end

    pop1 = copy(pop)        # Population's initial home position
    BestCost = Float64[]    # Leader fitness value in a current hunting period
    # X_best = copy(BestSol)  # Prey solution so far
    X_best = deepcopy(BestSol)
    Globest = Float64[]     # Prey fitness value so far

    t = 0                    # Hunting time counter
    it = 1                   # Iteration counter
    T = ceil(Int, D / 10) * 60  # Hunting time
    FEs = 0                  # Counter for function evaluations

    # Main loop
    while FEs <= MaxEEs
        # m = rand(1:n)        # Select a random member of cheetahs
        m = rand(2:n)        # Select a random member of cheetahs
        i0 = rand(1:n, m)  # Select m random members of cheetahs (Algorithm 1, L#9)
        for k in 1:m         # Algorithm 1, L#10
            i = rand(1:n)
            i = i0[k]

            # Neighbor agent selection
            if k == length(i0) && k != 1
                a = i0[k - 1]  # If k is the last element, select the previous one
            else
                a = i0[k + 1]  # Otherwise, select the next one
            end

            X = pop[i].Position    # The current position of i-th cheetah
            X1 = pop[a].Position    # The neighbor position
            Xb = BestSol.Position    # The leader position
            Xbest = X_best.Position  # The prey position

            kk = 0
            # Uncomment the following statements, it may improve the performance of CO
            # if i <= 2 && t > 2 && t > ceil(0.2 * T + 1) && abs(BestCost[t - 2] - BestCost[t - ceil(0.2 * T + 1)]) <= 0.0001 * Globest[t - 1]
            if i <= 2 && t > 2 && t > Int(ceil(0.2 * T + 1)) && abs(BestCost[t - 2] - BestCost[t - Int(ceil(0.2 * T + 1))]) <= 0.0001 * Globest[t - 1]
                X = X_best.Position
                kk = 0
            elseif i == 3
                X = BestSol.Position
                kk = -0.1 * rand() * t / T
            else
                kk = 0.25
            end

            if it % 100 == 0 || it == 1
                xd = randperm(D)
            end

            Z = copy(X)

            for j in xd  # Select arbitrary set of arrangements
                r_Hat = randn()  # Randomization parameter
                r1 = rand()
                alpha = (k == 1) ? 0.0001 * t / T * (ub[j] - lb[j]) : 0.0001 * t / T * abs(Xb[j] - X[j]) + 0.001 * round(rand() > 0.9)

                r = randn()
                r_Check = abs(r)^exp(r / 2) * sin(2 * π * r)  # Turning factor
                beta = X1[j] - X[j]  # Interaction factor

                h0 = exp(2 - 2 * t / T)
                H = abs(2 * r1 * h0 - h0)

                r2 = rand()
                r3 = kk + rand()

                # Strategy selection mechanism
                if r2 <= r3
                    r4 = 3 * rand()
                    if H > r4
                        Z[j] = X[j] + r_Hat^-1 * alpha  # Search
                    else
                        Z[j] = Xbest[j] + r_Check * beta  # Attack
                    end
                else
                    Z[j] = X[j]  # Sit & wait
                end
            end

            # Check the limits
            Z = clamp.(Z, lb, ub)

            # Evaluate the new position
            NewSol = Individual(Z, fobj(Z))
            if NewSol.Cost < pop[i].Cost
                pop[i] = NewSol
                if pop[i].Cost < BestSol.Cost
                    BestSol = pop[i]
                end
            end
            FEs += 1
        end

        t += 1  # Increment hunting time

        # Leave the prey and go back home
        if t > T && t - round(T) - 1 >= 1 && t > 2
            if abs(BestCost[t - 1] - BestCost[t - round(T) - 1]) <= abs(0.01 * BestCost[t - 1])
                best = X_best.Position
                j0 = rand(1:D, ceil(Int, D / 10 * rand()))
                best[j0] = lb[j0] .+ rand(length(j0)) .* (ub[j0] .- lb[j0])
                BestSol.Cost = fobj(best)
                BestSol.Position = best  # Leader's new position
                FEs += 1

                i0 = rand(1:n, round(1 * n))
                # Go back home
                pop[i0[n - m + 1:n]] = pop1[i0[1:m]]  # Some members back to their initial positions

                pop[i] = X_best  # Substitute the member i by the prey
                t = 1  # Reset the hunting time
            end
        end

        it += 1  # Increment iteration counter

        # Update the prey (global best) position
        if BestSol.Cost < X_best.Cost
            X_best = BestSol
        end
        push!(BestCost, BestSol.Cost)
        push!(Globest, X_best.Cost)

        # Display
        # if it % 500 == 0
        #     println(" FEs>> ", FEs, "   BestCost = ", Globest[end])
        # end
    end

    return X_best.Cost, X_best.Position, BestCost
end