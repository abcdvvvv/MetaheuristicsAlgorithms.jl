"""
Hu, Gang, Yuxuan Guo, Guo Wei, and Laith Abualigah. 
"Genghis Khan shark optimizer: a novel nature-inspired algorithm for engineering optimization." 
Advanced Engineering Informatics 58 (2023): 102210.
"""
function GKSO(n, Max_iter, lb, ub, dim, fhd)#(n, lb, ub, dim, fhd, Max_iter, args...)
    # start_time = time()

    # lb = ones(1, dim) .* lb  # Lower limit for variables
    # ub = ones(1, dim) .* ub  # Upper limit for variables
    lb = ones(dim) .* lb  # Lower limit for variables
    ub = ones(dim) .* ub  # Upper limit for variables
    VarSize = (dim)       # Size of Decision Variables Matrix

    # Initialization
    PopPos = zeros(n, dim)
    PopFit = zeros(n)
    for i in 1:n
        # PopPos[i, :] = rand(1, dim) .* (ub - lb) + lb
        PopPos[i, :] = rand(dim) .* (ub - lb) + lb
        # PopFit[i] = fhd(PopPos[i, :]', args...)
        # println("typeof(PopPos[i, :]')", typeof(PopPos[i, :]))
        # println("size(PopPos[i, :]')", size(PopPos[i, :]))
        PopFit[i] = fhd(vec(PopPos[i, :]'))
    end

    Best_score = Inf
    Best_pos = []

    for i in 1:n
        if PopFit[i] <= Best_score
            Best_score = PopFit[i]
            Best_pos = PopPos[i, :]
        end
    end

    curve = zeros(Max_iter)
    h = [0.1]

    # Main loop
    for it in 1:Max_iter
        h = append!(h, 1 - 2 * h[end]^4)
        p = 2 * (1 - (it / Max_iter)^(1 / 4)) + abs(h[end]) * ((it / Max_iter)^(1 / 4) - (it / Max_iter)^3)
        beta = 0.2 + (1.2 - 0.2) * (1 - (it / Max_iter)^3)^2
        alpha = abs(beta * sin((3 * π / 2 + sin(3 * π / 2 * beta))))

        # Hunting stage: Expand search scope and search for potential prey positions
        newPopPos = zeros(n, dim)
        newPopFit = zeros(n)
        for i in 1:n
            newPopPos[i, :] = PopPos[i, :] + (lb + rand() * (ub - lb)) / it
            # Handling boundary violations
            newPopPos[i, :] = clamp.(newPopPos[i, :], lb, ub)
            # Evaluate fitness
            # newPopFit[i] = fhd(newPopPos[i, :]', args...)
            newPopFit[i] = fhd(vec(newPopPos[i, :]'))
            # Compare the fitness
            if newPopFit[i] < PopFit[i]
                PopFit[i] = newPopFit[i]  # Update fitness
                PopPos[i, :] = newPopPos[i, :]  # Update positions
            end
        end

        # Best Position Attraction Effect: Approaching the Best Position
        GKS_Pos = zeros(n, dim)
        for i in 1:n
            s = (1.5 * PopFit[i]^rand())
            s = real(s)
            if i == 1
                newPopPos[i, :] = (Best_pos - PopPos[i, :]) * s
            else
                GKS_Pos[i, :] = (Best_pos - PopPos[i, :]) * s
                newPopPos[i, :] = (GKS_Pos[i, :] + newPopPos[i - 1, :]) / 2
            end
            # Handling boundary violations
            if all(newPopPos[i, :] .>= lb) && all(newPopPos[i, :] .<= ub)
                # newPopFit[i] = fhd(newPopPos[i, :]', args...)
                newPopFit[i] = fhd(vec(newPopPos[i, :]'))
                if newPopFit[i] < PopFit[i]
                    PopPos[i, :] = newPopPos[i, :]
                    PopFit[i] = newPopFit[i]
                end
            end
        end

        # Foraging stage: parabolic foraging
        for i in 1:n
            TF = (rand() > 0.5) * 2 - 1
            # newPopPos[i, :] = Best_pos + rand(1, dim) .* (Best_pos - PopPos[i, :]) + TF * p^2 * (Best_pos - PopPos[i, :])
            newPopPos[i, :] = Best_pos + rand(dim) .* (Best_pos - PopPos[i, :]) + TF * p^2 * (Best_pos - PopPos[i, :])
            # Handling boundary violations
            newPopPos[i, :] = clamp.(newPopPos[i, :], lb, ub)
            # newPopFit[i] = fhd(newPopPos[i, :]', args...)
            newPopFit[i] = fhd(vec(newPopPos[i, :]'))
            if newPopFit[i] < PopFit[i]
                PopFit[i] = newPopFit[i]
                PopPos[i, :] = newPopPos[i, :]
            end
        end

        # Self protection mechanism/dangerous escape behavior
        for i in 1:n
            # A1 = rand(1, n) .* n .|> floor .|> Int
            # r1, r2 = A1[1], A1[2]
            A1 = rand(1:n, n)  # Generate 'n' random indices between 1 and n
            r1, r2 = A1[1], A1[2]  # Assign two random indices from A1
            k = rand(1:n)  # Generate one random index between 1 and n
            if rand() < 0.5
                # k = rand() * n |> floor |> Int
                k = rand(1:n)
                f1, f2 = -1 + 2 * rand(), -1 + 2 * rand()
                ro = alpha * (2 * rand() - 1)
                # Xk = rand(1, dim) .* (ub - lb) + lb
                Xk = rand(dim) .* (ub - lb) + lb

                L1 = rand() < 0.5
                u1 = L1 * 2 * rand() + (1 - L1)
                u2 = L1 * rand() + (1 - L1)
                u3 = L1 * rand() + (1 - L1)
                L2 = rand() < 0.5
                Xp = (1 - L2) * PopPos[k, :] + L2 * Xk
                popi1 = rand(VarSize) .* (ub - lb) + lb
                popi2 = rand(VarSize) .* (ub - lb) + lb

                #
                A1 = rand(1:n, 2)  # Two random integers between 1 and n
                r1, r2 = A1[1], A1[2]
                k = rand(1:n)  # One random integer between 1 and n

                # # Print all the terms in the `if` block
                # println("r1: $r1, r2: $r2, k: $k")
                # println("u1: $u1, u2: $u2, u3: $u3")
                # println("f1: $f1, f2: $f2, ro: $ro")
                # println("Xp: $Xp")
                # println("popi1: $popi1, popi2: $popi2")
                # println("Best_pos: $Best_pos")
                # println("PopPos[r1, :]: ", PopPos[r1, :])
                # println("PopPos[r2, :]: ", PopPos[r2, :])

                # # Print the size of each term in the `if` block
                # println("Size of r1: ", size(r1))  # r1 and r2 are scalars, so this will just print ()
                # println("Size of r2: ", size(r2))
                # println("Size of k: ", size(k))    # k is a scalar too
                # println("Size of u1: ", size(u1))
                # println("Size of u2: ", size(u2))
                # println("Size of u3: ", size(u3))
                # println("Size of f1: ", size(f1))
                # println("Size of f2: ", size(f2))
                # println("Size of ro: ", size(ro))
                # println("Size of Xp: ", size(Xp))
                # println("Size of popi1: ", size(popi1))
                # println("Size of popi2: ", size(popi2))
                # println("Size of Best_pos: ", size(Best_pos))
                # println("Size of PopPos[r1, :]: ", size(PopPos[r1, :]))
                # println("Size of PopPos[r2, :]: ", size(PopPos[r2, :]))
                # println("Size of newPopPos[i, :]: ", size(newPopPos[i, :]))
                # println("Results ",f1 * (u1 * Best_pos - u2 * Xp) + f2 * ro * (u3 * (popi2 - popi1) + u2 * (PopPos[r1, :] - PopPos[r2, :])) / 2)

                if u1 < 0.5
                    newPopPos[i, :] = newPopPos[i, :] + f1 * (u1 * Best_pos - u2 * Xp) + f2 * ro * (u3 * (popi2 - popi1) + u2 * (PopPos[r1, :] - PopPos[r2, :])) / 2
                else
                    newPopPos[i, :] = Best_pos + f1 * (u1 * Best_pos - u2 * Xp) + f2 * ro * (u3 * (popi2 - popi1) + u2 * (PopPos[r1, :] - PopPos[r2, :])) / 2
                end
            end
            # Handling boundary violations
            newPopPos[i, :] = clamp.(newPopPos[i, :], lb, ub)
            # newPopFit[i] = fhd(newPopPos[i, :]', args...)
            newPopFit[i] = fhd(vec(newPopPos[i, :]'))
            if newPopFit[i] < PopFit[i]
                PopFit[i] = newPopFit[i]
                PopPos[i, :] = newPopPos[i, :]
            end
        end

        # Update the Best Position
        for i in 1:n
            if PopFit[i] < Best_score
                Best_score = PopFit[i]
                Best_pos = PopPos[i, :]
            end
        end

        curve[it] = Best_score
    end

    # elapsed_time = time() - start_time
    return Best_score, Best_pos, curve
end