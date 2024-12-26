"""
Khishe, Mohammad, and Mohammad Reza Mosavi. 
"Chimp optimization algorithm." 
Expert systems with applications 149 (2020): 113338.
"""

using Random


function ChOA(SearchAgents_no, Max_iter, lb, ub, dim, fobj)
    # Initialize Attacker, Barrier, Chaser, and Driver
    Attacker_pos = zeros(dim)
    Attacker_score = Inf  # Change to -Inf for maximization problems

    Barrier_pos = zeros(dim)
    Barrier_score = Inf  # Change to -Inf for maximization problems

    Chaser_pos = zeros(dim)
    Chaser_score = Inf  # Change to -Inf for maximization problems

    Driver_pos = zeros(dim)
    Driver_score = Inf  # Change to -Inf for maximization problems

    # Initialize the positions of search agents
    Positions = initialization(SearchAgents_no, dim, ub, lb)

    Convergence_curve = zeros(Max_iter)
    l = 0  # Loop counter

    # Main loop
    while l < Max_iter
        # for i in 1:size(Positions, 1)
        for i in axes(Positions, 1)
            # Return back the search agents that go beyond the boundaries of the search space
            Flag4ub = Positions[i, :] .> ub
            Flag4lb = Positions[i, :] .< lb
            Positions[i, :] = (Positions[i, :] .* .~(Flag4ub .+ Flag4lb)) .+ ub .* Flag4ub .+ lb .* Flag4lb

            # Calculate objective function for each search agent
            fitness = fobj(Positions[i, :])

            # Update Attacker, Barrier, Chaser, and Driver
            if fitness < Attacker_score
                Attacker_score = fitness
                Attacker_pos = Positions[i, :]
            elseif fitness > Attacker_score && fitness < Barrier_score
                Barrier_score = fitness
                Barrier_pos = Positions[i, :]
            elseif fitness > Attacker_score && fitness > Barrier_score && fitness < Chaser_score
                Chaser_score = fitness
                Chaser_pos = Positions[i, :]
            elseif fitness > Attacker_score && fitness > Barrier_score && fitness > Chaser_score && fitness > Driver_score
                Driver_score = fitness
                Driver_pos = Positions[i, :]
            end
        end

        f = 2 - l * (2 / Max_iter)  # a decreases linearly from 2 to 0

        # Dynamic Coefficient of f Vector as in Table 1
        C1G1 = 1.95 - ((2 * l^(1/3)) / (Max_iter^(1/3)))
        C2G1 = (2 * l^(1/3)) / (Max_iter^(1/3)) + 0.5

        C1G2 = 1.95 - ((2 * l^(1/3)) / (Max_iter^(1/3)))
        C2G2 = (2 * (l^3) / (Max_iter^3)) + 0.5

        C1G3 = (-2 * (l^3) / (Max_iter^3)) + 2.5
        C2G3 = (2 * l^(1/3)) / (Max_iter^(1/3)) + 0.5

        C1G4 = (-2 * (l^3) / (Max_iter^3)) + 2.5
        C2G4 = (2 * (l^3) / (Max_iter^3)) + 0.5

        # Update the position of search agents
        # for i in 1:size(Positions, 1)
        for i in axes(Positions, 1)
            for j in 1:dim
                r11 = C1G1 * rand()
                r12 = C2G1 * rand()

                r21 = C1G2 * rand()
                r22 = C2G2 * rand()

                r31 = C1G3 * rand()
                r32 = C2G3 * rand()

                r41 = C1G4 * rand()
                r42 = C2G4 * rand()

                A1 = 2 * f * r11 - f
                C1 = 2 * r12

                m = chaos(3, 1, 1)
                # println("size(m) ", size(m))
                # println("size(C1) ", size(C1))
                # println("size(A1) ", size(A1))
                # D_Attacker = abs(C1 * Attacker_pos[j] - m * Positions[i, j])
                D_Attacker = abs.(C1 * Attacker_pos[j] .- m * Positions[i, j])
                X1 = Attacker_pos[j] .- A1 * D_Attacker

                A2 = 2 * f * r21 - f
                C2 = 2 * r22
                # D_Barrier = abs(C2 * Barrier_pos[j] - m * Positions[i, j])
                D_Barrier = abs.(C2 * Barrier_pos[j] .- m * Positions[i, j])
                X2 = Barrier_pos[j] .- A2 * D_Barrier

                A3 = 2 * f * r31 - f
                C3 = 2 * r32
                # D_Chaser = abs(C3 * Chaser_pos[j] - m * Positions[i, j])
                D_Chaser = abs.(C3 * Chaser_pos[j] .- m * Positions[i, j])
                X3 = Chaser_pos[j] .- A3 * D_Chaser

                A4 = 2 * f * r41 - f
                C4 = 2 * r42
                # D_Driver = abs(C4 * Driver_pos[j] - m * Positions[i, j])
                D_Driver = abs.(C4 * Driver_pos[j] .- m * Positions[i, j])
                X4 = Driver_pos[j] .- A4 * D_Driver

                # println("X1 ", X1)

                # Positions[i, j] = (X1 + X2 + X3 + X4) / 4  # Update position
                Positions[i, j] = ((X1 + X2 + X3 + X4) / 4)[1]  # Update position
            end
        end

        l += 1
        Convergence_curve[l] = Attacker_score
    end

    return Attacker_score, Attacker_pos, Convergence_curve
end

function chaos(index, max_iter, Value)
    O = zeros(max_iter)
    x = zeros(max_iter + 1)
    x[1] = 0.7

    if index == 1
        # Chebyshev map
        for i in 1:max_iter
            x[i + 1] = cos(i * acos(x[i]))
            O[i] = ((x[i] + 1) * Value) / 2
        end
    elseif index == 2
        # Circle map
        a = 0.5
        b = 0.2
        for i in 1:max_iter
            x[i + 1] = mod(x[i] + b - (a / (2 * π)) * sin(2 * π * x[i]), 1)
            O[i] = x[i] * Value
        end
    elseif index == 3
        # Gauss/mouse map
        for i in 1:max_iter
            if x[i] == 0
                x[i + 1] = 0
            else
                x[i + 1] = mod(1 / x[i], 1)
            end
            O[i] = x[i] * Value
        end
    elseif index == 4
        # Iterative map
        a = 0.7
        for i in 1:max_iter
            x[i + 1] = sin((a * π) / x[i])
            O[i] = ((x[i] + 1) * Value) / 2
        end
    elseif index == 5
        # Logistic map
        a = 4
        for i in 1:max_iter
            x[i + 1] = a * x[i] * (1 - x[i])
            O[i] = x[i] * Value
        end
    elseif index == 6
        # Piecewise map
        P = 0.4
        for i in 1:max_iter
            if 0 <= x[i] < P
                x[i + 1] = x[i] / P
            elseif P <= x[i] < 0.5
                x[i + 1] = (x[i] - P) / (0.5 - P)
            elseif 0.5 <= x[i] < (1 - P)
                x[i + 1] = (1 - P - x[i]) / (0.5 - P)
            elseif (1 - P) <= x[i] < 1
                x[i + 1] = (1 - x[i]) / P
            end
            O[i] = x[i] * Value
        end
    elseif index == 7
        # Sine map
        for i in 1:max_iter
            x[i + 1] = sin(π * x[i])
            O[i] = x[i] * Value
        end
    elseif index == 8
        # Singer map
        u = 1.07
        for i in 1:max_iter
            x[i + 1] = u * (7.86 * x[i] - 23.31 * (x[i]^2) + 28.75 * (x[i]^3) - 13.302875 * (x[i]^4))
            O[i] = x[i] * Value
        end
    elseif index == 9
        # Sinusoidal map
        for i in 1:max_iter
            x[i + 1] = 2.3 * x[i]^2 * sin(π * x[i])
            O[i] = x[i] * Value
        end
    elseif index == 10
        # Tent map
        x[1] = 0.6
        for i in 1:max_iter
            if x[i] < 0.7
                x[i + 1] = x[i] / 0.7
            else
                x[i + 1] = (10 / 3) * (1 - x[i])
            end
            O[i] = x[i] * Value
        end
    end

    return O
end
