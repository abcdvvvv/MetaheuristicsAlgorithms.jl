"""
Abdel-Basset, Mohamed, Doaa El-Shahat, Mohammed Jameel, and Mohamed Abouhawwash. 
"Exponential distribution optimizer (EDO): a novel math-inspired algorithm for global optimization and engineering problems." 
Artificial Intelligence Review 56, no. 9 (2023): 9329-9400.
"""

function EDO(N, Max_iter, LB, UB, Dim, F_obj)
    # Initialize variables
    BestSol = zeros(Dim)  # Optimal solution
    BestFitness = Inf     # Optimal fitness value
    Xwinners = initialization(N, Dim, UB, LB)  # Initialize population
    Fitness = zeros(N)

    # Evaluate initial fitness
    for i in 1:N
        Fitness[i] = F_obj(Xwinners[i, :])
        if Fitness[i] < BestFitness
            BestFitness = Fitness[i]
            BestSol = Xwinners[i, :]
        end
    end

    Memoryless = copy(Xwinners)  # Old population
    iter = 1
    cgcurve = zeros(Max_iter)  # Convergence curve

    while iter <= Max_iter
        V = zeros(N, Dim)
        cgcurve[iter] = BestFitness

        # Sort fitness and solutions
        sorted_indices = sortperm(Fitness)
        Fitness = Fitness[sorted_indices]
        Xwinners = Xwinners[sorted_indices, :]

        # EDO parameters
        d = (1 - iter / Max_iter)  # Eq.(23)
        f = 2 * rand() - 1         # Eq.(17)
        a = f^10                   # Eq.(15)
        b = f^5                    # Eq.(16)
        c = d * f                  # Eq.(22)
        X_guide = mean(Xwinners[1:3, :], dims=1) |> vec  # Eq.(13)

        for i in 1:N
            alpha = rand()
            if alpha < 0.5
                # Exploitation
                if Memoryless[i, :] == Xwinners[i, :]
                    Mu = (X_guide .+ Memoryless[i, :]) ./ 2.0   # Eq.(19)
                    ExP_rate = 1.0 ./ Mu                       # Eq.(18)
                    variance = 1.0 ./ (ExP_rate .^ 2)          # Eq.(4)
                    V[i, :] = a .* (Memoryless[i, :] .- variance) .+ b .* X_guide  # Eq.(14)
                else
                    Mu = (X_guide .+ Memoryless[i, :]) ./ 2.0
                    ExP_rate = 1.0 ./ Mu
                    variance = 1.0 ./ (ExP_rate .^ 2)
                    phi = rand()
                    V[i, :] = b .* (Memoryless[i, :] .- variance) .+ log(phi) .* Xwinners[i, :]  # Eq.(14)
                end
            else
                # Exploration
                M = mean(Xwinners, dims=1) |> vec
                s = randperm(N)
                D1 = M .- Xwinners[s[1], :]  # Eq.(26)
                D2 = M .- Xwinners[s[2], :]  # Eq.(27)
                Z1 = M .- D1 .+ D2           # Eq.(24)
                Z2 = M .- D2 .+ D1           # Eq.(25)
                V[i, :] = Xwinners[i, :] .+ c .* Z1 .+ (1.0 - c) .* Z2 .- M  # Eq.(20)
            end

            # Check bounds of V
            V[i, :] = clamp.(V[i, :], LB, UB)
        end

        # Update Memoryless
        Memoryless .= V

        # Generate the next population
        V_Fitness = zeros(N)
        for i in 1:N
            V_Fitness[i] = F_obj(V[i, :])
            if V_Fitness[i] < Fitness[i]
                Xwinners[i, :] = V[i, :]
                Fitness[i] = V_Fitness[i]
                if Fitness[i] < BestFitness
                    BestFitness = Fitness[i]
                    BestSol = Xwinners[i, :]
                end
            end
        end

        # if iter % 1666 == 0
        #     println("Iteration $iter: Best Cost = $(BestFitness)")
        # end
        # println("Iteration $iter: Best Cost = $(BestFitness)")

        iter += 1
    end

    return BestFitness, BestSol, cgcurve
end
