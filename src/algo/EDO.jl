"""
Abdel-Basset, Mohamed, Doaa El-Shahat, Mohammed Jameel, and Mohamed Abouhawwash. 
"Exponential distribution optimizer (EDO): a novel math-inspired algorithm for global optimization and engineering problems." 
Artificial Intelligence Review 56, no. 9 (2023): 9329-9400.
"""

function EDO(N, Max_iter, LB, UB, Dim, F_obj)
    BestSol = zeros(Dim)  
    BestFitness = Inf     
    Xwinners = initialization(N, Dim, UB, LB)  
    Fitness = zeros(N)

    for i in 1:N
        Fitness[i] = F_obj(Xwinners[i, :])
        if Fitness[i] < BestFitness
            BestFitness = Fitness[i]
            BestSol = Xwinners[i, :]
        end
    end

    Memoryless = copy(Xwinners)  
    iter = 1
    cgcurve = zeros(Max_iter)  

    while iter <= Max_iter
        V = zeros(N, Dim)
        cgcurve[iter] = BestFitness

        sorted_indices = sortperm(Fitness)
        Fitness = Fitness[sorted_indices]
        Xwinners = Xwinners[sorted_indices, :]

        d = (1 - iter / Max_iter)  
        f = 2 * rand() - 1         
        a = f^10                   
        b = f^5                    
        c = d * f                  
        X_guide = mean(Xwinners[1:3, :], dims=1) |> vec  

        for i in 1:N
            alpha = rand()
            if alpha < 0.5
                if Memoryless[i, :] == Xwinners[i, :]
                    Mu = (X_guide .+ Memoryless[i, :]) ./ 2.0   
                    ExP_rate = 1.0 ./ Mu                       
                    variance = 1.0 ./ (ExP_rate .^ 2)          
                    V[i, :] = a .* (Memoryless[i, :] .- variance) .+ b .* X_guide  
                else
                    Mu = (X_guide .+ Memoryless[i, :]) ./ 2.0
                    ExP_rate = 1.0 ./ Mu
                    variance = 1.0 ./ (ExP_rate .^ 2)
                    phi = rand()
                    V[i, :] = b .* (Memoryless[i, :] .- variance) .+ log(phi) .* Xwinners[i, :]  
                end
            else
                M = mean(Xwinners, dims=1) |> vec
                s = randperm(N)
                D1 = M .- Xwinners[s[1], :]  
                D2 = M .- Xwinners[s[2], :]  
                Z1 = M .- D1 .+ D2           
                Z2 = M .- D2 .+ D1           
                V[i, :] = Xwinners[i, :] .+ c .* Z1 .+ (1.0 - c) .* Z2 .- M  
            end
            V[i, :] = clamp.(V[i, :], LB, UB)
        end


        Memoryless .= V


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

        iter += 1
    end

    return BestFitness, BestSol, cgcurve
end
