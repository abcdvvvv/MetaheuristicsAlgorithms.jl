"""
Abualigah, Laith, Mohamed Abd Elaziz, Putra Sumari, Zong Woo Geem, and Amir H. Gandomi. 
"Reptile Search Algorithm (RSA): A nature-inspired meta-heuristic optimizer." 
Expert Systems with Applications 191 (2022): 116158.
"""
function RSA(N, T, LB, UB, Dim, F_obj)
    Best_P = zeros(Dim)           
    Best_F = Inf                   
    X = initialization(N, Dim, UB, LB) 
    Xnew = zeros(N, Dim)
    Conv = zeros(T)                

    t = 1                          
    Alpha = 0.1                    
    Beta = 0.005                   
    Ffun = zeros(N)                
    Ffun_new = zeros(N)            

    for i in 1:N
        Ffun[i] = F_obj(X[i, :])  
        if Ffun[i] < Best_F
            Best_F = Ffun[i]
            Best_P = X[i, :]
        end
    end

    while t <= T  
        ES = 2 * randn() * (1 - (t / T))  
        for i in 2:N 
            for j in 1:Dim  
                R = (Best_P[j] - X[rand(1:N), j]) / (Best_P[j] + eps())
                P = Alpha + (X[i, j] - mean(X[i, :])) / (Best_P[j] * (UB - LB) + eps())
                Eta = Best_P[j] * P

                if t < T / 4
                    Xnew[i, j] = Best_P[j] - Eta * Beta - R * rand()
                elseif t < 2 * T / 4
                    Xnew[i, j] = Best_P[j] * X[rand(1:N), j] * ES * rand()
                elseif t < 3 * T / 4
                    Xnew[i, j] = Best_P[j] * P * rand()
                else
                    Xnew[i, j] = Best_P[j] - Eta * eps() - R * rand()
                end
            end
            
            Flag_UB = Xnew[i, :] .> UB
            Flag_LB = Xnew[i, :] .< LB
            Xnew[i, :] = (Xnew[i, :] .* .! (Flag_UB .| Flag_LB)) .+ (UB .* Flag_UB) .+ (LB .* Flag_LB)
            Ffun_new[i] = F_obj(Xnew[i, :])

            if Ffun_new[i] < Ffun[i]
                X[i, :] = Xnew[i, :]
                Ffun[i] = Ffun_new[i]
            end

            if Ffun[i] < Best_F
                Best_F = Ffun[i]
                Best_P = X[i, :]
            end
        end
  
        Conv[t] = Best_F  

        # if t % 50 == 0  # Print the best universe details after every 50 iterations
        #      println("At iteration $t the best solution fitness is $Best_F")
        # end

        t += 1
    end
    
    return Best_F, Best_P, Conv  
end