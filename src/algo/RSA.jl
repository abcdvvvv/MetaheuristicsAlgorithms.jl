"""
Abualigah, Laith, Mohamed Abd Elaziz, Putra Sumari, Zong Woo Geem, and Amir H. Gandomi. 
"Reptile Search Algorithm (RSA): A nature-inspired meta-heuristic optimizer." 
Expert Systems with Applications 191 (2022): 116158.
"""
function RSA(N, max_iter, lb, ub, dim, objfun)
    Best_P = zeros(dim)           
    Best_F = Inf                   
    X = initialization(N, dim, ub, lb) 
    Xnew = zeros(N, dim)
    Conv = zeros(max_iter)                

    t = 1                          
    Alpha = 0.1                    
    Beta = 0.005                   
    Ffun = zeros(N)                
    Ffun_new = zeros(N)            

    for i in 1:N
        Ffun[i] = objfun(X[i, :])  
        if Ffun[i] < Best_F
            Best_F = Ffun[i]
            Best_P = X[i, :]
        end
    end

    while t <= max_iter  
        ES = 2 * randn() * (1 - (t / max_iter))  
        for i in 2:N 
            for j in 1:dim  
                R = (Best_P[j] - X[rand(1:N), j]) / (Best_P[j] + eps())
                P = Alpha + (X[i, j] - mean(X[i, :])) / (Best_P[j] * (ub - lb) + eps())
                Eta = Best_P[j] * P

                if t < max_iter / 4
                    Xnew[i, j] = Best_P[j] - Eta * Beta - R * rand()
                elseif t < 2 * max_iter / 4
                    Xnew[i, j] = Best_P[j] * X[rand(1:N), j] * ES * rand()
                elseif t < 3 * max_iter / 4
                    Xnew[i, j] = Best_P[j] * P * rand()
                else
                    Xnew[i, j] = Best_P[j] - Eta * eps() - R * rand()
                end
            end
            
            Flag_UB = Xnew[i, :] .> ub
            Flag_LB = Xnew[i, :] .< lb
            Xnew[i, :] = (Xnew[i, :] .* .! (Flag_UB .| Flag_LB)) .+ (ub .* Flag_UB) .+ (lb .* Flag_LB)
            Ffun_new[i] = objfun(Xnew[i, :])

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