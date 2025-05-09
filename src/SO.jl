"""
Hashim, Fatma A., and Abdelazim G. Hussien. 
"Snake Optimizer: A novel meta-heuristic optimization algorithm." 
Knowledge-Based Systems 242 (2022): 108320.
"""

function SO(N::Int, T::Int, lb::Union{Int, AbstractVector}, ub::Union{Int, AbstractVector}, dim::Int, fobj::Function) #(N, T, fobj, dim, lb, ub) #SearchAgents_no, Max_iteration, lb, ub, dim, Chung_Reynolds #function SO(N, T, fobj, dim, lb, ub)
    vec_flag = [1, -1]
    Threshold = 0.25
    Thresold2 = 0.6
    C1 = 0.5
    C2 = 0.05
    C3 = 2
    if length(size(lb)) == 0
        lb = fill(lb, dim)  
    end
    if length(size(ub)) == 0
        ub = fill(ub, dim)  
    end

    X = initialization(SearchAgents_no, dim, ub, lb)
    gbest_t=zeros(T,1)
    fitness = zeros(N)
    for i in 1:N
        fitness[i] = fobj(X[i, :])
    end
    GYbest, gbest = findmin(fitness)
    Xfood = X[gbest, :]
    Nm = round(Int, N / 2)
    Nf = N - Nm
    Xm = X[1:Nm, :]
    Xnewm = zeros(Nm,dim)
    Xf = X[Nm + 1:N, :]
    Xnewf = zeros(Nf,dim)
    fitness_m = fitness[1:Nm]
    fitness_f = fitness[Nm + 1:N]
    fitnessBest_m, gbest1 = findmin(fitness_m)
    Xbest_m = Xm[gbest1, :]
    fitnessBest_f, gbest2 = findmin(fitness_f)
    Xbest_f = Xf[gbest2, :]
    for t in 1:T
        Temp = exp(-t / T)  
        Q = C1 * exp((t - T) / T)  
        if Q > 1
            Q = 1
        end
    
        if Q < Threshold
            for i in 1:Nm
                for j in 1:dim
                    rand_leader_index = rand(1:Nm)
                    X_randm = Xm[rand_leader_index, :]
                    flag_index = rand(1:2)
                    Flag = vec_flag[flag_index]
                    Am = exp(-fitness_m[rand_leader_index] / (fitness_m[i] + eps()))  
                    Xnewm[i, j] = X_randm[j] + Flag * C2 * Am * ((ub[j] - lb[j]) * rand() + lb[j])  
                end
            end
            for i in 1:Nf
                for j in 1:dim
                    rand_leader_index = rand(1:Nf)
                    X_randf = Xf[rand_leader_index, :]
                    flag_index = rand(1:2)
                    Flag = vec_flag[flag_index]
                    Af = exp(-fitness_f[rand_leader_index] / (fitness_f[i] + eps()))  
                    Xnewf[i, j] = X_randf[j] + Flag * C2 * Af * ((ub[j] - lb[j]) * rand() + lb[j])  
                end
            end
        else  
            if Temp > Thresold2  
                for i in 1:Nm
                    flag_index = rand(1:2)
                    Flag = vec_flag[flag_index]
                    for j in 1:dim
                        Xnewm[i, j] = Xfood[j] + C3 * Flag * Temp * rand() * (Xfood[j] - Xm[i, j])  
                    end
                end
                for i in 1:Nf
                    flag_index = rand(1:2)
                    Flag = vec_flag[flag_index]
                    for j in 1:dim
                        Xnewf[i, j] = Xfood[j] + Flag * C3 * Temp * rand() * (Xfood[j] - Xf[i, j])  
                    end
                end
            else  
                if rand() > 0.6  
                    for i in 1:Nm
                        for j in 1:dim
                            FM = exp(-fitnessBest_f / (fitness_m[i] + eps()))  
                            Xnewm[i, j] = Xm[i, j] + C3 * FM * rand() * (Q * Xbest_f[j] - Xm[i, j])  
                        end
                    end
                    for i in 1:Nf
                        for j in 1:dim
                            FF = exp(-fitnessBest_m / (fitness_f[i] + eps()))  
                            Xnewf[i, j] = Xf[i, j] + C3 * FF * rand() * (Q * Xbest_m[j] - Xf[i, j])  
                        end
                    end
                else  
                    for i in 1:Nm
                        for j in 1:dim
                            Mm = exp(-fitness_f[i] / (fitness_m[i] + eps()))  
                            Xnewm[i, j] = Xm[i, j] + C3 * rand() * Mm * (Q * Xf[i, j] - Xm[i, j])  
                        end
                    end
                    for i in 1:Nf
                        for j in 1:dim
                            Mf = exp(-fitness_m[i] / (fitness_f[i] + eps()))  
                            Xnewf[i, j] = Xf[i, j] + C3 * rand() * Mf * (Q * Xm[i, j] - Xf[i, j])  
                        end
                    end
                    flag_index = rand(1:2)
                    egg = vec_flag[flag_index]
                    if egg == 1
                        GYworst, gworst = findmax(fitness_m)
                        Xnewm[gworst, :] = lb .+ rand(dim) .* (ub .- lb)  
                        GYworst, gworst = findmax(fitness_f)
                        Xnewf[gworst, :] = lb .+ rand(dim) .* (ub .- lb)  
                    end
                end
            end
        end
        
        for j in 1:Nm
            Xnewm[j, :] = max.(min.( Xnewm[j, :], ub), lb)
            y = fobj(Xnewm[j, :])
            if y < fitness_m[j]
                fitness_m[j] = y
                Xm[j, :] = Xnewm[j, :]
            end
        end
    
        Ybest1, gbest1 = findmin(fitness_m)
    
        for j in 1:Nf
            Xnewf[j, :] = max.(min.( Xnewf[j, :], ub), lb)
            y = fobj(Xnewf[j, :])
            if y < fitness_f[j]
                fitness_f[j] = y
                Xf[j, :] = Xnewf[j, :]
            end
        end
    
        Ybest2, gbest2 = findmin(fitness_f)
    
        if Ybest1 < fitnessBest_m
            Xbest_m = Xm[gbest1, :]
            fitnessBest_m = Ybest1
        end
        if Ybest2 < fitnessBest_f
            Xbest_f = Xf[gbest2, :]
            fitnessBest_f = Ybest2
        end
        if Ybest1 < Ybest2
            gbest_t[t] = minimum(Ybest1)
        else
            gbest_t[t] = minimum(Ybest2)
        end
        if fitnessBest_m < fitnessBest_f
            GYbest = fitnessBest_m
            Xfood = Xbest_m
        else
            GYbest = fitnessBest_f
            Xfood = Xbest_f
        end
    end
    fval = GYbest
    return fval, Xfood, gbest_t
end