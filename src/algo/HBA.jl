"""
Hashim, Fatma A., Essam H. Houssein, Kashif Hussain, Mai S. Mabrouk, and Walid Al-Atabany. 
"Honey Badger Algorithm: New metaheuristic algorithm for solving optimization problems." 
Mathematics and Computers in Simulation 192 (2022): 84-110.
"""
function HBA(N, tmax, lb, ub, dim, objfunc) 
    beta = 6                      
    C = 2                         
    vec_flag = [1.0, -1.0]

    X = lb .+ rand(N, dim) .* (ub .- lb)
    fitness = ones(N) * Inf
    
    for i in 1:N
        fitness[i] = objfunc(vec(X[i,:]'))
    end
    
    GYbest, gbest = findmin(fitness)
    Xprey = X[gbest, :]
    Xnew = zeros(N, dim)

    CNVG = zeros(Float64, tmax)  

    for t in 1:tmax
        alpha = C * exp(-t / tmax)  
        I = Intensity(N, Xprey, X)   

        for i in 1:N
            r = rand()
            F = vec_flag[Int(floor(2 * rand())) + 1]

            for j in 1:dim
                di = Xprey[j] - X[i, j]

                if r < 0.5
                    r3 = rand()
                    r4 = rand()
                    r5 = rand()

                    
                    Xnew[i, j] = Xprey[j] + F * beta * I[i] * Xprey[j] + F * r3 * alpha * di * abs(cos(2 * π * r4) * (1 - cos(2 * π * r5)))
                    
                    
                else
                    r7 = rand()

                    Xnew[i, j] = Xprey[j] + F * r7 * alpha * di
                end
            end

            Xnew[i, :] = clamp.(Xnew[i, :], lb, ub)


            tempFitness = objfunc(vec(Xnew[i,:]'))
            if tempFitness < fitness[i]
                fitness[i] = tempFitness
                X[i, :] = Xnew[i, :]
            end
        end

        FU = X .> ub
        FL = X .< lb
        X = (X .* .! (FU .| FL)) .+ ub .* FU .+ lb .* FL

        Ybest, index = findmin(fitness)
        CNVG[t] = Ybest
        if Ybest < GYbest
            GYbest = Ybest
            Xprey = X[index, :]
        end
    end
    

    return GYbest, Xprey, CNVG
end

function fun_calcobjfunc(func, X)
    N = size(X, 1)               
    Y = zeros(Float64, N)        

    for i in 1:N
        Y[i] = func(vec(X[i, :]'))
    end

    return Y                     
end

function Intensity(N, Xprey, X)
    di = zeros(Float64, N)       
    S = zeros(Float64, N)        
    I = zeros(Float64, N)        
    eps = 1e-10                   

    for i in 1:N-1
        di[i] = norm(X[i, :] - Xprey .+ eps)^2
        S[i] = norm(X[i, :] - X[i + 1, :] .+ eps)^2
    end
    
    di[N] = norm(X[N, :] - Xprey .+ eps)^2
    S[N] = norm(X[N, :] - X[1, :] .+ eps)^2

    for i in 1:N
        r2 = rand()
        I[i] = r2 * S[i] / (4 * π * di[i])
    end
    
    return I                      
end