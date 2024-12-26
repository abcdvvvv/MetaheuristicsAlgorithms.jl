"""
Abdollahzadeh, Benyamin, Farhad Soleimanian Gharehchopogh, and Seyedali Mirjalili. 
"Artificial gorilla troops optimizer: a new nature‐inspired metaheuristic algorithm for global optimization problems." 
International Journal of Intelligent Systems 36, no. 10 (2021): 5887-5958.
"""
function GTO(pop_size, max_iter, lower_bound, upper_bound, variables_no, fobj)
    # Initialize Silverback
    Silverback = []
    Silverback_Score = Inf

    # Initialize the first random population of Gorillas
    X = initialization(pop_size, variables_no, upper_bound, lower_bound)

    # Initialize convergence curve
    convergence_curve = zeros(max_iter)

    # Evaluate fitness for the initial population
    Pop_Fit = zeros(pop_size)
    for i in 1:pop_size   
        Pop_Fit[i] = fobj(X[i, :])
        if Pop_Fit[i] < Silverback_Score 
            Silverback_Score = Pop_Fit[i] 
            Silverback = X[i, :]
        end
    end

    GX = copy(X)
    lb = ones(variables_no) * lower_bound 
    ub = ones(variables_no) * upper_bound 

    # Controlling parameters
    p = 0.03
    Beta = 3
    w = 0.8

    # Main loop
    for It in 1:max_iter 
        a = (cos(2 * rand()) + 1) * (1 - It / max_iter)
        C = a * (2 * rand() - 1) 

        # Exploration
        for i in 1:pop_size
            if rand() < p    
                GX[i, :] .= (ub - lb) .* rand() .+ lb
            else  
                if rand() >= 0.5
                    Z = rand(variables_no) * (2 * a) .- a  # Uniform distribution in range [-a, a]
                    H = Z .* X[i, :]   
                    GX[i, :] .= (rand() - a) * X[rand(1:pop_size), :] .+ C .* H 
                else   
                    GX[i, :] .= X[i, :] .- C .* (C * (X[i, :] .- GX[rand(1:pop_size), :]) .+ rand() * (X[i, :] .- GX[rand(1:pop_size), :])) 
                end
            end
        end       
       
        # GX = boundaryCheck(GX, lower_bound, upper_bound)
        GX .= clamp.(GX, lower_bound, upper_bound)

        # Group formation operation 
        for i in 1:pop_size
            New_Fit = fobj(GX[i, :])          
            if New_Fit < Pop_Fit[i]
                Pop_Fit[i] = New_Fit
                X[i, :] = GX[i, :]
            end
            if New_Fit < Silverback_Score 
                Silverback_Score = New_Fit 
                Silverback = GX[i, :]
            end
        end
    
        # Exploitation  
        for i in 1:pop_size
            if a >= w  
                g = 2^C
                delta = (abs.(mean(GX, dims=1)) .^ g) .^ (1/g)
                # GX[i, :] .= C * delta .* (X[i, :] .- Silverback) .+ X[i, :] 
                GX[i, :] .= C * delta[1, :] .* (X[i, :] .- Silverback) .+ X[i, :]
            else
                # h = rand() >= 0.5 ? randn(1, variables_no) : randn(1)
                h = rand() >= 0.5 ? randn() : randn()
                r1 = rand() 
                # println("Size of Silverback: ", size(Silverback))
                # println("Size of X[i, :]: ", size(X[i, :]))
                # println("Size of (2 * r1 - 1): ", size(2 * r1 - 1))
                # println("Size of (Beta * h): ", size(Beta * h))
                # GX[i, :] .= Silverback .- (Silverback * (2 * r1 - 1) .- X[i, :] * (2 * r1 - 1)) .* (Beta * h) 
                GX[i, :] .= Silverback .- (Silverback .* (2 * r1 - 1) .- X[i, :] .* (2 * r1 - 1)) .* (Beta * h)

            end
        end
   
        # GX = boundaryCheck(GX, lower_bound, upper_bound)
        GX .= clamp.(GX, lower_bound, upper_bound)

        # Group formation operation    
        for i in 1:pop_size
            New_Fit = fobj(GX[i, :])
            if New_Fit < Pop_Fit[i]
                Pop_Fit[i] = New_Fit
                X[i, :] = GX[i, :]
            end
            if New_Fit < Silverback_Score 
                Silverback_Score = New_Fit 
                Silverback = GX[i, :]
            end
        end
             
        convergence_curve[It] = Silverback_Score
        # println("In Iteration $It, best estimation of the global optimum is $(round(Silverback_Score, digits=4))")
    end 

    return Silverback_Score, Silverback, convergence_curve
end
