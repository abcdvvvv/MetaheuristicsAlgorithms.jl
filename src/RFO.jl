"""
Braik, M., & Al-Hiary, H. (2025). 
Rüppell’s fox optimizer: A novel meta-heuristic approach for solving global optimization problems. 
Cluster Computing, 28(5), 1-77.
"""

function RFO(rupplefoxes, itemax, lb, ub, dim, fobj)
    # Convergence curve
    ccurve = zeros(itemax)

    # f1 = plot(title = "Convergence characteristic curve",
    #         xlabel = L"Iteration",
    #         ylabel = L"fitness value",
    #         legend = false,
    #         grid = true,
    #         background_color = :white,
    #         fontfamily = "Times",
    #         guidefontsize = 10)

    # Generate initial solutions (position of rupple foxes)
    fposition = initialization(rupplefoxes, dim, ub, lb)

    # Initialize fitness arrays
    fPos = zeros(rupplefoxes, dim)
    fit = zeros(rupplefoxes)

    for i in 1:rupplefoxes
        fit[i] = fobj(fposition[i, :])
    end

    # Initial fitness of the random position of the rupple foxes
    fitness = copy(fit)

    # Find best initial solution
    fmin0, index = findmin(fit)

    # Initialize personal best and global best
    pbest = copy(fposition)
    fbest = copy(pbest[index, :])
    x = copy(pbest)

    # Parameters
    L = 100
    beta = 1e-10  # Constant for walk rate update
    vec_flag = [1, -1]


    for ite in 1:itemax

        h = 1 / (1 + exp((ite - itemax / 2) / L))  # eyesight decreasing
        s = 1 / (1 + exp((itemax / 2 - ite) / L))  # hearing increasing
        # smell = 0.1 / abs(acos(2 / (1 + exp((itemax / 2 - ite) / 100))))  # smell variation
        smell = 0.1 / abs(acos(clamp(2 / (1 + exp((itemax / 2 - ite) / 100)), -1.0, 1.0)))

    
        if rand() >= 0.5  # Daylight
            for i in 1:rupplefoxes
                ds = ceil.(Int, rupplefoxes * rand(rupplefoxes))
    
                if s >= h  # Eyesight > Hearing
                    if rand() >= 0.25
                        # First phase (sight dominates) - exploration
                        X_randm = pbest[ds[i], :]
                        rand_index = floor(Int, 4 * rand()) + 1
                        r_val = rand()
                        
                        fposition[i, :] = x[i, :] + rand_index * r_val * (X_randm - x[i, :]) + rand_index * r_val * (fbest - x[i, :])
                        
                        theta = rand() * 260 * 2 * π / 360
                        fposition[i, :] = frotate(fposition[i, :], X_randm, theta, dim)
                    else
                        # Second phase
                        flag_index = floor(Int, 2 * rand()) + 1
                        Flag = vec_flag[flag_index]
                        X_randm = pbest[ds[i], :]
                        fposition[i, :] = X_randm + beta * randn(1, dim)[1, :] * Flag
                    end
                else
                    # Hearing > Sight
                    if rand() >= 0.75
                        X_randm = pbest[ds[i], :]
                        rand_index = floor(Int, 4 * rand()) + 1
                        r_val = rand()
                        
                        fposition[i, :] = x[i, :] + rand_index * r_val * (X_randm - x[i, :]) + rand_index * r_val * (fbest - x[i, :])
                        
                        theta = rand() * 150 * π / 180
                        fposition[i, :] = frotate(fposition[i, :], X_randm, theta, dim)
                    else
                        flag_index = floor(Int, 2 * rand()) + 1
                        Flag = vec_flag[flag_index]
                        fposition[i, :] = pbest[ds[i], :] + beta * randn(1, dim)[1, :] * Flag
                    end
                end
            end
        else
            # Night
            for i in 1:rupplefoxes
                dr = floor.(Int, rupplefoxes * rand(rupplefoxes)) .+ 1
    
                if h >= s
                    if rand() >= 0.25
                        # First phase
                        X_randm = pbest[dr[i], :]
                        rand_index = floor(Int, 4 * rand()) + 1
                        r_val = rand()
                        
                        fposition[i, :] = x[i, :] + rand_index * r_val * (X_randm - x[i, :]) + rand_index * r_val * (fbest - x[i, :])
                        
                        theta = rand() * 150 * π / 360
                        fposition[i, :] = frotate(fposition[i, :], X_randm, theta, dim)
                    else
                        flag_index = floor(Int, 2 * rand()) + 1
                        Flag = vec_flag[flag_index]
                        fposition[i, :] = pbest[dr[i], :] + beta * randn(1, dim)[1, :] * Flag
                    end
                else
                    # Hearing > Sight
                    if rand() >= 0.75
                        X_randm = pbest[dr[i], :]
                        rand_index = floor(Int, 4 * rand()) + 1
                        r_val = rand()
                        
                        fposition[i, :] = x[i, :] + rand_index * r_val * (X_randm - x[i, :]) + rand_index * r_val * (fbest - x[i, :])
                        
                        theta = rand() * 260 * 2 * π / 180
                        fposition[i, :] = frotate(fposition[i, :], X_randm, theta, dim)
                    else
                        flag_index = floor(Int, 2 * rand()) + 1
                        Flag = vec_flag[flag_index]
                        fposition[i, :] = pbest[dr[i], :] + beta * randn(1, dim)[1, :] * Flag
                    end
                end
            end
        end
    end

    for i in 1:rupplefoxes
        if rand() >= smell
            ss = rand(1:rupplefoxes)
            X_randm1 = pbest[ss, :]
    
            xmin, xmax = 2, 4
            xr = xmin + rand() * (xmax - xmin)
            eps = abs((4 * rand()) - (rand() + rand())) / xr
    
            fposition[i, :] = x[i, :] .+ (X_randm1 .- x[i, :]) .* eps .+ eps .* (fbest .- x[i, :])
        else
            flag_index = rand(1:2)
            Flag = vec_flag[flag_index]
            ss = rand(1:rupplefoxes)
            fposition[i, :] = pbest[ss, :] .+ beta * randn(dim) * Flag
        end
    end
    
    # Animal behavior
    for i in 1:rupplefoxes
        if rand() >= h
            ab = rand(1:rupplefoxes)
            fposition[i, :] = pbest[ab, :] .+ beta * randn(dim)
        else
            ab = rand(1:rupplefoxes)
            X_randp = pbest[ab, :]
            Dista = 2 * (fbest .- fposition[i, :]) * rand()
            Distb = 3.0 * (X_randp .- pbest[i, :]) * rand()
            flag_index = rand(1:2)
            Flag = vec_flag[flag_index]
            if i == 1
                fposition[i, :] = fposition[i, :] .+ Dista .+ Distb * Flag
            else
                fPos[i, :] = fposition[i, :] .+ Dista .+ Distb * Flag
                fposition[i, :] = (fPos[i, :] .+ fposition[i-1, :]) ./ 2
            end
        end
    end
    
    # Worst-case exploration
    for i in 1:rupplefoxes
        flag_index = rand(1:2)
        fox = vec_flag[flag_index]
        if fox == 1
            _, gworst = findmax(fitness)
            fposition[gworst, :] .+= beta * randn(dim)
        end
    end
    
    # Handling boundary violations
    for i in 1:rupplefoxes
        fposition[i, :] = clamp.(fposition[i, :], lb, ub)
    end
    
    # Update bests and fitness
    for i in 1:rupplefoxes
        if all(fposition[i, :] .>= lb) && all(fposition[i, :] .<= ub)
            x[i, :] = fposition[i, :]
    
            ub_ = fposition[i, :] .> ub
            lb_ = fposition[i, :] .< lb
            fposition[i, :] = fposition[i, :] .* .!(ub_ .| lb_) .+ ub .* ub_ .+ lb .* lb_
    
            fit[i] = fobj(fposition[i, :])
    
            if fit[i] < fitness[i]
                pbest[i, :] = fposition[i, :]
                fitness[i] = fit[i]
            end
    
            if fitness[i] < fmin0
                fmin0 = fitness[i]
                fbest = pbest[i, :]
            end
        end
    end
    
    println("Iteration# $ite  Fitness= $fmin0")
    ccurve[ite] = fmin0    
    
end


function frotate(x::Vector{Float64}, y::Vector{Float64}, theta::Float64, dim::Int)
    cent = ceil(Int, dim / 2)

    # Create 2×N matrix of the points to rotate
    v = hcat(x, y)'  # 2×N matrix

    # Rotation center coordinates
    x_center = x[cent]
    y_center = y[cent]

    # Repeat the center as a 2×N matrix
    center = repeat([x_center, y_center], 1, length(x))

    # Rotation matrix
    R = [cos(theta) -sin(theta); sin(theta) cos(theta)]

    # Apply the rotation
    s = v .- center             # shift to origin
    so = R * s                  # rotate
    vo = so .+ center           # shift back

    return vec(vo[1, :])        # return rotated x-coordinates as a vector
end
