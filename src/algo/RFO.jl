"""
Braik, M., & Al-Hiary, H. (2025). 
Rüppell’s fox optimizer: A novel meta-heuristic approach for solving global optimization problems. 
Cluster Computing, 28(5), 1-77.
"""

# function RFO(rupplefoxes::Int, max_iter::Int, lb::Real, ub::Real, dim::Int, objfun::Function)
#     # Initialize variables
#     ccurve = zeros(max_iter)
#     vec_flag = [1, -1]
#     beta = 1e-10
#     L = 100

#     # Initialize population
#     fposition = initialization(rupplefoxes, dim, ub, lb)
#     x = copy(fposition)
#     fit = [objfun(fposition[i, :]) for i in 1:rupplefoxes]
#     fitness = copy(fit)

#     fmin0, index = findmin(fit)
#     pbest = copy(fposition)
#     fbest = copy(pbest[index, :])

#     for ite in 1:max_iter
#         h = 1 / (1 + exp((ite - max_iter / 2) / L))
#         s = 1 / (1 + exp((max_iter / 2 - ite) / L))
#         tmp = 2 / (1 + exp((max_iter / 2 - ite) / 100))
#         # smell = 0.1 / abs(acos(min(pi, max(-pi, tmp))))
#         smell = 0.1 / abs(acos(clamp(tmp, -1.0, 1.0)))

#         if rand() >= 0.5
#             # Daylight
#             for i in 1:rupplefoxes
#                 # ds = ceil.(Int, rupplefoxes * rand(rupplefoxes))
#                 ds = rand(1:rupplefoxes, rupplefoxes)
#                 if s >= h
#                     if rand() >= 0.25
#                         X_randm = pbest[ds[i], :]
#                         rand_index = floor(Int, 4 * rand() + 1) * rand()
#                         fposition[i, :] .= x[i, :] .+ rand_index .* (X_randm .- x[i, :]) .+ rand_index .* (fbest .- x[i, :])
#                         theta = rand() * 260 * 2 * pi / 360
#                         fposition[i, :] .= frotate(fposition[i, :], X_randm, theta, dim)
#                     else
#                         Flag = vec_flag[floor(Int, 2 * rand() + 1)]
#                         X_randm = pbest[ds[i], :]
#                         fposition[i, :] .= X_randm .+ beta * randn(dim) * Flag
#                     end
#                 else
#                     if rand() >= 0.75
#                         X_randm = pbest[ds[i], :]
#                         rand_index = floor(Int, 4 * rand() + 1) * rand()
#                         fposition[i, :] .= x[i, :] .+ rand_index .* (X_randm .- x[i, :]) .+ rand_index .* (fbest .- x[i, :])
#                         theta = rand() * 150 * pi / 180
#                         fposition[i, :] .= frotate(fposition[i, :], X_randm, theta, dim)
#                     else
#                         Flag = vec_flag[floor(Int, 2 * rand() + 1)]
#                         fposition[i, :] .= pbest[ds[i], :] .+ beta * randn(dim) * Flag
#                     end
#                 end
#             end
#         else
#             # Night
#             for i in 1:rupplefoxes
#                 dr = floor.(Int, rupplefoxes * rand(rupplefoxes)) .+ 1
#                 if h >= s
#                     if rand() >= 0.25
#                         X_randm = pbest[dr[i], :]
#                         rand_index = floor(Int, 4 * rand() + 1) * rand()
#                         fposition[i, :] .= x[i, :] .+ rand_index .* (X_randm .- x[i, :]) .+ rand_index .* (fbest .- x[i, :])
#                         theta = rand() * 150 * pi / 360
#                         fposition[i, :] .= frotate(fposition[i, :], X_randm, theta, dim)
#                     else
#                         Flag = vec_flag[floor(Int, 2 * rand() + 1)]
#                         fposition[i, :] .= pbest[dr[i], :] .+ beta * randn(dim) * Flag
#                     end
#                 else
#                     if rand() >= 0.75
#                         X_randm = pbest[dr[i], :]
#                         rand_index = floor(Int, 4 * rand() + 1) * rand()
#                         fposition[i, :] .= x[i, :] .+ rand_index .* (X_randm .- x[i, :]) .+ rand_index .* (fbest .- x[i, :])
#                         theta = rand() * 260 * 2 * pi / 180
#                         fposition[i, :] .= frotate(fposition[i, :], X_randm, theta, dim)
#                     else
#                         Flag = vec_flag[floor(Int, 2 * rand() + 1)]
#                         fposition[i, :] .= pbest[dr[i], :] .+ beta * randn(dim) * Flag
#                     end
#                 end
#             end
#         end

#         # Smell behavior
#         for i in 1:rupplefoxes
#             if rand() >= smell
#                 ss = floor.(Int, rand(rupplefoxes) .* 1) .+ 1
#                 X_randm1 = pbest[ss[i], :]
#                 xr = 2 + rand() * (4 - 2)
#                 eps = abs(4 * rand() - (1 * rand() + rand())) / xr
#                 fposition[i, :] .= x[i, :] .+ (X_randm1 .- x[i, :]) * eps .+ eps .* (fbest .- x[i, :])
#             else
#                 Flag = vec_flag[floor(Int, 2 * rand() + 1)]
#                 ss = floor.(Int, rand(rupplefoxes) .* 1) .+ 1
#                 fposition[i, :] .= pbest[ss[i], :] .+ beta * randn(dim) * Flag
#             end
#         end

#         # Animal behavior
#         for i in 1:rupplefoxes
#             ab = floor.(Int, rupplefoxes * rand(rupplefoxes)) .+ 1
#             if rand() >= h
#                 fposition[i, :] .= pbest[ab[i], :] .+ beta * randn(dim)
#             else
#                 X_randp = pbest[ab[i], :]
#                 Dista = 2 .* (fbest .- fposition[i, :]) * rand()
#                 Distb = 3.0 .* (X_randp .- pbest[i, :]) * rand()
#                 Flag = vec_flag[floor(Int, 2 * rand() + 1)]
#                 if i == 1
#                     fposition[i, :] .= fposition[i, :] .+ Dista .+ Distb * Flag
#                 else
#                     fPos_i = fposition[i, :] .+ Dista .+ Distb * Flag
#                     fposition[i, :] .= (fPos_i .+ fposition[i - 1, :]) ./ 2
#                 end
#             end
#         end

#         # Worst case exploration
#         for i in 1:rupplefoxes
#             if vec_flag[floor(Int, 2 * rand() + 1)] == 1
#                 _, gworst = findmax(fitness)
#                 fposition[gworst, :] .= fposition[gworst, :] .+ beta * randn(dim)
#             end
#         end

#         # Bound handling
#         for i in 1:rupplefoxes
#             fposition[i, :] .= clamp.(fposition[i, :], lb, ub)
#         end

#         # Fitness update
#         for i in 1:rupplefoxes
#             if all(fposition[i, :] .>= lb) && all(fposition[i, :] .<= ub)
#                 x[i, :] .= fposition[i, :]
#                 fit[i] = objfun(fposition[i, :])
#                 if fit[i] < fitness[i]
#                     pbest[i, :] .= fposition[i, :]
#                     fitness[i] = fit[i]
#                 end
#                 if fitness[i] < fmin0
#                     fmin0 = fitness[i]
#                     fbest .= pbest[i, :]
#                 end
#             end
#         end

#         println("Iteration# $ite  Fitness= $fmin0")
#         ccurve[ite] = fmin0
#     end

#     return fmin0, fbest, ccurve
# end

function RFO(rupplefoxes::Int, max_iter::Int,
    lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real},
    dim::Int, objfun::Function)
    # Preallocate
    ccurve = zeros(max_iter)
    vec_flag = [1, -1]
    beta = 1e-10
    L = 100.0

    # Initialize population
    fposition = initialization(rupplefoxes, dim, ub, lb)
    x = copy(fposition)

    # Evaluate initial fitness
    fit = zeros(rupplefoxes)
    for i = 1:rupplefoxes
        fit[i] = objfun(view(fposition, i, :))
    end
    fitness = copy(fit)

    # Initialize bests
    fmin0, idx = findmin(fit)
    pbest = copy(fposition)
    fbest = copy(pbest[idx, :])

    # Main loop
    for ite = 1:max_iter
        # Adaptive parameters
        h = 1.0 / (1 + exp((ite - max_iter / 2) / L))
        s = 1.0 / (1 + exp((max_iter / 2 - ite) / L))
        tmp = 2.0 / (1 + exp((max_iter / 2 - ite) / 100))
        smell = 0.1 / abs(acos(clamp(tmp, -1.0, 1.0)))

        # Daylight vs Night
        if rand() >= 0.5
            # Daylight
            ds = rand(1:rupplefoxes, rupplefoxes)
            for i = 1:rupplefoxes
                if s >= h
                    if rand() >= 0.25
                        Xr = view(pbest, ds[i], :)
                        ri = floor(Int, 4 * rand() + 1)
                        r = rand()
                        fposition[i, :] .= x[i, :] .+ ri * r * (Xr .- x[i, :]) .+ ri * r * (fbest .- x[i, :])
                        θ = rand() * 260 * π / 180
                        fposition[i, :] .= frotate(fposition[i, :], x[i, :], θ, dim)
                    else
                        Flag = vec_flag[rand(1:2)]
                        Xr = view(pbest, ds[i], :)
                        fposition[i, :] .= Xr .+ beta * randn(dim) * Flag
                    end
                else
                    if rand() >= 0.75
                        Xr = view(pbest, ds[i], :)
                        ri = floor(Int, 4 * rand() + 1)
                        r = rand()
                        fposition[i, :] .= x[i, :] .+ ri * r * (Xr .- x[i, :]) .+ ri * r * (fbest .- x[i, :])
                        θ = rand() * 150 * π / 180
                        fposition[i, :] .= frotate(fposition[i, :], x[i, :], θ, dim)
                    else
                        Flag = vec_flag[rand(1:2)]
                        fposition[i, :] .= view(pbest, ds[i], :) .+ beta * randn(dim) * Flag
                    end
                end
            end
        else
            # Night
            dr = rand(1:rupplefoxes, rupplefoxes)
            for i = 1:rupplefoxes
                if h >= s
                    if rand() >= 0.25
                        Xr = view(pbest, dr[i], :)
                        ri = floor(Int, 4 * rand() + 1)
                        r = rand()
                        fposition[i, :] .= x[i, :] .+ ri * r * (Xr .- x[i, :]) .+ ri * r * (fbest .- x[i, :])
                        θ = rand() * 150 * π / 180
                        fposition[i, :] .= frotate(fposition[i, :], x[i, :], θ, dim)
                    else
                        Flag = vec_flag[rand(1:2)]
                        fposition[i, :] .= view(pbest, dr[i], :) .+ beta * randn(dim) * Flag
                    end
                else
                    if rand() >= 0.75
                        Xr = view(pbest, dr[i], :)
                        ri = floor(Int, 4 * rand() + 1)
                        r = rand()
                        fposition[i, :] .= x[i, :] .+ ri * r * (Xr .- x[i, :]) .+ ri * r * (fbest .- x[i, :])
                        θ = rand() * 260 * π / 180
                        fposition[i, :] .= frotate(fposition[i, :], x[i, :], θ, dim)
                    else
                        Flag = vec_flag[rand(1:2)]
                        fposition[i, :] .= view(pbest, dr[i], :) .+ beta * randn(dim) * Flag
                    end
                end
            end
        end

        # Smell behavior
        ss = rand(1:rupplefoxes, rupplefoxes)
        for i = 1:rupplefoxes
            if rand() >= smell
                Xr1 = view(pbest, ss[i], :)
                xr = 2 + rand() * (4 - 2)
                eps = abs(4 * rand() - (rand() + rand())) / xr
                fposition[i, :] .= x[i, :] .+ eps * (Xr1 .- x[i, :]) .+ eps * (fbest .- x[i, :])
            else
                Flag = vec_flag[rand(1:2)]
                fposition[i, :] .= view(pbest, ss[i], :) .+ beta * randn(dim) * Flag
            end
        end

        # Animal behavior
        ab = rand(1:rupplefoxes, rupplefoxes)
        for i = 1:rupplefoxes
            if rand() >= h
                fposition[i, :] .= view(pbest, ab[i], :) .+ beta * randn(dim)
            else
                Xrp = view(pbest, ab[i], :)
                Dista = 2 * (fbest .- fposition[i, :]) * rand()
                Distb = 3 * (Xrp .- view(pbest, i, :)) * rand()
                Flag = vec_flag[rand(1:2)]
                if i == 1
                    fposition[i, :] .= fposition[i, :] .+ Dista .+ Distb * Flag
                else
                    tmpv = fposition[i, :] .+ Dista .+ Distb * Flag
                    fposition[i, :] .= (tmpv .+ fposition[i-1, :]) / 2
                end
            end
        end

        # Worst-case & boundary
        for i = 1:rupplefoxes
            if vec_flag[rand(1:2)] == 1
                _, gw = findmax(fitness)
                fposition[gw, :] .+= beta * randn(dim)
            end
            fposition[i, :] .= clamp.(fposition[i, :], lb, ub)
        end

        # Update fitness & bests
        for i = 1:rupplefoxes
            x[i, :] .= fposition[i, :]
            fit[i] = objfun(view(fposition, i, :))
            if fit[i] < fitness[i]
                pbest[i, :] .= fposition[i, :]
                fitness[i] = fit[i]
            end
            if fitness[i] < fmin0
                fmin0 = fitness[i]
                fbest .= pbest[i, :]
            end
        end

        ccurve[ite] = fmin0
    end

    return fmin0, fbest, ccurve
end

function frotate(x::Vector{T}, y::Vector{T}, θ::T, dim::Int) where {T<:Real}
    # center index
    cent = ceil(Int, dim / 2)

    # build 2×N matrix of points (each column is a point)
    v = [x'; y']  # 2×N

    # center of rotation
    x0, y0 = x[cent], y[cent]
    center = repeat([x0; y0], 1, length(x))  # 2×N

    # 2×2 rotation matrix
    R = [cos(θ) -sin(θ);
        sin(θ) cos(θ)]

    # apply rotation: shift to origin, rotate, shift back
    vo = R * (v .- center) .+ center

    # return the first row (rotated x-coordinates)
    return vec(vo[1, :])
end