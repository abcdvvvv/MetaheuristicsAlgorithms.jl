"""
Saremi, Shahrzad, Seyedali Mirjalili, and Andrew Lewis. 
"Grasshopper optimisation algorithm: theory and application." 
Advances in engineering software 105 (2017): 30-47.
"""

using LinearAlgebra

function distance(a, b)
    return sqrt((a[1] - b[1])^2 + (a[2] - b[2])^2)
end

function S_func(r)
    f = 0.5
    l = 1.5
    o = f * exp(-r / l) - exp(-r)  # Eq. (2.3) in the paper
    return o
end

function GOA(N, Max_iter, lb, ub, dim, fobj)

    # println("GOA is now estimating the global optimum for your problem...")

    flag = false
    if length(ub) == 1
        ub = ones(dim) * ub
        lb = ones(dim) * lb
    end

    if mod(dim, 2) != 0
        dim += 1
        ub = vcat(ub, 100)
        lb = vcat(lb, -100)
        flag = true
    end

    # Initialize the population of grasshoppers
    GrassHopperPositions = initialization(N, dim, ub, lb)
    GrassHopperFitness = zeros(N)

    fitness_history = zeros(N, Max_iter)
    position_history = zeros(N, Max_iter, dim)
    Convergence_curve = zeros(Max_iter)
    Trajectories = zeros(N, Max_iter)

    cMax = 1.0
    cMin = 0.00004

    # Calculate the fitness of initial grasshoppers
    for i in 1:N
        if flag
            GrassHopperFitness[i] = fobj(GrassHopperPositions[i, 1:end-1])
        else
            GrassHopperFitness[i] = fobj(GrassHopperPositions[i, :])
        end
        fitness_history[i, 1] = GrassHopperFitness[i]
        position_history[i, 1, :] = GrassHopperPositions[i, :]
        Trajectories[:, 1] = GrassHopperPositions[:, 1]
    end

    sorted_indexes = sortperm(GrassHopperFitness)
    Sorted_grasshopper = GrassHopperPositions[sorted_indexes, :]

    TargetPosition = Sorted_grasshopper[1, :]
    TargetFitness = GrassHopperFitness[sorted_indexes[1]]

    # Main loop
    for l in 2:Max_iter
        c = cMax - l * ((cMax - cMin) / Max_iter)

        GrassHopperPositions_temp = zeros(N, dim)

        for i in 1:N
            temp = GrassHopperPositions'
            S_i_total = zeros(dim)
            for k in 1:2:dim
                S_i = zeros(2)
                for j in 1:N
                    if i != j
                        Dist = distance(temp[k:k+1, j], temp[k:k+1, i])
                        r_ij_vec = (temp[k:k+1, j] - temp[k:k+1, i]) / (Dist + eps())
                        xj_xi = 2 + mod(Dist, 2)
                        s_ij = ((ub[k:k+1] - lb[k:k+1]) * c / 2) .* S_func(xj_xi) .* r_ij_vec
                        S_i += s_ij
                    end
                end
                S_i_total[k:k+1] = S_i
            end
            # X_new = c * S_i_total' + TargetPosition
            X_new = c * vec(S_i_total') + TargetPosition

            GrassHopperPositions_temp[i, :] = X_new'
        end

        GrassHopperPositions = GrassHopperPositions_temp

        for i in 1:N
            # Tp = GrassHopperPositions[i, :] .> ub'
            # Tm = GrassHopperPositions[i, :] .< lb'
            # GrassHopperPositions[i, :] = GrassHopperPositions[i, :] .* .~(Tp .+ Tm) .+ ub' .* Tp .+ lb' .* Tm
            # GrassHopperPositions[i, :] = max.(min.(GrassHopperPositions[i, :], ub'), lb')
            GrassHopperPositions[i, :] = max.(min.(GrassHopperPositions[i, :], ub), lb)

            if flag
                GrassHopperFitness[i] = fobj(GrassHopperPositions[i, 1:end-1])
            else
                GrassHopperFitness[i] = fobj(GrassHopperPositions[i, :])
            end

            fitness_history[i, l] = GrassHopperFitness[i]
            position_history[i, l, :] = GrassHopperPositions[i, :]
            Trajectories[:, l] = GrassHopperPositions[:, 1]

            if GrassHopperFitness[i] < TargetFitness
                TargetPosition = GrassHopperPositions[i, :]
                TargetFitness = GrassHopperFitness[i]
            end
        end

        Convergence_curve[l] = TargetFitness
        # println("In iteration #", l, ", target's objective = ", TargetFitness)
    end

    if flag
        TargetPosition = TargetPosition[1:dim-1]
    end

    return TargetFitness, TargetPosition, Convergence_curve, Trajectories, fitness_history, position_history
end

