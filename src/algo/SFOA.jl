"""
Zhong, C., Li, G., Meng, Z., Li, H., Yildiz, A. R., & Mirjalili, S. (2025). 
Starfish optimization algorithm (SFOA): a bio-inspired metaheuristic algorithm for global optimization compared with 100 optimizers. 
Neural Computing and Applications, 37(5), 3641-3683.
"""

# function SHO(N, Max_iterations, lowerbound, upperbound, dimension, fitness)

function SFOA(Npop, Max_it, lb, ub, nD, fobj)
    # Starfish Optimization Algorithm (SFOA)
    GP = 0.5  # parameter

    # Ensure bounds are vectors of length nD
    lb = length(lb) == 1 ? fill(lb, nD) : lb
    ub = length(ub) == 1 ? fill(ub, nD) : ub

    fvalbest = Inf
    Curve = zeros(Max_it)
    Xpos = [lb .+ rand(nD) .* (ub .- lb) for _ in 1:Npop]
    Fitness = [fobj(X) for X in Xpos]

    fvalbest, order = findmin(Fitness)
    xposbest = copy(Xpos[order])

    newX = [zeros(nD) for _ in 1:Npop]

    T = 1
    while T <= Max_it
        theta = π / 2 * T / Max_it
        tEO = (Max_it - T) / Max_it * cos(theta)

        if rand() < GP  # exploration
            for i in 1:Npop
                if nD > 5
                    jp1 = randperm(nD)[1:5]
                    for j in jp1
                        pm = (2rand() - 1) * π
                        if rand() < GP
                            newX[i][j] = Xpos[i][j] + pm * (xposbest[j] - Xpos[i][j]) * cos(theta)
                        else
                            newX[i][j] = Xpos[i][j] - pm * (xposbest[j] - Xpos[i][j]) * sin(theta)
                        end
                        if newX[i][j] > ub[j] || newX[i][j] < lb[j]
                            newX[i][j] = Xpos[i][j]
                        end
                    end
                else
                    jp2 = rand(1:nD)
                    im = randperm(Npop)
                    rand1 = 2rand() - 1
                    rand2 = 2rand() - 1
                    newX[i][jp2] = tEO * Xpos[i][jp2] + rand1 * (Xpos[im[1]][jp2] - Xpos[i][jp2]) + rand2 * (Xpos[im[2]][jp2] - Xpos[i][jp2])
                    if newX[i][jp2] > ub[jp2] || newX[i][jp2] < lb[jp2]
                        newX[i][jp2] = Xpos[i][jp2]
                    end
                end
                newX[i] = clamp.(newX[i], lb, ub)
            end
        else  # exploitation
            df = randperm(Npop)[1:5]
            dm = [xposbest .- Xpos[d] for d in df]
            for i in 1:Npop
                r1, r2 = rand(), rand()
                kp = randperm(5)[1:2]
                newX[i] = Xpos[i] .+ r1 .* dm[kp[1]] .+ r2 .* dm[kp[2]]
                if i == Npop
                    newX[i] = exp(-T * Npop / Max_it) .* Xpos[i]
                end
                newX[i] = clamp.(newX[i], lb, ub)
            end
        end

        # Fitness evaluation
        for i in 1:Npop
            newFit = fobj(newX[i])
            if newFit < Fitness[i]
                Fitness[i] = newFit
                Xpos[i] = copy(newX[i])
                if newFit < fvalbest
                    fvalbest = newFit
                    xposbest = copy(Xpos[i])
                end
            end
        end

        Curve[T] = fvalbest
        T += 1
    end

    #  return TargetFitness, TargetPosition, Convergence_curve,
    return fvalbest, xposbest, Curve 
end
