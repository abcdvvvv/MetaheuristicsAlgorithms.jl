"""
# References: 

- Naruei, Iraj, and Farshid Keynia. "A new optimization method based on COOT bird natural life model." Expert Systems with Applications 183 (2021): 115352.

"""
function COOT(objfun, lb::Vector{Float64}, ub::Vector{Float64}, npop::Integer, max_iter::Integer)
    dim = length(lb)
    if length(ub) == 1
        ub = ones(dim) * ub
        lb = ones(dim) * lb
    end

    NLeader = ceil(Int, 0.1 * npop)
    Ncoot = npop - NLeader
    Convergence_curve = zeros(max_iter)
    gBest = zeros(dim)
    gBestScore = Inf

    CootPos = rand(Ncoot, dim) .* (ub - lb)' .+ lb'
    CootFitness = zeros(Ncoot)

    LeaderPos = rand(NLeader, dim) .* (ub - lb)' .+ lb'
    LeaderFit = zeros(NLeader)

    for i = 1:Ncoot
        CootFitness[i] = objfun(CootPos[i, :])
        if gBestScore > CootFitness[i]
            gBestScore = CootFitness[i]
            gBest = CootPos[i, :]
        end
    end

    for i = 1:NLeader
        LeaderFit[i] = objfun(LeaderPos[i, :])
        if gBestScore > LeaderFit[i]
            gBestScore = LeaderFit[i]
            gBest = LeaderPos[i, :]
        end
    end

    Convergence_curve[1] = gBestScore
    l = 2

    while l <= max_iter
        B = 2 - l * (1 / max_iter)
        A = 1 - l * (1 / max_iter)

        for i = 1:Ncoot
            R = rand() < 0.5 ? -1 + 2 * rand() : (-1 .+ 2 * rand(dim))
            R1 = rand() < 0.5 ? rand() : rand(dim)
            k = 1 + mod(i - 1, NLeader)

            if rand() < 0.5
                CootPos[i, :] = 2 * R1 .* cos.(2 * π * R) .* (LeaderPos[k, :] - CootPos[i, :]) + LeaderPos[k, :]
            else
                if rand() < 0.5 && i != 1
                    CootPos[i, :] = (CootPos[i, :] + CootPos[i-1, :]) / 2
                else
                    Q = rand(dim) .* (ub - lb) .+ lb
                    CootPos[i, :] = CootPos[i, :] .+ A .* R1 .* (Q .- CootPos[i, :])
                end
            end

            CootPos[i, :] = max.(min.(CootPos[i, :], ub), lb)
        end

        for i = 1:Ncoot
            CootFitness[i] = objfun(CootPos[i, :])
            k = 1 + mod(i - 1, NLeader)

            if CootFitness[i] < LeaderFit[k]
                Temp = LeaderPos[k, :]
                TempFit = LeaderFit[k]
                LeaderFit[k] = CootFitness[i]
                LeaderPos[k, :] = CootPos[i, :]
                CootFitness[i] = TempFit
                CootPos[i, :] = Temp
            end
        end

        for i = 1:NLeader
            R = rand() < 0.5 ? fill(-1 + 2 * rand(), dim) : -1 .+ 2 * rand(dim)
            R3 = rand() < 0.5 ? rand() : rand(dim)

            if rand() < 0.5
                Temp = B * R3 .* cos.(2 * π * R) .* (gBest - LeaderPos[i, :]) + gBest
            else
                Temp = B * R3 .* cos.(2 * π * R) .* (gBest - LeaderPos[i, :]) - gBest
            end

            Temp = max.(min.(Temp, ub), lb)

            TempFit = objfun(Temp)

            if gBestScore > TempFit
                LeaderFit[i] = gBestScore
                LeaderPos[i, :] = gBest
                gBestScore = TempFit
                gBest = Temp
            end
        end

        Convergence_curve[l] = gBestScore
        l += 1
    end

    # return gBestScore, gBest, Convergence_curve
    return OptimizationResult(
        gBest,
        gBestScore,
        Convergence_curve)
end