"""
Braik, Malik, Abdelaziz Hammouri, Jaffar Atwan, Mohammed Azmi Al-Betar, and Mohammed A. Awadallah. 
"White Shark Optimizer: A novel bio-inspired meta-heuristic algorithm for global optimization problems." 
Knowledge-Based Systems 243 (2022): 108457.

"""
function WSO(whiteSharks, itemax, lb, ub, dim, fobj)
    ccurve = zeros(itemax)

    WSO_Positions = initialization(whiteSharks, dim, ub, lb)  

    v = zeros(size(WSO_Positions))

    fit = [fobj(WSO_Positions[i, :]) for i in 1:whiteSharks]

    fitness = copy(fit)  

    fmin0, index = findmin(fit)

    wbest = copy(WSO_Positions)  
    gbest = copy(WSO_Positions[index, :])  

    fmax = 0.75  
    fmin = 0.07  
    tau = 4.11
    mu = 2 / abs(2 - tau - sqrt(tau^2 - 4 * tau))
    pmin, pmax = 0.5, 1.5
    a0, a1, a2 = 6.250, 100, 0.0005

    for ite in 1:itemax
        mv = 1 / (a0 + exp((itemax / 2.0 - ite) / a1))
        s_s = abs((1 - exp(-a2 * ite / itemax)))

        p1 = pmax + (pmax - pmin) * exp(-(4 * ite / itemax)^2)
        p2 = pmin + (pmax - pmin) * exp(-(4 * ite / itemax)^2)

        nu = rand(1:whiteSharks, whiteSharks)

        for i in 1:whiteSharks
            rmin, rmax = 1.0, 3.0
            rr = rmin + rand() * (rmax - rmin)
            wr = abs(((2 * rand()) - (1 * rand() + rand())) / rr)
            v[i, :] = mu * v[i, :] + wr * (wbest[nu[i], :] - WSO_Positions[i, :])
        end

        for i in 1:whiteSharks
            f = fmin + (fmax - fmin) / (fmax + fmin)

            a = sign.(WSO_Positions[i, :] .- ub) .> 0
            b = sign.(WSO_Positions[i, :] .- lb) .< 0

            wo = .~(a .& b)

            if rand() < mv
                WSO_Positions[i, :] = WSO_Positions[i, :] .* .!wo + ub .* a + lb .* b
            else
                WSO_Positions[i, :] = WSO_Positions[i, :] + v[i, :] / f  
            end
        end

        for i in 1:whiteSharks
            for j in 1:dim
                if rand() < s_s
                    Dist = abs(rand() * (gbest[j] - 1 * WSO_Positions[i, j]))

                    if i == 1
                        WSO_Positions[i, j] = gbest[j] + rand() * Dist * sign(rand() - 0.5)
                    else
                        WSO_Pos = copy(WSO_Positions)
                        WSO_Pos[i, j] = gbest[j] + rand() * Dist * sign(rand() - 0.5)
                        WSO_Positions[i, j] = (WSO_Pos[i, j] + WSO_Positions[i - 1, j]) / 2 * rand()
                    end
                end
            end
        end

        for i in 1:whiteSharks
            if all(WSO_Positions[i, :] .>= lb) && all(WSO_Positions[i, :] .<= ub)
                fit[i] = fobj(WSO_Positions[i, :])

                if fit[i] < fitness[i]
                    wbest[i, :] = WSO_Positions[i, :]  
                    fitness[i] = fit[i]  
                end

                if fitness[i] < fmin0
                    fmin0 = fitness[i]
                    gbest = copy(wbest[index, :])  
                end
            end
        end
        ccurve[ite] = fmin0
    end

    return fmin0, gbest, ccurve
end