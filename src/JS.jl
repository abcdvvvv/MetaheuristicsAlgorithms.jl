"""
Chou, Jui-Sheng, and Dinh-Nhat Truong. 
"A novel metaheuristic optimizer inspired by behavior of jellyfish in ocean." 
Applied Mathematics and Computation 389 (2021): 125535.
"""
function JS(nPop, MaxIt, Lb, Ub, nd, CostFunction)

    nVar = nd                           
    VarSize = (nVar)                 
    if length(Lb) == 1
        VarMin = Lb .* ones(nd)      
        VarMax = Ub .* ones(nd)      
    else
        VarMin = Lb
        VarMax = Ub
    end


    popi = initialization(nPop, nd, VarMax, VarMin)
    it = 0

    BestSol = zeros(nd)

    fval = Inf

    popCost = zeros(nPop)
    for i in 1:nPop
        popCost[i] = CostFunction(popi[i, :])
    end

    fbestvl = zeros(MaxIt)

    for it in 1:MaxIt
        Meanvl = mean(popi, dims=1)[:]
        sorted_costs = sortperm(popCost)
        BestSol = popi[sorted_costs[1], :]
        BestCost = popCost[sorted_costs[1]]

        for i in 1:nPop
            Ar = (1 - it * ((1) / MaxIt)) * (2 * rand() - 1)

            if abs(Ar) >= 0.5
                newsol = popi[i, :] .+ rand(VarSize) .* (BestSol .- 3 * rand() * Meanvl)
                newsol = max.(min.(newsol, VarMax), VarMin)

                newsolCost = CostFunction(newsol)

                if newsolCost < popCost[i]
                    popi[i, :] = newsol
                    popCost[i] = newsolCost
                    if popCost[i] < BestCost
                        BestCost = popCost[i]
                        BestSol = popi[i, :]
                    end
                end
            else
                if rand() <= (1 - Ar)
                    j = rand(1:nPop)
                    while j == i
                        j = rand(1:nPop)
                    end
                    Step = popi[i, :] .- popi[j, :]
                    if popCost[j] < popCost[i]
                        Step = -Step
                    end
                    newsol = popi[i, :] .+ rand(VarSize) .* Step
                else
                    newsol = popi[i, :] .+ 0.1 * (VarMax .- VarMin) .* rand()
                end

                newsol = max.(min.(newsol, VarMax), VarMin)
                newsolCost = CostFunction(newsol)

                if newsolCost < popCost[i]
                    popi[i, :] = newsol
                    popCost[i] = newsolCost
                    if popCost[i] < BestCost
                        BestCost = popCost[i]
                        BestSol = popi[i, :]
                    end
                end
            end
        end

        fbestvl[it] = BestCost
    end

    u = BestSol
    fval = fbestvl[end]
    return fval, u, fbestvl
end
