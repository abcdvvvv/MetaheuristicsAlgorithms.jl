"""
Mirjalili, Seyedali, Amir H. Gandomi, Seyedeh Zahra Mirjalili, Shahrzad Saremi, Hossam Faris, and Seyed Mohammad Mirjalili. 
"Salp Swarm Algorithm: A bio-inspired optimizer for engineering design problems." 
Advances in engineering software 114 (2017): 163-191.
"""
function SSA(N::Int, max_iter::Int, lb::Union{Int,AbstractVector}, ub::Union{Int,AbstractVector}, dim::Int, objfun::Function)#(N, max_iter, lb, ub, dim, objfun)#(N, max_iter, objfun, dim, lb, ub)
    if size(ub, 1) == 1
        ub = ones(dim) * ub
        lb = ones(dim) * lb
    end

    Convergence_curve = zeros(max_iter)

    SalpPositions = initialization(N, dim, ub, lb)

    FoodPosition = zeros(dim)
    FoodFitness = Inf

    SalpFitness = zeros(N)
    for i in axes(SalpPositions, 1)
        SalpFitness[i] = objfun(SalpPositions[i, :])
    end

    sorted_indexes = sortperm(SalpFitness)
    Sorted_salps = SalpPositions[sorted_indexes, :]

    FoodPosition .= Sorted_salps[1, :]
    FoodFitness = SalpFitness[sorted_indexes[1]]

    l = 2
    while l <= max_iter
        c1 = 2 * exp(-((4 * l / max_iter)^2))
        for i = 1:size(SalpPositions, 1)
            SalpPositions = SalpPositions'

            if i <= N / 2
                for j = 1:dim
                    c2 = rand()
                    c3 = rand()
                    if c3 < 0.5
                        SalpPositions[i, j] = FoodPosition[j] + c1 * ((ub[j] - lb[j]) * c2 + lb[j])
                    else
                        SalpPositions[i, j] = FoodPosition[j] - c1 * ((ub[j] - lb[j]) * c2 + lb[j])
                    end
                end
            elseif i > N / 2 && i < N + 1
                point1 = SalpPositions[:, i-1]
                point2 = SalpPositions[:, i]
                SalpPositions[:, i] = (point2 .+ point1) / 2
            end

            SalpPositions = SalpPositions'
        end

        for i in axes(SalpPositions, 1)
            SalpPositions[i, :] = max.(min.(SalpPositions[i, :], ub), lb)

            SalpFitness[i] = objfun(SalpPositions[i, :])

            if SalpFitness[i] < FoodFitness
                FoodPosition .= SalpPositions[i, :]
                FoodFitness = SalpFitness[i]
            end
        end
        Convergence_curve[l] = FoodFitness
        l += 1
    end

    return FoodFitness, FoodPosition, Convergence_curve
end