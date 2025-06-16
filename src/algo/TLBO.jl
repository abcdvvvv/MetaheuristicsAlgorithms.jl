"""
Rao, R. Venkata, Vimal J. Savsani, and Dipakkumar P. Vakharia. 
"Teaching–learning-based optimization: a novel method for constrained mechanical design optimization problems." 
Computer-aided design 43, no. 3 (2011): 303-315.
"""
mutable struct individual
    Position::Vector{Float64}
    Cost::Float64
end

function TLBO(npop, max_iter, lb, ub, dim, objfun)
    VarSize = (dim)

    pop = [individual(rand(VarSize) * (ub - lb) .+ lb,
        objfun(vec(rand(VarSize) .* (ub - lb) .+ lb))) for _ = 1:npop]

    BestSol = individual([], Inf)

    for i = 1:npop
        pop[i].Cost = objfun(vec(pop[i].Position))
        if pop[i].Cost < BestSol.Cost
            BestSol = pop[i]
        end
    end

    BestCosts = zeros(max_iter)

    for it = 1:max_iter
        Mean = sum(p.Position for p in pop) / npop

        Teacher = pop[1]
        for i = 2:npop
            if pop[i].Cost < Teacher.Cost
                Teacher = pop[i]
            end
        end

        for i = 1:npop
            newsol = individual(zeros(VarSize), 0.0)

            TF = rand(1:2)

            newsol.Position = pop[i].Position .+ rand(VarSize) .* (Teacher.Position .- TF * Mean)

            newsol.Position = clamp.(newsol.Position, lb, ub)

            newsol.Cost = objfun(newsol.Position)

            if newsol.Cost < pop[i].Cost
                pop[i] = newsol
                if pop[i].Cost < BestSol.Cost
                    BestSol = pop[i]
                end
            end
        end

        for i = 1:npop
            A = collect(1:npop)
            deleteat!(A, i)
            j = A[rand(1:length(A))]

            Step = pop[i].Position .- pop[j].Position
            if pop[j].Cost < pop[i].Cost
                Step .= -Step
            end

            newsol = individual(zeros(VarSize), 0.0)

            newsol.Position = pop[i].Position .+ rand(VarSize) .* Step

            newsol.Position = clamp.(newsol.Position, lb, ub)

            newsol.Cost = objfun(newsol.Position)

            if newsol.Cost < pop[i].Cost
                pop[i] = newsol
                if pop[i].Cost < BestSol.Cost
                    BestSol = pop[i]
                end
            end
        end

        BestCosts[it] = BestSol.Cost
    end

    return BestSol.Cost, BestSol, BestCosts
end