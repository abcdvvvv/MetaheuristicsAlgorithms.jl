"""
Rao, R. Venkata, Vimal J. Savsani, and Dipakkumar P. Vakharia. 
"Teaching–learning-based optimization: a novel method for constrained mechanical design optimization problems." 
Computer-aided design 43, no. 3 (2011): 303-315.
"""
mutable struct Individual
    Position::Vector{Float64}
    Cost::Float64
end

function TLBO(nPop, MaxIt,VarMin, VarMax, nVar, CostFunction)
    VarSize = (nVar) 

    pop = [Individual(rand(VarSize) * (VarMax - VarMin) .+ VarMin, 
                    CostFunction(vec(rand(VarSize) .* (VarMax - VarMin) .+ VarMin))) for _ in 1:nPop]


    BestSol = Individual([], Inf)  

    for i in 1:nPop
        pop[i].Cost = CostFunction(vec(pop[i].Position))
        if pop[i].Cost < BestSol.Cost
            BestSol = pop[i]
        end
    end

    BestCosts = zeros(MaxIt)

    for it in 1:MaxIt
        
        Mean = sum(p.Position for p in pop) / nPop
        
        Teacher = pop[1]
        for i in 2:nPop
            if pop[i].Cost < Teacher.Cost
                Teacher = pop[i]
            end
        end
        
        for i in 1:nPop
            newsol = Individual(zeros(VarSize), 0.0)
            
            TF = rand(1:2)
            
            newsol.Position = pop[i].Position .+ rand(VarSize) .* (Teacher.Position .- TF * Mean)
            
            newsol.Position = clamp.(newsol.Position, VarMin, VarMax)
            
            newsol.Cost = CostFunction(newsol.Position)
            
            if newsol.Cost < pop[i].Cost
                pop[i] = newsol
                if pop[i].Cost < BestSol.Cost
                    BestSol = pop[i]
                end
            end
        end
        
        for i in 1:nPop
            A = collect(1:nPop)
            deleteat!(A, i)
            j = A[rand(1:length(A))]
            
            Step = pop[i].Position .- pop[j].Position
            if pop[j].Cost < pop[i].Cost
                Step .= -Step
            end
            
            newsol = Individual(zeros(VarSize), 0.0)
            
            newsol.Position = pop[i].Position .+ rand(VarSize) .* Step
            
            newsol.Position = clamp.(newsol.Position, VarMin, VarMax)
            
            newsol.Cost = CostFunction(newsol.Position)
            
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