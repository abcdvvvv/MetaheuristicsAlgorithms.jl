"""
Zhao, Weiguo, Liying Wang, and Seyedali Mirjalili. 
"Artificial hummingbird algorithm: A new bio-inspired optimizer with its engineering applications." 
Computer Methods in Applied Mechanics and Engineering 388 (2022): 114194.
"""

using Random

function AHA(nPop, MaxIt, Low, Up, Dim, FunIndex)#(FunIndex, MaxIt, nPop)
    
    PopPos = rand(nPop, Dim) .* (Up .- Low) .+ Low
    PopFit = zeros(nPop)

    for i in 1:nPop
        PopFit[i] = FunIndex(PopPos[i, :])
    end

    BestF = Inf
    BestX = []

    for i in 1:nPop
        if PopFit[i] <= BestF
            BestF = PopFit[i]
            BestX = PopPos[i, :]
        end
    end

    HisBestFit = zeros(MaxIt)
    VisitTable = zeros(nPop, nPop)
    VisitTable .= NaN
    for i in 1:nPop
        VisitTable[i, i] = NaN
    end

    for It in 1:MaxIt
        DirectVector = zeros(nPop, Dim) 

        for i in 1:nPop
            r = rand()
            if r < 1/3 
                RandDim = randperm(Dim)
                RandNum = Dim >= 3 ? ceil(Int, rand() * (Dim - 2) + 1) : ceil(Int, rand() * (Dim - 1) + 1)
                DirectVector[i, RandDim[1:RandNum]] .= 1
            else
                if r > 2/3 
                    DirectVector[i, :] .= 1.0
                else 
                    RandNum = ceil(Int, rand() * Dim)
                    DirectVector[i, RandNum] = 1.0
                end
            end

            if rand() < 0.5 
                MaxUnvisitedTime, TargetFoodIndex = findmax(VisitTable[i, :])
                MUT_Index = findall(VisitTable[i, :] .== MaxUnvisitedTime)

                if length(MUT_Index) > 1
                    _, Ind = findmin(PopFit[MUT_Index])
                    TargetFoodIndex = MUT_Index[Ind]
                end

                newPopPos = PopPos[TargetFoodIndex, :] .+ randn() * DirectVector[i, :] .* (PopPos[i, :] .- PopPos[TargetFoodIndex, :])
                newPopPos .= max.(min.(newPopPos, Up), Low)
                newPopFit = FunIndex(newPopPos)

                if newPopFit < PopFit[i]
                    PopFit[i] = newPopFit
                    PopPos[i, :] = newPopPos
                    VisitTable[i, :] .+= 1
                    VisitTable[i, TargetFoodIndex] = 0
                    VisitTable[:, i] .= maximum(VisitTable, dims=2) .+ 1
                    VisitTable[i, i] = NaN
                else
                    VisitTable[i, :] .+= 1
                    VisitTable[i, TargetFoodIndex] = 0
                end
            else 
                newPopPos = PopPos[i, :] .+ randn() * DirectVector[i, :] .* PopPos[i, :]
                newPopPos .= max.(min.(newPopPos, Up), Low)
                newPopFit = FunIndex(newPopPos)

                if newPopFit < PopFit[i]
                    PopFit[i] = newPopFit
                    PopPos[i, :] = newPopPos
                    VisitTable[i, :] .+= 1
                    VisitTable[:, i] .= maximum(VisitTable, dims=2) .+ 1
                    VisitTable[i, i] = NaN
                else
                    VisitTable[i, :] .+= 1
                end
            end
        end

        if It % (2 * nPop) == 0 
            _, MigrationIndex = findmax(PopFit)
            PopPos[MigrationIndex, :] = rand(1, Dim) .* (Up .- Low) .+ Low
            PopFit[MigrationIndex] = FunIndex(PopPos[MigrationIndex, :])
            VisitTable[MigrationIndex, :] .+= 1
            VisitTable[:, MigrationIndex] .= maximum(VisitTable, dims=2) .+ 1
            VisitTable[MigrationIndex, MigrationIndex] = NaN
        end

        for i in 1:nPop
            if PopFit[i] < BestF
                BestF = PopFit[i]
                BestX = PopPos[i, :]
            end
        end

        HisBestFit[It] = BestF
    end

    return  BestF, BestX,HisBestFit
end
