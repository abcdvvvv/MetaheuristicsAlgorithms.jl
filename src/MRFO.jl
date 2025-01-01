"""
Zhao, Weiguo, Zhenxing Zhang, and Liying Wang. 
"Manta ray foraging optimization: An effective bio-inspired optimizer for engineering applications." 
Engineering Applications of Artificial Intelligence 87 (2020): 103300.
"""
function MRFO(nPop, MaxIt, Low, Up, Dim, F_index)#(F_index, MaxIt, nPop)
    
    PopPos = [rand(Dim) .* (Up - Low) .+ Low for _ in 1:nPop]
    PopFit = [F_index(PopPos[i]) for i in 1:nPop]

    BestF = Inf
    BestX = []

    for i in 1:nPop
        if PopFit[i] <= BestF
            BestF = PopFit[i]
            BestX = PopPos[i]
        end
    end

    HisBestFit = zeros(MaxIt)

    for It in 1:MaxIt
        Coef = It / MaxIt
        
        if rand() < 0.5
            r1 = rand()
            Beta = 2 * exp(r1 * ((MaxIt - It + 1) / MaxIt)) * sin(2 * pi * r1)
            if Coef > rand()
                newPopPos1 = BestX .+ rand(Dim) .* (BestX .- PopPos[1]) .+ Beta .* (BestX .- PopPos[1])  # Equation (4)
            else
                IndivRand = rand(Dim) .* (Up - Low) .+ Low
                newPopPos1 = IndivRand .+ rand(Dim) .* (IndivRand .- PopPos[1]) .+ Beta .* (IndivRand .- PopPos[1])  # Equation (7)
            end
        else
            Alpha = 2 * rand(Dim) .* (-log.(rand(Dim))).^0.5
            newPopPos1 = PopPos[1] .+ rand(Dim) .* (BestX .- PopPos[1]) .+ Alpha .* (BestX .- PopPos[1])  # Equation (1)
        end
        
        newPopPos = [zeros(Dim) for _ in 1:nPop]
        newPopPos[1] = newPopPos1

        for i in 2:nPop
            if rand() < 0.5
                r1 = rand()
                Beta = 2 * exp(r1 * ((MaxIt - It + 1) / MaxIt)) * sin(2 * pi * r1)
                if Coef > rand()
                    newPopPos[i] = BestX .+ rand(Dim) .* (PopPos[i - 1] .- PopPos[i]) .+ Beta .* (BestX .- PopPos[i])  # Equation (4)
                else
                    IndivRand = rand(Dim) .* (Up - Low) .+ Low
                    newPopPos[i] = IndivRand .+ rand(Dim) .* (PopPos[i - 1] .- PopPos[i]) .+ Beta .* (IndivRand .- PopPos[i])  # Equation (7)
                end
            else
                Alpha = 2 * rand(Dim) .* (-log.(rand(Dim))).^0.5
                newPopPos[i] = PopPos[i] .+ rand(Dim) .* (PopPos[i - 1] .- PopPos[i]) .+ Alpha .* (BestX .- PopPos[i])  # Equation (1)
            end
        end
        
        newPopFit = zeros(nPop)
        for i in 1:nPop
            newPopPos[i] = max.(min.(newPopPos[i], Up), Low)
            newPopFit[i] = F_index(newPopPos[i])
            if newPopFit[i] < PopFit[i]
                PopFit[i] = newPopFit[i]
                PopPos[i] = newPopPos[i]
            end
        end
        
        S = 2
        for i in 1:nPop
            newPopPos[i] = PopPos[i] .+ S * (rand() * BestX .- rand() * PopPos[i])  # Equation (8)
        end

        for i in 1:nPop
            newPopPos[i] = max.(min.(newPopPos[i], Up), Low)
            newPopFit[i] = F_index(newPopPos[i])
            if newPopFit[i] < PopFit[i]
                PopFit[i] = newPopFit[i]
                PopPos[i] = newPopPos[i]
            end
        end

        for i in 1:nPop
            if PopFit[i] < BestF
                BestF = PopFit[i]
                BestX = PopPos[i]
            end
        end

        HisBestFit[It] = BestF
    end
    return BestF, BestX, HisBestFit
end