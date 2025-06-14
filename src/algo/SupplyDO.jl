"""
Zhao, Weiguo, Liying Wang, and Zhenxing Zhang. 
"Supply-demand-based optimization: A novel economics-inspired algorithm for global optimization." 
Ieee Access 7 (2019): 73182-73206.
"""
function SupplyDO(MarketSize, MaxIt, Low, Up, Dim, FunIndex)

    OneMarket = Dict(
        "CommPrice" => zeros(Dim),
        "CommPriceFit" => Inf,
        "CommQuantity" => zeros(Dim),
        "CommQuantityFit" => Inf
    )
    
    Market = fill(OneMarket, MarketSize)
    BestF = Inf
    BestX = []
    
    for i in 1:MarketSize
        Market[i]["CommPrice"] = rand(Dim) .* (Up - Low) .+ Low
        Market[i]["CommPriceFit"] = FunIndex(Market[i]["CommPrice"])
        Market[i]["CommQuantity"] = rand(Dim) .* (Up - Low) .+ Low
        Market[i]["CommQuantityFit"] = FunIndex(Market[i]["CommQuantity"])

        if Market[i]["CommQuantityFit"] <= Market[i]["CommPriceFit"]
            Market[i]["CommPriceFit"] = Market[i]["CommQuantityFit"]
            Market[i]["CommPrice"] = Market[i]["CommQuantity"]
        end
    end
    
    for i in 1:MarketSize
        if Market[i]["CommPriceFit"] <= BestF
            BestX = Market[i]["CommPrice"]
            BestF = Market[i]["CommPriceFit"]
        end
    end

    HisBestFit = zeros(MaxIt)

    for Iter in 1:MaxIt

        a = 2 * (MaxIt - Iter + 1) / MaxIt
        F = zeros(MarketSize)

        MeanQuantityFit = mean([Market[i]["CommQuantityFit"] for i in 1:MarketSize])
        for i in 1:MarketSize
            F[i] = abs(Market[i]["CommQuantityFit"] - MeanQuantityFit) + 1e-15
        end
        FQ = F / sum(F)

        MeanPriceFit = mean([Market[i]["CommPriceFit"] for i in 1:MarketSize])
        for i in 1:MarketSize
            F[i] = abs(Market[i]["CommPriceFit"] - MeanPriceFit) + 1e-15
        end
        FP = F / sum(F)

        MeanPrice = mean(hcat([Market[i]["CommPrice"] for i in 1:MarketSize]...), dims=2)'

        for i in 1:MarketSize
            Ind = rand() < 0.5 ? 1 : 2
            k = findfirst(x -> x >= rand(), cumsum(FQ))  
            CommQuantityEqu = Market[k]["CommQuantity"]

            Alpha = a * sin.(2 * pi * rand(Dim))
            Beta = 2 * cos.(2 * pi * rand(Dim))

            CommPriceEqu = ifelse(rand() > 0.5, rand() * MeanPrice, Market[findfirst(x -> x >= rand(), cumsum(FP))]["CommPrice"])

            NewCommQuantity = CommQuantityEqu .+ Alpha .* (Market[i]["CommPrice"] .- vec(CommPriceEqu))
            
            NewCommQuantity .= clamp.(NewCommQuantity, Low, Up)
            
            NewCommQuantityFit = FunIndex(NewCommQuantity)

            if NewCommQuantityFit <= Market[i]["CommQuantityFit"]
                Market[i]["CommQuantityFit"] = NewCommQuantityFit
                Market[i]["CommQuantity"] = NewCommQuantity
            end

            NewCommPrice = vec(CommPriceEqu) - Beta .* (NewCommQuantity .- CommQuantityEqu)
            NewCommPrice .= clamp.(NewCommPrice, Low, Up)
            NewCommPriceFit = FunIndex(NewCommPrice)

            if NewCommPriceFit <= Market[i]["CommPriceFit"]
                Market[i]["CommPriceFit"] = NewCommPriceFit
                Market[i]["CommPrice"] = NewCommPrice
            end
        end

        for i in 1:MarketSize
            if Market[i]["CommQuantityFit"] <= Market[i]["CommPriceFit"]
                Market[i]["CommPriceFit"] = Market[i]["CommQuantityFit"]
                Market[i]["CommPrice"] = Market[i]["CommQuantity"]
            end

            if Market[i]["CommPriceFit"] <= BestF
                BestX = Market[i]["CommPrice"]
                BestF = Market[i]["CommPriceFit"]
            end
        end

        HisBestFit[Iter] = BestF
    end

    return BestF, BestX, HisBestFit
end