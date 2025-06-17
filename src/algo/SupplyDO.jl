"""
Zhao, Weiguo, Liying Wang, and Zhenxing Zhang. 
"Supply-demand-based optimization: A novel economics-inspired algorithm for global optimization." 
Ieee Access 7 (2019): 73182-73206.
"""
function SupplyDO(MarketSize::Int, max_iter::Int, lb::Union{Real, AbstractVector}, ub::Union{Real, AbstractVector}, dim::Int, objfun)
    OneMarket = Dict(
        "CommPrice" => zeros(dim),
        "CommPriceFit" => Inf,
        "CommQuantity" => zeros(dim),
        "CommQuantityFit" => Inf,
    )

    Market = fill(OneMarket, MarketSize)
    best_f = Inf
    best_x = []

    for i = 1:MarketSize
        Market[i]["CommPrice"] = rand(dim) .* (ub - lb) .+ lb
        Market[i]["CommPriceFit"] = objfun(Market[i]["CommPrice"])
        Market[i]["CommQuantity"] = rand(dim) .* (ub - lb) .+ lb
        Market[i]["CommQuantityFit"] = objfun(Market[i]["CommQuantity"])

        if Market[i]["CommQuantityFit"] <= Market[i]["CommPriceFit"]
            Market[i]["CommPriceFit"] = Market[i]["CommQuantityFit"]
            Market[i]["CommPrice"] = Market[i]["CommQuantity"]
        end
    end

    for i = 1:MarketSize
        if Market[i]["CommPriceFit"] <= best_f
            best_x = Market[i]["CommPrice"]
            best_f = Market[i]["CommPriceFit"]
        end
    end

    his_best_fit = zeros(max_iter)

    for Iter = 1:max_iter
        a = 2 * (max_iter - Iter + 1) / max_iter
        F = zeros(MarketSize)

        MeanQuantityFit = mean([Market[i]["CommQuantityFit"] for i = 1:MarketSize])
        for i = 1:MarketSize
            F[i] = abs(Market[i]["CommQuantityFit"] - MeanQuantityFit) + 1e-15
        end
        FQ = F / sum(F)

        MeanPriceFit = mean([Market[i]["CommPriceFit"] for i = 1:MarketSize])
        for i = 1:MarketSize
            F[i] = abs(Market[i]["CommPriceFit"] - MeanPriceFit) + 1e-15
        end
        FP = F / sum(F)

        MeanPrice = mean(hcat([Market[i]["CommPrice"] for i = 1:MarketSize]...), dims=2)'

        for i = 1:MarketSize
            Ind = rand() < 0.5 ? 1 : 2
            k = findfirst(x -> x >= rand(), cumsum(FQ))
            CommQuantityEqu = Market[k]["CommQuantity"]

            Alpha = a * sin.(2 * pi * rand(dim))
            Beta = 2 * cos.(2 * pi * rand(dim))

            CommPriceEqu = ifelse(rand() > 0.5, rand() * MeanPrice, Market[findfirst(x -> x >= rand(), cumsum(FP))]["CommPrice"])

            NewCommQuantity = CommQuantityEqu .+ Alpha .* (Market[i]["CommPrice"] .- vec(CommPriceEqu))

            NewCommQuantity .= clamp.(NewCommQuantity, lb, ub)

            NewCommQuantityFit = objfun(NewCommQuantity)

            if NewCommQuantityFit <= Market[i]["CommQuantityFit"]
                Market[i]["CommQuantityFit"] = NewCommQuantityFit
                Market[i]["CommQuantity"] = NewCommQuantity
            end

            NewCommPrice = vec(CommPriceEqu) - Beta .* (NewCommQuantity .- CommQuantityEqu)
            NewCommPrice .= clamp.(NewCommPrice, lb, ub)
            NewCommPriceFit = objfun(NewCommPrice)

            if NewCommPriceFit <= Market[i]["CommPriceFit"]
                Market[i]["CommPriceFit"] = NewCommPriceFit
                Market[i]["CommPrice"] = NewCommPrice
            end
        end

        for i = 1:MarketSize
            if Market[i]["CommQuantityFit"] <= Market[i]["CommPriceFit"]
                Market[i]["CommPriceFit"] = Market[i]["CommQuantityFit"]
                Market[i]["CommPrice"] = Market[i]["CommQuantity"]
            end

            if Market[i]["CommPriceFit"] <= best_f
                best_x = Market[i]["CommPrice"]
                best_f = Market[i]["CommPriceFit"]
            end
        end

        his_best_fit[Iter] = best_f
    end

    # return best_f, best_x, his_best_fit
    return OptimizationResult(
        best_x,
        best_f,
        his_best_fit)
end