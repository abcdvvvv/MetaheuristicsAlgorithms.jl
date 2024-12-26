"""
Alsattar, Hassan A., A. A. Zaidan, and B. B. Zaidan. 
"Novel meta-heuristic bald eagle search optimisation algorithm." 
Artificial Intelligence Review 53 (2020): 2237-2264.
"""

function polr(a, R, N)
    th = a * π * rand(N)  # Generate theta
    r = th .+ R * rand(N)  # Calculate r

    xR = r .* sin.(th)  # Element-wise sin
    yR = r .* cos.(th)  # Element-wise cos

    # Normalize
    xR ./= maximum(abs.(xR))
    yR ./= maximum(abs.(yR))

    return xR, yR
end

function swoo_p(a, R, N)
    th = a * π * exp.(rand(N))  # Generate theta
    r = th  # Since r = th as per the MATLAB code

    xR = r .* sinh.(th)  # Element-wise sinh
    yR = r .* cosh.(th)  # Element-wise cosh

    # Normalize
    xR ./= maximum(abs.(xR))
    yR ./= maximum(abs.(yR))

    return xR, yR
end

function swoop(fobj, popPos, popCost, BestPos, BestCost, nPop, low, high)
    Mean = mean(popPos, dims=1)
    a = 10
    R = 1.5
    s1 = 0

    # Shuffle population
    A = randperm(nPop)
    popPos = popPos[A, :]
    popCost = popCost[A]

    # Assume swoo_p(a, R, nPop) generates x and y
    x, y = swoo_p(a, R, nPop)

    for i in 1:nPop
        # Step = popPos[i, :] .- 2 * Mean
        Step = popPos[i, :] .- 2 * Mean'
        Step1 = popPos[i, :] .- 2 * BestPos
        # newsolPos = rand(1, length(Mean)) .* BestPos .+ x[i] * Step .+ y[i] * Step1
        newsolPos = rand(length(Mean)) .* BestPos .+ x[i] * Step .+ y[i] * Step1
        # println("Size of Random Vector: ", size(rand(length(Mean))))

        # println("Size of First Term (random_vector .* BestPos): ", size(rand(length(Mean)) .* BestPos))

        # println("Size of Second Term (x[i] * Step): ", size(x[i] * Step))
        # println("Size of Second Term (x[i]): ", size(x[i] ))
        # println("Size of Second Term (Step): ", size(Step))
        # println("Size of Third Term (y[i] * Step1): ", size(y[i] * Step1))

        # println("Size of New Solution Position (newsolPos): ", size(rand(length(Mean)) .* BestPos .+ x[i] * Step .+ y[i] * Step1))
        newsolPos = clamp.(newsolPos, low, high)
        # println("___", size(newsolPos))
        newsolPos = vec(newsolPos)
        newsolCost = fobj(newsolPos)

        if newsolCost < popCost[i]
            popPos[i, :] = newsolPos
            popCost[i] = newsolCost
            s1 += 1

            if popCost[i] < BestCost
                BestPos = popPos[i, :]
                BestCost = popCost[i]
            end
        end
    end

    return popPos, popCost, BestPos, BestCost, s1
end

function search_space(fobj, popPos, popCost, BestPos, BestCost, nPop, low, high)

    # println("Hiiiii")
    Mean = mean(popPos, dims=1)
    a = 10
    R = 1.5
    s1 = 0

    # Shuffle population
    A = randperm(nPop)
    popPos = popPos[A, :]
    popCost = popCost[A]

    # Assume polr(a, R, nPop) generates x and y
    x, y = polr(a, R, nPop)

    for i in 1:nPop-1
        Step = popPos[i, :] .- popPos[i+1, :]
        Step1 = popPos[i, :] .- Mean'
        # println("vec(size(Step1)) ", size(Step1))
        newsolPos = popPos[i, :] .+ y[i] * vec(Step) .+ x[i] * Step1
        # println("Search_space: size(newsolPos)", size(newsolPos))
        newsolPos = vec(newsolPos)
        newsolPos = clamp.(newsolPos, low, high)
        newsolCost = fobj(newsolPos)

        if newsolCost < popCost[i]
            popPos[i, :] = newsolPos
            popCost[i] = newsolCost
            s1 += 1

            if popCost[i] < BestCost
                BestPos = popPos[i, :]
                BestCost = popCost[i]
            end
        end
    end

    return popPos, popCost, BestPos, BestCost, s1
end

function select_space(fobj, popPos, popCost, nPop, BestPos, BestCost, low, high, dim)
    Mean = mean(popPos, dims=1)
    lm = 2
    s1 = 0


    for i in 1:nPop
        newsolPos = BestPos .+ lm * rand(dim) .* (vec(Mean) .- popPos[i, :])
        newsolPos = clamp.(newsolPos, low, high)  # Clamp between low and high
        newsolPos = vec(newsolPos)
        newsolCost = fobj(vec(newsolPos))
        
        if newsolCost < popCost[i]
            popPos[i, :] = newsolPos
            popCost[i] = newsolCost
            s1 += 1

            if popCost[i] < BestCost
                BestPos = popPos[i, :]
                BestCost = popCost[i]
            end
        end
    end

    return popPos, popCost, BestPos, BestCost, s1
end

function BES(nPop, MaxIt, low, high, dim, fobj)

    # Initialize Best Solution
    BestCost = Inf
    BestPos = zeros(dim)

    # Population initialization
    popPos = low .+ (high - low) .* rand(nPop, dim)
    popCost = [fobj(popPos[i, :]) for i in 1:nPop]

    for i in 1:nPop
        if popCost[i] < BestCost
            BestPos = popPos[i, :]
            BestCost = popCost[i]
        end
    end
    # println([0, BestCost])

    Convergence_curve = zeros(MaxIt)

    # Main loop
    for t in 1:MaxIt
        # 1- select_space
        (popPos, popCost, BestPos, BestCost, s1) = select_space(fobj, popPos, popCost, nPop, BestPos, BestCost, low, high, dim)
        
        # 2- search in space
        # (fobj, popPos, popCost, BestPos, BestCost, nPop, low, high)
        (popPos, popCost, BestPos, BestCost, s2) = search_space(fobj, popPos, popCost, BestPos, BestCost, nPop, low, high)
        
        # 3- swoop
        (popPos, popCost, BestPos, BestCost, s3) = swoop(fobj, popPos, popCost, BestPos, BestCost, nPop, low, high)

        # Store convergence data
        Convergence_curve[t] = BestCost

        # println([t, BestCost])
    end

    return BestCost, BestPos, Convergence_curve
end
