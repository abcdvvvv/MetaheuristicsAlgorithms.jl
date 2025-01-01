"""
Zheng, Boli, Yi Chen, Chaofan Wang, Ali Asghar Heidari, Lei Liu, and Huiling Chen. 
"The moss growth optimization (MGO): concepts and performance." 
Journal of Computational Design and Engineering 11, no. 5 (2024): 184-221.
"""

function MossGO(SearchAgents_no, MaxIt, lb, ub, dim, fobj)
    FEs = 0
    MaxFEs = SearchAgents_no * MaxIt
    best_cost = Inf   
    best_M = zeros(dim)
    M = initialization(SearchAgents_no, dim, ub, lb) 
    costs = zeros(SearchAgents_no)

    for i in 1:SearchAgents_no
        costs[i] = fobj(M[i, :])
        FEs += 1
        if costs[i] < best_cost
            best_M .= M[i, :]
            best_cost = costs[i]
        end
    end

    Convergence_curve = Float64[]
    it = 1
    rec = 1

    w = 2
    rec_num = 10
    divide_num = Int(floor(dim / 4))
    d1 = 0.2

    newM = zeros(SearchAgents_no, dim)
    newM_cost = zeros(SearchAgents_no)
    rM = zeros(SearchAgents_no, dim, rec_num)  
    rM_cos = zeros(SearchAgents_no, rec_num)   

    while FEs < MaxFEs
        calPositions = copy(M)
        div_num = randperm(dim)

        for j in 1:max(divide_num, 1)
            th = best_M[div_num[j]]
            index = calPositions[:, div_num[j]] .> th
            if sum(index) < size(calPositions, 1) / 2
                index .= .~index
            end
            calPositions = calPositions[index, :]
        end

        D = repeat(best_M', size(calPositions, 1)) .- calPositions

        D_wind = sum(D, dims=1) ./ size(calPositions, 1)  
        D_wind = D_wind'

        

        beta = size(calPositions, 1) / SearchAgents_no
        gama = 1 / sqrt(1 - beta^2)
        step = w * (rand(length(D_wind)) .- 0.5) .* (1 - FEs / MaxFEs)
        step2 = 0.1 * w * (length(size(D_wind)) .- 0.5) .* (1 - FEs / MaxFEs) .* (1 + 1/2 * (1 + tanh(beta / gama)) * (1 - FEs / MaxFEs))
        step3 = 0.1 * (rand() - 0.5) * (1 - FEs / MaxFEs)
        rand_vals = rand(length(D_wind))  
        result = 1 ./ (1 .+ (0.5 .- 10 .* rand_vals))  
        act = actCal(result) 
        

        if rec == 1
            rM[:, :, rec] .= M
            rM_cos[:, rec] .= costs
            rec += 1
        end

        for i in 1:SearchAgents_no
            newM[i, :] .= M[i, :]
            if rand() > d1
                newM[i, :] = newM[i, :] .+ step .* D_wind
            else
                newM[i, :] = newM[i, :] .+ step2 .* D_wind
            end

            if rand() < 0.8
                if rand() > 0.5
                    newM[i, div_num[1]] = best_M[div_num[1]] + step3 * D_wind[div_num[1]]
                else
                    newM[i, :] .= (1 .- act) .* newM[i, :] .+ act .* best_M
                end
            end

            Flag4ub = newM[i, :] .> ub
            Flag4lb = newM[i, :] .< lb
            newM[i, :] .= (newM[i, :] .* .~(Flag4ub .| Flag4lb)) .+ ub .* Flag4ub .+ lb .* Flag4lb

            newM_cost[i] = fobj(newM[i, :])
            FEs += 1

            rM[i, :, rec] .= newM[i, :]
            rM_cos[i, rec] = newM_cost[i]

            if newM_cost[i] < best_cost
                best_M .= newM[i, :]
                best_cost = newM_cost[i]
            end
        end

        rec += 1
        if rec > rec_num || FEs >= MaxFEs
            lcost, Iindex = findmin(rM_cos, dims=2)
            Iindex = map(x -> x[2], Iindex)
            for i in 1:SearchAgents_no
                M[i, :] = rM[i, :, Iindex[i]]
            end
            costs .= lcost
            rec = 1
        end

        push!(Convergence_curve, best_cost)
        it += 1
    end

    return best_cost, best_M, Convergence_curve
end


function actCal(X)
    act = X .>= 0.5  # Apply element-wise comparison, return a boolean array
    return act
end