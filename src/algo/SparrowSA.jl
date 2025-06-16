"""
Jiankai Xue & Bo Shen 
A novel swarm intelligence optimization approach: sparrow search algorithm
Systems Science & Control Engineering, 8:1, 22-34, 
DOI: 10.1080/21642583.2019.1708830
(2020) 
"""
function SparrowSA(npop::Int, max_iter::Int, c, d, dim::Int, objfun)
    P_percent = 0.2
    pNum = round(Int, npop * P_percent)

    lb = fill(c, dim)
    ub = fill(d, dim)

    x = [lb .+ (ub .- lb) .* rand(dim) for _ = 1:npop]
    fit = [objfun(xi) for xi in x]
    pFit = copy(fit)
    pX = copy(x)

    fMin, bestI = findmin(fit)
    bestX = x[bestI]

    Convergence_curve = zeros(max_iter)

    for t = 1:max_iter
        sorted_idx = sortperm(pFit)
        worst_idx = argmax(pFit)
        worst = x[worst_idx]

        r2 = rand()
        if r2 < 0.8
            for i = 1:pNum
                r1 = rand()
                x[sorted_idx[i]] .= pX[sorted_idx[i]] .* exp(-i / (r1 * max_iter))
                x[sorted_idx[i]] .= clamp.(x[sorted_idx[i]], lb, ub)
                fit[sorted_idx[i]] = objfun(x[sorted_idx[i]])
            end
        else
            for i = 1:pNum
                x[sorted_idx[i]] .= pX[sorted_idx[i]] .+ randn(dim)
                x[sorted_idx[i]] .= clamp.(x[sorted_idx[i]], lb, ub)
                fit[sorted_idx[i]] = objfun(x[sorted_idx[i]])
            end
        end

        fMMin, bestII = findmin(fit)
        bestXX = x[bestII]

        for i = pNum+1:npop
            A = 2 .* floor.(rand(dim) * 2) .- 1
            if i > div(npop, 2)
                x[sorted_idx[i]] .= randn(dim) .* exp.((worst .- pX[sorted_idx[i]]) / i^2)
            else
                x[sorted_idx[i]] .= bestXX .+ abs.(pX[sorted_idx[i]] .- bestXX) .* A
            end
            x[sorted_idx[i]] .= clamp.(x[sorted_idx[i]], lb, ub)
            fit[sorted_idx[i]] = objfun(x[sorted_idx[i]])
        end

        random_indices = shuffle(sorted_idx[1:20])
        for j in random_indices
            if pFit[j] > fMin
                x[j] .= bestX .+ randn(dim) .* abs.(pX[j] .- bestX)
            else
                x[j] .= pX[j] .+ (2 * rand() - 1) * abs.(pX[j] .- worst) / (pFit[j] - maximum(pFit) + 1e-50)
            end
            x[j] .= clamp.(x[j], lb, ub)
            fit[j] = objfun(x[j])
        end

        for i = 1:npop
            if fit[i] < pFit[i]
                pFit[i] = fit[i]
                pX[i] = copy(x[i])
            end
            if pFit[i] < fMin
                fMin = pFit[i]
                bestX = copy(pX[i])
            end
        end

        Convergence_curve[t] = fMin
    end

    return fMin, bestX, Convergence_curve
end