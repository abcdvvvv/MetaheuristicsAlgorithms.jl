"""
Jiankai Xue & Bo Shen 
A novel swarm intelligence optimization approach: sparrow search algorithm
Systems Science & Control Engineering, 8:1, 22-34, 
DOI: 10.1080/21642583.2019.1708830
(2020) 
"""
function SparrowSA(pop::Int, M::Int, c, d, dim::Int, fobj)
    P_percent = 0.2            
    pNum = round(Int, pop * P_percent)   

    lb = fill(c, dim)
    ub = fill(d, dim)

    x = [lb .+ (ub .- lb) .* rand(dim) for _ in 1:pop]
    fit = [fobj(xi) for xi in x]
    pFit = copy(fit)
    pX = copy(x)

    fMin, bestI = findmin(fit)
    bestX = x[bestI]

    Convergence_curve = zeros(M)

    for t in 1:M
        sorted_idx = sortperm(pFit)
        worst_idx = argmax(pFit)
        worst = x[worst_idx]

        r2 = rand()
        if r2 < 0.8
            for i in 1:pNum
                r1 = rand()
                x[sorted_idx[i]] .= pX[sorted_idx[i]] .* exp(-i / (r1 * M))
                x[sorted_idx[i]] .= clamp.(x[sorted_idx[i]], lb, ub)
                fit[sorted_idx[i]] = fobj(x[sorted_idx[i]])
            end
        else
            for i in 1:pNum
                x[sorted_idx[i]] .= pX[sorted_idx[i]] .+ randn(dim)
                x[sorted_idx[i]] .= clamp.(x[sorted_idx[i]], lb, ub)
                fit[sorted_idx[i]] = fobj(x[sorted_idx[i]])
            end
        end

        fMMin, bestII = findmin(fit)
        bestXX = x[bestII]

        for i in pNum + 1:pop
            A = 2 .* floor.(rand(dim) * 2) .- 1
            if i > div(pop, 2)
                x[sorted_idx[i]] .= randn(dim) .* exp.((worst .- pX[sorted_idx[i]]) / i^2)
            else
                x[sorted_idx[i]] .= bestXX .+ abs.(pX[sorted_idx[i]] .- bestXX) .* A
            end
            x[sorted_idx[i]] .= clamp.(x[sorted_idx[i]], lb, ub)
            fit[sorted_idx[i]] = fobj(x[sorted_idx[i]])
        end

        random_indices = shuffle(sorted_idx[1:20])
        for j in random_indices
            if pFit[j] > fMin
                x[j] .= bestX .+ randn(dim) .* abs.(pX[j] .- bestX)
            else
                x[j] .= pX[j] .+ (2 * rand() - 1) * abs.(pX[j] .- worst) / (pFit[j] - maximum(pFit) + 1e-50)
            end
            x[j] .= clamp.(x[j], lb, ub)
            fit[j] = fobj(x[j])
        end

        for i in 1:pop
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