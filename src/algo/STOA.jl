"""
# References:
-  Dhiman, Gaurav, and Amandeep Kaur. 
"STOA: a bio-inspired based optimization algorithm for industrial engineering problems." 
Engineering Applications of Artificial Intelligence 82 (2019): 148-174.
"""
function STOA(npop::Int, max_iter::Int, lb::Union{Real, AbstractVector}, ub::Union{Real, AbstractVector}, dim::Int, objfun)
    position = zeros(Float64, dim)
    score = Inf

    positions = initialization(npop, dim, ub, lb)
    convergence = zeros(Float64, max_iter)

    l = 0

    while l < max_iter
        for i in axes(positions, 1)
            positions .= max.(lb, min.(positions, ub))

            fitness = objfun(positions[i, :])

            if fitness < score
                score = fitness
                position .= positions[i, :]
            end
        end

        Sa = 2 - l * (2 / max_iter)

        for i in axes(positions, 1)
            for j in axes(positions, 2)
                r1 = 0.5 * rand()
                r2 = 0.5 * rand()
                X1 = 2 * Sa * r1 - Sa
                b = 1
                ll = (Sa - 1) * rand() + 1
                D_alphs = Sa * positions[i, j] + X1 * (position[j] - positions[i, j])
                res = D_alphs * exp(b * ll) * cos(ll * 2 * π) + position[j]
                positions[i, j] = res
            end
        end
        l += 1
        convergence[l] = score
    end

    # return score, position, convergence
    return OptimizationResult(
        position,
        score,
        convergence)
end
