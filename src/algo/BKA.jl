"""
Wang, Jun, Wen-chuan Wang, Xiao-xue Hu, Lin Qiu, and Hong-fei Zang. 
"Black-winged kite algorithm: a nature-inspired meta-heuristic for solving benchmark functions and engineering problems." 
Artificial Intelligence Review 57, no. 4 (2024): 98.
"""

function BKA(pop, T, lb, ub, dim, fobj)
    # Initialize the locations of Blue Sheep
    p = 0.9
    r = rand()
    XPos = initialization(pop, dim, ub, lb) # Initial population
    XFit = [fobj(XPos[i, :]) for i in 1:pop]
    Convergence_curve = zeros(T)

    Best_Fitness_BKA = Inf
    Best_Pos_BKA = zeros(dim)

    for t in 1:T
        sorted_indexes = sortperm(XFit)
        XLeader_Pos = XPos[sorted_indexes[1], :]
        XLeader_Fit = XFit[sorted_indexes[1]]

        for i in 1:pop
            n = 0.05 * exp(-2 * (t / T)^2)
            if p < r
                XPosNew = XPos[i, :] .+ n .* (1 + sin(r)) .* XPos[i, :]
            else
                XPosNew = XPos[i, :] .* (n * (2 * rand(dim) .- 1) .+ 1)
            end
            XPosNew = clamp.(XPosNew, lb, ub) # Boundary checking

            XFit_New = fobj(XPosNew)
            if XFit_New < XFit[i]
                XPos[i, :] = XPosNew
                XFit[i] = XFit_New
            end

            m = 2 * sin(r + π / 2)
            s = rand(1:30)
            r_XFitness = XFit[s]
            ori_value = rand(dim)
            cauchy_value = tan.((ori_value .- 0.5) * π)

            if XFit[i] < r_XFitness
                XPosNew = XPos[i, :] .+ cauchy_value .* (XPos[i, :] .- XLeader_Pos)
            else
                XPosNew = XPos[i, :] .+ cauchy_value .* (XLeader_Pos .- m .* XPos[i, :])
            end
            XPosNew = clamp.(XPosNew, lb, ub) 

            XFit_New = fobj(XPosNew)
            if XFit_New < XFit[i]
                XPos[i, :] = XPosNew
                XFit[i] = XFit_New
            end
        end

        if minimum(XFit) < XLeader_Fit
            Best_Fitness_BKA = minimum(XFit)
            Best_Pos_BKA = XPos[argmin(XFit), :]
        else
            Best_Fitness_BKA = XLeader_Fit
            Best_Pos_BKA = XLeader_Pos
        end
        Convergence_curve[t] = Best_Fitness_BKA
    end

    return Best_Fitness_BKA, Best_Pos_BKA, Convergence_curve
end