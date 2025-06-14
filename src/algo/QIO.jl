"""
Zhao, Weiguo, Liying Wang, Zhenxing Zhang, Seyedali Mirjalili, Nima Khodadadi, and Qiang Ge. 
"Quadratic Interpolation Optimization (QIO): A new optimization algorithm based on generalized quadratic interpolation and its applications to real-world engineering problems." 
Computer Methods in Applied Mechanics and Engineering 417 (2023): 116446.
"""
function Interpolation(xi, xj, xk, fi, fj, fk, l, u)
    a = (xj^2 - xk^2) * fi + (xk^2 - xi^2) * fj + (xi^2 - xj^2) * fk
    b = 2 * ((xj - xk) * fi + (xk - xi) * fj + (xi - xj) * fk)

    L_xmin = a / (b + eps())  
    if isnan(L_xmin) || isinf(L_xmin) || L_xmin > u || L_xmin < l
        L_xmin = rand() * (u - l) + l  
    end
    return L_xmin
end


function GQI(a, b, c, fa, fb, fc, low, up)
    
    fabc = [fa, fb, fc]
    ind = sortperm(fabc)
    fijk = fabc[ind]

    fi, fj, fk = fijk
    dim = length(a)
    ai, bi, ci = ind


    L = zeros(dim)  

    for i in 1:dim
        x = [a[i], b[i], c[i]]
        xi, xj, xk = x[ai], x[bi], x[ci]

        if (xk >= xi && xi >= xj) || (xj >= xi && xi >= xk)
            L[i] = Interpolation(xi, xj, xk, fi, fj, fk, low[i], up[i])
        elseif (xk >= xj && xj >= xi)        
            I = Interpolation(xi, xj, xk, fi, fj, fk, low[i], up[i])
            if I < xj
                L[i] = I
            else
                L[i] = Interpolation(xi, xj, 3*xi - 2*xj, fi, fj, fk, low[i], up[i])
            end
        elseif (xi >= xj && xj >= xk)
            I = Interpolation(xi, xj, xk, fi, fj, fk, low[i], up[i])
            if I > xj
                L[i] = I
            else
                L[i] = Interpolation(xi, xj, 3*xi - 2*xj, fi, fj, fk, low[i], up[i])
            end
        elseif (xj >= xk && xk >= xi)
            L[i] = Interpolation(xi, 2*xi - xk, xk, fi, fj, fk, low[i], up[i])
        elseif (xi >= xk && xk >= xj)
            L[i] = Interpolation(xi, 2*xi - xk, xk, fi, fj, fk, low[i], up[i])
        end
    end
    return (dim == 1) ? L[1] : L
end

function QIO(nPop, MaxIt, Low, Up, Dim, FunIndex)

    IndividualsPos = zeros(nPop, Dim)
    IndividualsFit = zeros(nPop)

    if length(Low) == 1
        Low = ones(Dim) .* Low
        Up = ones(Dim) .* Up
    end

    for i in 1:nPop
        IndividualsPos[i, :] .= rand(Dim) .* (Up .- Low) .+ Low
        IndividualsFit[i] = FunIndex(IndividualsPos[i, :])
    end

    BestF = Inf
    BestX = []

    for i in 1:nPop
        if IndividualsFit[i] <= BestF
            BestF = IndividualsFit[i]
            BestX = IndividualsPos[i, :]
        end
    end

    HisBestF = zeros(MaxIt)

    for It in 1:MaxIt
        newIndividualPos = zeros(Dim)

        for i in 1:nPop
            if rand() > 0.5
                K = vcat(1:i-1, i+1:nPop)
                RandInd = randperm(nPop - 1)[1:3]
                K1 = K[RandInd[1]]
                K2 = K[RandInd[2]]
                K3 = K[RandInd[3]]
                f1 = IndividualsFit[i]
                f2 = IndividualsFit[K1]
                f3 = IndividualsFit[K2]

                for j in 1:Dim
                    x1 = IndividualsPos[i, j]
                    x2 = IndividualsPos[K1, j]
                    x3 = IndividualsPos[K2, j]
                    newIndividualPos[j] = GQI(x1, x2, x3, f1, f2, f3, Low[j], Up[j])
                end

                a = cos(pi/2 * It / MaxIt)
                b = 0.7 * a + 0.15 * a * (cos(5 * pi * It / MaxIt) + 1)
                w1 = 3 * b * randn()
                newIndividualPos .+= w1 .* (IndividualsPos[K3, :] .- newIndividualPos) .+ round(0.5 * (0.05 + rand())) * (log(rand() / rand()))

            else
                K = vcat(1:i-1, i+1:nPop)
                RandInd = randperm(nPop - 1)[1:2]
                K1 = K[RandInd[1]]
                K2 = K[RandInd[2]]
                f1 = IndividualsFit[K1]
                f2 = IndividualsFit[K2]
                f3 = BestF

                for j in 1:Dim
                    x1 = IndividualsPos[K[RandInd[1]], j]
                    x2 = IndividualsPos[K[RandInd[2]], j]
                    x3 = BestX[j]
                    newIndividualPos[j] = GQI(x1, x2, x3, f1, f2, f3, Low[j], Up[j])
                end

                w2 = 3 * (1 - (It - 1) / MaxIt) * randn()
                rD = rand(1:Dim)
                newIndividualPos .+= w2 * (BestX .- round(1 + rand()) * (Up .- Low) ./ (Up[rD] - Low[rD]) .* IndividualsPos[i, rD])
            end

            newIndividualPos = clamp.(newIndividualPos, Low, Up)
            newIndividualFit = FunIndex(newIndividualPos)
            if newIndividualFit < IndividualsFit[i]
                IndividualsFit[i] = newIndividualFit
                IndividualsPos[i, :] .= newIndividualPos
            end
        end

        for i in 1:nPop
            if IndividualsFit[i] < BestF
                BestF = IndividualsFit[i]
                BestX .= IndividualsPos[i, :]
            end
        end

        HisBestF[It] = BestF
    end

    return BestF, BestX, HisBestF
end
