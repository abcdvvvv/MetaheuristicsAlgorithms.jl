"""
Askari, Qamar, Mehreen Saeed, and Irfan Younas. 
"Heap-based optimizer inspired by corporate rank hierarchy for global optimization." 
Expert Systems with Applications 161 (2020): 113702.
"""
function colleaguesLimitsGenerator(degree::Int, searchAgents::Int)
    colleaguesLimits = zeros(Float64, searchAgents, 2)  
    for c in searchAgents:-1:1
        hi = ceil(log10(c * degree - c + 1) / log10(degree)) - 1
        lowerLim = ((degree * degree^(hi - 1) - 1) / (degree - 1) + 1)
        upperLim = (degree * degree^hi - 1) / (degree - 1)
        colleaguesLimits[c, 1] = lowerLim
        colleaguesLimits[c, 2] = upperLim
    end

    return colleaguesLimits  
end


function HBO(searchAgents, Max_iter, lb, ub, dim, fobj)
    cycles = floor(Int, Max_iter / 25)  
    degree = 3  

    treeHeight = ceil(log10(searchAgents * degree - searchAgents + 1) / log10(degree))
    fevals = 0

    Leader_pos = zeros(dim)
    Leader_score = Inf  
    

    Solutions = initialization(searchAgents, dim, ub, lb)

    
    fitnessHeap = fill(Inf, searchAgents, 2)  


    for c in 1:searchAgents
        fitness = fobj(Solutions[c, :])
        fevals += 1
        fitnessHeap[c, 1] = fitness
        fitnessHeap[c, 2] = c

        t = c
        while t > 1
            parentInd = Int(floor((t + 1) / degree))
            if fitnessHeap[t, 1] >= fitnessHeap[parentInd, 1]
                break
            else
                fitnessHeap[t, :], fitnessHeap[parentInd, :] = fitnessHeap[parentInd, :], fitnessHeap[t, :]
            end
            t = parentInd
        end

        if fitness <= Leader_score
            Leader_score = fitness
            Leader_pos .= Solutions[c, :]
        end
    end

    colleaguesLimits = colleaguesLimitsGenerator(degree, searchAgents)
    Convergence_curve = zeros(Max_iter)
    itPerCycle = Max_iter / cycles
    qtrCycle = itPerCycle / 4

    for it in 1:Max_iter
        gamma = (mod(it, itPerCycle)) / qtrCycle
        gamma = abs(2 - gamma)

        for c in searchAgents:-1:2
            if c == 1  
                continue
            else
                parentInd = div(c + 1, degree)
                curSol = Solutions[Int(fitnessHeap[c, 2]), :]
                parentSol = Solutions[Int(fitnessHeap[parentInd, 2]), :]  # Sol to be updated with reference to

                if colleaguesLimits[c, 2] > searchAgents
                    colleaguesLimits[c, 2] = searchAgents
                end

                colleagueInd = c
                while colleagueInd == c
                    colleagueInd = Int(rand(1:colleaguesLimits[c, 1]:colleaguesLimits[c, 2]))
                end
                colleagueSol = Solutions[Int(fitnessHeap[colleagueInd, 2]), :]

                for j in 1:dim
                    p1 = (1 - it / Max_iter)
                    p2 = p1 + (1 - p1) / 2
                    r = rand()
                    rn = (2 * rand() - 1)

                    if r < p1  
                        continue
                    elseif r < p2
                        D = abs(parentSol[j] - curSol[j])
                        curSol[j] = parentSol[j] + rn * gamma * D
                    else
                        if fitnessHeap[colleagueInd, 1] < fitnessHeap[c, 1]
                            D = abs(colleagueSol[j] - curSol[j])
                            curSol[j] = colleagueSol[j] + rn * gamma * D
                        else
                            D = abs(colleagueSol[j] - curSol[j])
                            curSol[j] = curSol[j] + rn * gamma * D
                        end
                    end
                end
            end

            Flag4ub = curSol .> ub
            Flag4lb = curSol .< lb
            curSol .= max.(lb, min.(curSol, ub))

            newFitness = fobj(curSol)
            fevals += 1
            if newFitness < fitnessHeap[c, 1]
                fitnessHeap[c, 1] = newFitness
                Solutions[Int(fitnessHeap[c, 2]), :] .= curSol
            end
            if newFitness < Leader_score
                Leader_score = newFitness
                Leader_pos .= curSol
            end

            t = c
            while t > 1
                parentInd = Int(floor((t + 1) / degree))
                if fitnessHeap[t, 1] >= fitnessHeap[parentInd, 1]
                    break
                else
                    fitnessHeap[t, :], fitnessHeap[parentInd, :] = fitnessHeap[parentInd, :], fitnessHeap[t, :]
                end
                t = parentInd
            end
        end
        Convergence_curve[it] = Leader_score
    end

    return Leader_score, Leader_pos, Convergence_curve
end