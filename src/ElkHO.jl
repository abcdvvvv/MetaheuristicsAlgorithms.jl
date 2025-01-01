"""
Al-Betar, M.A., Awadallah, M.A., Braik, M.S. et al. 
Elk herd optimizer: a novel nature-inspired metaheuristic algorithm. 
Artif Intell Rev 57, 48 (2024). 
https://doi.org/10.1007/s10462-023-10680-4
"""
function ElkHO(N, Max_iter, lb, ub, dim, fobj)

    if length(ub) == 1
        ub = ones(dim) * ub
        lb = ones(dim) * lb
    end

    MalesRate = 0.2 
    No_of_Males = round(Int, N * MalesRate)

    Convergence_curve = zeros(Max_iter)

    ElkHerd = initialization(N, dim, ub, lb)

    BestBull = zeros(dim)
    BestBullFitness = Inf

    ElkHerdFitness = [fobj(ElkHerd[i, :]) for i in 1:N]

    l = 1
    while l <= Max_iter

        sorted_indexes = sortperm(ElkHerdFitness)
        sorted_ELKS_fitness = ElkHerdFitness[sorted_indexes]

        NewElkHerd = copy(ElkHerd)
        NewElkHerdFitness = copy(ElkHerdFitness)

        BestBull = ElkHerd[sorted_indexes[1], :] 
        BestBullFitness = sorted_ELKS_fitness[1] 

        TransposeFitness = [1 / sorted_ELKS_fitness[i] for i in 1:No_of_Males]

        Familes = zeros(Int, N)
        for i in (No_of_Males + 1):N
            FemaleIndex = sorted_indexes[i] 
            randNumber = rand()
            MaleIndex = 0
            sum_fitness = 0.0

            for j in 1:No_of_Males
                sum_fitness += TransposeFitness[j] / sum(TransposeFitness)
                if sum_fitness > randNumber
                    MaleIndex = j
                    break
                end
            end
            Familes[FemaleIndex] = sorted_indexes[MaleIndex]
        end

        for i in 1:N
            # Male
            if Familes[i] == 0
                h = rand(1:N)
                for j in 1:dim
                    NewElkHerd[i, j] = ElkHerd[i, j] + rand() * (ElkHerd[h, j] - ElkHerd[i, j])
                    NewElkHerd[i, j] = clamp(NewElkHerd[i, j], lb[j], ub[j])
                end
            else
                h = rand(1:N)
                MaleIndex = Familes[i]
                hh = randperm(sum(Familes .== MaleIndex))
                h = 1 + floor(Int, (length(hh) - 1) * rand())
                for j in 1:dim
                    rd = -2 + 4 * rand()
                    NewElkHerd[i, j] = ElkHerd[i, j] + (ElkHerd[Familes[i], j] - ElkHerd[i, j]) + rd * (ElkHerd[h, j] - ElkHerd[i, j])
                end
            end
        end

        for i in 1:N
            NewElkHerdFitness[i] = fobj(NewElkHerd[i, :])
            if NewElkHerdFitness[i] < BestBullFitness
                BestBull = NewElkHerd[i, :]
                BestBullFitness = NewElkHerdFitness[i]
            end
        end

        NewPopulation = vcat(ElkHerd, NewElkHerd)
        NewFitness = [fobj(NewPopulation[i, :]) for i in axes(NewPopulation, 1)]

        sorted_indexes = sortperm(NewFitness)
        sorted_NewFitness = NewFitness[sorted_indexes]

        for i in 1:N
            ElkHerd[i, :] = NewPopulation[sorted_indexes[i], :]
            ElkHerdFitness[i] = sorted_NewFitness[i]
        end

        Convergence_curve[l] = BestBullFitness

        l += 1
    end

    return BestBullFitness, BestBull, Convergence_curve
end