"""
Özbay, Feyza Altunbey. 
"A modified seahorse optimization algorithm based on chaotic maps for solving global optimization and engineering problems." 
Engineering Science and Technology, an International Journal 41 (2023): 101408.
"""

function SeaHO(npop::Int, max_iter::Int, lb, ub, dim::Int, objfun)
    Sea_horses = initialization(npop, dim, ub, lb)
    Sea_horsesFitness = zeros(npop)
    fitness_history = zeros(npop, max_iter)
    population_history = zeros(npop, dim, max_iter)
    Convergence_curve = zeros(max_iter)
    Trajectories = zeros(npop, max_iter)

    for i = 1:npop
        Sea_horsesFitness[i] = objfun(Sea_horses[i, :])
        fitness_history[i, 1] = Sea_horsesFitness[i]
        population_history[i, :, 1] = Sea_horses[i, :]
        Trajectories[:, 1] .= Sea_horses[:, 1]
    end

    sorted_indexes = sortperm(Sea_horsesFitness)
    TargetPosition = Sea_horses[sorted_indexes[1], :]
    TargetFitness = Sea_horsesFitness[sorted_indexes[1]]
    Convergence_curve[1] = TargetFitness

    t = 1
    u = 0.05
    v = 0.05
    l = 0.05

    while t < max_iter + 1
        beta = randn(npop, dim)
        Elite = repeat(TargetPosition', npop, 1)
        r1 = randn(1, npop)
        Step_length = levy(npop, dim, 1.5)
        Sea_horses_new1 = similar(Sea_horses)
        Sea_horses_new2 = similar(Sea_horses)
        Si = zeros(npop ÷ 2, dim)

        for i = 1:npop
            for j = 1:dim
                if r1[i] > 0
                    r = rand()
                    theta = r * 2 * pi
                    row = u * exp(theta * v)
                    x = row * cos(theta)
                    y = row * sin(theta)
                    z = row * theta
                    Sea_horses_new1[i, j] = Sea_horses[i, j] + Step_length[i, j] * ((Elite[i, j] - Sea_horses[i, j]) * x * y * z + Elite[i, j])
                else
                    Sea_horses_new1[i, j] = Sea_horses[i, j] + rand() * l * beta[i, j] * (Sea_horses[i, j] - beta[i, j] * Elite[i, j])
                end
            end
        end

        Sea_horses_new1 = max.(min.(Sea_horses_new1, ub'), lb')

        r2 = rand(npop)
        for i = 1:npop
            for j = 1:dim
                alpha = (1 - t / max_iter)^(2 * t / max_iter)
                if r2[i] >= 0.1
                    Sea_horses_new2[i, j] = alpha * (Elite[i, j] - rand() * Sea_horses_new1[i, j]) + (1 - alpha) * Elite[i, j]
                else
                    Sea_horses_new2[i, j] = (1 - alpha) * (Sea_horses_new1[i, j] - rand() * Elite[i, j]) + alpha * Sea_horses_new1[i, j]
                end
            end
        end

        Sea_horses_new2 = max.(min.(Sea_horses_new2, ub'), lb')

        Sea_horsesFitness1 = [objfun(Sea_horses_new2[i, :]) for i = 1:npop]
        sorted_indexes = sortperm(Sea_horsesFitness1)

        Sea_horses_father = Sea_horses_new2[sorted_indexes[1:pop÷2], :]
        Sea_horses_mother = Sea_horses_new2[sorted_indexes[pop÷2+1:end], :]
        for k = 1:pop÷2
            r3 = rand()
            Si[k, :] = r3 * Sea_horses_father[k, :] + (1 - r3) * Sea_horses_mother[k, :]
        end

        Sea_horses_offspring = Si
        Sea_horses_offspring = max.(min.(Sea_horses_offspring, ub'), lb')
        Sea_horsesFitness2 = [objfun(Sea_horses_offspring[i, :]) for i = 1:pop÷2]

        Sea_horsesFitness = vcat(Sea_horsesFitness1, Sea_horsesFitness2)
        Sea_horses_new = vcat(Sea_horses_new2, Sea_horses_offspring)

        sorted_indexes = sortperm(Sea_horsesFitness)
        Sea_horses = Sea_horses_new[sorted_indexes[1:npop], :]
        SortfitbestN = Sea_horsesFitness[sorted_indexes[1:npop]]
        fitness_history[:, t] = SortfitbestN
        population_history[:, :, t] = Sea_horses
        Trajectories[:, t] = Sea_horses[:, 1]

        if SortfitbestN[1] < TargetFitness
            TargetPosition = Sea_horses[1, :]
            TargetFitness = SortfitbestN[1]
        end

        Convergence_curve[t] = TargetFitness
        t += 1
    end

    return TargetFitness, TargetPosition, Convergence_curve, Trajectories, fitness_history, population_history
end