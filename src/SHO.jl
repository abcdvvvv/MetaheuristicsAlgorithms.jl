"""
Dhiman, Gaurav, and Vijay Kumar. 
"Spotted hyena optimizer: a novel bio-inspired based metaheuristic technique for engineering applications." 
Advances in Engineering Software 114 (2017): 48-70.
"""
function noh(best_hyena_fitness)
    min_val = 0.5
    max_val = 1.0
    count = 0

    M = (max_val - min_val) * rand() + min_val
    M += best_hyena_fitness[1]

    for i in 2:lastindex(best_hyena_fitness)
        if M >= best_hyena_fitness[i]
            count += 1
        end
    end

    return count
end


function SHO(N, Max_iterations, lowerbound, upperbound, dimension, fitness)
    hyena_pos = initialization(N, dimension, upperbound, lowerbound)
    Convergence_curve = zeros(Max_iterations)
    Iteration = 1

    pre_population = zeros(N, dimension)
    best_hyenas = zeros(N, dimension)
    Best_hyena_pos = zeros(N)
    Best_hyena_score = Inf
    

    while Iteration < Max_iterations
        hyena_fitness = zeros(N)

        pre_fitness = zeros(N)
        best_hyena_fitness = 0

        for i in axes(hyena_pos, 1)
            H_ub = hyena_pos[i, :] .> upperbound
            H_lb = hyena_pos[i, :] .< lowerbound
            hyena_pos[i, :] = (hyena_pos[i, :] .* .~(H_ub .+ H_lb)) .+ upperbound .* H_ub .+ lowerbound .* H_lb
            hyena_fitness[i] = fitness(hyena_pos[i, :])
        end

        if Iteration == 1
            FS = sortperm(hyena_fitness)  
            fitness_sorted = hyena_fitness[FS]  
            sorted_population = hyena_pos[FS, :]
            best_hyenas = sorted_population
            best_hyena_fitness = fitness_sorted
        else
            double_population = vcat(pre_population, best_hyenas)
            double_fitness = vcat(pre_fitness, best_hyena_fitness)
            FS = sortperm(double_fitness)  
            double_fitness_sorted = sort(double_fitness)  
            double_sorted_population = double_population[FS, :]
            fitness_sorted = double_fitness_sorted[1:N]
            sorted_population = double_sorted_population[1:N, :]
            best_hyenas = sorted_population
            best_hyena_fitness = fitness_sorted
        end

        NOH = noh(best_hyena_fitness)

        Best_hyena_score = fitness_sorted[1]
        Best_hyena_pos = sorted_population[1, :]
        pre_population = copy(hyena_pos)
        pre_fitness = copy(hyena_fitness)

        a = 5 - Iteration * (5 / Max_iterations)
        CV = 0

        for i in axes(hyena_pos, 1)
            for i in axes(hyena_pos, 2)
                CV = 0
                for k in 1:NOH
                    r1 = rand()
                    r2 = rand()
                    Var1 = 2 * a * r1 - a
                    Var2 = 2 * r2
                    distance_to_hyena = abs(Var2 * sorted_population[k, j] - hyena_pos[i, j])
                    HYE = sorted_population[k, j] - Var1 * distance_to_hyena
                    CV += HYE
                end
                hyena_pos[i, j] = CV / (NOH + 1)
            end
        end

        Convergence_curve[Iteration] = Best_hyena_score
        Iteration += 1
    end

    return Best_hyena_score, Best_hyena_pos, Convergence_curve
end
