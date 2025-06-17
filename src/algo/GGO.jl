"""
El-Kenawy, El-Sayed M., Nima Khodadadi, Seyedali Mirjalili, Abdelaziz A. Abdelhamid, Marwa M. Eid, and Abdelhameed Ibrahim. 
"Greylag goose optimization: nature-inspired optimization algorithm." 
Expert Systems with Applications 238 (2024): 122147.
"""
function GGO(npop::Int, max_iter::Int, lb::Union{Real, AbstractVector}, ub::Union{Real, AbstractVector}, dim::Int, objfun)
    population = lb .+ rand(npop, dim) .* (ub .- lb)
    fitness = zeros(Float64, npop)
    Convergence = zeros(max_iter,1)

    for i in 1:npop
        fitness[i] = objfun(population[i, :])
    end

    for iter in 1:max_iter
        for i in 1:npop
            best_agent_index = argmin(fitness)

            new_solution = population[i, :] .+ rand(dim) .* (population[best_agent_index, :] .- population[i, :])

            new_solution = max.(min.(new_solution, ub), lb)

            new_fitness = objfun(new_solution)

            if new_fitness < fitness[i]
                population[i, :] .= new_solution
                fitness[i] = new_fitness
            end
        end
        Convergence[iter],_ = findmin(fitness) 
        println(Convergence[iter])
    end

    best_fitness, best_index = findmin(fitness)
    best_solution = population[best_index, :]
    
    return best_fitness, best_solution, Convergence
end
