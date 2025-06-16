"""
El-Kenawy, El-Sayed M., Nima Khodadadi, Seyedali Mirjalili, Abdelaziz A. Abdelhamid, Marwa M. Eid, and Abdelhameed Ibrahim. 
"Greylag goose optimization: nature-inspired optimization algorithm." 
Expert Systems with Applications 238 (2024): 122147.
"""
function GGO(num_agents::Int, max_iter::Int, lb::Union{Int, AbstractVector}, ub::Union{Int, AbstractVector}, num_variables::Int, objfun)
    population = lb .+ rand(num_agents, num_variables) .* (ub .- lb)
    fitness = zeros(Float64, num_agents)
    Convergence = zeros(max_iter,1)

    for i in 1:num_agents
        fitness[i] = objfun(population[i, :])
    end

    for iter in 1:max_iter
        for i in 1:num_agents
            best_agent_index = argmin(fitness)

            new_solution = population[i, :] .+ rand(num_variables) .* (population[best_agent_index, :] .- population[i, :])

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
