"""
El-Kenawy, El-Sayed M., Nima Khodadadi, Seyedali Mirjalili, Abdelaziz A. Abdelhamid, Marwa M. Eid, and Abdelhameed Ibrahim. 
"Greylag goose optimization: nature-inspired optimization algorithm." 
Expert Systems with Applications 238 (2024): 122147.
"""
function GGO(num_agents::Int, max_iter::Int, lb::Union{Int, AbstractVector}, ub::Union{Int, AbstractVector}, num_variables::Int, obj_func)
    # Initialize population
    population = lb .+ rand(num_agents, num_variables) .* (ub .- lb)
    fitness = zeros(Float64, num_agents)
    Convergence = zeros(max_iter,1)

    # Evaluate fitness of each agent
    for i in 1:num_agents
        fitness[i] = obj_func(population[i, :])
    end

    # Main loop
    for iter in 1:max_iter
        # Update position and fitness of each agent
        for i in 1:num_agents
            # Determine the best agent in the flock
            best_agent_index = argmin(fitness)

            # Generate a new solution by combining exploration and exploitation
            new_solution = population[i, :] .+ rand(num_variables) .* (population[best_agent_index, :] .- population[i, :])

            # Clip new solution to ensure it stays within bounds
            new_solution = max.(min.(new_solution, ub), lb)

            # Evaluate fitness of the new solution
            new_fitness = obj_func(new_solution)

            # Update if the new solution is better
            if new_fitness < fitness[i]
                population[i, :] .= new_solution
                fitness[i] = new_fitness
            end
        end
        Convergence[iter],_ = findmin(fitness) # minimum(fitness)
        println(Convergence[iter])
    end

    # Find the best solution in the final population
    best_fitness, best_index = findmin(fitness)
    best_solution = population[best_index, :]
    
    return best_fitness, best_solution, Convergence
end
