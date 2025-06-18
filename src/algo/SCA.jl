"""
# References:
-  Mirjalili, Seyedali. 
"SCA: a sine cosine algorithm for solving optimization problems." 
Knowledge-based systems 96 (2016): 120-133.
"""
function SCA(npop::Int, max_iter::Int, lb::Union{Int,AbstractVector}, ub::Union{Int,AbstractVector}, dim::Int, objfun::Function)
    println("SCA is optimizing your problem")

    X = initialization(npop, dim, ub, lb)

    Destination_position = zeros(dim)
    Destination_fitness = Inf

    Convergence_curve = zeros(max_iter)
    Objective_values = zeros(size(X, 1))

    for i in axes(X, 1)
        Objective_values[i] = objfun(X[i, :])
        if i == 1
            Destination_position .= X[i, :]
            Destination_fitness = Objective_values[i]
        elseif Objective_values[i] < Destination_fitness
            Destination_position .= X[i, :]
            Destination_fitness = Objective_values[i]
        end
    end

    t = 2
    while t <= max_iter
        a = 2
        r1 = a - t * (a / max_iter)

        for i in axes(X, 1)
            for j in axes(X, 2)
                r2 = 2 * pi * rand()
                r3 = 2 * rand()
                r4 = rand()

                if r4 < 0.5
                    X[i, j] += r1 * sin(r2) * abs(r3 * Destination_position[j] - X[i, j])
                else
                    X[i, j] += r1 * cos(r2) * abs(r3 * Destination_position[j] - X[i, j])
                end
            end
        end

        for i in axes(X, 1)
            X[i, :] = clamp.(X[i, :], lb, ub)

            Objective_values[i] = objfun(X[i, :])

            if Objective_values[i] < Destination_fitness
                Destination_position .= X[i, :]
                Destination_fitness = Objective_values[i]
            end
        end

        Convergence_curve[t] = Destination_fitness
        println("$t --> ", Convergence_curve[t])

        t += 1
    end

    return Destination_fitness, Destination_position, Convergence_curve
end