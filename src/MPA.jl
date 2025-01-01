"""
Faramarzi, Afshin, Mohammad Heidarinejad, Seyedali Mirjalili, and Amir H. Gandomi. 
"Marine Predators Algorithm: A nature-inspired metaheuristic." 
Expert systems with applications 152 (2020): 113377.
"""
function MPA(SearchAgents_no::Int, Max_iter::Int, lb::Union{Int, AbstractVector}, ub::Union{Int, AbstractVector}, dim::Int, fobj::Function)

    
    Top_predator_pos = zeros(dim)
    Top_predator_fit = Inf

    Convergence_curve = zeros(Max_iter)
    stepsize = zeros(SearchAgents_no, dim)
    fitness = fill(Inf, SearchAgents_no, 1)
    #
    fit_old = zeros(SearchAgents_no)
    #

    Prey = initialization(SearchAgents_no, dim, ub, lb)
    Prey_old = zeros(size(Prey))

    Xmin = ones(SearchAgents_no, dim) .* lb
    Xmax = ones(SearchAgents_no, dim) .* ub

    Iter = 0
    FADs = 0.2
    P = 0.5

    while Iter < Max_iter
        # Detecting top predator
        for i in 1:SearchAgents_no
            Flag4ub = Prey[i, :] .> ub
            Flag4lb = Prey[i, :] .< lb
            Prey[i, :] .= (Prey[i, :] .* .!(Flag4ub .| Flag4lb)) .+ ub .* Flag4ub .+ lb .* Flag4lb

            fitness[i] = fobj(Prey[i, :])

            if fitness[i] < Top_predator_fit
                Top_predator_fit = fitness[i]
                Top_predator_pos .= Prey[i, :]
            end
        end

        if Iter == 0
            fit_old = copy(fitness)
            Prey_old = copy(Prey)
        end

        Inx = fit_old .< fitness
        Indx = repeat(Inx, 1, dim)
        Prey .= Indx .* Prey_old .+ .~Indx .* Prey
        fitness .= Inx .* fit_old .+ .~Inx .* fitness

        fit_old = copy(fitness)
        Prey_old = copy(Prey)

        Elite = repeat(reshape(Top_predator_pos, 1, dim), SearchAgents_no, 1)
        CF = (1 - Iter / Max_iter)^(2 * Iter / Max_iter)

        RL = 0.05 * levy(SearchAgents_no, dim, 1.5)
        RB = randn(SearchAgents_no, dim)

        for i in 1:SearchAgents_no
            for j in 1:dim
                R = rand()

                if Iter < Max_iter / 3
                    stepsize[i, j] = RB[i, j] * (Elite[i, j] - RB[i, j] * Prey[i, j])
                    Prey[i, j] += P * R * stepsize[i, j]

                elseif Iter > Max_iter / 3 && Iter < 2 * Max_iter / 3
                    if i > SearchAgents_no / 2
                        stepsize[i, j] = RB[i, j] * (RB[i, j] * Elite[i, j] - Prey[i, j])
                        Prey[i, j] = Elite[i, j] + P * CF * stepsize[i, j]
                    else
                        stepsize[i, j] = RL[i, j] * (Elite[i, j] - RL[i, j] * Prey[i, j])
                        Prey[i, j] += P * R * stepsize[i, j]
                    end

                else
                    stepsize[i, j] = RL[i, j] * (RL[i, j] * Elite[i, j] - Prey[i, j])
                    Prey[i, j] = Elite[i, j] + P * CF * stepsize[i, j]
                end
            end
        end

        for i in 1:SearchAgents_no
            Flag4ub = Prey[i, :] .> ub
            Flag4lb = Prey[i, :] .< lb
            Prey[i, :] .= (Prey[i, :] .* .!(Flag4ub .| Flag4lb)) .+ ub .* Flag4ub .+ lb .* Flag4lb

            fitness[i] = fobj(Prey[i, :])

            if fitness[i] < Top_predator_fit
                Top_predator_fit = fitness[i]
                Top_predator_pos .= Prey[i, :]
            end
        end

        if Iter == 0
            fit_old = copy(fitness)
            Prey_old = copy(Prey)
        end

        Inx = fit_old .< fitness
        Indx = repeat(Inx, 1, dim)
        Prey .= Indx .* Prey_old .+ .~Indx .* Prey
        fitness .= Inx .* fit_old .+ .~Inx .* fitness

        fit_old = copy(fitness)
        Prey_old = copy(Prey)

        if rand() < FADs
            U = rand(SearchAgents_no, dim) .< FADs
            Prey += CF * ((Xmin .+ rand(SearchAgents_no, dim) .* (Xmax - Xmin)) .* U)
        else
            r = rand()
            Rs = SearchAgents_no
            stepsize = (FADs * (1 - r) + r) * (Prey[shuffle(1:Rs), :] - Prey[shuffle(1:Rs), :])
            Prey += stepsize
        end

        Iter += 1
        Convergence_curve[Iter] = Top_predator_fit
    end

    return Top_predator_fit, Top_predator_pos, Convergence_curve
end
