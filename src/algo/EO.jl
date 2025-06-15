"""
Faramarzi, Afshin, Mohammad Heidarinejad, Brent Stephens, and Seyedali Mirjalili. 
"Equilibrium optimizer: A novel optimization algorithm." 
Knowledge-based systems 191 (2020): 105190.
"""
function EO(Particles_no::Int, Max_iter::Int, lb::Union{Int, AbstractVector}, ub::Union{Int, AbstractVector}, dim, fobj)
    Convergence_curve = zeros(Max_iter)

    Ceq1 = zeros(1, dim); Ceq1_fit = Inf
    Ceq2 = zeros(1, dim); Ceq2_fit = Inf
    Ceq3 = zeros(1, dim); Ceq3_fit = Inf
    Ceq4 = zeros(1, dim); Ceq4_fit = Inf

    C = initialization(Particles_no, dim, ub, lb)
    C_old = zeros(size(C))
    Iter = 0; V = 1

    a1 = 2
    a2 = 1
    GP = 0.5

    while Iter < Max_iter
        fitness = zeros(Particles_no)
        fit_old = zeros(Particles_no)

        for i in 1:Particles_no
            Flag4ub = C[i, :] .> ub
            Flag4lb = C[i, :] .< lb
            C[i, :] = (C[i, :] .* .~(Flag4ub .+ Flag4lb)) .+ ub .* Flag4ub .+ lb .* Flag4lb

            fitness[i] = fobj(C[i, :])

            if fitness[i] < Ceq1_fit
                Ceq1_fit = fitness[i]; Ceq1 = C[i, :]
            elseif fitness[i] > Ceq1_fit && fitness[i] < Ceq2_fit
                Ceq2_fit = fitness[i]; Ceq2 = C[i, :]
            elseif fitness[i] > Ceq1_fit && fitness[i] > Ceq2_fit && fitness[i] < Ceq3_fit
                Ceq3_fit = fitness[i]; Ceq3 = C[i, :]
            elseif fitness[i] > Ceq1_fit && fitness[i] > Ceq2_fit && fitness[i] > Ceq3_fit && fitness[i] < Ceq4_fit
                Ceq4_fit = fitness[i]; Ceq4 = C[i, :]
            end
        end

        if Iter == 0
            fit_old = fitness; C_old = C
        end

        for i in 1:Particles_no
            if fit_old[i] < fitness[i]
                fitness[i] = fit_old[i]; C[i, :] = C_old[i, :]
            end
        end

        C_old = C; fit_old = fitness

        Ceq_ave = (Ceq1 + Ceq2 + Ceq3 + Ceq4) / 4
        C_pool = vcat(Ceq1, Ceq2, Ceq3, Ceq4, Ceq_ave)

        t = (1 - Iter / Max_iter) ^ (a2 * Iter / Max_iter)

        for i in 1:Particles_no
            lambda = rand(dim)
            r = rand(dim)

            Ceq = C_pool[rand(1:size(C_pool, 1)), :]
            F = a1 * sign.(r .- 0.5) .* (exp.(-lambda .* t) .- 1)
            r1 = rand(); r2 = rand()
            GCP = 0.5 * r1 * ones(dim) .* (r2 >= GP)
            G0 = GCP .* (Ceq .- lambda .* C[i, :])
            G = G0 .* F
            C[i, :] = Ceq .+ (C[i, :] .- Ceq) .* F .+ (G ./ lambda * V) .* (1 .- F)
        end

        Iter += 1
        Convergence_curve[Iter] = Ceq1_fit
    end
    
    return Ceq1_fit, Ceq1, Convergence_curve
end