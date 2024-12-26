"""
Heidari, Ali Asghar, Seyedali Mirjalili, Hossam Faris, Ibrahim Aljarah, Majdi Mafarja, and Huiling Chen. 
"Harris hawks optimization: Algorithm and applications." 
Future generation computer systems 97 (2019): 849-872.
"""

using Random
using SpecialFunctions  # For gamma function
using Statistics

function Levy(d::Int)
    beta = 1.5
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta)
    
    u = randn(d) * sigma
    v = randn(d)
    
    step = u ./ abs.(v).^(1 / beta)
    return step
end

function HHO(N::Int, T::Int, lb::Union{Int, AbstractVector}, ub::Union{Int, AbstractVector}, dim::Int, fobj::Function)
    println("HHO is now tackling your problem")

    # Initialize the location and Energy of the rabbit
    Rabbit_Location = zeros(dim)
    Rabbit_Energy = Inf

    # Initialize the locations of Harris' hawks
    X = initialization(N, dim, ub, lb)

    CNVG = zeros(T)

    t = 0  # Loop counter

    while t < T
        for i in 1:N
            # Check boundaries
            FU = X[i, :] .> ub
            FL = X[i, :] .< lb
            X[i, :] = (X[i, :] .* .~(FU .+ FL)) .+ ub .* FU .+ lb .* FL

            # Fitness of locations
            fitness = fobj(X[i, :])
            
            # Update the location of Rabbit
            if fitness < Rabbit_Energy
                Rabbit_Energy = fitness
                Rabbit_Location = copy(X[i, :])
            end
        end

        E1 = 2 * (1 - (t / T))  # Decreasing energy of rabbit

        # Update the location of Harris' hawks
        for i in 1:N
            E0 = 2 * rand() - 1  # -1 < E0 < 1
            Escaping_Energy = E1 * E0  # Escaping energy of rabbit

            if abs(Escaping_Energy) >= 1
                # Exploration phase
                q = rand()
                rand_Hawk_index = rand(1:N)
                X_rand = X[rand_Hawk_index, :]
                if q < 0.5
                    # Perch based on other family members
                    X[i, :] = X_rand - rand() * abs.(X_rand - 2 * rand() * X[i, :])
                else
                    # Perch on a random tall tree (random site inside group's home range)
                    X[i, :] = (Rabbit_Location .- vec(mean(X, dims=1))) .- rand() * ((ub - lb) * rand() + lb)
                end

            elseif abs(Escaping_Energy) < 1
                # Exploitation phase
                r = rand()  # Probability of each event

                if r >= 0.5 && abs(Escaping_Energy) < 0.5  # Hard besiege
                    X[i, :] = Rabbit_Location - Escaping_Energy * abs.(Rabbit_Location - X[i, :])
                elseif r >= 0.5 && abs(Escaping_Energy) >= 0.5  # Soft besiege
                    Jump_strength = 2 * (1 - rand())
                    X[i, :] = (Rabbit_Location - X[i, :]) - Escaping_Energy * abs.(Jump_strength * Rabbit_Location - X[i, :])
                elseif r < 0.5 && abs(Escaping_Energy) >= 0.5  # Soft besiege (rabbit escapes with zigzag motion)
                    Jump_strength = 2 * (1 - rand())
                    X1 = Rabbit_Location - Escaping_Energy * abs.(Jump_strength * Rabbit_Location - X[i, :])

                    if fobj(X1) < fobj(X[i, :])
                        X[i, :] = X1
                    else
                        X2 = Rabbit_Location - Escaping_Energy * abs.(Jump_strength * Rabbit_Location - X[i, :]) .+ rand(dim) .* Levy(dim)
                        if fobj(X2) < fobj(X[i, :])
                            X[i, :] = X2
                        end
                    end
                elseif r < 0.5 && abs(Escaping_Energy) < 0.5  # Hard besiege (zigzag escape)
                    Jump_strength = 2 * (1 - rand())
                    # X1 = Rabbit_Location - Escaping_Energy * abs.(Jump_strength * Rabbit_Location - mean(X, dims=1))
                    X1 = Rabbit_Location .- Escaping_Energy * abs.(Jump_strength * Rabbit_Location .- vec(mean(X, dims=1)))


                    if fobj(X1) < fobj(X[i, :])
                        X[i, :] = X1
                    else
                        X2 = Rabbit_Location - Escaping_Energy * abs.(Jump_strength * Rabbit_Location - vec(mean(X, dims=1))) .+ rand(dim) .* Levy(dim)
                        if fobj(X2) < fobj(X[i, :])
                            X[i, :] = X2
                        end
                    end
                end
            end
        end

        t += 1
        CNVG[t] = Rabbit_Energy
    end
    return Rabbit_Energy, Rabbit_Location, CNVG
end