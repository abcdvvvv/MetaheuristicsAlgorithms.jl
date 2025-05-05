"""
Houssein, Essam H., Diego Oliva, Nagwan Abdel Samee, Noha F. Mahmoud, and Marwa M. Emam. 
"Liver Cancer Algorithm: A novel bio-inspired optimizer." 
Computers in Biology and Medicine 165 (2023): 107389.
"""

function LCA(N, T, lb, ub, dim, fobj)
    println("LCA is now tackling your problem")
    
    Tumor_Location = zeros(Float64, dim)
    Tumor_Energy = Inf

    X = initialization(N, dim, ub, lb)

    CNVG = zeros(Float64, T)

    t = 0  

    q = 0

    while t < T

        for i in axes(X, 1)
            FU = X[i, :] .> ub
            FL = X[i, :] .< lb
            X[i, :] .= (X[i, :] .* .!(FU .| FL)) .+ ub .* FU .+ lb .* FL

            fitness = fobj(X[i, :])

            if fitness < Tumor_Energy
                Tumor_Energy = fitness
                Tumor_Location .= X[i, :]
            end
        end

        r = rand()
        v = r * t

        for i in axes(X, 1)
            f = 1
            l = rand()
            w = rand()

            if abs(v) <= 5
                q = rand()
                rand_Hawk_index = rand(1:N)
                X_rand = X[rand_Hawk_index, :]

                q = π / 6 * (l * w)^1.5

                if q < 2
                    X[i, :] .= (Tumor_Location .- mean(X, dims=1)') .- rand() * ((ub .- lb) .* rand() .+ lb)
                elseif q <= 2
                    X[i, :] .= (Tumor_Location .- mean(X, dims=1)) .- rand() * ((ub - lb) * rand() .+ lb)
                end

            elseif q >= 2 && q < 5
                p = 2 / 3  
                Jump_strength = v^p  
                X[i, :] .= (Tumor_Location .- X[i, :]) .- v * abs(Jump_strength * Tumor_Location .- X[i, :])

                X1 = Tumor_Location .- v * abs(Jump_strength * Tumor_Location .- X[i, :])
                X1 = MutationU(dim, T, Tumor_Location, t)

                X1 .= Tumor_Location .- v * abs(Jump_strength * Tumor_Location .- X[i, :])
                X1 = MutationU(dim, T, Tumor_Location, t)

                if fobj(X1') < fobj(X[i, :])  
                    X[i, :] .= X1
                else
                    X2 = Tumor_Location .- v * abs(Jump_strength * Tumor_Location .- X[i, :]) .+ rand(dim) .* Levy(dim)
                    X2 = MutationU(dim, T, Tumor_Location, t)

                    if fobj(X2') < fobj(X[i, :])  
                        X[i, :] .= X2
                    end

                    X3 = rand() * X2 .+ (1 - rand()) * X1
                    if fobj(X3') < fobj(X[i, :])  
                        X[i, :] .= X3
                    end
                end
            end
        end

        t += 1
        CNVG[t] = Tumor_Energy
    end

    return Tumor_Energy,Tumor_Location,CNVG
end