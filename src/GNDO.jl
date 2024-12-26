"""
Zhang, Yiying, Zhigang Jin, and Seyedali Mirjalili. 
"Generalized normal distribution optimization and its applications in parameter extraction of photovoltaic models." 
Energy Conversion and Management 224 (2020): 113301.
"""
function GNDO(n, t, lb, ub, d, obj)#(obj, n, d, lb, ub, t)
    x = lb .+ (ub .- lb) .* rand(n, d)
    
    bestFitness = Inf
    bestSol = zeros(d)
    # deta = zeros(d)
    cgcurve = zeros(t)

    fitness = zeros(n)  # Initialize fitness array with zeros
    
    for it in 1:t
        
        for i in 1:n
            fitness[i] = obj(x[i, :])  # Evaluate fitness for each individual
        end
        
        # Update best solution
        for i in 1:n
            if fitness[i] < bestFitness
                bestSol = x[i,:]
                bestFitness = fitness[i]
            end
        end
        cgcurve[it] = bestFitness
        
        mo = mean(x, dims=1)
        # mo = vec(mean(x, dims=1)')
        
        for i in 1:n
            # Select random unique indices a, b, c
            a, b, c = randperm(n)[1:3]
            while a == i || b == i || c == i || a == b || a == c || b == c
                # a, b, c = randperm(n, 3)
                a, b, c = randperm(n)[1:3]
            end
            
            # Eq. 24 and 25
            v1 = (fitness[a] < fitness[i]) ? (x[a] - x[i]) : (x[i] - x[a])
            v2 = (fitness[b] < fitness[c]) ? (x[b] - x[c]) : (x[c] - x[b])
            
            if rand() <= rand()
                # Eq. 19
                # u = (1/3) * (x[i] + bestSol + mo)
                u = (1/3) * (x[i,:] + bestSol + mo')
                # Eq. 20

                deta = sqrt.((1/3) .* ((x[i,:] - u).^2 + (bestSol - u).^2 + vec(mo' - u)).^2)

                
                # Random vectors for Eq. 21
                vc1 = rand(d)
                vc2 = rand(d)
                
                # Eq. 21
                Z1 = sqrt.(-log.(vc2)) .* cos.(2pi .* vc1)
                Z2 = sqrt.(-log.(vc2)) .* cos.(2pi .* vc1 .+ pi)
                
                a_rand = rand()
                b_rand = rand()
                
                if a_rand <= b_rand
                    eta = u + deta .* Z1
                else
                    eta = u + deta .* Z2
                end
                
                newsol = eta
            else
                # Eq. 23
                beta = rand()
                v = x[i] .+ beta .* abs.(randn(d)) .* v1 .+ (1 - beta) .* abs.(randn(d)) .* v2
                newsol = v
            end
            
            # Boundary constraints
            newsol = clamp.(newsol, lb, ub)
            newfitness = obj(vec(newsol))
            
            # Update if new solution is better
            if newfitness < fitness[i]
                x[i,:] = newsol
                fitness[i] = newfitness
                
                if fitness[i] < bestFitness
                    bestSol = x[i,:]
                    bestFitness = fitness[i]
                end
            end
        end
        
        cgcurve[it] = bestFitness
        # println("Iteration $it: Best Cost = $(bestFitness)")
    end
    
    return bestFitness, bestSol, cgcurve
end