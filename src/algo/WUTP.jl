"""
Braik, M., & Al-Hiary, H. (2025). 
A novel meta-heuristic optimization algorithm inspired by water uptake and transport in plants. 
Neural Computing and Applications, 1-82.
"""

function WUTP(no_particles::Int, max_iter::Int, lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real}, dim::Int, objfun::Function)
    # Preallocate
    ccurve = zeros(max_iter)
    # Initialize population
    y = initialization(no_particles, dim, ub, lb)
    # Initial velocity
    J = 0.1 .* copy(y)
    # Evaluate fitness

    # cost = [objfun(view(y, i, :)) for i in 1:no_particles]
    cost = zeros(no_particles)
    for i = 1:no_particles
        cost[i] = objfun(y[i, :])
    end
    fitness = copy(cost)
    # Initial best
    fmin0, idx0 = findmin(cost)
    yo = copy(y)
    pbest = copy(y)
    gbest = copy(y[idx0, :])
    # Parameters
    p = 0.5
    rho = 1000.0
    eta = 0.0018
    g = 9.81
    Lp = 1e-9
    D = 1e-9
    K = 1e-9
    a = 1.0

    for t = 1:max_iter
        λ = 1.0 - rand() / 2
        ν = 1e-7 / (1 + exp((t - max_iter / 2) / 100))
        rr = 0.1
        # Water in motion
        for i = 1:no_particles
            k = rand(1:no_particles)
            pb = view(pbest, k, :)
            c1, c2 = rand(), rand()
            J[i, :] .= λ .* J[i, :] .+ c1 * Lp * (y[i, :] .- pb) .+ c2 * rho * g * (y[i, :] .- yo[i, :])
        end
        # Horizontal flow
        for i = 1:no_particles, j = 1:dim
            r3, r4, r5 = rand(), rand(), rand()
            if r3 < p
                if r4 > 0.1
                    y[i, j] -= J[i, j] ./ K
                else
                    y[i, j] -= ν * (lb[j] - ((lb[j] - ub[j]) * rand()))
                end
            else
                if r5 > 0.1
                    y[i, j] -= J[i, j] / (2 * D * pi * a^2)
                else
                    y[i, j] -= ν * (lb[j] - ((lb[j] - ub[j]) * rand()))
                end
            end
        end
        # Vertical flow
        for i = 1:no_particles, j = 1:dim
            if rand() > p
                if rand() > 0.1
                    y[i, j] -= J[i, j] / K + K * rho * g * (y[i, j] - yo[i, j]) * rand()
                else
                    y[i, j] -= ν * (lb[j] - ((lb[j] - ub[j]) * rand()))
                end
            else
                if rand() > 0.1
                    y[i, j] -= J[i, j] / (2 * D * pi * a^2) + D * rho * g * (y[i, j] - yo[i, j]) * rand()
                else
                    y[i, j] -= ν * (lb[j] - ((lb[j] - ub[j]) * rand()))
                end
            end
        end
        # Uptake by roots
        for j = 1:dim, i = 1:no_particles
            k = rand(1:no_particles)
            pb = pbest[k, :]
            if rand() > p
                y[i, j] -= J[i, j] / K + K * rho * g * (y[i, j] - yo[i, j]) * rand()
            else
                y[i, j] -= J[i, j] / (K * pi * a^2) + rand() * (pb[j] - yo[i, j])
            end
        end
        # Root to xylem
        for j = 1:dim, i = 1:no_particles
            k = rand(1:no_particles)
            pb = pbest[k, :]
            if rand() < 0.1
                y[i, j] -= ν * (lb[j] - ((lb[j] - ub[j]) * rand()))
            else
                chi = 0.5
                y[i, j] -= J[i, j] / Lp + chi * (y[i, j] - yo[i, j]) * rand() + (pb[j] - y[i, j]) * rand()
            end
        end
        # Xylem to leaves
        for i = 1:no_particles
            if rand() < rr
                for j = 1:dim
                    y[i, j] -= ν * (lb[j] - ((lb[j] - ub[j]) * rand()))
                end
            else
                y[i, :] .+= J[i, :] .* (8 * eta / (pi * a^4))
            end
        end
        # Bound check
        for i = 1:no_particles, j = 1:dim
            y[i, j] = clamp(y[i, j], lb[j], ub[j])
        end
        # Update pbest & gbest
        for i = 1:no_particles
            cost[i] = objfun(view(y, i, :))
            if all(lb .<= y[i, :]) && all(y[i, :] .<= ub)
                yo[i, :] = y[i, :]
                if cost[i] < fitness[i]
                    pbest[i, :] = y[i, :]
                    fitness[i] = cost[i]
                end
            end
        end
        fmin, idx = findmin(fitness)
        if fmin < fmin0
            fmin0 = fmin
            gbest = pbest[idx, :]
        end
        ccurve[t] = fmin0
    end
    return fmin0, gbest, ccurve
end

# function WUTP(no_particles::Int, max_iter::Int, lb::Union{Int, AbstractVector}, ub::Union{Int, AbstractVector}, dim::Int, objfun)

#     # Ensure bounds are vectors of the correct dim
#     if length(lb) == 1
#         lb = fill(lb[1], dim)
#     end
#     if length(ub) == 1
#         ub = fill(ub[1], dim)
#     end

#     # Convergence curve
#     ccurve = zeros(max_iter)

#     cost = zeros(no_particles)

#     # Initial population and velocity
#     y = initialization(no_particles, dim, ub, lb)
#     J = 0.1 .* y

#     for i in 1:no_particles
#         cost[i] = objfun(y[i, :])
#     end

#     fitness = copy(cost)
#     fmin0, index = findmin(cost)
#     yo = copy(y)
#     pbest = copy(y)
#     gbest = copy(y[index, :])

#     # WUTP parameters
#     p = 0.5
#     rho = 1000.0
#     eta = 0.0018
#     g = 9.81
#     Lp = 1e-9
#     D = 1e-9
#     K = 1e-9
#     a = 1.0

#     for t in 1:max_iter
#         lambda = 1.0 - rand() / 2
#         nu = 1e-7 / (1 + exp((t - max_iter/2) / 100))
#         rr = 0.1
#         sigma = 1.0

#         # Water in motion
#         for i in 1:no_particles
#             rand_index = rand(1:no_particles)
#             pbeta = pbest[rand_index, :]
#             c1 = rand()
#             c2 = rand()
#             J[i, :] = lambda * J[i, :] + c1 * Lp * (y[i, :] - sigma * pbeta) + c2 * rho * g * (y[i, :] - yo[i, :])
#         end

#         # Horizontal flow through soil
#         for i in 1:no_particles
#             for j in 1:dim
#                 c3, c4, c5 = rand(), rand(), rand()
#                 if c3 < p
#                     if c4 > 0.1
#                         y[i, j] -= J[i, j] / K
#                     else
#                         y[i, j] -= nu * (lb[j] - ((lb[j] - ub[j]) * rand()))
#                     end
#                 else
#                     if c5 > 0.1
#                         y[i, j] -= J[i, j] / (2 * D * π * a^2)
#                     else
#                         y[i, j] -= nu * (lb[j] - ((lb[j] - ub[j]) * rand()))
#                     end
#                 end
#             end
#         end

#         # Vertical flow through soil
#         for i in 1:no_particles
#             for j in 1:dim
#                 if rand() > p
#                     if rand() > 0.1
#                         y[i, j] -= J[i, j] / K + K * rho * g * (y[i, j] - yo[i, j]) * rand()
#                     else
#                         y[i, j] -= nu * (lb[j] - ((lb[j] - ub[j]) * rand()))
#                     end
#                 else
#                     if rand() > 0.1
#                         y[i, j] -= J[i, j] / (2 * D * π * a^2) + D * rho * g * (y[i, j] - yo[i, j]) * rand()
#                     else
#                         y[i, j] -= nu * (lb[j] - ((lb[j] - ub[j]) * rand()))
#                     end
#                 end
#             end
#         end

#         # Uptake by roots
#         for j in 1:dim
#             for i in 1:no_particles
#                 rand_index = rand(1:no_particles)
#                 pbeta = pbest[rand_index, :]
#                 if rand() > p
#                     y[i, j] = y[i, j] - J[i, j] / K + K * rho * g * (y[i, j] - yo[i, j]) * rand()
#                 else
#                     y[i, j] = y[i, j] - J[i, j] / (K * π * a^2) + rand() * (pbeta[j] - yo[i, j])
#                 end
#             end
#         end

#         # Movement from root surface to xylem
#         chi = 0.5
#         for j in 1:dim
#             for i in 1:no_particles
#                 rand_index = rand(1:no_particles)
#                 pbeta = pbest[rand_index, :]
#                 if rand() < 0.1
#                     y[i, j] = y[i, j] - nu * (lb[j] - ((lb[j] - ub[j]) * rand()))
#                 else
#                     y[i, j] = y[i, j] - J[i, j] / Lp + chi * (y[i, j] - yo[i, j]) * rand() + (pbeta[j] - y[i, j]) * rand()
#                 end
#             end
#         end

#         # Flow through xylem
#         for i in 1:no_particles
#             if rand() < rr
#                 for j in 1:dim
#                     y[i, j] = y[i, j] - nu * (lb[j] - ((lb[j] - ub[j]) * rand()))
#                 end
#             else
#                 y[i, :] = y[i, :] + J[i, :] * (8 * eta / (π * a^4))
#             end
#         end

#         # Boundary check
#         for i in 1:no_particles
#             for j in 1:dim
#                 y[i, j] = clamp(y[i, j], lb[j], ub[j])
#             end
#         end

#         # Evaluate and update
#         for i in 1:no_particles
#             cost[i] = objfun(y[i, :])
#             if all(lb .<= y[i, :]) && all(y[i, :] .<= ub)
#                 yo[i, :] = y[i, :]
#                 if cost[i] < fitness[i]
#                     pbest[i, :] = y[i, :]
#                     fitness[i] = cost[i]
#                 end
#             end
#         end

#         fmin, index = findmin(fitness)
#         if fmin < fmin0
#             gbest = copy(pbest[index, :])
#             fmin0 = fmin
#         end

#         ccurve[t] = fmin0
#         println("fmin0 $t ", fmin0)
#     end

#     return fmin0, gbest, ccurve
# end
