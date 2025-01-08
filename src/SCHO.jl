"""
Bai, Jianfu, Yifei Li, Mingpo Zheng, Samir Khatir, Brahim Benaissa, Laith Abualigah, and Magd Abdel Wahab. 
"A sinh cosh optimizer." 
Knowledge-Based Systems 282 (2023): 111081.
"""

function SCHO(N, Max_iteration, lb, ub, dim, fobj)
    Destination_position = zeros(1, dim)
    Destination_fitness = Inf
    Destination_position_second = zeros(1, dim)
    Convergence_curve = zeros(Max_iteration)
    Position_sort = zeros(N, dim)

    u = 0.388
    m = 0.45
    n = 0.5
    p = 10
    q = 9
    Alpha = 4.6
    Beta = 1.55
    BS = floor(Int, Max_iteration / Beta)
    ct = 3.6
    T = floor(Int, Max_iteration / ct)
    BSi = 0
    BSi_temp = 0
    ub_2 = ub
    lb_2 = lb

    X = initialization(N, dim, ub, lb)
    Objective_values = zeros(N)

    for i in 1:N
        Objective_values[i] = fobj(X[i, :])
        if Objective_values[i] < Destination_fitness
            Destination_position = X[i, :]
            Destination_fitness = Objective_values[i]
        end
    end

    Convergence_curve[1] = Destination_fitness
    t = 2

    while t <= Max_iteration
        for i in 1:N
            for j in 1:dim
                cosh2 = (exp(t / Max_iteration) + exp(-t / Max_iteration)) / 2
                sinh2 = (exp(t / Max_iteration) - exp(-t / Max_iteration)) / 2
                r1 = rand()
                A = (p - q * (t / Max_iteration)^(cosh2 / sinh2)) * r1

                if t == BSi
                    ub_2 = Destination_position[j] + (1 - t / Max_iteration) * abs(Destination_position[j] - Destination_position_second[j])
                    lb_2 = Destination_position[j] - (1 - t / Max_iteration) * abs(Destination_position[j] - Destination_position_second[j])
                    ub_2 = min(ub_2, ub)
                    lb_2 = max(lb_2, lb)
                    X = initialization(N, dim, ub_2, lb_2)
                    BSi_temp = BSi
                    BSi = 0
                end

                if t <= T
                    r2, r3, r4, r5 = rand(), rand(), rand(), rand()
                    a1 = 3 * (-1.3 * t / Max_iteration + m)
                    sinh = (exp(r3) - exp(-r3)) / 2
                    cosh = (exp(r3) + exp(-r3)) / 2
                    W1 = r2 * a1 * (cosh + u * sinh - 1)

                    if A > 1
                        X[i, j] = Destination_position[j] + r4 * W1 * X[i, j] * (r5 <= 0.5 ? 1 : -1)
                    else
                        W3 = r2 * a1 * (cosh + u * sinh)
                        X[i, j] = Destination_position[j] + r4 * W3 * X[i, j] * (r5 <= 0.5 ? 1 : -1)
                    end
                else
                    r2, r3, r4, r5 = rand(), rand(), rand(), rand()
                    a2 = 2 * (-t / Max_iteration + n)
                    W2 = r2 * a2
                    sinh = (exp(r3) - exp(-r3)) / 2
                    cosh = (exp(r3) + exp(-r3)) / 2

                    if A < 1
                        X[i, j] += r5 * sinh / cosh * abs(W2 * Destination_position[j] - X[i, j])
                    else
                        X[i, j] += (r4 <= 0.5 ? abs(0.003 * W2 * Destination_position[j] - X[i, j]) : -abs(0.003 * W2 * Destination_position[j] - X[i, j]))
                    end
                end
            end
            BSi = BSi_temp
        end

        for i in 1:N
            X[i, :] = clamp.(X[i, :], lb_2, ub_2)

            Objective_values[i] = fobj(X[i, :])
            if Objective_values[i] < Destination_fitness
                Destination_position = X[i, :]
                Destination_fitness = Objective_values[i]
            end
        end

        if t == BS
            BSi = BS + 1
            BS += floor(Int, (Max_iteration - BS) / Alpha)
            sorted_idx = sortperm(Objective_values)
            Position_sort = X[sorted_idx, :]
            Destination_position_second = Position_sort[2, :]
        end

        Convergence_curve[t] = Destination_fitness
        t += 1
    end

    return Destination_fitness, Destination_position, Convergence_curve
end