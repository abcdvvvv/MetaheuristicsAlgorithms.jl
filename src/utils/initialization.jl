function initialization(npop::Int, dim::Int, ub::Union{Int,AbstractVector}, lb::Union{Int,AbstractVector})
    Boundary_no = size(ub, 1)  # number of boundaries
    X = zeros(Float64, npop, dim)

    if Boundary_no == 1
        X = rand(npop, dim) .* (ub - lb) .+ lb
    end

    # If each variable has a different lb and ub
    if Boundary_no > 1
        # X = zeros(npop, dim)
        for i = 1:dim
            ub_i = ub[i]
            lb_i = lb[i]
            X[:, i] = rand(npop) .* (ub_i - lb_i) .+ lb_i
        end
    end
    return X
end