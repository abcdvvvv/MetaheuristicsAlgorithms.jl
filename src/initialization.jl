function initialization(SearchAgents_no::Int, dim::Int, ub::Union{Int, AbstractVector}, lb::Union{Int, AbstractVector})
    Boundary_no = size(ub, 1)  # number of boundaries
    X = zeros(Float64, SearchAgents_no, dim)

    if Boundary_no == 1
        X = rand(SearchAgents_no, dim) .* (ub - lb) .+ lb
    end

    # If each variable has a different lb and ub
    if Boundary_no > 1
        # X = zeros(SearchAgents_no, dim)
        for i in 1:dim
            ub_i = ub[i]
            lb_i = lb[i]
            X[:, i] = rand(SearchAgents_no) .* (ub_i - lb_i) .+ lb_i
        end
    end
    return X
end