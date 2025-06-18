
addpath(directory) =
    map(file -> endswith(file, ".jl") && include(joinpath(directory, file)), readdir(directory))

function levy(d::Integer)
    beta = 1.5
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) /
             (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta)
    u = randn(d) * sigma
    v = randn(d)
    step = @. u / abs(v)^(1 / beta)
    return step
end

function levy(n::Integer, m::Integer, beta::AbstractFloat)
    num = gamma(1 + beta) * sin(pi * beta / 2)

    den = gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)

    sigma_u = (num / den)^(1 / beta)

    u = rand(Normal(0, sigma_u), n, m)

    v = rand(Normal(0, 1), n, m)

    z = @. u / abs(v)^(1 / beta)

    return z
end