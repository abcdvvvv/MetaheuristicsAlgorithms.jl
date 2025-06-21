function getH(g)
    return g <= 0 ? 0.0 : 1.0
end

# F1: Tension/compression spring design
function F1(x)
    cost = (x[3] + 2) * x[2] * x[1]^2
    g = [
        1 - ((x[3] * x[2]^3) / (71785 * x[1]^4)),
        (4*x[2]^2 - x[1]*x[2]) / (12566 * (x[2]*x[1]^3 - x[1]^4)) + 1 / (5108 * x[1]^2) - 1,
        1 - ((140.45 * x[1]) / (x[2]^2 * x[3])),
        (x[1] + x[2]) / 1.5 - 1
    ]
    lam = 1e15
    Z = sum(lam * gk^2 * getH(gk) for gk in g)
    return cost + Z
end

# F2: Pressure vessel design
function F2(x)
    cost = 0.6224*x[1]*x[3]*x[4] + 1.7781*x[2]*x[3]^2 + 3.1661*x[1]^2*x[4] + 19.84*x[1]^2*x[3]
    g = [
        -x[1] + 0.0193*x[3],
        -x[2] + 0.00954*x[3],
        -π*x[3]^2*x[4] - (4/3)*π*x[3]^3 + 1296000,
        x[4] - 240
    ]
    lam = 1e15
    Z = sum(lam * gk^2 * getH(gk) for gk in g)
    return cost + Z
end

# F3: Welded beam design
function F3(x)
    cost = 1.10471 * x[1]^2 * x[2] + 0.04811 * x[3] * x[4] * (14 + x[2])
    Q = 6000 * (14 + x[2]/2)
    D = sqrt(x[2]^2 / 4 + (x[1] + x[3])^2 / 4)
    J = 2 * x[1] * x[2] * sqrt(2) * (x[2]^2 / 12 + (x[1] + x[3])^2 / 4)
    alpha = 6000 / (sqrt(2) * x[1] * x[2])
    beta = Q * D / J
    tau = sqrt(alpha^2 + 2 * alpha * beta * x[2] / (2 * D) + beta^2)
    sigma = 504000 / (x[4] * x[3]^2)
    delta = 65856000 / (30e6 * x[4] * x[3]^3)
    F = 4.013 * 30e6 / 196 * sqrt(x[3]^2 * x[4]^6 / 36) * (1 - x[3] * sqrt(30/48) / 28)
    g = [
        tau - 13600,
        sigma - 30000,
        x[1] - x[4],
        0.10471 * x[1]^2 + 0.04811 * x[3] * x[4] * (14 + x[2]) - 5,
        0.125 - x[1],
        delta - 0.25,
        6000 - F
    ]
    lam = 1e15
    Z = sum(lam * gk^2 * getH(gk) for gk in g)
    return cost + Z
end

# F4: Speed reducer design
function F4(x)
    cost = 0.7854*x[1]*x[2]^2*(3.3333*x[3]^2 + 14.9334*x[3] - 43.0934) -
           1.508*x[1]*(x[6]^2 + x[7]^2) +
           7.4777*(x[6]^3 + x[7]^3) +
           0.7854*(x[4]*x[6]^2 + x[5]*x[7]^2)
    g = [
        27 / (x[1]*x[2]^2*x[3]) - 1,
        397.5 / (x[1]*x[2]^2*x[3]^2) - 1,
        1.93 * x[4]^3 / (x[2]*x[6]^4*x[3]) - 1,
        1.93 * x[5]^3 / (x[2]*x[7]^4*x[3]) - 1,
        sqrt((745*x[4]/(x[2]*x[3]))^2 + 16.9e6) / (110*x[6]^3) - 1,
        sqrt((745*x[5]/(x[2]*x[3]))^2 + 157.5e5) / (85*x[7]^3) - 1,
        x[2]*x[3]/40 - 1,
        5*x[2]/x[1] - 1,
        x[1]/(12*x[2]) - 1,
        (1.5*x[6]+1.9)/x[4] - 1,
        (1.1*x[7]+1.9)/x[5] - 1
    ]
    lam = 1e15
    Z = sum(lam * gk^2 * getH(gk) for gk in g)
    return cost + Z
end

# F5: Gear train design
function F5(x)
    x = round.(x)
    cost = (1/6.931 - (x[3]*x[2]) / (x[1]*x[4]))^2
    return cost
end

# F6: Three-bar truss design
function F6(x)
    cost = (2*sqrt(2)*x[1] + x[2]) * 100
    g = [
        (sqrt(2)*x[1] + x[2]) / (sqrt(2)*x[1]^2 + 2*x[1]*x[2])*2 - 2,
        x[2] / (sqrt(2)*x[1]^2 + 2*x[1]*x[2])*2 - 2,
        1 / (sqrt(2)*x[2] + x[1])*2 - 2
    ]
    lam = 1e15
    Z = sum(lam * gk^2 * getH(gk) for gk in g)
    return cost + Z
end

# F7: Rolling element bearing design
"""
    F3(x::Vector{Float64}) -> Float64

Welded Beam Design Optimization.

Minimizes cost of a welded beam subject to constraints on shear stress, normal stress, deflection, and geometry.

# Problem Source
This problem is a classical constrained engineering design problem used in various metaheuristic algorithm papers.

# Variables
- `x[1]`: Thickness of weld
- `x[2]`: Length of welded joint
- `x[3]`: Height of the beam
- `x[4]`: Width of the beam

# Constraints
Seven nonlinear inequality constraints.

# Returns
- Penalized objective function value (Float64)
"""
function Engineering_F7(x)
    x[3] = round(x[3])
    γ = x[2]/x[1]
    fc = 37.91*((1 + 1.04*((1 - γ)/(1 + γ))^1.72 * ((x[4]*(2x[5] - 1) / (x[5]*(2x[4] - 1)))^0.41))^(10/3))^-0.3 *
         (γ^0.3*(1 - γ)^1.39) / (1 + γ)^(1/3) * (2x[4]/(2x[4] - 1))^0.41
    cost = x[2] <= 25.4 ? -fc*x[3]^(2/3)*x[2]^1.8 : -3.647*fc*x[3]^(2/3)*x[2]^1.4
    D, d, Bw = 160.0, 90.0, 30.0
    T = D - d - 2*x[2]
    phio = 2π - 2*acos((((D - d)/2 - 3*(T/4))^2 + (D/2 - T/4 - x[2])^2 - (d/2 + T/4)^2) /
                      (2*((D - d)/2 - 3*(T/4))*(D/2 - T/4 - x[2])))
    g = [
        -phio / (2*asin(x[2]/x[1])) + x[3] - 1,
        -2*x[2] + x[6]*(D - d),
        -x[7]*(D - d) + 2*x[2],
        (0.5 - x[9])*(D + d) - x[1],
        -(0.5 + x[9])*(D + d) + x[1],
        -x[1] + 0.5*(D + d),
        -0.5*(D - x[1] - x[2]) + x[8]*x[2],
        x[10]*Bw - x[2],
        0.515 - x[4],
        0.515 - x[5]
    ]
    lam = 1e20
    Z = sum(lam * gk^2 * getH(gk) for gk in g)
    return cost + Z
end

# F8: Cantilever beam design
function F8(x)
    cost = 0.0624 * sum(x)
    g = 61/x[1]^3 + 37/x[2]^3 + 19/x[3]^3 + 7/x[4]^3 + 1/x[5]^3 - 1
    lam = 1e15
    Z = lam * g^2 * getH(g)
    return cost + Z
end

# F9: I-beam deflection
function F9(x)
    term1 = x[3] * (x[1] - 2*x[4])^3 / 12
    term2 = x[2] * x[4]^3 / 6
    term3 = 2 * x[2] * x[4] * ((x[1] - x[4])/2)^2
    cost = 5000 / (term1 + term2 + term3)
    g1 = 2*x[2]*x[4] + x[3]*(x[1] - 2*x[4]) - 300
    term1 = x[3]*(x[1] - 2*x[4])^3
    term2 = 2*x[2]*x[4]*(4*x[4]^2 + 3*x[1]*(x[1] - 2*x[4]))
    term3 = (x[1] - 2*x[4])*x[3]^3
    term4 = 2*x[4]*x[2]^3
    g2 = (18*x[1]*1e4)/(term1 + term2) + (15*x[2]*1e3)/(term3 + term4) - 6
    lam = 1e15
    Z = sum(lam * gk^2 * getH(gk) for gk in [g1, g2])
    return cost + Z
end
