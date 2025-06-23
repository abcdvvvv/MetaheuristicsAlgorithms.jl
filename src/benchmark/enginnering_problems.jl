function getH(g)
    return g <= 0 ? 0.0 : 1.0
end

# F1: Tension/compression spring design
"""
    F1(x::Vector{Float64}) -> Float64

Tension/Compression Spring Design Optimization.

Minimizes the weight of a tension/compression spring subject to constraints on shear stress, surge frequency, minimum deflection, and geometric limits.

# Problem Source
A well-known benchmark in constrained engineering design, commonly used in metaheuristic optimization literature.

# Variables
- `x[1]`: Wire diameter (d)
- `x[2]`: Mean coil diameter (D)
- `x[3]`: Number of active coils (N)

# Constraints
Four nonlinear inequality constraints:
- Shear stress constraint
- Surge frequency constraint
- Minimum deflection constraint
- Geometry-related limits

# Returns
- Penalized objective function value (Float64)
"""
function Engineering_F1(x)
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
"""
    F2(x::Vector{Float64}) -> Float64

Pressure Vessel Design Optimization.

Minimizes the total cost of a cylindrical pressure vessel, which includes material, forming, and welding costs, subject to constraints on thickness, volume, and stress.

# Problem Source
A classical benchmark problem in constrained engineering design, widely used in metaheuristic algorithm evaluations.

# Variables
- `x[1]`: Thickness of the shell (Ts)
- `x[2]`: Thickness of the head (Th)
- `x[3]`: Inner radius (R)
- `x[4]`: Length of the cylindrical section without head (L)

# Constraints
Four nonlinear inequality constraints:
- Stress constraints on thickness
- Volume constraint
- Geometrical bounds

# Returns
- Penalized objective function value (Float64)
"""
function Engineering_F2(x)
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
# raw"""
#     F3(x::Vector{Float64}) -> Float64

# Welded Beam Design Optimization.

# Minimizes cost of a welded beam subject to constraints on shear stress, normal stress, deflection, and geometry.

# $$
# \vec{z} = [z_1, z_2, z_3, z_4] = [h, l, t, b] \\
# \min_{\vec{z}} f(\vec{z}) = 1.10471 z_1^2 z_2 + 0.04811 z_3 z_4 (14 + z_2) \\
# \text{s.t.} \\
# g_1(\vec{z}) = \tau(z) - \tau_{\max} \leq 0 \\
# g_2(\vec{z}) = \sigma(z) - \sigma_{\max} \leq 0 \\
# g_3(\vec{z}) = z_1 - z_4 \leq 0 \\
# g_4(\vec{z}) = 0.10471 z_1^2 + 0.04811 z_3 z_4 (14 + z_2) - 5 \leq 0 \\
# g_5(\vec{z}) = 0.125 - z_1 \leq 0 \\
# g_6(\vec{z}) = \delta(z) - \delta_{\max} \leq 0 \\
# g_7(\vec{z}) = P - P_c(z) \leq 0 \\
# 0.1 \leq z_1, z_4 \leq 2,\quad 0.1 \leq z_2, z_3 \leq 10 \\
# \tau(z) = \sqrt{(\tau')^2 + 2\tau'\tau''\frac{z_2}{2R} + (\tau'')^2} \\
# \tau' = \frac{P}{\sqrt{2} z_1 z_2},\quad \tau'' = \frac{MR}{J} \\
# M = P \left( L + \frac{z_2}{2} \right),\quad R = \sqrt{ \frac{z_2^2}{4} + \left( \frac{z_1 + z_3}{2} \right)^2 } \\
# J = 2 \sqrt{2} z_1 z_2 \left[ \frac{z_2^2}{12} + \left( \frac{z_1 + z_3}{2} \right)^2 \right] \\
# \sigma(z) = \frac{6PL}{z_4 z_3^2},\quad \delta(z) = \frac{4PL^3}{E z_3^3 z_4} \\
# P_c(z) = \frac{4.013 E \sqrt{z_3^2 z_4^5 / 36}}{L^2} \left( 1 - \frac{z_3}{2L} \sqrt{\frac{E}{4G}} \right) \\
# P=6000~\text{lb},\quad L=14~\text{in},\quad E=30\times10^6~\text{psi} \\
# G=12\times10^6~\text{psi},\quad \tau_{\max}=13600~\text{psi},\quad \sigma_{\max}=30000~\text{psi},\quad \delta_{\max}=0.25~\text{in}
# $$

# # Problem Source
# This problem is a classical constrained engineering design problem used in various metaheuristic algorithm papers.

# # Variables
# - `x[1]`: Thickness of weld
# - `x[2]`: Length of welded joint
# - `x[3]`: Height of the beam
# - `x[4]`: Width of the beam

# # Constraints
# Seven nonlinear inequality constraints.

# # Returns
# - Penalized objective function value (Float64)
# """




# """
#     Engineering_F3(x::Vector{Float64}) -> Float64

# Welded Beam Design Optimization Problem.

# Minimizes the cost of a welded beam subject to constraints on shear stress, normal stress, deflection, and geometric properties.

# # Objective

# ``\\[
# \\vec{z} = [z_1, z_2, z_3, z_4] = [h, l, t, b] \\\\
# \\min_{\\vec{z}} f(\\vec{z}) = 1.10471 z_1^2 z_2 + 0.04811 z_3 z_4 (14 + z_2)
# \\]``

# # Constraints
 
# ``\\[
# \\begin{aligned}
# g_1(\\vec{z}) &= \\tau(z) - \\tau_{\\max} \\leq 0 \\\\
# g_2(\\vec{z}) &= \\sigma(z) - \\sigma_{\\max} \\leq 0 \\\\
# g_3(\\vec{z}) &= z_1 - z_4 \\leq 0 \\\\
# g_4(\\vec{z}) &= 0.10471 z_1^2 + 0.04811 z_3 z_4 (14 + z_2) - 5 \\leq 0 \\\\
# g_5(\\vec{z}) &= 0.125 - z_1 \\leq 0 \\\\
# g_6(\\vec{z}) &= \\delta(z) - \\delta_{\\max} \\leq 0 \\\\
# g_7(\\vec{z}) &= P - P_c(z) \\leq 0
# \\end{aligned}
# \\]``

# # Definitions

# ``\\[
# \\tau(z) = \\sqrt{(\\tau')^2 + 2\\tau'\\tau''\\frac{z_2}{2R} + (\\tau'')^2},\\quad
# \\tau' = \\frac{P}{\\sqrt{2} z_1 z_2},\\quad
# \\tau'' = \\frac{MR}{J}
# \\]``

# ``\\[
# M = P \\left( L + \\frac{z_2}{2} \\right),\\quad
# R = \\sqrt{ \\frac{z_2^2}{4} + \\left( \\frac{z_1 + z_3}{2} \\right)^2 }
# \\]``

# ``\\[
# J = 2 \\sqrt{2} z_1 z_2 \\left[ \\frac{z_2^2}{12} + \\left( \\frac{z_1 + z_3}{2} \\right)^2 \\right]
# \\]``

# ``\\[
# \\sigma(z) = \\frac{6PL}{z_4 z_3^2},\\quad
# \\delta(z) = \\frac{4PL^3}{E z_3^3 z_4}
# \\]``

# ``\\[
# P_c(z) = \\frac{4.013 E \\sqrt{z_3^2 z_4^5 / 36}}{L^2} \\left( 1 - \\frac{z_3}{2L} \\sqrt{\\frac{E}{4G}} \\right)
# \\]``

# # Constants

# - `P = 6000` lb
# - `L = 14` in
# - `E = 30×10⁶` psi
# - `G = 12×10⁶` psi
# - `τₘₐₓ = 13600` psi
# - `σₘₐₓ = 30000` psi
# - `δₘₐₓ = 0.25` in

# # Decision Variables

# - `x[1] = z₁`: Thickness of weld (h)
# - `x[2] = z₂`: Length of weld (l)
# - `x[3] = z₃`: Height of beam (t)
# - `x[4] = z₄`: Width of beam (b)

# # Returns

# - Penalized objective function value (`Float64`)
# """



raw"""
    Engineering_F3(x::Vector{Float64}) -> Float64

Welded Beam Design Optimization Problem.

Minimizes the cost of a welded beam subject to constraints on shear stress, normal stress, deflection, and geometric properties.
\begin{equation}\label{eq:WBD}
	\begin{aligned}
		&\text{Consider variable}& & \vec{z}=[z_{1},z_{2},z_{3},z_{4}]=[h,l,t,b].\\
		& \underset{\vec{z}}{\text{Minimize}}
		& & f(\vec{z})=1.10471 z_{1}^{2} z_{2}+0.04811 z_{3} z_{4}\left(14+z_{2}\right). \\
		& \text{Subject to} & & g_{1}(\vec{z})=\tau(z)-\tau_{\max } \leqslant 0.\\
		& & & g_{2}(\vec{z})=\sigma(z)-\sigma_{\max } \leqslant 0.\\
		& & & g_{3}(\vec{z})=z_{1}-z_{4} \leqslant 0.\\
		& & & g_{4}(\vec{z})=0.10471 z_{1}^{2}+ 0.04811 z_{3} z_{4}\left(14+z_{2}\right)-5 \leqslant 0.\\
		& & & g_{5}(\vec{z})=0.125-z_{1} \leqslant 0. \\
		& & & g_{6}(\vec{z})=\delta(z)-\delta_{\max } \leqslant 0. \\
		& & & g_{7}(\vec{z})=P-P_{c}(z) \leqslant 0. \\
		& \text{Variable range} & & 0.1 \leqslant z_{1},z_{4} \leqslant 2.\\
		& & & 0.1 \leqslant z_{2},z_{3} \leqslant 10.\\
		& \text{Where} & & \tau(z)=\sqrt{\left(\tau^{\prime}\right)^{2}+2 \tau^{\prime} \tau^{\prime \prime} \frac{z_{2}}{2 R}+\left(\tau^{\prime \prime}\right)^{2}}. \\
		& & & \tau^{\prime}=\frac{P}{\sqrt{2} z_{1} z_{2}},\quad
		\tau^{\prime \prime}=\frac{M R}{J}. \\
		& & & M=P\left(L+\frac{z_{2}}{2}\right). \\ 
		& & & R=\sqrt{\frac{z_{2}^{2}}{4}+\left(\frac{z_{1}+z_{3}}{2}\right)^{2}}. \\
		& & & J=2\left\{\sqrt{2} z_{1} z_{2}\left[\frac{z_{2}^{2}}{12}+\left(\frac{z_{1}+z_{3}}{2}\right)^{2}\right]\right\}. \\
		& & & \sigma(z)=\frac{6 P L}{z_{4} \chi_{3}^{2}}, \quad \delta(z)=\frac{4 P L^{3}}{E z_{3}^{3} z_{4}}. \\
		& & & P_{c}(z)=\frac{4.013 E \sqrt{\frac{z^{2} z^{5}}{36}}}{L^{2}}\left(1-\frac{z_{3}}{2 L} \sqrt{\frac{E}{4 G}}\right). \\
		& & & P=6000 lb, \quad L=14 in, \quad E=30 \times 10^{6} \mathrm{psi}. \\
		& & & G=12 \times 10^{6} \mathrm{psi}, \quad \tau_{\max }=13,600 ps. \\
		& & & \sigma_{\max }=30,000 \mathrm{psi}, \quad \delta_{\max }=0.25 in. \\
	\end{aligned}
\end{equation}

# Objective

``\\[
\\vec{z} = [z_1, z_2, z_3, z_4] = [h, l, t, b] \\\\
\\min_{\\vec{z}} f(\\vec{z}) = 1.10471 z_1^2 z_2 + 0.04811 z_3 z_4 (14 + z_2)
\\]``

# Constraints
 
``\\[
\\begin{aligned}
g_1(\\vec{z}) &= \\tau(z) - \\tau_{\\max} \\leq 0 \\\\
g_2(\\vec{z}) &= \\sigma(z) - \\sigma_{\\max} \\leq 0 \\\\
g_3(\\vec{z}) &= z_1 - z_4 \\leq 0 \\\\
g_4(\\vec{z}) &= 0.10471 z_1^2 + 0.04811 z_3 z_4 (14 + z_2) - 5 \\leq 0 \\\\
g_5(\\vec{z}) &= 0.125 - z_1 \\leq 0 \\\\
g_6(\\vec{z}) &= \\delta(z) - \\delta_{\\max} \\leq 0 \\\\
g_7(\\vec{z}) &= P - P_c(z) \\leq 0
\\end{aligned}
\\]``

# Definitions

``\\[
\\tau(z) = \\sqrt{(\\tau')^2 + 2\\tau'\\tau''\\frac{z_2}{2R} + (\\tau'')^2},\\quad
\\tau' = \\frac{P}{\\sqrt{2} z_1 z_2},\\quad
\\tau'' = \\frac{MR}{J}
\\]``

``\\[
M = P \\left( L + \\frac{z_2}{2} \\right),\\quad
R = \\sqrt{ \\frac{z_2^2}{4} + \\left( \\frac{z_1 + z_3}{2} \\right)^2 }
\\]``

``\\[
J = 2 \\sqrt{2} z_1 z_2 \\left[ \\frac{z_2^2}{12} + \\left( \\frac{z_1 + z_3}{2} \\right)^2 \\right]
\\]``

``\\[
\\sigma(z) = \\frac{6PL}{z_4 z_3^2},\\quad
\\delta(z) = \\frac{4PL^3}{E z_3^3 z_4}
\\]``

``\\[
P_c(z) = \\frac{4.013 E \\sqrt{z_3^2 z_4^5 / 36}}{L^2} \\left( 1 - \\frac{z_3}{2L} \\sqrt{\\frac{E}{4G}} \\right)
\\]``

# Constants

- `P = 6000` lb
- `L = 14` in
- `E = 30×10⁶` psi
- `G = 12×10⁶` psi
- `τₘₐₓ = 13600` psi
- `σₘₐₓ = 30000` psi
- `δₘₐₓ = 0.25` in

# Decision Variables

- `x[1] = z₁`: Thickness of weld (h)
- `x[2] = z₂`: Length of weld (l)
- `x[3] = z₃`: Height of beam (t)
- `x[4] = z₄`: Width of beam (b)

# Returns

- Penalized objective function value (`Float64`)
"""

function Engineering_F3(x)
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
"""
    F4(x::Vector{Float64}) -> Float64

Speed Reducer Design Optimization.

Minimizes the weight of a speed reducer subject to constraints on bending stress, surface stress, transverse deflections, and geometry.

# Problem Source
A standard benchmark problem in engineering design, commonly used to test constrained optimization algorithms.

# Variables
- `x[1]`: Face width (in)
- `x[2]`: Module of teeth (in)
- `x[3]`: Number of teeth
- `x[4]`: Length of the first shaft between bearings (in)
- `x[5]`: Length of the second shaft between bearings (in)
- `x[6]`: Diameter of the first shaft (in)
- `x[7]`: Diameter of the second shaft (in)

# Constraints
- Bending stress
- Surface stress
- Deflection of shafts
- Geometric and design constraints
- Seven nonlinear inequality constraints in total

# Returns
- Penalized objective function value (Float64)
"""
function Engineering_F4(x)
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
"""
    F5(x::Vector{Float64}) -> Float64

Gear Train Design Optimization.

Minimizes the error between an actual and a desired gear ratio in a simple four-gear train. All variables must be integers.

# Problem Source
A discrete constrained engineering design problem widely used to evaluate optimization algorithms that handle integer variables.

# Variables
- `x[1]`: Number of teeth on gear 1 (integer)
- `x[2]`: Number of teeth on gear 2 (integer)
- `x[3]`: Number of teeth on gear 3 (integer)
- `x[4]`: Number of teeth on gear 4 (integer)

# Constraints
- Each variable must be an integer in the range [12, 60]
- The gear ratio error must be minimized

# Returns
- Squared error between actual and desired gear ratio (Float64)
"""
function Engineering_F5(x)
    x = round.(x)
    cost = (1/6.931 - (x[3]*x[2]) / (x[1]*x[4]))^2
    return cost
end

# F6: Three-bar truss design
"""
    F6(x::Vector{Float64}) -> Float64

Three-Bar Truss Design Optimization.

Minimizes the weight of a three-bar truss structure subject to stress and displacement constraints.

# Problem Source
A classical structural optimization benchmark problem used in metaheuristic algorithm research.

# Variables
- `x[1]`: Cross-sectional area of the first bar (continuous)
- `x[2]`: Cross-sectional area of the second bar (continuous)

# Constraints
- Stress in each member must not exceed allowable limits
- Displacement constraints on the structure
- Variable bounds typically in the range [0.1, 10]

# Returns
- Penalized objective function value (Float64) representing the weight of the truss
"""
function Engineering_F6(x)
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
    F7(x::Vector{Float64}) -> Float64

Rolling Element Bearing Design Optimization.

Minimizes the bearing’s weight subject to constraints on stress, deflection, and geometry.

# Problem Source
A standard constrained engineering design problem often used to benchmark metaheuristic algorithms.

# Variables
- `x[1]`: Bearing inner radius
- `x[2]`: Bearing outer radius
- `x[3]`: Width of the bearing
- `x[4]`: Shaft diameter
- `x[5]`: Number of rolling elements

# Constraints
- Stress limits on the bearing components
- Deflection limits
- Geometric and manufacturing constraints

# Returns
- Penalized objective function value (Float64) reflecting the bearing weight or cost
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
"""
    F8(x::Vector{Float64}) -> Float64

Cantilever Beam Design Optimization.

Minimizes the weight of a cantilever beam subject to constraints on bending stress, deflection, and geometric dimensions.

# Problem Source
A classical constrained engineering design problem used in metaheuristic algorithm research.

# Variables
- `x[1]`: Width of the beam cross-section
- `x[2]`: Height of the beam cross-section
- `x[3]`: Length of the beam segment 1
- `x[4]`: Length of the beam segment 2
- `x[5]`: Length of the beam segment 3
- `x[6]`: Length of the beam segment 4

# Constraints
- Maximum bending stress constraints
- Deflection limits at the beam’s free end
- Geometric bounds on variables

# Returns
- Penalized objective function value (Float64), representing the beam weight
"""
function Engineering_F8(x)
    cost = 0.0624 * sum(x)
    g = 61/x[1]^3 + 37/x[2]^3 + 19/x[3]^3 + 7/x[4]^3 + 1/x[5]^3 - 1
    lam = 1e15
    Z = lam * g^2 * getH(g)
    return cost + Z
end

# F9: I-beam deflection
"""
    F9(x::Vector{Float64}) -> Float64

I-Beam Deflection Optimization.

Minimizes the weight of an I-beam subject to constraints on bending stress, shear stress, and deflection under load.

# Problem Source
A classical engineering design benchmark widely used in metaheuristic algorithm literature.

# Variables
- `x[1]`: Web height
- `x[2]`: Flange width
- `x[3]`: Web thickness
- `x[4]`: Flange thickness

# Constraints
- Bending stress limits
- Shear stress limits
- Maximum deflection allowed
- Geometric constraints

# Returns
- Penalized objective function value (Float64), representing the beam weight
"""
function Engineering_F9(x)
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
