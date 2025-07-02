# Ufun used in F12 and F13
function Ufun(x, a, k, m)
    return k .* ((x .- a).^m) .* (x .> a) + k .* ((-x .- a).^m) .* (x .< -a)
end
"""
F1()

Sphere function.

This is a basic unimodal test function used in benchmarking optimization algorithms.

**Equation:**

```math
f(\\mathbf{x}) = \\sum_{i=1}^n x_i^2
```

"""
F1(x) = sum(x .^ 2)

"""
    F2()
Sum of Absolute Values and Product of Absolute Values.

This is a basic test function used in optimization, combining both the sum and product of the absolute values of the input vector elements.

# Equation

```math
f(\\mathbf{x}) = \\sum_{i=1}^n |x_i| + \\prod_{i=1}^n |x_i|```
"""
F2(x) = sum(abs.(x)) + prod(abs.(x))

"""
    F3()
Cumulative Sum of Squares Function.

This test function computes the sum of the squares of cumulative sums of the input vector. It increases the dependency between variables and is used to test an algorithm’s ability to handle variable interactions.

# Equation

```math
f(\\mathbf{x}) = \\sum_{i=1}^n \\left( \\sum_{j=1}^i x_j \\right)^2```
"""
function F3(x)
    o = 0.0
    for i in 1:length(x)
        o += sum(x[1:i])^2
    end
    return o
end

"""
    F4()
Maximum Absolute Value Function.

This function returns the maximum of the absolute values of the input vector elements. It is used to test an optimizer's ability to minimize the worst-case (largest-magnitude) variable.

# Equation

```math
f(\\mathbf{x}) = \\max_{1 \\leq i \\leq n} |x_i|```
"""
F4(x) = maximum(abs.(x))

"""
    F5()
Rosenbrock Function.

A classic, non-convex test problem for optimization algorithms. It has a narrow, curved valley leading to the global minimum, which makes convergence difficult.

# Equation

```math
f(\\mathbf{x}) = \\sum_{i=1}^{n-1} \\left[ 100(x_{i+1} - x_i^2)^2 + (x_i - 1)^2 \\right]```
"""
function F5(x)
    return sum(100 .* (x[2:end] .- x[1:end-1].^2).^2 + (x[1:end-1] .- 1).^2)
end

"""
    F6()
Shifted Sphere Function.

This function is a variation of the Sphere function where each variable is shifted by 0.5 before squaring. It remains unimodal but shifts the global minimum from the origin.

# Equation

```math
f(\\mathbf{x}) = \\sum_{i=1}^n (x_i + 0.5)^2```
"""
F6(x) = sum(abs.(x .+ 0.5).^2)

"""
    F7()
Weighted Quartic Function with Noise.

This function adds a random noise term to a weighted sum of the fourth powers of the input variables. The noise introduces stochasticity, making it useful for testing robustness of optimization algorithms.

# Equation

```math
f(\\mathbf{x}) = \\sum_{i=1}^n i \\cdot x_i^4 + \\text{rand}()```
"""
function F7(x)
    dim = length(x)
    return sum((1:dim) .* (x.^4)) + rand()
end

"""
    F8()
Schwefel Function.

A widely used multimodal benchmark function with many local minima. It poses a challenge for optimization algorithms due to its deceptive landscape and large search space.

# Equation

```math
f(\\mathbf{x}) = \\sum_{i=1}^n -x_i \\cdot \\sin(\\sqrt{|x_i|})```
"""
F8(x) = sum(-x .* sin.(sqrt.(abs.(x))))

"""
    F9()
Rastrigin Function.

A highly multimodal benchmark function commonly used to evaluate the performance of global optimization algorithms. Its large number of local minima makes it particularly challenging.

# Equation

```math
f(\\mathbf{x}) = \\sum_{i=1}^n \\left( x_i^2 - 10 \\cos(2\\pi x_i) + 10 \\right)```
"""
function F9(x)
    dim = length(x)
    return sum(x.^2 .- 10 .* cos.(2π .* x)) + 10 * dim
end

"""
    F10()
Ackley Function.

A popular multimodal benchmark function used to test optimization algorithms. It features a nearly flat outer region and a large number of local minima, making convergence difficult.

# Equation

```math
f(\\mathbf{x}) = -20 \\exp\\left(-0.2 \\sqrt{\\frac{1}{n} \\sum_{i=1}^n x_i^2}\\right)
                - \\exp\\left(\\frac{1}{n} \\sum_{i=1}^n \\cos(2\\pi x_i)\\right)
                + 20 + e```
"""
function F10(x)
    dim = length(x)
    return -20 * exp(-0.2 * sqrt(sum(x.^2) / dim)) -
           exp(sum(cos.(2π .* x)) / dim) + 20 + exp(1)
end

"""
    F11()
Griewank Function.

A widely used multimodal test function with many widespread local minima, but a simple global minimum at the origin.

# Equation

```math
f(\\mathbf{x}) = \\frac{1}{4000} \\sum_{i=1}^n x_i^2 - \\prod_{i=1}^n \\cos\\left( \\frac{x_i}{\\sqrt{i}} \\right) + 1```
"""
function F11(x)
    dim = length(x)
    return sum(x.^2) / 4000 - prod(cos.(x ./ sqrt.(1:dim))) + 1
end

"""
    F12()
Penalized Function #1.

A multimodal benchmark function with penalization terms to enforce constraints, often used in optimization testing.

# Equation

```math
f(\\mathbf{x}) = \\frac{\\pi}{n} \\left[ 10 \\sin^2 \\left( \\pi \\left(1 + \\frac{x_1 + 1}{4} \\right) \\right) + \\sum_{i=1}^{n-1} \\left( \\frac{x_i + 1}{4} \\right)^2 \\left( 1 + 10 \\sin^2 \\left( \\pi \\left(1 + \\frac{x_{i+1} + 1}{4} \\right) \\right) \\right) + \\left( \\frac{x_n + 1}{4} \\right)^2 \\right] + \\sum_{i=1}^n U(x_i, 10, 100, 4)```
"""
function F12(x)
    dim = length(x)
    term1 = (π / dim) * (10 * sin(π * (1 + (x[1] + 1) / 4))^2)
    term2 = sum(((x[1:dim-1] .+ 1) ./ 4).^2 .* (1 .+ 10 .* sin.(π .* (1 .+ (x[2:dim] .+ 1) ./ 4)).^2))
    term3 = ((x[dim] + 1) / 4)^2
    return term1 + term2 + term3 + sum(Ufun(x, 10, 100, 4))
end


"""
    F13()
Penalized Function #2.

A multimodal benchmark function with penalization terms used to test optimization algorithms, featuring sine and quadratic terms.

# Equation

```math
f(\\mathbf{x}) = 0.1 \\left[ \\sin^2(3 \\pi x_1) + \\sum_{i=1}^{n-1} (x_i - 1)^2 (1 + \\sin^2(3 \\pi x_{i+1})) + (x_n - 1)^2 (1 + \\sin^2(2 \\pi x_n)) \\right] + \\sum_{i=1}^n U(x_i, 5, 100, 4)```
"""
function F13(x)
    dim = length(x)
    term1 = sin(3π * x[1])^2
    term2 = sum((x[1:dim-1] .- 1).^2 .* (1 .+ sin.(3π .* x[2:dim]).^2))
    term3 = (x[dim] - 1)^2 * (1 + sin(2π * x[dim])^2)
    return 0.1 * (term1 + term2 + term3) + sum(Ufun(x, 5, 100, 4))
end


"""
    F14()
"""
function F14(x)
    aS = hcat([-32, -16, 0, 16, 32,
               -32, -16, 0, 16, 32,
               -32, -16, 0, 16, 32,
               -32, -16, 0, 16, 32,
               -32, -16, 0, 16, 32]...,
              [-32, -32, -32, -32, -32,
               -16, -16, -16, -16, -16,
                0,  0,  0,  0,  0,
               16, 16, 16, 16, 16,
               32, 32, 32, 32, 32]...)
    bS = [sum((x .- aS[:, j]).^6) for j in 1:25]
    return (1 / (0.002 + sum(1 ./ (1:25 .+ bS))))^(-1)
end

"""
    F15()
"""
function F15(x)
    aK = [.1957, .1947, .1735, .16, .0844, .0627, .0456, .0342, .0323, .0235, .0246]
    bK = 1 ./ [.25, .5, 1, 2, 4, 6, 8, 10, 12, 14, 16]
    return sum((aK .- (x[1] .* (bK.^2 .+ x[2] .* bK)) ./ (bK.^2 .+ x[3] .* bK .+ x[4])).^2)
end

"""
    F16()
"""
function F16(x)
    return 4*x[1]^2 - 2.1*x[1]^4 + (x[1]^6)/3 + x[1]*x[2] - 4*x[2]^2 + 4*x[2]^4
end

"""
    F17()
"""
function F17(x)
    return (x[2] - x[1]^2 * 5.1 / (4π^2) + 5 / π * x[1] - 6)^2 +
           10 * (1 - 1 / (8π)) * cos(x[1]) + 10
end

"""
    F18()
"""
function F18(x)
    return (1 + (x[1] + x[2] + 1)^2 * (19 - 14x[1] + 3x[1]^2 - 14x[2] +
            6x[1]*x[2] + 3x[2]^2)) * (30 + (2x[1] - 3x[2])^2 * (18 - 32x[1] +
            12x[1]^2 + 48x[2] - 36x[1]*x[2] + 27x[2]^2))
end

"""
    F19()
"""
function F19(x)
    aH = [3 10 30; .1 10 35; 3 10 30; .1 10 35]
    cH = [1, 1.2, 3, 3.2]
    pH = [.3689 .1170 .2673;
          .4699 .4387 .7470;
          .1091 .8732 .5547;
          .0381 .5743 .8828]
    return -sum(cH[i] * exp(-sum(aH[i, :] .* (x .- pH[i, :]).^2)) for i in 1:4)
end

"""
    F20()
"""
function F20(x)
    aH = [10 3 17 3.5 1.7 8;
          .05 10 17 .1 8 14;
          3 3.5 1.7 10 17 8;
          17 8 .05 10 .1 14]
    cH = [1, 1.2, 3, 3.2]
    pH = [.1312 .1696 .5569 .0124 .8283 .5886;
          .2329 .4135 .8307 .3736 .1004 .9991;
          .2348 .1415 .3522 .2883 .3047 .6650;
          .4047 .8828 .8732 .5743 .1091 .0381]
    return -sum(cH[i] * exp(-sum(aH[i, :] .* (x .- pH[i, :]).^2)) for i in 1:4)
end

"""
    F21()
"""
function F21(x)
    aSH = [4 4 4 4;
           1 1 1 1;
           8 8 8 8;
           6 6 6 6;
           3 7 3 7;
           2 9 2 9;
           5 5 3 3;
           8 1 8 1;
           6 2 6 2;
           7 3.6 7 3.6]
    cSH = [.1, .2, .2, .4, .4, .6, .3, .7, .5, .5]
    return -sum((dot(x .- aSH[i, :], x .- aSH[i, :]) + cSH[i])^(-1) for i in 1:5)
end

"""
    F22()
"""
function F22(x)
    aSH = [4 4 4 4;
           1 1 1 1;
           8 8 8 8;
           6 6 6 6;
           3 7 3 7;
           2 9 2 9;
           5 5 3 3;
           8 1 8 1;
           6 2 6 2;
           7 3.6 7 3.6]
    cSH = [.1, .2, .2, .4, .4, .6, .3, .7, .5, .5]
    return -sum((dot(x .- aSH[i, :], x .- aSH[i, :]) + cSH[i])^(-1) for i in 1:7)
end

"""
    F23()
"""
function F23(x)
    aSH = [4 4 4 4;
           1 1 1 1;
           8 8 8 8;
           6 6 6 6;
           3 7 3 7;
           2 9 2 9;
           5 5 3 3;
           8 1 8 1;
           6 2 6 2;
           7 3.6 7 3.6]
    cSH = [.1, .2, .2, .4, .4, .6, .3, .7, .5, .5]
    return -sum((dot(x .- aSH[i, :], x .- aSH[i, :]) + cSH[i])^(-1) for i in 1:10)
end
