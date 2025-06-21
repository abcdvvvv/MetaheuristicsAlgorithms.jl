# Ufun used in F12 and F13
function Ufun(x, a, k, m)
    return k .* ((x .- a).^m) .* (x .> a) + k .* ((-x .- a).^m) .* (x .< -a)
end

F1(x) = sum(x .^ 2)

F2(x) = sum(abs.(x)) + prod(abs.(x))

function F3(x)
    o = 0.0
    for i in 1:length(x)
        o += sum(x[1:i])^2
    end
    return o
end

F4(x) = maximum(abs.(x))

function F5(x)
    return sum(100 .* (x[2:end] .- x[1:end-1].^2).^2 + (x[1:end-1] .- 1).^2)
end

F6(x) = sum(abs.(x .+ 0.5).^2)

function F7(x)
    dim = length(x)
    return sum((1:dim) .* (x.^4)) + rand()
end

F8(x) = sum(-x .* sin.(sqrt.(abs.(x))))

function F9(x)
    dim = length(x)
    return sum(x.^2 .- 10 .* cos.(2π .* x)) + 10 * dim
end

function F10(x)
    dim = length(x)
    return -20 * exp(-0.2 * sqrt(sum(x.^2) / dim)) -
           exp(sum(cos.(2π .* x)) / dim) + 20 + exp(1)
end

function F11(x)
    dim = length(x)
    return sum(x.^2) / 4000 - prod(cos.(x ./ sqrt.(1:dim))) + 1
end

function F12(x)
    dim = length(x)
    term1 = (π / dim) * (10 * sin(π * (1 + (x[1] + 1) / 4))^2)
    term2 = sum(((x[1:dim-1] .+ 1) ./ 4).^2 .* (1 .+ 10 .* sin.(π .* (1 .+ (x[2:dim] .+ 1) ./ 4)).^2))
    term3 = ((x[dim] + 1) / 4)^2
    return term1 + term2 + term3 + sum(Ufun(x, 10, 100, 4))
end

function F13(x)
    dim = length(x)
    term1 = sin(3π * x[1])^2
    term2 = sum((x[1:dim-1] .- 1).^2 .* (1 .+ sin.(3π .* x[2:dim]).^2))
    term3 = (x[dim] - 1)^2 * (1 + sin(2π * x[dim])^2)
    return 0.1 * (term1 + term2 + term3) + sum(Ufun(x, 5, 100, 4))
end

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

function F15(x)
    aK = [.1957, .1947, .1735, .16, .0844, .0627, .0456, .0342, .0323, .0235, .0246]
    bK = 1 ./ [.25, .5, 1, 2, 4, 6, 8, 10, 12, 14, 16]
    return sum((aK .- (x[1] .* (bK.^2 .+ x[2] .* bK)) ./ (bK.^2 .+ x[3] .* bK .+ x[4])).^2)
end

function F16(x)
    return 4*x[1]^2 - 2.1*x[1]^4 + (x[1]^6)/3 + x[1]*x[2] - 4*x[2]^2 + 4*x[2]^4
end

function F17(x)
    return (x[2] - x[1]^2 * 5.1 / (4π^2) + 5 / π * x[1] - 6)^2 +
           10 * (1 - 1 / (8π)) * cos(x[1]) + 10
end

function F18(x)
    return (1 + (x[1] + x[2] + 1)^2 * (19 - 14x[1] + 3x[1]^2 - 14x[2] +
            6x[1]*x[2] + 3x[2]^2)) * (30 + (2x[1] - 3x[2])^2 * (18 - 32x[1] +
            12x[1]^2 + 48x[2] - 36x[1]*x[2] + 27x[2]^2))
end

function F19(x)
    aH = [3 10 30; .1 10 35; 3 10 30; .1 10 35]
    cH = [1, 1.2, 3, 3.2]
    pH = [.3689 .1170 .2673;
          .4699 .4387 .7470;
          .1091 .8732 .5547;
          .0381 .5743 .8828]
    return -sum(cH[i] * exp(-sum(aH[i, :] .* (x .- pH[i, :]).^2)) for i in 1:4)
end

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
