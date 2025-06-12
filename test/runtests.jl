using MetaheuristicsAlgorithms
using Test
using Random

include("benchmark.jl")

@testset "MetaheuristicsAlgorithms.jl" begin
    @testset "AEFA on Eggholder function" begin
        Random.seed!(1)
        lb = [-512, -512]
        ub = [512, 512]
        algo = AEO(100, 300, lb, ub)

        while algo.update
            y = benchmark(algo.x; funNo=1)
            update_state!(algo, y)
        end
        @test algo.gbest == [-461.648054120324, -390.06329199871277]
        @test algo.gbest_val[end] == -775.8393103361666
    end

    @testset "AEO on Eggholder function" begin
        Random.seed!(1)
        lb = [-512, -512]
        ub = [512, 512]
        fun(x) = first(benchmark(x))
        BestF, BestX, _ = AEO(100, 300, lb, ub, 2, fun)

        @test BestF == -959.640662720851
        @test BestX == [512.0, 404.2318050941216]
    end
end
