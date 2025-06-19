@testset "WOA_struct" verbose = true begin 
    @testset "Ackley with WOA_struct, p = 2" verbose = true begin 
        lb = [-32.768 for i in 1:2]
        ub = [32.768 for i in 1:2]
        problem = OptimizationProblem(Ackley, lb, ub, 2)
        # result = AHA(100, 1500, lb, ub, Ackley)
        result = WOA(problem, 100, 1500)
        @test isapprox(result.BestX, [0.0 for i in 1:2], atol=1e-5)
        @test isapprox(result.BestF, 0.0, atol=1e-5)
    end 
    @testset "Ackley with WOA_struct, p = 5" verbose = true begin 
        lb = [-32.768 for i in 1:5]
        ub = [32.768 for i in 1:5]
        # result = AHA(100, 1500, lb, ub, Ackley)
        problem = OptimizationProblem(Ackley, lb, ub, 5)
        result = WOA(problem, 100, 1500)
        @test isapprox(result.BestX, [0.0 for i in 1:5], atol=1e-5)
        @test isapprox(result.BestF, 0.0, atol=1e-5)
    end
    @testset "Griewank with WOA_struct, p = 2" verbose = true begin 
        lb = [-600 for i in 1:2]
        ub = [600 for i in 1:2]
        # result = AHA(100, 1500, lb, ub, Griewank)
        problem = OptimizationProblem(Ackley, lb, ub, 2)
        result = WOA(problem, 100, 1500)
        @test isapprox(result.BestX, [0.0 for i in 1:2], atol=1e-5)
        @test isapprox(result.BestF, 0.0, atol=1e-5)
    end
end 
