@testset "AEO_struct" verbose = true begin 
    @testset "Ackley with AEO_struct, p = 2" verbose = true begin 
        lb = [-32.768 for i in 1:2]
        ub = [32.768 for i in 1:2]
        # result = AEO(100, 500, lb, ub, Ackley)
        problem = MyOptimization_Problem(Ackley, lb, ub, 2)
        result = AEO(problem, 100, 500)
        @test isapprox(result.bestX, [0.0 for i = 1:2], atol=1e-5)
        @test isapprox(result.bestF, 0.0, atol=1e-5)
    end 
    @testset "Ackley with AEO_struct, p = 5" verbose = true begin 
        lb = [-32.768 for i in 1:5]
        ub = [32.768 for i in 1:5]
        # result = AEO(100, 500, lb, ub, Ackley)
        problem = MyOptimization_Problem(Ackley, lb, ub, 2)
        result = AEO(problem, 100, 500)
        @test isapprox(result.bestX, [0.0 for i in 1:5], atol=1e-5)
        @test isapprox(result.bestF, 0.0, atol=1e-5)
    end
    @testset "Griewank with AEO_struct, p = 2" verbose = true begin 
        lb = [-600 for i in 1:2]
        ub = [600 for i in 1:2]
        # result = AEO(100, 500, lb, ub, Griewank)
        problem = MyOptimization_Problem(Ackley, lb, ub, 2)
        result = AEO(problem, 100, 500)
        @test isapprox(result.bestX, [0.0 for i in 1:2], atol=1e-5)
        @test isapprox(result.bestF, 0.0, atol=1e-5)
    end
end 