@testset "AEO" verbose = true begin 
    @testset "Ackley with AEO, p = 2" verbose = true begin 
        lb = [-32.768 for i in 1:2]
        ub = [32.768 for i in 1:2]
        result = AEO(100, 500, lb, ub, Ackley)
        @test isapprox(result.BestX, [0.0 for i in 1:2], atol=1e-5)
        @test isapprox(result.BestF, 0.0, atol=1e-5)
    end 
    @testset "Ackley with AEO, p = 5" verbose = true begin 
        lb = [-32.768 for i in 1:5]
        ub = [32.768 for i in 1:5]
        result = AEO(100, 500, lb, ub, Ackley)
        @test isapprox(result.BestX, [0.0 for i in 1:5], atol=1e-5)
        @test isapprox(result.BestF, 0.0, atol=1e-5)
    end
    @testset "Griewank with AEO, p = 2" verbose = true begin 
        lb = [-600 for i in 1:2]
        ub = [600 for i in 1:2]
        result = AEO(100, 500, lb, ub, Griewank)
        @test isapprox(result.BestX, [0.0 for i in 1:2], atol=1e-5)
        @test isapprox(result.BestF, 0.0, atol=1e-5)
    end
end 