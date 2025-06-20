@testset "DSO" verbose = true begin 
    @testset "Ackley with DSO, p = 2" verbose = true begin 
        lb = [-32.768 for i in 1:2]
        ub = [32.768 for i in 1:2]
        result = DSO(100, 500, lb, ub, Ackley)
        @test isapprox(result.bestX, [0.0 for i in 1:2], atol=1e-5)
        @test isapprox(result.bestF, 0.0, atol=1e-5)
    end 
    @testset "Ackley with DSO, p = 5" verbose = true begin 
        lb = [-32.768 for i in 1:5]
        ub = [32.768 for i in 1:5]
        result = DSO(100, 500, lb, ub, Ackley)
        @test isapprox(result.bestX, [0.0 for i in 1:5], atol=1e-5)
        @test isapprox(result.bestF, 0.0, atol=1e-5)
    end
    @testset "Griewank with DSO, p = 2" verbose = true begin 
        lb = [-600 for i in 1:2]
        ub = [600 for i in 1:2]
        result = DSO(100, 500, lb, ub, Griewank)
        @test isapprox(result.bestX, [0.0 for i in 1:2], atol=1e-5)
        @test isapprox(result.bestF, 0.0, atol=1e-5)
    end
end 