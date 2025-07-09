@testset "QIO" verbose = true begin
    @testset "Ackley with QIO, p = 2" verbose = true begin
        lb = [-32.768 for i = 1:2]
        ub = [32.768 for i = 1:2]
        result = QIO(Ackley, lb, ub, 100, 1000)
        @test isapprox(result.bestX, [0.0 for i = 1:2], atol=1e-5)
        @test isapprox(result.bestF, 0.0, atol=1e-5)
    end
    @testset "Ackley with QIO, p = 5" verbose = true begin
        lb = [-32.768 for i = 1:5]
        ub = [32.768 for i = 1:5]
        result = QIO(Ackley, lb, ub, 100, 1000)
        @test isapprox(result.bestX, [0.0 for i = 1:5], atol=1e-5)
        @test isapprox(result.bestF, 0.0, atol=1e-5)
    end
    @testset "Griewank with QIO, p = 2" verbose = true begin
        lb = [-600.0 for i = 1:2]
        ub = [600.0 for i = 1:2]
        result = QIO(Griewank, lb, ub, 100, 1000)
        @test isapprox(result.bestX, [0.0 for i = 1:2], atol=1e-5)
        @test isapprox(result.bestF, 0.0, atol=1e-5)
    end
end
