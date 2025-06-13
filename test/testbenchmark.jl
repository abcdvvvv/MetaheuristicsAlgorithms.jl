@testset "Test Ackley" verbose = true begin
    @testset "Ackley, p = 2" verbose = true begin
        result = Ackley([0.0, 0.0])
        @test isapprox(result, 0.0, atol=1e-5)
    end
    @testset "Ackley, p = 5" verbose = true begin
        result = Ackley([0.0, 0.0, 0.0, 0.0, 0.0])
        @test isapprox(result, 0.0, atol=1e-5)
    end
end

@testset "Test Griewank" verbose = true begin
    @testset "Griewank, p = 2" verbose = true begin
        result = Griewank([0.0, 0.0])
        @test isapprox(result, 0.0, atol=1e-5)
    end
    @testset "Griewank, p = 5" verbose = true begin
        result = Griewank([0.0, 0.0, 0.0, 0.0, 0.0])
        @test isapprox(result, 0.0, atol=1e-5)
    end
end