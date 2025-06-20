@testset "WOA_struct" verbose = true begin 
    @testset "Ackley with WOA_struct, p = 2" verbose = true begin 
        lb = [-32.768 for i in 1:2]
        ub = [32.768 for i in 1:2]
        problem = OptimizationProblem(Ackley, lb, ub, 2)
        # result = AHA(100, 1500, lb, ub, Ackley)
        result = WOA(problem, 100, 1500)
        @test isapprox(result.BestX, [0.0 for i in 1:2], atol=1e-5)
        @test isapprox(result.BestF, 0.0, atol=1e-5)
         p = plot(result.convergence_curve,
         xlabel="Iteration",
         ylabel="Best Fitness",
         title="Convergence Curve",
         legend=false)
         display(p)
    end 
    @testset "Ackley with WOA_struct, p = 5" verbose = true begin 
        lb = [-32.768 for i in 1:5]
        ub = [32.768 for i in 1:5]
        # result = AHA(100, 1500, lb, ub, Ackley)
        problem1 = OptimizationProblem(Ackley, lb, ub, 5)
        result = WOA(problem1, 100, 1500)
        @test isapprox(result.BestX, [0.0 for i in 1:5], atol=1e-5)
        @test isapprox(result.BestF, 0.0, atol=1e-5)
        p = plot(result.convergence_curve,
         xlabel="Iteration",
         ylabel="Best Fitness",
         title="Convergence Curve",
         legend=false)
         display(p)
    end
    @testset "Griewank with WOA_struct, p = 2" verbose = true begin 
        lb = [-600 for i in 1:2]
        ub = [600 for i in 1:2]
        # result = AHA(100, 1500, lb, ub, Griewank)
        problem2 = OptimizationProblem(Griewank, lb, ub, 2)
        result = WOA(problem2, 100, 1500)
        @test isapprox(result.BestX, [0.0 for i in 1:2], atol=1e-5)
        @test isapprox(result.BestF, 0.0, atol=1e-5)
        p = plot(result.convergence_curve,
         xlabel="Iteration",
         ylabel="Best Fitness",
         title="Convergence Curve",
         legend=false)
         display(p)
         savefig(p, "convergence.png")
    end
end 
