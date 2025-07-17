@testset "LSHADE_cnEpSin" verbose = true begin
    @testset "Ackley with LSHADE_cnEpSin, p = 2" verbose = true begin
        lb = [-32.768 for i = 1:2]
        ub = [32.768 for i = 1:2]
        result = LSHADE_cnEpSin(Ackley, lb, ub, 100, 1000)
        @test isapprox(result.bestX, [0.0 for i = 1:2], atol=1e-5)
        @test isapprox(result.bestF, 0.0, atol=1e-5)
    end
    @testset "Ackley with LSHADE_cnEpSin, p = 5" verbose = true begin
        lb = [-32.768 for i = 1:5]
        ub = [32.768 for i = 1:5]
        result = LSHADE_cnEpSin(Ackley, lb, ub, 100, 1000)
        @test isapprox(result.bestX, [0.0 for i = 1:5], atol=1e-5)
        @test isapprox(result.bestF, 0.0, atol=1e-5)
    end
    @testset "Griewank with LSHADE_cnEpSin, p = 2" verbose = true begin
        lb = [-600.0 for i = 1:2]
        ub = [600.0 for i = 1:2]
        result = LSHADE_cnEpSin(Griewank, lb, ub, 100, 1000)
        @test isapprox(result.bestX, [0.0 for i = 1:2], atol=1e-5)
        @test isapprox(result.bestF, 0.0, atol=1e-5)
    end
end

function updateArchive(archive_pop, archive_funvalues, archive_NP, pop, funvalue)
    # If archive size is zero, return original archive
    if archive_NP == 0
        return archive_pop, archive_funvalues
    end

    if size(pop, 1) != size(funvalue, 1)
        error("check it")
    end

    # Combine archive and new solutions
    popAll = vcat(archive_pop, pop)
    funvalues = vcat(archive_funvalues, funvalue)

    # Remove duplicate rows
    _, unique_idx = unique(popAll, dims=1, return_inds=true)
    popAll = popAll[sort(unique_idx), :]
    funvalues = funvalues[sort(unique_idx), :]

    if size(popAll, 1) <= archive_NP
        return popAll, funvalues
    else
        idx = randperm(size(popAll, 1))[1:archive_NP]
        return popAll[idx, :], funvalues[idx, :]
    end
end


function gnR1R2(NP1, NP2, r0)
    NP0 = length(r0)
    r1 = rand(1:NP1, NP0)

    for _ in 1:1000
        pos = r1 .== r0
        if all(!pos)
            break
        end
        r1[pos] = rand(1:NP1, sum(pos))
    end

    if any(r1 .== r0)
        error("Cannot generate r1 in 1000 iterations")
    end

    r2 = rand(1:NP2, NP0)

    for _ in 1:1000
        pos = (r2 .== r1) .| (r2 .== r0)
        if all(!pos)
            break
        end
        r2[pos] = rand(1:NP2, sum(pos))
    end

    if any((r2 .== r1) .| (r2 .== r0))
        error("Cannot generate r2 in 1000 iterations")
    end

    return r1, r2
end

