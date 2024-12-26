module MetaheuristicsAlgorithms

include("AEFA.jl")
include("AEO.jl")
include("AFT.jl")
include("AHA.jl")
include("ALO.jl")
include("AOArithmetic.jl")
include("APO.jl")
include("ARO.jl")
include("AVOA.jl")
include("BES.jl")
include("BKA.jl")
include("BO.jl")
include("BOA.jl")
include("ChOA.jl")
include("CO.jl")
include("CoatiOA.jl")
include("COOT.jl")
include("CSBO.jl")
include("DBO.jl")
include("DDAO.jl")
include("DMOA.jl")
include("DO.jl")
include("DSO.jl")
include("ECO.jl")
include("EDO.jl")
include("ElkHO.jl")
include("EO.jl")
include("ETO.jl")
include("FLA.jl")
include("FLoodA.jl")
include("FOX.jl")
include("GazelleOA.jl")
include("GBO.jl")
include("GEA.jl")
include("GJO.jl")
include("GKSO.jl")
include("GNDO.jl")
include("GO.jl")
include("GOA.jl")
include("GTO.jl")
include("GWO.jl")
include("HBA.jl")
include("HBO.jl")
include("HGS.jl")
include("HGSO.jl")
include("HHO.jl")
include("HikingOA.jl")
include("HO.jl")
include("HorseOA.jl")
##
include("initialization.jl")
##

##
using .AEFA, .AEO, .AFT, .AHA, .ALO, .AOArithmetic, .APO, .ARO, .AVOA
using .BES, .BKA, .BO, .BOA
using .ChOA, .CO, .CoatiOA, .COOT, .CSBO
using .DBO, .DDAO, .DMOA, .DO, .DSO
using .ECO, .EDO, .ElkHO, .EO, .ETO
using .FLA, .FLoodA, .FOX
using .GazelleOA, .GBO, .GEA, .GGO, .GJO, .GKSO, .GNDO, .GO, .GOA, .GTO, .GWO

# export AEFA,AEO, AFT, AHA, ALO, AOArithmetic, APO, ARO, AVOA
# export BES, BKA, BO, BOA
# export ChOA, CO, CoatiOA, COOT, CSBO
# export DBO, DDAO, DMOA, DO, DSO
# export ECO, EDO, ElkHO, EO, ETO
# export FLA, FLoodA, FOX
# export GazelleOA, GBO, GEA, GGO, GJO, GKSO, GNDO, GO, GOA, GTO, GWO
##
# dim = 30
# Max_iteration = 1000
# SearchAgents_no = 50
# lb = -100
# ub = 100
# tlt = "Chung Reynolds"
# i = 1
##
# BestPosition, BestValue, ConvergenceCurve = FPA(SearchAgents_no, Max_iteration, lb, ub, dim, Chung_Reynolds)
# println("BestValue: ", BestPosition)

end
