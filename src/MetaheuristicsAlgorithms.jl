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
include("INFO.jl")
include("IVYA.jl")
include("Jaya.jl")
include("JS.jl")
include("LCA.jl")
include("LFD.jl")
include("LPO.jl")
include("MossGO.jl")
include("MountainGO.jl")
include("MPA.jl")
include("MRFO.jl")
include("MVO.jl")
include("OOA.jl")
include("ParrotO.jl")
include("PDO.jl")
include("PKO.jl")
include("PoliticalO.jl")
include("PRO.jl")
include("PumaO.jl")
include("QIO.jl")
include("ROA.jl")
include("RSA.jl")
include("RSO.jl")
include("RUN.jl")
include("SBO.jl")
include("SBOA.jl")
include("SCHO.jl")
include("SeaHO.jl")
include("SHO.jl")
include("SMA.jl")
include("SnowOA.jl")
include("SO.jl")
include("SparrowSA.jl")
include("SSA.jl")
include("STOA.jl")
include("SupplyDO.jl")
include("TLBO.jl")
include("TLCO.jl")
include("TSA.jl")
include("TTAO.jl")
include("WHO.jl")
include("WO.jl")
include("WOA.jl")
include("WSO.jl")
include("YDSE.jl")
include("ZOA.jl")
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
using .HBA, .HBO, .HGS, .HGSO, .HikingOA, .HO, .HorseOA
using .INFO, .IVYA
using .Jaya, .JS
using .LCA, .LFD, .LPO
using .MossGO, .MountainGO, .MPA, .MRFO, .MVO
using .OOA
using .ParrotO, .PDO, .PKO, .PoliticalO, .PRO, .PumaO
using .QIO
using .ROA, .RSA, .RSO, .RUN
using .SBO, .SBOA, .SCHO, .SeaHO, .SHO, .SMA, .SnowOA, .SO, .SparrowSA, .SSA, .STOA, .SupplyDO
using .TLBO, .TLCO, .TSA, .TTAO
using .WHO, .WO, .WOA, .WSO
using .YDSE
using .ZOA

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
