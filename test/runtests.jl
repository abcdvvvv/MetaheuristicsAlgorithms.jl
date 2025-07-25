using MetaheuristicsAlgorithms
using Test
using Random
using Plots

include("benchfunctions.jl")

# include("testbenchmark.jl")
# #include("testalgo/testaefa.jl")
# include("testalgo/testaeo.jl")
# include("testalgo/testala.jl")
# include("testalgo/testalo.jl")
# #include("testalgo/testaoarithmetic.jl")
# #include("testalgo/testapo.jl")
# #include("testalgo/testaro.jl")
# #include("testalgo/testartemisinino.jl")
# #include("testalgo/testavoa.jl")
# include("testalgo/testBES.jl")
# include("testalgo/testBKA.jl")
# #include("testalgo/testBO.jl")
# include("testalgo/testBOA.jl")
# #include("testalgo/testCapSA.jl")
# #include("testalgo/testCDO.jl")
# #include("testalgo/testChameleonSA.jl")
# include("testalgo/testChOA.jl")
# include("testalgo/testCO.jl")
# #include("testalgo/testCoatiOA.jl")
# include("testalgo/testCOOT")
# #include("testalgo/testCSBO.jl")
# #include("testalgo/testDBO.jl")
# #include("testalgo/testDDAO.jl")
# #include("testalgo/testDMOA.jl")
# #include("testalgo/testDO.jl")
# include("testalgo/testdso.jl")
# include("testalgo/testECO.jl")
# #include("testalgo/testEDO.jl")
# #include("testalgo/testElkHO.jl")
# include("testalgo/testEO.jl")
# #include("testalgo/testESC.jl")
# #include("testalgo/testETO.jl")
# #include("testalgo/testFATA.jl")
# #include("testalgo/testFLA.jl")
# include("testalgo/testFLoodA.jl")
# #include("testalgo/testFOX.jl")
# #include("testalgo/testGazelleOA.jl")
# include("testalgo/testGBO.jl")
# #include("testalgo/testGEA.jl")
# #include("testalgo/testGGO.jl")
# include("testalgo/testGJO.jl")
# #include("testalgo/testGKSO.jl")
# include("testalgo/testGNDO.jl")
# include("testalgo/testGNDO.jl")
# #include("testalgo/testGO.jl")
# #include("testalgo/testGOA.jl")
# #include("testalgo/testGTO.jl")
# #include("testalgo/testGWO.jl")
# #include("testalgo/testHBA.jl")
# #include("testalgo/testHBO.jl")
# #include("testalgo/testHEOA.jl")
# #include("testalgo/testHGS.jl")
# #include("testalgo/testHGSO.jl")
# #include("testalgo/testHHO.jl")
# #include("testalgo/testHO.jl")
# include("testalgo/testHorseOA.jl")
# #include("testalgo/testINFO.jl")
# #include("testalgo/testIVYA.jl")
# #include("testalgo/testJaya.jl")
# #include("testalgo/testJS.jl")
# #include("testalgo/testLCA.jl")
# #include("testalgo/testLFD.jl")
# #include("testalgo/testLPO.jl")
# #include("testalgo/testLSHADE_cnEpSin.jl")
# #include("testalgo/testMossGO.jl")
# #include("testalgo/testMountainGO.jl")
# #include("testalgo/testMPA.jl")
# #include("testalgo/testOOA.jl")
# #include("testalgo/testParrotO.jl")
# #include("testalgo/testPDO.jl")
# #include("testalgo/testPKO.jl")
# #include("testalgo/testPLO.jl")
# #include("testalgo/testPoliticalO.jl")
# #include("testalgo/testPLO.jl")
# #include("testalgo/testPRO.jl")
# #include("testalgo/testPumaO.jl")
# #include("testalgo/testQIO.jl")
# #include("testalgo/testRBMO.jl")
# #include("testalgo/testRFO.jl")
# #include("testalgo/testRIME.jl")
# #include("testalgo/testROA.jl")
# #include("testalgo/testRSA.jl")
# #include("testalgo/testRSO.jl")
# include("testalgo/testRUN.jl")
# #include("testalgo/testSBO.jl")
# #include("testalgo/testSBOA.jl")
# #include("testalgo/testSCA.jl")
# #include("testalgo/testSCHO.jl")
# #include("testalgo/testSeaHO.jl")
# #include("testalgo/testSFOA.jl")
# #include("testalgo/testSHO.jl")
# include("testalgo/testSMA.jl")
# #include("testalgo/testSnowOA.jl")
# #include("testalgo/testSO.jl")
# #include("testalgo/testSOA.jl")
# #include("testalgo/testSparrowSA.jl")
# #include("testalgo/testSSA.jl")
# #include("testalgo/testSTOA.jl")
# #include("testalgo/testSuperbFOA.jl")
# #include("testalgo/testSupplyDO.jl")
# #include("testalgo/testTHRO.jl")
# #include("testalgo/testTLBO.jl")
# #include("testalgo/testTLCO.jl")
# #include("testalgo/testTOC.jl")
# include("testalgo/testTSA.jl")
# #include("testalgo/testTTAO.jl")
# include("testalgo/testWHO.jl")
# #include("testalgo/testWO.jl")
# #include("testalgo/testWOA.jl")
# #include("testalgo/testWOA.jl")
# #include("testalgo/testWUTP.jl")
# #include("testalgo/testYDSE.jl")
include("testalgo/testZOA.jl")



# This tests are disabled due to a possible bug (? maybe) in the AHA and AFT algorithms
# include("testaha.jl")
# include("testalgo/testaft.jl")


###
# include("testalgo/testaeostruct.jl")
# include("testoptimize.jl")
# include("testplot.jl")
# include("testcompare_algorithms.jl")

# include("testwoastruct.jl")



