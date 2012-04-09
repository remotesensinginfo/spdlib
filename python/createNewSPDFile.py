import spdpy

spdFileOut = spdpy.createSPDFile("test.spd")

spdFileOut.setNumBinsX(20)
spdFileOut.setNumBinsY(20)
spdFileOut.setBinSize(1)

spdWriter = spdpy.SPDPySeqWriter()
spdWriter.open(spdFileOut, "test.spd")


pulses = list()
for row in range(spdFileOut.numBinsY):
    print "Row:", str(row)
    for col in range(spdFileOut.numBinsX):
        spdWriter.writeDataColumn(pulses, col, row)

spdWriter.close(spdFileOut)
