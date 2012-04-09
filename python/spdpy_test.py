
from pylab import *
import spdpy

#spdFile = spdpy.openUPDFileHeader("/Users/pete/Desktop/p137_fw_GDA94_110118_mi_decomp_t9_e100_new_upd.upd")
#spdFile = spdpy.openUPDFileHeader("/Users/pete/Desktop/p137_fw_GDA94_110118_mi_decomp_upd.upd")

spdFile = spdpy.openSPDFileHeader("/Users/pete/Desktop/p137_fw_GDA94_110118_mi_decomp_1m_AHD_Test_spd.spd")

print 'Num Pulses = ', spdFile.numPulses
print 'Num points = ', spdFile.numPts

##pulses = spdpy.readUPD(spdFile, 12130, 10)

pulses = spdpy.readSPDPulsesRowCols(spdFile, 100, 0, 500)

speedLightNS = 0.299792458

for pulse in pulses:
    if pulse.numOfReceivedBins > 0 and pulse.numberOfReturns > 0:
        rangeVals = list()
        thresVals = list()
        axisDims = list()
        axisDims.append(pulse.rangeToWaveformStart)
        axisDims.append(pulse.rangeToWaveformStart+(pulse.numOfReceivedBins/speedLightNS))
        axisDims.append(0)
        axisDims.append(650)
        for i in range(len(pulse.received)):
            rangeVals.append((i/speedLightNS)+pulse.rangeToWaveformStart)
            thresVals.append(9)
        plot(rangeVals, pulse.received)
        plot(rangeVals, thresVals, linestyle='dashed', color='grey')
        if pulse.numberOfReturns > 0:
            ptRange = list()
            ptAmp = list()
            for pt in pulse.pts:
                ptRange.append(((pt.waveformOffset/1000)/speedLightNS) + pulse.rangeToWaveformStart)
                if pt.amplitudeReturn > 800:
                    ptAmp.append(799)
                else:
                    ptAmp.append(pt.amplitudeReturn)
            scatter(ptRange, ptAmp, marker='o', color='red')
        axis(axisDims)
        title(str("PulseID: ") + str(pulse.pulseID))
        show()
