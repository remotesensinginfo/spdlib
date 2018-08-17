#! /usr/bin/env python

############################################################################
# Copyright (c) 2011 Dr. Peter Bunting, Aberystwyth University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
#
#
# Author: Pete Bunting
# Email: pete.bunting@aber.ac.uk
# Date: 13/03/2011
# Version: 1.0
#
# History:
# 2011/03/13: Version 1.0 - Created.
# 2011/07/19: Plots recontructed waveform instead of just gaussian peak. John Armston.
#
#############################################################################

import os.path
import sys
import spdpy
from numpy import *
from pylab import *
import matplotlib.pyplot as plt

speedLightNS = 0.299792458

class SPDPlotPulses (object):

    def plotReceivedPulse(self, pulse, ampAxis, temporalBinSpacing):
        rangeVals = list()
        timeVals = list()
        thresVals = list()
        ampVals = list()
        axisDims = list()
        axisDims.append(0)
        axisDims.append(int(ampAxis))
        axisDims.append(pulse.rangeToWaveformStart+((pulse.numOfReceivedBins*temporalBinSpacing)*speedLightNS)/2)
        axisDims.append(pulse.rangeToWaveformStart)
        for i in range(len(pulse.received)):
            rangeVals.append(((i*temporalBinSpacing)*speedLightNS)/2+pulse.rangeToWaveformStart)
            timeVals.append(i*temporalBinSpacing)
            thresVals.append((pulse.receiveWaveNoiseThreshold*pulse.receiveWaveGain)+pulse.receiveWaveOffset)
            ampVals.append(((pulse.received[i])*pulse.receiveWaveGain)+pulse.receiveWaveOffset)
        plot(ampVals, rangeVals)
        plot(thresVals, rangeVals, linestyle='dashed', color='grey')
        if pulse.numberOfReturns > 0:
            ptRange = list()
            ptAmp = list()
            predVals = zeros(len(pulse.received))
            for pt in pulse.pts:
                ptRange.append(((pt.waveformOffset/1000.0)*speedLightNS)/2.0 + pulse.rangeToWaveformStart)
                ptWSD = (pt.widthReturn / (2.0*sqrt(2.0*log(2.0))) / 10.0)
                rangeOffset = array(timeVals) - pt.waveformOffset/1000.0
                predVals = predVals + (((pt.amplitudeReturn*pulse.receiveWaveGain)+pulse.receiveWaveOffset) * exp(-(rangeOffset**2 / (2.0*ptWSD**2))))
                ptAmp.append(((pt.amplitudeReturn + pulse.receiveWaveNoiseThreshold)*pulse.receiveWaveGain)+pulse.receiveWaveOffset)
            predVals = predVals + ((pulse.receiveWaveNoiseThreshold*pulse.receiveWaveGain)+pulse.receiveWaveOffset)
            scatter(ptAmp, ptRange, marker='o', color='red')
            plot(predVals, rangeVals, color='red')
        axis(axisDims)
        title(str("PulseID: ") + str(pulse.pulseID))
        xlabel("Received Signal")
        ylabel("Range from Sensor (m)")
        show()

    def plotTransmittedPulse(self, pulse, ampAxis, temporalBinSpacing):
        timeVals = list()
        gaussianVals = zeros(len(pulse.transmitted))
        ampVals = list()
        axisDims = list()
        axisDims.append(0)
        axisDims.append(pulse.numOfTransmittedBins*temporalBinSpacing)
        axisDims.append(0)
        axisDims.append(int(ampAxis))
        maxIdx = 0
        for i in range(len(pulse.transmitted)):
            if i == 0:
                maxIdx = i
            elif pulse.transmitted[i] > pulse.transmitted[maxIdx]:
                maxIdx = i

        for i in range(len(pulse.transmitted)):
            timeVals.append(i*temporalBinSpacing)
            ampVals.append(((pulse.transmitted[i])*pulse.transWaveGain)+pulse.transWaveOffset)
            ptWSD = (pulse.widthPulse / (2.0*sqrt(2.0*log(2.0))) / 10.0)
            gaussianVals[i] = (((pulse.amplitudePulse*pulse.transWaveGain)+pulse.transWaveOffset) * exp(-(((i-maxIdx)*temporalBinSpacing)**2 / (2.0*ptWSD**2))))
        plot(timeVals, ampVals, color='blue')
        plot(timeVals, gaussianVals, color='red')
        axis(axisDims)
        title(str("PulseID: ") + str(pulse.pulseID))
        xlabel("Time (ns)")
        ylabel("Transmitted Signal")
        show()

    def plotSPDFilePulses(self, inputFile, ampAxis, row, startCol, endCol):
        print("SPD File: " + inputFile)
        spdFile = spdpy.openSPDFileHeader(inputFile)
        print("Number of SPD Pulse: " + str(spdFile.numPulses))
        print("Number of SPD Point: " + str(spdFile.numPts))
        print("Index Size: " + str(spdFile.numBinsX) + " x " + str(spdFile.numBinsY))
        pulses = spdpy.readSPDPulsesRowCols(spdFile, row, startCol, endCol)
        print("Extracted " + str(len(pulses)) + " pulses.")
        for pulse in pulses:
            if pulse.numOfReceivedBins > 0 and pulse.numberOfReturns >= 0:
                self.plotReceivedPulse(pulse, ampAxis, spdFile.temporalBinSpacing)

    def plotUPDFilePulses(self, inputFile, ampAxis, startPulse, numPulses):
        print("UPD File: " + inputFile)
        spdFile = spdpy.openSPDFileHeader(inputFile)
        print("Number of SPD Pulse: " + str(spdFile.numPulses))
        print("Number of SPD Point: " + str(spdFile.numPts))
        pulses = spdpy.readSPDPulsesOffset(spdFile, startPulse, numPulses)
        print("Extracted " + str(len(pulses)) + " pulses.")
        for pulse in pulses:
            if pulse.numOfReceivedBins > 0 and pulse.numberOfReturns >= 0:
                self.plotReceivedPulse(pulse, ampAxis, spdFile.temporalBinSpacing)

    def plotSPDFileTransPulses(self, inputFile, ampAxis, row, startCol, endCol):
        print("SPD File: " + inputFile)
        spdFile = spdpy.openSPDFileHeader(inputFile)
        print("Number of SPD Pulse: " + str(spdFile.numPulses))
        print("Number of SPD Point: " + str(spdFile.numPts))
        print("Index Size: " + str(spdFile.numBinsX) + " x " + str(spdFile.numBinsY))
        pulses = spdpy.readSPDPulsesRowCols(spdFile, row, startCol, endCol)
        print("Extracted " + str(len(pulses)) + " pulses.")
        for pulse in pulses:
            if pulse.numOfTransmittedBins > 0:
                self.plotTransmittedPulse(pulse, ampAxis, spdFile.temporalBinSpacing)

    def plotUPDFileTransPulses(self, inputFile, ampAxis, startPulse, numPulses):
        print("UPD File: " + inputFile)
        spdFile = spdpy.openSPDFileHeader(inputFile)
        print("Number of SPD Pulse: " + str(spdFile.numPulses))
        print("Number of SPD Point: " + str(spdFile.numPts))
        pulses = spdpy.readSPDPulsesOffset(spdFile, startPulse, numPulses)
        print("Extracted " + str(len(pulses)) + " pulses.")
        for pulse in pulses:
            if pulse.numOfTransmittedBins > 0:
                self.plotTransmittedPulse(pulse, ampAxis, spdFile.temporalBinSpacing)

    def run(self):
        numArgs = len(sys.argv)
        if numArgs > 4:
            filetype = sys.argv[1].strip()
            inputFile = sys.argv[2].strip()
            ampAxis = sys.argv[3].strip()
            if filetype == "SPD":
                if numArgs == 7:
                    row = int(sys.argv[4].strip())
                    startCol = int(sys.argv[5].strip())
                    endCol = int(sys.argv[6].strip())
                    self.plotSPDFilePulses(inputFile, ampAxis, row, startCol, endCol)
                elif numArgs == 8:
                    row = int(sys.argv[4].strip())
                    startCol = int(sys.argv[5].strip())
                    endCol = int(sys.argv[6].strip())
                    if sys.argv[7] == '--trans':
                        self.plotSPDFileTransPulses(inputFile, ampAxis, row, startCol, endCol)
                else:
                    self.help()
                    exit()
            elif filetype == "UPD":
                if numArgs == 6:
                    startPulse = int(sys.argv[4].strip())
                    numPulses = int(sys.argv[5].strip())
                    self.plotUPDFilePulses(inputFile, ampAxis, startPulse, numPulses)
                elif numArgs == 7:
                    startPulse = int(sys.argv[4].strip())
                    numPulses = int(sys.argv[5].strip())
                    if sys.argv[7] == '--trans':
                        self.plotUPDFileTransPulses(inputFile, ampAxis, startPulse, numPulses)
                else:
                    self.help()
                    exit()
            else:
                self.help()
        else:
            self.help()

    def help(self):
        print('spdPlotPulses.py script generates plots from SPD/UPD files ')
        print('')
        print('Usage: python spdPlotPulses.py <SPD/UPD> <INPUT> <AMPLITUDE AXIS MAX=MAX()>')
        print('       [IF UPD <START PULSE> <NUM PULSES>] [--trans]')
        print('       [IF SPD <ROW> <START COL> <END COL>] [--trans]')
        print('\t<UPD/SPD> - Select whether input file is either UPD or SPD')
        print('\t<INPUT> - input SPD or UPD file')
        print('\t<THRESHOLD> - noise threshold for the waveform (plotted as a line)')
        print('\t --trans - plot the transmitted pulse')
        print('\t<AMP AXIS MAX> - The maximum value of the amplitude axis. Default is maximum value.')
        print('\nThis script was distributed with version 1.0.0 of SPDLib\'s')
        print('python bindings.')
        print('For maintainance email petebunting@mac.com')

if __name__ == '__main__':
    obj = SPDPlotPulses()
    obj.run()
