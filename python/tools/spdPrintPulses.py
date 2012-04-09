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
# Purpose:  A script to plot waveforms and the associated points
#           for either an SPD to UPD file.
#
# Author: Pete Bunting
# Email: pete.bunting@aber.ac.uk
# Date: 13/03/2011
# Version: 1.0
#
# History:
# 2011/03/13: Version 1.0 - Created.
# 2011/07/21: Version 1.1 - Updated with per pulse gain and offsets.
#
#############################################################################

import os.path
import sys
import spdpy
import math

class SPDPrintPulses (object):

    def printPulse(self, pulse):
        print "Pulse ID: ", pulse.pulseID
        print "Pulse GPS Time: ", pulse.gpsTime
        print "Pulse Index [x,y]: [", pulse.xIdx, ",", pulse.yIdx, "]"
        print "Pulse Origin [x,y,z,h]: [", pulse.x0, ",", pulse.y0, ",", pulse.z0, ",", pulse.h0, "]"
        print "Pulse Azimuth: ", math.degrees(pulse.azimuth)
        print "Pulse Zenith: ", math.degrees(pulse.zenith)
        print "Pulse Noise Thres: ", pulse.waveNoiseThreshold
        print "Pulse Num. Transmitted Bins: ", pulse.numOfTransmittedBins
        print "Transmitted Offset: ", pulse.transWaveOffset
        print "Transmitted Gain: ", pulse.transWaveGain
        print "Pulse Num. Received Bins: ", pulse.numOfReceivedBins
        print "Received Offset: ", pulse.receiveWaveOffset
        print "Received Gain: ", pulse.receiveWaveGain
        if pulse.numOfTransmittedBins > 0:
            print "Transmitted bins: "
            for val in pulse.transmitted:
                print (val*pulse.transWaveGain)+pulse.transWaveOffset, ",",
            #print "\n",
        if pulse.numOfReceivedBins > 0:
            print "Received bins: "
            for val in pulse.received:
                print (val*pulse.receiveWaveGain)+pulse.receiveWaveOffset, ",",
            #print "\n",
        print "Pulse Num Points: ", pulse.numberOfReturns
        if pulse.numberOfReturns > 0:
            for pt in pulse.pts:
                print "Return ID: ", pt.returnID
                print "[x,y,z,h]: [", pt.x, ",", pt.y, ",", pt.z, ",", pt.height, "]"
                print "Amplitude: ", pt.amplitudeReturn
                print "Width: ", pt.widthReturn
                print "Range: ", pt.range
                print "Class: ", pt.classification
                print "Wave Index: ", pt.wavePacketDescIdx
                print "Wave Offset: ", pt.waveformOffset
        print ""

    def printSPDFilePulses(self, inputFile, row, startCol, endCol):
        print "SPD File: ", inputFile
        spdFile = spdpy.openSPDFileHeader(inputFile)
        print "Number of SPD Pulse: ", spdFile.numPulses
        print "Number of SPD Point: ", spdFile.numPts
        print "Index Size: ", spdFile.numBinsX, " x ", spdFile.numBinsY
        pulses = spdpy.readSPDPulsesRowCols(spdFile, row, startCol, endCol)
        print "Extracted ", len(pulses), " pulses."
        for pulse in pulses:
            self.printPulse(pulse)


    def printUPDFilePulses(self, inputFile, startPulse, numPulses):
        print "UPD File: ", inputFile
        spdFile = spdpy.openUPDFileHeader(inputFile)
        print "Number of SPD Pulse: ", spdFile.numPulses
        print "Number of SPD Point: ", spdFile.numPts
        pulses = spdpy.readUPD(spdFile, startPulse, numPulses)
        print "Extracted ", len(pulses), " pulses."
        for pulse in pulses:
            self.printPulse(pulse)


    def run(self):
        numArgs = len(sys.argv)
        if numArgs > 3:
            filetype = sys.argv[1].strip()
            inputFile = sys.argv[2].strip()
            if filetype == "SPD":
                if numArgs != 6:
                    self.help()
                    exit()
                row = int(sys.argv[3].strip())
                startCol = int(sys.argv[4].strip())
                endCol = int(sys.argv[5].strip())
                self.printSPDFilePulses(inputFile, row, startCol, endCol)
            elif filetype == "UPD":
                if numArgs != 5:
                    self.help()
                    exit()
                startPulse = int(sys.argv[3].strip())
                numPulses = int(sys.argv[4].strip())
                self.printUPDFilePulses(inputFile, startPulse, numPulses)
            else:
                self.help()
        else:
            self.help()

    def help(self):
        print 'spdPrintPulses.py script prints the pulses from SPD/UPD files '
        print ''
        print 'Usage: python spdPlotPulses.py <SPD/UPD> <INPUT>'
        print '       [IF UPD <START PULSE> <NUM PULSES>]'
        print '       [IF SPD <ROW> <START COL> <END COL>]'
        print '\t<UPD/SPD> - Select whether input file is either UPD or SPD'
        print '\t<INPUT> - input SPD or UPD file'
        print '\nThis script was distributed with version 1.0.0 of SPDLib\'s'
        print 'python bindings.'
        print 'For maintainance email pete.bunting@aber.ac.uk'

if __name__ == '__main__':
    obj = SPDPrintPulses()
    obj.run()
