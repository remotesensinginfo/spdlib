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
# Purpose:  Module to read ASCII waveforms (line 1 transmitted, line 2 recieved) and convert to UPD
#
# Author: Peter Bunting
# Email: petebunting@mac.com
# Date: 28/11/2011
# Version: 1.0
#
# History:
# 2011/11/28: Version 1.0 - Created
#
#############################################################################

""" Module to read ASCII waveforms (line 1 transmitted, line 2 recieved) and convert to UPD """

from numpy import *
import os, sys, optparse
import spdpy


def parseLine(line):
    waveform = []
    commaSplit = line.split(',', line.count(','))
    for token in commaSplit:
        waveform.append(int(token.strip()))
    return waveform

def parseFileToList(file):
    asciiObj = open(file, 'r')
    outValues = []
    for eachLine in asciiObj:
        commaSplit = eachLine.split(',', eachLine.count(','))
        if len(commaSplit) > 1:
            outValues.append(int(commaSplit[1]))
    asciiObj.close()
    return outValues

def main(cmdargs):

    # Open UPD file
    spdOutFile = spdpy.createSPDFile(cmdargs.outputFile);
    updWriter = spdpy.UPDWriter()
    updWriter.open(spdOutFile,cmdargs.outputFile)
    outPulses = list()

    # Define contents this lidar dataset
    spdOutFile.setReceiveWaveformDefined(1)
    spdOutFile.setTransWaveformDefined(1)
    spdOutFile.setDecomposedPtDefined(0)
    spdOutFile.setDiscretePtDefined(0)
    spdOutFile.setOriginDefined(1)

    # Define scanner properties
    spdOutFile.setWavelength(1550.0)
    spdOutFile.setTemporalBinSpacing(1)

    # Open ASCII file and read header
    asciiObj = open(cmdargs.inputFile, 'r')

    transmittedWaveform = []
    receivedWaveform = []

    i = 0
    for eachLine in asciiObj:
        if i == 0:
            transmittedWaveform = parseLine(eachLine)
        elif i == 1:
            receivedWaveform = parseLine(eachLine)
        i = i + 1

    linearVals = parseFileToList(cmdargs.linearFile)
    delogVals = parseFileToList(cmdargs.delogFile)

    transmittedWaveformScale = []
    for val in transmittedWaveform:
        transmittedWaveformScale.append(delogVals[val])

    receivedWaveformScale = []
    for val in receivedWaveform:
        receivedWaveformScale.append(linearVals[val])

    pulse = spdpy.createSPDPulsePy()
    pulse.pulseID = 0
    pulse.x0 = 0
    pulse.y0 = 0
    pulse.z0 = 0
    pulse.azimuth = 0
    pulse.zenith = 0
    pulse.numOfTransmittedBins = len(transmittedWaveformScale)
    pulse.transmitted = transmittedWaveformScale
    pulse.transWaveGain = 1
    pulse.transWaveOffset = 0
    pulse.numOfReceivedBins = len(receivedWaveformScale)
    pulse.received = receivedWaveformScale
    pulse.receiveWaveGain = 1
    pulse.receiveWaveOffset = 0
    pulse.waveNoiseThreshold = 9

    outPulses.append(pulse)
    updWriter.writeData(outPulses)
    updWriter.close(spdOutFile)


# Command arguments
class CmdArgs:
  def __init__(self):
    p = optparse.OptionParser()
    p.add_option("-i","--inputFile", dest="inputFile", default=None, help="Input file (RiScan Pro *.dat file).")
    p.add_option("-o","--outputFile", dest="outputFile", default=None, help="Output UPD file (*.upd).")
    p.add_option("-l","--linear", dest="linearFile", default=None, help="Linear File Value.")
    p.add_option("-d","--delog", dest="delogFile", default=None, help="Delog File Value.")
    (options, args) = p.parse_args()
    self.__dict__.update(options.__dict__)

    if logical_or(self.inputFile is None, self.outputFile is None):
        p.print_help()
        print "Input and output filenames must be set."
        sys.exit()


# Run the script
if __name__ == "__main__":
    cmdargs = CmdArgs()
    main(cmdargs)


