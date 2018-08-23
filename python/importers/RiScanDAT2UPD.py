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
# Purpose:  Write the standard Riegl RiScan Pro ASCII export file format to UPD 
#
# Author: John Armston
# Email: j.armston@uq.edu.au
# Date: 23/11/2011
# Version: 1.0
#
# History:
# 2011/11/23: Version 1.0 - Created as an example for writing a Riegl ASCII
#                           dataset (exported from RIScan) to UPD.
#
#############################################################################

""" Module to read Riegl ASCII data and convert to UPD """

from numpy import *
import os, sys, optparse
import spdpy


def parseLine(header,line):
    
    asciiData = {}
    lparts = line.split(',')
    for i,item in enumerate(header):
        itemParts = item.split("[")
        if (item == "ID"):
            asciiData[item] = int(lparts[i])
        else:
            asciiData[itemParts[0]] = float(lparts[i])    
    return asciiData


def main(cmdargs):
    
    # Open UPD file 
    spdOutFile = spdpy.createSPDFile(cmdargs.outputFile);
    updWriter = spdpy.UPDWriter()
    updWriter.open(spdOutFile,cmdargs.outputFile)
    outPulses = list()

    # Define contents this lidar dataset
    spdOutFile.setReceiveWaveformDefined(0)
    spdOutFile.setTransWaveformDefined(0)
    spdOutFile.setDecomposedPtDefined(1)
    spdOutFile.setDiscretePtDefined(0)
    spdOutFile.setOriginDefined(1)

    # For the RIEGL ASCII format, we do not know what pulse each return
    # is linked to so each return is assigned to a unique pulse and we
    # set this attribute to TRUE so users know the return number are synthetic
    spdOutFile.setReturnNumsSynGen(1)

    # Define scanner properties
    spdOutFile.setSensorHeight(cmdargs.agh)
    spdOutFile.setWavelength(1550.0)

    # Open ASCII file and read header
    asciiObj = open(cmdargs.inputFile, 'r')
    header = asciiObj.readline().strip('\r\n').split(',')
    nPoints = int(asciiObj.readline().strip('\r\n'))
    spdOutFile.setNumPulses(nPoints)
    spdOutFile.setNumPts(nPoints)
    if (header.count("Red") > 0):
        spdOutFile.setRgbDefined(1)

    # Loop through each line of data
    outPulses = list()
    pulsesInBuffer = 0
    for i in range(nPoints):
        
        # Read and parse line
        line = asciiObj.readline().strip('\r\n')
        asciiData = parseLine(header,line)
        
        # Create pulse and add attributes
        pulse = spdpy.createSPDPulsePy()    
        pulse.pulseID = asciiData["ID"] 
        pulse.x0 = cmdargs.x0 
        pulse.y0 = cmdargs.y0
        pulse.z0 = cmdargs.z0    
        pulse.azimuth = asciiData["Phi"]
        pulse.zenith = asciiData["Theta"]

        # Create point and add attributes
        point = spdpy.createSPDPointPy()
        point.returnID = 0
        point.x = asciiData["X"]
        point.y = asciiData["Y"]
        point.z = asciiData["Z"]
        pulse.xIdx = asciiData["X"]
        pulse.yIdx = asciiData["Y"]
        point.range = asciiData["Range"]
        point.amplitudeReturn = asciiData["Amplitude"]
        #if asciiData.has_key("Reflectance"):
        #    point.user = asciiData["Reflectance"]
        if asciiData.has_key("Deviation"):
            point.widthReturn = asciiData["Deviation"]
        if asciiData.has_key("Red"):
            point.red = asciiData["Red"]
            point.green = asciiData["Green"]
            point.blue = asciiData["Blue"]   
            
        # Add point data to the pulse
        pulse.pts.append(point)
        pulse.numberOfReturns = 1
        pulse.pts_start_idx = i    
        
        # Update global statistics
        if i > 0:
            spdOutFile.setXMin(min(spdOutFile.xMin,asciiData["X"]))
            spdOutFile.setXMax(max(spdOutFile.xMax,asciiData["X"]))
            spdOutFile.setYMin(min(spdOutFile.yMin,asciiData["Y"]))
            spdOutFile.setYMax(max(spdOutFile.yMax,asciiData["Y"]))   
            spdOutFile.setZMin(min(spdOutFile.zMin,asciiData["Z"]))
            spdOutFile.setZMax(max(spdOutFile.zMax,asciiData["Z"])) 
            spdOutFile.setZenithMin(min(spdOutFile.zenithMin,asciiData["Theta"]))
            spdOutFile.setZenithMax(max(spdOutFile.zenithMax,asciiData["Theta"])) 
            spdOutFile.setAzimuthMin(min(spdOutFile.azimuthMin,asciiData["Phi"]))
            spdOutFile.setAzimuthMax(max(spdOutFile.azimuthMax,asciiData["Phi"])) 
            spdOutFile.setRangeMin(min(spdOutFile.rangeMin,asciiData["Range"]))
            spdOutFile.setRangeMax(max(spdOutFile.rangeMax,asciiData["Range"]))
        else:
            spdOutFile.setXMin(asciiData["X"])
            spdOutFile.setXMax(asciiData["X"])
            spdOutFile.setYMin(asciiData["Y"])
            spdOutFile.setYMax(asciiData["Y"])   
            spdOutFile.setZMin(asciiData["Z"])
            spdOutFile.setZMax(asciiData["Z"]) 
            spdOutFile.setZenithMin(asciiData["Theta"])
            spdOutFile.setZenithMax(asciiData["Theta"]) 
            spdOutFile.setAzimuthMin(asciiData["Phi"])
            spdOutFile.setAzimuthMax(asciiData["Phi"]) 
            spdOutFile.setRangeMin(asciiData["Range"])
            spdOutFile.setRangeMax(asciiData["Range"])            
        
        # If buffer size is reached, write out pulses
        outPulses.append(pulse)
        pulsesInBuffer += 1
        if (pulsesInBuffer == spdOutFile.pulseBlockSize or i == (nPoints-1)):
            try:
                updWriter.writeData(outPulses)
            except:
                raise IOError, "Error writing UPD File."    
            pulsesInBuffer = 0
            outPulses = list()
            
        # Let's monitor progress
        sys.stdout.write("Writing UPD file %s (%i%%)\r" % (cmdargs.outputFile, int((i+1) / float(nPoints) * 100.0)))

    # Close the input and output files
    asciiObj.close()
    updWriter.close(spdOutFile)


# Command arguments
class CmdArgs:
  def __init__(self):
    p = optparse.OptionParser()
    p.add_option("-i","--inputFile", dest="inputFile", default=None, help="Input file (RiScan Pro *.dat file).")
    p.add_option("-o","--outputFile", dest="outputFile", default=None, help="Output UPD file (*.upd).")
    p.add_option("--agh", dest="agh", type="float", default=1.6, help='Above ground height of sensor optical centre.')
    p.add_option("--x0", dest="x0", type="float", default=0.0, help='Sensor optical centre X.')
    p.add_option("--y0", dest="y0", type="float", default=0.0, help='Sensor optical centre Y.')
    p.add_option("--z0", dest="z0", type="float", default=0.0, help='Sensor optical centre Z.')
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
