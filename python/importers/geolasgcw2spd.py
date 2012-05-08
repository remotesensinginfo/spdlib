#!/usr/bin/env python

import numpy as np
import struct
import os
import sys
import optparse
import spdpy
from scipy import constants


def createSPDPulse(data, wfData, pulseID):
    """
    Create a pulse from LGC record
    """
    # Create pulse
    pulse = spdpy.createSPDPulsePy()    
    pulse.pulseID = pulseID
    pulse.wavelength = 1550.0
    pulse.numberOfReturns = 0
    pulse.GPStime = data[1]
    
    # Add coordinates
    pulse.x0 = data[2]
    pulse.y0 = data[3]
    pulse.z0 = data[4]
    magnitude = np.sqrt(data[5]**2 + data[6]**2 + data[7]**2)
    pulse.azimuth = np.arctan((data[5]/magnitude) / (data[6]/magnitude))
    if pulse.azimuth < 0:
        pulse.azimuth += 2 * np.pi
    pulse.zenith = np.arccos(data[7] / magnitude)

    # Received waveform
    rwfData = wfData[data[11]:]
    pulse.rangeToWaveformStart = ((constants.c / 1e9) * data[8]) / 2
    pulse.receiveWaveGain = 1.0
    pulse.receiveWaveOffset = 0.0
    pulse.numOfReceivedBins = data[10]
    pulse.received = [int(i) for i in rwfData]
    
    # Transmitted waveform
    twfData = wfData[:data[11]]
    pulse.transWaveGain = 1.0
    pulse.transWaveOffset = 0.0
    pulse.numOfTransmittedBins = data[11]
    pulse.transmitted = [int(i) for i in twfData]
    
    # Return result
    return pulse


def writeSPDPulses(spdObj, spdWriter, pulses):
    """
    Write a block of pulses
    """
    nPulses = len(pulses)
    try:
        spdWriter.writeData(pulses)
        spdObj.setNumPulses(spdObj.numPulses + nPulses)
    except:
        print "Error writing SPD File."
        sys.exit()


def main(cmdargs):
    """
    Convert the LGW and LGC files to SPD
    """
    # Create SPD file
    spdFile = cmdargs.lwfFile.replace(".lwf",".spd")
    spdObj = spdpy.createSPDFile(spdFile)
    spdWriter = spdpy.SPDPyNoIdxWriter()
    spdWriter.open(spdObj,spdFile)
    spdObj.setTransWaveformDefined(1)
    spdObj.setReceiveWaveformDefined(1)
    spdObj.setOriginDefined(1)
    spdObj.setTemporalBinSpacing(1.0)
    
    # Read binary data
    pulseBlockSize = 1e4
    lwfObj = open(cmdargs.lwfFile, "rb")    
    recordFormat = struct.Struct('Q d d d f f f f H H H B B')
    pulseID = 0
    pulses = list()
    pulsesInBuffer = 0
    with open(cmdargs.lgcFile, "rb") as lgcObj:
        while True:
            try:
                
                # Read pulse record
                dataStr = lgcObj.read(recordFormat.size)
                data = recordFormat.unpack_from(dataStr, 0)
                
                # Read waveform data
                lwfObj.seek(data[0],0)
                nSamples = data[9] + data[10]
                if data[11] > 0:
                    wfStr = lwfObj.read(nSamples * 2)
                    wfFormat = struct.Struct('%iH' % nSamples)
                else:
                    wfStr = lwfObj.read(nSamples)
                    wfFormat = struct.Struct('%iB' % nSamples)
                wfData = wfFormat.unpack_from(wfStr, 0)
                
                # Create the SPD record
                pulseID += 1
                pulse = createSPDPulse(data, wfData, pulseID)
                pulses.append(pulse)
                pulsesInBuffer += 1
                
                # Write pulses to file
                if pulsesInBuffer >= pulseBlockSize:
                    writeSPDPulses(spdObj, spdWriter, pulses)
                    pulses = list()
                    pulsesInBuffer = 0
                    if cmdargs.verbose:
                        sys.stdout.write("%i pulses imported\r" % pulseID)
                        sys.stdout.flush()
                    
            except:
                
                writeSPDPulses(spdObj, spdWriter, pulses)
                sys.stdout.write("%i pulses imported\n" % pulseID)
                sys.stdout.flush()
                break
    
    # Close files
    lwfObj.close()
    spdWriter.close(spdObj)


# Command arguments
class CmdArgs:
  def __init__(self):
    p = optparse.OptionParser()
    p.add_option("-w","--lwfFile", dest="lwfFile", default=None, help="Input LWF file (required).")
    p.add_option("-g","--lgcFile", dest="lgcFile", default=None, help="Input LGC file (required).")
    p.add_option("-v","--verbose", dest="verbose", default=False, action="store_true", help="Verbose output.")
    (options, args) = p.parse_args()
    self.__dict__.update(options.__dict__)
    
    if np.logical_or(self.lwfFile is None, self.lgcFile is None):
        p.print_help()
        print "Input and output filenames must be set."
        sys.exit()


# Run the script
if __name__ == "__main__":
    cmdargs = CmdArgs()
    main(cmdargs)
