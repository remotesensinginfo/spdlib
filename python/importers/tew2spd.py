#!/usr/bin/env python

############################################################################
# tew2spd.py
# spdlib
#
# Created by Mattias Nyström on 15/05/2012.
# Copyright (c) 2012 Mattias Nyström, Swedish University of Agricultural Sciences
#
 ############################################################################


import numpy as np
import array
import struct
import os
import sys
import optparse
import spdpy
from scipy import constants
import datetime
import time
#import math

#Constants:
spdSpeedOfLightM = 299792458
spdSpeedOfLightMns = 0.299792458
spdPiHalf = np.pi/2


class header:
    """Class defining the header of tew-file"""
    
    def __init__(self, headerTuple, headerTuple120=None):
        self.fileSignature = "".join(headerTuple[0:4]) # 4 bytes ("TEWF")    - File signature
        self.versionMajor = headerTuple[4] # 1 bytes                                    - Will be 1 if version is 1.20
        self.versionMinor = headerTuple[5] # 1 bytes                                    - Will be 20 if version is 1.20
        scanDate = datetime.datetime(headerTuple[6], 1, 1) + datetime.timedelta(headerTuple[7] - 1)
        #2 bytes                    - E.g. 2009
        # 2 bytes                   - Julian date of the year, (=Day of the year? E.g. 32 is Feb 1?) TODO
        self.scanYear = scanDate.year
        self.scanMonth = scanDate.month
        self.scanDay = scanDate.day
        self.headerSize = headerTuple[8] # 2 bytes                                 - The total size, in bytes, of the entire header block
        self.offsetToData = headerTuple[9] # 4 bytes                                - Total offset, in bytes, from the beginning of the file to the first data record.
        self.waveformDataFormat = headerTuple[10] # 1 byte (0-255)    - Waveform Data record format. TODO: Only supporting version 4. Check if WaveformDataFormat == 4 and give error otherwise.
        self.sampleLength = headerTuple[11] # 8 bytes                          - The distance, in meters, between the samples in the waveform.
        self.sampleLengthNs = self.sampleLength / spdSpeedOfLightMns #         - Calculated distance in nanoseconds.
        self.noOfWaveformDataRecs = headerTuple[12] # 4 bytes            - Number of waweform data records within the file.
        self.wfDataRecSize = headerTuple[13] # 2 bytes                          - Size of Waveform Data records in bytes. Will be set to 1 for the files I'm using while the data records have variable lengths.
        self.xScaleFactor = headerTuple[14] # 8 bytes                             - Overall offsets and scale factors that should be used to calculate the real coordinates from the relative coordinates in the data records.
        self.yScaleFactor = headerTuple[15] # 8 bytes                             - Calculate as: Ycoord = (Yrel * Yscale) + Yoffset
        self.zScaleFactor = headerTuple[16] # 8 bytes
        self.xOffset = headerTuple[17] # 8 bytes
        self.yOffset = headerTuple[18] # 8 bytes
        self.zOffset = headerTuple[19] # 8 bytes
        self.maxX = headerTuple[20] # 8 bytes                                         - Maximum and minimum coordinate values of the scanner origin.
        self.maxY = headerTuple[21] # 8 bytes
        self.maxZ = headerTuple[22] # 8 bytes
        self.minX = headerTuple[23] # 8 bytes
        self.minY = headerTuple[24] # 8 bytes
        self.minZ = headerTuple[25] # 8 bytes
        self.pulseLengthNs = headerTuple[26] # 1 byte                             - Pulse length of emitted pulse in nm. If the pulse length is unknown or variable it is set to 0.
        self.systemIdentifier = "".join(headerTuple[27:59]) # 32 bytes      - Specific string specifying the generationg system, e.g. "TopEye MKII"
        self.zeroLevel = headerTuple[59] # 4 bytes                                    - Only MkII. Zero level to apply to the waveform amplitudes to calculate intensity values, see Ch 1 scale factor.
        self.channel1ScaleFactor = headerTuple[60] # 4 bytes                   - Only MkII. Scale factor that should be applied the waveform amplitudes to calculate intensity values (I): I = (Awaveform + 127 - ZeroLevel ) * Channel1ScaleFactor
        self.channel2ScaleFactor = headerTuple[61] # 4 bytes
        self.channelSetup = headerTuple[62] # 1 bytes                             - Only MkII. The channel setup of the waveform receiver.
        self.noise = headerTuple[63] # 8 bytes                                          - Not used. Always 0.0.
        
        if(headerTuple120):
            self.wfCh1DistOffset = headerTuple120[0] # 4 bytes                            - Offset to apply to waveform ch 1 dist [m]. See waveform Ch1 data.
            self.wfCh2DistOffset = headerTuple120[1] # 4 bytes                            - Offset to apply to waveform ch 2 dist [m]. See waveform Ch2 data.
            self.ch1CorrectionTable = list(headerTuple120[2:258]) # 512 bytes    - Only MkIII. Correction tables are used as a look-up-table to retrieve the real intensity values.
                                                                                                            #This is used only for MKIII waveform data, for MkII waveform data the Zero Level and Scale Factor is used
                                                                                                            #to calculate intensity values. The intensity, for MkIII waveform, is calculated by indexing the correction table
                                                                                                            #with the waveform amplitude value (0-255). The trigger waveform is from Channel 1 and should use the same
                                                                                                            #correction table from Ch Correction Table.
                                                                                                            #Itr = Ch1CorrectionTable[ ATriggerWaveform ]
                                                                                                            #ICh1 = Ch1CorrectionTable[ ACh1Waveform ]
                                                                                                            #ICh2 = Ch2CorrectionTable[ ACh2Waveform ]
                                                                                                            #where ATriggerWaveform is an amplitude from the Trigger Waveform Data Array. ICh1 and ICh2 are the amplitude
                                                                                                            #values from the Waveform Ch1 and Ch2 data arrays. 
                    
            self.ch2CorrectionTable = list(headerTuple120[258:515]) #512 bytes
        
    
    def summary(self):
        print "\n*****************************************************************"
        print "                     Summary of tew-header"
        print "*****************************************************************"
        print "File signature:", self.fileSignature
        print "Version major:", self.versionMajor
        print "Version minor:", self.versionMinor
        print "Date:", self.scanYear, self.scanMonth, self.scanDay
        print "Header size:", self.headerSize
        print "Offset to data:", self.offsetToData
        print "Waveform data format:", self.waveformDataFormat
        print "Sample length (meter):", self.sampleLength, "(", self.sampleLengthNs, " ns)"
        print "No of waveform data records:", self.noOfWaveformDataRecs
        print "Waveform data record size:", self.wfDataRecSize
        print "X scale factor:", self.xScaleFactor
        print "Y scale factor:", self.yScaleFactor
        print "Z scale factor:", self.zScaleFactor
        print "X offset:", self.xOffset
        print "Y offset:", self.yOffset
        print "Z offset:", self.zOffset
        print "X max:", self.maxX
        print "Y max:", self.maxY
        print "Z max:", self.maxZ
        print "X min:", self.minX
        print "Y min:", self.minY
        print "Z min:", self.minZ
        print "Pulse length of emitted pulse (ns):", self.pulseLengthNs
        print "System identifier:", self.systemIdentifier
        print "Zero level (Mk II):", self.zeroLevel
        print "Channel 1 scale factor (Mk II):", self.channel1ScaleFactor
        print "Channel 2 scale factor (Mk II):", self.channel2ScaleFactor
        print "Channel setup (Mk II):", self.channelSetup
        print "Noise (not used, always 0.0):", self.noise
        
        if (self.versionMinor == 20):
            print "Waveform ch 1 offset (m):", self.wfCh1DistOffset
            print "Waveform ch 2 offset (m):", self.wfCh2DistOffset
            print "Ch 1 correction table size:", len(self.ch1CorrectionTable)
            print "Ch 2 correction table size:", len(self.ch2CorrectionTable)
            print "Ch 1 correction table:", self.ch1CorrectionTable
            print "Ch 2 correction table:", self.ch2CorrectionTable
            
            
class pulse:
    """Class defining a pulse in tew-file"""
    
    def __init__(self, pulse4):
        self.GPSTime = pulse4[0] # 8 bytes                               - GPS Time Stamp
        self.firstEchoRange = pulse4[1] # 4 bytes                         - Offset to first echo, in meters, extracted using the LRF manufacturer's algorithm.
                                                                                    #Calculate coordinate using: PfirstEcho = Pscanner + FirstEchoRange * Voutgoing
                                                                                    #Pscanner is the scanner origin in X,Y and Z.
        self.lastEchoRange = pulse4[2] # 4 bytes                         - Offset to last echo, in meters, extracted using the LRF manufacturer's algorithm.
        self.triggerWfLength = pulse4[3] # 2 bytes       - Size of Trigger Waveform data array.
        self.wfCh1Length = pulse4[4] # 2 b       - Size of waveform on Ch 1 data array.
        self.wfCh2Length = pulse4[5] # 2 b       - Size of waveform on Ch 2 data array.
        self.xyzRelOrigin = pulse4[6:9] # 12 bytes         - Position of scanner mirror in relative coordinates( XYZ direction )
                                                                                    #Real scanner mirror origin coordinates are calculated like this:
                                                                                    #Xorigin = ( XYZRelOrigin[1] * XScale ) + XOffset
        self.outgoingVector = pulse4[9:12] # 12 bytes    - Normalized vector pointing in the direction of the laser beam/waveform. The vector is expressed in X, Y, Z in 10-6 meters.
        self.noOfWfBlocksCh1 = pulse4[12] >> 4 # 0.5 bytes   - xxxx 0000 = Nr of WF blocks Ch1/LoCh
        self.noOfWfBlocksCh2 = pulse4[12] & 0xf # 0.5 bytes   - 0000 xxxx = Nr of WF blocks Ch2/HiCh



    def setTriggerWf(self, blockLength, blockOffset, wf):
        self.triggerWfBlockLength = blockLength # 1 byte                - The number of waveform bytes in this block.
        self.triggerWfBlockOffset = blockOffset # 2 bytes               - The sample number of the first sample in the block.
        self.triggerWf = list(wf) # array with triggerWfBlockLength bytes       - Array with waveform amplitude values
        self.triggerEchoOffset = 13.45 # TODO: Calculate distance to the trigger pulse.


    def setWf1(self, blockLength, blockOffset, wf):
        self.recWf1BlockLength = blockLength # 1 byte                - The number of waveform bytes in this block.
        self.recWf1BlockOffset = blockOffset # 2 bytes               - The sample number of the first sample in the block.
        self.recWf1 = list(wf) # array with recWf1BlockLength bytes   - Array with waveform amplitude values


    def setWf2(self, blockLength, blockOffset, wf):
        self.recWf2BlockLength = blockLength # 1 byte                - The number of waveform bytes in this block.
        self.recWf2BlockOffset = blockOffset # 2 bytes               - The sample number of the first sample in the block.
        self.recWf2 = list(wf) # array with recWf2BlockLength bytes   - Array with waveform amplitude values
        
    #Combine ch1 and ch2:
    def combineRecWf(self, header):
        self.recWf = self.recWf1 #TODO: Temporarily only using Wf1 and implement "combiner" later.
        self.recWfLength = self.recWf1BlockLength # Number of bins.
        self.recWfOffset = self.recWf1BlockOffset # The sample number of the first sample in the wf.
        self.recWfConstOffset = header.wfCh1DistOffset #Constant offset to start of the block (to where blockOffset start to count).


    def summary(self):
        print "\n*****************************************************************"
        print "                     Summary of tew-pulse"
        print "*****************************************************************"

        print "GPS time:", self.GPSTime
        print "First echo range:", self.firstEchoRange
        print "Last echo range:", self.lastEchoRange
        print "Size of trigger waveform data array:", self.triggerWfLength
        print "Size of ch 1 waveform data array:", self.wfCh1Length
        print "Size of ch 2 waveform data array:", self.wfCh2Length
        print "Position of scanner mirror in rel coord:", self.xyzRelOrigin
        print "Normalized outgoing vector:", self.outgoingVector
        
        range = np.sqrt(self.outgoingVector[0]**2 + self.outgoingVector[1]**2 + self.outgoingVector[2]**2)
        print "Length of normalized outgoing vector (calculated):", range
        print "Azimuth (calculated):", np.arctan2(self.outgoingVector[1], self.outgoingVector[0])
        print "Zenith (calculated):", np.arccos(self.outgoingVector[2] / range)
        
        print "Number of waveform blocks ch 1:", self.noOfWfBlocksCh1
        print "Number of waveform blocks ch 2:", self.noOfWfBlocksCh2
        
        print "Trigger Wf, block length:", self.triggerWfBlockLength
        print "Trigger Wf, block offset (bins):", self.triggerWfBlockOffset
        print "Trigger Wf:", self.triggerWf
        
        if(self.noOfWfBlocksCh1>0):
            print "Received Wf1, block length:", self.recWf1BlockLength
            print "Received Wf1, block offset (bins):", self.recWf1BlockOffset
            print "Received Wf1 (corrected):", self.recWf1

        if(self.noOfWfBlocksCh2>0):
            print "Received Wf2, block length:", self.recWf2BlockLength
            print "Received Wf2, block offset (bins):", self.recWf2BlockOffset
            print "Received Wf2 (corrected):", self.recWf2
        




def createSPDPulse(tewHeader, tewPulse, pulseID):
    """
    Create a pulse from TEW-record.
    """
    # Create pulse
    pulse = spdpy.createSPDPulsePy()    
    pulse.pulseID = pulseID
    pulse.wavelength = 1550.0
    pulse.numberOfReturns = 0
    pulse.gpsTime = long(tewPulse.GPSTime*1e9) #Convert to long and store as nanoseconds.
    
    # Add origin coordinates
    pulse.x0 = tewPulse.xyzRelOrigin[0] * tewHeader.xScaleFactor + tewHeader.xOffset
    pulse.y0 = tewPulse.xyzRelOrigin[1] * tewHeader.yScaleFactor + tewHeader.yOffset
    pulse.z0 = tewPulse.xyzRelOrigin[2] * tewHeader.zScaleFactor + tewHeader.zOffset
    range = np.sqrt(tewPulse.outgoingVector[0]**2 + tewPulse.outgoingVector[1]**2 + tewPulse.outgoingVector[2]**2)
    pulse.azimuth = np.arctan2(tewPulse.outgoingVector[1], tewPulse.outgoingVector[0]) #Will have a value between -pi..pi. Seem to be like that in spdCommon.cpp when converting cartesian to spherical coordinates.
    pulse.zenith = np.arccos(tewPulse.outgoingVector[2] / range) + spdPiHalf

    # Received waveform
    pulse.rangeToWaveformStart = (tewPulse.recWfOffset - (tewPulse.triggerWfBlockOffset + tewPulse.triggerEchoOffset)) * tewHeader.sampleLength + tewPulse.recWfConstOffset
    pulse.receiveWaveGain = 1.0
    pulse.receiveWaveOffset = 0.0
    pulse.numOfReceivedBins = tewPulse.recWfLength
    pulse.received = [int(i) for i in tewPulse.recWf]
    
    # Transmitted waveform
    pulse.transWaveGain = 1.0
    pulse.transWaveOffset = 0.0
    pulse.numOfTransmittedBins = tewPulse.triggerWfBlockLength
    pulse.transmitted = [int(i) for i in tewPulse.triggerWf]
    
    if False:
        print "x0: ", pulse.x0
        print "y0: ", pulse.y0
        print "z0: ", pulse.z0
        print "tewPulse.outgoingVector[2] :", tewPulse.outgoingVector[2] 
        print "range:", range
        print "azimuth:", pulse.azimuth
        print "zenith:", pulse.zenith
        print "rangeToWaveformStart:", pulse.rangeToWaveformStart

    
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
    Convert TEW files to SPD
    """
    # Create SPD file
    spdFile = cmdargs.tewFile.replace(".tew",".spd")
    spdObj = spdpy.createSPDFile(spdFile)
    spdWriter = spdpy.SPDPyNoIdxWriter()
    spdWriter.open(spdObj,spdFile)
    
    # Set SPD header values
    spdObj.setTransWaveformDefined(1)
    spdObj.setReceiveWaveformDefined(1)
    spdObj.setOriginDefined(1)

    
    # Read tew-header
    tewObj = open(cmdargs.tewFile, "rb")
    
    recordFormat115 = struct.Struct('=4c B B H H H I B d I h d d d d d d d d d d d d B 32c f f f B d') #= is to have exact size of the data type.

    if (recordFormat115.size != 181):
        sys.stderr.write( "\nWrong size (" + str(recordFormat115.size) + " bytes) of Python defined header for TEW 1.15. Should be 181 bytes. Please check source code.\n" )
        sys.exit()

    headerStr115 = tewObj.read(recordFormat115.size)
    headerTuple115 = recordFormat115.unpack_from(headerStr115, 0)
    
    if (headerTuple115[5] == 20):
        recordFormat120 = struct.Struct('=f f 256H 256H')
        headerStr120 = tewObj.read(recordFormat120.size)
        headerTuple120 = recordFormat120.unpack_from(headerStr120, 0)
        tewHeader = header(headerTuple115, headerTuple120)
    else:
        tewHeader = header(headerTuple115)
    
    tewHeader.summary()
    
    
    # Set SPD header values
    spdObj.setSystemIdentifier(tewHeader.systemIdentifier)
    spdObj.setTemporalBinSpacing(tewHeader.sampleLengthNs)
    spdObj.setYearOfCapture(tewHeader.scanYear)
    spdObj.setMonthOfCapture(tewHeader.scanMonth)
    spdObj.setDayOfCapture(tewHeader.scanDay)
    spdObj.setHourOfCapture(0)
    spdObj.setMinuteOfCapture(0)
    spdObj.setSecondOfCapture(0)
    
    
    spdPulsesInBuffer = 0 #Count number of spdPulses in buffer before saving to file.
    spdPulseBlockSize = 10000 #Number of pulses before saving to file.
    spdPulses = list()
    time0 = time.time() #Timers used to calculate time to finish.
    time1 = 0 #Timers used to calculate time to finish.
   
    for pulseId in xrange(0, 2): #tewHeader.noOfWaveformDataRecs):
        #Pulses
        pulses = list()
        pulsesInBuffer = 0
        
        
        # Read tew-pulse header
        pulse4Format = struct.Struct('=d f f H H H 3l 3l B')
        pulse4str = tewObj.read(pulse4Format.size)
        pulse4 = pulse4Format.unpack_from(pulse4str, 0)
        
        tewPulse = pulse(pulse4)
        
        
        # Read tew trigger pulse header
        triggerHeaderFormat = struct.Struct('=B H') #Block length / Block offset.
        triggerHeaderStr = tewObj.read(triggerHeaderFormat.size)
        triggerHeader = triggerHeaderFormat.unpack_from(triggerHeaderStr, 0)
        
        # Read tew trigger waveform
        triggerWfFormat = array.array('B')
        triggerWfFormat.read(tewObj, triggerHeader[0])
        triggerWf = np.array(triggerWfFormat, dtype=np.uint)
        
        #Apply correction table (should not be done for trigger wf?)
        #triggerWf = np.zeros(len(triggerWfTemp), dtype=np.uint)
        #for j in xrange(0, len(triggerWfTemp)):
        #    triggerWf[j] = tewHeader.ch1CorrectionTable[triggerWfTemp[j ]]
            
        #Store header and pulse for trigger waveform
        tewPulse.setTriggerWf(triggerHeader[0], triggerHeader[1], triggerWf)
        

        for i in xrange(1, tewPulse.noOfWfBlocksCh1+1): #TODO: Need to combine many blocks into one. At the moment, only the last block is stored.
            # Read tew received waveform 1 (low ch) header
            recWf1HeaderFormat = struct.Struct('=B H') #Block length / Block offset.
            recWf1HeaderStr = tewObj.read(recWf1HeaderFormat.size)
            recWf1Header = recWf1HeaderFormat.unpack_from(recWf1HeaderStr, 0)
            
            # Read tew received waveform 1 (low ch = sensitive)
            recWf1Format = array.array('B')
            recWf1Format.read(tewObj, recWf1Header[0])
            recWf1Temp = np.array(recWf1Format, dtype=np.uint)
            
            # Apply correction table
            recWf1 = np.zeros(len(recWf1Temp), dtype=np.uint)
            for j in xrange(0, len(recWf1Temp)):
                recWf1[j] = tewHeader.ch1CorrectionTable[recWf1Temp[j]]
            
            #Store header and pulse for received waveform
            tewPulse.setWf1(recWf1Header[0], recWf1Header[1], recWf1)
        
        
        for i in xrange(1, tewPulse.noOfWfBlocksCh2+1): #TODO: Lägg till nåt som syr ihop fler än 1 till samma paket... och därefter kombinerar Ch1 och Ch2.
            # Read tew received waveform 2 (high ch) header
            recWf2HeaderFormat = struct.Struct('=B H') #Block length / Block offset.
            recWf2HeaderStr = tewObj.read(recWf2HeaderFormat.size)
            recWf2Header = recWf2HeaderFormat.unpack_from(recWf2HeaderStr, 0)
            
            # Read tew received waveform 2 (high ch)
            recWf2Format = array.array('B')
            recWf2Format.read(tewObj, recWf2Header[0])
            recWf2Temp = np.array(recWf2Format, dtype=np.uint)
            
            # Apply correction table
            recWf2 = np.zeros(len(recWf2Temp), dtype=np.uint)
            for j in xrange(0, len(recWf2Temp)):
                recWf2[j] = tewHeader.ch2CorrectionTable[recWf2Temp[j]]

            #Store header and pulse for received waveform
            tewPulse.setWf2(recWf2Header[0], recWf2Header[1], recWf2)
            
        tewPulse.combineRecWf(tewHeader) #Combine ch1 and ch2. TODO: Only take ch 1 at the moment.
        
        tewPulse.summary() #Print summary about the tew-pulse header
        
        
        #####################################
        #Store the waveform as SPDPulse.
        #####################################
        spdPulse = createSPDPulse(tewHeader, tewPulse, pulseId)
        spdPulses.append(spdPulse)
        spdPulsesInBuffer += 1
        
        # Write pulses to file
        if spdPulsesInBuffer >= spdPulseBlockSize:
            writeSPDPulses(spdObj, spdWriter, spdPulses)
            spdPulses = list()
            spdPulsesInBuffer = 0
            if cmdargs.verbose:
                pulsesImported = pulseId + 1
                time1 = time.time()
                estimatedTimeToFinish = ((time1-time0)/spdPulseBlockSize * (tewHeader.noOfWaveformDataRecs-pulsesImported))/60.0
                sys.stdout.write("%i pulses imported" % pulsesImported)
                sys.stdout.write(" (Estimated %0.1f min to finish)\r" % estimatedTimeToFinish)
                sys.stdout.flush()
                time0 = time.time()
                
    #End of for reading pulses.
    
    sys.stdout.write("\n") #New line before exiting compilation.
    print "Pulses still in buffer:", spdPulsesInBuffer
    
    #Write SPDPulses that still are in memory (if any)
    if spdPulsesInBuffer>0:
        writeSPDPulses(spdObj, spdWriter, spdPulses)
        print "Just wrote the last pulses in buffer."
        
 
    # Close files
    tewObj.close()
    spdWriter.close(spdObj)


# Command arguments
class CmdArgs:
  def __init__(self):
    p = optparse.OptionParser()
    p.add_option("-i","--inputFile", dest="tewFile", default=None, help="Input TEW-file (required)")
    p.add_option("-v","--verbose", dest="verbose", default=False, action="store_true", help="Verbose output.")
    (options, args) = p.parse_args()
    self.__dict__.update(options.__dict__)
    
    if (self.tewFile is None):
        p.print_help()
        print "Input filename must be set."
        sys.exit()


# Run the script
if __name__ == "__main__":
    cmdargs = CmdArgs()
    main(cmdargs)
