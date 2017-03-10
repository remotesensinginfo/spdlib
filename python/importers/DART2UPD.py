#! /usr/bin/env python
# -*- coding: utf-8 -*-

############################################################################
#This file convert the current dart output file into a UPD file which can be implemented in SPDlib
############################################################################

import re
import sys
import os
import os.path
import struct
import math
import warnings
import spdpy
from datetime import date
import argparse

speedOfLightPerNS=0.299792458

hearder_length=90            # 50+12+28
waveform_parameter_length=104 #

hearder_format="=50s2I4?2d3I"
waveform_parameter_format="=11d4I"
to8bit_format="=b"

#Predefined Parameters:
#These parameters are quite common in LIDAR data, but they are not recorded in simulated DART data, so they are pre-fixed
#scanDirectionFlag=1
#edgeFlightLineFlag=0
#gpsTime=100
#receiveWaveOffset = 0   #volt offset




#Start of the class:
class DART2UPD (object): 

    
    def __init__(self):
      self.ifFixedGain = False
      self.fixedGain = 0.0
      self.receiveWaveOffset = 0
      self.waveNoiseThreshold = 17
      self.maxOutput = 125   #in order to play a role like automatic gain control
      self.ifSolarNoise = False
      self.snFile = 'solar_noise.txt'
      self.snMap = [[0.0]]
      self.ifOutputImagePerBin = False
      self.imagePerBinPath = '~/'
      self.ifOutputNbPhotonPerPulse = False
      self.txtNbPhotonPerPulse = 'nbPhotonPerPulse.txt'
      self.txtNbPhotonMap = [[0.0]]
      self.containerImagePerBin = [[[0.0]]]
      
    def readSolarNoiseFile(self):
      print('reading solar noise file: ',self.snFile)
      with open(self.snFile,'r') as f:
        for row in f.readlines():
          content=row.split()
#          print(int(content[0]), int(content[1]), float(content[5]))
          self.addToSnMap(int(content[0]), int(content[1]), float(content[5]))

    def addToSnMap(self, indX, indY, snValue):
      if len(self.snMap)<indX+1:
        for i in range(len(self.snMap),indX+1):
          self.snMap.append([0.0])
      if len(self.snMap[indX])<indY+1:
        for i in range(len(self.snMap[indX]),indY+1):
          self.snMap[indX].append([0.0])
      self.snMap[indX][indY]=snValue

    def addToNbPhotonPerPulseMap(self, indX, indY, nbPhoton):
      if len(self.txtNbPhotonMap)<indX+1:
        for i in range(len(self.txtNbPhotonMap),indX+1):
          self.txtNbPhotonMap.append([0.0])
      if len(self.txtNbPhotonMap[indX])<indY+1:
        for i in range(len(self.txtNbPhotonMap[indX]),indY+1):
          self.txtNbPhotonMap[indX].append([0.0])
      self.txtNbPhotonMap[indX][indY]=nbPhoton
      
    def addToContainerImagePerBin(self, indBin, indX, indY, pixelV):
      if len(self.containerImagePerBin) < indBin+1:
        for i in range(len(self.containerImagePerBin),indBin+1):
          self.containerImagePerBin.append([[0.0]])
      if len(self.containerImagePerBin[indBin]) < indX+1:
        for i in range(len(self.containerImagePerBin[indBin]), indX+1):
          self.containerImagePerBin[indBin].append([0.0])
      if len(self.containerImagePerBin[indBin][indX]) < indY+1:
        for i in range(len(self.containerImagePerBin[indBin][indX]),indY+1):
          self.containerImagePerBin[indBin][indX].append(0.0)
      self.containerImagePerBin[indBin][indX][indY]=pixelV
          
    def writeImagesPerBin(self):
      
      try:
        os.stat(self.imagePerBinPath)
      except:
        os.mkdir(self.imagePerBinPath)
      
      for i in range(len(self.containerImagePerBin)):
        
        sumCnt=0
        for j in range(len(self.containerImagePerBin[i])):
          for k in range(len(self.containerImagePerBin[i][j])):
            sumCnt += self.containerImagePerBin[i][j][k]
            
        if sumCnt != 0:
          outputFileName=self.imagePerBinPath + 'lidar' + str(i) + '.txt'
          with open(outputFileName,'w') as f:
            print('Open file '+outputFileName+' for writing the lidar intensity perbin'+ "\n")
            f.writelines(' '.join(str(k) for k in j) + '\n' for j in self.containerImagePerBin[i])
          
          
    def readDARTLidarImageBinaryFile(self, dartFileName, updFileName):
        
        if self.ifSolarNoise:
          self.readSolarNoiseFile()
      
        spdOutFile = spdpy.createSPDFile(updFileName);
        updWriter = spdpy.SPDPyNoIdxWriter()
        updWriter.open(spdOutFile, updFileName)

        dartfile = open(dartFileName, "rb")

        try:
            hearder_record = dartfile.read(hearder_length)
        except:
            tb = sys.exc_info()[2]
            print("Reading failed on input DART file " + dartFileName + "\n")
            sys.exit(1)
        #end try
        
        header_data=struct.unpack(hearder_format,hearder_record);
        print("Input Data Information:")
        print("  Version: ", header_data[0][0:42])
        
        #Parameters about the locations of the waveform record
        ifFormatFloat=header_data[3]
        ifExistNonConv=header_data[4]
        ifExistFirstOrder=header_data[5]
        ifExistStats=header_data[6]
        
        nbBinsConvolved=header_data[9]
        nbBinsNonConvolved=header_data[10]
        
        #Calculate the offset of waveforms between pulses
        posOffsetPerPulse=0
        nbBytes=8
        waveIterFormat="=%dd"
        if ifFormatFloat:
            nbBytes=4
            waveIterFormat="=%df"

        if ifExistStats:
            posOffsetPerPulse+=9+41*nbBytes-1
        if ifExistNonConv:
            if ifExistFirstOrder:
                posOffsetPerPulse+=nbBytes*(2*nbBinsNonConvolved+nbBinsConvolved)
            else:
                posOffsetPerPulse+=nbBytes*nbBinsNonConvolved
        else:
            if ifExistFirstOrder:
                posOffsetPerPulse+=nbBytes*nbBinsConvolved
        
        convolved_Length=nbBytes*nbBinsConvolved  #The length of convolved waveform data in bytes
                
        #Pulse Global Parameters:
        timeStep=header_data[7]
        distStep=header_data[8]
        nbPulses=nbBinsEachNonConvolved=header_data[11]
        
        print('Total number of pulses: ',nbPulses)
        
        spdOutFile.setTemporalBinSpacing(timeStep)
        
        #print(ifFormatFloat, ifExistNonConv, ifExistFirstOrder, ifExistStats, nbBinsConvolved, nbBinsNonConvolved)
        #print(timeStep, distStep, nbPulses)
        
        #read and convert parameters:
        minX=minY=1e10
        maxX=maxY=-1e10
        
        tmp = dartfile.tell()
        
        outPulses = list()
        feedback = int(nbPulses/10.0)
        countPulses = 0
        pulsesInBuffer = 0
        
        tmpIndicatorPast = 0
        maxCntGlobe=1e-8
        
        for cnt in range(nbPulses):
            dartfile.seek(tmp)
            try:
                pulseInfo = dartfile.read(waveform_parameter_length)
            except:
                tb = sys.exc_info()[2]
                raise IOError("Reading failed on input DART LIDARIMAGE file while reading parameter of each pulse" + dartFileName, tb)
            #end try
            pulse_info=struct.unpack(waveform_parameter_format, pulseInfo)
            try:
                wave = dartfile.read(convolved_Length)
            except:
                tb = sys.exc_info()[2]
                raise IOError("Reading failed on input DART LIDARIMAGE file while reading waveform of each pulse" + dartFileName, tb)
            #end try
            wave_data = list(struct.unpack(waveIterFormat %nbBinsConvolved, wave))
            
            if (self.ifOutputNbPhotonPerPulse):
              sum_wave_data = 0
              for j in range(len(wave_data)):
                sum_wave_data += wave_data[j]
              self.addToNbPhotonPerPulseMap(pulse_info[12], pulse_info[13], sum_wave_data);
            
            #ajust the gain and output
            maxCount=1e-8
            thresCount=1   #set 1 to be the minimum number of photons can be detected by the sensor.
            for j in range(len(wave_data)):
              if self.ifSolarNoise:
                wave_data[j]+=self.snMap[pulse_info[12]][pulse_info[13]]
              if wave_data[j]>maxCount:
                maxCount=wave_data[j]
            
            if maxCount==1e-8:
              tmp = dartfile.tell() #saves current position in input file before jump to the data of the next waveform
              tmp += posOffsetPerPulse
              continue
              
            if (maxCntGlobe<maxCount):
              maxCntGlobe=maxCount
            #if maxCount>thresCount:
                        
            if (self.ifFixedGain):
              receiveWaveGain=float(self.fixedGain)
            else:
              receiveWaveGain=float(self.maxOutput)/maxCount
            #maxOutput/maxCount
            #else:
              #receiveWaveGain=1
              
            #print(receiveWaveGain)

            
            #print(pulse_info[3], pulse_info[4], pulse_info[5], pulse_info[6], pulse_info[7], pulse_info[8])
            
            
            pulse = spdpy.createSPDPulsePy()
            pulse.pulseID = pulse_info[14]
            pulse.gpsTime = pulse_info[14]
            #pulse.scanDirectionFlag = scanDirectionFlag
            #pulse.edgeFlightLineFlag = edgeFlightLineFlag
            #pulse.gpsTime=gpsTime
            pulse.receiveWaveGain = receiveWaveGain
            pulse.receiveWaveOffset = self.receiveWaveOffset
            pulse.waveNoiseThreshold = self.waveNoiseThreshold
            
            pulse.scanAngleRank=round(math.degrees(pulse_info[0]))
            
            nbBinsToCenterFOV=pulse_info[11]
            distToCenterFOV=pulse_info[11]*distStep

            distToBeginWave=distToCenterFOV+speedOfLightPerNS*pulse_info[9] #pulse_info[10] is negative
                        
            pulse.x0 = pulse_info[6] + pulse_info[3]/2*distToBeginWave #Divided by 2 change from distance to waveform (2 way)
            pulse.y0 = pulse_info[7] + pulse_info[4]/2*distToBeginWave
            pulse.z0 = pulse_info[8] + pulse_info[5]/2*distToBeginWave            

            #print(pulse_info)
            
            #print(pulse.x0, pulse.y0, pulse.z0, distToBeginWave)
            
            if pulse.x0 < minX:
                minX = pulse.x0
            elif pulse.x0 > maxX:
                maxX = pulse.x0

            if pulse.y0 < minY:
                minY = pulse.y0
            elif pulse.y0 > maxY:
                maxY = pulse.y0
            
            aPtX = pulse.x0+pulse_info[3] * distStep * nbBinsConvolved/2
            aPtY = pulse.y0+pulse_info[4] * distStep * nbBinsConvolved/2
            aPtZ = pulse.z0+pulse_info[5] * distStep * nbBinsConvolved/2
            
            if aPtX < minX:
                minX = aPtX
            elif aPtX > maxX:
                maxX = aPtX

            if aPtY < minY:
                minY = aPtY
            elif aPtY > maxY:
                maxY = aPtY
                        
            #zenith=pulse_info[1]
            
            #azimuth=pulse_info[2]
            
            #print(zenith, azimuth)
                        
            tempX = aPtX - pulse.x0
            tempY = aPtY - pulse.y0
            tempZ = aPtZ - pulse.z0

            ptRange = math.sqrt((tempX * tempX) + (tempY * tempY) + (tempZ * tempZ))
            zenith = math.acos(tempZ/(ptRange))
            tempAzimuth = math.atan2(tempX, tempY)
            azimuth = 0
            if tempAzimuth < 0:
                azimuth = (2.0 * math.pi) + tempAzimuth
            else:
                azimuth = tempAzimuth;

            #print(pulse_info[1], pulse_info[2], zenith, azimuth)
            
            pulse.azimuth = azimuth
            pulse.zenith = zenith
            
            pulse.xIdx = pulse.x0
            pulse.yIdx = pulse.y0
            
            #print("xxx", nbBinsConvolved)
            
            #print(receiveWaveGain)
            
            for i in range(nbBinsConvolved):      
                c=int(wave_data[i]*receiveWaveGain)  #approximated count < 128
                #print(c,end=' ')
                if c>self.maxOutput:
                  c=self.maxOutput
                c_s=chr(c)
                c_b=struct.unpack(to8bit_format,bytearray(c_s, 'utf8'))
                pulse.received.append(c_b[0])
                
                if self.ifOutputImagePerBin:
                  self.addToContainerImagePerBin(i, pulse_info[12], pulse_info[13], c);
#                print(pulse.received)
                        
            pulse.numOfReceivedBins=nbBinsConvolved

              
            outPulses.append(pulse)
            countPulses+=1
            pulsesInBuffer+=1
              

            
            tmp = dartfile.tell() #saves current position in input file before jump to the data of the next waveform
            tmp += posOffsetPerPulse;

            if (pulsesInBuffer >= spdOutFile.pulseBlockSize or cnt == (nbPulses-1)):
                try:
                    updWriter.writeData(outPulses)
                except:
                    raise IOError("Error writing UPD File.")
                pulsesInBuffer = 0
                del outPulses
                outPulses = list()
            if ((not feedback==0) and (pulsesInBuffer / feedback) > tmpIndicatorPast):
                tmpIndicatorNew = int(pulsesInBuffer / feedback)
                for i in range(tmpIndicatorPast, tmpIndicatorNew):
                  print(i*10,'%......')
                  sys.stdout.flush()
                tmpIndicatorPast=tmpIndicatorNew
                
                #end of iterative reading the waveform
                
        print('100%')
        print("minX:",minX,end=' ')
        print("maxX:",maxX,end=' ')
        print("minY",minY,end=' ')
        print("maxY",maxY)
        #minX-=100
        #minY-=100
        #maxX+=100
        #maxY+=100
        spdOutFile.setXMin(minX)
        spdOutFile.setXMax(maxX)
        spdOutFile.setYMin(minY)
        spdOutFile.setYMax(maxY)
        
        if self.ifOutputImagePerBin:
          print("Write images per bin...\n")
          self.writeImagesPerBin()
        
        if (self.ifOutputNbPhotonPerPulse):
          print("Write number of recieved photons per Pulse...\n")
          with open(self.txtNbPhotonPerPulse,'w') as f:
            print('Open file '+self.txtNbPhotonPerPulse+' for writing the number of photons per pulse'+ "\n")
            f.writelines(' '.join(str(k) for k in j) + '\n' for j in self.txtNbPhotonMap)
        
        if countPulses==0:
            print("\nNo wave forms have been found...")
            print("Please check your data and try again\n")
        else:
            print("\nNumber of extracted waves: ", countPulses)
            print("\nGlobal maximum number of photons per bin: ", maxCntGlobe)
        
        dartfile.close()
        spdOutFile.setReceiveWaveformDefined(1)
        spdOutFile.setTransWaveformDefined(0)
        spdOutFile.setDecomposedPtDefined(0)
        spdOutFile.setDiscretePtDefined(0)
        spdOutFile.setOriginDefined(1)

        updWriter.close(spdOutFile)
        
    def run(self):
        parser=argparse.ArgumentParser(description='DART2UPD.py script converts a DART LIDAR multi-pulse output file to a UPD File.')
        
        parser.add_argument('inputFile', type=str, help='input DART LIDAR binary file')
        parser.add_argument('outputFile', type=str, help='output UPD file')
        
        parser.add_argument('-snf','--solarNoiseFile',type=str, help='set the solar noise input file')
        parser.add_argument('-g','--gain',type=float, help='set a fixed digital gain for the amplitude->volts conversion')
        parser.add_argument('-off','--offset',type=int, help='set a fixed digital offset for the amplitude->volts conversion')
        
        parser.add_argument('-nthes','--noiseThreshold',type=int, help='set the noise threshold')
        parser.add_argument('-maxc','--maxCount',type=int, help='set the noise threshold')
        
        parser.add_argument('-impb','--imagePerBin',type=str, help='set the output path for lidar images per bin (text format)')
        
        parser.add_argument('-nppp','--numPhPerPulse',type=str, help='set the output file of total number of photons per pulse')
        
        args=parser.parse_args()

        if args.inputFile:
          print('Input file: '+args.inputFile)
        
        if args.outputFile:
          print('Output file: '+args.outputFile)
        
        if args.solarNoiseFile:
          print('Solar noise file: '+args.solarNoiseFile)
          self.snFile = args.solarNoiseFile
          self.ifSolarNoise = True
          
        if args.gain:
          self.ifFixedGain = True
          self.fixedGain = args.gain
          print('Digital gain: ',args.gain)
        
        if args.offset:
          self.receiveWaveOffset=args.offset
          print('Digital offset: ',args.offset)
            
        if args.noiseThreshold:
          self.noiseThreshold=args.noiseThreshold
          print('Noise threshold: ',args.noiseThreshold)
          
        if args.maxCount:
          self.maxOutput=args.maxCount
          print('Maximum count number: ',args.maxCount)
        
        if args.imagePerBin:
          self.ifOutputImagePerBin = True
          self.imagePerBinPath = args.imagePerBin
          
        if args.numPhPerPulse:
          self.ifOutputNbPhotonPerPulse = True
          self.txtNbPhotonPerPulse=args.numPhPerPulse
        
        self.readDARTLidarImageBinaryFile(args.inputFile, args.outputFile)
        
        
        
        

#    def help(self):
#        print 'DART2UPD.py script converts a DART LIDAR IMAGE output file to a UPD File.'
#        print 'DART2UPD.py inputDARTLidarImageFile outputUPDFile'
        
#        print 'Print LAS header info:'
#        print 'LASWave2UPD.py -header <input file>'
#        print 'Convert Waveforms Only:'
#        print 'LASWave2UPD.py -waves <input file> <output file>'
#        print 'Convert Points Only (each point creates new pulse):'
#        print 'LASWave2UPD.py -pulpoints <input file> <output file>'
#        print 'Debug Waveforms - print to console:'
#        print 'LASWave2UPD.py -wavesdb <input file> <start point> <num points>'
#        print 'Debug Points -print to console:'
#        print 'LASWave2UPD.py -pointsdb <input file> <start point> <num points>'
#        print ''
#        print '\nThis script was distributed with version 3.0.0 of SPDLib\'s'
#        print 'python bindings.'
#        print 'For maintainance email spdlib-develop@lists.sourceforge.net'

if __name__ == '__main__':
    obj = DART2UPD()
    obj.run()


        
