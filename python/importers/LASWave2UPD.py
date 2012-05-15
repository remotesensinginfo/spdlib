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
# Date: 21/07/2011
# Version: 1.0
#
# History:
# 2011/07/21: Version 1.0 - Created using a script from PML as the basis.
#
#############################################################################

""" Module to read Fullwaveform LiDAR files format LAS 1.3 and convert to UPD """

import re
import sys
import os
import os.path
import struct
import math
import warnings
import spdpy
from datetime import date

public_header_length = 235 # Public Header length in bytes.
VbleRec_header_length = 54 # Variable Length Record Header length in bytes
EVLR_length = 60 #Extended Variable Lenght Record Header, in Version 1.3 the only EVLR is waveform data packets
point_data_length = 57
light_speed = 0.299792458 #meters per nanosecond
point_scale_factors=[] #Contained in  the file header, these scale factors must be applied to each point x,y,z
point_offsets=[] #Contained in  the file header, these scale factors must be applied to each point x,y,z
pub_head_format = "=4sHHlHH8sBB32s32sHHHLLBHL5L12dQ"
VbleRec_head_format="=H16sHH32s"
EVLR_format="=H16sHQ32s"
point_data_format="=3lHBBbBBBdBQL4f" #Note it should be 3lHBBbHBdBQL4f but User data (field [7]) is decomposed in two :0,gain
wv_packet_format = "=cclldd"

class LASWave2UPD (object):

    ################################################################
    #Function readLASHeader
    # Reads an LAS 1.3 file into a list of records (only saves records headers)
    # V0.0: only reads point records type 4 (it doesn't even check if it's a different type)
    #
    # Arguments:
    #  filename: Name of LAS file to read
    #
    # Returns:
    #  headdata: Header values
    #
    ################################################################
    # Public Header(usually 243 bytes)
    ################################################################
    # 0 File Signature ("LASF")              char[4]            4 bytes  *
    # 4 File Source ID                       unsigned short     2 bytes  *
    # 6 Global Encoding                      unsigned short     2 bytes  *
    # 8 Project ID - GUID data 1             unsigned long      4 bytes
    # 12 Project ID - GUID data 2             unsigned short     2 byte
    # 14 Project ID - GUID data 3             unsigned short     2 byte
    # 16 Project ID - GUID data 4             unsigned char[8]   8 bytes
    # 24 Version Major                        unsigned char      1 byte   *
    # 25 Version Minor                        unsigned char      1 byte   *
    # 26 System Identifier                    char[32]           32 bytes *
    # 58 Generating Software                  char[32]           32 bytes *
    # 90 File Creation Day of Year            unsigned short     2 bytes  *
    # 92 File Creation Year                   unsigned short     2 bytes  *
    # 94 Header Size                          unsigned short     2 bytes  *
    # 96 Offset to point data                 unsigned long      4 bytes  *
    # 100 Number of Variable Length Records    unsigned long      4 bytes  *
    # 104 Point Data Format ID (0-99 for spec) unsigned char      1 byte   *
    # 105 Point Data Record Length             unsigned short     2 bytes  *
    # 107 Number of point records              unsigned long      4 bytes  *
    # 111 Number of points by return           unsigned long[7]   28 bytes *
    # 139 X scale factor                       Double             8 bytes  *
    # 147 Y scale factor                       Double             8 bytes  *
    # 155 Z scale factor                       Double             8 bytes  *
    # 163 X offset                             Double             8 bytes  *
    # 171 Y offset                             Double             8 bytes  *
    # 179 Z offset                             Double             8 bytes  *
    # 187 Max X                                Double             8 bytes  *
    # 195 Min X                                Double             8 bytes  *
    # 203 Max Y                                Double             8 bytes  *
    # 211 Min Y                                Double             8 bytes  *
    # 219 Max Z                                Double             8 bytes  *
    # 227 Min Z                                Double             8 bytes  *
    # 235 Start of Waveform Data Packet Record Unsigned long long 8 bytes  *
    ################################################################
    def readLASHeader(self, filename):
        tb=None

        # Check given file exists

        if (not os.path.isfile(filename)):
            print "\nFile " + filename + " does not exist"
            print "Please check your file location and try again \n"
            sys.exit(1)
        #end if

        # Check the file ends in *.LAS
        basename, extension = os.path.splitext(filename)

        if not (extension == ".LAS" or extension == ".las"):
            print "\nFile " + filename + " is not a *.LAS or *.las file"
            print "Please specify a LAS file on the command line\n"
            sys.exit(1)
        #end if

        # Open the LAS file
        try:
            lasfile = open(filename, "rb")
        except IOError:
            print "\nCould not open LAS file " + filename
            print "Please check your file permissions and try again\n"
            sys.exit(1)
        #end try

        # Read the public header
        try:
            record = lasfile.read(public_header_length)
        except:
            tb = sys.exc_info()[2]
            print "\nReading failed on input LAS file " + filename
            sys.exit(1)
        #end try

        # Unpack data from binary to list and append to output
        headdata = struct.unpack(pub_head_format, record)

        lasfile.close()

        # Only LAS 1.3 (full wave form data) implemented
        version=str(headdata[7])+'.'+str(+headdata[8])

        if version != '1.3':
            print "\nSpecified file is a LAS %s.%s file, not a LAS 1.3 file" % (headdata[7], headdata[8])
            print "Please check your file and try again \n"
            sys.exit(1)
        #end if

        return(headdata)
    #end function

    ################################################################
    # Function printLASHeader
    # Prints Public Header formatted from LAS 1.3 file.
    #
    # Arguments:
    #  headdata: header as returned by ReadLASHeader
    #  filename: filename of LAS1.3 file
    ################################################################
    def printLASHeader(self, headdata,filename):
        print "\nHeader file of " , filename
        print "File Signature (\"LASF\") ", headdata[0]
        print "File Source ID  ", headdata[1]
        print "Global Encoding ", headdata[2]
        print "Project ID - GUID data 1 ", headdata[3]
        print "Project ID - GUID data 2 ", headdata[4]
        print "Project ID - GUID data 3 ", headdata[5]
        print "Project ID - GUID data 4 ", headdata[6]
        print "Version Major %c" % headdata[7]
        print "Version Minor ", headdata[8]
        print "System Identifier ", headdata[9]
        print "Generating Software ", headdata[10]
        print "File Creation Day of Year  ", headdata[11]
        print "File Creation Year ", headdata[12]
        print "Header Size    ", headdata[13]
        print "Offset to point data ", headdata[14]
        print "Number of Variable Length Records ", headdata[15]
        print "Point Data Format ID (0-99 for spec) ", headdata[16]
        print "Point Data Record Length ", headdata[17]
        print "Number of point records ", headdata[18]
        print "Number of points by return  ", headdata[19:23]
        print "X scale factor ", headdata[24]
        print "Y scale factor  ", headdata[25]
        print "Z scale factor ", headdata[26]
        print "X offset  ", headdata[27]
        print "Y offset  ", headdata[28]
        print "Z offset ", headdata[29]
        print "Max X  ", headdata[30]
        print "Min X  ", headdata[31]
        print "Max Y  ", headdata[32]
        print "Min Y ", headdata[33]
        print "Max Z  ", headdata[34]
        print "Min Z  ", headdata[35]
        print "Start of Waveform Data Packet Record ", headdata[36] , "\n"
    #end function

    ################################################################
    # Function readLASWaves
    # Function that extracts waveforms (only!) from LAS 1.3 files.
    #
    # Arguments:
    #  headdata: header as returned by ReadLASHeader.
    #  inputFile: LAS 1.3 file to extract data from.
    #  outputFile: Output UPD file.
    #
    ################################################################
    def readLASWaves(self, headdata, inputFile, outputFile):
        spdOutFile = spdpy.createSPDFile(outputFile);
        updWriter = spdpy.SPDPyNoIdxWriter()
        updWriter.open(spdOutFile, outputFile)
        record = ""
        tb=None

        lasfile = open(inputFile, "rb")

        try:
            record = lasfile.read(public_header_length)
        except:
            tb = sys.exc_info()[2]
            print "Reading failed on input LAS file " + inputFile + "\n"
            sys.exit(1)
        #end try

        minX = headdata[31]
        maxX = headdata[30]
        minY = headdata[33]
        maxY = headdata[32]

        #printPubHeader(headdata)
        point_scale_factors.append(headdata[24]) # X scale factor
        point_scale_factors.append(headdata[25]) # Y scale factor
        point_scale_factors.append(headdata[26]) # Z scale factor
        point_offsets.append(headdata[27]) # X offset
        point_offsets.append(headdata[28]) # Y offset
        point_offsets.append(headdata[29]) # Z offset

        # Read as many records as indicated on the header
        N_vble_rec= headdata[15]

        for v_rec in range(N_vble_rec):
            try:
                v_record = lasfile.read(VbleRec_header_length)
            except:
                tb = sys.exc_info()[2]
                raise IOError, "Reading failed on input LAS file while reading Vble length record " + inputFile, tb
            #end try

            headdata_rec = struct.unpack(VbleRec_head_format, v_record)

            Rec_length = headdata_rec[3]
            skip_record = lasfile.read(Rec_length)

            #If RecordID= 1001 it is the intensity histogram
            if (headdata_rec[2] == 1001):
                i_hist = struct.unpack("=%dl" %(Rec_length/4),skip_record)
                #print i_hist
            #end if

            #If RecordID= 1002 it is Leica mission info containing
            #0_ Laser Pulserate: 1 Hz
            #1_ Field Of View: 0.1 degrees
            #2_ Scanner Offset: 1 ticks
            #3_ Scan Rate:  0.1 Hz
            #4_ Flying Altitude (AGL): meters
            #5_ GPS Week Number at start of line:   week
            #6_ GPS Seconds of Week at start of line: seconds
            #7_ Reserved
            #NOTE Leica definition says this record contains 26 bytes but it actually contains only 22 so not sure what is what...
            # By comparisson with FlightLineLog fields 0,3,4,5 and 6 are ok


            if (headdata_rec[2] == 1002):
                mis_info = struct.unpack("=lHHhhhll",skip_record)
                #print "mission info", mis_info
                laser_pulse_rate=mis_info[0]
                spdOutFile.setPulseRepetitionFreq(laser_pulse_rate)
                field_of_view=mis_info[1]
                scanner_offset=mis_info[2]
                scan_rate=mis_info[3]
                spdOutFile.setSensorScanRate(scan_rate)
                fly_altitude=mis_info[4] # Corresponds to the Nadir Range in the FlightLineLog
                spdOutFile.setSensorHeight(fly_altitude)
            #end if

            #If RecordID= 1003 it is User inputs containing:
            # IMU Roll Correction
            # IMU Pitch Correction
            # IMU Heading Correction
            # POS Time Offset
            # Range Offset - Return 1
            # Range Offset - Return 2
            # Range Offset - Return 3
            # Range Offset - Return 4
            # Range Offset - Return 5
            # Elevation Offset
            # Scan Angle Correction
            # Encoder Latency
            # Torsion Constant
            # Scanner Ticks Per Revolution
            # Low Altitude Temperature
            # High Altitude Temperature
            # Low Altitude
            # High Altitude
            # Temperature Gradient

            if (headdata_rec[2] == 1003):
                user_info = struct.unpack("=3l9hll4hd",skip_record)
                #print "userinfo",user_info
            #end if

            if (headdata_rec[2] == 34735):
                #struct.calcsize(skip_record)
                projection_info = struct.unpack("28H",skip_record)
            #print "Projection", projection_info

            #If RecordID>= 100 it is a waveform Packet Descriptor

            if (headdata_rec[2] >= 100) and (headdata_rec[2] < 356):
                wv_info = struct.unpack("=cclldd",skip_record)
                spdOutFile.setTemporalBinSpacing(wv_info[3]/1000.0)
                #print wv_info
            #end if
        #end for

        # Read points
        outPulses = list()
        Size_points = headdata[17]
        N_points = headdata[18]
        Offset_points = headdata[14]
        Offset_EVLRH = headdata[36]

        # Move to first point
        lasfile.seek(Offset_points)
        countAllPoints = 0
        countPulses = 0
        pulsesInBuffer = 0;
        c_point=[0,0]
        print "Starting to process %d points" %N_points
        feedback = int(N_points/100)

        for p in range(N_points):
            Point = lasfile.read(Size_points)
            point_info = struct.unpack(point_data_format,Point)

            wave_desc = point_info[11]
            if wave_desc <> 0: # if there is waveform asociated to this point
                pulse = spdpy.createSPDPulsePy()
                pulse.pulseID = countPulses

                scan_dir=(point_info[4]&64)/64
                pulse.scanDirectionFlag = scan_dir
                edge_fl=(point_info[4]&128)/128
                pulse.edgeFlightLineFlag = edge_fl

                wavedata = []
                wavedata.append(point_info)
                wave_offset = Offset_EVLRH + point_info[12]
                wave_size = point_info[13]

                tmp = lasfile.tell() #saves current position in input file before jumpin to wave info
                lasfile.seek(wave_offset)
                wave_dat = lasfile.read(wave_size)
                wave_data = struct.unpack("=%db" %wave_size, wave_dat)
                wavedata.append(wave_data)

                pulse.gpsTime = int(wavedata[0][10])
                pulse.scanAngleRank = wavedata[0][6]

                pulse.receiveWaveGain = wavedata[0][8]*wv_info[4]
                pulse.receiveWaveOffset = float(wv_info[5])
                pulse.waveNoiseThreshold = 17

                w_point=[0,0,0]
                w_point[0]=wavedata[0][0]*point_scale_factors[0]+point_offsets[0]
                w_point[1]=wavedata[0][1]*point_scale_factors[1]+point_offsets[1]
                w_point[2]=wavedata[0][2]*point_scale_factors[2]+point_offsets[2]

                pulse.x0 = float(w_point[0]+ wavedata[0][15]*wavedata[0][14])
                pulse.y0 = float(w_point[1]+ wavedata[0][16]*wavedata[0][14])
                pulse.z0 = float(w_point[2]+ wavedata[0][17]*wavedata[0][14])

                if pulse.x0 < minX:
                    minX = pulse.x0
                elif pulse.x0 > maxX:
                    maxX = pulse.x0

                if pulse.y0 < minY:
                    minY = pulse.y0
                elif pulse.y0 > maxY:
                    maxY = pulse.y0

                aPtX = pulse.x0-(wavedata[0][15]*1000*(wv_info[3]/1000.0)) * wave_size
                aPtY = pulse.y0-(wavedata[0][16]*1000*(wv_info[3]/1000.0)) * wave_size
                aPtZ = pulse.z0-(wavedata[0][17]*1000*(wv_info[3]/1000.0)) * wave_size

                if aPtX < minX:
                    minX = aPtX
                elif aPtX > maxX:
                    maxX = aPtX

                if aPtY < minY:
                    minY = aPtY
                elif aPtY > maxY:
                    maxY = aPtY

                tempX = aPtX - pulse.x0;
                tempY = aPtY - pulse.y0;
                tempZ = aPtZ - pulse.z0;

                ptRange = math.sqrt((tempX * tempX) + (tempY * tempY) + (tempZ * tempZ));
                zenith = math.acos(tempZ/(ptRange));
                azimuth = math.atan2(tempY, tempX);

                pulse.azimuth = azimuth
                pulse.zenith = zenith

                pulse.xIdx = pulse.x0
                pulse.yIdx = pulse.y0

                for i in range(len(wave_data)):
                    if wave_data[i] < 0:
                        pulse.received.append(0)
                    else:
                        pulse.received.append(wave_data[i])

                pulse.numOfReceivedBins = wave_size

                outPulses.append(pulse)

                lasfile.seek(tmp) # Goes back to next point in file
                countPulses+=1
                pulsesInBuffer+=1

            if (pulsesInBuffer >= spdOutFile.pulseBlockSize or p == (N_points-1)):
                try:
                    updWriter.writeData(outPulses)
                except:
                    raise IOError, "Error writing UPD File."
                pulsesInBuffer = 0
                del outPulses
                outPulses = list()

            if (countAllPoints % feedback) == 0:
                print ".", countAllPoints, ".",
                sys.stdout.flush()

            countAllPoints+=1
            #end if
        #end for

        spdOutFile.setXMin(minX)
        spdOutFile.setXMax(maxX)
        spdOutFile.setYMin(minY)
        spdOutFile.setYMax(maxY)

        if countPulses==0:
            print "\nNo wave forms have been found..."
            print "Please check your data and try again\n"
        else:
            print "\nNumber of extracted waves: ", countPulses
            print countAllPoints-countPulses, " did not have waveforms."
        #end if

        # After points, read EVLR
        try:
            evlr_record = lasfile.read(EVLR_length)
        except:
            tb = sys.exc_info()[2] # Get traceback (causes circular reference to clean up later)
            raise IOError, "Reading failed on input LAS file while reading Vble length record " + inputFile, tb
        #end try

        evlr_data = struct.unpack("=H16sHQ32s", evlr_record)
        #It doesn't contain anything useful really...
        #print "Extended Variable Length Record Header:", evlr_data

        lasfile.close()
        spdOutFile.setReceiveWaveformDefined(1)
        spdOutFile.setTransWaveformDefined(0)
        spdOutFile.setDecomposedPtDefined(0)
        spdOutFile.setDiscretePtDefined(1)
        spdOutFile.setOriginDefined(1)

        updWriter.close(spdOutFile)
    #end function

    ################################################################
    # Function readPrintLASWaves
    # Function that extracts waveforms (only!) from LAS 1.3 files.
    #
    # Arguments:
    #  headdata: header as returned by ReadLASHeader.
    #  inputFile: LAS 1.3 file to extract data from.
    #  outputFile: Output UPD file.
    #  startPt: first point to be printed.
    #  numPts: number of points to be printed.
    #
    ################################################################
    def readPrintLASWaves(self, headdata, inputFile, startPt, numPts):
        record = ""
        tb=None

        lasfile = open(inputFile, "rb")

        try:
            record = lasfile.read(public_header_length)
        except:
            tb = sys.exc_info()[2]
            print "Reading failed on input LAS file " + inputFile + "\n"
            sys.exit(1)
        #end try

        minX = headdata[31]
        maxX = headdata[30]
        minY = headdata[33]
        maxY = headdata[32]

        print "Extent [minX, maxX, minY, maxY][", minX, ",", maxX, ",", minY, ",", maxY, "]"

        #printPubHeader(headdata)
        point_scale_factors.append(headdata[24]) # X scale factor
        point_scale_factors.append(headdata[25]) # Y scale factor
        point_scale_factors.append(headdata[26]) # Z scale factor
        point_offsets.append(headdata[27]) # X offset
        point_offsets.append(headdata[28]) # Y offset
        point_offsets.append(headdata[29]) # Z offset

        # Read as many records as indicated on the header
        N_vble_rec= headdata[15]

        for v_rec in range(N_vble_rec):
            try:
                v_record = lasfile.read(VbleRec_header_length)
            except:
                tb = sys.exc_info()[2]
                raise IOError, "Reading failed on input LAS file while reading Vble length record " + inputFile, tb
            #end try

            headdata_rec = struct.unpack(VbleRec_head_format, v_record)

            Rec_length = headdata_rec[3]
            skip_record = lasfile.read(Rec_length)

            #If RecordID= 1001 it is the intensity histogram
            if (headdata_rec[2] == 1001):
                i_hist = struct.unpack("=%dl" %(Rec_length/4),skip_record)
                #print i_hist
            #end if

            #If RecordID= 1002 it is Leica mission info containing
            #0_ Laser Pulserate: 1 Hz
            #1_ Field Of View: 0.1 degrees
            #2_ Scanner Offset: 1 ticks
            #3_ Scan Rate:  0.1 Hz
            #4_ Flying Altitude (AGL): meters
            #5_ GPS Week Number at start of line:   week
            #6_ GPS Seconds of Week at start of line: seconds
            #7_ Reserved
            #NOTE Leica definition says this record contains 26 bytes but it actually contains only 22 so not sure what is what...
            # By comparisson with FlightLineLog fields 0,3,4,5 and 6 are ok
            if (headdata_rec[2] == 1002):
                mis_info = struct.unpack("=lHHhhhll",skip_record)
                #print "mission info", mis_info
                laser_pulse_rate=mis_info[0]
                print "Laser Pulse Rate: ", laser_pulse_rate
                field_of_view=mis_info[1]
                scanner_offset=mis_info[2]
                scan_rate=mis_info[3]
                print "Sensor Scan Rate: ", scan_rate
                fly_altitude=mis_info[4] # Corresponds to the Nadir Range in the FlightLineLog
                print "Sensor Height: ", fly_altitude
            #end if

            #If RecordID= 1003 it is User inputs containing:
            # IMU Roll Correction
            # IMU Pitch Correction
            # IMU Heading Correction
            # POS Time Offset
            # Range Offset - Return 1
            # Range Offset - Return 2
            # Range Offset - Return 3
            # Range Offset - Return 4
            # Range Offset - Return 5
            # Elevation Offset
            # Scan Angle Correction
            # Encoder Latency
            # Torsion Constant
            # Scanner Ticks Per Revolution
            # Low Altitude Temperature
            # High Altitude Temperature
            # Low Altitude
            # High Altitude
            # Temperature Gradient
            if (headdata_rec[2] == 1003):
                user_info = struct.unpack("=3l9hll4hd",skip_record)
                #print "userinfo",user_info
            #end if

            if (headdata_rec[2] == 34735):
                #struct.calcsize(skip_record)
                projection_info = struct.unpack("28H",skip_record)
            #print "Projection", projection_info

            #If RecordID>= 100 it is a waveform Packet Descriptor
            if (headdata_rec[2] >= 100) and (headdata_rec[2] < 356):
                wv_info = struct.unpack("=cclldd",skip_record)
                print "Temporal Bin Spacing: ", wv_info[3]/1000.0
                #print wv_info
            #end if
        #end for

        # Read points
        outPulses = list()
        Size_points = headdata[17]
        N_points = headdata[18]
        Offset_points = headdata[14]
        Offset_EVLRH = headdata[36]

        # Move to first point
        lasfile.seek(Offset_points)
        countAllPoints = 0
        c_point=[0,0]
        print "Starting to process %d points" %N_points
        printDBInfo = False
        countPrintedPoints = 0
        countPulses = 0

        for p in range(N_points):
            Point = lasfile.read(Size_points)
            point_info = struct.unpack(point_data_format,Point)

            wave_desc = point_info[11]
            if wave_desc <> 0: # if there is waveform asociated to this point
                if countAllPoints >= startPt:
                    printDBInfo = True

                if printDBInfo:
                    print "Point : ", countAllPoints
                    print "GPS Time: ", point_info[10]
                    print "XYZ: [", (point_info[0]*point_scale_factors[0])+point_offsets[0], ",", (point_info[1]*point_scale_factors[1])+point_offsets[1], ",", (point_info[2]*point_scale_factors[2])+point_offsets[2], "]"
                    print "Intensity: ", point_info[3]
                    print "Return No: ", (point_info[4]&7)
                    print "No of Returns: ", (point_info[4] & 56)
                    print "Scan Direction Flag: ", (point_info[4]&64)/64
                    print "Edge of flightline flag: ", (point_info[4]&128)/128
                    countPrintedPoints=countPrintedPoints+1

                wavedata = []
                wavedata.append(point_info)
                wave_offset = Offset_EVLRH + point_info[12]
                wave_size = point_info[13]

                tmp = lasfile.tell() #saves current position in input file before jumpin to wave info
                lasfile.seek(wave_offset)
                wave_dat = lasfile.read(wave_size)
                wave_data = struct.unpack("=%db" %wave_size, wave_dat)
                wavedata.append(wave_data)

                if printDBInfo:
                    print "AGC: ", wavedata[0][8]
                    print "Received Gain: ", wv_info[4]
                    print "Offset: ", wv_info[5]
                    print "Noise Threshold: ", (17*wavedata[0][8]*wv_info[4])+wv_info[5]

                w_point=[0,0,0]
                w_point[0]=wavedata[0][0]*point_scale_factors[0]+point_offsets[0]
                w_point[1]=wavedata[0][1]*point_scale_factors[1]+point_offsets[1]
                w_point[2]=wavedata[0][2]*point_scale_factors[2]+point_offsets[2]

                aPtX = float(w_point[0]+ wavedata[0][15]*wavedata[0][14])-(wavedata[0][15]*1000*(wv_info[3]/1000.0)) * wave_size
                aPtY = float(w_point[1]+ wavedata[0][16]*wavedata[0][14])-(wavedata[0][16]*1000*(wv_info[3]/1000.0)) * wave_size
                aPtZ = float(w_point[2]+ wavedata[0][17]*wavedata[0][14])-(wavedata[0][17]*1000*(wv_info[3]/1000.0)) * wave_size

                tempX = aPtX - float(w_point[0]+ wavedata[0][15]*wavedata[0][14]);
                tempY = aPtY - float(w_point[1]+ wavedata[0][16]*wavedata[0][14]);
                tempZ = aPtZ - float(w_point[2]+ wavedata[0][17]*wavedata[0][14]);

                ptRange = math.sqrt((tempX * tempX) + (tempY * tempY) + (tempZ * tempZ));
                zenith = math.acos(tempZ/(ptRange));
                azimuth = math.atan2(tempY, tempX);

                if printDBInfo:
                    print "Origin: [X,Y,Z]: [", float(w_point[0]+ wavedata[0][15]*wavedata[0][14]), ",", float(w_point[1]+ wavedata[0][16]*wavedata[0][14]), ",", float(w_point[2]+ wavedata[0][17]*wavedata[0][14]), "]"
                    print "X Offset: ", wavedata[0][15]*1000*(wv_info[3]/1000.0) # *1000 to convert from km to meters *sampling to get offset between each sample
                    print "Y Offset: ", wavedata[0][16]*1000*(wv_info[3]/1000.0) # *1000 to convert from km to meters *sampling to get offset between each sample
                    print "Z Offset: ", wavedata[0][17]*1000*(wv_info[3]/1000.0) # *1000 to convert from km to meters *sampling to get offset between each sample
                    print "Arb Point: [", aPtX, ",", aPtY, ",", aPtZ, "]"
                    print "Azimuth: ", azimuth
                    print "Zenith: ", zenith


                if printDBInfo:
                    print "Waveform Offset: ", wave_offset
                    print "Waveform Size: ", wave_size
                    for i in range(len(wave_data)):
                        if i == 0:
                            print wave_data[i],
                        else:
                            print ",", wave_data[i],
                    print "\n"

                lasfile.seek(tmp) # Goes back to next point in file
                countPulses+=1

            if countPrintedPoints >= numPts:
                break;

            countAllPoints+=1
            #end if
        #end for

        if countPulses==0:
            print "\nNo wave forms have been found..."
            print "Please check your data and try again\n"
        else:
            print "\nNumber of extracted pulses: ", countPulses
            print countAllPoints-countPulses, " did not have waveforms."
        #end if

        # After points, read EVLR
        try:
            evlr_record = lasfile.read(EVLR_length)
        except:
            tb = sys.exc_info()[2] # Get traceback (causes circular reference to clean up later)
            raise IOError, "Reading failed on input LAS file while reading Vble length record " + inputFile, tb
        #end try

        evlr_data = struct.unpack("=H16sHQ32s", evlr_record)
        #It doesn't contain anything useful really...
        #print "Extended Variable Length Record Header:", evlr_data

        lasfile.close()
    #end function

    ################################################################
    # Function readLASPoints
    # Function that extracts points (only!) as individual pulses
    # from LAS 1.3 files.
    #
    # Arguments:
    #  headdata: header as returned by ReadLASHeader.
    #  inputFile: LAS 1.3 file to extract data from.
    #  outputFile: Output UPD file.
    #
    ################################################################
    def readLASPoints(self, headdata, inputFile, outputFile):
        spdOutFile = spdpy.createSPDFile(outputFile);
        updWriter = spdpy.SPDPyNoIdxWriter()
        updWriter.open(spdOutFile, outputFile)
        record = ""
        tb=None

        lasfile = open(inputFile, "rb")

        try:
            record = lasfile.read(public_header_length)
        except:
            tb = sys.exc_info()[2]
            print "Reading failed on input LAS file " + inputFile + "\n"
            sys.exit(1)
        #end try

        minX = headdata[31]
        maxX = headdata[30]
        minY = headdata[33]
        maxY = headdata[32]

        #printPubHeader(headdata)
        point_scale_factors.append(headdata[24]) # X scale factor
        point_scale_factors.append(headdata[25]) # Y scale factor
        point_scale_factors.append(headdata[26]) # Z scale factor
        point_offsets.append(headdata[27]) # X offset
        point_offsets.append(headdata[28]) # Y offset
        point_offsets.append(headdata[29]) # Z offset

        # Read as many records as indicated on the header
        N_vble_rec= headdata[15]

        for v_rec in range(N_vble_rec):
            try:
                v_record = lasfile.read(VbleRec_header_length)
            except:
                tb = sys.exc_info()[2]
                raise IOError, "Reading failed on input LAS file while reading Vble length record " + inputFile, tb
            #end try

            headdata_rec = struct.unpack(VbleRec_head_format, v_record)

            Rec_length = headdata_rec[3]
            skip_record = lasfile.read(Rec_length)

            #If RecordID= 1001 it is the intensity histogram
            if (headdata_rec[2] == 1001):
                i_hist = struct.unpack("=%dl" %(Rec_length/4),skip_record)
                #print i_hist
            #end if

            #If RecordID= 1002 it is Leica mission info containing
            #0_ Laser Pulserate: 1 Hz
            #1_ Field Of View: 0.1 degrees
            #2_ Scanner Offset: 1 ticks
            #3_ Scan Rate:  0.1 Hz
            #4_ Flying Altitude (AGL): meters
            #5_ GPS Week Number at start of line:   week
            #6_ GPS Seconds of Week at start of line: seconds
            #7_ Reserved
            #NOTE Leica definition says this record contains 26 bytes but it actually contains only 22 so not sure what is what...
            # By comparisson with FlightLineLog fields 0,3,4,5 and 6 are ok
            if (headdata_rec[2] == 1002):
                mis_info = struct.unpack("=lHHhhhll",skip_record)
                #print "mission info", mis_info
                laser_pulse_rate=mis_info[0]
                spdOutFile.setPulseRepetitionFreq(laser_pulse_rate)
                field_of_view=mis_info[1]
                scanner_offset=mis_info[2]
                scan_rate=mis_info[3]
                spdOutFile.setSensorScanRate(scan_rate)
                fly_altitude=mis_info[4] # Corresponds to the Nadir Range in the FlightLineLog
                spdOutFile.setSensorHeight(fly_altitude)
            #end if

            #If RecordID= 1003 it is User inputs containing:
            # IMU Roll Correction
            # IMU Pitch Correction
            # IMU Heading Correction
            # POS Time Offset
            # Range Offset - Return 1
            # Range Offset - Return 2
            # Range Offset - Return 3
            # Range Offset - Return 4
            # Range Offset - Return 5
            # Elevation Offset
            # Scan Angle Correction
            # Encoder Latency
            # Torsion Constant
            # Scanner Ticks Per Revolution
            # Low Altitude Temperature
            # High Altitude Temperature
            # Low Altitude
            # High Altitude
            # Temperature Gradient
            if (headdata_rec[2] == 1003):
                user_info = struct.unpack("=3l9hll4hd",skip_record)
                #print "userinfo",user_info
            #end if

            if (headdata_rec[2] == 34735):
                #struct.calcsize(skip_record)
                projection_info = struct.unpack("28H",skip_record)
            #print "Projection", projection_info

            #If RecordID>= 100 it is a waveform Packet Descriptor
            if (headdata_rec[2] >= 100) and (headdata_rec[2] < 356):
                wv_info = struct.unpack("=cclldd",skip_record)
                spdOutFile.setTemporalBinSpacing(wv_info[3]/1000.0)
                #print wv_info
            #end if
        #end for

        # Read points
        outPulses = list()
        Size_points = headdata[17]
        N_points = headdata[18]
        Offset_points = headdata[14]
        Offset_EVLRH = headdata[36]

        # Move to first point
        lasfile.seek(Offset_points)
        countAllPoints = 0
        pulsesInBuffer = 0;
        c_point=[0,0]
        print "Starting to process %d points" %N_points
        feedback = int(N_points/100)

        for p in range(N_points):
            Point = lasfile.read(Size_points)
            point_info = struct.unpack(point_data_format,Point)

            pulse = spdpy.createSPDPulsePy()
            pulse.pulseID = countAllPoints
            pulse.scanDirectionFlag = (point_info[4]&64)/64
            pulse.edgeFlightLineFlag = (point_info[4]&128)/128
            pulse.gpsTime = int(point_info[10])
            pulse.scanAngleRank = point_info[6]

            pulse.numberOfReturns = 1
            spdPoint = spdpy.createSPDPointPy()
            spdPoint.pointID = 0
            spdPoint.x = (point_info[0]*point_scale_factors[0])+point_offsets[0]
            spdPoint.y = (point_info[1]*point_scale_factors[1])+point_offsets[1]
            spdPoint.z = (point_info[2]*point_scale_factors[2])+point_offsets[2]
            spdPoint.amplitudeReturn = point_info[3]

            if spdPoint.x < minX:
                minX = spdPoint.x
            elif spdPoint.x > maxX:
                maxX = spdPoint.x

            if spdPoint.y < minY:
                minY = spdPoint.y
            elif spdPoint.y > maxY:
                maxY = spdPoint.y

            pulse.pts = list()
            pulse.pts.append(spdPoint)

            pulse.xIdx = spdPoint.x
            pulse.yIdx = spdPoint.y

            outPulses.append(pulse)

            pulsesInBuffer+=1

            if pulsesInBuffer >= spdOutFile.pulseBlockSize:
                try:
                    updWriter.writeData(outPulses)
                except:
                    raise IOError, "Error writing UPD File."
                pulsesInBuffer = 0
                del outPulses
                outPulses = list()
                #print "Written Data"

            if (countAllPoints % feedback) == 0:
                print ".", countAllPoints, ".",
                sys.stdout.flush()

            countAllPoints+=1
        #end for


        if pulsesInBuffer > 0:
            try:
                updWriter.writeData(outPulses)
            except:
                raise IOError, "Error writing UPD File."
            pulsesInBuffer = 0
            del outPulses
            outPulses = list()

        spdOutFile.setXMin(minX)
        spdOutFile.setXMax(maxX)
        spdOutFile.setYMin(minY)
        spdOutFile.setYMax(maxY)

        if countAllPoints==0:
            print "\nNo points have been found..."
            print "Please check your data and try again\n"
        else:
            print "\nNumber of extracted pulses: ", countAllPoints
        #end if

        # After points, read EVLR
        try:
            evlr_record = lasfile.read(EVLR_length)
        except:
            tb = sys.exc_info()[2] # Get traceback (causes circular reference to clean up later)
            raise IOError, "Reading failed on input LAS file while reading Vble length record " + inputFile, tb
        #end try

        evlr_data = struct.unpack("=H16sHQ32s", evlr_record)
        #It doesn't contain anything useful really...
        #print "Extended Variable Length Record Header:", evlr_data

        lasfile.close()
        spdOutFile.setReceiveWaveformDefined(0)
        spdOutFile.setTransWaveformDefined(0)
        spdOutFile.setDecomposedPtDefined(0)
        spdOutFile.setDiscretePtDefined(1)

        updWriter.close(spdOutFile)
    #end function

    ################################################################
    # Function readPrintLASPoints
    # Function that extracts points (only!) as individual pulses
    # from LAS 1.3 files.
    #
    # Arguments:
    #  headdata: header as returned by ReadLASHeader.
    #  inputFile: LAS 1.3 file to extract data from.
    #  startPt: first point to be printed.
    #  numPts: number of points to be printed.
    #
    ################################################################
    def readPrintLASPoints(self, headdata, inputFile, startPt, numPts):
        record = ""
        tb=None

        lasfile = open(inputFile, "rb")

        try:
            record = lasfile.read(public_header_length)
        except:
            tb = sys.exc_info()[2]
            print "Reading failed on input LAS file " + inputFile + "\n"
            sys.exit(1)
        #end try

        minX = headdata[31]
        maxX = headdata[30]
        minY = headdata[33]
        maxY = headdata[32]

        print "Extent [minX, maxX, minY, maxY][", minX, ",", maxX, ",", minY, ",", maxY, "]"

        #printPubHeader(headdata)
        point_scale_factors.append(headdata[24]) # X scale factor
        point_scale_factors.append(headdata[25]) # Y scale factor
        point_scale_factors.append(headdata[26]) # Z scale factor
        point_offsets.append(headdata[27]) # X offset
        point_offsets.append(headdata[28]) # Y offset
        point_offsets.append(headdata[29]) # Z offset

        # Read as many records as indicated on the header
        N_vble_rec= headdata[15]

        for v_rec in range(N_vble_rec):
            try:
                v_record = lasfile.read(VbleRec_header_length)
            except:
                tb = sys.exc_info()[2]
                raise IOError, "Reading failed on input LAS file while reading Vble length record " + inputFile, tb
            #end try

            headdata_rec = struct.unpack(VbleRec_head_format, v_record)

            Rec_length = headdata_rec[3]
            skip_record = lasfile.read(Rec_length)

            #If RecordID= 1001 it is the intensity histogram
            if (headdata_rec[2] == 1001):
                i_hist = struct.unpack("=%dl" %(Rec_length/4),skip_record)
                #print i_hist
            #end if

            #If RecordID= 1002 it is Leica mission info containing
            #0_ Laser Pulserate: 1 Hz
            #1_ Field Of View: 0.1 degrees
            #2_ Scanner Offset: 1 ticks
            #3_ Scan Rate:  0.1 Hz
            #4_ Flying Altitude (AGL): meters
            #5_ GPS Week Number at start of line:   week
            #6_ GPS Seconds of Week at start of line: seconds
            #7_ Reserved
            #NOTE Leica definition says this record contains 26 bytes but it actually contains only 22 so not sure what is what...
            # By comparisson with FlightLineLog fields 0,3,4,5 and 6 are ok
            if (headdata_rec[2] == 1002):
                mis_info = struct.unpack("=lHHhhhll",skip_record)
                #print "mission info", mis_info
                laser_pulse_rate=mis_info[0]
                print "Laser Pulse Rate: ", laser_pulse_rate
                field_of_view=mis_info[1]
                scanner_offset=mis_info[2]
                scan_rate=mis_info[3]
                print "Sensor Scan Rate: ", scan_rate
                fly_altitude=mis_info[4] # Corresponds to the Nadir Range in the FlightLineLog
                print "Sensor Height: ", fly_altitude
            #end if

            #If RecordID= 1003 it is User inputs containing:
            # IMU Roll Correction
            # IMU Pitch Correction
            # IMU Heading Correction
            # POS Time Offset
            # Range Offset - Return 1
            # Range Offset - Return 2
            # Range Offset - Return 3
            # Range Offset - Return 4
            # Range Offset - Return 5
            # Elevation Offset
            # Scan Angle Correction
            # Encoder Latency
            # Torsion Constant
            # Scanner Ticks Per Revolution
            # Low Altitude Temperature
            # High Altitude Temperature
            # Low Altitude
            # High Altitude
            # Temperature Gradient
            if (headdata_rec[2] == 1003):
                user_info = struct.unpack("=3l9hll4hd",skip_record)
                #print "userinfo",user_info
            #end if

            if (headdata_rec[2] == 34735):
                #struct.calcsize(skip_record)
                projection_info = struct.unpack("28H",skip_record)
            #print "Projection", projection_info

            #If RecordID>= 100 it is a waveform Packet Descriptor
            if (headdata_rec[2] >= 100) and (headdata_rec[2] < 356):
                wv_info = struct.unpack("=cclldd",skip_record)
                print "Temporal Bin Spacing: ", wv_info[3]/1000.0
                #print wv_info
            #end if
        #end for

        # Read points
        outPulses = list()
        Size_points = headdata[17]
        N_points = headdata[18]
        Offset_points = headdata[14]
        Offset_EVLRH = headdata[36]

        # Move to first point
        lasfile.seek(Offset_points)
        countAllPoints = 0
        c_point=[0,0]
        print "Starting to process %d points" %N_points
        printDBInfo = False
        countPrintedPoints = 0

        for p in range(N_points):
            Point = lasfile.read(Size_points)
            point_info = struct.unpack(point_data_format,Point)

            if countAllPoints >= startPt:
                printDBInfo = True

            if printDBInfo:
                print "Point : ", countAllPoints
                print "GPS Time: ", point_info[10]
                print "XYZ: [", (point_info[0]*point_scale_factors[0])+point_offsets[0], ",", (point_info[1]*point_scale_factors[1])+point_offsets[1], ",", (point_info[2]*point_scale_factors[2])+point_offsets[2], "]"
                print "Intensity: ", point_info[3]
                print "Return No: ", (point_info[4]&7)
                print "No of Returns: ", (point_info[4] & 56)
                print "Scan Direction Flag: ", (point_info[4]&64)/64
                print "Edge of flightline flag: ", (point_info[4]&128)/128
                print ''
                countPrintedPoints=countPrintedPoints+1

            if countPrintedPoints >= numPts:
                break;
            countAllPoints+=1
        #end for

        if countAllPoints==0:
            print "\nNo points have been found..."
            print "Please check your data and try again\n"
        else:
            print "\nNumber of extracted point: ", countAllPoints
        #end if

        # After points, read EVLR
        try:
            evlr_record = lasfile.read(EVLR_length)
        except:
            tb = sys.exc_info()[2] # Get traceback (causes circular reference to clean up later)
            raise IOError, "Reading failed on input LAS file while reading Vble length record " + inputFile, tb
        #end try

        evlr_data = struct.unpack("=H16sHQ32s", evlr_record)
        #It doesn't contain anything useful really...
        #print "Extended Variable Length Record Header:", evlr_data

        lasfile.close()
    #end function


    def run(self):
        numArgs = len(sys.argv)
        if numArgs == 4:
            optionFlag = sys.argv[1].strip()
            inputFile = sys.argv[2].strip()
            outputFile = sys.argv[3].strip()

            lasHeaderData = self.readLASHeader(inputFile)

            if optionFlag == "-waves":
                self.readLASWaves(lasHeaderData, inputFile, outputFile)
            elif optionFlag == "-pulpoints":
                self.readLASPoints(lasHeaderData, inputFile, outputFile)
            else:
                self.help()
        elif numArgs == 3:
            optionFlag = sys.argv[1].strip()
            inputFile = sys.argv[2].strip()

            lasHeaderData = self.readLASHeader(inputFile)

            if optionFlag == "-header":
                self.printLASHeader(lasHeaderData, inputFile)
            else:
                self.help()
        elif numArgs == 5:
            optionFlag = sys.argv[1].strip()
            inputFile = sys.argv[2].strip()
            startPoint = int(sys.argv[3].strip())
            numPoints = int(sys.argv[4].strip())

            lasHeaderData = self.readLASHeader(inputFile)

            if optionFlag == "-wavesdb":
                self.readPrintLASWaves(lasHeaderData, inputFile, startPoint, numPoints)
            elif optionFlag == "-pointsdb":
                self.readPrintLASPoints(lasHeaderData, inputFile, startPoint, numPoints)
            else:
                self.help()
        else:
            self.help()

    def help(self):
        print 'LASWave2UPD.py script converts a LAS 1.3 file to a UPD File.'
        print ''
        print 'Print LAS header info:'
        print 'LASWave2UPD.py -header <input file>'
        print 'Convert Waveforms Only:'
        print 'LASWave2UPD.py -waves <input file> <output file>'
        print 'Convert Points Only (each point creates new pulse):'
        print 'LASWave2UPD.py -pulpoints <input file> <output file>'
        print 'Debug Waveforms - print to console:'
        print 'LASWave2UPD.py -wavesdb <input file> <start point> <num points>'
        print 'Debug Points -print to console:'
        print 'LASWave2UPD.py -pointsdb <input file> <start point> <num points>'
        print ''
        print '\nThis script was distributed with version 3.0.0 of SPDLib\'s'
        print 'python bindings.'
        print 'For maintainance email spdlib-develop@lists.sourceforge.net'

if __name__ == '__main__':
    obj = LASWave2UPD()
    obj.run()

