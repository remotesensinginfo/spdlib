/*
 *  SPDPulse.cpp
 *  spdlib
 *
 *  Created by Pete Bunting on 27/11/2010.
 *  Copyright 2010 SPDLib. All rights reserved.
 * 
 *  This file is part of SPDLib.
 *
 *  Permission is hereby granted, free of charge, to any person 
 *  obtaining a copy of this software and associated documentation 
 *  files (the "Software"), to deal in the Software without restriction, 
 *  including without limitation the rights to use, copy, modify, 
 *  merge, publish, distribute, sublicense, and/or sell copies of the 
 *  Software, and to permit persons to whom the Software is furnished 
 *  to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be 
 *  included in all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
 *  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
 *  OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
 *  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR 
 *  ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF 
 *  CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION 
 *  WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#include "spd/SPDPulse.h"


namespace spdlib
{
	
	SPDPulseUtils::SPDPulseUtils()
	{
		
	}
	
	H5::CompType* SPDPulseUtils::createSPDPulseH5V1DataTypeDisk()
	{
		H5::CompType *spdPulseDataType = new H5::CompType( sizeof(SPDPulseH5V1) );
		spdPulseDataType->insertMember(PULSEMEMBERNAME_PULSE_ID, HOFFSET(SPDPulseH5V1, pulseID), H5::PredType::STD_U64LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_GPS_TIME, HOFFSET(SPDPulseH5V1, gpsTime), H5::PredType::IEEE_F64LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_X_ORIGIN, HOFFSET(SPDPulseH5V1, x0), H5::PredType::IEEE_F64LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_Y_ORIGIN, HOFFSET(SPDPulseH5V1, y0), H5::PredType::IEEE_F64LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_Z_ORIGIN, HOFFSET(SPDPulseH5V1, z0), H5::PredType::IEEE_F32LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_H_ORIGIN, HOFFSET(SPDPulseH5V1, h0), H5::PredType::IEEE_F32LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_X_IDX, HOFFSET(SPDPulseH5V1, xIdx), H5::PredType::IEEE_F64LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_Y_IDX, HOFFSET(SPDPulseH5V1, yIdx), H5::PredType::IEEE_F64LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_AZIMUTH, HOFFSET(SPDPulseH5V1, azimuth), H5::PredType::IEEE_F32LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_ZENITH, HOFFSET(SPDPulseH5V1, zenith), H5::PredType::IEEE_F32LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_NUMBER_OF_RETURNS, HOFFSET(SPDPulseH5V1, numberOfReturns), H5::PredType::STD_U8LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_NUMBER_OF_WAVEFORM_TRANSMITTED_BINS, HOFFSET(SPDPulseH5V1, numOfTransmittedBins), H5::PredType::STD_U16LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_NUMBER_OF_WAVEFORM_RECEIVED_BINS, HOFFSET(SPDPulseH5V1, numOfReceivedBins), H5::PredType::STD_U16LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_RANGE_TO_WAVEFORM_START, HOFFSET(SPDPulseH5V1, rangeToWaveformStart), H5::PredType::IEEE_F32LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_AMPLITUDE_PULSE, HOFFSET(SPDPulseH5V1, amplitudePulse), H5::PredType::IEEE_F32LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_WIDTH_PULSE, HOFFSET(SPDPulseH5V1, widthPulse), H5::PredType::IEEE_F32LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_USER, HOFFSET(SPDPulseH5V1, user), H5::PredType::STD_U32LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_SOURCE_ID, HOFFSET(SPDPulseH5V1, sourceID), H5::PredType::STD_U16LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_EDGE_FLIGHT_LINE_FLAG, HOFFSET(SPDPulseH5V1, edgeFlightLineFlag), H5::PredType::STD_U8LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_SCAN_DIRECTION_FLAG, HOFFSET(SPDPulseH5V1, scanDirectionFlag), H5::PredType::STD_U8LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_SCAN_ANGLE_RANK, HOFFSET(SPDPulseH5V1, scanAngleRank), H5::PredType::IEEE_F32LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_WAVE_NOISE_THRES, HOFFSET(SPDPulseH5V1, waveNoiseThreshold), H5::PredType::IEEE_F32LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_RECEIVE_WAVE_GAIN, HOFFSET(SPDPulseH5V1, receiveWaveGain), H5::PredType::IEEE_F32LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_RECEIVE_WAVE_OFFSET, HOFFSET(SPDPulseH5V1, receiveWaveOffset), H5::PredType::IEEE_F32LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_TRANS_WAVE_GAIN, HOFFSET(SPDPulseH5V1, transWaveGain), H5::PredType::IEEE_F32LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_TRANS_WAVE_OFFSET, HOFFSET(SPDPulseH5V1, transWaveOffset), H5::PredType::IEEE_F32LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_PTS_START_IDX, HOFFSET(SPDPulseH5V1, ptsStartIdx), H5::PredType::STD_U64LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_RECEIVED_START_IDX, HOFFSET(SPDPulseH5V1, receivedStartIdx), H5::PredType::STD_U64LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_TRANSMITTED_START_IDX, HOFFSET(SPDPulseH5V1, transmittedStartIdx), H5::PredType::STD_U64LE);
		
		return spdPulseDataType;
	}
	
	H5::CompType* SPDPulseUtils::createSPDPulseH5V1DataTypeMemory()
	{
		H5::CompType *spdPulseDataType = new H5::CompType( sizeof(SPDPulseH5V1) );
		spdPulseDataType->insertMember(PULSEMEMBERNAME_PULSE_ID, HOFFSET(SPDPulseH5V1, pulseID), H5::PredType::NATIVE_ULLONG);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_GPS_TIME, HOFFSET(SPDPulseH5V1, gpsTime), H5::PredType::NATIVE_DOUBLE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_X_ORIGIN, HOFFSET(SPDPulseH5V1, x0), H5::PredType::NATIVE_DOUBLE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_Y_ORIGIN, HOFFSET(SPDPulseH5V1, y0), H5::PredType::NATIVE_DOUBLE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_Z_ORIGIN, HOFFSET(SPDPulseH5V1, z0), H5::PredType::NATIVE_FLOAT);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_H_ORIGIN, HOFFSET(SPDPulseH5V1, h0), H5::PredType::NATIVE_FLOAT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_X_IDX, HOFFSET(SPDPulseH5V1, xIdx), H5::PredType::NATIVE_DOUBLE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_Y_IDX, HOFFSET(SPDPulseH5V1, yIdx), H5::PredType::NATIVE_DOUBLE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_AZIMUTH, HOFFSET(SPDPulseH5V1, azimuth), H5::PredType::NATIVE_FLOAT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_ZENITH, HOFFSET(SPDPulseH5V1, zenith), H5::PredType::NATIVE_FLOAT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_NUMBER_OF_RETURNS, HOFFSET(SPDPulseH5V1, numberOfReturns), H5::PredType::NATIVE_UINT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_NUMBER_OF_WAVEFORM_TRANSMITTED_BINS, HOFFSET(SPDPulseH5V1, numOfTransmittedBins), H5::PredType::NATIVE_UINT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_NUMBER_OF_WAVEFORM_RECEIVED_BINS, HOFFSET(SPDPulseH5V1, numOfReceivedBins), H5::PredType::NATIVE_UINT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_RANGE_TO_WAVEFORM_START, HOFFSET(SPDPulseH5V1, rangeToWaveformStart), H5::PredType::NATIVE_FLOAT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_AMPLITUDE_PULSE, HOFFSET(SPDPulseH5V1, amplitudePulse), H5::PredType::NATIVE_FLOAT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_WIDTH_PULSE, HOFFSET(SPDPulseH5V1, widthPulse), H5::PredType::NATIVE_FLOAT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_USER, HOFFSET(SPDPulseH5V1, user), H5::PredType::NATIVE_ULONG);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_SOURCE_ID, HOFFSET(SPDPulseH5V1, sourceID), H5::PredType::NATIVE_UINT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_EDGE_FLIGHT_LINE_FLAG, HOFFSET(SPDPulseH5V1, edgeFlightLineFlag), H5::PredType::NATIVE_UINT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_SCAN_DIRECTION_FLAG, HOFFSET(SPDPulseH5V1, scanDirectionFlag), H5::PredType::NATIVE_UINT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_SCAN_ANGLE_RANK, HOFFSET(SPDPulseH5V1, scanAngleRank), H5::PredType::NATIVE_FLOAT);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_WAVE_NOISE_THRES, HOFFSET(SPDPulseH5V1, waveNoiseThreshold), H5::PredType::NATIVE_FLOAT);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_RECEIVE_WAVE_GAIN, HOFFSET(SPDPulseH5V1, receiveWaveGain), H5::PredType::NATIVE_FLOAT);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_RECEIVE_WAVE_OFFSET, HOFFSET(SPDPulseH5V1, receiveWaveOffset), H5::PredType::NATIVE_FLOAT);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_TRANS_WAVE_GAIN, HOFFSET(SPDPulseH5V1, transWaveGain), H5::PredType::NATIVE_FLOAT);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_TRANS_WAVE_OFFSET, HOFFSET(SPDPulseH5V1, transWaveOffset), H5::PredType::NATIVE_FLOAT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_PTS_START_IDX, HOFFSET(SPDPulseH5V1, ptsStartIdx), H5::PredType::NATIVE_ULLONG);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_RECEIVED_START_IDX, HOFFSET(SPDPulseH5V1, receivedStartIdx), H5::PredType::NATIVE_ULLONG);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_TRANSMITTED_START_IDX, HOFFSET(SPDPulseH5V1, transmittedStartIdx), H5::PredType::NATIVE_ULLONG);
		
		return spdPulseDataType;
	}
    
    H5::CompType* SPDPulseUtils::createSPDPulseH5V2DataTypeDisk()
	{
		H5::CompType *spdPulseDataType = new H5::CompType( sizeof(SPDPulseH5V2) );
		spdPulseDataType->insertMember(PULSEMEMBERNAME_PULSE_ID, HOFFSET(SPDPulseH5V2, pulseID), H5::PredType::STD_U64LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_GPS_TIME, HOFFSET(SPDPulseH5V2, gpsTime), H5::PredType::STD_U64LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_X_ORIGIN, HOFFSET(SPDPulseH5V2, x0), H5::PredType::IEEE_F64LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_Y_ORIGIN, HOFFSET(SPDPulseH5V2, y0), H5::PredType::IEEE_F64LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_Z_ORIGIN, HOFFSET(SPDPulseH5V2, z0), H5::PredType::IEEE_F32LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_H_ORIGIN, HOFFSET(SPDPulseH5V2, h0), H5::PredType::IEEE_F32LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_X_IDX, HOFFSET(SPDPulseH5V2, xIdx), H5::PredType::IEEE_F64LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_Y_IDX, HOFFSET(SPDPulseH5V2, yIdx), H5::PredType::IEEE_F64LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_AZIMUTH, HOFFSET(SPDPulseH5V2, azimuth), H5::PredType::IEEE_F32LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_ZENITH, HOFFSET(SPDPulseH5V2, zenith), H5::PredType::IEEE_F32LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_NUMBER_OF_RETURNS, HOFFSET(SPDPulseH5V2, numberOfReturns), H5::PredType::STD_U8LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_NUMBER_OF_WAVEFORM_TRANSMITTED_BINS, HOFFSET(SPDPulseH5V2, numOfTransmittedBins), H5::PredType::STD_U16LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_NUMBER_OF_WAVEFORM_RECEIVED_BINS, HOFFSET(SPDPulseH5V2, numOfReceivedBins), H5::PredType::STD_U16LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_RANGE_TO_WAVEFORM_START, HOFFSET(SPDPulseH5V2, rangeToWaveformStart), H5::PredType::IEEE_F32LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_AMPLITUDE_PULSE, HOFFSET(SPDPulseH5V2, amplitudePulse), H5::PredType::IEEE_F32LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_WIDTH_PULSE, HOFFSET(SPDPulseH5V2, widthPulse), H5::PredType::IEEE_F32LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_USER, HOFFSET(SPDPulseH5V2, user), H5::PredType::STD_U32LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_SOURCE_ID, HOFFSET(SPDPulseH5V2, sourceID), H5::PredType::STD_U16LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_SCANLINE, HOFFSET(SPDPulseH5V2, scanline), H5::PredType::STD_U32LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_SCANLINE_IDX, HOFFSET(SPDPulseH5V2, scanlineIdx), H5::PredType::STD_U16LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_RECEIVE_WAVE_NOISE_THRES, HOFFSET(SPDPulseH5V2, receiveWaveNoiseThreshold), H5::PredType::IEEE_F32LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_TRANS_WAVE_NOISE_THRES, HOFFSET(SPDPulseH5V2, transWaveNoiseThres), H5::PredType::IEEE_F32LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_WAVELENGTH, HOFFSET(SPDPulseH5V2, wavelength), H5::PredType::IEEE_F32LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_RECEIVE_WAVE_GAIN, HOFFSET(SPDPulseH5V2, receiveWaveGain), H5::PredType::IEEE_F32LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_RECEIVE_WAVE_OFFSET, HOFFSET(SPDPulseH5V2, receiveWaveOffset), H5::PredType::IEEE_F32LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_TRANS_WAVE_GAIN, HOFFSET(SPDPulseH5V2, transWaveGain), H5::PredType::IEEE_F32LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_TRANS_WAVE_OFFSET, HOFFSET(SPDPulseH5V2, transWaveOffset), H5::PredType::IEEE_F32LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_PTS_START_IDX, HOFFSET(SPDPulseH5V2, ptsStartIdx), H5::PredType::STD_U64LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_RECEIVED_START_IDX, HOFFSET(SPDPulseH5V2, receivedStartIdx), H5::PredType::STD_U64LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_TRANSMITTED_START_IDX, HOFFSET(SPDPulseH5V2, transmittedStartIdx), H5::PredType::STD_U64LE);
		
		return spdPulseDataType;
	}
	
	H5::CompType* SPDPulseUtils::createSPDPulseH5V2DataTypeMemory()
	{
		H5::CompType *spdPulseDataType = new H5::CompType( sizeof(SPDPulseH5V2) );
		spdPulseDataType->insertMember(PULSEMEMBERNAME_PULSE_ID, HOFFSET(SPDPulseH5V2, pulseID), H5::PredType::NATIVE_ULLONG);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_GPS_TIME, HOFFSET(SPDPulseH5V2, gpsTime), H5::PredType::NATIVE_ULLONG);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_X_ORIGIN, HOFFSET(SPDPulseH5V2, x0), H5::PredType::NATIVE_DOUBLE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_Y_ORIGIN, HOFFSET(SPDPulseH5V2, y0), H5::PredType::NATIVE_DOUBLE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_Z_ORIGIN, HOFFSET(SPDPulseH5V2, z0), H5::PredType::NATIVE_FLOAT);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_H_ORIGIN, HOFFSET(SPDPulseH5V2, h0), H5::PredType::NATIVE_FLOAT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_X_IDX, HOFFSET(SPDPulseH5V2, xIdx), H5::PredType::NATIVE_DOUBLE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_Y_IDX, HOFFSET(SPDPulseH5V2, yIdx), H5::PredType::NATIVE_DOUBLE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_AZIMUTH, HOFFSET(SPDPulseH5V2, azimuth), H5::PredType::NATIVE_FLOAT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_ZENITH, HOFFSET(SPDPulseH5V2, zenith), H5::PredType::NATIVE_FLOAT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_NUMBER_OF_RETURNS, HOFFSET(SPDPulseH5V2, numberOfReturns), H5::PredType::NATIVE_UINT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_NUMBER_OF_WAVEFORM_TRANSMITTED_BINS, HOFFSET(SPDPulseH5V2, numOfTransmittedBins), H5::PredType::NATIVE_UINT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_NUMBER_OF_WAVEFORM_RECEIVED_BINS, HOFFSET(SPDPulseH5V2, numOfReceivedBins), H5::PredType::NATIVE_UINT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_RANGE_TO_WAVEFORM_START, HOFFSET(SPDPulseH5V2, rangeToWaveformStart), H5::PredType::NATIVE_FLOAT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_AMPLITUDE_PULSE, HOFFSET(SPDPulseH5V2, amplitudePulse), H5::PredType::NATIVE_FLOAT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_WIDTH_PULSE, HOFFSET(SPDPulseH5V2, widthPulse), H5::PredType::NATIVE_FLOAT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_USER, HOFFSET(SPDPulseH5V2, user), H5::PredType::NATIVE_ULONG);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_SOURCE_ID, HOFFSET(SPDPulseH5V2, sourceID), H5::PredType::NATIVE_UINT);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_SCANLINE, HOFFSET(SPDPulseH5V2, scanline), H5::PredType::NATIVE_ULONG);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_SCANLINE_IDX, HOFFSET(SPDPulseH5V2, scanlineIdx), H5::PredType::NATIVE_UINT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_RECEIVE_WAVE_NOISE_THRES, HOFFSET(SPDPulseH5V2, receiveWaveNoiseThreshold), H5::PredType::NATIVE_FLOAT);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_TRANS_WAVE_NOISE_THRES, HOFFSET(SPDPulseH5V2, transWaveNoiseThres), H5::PredType::NATIVE_FLOAT);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_WAVELENGTH, HOFFSET(SPDPulseH5V2, wavelength), H5::PredType::NATIVE_FLOAT);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_RECEIVE_WAVE_GAIN, HOFFSET(SPDPulseH5V2, receiveWaveGain), H5::PredType::NATIVE_FLOAT);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_RECEIVE_WAVE_OFFSET, HOFFSET(SPDPulseH5V2, receiveWaveOffset), H5::PredType::NATIVE_FLOAT);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_TRANS_WAVE_GAIN, HOFFSET(SPDPulseH5V2, transWaveGain), H5::PredType::NATIVE_FLOAT);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_TRANS_WAVE_OFFSET, HOFFSET(SPDPulseH5V2, transWaveOffset), H5::PredType::NATIVE_FLOAT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_PTS_START_IDX, HOFFSET(SPDPulseH5V2, ptsStartIdx), H5::PredType::NATIVE_ULLONG);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_RECEIVED_START_IDX, HOFFSET(SPDPulseH5V2, receivedStartIdx), H5::PredType::NATIVE_ULLONG);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_TRANSMITTED_START_IDX, HOFFSET(SPDPulseH5V2, transmittedStartIdx), H5::PredType::NATIVE_ULLONG);
		
		return spdPulseDataType;
	}

	void SPDPulseUtils::initSPDPulse(SPDPulse *pl)
	{		
		pl->x0 = 0;
		pl->y0 = 0;
		pl->z0 = 0;
		pl->h0 = 0;
		pl->xIdx = 0;
		pl->yIdx = 0;
		pl->azimuth = 0;
		pl->zenith = 0;
		pl->pulseID = 0;
		pl->gpsTime = 0;
		pl->numberOfReturns = 0;
		pl->numOfTransmittedBins = 0;
		pl->numOfReceivedBins = 0;
		pl->rangeToWaveformStart = 0;
		pl->amplitudePulse = 0;
		pl->widthPulse = 0;
		pl->user = 0;
		pl->sourceID = 0;
        pl->scanline = 0;
        pl->scanlineIdx = 0;
		pl->edgeFlightLineFlag = SPD_UNDEFINED;
		pl->scanDirectionFlag = SPD_UNDEFINED;
		pl->scanAngleRank = 0;
        pl->receiveWaveNoiseThreshold = 0;
        pl->transWaveNoiseThres = 0;
        pl->wavelength = 0;
        pl->receiveWaveGain = 1;
        pl->receiveWaveOffset = 0;
        pl->transWaveGain = 1;
        pl->transWaveOffset = 0;
		pl->pts = new std::vector<SPDPoint*>();
		pl->transmitted = NULL;
		pl->received = NULL;
	}
    
	void SPDPulseUtils::initSPDPulseH5(SPDPulseH5V1 *pl)
	{		
		pl->x0 = 0;
		pl->y0 = 0;
		pl->z0 = 0;	
        pl->h0 = 0;	
		pl->xIdx = 0;
		pl->yIdx = 0;
		pl->azimuth = 0;
		pl->zenith = 0;
		pl->pulseID = 0;
		pl->gpsTime = 0;
		pl->numberOfReturns = 0;
		pl->numOfTransmittedBins = 0;
		pl->numOfReceivedBins = 0;
		pl->rangeToWaveformStart = 0;
		pl->amplitudePulse = 0;
		pl->widthPulse = 0;
		pl->user = 0;
		pl->sourceID = 0;
		pl->edgeFlightLineFlag = SPD_UNDEFINED;
		pl->scanDirectionFlag = SPD_UNDEFINED;
		pl->scanAngleRank = 0;
        pl->waveNoiseThreshold = 0;
        pl->receiveWaveGain = 1;
        pl->receiveWaveOffset = 0;
        pl->transWaveGain = 1;
        pl->transWaveOffset = 0;
		pl->ptsStartIdx = 0;
		pl->receivedStartIdx = 0;
		pl->transmittedStartIdx = 0;
	}
    
    void SPDPulseUtils::initSPDPulseH5(SPDPulseH5V2 *pl)
	{		
		pl->x0 = 0;
		pl->y0 = 0;
		pl->z0 = 0;	
        pl->h0 = 0;	
		pl->xIdx = 0;
		pl->yIdx = 0;
		pl->azimuth = 0;
		pl->zenith = 0;
		pl->pulseID = 0;
		pl->gpsTime = 0;
		pl->numberOfReturns = 0;
		pl->numOfTransmittedBins = 0;
		pl->numOfReceivedBins = 0;
		pl->rangeToWaveformStart = 0;
		pl->amplitudePulse = 0;
		pl->widthPulse = 0;
		pl->user = 0;
		pl->sourceID = 0;
        pl->scanline = 0;
        pl->scanlineIdx = 0;
        pl->receiveWaveNoiseThreshold = 0;
        pl->transWaveNoiseThres = 0;
        pl->wavelength = 0;
        pl->receiveWaveGain = 1;
        pl->receiveWaveOffset = 0;
        pl->transWaveGain = 1;
        pl->transWaveOffset = 0;
		pl->ptsStartIdx = 0;
		pl->receivedStartIdx = 0;
		pl->transmittedStartIdx = 0;
	}
	
	void SPDPulseUtils::deleteSPDPulse(SPDPulse *pl)
	{
		if(pl != NULL)
        {
            if(pl->numberOfReturns > 0)
            {
                for(boost::uint_fast16_t i = 0; i < pl->pts->size(); ++i)
                {
                    delete pl->pts->at(i);
                }
            }
            delete pl->pts;
            
            if(pl->numOfTransmittedBins > 0)
            {
                delete[] pl->transmitted;
            }
            
            if(pl->numOfReceivedBins > 0)
            {
                delete[] pl->received;
            }
            
            delete pl;
        }
	}
    
	SPDPulse* SPDPulseUtils::createSPDPulseCopy(SPDPulse *pl)
	{
		SPDPulse *pl_out = new SPDPulse();
		pl_out->x0 = pl->x0;
		pl_out->y0 = pl->y0;
		pl_out->z0 = pl->z0;
        pl_out->h0 = pl->h0;
		pl_out->xIdx = pl->xIdx;
		pl_out->yIdx = pl->yIdx;
		pl_out->azimuth = pl->azimuth;
		pl_out->zenith = pl->zenith;
		pl_out->pulseID = pl->pulseID;
		pl_out->gpsTime = pl->gpsTime;
		pl_out->numberOfReturns = pl->numberOfReturns;
		pl_out->pts = pl->pts;
		pl_out->transmitted = pl->transmitted;
		pl_out->received = pl->received;
		pl_out->numOfTransmittedBins = pl->numOfTransmittedBins;
		pl_out->numOfReceivedBins = pl->numOfReceivedBins;
		pl_out->rangeToWaveformStart = pl->rangeToWaveformStart;
		pl_out->amplitudePulse = pl->amplitudePulse;
		pl_out->widthPulse = pl->widthPulse;
		pl_out->user = pl->user;
		pl_out->sourceID = pl->sourceID;
        pl_out->scanline = pl->scanline;
        pl_out->scanlineIdx = pl->scanlineIdx;
		pl_out->edgeFlightLineFlag = pl->edgeFlightLineFlag;
		pl_out->scanDirectionFlag = pl->scanDirectionFlag;
		pl_out->scanAngleRank = pl->scanAngleRank;
        pl_out->receiveWaveNoiseThreshold = pl->receiveWaveNoiseThreshold;
        pl_out->transWaveNoiseThres = pl->transWaveNoiseThres;
        pl_out->wavelength = pl->wavelength;
        pl_out->receiveWaveGain = pl->receiveWaveGain;
        pl_out->receiveWaveOffset = pl->receiveWaveOffset;
        pl_out->transWaveGain = pl->transWaveGain;
        pl_out->transWaveOffset = pl->transWaveOffset;
		return pl_out;
	}
	
	void SPDPulseUtils::copySPDPulseTo(SPDPulse *pl, SPDPulse *pl_out)
	{
		pl_out->x0 = pl->x0;
		pl_out->y0 = pl->y0;
		pl_out->z0 = pl->z0;
        pl_out->h0 = pl->h0;
		pl_out->xIdx = pl->xIdx;
		pl_out->yIdx = pl->yIdx;
		pl_out->azimuth = pl->azimuth;
		pl_out->zenith = pl->zenith;
		pl_out->pulseID = pl->pulseID;
		pl_out->gpsTime = pl->gpsTime;
		pl_out->numberOfReturns = pl->numberOfReturns;
		pl_out->pts = pl->pts;
		pl_out->transmitted = pl->transmitted;
		pl_out->received = pl->received;
		pl_out->numOfTransmittedBins = pl->numOfTransmittedBins;
		pl_out->numOfReceivedBins = pl->numOfReceivedBins;
		pl_out->rangeToWaveformStart = pl->rangeToWaveformStart;
		pl_out->amplitudePulse = pl->amplitudePulse;
		pl_out->widthPulse = pl->widthPulse;
		pl_out->user = pl->user;
		pl_out->sourceID = pl->sourceID;
        pl_out->scanline = pl->scanline;
        pl_out->scanlineIdx = pl->scanlineIdx;
		pl_out->edgeFlightLineFlag = pl->edgeFlightLineFlag;
		pl_out->scanDirectionFlag = pl->scanDirectionFlag;
		pl_out->scanAngleRank = pl->scanAngleRank;
        pl_out->receiveWaveNoiseThreshold = pl->receiveWaveNoiseThreshold;
        pl_out->transWaveNoiseThres = pl->transWaveNoiseThres;
        pl_out->wavelength = pl->wavelength;
        pl_out->receiveWaveGain = pl->receiveWaveGain;
        pl_out->receiveWaveOffset = pl->receiveWaveOffset;
        pl_out->transWaveGain = pl->transWaveGain;
        pl_out->transWaveOffset = pl->transWaveOffset;
	}
	
	SPDPulse* SPDPulseUtils::createSPDPulseDeepCopy(SPDPulse *pl)
	{
		SPDPointUtils ptUtils;
		
		SPDPulse *pl_out = new SPDPulse();
		pl_out->x0 = pl->x0;
		pl_out->y0 = pl->y0;
		pl_out->z0 = pl->z0;
        pl_out->h0 = pl->h0;
		pl_out->xIdx = pl->xIdx;
		pl_out->yIdx = pl->yIdx;
		pl_out->azimuth = pl->azimuth;
		pl_out->zenith = pl->zenith;
		pl_out->pulseID = pl->pulseID;
		pl_out->gpsTime = pl->gpsTime;
		pl_out->numberOfReturns = pl->numberOfReturns;
		pl_out->pts = new std::vector<SPDPoint*>();
        if(pl->numberOfReturns > 0)
        {
            pl_out->pts->reserve(pl->numberOfReturns);
            //for(boost::uint_fast16_t i = 0; i < pl->numberOfReturns; ++i)
            for(std::vector<SPDPoint*>::iterator iterPts = pl->pts->begin(); iterPts != pl->pts->end(); ++iterPts)
            {
                pl_out->pts->push_back(ptUtils.createSPDPointCopy(*iterPts));//(pl->pts->at(i)));
            }
        }
		pl_out->transmitted = new boost::uint_fast32_t[pl->numOfTransmittedBins];
		for(boost::uint_fast16_t i = 0; i < pl->numOfTransmittedBins; ++i)
		{
			pl_out->transmitted[i] = pl->transmitted[i];
		}
		pl_out->received = new boost::uint_fast32_t[pl->numOfReceivedBins];
		for(boost::uint_fast16_t i = 0; i < pl->numOfReceivedBins; ++i)
		{
			pl_out->received[i] = pl->received[i];
		}
		pl_out->numOfTransmittedBins = pl->numOfTransmittedBins;
		pl_out->numOfReceivedBins = pl->numOfReceivedBins;
		pl_out->rangeToWaveformStart = pl->rangeToWaveformStart;
		pl_out->amplitudePulse = pl->amplitudePulse;
		pl_out->widthPulse = pl->widthPulse;
		pl_out->user = pl->user;
		pl_out->sourceID = pl->sourceID;
        pl_out->scanline = pl->scanline;
        pl_out->scanlineIdx = pl->scanlineIdx;
		pl_out->edgeFlightLineFlag = pl->edgeFlightLineFlag;
		pl_out->scanDirectionFlag = pl->scanDirectionFlag;
		pl_out->scanAngleRank = pl->scanAngleRank;
        pl_out->receiveWaveNoiseThreshold = pl->receiveWaveNoiseThreshold;
        pl_out->transWaveNoiseThres = pl->transWaveNoiseThres;
        pl_out->wavelength = pl->wavelength;
        pl_out->receiveWaveGain = pl->receiveWaveGain;
        pl_out->receiveWaveOffset = pl->receiveWaveOffset;
        pl_out->transWaveGain = pl->transWaveGain;
        pl_out->transWaveOffset = pl->transWaveOffset;
		return pl_out;
	}
	
	void SPDPulseUtils::deepCopySPDPulseTo(SPDPulse *pl, SPDPulse *pl_out)
	{
		SPDPointUtils ptUtils;
		
		pl_out->x0 = pl->x0;
		pl_out->y0 = pl->y0;
		pl_out->z0 = pl->z0;
        pl_out->h0 = pl->h0;
		pl_out->xIdx = pl->xIdx;
		pl_out->yIdx = pl->yIdx;
		pl_out->azimuth = pl->azimuth;
		pl_out->zenith = pl->zenith;
		pl_out->pulseID = pl->pulseID;
		pl_out->gpsTime = pl->gpsTime;
		pl_out->numberOfReturns = pl->numberOfReturns;
		pl_out->pts = new std::vector<SPDPoint*>();
		pl_out->pts->reserve(pl->numberOfReturns);
		for(boost::uint_fast16_t i = 0; i < pl->numberOfReturns; ++i)
		{
			pl_out->pts->at(i) = ptUtils.createSPDPointCopy(pl->pts->at(i));
		}
		pl_out->transmitted = new boost::uint_fast32_t[pl->numOfTransmittedBins];
		for(boost::uint_fast16_t i = 0; i < pl->numOfTransmittedBins; ++i)
		{
			pl_out->transmitted[i] = pl->transmitted[i];
		}
		pl_out->received = new boost::uint_fast32_t[pl->numOfReceivedBins];
		for(boost::uint_fast16_t i = 0; i < pl->numOfReceivedBins; ++i)
		{
			pl_out->received[i] = pl->received[i];
		}
		pl_out->numOfTransmittedBins = pl->numOfTransmittedBins;
		pl_out->numOfReceivedBins = pl->numOfReceivedBins;
		pl_out->rangeToWaveformStart = pl->rangeToWaveformStart;
		pl_out->amplitudePulse = pl->amplitudePulse;
		pl_out->widthPulse = pl->widthPulse;
		pl_out->user = pl->user;
		pl_out->sourceID = pl->sourceID;
        pl_out->scanline = pl->scanline;
        pl_out->scanlineIdx = pl->scanlineIdx;
		pl_out->edgeFlightLineFlag = pl->edgeFlightLineFlag;
		pl_out->scanDirectionFlag = pl->scanDirectionFlag;
		pl_out->scanAngleRank = pl->scanAngleRank;
        pl_out->receiveWaveNoiseThreshold = pl->receiveWaveNoiseThreshold;
        pl_out->transWaveNoiseThres = pl->transWaveNoiseThres;
        pl_out->wavelength = pl->wavelength;
        pl_out->receiveWaveGain = pl->receiveWaveGain;
        pl_out->receiveWaveOffset = pl->receiveWaveOffset;
        pl_out->transWaveGain = pl->transWaveGain;
        pl_out->transWaveOffset = pl->transWaveOffset;
	}
	
	SPDPulseH5V1* SPDPulseUtils::createSPDPulseH5V1Copy(SPDPulse *pl)
	{
		SPDPulseH5V1 *pl_out = new SPDPulseH5V1();
		pl_out->x0 = pl->x0;
		pl_out->y0 = pl->y0;
		pl_out->z0 = pl->z0;
        pl_out->h0 = pl->h0;
		pl_out->xIdx = pl->xIdx;
		pl_out->yIdx = pl->yIdx;
		pl_out->azimuth = pl->azimuth;
		pl_out->zenith = pl->zenith;
		pl_out->pulseID = pl->pulseID;
		pl_out->gpsTime = pl->gpsTime;
		pl_out->numberOfReturns = pl->numberOfReturns;
		pl_out->numOfTransmittedBins = pl->numOfTransmittedBins;
		pl_out->numOfReceivedBins = pl->numOfReceivedBins;
		pl_out->rangeToWaveformStart = pl->rangeToWaveformStart;
		pl_out->amplitudePulse = pl->amplitudePulse;
		pl_out->widthPulse = pl->widthPulse;
		pl_out->user = pl->user;
		pl_out->sourceID = pl->sourceID;
		pl_out->edgeFlightLineFlag = pl->edgeFlightLineFlag;
		pl_out->scanDirectionFlag = pl->scanDirectionFlag;
		pl_out->scanAngleRank = pl->scanAngleRank;
        pl_out->waveNoiseThreshold = pl->receiveWaveNoiseThreshold;
        pl_out->receiveWaveGain = pl->receiveWaveGain;
        pl_out->receiveWaveOffset = pl->receiveWaveOffset;
        pl_out->transWaveGain = pl->transWaveGain;
        pl_out->transWaveOffset = pl->transWaveOffset;
		return pl_out;
	}
	
	void SPDPulseUtils::copySPDPulseToSPDPulseH5(SPDPulse *pl, SPDPulseH5V1 *pl_out)
	{
		pl_out->x0 = pl->x0;
		pl_out->y0 = pl->y0;
		pl_out->z0 = pl->z0;
        pl_out->h0 = pl->h0;
		pl_out->xIdx = pl->xIdx;
		pl_out->yIdx = pl->yIdx;
		pl_out->azimuth = pl->azimuth;
		pl_out->zenith = pl->zenith;
		pl_out->pulseID = pl->pulseID;
		pl_out->gpsTime = pl->gpsTime;
		pl_out->numberOfReturns = pl->numberOfReturns;
		pl_out->numOfTransmittedBins = pl->numOfTransmittedBins;
		pl_out->numOfReceivedBins = pl->numOfReceivedBins;
		pl_out->rangeToWaveformStart = pl->rangeToWaveformStart;
		pl_out->amplitudePulse = pl->amplitudePulse;
		pl_out->widthPulse = pl->widthPulse;
		pl_out->user = pl->user;
		pl_out->sourceID = pl->sourceID;
		pl_out->edgeFlightLineFlag = pl->edgeFlightLineFlag;
		pl_out->scanDirectionFlag = pl->scanDirectionFlag;
		pl_out->scanAngleRank = pl->scanAngleRank;
        pl_out->waveNoiseThreshold = pl->receiveWaveNoiseThreshold;
        pl_out->receiveWaveGain = pl->receiveWaveGain;
        pl_out->receiveWaveOffset = pl->receiveWaveOffset;
        pl_out->transWaveGain = pl->transWaveGain;
        pl_out->transWaveOffset = pl->transWaveOffset;
	}
	
	SPDPulse* SPDPulseUtils::createSPDPulseCopyFromH5(SPDPulseH5V1 *pl)
	{
		SPDPulse *pl_out = new SPDPulse();
		pl_out->x0 = pl->x0;
		pl_out->y0 = pl->y0;
		pl_out->z0 = pl->z0;
        pl_out->h0 = pl->h0;
		pl_out->xIdx = pl->xIdx;
		pl_out->yIdx = pl->yIdx;
		pl_out->azimuth = pl->azimuth;
		pl_out->zenith = pl->zenith;
		pl_out->pulseID = pl->pulseID;
		pl_out->gpsTime = pl->gpsTime;
		pl_out->numberOfReturns = pl->numberOfReturns;
		pl_out->numOfTransmittedBins = pl->numOfTransmittedBins;
		pl_out->numOfReceivedBins = pl->numOfReceivedBins;
		pl_out->rangeToWaveformStart = pl->rangeToWaveformStart;
		pl_out->amplitudePulse = pl->amplitudePulse;
		pl_out->widthPulse = pl->widthPulse;
		pl_out->user = pl->user;
		pl_out->sourceID = pl->sourceID;
		pl_out->edgeFlightLineFlag = pl->edgeFlightLineFlag;
		pl_out->scanDirectionFlag = pl->scanDirectionFlag;
		pl_out->scanAngleRank = pl->scanAngleRank;
        pl_out->receiveWaveNoiseThreshold = pl->waveNoiseThreshold;
        pl_out->receiveWaveGain = pl->receiveWaveGain;
        pl_out->receiveWaveOffset = pl->receiveWaveOffset;
        pl_out->transWaveGain = pl->transWaveGain;
        pl_out->transWaveOffset = pl->transWaveOffset;
		return pl_out;
	}
	
	void SPDPulseUtils::copySPDPulseH5ToSPDPulse(SPDPulseH5V1 *pl, SPDPulse *pl_out)
	{
		pl_out->x0 = pl->x0;
		pl_out->y0 = pl->y0;
		pl_out->z0 = pl->z0;
        pl_out->h0 = pl->h0;
		pl_out->xIdx = pl->xIdx;
		pl_out->yIdx = pl->yIdx;
		pl_out->azimuth = pl->azimuth;
		pl_out->zenith = pl->zenith;
		pl_out->pulseID = pl->pulseID;
		pl_out->gpsTime = pl->gpsTime;
		pl_out->numberOfReturns = pl->numberOfReturns;
		pl_out->numOfTransmittedBins = pl->numOfTransmittedBins;
		pl_out->numOfReceivedBins = pl->numOfReceivedBins;
		pl_out->rangeToWaveformStart = pl->rangeToWaveformStart;
		pl_out->amplitudePulse = pl->amplitudePulse;
		pl_out->widthPulse = pl->widthPulse;
		pl_out->user = pl->user;
		pl_out->sourceID = pl->sourceID;
		pl_out->edgeFlightLineFlag = pl->edgeFlightLineFlag;
		pl_out->scanDirectionFlag = pl->scanDirectionFlag;
		pl_out->scanAngleRank = pl->scanAngleRank;
        pl_out->receiveWaveNoiseThreshold = pl->waveNoiseThreshold;
        pl_out->receiveWaveGain = pl->receiveWaveGain;
        pl_out->receiveWaveOffset = pl->receiveWaveOffset;
        pl_out->transWaveGain = pl->transWaveGain;
        pl_out->transWaveOffset = pl->transWaveOffset;
	}
    
    SPDPulseH5V2* SPDPulseUtils::createSPDPulseH5V2Copy(SPDPulse *pl)
	{
		SPDPulseH5V2 *pl_out = new SPDPulseH5V2();
		pl_out->x0 = pl->x0;
		pl_out->y0 = pl->y0;
		pl_out->z0 = pl->z0;
        pl_out->h0 = pl->h0;
		pl_out->xIdx = pl->xIdx;
		pl_out->yIdx = pl->yIdx;
		pl_out->azimuth = pl->azimuth;
		pl_out->zenith = pl->zenith;
		pl_out->pulseID = pl->pulseID;
		pl_out->gpsTime = pl->gpsTime;
		pl_out->numberOfReturns = pl->numberOfReturns;
		pl_out->numOfTransmittedBins = pl->numOfTransmittedBins;
		pl_out->numOfReceivedBins = pl->numOfReceivedBins;
		pl_out->rangeToWaveformStart = pl->rangeToWaveformStart;
		pl_out->amplitudePulse = pl->amplitudePulse;
		pl_out->widthPulse = pl->widthPulse;
		pl_out->user = pl->user;
		pl_out->sourceID = pl->sourceID;
        pl_out->scanline = pl->scanline;
        pl_out->scanlineIdx = pl->scanlineIdx;
        pl_out->receiveWaveNoiseThreshold = pl->receiveWaveNoiseThreshold;
        pl_out->transWaveNoiseThres = pl->transWaveNoiseThres;
        pl_out->wavelength = pl->wavelength;
        pl_out->receiveWaveGain = pl->receiveWaveGain;
        pl_out->receiveWaveOffset = pl->receiveWaveOffset;
        pl_out->transWaveGain = pl->transWaveGain;
        pl_out->transWaveOffset = pl->transWaveOffset;
		return pl_out;
	}
	
	void SPDPulseUtils::copySPDPulseToSPDPulseH5(SPDPulse *pl, SPDPulseH5V2 *pl_out)
	{
		pl_out->x0 = pl->x0;
		pl_out->y0 = pl->y0;
		pl_out->z0 = pl->z0;
        pl_out->h0 = pl->h0;
		pl_out->xIdx = pl->xIdx;
		pl_out->yIdx = pl->yIdx;
		pl_out->azimuth = pl->azimuth;
		pl_out->zenith = pl->zenith;
		pl_out->pulseID = pl->pulseID;
		pl_out->gpsTime = pl->gpsTime;
		pl_out->numberOfReturns = pl->numberOfReturns;
		pl_out->numOfTransmittedBins = pl->numOfTransmittedBins;
		pl_out->numOfReceivedBins = pl->numOfReceivedBins;
		pl_out->rangeToWaveformStart = pl->rangeToWaveformStart;
		pl_out->amplitudePulse = pl->amplitudePulse;
		pl_out->widthPulse = pl->widthPulse;
		pl_out->user = pl->user;
		pl_out->sourceID = pl->sourceID;
        pl_out->scanline = pl->scanline;
        pl_out->scanlineIdx = pl->scanlineIdx;
		pl_out->receiveWaveNoiseThreshold = pl->receiveWaveNoiseThreshold;
        pl_out->transWaveNoiseThres = pl->transWaveNoiseThres;
        pl_out->wavelength = pl->wavelength;
        pl_out->receiveWaveGain = pl->receiveWaveGain;
        pl_out->receiveWaveOffset = pl->receiveWaveOffset;
        pl_out->transWaveGain = pl->transWaveGain;
        pl_out->transWaveOffset = pl->transWaveOffset;
	}
	
	SPDPulse* SPDPulseUtils::createSPDPulseCopyFromH5(SPDPulseH5V2 *pl)
	{
		SPDPulse *pl_out = new SPDPulse();
		pl_out->x0 = pl->x0;
		pl_out->y0 = pl->y0;
		pl_out->z0 = pl->z0;
        pl_out->h0 = pl->h0;
		pl_out->xIdx = pl->xIdx;
		pl_out->yIdx = pl->yIdx;
		pl_out->azimuth = pl->azimuth;
		pl_out->zenith = pl->zenith;
		pl_out->pulseID = pl->pulseID;
		pl_out->gpsTime = pl->gpsTime;
		pl_out->numberOfReturns = pl->numberOfReturns;
		pl_out->numOfTransmittedBins = pl->numOfTransmittedBins;
		pl_out->numOfReceivedBins = pl->numOfReceivedBins;
		pl_out->rangeToWaveformStart = pl->rangeToWaveformStart;
		pl_out->amplitudePulse = pl->amplitudePulse;
		pl_out->widthPulse = pl->widthPulse;
		pl_out->user = pl->user;
		pl_out->sourceID = pl->sourceID;
        pl_out->scanline = pl->scanline;
        pl_out->scanlineIdx = pl->scanlineIdx;
		pl_out->receiveWaveNoiseThreshold = pl->receiveWaveNoiseThreshold;
        pl_out->transWaveNoiseThres = pl->transWaveNoiseThres;
        pl_out->wavelength = pl->wavelength;
        pl_out->receiveWaveGain = pl->receiveWaveGain;
        pl_out->receiveWaveOffset = pl->receiveWaveOffset;
        pl_out->transWaveGain = pl->transWaveGain;
        pl_out->transWaveOffset = pl->transWaveOffset;
		return pl_out;
	}
	
	void SPDPulseUtils::copySPDPulseH5ToSPDPulse(SPDPulseH5V2 *pl, SPDPulse *pl_out)
	{
		pl_out->x0 = pl->x0;
		pl_out->y0 = pl->y0;
		pl_out->z0 = pl->z0;
        pl_out->h0 = pl->h0;
		pl_out->xIdx = pl->xIdx;
		pl_out->yIdx = pl->yIdx;
		pl_out->azimuth = pl->azimuth;
		pl_out->zenith = pl->zenith;
		pl_out->pulseID = pl->pulseID;
		pl_out->gpsTime = pl->gpsTime;
		pl_out->numberOfReturns = pl->numberOfReturns;
		pl_out->numOfTransmittedBins = pl->numOfTransmittedBins;
		pl_out->numOfReceivedBins = pl->numOfReceivedBins;
		pl_out->rangeToWaveformStart = pl->rangeToWaveformStart;
		pl_out->amplitudePulse = pl->amplitudePulse;
		pl_out->widthPulse = pl->widthPulse;
		pl_out->user = pl->user;
		pl_out->sourceID = pl->sourceID;
        pl_out->scanline = pl->scanline;
        pl_out->scanlineIdx = pl->scanlineIdx;
		pl_out->receiveWaveNoiseThreshold = pl->receiveWaveNoiseThreshold;
        pl_out->transWaveNoiseThres = pl->transWaveNoiseThres;
        pl_out->wavelength = pl->wavelength;
        pl_out->receiveWaveGain = pl->receiveWaveGain;
        pl_out->receiveWaveOffset = pl->receiveWaveOffset;
        pl_out->transWaveGain = pl->transWaveGain;
        pl_out->transWaveOffset = pl->transWaveOffset;
	}
	
	SPDPulseUtils::~SPDPulseUtils()
	{
		
	}
}
