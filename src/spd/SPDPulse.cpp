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
	
	CompType* SPDPulseUtils::createSPDPulseH5V1DataTypeDisk()
	{
		CompType *spdPulseDataType = new CompType( sizeof(SPDPulseH5V1) );
		spdPulseDataType->insertMember(PULSEMEMBERNAME_PULSE_ID, HOFFSET(SPDPulseH5V1, pulseID), PredType::STD_U64LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_GPS_TIME, HOFFSET(SPDPulseH5V1, gpsTime), PredType::IEEE_F64LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_X_ORIGIN, HOFFSET(SPDPulseH5V1, x0), PredType::IEEE_F64LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_Y_ORIGIN, HOFFSET(SPDPulseH5V1, y0), PredType::IEEE_F64LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_Z_ORIGIN, HOFFSET(SPDPulseH5V1, z0), PredType::IEEE_F32LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_H_ORIGIN, HOFFSET(SPDPulseH5V1, h0), PredType::IEEE_F32LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_X_IDX, HOFFSET(SPDPulseH5V1, xIdx), PredType::IEEE_F64LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_Y_IDX, HOFFSET(SPDPulseH5V1, yIdx), PredType::IEEE_F64LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_AZIMUTH, HOFFSET(SPDPulseH5V1, azimuth), PredType::IEEE_F32LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_ZENITH, HOFFSET(SPDPulseH5V1, zenith), PredType::IEEE_F32LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_NUMBER_OF_RETURNS, HOFFSET(SPDPulseH5V1, numberOfReturns), PredType::STD_U8LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_NUMBER_OF_WAVEFORM_TRANSMITTED_BINS, HOFFSET(SPDPulseH5V1, numOfTransmittedBins), PredType::STD_U16LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_NUMBER_OF_WAVEFORM_RECEIVED_BINS, HOFFSET(SPDPulseH5V1, numOfReceivedBins), PredType::STD_U16LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_RANGE_TO_WAVEFORM_START, HOFFSET(SPDPulseH5V1, rangeToWaveformStart), PredType::IEEE_F32LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_AMPLITUDE_PULSE, HOFFSET(SPDPulseH5V1, amplitudePulse), PredType::IEEE_F32LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_WIDTH_PULSE, HOFFSET(SPDPulseH5V1, widthPulse), PredType::IEEE_F32LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_USER, HOFFSET(SPDPulseH5V1, user), PredType::STD_U32LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_SOURCE_ID, HOFFSET(SPDPulseH5V1, sourceID), PredType::STD_U16LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_EDGE_FLIGHT_LINE_FLAG, HOFFSET(SPDPulseH5V1, edgeFlightLineFlag), PredType::STD_U8LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_SCAN_DIRECTION_FLAG, HOFFSET(SPDPulseH5V1, scanDirectionFlag), PredType::STD_U8LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_SCAN_ANGLE_RANK, HOFFSET(SPDPulseH5V1, scanAngleRank), PredType::IEEE_F32LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_WAVE_NOISE_THRES, HOFFSET(SPDPulseH5V1, waveNoiseThreshold), PredType::IEEE_F32LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_RECEIVE_WAVE_GAIN, HOFFSET(SPDPulseH5V1, receiveWaveGain), PredType::IEEE_F32LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_RECEIVE_WAVE_OFFSET, HOFFSET(SPDPulseH5V1, receiveWaveOffset), PredType::IEEE_F32LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_TRANS_WAVE_GAIN, HOFFSET(SPDPulseH5V1, transWaveGain), PredType::IEEE_F32LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_TRANS_WAVE_OFFSET, HOFFSET(SPDPulseH5V1, transWaveOffset), PredType::IEEE_F32LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_PTS_START_IDX, HOFFSET(SPDPulseH5V1, ptsStartIdx), PredType::STD_U64LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_RECEIVED_START_IDX, HOFFSET(SPDPulseH5V1, receivedStartIdx), PredType::STD_U64LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_TRANSMITTED_START_IDX, HOFFSET(SPDPulseH5V1, transmittedStartIdx), PredType::STD_U64LE);
		
		return spdPulseDataType;
	}
	
	CompType* SPDPulseUtils::createSPDPulseH5V1DataTypeMemory()
	{
		CompType *spdPulseDataType = new CompType( sizeof(SPDPulseH5V1) );
		spdPulseDataType->insertMember(PULSEMEMBERNAME_PULSE_ID, HOFFSET(SPDPulseH5V1, pulseID), PredType::NATIVE_ULLONG);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_GPS_TIME, HOFFSET(SPDPulseH5V1, gpsTime), PredType::NATIVE_DOUBLE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_X_ORIGIN, HOFFSET(SPDPulseH5V1, x0), PredType::NATIVE_DOUBLE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_Y_ORIGIN, HOFFSET(SPDPulseH5V1, y0), PredType::NATIVE_DOUBLE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_Z_ORIGIN, HOFFSET(SPDPulseH5V1, z0), PredType::NATIVE_FLOAT);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_H_ORIGIN, HOFFSET(SPDPulseH5V1, h0), PredType::NATIVE_FLOAT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_X_IDX, HOFFSET(SPDPulseH5V1, xIdx), PredType::NATIVE_DOUBLE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_Y_IDX, HOFFSET(SPDPulseH5V1, yIdx), PredType::NATIVE_DOUBLE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_AZIMUTH, HOFFSET(SPDPulseH5V1, azimuth), PredType::NATIVE_FLOAT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_ZENITH, HOFFSET(SPDPulseH5V1, zenith), PredType::NATIVE_FLOAT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_NUMBER_OF_RETURNS, HOFFSET(SPDPulseH5V1, numberOfReturns), PredType::NATIVE_UINT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_NUMBER_OF_WAVEFORM_TRANSMITTED_BINS, HOFFSET(SPDPulseH5V1, numOfTransmittedBins), PredType::NATIVE_UINT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_NUMBER_OF_WAVEFORM_RECEIVED_BINS, HOFFSET(SPDPulseH5V1, numOfReceivedBins), PredType::NATIVE_UINT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_RANGE_TO_WAVEFORM_START, HOFFSET(SPDPulseH5V1, rangeToWaveformStart), PredType::NATIVE_FLOAT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_AMPLITUDE_PULSE, HOFFSET(SPDPulseH5V1, amplitudePulse), PredType::NATIVE_FLOAT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_WIDTH_PULSE, HOFFSET(SPDPulseH5V1, widthPulse), PredType::NATIVE_FLOAT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_USER, HOFFSET(SPDPulseH5V1, user), PredType::NATIVE_ULONG);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_SOURCE_ID, HOFFSET(SPDPulseH5V1, sourceID), PredType::NATIVE_UINT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_EDGE_FLIGHT_LINE_FLAG, HOFFSET(SPDPulseH5V1, edgeFlightLineFlag), PredType::NATIVE_UINT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_SCAN_DIRECTION_FLAG, HOFFSET(SPDPulseH5V1, scanDirectionFlag), PredType::NATIVE_UINT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_SCAN_ANGLE_RANK, HOFFSET(SPDPulseH5V1, scanAngleRank), PredType::NATIVE_FLOAT);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_WAVE_NOISE_THRES, HOFFSET(SPDPulseH5V1, waveNoiseThreshold), PredType::NATIVE_FLOAT);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_RECEIVE_WAVE_GAIN, HOFFSET(SPDPulseH5V1, receiveWaveGain), PredType::NATIVE_FLOAT);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_RECEIVE_WAVE_OFFSET, HOFFSET(SPDPulseH5V1, receiveWaveOffset), PredType::NATIVE_FLOAT);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_TRANS_WAVE_GAIN, HOFFSET(SPDPulseH5V1, transWaveGain), PredType::NATIVE_FLOAT);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_TRANS_WAVE_OFFSET, HOFFSET(SPDPulseH5V1, transWaveOffset), PredType::NATIVE_FLOAT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_PTS_START_IDX, HOFFSET(SPDPulseH5V1, ptsStartIdx), PredType::NATIVE_ULLONG);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_RECEIVED_START_IDX, HOFFSET(SPDPulseH5V1, receivedStartIdx), PredType::NATIVE_ULLONG);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_TRANSMITTED_START_IDX, HOFFSET(SPDPulseH5V1, transmittedStartIdx), PredType::NATIVE_ULLONG);
		
		return spdPulseDataType;
	}
    
    CompType* SPDPulseUtils::createSPDPulseH5V2DataTypeDisk()
	{
		CompType *spdPulseDataType = new CompType( sizeof(SPDPulseH5V2) );
		spdPulseDataType->insertMember(PULSEMEMBERNAME_PULSE_ID, HOFFSET(SPDPulseH5V2, pulseID), PredType::STD_U64LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_GPS_TIME, HOFFSET(SPDPulseH5V2, gpsTime), PredType::STD_U64LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_X_ORIGIN, HOFFSET(SPDPulseH5V2, x0), PredType::IEEE_F64LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_Y_ORIGIN, HOFFSET(SPDPulseH5V2, y0), PredType::IEEE_F64LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_Z_ORIGIN, HOFFSET(SPDPulseH5V2, z0), PredType::IEEE_F32LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_H_ORIGIN, HOFFSET(SPDPulseH5V2, h0), PredType::IEEE_F32LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_X_IDX, HOFFSET(SPDPulseH5V2, xIdx), PredType::IEEE_F64LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_Y_IDX, HOFFSET(SPDPulseH5V2, yIdx), PredType::IEEE_F64LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_AZIMUTH, HOFFSET(SPDPulseH5V2, azimuth), PredType::IEEE_F32LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_ZENITH, HOFFSET(SPDPulseH5V2, zenith), PredType::IEEE_F32LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_NUMBER_OF_RETURNS, HOFFSET(SPDPulseH5V2, numberOfReturns), PredType::STD_U8LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_NUMBER_OF_WAVEFORM_TRANSMITTED_BINS, HOFFSET(SPDPulseH5V2, numOfTransmittedBins), PredType::STD_U16LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_NUMBER_OF_WAVEFORM_RECEIVED_BINS, HOFFSET(SPDPulseH5V2, numOfReceivedBins), PredType::STD_U16LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_RANGE_TO_WAVEFORM_START, HOFFSET(SPDPulseH5V2, rangeToWaveformStart), PredType::IEEE_F32LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_AMPLITUDE_PULSE, HOFFSET(SPDPulseH5V2, amplitudePulse), PredType::IEEE_F32LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_WIDTH_PULSE, HOFFSET(SPDPulseH5V2, widthPulse), PredType::IEEE_F32LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_USER, HOFFSET(SPDPulseH5V2, user), PredType::STD_U32LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_SOURCE_ID, HOFFSET(SPDPulseH5V2, sourceID), PredType::STD_U16LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_SCANLINE, HOFFSET(SPDPulseH5V2, scanline), PredType::STD_U32LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_SCANLINE_IDX, HOFFSET(SPDPulseH5V2, scanlineIdx), PredType::STD_U16LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_RECEIVE_WAVE_NOISE_THRES, HOFFSET(SPDPulseH5V2, receiveWaveNoiseThreshold), PredType::IEEE_F32LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_TRANS_WAVE_NOISE_THRES, HOFFSET(SPDPulseH5V2, transWaveNoiseThres), PredType::IEEE_F32LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_WAVELENGTH, HOFFSET(SPDPulseH5V2, wavelength), PredType::IEEE_F32LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_RECEIVE_WAVE_GAIN, HOFFSET(SPDPulseH5V2, receiveWaveGain), PredType::IEEE_F32LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_RECEIVE_WAVE_OFFSET, HOFFSET(SPDPulseH5V2, receiveWaveOffset), PredType::IEEE_F32LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_TRANS_WAVE_GAIN, HOFFSET(SPDPulseH5V2, transWaveGain), PredType::IEEE_F32LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_TRANS_WAVE_OFFSET, HOFFSET(SPDPulseH5V2, transWaveOffset), PredType::IEEE_F32LE);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_PTS_START_IDX, HOFFSET(SPDPulseH5V2, ptsStartIdx), PredType::STD_U64LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_RECEIVED_START_IDX, HOFFSET(SPDPulseH5V2, receivedStartIdx), PredType::STD_U64LE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_TRANSMITTED_START_IDX, HOFFSET(SPDPulseH5V2, transmittedStartIdx), PredType::STD_U64LE);
		
		return spdPulseDataType;
	}
	
	CompType* SPDPulseUtils::createSPDPulseH5V2DataTypeMemory()
	{
		CompType *spdPulseDataType = new CompType( sizeof(SPDPulseH5V2) );
		spdPulseDataType->insertMember(PULSEMEMBERNAME_PULSE_ID, HOFFSET(SPDPulseH5V2, pulseID), PredType::NATIVE_ULLONG);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_GPS_TIME, HOFFSET(SPDPulseH5V2, gpsTime), PredType::NATIVE_ULLONG);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_X_ORIGIN, HOFFSET(SPDPulseH5V2, x0), PredType::NATIVE_DOUBLE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_Y_ORIGIN, HOFFSET(SPDPulseH5V2, y0), PredType::NATIVE_DOUBLE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_Z_ORIGIN, HOFFSET(SPDPulseH5V2, z0), PredType::NATIVE_FLOAT);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_H_ORIGIN, HOFFSET(SPDPulseH5V2, h0), PredType::NATIVE_FLOAT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_X_IDX, HOFFSET(SPDPulseH5V2, xIdx), PredType::NATIVE_DOUBLE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_Y_IDX, HOFFSET(SPDPulseH5V2, yIdx), PredType::NATIVE_DOUBLE);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_AZIMUTH, HOFFSET(SPDPulseH5V2, azimuth), PredType::NATIVE_FLOAT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_ZENITH, HOFFSET(SPDPulseH5V2, zenith), PredType::NATIVE_FLOAT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_NUMBER_OF_RETURNS, HOFFSET(SPDPulseH5V2, numberOfReturns), PredType::NATIVE_UINT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_NUMBER_OF_WAVEFORM_TRANSMITTED_BINS, HOFFSET(SPDPulseH5V2, numOfTransmittedBins), PredType::NATIVE_UINT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_NUMBER_OF_WAVEFORM_RECEIVED_BINS, HOFFSET(SPDPulseH5V2, numOfReceivedBins), PredType::NATIVE_UINT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_RANGE_TO_WAVEFORM_START, HOFFSET(SPDPulseH5V2, rangeToWaveformStart), PredType::NATIVE_FLOAT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_AMPLITUDE_PULSE, HOFFSET(SPDPulseH5V2, amplitudePulse), PredType::NATIVE_FLOAT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_WIDTH_PULSE, HOFFSET(SPDPulseH5V2, widthPulse), PredType::NATIVE_FLOAT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_USER, HOFFSET(SPDPulseH5V2, user), PredType::NATIVE_ULONG);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_SOURCE_ID, HOFFSET(SPDPulseH5V2, sourceID), PredType::NATIVE_UINT);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_SCANLINE, HOFFSET(SPDPulseH5V2, scanline), PredType::NATIVE_ULONG);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_SCANLINE_IDX, HOFFSET(SPDPulseH5V2, scanlineIdx), PredType::NATIVE_UINT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_RECEIVE_WAVE_NOISE_THRES, HOFFSET(SPDPulseH5V2, receiveWaveNoiseThreshold), PredType::NATIVE_FLOAT);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_TRANS_WAVE_NOISE_THRES, HOFFSET(SPDPulseH5V2, transWaveNoiseThres), PredType::NATIVE_FLOAT);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_WAVELENGTH, HOFFSET(SPDPulseH5V2, wavelength), PredType::NATIVE_FLOAT);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_RECEIVE_WAVE_GAIN, HOFFSET(SPDPulseH5V2, receiveWaveGain), PredType::NATIVE_FLOAT);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_RECEIVE_WAVE_OFFSET, HOFFSET(SPDPulseH5V2, receiveWaveOffset), PredType::NATIVE_FLOAT);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_TRANS_WAVE_GAIN, HOFFSET(SPDPulseH5V2, transWaveGain), PredType::NATIVE_FLOAT);
        spdPulseDataType->insertMember(PULSEMEMBERNAME_TRANS_WAVE_OFFSET, HOFFSET(SPDPulseH5V2, transWaveOffset), PredType::NATIVE_FLOAT);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_PTS_START_IDX, HOFFSET(SPDPulseH5V2, ptsStartIdx), PredType::NATIVE_ULLONG);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_RECEIVED_START_IDX, HOFFSET(SPDPulseH5V2, receivedStartIdx), PredType::NATIVE_ULLONG);
		spdPulseDataType->insertMember(PULSEMEMBERNAME_TRANSMITTED_START_IDX, HOFFSET(SPDPulseH5V2, transmittedStartIdx), PredType::NATIVE_ULLONG);
		
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
		pl->pts = new vector<SPDPoint*>();
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
		pl_out->pts = new vector<SPDPoint*>();
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
		pl_out->pts = new vector<SPDPoint*>();
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
