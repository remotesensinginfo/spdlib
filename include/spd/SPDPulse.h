 /*
  *  SPDPulse.h
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


#ifndef SPDPulse_H
#define SPDPulse_H

#include <iostream>
#include <string>
#include <vector>

#include <boost/cstdint.hpp>

#include "H5Cpp.h"

#include "spd/SPDPoint.h"
#include "spd/SPDCommon.h"

using namespace std;
using namespace H5;

namespace spdlib
{
	static const string PULSEMEMBERNAME_PULSE_ID( "PULSE_ID" );
	static const string PULSEMEMBERNAME_GPS_TIME( "GPS_TIME" );
	static const string PULSEMEMBERNAME_X_ORIGIN( "X_ORIGIN" );
	static const string PULSEMEMBERNAME_Y_ORIGIN( "Y_ORIGIN" );
	static const string PULSEMEMBERNAME_Z_ORIGIN( "Z_ORIGIN" );
    static const string PULSEMEMBERNAME_H_ORIGIN( "H_ORIGIN" );
	static const string PULSEMEMBERNAME_X_IDX( "X_IDX" );
	static const string PULSEMEMBERNAME_Y_IDX( "Y_IDX" );
	static const string PULSEMEMBERNAME_AZIMUTH( "AZIMUTH" );
	static const string PULSEMEMBERNAME_ZENITH( "ZENITH" );
	static const string PULSEMEMBERNAME_NUMBER_OF_RETURNS( "NUMBER_OF_RETURNS" );
	static const string PULSEMEMBERNAME_NUMBER_OF_WAVEFORM_TRANSMITTED_BINS( "NUMBER_OF_WAVEFORM_TRANSMITTED_BINS" );
	static const string PULSEMEMBERNAME_NUMBER_OF_WAVEFORM_RECEIVED_BINS( "NUMBER_OF_WAVEFORM_RECEIVED_BINS" );
	static const string PULSEMEMBERNAME_RANGE_TO_WAVEFORM_START( "RANGE_TO_WAVEFORM_START" );
	static const string PULSEMEMBERNAME_AMPLITUDE_PULSE( "AMPLITUDE_PULSE" );
	static const string PULSEMEMBERNAME_WIDTH_PULSE( "WIDTH_PULSE" );
	static const string PULSEMEMBERNAME_USER( "USER_FIELD" );
	static const string PULSEMEMBERNAME_SOURCE_ID( "SOURCE_ID" );
	static const string PULSEMEMBERNAME_EDGE_FLIGHT_LINE_FLAG( "EDGE_FLIGHT_LINE_FLAG" );
	static const string PULSEMEMBERNAME_SCAN_DIRECTION_FLAG( "SCAN_DIRECTION_FLAG" );
	static const string PULSEMEMBERNAME_SCAN_ANGLE_RANK( "SCAN_ANGLE_RANK" );
    static const string PULSEMEMBERNAME_SCANLINE( "SCANLINE" );
    static const string PULSEMEMBERNAME_SCANLINE_IDX( "SCANLINE_IDX" );    
    static const string PULSEMEMBERNAME_WAVE_NOISE_THRES( "WAVE_NOISE_THRES" );
    static const string PULSEMEMBERNAME_RECEIVE_WAVE_NOISE_THRES( "RECEIVE_WAVE_NOISE_THRES" );
    static const string PULSEMEMBERNAME_TRANS_WAVE_NOISE_THRES( "TRANS_WAVE_NOISE_THRES" );
    static const string PULSEMEMBERNAME_WAVELENGTH( "WAVELENGTH" );
    static const string PULSEMEMBERNAME_RECEIVE_WAVE_GAIN( "RECEIVE_WAVE_GAIN" );
    static const string PULSEMEMBERNAME_RECEIVE_WAVE_OFFSET( "RECEIVE_WAVE_OFFSET" );
    static const string PULSEMEMBERNAME_TRANS_WAVE_GAIN( "TRANS_WAVE_GAIN" );
    static const string PULSEMEMBERNAME_TRANS_WAVE_OFFSET( "TRANS_WAVE_OFFSET" );
	static const string PULSEMEMBERNAME_PTS_START_IDX( "PTS_START_IDX" );
	static const string PULSEMEMBERNAME_RECEIVED_START_IDX( "RECEIVED_START_IDX" );
	static const string PULSEMEMBERNAME_TRANSMITTED_START_IDX( "TRANSMITTED_START_IDX" );
	
	struct SPDPulse
	{
		/**
		 * The pulse ID - A unique ID for each pulse
		 */
        boost::uint_fast64_t pulseID;
		/**
		 * GPS time is the time at which the point was aquired.
		 */
		boost::uint_fast64_t gpsTime;
		/**
		 * X Coordinate origin of the pulse
		 */
		double x0;
		/**
		 * Y Coordinate origin of the pulse
		 */
		double y0;
		/**
		 * Z Coordinate origin of the pulse
		 */
		float z0;
        /**
		 * Height coordinate origin of the pulse
		 */
		float h0;
		/**
		 * X Coordinate used for indexing.
		 */
		double xIdx;
		/**
		 * Y Coordinate used for indexing.
		 */
		double yIdx;
		/**
		 * Azimuth used for Spherical coordinates
		 */
		float azimuth;
		/**
		 * Zenith used for Spherical coordinates
		 */
		float zenith;
		/**
		 * The number of returns within a pulse
		 */
        boost::uint_fast16_t numberOfReturns;
		/**
		 * List of points within the pulse
		 */
		vector<SPDPoint*> *pts;
		/**
		 * Out going energy pulse
		 */
        boost::uint_fast32_t *transmitted;
		/**
		 * In coming energy pulse
		 */
        boost::uint_fast32_t *received;
		/**
		 * Number of bins within tranmitted pulse
		 */
        boost::uint_fast16_t numOfTransmittedBins;
		/**
		 * Number of bins within received pulse
		 */
        boost::uint_fast16_t numOfReceivedBins;
		/**
		 * Range to the return
		 */
		float rangeToWaveformStart;
		/**
		 * Amplitude (intensity) transmitted from the sensor
		 */
		float amplitudePulse;
		/**
		 * Width of the pulse transmitted from the sensor
		 */
		float widthPulse;
		/**
		 * A user defined field
		 */
        boost::uint_fast32_t user;
		/**
		 * An ID of the source (i.e., different flight lines)
		 */
        boost::uint_fast16_t sourceID;
		/**
		 * A variable to identify whether the point was on the edge of the scan 
		 * (i.e., the laser was changing direction)
		 */
        boost::uint_fast16_t edgeFlightLineFlag;
		/**
		 * A variable to identify the direction the scanner is moving in.
		 * 
		 * Note: A positive scan direction is a scan moving from the left side 
		 * of the in-track direction to the right side and negative the opposite
		 */
        boost::uint_fast16_t scanDirectionFlag;
		/**
		 * The angle of the laser (including aircraft roll) when the pulse was
		 * emitted. Values can range from -90 to +90.
		 *
		 * Note: The scan angle is an angle based on 0 degrees being nadir, and –90 
		 * degrees to the left side of the aircraft in the direction of flight.
		 */
		float scanAngleRank;
        /**
		 * A variable to identify the scanline the file originated from.
		 */
        boost::uint_fast32_t scanline;
        /**
		 * A variable to identify the position of the pulse along the scanline
		 */
        boost::uint_fast16_t scanlineIdx;
        /**
         * A floating point value defining the noise threshold to be used when processing
         * the received waveform associated with this pulse.
         */
        float receiveWaveNoiseThreshold;
        /**
         * A floating point value defining the noise threshold to be used when processing
         * the transmitted waveform associated with this pulse.
         */
        float transWaveNoiseThres;
        /**
         * The wavelength of the pulse
         */
        float wavelength;
        /**
         * A floating point value defining the gain value to be used when processing
         * the recieved waveform associated with this pulse.
         */
        float receiveWaveGain;
        /**
         * A floating point value defining the offset value to be used when processing
         * the recieved waveform associated with this pulse.
         */
        float receiveWaveOffset;
        /**
         * A floating point value defining the gain value to be used when processing
         * the transmitted waveform associated with this pulse.
         */
        float transWaveGain;
        /**
         * A floating point value defining the offset value to be used when processing
         * the transmitted waveform associated with this pulse.
         */
        float transWaveOffset;
        
		friend ostream& operator<<(ostream& stream, SPDPulse &obj)
		{
			stream << "Pulse ID: " << obj.pulseID << " from time " << obj.gpsTime << endl;
			stream << "Origin [" << obj.x0 << "," << obj.y0 << "," << obj.z0 << "]\n";
			stream << "Index (Cartisan): [" << obj.xIdx << "," << obj.yIdx << "]\n";
			stream << "Index (Spherical): [" << obj.azimuth << "," << obj.zenith << "]\n";
			stream << "Number of Returns: " << obj.numberOfReturns << endl;
			stream << "Num. Transmitted Bins: " << obj.numOfTransmittedBins << endl;
			stream << "Num. Received Bins: " << obj.numOfReceivedBins << endl;
			stream << "Range to Return: " << obj.rangeToWaveformStart << endl;
			stream << "Amplitude of Pulse: " << obj.amplitudePulse << endl;
			stream << "Width of Pulse: " << obj.widthPulse << endl;
			stream << "User Field: " << obj.user << endl;
			stream << "Source ID: " << obj.sourceID << endl;
			stream << "Edge of Flight line flag: " << obj.edgeFlightLineFlag << endl;
			stream << "Scan direction flag: " << obj.scanDirectionFlag << endl;
			stream << "Scan angle Rank: " << obj.scanAngleRank << endl;
            stream << "Wavelength: " << obj.wavelength << endl;
            stream << "Received Waveform Noise Threshold: " << obj.receiveWaveNoiseThreshold << endl;
            stream << "Transmitted Waveform Noise Threshold: " << obj.transWaveNoiseThres << endl;
            stream << "Received Waveform Gain: " << obj.receiveWaveGain << endl;
            stream << "Received Waveform Offset: " << obj.receiveWaveOffset << endl;
            stream << "Transmitted Waveform Gain: " << obj.transWaveGain << endl;
            stream << "Transmitted Waveform Offset: " << obj.transWaveOffset << endl;
			for(uint_fast16_t i = 0; i < obj.numberOfReturns; ++i)
			{
				cout << obj.pts->at(i);
			}
			cout << "Transmitted: ";
			for(uint_fast16_t i = 0; i < obj.numOfTransmittedBins; ++i)
			{
				if(i == 0)
				{
					cout << obj.transmitted[i];
				}
				else
				{
					cout << "," << obj.transmitted[i];
				}
			}
			cout << endl;
			cout << "Received: ";
			for(uint_fast16_t i = 0; i < obj.numOfReceivedBins; ++i)
			{
				if(i == 0)
				{
					cout << obj.received[i];
				}
				else
				{
					cout << "," << obj.received[i];
				}
			}
			cout << endl;
			return stream;
		};
		
		friend ostream& operator<<(ostream& stream, SPDPulse *obj)
		{
			stream << "Pulse ID: " << obj->pulseID  << " from time " << obj->gpsTime << endl;
			stream << "Origin [" << obj->x0 << "," << obj->y0 << "," << obj->z0 << "]\n";
			stream << "Index (Cartisan): [" << obj->xIdx << "," << obj->yIdx << "]\n";
			stream << "Index (Spherical): [" << obj->azimuth << "," << obj->zenith << "]\n";
			stream << "Number of Returns: " << obj->numberOfReturns << endl;
			stream << "Num. Transmitted Bins: " << obj->numOfTransmittedBins << endl;
			stream << "Num. Received Bins: " << obj->numOfReceivedBins << endl;
			stream << "Range to Return: " << obj->rangeToWaveformStart << endl;
			stream << "Amplitude of Pulse: " << obj->amplitudePulse << endl;
			stream << "Width of Pulse: " << obj->widthPulse << endl;
			stream << "User Field: " << obj->user << endl;
			stream << "Source ID: " << obj->sourceID << endl;
			stream << "Edge of Flight line flag: " << obj->edgeFlightLineFlag << endl;
			stream << "Scan direction flag: " << obj->scanDirectionFlag << endl;
			stream << "Scan angle Rank: " << obj->scanAngleRank << endl;
            stream << "Wavelength: " << obj->wavelength << endl;
            stream << "Received Waveform Noise Threshold: " << obj->receiveWaveNoiseThreshold << endl;
            stream << "Transmitted Waveform Noise Threshold: " << obj->transWaveNoiseThres << endl;
            stream << "Received Waveform Gain: " << obj->receiveWaveGain << endl;
            stream << "Received Waveform Offset: " << obj->receiveWaveOffset << endl;
            stream << "Transmitted Waveform Gain: " << obj->transWaveGain << endl;
            stream << "Transmitted Waveform Offset: " << obj->transWaveOffset << endl;
			for(boost::uint_fast16_t i = 0; i < obj->numberOfReturns; ++i)
			{
				cout << obj->pts->at(i);
			}
			cout << "Transmitted: ";
			for(boost::uint_fast16_t i = 0; i < obj->numOfTransmittedBins; ++i)
			{
				if(i == 0)
				{
					cout << obj->transmitted[i];
				}
				else
				{
					cout << "," << obj->transmitted[i];
				}
			}
			cout << endl;
			cout << "Received: ";
			for(boost::uint_fast16_t i = 0; i < obj->numOfReceivedBins; ++i)
			{
				if(i == 0)
				{
					cout << obj->received[i];
				}
				else
				{
					cout << "," << obj->received[i];
				}
			}
			cout << endl;
			return stream;
		};
	};
	
	struct SPDPulseH5V1
	{
        /**
		 * GPS time is the time at which the point was aquired.
		 */
		double gpsTime;
		/**
		 * The pulse ID - A unique ID for each pulse
		 */
		unsigned long long pulseID;
		/**
		 * X Coordinate origin of the pulse
		 */
		double x0;
		/**
		 * Y Coordinate origin of the pulse
		 */
		double y0;
		/**
		 * Z Coordinate origin of the pulse
		 */
		float z0;
        /**
		 * Height coordinate origin of the pulse
		 */
		float h0;
		/**
		 * X Coordinate used for indexing.
		 */
		double xIdx;
		/**
		 * Y Coordinate used for indexing.
		 */
		double yIdx;
		/**
		 * Azimuth used for Spherical coordinates
		 */
		float azimuth;
		/**
		 * Zenith used for Spherical coordinates
		 */
		float zenith;
		/**
		 * The number of returns within a pulse
		 */
		unsigned int numberOfReturns;
		/**
		 * Number of bins within tranmitted pulse
		 */
		unsigned int numOfTransmittedBins;
		/**
		 * Number of bins within received pulse
		 */
		unsigned int numOfReceivedBins;
		/**
		 * Range to the return
		 */
		float rangeToWaveformStart;
		/**
		 * Amplitude transmitted from the sensor
		 */
		float amplitudePulse;
		/**
		 * Width of the pulse transmitted from the sensor
		 */
		float widthPulse;
		/**
		 * A user defined field
		 */
		unsigned long user;
		/**
		 * An ID of the source (i.e., different flight lines)
		 */
		unsigned int sourceID;
		/**
		 * A variable to identify whether the point was on the edge of the scan 
		 * (i.e., the laser was changing direction)
		 */
		unsigned int edgeFlightLineFlag;
		/**
		 * A variable to identify the direction the scanner is moving in.
		 * 
		 * Note: A positive scan direction is a scan moving from the left side 
		 * of the in-track direction to the right side and negative the opposite
		 */
		unsigned int scanDirectionFlag;
		/**
		 * The angle of the laser (including aircraft roll) when the pulse was
		 * emitted. Values can range from -90 to +90.
		 *
		 * Note: The scan angle is an angle based on 0 degrees being nadir, and –90 
		 * degrees to the left side of the aircraft in the direction of flight.
		 */
		float scanAngleRank;
        /**
         * A floating point value defining the noise threshold to be used when processing
         * the received waveform associated with this pulse.
         */
        float waveNoiseThreshold;
        /**
         * A floating point value defining the gain value to be used when processing
         * the recieved waveform associated with this pulse.
         */
        float receiveWaveGain;
        /**
         * A floating point value defining the offset value to be used when processing
         * the recieved waveform associated with this pulse.
         */
        float receiveWaveOffset;
        /**
         * A floating point value defining the gain value to be used when processing
         * the transmitted waveform associated with this pulse.
         */
        float transWaveGain;
        /**
         * A floating point value defining the offset value to be used when processing
         * the transmitted waveform associated with this pulse.
         */
        float transWaveOffset;
		/**
		 * The starting index of the points in the point list
		 */
		unsigned long long ptsStartIdx;
		/**
		 * The starting index of the transmitted values
		 */
		unsigned long long transmittedStartIdx;
		/**
		 * The starting index of the received values
		 */
		unsigned long long receivedStartIdx;
		
		friend ostream& operator<<(ostream& stream, SPDPulseH5V1 &obj)
		{
			stream << "Pulse ID: " << obj.pulseID << " from time " << obj.gpsTime  << endl;
			stream << "Origin [" << obj.x0 << "," << obj.y0 << "," << obj.z0 << "]\n";
			stream << "Index (Cartisan): [" << obj.xIdx << "," << obj.yIdx << "]\n";
			stream << "Index (Spherical): [" << obj.azimuth << "," << obj.zenith << "]\n";
			stream << "Number of Returns: " << obj.numberOfReturns << endl;
			stream << "Num. Out Values: " << obj.numOfTransmittedBins << endl;
			stream << "Num. In Values: " << obj.numOfReceivedBins << endl;
			stream << "Range to Return: " << obj.rangeToWaveformStart << endl;
			stream << "Amplitude of Pulse: " << obj.amplitudePulse << endl;
			stream << "Width of Pulse: " << obj.widthPulse << endl;
			stream << "User Field: " << obj.user << endl;
			stream << "Source ID: " << obj.sourceID << endl;
			stream << "Edge of Flight line flag: " << obj.edgeFlightLineFlag << endl;
			stream << "Scan direction flag: " << obj.scanDirectionFlag << endl;
			stream << "Scan angle Rank: " << obj.scanAngleRank << endl;
            stream << "Waveform Noise Threshold: " << obj.waveNoiseThreshold << endl;
            stream << "Received Waveform Gain: " << obj.receiveWaveGain << endl;
            stream << "Received Waveform Offset: " << obj.receiveWaveOffset << endl;
            stream << "Transmitted Waveform Gain: " << obj.transWaveGain << endl;
            stream << "Transmitted Waveform Offset: " << obj.transWaveOffset << endl;
			stream << "Pts start index: " << obj.ptsStartIdx << endl;
			stream << "Receievd start index: " << obj.receivedStartIdx << endl;
			stream << "Transmitted start index: " << obj.transmittedStartIdx << endl;
			return stream;
		};
		
		friend ostream& operator<<(ostream& stream, SPDPulseH5V1 *obj)
		{
			stream << "Pulse ID: " << obj->pulseID << " from time " << obj->gpsTime  << endl;
			stream << "Origin [" << obj->x0 << "," << obj->y0 << "," << obj->z0 << "]\n";
			stream << "Index (Cartisan): [" << obj->xIdx << "," << obj->yIdx << "]\n";
			stream << "Index (Spherical): [" << obj->azimuth << "," << obj->zenith << "]\n";
			stream << "Number of Returns: " << obj->numberOfReturns << endl;
			stream << "Num. Out Values: " << obj->numOfTransmittedBins << endl;
			stream << "Num. In Values: " << obj->numOfReceivedBins << endl;
			stream << "Range to Return: " << obj->rangeToWaveformStart << endl;
			stream << "Amplitude of Pulse: " << obj->amplitudePulse << endl;
			stream << "Width of Pulse: " << obj->widthPulse << endl;
			stream << "User Field: " << obj->user << endl;
			stream << "Source ID: " << obj->sourceID << endl;
			stream << "Edge of Flight line flag: " << obj->edgeFlightLineFlag << endl;
			stream << "Scan direction flag: " << obj->scanDirectionFlag << endl;
			stream << "Scan angle Rank: " << obj->scanAngleRank << endl;
            stream << "Waveform Noise Threshold: " << obj->waveNoiseThreshold << endl;
            stream << "Received Waveform Gain: " << obj->receiveWaveGain << endl;
            stream << "Received Waveform Offset: " << obj->receiveWaveOffset << endl;
            stream << "Transmitted Waveform Gain: " << obj->transWaveGain << endl;
            stream << "Transmitted Waveform Offset: " << obj->transWaveOffset << endl;
			stream << "Pts start index: " << obj->ptsStartIdx << endl;
			stream << "Received start index: " << obj->receivedStartIdx << endl;
			stream << "Transmitted start index: " << obj->transmittedStartIdx << endl;
			return stream;
		};
	};
    
    struct SPDPulseH5V2
	{
        /**
		 * GPS time is the time at which the point was aquired.
		 */
		unsigned long long gpsTime;
		/**
		 * The pulse ID - A unique ID for each pulse
		 */
		unsigned long long pulseID;
		/**
		 * X Coordinate origin of the pulse
		 */
		double x0;
		/**
		 * Y Coordinate origin of the pulse
		 */
		double y0;
		/**
		 * Z Coordinate origin of the pulse
		 */
		float z0;
        /**
		 * Height coordinate origin of the pulse
		 */
		float h0;
		/**
		 * X Coordinate used for indexing.
		 */
		double xIdx;
		/**
		 * Y Coordinate used for indexing.
		 */
		double yIdx;
		/**
		 * Azimuth used for Spherical coordinates
		 */
		float azimuth;
		/**
		 * Zenith used for Spherical coordinates
		 */
		float zenith;
		/**
		 * The number of returns within a pulse
		 */
		unsigned int numberOfReturns;
		/**
		 * Number of bins within tranmitted pulse
		 */
		unsigned int numOfTransmittedBins;
		/**
		 * Number of bins within received pulse
		 */
		unsigned int numOfReceivedBins;
		/**
		 * Range to the return
		 */
		float rangeToWaveformStart;
		/**
		 * Amplitude transmitted from the sensor
		 */
		float amplitudePulse;
		/**
		 * Width of the pulse transmitted from the sensor
		 */
		float widthPulse;
		/**
		 * A user defined field
		 */
		unsigned long user;
		/**
		 * An ID of the source (i.e., different flight lines)
		 */
		unsigned int sourceID;
        /**
		 * A variable to identify the scanline the file originated from.
		 */
        unsigned long scanline;
        /**
		 * A variable to identify the position of the pulse along the scanline
		 */
        unsigned int scanlineIdx;
        /**
         * A floating point value defining the noise threshold to be used when processing
         * the received waveform associated with this pulse.
         */
        float receiveWaveNoiseThreshold;
        /**
         * A floating point value defining the noise threshold to be used when processing
         * the transmitted waveform associated with this pulse.
         */
        float transWaveNoiseThres;
        /**
         * The wavelength of the pulse
         */
        float wavelength;
        /**
         * A floating point value defining the gain value to be used when processing
         * the recieved waveform associated with this pulse.
         */
        float receiveWaveGain;
        /**
         * A floating point value defining the offset value to be used when processing
         * the recieved waveform associated with this pulse.
         */
        float receiveWaveOffset;
        /**
         * A floating point value defining the gain value to be used when processing
         * the transmitted waveform associated with this pulse.
         */
        float transWaveGain;
        /**
         * A floating point value defining the offset value to be used when processing
         * the transmitted waveform associated with this pulse.
         */
        float transWaveOffset;
		/**
		 * The starting index of the points in the point list
		 */
		unsigned long long ptsStartIdx;
		/**
		 * The starting index of the transmitted values
		 */
		unsigned long long transmittedStartIdx;
		/**
		 * The starting index of the received values
		 */
		unsigned long long receivedStartIdx;
		
		friend ostream& operator<<(ostream& stream, SPDPulseH5V2 &obj)
		{
			stream << "Pulse ID: " << obj.pulseID << " from time " << obj.gpsTime  << endl;
			stream << "Origin [" << obj.x0 << "," << obj.y0 << "," << obj.z0 << "]\n";
			stream << "Index (Cartisan): [" << obj.xIdx << "," << obj.yIdx << "]\n";
			stream << "Index (Spherical): [" << obj.azimuth << "," << obj.zenith << "]\n";
			stream << "Number of Returns: " << obj.numberOfReturns << endl;
			stream << "Num. Out Values: " << obj.numOfTransmittedBins << endl;
			stream << "Num. In Values: " << obj.numOfReceivedBins << endl;
			stream << "Range to Return: " << obj.rangeToWaveformStart << endl;
			stream << "Amplitude of Pulse: " << obj.amplitudePulse << endl;
			stream << "Width of Pulse: " << obj.widthPulse << endl;
			stream << "User Field: " << obj.user << endl;
			stream << "Source ID: " << obj.sourceID << endl;
            stream << "Wavelength: " << obj.wavelength << endl;
            stream << "Received Waveform Noise Threshold: " << obj.receiveWaveNoiseThreshold << endl;
            stream << "Transmitted Waveform Noise Threshold: " << obj.transWaveNoiseThres << endl;
            stream << "Received Waveform Gain: " << obj.receiveWaveGain << endl;
            stream << "Received Waveform Offset: " << obj.receiveWaveOffset << endl;
            stream << "Transmitted Waveform Gain: " << obj.transWaveGain << endl;
            stream << "Transmitted Waveform Offset: " << obj.transWaveOffset << endl;
			stream << "Pts start index: " << obj.ptsStartIdx << endl;
			stream << "Receievd start index: " << obj.receivedStartIdx << endl;
			stream << "Transmitted start index: " << obj.transmittedStartIdx << endl;
			return stream;
		};
		
		friend ostream& operator<<(ostream& stream, SPDPulseH5V2 *obj)
		{
			stream << "Pulse ID: " << obj->pulseID << " from time " << obj->gpsTime  << endl;
			stream << "Origin [" << obj->x0 << "," << obj->y0 << "," << obj->z0 << "]\n";
			stream << "Index (Cartisan): [" << obj->xIdx << "," << obj->yIdx << "]\n";
			stream << "Index (Spherical): [" << obj->azimuth << "," << obj->zenith << "]\n";
			stream << "Number of Returns: " << obj->numberOfReturns << endl;
			stream << "Num. Out Values: " << obj->numOfTransmittedBins << endl;
			stream << "Num. In Values: " << obj->numOfReceivedBins << endl;
			stream << "Range to Return: " << obj->rangeToWaveformStart << endl;
			stream << "Amplitude of Pulse: " << obj->amplitudePulse << endl;
			stream << "Width of Pulse: " << obj->widthPulse << endl;
			stream << "User Field: " << obj->user << endl;
			stream << "Source ID: " << obj->sourceID << endl;
            stream << "Wavelength: " << obj->wavelength << endl;
            stream << "Received Waveform Noise Threshold: " << obj->receiveWaveNoiseThreshold << endl;
            stream << "Transmitted Waveform Noise Threshold: " << obj->transWaveNoiseThres << endl;
            stream << "Received Waveform Gain: " << obj->receiveWaveGain << endl;
            stream << "Received Waveform Offset: " << obj->receiveWaveOffset << endl;
            stream << "Transmitted Waveform Gain: " << obj->transWaveGain << endl;
            stream << "Transmitted Waveform Offset: " << obj->transWaveOffset << endl;
			stream << "Pts start index: " << obj->ptsStartIdx << endl;
			stream << "Received start index: " << obj->receivedStartIdx << endl;
			stream << "Transmitted start index: " << obj->transmittedStartIdx << endl;
			return stream;
		};
	};
	
	/**
	 * Provides useful utilities for manipulating an SPDPulse.
	 */
	class SPDPulseUtils
	{
	public:
		SPDPulseUtils();
		CompType* createSPDPulseH5V1DataTypeDisk();
		CompType* createSPDPulseH5V1DataTypeMemory();
        CompType* createSPDPulseH5V2DataTypeDisk();
		CompType* createSPDPulseH5V2DataTypeMemory();
		void initSPDPulse(SPDPulse *pl);
		void initSPDPulseH5(SPDPulseH5V1 *pl);
        void initSPDPulseH5(SPDPulseH5V2 *pl);
		static void deleteSPDPulse(SPDPulse *pl);
		SPDPulse* createSPDPulseCopy(SPDPulse *pl);
		void copySPDPulseTo(SPDPulse *pl, SPDPulse *pl_out);
		SPDPulse* createSPDPulseDeepCopy(SPDPulse *pl);
		void deepCopySPDPulseTo(SPDPulse *pl, SPDPulse *pl_out);
        
        /* For Pulse Version 1 */
		SPDPulseH5V1* createSPDPulseH5V1Copy(SPDPulse *pl);
		void copySPDPulseToSPDPulseH5(SPDPulse *pl, SPDPulseH5V1 *pl_out);
		SPDPulse* createSPDPulseCopyFromH5(SPDPulseH5V1 *pl);
		void copySPDPulseH5ToSPDPulse(SPDPulseH5V1 *pl, SPDPulse *pl_out);
        
        /* For Pulse Version 2 */
        SPDPulseH5V2* createSPDPulseH5V2Copy(SPDPulse *pl);
		void copySPDPulseToSPDPulseH5(SPDPulse *pl, SPDPulseH5V2 *pl_out);
		SPDPulse* createSPDPulseCopyFromH5(SPDPulseH5V2 *pl);
		void copySPDPulseH5ToSPDPulse(SPDPulseH5V2 *pl, SPDPulse *pl_out);
		
		~SPDPulseUtils();		
	};
	
}

#endif

