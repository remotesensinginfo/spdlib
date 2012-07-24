 /*
  *  SPDPoint.h
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

#ifndef SPDPoint_H
#define SPDPoint_H

#include <iostream>
#include <string>
#include <math.h>
#include <vector>

#include <boost/cstdint.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include "H5Cpp.h"

#include "spd/SPDCommon.h"
#include "spd/SPDProcessingException.h"

namespace spdlib
{
	static const std::string POINTMEMBERNAME_X( "X" );
	static const std::string POINTMEMBERNAME_Y( "Y" );
	static const std::string POINTMEMBERNAME_Z( "Z" );
	static const std::string POINTMEMBERNAME_HEIGHT( "HEIGHT" );
	static const std::string POINTMEMBERNAME_RANGE( "RANGE" );
	static const std::string POINTMEMBERNAME_AMPLITUDE_RETURN( "AMPLITUDE_RETURN" );
	static const std::string POINTMEMBERNAME_WIDTH_RETURN( "WIDTH_RETURN" );
	static const std::string POINTMEMBERNAME_RED( "RED" );
	static const std::string POINTMEMBERNAME_GREEN( "GREEN" );
	static const std::string POINTMEMBERNAME_BLUE( "BLUE" );
	static const std::string POINTMEMBERNAME_CLASSIFICATION( "CLASSIFICATION" );
	static const std::string POINTMEMBERNAME_RETURN_ID( "RETURN_ID" );
	static const std::string POINTMEMBERNAME_NUMBER_OF_RETURNS( "NUMBER_OF_RETURNS" );
	static const std::string POINTMEMBERNAME_GPS_TIME( "GPS_TIME" );
	static const std::string POINTMEMBERNAME_USER( "USER_FIELD" );
	static const std::string POINTMEMBERNAME_MODEL_KEY_POINT( "MODEL_KEY_POINT" );
	static const std::string POINTMEMBERNAME_LOW_POINT( "LOW_POINT" );
	static const std::string POINTMEMBERNAME_OVERLAP( "OVERLAP" );
	static const std::string POINTMEMBERNAME_IGNORE( "IGNORE" );
	static const std::string POINTMEMBERNAME_WAVE_PACKET_DESC_IDX( "WAVE_PACKET_DESC_IDX" );
	static const std::string POINTMEMBERNAME_WAVEFORM_OFFSET( "WAVEFORM_OFFSET" );

	struct SPDPoint
	{
		/**
		 * The return ID - A unique ID for each point within a pulse. 
		 * (In order recieved)
		 */
        boost::uint_fast16_t returnID;
		/**
		 * GPS time is the time at which the point was aquired.
		 */
		boost::uint_fast64_t gpsTime;
		/**
		 * Cartesian X Coordinate.
		 */
		double x;
		/**
		 * Cartesian Y Coordinate.
		 */
		double y;
		/**
		 * Cartesian Z Coordinate.
		 */
		float z;
		/**
		 * height about the ground (i.e., following removal of the ground surface)
		 */
		float height;
		/**
		 * Cartesian: Unused.
		 * Spherical: Range from the sensor
		 * Cylindrical: Range from the sensor
		 */
		float range;
		/**
		 * Amplitude of the signal returned to the sensor from this point
		 */
		float amplitudeReturn;
		/**
		 * Width of the signal returned to the sensor from this point
		 */
		float widthReturn;
		/**
		 * Red value - for visualisation.
		 */
        boost::uint_fast16_t red;
		/**
		 * Green value - for visualisation.
		 */
        boost::uint_fast16_t green;
		/**
		 * Blue value - for visualisation.
		 */
        boost::uint_fast16_t blue;
		/**
		 * Classification association.
		 */
        boost::uint_fast16_t classification;
		/**
		 * A user defined field
		 */
        boost::uint_fast32_t user;
		/**
		 * Model key point
		 */
        boost::int_fast16_t modelKeyPoint;
		/**
		 * Low Point
		 */
        boost::int_fast16_t lowPoint;
		/**
		 * In Overlapping region
		 */
        boost::int_fast16_t overlap;
		/**
		 * Define whether a point should be ignore during processing
		 */
        boost::int_fast16_t ignore;
		/**
		 * This value indicates a user defined record that describes the waveform packet. 
		 * A value of zero indicates that there is no waveform data associated with the lidar return
		 */
        boost::int_fast16_t wavePacketDescIdx;
		/**
		 * Offset in picoseconds from the first digitized value to the location with the waveform 
		 * packet that the associated return was detected
		 */
        boost::uint_fast32_t waveformOffset;
		
		friend std::ostream& operator<<(std::ostream& stream, SPDPoint &obj)
		{
			stream << "Return " << obj.returnID << std::endl;
			stream << "XYZI: [" << obj.x << "," << obj.y << "," << obj.z << "] " << obj.amplitudeReturn << std::endl;
			stream << "height: " << obj.height << std::endl;
			stream << "Range: " << obj.range << std::endl;
			stream << "Pulse Width: " << obj.widthReturn << std::endl;
			stream << "Classification: " << obj.classification << std::endl;
			stream << "RGB: [" << obj.red << "," << obj.green << "," << obj.blue << "]" << std::endl;
			stream << "GPS Time: " << obj.gpsTime << std::endl;
			stream << "User: " << obj.user << std::endl;
			return stream;
		};
		
		friend std::ostream& operator<<(std::ostream& stream, SPDPoint *obj)
		{
			stream << "Return " << obj->returnID << std::endl;
			stream << "XYZI: [" << obj->x << "," << obj->y << "," << obj->z << "] " << obj->amplitudeReturn << std::endl;
			stream << "height: " << obj->height << std::endl;
			stream << "Range: " << obj->range << std::endl;
			stream << "Pulse Width: " << obj->widthReturn << std::endl;
			stream << "Classification: " << obj->classification << std::endl;
			stream << "RGB: [" << obj->red << "," << obj->green << "," << obj->blue << "]" << std::endl;
			stream << "GPS Time: " << obj->gpsTime << std::endl;
			stream << "User: " << obj->user << std::endl;
			return stream;
		};
	};
	
	struct SPDPointH5V1
	{
		unsigned int returnID;
		/**
		 * GPS time is the time at which the point was aquired.
		 */
		double gpsTime;
		/**
		 * Cartesian X Coordinate.
		 */
		double x;
		/**
		 * Cartesian Y Coordinate.
		 */
		double y;
		/**
		 * Cartesian Z Coordinate.
		 */
		float z;
		/**
		 * height about the ground (i.e., following removal of the ground surface)
		 */
		float height;
		/**
		 * Cartesian: Unused.
		 * Spherical: Range from the sensor
		 * Cylindrical: Range from the sensor
		 */
		float range;
		/**
		 * Amplitude of the signal returned to the sensor from this point
		 */
		float amplitudeReturn;
		/**
		 * Width of the signal returned to the sensor from this point
		 */
		float widthReturn;
		/**
		 * Red value - for visualisation.
		 */
		unsigned int red;
		/**
		 * Green value - for visualisation.
		 */
		unsigned int green;
		/**
		 * Blue value - for visualisation.
		 */
		unsigned int blue;
		/**
		 * Classification association.
		 */
		unsigned int classification;
		/**
		 * A user defined field
		 */
		unsigned long user;
		/**
		 * Model key point
		 */
		unsigned int modelKeyPoint;
		/**
		 * Low Point
		 */
		unsigned int lowPoint;
		/**
		 * In Overlapping region
		 */
		unsigned int overlap;
		/**
		 * Define whether a point should be ignore during processing
		 */
		unsigned int ignore;
		/**
		 * This value indicates a user defined record that describes the waveform packet. 
		 * A value of zero indicates that there is no waveform data associated with the lidar return
		 */
		unsigned int wavePacketDescIdx;
		/**
		 * Offset in picoseconds from the first digitized value to the location with the waveform 
		 * packet that the associated return was detected
		 */
		unsigned long waveformOffset;
		
		friend std::ostream& operator<<(std::ostream& stream, SPDPointH5V1 &obj)
		{
			stream << "Return " << obj.returnID << std::endl;
			stream << "XYZI: [" << obj.x << "," << obj.y << "," << obj.z << "] " << obj.amplitudeReturn << std::endl;
			stream << "height: " << obj.height << std::endl;
			stream << "Range: " << obj.range << std::endl;
			stream << "Pulse Width: " << obj.widthReturn << std::endl;
			stream << "Classification: " << obj.classification << std::endl;
			stream << "RGB: [" << obj.red << "," << obj.green << "," << obj.blue << "]" << std::endl;
			stream << "GPS Time: " << obj.gpsTime << std::endl;
			stream << "User: " << obj.user << std::endl;
			return stream;
		};
		
		friend std::ostream& operator<<(std::ostream& stream, SPDPointH5V1 *obj)
		{
			stream << "Return " << obj->returnID << std::endl;
			stream << "XYZI: [" << obj->x << "," << obj->y << "," << obj->z << "] " << obj->amplitudeReturn << std::endl;
			stream << "height: " << obj->height << std::endl;
			stream << "Range: " << obj->range << std::endl;
			stream << "Pulse Width: " << obj->widthReturn << std::endl;
			stream << "Classification: " << obj->classification << std::endl;
			stream << "RGB: [" << obj->red << "," << obj->green << "," << obj->blue << "]" << std::endl;
			stream << "GPS Time: " << obj->gpsTime << std::endl;
			stream << "User: " << obj->user << std::endl;
			return stream;
		};
	};
	
    struct SPDPointH5V2
	{
		unsigned int returnID;
		/**
		 * GPS time is the time at which the point was aquired.
		 */
		unsigned long long gpsTime;
		/**
		 * Cartesian X Coordinate.
		 */
		double x;
		/**
		 * Cartesian Y Coordinate.
		 */
		double y;
		/**
		 * Cartesian Z Coordinate.
		 */
		float z;
		/**
		 * height about the ground (i.e., following removal of the ground surface)
		 */
		float height;
		/**
		 * Cartesian: Unused.
		 * Spherical: Range from the sensor
		 * Cylindrical: Range from the sensor
		 */
		float range;
		/**
		 * Amplitude of the signal returned to the sensor from this point
		 */
		float amplitudeReturn;
		/**
		 * Width of the signal returned to the sensor from this point
		 */
		float widthReturn;
		/**
		 * Red value - for visualisation.
		 */
		unsigned int red;
		/**
		 * Green value - for visualisation.
		 */
		unsigned int green;
		/**
		 * Blue value - for visualisation.
		 */
		unsigned int blue;
		/**
		 * Classification association.
		 */
		unsigned int classification;
		/**
		 * A user defined field
		 */
		unsigned long user;
		/**
		 * Define whether a point should be ignore during processing
		 */
		unsigned int ignore;
		/**
		 * This value indicates a user defined record that describes the waveform packet. 
		 * A value of zero indicates that there is no waveform data associated with the lidar return
		 */
		unsigned int wavePacketDescIdx;
		/**
		 * Offset in picoseconds from the first digitized value to the location with the waveform 
		 * packet that the associated return was detected
		 */
		unsigned long waveformOffset;
		
		friend std::ostream& operator<<(std::ostream& stream, SPDPointH5V2 &obj)
		{
			stream << "Return " << obj.returnID << std::endl;
			stream << "XYZI: [" << obj.x << "," << obj.y << "," << obj.z << "] " << obj.amplitudeReturn << std::endl;
			stream << "height: " << obj.height << std::endl;
			stream << "Range: " << obj.range << std::endl;
			stream << "Pulse Width: " << obj.widthReturn << std::endl;
			stream << "Classification: " << obj.classification << std::endl;
			stream << "RGB: [" << obj.red << "," << obj.green << "," << obj.blue << "]" << std::endl;
			stream << "GPS Time: " << obj.gpsTime << std::endl;
			stream << "User: " << obj.user << std::endl;
			return stream;
		};
		
		friend std::ostream& operator<<(std::ostream& stream, SPDPointH5V2 *obj)
		{
			stream << "Return " << obj->returnID << std::endl;
			stream << "XYZI: [" << obj->x << "," << obj->y << "," << obj->z << "] " << obj->amplitudeReturn << std::endl;
			stream << "height: " << obj->height << std::endl;
			stream << "Range: " << obj->range << std::endl;
			stream << "Pulse Width: " << obj->widthReturn << std::endl;
			stream << "Classification: " << obj->classification << std::endl;
			stream << "RGB: [" << obj->red << "," << obj->green << "," << obj->blue << "]" << std::endl;
			stream << "GPS Time: " << obj->gpsTime << std::endl;
			stream << "User: " << obj->user << std::endl;
			return stream;
		};
	};
    
	/**
	 * Provides useful utilities for manipulating an SPDPoint.
	 */
	class SPDPointUtils
	{
	public:
		SPDPointUtils();
		double distanceXY(SPDPoint *pt1, SPDPoint *pt2);
		double distanceXY(double x, double y, SPDPoint *pt);
		double distanceXYZ(SPDPoint *pt1, SPDPoint *pt2);
		double distanceXYZ(double x, double y, double z, SPDPoint *pt);
		H5::CompType* createSPDPointV1DataTypeDisk();
		H5::CompType* createSPDPointV1DataTypeMemory();
        H5::CompType* createSPDPointV2DataTypeDisk();
		H5::CompType* createSPDPointV2DataTypeMemory();        
		void initSPDPoint(SPDPoint *pt);
		void initSPDPoint(SPDPointH5V1 *pt);
        void initSPDPoint(SPDPointH5V2 *pt);
		SPDPoint* createSPDPointCopy(SPDPoint *pt);
        void copySPDPointTo(SPDPoint *pt, SPDPoint *pt_out);
        
        /* For Point Version 1 */
		SPDPoint* createSPDPointCopy(SPDPointH5V1 *pt);
		SPDPointH5V1* createSPDPointH5Copy(SPDPointH5V1 *pt);
		SPDPointH5V1* createSPDPointH5V1Copy(SPDPoint *pt);
		void copySPDPointTo(SPDPoint *pt, SPDPointH5V1 *pt_out);
		void copySPDPointH5To(SPDPointH5V1 *pt, SPDPointH5V1 *pt_out);
		void copySPDPointH5To(SPDPointH5V1 *pt, SPDPoint *pt_out);
        
        /* For Point Version 2 */
		SPDPoint* createSPDPointCopy(SPDPointH5V2 *pt);
		SPDPointH5V2* createSPDPointH5Copy(SPDPointH5V2 *pt);
		SPDPointH5V2* createSPDPointH5V2Copy(SPDPoint *pt);
		void copySPDPointTo(SPDPoint *pt, SPDPointH5V2 *pt_out);
		void copySPDPointH5To(SPDPointH5V2 *pt, SPDPointH5V2 *pt_out);
		void copySPDPointH5To(SPDPointH5V2 *pt, SPDPoint *pt_out);
        
        void verticalHeightBinPoints(std::vector<SPDPoint*> *pts, std::vector<SPDPoint*> **bins,boost::uint_fast32_t numBins, float min, float max, bool ignorePtsOverMax, bool ignoreGrd, float minHeightThres)throw(SPDProcessingException);
        void verticalElevationBinPoints(std::vector<SPDPoint*> *pts, std::vector<SPDPoint*> **bins,boost::uint_fast32_t numBins, float min, float max)throw(SPDProcessingException);
		~SPDPointUtils();		
	};
	
	inline bool cmpSPDPointTime(SPDPoint *pt1, SPDPoint *pt2) 
	{
		return pt1->gpsTime > pt2->gpsTime;
	}
}

#endif

