/*
 *  SPDPoint.cpp
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

#include "spd/SPDPoint.h"

namespace spdlib
{
	
	SPDPointUtils::SPDPointUtils()
	{
		
	}
	
	double SPDPointUtils::distanceXY(SPDPoint *pt1, SPDPoint *pt2)
	{
		double diffX = pt1->x - pt2->x;
		double diffY = pt1->y - pt2->y;
		return sqrt((diffX*diffX)+(diffY*diffY));
	}
	
	double SPDPointUtils::distanceXY(double x, double y, SPDPoint *pt)
	{
		double diffX = x - pt->x;
		double diffY = y - pt->y;
		return sqrt((diffX*diffX)+(diffY*diffY));
	}
	
	double SPDPointUtils::distanceXYZ(SPDPoint *pt1, SPDPoint *pt2)
	{
		double diffX = pt1->x - pt2->x;
		double diffY = pt1->y - pt2->y;
		double diffZ = pt1->z - pt2->z;
		return sqrt((diffX*diffX)+(diffY*diffY)+(diffZ*diffZ));
	}
	
	double SPDPointUtils::distanceXYZ(double x, double y, double z, SPDPoint *pt)
	{
		double diffX = x - pt->x;
		double diffY = y - pt->y;
		double diffZ = z - pt->z;
		return sqrt((diffX*diffX)+(diffY*diffY)+(diffZ*diffZ));
	}
	
	H5::CompType* SPDPointUtils::createSPDPointV1DataTypeDisk()
	{
		H5::CompType *spdPointDataType = new H5::CompType( sizeof(SPDPointH5V1) );
		spdPointDataType->insertMember(POINTMEMBERNAME_RETURN_ID, HOFFSET(SPDPointH5V1, returnID), H5::PredType::STD_U8LE);
		spdPointDataType->insertMember(POINTMEMBERNAME_GPS_TIME, HOFFSET(SPDPointH5V1, gpsTime), H5::PredType::IEEE_F64LE);
		spdPointDataType->insertMember(POINTMEMBERNAME_X, HOFFSET(SPDPointH5V1, x), H5::PredType::IEEE_F64LE);
		spdPointDataType->insertMember(POINTMEMBERNAME_Y, HOFFSET(SPDPointH5V1, y), H5::PredType::IEEE_F64LE);
		spdPointDataType->insertMember(POINTMEMBERNAME_Z, HOFFSET(SPDPointH5V1, z), H5::PredType::IEEE_F32LE);
		spdPointDataType->insertMember(POINTMEMBERNAME_HEIGHT, HOFFSET(SPDPointH5V1, height), H5::PredType::IEEE_F32LE);
		spdPointDataType->insertMember(POINTMEMBERNAME_RANGE, HOFFSET(SPDPointH5V1, range), H5::PredType::IEEE_F32LE);
		spdPointDataType->insertMember(POINTMEMBERNAME_AMPLITUDE_RETURN, HOFFSET(SPDPointH5V1, amplitudeReturn), H5::PredType::IEEE_F32LE);
		spdPointDataType->insertMember(POINTMEMBERNAME_WIDTH_RETURN, HOFFSET(SPDPointH5V1, widthReturn), H5::PredType::IEEE_F32LE);		
		spdPointDataType->insertMember(POINTMEMBERNAME_RED, HOFFSET(SPDPointH5V1, red), H5::PredType::STD_U16LE);
		spdPointDataType->insertMember(POINTMEMBERNAME_GREEN, HOFFSET(SPDPointH5V1, green), H5::PredType::STD_U16LE);
		spdPointDataType->insertMember(POINTMEMBERNAME_BLUE, HOFFSET(SPDPointH5V1, blue), H5::PredType::STD_U16LE);
		spdPointDataType->insertMember(POINTMEMBERNAME_CLASSIFICATION, HOFFSET(SPDPointH5V1, classification), H5::PredType::STD_U8LE);
		spdPointDataType->insertMember(POINTMEMBERNAME_USER, HOFFSET(SPDPointH5V1, user), H5::PredType::STD_U32LE);
		spdPointDataType->insertMember(POINTMEMBERNAME_MODEL_KEY_POINT, HOFFSET(SPDPointH5V1, modelKeyPoint), H5::PredType::STD_U8LE);
		spdPointDataType->insertMember(POINTMEMBERNAME_LOW_POINT, HOFFSET(SPDPointH5V1, lowPoint), H5::PredType::STD_U8LE);
		spdPointDataType->insertMember(POINTMEMBERNAME_OVERLAP, HOFFSET(SPDPointH5V1, overlap), H5::PredType::STD_U8LE);
		spdPointDataType->insertMember(POINTMEMBERNAME_IGNORE, HOFFSET(SPDPointH5V1, ignore), H5::PredType::STD_U8LE);
		spdPointDataType->insertMember(POINTMEMBERNAME_WAVE_PACKET_DESC_IDX, HOFFSET(SPDPointH5V1, wavePacketDescIdx), H5::PredType::STD_I16LE);
		spdPointDataType->insertMember(POINTMEMBERNAME_WAVEFORM_OFFSET, HOFFSET(SPDPointH5V1, waveformOffset), H5::PredType::STD_U32LE);
		return spdPointDataType;
	}
	
	H5::CompType* SPDPointUtils::createSPDPointV1DataTypeMemory()
	{
		H5::CompType *spdPointDataType = new H5::CompType( sizeof(SPDPointH5V1) );
		spdPointDataType->insertMember(POINTMEMBERNAME_RETURN_ID, HOFFSET(SPDPointH5V1, returnID), H5::PredType::NATIVE_UINT);
		spdPointDataType->insertMember(POINTMEMBERNAME_GPS_TIME, HOFFSET(SPDPointH5V1, gpsTime), H5::PredType::NATIVE_DOUBLE);
		spdPointDataType->insertMember(POINTMEMBERNAME_X, HOFFSET(SPDPointH5V1, x), H5::PredType::NATIVE_DOUBLE);
		spdPointDataType->insertMember(POINTMEMBERNAME_Y, HOFFSET(SPDPointH5V1, y), H5::PredType::NATIVE_DOUBLE);
		spdPointDataType->insertMember(POINTMEMBERNAME_Z, HOFFSET(SPDPointH5V1, z), H5::PredType::NATIVE_FLOAT);
		spdPointDataType->insertMember(POINTMEMBERNAME_HEIGHT, HOFFSET(SPDPointH5V1, height), H5::PredType::NATIVE_FLOAT);
		spdPointDataType->insertMember(POINTMEMBERNAME_RANGE, HOFFSET(SPDPointH5V1, range), H5::PredType::NATIVE_FLOAT);
		spdPointDataType->insertMember(POINTMEMBERNAME_AMPLITUDE_RETURN, HOFFSET(SPDPointH5V1, amplitudeReturn), H5::PredType::NATIVE_FLOAT);
		spdPointDataType->insertMember(POINTMEMBERNAME_WIDTH_RETURN, HOFFSET(SPDPointH5V1, widthReturn), H5::PredType::NATIVE_FLOAT);		
		spdPointDataType->insertMember(POINTMEMBERNAME_RED, HOFFSET(SPDPointH5V1, red), H5::PredType::NATIVE_UINT);
		spdPointDataType->insertMember(POINTMEMBERNAME_GREEN, HOFFSET(SPDPointH5V1, green), H5::PredType::NATIVE_UINT);
		spdPointDataType->insertMember(POINTMEMBERNAME_BLUE, HOFFSET(SPDPointH5V1, blue), H5::PredType::NATIVE_UINT);
		spdPointDataType->insertMember(POINTMEMBERNAME_CLASSIFICATION, HOFFSET(SPDPointH5V1, classification), H5::PredType::NATIVE_UINT);
		spdPointDataType->insertMember(POINTMEMBERNAME_USER, HOFFSET(SPDPointH5V1, user), H5::PredType::NATIVE_ULONG);
		spdPointDataType->insertMember(POINTMEMBERNAME_MODEL_KEY_POINT, HOFFSET(SPDPointH5V1, modelKeyPoint), H5::PredType::NATIVE_UINT);
		spdPointDataType->insertMember(POINTMEMBERNAME_LOW_POINT, HOFFSET(SPDPointH5V1, lowPoint), H5::PredType::NATIVE_UINT);
		spdPointDataType->insertMember(POINTMEMBERNAME_OVERLAP, HOFFSET(SPDPointH5V1, overlap), H5::PredType::NATIVE_UINT);
		spdPointDataType->insertMember(POINTMEMBERNAME_IGNORE, HOFFSET(SPDPointH5V1, ignore), H5::PredType::NATIVE_UINT);
		spdPointDataType->insertMember(POINTMEMBERNAME_WAVE_PACKET_DESC_IDX, HOFFSET(SPDPointH5V1, wavePacketDescIdx), H5::PredType::NATIVE_UINT);
		spdPointDataType->insertMember(POINTMEMBERNAME_WAVEFORM_OFFSET, HOFFSET(SPDPointH5V1, waveformOffset), H5::PredType::NATIVE_ULONG);
		return spdPointDataType;
	}
    
    H5::CompType* SPDPointUtils::createSPDPointV2DataTypeDisk()
	{
		H5::CompType *spdPointDataType = new H5::CompType( sizeof(SPDPointH5V2) );
		spdPointDataType->insertMember(POINTMEMBERNAME_RETURN_ID, HOFFSET(SPDPointH5V2, returnID), H5::PredType::STD_U8LE);
		spdPointDataType->insertMember(POINTMEMBERNAME_GPS_TIME, HOFFSET(SPDPointH5V2, gpsTime), H5::PredType::IEEE_F64LE);
		spdPointDataType->insertMember(POINTMEMBERNAME_X, HOFFSET(SPDPointH5V2, x), H5::PredType::IEEE_F64LE);
		spdPointDataType->insertMember(POINTMEMBERNAME_Y, HOFFSET(SPDPointH5V2, y), H5::PredType::IEEE_F64LE);
		spdPointDataType->insertMember(POINTMEMBERNAME_Z, HOFFSET(SPDPointH5V2, z), H5::PredType::IEEE_F32LE);
		spdPointDataType->insertMember(POINTMEMBERNAME_HEIGHT, HOFFSET(SPDPointH5V2, height), H5::PredType::IEEE_F32LE);
		spdPointDataType->insertMember(POINTMEMBERNAME_RANGE, HOFFSET(SPDPointH5V2, range), H5::PredType::IEEE_F32LE);
		spdPointDataType->insertMember(POINTMEMBERNAME_AMPLITUDE_RETURN, HOFFSET(SPDPointH5V2, amplitudeReturn), H5::PredType::IEEE_F32LE);
		spdPointDataType->insertMember(POINTMEMBERNAME_WIDTH_RETURN, HOFFSET(SPDPointH5V2, widthReturn), H5::PredType::IEEE_F32LE);		
		spdPointDataType->insertMember(POINTMEMBERNAME_RED, HOFFSET(SPDPointH5V2, red), H5::PredType::STD_U16LE);
		spdPointDataType->insertMember(POINTMEMBERNAME_GREEN, HOFFSET(SPDPointH5V2, green), H5::PredType::STD_U16LE);
		spdPointDataType->insertMember(POINTMEMBERNAME_BLUE, HOFFSET(SPDPointH5V2, blue), H5::PredType::STD_U16LE);
		spdPointDataType->insertMember(POINTMEMBERNAME_CLASSIFICATION, HOFFSET(SPDPointH5V2, classification), H5::PredType::STD_U8LE);
		spdPointDataType->insertMember(POINTMEMBERNAME_USER, HOFFSET(SPDPointH5V2, user), H5::PredType::STD_U32LE);
		spdPointDataType->insertMember(POINTMEMBERNAME_IGNORE, HOFFSET(SPDPointH5V2, ignore), H5::PredType::STD_U8LE);
		spdPointDataType->insertMember(POINTMEMBERNAME_WAVE_PACKET_DESC_IDX, HOFFSET(SPDPointH5V2, wavePacketDescIdx), H5::PredType::STD_I16LE);
		spdPointDataType->insertMember(POINTMEMBERNAME_WAVEFORM_OFFSET, HOFFSET(SPDPointH5V2, waveformOffset), H5::PredType::STD_U32LE);
		return spdPointDataType;
	}
	
	H5::CompType* SPDPointUtils::createSPDPointV2DataTypeMemory()
	{
		H5::CompType *spdPointDataType = new H5::CompType( sizeof(SPDPointH5V2) );
		spdPointDataType->insertMember(POINTMEMBERNAME_RETURN_ID, HOFFSET(SPDPointH5V2, returnID), H5::PredType::NATIVE_UINT);
		spdPointDataType->insertMember(POINTMEMBERNAME_GPS_TIME, HOFFSET(SPDPointH5V2, gpsTime), H5::PredType::NATIVE_DOUBLE);
		spdPointDataType->insertMember(POINTMEMBERNAME_X, HOFFSET(SPDPointH5V2, x), H5::PredType::NATIVE_DOUBLE);
		spdPointDataType->insertMember(POINTMEMBERNAME_Y, HOFFSET(SPDPointH5V2, y), H5::PredType::NATIVE_DOUBLE);
		spdPointDataType->insertMember(POINTMEMBERNAME_Z, HOFFSET(SPDPointH5V2, z), H5::PredType::NATIVE_FLOAT);
		spdPointDataType->insertMember(POINTMEMBERNAME_HEIGHT, HOFFSET(SPDPointH5V2, height), H5::PredType::NATIVE_FLOAT);
		spdPointDataType->insertMember(POINTMEMBERNAME_RANGE, HOFFSET(SPDPointH5V2, range), H5::PredType::NATIVE_FLOAT);
		spdPointDataType->insertMember(POINTMEMBERNAME_AMPLITUDE_RETURN, HOFFSET(SPDPointH5V2, amplitudeReturn), H5::PredType::NATIVE_FLOAT);
		spdPointDataType->insertMember(POINTMEMBERNAME_WIDTH_RETURN, HOFFSET(SPDPointH5V2, widthReturn), H5::PredType::NATIVE_FLOAT);		
		spdPointDataType->insertMember(POINTMEMBERNAME_RED, HOFFSET(SPDPointH5V2, red), H5::PredType::NATIVE_UINT);
		spdPointDataType->insertMember(POINTMEMBERNAME_GREEN, HOFFSET(SPDPointH5V2, green), H5::PredType::NATIVE_UINT);
		spdPointDataType->insertMember(POINTMEMBERNAME_BLUE, HOFFSET(SPDPointH5V2, blue), H5::PredType::NATIVE_UINT);
		spdPointDataType->insertMember(POINTMEMBERNAME_CLASSIFICATION, HOFFSET(SPDPointH5V2, classification), H5::PredType::NATIVE_UINT);
		spdPointDataType->insertMember(POINTMEMBERNAME_USER, HOFFSET(SPDPointH5V2, user), H5::PredType::NATIVE_ULONG);
		spdPointDataType->insertMember(POINTMEMBERNAME_IGNORE, HOFFSET(SPDPointH5V2, ignore), H5::PredType::NATIVE_UINT);
		spdPointDataType->insertMember(POINTMEMBERNAME_WAVE_PACKET_DESC_IDX, HOFFSET(SPDPointH5V2, wavePacketDescIdx), H5::PredType::NATIVE_UINT);
		spdPointDataType->insertMember(POINTMEMBERNAME_WAVEFORM_OFFSET, HOFFSET(SPDPointH5V2, waveformOffset), H5::PredType::NATIVE_ULONG);
		return spdPointDataType;
	}
	
	void SPDPointUtils::initSPDPoint(SPDPoint *pt)
	{
		pt->x = 0;
		pt->y = 0;
		pt->z = 0;
		pt->height = 0;
		pt->range = 0;
		pt->amplitudeReturn = 0;
		pt->widthReturn = 0;
		pt->red = 0;
		pt->green = 0;
		pt->blue = 0;
		pt->classification = SPD_UNCLASSIFIED;
		pt->returnID = 0;
		pt->gpsTime = 0;
		pt->user = 0;
		pt->modelKeyPoint = SPD_FALSE;
		pt->lowPoint = SPD_FALSE;
		pt->overlap = SPD_FALSE;
		pt->ignore = SPD_FALSE;
		pt->wavePacketDescIdx = 0;
		pt->waveformOffset = 0;
	}
	
	void SPDPointUtils::initSPDPoint(SPDPointH5V1 *pt)
	{
		pt->x = 0;
		pt->y = 0;
		pt->z = 0;
		pt->height = 0;
		pt->range = 0;
		pt->amplitudeReturn = 0;
		pt->widthReturn = 0;
		pt->red = 0;
		pt->green = 0;
		pt->blue = 0;
		pt->classification = SPD_UNCLASSIFIED;
		pt->returnID = 0;
		pt->gpsTime = 0;
		pt->user = 0;
		pt->modelKeyPoint = SPD_FALSE;
		pt->lowPoint = SPD_FALSE;
		pt->overlap = SPD_FALSE;
		pt->ignore = SPD_FALSE;
		pt->wavePacketDescIdx = 0;
		pt->waveformOffset = 0;
	}
    
    void SPDPointUtils::initSPDPoint(SPDPointH5V2 *pt)
	{
		pt->x = 0;
		pt->y = 0;
		pt->z = 0;
		pt->height = 0;
		pt->range = 0;
		pt->amplitudeReturn = 0;
		pt->widthReturn = 0;
		pt->red = 0;
		pt->green = 0;
		pt->blue = 0;
		pt->classification = SPD_UNCLASSIFIED;
		pt->returnID = 0;
		pt->gpsTime = 0;
		pt->user = 0;
		pt->ignore = SPD_FALSE;
		pt->wavePacketDescIdx = 0;
		pt->waveformOffset = 0;
	}
	
	SPDPoint* SPDPointUtils::createSPDPointCopy(SPDPoint *pt)
	{
		SPDPoint *pt_out = new SPDPoint();
		pt_out->x = pt->x;
		pt_out->y = pt->y;
		pt_out->z = pt->z;
		pt_out->height = pt->height;
		pt_out->range = pt->range;
		pt_out->amplitudeReturn = pt->amplitudeReturn;
		pt_out->widthReturn = pt->widthReturn;
		pt_out->red = pt->red;
		pt_out->green = pt->green;
		pt_out->blue = pt->blue;
		pt_out->classification = pt->classification;
		pt_out->returnID = pt->returnID;
		pt_out->gpsTime = pt->gpsTime;
		pt_out->user = pt->user;
		pt_out->modelKeyPoint = pt->modelKeyPoint;
		pt_out->lowPoint = pt->lowPoint;
		pt_out->overlap = pt->overlap;
		pt_out->ignore = pt->ignore;
		pt_out->wavePacketDescIdx = pt->wavePacketDescIdx;
		pt_out->waveformOffset = pt->waveformOffset;
		
		return pt_out;
	}
    
    void SPDPointUtils::copySPDPointTo(SPDPoint *pt, SPDPoint *pt_out)
	{
		pt_out->x = pt->x;
		pt_out->y = pt->y;
		pt_out->z = pt->z;
		pt_out->height = pt->height;
		pt_out->range = pt->range;
		pt_out->amplitudeReturn = pt->amplitudeReturn;
		pt_out->widthReturn = pt->widthReturn;
		pt_out->red = pt->red;
		pt_out->green = pt->green;
		pt_out->blue = pt->blue;
		pt_out->classification = pt->classification;
		pt_out->returnID = pt->returnID;
		pt_out->gpsTime = pt->gpsTime;
		pt_out->user = pt->user;
		pt_out->modelKeyPoint = pt->modelKeyPoint;
		pt_out->lowPoint = pt->lowPoint;
		pt_out->overlap = pt->overlap;
		pt_out->ignore = pt->ignore;
		pt_out->wavePacketDescIdx = pt->wavePacketDescIdx;
		pt_out->waveformOffset = pt->waveformOffset;
	}
	
	SPDPoint* SPDPointUtils::createSPDPointCopy(SPDPointH5V1 *pt)
	{
		SPDPoint *pt_out = new SPDPoint();
		pt_out->x = pt->x;
		pt_out->y = pt->y;
		pt_out->z = pt->z;
		pt_out->height = pt->height;
		pt_out->range = pt->range;
		pt_out->amplitudeReturn = pt->amplitudeReturn;
		pt_out->widthReturn = pt->widthReturn;
		pt_out->red = pt->red;
		pt_out->green = pt->green;
		pt_out->blue = pt->blue;
		pt_out->classification = pt->classification;
		pt_out->returnID = pt->returnID;
		pt_out->gpsTime = pt->gpsTime;
		pt_out->user = pt->user;
		pt_out->modelKeyPoint = pt->modelKeyPoint;
		pt_out->lowPoint = pt->lowPoint;
		pt_out->overlap = pt->overlap;
		pt_out->ignore = pt->ignore;
		pt_out->wavePacketDescIdx = pt->wavePacketDescIdx;
		pt_out->waveformOffset = pt->waveformOffset;
		
		return pt_out;
	}
	
	SPDPointH5V1* SPDPointUtils::createSPDPointH5Copy(SPDPointH5V1 *pt)
	{
		SPDPointH5V1 *pt_out = new SPDPointH5V1();
		pt_out->x = pt->x;
		pt_out->y = pt->y;
		pt_out->z = pt->z;
		pt_out->height = pt->height;
		pt_out->range = pt->range;
		pt_out->amplitudeReturn = pt->amplitudeReturn;
		pt_out->widthReturn = pt->widthReturn;
		pt_out->red = pt->red;
		pt_out->green = pt->green;
		pt_out->blue = pt->blue;
		pt_out->classification = pt->classification;
		pt_out->returnID = pt->returnID;
		pt_out->gpsTime = pt->gpsTime;
		pt_out->user = pt->user;
		pt_out->modelKeyPoint = pt->modelKeyPoint;
		pt_out->lowPoint = pt->lowPoint;
		pt_out->overlap = pt->overlap;
		pt_out->ignore = pt->ignore;
		pt_out->wavePacketDescIdx = pt->wavePacketDescIdx;
		pt_out->waveformOffset = pt->waveformOffset;
		
		return pt_out;
	}
	
	SPDPointH5V1* SPDPointUtils::createSPDPointH5V1Copy(SPDPoint *pt)
	{
		SPDPointH5V1 *pt_out = new SPDPointH5V1();
		pt_out->x = pt->x;
		pt_out->y = pt->y;
		pt_out->z = pt->z;
		pt_out->height = pt->height;
		pt_out->range = pt->range;
		pt_out->amplitudeReturn = pt->amplitudeReturn;
		pt_out->widthReturn = pt->widthReturn;
		pt_out->red = pt->red;
		pt_out->green = pt->green;
		pt_out->blue = pt->blue;
		pt_out->classification = pt->classification;
		pt_out->returnID = pt->returnID;
		pt_out->gpsTime = pt->gpsTime;
		pt_out->user = pt->user;
		pt_out->modelKeyPoint = pt->modelKeyPoint;
		pt_out->lowPoint = pt->lowPoint;
		pt_out->overlap = pt->overlap;
		pt_out->ignore = pt->ignore;
		pt_out->wavePacketDescIdx = pt->wavePacketDescIdx;
		pt_out->waveformOffset = pt->waveformOffset;
		
		return pt_out;
	}	
	
	void SPDPointUtils::copySPDPointTo(SPDPoint *pt, SPDPointH5V1 *pt_out)
	{
		pt_out->x = pt->x;
		pt_out->y = pt->y;
		pt_out->z = pt->z;
		pt_out->height = pt->height;
		pt_out->range = pt->range;
		pt_out->amplitudeReturn = pt->amplitudeReturn;
		pt_out->widthReturn = pt->widthReturn;
		pt_out->red = pt->red;
		pt_out->green = pt->green;
		pt_out->blue = pt->blue;
		pt_out->classification = pt->classification;
		pt_out->returnID = pt->returnID;
		pt_out->gpsTime = pt->gpsTime;
		pt_out->user = pt->user;
		pt_out->modelKeyPoint = pt->modelKeyPoint;
		pt_out->lowPoint = pt->lowPoint;
		pt_out->overlap = pt->overlap;
		pt_out->ignore = pt->ignore;
		pt_out->wavePacketDescIdx = pt->wavePacketDescIdx;
		pt_out->waveformOffset = pt->waveformOffset;
	}
	
	void SPDPointUtils::copySPDPointH5To(SPDPointH5V1 *pt, SPDPointH5V1 *pt_out)
	{
		pt_out->x = pt->x;
		pt_out->y = pt->y;
		pt_out->z = pt->z;
		pt_out->height = pt->height;
		pt_out->range = pt->range;
		pt_out->amplitudeReturn = pt->amplitudeReturn;
		pt_out->widthReturn = pt->widthReturn;
		pt_out->red = pt->red;
		pt_out->green = pt->green;
		pt_out->blue = pt->blue;
		pt_out->classification = pt->classification;
		pt_out->returnID = pt->returnID;
		pt_out->gpsTime = pt->gpsTime;
		pt_out->user = pt->user;
		pt_out->modelKeyPoint = pt->modelKeyPoint;
		pt_out->lowPoint = pt->lowPoint;
		pt_out->overlap = pt->overlap;
		pt_out->ignore = pt->ignore;
		pt_out->wavePacketDescIdx = pt->wavePacketDescIdx;
		pt_out->waveformOffset = pt->waveformOffset;
	}
	
	void SPDPointUtils::copySPDPointH5To(SPDPointH5V1 *pt, SPDPoint *pt_out)
	{
		pt_out->x = pt->x;
		pt_out->y = pt->y;
		pt_out->z = pt->z;
		pt_out->height = pt->height;
		pt_out->range = pt->range;
		pt_out->amplitudeReturn = pt->amplitudeReturn;
		pt_out->widthReturn = pt->widthReturn;
		pt_out->red = pt->red;
		pt_out->green = pt->green;
		pt_out->blue = pt->blue;
		pt_out->classification = pt->classification;
		pt_out->returnID = pt->returnID;
		pt_out->gpsTime = pt->gpsTime;
		pt_out->user = pt->user;
		pt_out->modelKeyPoint = pt->modelKeyPoint;
		pt_out->lowPoint = pt->lowPoint;
		pt_out->overlap = pt->overlap;
		pt_out->ignore = pt->ignore;
		pt_out->wavePacketDescIdx = pt->wavePacketDescIdx;
		pt_out->waveformOffset = pt->waveformOffset;
	}

    SPDPoint* SPDPointUtils::createSPDPointCopy(SPDPointH5V2 *pt)
	{
		SPDPoint *pt_out = new SPDPoint();
		pt_out->x = pt->x;
		pt_out->y = pt->y;
		pt_out->z = pt->z;
		pt_out->height = pt->height;
		pt_out->range = pt->range;
		pt_out->amplitudeReturn = pt->amplitudeReturn;
		pt_out->widthReturn = pt->widthReturn;
		pt_out->red = pt->red;
		pt_out->green = pt->green;
		pt_out->blue = pt->blue;
		pt_out->classification = pt->classification;
		pt_out->returnID = pt->returnID;
		pt_out->gpsTime = pt->gpsTime;
		pt_out->user = pt->user;
		pt_out->ignore = pt->ignore;
		pt_out->wavePacketDescIdx = pt->wavePacketDescIdx;
		pt_out->waveformOffset = pt->waveformOffset;
		
		return pt_out;
	}
	
	SPDPointH5V2* SPDPointUtils::createSPDPointH5Copy(SPDPointH5V2 *pt)
	{
		SPDPointH5V2 *pt_out = new SPDPointH5V2();
		pt_out->x = pt->x;
		pt_out->y = pt->y;
		pt_out->z = pt->z;
		pt_out->height = pt->height;
		pt_out->range = pt->range;
		pt_out->amplitudeReturn = pt->amplitudeReturn;
		pt_out->widthReturn = pt->widthReturn;
		pt_out->red = pt->red;
		pt_out->green = pt->green;
		pt_out->blue = pt->blue;
		pt_out->classification = pt->classification;
		pt_out->returnID = pt->returnID;
		pt_out->gpsTime = pt->gpsTime;
		pt_out->user = pt->user;
		pt_out->ignore = pt->ignore;
		pt_out->wavePacketDescIdx = pt->wavePacketDescIdx;
		pt_out->waveformOffset = pt->waveformOffset;
		
		return pt_out;
	}
	
	SPDPointH5V2* SPDPointUtils::createSPDPointH5V2Copy(SPDPoint *pt)
	{
		SPDPointH5V2 *pt_out = new SPDPointH5V2();
		pt_out->x = pt->x;
		pt_out->y = pt->y;
		pt_out->z = pt->z;
		pt_out->height = pt->height;
		pt_out->range = pt->range;
		pt_out->amplitudeReturn = pt->amplitudeReturn;
		pt_out->widthReturn = pt->widthReturn;
		pt_out->red = pt->red;
		pt_out->green = pt->green;
		pt_out->blue = pt->blue;
		pt_out->classification = pt->classification;
		pt_out->returnID = pt->returnID;
		pt_out->gpsTime = pt->gpsTime;
		pt_out->user = pt->user;
		pt_out->ignore = pt->ignore;
		pt_out->wavePacketDescIdx = pt->wavePacketDescIdx;
		pt_out->waveformOffset = pt->waveformOffset;
		
		return pt_out;
	}	
	
	void SPDPointUtils::copySPDPointTo(SPDPoint *pt, SPDPointH5V2 *pt_out)
	{
		pt_out->x = pt->x;
		pt_out->y = pt->y;
		pt_out->z = pt->z;
		pt_out->height = pt->height;
		pt_out->range = pt->range;
		pt_out->amplitudeReturn = pt->amplitudeReturn;
		pt_out->widthReturn = pt->widthReturn;
		pt_out->red = pt->red;
		pt_out->green = pt->green;
		pt_out->blue = pt->blue;
		pt_out->classification = pt->classification;
		pt_out->returnID = pt->returnID;
		pt_out->gpsTime = pt->gpsTime;
		pt_out->user = pt->user;
		pt_out->ignore = pt->ignore;
		pt_out->wavePacketDescIdx = pt->wavePacketDescIdx;
		pt_out->waveformOffset = pt->waveformOffset;
	}
	
	void SPDPointUtils::copySPDPointH5To(SPDPointH5V2 *pt, SPDPointH5V2 *pt_out)
	{
		pt_out->x = pt->x;
		pt_out->y = pt->y;
		pt_out->z = pt->z;
		pt_out->height = pt->height;
		pt_out->range = pt->range;
		pt_out->amplitudeReturn = pt->amplitudeReturn;
		pt_out->widthReturn = pt->widthReturn;
		pt_out->red = pt->red;
		pt_out->green = pt->green;
		pt_out->blue = pt->blue;
		pt_out->classification = pt->classification;
		pt_out->returnID = pt->returnID;
		pt_out->gpsTime = pt->gpsTime;
		pt_out->user = pt->user;
		pt_out->ignore = pt->ignore;
		pt_out->wavePacketDescIdx = pt->wavePacketDescIdx;
		pt_out->waveformOffset = pt->waveformOffset;
	}
	
	void SPDPointUtils::copySPDPointH5To(SPDPointH5V2 *pt, SPDPoint *pt_out)
	{
		pt_out->x = pt->x;
		pt_out->y = pt->y;
		pt_out->z = pt->z;
		pt_out->height = pt->height;
		pt_out->range = pt->range;
		pt_out->amplitudeReturn = pt->amplitudeReturn;
		pt_out->widthReturn = pt->widthReturn;
		pt_out->red = pt->red;
		pt_out->green = pt->green;
		pt_out->blue = pt->blue;
		pt_out->classification = pt->classification;
		pt_out->returnID = pt->returnID;
		pt_out->gpsTime = pt->gpsTime;
		pt_out->user = pt->user;
		pt_out->ignore = pt->ignore;
		pt_out->wavePacketDescIdx = pt->wavePacketDescIdx;
		pt_out->waveformOffset = pt->waveformOffset;
	}
    
    void SPDPointUtils::verticalHeightBinPoints(std::vector<SPDPoint*> *pts, std::vector<SPDPoint*> **bins, boost::uint_fast32_t numBins, float min, float max, bool ignorePtsOverMax, bool ignoreGrd, float minHeightThres)throw(SPDProcessingException)
    {
        try 
        {
            float interval = (max-min)/numBins;
            float diff = 0;
            boost::uint_fast32_t idx = 0;
            
            for(std::vector<SPDPoint*>::iterator iterPts = pts->begin(); iterPts != pts->end(); ++iterPts)
            {
                if(ignoreGrd & ((*iterPts)->classification != SPD_GROUND))
                {
                    if((*iterPts)->height > minHeightThres)
                    {
                        diff = ((*iterPts)->height - min)/interval;
                                                
                        try 
                        {
                            idx = boost::numeric_cast<boost::uint_fast32_t>(diff);
                        }
                        catch(boost::numeric::negative_overflow& e) 
                        {
                            throw SPDProcessingException(e.what());
                        }
                        catch(boost::numeric::positive_overflow& e) 
                        {
                            throw SPDProcessingException(e.what());
                        }
                        catch(boost::numeric::bad_numeric_cast& e) 
                        {
                            throw SPDProcessingException(e.what());
                        }
                        
                        if((idx > ((numBins)-1)) & !ignorePtsOverMax)
                        {
                            std::cout << "Height: = " << (*iterPts)->height << std::endl;
                            std::cout << "Diff = " << diff << std::endl;
                            std::cout << "Index = " << idx << std::endl;
                            std::cout << "Num Bins = " << numBins << std::endl;
                            throw SPDProcessingException("Did not find index within range.");
                        }
                        else if(idx < numBins)
                        {
                            bins[idx]->push_back(*iterPts);
                            //std::cout << "bins[idx]->size() = " << bins[idx]->size() << std::endl;
                        }
                    }
                }
                else if(!ignoreGrd)
                {
                    if((*iterPts)->height > minHeightThres)
                    {
                        diff = ((*iterPts)->height - min)/interval;
                                                
                        try 
                        {
                            idx = boost::numeric_cast<boost::uint_fast32_t>(diff);
                        }
                        catch(boost::numeric::negative_overflow& e) 
                        {
                            throw SPDProcessingException(e.what());
                        }
                        catch(boost::numeric::positive_overflow& e) 
                        {
                            throw SPDProcessingException(e.what());
                        }
                        catch(boost::numeric::bad_numeric_cast& e) 
                        {
                            throw SPDProcessingException(e.what());
                        }
                                                
                        if((idx > ((numBins)-1)) & !ignorePtsOverMax)
                        {
                            std::cout << "Height: = " << (*iterPts)->height << std::endl;
                            std::cout << "Diff = " << diff << std::endl;
                            std::cout << "Index = " << idx << std::endl;
                            std::cout << "Num Bins = " << numBins << std::endl;
                            throw SPDProcessingException("Did not find index within range.");
                        }
                        else if(idx < numBins)
                        {
                            bins[idx]->push_back(*iterPts);
                        }
                    }
                }
            }
        }
        catch (SPDProcessingException &e) 
        {
            throw e;
        }
    }
	
    void SPDPointUtils::verticalElevationBinPoints(std::vector<SPDPoint*> *pts, std::vector<SPDPoint*> **bins, boost::uint_fast32_t numBins, float min, float max)throw(SPDProcessingException)
    {
        try 
        {
            float interval = (max-min)/numBins;
            float diff = 0;
            boost::uint_fast32_t idx = 0;
            
            for(std::vector<SPDPoint*>::iterator iterPts = pts->begin(); iterPts != pts->end(); ++iterPts)
            {
                diff = ((*iterPts)->z - min)/interval;
                
                try 
				{
					idx = boost::numeric_cast<boost::uint_fast32_t>(diff);
				}
				catch(boost::numeric::negative_overflow& e) 
				{
					throw SPDProcessingException(e.what());
				}
				catch(boost::numeric::positive_overflow& e) 
				{
					throw SPDProcessingException(e.what());
				}
				catch(boost::numeric::bad_numeric_cast& e) 
				{
					throw SPDProcessingException(e.what());
				}
				
				if(idx > ((numBins)-1))
				{
					std::cout << "Z: = " << (*iterPts)->z << std::endl;
					std::cout << "Diff = " << diff << std::endl;
					std::cout << "Index = " << idx << std::endl;
					std::cout << "Num Bins = " << numBins << std::endl;
					throw SPDProcessingException("Did not find index within range.");
				}
                
                bins[idx]->push_back(*iterPts);
            }
        }
        catch (SPDProcessingException &e) 
        {
            throw e;
        }
    }
    
	SPDPointUtils::~SPDPointUtils()
	{
		
	}
}

