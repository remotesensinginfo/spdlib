/*
 *  SPDFile.cpp
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

#include "spd/SPDFile.h"

namespace spdlib
{
	
	SPDFile::SPDFile(std::string filepath) : filepath(""), 
	spatialreference(""), 
	indexType(SPD_CARTESIAN_IDX),
	discretePtDefined(SPD_FALSE),
	decomposedPtDefined(SPD_FALSE),
	transWaveformDefined(SPD_FALSE),
    receiveWaveformDefined(SPD_FALSE),
	majorSPDVersion(2),
	minorSPDVersion(1),
    pointVersion(2),
    pulseVersion(2),
	generatingSoftware("SPDLIB"),
	systemIdentifier("UNDEFINED"),
	fileSignature("SPDFILE"),
	yearOfCreation(0),
	monthOfCreation(0),
	dayOfCreation(0),
	hourOfCreation(0),
	minuteOfCreation(0),
	secondOfCreation(0),
	yearOfCapture(0),
	monthOfCapture(0),
	dayOfCapture(0),
	hourOfCapture(0),
	minuteOfCapture(0),
	secondOfCapture(0),
	numPts(0),
	numPulses(0),
	userMetaField(""),
	xMin(0),
	xMax(0),
	yMin(0),
	yMax(0),
	zMin(0),
	zMax(0),
	zenithMin(0),
	zenithMax(0),
	azimuthMin(0),
	azimuthMax(0),
	rangeMin(0),
	rangeMax(0),
    scanlineMin(0),
    scanlineMax(0),
    scanlineIdxMin(0),
    scanlineIdxMax(0),   
	binSize(0),
	numBinsX(0),
	numBinsY(0),
	wavelengths(),
    bandwidths(),
    numOfWavelengths(0),
	pulseRepetitionFreq(0),
	beamDivergence(0),
	sensorHeight(0),
	footprint(0),
	maxScanAngle(0),
	rgbDefined(SPD_FALSE),
	pulseBlockSize(250),
	pointBlockSize(250),
	receivedBlockSize(250),
	transmittedBlockSize(250),
    waveformBitRes(SPD_32_BIT_WAVE),
	temporalBinSpacing(0),
	returnNumsSynGen(SPD_FALSE),
	heightDefined(SPD_FALSE),
	sensorSpeed(0),
	sensorScanRate(0),
	pointDensity(0),
	pulseDensity(0),
	pulseCrossTrackSpacing(0),
	pulseAlongTrackSpacing(0),
	originDefined(SPD_FALSE),
	pulseAngularSpacingAzimuth(0),
	pulseAngularSpacingZenith(0),
	pulseIdxMethod(SPD_FIRST_RETURN),
    sensorApertureSize(0),
    pulseEnergy(0),
    fieldOfView(0)
	{
		this->filepath = filepath;
		
		time_t rawtime = time (NULL);
		struct tm *timeinfo;
		timeinfo = localtime ( &rawtime );
		
		yearOfCreation = 1900 + timeinfo->tm_year;
		monthOfCreation = timeinfo->tm_mon + 1;
		dayOfCreation = timeinfo->tm_mday;
		hourOfCreation = timeinfo->tm_hour;
		minuteOfCreation = timeinfo->tm_min;
		secondOfCreation = timeinfo->tm_sec;
	}
	
	void SPDFile::copyAttributesTo(SPDFile *spdFile)
	{
		spdFile->spatialreference = this->spatialreference;
		spdFile->indexType = this->indexType;
		spdFile->discretePtDefined = this->discretePtDefined;
		spdFile->decomposedPtDefined = this->decomposedPtDefined;
		spdFile->transWaveformDefined = this->transWaveformDefined;
        spdFile->receiveWaveformDefined = this->receiveWaveformDefined;
		spdFile->majorSPDVersion = this->majorSPDVersion;
		spdFile->minorSPDVersion = this->minorSPDVersion;
        spdFile->pointVersion = this->pointVersion;
        spdFile->pulseVersion = this->pulseVersion;        
		spdFile->generatingSoftware = this->generatingSoftware;
		spdFile->systemIdentifier = this->systemIdentifier;
		spdFile->fileSignature = this->fileSignature;
		spdFile->yearOfCapture = this->yearOfCapture;
		spdFile->monthOfCapture =  this->monthOfCapture;
		spdFile->dayOfCapture = this->dayOfCapture;
		spdFile->hourOfCapture = this->hourOfCapture;
		spdFile->minuteOfCapture = this->minuteOfCapture;
		spdFile->secondOfCapture = this->secondOfCapture;
		spdFile->numPts = this->numPts;
		spdFile->numPulses = this->numPulses;
		spdFile->userMetaField = this->userMetaField;
		spdFile->xMin = this->xMin;
		spdFile->xMax = this->xMax;
		spdFile->yMin = this->yMin;
		spdFile->yMax = this->yMax;
		spdFile->zMin = this->zMin;
		spdFile->zMax = this->zMax;
		spdFile->zenithMin = this->zenithMin;
		spdFile->zenithMax = this->zenithMax;
		spdFile->azimuthMin = this->azimuthMin;
		spdFile->azimuthMax = this->azimuthMax;
		spdFile->rangeMin = this->rangeMin;
		spdFile->rangeMax = this->rangeMax;		
    	spdFile->scanlineMin = this->scanlineMin;
    	spdFile->scanlineMax = this->scanlineMax;
    	spdFile->scanlineIdxMin = this->scanlineIdxMin;
    	spdFile->scanlineIdxMax =  this->scanlineIdxMax;        
        spdFile->binSize = this->binSize;
		spdFile->numBinsX = this->numBinsX;
		spdFile->numBinsY = this->numBinsY;
		spdFile->wavelengths = this->wavelengths;
        spdFile->bandwidths = this->bandwidths;
        spdFile->numOfWavelengths = this->numOfWavelengths;
		spdFile->pulseRepetitionFreq = this->pulseRepetitionFreq;
		spdFile->beamDivergence = this->beamDivergence;
		spdFile->sensorHeight = this->sensorHeight;
		spdFile->footprint = this->footprint;
		spdFile->maxScanAngle = this->maxScanAngle;
		spdFile->rgbDefined = this->rgbDefined;
		spdFile->pulseBlockSize = this->pulseBlockSize;
		spdFile->pointBlockSize = this->pointBlockSize;
		spdFile->receivedBlockSize = this->receivedBlockSize;
		spdFile->transmittedBlockSize = this->transmittedBlockSize;
        spdFile->waveformBitRes = this->waveformBitRes;
		spdFile->temporalBinSpacing = this->temporalBinSpacing;
		spdFile->returnNumsSynGen = this->returnNumsSynGen;
		spdFile->heightDefined = this->heightDefined;
		spdFile->sensorSpeed = this->sensorSpeed;
		spdFile->sensorScanRate = this->sensorScanRate;
		spdFile->pointDensity = this->pointDensity;
		spdFile->pulseDensity = this->pulseDensity;
		spdFile->pulseCrossTrackSpacing = this->pulseCrossTrackSpacing;
		spdFile->pulseAlongTrackSpacing = this->pulseAlongTrackSpacing;
		spdFile->originDefined = this->originDefined;
		spdFile->pulseAngularSpacingAzimuth = this->pulseAngularSpacingAzimuth;
		spdFile->pulseAngularSpacingZenith = this->pulseAngularSpacingZenith;
		spdFile->pulseIdxMethod = this->pulseIdxMethod;
        spdFile->sensorApertureSize = this->sensorApertureSize;
        spdFile->pulseEnergy = this->pulseEnergy;
        spdFile->fieldOfView = this->fieldOfView;
	}
	
	void SPDFile::copyAttributesFrom(SPDFile *spdFile)
	{
		this->spatialreference = spdFile->spatialreference;
		this->indexType = spdFile->indexType;
		this->discretePtDefined = spdFile->discretePtDefined;
		this->decomposedPtDefined = spdFile->decomposedPtDefined;
		this->transWaveformDefined = spdFile->transWaveformDefined;
        this->receiveWaveformDefined = spdFile->receiveWaveformDefined;
		this->majorSPDVersion = spdFile->majorSPDVersion;
		this->minorSPDVersion = spdFile->minorSPDVersion;
        this->pointVersion = spdFile->pointVersion;
        this->pulseVersion = spdFile->pulseVersion;
		this->generatingSoftware = spdFile->generatingSoftware;
		this->systemIdentifier = spdFile->systemIdentifier;
		this->fileSignature = spdFile->fileSignature;
		this->yearOfCapture = spdFile->yearOfCapture;
		this->monthOfCapture =  spdFile->monthOfCapture;
		this->dayOfCapture = spdFile->dayOfCapture;
		this->hourOfCapture = spdFile->hourOfCapture;
		this->minuteOfCapture = spdFile->minuteOfCapture;
		this->secondOfCapture = spdFile->secondOfCapture;
		this->numPts = spdFile->numPts;
		this->numPulses = spdFile->numPulses;
		this->userMetaField = spdFile->userMetaField;
		this->xMin = spdFile->xMin;
		this->xMax = spdFile->xMax;
		this->yMin = spdFile->yMin;
		this->yMax = spdFile->yMax;
		this->zMin = spdFile->zMin;
		this->zMax = spdFile->zMax;
		this->zenithMin = spdFile->zenithMin;
		this->zenithMax = spdFile->zenithMax;
		this->azimuthMin = spdFile->azimuthMin;
		this->azimuthMax = spdFile->azimuthMax;
		this->rangeMin = spdFile->rangeMin;
		this->rangeMax = spdFile->rangeMax;
    	this->scanlineMin = spdFile->scanlineMin;
    	this->scanlineMax = spdFile->scanlineMax;
    	this->scanlineIdxMin = spdFile->scanlineIdxMin;
    	this->scanlineIdxMax =  spdFile->scanlineIdxMax;        
		this->binSize = spdFile->binSize;
		this->numBinsX = spdFile->numBinsX;
		this->numBinsY = spdFile->numBinsY;
		this->wavelengths = spdFile->wavelengths;
        this->bandwidths = spdFile->bandwidths;
        this->numOfWavelengths = spdFile->numOfWavelengths;
		this->pulseRepetitionFreq = spdFile->pulseRepetitionFreq;
		this->beamDivergence = spdFile->beamDivergence;
		this->sensorHeight = spdFile->sensorHeight;
		this->footprint = spdFile->footprint;
		this->maxScanAngle = spdFile->maxScanAngle;
		this->rgbDefined = spdFile->rgbDefined;
		this->pulseBlockSize = spdFile->pulseBlockSize;
		this->pointBlockSize = spdFile->pointBlockSize;
		this->receivedBlockSize = spdFile->receivedBlockSize;
		this->transmittedBlockSize = spdFile->transmittedBlockSize;
        this->waveformBitRes = spdFile->waveformBitRes;
		this->temporalBinSpacing = spdFile->temporalBinSpacing;
		this->returnNumsSynGen = spdFile->returnNumsSynGen;
		this->heightDefined = spdFile->heightDefined;
		this->sensorSpeed = spdFile->sensorSpeed;
		this->sensorScanRate = spdFile->sensorScanRate;
		this->pointDensity = spdFile->pointDensity;
		this->pulseDensity = spdFile->pulseDensity;
		this->pulseCrossTrackSpacing = spdFile->pulseCrossTrackSpacing;
		this->pulseAlongTrackSpacing = spdFile->pulseAlongTrackSpacing;
		this->originDefined = spdFile->originDefined;
		this->pulseAngularSpacingAzimuth = spdFile->pulseAngularSpacingAzimuth;
		this->pulseAngularSpacingZenith = spdFile->pulseAngularSpacingZenith;
		this->pulseIdxMethod = spdFile->pulseIdxMethod;
        this->sensorApertureSize = spdFile->sensorApertureSize;
        this->pulseEnergy = spdFile->pulseEnergy;
        this->fieldOfView = spdFile->fieldOfView;
	}
    
    void SPDFile::copyAttributesFromTemplate(SPDFile *spdFile)
	{
		this->spatialreference = spdFile->spatialreference;
		this->discretePtDefined = spdFile->discretePtDefined;
		this->decomposedPtDefined = spdFile->decomposedPtDefined;
		this->transWaveformDefined = spdFile->transWaveformDefined;
        this->receiveWaveformDefined = spdFile->receiveWaveformDefined;
        this->yearOfCapture = spdFile->yearOfCapture;
		this->monthOfCapture =  spdFile->monthOfCapture;
		this->dayOfCapture = spdFile->dayOfCapture;
		this->hourOfCapture = spdFile->hourOfCapture;
		this->minuteOfCapture = spdFile->minuteOfCapture;
		this->secondOfCapture = spdFile->secondOfCapture;
		this->numPts = 0;
		this->numPulses = 0;
		this->userMetaField = spdFile->userMetaField;
		this->wavelengths = spdFile->wavelengths;
        this->bandwidths = spdFile->bandwidths;
        this->numOfWavelengths = spdFile->numOfWavelengths;
		this->pulseRepetitionFreq = spdFile->pulseRepetitionFreq;
		this->beamDivergence = spdFile->beamDivergence;
		this->sensorHeight = spdFile->sensorHeight;
		this->footprint = spdFile->footprint;
		this->maxScanAngle = spdFile->maxScanAngle;
		this->rgbDefined = spdFile->rgbDefined;
		this->pulseBlockSize = spdFile->pulseBlockSize;
		this->pointBlockSize = spdFile->pointBlockSize;
		this->receivedBlockSize = spdFile->receivedBlockSize;
		this->transmittedBlockSize = spdFile->transmittedBlockSize;
        this->waveformBitRes = spdFile->waveformBitRes;
		this->temporalBinSpacing = spdFile->temporalBinSpacing;
		this->returnNumsSynGen = spdFile->returnNumsSynGen;
		this->heightDefined = spdFile->heightDefined;
		this->sensorSpeed = spdFile->sensorSpeed;
		this->sensorScanRate = spdFile->sensorScanRate;
		this->pointDensity = spdFile->pointDensity;
		this->pulseDensity = spdFile->pulseDensity;
		this->pulseCrossTrackSpacing = spdFile->pulseCrossTrackSpacing;
		this->pulseAlongTrackSpacing = spdFile->pulseAlongTrackSpacing;
		this->originDefined = spdFile->originDefined;
		this->pulseAngularSpacingAzimuth = spdFile->pulseAngularSpacingAzimuth;
		this->pulseAngularSpacingZenith = spdFile->pulseAngularSpacingZenith;
		this->pulseIdxMethod = spdFile->pulseIdxMethod;
        this->sensorApertureSize = spdFile->sensorApertureSize;
        this->pulseEnergy = spdFile->pulseEnergy;
        this->fieldOfView = spdFile->fieldOfView;
	}
	
	bool SPDFile::checkCompatibility(SPDFile *spdFile)
	{
        if(this->majorSPDVersion != spdFile->majorSPDVersion)
        {
            return false;
        }
        
        if(this->minorSPDVersion != spdFile->minorSPDVersion)
        {
            return false;
        }
        
        if(this->pointVersion != spdFile->pointVersion)
        {
            return false;
        }
        
        if(this->pulseVersion != spdFile->pulseVersion)
        {
            return false;
        }
        
		if(this->spatialreference != spdFile->spatialreference)
		{
			return false;
		}
		
		if(this->indexType != spdFile->indexType)
		{
			return false;
		}
		
		if(this->discretePtDefined != spdFile->discretePtDefined)
		{
			return false;
		}
		
		if(this->decomposedPtDefined != spdFile->decomposedPtDefined)
		{
			return false;
		}
		
		if(this->transWaveformDefined != spdFile->transWaveformDefined)
		{
			return false;
		}
        
        if(this->receiveWaveformDefined != spdFile->receiveWaveformDefined)
		{
			return false;
		}
        
        if(this->numOfWavelengths != spdFile->numOfWavelengths)
		{
			return false;
		}
		
        if(this->wavelengths.size() != spdFile->wavelengths.size())
		{
			return false;
		}
        
        for(boost::int_fast16_t i = 0; i < this->wavelengths.size(); ++i)
        {
            if(!compare_double(this->wavelengths[i], spdFile->wavelengths[i]))
            {
                return false;
            }
        }
        
        if(this->bandwidths.size() != spdFile->bandwidths.size())
		{
			return false;
		}
        
        for(boost::int_fast16_t i = 0; i < this->bandwidths.size(); ++i)
        {
            if(!compare_double(this->bandwidths[i], spdFile->bandwidths[i]))
            {
                return false;
            }
        }
		
		if(!compare_double(this->pulseRepetitionFreq, spdFile->pulseRepetitionFreq))
		{
			return false;
		}
		
		if(!compare_double(this->beamDivergence, spdFile->beamDivergence))
		{
			return false;
		}
		
		if(!compare_double(this->sensorHeight, spdFile->sensorHeight))
		{
			return false;
		}
		
		if(!compare_double(this->footprint, spdFile->footprint))
		{
			return false;
		}
		
		if(!compare_double(this->maxScanAngle, spdFile->maxScanAngle))
		{
			return false;
		}
		
		if(this->rgbDefined != spdFile->rgbDefined)
		{
			return false;
		}
		
		if(!compare_double(this->temporalBinSpacing, spdFile->temporalBinSpacing))
		{
			return false;
		}
		
		if(this->returnNumsSynGen != spdFile->returnNumsSynGen)
		{
			return false;
		}
		
		if(this->heightDefined != spdFile->heightDefined)
		{
			return false;
		}
		
		if(!compare_double(this->sensorSpeed, spdFile->sensorSpeed))
		{
			return false;
		}
		
		if(!compare_double(this->sensorScanRate, spdFile->sensorScanRate))
		{
			return false;
		}
		
		if(!compare_double(this->pointDensity, spdFile->pointDensity))
		{
			return false;
		}
		
		if(!compare_double(this->pulseDensity, spdFile->pulseDensity))
		{
			return false;
		}
		
		if(!compare_double(this->pulseCrossTrackSpacing, spdFile->pulseCrossTrackSpacing))
		{
			return false;
		}
		
		if(!compare_double(this->pulseAlongTrackSpacing, spdFile->pulseAlongTrackSpacing))
		{
			return false;
		}
		
		if(this->originDefined != spdFile->originDefined)
		{
			return false;
		}
		
		if(!compare_double(this->pulseAngularSpacingAzimuth, spdFile->pulseAngularSpacingAzimuth))
		{
			return false;
		}
		
		if(!compare_double(this->pulseAngularSpacingZenith, spdFile->pulseAngularSpacingZenith))
		{
			return false;
		}
		
		if(this->pulseIdxMethod != spdFile->pulseIdxMethod)
		{
			return false;
		}
		
		if(!compare_double(this->binSize, spdFile->binSize))
		{
			return false;
		}
		
		if(this->numBinsX != spdFile->numBinsX)
		{
			return false;
		}
		
		if(this->numBinsY != spdFile->numBinsY)
		{
			return false;
		}
		
		return true;
	}
	
	bool SPDFile::checkCompatibilityExpandExtent(SPDFile *spdFile)
	{
        if(this->majorSPDVersion != spdFile->majorSPDVersion)
        {
            std::cout << "SPD Major version is different\n";
            return false;
        }
        
        if(this->minorSPDVersion != spdFile->minorSPDVersion)
        {
            std::cout << "SPD Minor version is different\n";
            return false;
        }
        
        if(this->pointVersion != spdFile->pointVersion)
        {
            std::cout << "Point version is different\n";
            return false;
        }
        
        if(this->pulseVersion != spdFile->pulseVersion)
        {
            std::cout << "Pulse version is different\n";
            return false;
        }
        
		if(this->spatialreference != spdFile->spatialreference)
		{
            std::cout << "Spatial Reference is different\n";
			return false;
		}
		
		if(this->indexType != spdFile->indexType)
		{
            std::cout << "Index is different\n";
			return false;
		}
		
		if(this->discretePtDefined != spdFile->discretePtDefined)
		{
            std::cout << "definiation of discrete returns is different\n";
			return false;
		}
		
		if(this->decomposedPtDefined != spdFile->decomposedPtDefined)
		{
            std::cout << "definiation of decomposed returns is different\n";
			return false;
		}
		
		if(this->transWaveformDefined != spdFile->transWaveformDefined)
		{
            std::cout << "definiation of transmitted waveform is different\n";
			return false;
		}
        
        if(this->receiveWaveformDefined != spdFile->receiveWaveformDefined)
		{
            std::cout << "definiation of received waveform is different\n";
			return false;
		}
        
        if(this->numOfWavelengths != spdFile->numOfWavelengths)
		{
            std::cout << "Number of wavelengths is different\n";
			return false;
		}
		
        if(this->wavelengths.size() != spdFile->wavelengths.size())
		{
            std::cout << "Number of wavelengths in lists different\n";
			return false;
		}
        
        for(boost::int_fast16_t i = 0; i < this->wavelengths.size(); ++i)
        {
            if(!compare_double(this->wavelengths[i], spdFile->wavelengths[i]))
            {
                std::cout << "Wavelengths are different\n";
                return false;
            }
        }
        
        if(this->bandwidths.size() != spdFile->bandwidths.size())
		{
            std::cout << "Number of bandwidths in lists different\n";
			return false;
		}
        
        for(boost::int_fast16_t i = 0; i < this->bandwidths.size(); ++i)
        {
            if(!compare_double(this->bandwidths[i], spdFile->bandwidths[i]))
            {
                std::cout << "Bandwidths are different\n";
                return false;
            }
        }
		
		if(!compare_double(this->pulseRepetitionFreq, spdFile->pulseRepetitionFreq))
		{
            std::cout << "Pulse Repetition Freq is different\n";
			return false;
		}
		
		if(!compare_double(this->beamDivergence, spdFile->beamDivergence))
		{
            std::cout << "Beam Divergence is different\n";
			return false;
		}
		
		if(!compare_double(this->sensorHeight, spdFile->sensorHeight))
		{
            std::cout << "Sensor Height is different\n";
			return false;
		}
		
		if(!compare_double(this->footprint, spdFile->footprint))
		{
            std::cout << "Footprint is different\n";
			return false;
		}
		
		if(!compare_double(this->maxScanAngle, spdFile->maxScanAngle))
		{
            std::cout << "Max angle is different\n";
			return false;
		}
		
		if(this->rgbDefined != spdFile->rgbDefined)
		{
            std::cout << "RGB defined is different\n";
			return false;
		}
		
		if(!compare_double(this->temporalBinSpacing, spdFile->temporalBinSpacing))
		{
            std::cout << "Temporal bin spacing is different\n";
			return false;
		}
		
		if(this->returnNumsSynGen != spdFile->returnNumsSynGen)
		{
            std::cout << "Returns number synthetically generated is different\n";
			return false;
		}
		
		if(this->heightDefined != spdFile->heightDefined)
		{
            std::cout << "Height defined\n";
			return false;
		}
		
		if(!compare_double(this->sensorSpeed, spdFile->sensorSpeed))
		{
            std::cout << "Sensor Speed is different\n";
			return false;
		}
		
		if(!compare_double(this->sensorScanRate, spdFile->sensorScanRate))
		{
            std::cout << "Sensor Scan Rate is different\n";
			return false;
		}
		
		if(!compare_double(this->pointDensity, spdFile->pointDensity))
		{
            std::cout << "Point Density is different\n";
			return false;
		}
		
		if(!compare_double(this->pulseDensity, spdFile->pulseDensity))
		{
            std::cout << "Pulse Density is different\n";
			return false;
		}
		
		if(!compare_double(this->pulseCrossTrackSpacing, spdFile->pulseCrossTrackSpacing))
		{
            std::cout << "Cross Track Spacing is different\n";
			return false;
		}
		
		if(!compare_double(this->pulseAlongTrackSpacing, spdFile->pulseAlongTrackSpacing))
		{
            std::cout << "Pulse along track is different\n";
			return false;
		}
		
		if(this->originDefined != spdFile->originDefined)
		{
            std::cout << "Origin defined is different\n";
			return false;
		}
		
		if(!compare_double(this->pulseAngularSpacingAzimuth, spdFile->pulseAngularSpacingAzimuth))
		{
            std::cout << "Pulse Angular Spacing Azimuth is different\n";
			return false;
		}
		
		if(!compare_double(this->pulseAngularSpacingZenith, spdFile->pulseAngularSpacingZenith))
		{
            std::cout << "Pulse Angular Spacing Zenith is different\n";
			return false;
		}
		
		if(this->pulseIdxMethod != spdFile->pulseIdxMethod)
		{
            std::cout << "Pulse Index method is different\n";
			return false;
		}
		
		if(!compare_double(this->binSize, spdFile->binSize))
		{
            std::cout << "Pulse bin size is different\n";
			return false;
		}
		
		if(spdFile->xMin < this->xMin)
		{
			this->xMin = spdFile->xMin;
		}
		
		if(spdFile->xMax > this->xMax)
		{
			this->xMax = spdFile->xMax;
		}
		
		if(spdFile->yMin < this->yMin)
		{
			this->yMin = spdFile->yMin;
		}
		
		if(spdFile->yMax > this->yMax)
		{
			this->yMax = spdFile->yMax;
		}
		
		if(spdFile->zMin < this->zMin)
		{
			this->zMin = spdFile->zMin;
		}
		
		if(spdFile->zMax > this->zMax)
		{
			this->zMax = spdFile->zMax;
		}
		
		if(spdFile->zenithMin < this->zenithMin)
		{
			this->zenithMin = spdFile->zenithMin;
		}
		
		if(spdFile->zenithMax > this->zenithMax)
		{
			this->zenithMax = spdFile->zenithMax;
		}
		
		if(spdFile->azimuthMin < this->azimuthMin)
		{
			this->azimuthMin = spdFile->azimuthMin;
		}
		
		if(spdFile->azimuthMax > this->azimuthMax)
		{
			this->azimuthMax = spdFile->azimuthMax;
		}
		
		if(spdFile->rangeMin < this->rangeMin)
		{
			this->rangeMin = spdFile->rangeMin;
		}
		
		if(spdFile->rangeMax > this->rangeMax)
		{
			this->rangeMax = spdFile->rangeMax;
		}
		
		if(spdFile->scanlineMin < this->scanlineMin)
		{
			this->scanlineMin = spdFile->scanlineMin;
		}
		
		if(spdFile->scanlineMax > this->scanlineMax)
		{
			this->scanlineMax = spdFile->scanlineMax;
		}
		
		if(spdFile->scanlineIdxMin < this->scanlineIdxMin)
		{
			this->scanlineIdxMin = spdFile->scanlineIdxMin;
		}
		
		if(spdFile->scanlineIdxMax > this->scanlineIdxMax)
		{
			this->scanlineIdxMax = spdFile->scanlineIdxMax;
		}
		
		return true;
	}
    
    bool SPDFile::checkCompatibilityGeneralCheckExpandExtent(SPDFile *spdFile)
	{
        if(this->majorSPDVersion != spdFile->majorSPDVersion)
        {
            std::cout << "SPD Major version is different\n";
            return false;
        }
        
        if(this->minorSPDVersion != spdFile->minorSPDVersion)
        {
            std::cout << "SPD Minor version is different\n";
            return false;
        }
        
        if(this->pointVersion != spdFile->pointVersion)
        {
            std::cout << "Point version is different\n";
            return false;
        }
        
        if(this->pulseVersion != spdFile->pulseVersion)
        {
            std::cout << "Pulse version is different\n";
            return false;
        }
        
		if(this->spatialreference != spdFile->spatialreference)
		{
            std::cout << "Spatial Reference is different\n";
			return false;
		}
		
		if(this->indexType != spdFile->indexType)
		{
            std::cout << "Index is different\n";
			return false;
		}
		
		if(this->discretePtDefined != spdFile->discretePtDefined)
		{
            std::cout << "definiation of discrete returns is different\n";
			return false;
		}
		
		if(this->decomposedPtDefined != spdFile->decomposedPtDefined)
		{
            std::cout << "definiation of decomposed returns is different\n";
			return false;
		}
		
		if(this->transWaveformDefined != spdFile->transWaveformDefined)
		{
            std::cout << "definiation of transmitted waveform is different\n";
			return false;
		}
        
        if(this->receiveWaveformDefined != spdFile->receiveWaveformDefined)
		{
            std::cout << "definiation of received waveform is different\n";
			return false;
		}
		
		if(this->numOfWavelengths != spdFile->numOfWavelengths)
		{
            std::cout << "Number of wavelengths is different\n";
			return false;
		}
		
        if(this->wavelengths.size() != spdFile->wavelengths.size())
		{
            std::cout << "Number of wavelengths in lists different\n";
			return false;
		}
        
        for(boost::int_fast16_t i = 0; i < this->wavelengths.size(); ++i)
        {
            if(!compare_double(this->wavelengths[i], spdFile->wavelengths[i]))
            {
                std::cout << "Wavelengths are different\n";
                return false;
            }
        }
        
        if(this->bandwidths.size() != spdFile->bandwidths.size())
		{
            std::cout << "Number of bandwidths in lists different\n";
			return false;
		}
        
        for(boost::int_fast16_t i = 0; i < this->bandwidths.size(); ++i)
        {
            if(!compare_double(this->bandwidths[i], spdFile->bandwidths[i]))
            {
                std::cout << "Bandwidths are different\n";
                return false;
            }
        }
		
		if(!compare_double(this->pulseRepetitionFreq, spdFile->pulseRepetitionFreq))
		{
            std::cout << "Pulse Repetition Freq is different\n";
			return false;
		}
		
		if(!compare_double(this->beamDivergence, spdFile->beamDivergence))
		{
            std::cout << "Beam Divergence is different\n";
			return false;
		}
		
		if(!compare_double(this->sensorHeight, spdFile->sensorHeight))
		{
            std::cout << "Sensor Height is different\n";
			return false;
		}
		
		if(!compare_double(this->footprint, spdFile->footprint))
		{
            std::cout << "Footprint is different\n";
			return false;
		}
		
		if(!compare_double(this->maxScanAngle, spdFile->maxScanAngle))
		{
            std::cout << "Max angle is different\n";
			return false;
		}
		
		if(this->rgbDefined != spdFile->rgbDefined)
		{
            std::cout << "RGB defined is different\n";
			return false;
		}
		
		if(!compare_double(this->temporalBinSpacing, spdFile->temporalBinSpacing))
		{
            std::cout << "Temporal bin spacing is different\n";
			return false;
		}
		
		if(this->returnNumsSynGen != spdFile->returnNumsSynGen)
		{
            std::cout << "Returns number synthetically generated is different\n";
			return false;
		}
		
		if(this->heightDefined != spdFile->heightDefined)
		{
            std::cout << "Height defined\n";
			return false;
		}
		
		if(!compare_double(this->sensorSpeed, spdFile->sensorSpeed))
		{
            std::cout << "Sensor Speed is different\n";
			return false;
		}
		
		if(!compare_double(this->sensorScanRate, spdFile->sensorScanRate))
		{
            std::cout << "Sensor Scan Rate is different\n";
			return false;
		}
		
		if(!compare_double(this->pointDensity, spdFile->pointDensity))
		{
            std::cout << "Point Density is different\n";
			return false;
		}
		
		if(!compare_double(this->pulseDensity, spdFile->pulseDensity))
		{
            std::cout << "Pulse Density is different\n";
			return false;
		}
		
		if(!compare_double(this->pulseCrossTrackSpacing, spdFile->pulseCrossTrackSpacing))
		{
            std::cout << "Cross Track Spacing is different\n";
			return false;
		}
		
		if(!compare_double(this->pulseAlongTrackSpacing, spdFile->pulseAlongTrackSpacing))
		{
            std::cout << "Pulse along track is different\n";
			return false;
		}
		
		if(this->originDefined != spdFile->originDefined)
		{
            std::cout << "Origin defined is different\n";
			return false;
		}
		
		if(!compare_double(this->pulseAngularSpacingAzimuth, spdFile->pulseAngularSpacingAzimuth))
		{
            std::cout << "Pulse Angular Spacing Azimuth is different\n";
			return false;
		}
		
		if(!compare_double(this->pulseAngularSpacingZenith, spdFile->pulseAngularSpacingZenith))
		{
            std::cout << "Pulse Angular Spacing Zenith is different\n";
			return false;
		}
		
		if(this->pulseIdxMethod != spdFile->pulseIdxMethod)
		{
            std::cout << "Pulse Index method is different\n";
			return false;
		}
		
		if(spdFile->xMin < this->xMin)
		{
			this->xMin = spdFile->xMin;
		}
		
		if(spdFile->xMax > this->xMax)
		{
			this->xMax = spdFile->xMax;
		}
		
		if(spdFile->yMin < this->yMin)
		{
			this->yMin = spdFile->yMin;
		}
		
		if(spdFile->yMax > this->yMax)
		{
			this->yMax = spdFile->yMax;
		}
		
		if(spdFile->zMin < this->zMin)
		{
			this->zMin = spdFile->zMin;
		}
		
		if(spdFile->zMax > this->zMax)
		{
			this->zMax = spdFile->zMax;
		}
		
		if(spdFile->zenithMin < this->zenithMin)
		{
			this->zenithMin = spdFile->zenithMin;
		}
		
		if(spdFile->zenithMax > this->zenithMax)
		{
			this->zenithMax = spdFile->zenithMax;
		}
		
		if(spdFile->azimuthMin < this->azimuthMin)
		{
			this->azimuthMin = spdFile->azimuthMin;
		}
		
		if(spdFile->azimuthMax > this->azimuthMax)
		{
			this->azimuthMax = spdFile->azimuthMax;
		}
		
		if(spdFile->rangeMin < this->rangeMin)
		{
			this->rangeMin = spdFile->rangeMin;
		}
		
		if(spdFile->rangeMax > this->rangeMax)
		{
			this->rangeMax = spdFile->rangeMax;
		}
		
		if(spdFile->scanlineMin < this->scanlineMin)
		{
			this->scanlineMin = spdFile->scanlineMin;
		}
		
		if(spdFile->scanlineMax > this->scanlineMax)
		{
			this->scanlineMax = spdFile->scanlineMax;
		}
		
		if(spdFile->scanlineIdxMin < this->scanlineIdxMin)
		{
			this->scanlineIdxMin = spdFile->scanlineIdxMin;
		}
		
		if(spdFile->scanlineIdxMax > this->scanlineIdxMax)
		{
			this->scanlineIdxMax = spdFile->scanlineIdxMax;
		}		
		
		return true;
	}
    
    void SPDFile::expandExtent(SPDFile *spdFile)
    {
        if(spdFile->xMin < this->xMin)
		{
			this->xMin = spdFile->xMin;
		}
		
		if(spdFile->xMax > this->xMax)
		{
			this->xMax = spdFile->xMax;
		}
		
		if(spdFile->yMin < this->yMin)
		{
			this->yMin = spdFile->yMin;
		}
		
		if(spdFile->yMax > this->yMax)
		{
			this->yMax = spdFile->yMax;
		}
		
		if(spdFile->zMin < this->zMin)
		{
			this->zMin = spdFile->zMin;
		}
		
		if(spdFile->zMax > this->zMax)
		{
			this->zMax = spdFile->zMax;
		}
		
		if(spdFile->zenithMin < this->zenithMin)
		{
			this->zenithMin = spdFile->zenithMin;
		}
		
		if(spdFile->zenithMax > this->zenithMax)
		{
			this->zenithMax = spdFile->zenithMax;
		}
		
		if(spdFile->azimuthMin < this->azimuthMin)
		{
			this->azimuthMin = spdFile->azimuthMin;
		}
		
		if(spdFile->azimuthMax > this->azimuthMax)
		{
			this->azimuthMax = spdFile->azimuthMax;
		}
		
		if(spdFile->rangeMin < this->rangeMin)
		{
			this->rangeMin = spdFile->rangeMin;
		}
		
		if(spdFile->rangeMax > this->rangeMax)
		{
			this->rangeMax = spdFile->rangeMax;
		}
		
		if(spdFile->scanlineMin < this->scanlineMin)
		{
			this->scanlineMin = spdFile->scanlineMin;
		}
		
		if(spdFile->scanlineMax > this->scanlineMax)
		{
			this->scanlineMax = spdFile->scanlineMax;
		}
		
		if(spdFile->scanlineIdxMin < this->scanlineIdxMin)
		{
			this->scanlineIdxMin = spdFile->scanlineIdxMin;
		}
		
		if(spdFile->scanlineIdxMax > this->scanlineIdxMax)
		{
			this->scanlineIdxMax = spdFile->scanlineIdxMax;
		}
		
    }
	
	std::ostream& operator<<(std::ostream& stream, SPDFile &obj)
	{
		stream << "File Path: " << obj.getFilePath() << std::endl;
        stream << "File Type: ";
        if(obj.fileType == SPD_SEQ_TYPE)
        {
            stream << "Sequencial index\n";
        }
        else if(obj.fileType == SPD_SEQ_TYPE)
        {
            stream << "Non-Sequencial index\n";
        }
        else if(obj.fileType == SPD_SEQ_TYPE)
        {
            stream << "No index\n";
        }
        else
        {
            stream << std::endl;
        }
		stream << "File Format Version: " <<  obj.majorSPDVersion << "." << obj.minorSPDVersion << std::endl;
        stream << "Point Version: " << obj.pointVersion << std::endl;
        stream << "Pulse Version: " << obj.pulseVersion << std::endl;
		stream << "Generating Software: " <<  obj.generatingSoftware << std::endl;
		stream << "File Signature: " << obj.fileSignature << std::endl;
        stream << "Spatial Reference: " <<  obj.spatialreference << std::endl;
		stream << "Creation Time (YYYY:MM:DD HH:MM:SS): " << obj.yearOfCreation << ":" << obj.monthOfCreation << ":" << obj.dayOfCreation << ":" << obj.hourOfCreation << ":"<< obj.minuteOfCreation << ":"<< obj.secondOfCreation << std::endl;
		stream << "Capture Time (YYYY:MM:DD HH:MM:SS): " << obj.yearOfCapture << ":" << obj.monthOfCapture << ":" << obj.dayOfCapture << ":" << obj.hourOfCapture << ":" << obj.minuteOfCapture << ":" << obj.secondOfCapture << std::endl;
        stream << "Index Type: ";
        if(obj.indexType == SPD_NO_IDX)
        {
            stream << "No Index\n";
        }
        else if(obj.indexType == SPD_CARTESIAN_IDX)
        {
            stream << "Cartesian\n";
        }
        else if(obj.indexType == SPD_SPHERICAL_IDX)
        {
            stream << "Spherical\n";
        }
        else if(obj.indexType == SPD_CYLINDRICAL_IDX)
        {
            stream << "Cylindrical\n";
        }
        else if(obj.indexType == SPD_POLAR_IDX)
        {
            stream << "Polar\n";
        }
        else if(obj.indexType == SPD_SCAN_IDX)
        {
            stream << "Scan\n";
        }
        else 
        {
            stream << std::endl;
        }
        stream << "The file contains: \n";
        if(obj.discretePtDefined == SPD_TRUE)
        {
            stream << "\t Contains discrete returns\n";
        }
        if(obj.decomposedPtDefined == SPD_TRUE)
        {
            stream << "\t Contains decomposed returns\n";
        }
        if(obj.transWaveformDefined == SPD_TRUE)
        {
            stream << "\t Contains transmitted waveforms\n";
        }
        if(obj.receiveWaveformDefined == SPD_TRUE)
        {
            stream << "\t Contains received waveforms\n";
        }
        stream << "Number of Points = " << obj.numPts << std::endl;
		stream << "Number of Pulses = " << obj.numPulses << std::endl;
		stream << "BBOX [xmin, ymin, xmax, ymax]: [" << obj.xMin << "," << obj.yMin << "," << obj.xMax << "," << obj.yMax << "]\n";
		stream << "BBOX [Azimuth Min, Zenith Min, Azimuth Max, Zenith Max]: [" << obj.azimuthMin << "," << obj.zenithMin << "," << obj.azimuthMax << "," << obj.zenithMax << "]\n";
		stream << "Z : [" << obj.zMin << "," << obj.zMax << "]\n";
        stream << "Range: [" << obj.rangeMin << "," << obj.rangeMax << "]\n";
        stream << "Scanline: [" << obj.scanlineMin << "," << obj.scanlineMax << "]\n";
        stream << "Scanline Idx: [" << obj.scanlineIdxMin << "," << obj.scanlineIdxMax << "]\n";
		stream << "Gridding [xSize,ySize] Bin Size: [" << obj.numBinsX << "," << obj.numBinsY << "] " << obj.binSize << std::endl;
        if(obj.wavelengths.size() > 0)
        {
            stream << "Wavelengths:\n";
            for(std::vector<float>::iterator iterWavel = obj.wavelengths.begin(); iterWavel != obj.wavelengths.end(); ++iterWavel)
            {
                stream << "\t" << *iterWavel << std::endl;
            }
        }
        else
        {
            stream << "Wavelengths: \n\tUnknown\n";
        }
        if(obj.bandwidths.size() > 0)
        {
            stream << "Bandwidths:\n";
            for(std::vector<float>::iterator iterBand = obj.bandwidths.begin(); iterBand != obj.bandwidths.end(); ++iterBand)
            {
                stream << "\t" << *iterBand << std::endl;
            }
        }
        else
        {
            stream << "Bandwidths: \n\tUnknown\n";
        }
        stream << "Pulse Repetition Freq: " << obj.pulseRepetitionFreq << std::endl;
        stream << "Beam Divergance: " << obj.beamDivergence << std::endl;
        stream << "Sensor Height: " << obj.sensorHeight << std::endl;
        stream << "Footprint: " << obj.footprint << std::endl;
        stream << "Max. Scan Angle: " << obj.maxScanAngle << std::endl;
        stream << "Waveform Bin Resolution: " << obj.waveformBitRes << std::endl;
        stream << "Temporal Waveform Bin Spacing: " << obj.temporalBinSpacing << std::endl;
        stream << "Sensor Speed: " << obj.sensorSpeed << std::endl;
        stream << "Sensor Scanrate: " << obj.sensorScanRate << std::endl;
        stream << "Pulse Density: " << obj.pulseDensity << std::endl;
        stream << "Point Density: " << obj.pointDensity << std::endl;
        stream << "Pulse cross track spacing: " << obj.pulseCrossTrackSpacing << std::endl;
        stream << "Pulse along track spacing: " << obj.pulseAlongTrackSpacing << std::endl;
        stream << "Pulse angular spacing azimuth: " << obj.pulseAngularSpacingAzimuth << std::endl;
        stream << "Pulse angular spacing zenith: " << obj.pulseAngularSpacingZenith << std::endl;
        stream << "Sensor Aperture Size: " << obj.sensorApertureSize << std::endl;
        stream << "Pulse Energy: " << obj.pulseEnergy << std::endl;
        stream << "Field of view: " << obj.fieldOfView << std::endl;
		stream << "User Meta Data: " << obj.userMetaField << std::endl;
		return stream;
	}
	
	std::ostream& operator<<(std::ostream& stream, SPDFile *obj)
	{
		stream << "File Path: " << obj->getFilePath() << std::endl;
        stream << "File Type: ";
        if(obj->fileType == SPD_SEQ_TYPE)
        {
            stream << "Sequencial index\n";
        }
        else if(obj->fileType == SPD_SEQ_TYPE)
        {
            stream << "Non-Sequencial index\n";
        }
        else if(obj->fileType == SPD_SEQ_TYPE)
        {
            stream << "No index\n";
        }
        else
        {
            stream << std::endl;
        }
		stream << "File Format Version: " <<  obj->majorSPDVersion << "." << obj->minorSPDVersion << std::endl;
        stream << "Point Version: " << obj->pointVersion << std::endl;
        stream << "Pulse Version: " << obj->pulseVersion << std::endl;
		stream << "Generating Software: " <<  obj->generatingSoftware << std::endl;
		stream << "File Signature: " << obj->fileSignature << std::endl;
        stream << "Spatial Reference: " <<  obj->spatialreference << std::endl;
		stream << "Creation Time (YYYY:MM:DD HH:MM:SS): " << obj->yearOfCreation << ":" << obj->monthOfCreation << ":" << obj->dayOfCreation << ":" << obj->hourOfCreation << ":"<< obj->minuteOfCreation << ":"<< obj->secondOfCreation << std::endl;
		stream << "Capture Time (YYYY:MM:DD HH:MM:SS): " << obj->yearOfCapture << ":" << obj->monthOfCapture << ":" << obj->dayOfCapture << ":" << obj->hourOfCapture << ":" << obj->minuteOfCapture << ":" << obj->secondOfCapture << std::endl;
        stream << "Index Type: ";
        if(obj->indexType == SPD_NO_IDX)
        {
            stream << "No Index\n";
        }
        else if(obj->indexType == SPD_CARTESIAN_IDX)
        {
            stream << "Cartesian\n";
        }
        else if(obj->indexType == SPD_SPHERICAL_IDX)
        {
            stream << "Spherical\n";
        }
        else if(obj->indexType == SPD_CYLINDRICAL_IDX)
        {
            stream << "Cylindrical\n";
        }
        else if(obj->indexType == SPD_POLAR_IDX)
        {
            stream << "Polar\n";
        }
        else if(obj->indexType == SPD_SCAN_IDX)
        {
            stream << "Scan\n";
        }
        else 
        {
            stream << std::endl;
        }
        stream << "The file contains: \n";
        if(obj->discretePtDefined == SPD_TRUE)
        {
            stream << "\t Contains discrete returns\n";
        }
        if(obj->decomposedPtDefined == SPD_TRUE)
        {
            stream << "\t Contains decomposed returns\n";
        }
        if(obj->transWaveformDefined == SPD_TRUE)
        {
            stream << "\t Contains transmitted waveforms\n";
        }
        if(obj->receiveWaveformDefined == SPD_TRUE)
        {
            stream << "\t Contains received waveforms\n";
        }
        stream << "Number of Points = " << obj->numPts << std::endl;
		stream << "Number of Pulses = " << obj->numPulses << std::endl;
		stream << "BBOX [xmin, ymin, xmax, ymax]: [" << obj->xMin << "," << obj->yMin << "," << obj->xMax << "," << obj->yMax << "]\n";
		stream << "BBOX [Azimuth Min, Zenith Min, Azimuth Max, Zenith Max]: [" << obj->azimuthMin << "," << obj->zenithMin << "," << obj->azimuthMax << "," << obj->zenithMax << "]\n";
		stream << "Z : [" << obj->zMin << "," << obj->zMax << "]\n";
        stream << "Range: [" << obj->rangeMin << "," << obj->rangeMax << "]\n";
        stream << "Scanline: [" << obj->scanlineMin << "," << obj->scanlineMax << "]\n";
        stream << "Scanline Idx: [" << obj->scanlineIdxMin << "," << obj->scanlineIdxMax << "]\n";
		stream << "Gridding [xSize,ySize] Bin Size: [" << obj->numBinsX << "," << obj->numBinsY << "] " << obj->binSize << std::endl;
        if(obj->wavelengths.size() > 0)
        {
            stream << "Wavelengths:\n";
            for(std::vector<float>::iterator iterWavel = obj->wavelengths.begin(); iterWavel != obj->wavelengths.end(); ++iterWavel)
            {
                stream << "\t" << *iterWavel << std::endl;
            }
        }
        else
        {
            stream << "Wavelengths: \n\tUnknown\n";
        }
        if(obj->bandwidths.size() > 0)
        {
            stream << "Bandwidths:\n";
            for(std::vector<float>::iterator iterBand = obj->bandwidths.begin(); iterBand != obj->bandwidths.end(); ++iterBand)
            {
                stream << "\t" << *iterBand << std::endl;
            }
        }
        else
        {
            stream << "Bandwidths: \n\tUnknown\n";
        }
        stream << "Pulse Repetition Freq: " << obj->pulseRepetitionFreq << std::endl;
        stream << "Beam Divergance: " << obj->beamDivergence << std::endl;
        stream << "Sensor Height: " << obj->sensorHeight << std::endl;
        stream << "Footprint: " << obj->footprint << std::endl;
        stream << "Max. Scan Angle: " << obj->maxScanAngle << std::endl;
        stream << "Waveform Bin Resolution: " << obj->waveformBitRes << std::endl;
        stream << "Temporal Waveform Bin Spacing: " << obj->temporalBinSpacing << std::endl;
        stream << "Sensor Speed: " << obj->sensorSpeed << std::endl;
        stream << "Sensor Scanrate: " << obj->sensorScanRate << std::endl;
        stream << "Pulse Density: " << obj->pulseDensity << std::endl;
        stream << "Point Density: " << obj->pointDensity << std::endl;
        stream << "Pulse cross track spacing: " << obj->pulseCrossTrackSpacing << std::endl;
        stream << "Pulse along track spacing: " << obj->pulseAlongTrackSpacing << std::endl;
        stream << "Pulse angular spacing azimuth: " << obj->pulseAngularSpacingAzimuth << std::endl;
        stream << "Pulse angular spacing zenith: " << obj->pulseAngularSpacingZenith << std::endl;
        stream << "Sensor Aperture Size: " << obj->sensorApertureSize << std::endl;
        stream << "Pulse Energy: " << obj->pulseEnergy << std::endl;
        stream << "Field of view: " << obj->fieldOfView << std::endl;
		stream << "User Meta Data: " << obj->userMetaField << std::endl;
		
		return stream;
	}
	
	
	SPDFile::~SPDFile()
	{
		
	}
    
    
    double* SPDFileProcessingUtilities::calcCartesainOverlap(SPDFile **spdFiles, boost::uint_fast16_t numOfFiles) 
    {
        /**
         * overlap[0] = min X
         * overlap[1] = max X
         * overlap[2] = min Y
         * overlap[3] = max Y
         * overlap[4] = min Z
         * overlap[5] = max Z
         */
        double *overlap = new double[6];
        bool first = true;
        for(boost::uint_fast16_t i = 0; i < numOfFiles; ++i)
        {
            if(first)
            {
                overlap[0] = spdFiles[i]->getXMin();
                overlap[1] = spdFiles[i]->getXMax();
                overlap[2] = spdFiles[i]->getYMin();
                overlap[3] = spdFiles[i]->getYMax();
                overlap[4] = spdFiles[i]->getZMin();
                overlap[5] = spdFiles[i]->getZMax();
                first = false;
            }
            else
            {
                if(spdFiles[i]->getXMin() > overlap[0])
                {
                    overlap[0] = spdFiles[i]->getXMin();
                }
                
                if(spdFiles[i]->getXMax() < overlap[1])
                {
                    overlap[1] = spdFiles[i]->getXMax();
                }
                
                if(spdFiles[i]->getYMin() > overlap[2])
                {
                    overlap[2] = spdFiles[i]->getYMin();
                }
                
                if(spdFiles[i]->getYMax() < overlap[3])
                {
                    overlap[3] = spdFiles[i]->getYMax();
                }
                
                if(spdFiles[i]->getZMin() > overlap[4])
                {
                    overlap[4] = spdFiles[i]->getZMin();
                }
                
                if(spdFiles[i]->getZMax() < overlap[5])
                {
                    overlap[5] = spdFiles[i]->getZMax();
                }
                
            }
        }
        
        return overlap;
    }
    
    double* SPDFileProcessingUtilities::calcSphericalOverlap(SPDFile **spdFiles, boost::uint_fast16_t numOfFiles) 
    {
        /**
         * overlap[0] = min Azimuth
         * overlap[1] = max Azimuth
         * overlap[2] = min Zenith
         * overlap[3] = max Zenith
         * overlap[4] = min Range
         * overlap[5] = max Range
         */
        double *overlap = new double[6];
        bool first = true;
        for(boost::uint_fast16_t i = 0; i < numOfFiles; ++i)
        {
            if(first)
            {
                overlap[0] = spdFiles[i]->getAzimuthMin();
                overlap[1] = spdFiles[i]->getAzimuthMax();
                overlap[2] = spdFiles[i]->getZenithMin();
                overlap[3] = spdFiles[i]->getZenithMax();
                overlap[4] = spdFiles[i]->getRangeMin();
                overlap[5] = spdFiles[i]->getRangeMax();
                first = false;
            }
            else
            {
                if(spdFiles[i]->getAzimuthMin() > overlap[0])
                {
                    overlap[0] = spdFiles[i]->getAzimuthMin();
                }
                
                if(spdFiles[i]->getAzimuthMax() < overlap[1])
                {
                    overlap[1] = spdFiles[i]->getAzimuthMax();
                }
                
                if(spdFiles[i]->getZenithMin() > overlap[2])
                {
                    overlap[2] = spdFiles[i]->getZenithMin();
                }
                
                if(spdFiles[i]->getZenithMax() < overlap[3])
                {
                    overlap[3] = spdFiles[i]->getZenithMax();
                }
                
                if(spdFiles[i]->getRangeMin() > overlap[4])
                {
                    overlap[4] = spdFiles[i]->getRangeMin();
                }
                
                if(spdFiles[i]->getRangeMax() < overlap[5])
                {
                    overlap[5] = spdFiles[i]->getRangeMax();
                }
                
            }
        }
        
        return overlap;
    }
}
