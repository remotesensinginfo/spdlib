/*
 *  SPDFile.h
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

#ifndef SPDFile_H
#define SPDFile_H

#include <iostream>
#include <string>
#include <time.h>
#include <vector>

#include <boost/cstdint.hpp>

#include "spd/SPDPoint.h"
#include "spd/SPDPulse.h"
#include "spd/SPDCommon.h"
#include "spd/SPDProcessingException.h"

namespace spdlib
{
	
	static const std::string GROUPNAME_HEADER( "/HEADER" );
	static const std::string GROUPNAME_INDEX( "/INDEX" );
	static const std::string GROUPNAME_QUICKLOOK( "/QUICKLOOK" );
	static const std::string GROUPNAME_DATA( "/DATA" );
	
	static const std::string SPDFILE_DATASETNAME_SPATIAL_REFERENCE( "/HEADER/SPATIAL_REFERENCE" );
	static const std::string SPDFILE_DATASETNAME_INDEX_TYPE( "/HEADER/INDEX_TYPE" );
	static const std::string SPDFILE_DATASETNAME_DISCRETE_PT_DEFINED( "/HEADER/DEFINED_DISCRETE_PT" );
	static const std::string SPDFILE_DATASETNAME_DECOMPOSED_PT_DEFINED( "/HEADER/DEFINED_DECOMPOSED_PT" );
	static const std::string SPDFILE_DATASETNAME_TRANS_WAVEFORM_DEFINED( "/HEADER/DEFINED_TRANS_WAVEFORM" );
    static const std::string SPDFILE_DATASETNAME_RECEIVE_WAVEFORM_DEFINED( "/HEADER/DEFINED_RECEIVE_WAVEFORM" );
	static const std::string SPDFILE_DATASETNAME_MAJOR_VERSION( "/HEADER/VERSION_MAJOR_SPD" );
	static const std::string SPDFILE_DATASETNAME_MINOR_VERSION( "/HEADER/VERSION_MINOR_SPD" );
    static const std::string SPDFILE_DATASETNAME_POINT_VERSION( "/HEADER/VERSION_POINT" );
    static const std::string SPDFILE_DATASETNAME_PULSE_VERSION( "/HEADER/VERSION_PULSE" );
	static const std::string SPDFILE_DATASETNAME_GENERATING_SOFTWARE( "/HEADER/GENERATING_SOFTWARE" );
	static const std::string SPDFILE_DATASETNAME_SYSTEM_IDENTIFIER( "/HEADER/SYSTEM_IDENTIFIER" );
	static const std::string SPDFILE_DATASETNAME_FILE_SIGNATURE( "/HEADER/FILE_SIGNATURE" );
	static const std::string SPDFILE_DATASETNAME_YEAR_OF_CREATION( "/HEADER/CREATION_YEAR_OF" );
	static const std::string SPDFILE_DATASETNAME_MONTH_OF_CREATION( "/HEADER/CREATION_MONTH_OF" );
	static const std::string SPDFILE_DATASETNAME_DAY_OF_CREATION( "/HEADER/CREATION_DAY_OF" );
	static const std::string SPDFILE_DATASETNAME_HOUR_OF_CREATION( "/HEADER/CREATION_HOUR_OF" );
	static const std::string SPDFILE_DATASETNAME_MINUTE_OF_CREATION( "/HEADER/CREATION_MINUTE_OF" );
	static const std::string SPDFILE_DATASETNAME_SECOND_OF_CREATION( "/HEADER/CREATION_SECOND_OF" );
    static const std::string SPDFILE_DATASETNAME_DATE_CREATION( "/HEADER/CREATION_DATE" );
    static const std::string SPDFILE_DATASETNAME_TIME_CREATION( "/HEADER/CREATION_TIME" );
	static const std::string SPDFILE_DATASETNAME_YEAR_OF_CAPTURE( "/HEADER/CAPTURE_YEAR_OF" );
	static const std::string SPDFILE_DATASETNAME_MONTH_OF_CAPTURE( "/HEADER/CAPTURE_MONTH_OF" );
	static const std::string SPDFILE_DATASETNAME_DAY_OF_CAPTURE( "/HEADER/CAPTURE_DAY_OF" );
	static const std::string SPDFILE_DATASETNAME_HOUR_OF_CAPTURE( "/HEADER/CAPTURE_HOUR_OF" );
	static const std::string SPDFILE_DATASETNAME_MINUTE_OF_CAPTURE( "/HEADER/CAPTURE_MINUTE_OF" );
	static const std::string SPDFILE_DATASETNAME_SECOND_OF_CAPTURE( "/HEADER/CAPTURE_SECOND_OF" );
	static const std::string SPDFILE_DATASETNAME_NUMBER_OF_POINTS( "/HEADER/NUMBER_OF_POINTS" );
	static const std::string SPDFILE_DATASETNAME_NUMBER_OF_PULSES( "/HEADER/NUMBER_OF_PULSES" );
	static const std::string SPDFILE_DATASETNAME_USER_META_DATA( "/HEADER/USER_META_DATA" );
	static const std::string SPDFILE_DATASETNAME_X_MIN( "/HEADER/X_MIN" );
	static const std::string SPDFILE_DATASETNAME_X_MAX( "/HEADER/X_MAX" );
	static const std::string SPDFILE_DATASETNAME_Y_MIN( "/HEADER/Y_MIN" );
	static const std::string SPDFILE_DATASETNAME_Y_MAX( "/HEADER/Y_MAX" );
	static const std::string SPDFILE_DATASETNAME_Z_MIN( "/HEADER/Z_MIN" );
	static const std::string SPDFILE_DATASETNAME_Z_MAX( "/HEADER/Z_MAX" );
	static const std::string SPDFILE_DATASETNAME_ZENITH_MIN( "/HEADER/ZENITH_MIN" );
	static const std::string SPDFILE_DATASETNAME_ZENITH_MAX( "/HEADER/ZENITH_MAX" );
	static const std::string SPDFILE_DATASETNAME_AZIMUTH_MIN( "/HEADER/AZIMUTH_MIN" );
	static const std::string SPDFILE_DATASETNAME_AZIMUTH_MAX( "/HEADER/AZIMUTH_MAX" );
	static const std::string SPDFILE_DATASETNAME_RANGE_MIN( "/HEADER/RANGE_MIN" );
	static const std::string SPDFILE_DATASETNAME_RANGE_MAX( "/HEADER/RANGE_MAX" );
    static const std::string SPDFILE_DATASETNAME_SCANLINE_MIN( "/HEADER/SCANLINE_MIN" );
	static const std::string SPDFILE_DATASETNAME_SCANLINE_MAX( "/HEADER/SCANLINE_MAX" );
	static const std::string SPDFILE_DATASETNAME_SCANLINE_IDX_MIN( "/HEADER/SCANLINE_IDX_MIN" );
	static const std::string SPDFILE_DATASETNAME_SCANLINE_IDX_MAX( "/HEADER/SCANLINE_IDX_MAX" );
	static const std::string SPDFILE_DATASETNAME_BIN_SIZE( "/HEADER/BIN_SIZE" );
	static const std::string SPDFILE_DATASETNAME_NUMBER_BINS_X( "/HEADER/NUMBER_BINS_X" );
	static const std::string SPDFILE_DATASETNAME_NUMBER_BINS_Y( "/HEADER/NUMBER_BINS_Y" );
	static const std::string SPDFILE_DATASETNAME_WAVELENGTH( "/HEADER/WAVELENGTH" );
    static const std::string SPDFILE_DATASETNAME_WAVELENGTHS( "/HEADER/WAVELENGTHS" );
    static const std::string SPDFILE_DATASETNAME_BANDWIDTHS( "/HEADER/BANDWIDTHS" );
    static const std::string SPDFILE_DATASETNAME_NUM_OF_WAVELENGTHS( "/HEADER/NUM_OF_WAVELENGTHS" );
	static const std::string SPDFILE_DATASETNAME_PULSE_REPETITION_FREQ( "/HEADER/SENSOR_PULSE_REPETITION_FREQ" );
	static const std::string SPDFILE_DATASETNAME_BEAM_DIVERGENCE( "/HEADER/SENSOR_BEAM_DIVERGENCE" );
	static const std::string SPDFILE_DATASETNAME_SENSOR_HEIGHT( "/HEADER/SENSOR_HEIGHT" );
	static const std::string SPDFILE_DATASETNAME_FOOTPRINT( "/HEADER/PULSE_FOOTPRINT" );
	static const std::string SPDFILE_DATASETNAME_MAX_SCAN_ANGLE( "/HEADER/SENSOR_MAX_SCAN_ANGLE" );
	static const std::string SPDFILE_DATASETNAME_RGB_DEFINED( "/HEADER/DEFINED_RGB" );
	static const std::string SPDFILE_DATASETNAME_PULSE_BLOCK_SIZE( "/HEADER/BLOCK_SIZE_PULSE" );
	static const std::string SPDFILE_DATASETNAME_POINT_BLOCK_SIZE( "/HEADER/BLOCK_SIZE_POINT" );
	static const std::string SPDFILE_DATASETNAME_RECEIVED_BLOCK_SIZE( "/HEADER/BLOCK_SIZE_RECEIVED" );
	static const std::string SPDFILE_DATASETNAME_TRANSMITTED_BLOCK_SIZE( "/HEADER/BLOCK_SIZE_TRANSMITTED" );
    static const std::string SPDFILE_DATASETNAME_WAVEFORM_BIT_RES( "/HEADER/WAVEFORM_BIT_RES" );
	static const std::string SPDFILE_DATASETNAME_TEMPORAL_BIN_SPACING( "/HEADER/SENSOR_TEMPORAL_BIN_SPACING" );
	static const std::string SPDFILE_DATASETNAME_RETURN_NUMBERS_SYN_GEN( "/HEADER/RETURN_NUMBERS_SYN_GEN" );
	static const std::string SPDFILE_DATASETNAME_HEIGHT_DEFINED( "/HEADER/DEFINED_HEIGHT" );
	static const std::string SPDFILE_DATASETNAME_SENSOR_SPEED( "/HEADER/SENSOR_SPEED" );
	static const std::string SPDFILE_DATASETNAME_SENSOR_SCAN_RATE( "/HEADER/SENSOR_SCAN_RATE" );
	static const std::string SPDFILE_DATASETNAME_POINT_DENSITY( "/HEADER/POINT_DENSITY" );
	static const std::string SPDFILE_DATASETNAME_PULSE_DENSITY( "/HEADER/PULSE_DENSITY" );
	static const std::string SPDFILE_DATASETNAME_PULSE_CROSS_TRACK_SPACING( "/HEADER/PULSE_CROSS_TRACK_SPACING" );
	static const std::string SPDFILE_DATASETNAME_PULSE_ALONG_TRACK_SPACING( "/HEADER/PULSE_ALONG_TRACK_SPACING" );
	static const std::string SPDFILE_DATASETNAME_ORIGIN_DEFINED( "/HEADER/DEFINED_ORIGIN" );
	static const std::string SPDFILE_DATASETNAME_PULSE_ANGULAR_SPACING_AZIMUTH( "/HEADER/PULSE_ANGULAR_SPACING_AZIMUTH" );
	static const std::string SPDFILE_DATASETNAME_PULSE_ANGULAR_SPACING_ZENITH( "/HEADER/PULSE_ANGULAR_SPACING_ZENITH" );
	static const std::string SPDFILE_DATASETNAME_PULSE_INDEX_METHOD( "/HEADER/PULSE_INDEX_METHOD" );
    static const std::string SPDFILE_DATASETNAME_FILE_TYPE( "/HEADER/FILE_TYPE" );
    static const std::string SPDFILE_DATASETNAME_SENSOR_APERTURE_SIZE( "/HEADER/SENSOR_APERTURE_SIZE" );
    static const std::string SPDFILE_DATASETNAME_PULSE_ENERGY( "/HEADER/PULSE_ENERGY" );
    static const std::string SPDFILE_DATASETNAME_FIELD_OF_VIEW( "/HEADER/FIELD_OF_VIEW" );
    
	static const std::string SPDFILE_DATASETNAME_PLS_PER_BIN( "/INDEX/PLS_PER_BIN" );
	static const std::string SPDFILE_DATASETNAME_BIN_OFFSETS( "/INDEX/BIN_OFFSETS" );
	
	static const std::string SPDFILE_DATASETNAME_QKLIMAGE( "/QUICKLOOK/IMAGE" );
	
	static const std::string SPDFILE_DATASETNAME_PULSES( "/DATA/PULSES" );
	static const std::string SPDFILE_DATASETNAME_POINTS( "/DATA/POINTS" );
	static const std::string SPDFILE_DATASETNAME_RECEIVED( "/DATA/RECEIVED" );
	static const std::string SPDFILE_DATASETNAME_TRANSMITTED( "/DATA/TRANSMITTED" );
	
	static const std::string ATTRIBUTENAME_CLASS( "CLASS" );
	static const std::string ATTRIBUTENAME_IMAGE_VERSION( "IMAGE_VERSION" );
    
	class SPDFile
	{		
	public:
		SPDFile(std::string filepath);
		
		void copyAttributesTo(SPDFile *spdFile);
		void copyAttributesFrom(SPDFile *spdFile);
		bool checkCompatibility(SPDFile *spdFile);
		bool checkCompatibilityExpandExtent(SPDFile *spdFile);
        bool checkCompatibilityGeneralCheckExpandExtent(SPDFile *spdFile);
        void expandExtent(SPDFile *spdFile);
		
        void setFilePath(std::string filepath){this->filepath = filepath;};
		std::string getFilePath(){return filepath;};
		
		void setSpatialReference(std::string spatialReference){this->spatialreference = spatialReference;};
		std::string getSpatialReference(){return spatialreference;};
		
		void setIndexType(boost::uint_fast16_t indexType){this->indexType = indexType;};
        boost::uint_fast16_t getIndexType(){return indexType;};
        
        void setFileType(boost::uint_fast16_t fileType){this->fileType = fileType;};
        boost::uint_fast16_t getFileType(){return fileType;};
		
		void setDiscretePtDefined(boost::int_fast16_t discretePtDefined){this->discretePtDefined = discretePtDefined;};
        boost::int_fast16_t getDiscretePtDefined(){return discretePtDefined;};
		
		void setDecomposedPtDefined(boost::int_fast16_t decomposedPtDefined){this->decomposedPtDefined = decomposedPtDefined;};
        boost::int_fast16_t getDecomposedPtDefined(){return decomposedPtDefined;};
		
		void setTransWaveformDefined(boost::int_fast16_t transWaveformDefined){this->transWaveformDefined = transWaveformDefined;};
        boost::int_fast16_t getTransWaveformDefined(){return transWaveformDefined;};
        
		void setReceiveWaveformDefined(boost::int_fast16_t receiveWaveformDefined){this->receiveWaveformDefined = receiveWaveformDefined;};
        boost::int_fast16_t getReceiveWaveformDefined(){return receiveWaveformDefined;};
        
		void setMajorSPDVersion(boost::uint_fast16_t majorSPDVersion){this->majorSPDVersion = majorSPDVersion;};
        boost::uint_fast16_t getMajorSPDVersion(){return majorSPDVersion;};
		
		void setMinorSPDVersion(boost::uint_fast16_t minorSPDVersion){this->minorSPDVersion = minorSPDVersion;};
        boost::uint_fast16_t getMinorSPDVersion(){return minorSPDVersion;};
        
        void setPointVersion(boost::uint_fast16_t pointVersion){this->pointVersion = pointVersion;};
        boost::uint_fast16_t getPointVersion(){return pointVersion;};
        
        void setPulseVersion(boost::uint_fast16_t pulseVersion){this->pulseVersion = pulseVersion;};
        boost::uint_fast16_t getPulseVersion(){return pulseVersion;};
		
		void setGeneratingSoftware(std::string generatingSoftware){this->generatingSoftware = generatingSoftware;};
		std::string getGeneratingSoftware(){return generatingSoftware;};
		
		void setSystemIdentifier(std::string systemIdentifier){this->systemIdentifier = systemIdentifier;};
		std::string getSystemIdentifier(){return systemIdentifier;};
		
		void setFileSignature(std::string fileSignature){this->fileSignature = fileSignature;};
		std::string getFileSignature(){return fileSignature;};
		
		void setYearOfCreation(boost::uint_fast16_t year){this->yearOfCreation = year;};
        boost::uint_fast16_t getYearOfCreation(){return yearOfCreation;};
		
		void setMonthOfCreation(boost::uint_fast16_t month){this->monthOfCreation = month;};
        boost::uint_fast16_t getMonthOfCreation(){return monthOfCreation;};
		
		void setDayOfCreation(boost::uint_fast16_t day){this->dayOfCreation = day;};
        boost::uint_fast16_t getDayOfCreation(){return dayOfCreation;};
		
		void setHourOfCreation(boost::uint_fast16_t hourOfCreation){this->hourOfCreation = hourOfCreation;};
        boost::uint_fast16_t getHourOfCreation(){return hourOfCreation;};
		
		void setMinuteOfCreation(boost::uint_fast16_t minuteOfCreation){this->minuteOfCreation = minuteOfCreation;};
        boost::uint_fast16_t getMinuteOfCreation(){return minuteOfCreation;};
		
		void setSecondOfCreation(boost::uint_fast16_t secondOfCreation){this->secondOfCreation = secondOfCreation;};
        boost::uint_fast16_t getSecondOfCreation(){return secondOfCreation;};
		
		void setYearOfCapture(boost::uint_fast16_t year){this->yearOfCapture = year;};
        boost::uint_fast16_t getYearOfCapture(){return yearOfCapture;};
		
		void setMonthOfCapture(boost::uint_fast16_t month){this->monthOfCapture = month;};
        boost::uint_fast16_t getMonthOfCapture(){return monthOfCapture;};
		
		void setDayOfCapture(boost::uint_fast16_t day){this->dayOfCapture = day;};
        boost::uint_fast16_t getDayOfCapture(){return dayOfCapture;};
		
		void setHourOfCapture(boost::uint_fast16_t hourOfCapture){this->hourOfCapture = hourOfCapture;};
        boost::uint_fast16_t getHourOfCapture(){return hourOfCapture;};
		
		void setMinuteOfCapture(boost::uint_fast16_t minuteOfCapture){this->minuteOfCapture = minuteOfCapture;};
        boost::uint_fast16_t getMinuteOfCapture(){return minuteOfCapture;};
		
		void setSecondOfCapture(boost::uint_fast16_t secondOfCapture){this->secondOfCapture = secondOfCapture;};
        boost::uint_fast16_t getSecondOfCapture(){return secondOfCapture;};
		
		void setNumberOfPoints(boost::uint_fast64_t numPts){this->numPts = numPts;};
        boost::uint_fast64_t getNumberOfPoints(){return numPts;};
		
		void setNumberOfPulses(boost::uint_fast64_t numPulses){this->numPulses = numPulses;};
        boost::uint_fast64_t getNumberOfPulses(){return numPulses;};
		
		void setUserMetaField(std::string userMetaField){this->userMetaField = userMetaField;};
		std::string getUserMetaField(){return userMetaField;};
		
		void setXMin(double xMin){this->xMin = xMin;};
		double getXMin(){return xMin;};
		
		void setXMax(double xMax){this->xMax = xMax;};
		double getXMax(){return xMax;};
		
		void setYMin(double yMin){this->yMin = yMin;};
		double getYMin(){return yMin;};
		
		void setYMax(double yMax){this->yMax = yMax;};
		double getYMax(){return yMax;};
		
		void setZMin(double zMin){this->zMin = zMin;};
		double getZMin(){return zMin;};
		
		void setZMax(double zMax){this->zMax = zMax;};
		double getZMax(){return zMax;};
		
		void setBoundingBox(double xMin, double xMax, double yMin, double yMax)
		{
			this->xMin = xMin;
			this->xMax = xMax;
			this->yMin = yMin;
			this->yMax = yMax;
		};
		
		void setBoundingVolume(double xMin, double xMax, double yMin, double yMax, double zMin, double zMax)
		{
			this->xMin = xMin;
			this->xMax = xMax;
			this->yMin = yMin;
			this->yMax = yMax;
			this->zMin = zMin;
			this->zMax = zMax;
		};
		
		void setZenithMin(double zenith){this->zenithMin = zenith;};
		double getZenithMin(){return zenithMin;};
		
		void setZenithMax(double zenith){this->zenithMax = zenith;};
		double getZenithMax(){return zenithMax;};
		
		void setAzimuthMin(double azimuth){this->azimuthMin = azimuth;};
		double getAzimuthMin(){return azimuthMin;};
		
		void setAzimuthMax(double azimuth){this->azimuthMax = azimuth;};
		double getAzimuthMax(){return azimuthMax;};
		
		void setRangeMin(double range){this->rangeMin = range;};
		double getRangeMin(){return rangeMin;};
		
		void setRangeMax(double range){this->rangeMax = range;};
		double getRangeMax(){return rangeMax;};
		
		void setBoundingBoxSpherical(double zenithMin, double zenithMax, double azimuthMin, double azimuthMax)
		{
			this->zenithMin = zenithMin;
			this->zenithMax = zenithMax;
			this->azimuthMin = azimuthMin;
			this->azimuthMax = azimuthMax;
		};
		
		void setBoundingVolumeSpherical(double zenithMin, double zenithMax, double azimuthMin, double azimuthMax, double rangeMin, double rangeMax)
		{
			this->zenithMin = zenithMin;
			this->zenithMax = zenithMax;
			this->azimuthMin = azimuthMin;
			this->azimuthMax = azimuthMax;
			this->rangeMin = rangeMin;
			this->rangeMax = rangeMax;
		};
        
        void setScanlineMin(double scanline){this->scanlineMin = scanline;};
		double getScanlineMin(){return scanlineMin;};
		
		void setScanlineMax(double scanline){this->scanlineMax = scanline;};
		double getScanlineMax(){return scanlineMax;};
		
		void setScanlineIdxMin(double scanlineIdx){this->scanlineIdxMin = scanlineIdx;};
		double getScanlineIdxMin(){return scanlineIdxMin;};
		
		void setScanlineIdxMax(double scanlineIdx){this->scanlineIdxMax = scanlineIdx;};
		double getScanlineIdxMax(){return scanlineIdxMax;};
        
        void setBoundingBoxScanline(double scanlineMin, double scanlineMax, double scanlineIdxMin, double scanlineIdxMax)
		{
			this->scanlineMin = scanlineMin;
			this->scanlineMax = scanlineMax;
			this->scanlineIdxMin = scanlineIdxMin;
			this->scanlineIdxMax = scanlineIdxMax;
		};
		
		void setBinSize(float binSize){this->binSize = binSize;};
		float getBinSize(){return binSize;};
		
		void setNumberBinsX(boost::uint_fast32_t numBinsX){this->numBinsX = numBinsX;};
        boost::uint_fast32_t getNumberBinsX(){return numBinsX;};
		
		void setNumberBinsY(boost::uint_fast32_t numBinsY){this->numBinsY = numBinsY;};
        boost::uint_fast32_t getNumberBinsY(){return numBinsY;};
		
		void setWavelengths(std::vector<float> wavelengths){this->wavelengths = wavelengths;};
		std::vector<float>* getWavelengths(){return &wavelengths;};
        
        void setBandwidths(std::vector<float> bandwidths){this->bandwidths = bandwidths;};
		std::vector<float>* getBandwidths(){return &bandwidths;};
        
        void setNumOfWavelengths(boost::uint_fast16_t numOfWavelengths){this->numOfWavelengths = numOfWavelengths;};
		boost::uint_fast16_t getNumOfWavelengths(){return numOfWavelengths;};
		
		void setPulseRepetitionFreq(float pulseRepetitionFreq){this->pulseRepetitionFreq = pulseRepetitionFreq;};
		float getPulseRepetitionFreq(){return pulseRepetitionFreq;};
		
		void setBeamDivergence(float beamDivergence){this->beamDivergence = beamDivergence;};
		float getBeamDivergence(){return beamDivergence;};
		
		void setSensorHeight(double sensorHeight){this->sensorHeight = sensorHeight;};
		double getSensorHeight(){return sensorHeight;};
		
		void setFootprint(float footprint){this->footprint = footprint;};
		float getFootprint(){return footprint;};
		
		void setMaxScanAngle(float maxScanAngle){this->maxScanAngle = maxScanAngle;};
		float getMaxScanAngle(){return maxScanAngle;};
		
		void setRGBDefined(boost::int_fast16_t rgbDefined){this->rgbDefined = rgbDefined;};
        boost::int_fast16_t getRGBDefined(){return rgbDefined;};
		
		void setPulseBlockSize(boost::uint_fast16_t pulseBlockSize){this->pulseBlockSize = pulseBlockSize;};
        boost::uint_fast16_t getPulseBlockSize(){return pulseBlockSize;};
		
		void setPointBlockSize(boost::uint_fast16_t pointBlockSize){this->pointBlockSize = pointBlockSize;};
        boost::uint_fast16_t getPointBlockSize(){return pointBlockSize;};
		
		void setReceivedBlockSize(boost::uint_fast16_t receivedBlockSize){this->receivedBlockSize = receivedBlockSize;};
        boost::uint_fast16_t getReceivedBlockSize(){return receivedBlockSize;};
		
		void setTransmittedBlockSize(boost::uint_fast16_t transmittedBlockSize){this->transmittedBlockSize = transmittedBlockSize;};
        boost::uint_fast16_t getTransmittedBlockSize(){return transmittedBlockSize;};
        
        void setWaveformBitRes(boost::uint_fast16_t waveformBitRes){this->waveformBitRes = waveformBitRes;};
        boost::uint_fast16_t getWaveformBitRes(){return waveformBitRes;};
		
		void setTemporalBinSpacing(double temporalBinSpacing){this->temporalBinSpacing = temporalBinSpacing;};
		double getTemporalBinSpacing(){return temporalBinSpacing;};
		
		void setReturnNumsSynGen(boost::int_fast16_t returnNumsSynGen){this->returnNumsSynGen = returnNumsSynGen;};
        boost::int_fast16_t getReturnNumsSynGen(){return returnNumsSynGen;};
		
		void setHeightDefined(boost::int_fast16_t heightDefined){this->heightDefined = heightDefined;};
        boost::int_fast16_t getHeightDefined(){return heightDefined;};
		
		void setSensorSpeed(float sensorSpeed){this->sensorSpeed = sensorSpeed;};
		float getSensorSpeed(){return sensorSpeed;};
        
		void setSensorScanRate(float sensorScanRate){this->sensorScanRate = sensorScanRate;};
		float getSensorScanRate(){return sensorScanRate;};
        
		void setPointDensity(float pointDensity){this->pointDensity = pointDensity;};
		float getPointDensity(){return pointDensity;};
        
		void setPulseDensity(float pulseDensity){this->pulseDensity = pulseDensity;};
		float getPulseDensity(){return pulseDensity;};
		
		void setPulseCrossTrackSpacing(float pulseCrossTrackSpacing){this->pulseCrossTrackSpacing = pulseCrossTrackSpacing;};
		float getPulseCrossTrackSpacing(){return pulseCrossTrackSpacing;};
		
		void setPulseAlongTrackSpacing(float pulseAlongTrackSpacing){this->pulseAlongTrackSpacing = pulseAlongTrackSpacing;};
		float getPulseAlongTrackSpacing(){return pulseAlongTrackSpacing;};
		
		void setOriginDefined(boost::int_fast16_t originDefined){this->originDefined = originDefined;};
        boost::int_fast16_t getOriginDefined(){return originDefined;};
		
		void setPulseAngularSpacingAzimuth(float pulseAngularSpacingAzimuth){this->pulseAngularSpacingAzimuth = pulseAngularSpacingAzimuth;};
		float getPulseAngularSpacingAzimuth(){return pulseAngularSpacingAzimuth;};
		
		void setPulseAngularSpacingZenith(float pulseAngularSpacingZenith){this->pulseAngularSpacingZenith = pulseAngularSpacingZenith;};
		float getPulseAngularSpacingZenith(){return pulseAngularSpacingZenith;};
		
		void setPulseIdxMethod(boost::int_fast16_t pulseIdxMethod){this->pulseIdxMethod = pulseIdxMethod;};
        boost::int_fast16_t getPulseIdxMethod(){return pulseIdxMethod;};
        
        void setSensorApertureSize(float sensorApertureSize){this->sensorApertureSize = sensorApertureSize;};
		float getSensorApertureSize(){return sensorApertureSize;};
        
        void setPulseEnergy(float pulseEnergy){this->pulseEnergy = pulseEnergy;};
		float getPulseEnergy(){return pulseEnergy;};
        
        void setFieldOfView(float fieldOfView){this->fieldOfView = fieldOfView;};
		float getFieldOfView(){return fieldOfView;};
		
		friend std::ostream& operator<<(std::ostream& stream, SPDFile &obj);
		friend std::ostream& operator<<(std::ostream& stream, SPDFile *obj);
		
		~SPDFile();
		
	protected:
		/**
		 * The file name and location of the system.
		 */
		std::string filepath;
		/**
		 * The spatial reference for the data as a proj4 string.
		 */
		std::string spatialreference;
		/**
		 * Define how the points are indexed.
		 */ 
        boost::uint_fast16_t indexType;
        /**
		 * Define the file type (SPD_SEQ, SPD_UNSEQ and UPD).
		 */ 
        boost::uint_fast16_t fileType;
		/**
		 * Are discrete returns defined within this file.
		 */
        boost::int_fast16_t discretePtDefined;
		/**
		 * Are decomposed discrete returns defined within this file.
		 */
        boost::int_fast16_t decomposedPtDefined;
		/**
		 * Are transmitted waveforms defined within this file.
		 */
        boost::int_fast16_t transWaveformDefined;
        /**
		 * Are received waveforms defined within this file.
		 */
        boost::int_fast16_t receiveWaveformDefined;
		/**
		 * Major Version to which this SPD file adheres.
		 */
        boost::uint_fast16_t majorSPDVersion;
		/**
		 * Minor Version to which this SPD file adheres.
		 */
        boost::uint_fast16_t minorSPDVersion;
        /**
		 * Point Version contained within this SPD File.
		 */
        boost::uint_fast16_t pointVersion;
        /**
		 * Pulse Version contained within this SPD File.
		 */
        boost::uint_fast16_t pulseVersion;
		/**
		 * The software from which this file was created.
		 */
		std::string generatingSoftware;
		/**
		 * The sensor which collected the data
		 */
		std::string systemIdentifier;
		/**
		 * The file signature of the file type ('SPDFile'). 
		 */
		std::string fileSignature;
		/**
		 * The year of creation
		 */
        boost::uint_fast16_t yearOfCreation;
		/**
		 * The month of creation
		 */ 
        boost::uint_fast16_t monthOfCreation;
		/**
		 * The day of creation
		 */
        boost::uint_fast16_t dayOfCreation;
		/**
		 * The hour of creation 
		 */
        boost::uint_fast16_t hourOfCreation;
		/**
		 * The minute of creation 
		 */
        boost::uint_fast16_t minuteOfCreation;
		/**
		 * The second of creation 
		 */
        boost::uint_fast16_t secondOfCreation;
		/**
		 * The year of capture
		 */
        boost::uint_fast16_t yearOfCapture;
		/**
		 * The month of capture
		 */ 
        boost::uint_fast16_t monthOfCapture;
		/**
		 * The day of capture
		 */
        boost::uint_fast16_t dayOfCapture;
		/**
		 * The hour of capture 
		 */
        boost::uint_fast16_t hourOfCapture;
		/**
		 * The minute of capture 
		 */
        boost::uint_fast16_t minuteOfCapture;
		/**
		 * The second of capture 
		 */
        boost::uint_fast16_t secondOfCapture;		
		/**
		 * The number of points within the file.
		 */
        boost::uint_fast64_t numPts;
		/**
		 * The number of points within the file.
		 */
        boost::uint_fast64_t numPulses;
		/**
		 * A text attribute the user can use to store
		 * further information (meta-data) as required.
		 */
		std::string userMetaField;
		/**
		 * The minimum X within the scenes bounding box (Cartesian)
		 */
		double xMin;
		/**
		 * The maximum X within the scenes bounding box (Cartesian)
		 */
		double xMax;
		/**
		 * The minimum Y within the scenes bounding box (Cartesian)
		 */
		double yMin;
		/**
		 * The maximum Y within the scenes bounding box (Cartesian)
		 */
		double yMax;
		/**
		 * The minimum Z within the scenes bounding box (Cartesian)
		 */
		double zMin;
		/**
		 * The maximum Z within the scenes bounding box (Cartesian)
		 */
		double zMax;
		/**
		 * The minimum zenith within the scenes bounding box (Spherical)
		 */
		double zenithMin;
		/**
		 * The maximum zenith within the scenes bounding box (Spherical)
		 */
		double zenithMax;
		/**
		 * The minimum azimuth within the scenes bounding box (Spherical)
		 */
		double azimuthMin;
		/**
		 * The maximum azimuth within the scenes bounding box (Spherical)
		 */
		double azimuthMax;
		/**
		 * The minimum range within the scenes bounding box (Spherical)
		 */
		double rangeMin;
		/**
		 * The maximum range within the scenes bounding box (Spherical)
		 */
		double rangeMax;
        /**
		 * The minimum scanline within the scene
		 */
		double scanlineMin;
		/**
		 * The maximum scanline within the scene
		 */
		double scanlineMax;
		/**
		 * The minimum scanline index within the scene
		 */
		double scanlineIdxMin;
		/**
		 * The maximum scanline index within the scene
		 */
		double scanlineIdxMax;
		/**
		 * The bin Size of the files index
		 */
		float binSize;
		/**
		 * The number of bins in the X axis
		 */
        boost::uint_fast32_t numBinsX;
		/**
		 * The number of bins in the Y axis
		 */
        boost::uint_fast32_t numBinsY;
		/**
		 * The wavelengths (in nm) of the sensor used to capture the data
		 */
		std::vector<float> wavelengths;
        /**
		 * The bandwidth used for each wavelength by the sensor used to capture the data
		 */
		std::vector<float> bandwidths;
        /**
		 * The number of wavelengths stored within the file.
		 */
        boost::uint_fast16_t numOfWavelengths;
		/**
		 * The pulse repetition frequency of the sensor used to capture the data
		 */
		float pulseRepetitionFreq;
		/**
		 * The beam divergence of the sensor used to capture the data
		 */
		float beamDivergence;
		/**
		 * The nominal flying height (airborne) or height of scanner (terrestrial)
		 */
		double sensorHeight;
		/**
		 * The nominal footprint of the laser used to capture the data
		 */
		float footprint;
		/**
		 * The maximum scan angle of the sensor used to capture the data
		 */
		float maxScanAngle;
		/**
		 * Are RGB values attached to the points
		 */
        boost::int_fast16_t rgbDefined;
		/**
		 * Pulse compression block size
		 */
        boost::uint_fast16_t pulseBlockSize;
		/**
		 * Point compression block size
		 */
        boost::uint_fast16_t pointBlockSize;
		/**
		 * Received compression block size
		 */
        boost::uint_fast16_t receivedBlockSize;
		/**
		 * Transmitted compression block size
		 */
        boost::uint_fast16_t transmittedBlockSize;
        /**
         * The number of bits used to store the waveforms.
         */
        boost::uint_fast16_t waveformBitRes;
		/**
		 * Time between waveform bins
		 */
		double temporalBinSpacing;
		/**
		 * Return numbers synthetically generated
		 */
        boost::int_fast16_t returnNumsSynGen;
		/**
		 * Height defined
		 */
        boost::int_fast16_t heightDefined;
		/**
		 * Sensor Speed
		 */
		float sensorSpeed;
		/**
		 * Sensor scan rate
 		 */
		float sensorScanRate;
		/**
		 * Point density
		 */
		float pointDensity;
		/**
		 * Pulse density
		 */
		float pulseDensity;
		/**
		 * Pulse cross track spacing
		 */
		float pulseCrossTrackSpacing;
		/**
		 * Pulse along track spacing
		 */
		float pulseAlongTrackSpacing;
		/**
		 * Pulse origin defined
		 */
        boost::int_fast16_t originDefined;
		/**
		 * Pulse angular spacing in the azimuth
		 */
		float pulseAngularSpacingAzimuth;
		/**
		 * Pulse angular spacing in the zenith
		 */
		float pulseAngularSpacingZenith;
		/**
		 * The method by which the points are indexed 
		 * (i.e., first return or last return etc).
		 */
        boost::int_fast16_t pulseIdxMethod;        
        /**
         * Sensor Aperture size
         */
        float sensorApertureSize;
        /**
         * Pulse Energy (mj)
         */
        float pulseEnergy;
        /**
         * Field of view of the sensor
         */
        float fieldOfView;
	};
    
    
    class SPDFileProcessingUtilities
    {
    public:
        SPDFileProcessingUtilities(){};
        /**
         * overlap[0] = min X
         * overlap[1] = max X
         * overlap[2] = min Y
         * overlap[3] = max Y
         * overlap[4] = min Z
         * overlap[5] = max Z
         */
        double* calcCartesainOverlap(SPDFile **spdFiles,boost::uint_fast16_t numOfFiles) throw(SPDProcessingException);
        /**
         * overlap[0] = min Azimuth
         * overlap[1] = max Azimuth
         * overlap[2] = min Zenith
         * overlap[3] = max Zenith
         * overlap[4] = min Range
         * overlap[5] = max Range
         */
        double* calcSphericalOverlap(SPDFile **spdFiles,boost::uint_fast16_t numOfFiles) throw(SPDProcessingException);
        ~SPDFileProcessingUtilities(){};
    };
}

#endif



