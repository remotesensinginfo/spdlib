/*
 *  SPDSeqFileWriter.cpp
 *  spdlib
 *
 *  Created by Pete Bunting on 28/11/2010.
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

#include "spd/SPDFileWriter.h"


namespace spdlib
{
    
    void SPDFileWriter::writeHeaderInfo(H5File *spdOutH5File, SPDFile *spdFile) throw(SPDIOException)
    {
        try 
		{
			IntType int16bitDataTypeDisk( PredType::STD_I16LE );
			IntType uint16bitDataTypeDisk( PredType::STD_U16LE );
            IntType uint32bitDataType( PredType::STD_U32LE );
			IntType uint64bitDataTypeDisk( PredType::STD_U64LE );
			FloatType floatDataTypeDisk( PredType::IEEE_F32LE );
			FloatType doubleDataTypeDisk( PredType::IEEE_F64LE );
			
			float outFloatDataValue[1];
			double outDoubleDataValue[1];
			int out16bitintDataValue[1];
			unsigned int out16bitUintDataValue[1];
            unsigned long out32bitUintDataValue[1];
			unsigned long long out64bitUintDataValue[1];
			
			unsigned int numLinesStr = 1;
			unsigned int rankStr = 1;
			const char **wStrdata = NULL;
			
			hsize_t	dims1Str[1];
			dims1Str[0] = numLinesStr;
			DataSpace dataspaceStrAll(rankStr, dims1Str);
			StrType strTypeAll(0, H5T_VARIABLE);
			
			hsize_t dimsValue[1];
			dimsValue[0] = 1;
			DataSpace singleValueDataSpace(1, dimsValue);
			
			if((H5T_STRING!=H5Tget_class(strTypeAll.getId())) || (!H5Tis_variable_str(strTypeAll.getId())))
			{
				throw SPDIOException("The string data type defined is not variable.");
			}
            
            DataSet datasetSpatialReference = spdOutH5File->createDataSet(SPDFILE_DATASETNAME_SPATIAL_REFERENCE, strTypeAll, dataspaceStrAll);
			wStrdata = new const char*[numLinesStr];
			wStrdata[0] = spdFile->getSpatialReference().c_str();			
			datasetSpatialReference.write((void*)wStrdata, strTypeAll);
			datasetSpatialReference.close();
			delete[] wStrdata;
			
            DataSet datasetFileType = spdOutH5File->createDataSet(SPDFILE_DATASETNAME_FILE_TYPE, uint16bitDataTypeDisk, singleValueDataSpace);
            out16bitUintDataValue[0] = spdFile->getFileType();
			datasetFileType.write( out16bitUintDataValue, PredType::NATIVE_UINT );
            
            DataSet datasetPulseIndexMethod = spdOutH5File->createDataSet(SPDFILE_DATASETNAME_PULSE_INDEX_METHOD, uint16bitDataTypeDisk, singleValueDataSpace);
            out16bitUintDataValue[0] = spdFile->getIndexType();
			datasetPulseIndexMethod.write( out16bitUintDataValue, PredType::NATIVE_UINT );
            
            DataSet datasetDiscreteDefined = spdOutH5File->createDataSet(SPDFILE_DATASETNAME_DISCRETE_PT_DEFINED, int16bitDataTypeDisk, singleValueDataSpace);
			out16bitintDataValue[0] = spdFile->getDiscretePtDefined();
			datasetDiscreteDefined.write( out16bitintDataValue, PredType::NATIVE_INT );
			
            DataSet datasetDecomposedDefined = spdOutH5File->createDataSet(SPDFILE_DATASETNAME_DECOMPOSED_PT_DEFINED, int16bitDataTypeDisk, singleValueDataSpace);
			out16bitintDataValue[0] = spdFile->getDecomposedPtDefined();
			datasetDecomposedDefined.write( out16bitintDataValue, PredType::NATIVE_INT );
			
            DataSet datasetTransWaveformDefined = spdOutH5File->createDataSet(SPDFILE_DATASETNAME_TRANS_WAVEFORM_DEFINED, int16bitDataTypeDisk, singleValueDataSpace);
			out16bitintDataValue[0] = spdFile->getTransWaveformDefined();
			datasetTransWaveformDefined.write( out16bitintDataValue, PredType::NATIVE_INT );
            
            DataSet datasetReceiveWaveformDefined = spdOutH5File->createDataSet(SPDFILE_DATASETNAME_RECEIVE_WAVEFORM_DEFINED, int16bitDataTypeDisk, singleValueDataSpace);
            out16bitintDataValue[0] = spdFile->getReceiveWaveformDefined();
			datasetReceiveWaveformDefined.write( out16bitintDataValue, PredType::NATIVE_INT );
            
            DataSet datasetIndexType = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_INDEX_TYPE, uint16bitDataTypeDisk, singleValueDataSpace);
            out16bitUintDataValue[0] = spdFile->getIndexType();
			datasetIndexType.write( out16bitUintDataValue, PredType::NATIVE_UINT );
            
            DataSet datasetMajorVersion = spdOutH5File->createDataSet(SPDFILE_DATASETNAME_MAJOR_VERSION, uint16bitDataTypeDisk, singleValueDataSpace);
			out16bitUintDataValue[0] = spdFile->getMajorSPDVersion();
			datasetMajorVersion.write( out16bitUintDataValue, PredType::NATIVE_UINT );
			
            DataSet datasetMinorVersion = spdOutH5File->createDataSet(SPDFILE_DATASETNAME_MINOR_VERSION, uint16bitDataTypeDisk, singleValueDataSpace);
			out16bitUintDataValue[0] = spdFile->getMinorSPDVersion();
			datasetMinorVersion.write( out16bitUintDataValue, PredType::NATIVE_UINT );
            
            DataSet datasetPointVersion = spdOutH5File->createDataSet(SPDFILE_DATASETNAME_POINT_VERSION, uint16bitDataTypeDisk, singleValueDataSpace);
            out16bitUintDataValue[0] = spdFile->getPointVersion();
			datasetPointVersion.write( out16bitUintDataValue, PredType::NATIVE_UINT );
            
            DataSet datasetPulseVersion = spdOutH5File->createDataSet(SPDFILE_DATASETNAME_PULSE_VERSION, uint16bitDataTypeDisk, singleValueDataSpace);
            out16bitUintDataValue[0] = spdFile->getPulseVersion();
			datasetPulseVersion.write( out16bitUintDataValue, PredType::NATIVE_UINT );
			
            DataSet datasetGeneratingSoftware = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_GENERATING_SOFTWARE, strTypeAll, dataspaceStrAll);
			wStrdata = new const char*[numLinesStr];
			wStrdata[0] = spdFile->getGeneratingSoftware().c_str();			
			datasetGeneratingSoftware.write((void*)wStrdata, strTypeAll);
			datasetGeneratingSoftware.close();
			delete[] wStrdata;
			
            DataSet datasetSystemIdentifier = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_SYSTEM_IDENTIFIER, strTypeAll, dataspaceStrAll);
			wStrdata = new const char*[numLinesStr];
			wStrdata[0] = spdFile->getSystemIdentifier().c_str();
			datasetSystemIdentifier.write((void*)wStrdata, strTypeAll);
			datasetSystemIdentifier.close();
			delete[] wStrdata;
			
            DataSet datasetFileSignature = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_FILE_SIGNATURE, strTypeAll, dataspaceStrAll);
			wStrdata = new const char*[numLinesStr];
			wStrdata[0] = spdFile->getFileSignature().c_str();			
			datasetFileSignature.write((void*)wStrdata, strTypeAll);
			datasetFileSignature.close();
			delete[] wStrdata;
			
            DataSet datasetYearOfCreation = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_YEAR_OF_CREATION, uint16bitDataTypeDisk, singleValueDataSpace);
			out16bitUintDataValue[0] = spdFile->getYearOfCreation();
			datasetYearOfCreation.write( out16bitUintDataValue, PredType::NATIVE_UINT );
			
            DataSet datasetMonthOfCreation = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_MONTH_OF_CREATION, uint16bitDataTypeDisk, singleValueDataSpace);
			out16bitUintDataValue[0] = spdFile->getMonthOfCreation();
			datasetMonthOfCreation.write( out16bitUintDataValue, PredType::NATIVE_UINT );
			
            DataSet datasetDayOfCreation = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_DAY_OF_CREATION, uint16bitDataTypeDisk, singleValueDataSpace);
			out16bitUintDataValue[0] = spdFile->getDayOfCreation();
			datasetDayOfCreation.write( out16bitUintDataValue, PredType::NATIVE_UINT );
			
            DataSet datasetHourOfCreation = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_HOUR_OF_CREATION, uint16bitDataTypeDisk, singleValueDataSpace);
			out16bitUintDataValue[0] = spdFile->getHourOfCreation();
			datasetHourOfCreation.write( out16bitUintDataValue, PredType::NATIVE_UINT );
			
            DataSet datasetMinuteOfCreation = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_MINUTE_OF_CREATION, uint16bitDataTypeDisk, singleValueDataSpace);
			out16bitUintDataValue[0] = spdFile->getMinuteOfCreation();
			datasetMinuteOfCreation.write( out16bitUintDataValue, PredType::NATIVE_UINT );
			
            DataSet datasetSecondOfCreation = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_SECOND_OF_CREATION, uint16bitDataTypeDisk, singleValueDataSpace);
			out16bitUintDataValue[0] = spdFile->getSecondOfCreation();
			datasetSecondOfCreation.write( out16bitUintDataValue, PredType::NATIVE_UINT );
			
            DataSet datasetYearOfCapture = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_YEAR_OF_CAPTURE, uint16bitDataTypeDisk, singleValueDataSpace);
			out16bitUintDataValue[0] = spdFile->getYearOfCapture();
			datasetYearOfCapture.write( out16bitUintDataValue, PredType::NATIVE_UINT );
			
            DataSet datasetMonthOfCapture = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_MONTH_OF_CAPTURE, uint16bitDataTypeDisk, singleValueDataSpace);
			out16bitUintDataValue[0] = spdFile->getMonthOfCapture();
			datasetMonthOfCapture.write( out16bitUintDataValue, PredType::NATIVE_UINT );
			
            DataSet datasetDayOfCapture = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_DAY_OF_CAPTURE, uint16bitDataTypeDisk, singleValueDataSpace);
			out16bitUintDataValue[0] = spdFile->getDayOfCapture();
			datasetDayOfCapture.write( out16bitUintDataValue, PredType::NATIVE_UINT );
			
            DataSet datasetHourOfCapture = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_HOUR_OF_CAPTURE, uint16bitDataTypeDisk, singleValueDataSpace);
			out16bitUintDataValue[0] = spdFile->getHourOfCapture();
			datasetHourOfCapture.write( out16bitUintDataValue, PredType::NATIVE_UINT );
			
            DataSet datasetMinuteOfCapture = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_MINUTE_OF_CAPTURE, uint16bitDataTypeDisk, singleValueDataSpace);
			out16bitUintDataValue[0] = spdFile->getMinuteOfCapture();
			datasetMinuteOfCapture.write( out16bitUintDataValue, PredType::NATIVE_UINT );
			
            DataSet datasetSecondOfCapture = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_SECOND_OF_CAPTURE, uint16bitDataTypeDisk, singleValueDataSpace);
			out16bitUintDataValue[0] = spdFile->getSecondOfCapture();
			datasetSecondOfCapture.write( out16bitUintDataValue, PredType::NATIVE_UINT );
			
            DataSet datasetNumberOfPoints = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_NUMBER_OF_POINTS, uint64bitDataTypeDisk, singleValueDataSpace);
			out64bitUintDataValue[0] = spdFile->getNumberOfPoints();
			datasetNumberOfPoints.write( out64bitUintDataValue, PredType::NATIVE_ULLONG );
			
            DataSet datasetNumberOfPulses = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_NUMBER_OF_PULSES, uint64bitDataTypeDisk, singleValueDataSpace);
			out64bitUintDataValue[0] = spdFile->getNumberOfPulses();
			datasetNumberOfPulses.write( out64bitUintDataValue, PredType::NATIVE_ULLONG );
			
            DataSet datasetUserMetaData = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_USER_META_DATA, strTypeAll, dataspaceStrAll);
			wStrdata = new const char*[numLinesStr];
			wStrdata[0] = spdFile->getUserMetaField().c_str();			
			datasetUserMetaData.write((void*)wStrdata, strTypeAll);
			datasetUserMetaData.close();
			delete[] wStrdata;
            
            DataSet datasetXMin = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_X_MIN, doubleDataTypeDisk, singleValueDataSpace );
			outDoubleDataValue[0] = spdFile->getXMin();
			datasetXMin.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
            
            DataSet datasetXMax = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_X_MAX, doubleDataTypeDisk, singleValueDataSpace );
			outDoubleDataValue[0] = spdFile->getXMax();
			datasetXMax.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
			
            DataSet datasetYMin = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_Y_MIN, doubleDataTypeDisk, singleValueDataSpace );
			outDoubleDataValue[0] = spdFile->getYMin();
			datasetYMin.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
			
            DataSet datasetYMax = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_Y_MAX, doubleDataTypeDisk, singleValueDataSpace );
			outDoubleDataValue[0] = spdFile->getYMax();
			datasetYMax.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
			
            DataSet datasetZMin = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_Z_MIN, doubleDataTypeDisk, singleValueDataSpace );
			outDoubleDataValue[0] = spdFile->getZMin();
			datasetZMin.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
			
            DataSet datasetZMax = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_Z_MAX, doubleDataTypeDisk, singleValueDataSpace );
			outDoubleDataValue[0] = spdFile->getZMax();
			datasetZMax.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
            
            DataSet datasetZenithMin = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_ZENITH_MIN, doubleDataTypeDisk, singleValueDataSpace );
			outDoubleDataValue[0] = spdFile->getZenithMin();
			datasetZenithMin.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
			
            DataSet datasetZenithMax = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_ZENITH_MAX, doubleDataTypeDisk, singleValueDataSpace );
			outDoubleDataValue[0] = spdFile->getZenithMax();
			datasetZenithMax.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
			
            DataSet datasetAzimuthMin = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_AZIMUTH_MIN, doubleDataTypeDisk, singleValueDataSpace );
			outDoubleDataValue[0] = spdFile->getAzimuthMin();
			datasetAzimuthMin.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
			
            DataSet datasetAzimuthMax = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_AZIMUTH_MAX, doubleDataTypeDisk, singleValueDataSpace );
			outDoubleDataValue[0] = spdFile->getAzimuthMax();
			datasetAzimuthMax.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
			
            DataSet datasetRangeMin = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_RANGE_MIN, doubleDataTypeDisk, singleValueDataSpace );
			outDoubleDataValue[0] = spdFile->getRangeMin();
			datasetRangeMin.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
			
            DataSet datasetRangeMax = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_RANGE_MAX, doubleDataTypeDisk, singleValueDataSpace );
			outDoubleDataValue[0] = spdFile->getRangeMax();
			datasetRangeMax.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
            
            DataSet datasetScanlineMin = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_SCANLINE_MIN, doubleDataTypeDisk, singleValueDataSpace );
			outDoubleDataValue[0] = spdFile->getScanlineMin();
			datasetScanlineMin.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
			
            DataSet datasetScanlineMax = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_SCANLINE_MAX, doubleDataTypeDisk, singleValueDataSpace );
			outDoubleDataValue[0] = spdFile->getScanlineMax();
			datasetScanlineMax.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
			
            DataSet datasetScanlineIdxMin = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_SCANLINE_IDX_MIN, doubleDataTypeDisk, singleValueDataSpace );
			outDoubleDataValue[0] = spdFile->getScanlineIdxMin();
			datasetScanlineIdxMin.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
			
            DataSet datasetScanlineIdxMax = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_SCANLINE_IDX_MAX, doubleDataTypeDisk, singleValueDataSpace );
			outDoubleDataValue[0] = spdFile->getScanlineIdxMax();
			datasetScanlineIdxMax.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
            
            DataSet datasetPulseRepFreq = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_PULSE_REPETITION_FREQ, floatDataTypeDisk, singleValueDataSpace );
			outFloatDataValue[0] = spdFile->getPulseRepetitionFreq();
			datasetPulseRepFreq.write( outFloatDataValue, PredType::NATIVE_FLOAT );
			
            DataSet datasetBeamDivergence = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_BEAM_DIVERGENCE, floatDataTypeDisk, singleValueDataSpace );
			outFloatDataValue[0] = spdFile->getBeamDivergence();
			datasetBeamDivergence.write( outFloatDataValue, PredType::NATIVE_FLOAT );
			
            DataSet datasetSensorHeight = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_SENSOR_HEIGHT, doubleDataTypeDisk, singleValueDataSpace );
			outDoubleDataValue[0] = spdFile->getSensorHeight();
			datasetSensorHeight.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
			
            DataSet datasetFootprint = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_FOOTPRINT, floatDataTypeDisk, singleValueDataSpace );
			outFloatDataValue[0] = spdFile->getFootprint();
			datasetFootprint.write( outFloatDataValue, PredType::NATIVE_FLOAT );
			
            DataSet datasetMaxScanAngle = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_MAX_SCAN_ANGLE, floatDataTypeDisk, singleValueDataSpace );
			outFloatDataValue[0] = spdFile->getMaxScanAngle();
			datasetMaxScanAngle.write( outFloatDataValue, PredType::NATIVE_FLOAT );
			
            DataSet datasetRGBDefined = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_RGB_DEFINED, int16bitDataTypeDisk, singleValueDataSpace);
			out16bitintDataValue[0] = spdFile->getRGBDefined();
			datasetRGBDefined.write( out16bitintDataValue, PredType::NATIVE_INT );
			
            DataSet datasetPulseBlockSize = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_PULSE_BLOCK_SIZE, uint16bitDataTypeDisk, singleValueDataSpace);
			out16bitUintDataValue[0] = spdFile->getPulseBlockSize();
			datasetPulseBlockSize.write( out16bitUintDataValue, PredType::NATIVE_UINT );
			
            DataSet datasetPointsBlockSize = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_POINT_BLOCK_SIZE, uint16bitDataTypeDisk, singleValueDataSpace);
			out16bitUintDataValue[0] = spdFile->getPointBlockSize();
			datasetPointsBlockSize.write( out16bitUintDataValue, PredType::NATIVE_UINT );
			
            DataSet datasetReceivedBlockSize = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_RECEIVED_BLOCK_SIZE, uint16bitDataTypeDisk, singleValueDataSpace);
			out16bitUintDataValue[0] = spdFile->getReceivedBlockSize();
			datasetReceivedBlockSize.write( out16bitUintDataValue, PredType::NATIVE_UINT );
			
            DataSet datasetTransmittedBlockSize = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_TRANSMITTED_BLOCK_SIZE, uint16bitDataTypeDisk, singleValueDataSpace);
			out16bitUintDataValue[0] = spdFile->getTransmittedBlockSize();
			datasetTransmittedBlockSize.write( out16bitUintDataValue, PredType::NATIVE_UINT );
            
            DataSet datasetWaveformBitRes = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_WAVEFORM_BIT_RES, uint16bitDataTypeDisk, singleValueDataSpace);
            out16bitUintDataValue[0] = spdFile->getWaveformBitRes();
			datasetWaveformBitRes.write( out16bitUintDataValue, PredType::NATIVE_UINT );
            
            DataSet datasetTemporalBinSpacing = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_TEMPORAL_BIN_SPACING, doubleDataTypeDisk, singleValueDataSpace);
			outDoubleDataValue[0] = spdFile->getTemporalBinSpacing();
			datasetTemporalBinSpacing.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
			
            DataSet datasetReturnNumsSynGen = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_RETURN_NUMBERS_SYN_GEN, int16bitDataTypeDisk, singleValueDataSpace);
			out16bitintDataValue[0] = spdFile->getReturnNumsSynGen();
			datasetReturnNumsSynGen.write( out16bitintDataValue, PredType::NATIVE_INT );
            
            DataSet datasetHeightDefined = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_HEIGHT_DEFINED, int16bitDataTypeDisk, singleValueDataSpace);
			out16bitintDataValue[0] = spdFile->getHeightDefined();
			datasetHeightDefined.write( out16bitintDataValue, PredType::NATIVE_INT );
			
            DataSet datasetSensorSpeed = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_SENSOR_SPEED, floatDataTypeDisk, singleValueDataSpace);
			outFloatDataValue[0] = spdFile->getSensorSpeed();
			datasetSensorSpeed.write( outFloatDataValue, PredType::NATIVE_FLOAT );
			
            DataSet datasetSensorScanRate = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_SENSOR_SCAN_RATE, floatDataTypeDisk, singleValueDataSpace);
			outFloatDataValue[0] = spdFile->getSensorScanRate();
			datasetSensorScanRate.write( outFloatDataValue, PredType::NATIVE_FLOAT );
			
            DataSet datasetPointDensity = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_POINT_DENSITY, floatDataTypeDisk, singleValueDataSpace);
			outFloatDataValue[0] = spdFile->getPointDensity();
			datasetPointDensity.write( outFloatDataValue, PredType::NATIVE_FLOAT );
			
            DataSet datasetPulseDensity = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_PULSE_DENSITY, floatDataTypeDisk, singleValueDataSpace);
			outFloatDataValue[0] = spdFile->getPulseDensity();
			datasetPulseDensity.write( outFloatDataValue, PredType::NATIVE_FLOAT );
			
            DataSet datasetPulseCrossTrackSpacing = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_PULSE_CROSS_TRACK_SPACING, floatDataTypeDisk, singleValueDataSpace);
			outFloatDataValue[0] = spdFile->getPulseCrossTrackSpacing();
			datasetPulseCrossTrackSpacing.write( outFloatDataValue, PredType::NATIVE_FLOAT );
			
            DataSet datasetPulseAlongTrackSpacing = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_PULSE_ALONG_TRACK_SPACING, floatDataTypeDisk, singleValueDataSpace);
			outFloatDataValue[0] = spdFile->getPulseAlongTrackSpacing();
			datasetPulseAlongTrackSpacing.write( outFloatDataValue, PredType::NATIVE_FLOAT );
			
            DataSet datasetOriginDefined = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_ORIGIN_DEFINED, int16bitDataTypeDisk, singleValueDataSpace);
			out16bitintDataValue[0] = spdFile->getOriginDefined();
			datasetOriginDefined.write( out16bitintDataValue, PredType::NATIVE_INT );
            
            DataSet datasetPulseAngularSpacingAzimuth = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_PULSE_ANGULAR_SPACING_AZIMUTH, floatDataTypeDisk, singleValueDataSpace);
			outFloatDataValue[0] = spdFile->getPulseAngularSpacingAzimuth();
			datasetPulseAngularSpacingAzimuth.write( outFloatDataValue, PredType::NATIVE_FLOAT );
			
            DataSet datasetPulseAngularSpacingZenith = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_PULSE_ANGULAR_SPACING_ZENITH, floatDataTypeDisk, singleValueDataSpace);
			outFloatDataValue[0] = spdFile->getPulseAngularSpacingZenith();
			datasetPulseAngularSpacingZenith.write( outFloatDataValue, PredType::NATIVE_FLOAT );
            
            DataSet datasetSensorApertureSize = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_SENSOR_APERTURE_SIZE, floatDataTypeDisk, singleValueDataSpace);
            outFloatDataValue[0] = spdFile->getSensorApertureSize();
			datasetSensorApertureSize.write( outFloatDataValue, PredType::NATIVE_FLOAT );
            
            DataSet datasetPulseEnergy = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_PULSE_ENERGY, floatDataTypeDisk, singleValueDataSpace);
            outFloatDataValue[0] = spdFile->getPulseEnergy();
			datasetPulseEnergy.write( outFloatDataValue, PredType::NATIVE_FLOAT );
            
            DataSet datasetFieldOfView = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_FIELD_OF_VIEW, floatDataTypeDisk, singleValueDataSpace);
            outFloatDataValue[0] = spdFile->getFieldOfView();
			datasetFieldOfView.write( outFloatDataValue, PredType::NATIVE_FLOAT );
            
            if(spdFile->getNumOfWavelengths() == 0)
            {
                spdFile->setNumOfWavelengths(1);
                spdFile->getWavelengths()->push_back(0);
                spdFile->getBandwidths()->push_back(0);
            }
            
            DataSet datasetNumOfWavelengths = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_NUM_OF_WAVELENGTHS, uint16bitDataTypeDisk, singleValueDataSpace );
            out16bitUintDataValue[0] = spdFile->getNumOfWavelengths();
			datasetNumOfWavelengths.write( out16bitUintDataValue, PredType::NATIVE_UINT );
            
            hsize_t dimsWavelengthsValue[1];
			dimsWavelengthsValue[0] = spdFile->getNumOfWavelengths();
			DataSpace wavelengthsDataSpace(1, dimsWavelengthsValue);
            
            DataSet datasetWavelengths = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_WAVELENGTHS, floatDataTypeDisk, wavelengthsDataSpace );
            datasetWavelengths.write( &spdFile->getWavelengths()[0], PredType::NATIVE_FLOAT );
                        
			DataSet datasetBandwidths = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_BANDWIDTHS, floatDataTypeDisk, wavelengthsDataSpace );
            datasetBandwidths.write( &spdFile->getBandwidths()[0], PredType::NATIVE_FLOAT );
                        
            if(spdFile->getFileType() != SPD_UPD_TYPE)
            {
                DataSet datasetBinSize = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_BIN_SIZE, floatDataTypeDisk, singleValueDataSpace );
                outFloatDataValue[0] = spdFile->getBinSize();
                datasetBinSize.write( outFloatDataValue, PredType::NATIVE_FLOAT );
			
                DataSet datasetNumberBinsX = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_NUMBER_BINS_X, uint32bitDataType, singleValueDataSpace);			
                out32bitUintDataValue[0] = spdFile->getNumberBinsX();
                datasetNumberBinsX.write( out32bitUintDataValue, PredType::NATIVE_ULONG );
			
                DataSet datasetNumberBinsY = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_NUMBER_BINS_Y, uint32bitDataType, singleValueDataSpace);
                out32bitUintDataValue[0] = spdFile->getNumberBinsY();
                datasetNumberBinsY.write( out32bitUintDataValue, PredType::NATIVE_ULONG );
            }
			
		}
		catch( FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataTypeIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch(SPDIOException &e)
		{
			throw e;
		}
    }
    
    void SPDFileWriter::updateHeaderInfo(H5File *spdOutH5File, SPDFile *spdFile) throw(SPDIOException)
	{
		float outFloatDataValue[1];
		double outDoubleDataValue[1];
		int out16bitintDataValue[1];
		unsigned int out16bitUintDataValue[1];
        unsigned long out32bitUintDataValue[1];
		unsigned long long out64bitUintDataValue[1];
		
		unsigned int numLinesStr = 1;
		const char **wStrdata = NULL;
		
		hsize_t	dims1Str[1];
		dims1Str[0] = numLinesStr;
		
		StrType strTypeAll(0, H5T_VARIABLE);
		
		const H5std_string spdFilePath( spdFile->getFilePath() );
		try 
		{
			Exception::dontPrint();
			
			// Create File..
			H5File *spdH5File = new H5File( spdFilePath, H5F_ACC_RDWR );
			
			if((H5T_STRING!=H5Tget_class(strTypeAll.getId())) || (!H5Tis_variable_str(strTypeAll.getId())))
			{
				throw SPDIOException("The string data type defined is not variable.");
			}
            
            
            try 
            {
                DataSet datasetMajorVersion = spdH5File->openDataSet( SPDFILE_DATASETNAME_MAJOR_VERSION );
                out16bitUintDataValue[0] = spdFile->getMajorSPDVersion();
                datasetMajorVersion.write( out16bitUintDataValue, PredType::NATIVE_UINT );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("The SPD major version header value was not provided.");
            }
            
            try 
            {
                DataSet datasetMinorVersion = spdH5File->openDataSet( SPDFILE_DATASETNAME_MINOR_VERSION );
                out16bitUintDataValue[0] = spdFile->getMinorSPDVersion();
                datasetMinorVersion.write( out16bitUintDataValue, PredType::NATIVE_UINT );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("The SPD minor version header value was not provided.");
            }
            
            try 
            {
                DataSet datasetPointVersion = spdH5File->openDataSet( SPDFILE_DATASETNAME_POINT_VERSION );
                out16bitUintDataValue[0] = spdFile->getPointVersion();
                datasetPointVersion.write( out16bitUintDataValue, PredType::NATIVE_UINT );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("The SPD point version header value was not provided.");
            }
            
            try 
            {
                DataSet datasetPulseVersion = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_VERSION );
                out16bitUintDataValue[0] = spdFile->getPulseVersion();
                datasetPulseVersion.write( out16bitUintDataValue, PredType::NATIVE_UINT );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("The SPD pulse version header value was not provided.");
            }
            
            try 
            {
                DataSet datasetSpatialReference = spdH5File->openDataSet( SPDFILE_DATASETNAME_SPATIAL_REFERENCE );
                wStrdata = new const char*[numLinesStr];
                wStrdata[0] = spdFile->getSpatialReference().c_str();			
                datasetSpatialReference.write((void*)wStrdata, strTypeAll);
                datasetSpatialReference.close();
                delete[] wStrdata;
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Spatial reference header value is not represent.");
            }
            
            try 
            {
                DataSet datasetFileType = spdH5File->openDataSet( SPDFILE_DATASETNAME_FILE_TYPE );
                out16bitUintDataValue[0] = spdFile->getFileType();
                datasetFileType.write( out16bitUintDataValue, PredType::NATIVE_UINT );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("File type header value not present.");
            }
            
            try 
            {
                DataSet datasetIndexType = spdH5File->openDataSet( SPDFILE_DATASETNAME_INDEX_TYPE );
                out16bitUintDataValue[0] = spdFile->getIndexType();
                datasetIndexType.write( out16bitUintDataValue, PredType::NATIVE_UINT );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Index type header value not provided.");
            }
            
            try 
            {
                DataSet datasetDiscreteDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_DISCRETE_PT_DEFINED );
                out16bitintDataValue[0] = spdFile->getDiscretePtDefined();
                datasetDiscreteDefined.write( out16bitintDataValue, PredType::NATIVE_INT );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Discrete Point Defined header value not provided.");
            }
            
            try 
            {
                DataSet datasetDecomposedDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_DECOMPOSED_PT_DEFINED );
                out16bitintDataValue[0] = spdFile->getDecomposedPtDefined();
                datasetDecomposedDefined.write( out16bitintDataValue, PredType::NATIVE_INT );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Decomposed Point Defined header value not provided.");
            }
            
            try 
            {
                DataSet datasetTransWaveformDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_TRANS_WAVEFORM_DEFINED );
                out16bitintDataValue[0] = spdFile->getTransWaveformDefined();
                datasetTransWaveformDefined.write( out16bitintDataValue, PredType::NATIVE_INT );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Transmitted Waveform Defined header value not provided.");
            }
            
            try 
            {
                DataSet datasetReceiveWaveformDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_RECEIVE_WAVEFORM_DEFINED );
                out16bitintDataValue[0] = spdFile->getReceiveWaveformDefined();
                datasetReceiveWaveformDefined.write( out16bitintDataValue, PredType::NATIVE_INT );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Received Waveform Defined header value not provided.");
            }
            
            try 
            {
                DataSet datasetGeneratingSoftware = spdH5File->openDataSet( SPDFILE_DATASETNAME_GENERATING_SOFTWARE );
                wStrdata = new const char*[numLinesStr];
                wStrdata[0] = spdFile->getGeneratingSoftware().c_str();			
                datasetGeneratingSoftware.write((void*)wStrdata, strTypeAll);
                datasetGeneratingSoftware.close();
                delete[] wStrdata;
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Generating software header value not provided.");
            }
            
            try 
            {
                DataSet datasetSystemIdentifier = spdH5File->openDataSet( SPDFILE_DATASETNAME_SYSTEM_IDENTIFIER );
                wStrdata = new const char*[numLinesStr];
                wStrdata[0] = spdFile->getSystemIdentifier().c_str();			
                datasetSystemIdentifier.write((void*)wStrdata, strTypeAll);
                datasetSystemIdentifier.close();
                delete[] wStrdata;
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("System identifier header value not provided.");
            }
            
            try 
            {
                DataSet datasetFileSignature = spdH5File->openDataSet( SPDFILE_DATASETNAME_FILE_SIGNATURE );
                wStrdata = new const char*[numLinesStr];
                wStrdata[0] = spdFile->getFileSignature().c_str();			
                datasetFileSignature.write((void*)wStrdata, strTypeAll);
                datasetFileSignature.close();
                delete[] wStrdata;
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("File signature header value not provided.");
            }
            
            try 
            {
                DataSet datasetYearOfCreation = spdH5File->openDataSet( SPDFILE_DATASETNAME_YEAR_OF_CREATION );
                DataSet datasetMonthOfCreation = spdH5File->openDataSet( SPDFILE_DATASETNAME_MONTH_OF_CREATION );
                DataSet datasetDayOfCreation = spdH5File->openDataSet( SPDFILE_DATASETNAME_DAY_OF_CREATION );
                DataSet datasetHourOfCreation = spdH5File->openDataSet( SPDFILE_DATASETNAME_HOUR_OF_CREATION );
                DataSet datasetMinuteOfCreation = spdH5File->openDataSet( SPDFILE_DATASETNAME_MINUTE_OF_CREATION );
                DataSet datasetSecondOfCreation = spdH5File->openDataSet( SPDFILE_DATASETNAME_SECOND_OF_CREATION );
                
                out16bitUintDataValue[0] = spdFile->getYearOfCreation();
                datasetYearOfCreation.write( out16bitUintDataValue, PredType::NATIVE_UINT );
                
                out16bitUintDataValue[0] = spdFile->getMonthOfCreation();
                datasetMonthOfCreation.write( out16bitUintDataValue, PredType::NATIVE_UINT );
                
                out16bitUintDataValue[0] = spdFile->getDayOfCreation();
                datasetDayOfCreation.write( out16bitUintDataValue, PredType::NATIVE_UINT );
                
                out16bitUintDataValue[0] = spdFile->getHourOfCreation();
                datasetHourOfCreation.write( out16bitUintDataValue, PredType::NATIVE_UINT );
                
                out16bitUintDataValue[0] = spdFile->getMinuteOfCreation();
                datasetMinuteOfCreation.write( out16bitUintDataValue, PredType::NATIVE_UINT );
                
                out16bitUintDataValue[0] = spdFile->getSecondOfCreation();
                datasetSecondOfCreation.write( out16bitUintDataValue, PredType::NATIVE_UINT );;
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Date of file creation header values not provided.");
            }
            
            try 
            {
                DataSet datasetYearOfCapture = spdH5File->openDataSet( SPDFILE_DATASETNAME_YEAR_OF_CAPTURE );
                DataSet datasetMonthOfCapture = spdH5File->openDataSet( SPDFILE_DATASETNAME_MONTH_OF_CAPTURE );
                DataSet datasetDayOfCapture = spdH5File->openDataSet( SPDFILE_DATASETNAME_DAY_OF_CAPTURE );
                DataSet datasetHourOfCapture = spdH5File->openDataSet( SPDFILE_DATASETNAME_HOUR_OF_CAPTURE );
                DataSet datasetMinuteOfCapture = spdH5File->openDataSet( SPDFILE_DATASETNAME_MINUTE_OF_CAPTURE );
                DataSet datasetSecondOfCapture = spdH5File->openDataSet( SPDFILE_DATASETNAME_SECOND_OF_CAPTURE );
                
                out16bitUintDataValue[0] = spdFile->getYearOfCapture();
                datasetYearOfCapture.write( out16bitUintDataValue, PredType::NATIVE_UINT );
                
                out16bitUintDataValue[0] = spdFile->getMonthOfCapture();
                datasetMonthOfCapture.write( out16bitUintDataValue, PredType::NATIVE_UINT );
                
                out16bitUintDataValue[0] = spdFile->getDayOfCapture();
                datasetDayOfCapture.write( out16bitUintDataValue, PredType::NATIVE_UINT );
                
                out16bitUintDataValue[0] = spdFile->getHourOfCapture();
                datasetHourOfCapture.write( out16bitUintDataValue, PredType::NATIVE_UINT );
                
                out16bitUintDataValue[0] = spdFile->getMinuteOfCapture();
                datasetMinuteOfCapture.write( out16bitUintDataValue, PredType::NATIVE_UINT );
                
                out16bitUintDataValue[0] = spdFile->getSecondOfCapture();
                datasetSecondOfCapture.write( out16bitUintDataValue, PredType::NATIVE_UINT );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Date/Time of capture header values not provided.");
            }
            
            try 
            {
                DataSet datasetNumberOfPoints = spdH5File->openDataSet( SPDFILE_DATASETNAME_NUMBER_OF_POINTS );
                out64bitUintDataValue[0] = spdFile->getNumberOfPoints();
                datasetNumberOfPoints.write( out64bitUintDataValue, PredType::NATIVE_ULLONG );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Number of points header value not provided.");
            }
            
            try 
            {
                DataSet datasetNumberOfPulses = spdH5File->openDataSet( SPDFILE_DATASETNAME_NUMBER_OF_PULSES );
                out64bitUintDataValue[0] = spdFile->getNumberOfPulses();
                datasetNumberOfPulses.write( out64bitUintDataValue, PredType::NATIVE_ULLONG );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Number of pulses header value not provided.");
            }
            
            try 
            {
                DataSet datasetUserMetaData = spdH5File->openDataSet( SPDFILE_DATASETNAME_USER_META_DATA );
                wStrdata = new const char*[numLinesStr];
                wStrdata[0] = spdFile->getUserMetaField().c_str();			
                datasetUserMetaData.write((void*)wStrdata, strTypeAll);
                datasetUserMetaData.close();
                delete[] wStrdata;
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("User metadata header value not provided.");
            }
            
            try 
            {
                DataSet datasetXMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_X_MIN );
                DataSet datasetXMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_X_MAX );
                DataSet datasetYMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_Y_MIN );
                DataSet datasetYMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_Y_MAX );
                DataSet datasetZMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_Z_MIN );
                DataSet datasetZMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_Z_MAX );
                
                outDoubleDataValue[0] = spdFile->getXMin();
                datasetXMin.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
                
                outDoubleDataValue[0] = spdFile->getXMax();
                datasetXMax.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
                
                outDoubleDataValue[0] = spdFile->getYMin();
                datasetYMin.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
                
                outDoubleDataValue[0] = spdFile->getYMax();
                datasetYMax.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
                
                outDoubleDataValue[0] = spdFile->getZMin();
                datasetZMin.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
                
                outDoubleDataValue[0] = spdFile->getZMax();
                datasetZMax.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Dataset bounding volume header values not provided.");
            }
            
            try 
            {
                DataSet datasetZenithMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_ZENITH_MIN );
                DataSet datasetZenithMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_ZENITH_MAX );
                DataSet datasetAzimuthMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_AZIMUTH_MIN );
                DataSet datasetAzimuthMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_AZIMUTH_MAX );
                DataSet datasetRangeMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_RANGE_MIN );
                DataSet datasetRangeMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_RANGE_MAX );
                
                outDoubleDataValue[0] = spdFile->getZenithMin();
                datasetZenithMin.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
                
                outDoubleDataValue[0] = spdFile->getZenithMax();
                datasetZenithMax.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );;
                
                outDoubleDataValue[0] = spdFile->getAzimuthMin();
                datasetAzimuthMin.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
                
                outDoubleDataValue[0] = spdFile->getAzimuthMax();
                datasetAzimuthMax.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
                
                outDoubleDataValue[0] = spdFile->getRangeMin();
                datasetRangeMin.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
                
                outDoubleDataValue[0] = spdFile->getRangeMax();
                datasetRangeMax.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Bounding spherical volume header values not provided.");
            }
            
            try 
            {
                DataSet datasetScanlineMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_SCANLINE_MIN );
                DataSet datasetScanlineMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_SCANLINE_MAX );
                DataSet datasetScanlineIdxMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_SCANLINE_IDX_MIN );
                DataSet datasetScanlineIdxMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_SCANLINE_IDX_MAX );
                
                outDoubleDataValue[0] = spdFile->getScanlineMin();
                datasetScanlineMin.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
                
                outDoubleDataValue[0] = spdFile->getScanlineMax();
                datasetScanlineMax.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );;
                
                outDoubleDataValue[0] = spdFile->getScanlineIdxMin();
                datasetScanlineIdxMin.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
                
                outDoubleDataValue[0] = spdFile->getScanlineIdxMax();
                datasetScanlineIdxMax.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
            } 
            catch ( Exception &e) 
            {
                FloatType doubleDataTypeDisk( PredType::IEEE_F64LE );
                hsize_t dimsValue[1];
                dimsValue[0] = 1;
                DataSpace singleValueDataSpace(1, dimsValue);
                
                DataSet datasetScanlineMin = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_SCANLINE_MIN, doubleDataTypeDisk, singleValueDataSpace );
                outDoubleDataValue[0] = spdFile->getScanlineMin();
                datasetScanlineMin.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
                
                DataSet datasetScanlineMax = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_SCANLINE_MAX, doubleDataTypeDisk, singleValueDataSpace );
                outDoubleDataValue[0] = spdFile->getScanlineMax();
                datasetScanlineMax.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
                
                DataSet datasetScanlineIdxMin = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_SCANLINE_IDX_MIN, doubleDataTypeDisk, singleValueDataSpace );
                outDoubleDataValue[0] = spdFile->getScanlineIdxMin();
                datasetScanlineIdxMin.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
                
                DataSet datasetScanlineIdxMax = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_SCANLINE_IDX_MAX, doubleDataTypeDisk, singleValueDataSpace );
                outDoubleDataValue[0] = spdFile->getScanlineIdxMax();
                datasetScanlineIdxMax.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
            }
            
            if(spdFile->getFileType() != SPD_UPD_TYPE)
            {
                try 
                {
                    DataSet datasetBinSize = spdH5File->openDataSet( SPDFILE_DATASETNAME_BIN_SIZE );
                    outFloatDataValue[0] = spdFile->getBinSize();
                    datasetBinSize.write( outFloatDataValue, PredType::NATIVE_FLOAT );
                } 
                catch ( Exception &e) 
                {
                    throw SPDIOException("Bin size header value not provided.");
                }
                
                try 
                {
                    DataSet datasetNumberBinsX = spdH5File->openDataSet( SPDFILE_DATASETNAME_NUMBER_BINS_X );
                    out32bitUintDataValue[0] = spdFile->getNumberBinsX();
                    datasetNumberBinsX.write( out32bitUintDataValue, PredType::NATIVE_ULONG );
                } 
                catch ( Exception &e) 
                {
                    throw SPDIOException("Number of X bins header value not provided.");
                }
                
                try 
                {
                    DataSet datasetNumberBinsY = spdH5File->openDataSet( SPDFILE_DATASETNAME_NUMBER_BINS_Y );
                    out32bitUintDataValue[0] = spdFile->getNumberBinsY();
                    datasetNumberBinsY.write( out32bitUintDataValue, PredType::NATIVE_ULONG );
                } 
                catch ( Exception &e) 
                {
                    throw SPDIOException("Number of Y bins header value not provided.");
                }
            }
            
            try 
            {
                DataSet datasetPulseRepFreq = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_REPETITION_FREQ );
                outFloatDataValue[0] = spdFile->getPulseRepetitionFreq();
                datasetPulseRepFreq.write( outFloatDataValue, PredType::NATIVE_FLOAT );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Pulse repetition frequency header value not provided.");
            }
            
            try 
            {
                DataSet datasetBeamDivergence = spdH5File->openDataSet( SPDFILE_DATASETNAME_BEAM_DIVERGENCE );
                outFloatDataValue[0] = spdFile->getBeamDivergence();
                datasetBeamDivergence.write( outFloatDataValue, PredType::NATIVE_FLOAT );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Beam divergence header value not provided.");
            }
            
            try 
            {
                DataSet datasetSensorHeight = spdH5File->openDataSet( SPDFILE_DATASETNAME_SENSOR_HEIGHT );
                outDoubleDataValue[0] = spdFile->getSensorHeight();
                datasetSensorHeight.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Sensor height header value not provided.");
            }
            
            try 
            {
                DataSet datasetFootprint = spdH5File->openDataSet( SPDFILE_DATASETNAME_FOOTPRINT );
                outFloatDataValue[0] = spdFile->getFootprint();
                datasetFootprint.write( outFloatDataValue, PredType::NATIVE_FLOAT );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Footprint header value not provided.");
            }
            
            try 
            {
                DataSet datasetMaxScanAngle = spdH5File->openDataSet( SPDFILE_DATASETNAME_MAX_SCAN_ANGLE );
                outFloatDataValue[0] = spdFile->getMaxScanAngle();
                datasetMaxScanAngle.write( outFloatDataValue, PredType::NATIVE_FLOAT );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Max scan angle header value not provided.");
            }
            
            try 
            {
                DataSet datasetRGBDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_RGB_DEFINED );
                out16bitintDataValue[0] = spdFile->getRGBDefined();
                datasetRGBDefined.write( out16bitintDataValue, PredType::NATIVE_INT );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("RGB defined header value not provided.");
            }
            
            try 
            {
                DataSet datasetPulseBlockSize = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_BLOCK_SIZE );
                out16bitUintDataValue[0] = spdFile->getPulseBlockSize();
                datasetPulseBlockSize.write( out16bitUintDataValue, PredType::NATIVE_UINT );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Pulse block size header value not provided.");
            }
            
            try 
            {
                DataSet datasetPointsBlockSize = spdH5File->openDataSet( SPDFILE_DATASETNAME_POINT_BLOCK_SIZE );
                out16bitUintDataValue[0] = spdFile->getPointBlockSize();
                datasetPointsBlockSize.write( out16bitUintDataValue, PredType::NATIVE_UINT );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Point block size header value not provided.");
            }
            
            try 
            {
                DataSet datasetReceivedBlockSize = spdH5File->openDataSet( SPDFILE_DATASETNAME_RECEIVED_BLOCK_SIZE );
                out16bitUintDataValue[0] = spdFile->getReceivedBlockSize();
                datasetReceivedBlockSize.write( out16bitUintDataValue, PredType::NATIVE_UINT );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Received waveform block size header value not provided.");
            }
            
            try 
            {
                DataSet datasetTransmittedBlockSize = spdH5File->openDataSet( SPDFILE_DATASETNAME_TRANSMITTED_BLOCK_SIZE );
                out16bitUintDataValue[0] = spdFile->getTransmittedBlockSize();
                datasetTransmittedBlockSize.write( out16bitUintDataValue, PredType::NATIVE_UINT );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Transmitted waveform block size header value not provided.");
            }
            
            try 
            {
                DataSet datasetWaveformBitRes = spdH5File->openDataSet( SPDFILE_DATASETNAME_WAVEFORM_BIT_RES );
                out16bitUintDataValue[0] = spdFile->getWaveformBitRes();
                datasetWaveformBitRes.write( out16bitUintDataValue, PredType::NATIVE_UINT );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Waveform bit resolution header value not provided.");
            }
            
            try 
            {
                DataSet datasetTemporalBinSpacing = spdH5File->openDataSet( SPDFILE_DATASETNAME_TEMPORAL_BIN_SPACING );
                outDoubleDataValue[0] = spdFile->getTemporalBinSpacing();
                datasetTemporalBinSpacing.write( outDoubleDataValue, PredType::NATIVE_DOUBLE );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Temporal bin spacing header value not provided.");
            }
            
            try 
            {
                DataSet datasetReturnNumsSynGen = spdH5File->openDataSet( SPDFILE_DATASETNAME_RETURN_NUMBERS_SYN_GEN );
                out16bitintDataValue[0] = spdFile->getReturnNumsSynGen();
                datasetReturnNumsSynGen.write( out16bitintDataValue, PredType::NATIVE_INT );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Return number synthetically generated header value not provided.");
            }
            
            try 
            {
                DataSet datasetHeightDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_HEIGHT_DEFINED );
                out16bitintDataValue[0] = spdFile->getHeightDefined();
                datasetHeightDefined.write( out16bitintDataValue, PredType::NATIVE_INT );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Height fields defined header value not provided.");
            }
            
            try 
            {
                DataSet datasetSensorSpeed = spdH5File->openDataSet( SPDFILE_DATASETNAME_SENSOR_SPEED );
                outFloatDataValue[0] = spdFile->getSensorSpeed();
                datasetSensorSpeed.write( outFloatDataValue, PredType::NATIVE_FLOAT );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Sensor speed header value not provided.");
            }
            
            try 
            {
                DataSet datasetSensorScanRate = spdH5File->openDataSet( SPDFILE_DATASETNAME_SENSOR_SCAN_RATE );
                outFloatDataValue[0] = spdFile->getSensorScanRate();
                datasetSensorScanRate.write( outFloatDataValue, PredType::NATIVE_FLOAT );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Sensor Scan Rate header value not provided.");
            }
            
            try 
            {
                DataSet datasetPointDensity = spdH5File->openDataSet( SPDFILE_DATASETNAME_POINT_DENSITY );
                outFloatDataValue[0] = spdFile->getPointDensity();
                datasetPointDensity.write( outFloatDataValue, PredType::NATIVE_FLOAT );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Point density header value not provided.");
            }
            
            try 
            {
                DataSet datasetPulseDensity = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_DENSITY );
                outFloatDataValue[0] = spdFile->getPulseDensity();
                datasetPulseDensity.write( outFloatDataValue, PredType::NATIVE_FLOAT );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Pulse density header value not provided.");
            }
            
            try 
            {
                DataSet datasetPulseCrossTrackSpacing = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_CROSS_TRACK_SPACING );
                outFloatDataValue[0] = spdFile->getPulseCrossTrackSpacing();
                datasetPulseCrossTrackSpacing.write( outFloatDataValue, PredType::NATIVE_FLOAT );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Cross track spacing header value not provided.");
            }
            
            try 
            {
                DataSet datasetPulseAlongTrackSpacing = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_ALONG_TRACK_SPACING );
                outFloatDataValue[0] = spdFile->getPulseAlongTrackSpacing();
                datasetPulseAlongTrackSpacing.write( outFloatDataValue, PredType::NATIVE_FLOAT );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Along track spacing header value not provided.");
            }
            
            try 
            {
                DataSet datasetOriginDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_ORIGIN_DEFINED );
                out16bitintDataValue[0] = spdFile->getOriginDefined();
                datasetOriginDefined.write( out16bitintDataValue, PredType::NATIVE_INT );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Origin defined header value not provided.");
            }
            
            try 
            {
                DataSet datasetPulseAngularSpacingAzimuth = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_ANGULAR_SPACING_AZIMUTH );
                outFloatDataValue[0] = spdFile->getPulseAngularSpacingAzimuth();
                datasetPulseAngularSpacingAzimuth.write( outFloatDataValue, PredType::NATIVE_FLOAT );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Angular azimuth spacing header value not provided.");
            }
            
            try 
            {
                DataSet datasetPulseAngularSpacingZenith = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_ANGULAR_SPACING_ZENITH );
                outFloatDataValue[0] = spdFile->getPulseAngularSpacingZenith();
                datasetPulseAngularSpacingZenith.write( outFloatDataValue, PredType::NATIVE_FLOAT );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Angular Zenith spacing header value not provided.");
            }
            
            try 
            {
                DataSet datasetPulseIndexMethod = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_INDEX_METHOD );
                out16bitUintDataValue[0] = spdFile->getIndexType();
                datasetPulseIndexMethod.write( out16bitUintDataValue, PredType::NATIVE_UINT );
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Method of indexing header value not provided.");
            }
            
            try 
            {
                DataSet datasetSensorApertureSize = spdH5File->openDataSet( SPDFILE_DATASETNAME_SENSOR_APERTURE_SIZE );
                outFloatDataValue[0] = spdFile->getSensorApertureSize();
                datasetSensorApertureSize.write( outFloatDataValue, PredType::NATIVE_FLOAT );
            } 
            catch ( Exception &e) 
            {
                float outFloatDataValue[1];
                FloatType floatDataTypeDisk( PredType::IEEE_F32LE );
                hsize_t dimsValue[1];
                dimsValue[0] = 1;
                DataSpace singleValueDataSpace(1, dimsValue);
                DataSet datasetSensorApertureSize = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_SENSOR_APERTURE_SIZE, floatDataTypeDisk, singleValueDataSpace);
                outFloatDataValue[0] = spdFile->getSensorApertureSize();
                datasetSensorApertureSize.write( outFloatDataValue, PredType::NATIVE_FLOAT );
            }
            
            try 
            {
                DataSet datasetPulseEnergy = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_ENERGY );
                outFloatDataValue[0] = spdFile->getPulseEnergy();
                datasetPulseEnergy.write( outFloatDataValue, PredType::NATIVE_FLOAT );
            } 
            catch ( Exception &e) 
            {
                float outFloatDataValue[1];
                FloatType floatDataTypeDisk( PredType::IEEE_F32LE );
                hsize_t dimsValue[1];
                dimsValue[0] = 1;
                DataSpace singleValueDataSpace(1, dimsValue);
                DataSet datasetPulseEnergy = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_PULSE_ENERGY, floatDataTypeDisk, singleValueDataSpace);
                outFloatDataValue[0] = spdFile->getPulseEnergy();
                datasetPulseEnergy.write( outFloatDataValue, PredType::NATIVE_FLOAT );
            }
            
            try 
            {
                DataSet datasetFieldOfView = spdH5File->openDataSet( SPDFILE_DATASETNAME_FIELD_OF_VIEW );
                outFloatDataValue[0] = spdFile->getFieldOfView();
                datasetFieldOfView.write( outFloatDataValue, PredType::NATIVE_FLOAT );
            } 
            catch ( Exception &e) 
            {
                float outFloatDataValue[1];
                FloatType floatDataTypeDisk( PredType::IEEE_F32LE );
                hsize_t dimsValue[1];
                dimsValue[0] = 1;
                DataSpace singleValueDataSpace(1, dimsValue);
                DataSet datasetFieldOfView = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_FIELD_OF_VIEW, floatDataTypeDisk, singleValueDataSpace);
                outFloatDataValue[0] = spdFile->getFieldOfView();
                datasetFieldOfView.write( outFloatDataValue, PredType::NATIVE_FLOAT );
            }
            
            try 
            {
                DataSet datasetNumOfWavelengths = spdH5File->openDataSet( SPDFILE_DATASETNAME_NUM_OF_WAVELENGTHS );
                out16bitUintDataValue[0] = spdFile->getNumOfWavelengths();
                datasetNumOfWavelengths.write( out16bitUintDataValue, PredType::NATIVE_UINT );
                
                if(spdFile->getNumOfWavelengths() > 0)
                {
                    hsize_t dimsWavelengthsValue[1];
                    dimsWavelengthsValue[0] = spdFile->getNumOfWavelengths();
                    DataSpace wavelengthsDataSpace(1, dimsWavelengthsValue);
                    
                    DataSet datasetWavelengths = spdH5File->openDataSet( SPDFILE_DATASETNAME_WAVELENGTHS );
                    datasetWavelengths.write( &spdFile->getWavelengths()[0], PredType::NATIVE_FLOAT );
                    
                    DataSet datasetBandwidths = spdH5File->openDataSet( SPDFILE_DATASETNAME_BANDWIDTHS );
                    datasetBandwidths.write( &spdFile->getBandwidths()[0], PredType::NATIVE_FLOAT );
                }
                else
                {
                    vector<float> wavelengths;
                    spdFile->setWavelengths(wavelengths);
                    vector<float> bandwidths;
                    spdFile->setBandwidths(bandwidths);
                }
                
            } 
            catch ( Exception &e) 
            {
                DataSet datasetWavelength = spdH5File->openDataSet( SPDFILE_DATASETNAME_WAVELENGTH );
                if(spdFile->getNumOfWavelengths() > 0)
                {
                    outFloatDataValue[0] = spdFile->getWavelengths()->front();
                    datasetWavelength.write( outFloatDataValue, PredType::NATIVE_FLOAT );
                }
                else
                {
                    outFloatDataValue[0] = 0;
                    datasetWavelength.write( outFloatDataValue, PredType::NATIVE_FLOAT );
                }
                
                IntType uint16bitDataTypeDisk( PredType::STD_U16LE );
                hsize_t dimsValue[1];
                dimsValue[0] = 1;
                DataSpace singleValueDataSpace(1, dimsValue);
                DataSet datasetNumOfWavelengths = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_NUM_OF_WAVELENGTHS, uint16bitDataTypeDisk, singleValueDataSpace );
                out16bitUintDataValue[0] = spdFile->getNumOfWavelengths();
                datasetNumOfWavelengths.write( out16bitUintDataValue, PredType::NATIVE_UINT );
                
                hsize_t dimsWavelengthsValue[1];
                dimsWavelengthsValue[0] = spdFile->getNumOfWavelengths();
                DataSpace wavelengthsDataSpace(1, dimsWavelengthsValue);
                FloatType floatDataTypeDisk( PredType::IEEE_F32LE );
                
                DataSet datasetWavelengths = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_WAVELENGTHS, floatDataTypeDisk, wavelengthsDataSpace );
                datasetWavelengths.write( &spdFile->getWavelengths()[0], PredType::NATIVE_FLOAT );
                DataSet datasetBandwidths = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_BANDWIDTHS, floatDataTypeDisk, wavelengthsDataSpace );
                datasetBandwidths.write( &spdFile->getBandwidths()[0], PredType::NATIVE_FLOAT );
            }
			spdH5File->flush(H5F_SCOPE_GLOBAL);
			spdH5File->close();
			delete spdH5File;
			
		}
		catch( FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataTypeIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
        catch( SPDIOException &e )
		{
			throw e;
		}
	}
    
    
    
    
    
    
    
	SPDSeqFileWriter::SPDSeqFileWriter() : SPDDataExporter("SPD-SEQ"), spdOutH5File(NULL), pulsesDataset(NULL), spdPulseDataType(NULL), pointsDataset(NULL), spdPointDataType(NULL), datasetPlsPerBin(NULL), datasetBinsOffset(NULL), receivedDataset(NULL), transmittedDataset(NULL), datasetQuicklook(NULL), numPulses(0), numPts(0), numTransVals(0), numReceiveVals(0), firstColumn(true), nextCol(0), nextRow(0), numCols(0), numRows(0)
	{
		
	}
	
	SPDSeqFileWriter::SPDSeqFileWriter(const SPDDataExporter &dataExporter) throw(SPDException) : SPDDataExporter(dataExporter), spdOutH5File(NULL), pulsesDataset(NULL), spdPulseDataType(NULL), pointsDataset(NULL), spdPointDataType(NULL), datasetPlsPerBin(NULL), datasetBinsOffset(NULL), receivedDataset(NULL), transmittedDataset(NULL), datasetQuicklook(NULL), numPulses(0), numPts(0), numTransVals(0), numReceiveVals(0), firstColumn(true), nextCol(0), nextRow(0), numCols(0), numRows(0)
	{
		if(fileOpened)
		{
			throw SPDException("Cannot make a copy of a file exporter when a file is open.");
		}
	}
	
	SPDSeqFileWriter& SPDSeqFileWriter::operator=(const SPDSeqFileWriter& dataExporter) throw(SPDException)
	{
		if(fileOpened)
		{
			throw SPDException("Cannot make a copy of a file exporter when a file is open.");
		}
		
		this->spdFile = dataExporter.spdFile;
		this->outputFile = dataExporter.outputFile;
		return *this;
	}
    
    SPDDataExporter* SPDSeqFileWriter::getInstance()
    {
        return new SPDSeqFileWriter();
    }
	
	bool SPDSeqFileWriter::open(SPDFile *spdFile, string outputFile) throw(SPDIOException)
	{
		SPDPulseUtils pulseUtils;
		SPDPointUtils pointUtils;
		this->spdFile = spdFile;
		this->outputFile = outputFile;
		
		const H5std_string spdFilePath( outputFile );
		
		try 
		{
			Exception::dontPrint();
			
			// Create File..
			spdOutH5File = new H5File( spdFilePath, H5F_ACC_TRUNC );
			
			// Create Groups..
			spdOutH5File->createGroup( GROUPNAME_HEADER );
			spdOutH5File->createGroup( GROUPNAME_INDEX );
			spdOutH5File->createGroup( GROUPNAME_QUICKLOOK );
			spdOutH5File->createGroup( GROUPNAME_DATA );
			
			this->numCols = spdFile->getNumberBinsX();
			this->numRows = spdFile->getNumberBinsY();
			
			// Create DataType, DataSpace and Dataset for Pulses
			hsize_t initDimsPulseDS[1];
			initDimsPulseDS[0] = 0;
			hsize_t maxDimsPulseDS[1];
			maxDimsPulseDS[0] = H5S_UNLIMITED;
			DataSpace pulseDataSpace = DataSpace(1, initDimsPulseDS, maxDimsPulseDS);
			
			hsize_t dimsPulseChunk[1];
			dimsPulseChunk[0] = spdFile->getPulseBlockSize();
			
			DSetCreatPropList creationPulseDSPList;
			creationPulseDSPList.setChunk(1, dimsPulseChunk);
            creationPulseDSPList.setShuffle();
			creationPulseDSPList.setDeflate(SPD_DEFLATE);
            
            CompType *spdPulseDataTypeDisk = NULL;
            if(spdFile->getPulseVersion() == 1)
            {
                SPDPulseH5V1 spdPulse = SPDPulseH5V1();
                pulseUtils.initSPDPulseH5(&spdPulse);
                spdPulseDataTypeDisk = pulseUtils.createSPDPulseH5V1DataTypeDisk();
                spdPulseDataTypeDisk->pack();
                spdPulseDataType = pulseUtils.createSPDPulseH5V1DataTypeMemory();
                creationPulseDSPList.setFillValue( *spdPulseDataTypeDisk, &spdPulse);
            }
            else if(spdFile->getPulseVersion() == 2)
            {
                SPDPulseH5V2 spdPulse = SPDPulseH5V2();
                pulseUtils.initSPDPulseH5(&spdPulse);
                spdPulseDataTypeDisk = pulseUtils.createSPDPulseH5V2DataTypeDisk();
                spdPulseDataTypeDisk->pack();
                spdPulseDataType = pulseUtils.createSPDPulseH5V2DataTypeMemory();
                creationPulseDSPList.setFillValue( *spdPulseDataTypeDisk, &spdPulse);
            }
            else
            {
                throw SPDIOException("Did not recognise the Pulse version.");
            }
			pulsesDataset = new DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_PULSES, *spdPulseDataTypeDisk, pulseDataSpace, creationPulseDSPList));
			
			// Create DataType, DataSpace and Dataset for Points
			hsize_t initDimsPtsDS[1];
			initDimsPtsDS[0] = 0;
			hsize_t maxDimsPtsDS[1];
			maxDimsPtsDS[0] = H5S_UNLIMITED;
			DataSpace ptsDataSpace = DataSpace(1, initDimsPtsDS, maxDimsPtsDS);
			
			hsize_t dimsPtsChunk[1];
			dimsPtsChunk[0] = spdFile->getPointBlockSize();
			
			DSetCreatPropList creationPtsDSPList;
			creationPtsDSPList.setChunk(1, dimsPtsChunk);			
			creationPtsDSPList.setShuffle();
            creationPtsDSPList.setDeflate(SPD_DEFLATE);
            
            CompType *spdPointDataTypeDisk = NULL;
            if(spdFile->getPointVersion() == 1)
            {
                SPDPointH5V1 spdPoint = SPDPointH5V1();
                pointUtils.initSPDPoint(&spdPoint);
                spdPointDataTypeDisk = pointUtils.createSPDPointV1DataTypeDisk();
                spdPointDataTypeDisk->pack();
                spdPointDataType = pointUtils.createSPDPointV1DataTypeMemory();
                creationPtsDSPList.setFillValue( *spdPointDataTypeDisk, &spdPoint);
            }
            else if(spdFile->getPointVersion() == 2)
            {
                SPDPointH5V2 spdPoint = SPDPointH5V2();
                pointUtils.initSPDPoint(&spdPoint);
                spdPointDataTypeDisk = pointUtils.createSPDPointV2DataTypeDisk();
                spdPointDataTypeDisk->pack();
                spdPointDataType = pointUtils.createSPDPointV2DataTypeMemory();
                creationPtsDSPList.setFillValue( *spdPointDataTypeDisk, &spdPoint);
            }
            else
            {
                throw SPDIOException("Did not recognise the Point version");
            }
			pointsDataset = new DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_POINTS, *spdPointDataTypeDisk, ptsDataSpace, creationPtsDSPList));
			
			// Create transmitted and received DataSpace and Dataset
			hsize_t initDimsWaveformDS[1];
			initDimsWaveformDS[0] = 0;
			hsize_t maxDimsWaveformDS[1];
			maxDimsWaveformDS[0] = H5S_UNLIMITED;
			DataSpace waveformDataSpace = DataSpace(1, initDimsWaveformDS, maxDimsWaveformDS);
			
			hsize_t dimsReceivedChunk[1];
			dimsReceivedChunk[0] = spdFile->getReceivedBlockSize();
            
			hsize_t dimsTransmittedChunk[1];
			dimsTransmittedChunk[0] = spdFile->getTransmittedBlockSize();
			
            if(spdFile->getWaveformBitRes() == SPD_32_BIT_WAVE)
            {
                IntType intU32DataType( PredType::STD_U32LE );
                intU32DataType.setOrder( H5T_ORDER_LE );
                
                boost::uint_fast32_t fillValueUInt = 0;
                DSetCreatPropList creationReceivedDSPList;
                creationReceivedDSPList.setChunk(1, dimsReceivedChunk);
                creationReceivedDSPList.setShuffle();
                creationReceivedDSPList.setDeflate(SPD_DEFLATE);
                creationReceivedDSPList.setFillValue( PredType::STD_U32LE, &fillValueUInt);
                
                DSetCreatPropList creationTransmittedDSPList;
                creationTransmittedDSPList.setChunk(1, dimsTransmittedChunk);			
                creationTransmittedDSPList.setShuffle();
                creationTransmittedDSPList.setDeflate(SPD_DEFLATE);
                creationTransmittedDSPList.setFillValue( PredType::STD_U32LE, &fillValueUInt);
                
                receivedDataset = new DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_RECEIVED, intU32DataType, waveformDataSpace, creationReceivedDSPList));
                transmittedDataset = new DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_TRANSMITTED, intU32DataType, waveformDataSpace, creationTransmittedDSPList));
            }
            else if(spdFile->getWaveformBitRes() == SPD_16_BIT_WAVE)
            {
                IntType intU16DataType( PredType::STD_U16LE );
                intU16DataType.setOrder( H5T_ORDER_LE );
                
                boost::uint_fast32_t fillValueUInt = 0;
                DSetCreatPropList creationReceivedDSPList;
                creationReceivedDSPList.setChunk(1, dimsReceivedChunk);			
                creationReceivedDSPList.setShuffle();
                creationReceivedDSPList.setDeflate(SPD_DEFLATE);
                creationReceivedDSPList.setFillValue( PredType::STD_U16LE, &fillValueUInt);
                
                DSetCreatPropList creationTransmittedDSPList;
                creationTransmittedDSPList.setChunk(1, dimsTransmittedChunk);			
                creationTransmittedDSPList.setShuffle();
                creationTransmittedDSPList.setDeflate(SPD_DEFLATE);
                creationTransmittedDSPList.setFillValue( PredType::STD_U16LE, &fillValueUInt);
                
                receivedDataset = new DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_RECEIVED, intU16DataType, waveformDataSpace, creationReceivedDSPList));
                transmittedDataset = new DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_TRANSMITTED, intU16DataType, waveformDataSpace, creationTransmittedDSPList));
            }
            else if(spdFile->getWaveformBitRes() == SPD_8_BIT_WAVE)
            {
                IntType intU8DataType( PredType::STD_U8LE );
                intU8DataType.setOrder( H5T_ORDER_LE );
                
                boost::uint_fast32_t fillValueUInt = 0;
                DSetCreatPropList creationReceivedDSPList;
                creationReceivedDSPList.setChunk(1, dimsReceivedChunk);			
                creationReceivedDSPList.setShuffle();
                creationReceivedDSPList.setDeflate(SPD_DEFLATE);
                creationReceivedDSPList.setFillValue( PredType::STD_U8LE, &fillValueUInt);
                
                DSetCreatPropList creationTransmittedDSPList;
                creationTransmittedDSPList.setChunk(1, dimsTransmittedChunk);			
                creationTransmittedDSPList.setShuffle();
                creationTransmittedDSPList.setDeflate(SPD_DEFLATE);
                creationTransmittedDSPList.setFillValue( PredType::STD_U8LE, &fillValueUInt);
                
                receivedDataset = new DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_RECEIVED, intU8DataType, waveformDataSpace, creationReceivedDSPList));
                transmittedDataset = new DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_TRANSMITTED, intU8DataType, waveformDataSpace, creationTransmittedDSPList));
            }
            else
            {
                throw SPDIOException("Waveform bit resolution is unknown.");
            }
			
			// Create Reference datasets and dataspaces		
			IntType intU64DataType( PredType::STD_U64LE );
            intU64DataType.setOrder( H5T_ORDER_LE );
            IntType intU32DataType( PredType::STD_U32LE );
            intU32DataType.setOrder( H5T_ORDER_LE );
			
			hsize_t initDimsIndexDS[2];
			initDimsIndexDS[0] = numRows;
			initDimsIndexDS[1] = numCols;
			DataSpace indexDataSpace(2, initDimsIndexDS);
			
			hsize_t dimsIndexChunk[2];
			dimsIndexChunk[0] = 1;
			dimsIndexChunk[1] = numCols;
			
			boost::uint_fast32_t fillValue32bit = 0;
			DSetCreatPropList initParamsIndexPulsesPerBin;
			initParamsIndexPulsesPerBin.setChunk(2, dimsIndexChunk);			
			initParamsIndexPulsesPerBin.setShuffle();
            initParamsIndexPulsesPerBin.setDeflate(SPD_DEFLATE);
			initParamsIndexPulsesPerBin.setFillValue( PredType::STD_U32LE, &fillValue32bit);
			
			boost::uint_fast64_t fillValue64bit = 0;
			DSetCreatPropList initParamsIndexOffset;
			initParamsIndexOffset.setChunk(2, dimsIndexChunk);			
			initParamsIndexOffset.setShuffle();
            initParamsIndexOffset.setDeflate(SPD_DEFLATE);
			initParamsIndexOffset.setFillValue( PredType::STD_U64LE, &fillValue64bit);
			
			datasetPlsPerBin = new DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_PLS_PER_BIN, intU32DataType, indexDataSpace, initParamsIndexPulsesPerBin ));
			datasetBinsOffset = new DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_BIN_OFFSETS, intU64DataType, indexDataSpace, initParamsIndexOffset ));
			
			// Created Quicklook datasets and dataspaces
			FloatType floatDataType( PredType::IEEE_F32LE );
			
			hsize_t initDimsQuicklookDS[2];
			initDimsQuicklookDS[0] = numRows;
			initDimsQuicklookDS[1] = numCols;
			DataSpace quicklookDataSpace(2, initDimsQuicklookDS);
			
			hsize_t dimsQuicklookChunk[2];
			dimsQuicklookChunk[0] = 1;
			dimsQuicklookChunk[1] = numCols;
			
			float fillValueFloatQKL = 0;
			DSetCreatPropList initParamsQuicklook;
			initParamsQuicklook.setChunk(2, dimsQuicklookChunk);			
			initParamsQuicklook.setShuffle();
            initParamsQuicklook.setDeflate(SPD_DEFLATE);
			initParamsQuicklook.setFillValue( PredType::IEEE_F32LE, &fillValueFloatQKL);
			
			datasetQuicklook = new DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_QKLIMAGE, floatDataType, quicklookDataSpace, initParamsQuicklook ));

			this->nextCol = 0;
			this->nextRow = 0;
			this->numPts = 0;
			firstColumn = true;
			fileOpened = true;
            
            xMinWritten = 0;
            yMinWritten = 0;
            zMinWritten = 0;
            xMaxWritten = 0;
            yMaxWritten = 0;
            zMaxWritten = 0;
            azMinWritten = 0;
            zenMinWritten = 0;
            ranMinWritten = 0;
            azMaxWritten = 0;
            zenMaxWritten = 0;
            ranMaxWritten = 0;
			firstReturn = true;
            firstPulse = true;
            
            bufIdxCol = 0;
            bufIdxRow = 0;
            numPulsesForBuf = 0;
            
            plsBuffer = new vector<SPDPulse*>();
            plsBuffer->reserve(spdFile->getPulseBlockSize());
            
            qkBuffer = new float[spdFile->getNumberBinsX()];
            plsInColBuf = new unsigned long[spdFile->getNumberBinsX()];
            plsOffsetBuf = new unsigned long long[spdFile->getNumberBinsX()];
            
		}
		catch( FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataTypeIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		
		return fileOpened;
	}
	
	void SPDSeqFileWriter::writeDataColumn(list<SPDPulse*> *pls, boost::uint_fast32_t col, boost::uint_fast32_t row)throw(SPDIOException)
	{
		SPDPulseUtils pulseUtils;
		SPDPointUtils pointUtils;
        
		if(!fileOpened)
		{
			throw SPDIOException("SPD (HDF5) file not open, cannot finalise.");
		}
        
        if(col >= numCols)
		{
			cout << "Number of Columns = " << numCols << endl;
			cout << col << endl;
			throw SPDIOException("The column you have specified it not within the current file.");
		}
		
		if(row >= numRows)
		{
			cout << "Number of Columns = " << numRows << endl;
			cout << row << endl;
			throw SPDIOException("The row you have specified it not within the current file.");
		}
		
        if((row != this->nextRow) & (col != this->nextCol))
        {
            cout << "The next expected row/column was[" << nextRow << "," << nextCol << "] [" << row << "," << col << "] was provided\n";
            
            throw SPDIOException("The column and row provided were not what was expected.");
        }
		
        if(col != bufIdxCol)
        {
            throw SPDIOException("The index buffer index and the column provided are out of sync.");
        }
        
		try 
		{
			Exception::dontPrint();
			
            list<SPDPulse*>::iterator iterInPls;
            float qkVal = 0;
            boost::uint_fast16_t numVals = 0;
            bool first = true;
            
            // Calculate the Quicklook value
            if(pls->size() > 0)
            {
                if((spdFile->getDecomposedPtDefined() == SPD_TRUE) | (spdFile->getDiscretePtDefined() == SPD_TRUE))
                {
                    for(iterInPls = pls->begin(); iterInPls != pls->end(); ++iterInPls)
                    {
                        if((*iterInPls)->numberOfReturns > 0)
                        {
                            for(vector<SPDPoint*>::iterator iterPts = (*iterInPls)->pts->begin(); iterPts != (*iterInPls)->pts->end(); ++iterPts)
                            {
                                if(spdFile->getIndexType() == SPD_CARTESIAN_IDX)
                                {
                                    if(first)
                                    {
                                        qkVal = (*iterPts)->z;
                                        first = false;
                                    }
                                    else if(qkVal < (*iterPts)->z)
                                    {
                                        qkVal = (*iterPts)->z;
                                    }
                                }
                                else
                                {
                                    if(first)
                                    {
                                        qkVal = (*iterPts)->range;
                                        first = false;
                                    }
                                    else if(qkVal > (*iterPts)->range)
                                    {
                                        qkVal = (*iterPts)->range;
                                    }
                                }
                            }
                        }
                    }
                }
                else if(spdFile->getReceiveWaveformDefined() == SPD_TRUE)
                {
                    for(iterInPls = pls->begin(); iterInPls != pls->end(); ++iterInPls)
                    {
                        for(boost::uint_fast32_t i = 0; i < (*iterInPls)->numOfReceivedBins; ++i)
                        {
                            qkVal += (*iterInPls)->received[i];
                            ++numVals;
                        }
                    }
                    qkVal = qkVal/numVals;
                }
                else
                {
                    qkVal = pls->size();
                }
            }
            
            qkBuffer[bufIdxCol] = qkVal;
            plsInColBuf[bufIdxCol] = pls->size();
            plsOffsetBuf[bufIdxCol] = numPulsesForBuf;
            numPulsesForBuf += pls->size();
            ++bufIdxCol;
			
			for(iterInPls = pls->begin(); iterInPls != pls->end(); ++iterInPls)
			{
				plsBuffer->push_back(*iterInPls);

				if(plsBuffer->size() == spdFile->getPulseBlockSize() )
				{
					unsigned long numPulsesInCol = plsBuffer->size();
					unsigned long numPointsInCol = 0;
					unsigned long numTransValsInCol = 0;
					unsigned long numReceiveValsInCol = 0;
					
					vector<SPDPulse*>::iterator iterPulses;
					for(iterPulses = plsBuffer->begin(); iterPulses != plsBuffer->end(); ++iterPulses)
					{
						numPointsInCol += (*iterPulses)->numberOfReturns;
						numTransValsInCol += (*iterPulses)->numOfTransmittedBins;
						numReceiveValsInCol += (*iterPulses)->numOfReceivedBins;
					}
					
					void *spdPulses = NULL;
					void *spdPoints = NULL;
                    
                    if(spdFile->getPulseVersion() == 1)
                    {
                        spdPulses = new SPDPulseH5V1[numPulsesInCol];
                    }
                    else if(spdFile->getPulseVersion() == 2)
                    {
                        spdPulses = new SPDPulseH5V2[numPulsesInCol];
                    }
                    
                    if(spdFile->getPointVersion() == 1)
                    {
                        spdPoints = new SPDPointH5V1[numPointsInCol];
                    }
                    else if(spdFile->getPointVersion() == 2)
                    {
                        spdPoints = new SPDPointH5V2[numPointsInCol];
                    }
                    
					unsigned long *transmittedValues = new unsigned long[numTransValsInCol];
					unsigned long *receivedValues = new unsigned long[numReceiveValsInCol];
					
					unsigned long long pulseCounter = 0;
					unsigned long long pointCounter = 0;
					unsigned long long transValCounter = 0;
					unsigned long long receiveValCounter = 0;
                    
					for(iterPulses = plsBuffer->begin(); iterPulses != plsBuffer->end(); ++iterPulses)
					{
                        if(spdFile->getPulseVersion() == 1)
                        {
                            SPDPulseH5V1 *pulseObj = &((SPDPulseH5V1 *)spdPulses)[pulseCounter];
                            pulseUtils.copySPDPulseToSPDPulseH5((*iterPulses), pulseObj);
                            pulseObj->ptsStartIdx = (numPts + pointCounter);
                            pulseObj->transmittedStartIdx = (numTransVals + transValCounter);
                            pulseObj->receivedStartIdx = (numReceiveVals + receiveValCounter);
                        }
                        else if(spdFile->getPulseVersion() == 2)
                        {
                            SPDPulseH5V2 *pulseObj = &((SPDPulseH5V2 *)spdPulses)[pulseCounter];
                            pulseUtils.copySPDPulseToSPDPulseH5((*iterPulses), pulseObj);
                            pulseObj->ptsStartIdx = (numPts + pointCounter);
                            pulseObj->transmittedStartIdx = (numTransVals + transValCounter);
                            pulseObj->receivedStartIdx = (numReceiveVals + receiveValCounter);
                        }
						
                        if(firstPulse)
                        {
                            azMinWritten = (*iterPulses)->azimuth;
                            zenMinWritten = (*iterPulses)->zenith;
                            azMaxWritten = (*iterPulses)->azimuth;
                            zenMaxWritten = (*iterPulses)->zenith;
                            
                            firstPulse = false;
                        }
                        else
                        {
                            if((*iterPulses)->azimuth < azMinWritten)
                            {
                                azMinWritten = (*iterPulses)->azimuth;
                            }
                            else if((*iterPulses)->azimuth > azMaxWritten)
                            {
                                azMaxWritten = (*iterPulses)->azimuth;
                            }
                            
                            if((*iterPulses)->zenith < zenMinWritten)
                            {
                                zenMinWritten = (*iterPulses)->zenith;
                            }
                            else if((*iterPulses)->zenith > zenMaxWritten)
                            {
                                zenMaxWritten = (*iterPulses)->zenith;
                            }
                        }
                        
						for(boost::uint_fast16_t n = 0; n < (*iterPulses)->numberOfReturns; ++n)
						{
                            if(spdFile->getPointVersion() == 1)
                            {
                                pointUtils.copySPDPointTo((*iterPulses)->pts->at(n), &((SPDPointH5V1 *)spdPoints)[pointCounter++]);
                            }
                            else if(spdFile->getPointVersion() == 2)
                            {
                                pointUtils.copySPDPointTo((*iterPulses)->pts->at(n), &((SPDPointH5V2 *)spdPoints)[pointCounter++]);
                            }
                            
                            if(firstReturn)
                            {
                                xMinWritten = (*iterPulses)->pts->at(n)->x;
                                yMinWritten = (*iterPulses)->pts->at(n)->y;
                                zMinWritten = (*iterPulses)->pts->at(n)->z;
                                xMaxWritten = (*iterPulses)->pts->at(n)->x;
                                yMaxWritten = (*iterPulses)->pts->at(n)->y;
                                zMaxWritten = (*iterPulses)->pts->at(n)->z;
                                
                                ranMinWritten = (*iterPulses)->pts->at(n)->range;
                                ranMaxWritten = (*iterPulses)->pts->at(n)->range;
                                
                                firstReturn = false;
                            }
                            else
                            {
                                if((*iterPulses)->pts->at(n)->x < xMinWritten)
                                {
                                    xMinWritten = (*iterPulses)->pts->at(n)->x;
                                }
                                else if((*iterPulses)->pts->at(n)->x > xMaxWritten)
                                {
                                    xMaxWritten = (*iterPulses)->pts->at(n)->x;
                                }
                                
                                if((*iterPulses)->pts->at(n)->y < yMinWritten)
                                {
                                    yMinWritten = (*iterPulses)->pts->at(n)->y;
                                }
                                else if((*iterPulses)->pts->at(n)->y > yMaxWritten)
                                {
                                    yMaxWritten = (*iterPulses)->pts->at(n)->y;
                                }
                                
                                if((*iterPulses)->pts->at(n)->z < zMinWritten)
                                {
                                    zMinWritten = (*iterPulses)->pts->at(n)->z;
                                }
                                else if((*iterPulses)->pts->at(n)->z > zMaxWritten)
                                {
                                    zMaxWritten = (*iterPulses)->pts->at(n)->z;
                                }
                                
                                if((*iterPulses)->pts->at(n)->range < ranMinWritten)
                                {
                                    ranMinWritten = (*iterPulses)->pts->at(n)->range;
                                }
                                else if((*iterPulses)->pts->at(n)->range > ranMaxWritten)
                                {
                                    ranMaxWritten = (*iterPulses)->pts->at(n)->range;
                                }
                            }
						}
						
						for(boost::uint_fast16_t n = 0; n < (*iterPulses)->numOfTransmittedBins; ++n)
						{
							transmittedValues[transValCounter++] = (*iterPulses)->transmitted[n];
						}
						
						for(boost::uint_fast16_t n = 0; n < (*iterPulses)->numOfReceivedBins; ++n)
						{
							receivedValues[receiveValCounter++] = (*iterPulses)->received[n];
						}
						
						++pulseCounter;
						SPDPulseUtils::deleteSPDPulse(*iterPulses);
					}
                    plsBuffer->clear();
					
					// Write Pulses to disk
					hsize_t extendPulsesDatasetTo[1];
					extendPulsesDatasetTo[0] = this->numPulses + numPulsesInCol;
					pulsesDataset->extend( extendPulsesDatasetTo );
					
					hsize_t pulseDataOffset[1];
					pulseDataOffset[0] = this->numPulses;
					hsize_t pulseDataDims[1];
					pulseDataDims[0] = numPulsesInCol;
					
					DataSpace pulseWriteDataSpace = pulsesDataset->getSpace();
					pulseWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pulseDataDims, pulseDataOffset);
					DataSpace newPulsesDataspace = DataSpace(1, pulseDataDims);
					
					pulsesDataset->write(spdPulses, *spdPulseDataType, newPulsesDataspace, pulseWriteDataSpace);
					
					// Write Points to Disk
					if(numPointsInCol > 0)
					{
						hsize_t extendPointsDatasetTo[1];
						extendPointsDatasetTo[0] = this->numPts + numPointsInCol;
						pointsDataset->extend( extendPointsDatasetTo );
						
						hsize_t pointsDataOffset[1];
						pointsDataOffset[0] = this->numPts;
						hsize_t pointsDataDims[1];
						pointsDataDims[0] = numPointsInCol;
						
						DataSpace pointWriteDataSpace = pointsDataset->getSpace();
						pointWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pointsDataDims, pointsDataOffset);
						DataSpace newPointsDataspace = DataSpace(1, pointsDataDims);
						
						pointsDataset->write(spdPoints, *spdPointDataType, newPointsDataspace, pointWriteDataSpace);
					}
					
					// Write Transmitted Values to Disk
					if(numTransValsInCol > 0)
					{
						hsize_t extendTransDatasetTo[1];
						extendTransDatasetTo[0] = this->numTransVals + numTransValsInCol;
						transmittedDataset->extend( extendTransDatasetTo );
						
						hsize_t transDataOffset[1];
						transDataOffset[0] = this->numTransVals;
						hsize_t transDataDims[1];
						transDataDims[0] = numTransValsInCol;
						
						DataSpace transWriteDataSpace = transmittedDataset->getSpace();
						transWriteDataSpace.selectHyperslab(H5S_SELECT_SET, transDataDims, transDataOffset);
						DataSpace newTransDataspace = DataSpace(1, transDataDims);
						
						transmittedDataset->write(transmittedValues, PredType::NATIVE_ULONG, newTransDataspace, transWriteDataSpace);
					}
					
					// Write Recieved Values to Disk
					if(numReceiveValsInCol > 0)
					{
						hsize_t extendReceiveDatasetTo[1];
						extendReceiveDatasetTo[0] = this->numReceiveVals + numReceiveValsInCol;
						receivedDataset->extend( extendReceiveDatasetTo );
						
						hsize_t receivedDataOffset[1];
						receivedDataOffset[0] = this->numReceiveVals;
						hsize_t receivedDataDims[1];
						receivedDataDims[0] = numReceiveValsInCol;
						
						DataSpace receivedWriteDataSpace = receivedDataset->getSpace();
						receivedWriteDataSpace.selectHyperslab(H5S_SELECT_SET, receivedDataDims, receivedDataOffset);
						DataSpace newReceivedDataspace = DataSpace(1, receivedDataDims);
						
						receivedDataset->write(receivedValues, PredType::NATIVE_ULONG, newReceivedDataspace, receivedWriteDataSpace);
					}
					
					// Delete tempory arrarys once written to disk.
                    if(spdFile->getPointVersion() == 1)
                    {
                        delete[] reinterpret_cast<SPDPointH5V1*>(spdPoints);
                    }
                    else if(spdFile->getPointVersion() == 2)
                    {
                        delete[] reinterpret_cast<SPDPointH5V2*>(spdPoints);
                    }
                    
                    if(spdFile->getPulseVersion() == 1)
                    {
                        delete[] reinterpret_cast<SPDPulseH5V1*>(spdPulses);
                    }
                    else if(spdFile->getPulseVersion() == 2)
                    {
                        delete[] reinterpret_cast<SPDPulseH5V2*>(spdPulses);
                    }
                    
					delete[] transmittedValues;
					delete[] receivedValues;
					
					numPulses += numPulsesInCol;
					numPts += numPointsInCol;
					numTransVals += numTransValsInCol;
					numReceiveVals += numReceiveValsInCol;
				}                
			}
            pls->clear();
            
			if(bufIdxCol == spdFile->getNumberBinsX())
            {
                // Write QK image and index lines.
                
                DataSpace plsWritePlsPerBinDataSpace = datasetPlsPerBin->getSpace();
                DataSpace plsWriteOffsetsDataSpace = datasetBinsOffset->getSpace();
                DataSpace plsWriteQKLDataSpace = datasetQuicklook->getSpace();
                
                hsize_t dataIndexOffset[2];
                dataIndexOffset[0] = bufIdxRow;
                dataIndexOffset[1] = 0;
                hsize_t dataIndexDims[2];
                dataIndexDims[0] = 1;
                dataIndexDims[1] = spdFile->getNumberBinsX();
                DataSpace newIndexDataspace = DataSpace(2, dataIndexDims);
                
                plsWritePlsPerBinDataSpace.selectHyperslab( H5S_SELECT_SET, dataIndexDims, dataIndexOffset );
                plsWriteOffsetsDataSpace.selectHyperslab( H5S_SELECT_SET, dataIndexDims, dataIndexOffset );
                plsWriteQKLDataSpace.selectHyperslab( H5S_SELECT_SET, dataIndexDims, dataIndexOffset );
                
                datasetPlsPerBin->write( plsInColBuf, PredType::NATIVE_ULONG, newIndexDataspace, plsWritePlsPerBinDataSpace );
                datasetBinsOffset->write( plsOffsetBuf, PredType::NATIVE_ULLONG, newIndexDataspace, plsWriteOffsetsDataSpace );
                datasetQuicklook->write( qkBuffer, PredType::NATIVE_FLOAT, newIndexDataspace, plsWriteQKLDataSpace );
                
                bufIdxCol = 0;
                ++bufIdxRow;
            }
		}
		catch( FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataTypeIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch(SPDIOException &e)
		{
			throw e;
		}
        
        ++nextCol;
        if(nextCol == spdFile->getNumberBinsX())
        {
            nextCol = 0;
            ++nextRow;
        }
		
        if(nextRow > spdFile->getNumberBinsY())
        {
            throw SPDIOException("The number of rows has exceeded the number specified in the SPDFile.");
        }
	}
	
	void SPDSeqFileWriter::writeDataColumn(vector<SPDPulse*> *pls, boost::uint_fast32_t col, boost::uint_fast32_t row)throw(SPDIOException)
	{
        SPDPulseUtils pulseUtils;
		SPDPointUtils pointUtils;
        
		if(!fileOpened)
		{
			throw SPDIOException("SPD (HDF5) file not open, cannot finalise.");
		}
        
        if(col >= numCols)
		{
			cout << "Number of Columns = " << numCols << endl;
			cout << col << endl;
			throw SPDIOException("The column you have specified it not within the current file.");
		}
		
		if(row >= numRows)
		{
			cout << "Number of Columns = " << numRows << endl;
			cout << row << endl;
			throw SPDIOException("The row you have specified it not within the current file.");
		}
		
        if((row != this->nextRow) & (col != this->nextCol))
        {
            cout << "The next expected row/column was[" << nextRow << "," << nextCol << "] [" << row << "," << col << "] was provided\n";
            
            throw SPDIOException("The column and row provided were not what was expected.");
        }
		
        if(col != bufIdxCol)
        {
            throw SPDIOException("The index buffer index and the column provided are out of sync.");
        }
        
		try 
		{
			Exception::dontPrint();
			
            vector<SPDPulse*>::iterator iterInPls;
            float qkVal = 0;
            boost::uint_fast16_t numVals = 0;
            bool first = true;
            
            // Calculate the Quicklook value
            if(pls->size() > 0)
            {
                if((spdFile->getDecomposedPtDefined() == SPD_TRUE) | (spdFile->getDiscretePtDefined() == SPD_TRUE))
                {
                    for(iterInPls = pls->begin(); iterInPls != pls->end(); ++iterInPls)
                    {
                        if((*iterInPls)->numberOfReturns > 0)
                        {
                            for(vector<SPDPoint*>::iterator iterPts = (*iterInPls)->pts->begin(); iterPts != (*iterInPls)->pts->end(); ++iterPts)
                            {
                                if(spdFile->getIndexType() == SPD_CARTESIAN_IDX)
                                {
                                    if(first)
                                    {
                                        qkVal = (*iterPts)->z;
                                        first = false;
                                    }
                                    else if(qkVal < (*iterPts)->z)
                                    {
                                        qkVal = (*iterPts)->z;
                                    }
                                }
                                else
                                {
                                    if(first)
                                    {
                                        qkVal = (*iterPts)->range;
                                        first = false;
                                    }
                                    else if(qkVal > (*iterPts)->range)
                                    {
                                        qkVal = (*iterPts)->range;
                                    }
                                }
                            }
                        }
                    }
                }
                else if(spdFile->getReceiveWaveformDefined() == SPD_TRUE)
                {
                    for(iterInPls = pls->begin(); iterInPls != pls->end(); ++iterInPls)
                    {
                        for(boost::uint_fast32_t i = 0; i < (*iterInPls)->numOfReceivedBins; ++i)
                        {
                            qkVal += (*iterInPls)->received[i];
                            ++numVals;
                        }
                    }
                    qkVal = qkVal/numVals;
                }
                else
                {
                    qkVal = pls->size();
                }
            }
            
            qkBuffer[bufIdxCol] = qkVal;
            plsInColBuf[bufIdxCol] = pls->size();
            plsOffsetBuf[bufIdxCol] = numPulsesForBuf;
            numPulsesForBuf += pls->size();
            ++bufIdxCol; 
			
			for(iterInPls = pls->begin(); iterInPls != pls->end(); ++iterInPls)
			{
				plsBuffer->push_back(*iterInPls);

				if(plsBuffer->size() == spdFile->getPulseBlockSize() )
				{
					unsigned long numPulsesLocal = plsBuffer->size();
					unsigned long numPointsLocal = 0;
					unsigned long numTransValsLocal = 0;
					unsigned long numReceiveValsLocal = 0;
					
					vector<SPDPulse*>::iterator iterPulses;
					for(iterPulses = plsBuffer->begin(); iterPulses != plsBuffer->end(); ++iterPulses)
					{
						numPointsLocal += (*iterPulses)->numberOfReturns;
						numTransValsLocal += (*iterPulses)->numOfTransmittedBins;
						numReceiveValsLocal += (*iterPulses)->numOfReceivedBins;
					}
					
					void *spdPulses = NULL;
					void *spdPoints = NULL;
                    
                    if(spdFile->getPulseVersion() == 1)
                    {
                        spdPulses = new SPDPulseH5V1[numPulsesLocal];
                    }
                    else if(spdFile->getPulseVersion() == 2)
                    {
                        spdPulses = new SPDPulseH5V2[numPulsesLocal];
                    }
                    
                    if(spdFile->getPointVersion() == 1)
                    {
                        spdPoints = new SPDPointH5V1[numPointsLocal];
                    }
                    else if(spdFile->getPointVersion() == 2)
                    {
                        spdPoints = new SPDPointH5V2[numPointsLocal];
                    }
                    
					unsigned long *transmittedValues = new unsigned long[numTransValsLocal];
					unsigned long *receivedValues = new unsigned long[numReceiveValsLocal];
					
					unsigned long long pulseCounter = 0;
					unsigned long long pointCounter = 0;
					unsigned long long transValCounter = 0;
					unsigned long long receiveValCounter = 0;
                    
					for(iterPulses = plsBuffer->begin(); iterPulses != plsBuffer->end(); ++iterPulses)
					{
                        if(spdFile->getPulseVersion() == 1)
                        {
                            SPDPulseH5V1 *pulseObj = &((SPDPulseH5V1 *)spdPulses)[pulseCounter];
                            pulseUtils.copySPDPulseToSPDPulseH5((*iterPulses), pulseObj);
                            pulseObj->ptsStartIdx = (numPts + pointCounter);
                            pulseObj->transmittedStartIdx = (numTransVals + transValCounter);
                            pulseObj->receivedStartIdx = (numReceiveVals + receiveValCounter);
                        }
                        else if(spdFile->getPulseVersion() == 2)
                        {
                            SPDPulseH5V2 *pulseObj = &((SPDPulseH5V2 *)spdPulses)[pulseCounter];
                            pulseUtils.copySPDPulseToSPDPulseH5((*iterPulses), pulseObj);
                            pulseObj->ptsStartIdx = (numPts + pointCounter);
                            pulseObj->transmittedStartIdx = (numTransVals + transValCounter);
                            pulseObj->receivedStartIdx = (numReceiveVals + receiveValCounter);
                        }
						
                        if(firstPulse)
                        {
                            azMinWritten = (*iterPulses)->azimuth;
                            zenMinWritten = (*iterPulses)->zenith;
                            azMaxWritten = (*iterPulses)->azimuth;
                            zenMaxWritten = (*iterPulses)->zenith;
                            
                            firstPulse = false;
                        }
                        else
                        {
                            if((*iterPulses)->azimuth < azMinWritten)
                            {
                                azMinWritten = (*iterPulses)->azimuth;
                            }
                            else if((*iterPulses)->azimuth > azMaxWritten)
                            {
                                azMaxWritten = (*iterPulses)->azimuth;
                            }
                            
                            if((*iterPulses)->zenith < zenMinWritten)
                            {
                                zenMinWritten = (*iterPulses)->zenith;
                            }
                            else if((*iterPulses)->zenith > zenMaxWritten)
                            {
                                zenMaxWritten = (*iterPulses)->zenith;
                            }
                        }
                        
						for(boost::uint_fast16_t n = 0; n < (*iterPulses)->numberOfReturns; ++n)
						{
                            if(spdFile->getPointVersion() == 1)
                            {
                                pointUtils.copySPDPointTo((*iterPulses)->pts->at(n), &((SPDPointH5V1 *)spdPoints)[pointCounter++]);
                            }
                            else if(spdFile->getPointVersion() == 2)
                            {
                                pointUtils.copySPDPointTo((*iterPulses)->pts->at(n), &((SPDPointH5V2 *)spdPoints)[pointCounter++]);
                            }
                            
                            if(firstReturn)
                            {
                                xMinWritten = (*iterPulses)->pts->at(n)->x;
                                yMinWritten = (*iterPulses)->pts->at(n)->y;
                                zMinWritten = (*iterPulses)->pts->at(n)->z;
                                xMaxWritten = (*iterPulses)->pts->at(n)->x;
                                yMaxWritten = (*iterPulses)->pts->at(n)->y;
                                zMaxWritten = (*iterPulses)->pts->at(n)->z;
                                
                                ranMinWritten = (*iterPulses)->pts->at(n)->range;
                                ranMaxWritten = (*iterPulses)->pts->at(n)->range;
                                
                                firstReturn = false;
                            }
                            else
                            {
                                if((*iterPulses)->pts->at(n)->x < xMinWritten)
                                {
                                    xMinWritten = (*iterPulses)->pts->at(n)->x;
                                }
                                else if((*iterPulses)->pts->at(n)->x > xMaxWritten)
                                {
                                    xMaxWritten = (*iterPulses)->pts->at(n)->x;
                                }
                                
                                if((*iterPulses)->pts->at(n)->y < yMinWritten)
                                {
                                    yMinWritten = (*iterPulses)->pts->at(n)->y;
                                }
                                else if((*iterPulses)->pts->at(n)->y > yMaxWritten)
                                {
                                    yMaxWritten = (*iterPulses)->pts->at(n)->y;
                                }
                                
                                if((*iterPulses)->pts->at(n)->z < zMinWritten)
                                {
                                    zMinWritten = (*iterPulses)->pts->at(n)->z;
                                }
                                else if((*iterPulses)->pts->at(n)->z > zMaxWritten)
                                {
                                    zMaxWritten = (*iterPulses)->pts->at(n)->z;
                                }
                                
                                if((*iterPulses)->pts->at(n)->range < ranMinWritten)
                                {
                                    ranMinWritten = (*iterPulses)->pts->at(n)->range;
                                }
                                else if((*iterPulses)->pts->at(n)->range > ranMaxWritten)
                                {
                                    ranMaxWritten = (*iterPulses)->pts->at(n)->range;
                                }
                            }
						}
						
						for(boost::uint_fast16_t n = 0; n < (*iterPulses)->numOfTransmittedBins; ++n)
						{
							transmittedValues[transValCounter++] = (*iterPulses)->transmitted[n];
						}
						
						for(boost::uint_fast16_t n = 0; n < (*iterPulses)->numOfReceivedBins; ++n)
						{
							receivedValues[receiveValCounter++] = (*iterPulses)->received[n];
						}
						
						++pulseCounter;
						SPDPulseUtils::deleteSPDPulse(*iterPulses);
					}
                    plsBuffer->clear();
					
					// Write Pulses to disk
					hsize_t extendPulsesDatasetTo[1];
					extendPulsesDatasetTo[0] = this->numPulses + numPulsesLocal;
					pulsesDataset->extend( extendPulsesDatasetTo );
					
					hsize_t pulseDataOffset[1];
					pulseDataOffset[0] = this->numPulses;
					hsize_t pulseDataDims[1];
					pulseDataDims[0] = numPulsesLocal;
					
					DataSpace pulseWriteDataSpace = pulsesDataset->getSpace();
					pulseWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pulseDataDims, pulseDataOffset);
					DataSpace newPulsesDataspace = DataSpace(1, pulseDataDims);
					
					pulsesDataset->write(spdPulses, *spdPulseDataType, newPulsesDataspace, pulseWriteDataSpace);
					
					// Write Points to Disk
					if(numPointsLocal > 0)
					{
						hsize_t extendPointsDatasetTo[1];
						extendPointsDatasetTo[0] = this->numPts + numPointsLocal;
						pointsDataset->extend( extendPointsDatasetTo );
						
						hsize_t pointsDataOffset[1];
						pointsDataOffset[0] = this->numPts;
						hsize_t pointsDataDims[1];
						pointsDataDims[0] = numPointsLocal;
						
						DataSpace pointWriteDataSpace = pointsDataset->getSpace();
						pointWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pointsDataDims, pointsDataOffset);
						DataSpace newPointsDataspace = DataSpace(1, pointsDataDims);
						
						pointsDataset->write(spdPoints, *spdPointDataType, newPointsDataspace, pointWriteDataSpace);
					}
					
					// Write Transmitted Values to Disk
					if(numTransValsLocal > 0)
					{
						hsize_t extendTransDatasetTo[1];
						extendTransDatasetTo[0] = this->numTransVals + numTransValsLocal;
						transmittedDataset->extend( extendTransDatasetTo );
						
						hsize_t transDataOffset[1];
						transDataOffset[0] = this->numTransVals;
						hsize_t transDataDims[1];
						transDataDims[0] = numTransValsLocal;
						
						DataSpace transWriteDataSpace = transmittedDataset->getSpace();
						transWriteDataSpace.selectHyperslab(H5S_SELECT_SET, transDataDims, transDataOffset);
						DataSpace newTransDataspace = DataSpace(1, transDataDims);
						
						transmittedDataset->write(transmittedValues, PredType::NATIVE_ULONG, newTransDataspace, transWriteDataSpace);
					}
					
					// Write Recieved Values to Disk
					if(numReceiveValsLocal > 0)
					{
						hsize_t extendReceiveDatasetTo[1];
						extendReceiveDatasetTo[0] = this->numReceiveVals + numReceiveValsLocal;
						receivedDataset->extend( extendReceiveDatasetTo );
						
						hsize_t receivedDataOffset[1];
						receivedDataOffset[0] = this->numReceiveVals;
						hsize_t receivedDataDims[1];
						receivedDataDims[0] = numReceiveValsLocal;
						
						DataSpace receivedWriteDataSpace = receivedDataset->getSpace();
						receivedWriteDataSpace.selectHyperslab(H5S_SELECT_SET, receivedDataDims, receivedDataOffset);
						DataSpace newReceivedDataspace = DataSpace(1, receivedDataDims);
						
						receivedDataset->write(receivedValues, PredType::NATIVE_ULONG, newReceivedDataspace, receivedWriteDataSpace);
					}
					
					// Delete tempory arrarys once written to disk.
                    if(spdFile->getPointVersion() == 1)
                    {
                        delete[] reinterpret_cast<SPDPointH5V1*>(spdPoints);
                    }
                    else if(spdFile->getPointVersion() == 2)
                    {
                        delete[] reinterpret_cast<SPDPointH5V2*>(spdPoints);
                    }
                    
                    if(spdFile->getPulseVersion() == 1)
                    {
                        delete[] reinterpret_cast<SPDPulseH5V1*>(spdPulses);
                    }
                    else if(spdFile->getPulseVersion() == 2)
                    {
                        delete[] reinterpret_cast<SPDPulseH5V2*>(spdPulses);
                    }
                    
					delete[] transmittedValues;
					delete[] receivedValues;
					
					numPulses += numPulsesLocal;
					numPts += numPointsLocal;
					numTransVals += numTransValsLocal;
					numReceiveVals += numReceiveValsLocal;
				}                
			}
            pls->clear();
			            
			if(bufIdxCol == spdFile->getNumberBinsX())
            {
                // Write QK image and index lines.
                
                DataSpace plsWritePlsPerBinDataSpace = datasetPlsPerBin->getSpace();
                DataSpace plsWriteOffsetsDataSpace = datasetBinsOffset->getSpace();
                DataSpace plsWriteQKLDataSpace = datasetQuicklook->getSpace();
                
                hsize_t dataIndexOffset[2];
                dataIndexOffset[0] = bufIdxRow;
                dataIndexOffset[1] = 0;
                hsize_t dataIndexDims[2];
                dataIndexDims[0] = 1;
                dataIndexDims[1] = spdFile->getNumberBinsX();
                DataSpace newIndexDataspace = DataSpace(2, dataIndexDims);
                
                plsWritePlsPerBinDataSpace.selectHyperslab( H5S_SELECT_SET, dataIndexDims, dataIndexOffset );
                plsWriteOffsetsDataSpace.selectHyperslab( H5S_SELECT_SET, dataIndexDims, dataIndexOffset );
                plsWriteQKLDataSpace.selectHyperslab( H5S_SELECT_SET, dataIndexDims, dataIndexOffset );
                
                datasetPlsPerBin->write( plsInColBuf, PredType::NATIVE_ULONG, newIndexDataspace, plsWritePlsPerBinDataSpace );
                datasetBinsOffset->write( plsOffsetBuf, PredType::NATIVE_ULLONG, newIndexDataspace, plsWriteOffsetsDataSpace );
                datasetQuicklook->write( qkBuffer, PredType::NATIVE_FLOAT, newIndexDataspace, plsWriteQKLDataSpace );
                
                
                bufIdxCol = 0;
                ++bufIdxRow;
            }
		}
		catch( FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataTypeIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch(SPDIOException &e)
		{
			throw e;
		}
        
        ++nextCol;
        if(nextCol == spdFile->getNumberBinsX())
        {
            nextCol = 0;
            ++nextRow;
        }
		
        if(nextRow > spdFile->getNumberBinsY())
        {
            throw SPDIOException("The number of rows has exceeded the number specified in the SPDFile.");
        }
	}
	
	void SPDSeqFileWriter::finaliseClose() throw(SPDIOException)
	{
		if(!fileOpened)
		{
			throw SPDIOException("SPD (HDF5) file not open, cannot finalise.");
		}
        
        SPDPulseUtils pulseUtils;
		SPDPointUtils pointUtils;
		
		try 
		{
			Exception::dontPrint();
			
			if(plsBuffer->size() > 0 )
			{
				unsigned long numPulsesInCol = plsBuffer->size();
				unsigned long numPointsInCol = 0;
				unsigned long numTransValsInCol = 0;
				unsigned long numReceiveValsInCol = 0;
				
				vector<SPDPulse*>::iterator iterPulses;
				for(iterPulses = plsBuffer->begin(); iterPulses != plsBuffer->end(); ++iterPulses)
				{
					numPointsInCol += (*iterPulses)->numberOfReturns;
					numTransValsInCol += (*iterPulses)->numOfTransmittedBins;
					numReceiveValsInCol += (*iterPulses)->numOfReceivedBins;
				}
                
                void *spdPulses = NULL;
                void *spdPoints = NULL;
                
                if(spdFile->getPulseVersion() == 1)
                {
                    spdPulses = new SPDPulseH5V1[numPulsesInCol];
                }
                else if(spdFile->getPulseVersion() == 2)
                {
                    spdPulses = new SPDPulseH5V2[numPulsesInCol];
                }
                
                if(spdFile->getPointVersion() == 1)
                {
                    spdPoints = new SPDPointH5V1[numPointsInCol];
                }
                else if(spdFile->getPointVersion() == 2)
                {
                    spdPoints = new SPDPointH5V2[numPointsInCol];
                }
                
				unsigned long *transmittedValues = new unsigned long[numTransValsInCol];
				unsigned long *receivedValues = new unsigned long[numReceiveValsInCol];
				
				unsigned long pulseCounter = 0;
				unsigned long pointCounter = 0;
				unsigned long transValCounter = 0;
				unsigned long receiveValCounter = 0;
				
				for(iterPulses = plsBuffer->begin(); iterPulses != plsBuffer->end(); ++iterPulses)
				{
					if(spdFile->getPulseVersion() == 1)
                    {
                        SPDPulseH5V1 *pulseObj = &((SPDPulseH5V1 *)spdPulses)[pulseCounter];
                        pulseUtils.copySPDPulseToSPDPulseH5((*iterPulses), pulseObj);
                        pulseObj->ptsStartIdx = (numPts + pointCounter);
                        pulseObj->transmittedStartIdx = (numTransVals + transValCounter);
                        pulseObj->receivedStartIdx = (numReceiveVals + receiveValCounter);
                    }
                    else if(spdFile->getPulseVersion() == 2)
                    {
                        SPDPulseH5V2 *pulseObj = &((SPDPulseH5V2 *)spdPulses)[pulseCounter];
                        pulseUtils.copySPDPulseToSPDPulseH5((*iterPulses), pulseObj);
                        pulseObj->ptsStartIdx = (numPts + pointCounter);
                        pulseObj->transmittedStartIdx = (numTransVals + transValCounter);
                        pulseObj->receivedStartIdx = (numReceiveVals + receiveValCounter);
                    }
					
                    if(firstPulse)
                    {
                        azMinWritten = (*iterPulses)->azimuth;
                        zenMinWritten = (*iterPulses)->zenith;
                        azMaxWritten = (*iterPulses)->azimuth;
                        zenMaxWritten = (*iterPulses)->zenith;
                        
                        firstPulse = false;
                    }
                    else
                    {
                        if((*iterPulses)->azimuth < azMinWritten)
                        {
                            azMinWritten = (*iterPulses)->azimuth;
                        }
                        else if((*iterPulses)->azimuth > azMaxWritten)
                        {
                            azMaxWritten = (*iterPulses)->azimuth;
                        }
                        
                        if((*iterPulses)->zenith < zenMinWritten)
                        {
                            zenMinWritten = (*iterPulses)->zenith;
                        }
                        else if((*iterPulses)->zenith > zenMaxWritten)
                        {
                            zenMaxWritten = (*iterPulses)->zenith;
                        }
                    }
                    
					for(boost::uint_fast16_t n = 0; n < (*iterPulses)->numberOfReturns; ++n)
					{
						if(spdFile->getPointVersion() == 1)
                        {
                            pointUtils.copySPDPointTo((*iterPulses)->pts->at(n), &((SPDPointH5V1 *)spdPoints)[pointCounter++]);
                        }
                        else if(spdFile->getPointVersion() == 2)
                        {
                            pointUtils.copySPDPointTo((*iterPulses)->pts->at(n), &((SPDPointH5V2 *)spdPoints)[pointCounter++]);
                        }
                        
                        if(firstReturn)
                        {
                            xMinWritten = (*iterPulses)->pts->at(n)->x;
                            yMinWritten = (*iterPulses)->pts->at(n)->y;
                            zMinWritten = (*iterPulses)->pts->at(n)->z;
                            xMaxWritten = (*iterPulses)->pts->at(n)->x;
                            yMaxWritten = (*iterPulses)->pts->at(n)->y;
                            zMaxWritten = (*iterPulses)->pts->at(n)->z;
                            
                            ranMinWritten = (*iterPulses)->pts->at(n)->range;
                            ranMaxWritten = (*iterPulses)->pts->at(n)->range;
                            
                            firstReturn = false;
                        }
                        else
                        {
                            if((*iterPulses)->pts->at(n)->x < xMinWritten)
                            {
                                xMinWritten = (*iterPulses)->pts->at(n)->x;
                            }
                            else if((*iterPulses)->pts->at(n)->x > xMaxWritten)
                            {
                                xMaxWritten = (*iterPulses)->pts->at(n)->x;
                            }
                            
                            if((*iterPulses)->pts->at(n)->y < yMinWritten)
                            {
                                yMinWritten = (*iterPulses)->pts->at(n)->y;
                            }
                            else if((*iterPulses)->pts->at(n)->y > yMaxWritten)
                            {
                                yMaxWritten = (*iterPulses)->pts->at(n)->y;
                            }
                            
                            if((*iterPulses)->pts->at(n)->z < zMinWritten)
                            {
                                zMinWritten = (*iterPulses)->pts->at(n)->z;
                            }
                            else if((*iterPulses)->pts->at(n)->z > zMaxWritten)
                            {
                                zMaxWritten = (*iterPulses)->pts->at(n)->z;
                            }
                            
                            if((*iterPulses)->pts->at(n)->range < ranMinWritten)
                            {
                                ranMinWritten = (*iterPulses)->pts->at(n)->range;
                            }
                            else if((*iterPulses)->pts->at(n)->range > ranMaxWritten)
                            {
                                ranMaxWritten = (*iterPulses)->pts->at(n)->range;
                            }
                        }
					}
					
					for(boost::uint_fast16_t n = 0; n < (*iterPulses)->numOfTransmittedBins; ++n)
					{
						transmittedValues[transValCounter++] = (*iterPulses)->transmitted[n];
					}
					
					for(boost::uint_fast16_t n = 0; n < (*iterPulses)->numOfReceivedBins; ++n)
					{
						receivedValues[receiveValCounter++] = (*iterPulses)->received[n];
					}
					
					++pulseCounter;
					SPDPulseUtils::deleteSPDPulse(*iterPulses);
				}
				plsBuffer->clear();
                
				// Write Pulses to disk
				hsize_t extendPulsesDatasetTo[1];
				extendPulsesDatasetTo[0] = this->numPulses + numPulsesInCol;
				pulsesDataset->extend( extendPulsesDatasetTo );
				
				hsize_t pulseDataOffset[1];
				pulseDataOffset[0] = this->numPulses;
				hsize_t pulseDataDims[1];
				pulseDataDims[0] = numPulsesInCol;
				
				DataSpace pulseWriteDataSpace = pulsesDataset->getSpace();
				pulseWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pulseDataDims, pulseDataOffset);
				DataSpace newPulsesDataspace = DataSpace(1, pulseDataDims);
				
				pulsesDataset->write(spdPulses, *spdPulseDataType, newPulsesDataspace, pulseWriteDataSpace);
				
				// Write Points to Disk
				if(numPointsInCol > 0)
				{
					hsize_t extendPointsDatasetTo[1];
					extendPointsDatasetTo[0] = this->numPts + numPointsInCol;
					pointsDataset->extend( extendPointsDatasetTo );
					
					hsize_t pointsDataOffset[1];
					pointsDataOffset[0] = this->numPts;
					hsize_t pointsDataDims[1];
					pointsDataDims[0] = numPointsInCol;
					
					DataSpace pointWriteDataSpace = pointsDataset->getSpace();
					pointWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pointsDataDims, pointsDataOffset);
					DataSpace newPointsDataspace = DataSpace(1, pointsDataDims);
					
					pointsDataset->write(spdPoints, *spdPointDataType, newPointsDataspace, pointWriteDataSpace);
				}
				
				// Write Transmitted Values to Disk
				if(numTransValsInCol > 0)
				{
					hsize_t extendTransDatasetTo[1];
					extendTransDatasetTo[0] = this->numTransVals + numTransValsInCol;
					transmittedDataset->extend( extendTransDatasetTo );
					
					hsize_t transDataOffset[1];
					transDataOffset[0] = this->numTransVals;
					hsize_t transDataDims[1];
					transDataDims[0] = numTransValsInCol;
					
					DataSpace transWriteDataSpace = transmittedDataset->getSpace();
					transWriteDataSpace.selectHyperslab(H5S_SELECT_SET, transDataDims, transDataOffset);
					DataSpace newTransDataspace = DataSpace(1, transDataDims);
					
					transmittedDataset->write(transmittedValues, PredType::NATIVE_ULONG, newTransDataspace, transWriteDataSpace);
				}
				
				// Write Recieved Values to Disk
				if(numReceiveValsInCol > 0)
				{
					hsize_t extendReceiveDatasetTo[1];
					extendReceiveDatasetTo[0] = this->numReceiveVals + numReceiveValsInCol;
					receivedDataset->extend( extendReceiveDatasetTo );
					
					hsize_t receivedDataOffset[1];
					receivedDataOffset[0] = this->numReceiveVals;
					hsize_t receivedDataDims[1];
					receivedDataDims[0] = numReceiveValsInCol;
					
					DataSpace receivedWriteDataSpace = receivedDataset->getSpace();
					receivedWriteDataSpace.selectHyperslab(H5S_SELECT_SET, receivedDataDims, receivedDataOffset);
					DataSpace newReceivedDataspace = DataSpace(1, receivedDataDims);
					
					receivedDataset->write(receivedValues, PredType::NATIVE_ULONG, newReceivedDataspace, receivedWriteDataSpace);
				}
				
				// Delete tempory arrarys once written to disk.
				if(spdFile->getPointVersion() == 1)
                {
                    delete[] reinterpret_cast<SPDPointH5V1*>(spdPoints);
                }
                else if(spdFile->getPointVersion() == 2)
                {
                    delete[] reinterpret_cast<SPDPointH5V2*>(spdPoints);
                }
                
                if(spdFile->getPulseVersion() == 1)
                {
                    delete[] reinterpret_cast<SPDPulseH5V1*>(spdPulses);
                }
                else if(spdFile->getPulseVersion() == 2)
                {
                    delete[] reinterpret_cast<SPDPulseH5V2*>(spdPulses);
                }
                
				delete[] transmittedValues;
				delete[] receivedValues;
				
				numPulses += numPulsesInCol;
				numPts += numPointsInCol;
				numTransVals += numTransValsInCol;
				numReceiveVals += numReceiveValsInCol;
			}            
		}
		catch( FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataTypeIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch(SPDIOException &e)
		{
			throw e;
		}
        
        
		
		try 
		{
            if(!firstReturn)
            {
                if(spdFile->getIndexType() == SPD_CARTESIAN_IDX)
                {
                    spdFile->setZMin(zMinWritten);
                    spdFile->setZMax(zMaxWritten);
                    spdFile->setBoundingVolumeSpherical(zenMinWritten, zenMaxWritten, azMinWritten, azMaxWritten, ranMinWritten, ranMaxWritten);
                }
                else if(spdFile->getIndexType() == SPD_SPHERICAL_IDX)
                {
                    spdFile->setBoundingVolume(xMinWritten, xMaxWritten, yMinWritten, yMaxWritten, zMinWritten, zMaxWritten);
                    spdFile->setRangeMin(ranMinWritten);
                    spdFile->setRangeMax(ranMaxWritten);
                }
                else if(spdFile->getIndexType() == SPD_POLAR_IDX)
                {
                    spdFile->setBoundingVolume(xMinWritten, xMaxWritten, yMinWritten, yMaxWritten, zMinWritten, zMaxWritten);
                    spdFile->setRangeMin(ranMinWritten);
                    spdFile->setRangeMax(ranMaxWritten);
                }
                else
                {
                    // Do nothing... 
                }
            }
            
			spdFile->setNumberOfPoints(numPts);
            spdFile->setNumberOfPulses(numPulses);
            
            spdFile->setFileType(SPD_SEQ_TYPE);
            
            this->writeHeaderInfo(spdOutH5File, spdFile);

			// Write attributes to Quicklook
			StrType strdatatypeLen6(PredType::C_S1, 6);
			StrType strdatatypeLen4(PredType::C_S1, 4);
			const H5std_string strClassVal ("IMAGE");
			const H5std_string strImgVerVal ("1.2");
			
			DataSpace attr_dataspace = DataSpace(H5S_SCALAR);
			
			Attribute classAttribute = datasetQuicklook->createAttribute(ATTRIBUTENAME_CLASS, strdatatypeLen6, attr_dataspace);
			classAttribute.write(strdatatypeLen6, strClassVal); 
			
			Attribute imgVerAttribute = datasetQuicklook->createAttribute(ATTRIBUTENAME_IMAGE_VERSION, strdatatypeLen4, attr_dataspace);
			imgVerAttribute.write(strdatatypeLen4, strImgVerVal);
			
			delete pulsesDataset;
			delete spdPulseDataType;
			delete pointsDataset;
			delete spdPointDataType;
			delete datasetPlsPerBin;
			delete datasetBinsOffset;
			delete receivedDataset;
			delete transmittedDataset;
			delete datasetQuicklook;
			
			spdOutH5File->flush(H5F_SCOPE_GLOBAL);
			spdOutH5File->close();	
			delete spdOutH5File;
			fileOpened = false;
		}
		catch( FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataTypeIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch(SPDIOException &e)
		{
			throw e;
		}
	}

	
	bool SPDSeqFileWriter::requireGrid()
	{
		return true;
	}
	
	bool SPDSeqFileWriter::needNumOutPts()
	{
		return false;
	}
	
	SPDSeqFileWriter::~SPDSeqFileWriter()
	{
		if(fileOpened)
		{
			try 
			{
				this->finaliseClose();
			}
			catch (SPDIOException &e) 
			{
				cerr << "WARNING: " << e.what() << endl;
			}
		}
	}
    
    
    
    
    
    
    
    
    
    
    SPDNonSeqFileWriter::SPDNonSeqFileWriter() : SPDDataExporter("SPD-NSQ"), spdOutH5File(NULL), pulsesDataset(NULL), spdPulseDataType(NULL), pointsDataset(NULL), spdPointDataType(NULL), datasetPlsPerBin(NULL), datasetBinsOffset(NULL), receivedDataset(NULL), transmittedDataset(NULL), datasetQuicklook(NULL), numPulses(0), numPts(0), numTransVals(0), numReceiveVals(0), firstColumn(true), numCols(0), numRows(0)
	{
		
	}
	
	SPDNonSeqFileWriter::SPDNonSeqFileWriter(const SPDDataExporter &dataExporter) throw(SPDException) : SPDDataExporter(dataExporter), spdOutH5File(NULL), pulsesDataset(NULL), spdPulseDataType(NULL), pointsDataset(NULL), spdPointDataType(NULL), datasetPlsPerBin(NULL), datasetBinsOffset(NULL), receivedDataset(NULL), transmittedDataset(NULL), datasetQuicklook(NULL), numPulses(0), numPts(0), numTransVals(0), numReceiveVals(0), firstColumn(true), numCols(0), numRows(0)
	{
		if(fileOpened)
		{
			throw SPDException("Cannot make a copy of a file exporter when a file is open.");
		}
	}
	
	SPDNonSeqFileWriter& SPDNonSeqFileWriter::operator=(const SPDNonSeqFileWriter& dataExporter) throw(SPDException)
	{
		if(fileOpened)
		{
			throw SPDException("Cannot make a copy of a file exporter when a file is open.");
		}
		
		this->spdFile = dataExporter.spdFile;
		this->outputFile = dataExporter.outputFile;
		return *this;
	}
    
    SPDDataExporter* SPDNonSeqFileWriter::getInstance()
    {
        return new SPDNonSeqFileWriter();
    }
	
	bool SPDNonSeqFileWriter::open(SPDFile *spdFile, string outputFile) throw(SPDIOException)
	{
		SPDPulseUtils pulseUtils;
		SPDPointUtils pointUtils;
		this->spdFile = spdFile;
		this->outputFile = outputFile;
		
		const H5std_string spdFilePath( outputFile );
		
		try 
		{
			Exception::dontPrint();
			
			// Create File..
			spdOutH5File = new H5File( spdFilePath, H5F_ACC_TRUNC );
			
			// Create Groups..
			spdOutH5File->createGroup( GROUPNAME_HEADER );
			spdOutH5File->createGroup( GROUPNAME_INDEX );
			spdOutH5File->createGroup( GROUPNAME_QUICKLOOK );
			spdOutH5File->createGroup( GROUPNAME_DATA );
			
			this->numCols = spdFile->getNumberBinsX();
			this->numRows = spdFile->getNumberBinsY();
			
			// Create DataType, DataSpace and Dataset for Pulses
			hsize_t initDimsPulseDS[1];
			initDimsPulseDS[0] = 0;
			hsize_t maxDimsPulseDS[1];
			maxDimsPulseDS[0] = H5S_UNLIMITED;
			DataSpace pulseDataSpace = DataSpace(1, initDimsPulseDS, maxDimsPulseDS);
			
			hsize_t dimsPulseChunk[1];
			dimsPulseChunk[0] = spdFile->getPulseBlockSize();
			
			DSetCreatPropList creationPulseDSPList;
			creationPulseDSPList.setChunk(1, dimsPulseChunk);			
			creationPulseDSPList.setShuffle();
            creationPulseDSPList.setDeflate(SPD_DEFLATE);
            
            CompType *spdPulseDataTypeDisk = NULL;
            if(spdFile->getPulseVersion() == 1)
            {
                SPDPulseH5V1 spdPulse = SPDPulseH5V1();
                pulseUtils.initSPDPulseH5(&spdPulse);
                spdPulseDataTypeDisk = pulseUtils.createSPDPulseH5V1DataTypeDisk();
                spdPulseDataTypeDisk->pack();
                spdPulseDataType = pulseUtils.createSPDPulseH5V1DataTypeMemory();
                creationPulseDSPList.setFillValue( *spdPulseDataTypeDisk, &spdPulse);
            }
            else if(spdFile->getPulseVersion() == 2)
            {
                SPDPulseH5V2 spdPulse = SPDPulseH5V2();
                pulseUtils.initSPDPulseH5(&spdPulse);
                spdPulseDataTypeDisk = pulseUtils.createSPDPulseH5V2DataTypeDisk();
                spdPulseDataTypeDisk->pack();
                spdPulseDataType = pulseUtils.createSPDPulseH5V2DataTypeMemory();
                creationPulseDSPList.setFillValue( *spdPulseDataTypeDisk, &spdPulse);
            }
            else
            {
                throw SPDIOException("Did not recognise the Pulse version.");
            }
			pulsesDataset = new DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_PULSES, *spdPulseDataTypeDisk, pulseDataSpace, creationPulseDSPList));
			
			
			// Create DataType, DataSpace and Dataset for Points
			hsize_t initDimsPtsDS[1];
			initDimsPtsDS[0] = 0;
			hsize_t maxDimsPtsDS[1];
			maxDimsPtsDS[0] = H5S_UNLIMITED;
			DataSpace ptsDataSpace = DataSpace(1, initDimsPtsDS, maxDimsPtsDS);
			
			hsize_t dimsPtsChunk[1];
			dimsPtsChunk[0] = spdFile->getPointBlockSize();
			
			DSetCreatPropList creationPtsDSPList;
			creationPtsDSPList.setChunk(1, dimsPtsChunk);			
            creationPtsDSPList.setShuffle();
			creationPtsDSPList.setDeflate(SPD_DEFLATE);
            
            CompType *spdPointDataTypeDisk = NULL;
            if(spdFile->getPointVersion() == 1)
            {
                SPDPointH5V1 spdPoint = SPDPointH5V1();
                pointUtils.initSPDPoint(&spdPoint);
                spdPointDataTypeDisk = pointUtils.createSPDPointV1DataTypeDisk();
                spdPointDataTypeDisk->pack();
                spdPointDataType = pointUtils.createSPDPointV1DataTypeMemory();
                creationPtsDSPList.setFillValue( *spdPointDataTypeDisk, &spdPoint);
            }
            else if(spdFile->getPointVersion() == 2)
            {
                SPDPointH5V2 spdPoint = SPDPointH5V2();
                pointUtils.initSPDPoint(&spdPoint);
                spdPointDataTypeDisk = pointUtils.createSPDPointV2DataTypeDisk();
                spdPointDataTypeDisk->pack();
                spdPointDataType = pointUtils.createSPDPointV2DataTypeMemory();
                creationPtsDSPList.setFillValue( *spdPointDataTypeDisk, &spdPoint);
            }
            else
            {
                throw SPDIOException("Did not recognise the Point version");
            }
			pointsDataset = new DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_POINTS, *spdPointDataTypeDisk, ptsDataSpace, creationPtsDSPList));
			
			// Create incoming and outgoing DataSpace and Dataset
			hsize_t initDimsWaveformDS[1];
			initDimsWaveformDS[0] = 0;
			hsize_t maxDimsWaveformDS[1];
			maxDimsWaveformDS[0] = H5S_UNLIMITED;
			DataSpace waveformDataSpace = DataSpace(1, initDimsWaveformDS, maxDimsWaveformDS);
			
			hsize_t dimsReceivedChunk[1];
			dimsReceivedChunk[0] = spdFile->getReceivedBlockSize();
            
			hsize_t dimsTransmittedChunk[1];
			dimsTransmittedChunk[0] = spdFile->getTransmittedBlockSize();
			
            if(spdFile->getWaveformBitRes() == SPD_32_BIT_WAVE)
            {
                IntType intU32DataType( PredType::STD_U32LE );
                intU32DataType.setOrder( H5T_ORDER_LE );
                
                boost::uint_fast32_t fillValueUInt = 0;
                DSetCreatPropList creationReceivedDSPList;
                creationReceivedDSPList.setChunk(1, dimsReceivedChunk);	
                creationReceivedDSPList.setShuffle();
                creationReceivedDSPList.setDeflate(SPD_DEFLATE);
                creationReceivedDSPList.setFillValue( PredType::STD_U32LE, &fillValueUInt);
                
                DSetCreatPropList creationTransmittedDSPList;
                creationTransmittedDSPList.setChunk(1, dimsTransmittedChunk);			
                creationTransmittedDSPList.setShuffle();
                creationTransmittedDSPList.setDeflate(SPD_DEFLATE);
                creationTransmittedDSPList.setFillValue( PredType::STD_U32LE, &fillValueUInt);
                
                receivedDataset = new DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_RECEIVED, intU32DataType, waveformDataSpace, creationReceivedDSPList));
                transmittedDataset = new DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_TRANSMITTED, intU32DataType, waveformDataSpace, creationTransmittedDSPList));
            }
            else if(spdFile->getWaveformBitRes() == SPD_16_BIT_WAVE)
            {
                IntType intU16DataType( PredType::STD_U16LE );
                intU16DataType.setOrder( H5T_ORDER_LE );
                
                boost::uint_fast32_t fillValueUInt = 0;
                DSetCreatPropList creationReceivedDSPList;
                creationReceivedDSPList.setChunk(1, dimsReceivedChunk);			
                creationReceivedDSPList.setShuffle();
                creationReceivedDSPList.setDeflate(SPD_DEFLATE);
                creationReceivedDSPList.setFillValue( PredType::STD_U16LE, &fillValueUInt);
                
                DSetCreatPropList creationTransmittedDSPList;
                creationTransmittedDSPList.setChunk(1, dimsTransmittedChunk);			
                creationTransmittedDSPList.setShuffle();
                creationTransmittedDSPList.setDeflate(SPD_DEFLATE);
                creationTransmittedDSPList.setFillValue( PredType::STD_U16LE, &fillValueUInt);
                
                receivedDataset = new DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_RECEIVED, intU16DataType, waveformDataSpace, creationReceivedDSPList));
                transmittedDataset = new DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_TRANSMITTED, intU16DataType, waveformDataSpace, creationTransmittedDSPList));
            }
            else if(spdFile->getWaveformBitRes() == SPD_8_BIT_WAVE)
            {
                IntType intU8DataType( PredType::STD_U8LE );
                intU8DataType.setOrder( H5T_ORDER_LE );
                
                boost::uint_fast32_t fillValueUInt = 0;
                DSetCreatPropList creationReceivedDSPList;
                creationReceivedDSPList.setChunk(1, dimsReceivedChunk);			
                creationReceivedDSPList.setShuffle();
                creationReceivedDSPList.setDeflate(SPD_DEFLATE);
                creationReceivedDSPList.setFillValue( PredType::STD_U8LE, &fillValueUInt);
                
                DSetCreatPropList creationTransmittedDSPList;
                creationTransmittedDSPList.setChunk(1, dimsTransmittedChunk);			
                creationTransmittedDSPList.setShuffle();
                creationTransmittedDSPList.setDeflate(SPD_DEFLATE);
                creationTransmittedDSPList.setFillValue( PredType::STD_U8LE, &fillValueUInt);
                
                receivedDataset = new DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_RECEIVED, intU8DataType, waveformDataSpace, creationReceivedDSPList));
                transmittedDataset = new DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_TRANSMITTED, intU8DataType, waveformDataSpace, creationTransmittedDSPList));
            }
            else
            {
                throw SPDIOException("Waveform bit resolution is unknown.");
            }
			
			// Create Reference datasets and dataspaces		
			IntType intU64DataType( PredType::STD_U64LE );
            intU64DataType.setOrder( H5T_ORDER_LE );
            IntType intU32DataType( PredType::STD_U32LE );
            intU32DataType.setOrder( H5T_ORDER_LE );
			
			hsize_t initDimsIndexDS[2];
			initDimsIndexDS[0] = numRows;
			initDimsIndexDS[1] = numCols;
			DataSpace indexDataSpace(2, initDimsIndexDS);
			
			hsize_t dimsIndexChunk[2];
			dimsIndexChunk[0] = 1;
			dimsIndexChunk[1] = numCols;
			
			boost::uint_fast32_t fillValue32bit = 0;
			DSetCreatPropList initParamsIndexPulsesPerBin;
			initParamsIndexPulsesPerBin.setChunk(2, dimsIndexChunk);			
			initParamsIndexPulsesPerBin.setShuffle();
            initParamsIndexPulsesPerBin.setDeflate(SPD_DEFLATE);
			initParamsIndexPulsesPerBin.setFillValue( PredType::STD_U32LE, &fillValue32bit);
			
			boost::uint_fast64_t fillValue64bit = 0;
			DSetCreatPropList initParamsIndexOffset;
			initParamsIndexOffset.setChunk(2, dimsIndexChunk);			
			initParamsIndexOffset.setShuffle();
            initParamsIndexOffset.setDeflate(SPD_DEFLATE);
			initParamsIndexOffset.setFillValue( PredType::STD_U64LE, &fillValue64bit);
			
			datasetPlsPerBin = new DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_PLS_PER_BIN, intU32DataType, indexDataSpace, initParamsIndexPulsesPerBin ));
			datasetBinsOffset = new DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_BIN_OFFSETS, intU64DataType, indexDataSpace, initParamsIndexOffset ));
			
			// Created Quicklook datasets and dataspaces
			FloatType floatDataType( PredType::IEEE_F32LE );
			
			hsize_t initDimsQuicklookDS[2];
			initDimsQuicklookDS[0] = numRows;
			initDimsQuicklookDS[1] = numCols;
			DataSpace quicklookDataSpace(2, initDimsQuicklookDS);
			
			hsize_t dimsQuicklookChunk[2];
			dimsQuicklookChunk[0] = 1;
			dimsQuicklookChunk[1] = numCols;
			
			float fillValueFloatQKL = 0;
			DSetCreatPropList initParamsQuicklook;
			initParamsQuicklook.setChunk(2, dimsQuicklookChunk);			
            initParamsQuicklook.setShuffle();
			initParamsQuicklook.setDeflate(SPD_DEFLATE);
			initParamsQuicklook.setFillValue( PredType::IEEE_F32LE, &fillValueFloatQKL);
			
			datasetQuicklook = new DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_QKLIMAGE, floatDataType, quicklookDataSpace, initParamsQuicklook ));
            
			this->numPts = 0;
			firstColumn = true;
			fileOpened = true;
            
            xMinWritten = 0;
            yMinWritten = 0;
            zMinWritten = 0;
            xMaxWritten = 0;
            yMaxWritten = 0;
            zMaxWritten = 0;
            azMinWritten = 0;
            zenMinWritten = 0;
            ranMinWritten = 0;
            azMaxWritten = 0;
            zenMaxWritten = 0;
            ranMaxWritten = 0;
			firstReturn = true;
            firstPulse = true;
            
            plsBuffer = new vector<SPDPulse*>();
            plsBuffer->reserve(spdFile->getPulseBlockSize());
            plsOffset = 0;
            
		}
		catch( FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataTypeIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		
		return fileOpened;
	}
	
	void SPDNonSeqFileWriter::writeDataColumn(list<SPDPulse*> *pls, boost::uint_fast32_t col, boost::uint_fast32_t row)throw(SPDIOException)
	{
		SPDPulseUtils pulseUtils;
		SPDPointUtils pointUtils;
        
		if(!fileOpened)
		{
			throw SPDIOException("SPD (HDF5) file not open, cannot finalise.");
		}
        
        if(col >= numCols)
		{
			cout << "Number of Columns = " << numCols << endl;
			cout << col << endl;
			throw SPDIOException("The column you have specified it not within the current file.");
		}
		
		if(row >= numRows)
		{
			cout << "Number of Columns = " << numRows << endl;
			cout << row << endl;
			throw SPDIOException("The row you have specified it not within the current file.");
		}
        
		try 
		{
			Exception::dontPrint();
			
            list<SPDPulse*>::iterator iterInPls;
            float qkVal = 0;
            boost::uint_fast16_t numVals = 0;
            bool first = true;
            
            // Calculate the Quicklook value
            if(pls->size() > 0)
            {
                if((spdFile->getDecomposedPtDefined() == SPD_TRUE) | (spdFile->getDiscretePtDefined() == SPD_TRUE))
                {
                    for(iterInPls = pls->begin(); iterInPls != pls->end(); ++iterInPls)
                    {
                        if((*iterInPls)->numberOfReturns > 0)
                        {
                            for(vector<SPDPoint*>::iterator iterPts = (*iterInPls)->pts->begin(); iterPts != (*iterInPls)->pts->end(); ++iterPts)
                            {
                                if(spdFile->getIndexType() == SPD_CARTESIAN_IDX)
                                {
                                    if(first)
                                    {
                                        qkVal = (*iterPts)->z;
                                        first = false;
                                    }
                                    else if(qkVal < (*iterPts)->z)
                                    {
                                        qkVal = (*iterPts)->z;
                                    }
                                }
                                else
                                {
                                    if(first)
                                    {
                                        qkVal = (*iterPts)->range;
                                        first = false;
                                    }
                                    else if(qkVal > (*iterPts)->range)
                                    {
                                        qkVal = (*iterPts)->range;
                                    }
                                }
                            }
                        }
                    }
                }
                else if(spdFile->getReceiveWaveformDefined() == SPD_TRUE)
                {
                    for(iterInPls = pls->begin(); iterInPls != pls->end(); ++iterInPls)
                    {
                        for(boost::uint_fast32_t i = 0; i < (*iterInPls)->numOfReceivedBins; ++i)
                        {
                            qkVal += (*iterInPls)->received[i];
                            ++numVals;
                        }
                    }
                    qkVal = qkVal/numVals;
                }
                else
                {
                    qkVal = pls->size();
                }
            }
            
            // Write the Quicklook value and index information.
            DataSpace plsWritePlsPerBinDataSpace = datasetPlsPerBin->getSpace();
            DataSpace plsWriteOffsetsDataSpace = datasetBinsOffset->getSpace();
            DataSpace plsWriteQKLDataSpace = datasetQuicklook->getSpace();
            
            hsize_t dataIndexOffset[2];
            dataIndexOffset[0] = row;
            dataIndexOffset[1] = col;
            hsize_t dataIndexDims[2];
            dataIndexDims[0] = 1;
            dataIndexDims[1] = 1;
            DataSpace newIndexDataspace = DataSpace(2, dataIndexDims);
            
            plsWritePlsPerBinDataSpace.selectHyperslab( H5S_SELECT_SET, dataIndexDims, dataIndexOffset );
            plsWriteOffsetsDataSpace.selectHyperslab( H5S_SELECT_SET, dataIndexDims, dataIndexOffset );
            plsWriteQKLDataSpace.selectHyperslab( H5S_SELECT_SET, dataIndexDims, dataIndexOffset );
            
            unsigned long plsInBin = pls->size();
            
            datasetPlsPerBin->write( &plsInBin, PredType::NATIVE_ULONG, newIndexDataspace, plsWritePlsPerBinDataSpace );
            datasetBinsOffset->write( &plsOffset, PredType::NATIVE_ULLONG, newIndexDataspace, plsWriteOffsetsDataSpace );
            datasetQuicklook->write( &qkVal, PredType::NATIVE_FLOAT, newIndexDataspace, plsWriteQKLDataSpace );
            
            plsOffset += pls->size();
			
			for(iterInPls = pls->begin(); iterInPls != pls->end(); ++iterInPls)
			{
				plsBuffer->push_back(*iterInPls);
				if(plsBuffer->size() == spdFile->getPulseBlockSize() )
				{
					unsigned long numPulsesInCol = plsBuffer->size();
					unsigned long numPointsInCol = 0;
					unsigned long numTransValsInCol = 0;
					unsigned long numReceiveValsInCol = 0;
					
					vector<SPDPulse*>::iterator iterPulses;
					for(iterPulses = plsBuffer->begin(); iterPulses != plsBuffer->end(); ++iterPulses)
					{
						numPointsInCol += (*iterPulses)->numberOfReturns;
						numTransValsInCol += (*iterPulses)->numOfTransmittedBins;
						numReceiveValsInCol += (*iterPulses)->numOfReceivedBins;
					}
					
					void *spdPulses = NULL;
					void *spdPoints = NULL;
                    
                    if(spdFile->getPulseVersion() == 1)
                    {
                        spdPulses = new SPDPulseH5V1[numPulsesInCol];
                    }
                    else if(spdFile->getPulseVersion() == 2)
                    {
                        spdPulses = new SPDPulseH5V2[numPulsesInCol];
                    }
                    
                    if(spdFile->getPointVersion() == 1)
                    {
                        spdPoints = new SPDPointH5V1[numPointsInCol];
                    }
                    else if(spdFile->getPointVersion() == 2)
                    {
                        spdPoints = new SPDPointH5V2[numPointsInCol];
                    }
                    
					unsigned long *transmittedValues = new unsigned long[numTransValsInCol];
					unsigned long *receivedValues = new unsigned long[numReceiveValsInCol];
					
					unsigned long long pulseCounter = 0;
					unsigned long long pointCounter = 0;
					unsigned long long transValCounter = 0;
					unsigned long long receiveValCounter = 0;
                    
					for(iterPulses = plsBuffer->begin(); iterPulses != plsBuffer->end(); ++iterPulses)
					{
                        if(spdFile->getPulseVersion() == 1)
                        {
                            SPDPulseH5V1 *pulseObj = &((SPDPulseH5V1 *)spdPulses)[pulseCounter];
                            pulseUtils.copySPDPulseToSPDPulseH5((*iterPulses), pulseObj);
                            pulseObj->ptsStartIdx = (numPts + pointCounter);
                            pulseObj->transmittedStartIdx = (numTransVals + transValCounter);
                            pulseObj->receivedStartIdx = (numReceiveVals + receiveValCounter);
                        }
                        else if(spdFile->getPulseVersion() == 2)
                        {
                            SPDPulseH5V2 *pulseObj = &((SPDPulseH5V2 *)spdPulses)[pulseCounter];
                            pulseUtils.copySPDPulseToSPDPulseH5((*iterPulses), pulseObj);
                            pulseObj->ptsStartIdx = (numPts + pointCounter);
                            pulseObj->transmittedStartIdx = (numTransVals + transValCounter);
                            pulseObj->receivedStartIdx = (numReceiveVals + receiveValCounter);
                        }
						
                        if(firstPulse)
                        {
                            azMinWritten = (*iterPulses)->azimuth;
                            zenMinWritten = (*iterPulses)->zenith;
                            azMaxWritten = (*iterPulses)->azimuth;
                            zenMaxWritten = (*iterPulses)->zenith;
                            
                            firstPulse = false;
                        }
                        else
                        {
                            if((*iterPulses)->azimuth < azMinWritten)
                            {
                                azMinWritten = (*iterPulses)->azimuth;
                            }
                            else if((*iterPulses)->azimuth > azMaxWritten)
                            {
                                azMaxWritten = (*iterPulses)->azimuth;
                            }
                            
                            if((*iterPulses)->zenith < zenMinWritten)
                            {
                                zenMinWritten = (*iterPulses)->zenith;
                            }
                            else if((*iterPulses)->zenith > zenMaxWritten)
                            {
                                zenMaxWritten = (*iterPulses)->zenith;
                            }
                        }

						for(boost::uint_fast16_t n = 0; n < (*iterPulses)->numberOfReturns; ++n)
						{
                            if(spdFile->getPointVersion() == 1)
                            {
                                pointUtils.copySPDPointTo((*iterPulses)->pts->at(n), &((SPDPointH5V1 *)spdPoints)[pointCounter++]);
                            }
                            else if(spdFile->getPointVersion() == 2)
                            {
                                pointUtils.copySPDPointTo((*iterPulses)->pts->at(n), &((SPDPointH5V2 *)spdPoints)[pointCounter++]);
                            }
                            
                            if(firstReturn)
                            {
                                xMinWritten = (*iterPulses)->pts->at(n)->x;
                                yMinWritten = (*iterPulses)->pts->at(n)->y;
                                zMinWritten = (*iterPulses)->pts->at(n)->z;
                                xMaxWritten = (*iterPulses)->pts->at(n)->x;
                                yMaxWritten = (*iterPulses)->pts->at(n)->y;
                                zMaxWritten = (*iterPulses)->pts->at(n)->z;
                                
                                ranMinWritten = (*iterPulses)->pts->at(n)->range;
                                ranMaxWritten = (*iterPulses)->pts->at(n)->range;
                                
                                firstReturn = false;
                            }
                            else
                            {
                                if((*iterPulses)->pts->at(n)->x < xMinWritten)
                                {
                                    xMinWritten = (*iterPulses)->pts->at(n)->x;
                                }
                                else if((*iterPulses)->pts->at(n)->x > xMaxWritten)
                                {
                                    xMaxWritten = (*iterPulses)->pts->at(n)->x;
                                }
                                
                                if((*iterPulses)->pts->at(n)->y < yMinWritten)
                                {
                                    yMinWritten = (*iterPulses)->pts->at(n)->y;
                                }
                                else if((*iterPulses)->pts->at(n)->y > yMaxWritten)
                                {
                                    yMaxWritten = (*iterPulses)->pts->at(n)->y;
                                }
                                
                                if((*iterPulses)->pts->at(n)->z < zMinWritten)
                                {
                                    zMinWritten = (*iterPulses)->pts->at(n)->z;
                                }
                                else if((*iterPulses)->pts->at(n)->z > zMaxWritten)
                                {
                                    zMaxWritten = (*iterPulses)->pts->at(n)->z;
                                }
                                
                                if((*iterPulses)->pts->at(n)->range < ranMinWritten)
                                {
                                    ranMinWritten = (*iterPulses)->pts->at(n)->range;
                                }
                                else if((*iterPulses)->pts->at(n)->range > ranMaxWritten)
                                {
                                    ranMaxWritten = (*iterPulses)->pts->at(n)->range;
                                }
                            }
						}
						
						for(boost::uint_fast16_t n = 0; n < (*iterPulses)->numOfTransmittedBins; ++n)
						{
							transmittedValues[transValCounter++] = (*iterPulses)->transmitted[n];
						}
						
						for(boost::uint_fast16_t n = 0; n < (*iterPulses)->numOfReceivedBins; ++n)
						{
							receivedValues[receiveValCounter++] = (*iterPulses)->received[n];
						}
						
						++pulseCounter;
						SPDPulseUtils::deleteSPDPulse(*iterPulses);
					}
                    plsBuffer->clear();
					
					// Write Pulses to disk
					hsize_t extendPulsesDatasetTo[1];
					extendPulsesDatasetTo[0] = this->numPulses + numPulsesInCol;
					pulsesDataset->extend( extendPulsesDatasetTo );
					
					hsize_t pulseDataOffset[1];
					pulseDataOffset[0] = this->numPulses;
					hsize_t pulseDataDims[1];
					pulseDataDims[0] = numPulsesInCol;
					
					DataSpace pulseWriteDataSpace = pulsesDataset->getSpace();
					pulseWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pulseDataDims, pulseDataOffset);
					DataSpace newPulsesDataspace = DataSpace(1, pulseDataDims);
					
					pulsesDataset->write(spdPulses, *spdPulseDataType, newPulsesDataspace, pulseWriteDataSpace);
					
					// Write Points to Disk
					if(numPointsInCol > 0)
					{
						hsize_t extendPointsDatasetTo[1];
						extendPointsDatasetTo[0] = this->numPts + numPointsInCol;
						pointsDataset->extend( extendPointsDatasetTo );
						
						hsize_t pointsDataOffset[1];
						pointsDataOffset[0] = this->numPts;
						hsize_t pointsDataDims[1];
						pointsDataDims[0] = numPointsInCol;
						
						DataSpace pointWriteDataSpace = pointsDataset->getSpace();
						pointWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pointsDataDims, pointsDataOffset);
						DataSpace newPointsDataspace = DataSpace(1, pointsDataDims);
						
						pointsDataset->write(spdPoints, *spdPointDataType, newPointsDataspace, pointWriteDataSpace);
					}
					
					// Write Transmitted Values to Disk
					if(numTransValsInCol > 0)
					{
						hsize_t extendTransDatasetTo[1];
						extendTransDatasetTo[0] = this->numTransVals + numTransValsInCol;
						transmittedDataset->extend( extendTransDatasetTo );
						
						hsize_t transDataOffset[1];
						transDataOffset[0] = this->numTransVals;
						hsize_t transDataDims[1];
						transDataDims[0] = numTransValsInCol;
						
						DataSpace transWriteDataSpace = transmittedDataset->getSpace();
						transWriteDataSpace.selectHyperslab(H5S_SELECT_SET, transDataDims, transDataOffset);
						DataSpace newTransDataspace = DataSpace(1, transDataDims);
						
						transmittedDataset->write(transmittedValues, PredType::NATIVE_ULONG, newTransDataspace, transWriteDataSpace);
					}
					
					// Write Recieved Values to Disk
					if(numReceiveValsInCol > 0)
					{
						hsize_t extendReceiveDatasetTo[1];
						extendReceiveDatasetTo[0] = this->numReceiveVals + numReceiveValsInCol;
						receivedDataset->extend( extendReceiveDatasetTo );
						
						hsize_t receivedDataOffset[1];
						receivedDataOffset[0] = this->numReceiveVals;
						hsize_t receivedDataDims[1];
						receivedDataDims[0] = numReceiveValsInCol;
						
						DataSpace receivedWriteDataSpace = receivedDataset->getSpace();
						receivedWriteDataSpace.selectHyperslab(H5S_SELECT_SET, receivedDataDims, receivedDataOffset);
						DataSpace newReceivedDataspace = DataSpace(1, receivedDataDims);
						
						receivedDataset->write(receivedValues, PredType::NATIVE_ULONG, newReceivedDataspace, receivedWriteDataSpace);
					}
					
					// Delete tempory arrarys once written to disk.
                    if(spdFile->getPointVersion() == 1)
                    {
                        delete[] reinterpret_cast<SPDPointH5V1*>(spdPoints);
                    }
                    else if(spdFile->getPointVersion() == 2)
                    {
                        delete[] reinterpret_cast<SPDPointH5V2*>(spdPoints);
                    }
                    
                    if(spdFile->getPulseVersion() == 1)
                    {
                        delete[] reinterpret_cast<SPDPulseH5V1*>(spdPulses);
                    }
                    else if(spdFile->getPulseVersion() == 2)
                    {
                        delete[] reinterpret_cast<SPDPulseH5V2*>(spdPulses);
                    }
                    
					delete[] transmittedValues;
					delete[] receivedValues;
					
					numPulses += numPulsesInCol;
					numPts += numPointsInCol;
					numTransVals += numTransValsInCol;
					numReceiveVals += numReceiveValsInCol;
				}                
			}
            pls->clear();
		}
		catch( FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataTypeIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch(SPDIOException &e)
		{
			throw e;
		}
	}
	
	void SPDNonSeqFileWriter::writeDataColumn(vector<SPDPulse*> *pls, boost::uint_fast32_t col, boost::uint_fast32_t row)throw(SPDIOException)
	{
        SPDPulseUtils pulseUtils;
		SPDPointUtils pointUtils;
        
		if(!fileOpened)
		{
			throw SPDIOException("SPD (HDF5) file not open, cannot finalise.");
		}
        
        if(col >= numCols)
		{
			cout << "Number of Columns = " << numCols << endl;
			cout << col << endl;
			throw SPDIOException("The column you have specified it not within the current file.");
		}
		
		if(row >= numRows)
		{
			cout << "Number of Columns = " << numRows << endl;
			cout << row << endl;
			throw SPDIOException("The row you have specified it not within the current file.");
		}
        
		try 
		{
			Exception::dontPrint();
			
            vector<SPDPulse*>::iterator iterInPls;
            float qkVal = 0;
            boost::uint_fast16_t numVals = 0;
            bool first = true;
            
            // Calculate the Quicklook value
            if(pls->size() > 0)
            {
                if((spdFile->getDecomposedPtDefined() == SPD_TRUE) | (spdFile->getDiscretePtDefined() == SPD_TRUE))
                {
                    for(iterInPls = pls->begin(); iterInPls != pls->end(); ++iterInPls)
                    {
                        if((*iterInPls)->numberOfReturns > 0)
                        {
                            for(vector<SPDPoint*>::iterator iterPts = (*iterInPls)->pts->begin(); iterPts != (*iterInPls)->pts->end(); ++iterPts)
                            {
                                if(spdFile->getIndexType() == SPD_CARTESIAN_IDX)
                                {
                                    if(first)
                                    {
                                        qkVal = (*iterPts)->z;
                                        first = false;
                                    }
                                    else if(qkVal < (*iterPts)->z)
                                    {
                                        qkVal = (*iterPts)->z;
                                    }
                                }
                                else
                                {
                                    if(first)
                                    {
                                        qkVal = (*iterPts)->range;
                                        first = false;
                                    }
                                    else if(qkVal > (*iterPts)->range)
                                    {
                                        qkVal = (*iterPts)->range;
                                    }
                                }
                            }
                        }
                    }
                }
                else if(spdFile->getReceiveWaveformDefined() == SPD_TRUE)
                {
                    for(iterInPls = pls->begin(); iterInPls != pls->end(); ++iterInPls)
                    {
                        for(boost::uint_fast32_t i = 0; i < (*iterInPls)->numOfReceivedBins; ++i)
                        {
                            qkVal += (*iterInPls)->received[i];
                            ++numVals;
                        }
                    }
                    qkVal = qkVal/numVals;
                }
                else
                {
                    qkVal = pls->size();
                }
            }
            
            // Write the Quicklook value and index information.
            DataSpace plsWritePlsPerBinDataSpace = datasetPlsPerBin->getSpace();
            DataSpace plsWriteOffsetsDataSpace = datasetBinsOffset->getSpace();
            DataSpace plsWriteQKLDataSpace = datasetQuicklook->getSpace();
            
            hsize_t dataIndexOffset[2];
            dataIndexOffset[0] = row;
            dataIndexOffset[1] = col;
            hsize_t dataIndexDims[2];
            dataIndexDims[0] = 1;
            dataIndexDims[1] = 1;
            DataSpace newIndexDataspace = DataSpace(2, dataIndexDims);
            
            plsWritePlsPerBinDataSpace.selectHyperslab( H5S_SELECT_SET, dataIndexDims, dataIndexOffset );
            plsWriteOffsetsDataSpace.selectHyperslab( H5S_SELECT_SET, dataIndexDims, dataIndexOffset );
            plsWriteQKLDataSpace.selectHyperslab( H5S_SELECT_SET, dataIndexDims, dataIndexOffset );
            
            unsigned long plsInBin = pls->size();
            
            datasetPlsPerBin->write( &plsInBin, PredType::NATIVE_ULONG, newIndexDataspace, plsWritePlsPerBinDataSpace );
            datasetBinsOffset->write( &plsOffset, PredType::NATIVE_ULLONG, newIndexDataspace, plsWriteOffsetsDataSpace );
            datasetQuicklook->write( &qkVal, PredType::NATIVE_FLOAT, newIndexDataspace, plsWriteQKLDataSpace );
            
            plsOffset += pls->size();
			
			for(iterInPls = pls->begin(); iterInPls != pls->end(); ++iterInPls)
			{
				plsBuffer->push_back(*iterInPls);

				if(plsBuffer->size() == spdFile->getPulseBlockSize() )
				{
					unsigned long numPulsesInCol = plsBuffer->size();
					unsigned long numPointsInCol = 0;
					unsigned long numTransValsInCol = 0;
					unsigned long numReceiveValsInCol = 0;
					
					vector<SPDPulse*>::iterator iterPulses;
					for(iterPulses = plsBuffer->begin(); iterPulses != plsBuffer->end(); ++iterPulses)
					{
						numPointsInCol += (*iterPulses)->numberOfReturns;
						numTransValsInCol += (*iterPulses)->numOfTransmittedBins;
						numReceiveValsInCol += (*iterPulses)->numOfReceivedBins;
					}
					
					void *spdPulses = NULL;
					void *spdPoints = NULL;
                    
                    if(spdFile->getPulseVersion() == 1)
                    {
                        spdPulses = new SPDPulseH5V1[numPulsesInCol];
                    }
                    else if(spdFile->getPulseVersion() == 2)
                    {
                        spdPulses = new SPDPulseH5V2[numPulsesInCol];
                    }
                    
                    if(spdFile->getPointVersion() == 1)
                    {
                        spdPoints = new SPDPointH5V1[numPointsInCol];
                    }
                    else if(spdFile->getPointVersion() == 2)
                    {
                        spdPoints = new SPDPointH5V2[numPointsInCol];
                    }
                    
					unsigned long *transmittedValues = new unsigned long[numTransValsInCol];
					unsigned long *receivedValues = new unsigned long[numReceiveValsInCol];
					
					unsigned long long pulseCounter = 0;
					unsigned long long pointCounter = 0;
					unsigned long long transValCounter = 0;
					unsigned long long receiveValCounter = 0;
                    
					for(iterPulses = plsBuffer->begin(); iterPulses != plsBuffer->end(); ++iterPulses)
					{
                        if(spdFile->getPulseVersion() == 1)
                        {
                            SPDPulseH5V1 *pulseObj = &((SPDPulseH5V1 *)spdPulses)[pulseCounter];
                            pulseUtils.copySPDPulseToSPDPulseH5((*iterPulses), pulseObj);
                            pulseObj->ptsStartIdx = (numPts + pointCounter);
                            pulseObj->transmittedStartIdx = (numTransVals + transValCounter);
                            pulseObj->receivedStartIdx = (numReceiveVals + receiveValCounter);
                        }
                        else if(spdFile->getPulseVersion() == 2)
                        {
                            SPDPulseH5V2 *pulseObj = &((SPDPulseH5V2 *)spdPulses)[pulseCounter];
                            pulseUtils.copySPDPulseToSPDPulseH5((*iterPulses), pulseObj);
                            pulseObj->ptsStartIdx = (numPts + pointCounter);
                            pulseObj->transmittedStartIdx = (numTransVals + transValCounter);
                            pulseObj->receivedStartIdx = (numReceiveVals + receiveValCounter);
                        }
						
                        if(firstPulse)
                        {
                            azMinWritten = (*iterPulses)->azimuth;
                            zenMinWritten = (*iterPulses)->zenith;
                            azMaxWritten = (*iterPulses)->azimuth;
                            zenMaxWritten = (*iterPulses)->zenith;
                            
                            firstPulse = false;
                        }
                        else
                        {
                            if((*iterPulses)->azimuth < azMinWritten)
                            {
                                azMinWritten = (*iterPulses)->azimuth;
                            }
                            else if((*iterPulses)->azimuth > azMaxWritten)
                            {
                                azMaxWritten = (*iterPulses)->azimuth;
                            }
                            
                            if((*iterPulses)->zenith < zenMinWritten)
                            {
                                zenMinWritten = (*iterPulses)->zenith;
                            }
                            else if((*iterPulses)->zenith > zenMaxWritten)
                            {
                                zenMaxWritten = (*iterPulses)->zenith;
                            }
                        }
                        
						for(boost::uint_fast16_t n = 0; n < (*iterPulses)->numberOfReturns; ++n)
						{
                            if(spdFile->getPointVersion() == 1)
                            {
                                pointUtils.copySPDPointTo((*iterPulses)->pts->at(n), &((SPDPointH5V1 *)spdPoints)[pointCounter++]);
                            }
                            else if(spdFile->getPointVersion() == 2)
                            {
                                pointUtils.copySPDPointTo((*iterPulses)->pts->at(n), &((SPDPointH5V2 *)spdPoints)[pointCounter++]);
                            }
                            
                            if(firstReturn)
                            {
                                xMinWritten = (*iterPulses)->pts->at(n)->x;
                                yMinWritten = (*iterPulses)->pts->at(n)->y;
                                zMinWritten = (*iterPulses)->pts->at(n)->z;
                                xMaxWritten = (*iterPulses)->pts->at(n)->x;
                                yMaxWritten = (*iterPulses)->pts->at(n)->y;
                                zMaxWritten = (*iterPulses)->pts->at(n)->z;
                                
                                ranMinWritten = (*iterPulses)->pts->at(n)->range;
                                ranMaxWritten = (*iterPulses)->pts->at(n)->range;
                                
                                firstReturn = false;
                            }
                            else
                            {
                                if((*iterPulses)->pts->at(n)->x < xMinWritten)
                                {
                                    xMinWritten = (*iterPulses)->pts->at(n)->x;
                                }
                                else if((*iterPulses)->pts->at(n)->x > xMaxWritten)
                                {
                                    xMaxWritten = (*iterPulses)->pts->at(n)->x;
                                }
                                
                                if((*iterPulses)->pts->at(n)->y < yMinWritten)
                                {
                                    yMinWritten = (*iterPulses)->pts->at(n)->y;
                                }
                                else if((*iterPulses)->pts->at(n)->y > yMaxWritten)
                                {
                                    yMaxWritten = (*iterPulses)->pts->at(n)->y;
                                }
                                
                                if((*iterPulses)->pts->at(n)->z < zMinWritten)
                                {
                                    zMinWritten = (*iterPulses)->pts->at(n)->z;
                                }
                                else if((*iterPulses)->pts->at(n)->z > zMaxWritten)
                                {
                                    zMaxWritten = (*iterPulses)->pts->at(n)->z;
                                }
                                
                                if((*iterPulses)->pts->at(n)->range < ranMinWritten)
                                {
                                    ranMinWritten = (*iterPulses)->pts->at(n)->range;
                                }
                                else if((*iterPulses)->pts->at(n)->range > ranMaxWritten)
                                {
                                    ranMaxWritten = (*iterPulses)->pts->at(n)->range;
                                }
                            }
						}
						
						for(boost::uint_fast16_t n = 0; n < (*iterPulses)->numOfTransmittedBins; ++n)
						{
							transmittedValues[transValCounter++] = (*iterPulses)->transmitted[n];
						}
						
						for(boost::uint_fast16_t n = 0; n < (*iterPulses)->numOfReceivedBins; ++n)
						{
							receivedValues[receiveValCounter++] = (*iterPulses)->received[n];
						}
						
						++pulseCounter;
						SPDPulseUtils::deleteSPDPulse(*iterPulses);
					}
                    plsBuffer->clear();
					
					// Write Pulses to disk
					hsize_t extendPulsesDatasetTo[1];
					extendPulsesDatasetTo[0] = this->numPulses + numPulsesInCol;
					pulsesDataset->extend( extendPulsesDatasetTo );
					
					hsize_t pulseDataOffset[1];
					pulseDataOffset[0] = this->numPulses;
					hsize_t pulseDataDims[1];
					pulseDataDims[0] = numPulsesInCol;
					
					DataSpace pulseWriteDataSpace = pulsesDataset->getSpace();
					pulseWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pulseDataDims, pulseDataOffset);
					DataSpace newPulsesDataspace = DataSpace(1, pulseDataDims);
					
					pulsesDataset->write(spdPulses, *spdPulseDataType, newPulsesDataspace, pulseWriteDataSpace);
					
					// Write Points to Disk
					if(numPointsInCol > 0)
					{
						hsize_t extendPointsDatasetTo[1];
						extendPointsDatasetTo[0] = this->numPts + numPointsInCol;
						pointsDataset->extend( extendPointsDatasetTo );
						
						hsize_t pointsDataOffset[1];
						pointsDataOffset[0] = this->numPts;
						hsize_t pointsDataDims[1];
						pointsDataDims[0] = numPointsInCol;
						
						DataSpace pointWriteDataSpace = pointsDataset->getSpace();
						pointWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pointsDataDims, pointsDataOffset);
						DataSpace newPointsDataspace = DataSpace(1, pointsDataDims);
						
						pointsDataset->write(spdPoints, *spdPointDataType, newPointsDataspace, pointWriteDataSpace);
					}
					
					// Write Transmitted Values to Disk
					if(numTransValsInCol > 0)
					{
						hsize_t extendTransDatasetTo[1];
						extendTransDatasetTo[0] = this->numTransVals + numTransValsInCol;
						transmittedDataset->extend( extendTransDatasetTo );
						
						hsize_t transDataOffset[1];
						transDataOffset[0] = this->numTransVals;
						hsize_t transDataDims[1];
						transDataDims[0] = numTransValsInCol;
						
						DataSpace transWriteDataSpace = transmittedDataset->getSpace();
						transWriteDataSpace.selectHyperslab(H5S_SELECT_SET, transDataDims, transDataOffset);
						DataSpace newTransDataspace = DataSpace(1, transDataDims);
						
						transmittedDataset->write(transmittedValues, PredType::NATIVE_ULONG, newTransDataspace, transWriteDataSpace);
					}
					
					// Write Recieved Values to Disk
					if(numReceiveValsInCol > 0)
					{
						hsize_t extendReceiveDatasetTo[1];
						extendReceiveDatasetTo[0] = this->numReceiveVals + numReceiveValsInCol;
						receivedDataset->extend( extendReceiveDatasetTo );
						
						hsize_t receivedDataOffset[1];
						receivedDataOffset[0] = this->numReceiveVals;
						hsize_t receivedDataDims[1];
						receivedDataDims[0] = numReceiveValsInCol;
						
						DataSpace receivedWriteDataSpace = receivedDataset->getSpace();
						receivedWriteDataSpace.selectHyperslab(H5S_SELECT_SET, receivedDataDims, receivedDataOffset);
						DataSpace newReceivedDataspace = DataSpace(1, receivedDataDims);
						
						receivedDataset->write(receivedValues, PredType::NATIVE_ULONG, newReceivedDataspace, receivedWriteDataSpace);
					}
					
					// Delete tempory arrarys once written to disk.
                    if(spdFile->getPointVersion() == 1)
                    {
                        delete[] reinterpret_cast<SPDPointH5V1*>(spdPoints);
                    }
                    else if(spdFile->getPointVersion() == 2)
                    {
                        delete[] reinterpret_cast<SPDPointH5V2*>(spdPoints);
                    }
                    
                    if(spdFile->getPulseVersion() == 1)
                    {
                        delete[] reinterpret_cast<SPDPulseH5V1*>(spdPulses);
                    }
                    else if(spdFile->getPulseVersion() == 2)
                    {
                        delete[] reinterpret_cast<SPDPulseH5V2*>(spdPulses);
                    }
                    
					delete[] transmittedValues;
					delete[] receivedValues;
					
					numPulses += numPulsesInCol;
					numPts += numPointsInCol;
					numTransVals += numTransValsInCol;
					numReceiveVals += numReceiveValsInCol;
				}                
			}
            pls->clear();
		}
		catch( FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataTypeIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch(SPDIOException &e)
		{
			throw e;
		}
	}
	
	void SPDNonSeqFileWriter::finaliseClose() throw(SPDIOException)
	{
		if(!fileOpened)
		{
			throw SPDIOException("SPD (HDF5) file not open, cannot finalise.");
		}
        
        SPDPulseUtils pulseUtils;
		SPDPointUtils pointUtils;
		
		try 
		{
			Exception::dontPrint();
			
			if(plsBuffer->size() > 0 )
			{
				unsigned long numPulsesInCol = plsBuffer->size();
				unsigned long numPointsInCol = 0;
				unsigned long numTransValsInCol = 0;
				unsigned long numReceiveValsInCol = 0;
				
				vector<SPDPulse*>::iterator iterPulses;
				for(iterPulses = plsBuffer->begin(); iterPulses != plsBuffer->end(); ++iterPulses)
				{
					numPointsInCol += (*iterPulses)->numberOfReturns;
					numTransValsInCol += (*iterPulses)->numOfTransmittedBins;
					numReceiveValsInCol += (*iterPulses)->numOfReceivedBins;
				}
                
                void *spdPulses = NULL;
                void *spdPoints = NULL;
                
                if(spdFile->getPulseVersion() == 1)
                {
                    spdPulses = new SPDPulseH5V1[numPulsesInCol];
                }
                else if(spdFile->getPulseVersion() == 2)
                {
                    spdPulses = new SPDPulseH5V2[numPulsesInCol];
                }
                
                if(spdFile->getPointVersion() == 1)
                {
                    spdPoints = new SPDPointH5V1[numPointsInCol];
                }
                else if(spdFile->getPointVersion() == 2)
                {
                    spdPoints = new SPDPointH5V2[numPointsInCol];
                }
                
				unsigned long *transmittedValues = new unsigned long[numTransValsInCol];
				unsigned long *receivedValues = new unsigned long[numReceiveValsInCol];
				
				unsigned long pulseCounter = 0;
				unsigned long pointCounter = 0;
				unsigned long transValCounter = 0;
				unsigned long receiveValCounter = 0;
				
				for(iterPulses = plsBuffer->begin(); iterPulses != plsBuffer->end(); ++iterPulses)
				{
					if(spdFile->getPulseVersion() == 1)
                    {
                        SPDPulseH5V1 *pulseObj = &((SPDPulseH5V1 *)spdPulses)[pulseCounter];
                        pulseUtils.copySPDPulseToSPDPulseH5((*iterPulses), pulseObj);
                        pulseObj->ptsStartIdx = (numPts + pointCounter);
                        pulseObj->transmittedStartIdx = (numTransVals + transValCounter);
                        pulseObj->receivedStartIdx = (numReceiveVals + receiveValCounter);
                    }
                    else if(spdFile->getPulseVersion() == 2)
                    {
                        SPDPulseH5V2 *pulseObj = &((SPDPulseH5V2 *)spdPulses)[pulseCounter];
                        pulseUtils.copySPDPulseToSPDPulseH5((*iterPulses), pulseObj);
                        pulseObj->ptsStartIdx = (numPts + pointCounter);
                        pulseObj->transmittedStartIdx = (numTransVals + transValCounter);
                        pulseObj->receivedStartIdx = (numReceiveVals + receiveValCounter);
                    }
					
                    if(firstPulse)
                    {
                        azMinWritten = (*iterPulses)->azimuth;
                        zenMinWritten = (*iterPulses)->zenith;
                        azMaxWritten = (*iterPulses)->azimuth;
                        zenMaxWritten = (*iterPulses)->zenith;
                        
                        firstPulse = false;
                    }
                    else
                    {
                        if((*iterPulses)->azimuth < azMinWritten)
                        {
                            azMinWritten = (*iterPulses)->azimuth;
                        }
                        else if((*iterPulses)->azimuth > azMaxWritten)
                        {
                            azMaxWritten = (*iterPulses)->azimuth;
                        }
                        
                        if((*iterPulses)->zenith < zenMinWritten)
                        {
                            zenMinWritten = (*iterPulses)->zenith;
                        }
                        else if((*iterPulses)->zenith > zenMaxWritten)
                        {
                            zenMaxWritten = (*iterPulses)->zenith;
                        }
                    }
                    
					for(boost::uint_fast16_t n = 0; n < (*iterPulses)->numberOfReturns; ++n)
					{
						if(spdFile->getPointVersion() == 1)
                        {
                            pointUtils.copySPDPointTo((*iterPulses)->pts->at(n), &((SPDPointH5V1 *)spdPoints)[pointCounter++]);
                        }
                        else if(spdFile->getPointVersion() == 2)
                        {
                            pointUtils.copySPDPointTo((*iterPulses)->pts->at(n), &((SPDPointH5V2 *)spdPoints)[pointCounter++]);
                        }
                        
                        if(firstReturn)
                        {
                            xMinWritten = (*iterPulses)->pts->at(n)->x;
                            yMinWritten = (*iterPulses)->pts->at(n)->y;
                            zMinWritten = (*iterPulses)->pts->at(n)->z;
                            xMaxWritten = (*iterPulses)->pts->at(n)->x;
                            yMaxWritten = (*iterPulses)->pts->at(n)->y;
                            zMaxWritten = (*iterPulses)->pts->at(n)->z;
                            
                            ranMinWritten = (*iterPulses)->pts->at(n)->range;
                            ranMaxWritten = (*iterPulses)->pts->at(n)->range;
                            
                            firstReturn = false;
                        }
                        else
                        {
                            if((*iterPulses)->pts->at(n)->x < xMinWritten)
                            {
                                xMinWritten = (*iterPulses)->pts->at(n)->x;
                            }
                            else if((*iterPulses)->pts->at(n)->x > xMaxWritten)
                            {
                                xMaxWritten = (*iterPulses)->pts->at(n)->x;
                            }
                            
                            if((*iterPulses)->pts->at(n)->y < yMinWritten)
                            {
                                yMinWritten = (*iterPulses)->pts->at(n)->y;
                            }
                            else if((*iterPulses)->pts->at(n)->y > yMaxWritten)
                            {
                                yMaxWritten = (*iterPulses)->pts->at(n)->y;
                            }
                            
                            if((*iterPulses)->pts->at(n)->z < zMinWritten)
                            {
                                zMinWritten = (*iterPulses)->pts->at(n)->z;
                            }
                            else if((*iterPulses)->pts->at(n)->z > zMaxWritten)
                            {
                                zMaxWritten = (*iterPulses)->pts->at(n)->z;
                            }
                            
                            if((*iterPulses)->pts->at(n)->range < ranMinWritten)
                            {
                                ranMinWritten = (*iterPulses)->pts->at(n)->range;
                            }
                            else if((*iterPulses)->pts->at(n)->range > ranMaxWritten)
                            {
                                ranMaxWritten = (*iterPulses)->pts->at(n)->range;
                            }
                        }
					}
					
					for(boost::uint_fast16_t n = 0; n < (*iterPulses)->numOfTransmittedBins; ++n)
					{
						transmittedValues[transValCounter++] = (*iterPulses)->transmitted[n];
					}
					
					for(boost::uint_fast16_t n = 0; n < (*iterPulses)->numOfReceivedBins; ++n)
					{
						receivedValues[receiveValCounter++] = (*iterPulses)->received[n];
					}
					
					++pulseCounter;
					SPDPulseUtils::deleteSPDPulse(*iterPulses);
				}
				plsBuffer->clear();
                
				// Write Pulses to disk
				hsize_t extendPulsesDatasetTo[1];
				extendPulsesDatasetTo[0] = this->numPulses + numPulsesInCol;
				pulsesDataset->extend( extendPulsesDatasetTo );
				
				hsize_t pulseDataOffset[1];
				pulseDataOffset[0] = this->numPulses;
				hsize_t pulseDataDims[1];
				pulseDataDims[0] = numPulsesInCol;
				
				DataSpace pulseWriteDataSpace = pulsesDataset->getSpace();
				pulseWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pulseDataDims, pulseDataOffset);
				DataSpace newPulsesDataspace = DataSpace(1, pulseDataDims);
				
				pulsesDataset->write(spdPulses, *spdPulseDataType, newPulsesDataspace, pulseWriteDataSpace);
				
				// Write Points to Disk
				if(numPointsInCol > 0)
				{
					hsize_t extendPointsDatasetTo[1];
					extendPointsDatasetTo[0] = this->numPts + numPointsInCol;
					pointsDataset->extend( extendPointsDatasetTo );
					
					hsize_t pointsDataOffset[1];
					pointsDataOffset[0] = this->numPts;
					hsize_t pointsDataDims[1];
					pointsDataDims[0] = numPointsInCol;
					
					DataSpace pointWriteDataSpace = pointsDataset->getSpace();
					pointWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pointsDataDims, pointsDataOffset);
					DataSpace newPointsDataspace = DataSpace(1, pointsDataDims);
					
					pointsDataset->write(spdPoints, *spdPointDataType, newPointsDataspace, pointWriteDataSpace);
				}
				
				// Write Transmitted Values to Disk
				if(numTransValsInCol > 0)
				{
					hsize_t extendTransDatasetTo[1];
					extendTransDatasetTo[0] = this->numTransVals + numTransValsInCol;
					transmittedDataset->extend( extendTransDatasetTo );
					
					hsize_t transDataOffset[1];
					transDataOffset[0] = this->numTransVals;
					hsize_t transDataDims[1];
					transDataDims[0] = numTransValsInCol;
					
					DataSpace transWriteDataSpace = transmittedDataset->getSpace();
					transWriteDataSpace.selectHyperslab(H5S_SELECT_SET, transDataDims, transDataOffset);
					DataSpace newTransDataspace = DataSpace(1, transDataDims);
					
					transmittedDataset->write(transmittedValues, PredType::NATIVE_ULONG, newTransDataspace, transWriteDataSpace);
				}
				
				// Write Recieved Values to Disk
				if(numReceiveValsInCol > 0)
				{
					hsize_t extendReceiveDatasetTo[1];
					extendReceiveDatasetTo[0] = this->numReceiveVals + numReceiveValsInCol;
					receivedDataset->extend( extendReceiveDatasetTo );
					
					hsize_t receivedDataOffset[1];
					receivedDataOffset[0] = this->numReceiveVals;
					hsize_t receivedDataDims[1];
					receivedDataDims[0] = numReceiveValsInCol;
					
					DataSpace receivedWriteDataSpace = receivedDataset->getSpace();
					receivedWriteDataSpace.selectHyperslab(H5S_SELECT_SET, receivedDataDims, receivedDataOffset);
					DataSpace newReceivedDataspace = DataSpace(1, receivedDataDims);
					
					receivedDataset->write(receivedValues, PredType::NATIVE_ULONG, newReceivedDataspace, receivedWriteDataSpace);
				}
				
				// Delete tempory arrarys once written to disk.
				if(spdFile->getPointVersion() == 1)
                {
                    delete[] reinterpret_cast<SPDPointH5V1*>(spdPoints);
                }
                else if(spdFile->getPointVersion() == 2)
                {
                    delete[] reinterpret_cast<SPDPointH5V2*>(spdPoints);
                }
                
                if(spdFile->getPulseVersion() == 1)
                {
                    delete[] reinterpret_cast<SPDPulseH5V1*>(spdPulses);
                }
                else if(spdFile->getPulseVersion() == 2)
                {
                    delete[] reinterpret_cast<SPDPulseH5V2*>(spdPulses);
                }
                
				delete[] transmittedValues;
				delete[] receivedValues;
				
				numPulses += numPulsesInCol;
				numPts += numPointsInCol;
				numTransVals += numTransValsInCol;
				numReceiveVals += numReceiveValsInCol;
			}            
		}
		catch( FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataTypeIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch(SPDIOException &e)
		{
			throw e;
		}
        
        
		
		try 
		{
            if(!firstReturn)
            {
                if(spdFile->getIndexType() == SPD_CARTESIAN_IDX)
                {
                    spdFile->setZMin(zMinWritten);
                    spdFile->setZMax(zMaxWritten);
                    spdFile->setBoundingVolumeSpherical(zenMinWritten, zenMaxWritten, azMinWritten, azMaxWritten, ranMinWritten, ranMaxWritten);
                }
                else if(spdFile->getIndexType() == SPD_SPHERICAL_IDX)
                {
                    spdFile->setBoundingVolume(xMinWritten, xMaxWritten, yMinWritten, yMaxWritten, zMinWritten, zMaxWritten);
                    spdFile->setRangeMin(ranMinWritten);
                    spdFile->setRangeMax(ranMaxWritten);
                }
                else if(spdFile->getIndexType() == SPD_POLAR_IDX)
                {
                    spdFile->setBoundingVolume(xMinWritten, xMaxWritten, yMinWritten, yMaxWritten, zMinWritten, zMaxWritten);
                    spdFile->setRangeMin(ranMinWritten);
                    spdFile->setRangeMax(ranMaxWritten);
                }
                else
                {
                    // Do nothing... 
                }
            }
            
			spdFile->setNumberOfPoints(numPts);
            spdFile->setNumberOfPulses(numPulses);
            
            spdFile->setFileType(SPD_NONSEQ_TYPE);
            
            spdFile->setIndexType(SPD_NO_IDX);
            
            this->writeHeaderInfo(spdOutH5File, spdFile);
            
			// Write attributes to Quicklook
			StrType strdatatypeLen6(PredType::C_S1, 6);
			StrType strdatatypeLen4(PredType::C_S1, 4);
			const H5std_string strClassVal ("IMAGE");
			const H5std_string strImgVerVal ("1.2");
			
			DataSpace attr_dataspace = DataSpace(H5S_SCALAR);
			
			Attribute classAttribute = datasetQuicklook->createAttribute(ATTRIBUTENAME_CLASS, strdatatypeLen6, attr_dataspace);
			classAttribute.write(strdatatypeLen6, strClassVal); 
			
			Attribute imgVerAttribute = datasetQuicklook->createAttribute(ATTRIBUTENAME_IMAGE_VERSION, strdatatypeLen4, attr_dataspace);
			imgVerAttribute.write(strdatatypeLen4, strImgVerVal);
			
			delete pulsesDataset;
			delete spdPulseDataType;
			delete pointsDataset;
			delete spdPointDataType;
			delete datasetPlsPerBin;
			delete datasetBinsOffset;
			delete receivedDataset;
			delete transmittedDataset;
			delete datasetQuicklook;
			
			spdOutH5File->flush(H5F_SCOPE_GLOBAL);
			spdOutH5File->close();	
			delete spdOutH5File;
			fileOpened = false;
		}
		catch( FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataTypeIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch(SPDIOException &e)
		{
			throw e;
		}
	}
    
	
	bool SPDNonSeqFileWriter::requireGrid()
	{
		return true;
	}
	
	bool SPDNonSeqFileWriter::needNumOutPts()
	{
		return false;
	}
	
	SPDNonSeqFileWriter::~SPDNonSeqFileWriter()
	{
		if(fileOpened)
		{
			try 
			{
				this->finaliseClose();
			}
			catch (SPDIOException &e) 
			{
				cerr << "WARNING: " << e.what() << endl;
			}
		}
	}
    
    
    
    
    
    
    
    SPDNoIdxFileWriter::SPDNoIdxFileWriter() : SPDDataExporter("UPD"), spdOutH5File(NULL), pulsesDataset(NULL), spdPulseDataType(NULL), pointsDataset(NULL), spdPointDataType(NULL), receivedDataset(NULL), transmittedDataset(NULL), numPulses(0), numPts(0), numTransVals(0), numReceiveVals(0)
	{
		
	}
    
	SPDNoIdxFileWriter::SPDNoIdxFileWriter(const SPDDataExporter &dataExporter) throw(SPDException) : SPDDataExporter(dataExporter), spdOutH5File(NULL), pulsesDataset(NULL), spdPulseDataType(NULL), pointsDataset(NULL), spdPointDataType(NULL), receivedDataset(NULL), transmittedDataset(NULL), numPulses(0), numPts(0), numTransVals(0), numReceiveVals(0)
	{
		if(fileOpened)
		{
			throw SPDException("Cannot make a copy of a file exporter when a file is open.");
		}
	}
    
	SPDNoIdxFileWriter& SPDNoIdxFileWriter::operator=(const SPDNoIdxFileWriter& dataExporter) throw(SPDException)
	{
		if(fileOpened)
		{
			throw SPDException("Cannot make a copy of a file exporter when a file is open.");
		}
		
		this->spdFile = dataExporter.spdFile;
		this->outputFile = dataExporter.outputFile;
		return *this;
	}
    
    SPDDataExporter* SPDNoIdxFileWriter::getInstance()
    {
        return new SPDNoIdxFileWriter();
    }
    
	bool SPDNoIdxFileWriter::open(SPDFile *spdFile, string outputFile) throw(SPDIOException)
	{
		SPDPulseUtils pulseUtils;
		SPDPointUtils pointUtils;
		this->spdFile = spdFile;
		this->outputFile = outputFile;
		
		const H5std_string spdFilePath( outputFile );
		
		try 
		{
			Exception::dontPrint();
			
			// Create File..
			spdOutH5File = new H5File( spdFilePath, H5F_ACC_TRUNC );
			
			// Create Groups..
			spdOutH5File->createGroup( GROUPNAME_HEADER );
			spdOutH5File->createGroup( GROUPNAME_DATA );
			
			// Create DataType, DataSpace and Dataset for Pulses
            hsize_t initDimsPulseDS[1];
			initDimsPulseDS[0] = 0;
			hsize_t maxDimsPulseDS[1];
			maxDimsPulseDS[0] = H5S_UNLIMITED;
			DataSpace pulseDataSpace = DataSpace(1, initDimsPulseDS, maxDimsPulseDS);
			
			hsize_t dimsPulseChunk[1];
			dimsPulseChunk[0] = spdFile->getPulseBlockSize();
			
			DSetCreatPropList creationPulseDSPList;
			creationPulseDSPList.setChunk(1, dimsPulseChunk);			
			creationPulseDSPList.setShuffle();
            creationPulseDSPList.setDeflate(SPD_DEFLATE);
            
            CompType *spdPulseDataTypeDisk = NULL;
            if(spdFile->getPulseVersion() == 1)
            {
                SPDPulseH5V1 spdPulse = SPDPulseH5V1();
                pulseUtils.initSPDPulseH5(&spdPulse);
                spdPulseDataTypeDisk = pulseUtils.createSPDPulseH5V1DataTypeDisk();
                spdPulseDataTypeDisk->pack();
                spdPulseDataType = pulseUtils.createSPDPulseH5V1DataTypeMemory();
                creationPulseDSPList.setFillValue( *spdPulseDataTypeDisk, &spdPulse);
            }
            else if(spdFile->getPulseVersion() == 2)
            {
                SPDPulseH5V2 spdPulse = SPDPulseH5V2();
                pulseUtils.initSPDPulseH5(&spdPulse);
                spdPulseDataTypeDisk = pulseUtils.createSPDPulseH5V2DataTypeDisk();
                spdPulseDataTypeDisk->pack();
                spdPulseDataType = pulseUtils.createSPDPulseH5V2DataTypeMemory();
                creationPulseDSPList.setFillValue( *spdPulseDataTypeDisk, &spdPulse);
            }
            else
            {
                throw SPDIOException("Did not recognise the Pulse version.");
            }
			pulsesDataset = new DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_PULSES, *spdPulseDataTypeDisk, pulseDataSpace, creationPulseDSPList));

			// Create DataType, DataSpace and Dataset for Points
            hsize_t initDimsPtsDS[1];
			initDimsPtsDS[0] = 0;
			hsize_t maxDimsPtsDS[1];
			maxDimsPtsDS[0] = H5S_UNLIMITED;
			DataSpace ptsDataSpace = DataSpace(1, initDimsPtsDS, maxDimsPtsDS);
			
			hsize_t dimsPtsChunk[1];
			dimsPtsChunk[0] = spdFile->getPointBlockSize();
			
			DSetCreatPropList creationPtsDSPList;
			creationPtsDSPList.setChunk(1, dimsPtsChunk);			
			creationPtsDSPList.setShuffle();
            creationPtsDSPList.setDeflate(SPD_DEFLATE);
            
            CompType *spdPointDataTypeDisk = NULL;
            if(spdFile->getPointVersion() == 1)
            {
                SPDPointH5V1 spdPoint = SPDPointH5V1();
                pointUtils.initSPDPoint(&spdPoint);
                spdPointDataTypeDisk = pointUtils.createSPDPointV1DataTypeDisk();
                spdPointDataTypeDisk->pack();
                spdPointDataType = pointUtils.createSPDPointV1DataTypeMemory();
                creationPtsDSPList.setFillValue( *spdPointDataTypeDisk, &spdPoint);
            }
            else if(spdFile->getPointVersion() == 2)
            {
                SPDPointH5V2 spdPoint = SPDPointH5V2();
                pointUtils.initSPDPoint(&spdPoint);
                spdPointDataTypeDisk = pointUtils.createSPDPointV2DataTypeDisk();
                spdPointDataTypeDisk->pack();
                spdPointDataType = pointUtils.createSPDPointV2DataTypeMemory();
                creationPtsDSPList.setFillValue( *spdPointDataTypeDisk, &spdPoint);
            }
            else
            {
                throw SPDIOException("Did not recognise the Point version");
            }
			pointsDataset = new DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_POINTS, *spdPointDataTypeDisk, ptsDataSpace, creationPtsDSPList));
			
			// Create transmitted and received DataSpace and Dataset
			hsize_t initDimsWaveformDS[1];
			initDimsWaveformDS[0] = 0;
			hsize_t maxDimsWaveformDS[1];
			maxDimsWaveformDS[0] = H5S_UNLIMITED;
			DataSpace waveformDataSpace = DataSpace(1, initDimsWaveformDS, maxDimsWaveformDS);
			
			hsize_t dimsReceivedChunk[1];
			dimsReceivedChunk[0] = spdFile->getReceivedBlockSize();
            
			hsize_t dimsTransmittedChunk[1];
			dimsTransmittedChunk[0] = spdFile->getTransmittedBlockSize();
			
            if(spdFile->getWaveformBitRes() == SPD_32_BIT_WAVE)
            {
                IntType intU32DataType( PredType::STD_U32LE );
                intU32DataType.setOrder( H5T_ORDER_LE );
                
                boost::uint_fast32_t fillValueUInt = 0;
                DSetCreatPropList creationReceivedDSPList;
                creationReceivedDSPList.setChunk(1, dimsReceivedChunk);			
                creationReceivedDSPList.setShuffle();
                creationReceivedDSPList.setDeflate(SPD_DEFLATE);
                creationReceivedDSPList.setFillValue( PredType::STD_U32LE, &fillValueUInt);
                
                DSetCreatPropList creationTransmittedDSPList;
                creationTransmittedDSPList.setChunk(1, dimsTransmittedChunk);			
                creationTransmittedDSPList.setShuffle();
                creationTransmittedDSPList.setDeflate(SPD_DEFLATE);
                creationTransmittedDSPList.setFillValue( PredType::STD_U32LE, &fillValueUInt);
                
                receivedDataset = new DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_RECEIVED, intU32DataType, waveformDataSpace, creationReceivedDSPList));
                transmittedDataset = new DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_TRANSMITTED, intU32DataType, waveformDataSpace, creationTransmittedDSPList));
            }
            else if(spdFile->getWaveformBitRes() == SPD_16_BIT_WAVE)
            {
                IntType intU16DataType( PredType::STD_U16LE );
                intU16DataType.setOrder( H5T_ORDER_LE );
                
                boost::uint_fast32_t fillValueUInt = 0;
                DSetCreatPropList creationReceivedDSPList;
                creationReceivedDSPList.setChunk(1, dimsReceivedChunk);			
                creationReceivedDSPList.setShuffle();
                creationReceivedDSPList.setDeflate(SPD_DEFLATE);
                creationReceivedDSPList.setFillValue( PredType::STD_U16LE, &fillValueUInt);
                
                DSetCreatPropList creationTransmittedDSPList;
                creationTransmittedDSPList.setChunk(1, dimsTransmittedChunk);			
                creationTransmittedDSPList.setShuffle();
                creationTransmittedDSPList.setDeflate(SPD_DEFLATE);
                creationTransmittedDSPList.setFillValue( PredType::STD_U16LE, &fillValueUInt);
                
                receivedDataset = new DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_RECEIVED, intU16DataType, waveformDataSpace, creationReceivedDSPList));
                transmittedDataset = new DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_TRANSMITTED, intU16DataType, waveformDataSpace, creationTransmittedDSPList));
            }
            else if(spdFile->getWaveformBitRes() == SPD_8_BIT_WAVE)
            {
                IntType intU8DataType( PredType::STD_U8LE );
                intU8DataType.setOrder( H5T_ORDER_LE );
                
                boost::uint_fast32_t fillValueUInt = 0;
                DSetCreatPropList creationReceivedDSPList;
                creationReceivedDSPList.setChunk(1, dimsReceivedChunk);			
                creationReceivedDSPList.setShuffle();
                creationReceivedDSPList.setDeflate(SPD_DEFLATE);
                creationReceivedDSPList.setFillValue( PredType::STD_U8LE, &fillValueUInt);
                
                DSetCreatPropList creationTransmittedDSPList;
                creationTransmittedDSPList.setChunk(1, dimsTransmittedChunk);			
                creationTransmittedDSPList.setShuffle();
                creationTransmittedDSPList.setDeflate(SPD_DEFLATE);
                creationTransmittedDSPList.setFillValue( PredType::STD_U8LE, &fillValueUInt);
                
                receivedDataset = new DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_RECEIVED, intU8DataType, waveformDataSpace, creationReceivedDSPList));
                transmittedDataset = new DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_TRANSMITTED, intU8DataType, waveformDataSpace, creationTransmittedDSPList));
            }
            else
            {
                throw SPDIOException("Waveform bit resolution is unknown.");
            }
            
			
			fileOpened = true;
			
			plsBuffer = new vector<SPDPulse*>();
            plsBuffer->reserve(spdFile->getPulseBlockSize());
		}
		catch( FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataTypeIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
        
        xMinWritten = 0;
        yMinWritten = 0;
        zMinWritten = 0;
        xMaxWritten = 0;
        yMaxWritten = 0;
        zMaxWritten = 0;
        azMinWritten = 0;
        zenMinWritten = 0;
        ranMinWritten = 0;
        azMaxWritten = 0;
        zenMaxWritten = 0;
        ranMaxWritten = 0;
        firstReturn = true;
        firstPulse = true;
        firstWaveform = true;
		
		return fileOpened;
	}
    
	void SPDNoIdxFileWriter::writeDataColumn(list<SPDPulse*> *plsIn, boost::uint_fast32_t col, boost::uint_fast32_t row)throw(SPDIOException)
	{
		SPDPulseUtils pulseUtils;
		SPDPointUtils pointUtils;
        
		if(!fileOpened)
		{
			throw SPDIOException("SPD (HDF5) file not open, cannot finalise.");
		}
		
		try 
		{
			Exception::dontPrint();
			
			list<SPDPulse*>::iterator iterInPls;
			for(iterInPls = plsIn->begin(); iterInPls != plsIn->end(); ++iterInPls)
			{
				plsBuffer->push_back(*iterInPls);
				if(plsBuffer->size() == spdFile->getPulseBlockSize() )
				{
					unsigned long numPulsesInCol = plsBuffer->size();
					unsigned long numPointsInCol = 0;
					unsigned long numTransValsInCol = 0;
					unsigned long numReceiveValsInCol = 0;
					
					vector<SPDPulse*>::iterator iterPulses;
					for(iterPulses = plsBuffer->begin(); iterPulses != plsBuffer->end(); ++iterPulses)
					{
						numPointsInCol += (*iterPulses)->numberOfReturns;
						numTransValsInCol += (*iterPulses)->numOfTransmittedBins;
						numReceiveValsInCol += (*iterPulses)->numOfReceivedBins;
					}
					
					void *spdPulses = NULL;
					void *spdPoints = NULL;
                    
                    if(spdFile->getPulseVersion() == 1)
                    {
                        spdPulses = new SPDPulseH5V1[numPulsesInCol];
                    }
                    else if(spdFile->getPulseVersion() == 2)
                    {
                        spdPulses = new SPDPulseH5V2[numPulsesInCol];
                    }
                    
                    if(spdFile->getPointVersion() == 1)
                    {
                        spdPoints = new SPDPointH5V1[numPointsInCol];
                    }
                    else if(spdFile->getPointVersion() == 2)
                    {
                        spdPoints = new SPDPointH5V2[numPointsInCol];
                    }
                    
					unsigned long *transmittedValues = new unsigned long[numTransValsInCol];
					unsigned long *receivedValues = new unsigned long[numReceiveValsInCol];
					
					unsigned long long pulseCounter = 0;
					unsigned long long pointCounter = 0;
					unsigned long long transValCounter = 0;
					unsigned long long receiveValCounter = 0;
                    
					for(iterPulses = plsBuffer->begin(); iterPulses != plsBuffer->end(); ++iterPulses)
					{
                        if(spdFile->getPulseVersion() == 1)
                        {
                            SPDPulseH5V1 *pulseObj = &((SPDPulseH5V1 *)spdPulses)[pulseCounter];
                            pulseUtils.copySPDPulseToSPDPulseH5((*iterPulses), pulseObj);
                            pulseObj->ptsStartIdx = (numPts + pointCounter);
                            pulseObj->transmittedStartIdx = (numTransVals + transValCounter);
                            pulseObj->receivedStartIdx = (numReceiveVals + receiveValCounter);
                        }
                        else if(spdFile->getPulseVersion() == 2)
                        {
                            SPDPulseH5V2 *pulseObj = &((SPDPulseH5V2 *)spdPulses)[pulseCounter];
                            pulseUtils.copySPDPulseToSPDPulseH5((*iterPulses), pulseObj);
                            pulseObj->ptsStartIdx = (numPts + pointCounter);
                            pulseObj->transmittedStartIdx = (numTransVals + transValCounter);
                            pulseObj->receivedStartIdx = (numReceiveVals + receiveValCounter);
                        }
						
                        if(firstPulse)
                        {
                            azMinWritten = (*iterPulses)->azimuth;
                            zenMinWritten = (*iterPulses)->zenith;
                            azMaxWritten = (*iterPulses)->azimuth;
                            zenMaxWritten = (*iterPulses)->zenith;
                            
                            firstPulse = false;
                        }
                        else
                        {
                            if((*iterPulses)->azimuth < azMinWritten)
                            {
                                azMinWritten = (*iterPulses)->azimuth;
                            }
                            else if((*iterPulses)->azimuth > azMaxWritten)
                            {
                                azMaxWritten = (*iterPulses)->azimuth;
                            }
                            
                            if((*iterPulses)->zenith < zenMinWritten)
                            {
                                zenMinWritten = (*iterPulses)->zenith;
                            }
                            else if((*iterPulses)->zenith > zenMaxWritten)
                            {
                                zenMaxWritten = (*iterPulses)->zenith;
                            }
                        }
                        
						for(boost::uint_fast16_t n = 0; n < (*iterPulses)->numberOfReturns; ++n)
						{
                            if(spdFile->getPointVersion() == 1)
                            {
                                pointUtils.copySPDPointTo((*iterPulses)->pts->at(n), &((SPDPointH5V1 *)spdPoints)[pointCounter++]);
                            }
                            else if(spdFile->getPointVersion() == 2)
                            {
                                pointUtils.copySPDPointTo((*iterPulses)->pts->at(n), &((SPDPointH5V2 *)spdPoints)[pointCounter++]);
                            }
                            
                            if(firstReturn)
                            {
                                xMinWritten = (*iterPulses)->pts->at(n)->x;
                                yMinWritten = (*iterPulses)->pts->at(n)->y;
                                zMinWritten = (*iterPulses)->pts->at(n)->z;
                                xMaxWritten = (*iterPulses)->pts->at(n)->x;
                                yMaxWritten = (*iterPulses)->pts->at(n)->y;
                                zMaxWritten = (*iterPulses)->pts->at(n)->z;
                                
                                ranMinWritten = (*iterPulses)->pts->at(n)->range;
                                ranMaxWritten = (*iterPulses)->pts->at(n)->range;
                                
                                firstReturn = false;
                            }
                            else
                            {
                                if((*iterPulses)->pts->at(n)->x < xMinWritten)
                                {
                                    xMinWritten = (*iterPulses)->pts->at(n)->x;
                                }
                                else if((*iterPulses)->pts->at(n)->x > xMaxWritten)
                                {
                                    xMaxWritten = (*iterPulses)->pts->at(n)->x;
                                }
                                
                                if((*iterPulses)->pts->at(n)->y < yMinWritten)
                                {
                                    yMinWritten = (*iterPulses)->pts->at(n)->y;
                                }
                                else if((*iterPulses)->pts->at(n)->y > yMaxWritten)
                                {
                                    yMaxWritten = (*iterPulses)->pts->at(n)->y;
                                }
                                
                                if((*iterPulses)->pts->at(n)->z < zMinWritten)
                                {
                                    zMinWritten = (*iterPulses)->pts->at(n)->z;
                                }
                                else if((*iterPulses)->pts->at(n)->z > zMaxWritten)
                                {
                                    zMaxWritten = (*iterPulses)->pts->at(n)->z;
                                }
                                
                                if((*iterPulses)->pts->at(n)->range < ranMinWritten)
                                {
                                    ranMinWritten = (*iterPulses)->pts->at(n)->range;
                                }
                                else if((*iterPulses)->pts->at(n)->range > ranMaxWritten)
                                {
                                    ranMaxWritten = (*iterPulses)->pts->at(n)->range;
                                }
                            }
						}
						
						for(boost::uint_fast16_t n = 0; n < (*iterPulses)->numOfTransmittedBins; ++n)
						{
							transmittedValues[transValCounter++] = (*iterPulses)->transmitted[n];
						}
						
						for(boost::uint_fast16_t n = 0; n < (*iterPulses)->numOfReceivedBins; ++n)
						{
							receivedValues[receiveValCounter++] = (*iterPulses)->received[n];
						}
						
						++pulseCounter;
						SPDPulseUtils::deleteSPDPulse(*iterPulses);
					}
					plsBuffer->clear();
                    
					// Write Pulses to disk
					hsize_t extendPulsesDatasetTo[1];
					extendPulsesDatasetTo[0] = this->numPulses + numPulsesInCol;
					pulsesDataset->extend( extendPulsesDatasetTo );
					
					hsize_t pulseDataOffset[1];
					pulseDataOffset[0] = this->numPulses;
					hsize_t pulseDataDims[1];
					pulseDataDims[0] = numPulsesInCol;
					
					DataSpace pulseWriteDataSpace = pulsesDataset->getSpace();
					pulseWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pulseDataDims, pulseDataOffset);
					DataSpace newPulsesDataspace = DataSpace(1, pulseDataDims);
					
					pulsesDataset->write(spdPulses, *spdPulseDataType, newPulsesDataspace, pulseWriteDataSpace);
					
					// Write Points to Disk
					if(numPointsInCol > 0)
					{
						hsize_t extendPointsDatasetTo[1];
						extendPointsDatasetTo[0] = this->numPts + numPointsInCol;
						pointsDataset->extend( extendPointsDatasetTo );
						
						hsize_t pointsDataOffset[1];
						pointsDataOffset[0] = this->numPts;
						hsize_t pointsDataDims[1];
						pointsDataDims[0] = numPointsInCol;
						
						DataSpace pointWriteDataSpace = pointsDataset->getSpace();
						pointWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pointsDataDims, pointsDataOffset);
						DataSpace newPointsDataspace = DataSpace(1, pointsDataDims);
						
						pointsDataset->write(spdPoints, *spdPointDataType, newPointsDataspace, pointWriteDataSpace);
					}
					
					// Write Transmitted Values to Disk
					if(numTransValsInCol > 0)
					{
						hsize_t extendTransDatasetTo[1];
						extendTransDatasetTo[0] = this->numTransVals + numTransValsInCol;
						transmittedDataset->extend( extendTransDatasetTo );
						
						hsize_t transDataOffset[1];
						transDataOffset[0] = this->numTransVals;
						hsize_t transDataDims[1];
						transDataDims[0] = numTransValsInCol;
						
						DataSpace transWriteDataSpace = transmittedDataset->getSpace();
						transWriteDataSpace.selectHyperslab(H5S_SELECT_SET, transDataDims, transDataOffset);
						DataSpace newTransDataspace = DataSpace(1, transDataDims);
						
						transmittedDataset->write(transmittedValues, PredType::NATIVE_ULONG, newTransDataspace, transWriteDataSpace);
					}
					
					// Write Recieved Values to Disk
					if(numReceiveValsInCol > 0)
					{
						hsize_t extendReceiveDatasetTo[1];
						extendReceiveDatasetTo[0] = this->numReceiveVals + numReceiveValsInCol;
						receivedDataset->extend( extendReceiveDatasetTo );
						
						hsize_t receivedDataOffset[1];
						receivedDataOffset[0] = this->numReceiveVals;
						hsize_t receivedDataDims[1];
						receivedDataDims[0] = numReceiveValsInCol;
						
						DataSpace receivedWriteDataSpace = receivedDataset->getSpace();
						receivedWriteDataSpace.selectHyperslab(H5S_SELECT_SET, receivedDataDims, receivedDataOffset);
						DataSpace newReceivedDataspace = DataSpace(1, receivedDataDims);
						
						receivedDataset->write(receivedValues, PredType::NATIVE_ULONG, newReceivedDataspace, receivedWriteDataSpace);
					}
					
					// Delete tempory arrarys once written to disk.
                    if(spdFile->getPointVersion() == 1)
                    {
                        delete[] reinterpret_cast<SPDPointH5V1*>(spdPoints);
                    }
                    else if(spdFile->getPointVersion() == 2)
                    {
                        delete[] reinterpret_cast<SPDPointH5V2*>(spdPoints);
                    }
                    
                    if(spdFile->getPulseVersion() == 1)
                    {
                        delete[] reinterpret_cast<SPDPulseH5V1*>(spdPulses);
                    }
                    else if(spdFile->getPulseVersion() == 2)
                    {
                        delete[] reinterpret_cast<SPDPulseH5V2*>(spdPulses);
                    }
                    
					delete[] transmittedValues;
					delete[] receivedValues;
					
					numPulses += numPulsesInCol;
					numPts += numPointsInCol;
					numTransVals += numTransValsInCol;
					numReceiveVals += numReceiveValsInCol;
				}
			}
			plsIn->clear();
			
		}
		catch( FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataTypeIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch(SPDIOException &e)
		{
			throw e;
		}
	}
	
	void SPDNoIdxFileWriter::writeDataColumn(vector<SPDPulse*> *plsIn, boost::uint_fast32_t col, boost::uint_fast32_t row)throw(SPDIOException)
	{
		SPDPulseUtils pulseUtils;
		SPDPointUtils pointUtils;
        
		if(!fileOpened)
		{
			throw SPDIOException("SPD (HDF5) file not open, cannot finalise.");
		}
		
        
		try 
		{
			Exception::dontPrint();
			
			vector<SPDPulse*>::iterator iterInPls;
			for(iterInPls = plsIn->begin(); iterInPls != plsIn->end(); ++iterInPls)
			{
				plsBuffer->push_back(*iterInPls);
				if(plsBuffer->size() == spdFile->getPulseBlockSize())
				{
                    //cout << "Writing buffer (" << numPulses <<  " Pulses)\n";
					unsigned long numPulsesInCol = plsBuffer->size();
					unsigned long numPointsInCol = 0;
					unsigned long numTransValsInCol = 0;
					unsigned long numReceiveValsInCol = 0;
					
					vector<SPDPulse*>::iterator iterPulses;
					for(iterPulses = plsBuffer->begin(); iterPulses != plsBuffer->end(); ++iterPulses)
					{
						numPointsInCol += (*iterPulses)->numberOfReturns;
						numTransValsInCol += (*iterPulses)->numOfTransmittedBins;
						numReceiveValsInCol += (*iterPulses)->numOfReceivedBins;
					}
					
					void *spdPulses = NULL;
					void *spdPoints = NULL;
                    
                    if(spdFile->getPulseVersion() == 1)
                    {
                        spdPulses = new SPDPulseH5V1[numPulsesInCol];
                    }
                    else if(spdFile->getPulseVersion() == 2)
                    {
                        spdPulses = new SPDPulseH5V2[numPulsesInCol];
                    }
                    
                    if(spdFile->getPointVersion() == 1)
                    {
                        spdPoints = new SPDPointH5V1[numPointsInCol];
                    }
                    else if(spdFile->getPointVersion() == 2)
                    {
                        spdPoints = new SPDPointH5V2[numPointsInCol];
                    }
                    
					unsigned long *transmittedValues = new unsigned long[numTransValsInCol];
					unsigned long *receivedValues = new unsigned long[numReceiveValsInCol];
					
					unsigned long long pulseCounter = 0;
					unsigned long long pointCounter = 0;
					unsigned long long transValCounter = 0;
					unsigned long long receiveValCounter = 0;
                    
					for(iterPulses = plsBuffer->begin(); iterPulses != plsBuffer->end(); ++iterPulses)
					{
                        if(spdFile->getPulseVersion() == 1)
                        {
                            SPDPulseH5V1 *pulseObj = &((SPDPulseH5V1 *)spdPulses)[pulseCounter];
                            pulseUtils.copySPDPulseToSPDPulseH5((*iterPulses), pulseObj);
                            pulseObj->ptsStartIdx = (numPts + pointCounter);
                            pulseObj->transmittedStartIdx = (numTransVals + transValCounter);
                            pulseObj->receivedStartIdx = (numReceiveVals + receiveValCounter);
                        }
                        else if(spdFile->getPulseVersion() == 2)
                        {
                            SPDPulseH5V2 *pulseObj = &((SPDPulseH5V2 *)spdPulses)[pulseCounter];
                            pulseUtils.copySPDPulseToSPDPulseH5((*iterPulses), pulseObj);
                            pulseObj->ptsStartIdx = (numPts + pointCounter);
                            pulseObj->transmittedStartIdx = (numTransVals + transValCounter);
                            pulseObj->receivedStartIdx = (numReceiveVals + receiveValCounter);
                        }
						
                        if(firstPulse)
                        {
                            azMinWritten = (*iterPulses)->azimuth;
                            zenMinWritten = (*iterPulses)->zenith;
                            azMaxWritten = (*iterPulses)->azimuth;
                            zenMaxWritten = (*iterPulses)->zenith;
                            
                            firstPulse = false;
                        }
                        else
                        {
                            if((*iterPulses)->azimuth < azMinWritten)
                            {
                                azMinWritten = (*iterPulses)->azimuth;
                            }
                            else if((*iterPulses)->azimuth > azMaxWritten)
                            {
                                azMaxWritten = (*iterPulses)->azimuth;
                            }
                            
                            if((*iterPulses)->zenith < zenMinWritten)
                            {
                                zenMinWritten = (*iterPulses)->zenith;
                            }
                            else if((*iterPulses)->zenith > zenMaxWritten)
                            {
                                zenMaxWritten = (*iterPulses)->zenith;
                            }
                        }
                        
						for(boost::uint_fast16_t n = 0; n < (*iterPulses)->numberOfReturns; ++n)
						{
                            if(spdFile->getPointVersion() == 1)
                            {
                                pointUtils.copySPDPointTo((*iterPulses)->pts->at(n), &((SPDPointH5V1 *)spdPoints)[pointCounter++]);
                            }
                            else if(spdFile->getPointVersion() == 2)
                            {
                                pointUtils.copySPDPointTo((*iterPulses)->pts->at(n), &((SPDPointH5V2 *)spdPoints)[pointCounter++]);
                            }
                            
                            if(firstReturn)
                            {
                                xMinWritten = (*iterPulses)->pts->at(n)->x;
                                yMinWritten = (*iterPulses)->pts->at(n)->y;
                                zMinWritten = (*iterPulses)->pts->at(n)->z;
                                xMaxWritten = (*iterPulses)->pts->at(n)->x;
                                yMaxWritten = (*iterPulses)->pts->at(n)->y;
                                zMaxWritten = (*iterPulses)->pts->at(n)->z;
                                
                                ranMinWritten = (*iterPulses)->pts->at(n)->range;
                                ranMaxWritten = (*iterPulses)->pts->at(n)->range;
                                
                                firstReturn = false;
                            }
                            else
                            {
                                if((*iterPulses)->pts->at(n)->x < xMinWritten)
                                {
                                    xMinWritten = (*iterPulses)->pts->at(n)->x;
                                }
                                else if((*iterPulses)->pts->at(n)->x > xMaxWritten)
                                {
                                    xMaxWritten = (*iterPulses)->pts->at(n)->x;
                                }
                                
                                if((*iterPulses)->pts->at(n)->y < yMinWritten)
                                {
                                    yMinWritten = (*iterPulses)->pts->at(n)->y;
                                }
                                else if((*iterPulses)->pts->at(n)->y > yMaxWritten)
                                {
                                    yMaxWritten = (*iterPulses)->pts->at(n)->y;
                                }
                                
                                if((*iterPulses)->pts->at(n)->z < zMinWritten)
                                {
                                    zMinWritten = (*iterPulses)->pts->at(n)->z;
                                }
                                else if((*iterPulses)->pts->at(n)->z > zMaxWritten)
                                {
                                    zMaxWritten = (*iterPulses)->pts->at(n)->z;
                                }
                                
                                if((*iterPulses)->pts->at(n)->range < ranMinWritten)
                                {
                                    ranMinWritten = (*iterPulses)->pts->at(n)->range;
                                }
                                else if((*iterPulses)->pts->at(n)->range > ranMaxWritten)
                                {
                                    ranMaxWritten = (*iterPulses)->pts->at(n)->range;
                                }
                            }
                            
                            /*cout << "Pulse " << ++counter << endl;
                            cout << "\txMinWritten = " << xMinWritten << endl;
                            cout << "\txMaxWritten = " << xMaxWritten << endl;
                            cout << "\tyMinWritten = " << yMinWritten << endl;
                            cout << "\tyMaxWritten = " << yMaxWritten << endl;
                            cout << "\tzMinWritten = " << zMinWritten << endl;
                            cout << "\tzMaxWritten = " << zMaxWritten << endl << endl;*/
						}
						
						for(boost::uint_fast16_t n = 0; n < (*iterPulses)->numOfTransmittedBins; ++n)
						{
							transmittedValues[transValCounter++] = (*iterPulses)->transmitted[n];
						}
						
						for(boost::uint_fast16_t n = 0; n < (*iterPulses)->numOfReceivedBins; ++n)
						{
							receivedValues[receiveValCounter++] = (*iterPulses)->received[n];
						}
						
						++pulseCounter;
						SPDPulseUtils::deleteSPDPulse(*iterPulses);
					}
                    plsBuffer->clear();
					
					// Write Pulses to disk
					hsize_t extendPulsesDatasetTo[1];
					extendPulsesDatasetTo[0] = this->numPulses + numPulsesInCol;
					pulsesDataset->extend( extendPulsesDatasetTo );
					
					hsize_t pulseDataOffset[1];
					pulseDataOffset[0] = this->numPulses;
					hsize_t pulseDataDims[1];
					pulseDataDims[0] = numPulsesInCol;
					
					DataSpace pulseWriteDataSpace = pulsesDataset->getSpace();
					pulseWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pulseDataDims, pulseDataOffset);
					DataSpace newPulsesDataspace = DataSpace(1, pulseDataDims);
					
					pulsesDataset->write(spdPulses, *spdPulseDataType, newPulsesDataspace, pulseWriteDataSpace);
					
					// Write Points to Disk
					if(numPointsInCol > 0)
					{
						hsize_t extendPointsDatasetTo[1];
						extendPointsDatasetTo[0] = this->numPts + numPointsInCol;
						pointsDataset->extend( extendPointsDatasetTo );
						
						hsize_t pointsDataOffset[1];
						pointsDataOffset[0] = this->numPts;
						hsize_t pointsDataDims[1];
						pointsDataDims[0] = numPointsInCol;
						
						DataSpace pointWriteDataSpace = pointsDataset->getSpace();
						pointWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pointsDataDims, pointsDataOffset);
						DataSpace newPointsDataspace = DataSpace(1, pointsDataDims);
						
						pointsDataset->write(spdPoints, *spdPointDataType, newPointsDataspace, pointWriteDataSpace);
					}
					
					// Write Transmitted Values to Disk
					if(numTransValsInCol > 0)
					{
						hsize_t extendTransDatasetTo[1];
						extendTransDatasetTo[0] = this->numTransVals + numTransValsInCol;
						transmittedDataset->extend( extendTransDatasetTo );
						
						hsize_t transDataOffset[1];
						transDataOffset[0] = this->numTransVals;
						hsize_t transDataDims[1];
						transDataDims[0] = numTransValsInCol;
						
						DataSpace transWriteDataSpace = transmittedDataset->getSpace();
						transWriteDataSpace.selectHyperslab(H5S_SELECT_SET, transDataDims, transDataOffset);
						DataSpace newTransDataspace = DataSpace(1, transDataDims);
						
						transmittedDataset->write(transmittedValues, PredType::NATIVE_ULONG, newTransDataspace, transWriteDataSpace);
					}
					
					// Write Recieved Values to Disk
					if(numReceiveValsInCol > 0)
					{
						hsize_t extendReceiveDatasetTo[1];
						extendReceiveDatasetTo[0] = this->numReceiveVals + numReceiveValsInCol;
						receivedDataset->extend( extendReceiveDatasetTo );
						
						hsize_t receivedDataOffset[1];
						receivedDataOffset[0] = this->numReceiveVals;
						hsize_t receivedDataDims[1];
						receivedDataDims[0] = numReceiveValsInCol;
						
						DataSpace receivedWriteDataSpace = receivedDataset->getSpace();
						receivedWriteDataSpace.selectHyperslab(H5S_SELECT_SET, receivedDataDims, receivedDataOffset);
						DataSpace newReceivedDataspace = DataSpace(1, receivedDataDims);
						
						receivedDataset->write(receivedValues, PredType::NATIVE_ULONG, newReceivedDataspace, receivedWriteDataSpace);
					}
					
					// Delete tempory arrarys once written to disk.
                    if(spdFile->getPointVersion() == 1)
                    {
                        delete[] reinterpret_cast<SPDPointH5V1*>(spdPoints);
                    }
                    else if(spdFile->getPointVersion() == 2)
                    {
                        delete[] reinterpret_cast<SPDPointH5V2*>(spdPoints);
                    }
                    
                    if(spdFile->getPulseVersion() == 1)
                    {
                        delete[] reinterpret_cast<SPDPulseH5V1*>(spdPulses);
                    }
                    else if(spdFile->getPulseVersion() == 2)
                    {
                        delete[] reinterpret_cast<SPDPulseH5V2*>(spdPulses);
                    }
                    
					delete[] transmittedValues;
					delete[] receivedValues;
					
					numPulses += numPulsesInCol;
					numPts += numPointsInCol;
					numTransVals += numTransValsInCol;
					numReceiveVals += numReceiveValsInCol;
				}                
			}
            plsBuffer->clear();
		}
		catch( FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataTypeIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch(SPDIOException &e)
		{
			throw e;
		}
	}
    
	void SPDNoIdxFileWriter::finaliseClose() throw(SPDIOException)
	{
		if(!fileOpened)
		{
			throw SPDIOException("SPD (HDF5) file not open, cannot finalise.");
		}
		
		SPDPulseUtils pulseUtils;
		SPDPointUtils pointUtils;
		
		try 
		{
			Exception::dontPrint();
			
			if(plsBuffer->size() > 0 )
			{
				unsigned long numPulsesInCol = plsBuffer->size();
				unsigned long numPointsInCol = 0;
				unsigned long numTransValsInCol = 0;
				unsigned long numReceiveValsInCol = 0;
				
				vector<SPDPulse*>::iterator iterPulses;
				for(iterPulses = plsBuffer->begin(); iterPulses != plsBuffer->end(); ++iterPulses)
				{
					numPointsInCol += (*iterPulses)->numberOfReturns;
					numTransValsInCol += (*iterPulses)->numOfTransmittedBins;
					numReceiveValsInCol += (*iterPulses)->numOfReceivedBins;
				}
                
                void *spdPulses = NULL;
                void *spdPoints = NULL;
                
                if(spdFile->getPulseVersion() == 1)
                {
                    spdPulses = new SPDPulseH5V1[numPulsesInCol];
                }
                else if(spdFile->getPulseVersion() == 2)
                {
                    spdPulses = new SPDPulseH5V2[numPulsesInCol];
                }
                
                if(spdFile->getPointVersion() == 1)
                {
                    spdPoints = new SPDPointH5V1[numPointsInCol];
                }
                else if(spdFile->getPointVersion() == 2)
                {
                    spdPoints = new SPDPointH5V2[numPointsInCol];
                }
                                
				unsigned long *transmittedValues = new unsigned long[numTransValsInCol];
				unsigned long *receivedValues = new unsigned long[numReceiveValsInCol];
				
				unsigned long pulseCounter = 0;
				unsigned long pointCounter = 0;
				unsigned long transValCounter = 0;
				unsigned long receiveValCounter = 0;
				
				for(iterPulses = plsBuffer->begin(); iterPulses != plsBuffer->end(); ++iterPulses)
				{
					if(spdFile->getPulseVersion() == 1)
                    {
                        SPDPulseH5V1 *pulseObj = &((SPDPulseH5V1 *)spdPulses)[pulseCounter];
                        pulseUtils.copySPDPulseToSPDPulseH5((*iterPulses), pulseObj);
                        pulseObj->ptsStartIdx = (numPts + pointCounter);
                        pulseObj->transmittedStartIdx = (numTransVals + transValCounter);
                        pulseObj->receivedStartIdx = (numReceiveVals + receiveValCounter);
                    }
                    else if(spdFile->getPulseVersion() == 2)
                    {
                        SPDPulseH5V2 *pulseObj = &((SPDPulseH5V2 *)spdPulses)[pulseCounter];
                        pulseUtils.copySPDPulseToSPDPulseH5((*iterPulses), pulseObj);
                        pulseObj->ptsStartIdx = (numPts + pointCounter);
                        pulseObj->transmittedStartIdx = (numTransVals + transValCounter);
                        pulseObj->receivedStartIdx = (numReceiveVals + receiveValCounter);
                    }
					
                    if(firstPulse)
                    {
                        azMinWritten = (*iterPulses)->azimuth;
                        zenMinWritten = (*iterPulses)->zenith;
                        azMaxWritten = (*iterPulses)->azimuth;
                        zenMaxWritten = (*iterPulses)->zenith;
                        
                        firstPulse = false;
                    }
                    else
                    {
                        if((*iterPulses)->azimuth < azMinWritten)
                        {
                            azMinWritten = (*iterPulses)->azimuth;
                        }
                        else if((*iterPulses)->azimuth > azMaxWritten)
                        {
                            azMaxWritten = (*iterPulses)->azimuth;
                        }
                        
                        if((*iterPulses)->zenith < zenMinWritten)
                        {
                            zenMinWritten = (*iterPulses)->zenith;
                        }
                        else if((*iterPulses)->zenith > zenMaxWritten)
                        {
                            zenMaxWritten = (*iterPulses)->zenith;
                        }
                    }
                    
					for(boost::uint_fast16_t n = 0; n < (*iterPulses)->numberOfReturns; ++n)
					{
						if(spdFile->getPointVersion() == 1)
                        {
                            pointUtils.copySPDPointTo((*iterPulses)->pts->at(n), &((SPDPointH5V1 *)spdPoints)[pointCounter++]);
                        }
                        else if(spdFile->getPointVersion() == 2)
                        {
                            pointUtils.copySPDPointTo((*iterPulses)->pts->at(n), &((SPDPointH5V2 *)spdPoints)[pointCounter++]);
                        }
                        
                        if(firstReturn)
                        {
                            xMinWritten = (*iterPulses)->pts->at(n)->x;
                            yMinWritten = (*iterPulses)->pts->at(n)->y;
                            zMinWritten = (*iterPulses)->pts->at(n)->z;
                            xMaxWritten = (*iterPulses)->pts->at(n)->x;
                            yMaxWritten = (*iterPulses)->pts->at(n)->y;
                            zMaxWritten = (*iterPulses)->pts->at(n)->z;
                            
                            ranMinWritten = (*iterPulses)->pts->at(n)->range;
                            ranMaxWritten = (*iterPulses)->pts->at(n)->range;
                            
                            firstReturn = false;
                        }
                        else
                        {
                            if((*iterPulses)->pts->at(n)->x < xMinWritten)
                            {
                                xMinWritten = (*iterPulses)->pts->at(n)->x;
                            }
                            else if((*iterPulses)->pts->at(n)->x > xMaxWritten)
                            {
                                xMaxWritten = (*iterPulses)->pts->at(n)->x;
                            }
                            
                            if((*iterPulses)->pts->at(n)->y < yMinWritten)
                            {
                                yMinWritten = (*iterPulses)->pts->at(n)->y;
                            }
                            else if((*iterPulses)->pts->at(n)->y > yMaxWritten)
                            {
                                yMaxWritten = (*iterPulses)->pts->at(n)->y;
                            }
                            
                            if((*iterPulses)->pts->at(n)->z < zMinWritten)
                            {
                                zMinWritten = (*iterPulses)->pts->at(n)->z;
                            }
                            else if((*iterPulses)->pts->at(n)->z > zMaxWritten)
                            {
                                zMaxWritten = (*iterPulses)->pts->at(n)->z;
                            }
                            
                            if((*iterPulses)->pts->at(n)->range < ranMinWritten)
                            {
                                ranMinWritten = (*iterPulses)->pts->at(n)->range;
                            }
                            else if((*iterPulses)->pts->at(n)->range > ranMaxWritten)
                            {
                                ranMaxWritten = (*iterPulses)->pts->at(n)->range;
                            }
                        }
					}
					
					for(boost::uint_fast16_t n = 0; n < (*iterPulses)->numOfTransmittedBins; ++n)
					{
						transmittedValues[transValCounter++] = (*iterPulses)->transmitted[n];
					}
					
					for(boost::uint_fast16_t n = 0; n < (*iterPulses)->numOfReceivedBins; ++n)
					{
						receivedValues[receiveValCounter++] = (*iterPulses)->received[n];
					}
					
					++pulseCounter;
					SPDPulseUtils::deleteSPDPulse(*iterPulses);
				}
                plsBuffer->clear();
				
				// Write Pulses to disk
				hsize_t extendPulsesDatasetTo[1];
				extendPulsesDatasetTo[0] = this->numPulses + numPulsesInCol;
				pulsesDataset->extend( extendPulsesDatasetTo );
				
				hsize_t pulseDataOffset[1];
				pulseDataOffset[0] = this->numPulses;
				hsize_t pulseDataDims[1];
				pulseDataDims[0] = numPulsesInCol;
				
				DataSpace pulseWriteDataSpace = pulsesDataset->getSpace();
				pulseWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pulseDataDims, pulseDataOffset);
				DataSpace newPulsesDataspace = DataSpace(1, pulseDataDims);
				
				pulsesDataset->write(spdPulses, *spdPulseDataType, newPulsesDataspace, pulseWriteDataSpace);
				
				// Write Points to Disk
				if(numPointsInCol > 0)
				{
					hsize_t extendPointsDatasetTo[1];
					extendPointsDatasetTo[0] = this->numPts + numPointsInCol;
					pointsDataset->extend( extendPointsDatasetTo );
					
					hsize_t pointsDataOffset[1];
					pointsDataOffset[0] = this->numPts;
					hsize_t pointsDataDims[1];
					pointsDataDims[0] = numPointsInCol;
					
					DataSpace pointWriteDataSpace = pointsDataset->getSpace();
					pointWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pointsDataDims, pointsDataOffset);
					DataSpace newPointsDataspace = DataSpace(1, pointsDataDims);
					
					pointsDataset->write(spdPoints, *spdPointDataType, newPointsDataspace, pointWriteDataSpace);
				}
				
				// Write Transmitted Values to Disk
				if(numTransValsInCol > 0)
				{
					hsize_t extendTransDatasetTo[1];
					extendTransDatasetTo[0] = this->numTransVals + numTransValsInCol;
					transmittedDataset->extend( extendTransDatasetTo );
					
					hsize_t transDataOffset[1];
					transDataOffset[0] = this->numTransVals;
					hsize_t transDataDims[1];
					transDataDims[0] = numTransValsInCol;
					
					DataSpace transWriteDataSpace = transmittedDataset->getSpace();
					transWriteDataSpace.selectHyperslab(H5S_SELECT_SET, transDataDims, transDataOffset);
					DataSpace newTransDataspace = DataSpace(1, transDataDims);
					
					transmittedDataset->write(transmittedValues, PredType::NATIVE_ULONG, newTransDataspace, transWriteDataSpace);
				}
				
				// Write Recieved Values to Disk
				if(numReceiveValsInCol > 0)
				{
					hsize_t extendReceiveDatasetTo[1];
					extendReceiveDatasetTo[0] = this->numReceiveVals + numReceiveValsInCol;
					receivedDataset->extend( extendReceiveDatasetTo );
					
					hsize_t receivedDataOffset[1];
					receivedDataOffset[0] = this->numReceiveVals;
					hsize_t receivedDataDims[1];
					receivedDataDims[0] = numReceiveValsInCol;
					
					DataSpace receivedWriteDataSpace = receivedDataset->getSpace();
					receivedWriteDataSpace.selectHyperslab(H5S_SELECT_SET, receivedDataDims, receivedDataOffset);
					DataSpace newReceivedDataspace = DataSpace(1, receivedDataDims);
					
					receivedDataset->write(receivedValues, PredType::NATIVE_ULONG, newReceivedDataspace, receivedWriteDataSpace);
				}
				
				// Delete tempory arrarys once written to disk.
				if(spdFile->getPointVersion() == 1)
                {
                    delete[] reinterpret_cast<SPDPointH5V1*>(spdPoints);
                }
                else if(spdFile->getPointVersion() == 2)
                {
                    delete[] reinterpret_cast<SPDPointH5V2*>(spdPoints);
                }
                
                if(spdFile->getPulseVersion() == 1)
                {
                    delete[] reinterpret_cast<SPDPulseH5V1*>(spdPulses);
                }
                else if(spdFile->getPulseVersion() == 2)
                {
                    delete[] reinterpret_cast<SPDPulseH5V2*>(spdPulses);
                }

				delete[] transmittedValues;
				delete[] receivedValues;
				
				numPulses += numPulsesInCol;
				numPts += numPointsInCol;
				numTransVals += numTransValsInCol;
				numReceiveVals += numReceiveValsInCol;
			}            
		}
		catch( FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataTypeIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch(SPDIOException &e)
		{
			throw e;
		}
				
		try 
		{
            if(!firstPulse)
            {
                spdFile->setAzimuthMax(azMaxWritten);
                spdFile->setAzimuthMin(azMinWritten);
                spdFile->setZenithMax(zenMaxWritten);
                spdFile->setZenithMin(zenMinWritten);
            }
            
            if(!firstReturn)
            {
                spdFile->setBoundingVolume(xMinWritten, xMaxWritten, yMinWritten, yMaxWritten, zMinWritten, zMaxWritten);
                spdFile->setBoundingVolumeSpherical(zenMinWritten, zenMaxWritten, azMinWritten, azMaxWritten, ranMinWritten, ranMaxWritten);
            }
            
            spdFile->setNumberOfPoints(numPts);
            spdFile->setNumberOfPulses(numPulses);
            
            spdFile->setFileType(SPD_UPD_TYPE);
            
            spdFile->setIndexType(SPD_NO_IDX);
            
            //cout << "spdFile:\n" << spdFile << endl;
            
            this->writeHeaderInfo(spdOutH5File, spdFile);
            
            delete pulsesDataset;
			delete spdPulseDataType;
			delete pointsDataset;
			delete spdPointDataType;
			delete receivedDataset;
			delete transmittedDataset;
			
			spdOutH5File->flush(H5F_SCOPE_GLOBAL);
			spdOutH5File->close();	
			delete spdOutH5File;
			fileOpened = false;
		}
		catch( FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataTypeIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch(SPDIOException &e)
		{
			throw e;
		}
	}
    
	
    
	bool SPDNoIdxFileWriter::requireGrid()
	{
		return false;
	}
    
	bool SPDNoIdxFileWriter::needNumOutPts()
	{
		return false;
	}
    
	SPDNoIdxFileWriter::~SPDNoIdxFileWriter()
	{
		if(fileOpened)
		{
			try 
			{
				this->finaliseClose();
			}
			catch (SPDIOException &e) 
			{
				cerr << "WARNING: " << e.what() << endl;
			}
		}
	}
}


