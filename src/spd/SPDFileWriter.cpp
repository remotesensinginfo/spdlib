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
    
    void SPDFileWriter::writeHeaderInfo(H5::H5File* spdOutH5File, SPDFile *spdFile) 
    {
        try 
		{
			H5::IntType int16bitDataTypeDisk( H5::PredType::STD_I16LE );
			H5::IntType uint16bitDataTypeDisk( H5::PredType::STD_U16LE );
            H5::IntType uint32bitDataType( H5::PredType::STD_U32LE );
			H5::IntType uint64bitDataTypeDisk( H5::PredType::STD_U64LE );
			H5::FloatType floatDataTypeDisk( H5::PredType::IEEE_F32LE );
			H5::FloatType doubleDataTypeDisk( H5::PredType::IEEE_F64LE );
			
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
			H5::DataSpace dataspaceStrAll(rankStr, dims1Str);
			H5::StrType strTypeAll(0, H5T_VARIABLE);
			
			hsize_t dimsValue[1];
			dimsValue[0] = 1;
			H5::DataSpace singleValueDataspace(1, dimsValue);
			
			if((H5T_STRING!=H5Tget_class(strTypeAll.getId())) || (!H5Tis_variable_str(strTypeAll.getId())))
			{
				throw SPDIOException("The string data type defined is not variable.");
			}
            
            H5::DataSet datasetSpatialReference = spdOutH5File->createDataSet(SPDFILE_DATASETNAME_SPATIAL_REFERENCE, strTypeAll, dataspaceStrAll);
            std::string spatRefStr = spdFile->getSpatialReference();
            char *spatRefCStr = new char [spatRefStr.length()+1];
            std::strcpy (spatRefCStr, spatRefStr.c_str());
			datasetSpatialReference.write((void*)&spatRefCStr, strTypeAll);
			datasetSpatialReference.close();
			delete[] spatRefCStr;
			
            H5::DataSet datasetFileType = spdOutH5File->createDataSet(SPDFILE_DATASETNAME_FILE_TYPE, uint16bitDataTypeDisk, singleValueDataspace);
            out16bitUintDataValue[0] = spdFile->getFileType();
			datasetFileType.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
            
            H5::DataSet datasetPulseIndexMethod = spdOutH5File->createDataSet(SPDFILE_DATASETNAME_PULSE_INDEX_METHOD, uint16bitDataTypeDisk, singleValueDataspace);
            out16bitUintDataValue[0] = spdFile->getIndexType();
			datasetPulseIndexMethod.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
            
            H5::DataSet datasetDiscreteDefined = spdOutH5File->createDataSet(SPDFILE_DATASETNAME_DISCRETE_PT_DEFINED, int16bitDataTypeDisk, singleValueDataspace);
			out16bitintDataValue[0] = spdFile->getDiscretePtDefined();
			datasetDiscreteDefined.write( out16bitintDataValue, H5::PredType::NATIVE_INT );
			
            H5::DataSet datasetDecomposedDefined = spdOutH5File->createDataSet(SPDFILE_DATASETNAME_DECOMPOSED_PT_DEFINED, int16bitDataTypeDisk, singleValueDataspace);
			out16bitintDataValue[0] = spdFile->getDecomposedPtDefined();
			datasetDecomposedDefined.write( out16bitintDataValue, H5::PredType::NATIVE_INT );
			
            H5::DataSet datasetTransWaveformDefined = spdOutH5File->createDataSet(SPDFILE_DATASETNAME_TRANS_WAVEFORM_DEFINED, int16bitDataTypeDisk, singleValueDataspace);
			out16bitintDataValue[0] = spdFile->getTransWaveformDefined();
			datasetTransWaveformDefined.write( out16bitintDataValue, H5::PredType::NATIVE_INT );
            
            H5::DataSet datasetReceiveWaveformDefined = spdOutH5File->createDataSet(SPDFILE_DATASETNAME_RECEIVE_WAVEFORM_DEFINED, int16bitDataTypeDisk, singleValueDataspace);
            out16bitintDataValue[0] = spdFile->getReceiveWaveformDefined();
			datasetReceiveWaveformDefined.write( out16bitintDataValue, H5::PredType::NATIVE_INT );
            
            H5::DataSet datasetIndexType = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_INDEX_TYPE, uint16bitDataTypeDisk, singleValueDataspace);
            out16bitUintDataValue[0] = spdFile->getIndexType();
			datasetIndexType.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
            
            H5::DataSet datasetMajorVersion = spdOutH5File->createDataSet(SPDFILE_DATASETNAME_MAJOR_VERSION, uint16bitDataTypeDisk, singleValueDataspace);
			out16bitUintDataValue[0] = spdFile->getMajorSPDVersion();
			datasetMajorVersion.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
			
            H5::DataSet datasetMinorVersion = spdOutH5File->createDataSet(SPDFILE_DATASETNAME_MINOR_VERSION, uint16bitDataTypeDisk, singleValueDataspace);
			out16bitUintDataValue[0] = spdFile->getMinorSPDVersion();
			datasetMinorVersion.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
            
            H5::DataSet datasetPointVersion = spdOutH5File->createDataSet(SPDFILE_DATASETNAME_POINT_VERSION, uint16bitDataTypeDisk, singleValueDataspace);
            out16bitUintDataValue[0] = spdFile->getPointVersion();
			datasetPointVersion.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
            
            H5::DataSet datasetPulseVersion = spdOutH5File->createDataSet(SPDFILE_DATASETNAME_PULSE_VERSION, uint16bitDataTypeDisk, singleValueDataspace);
            out16bitUintDataValue[0] = spdFile->getPulseVersion();
			datasetPulseVersion.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
			
            H5::DataSet datasetGeneratingSoftware = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_GENERATING_SOFTWARE, strTypeAll, dataspaceStrAll);
			wStrdata = new const char*[numLinesStr];
			wStrdata[0] = spdFile->getGeneratingSoftware().c_str();			
			datasetGeneratingSoftware.write((void*)wStrdata, strTypeAll);
			datasetGeneratingSoftware.close();
			delete[] wStrdata;
			
            H5::DataSet datasetSystemIdentifier = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_SYSTEM_IDENTIFIER, strTypeAll, dataspaceStrAll);
			wStrdata = new const char*[numLinesStr];
			wStrdata[0] = spdFile->getSystemIdentifier().c_str();
			datasetSystemIdentifier.write((void*)wStrdata, strTypeAll);
			datasetSystemIdentifier.close();
			delete[] wStrdata;
			
            H5::DataSet datasetFileSignature = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_FILE_SIGNATURE, strTypeAll, dataspaceStrAll);
			wStrdata = new const char*[numLinesStr];
			wStrdata[0] = spdFile->getFileSignature().c_str();			
			datasetFileSignature.write((void*)wStrdata, strTypeAll);
			datasetFileSignature.close();
			delete[] wStrdata;
			
            H5::DataSet datasetYearOfCreation = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_YEAR_OF_CREATION, uint16bitDataTypeDisk, singleValueDataspace);
			out16bitUintDataValue[0] = spdFile->getYearOfCreation();
			datasetYearOfCreation.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
			
            H5::DataSet datasetMonthOfCreation = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_MONTH_OF_CREATION, uint16bitDataTypeDisk, singleValueDataspace);
			out16bitUintDataValue[0] = spdFile->getMonthOfCreation();
			datasetMonthOfCreation.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
			
            H5::DataSet datasetDayOfCreation = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_DAY_OF_CREATION, uint16bitDataTypeDisk, singleValueDataspace);
			out16bitUintDataValue[0] = spdFile->getDayOfCreation();
			datasetDayOfCreation.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
			
            H5::DataSet datasetHourOfCreation = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_HOUR_OF_CREATION, uint16bitDataTypeDisk, singleValueDataspace);
			out16bitUintDataValue[0] = spdFile->getHourOfCreation();
			datasetHourOfCreation.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
			
            H5::DataSet datasetMinuteOfCreation = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_MINUTE_OF_CREATION, uint16bitDataTypeDisk, singleValueDataspace);
			out16bitUintDataValue[0] = spdFile->getMinuteOfCreation();
			datasetMinuteOfCreation.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
			
            H5::DataSet datasetSecondOfCreation = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_SECOND_OF_CREATION, uint16bitDataTypeDisk, singleValueDataspace);
			out16bitUintDataValue[0] = spdFile->getSecondOfCreation();
			datasetSecondOfCreation.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
			
            H5::DataSet datasetYearOfCapture = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_YEAR_OF_CAPTURE, uint16bitDataTypeDisk, singleValueDataspace);
			out16bitUintDataValue[0] = spdFile->getYearOfCapture();
			datasetYearOfCapture.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
			
            H5::DataSet datasetMonthOfCapture = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_MONTH_OF_CAPTURE, uint16bitDataTypeDisk, singleValueDataspace);
			out16bitUintDataValue[0] = spdFile->getMonthOfCapture();
			datasetMonthOfCapture.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
			
            H5::DataSet datasetDayOfCapture = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_DAY_OF_CAPTURE, uint16bitDataTypeDisk, singleValueDataspace);
			out16bitUintDataValue[0] = spdFile->getDayOfCapture();
			datasetDayOfCapture.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
			
            H5::DataSet datasetHourOfCapture = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_HOUR_OF_CAPTURE, uint16bitDataTypeDisk, singleValueDataspace);
			out16bitUintDataValue[0] = spdFile->getHourOfCapture();
			datasetHourOfCapture.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
			
            H5::DataSet datasetMinuteOfCapture = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_MINUTE_OF_CAPTURE, uint16bitDataTypeDisk, singleValueDataspace);
			out16bitUintDataValue[0] = spdFile->getMinuteOfCapture();
			datasetMinuteOfCapture.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
			
            H5::DataSet datasetSecondOfCapture = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_SECOND_OF_CAPTURE, uint16bitDataTypeDisk, singleValueDataspace);
			out16bitUintDataValue[0] = spdFile->getSecondOfCapture();
			datasetSecondOfCapture.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
			
            H5::DataSet datasetNumberOfPoints = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_NUMBER_OF_POINTS, uint64bitDataTypeDisk, singleValueDataspace);
			out64bitUintDataValue[0] = spdFile->getNumberOfPoints();
			datasetNumberOfPoints.write( out64bitUintDataValue, H5::PredType::NATIVE_ULLONG );
			
            H5::DataSet datasetNumberOfPulses = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_NUMBER_OF_PULSES, uint64bitDataTypeDisk, singleValueDataspace);
			out64bitUintDataValue[0] = spdFile->getNumberOfPulses();
			datasetNumberOfPulses.write( out64bitUintDataValue, H5::PredType::NATIVE_ULLONG );
			
            H5::DataSet datasetUserMetaData = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_USER_META_DATA, strTypeAll, dataspaceStrAll);
			wStrdata = new const char*[numLinesStr];
			wStrdata[0] = spdFile->getUserMetaField().c_str();			
			datasetUserMetaData.write((void*)wStrdata, strTypeAll);
			datasetUserMetaData.close();
			delete[] wStrdata;
            
            H5::DataSet datasetXMin = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_X_MIN, doubleDataTypeDisk, singleValueDataspace);
			outDoubleDataValue[0] = spdFile->getXMin();
			datasetXMin.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
            
            H5::DataSet datasetXMax = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_X_MAX, doubleDataTypeDisk, singleValueDataspace);
			outDoubleDataValue[0] = spdFile->getXMax();
			datasetXMax.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
			
            H5::DataSet datasetYMin = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_Y_MIN, doubleDataTypeDisk, singleValueDataspace);
			outDoubleDataValue[0] = spdFile->getYMin();
			datasetYMin.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
			
            H5::DataSet datasetYMax = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_Y_MAX, doubleDataTypeDisk, singleValueDataspace);
			outDoubleDataValue[0] = spdFile->getYMax();
			datasetYMax.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
			
            H5::DataSet datasetZMin = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_Z_MIN, doubleDataTypeDisk, singleValueDataspace);
			outDoubleDataValue[0] = spdFile->getZMin();
			datasetZMin.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
			
            H5::DataSet datasetZMax = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_Z_MAX, doubleDataTypeDisk, singleValueDataspace);
			outDoubleDataValue[0] = spdFile->getZMax();
			datasetZMax.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
            
            H5::DataSet datasetZenithMin = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_ZENITH_MIN, doubleDataTypeDisk, singleValueDataspace);
			outDoubleDataValue[0] = spdFile->getZenithMin();
			datasetZenithMin.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
			
            H5::DataSet datasetZenithMax = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_ZENITH_MAX, doubleDataTypeDisk, singleValueDataspace);
			outDoubleDataValue[0] = spdFile->getZenithMax();
			datasetZenithMax.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
			
            H5::DataSet datasetAzimuthMin = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_AZIMUTH_MIN, doubleDataTypeDisk, singleValueDataspace);
			outDoubleDataValue[0] = spdFile->getAzimuthMin();
			datasetAzimuthMin.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
			
            H5::DataSet datasetAzimuthMax = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_AZIMUTH_MAX, doubleDataTypeDisk, singleValueDataspace);
			outDoubleDataValue[0] = spdFile->getAzimuthMax();
			datasetAzimuthMax.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
			
            H5::DataSet datasetRangeMin = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_RANGE_MIN, doubleDataTypeDisk, singleValueDataspace);
			outDoubleDataValue[0] = spdFile->getRangeMin();
			datasetRangeMin.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
			
            H5::DataSet datasetRangeMax = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_RANGE_MAX, doubleDataTypeDisk, singleValueDataspace);
			outDoubleDataValue[0] = spdFile->getRangeMax();
			datasetRangeMax.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
            
            H5::DataSet datasetScanlineMin = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_SCANLINE_MIN, doubleDataTypeDisk, singleValueDataspace);
			outDoubleDataValue[0] = spdFile->getScanlineMin();
			datasetScanlineMin.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
			
            H5::DataSet datasetScanlineMax = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_SCANLINE_MAX, doubleDataTypeDisk, singleValueDataspace);
			outDoubleDataValue[0] = spdFile->getScanlineMax();
			datasetScanlineMax.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
			
            H5::DataSet datasetScanlineIdxMin = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_SCANLINE_IDX_MIN, doubleDataTypeDisk, singleValueDataspace);
			outDoubleDataValue[0] = spdFile->getScanlineIdxMin();
			datasetScanlineIdxMin.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
			
            H5::DataSet datasetScanlineIdxMax = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_SCANLINE_IDX_MAX, doubleDataTypeDisk, singleValueDataspace);
			outDoubleDataValue[0] = spdFile->getScanlineIdxMax();
			datasetScanlineIdxMax.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
            
            H5::DataSet datasetPulseRepFreq = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_PULSE_REPETITION_FREQ, floatDataTypeDisk, singleValueDataspace);
			outFloatDataValue[0] = spdFile->getPulseRepetitionFreq();
			datasetPulseRepFreq.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
			
            H5::DataSet datasetBeamDivergence = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_BEAM_DIVERGENCE, floatDataTypeDisk, singleValueDataspace);
			outFloatDataValue[0] = spdFile->getBeamDivergence();
			datasetBeamDivergence.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
			
            H5::DataSet datasetSensorHeight = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_SENSOR_HEIGHT, doubleDataTypeDisk, singleValueDataspace);
			outDoubleDataValue[0] = spdFile->getSensorHeight();
			datasetSensorHeight.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
			
            H5::DataSet datasetFootprint = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_FOOTPRINT, floatDataTypeDisk, singleValueDataspace);
			outFloatDataValue[0] = spdFile->getFootprint();
			datasetFootprint.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
			
            H5::DataSet datasetMaxScanAngle = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_MAX_SCAN_ANGLE, floatDataTypeDisk, singleValueDataspace);
			outFloatDataValue[0] = spdFile->getMaxScanAngle();
			datasetMaxScanAngle.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
			
            H5::DataSet datasetRGBDefined = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_RGB_DEFINED, int16bitDataTypeDisk, singleValueDataspace);
			out16bitintDataValue[0] = spdFile->getRGBDefined();
			datasetRGBDefined.write( out16bitintDataValue, H5::PredType::NATIVE_INT );
			
            H5::DataSet datasetPulseBlockSize = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_PULSE_BLOCK_SIZE, uint16bitDataTypeDisk, singleValueDataspace);
			out16bitUintDataValue[0] = spdFile->getPulseBlockSize();
			datasetPulseBlockSize.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
			
            H5::DataSet datasetPointsBlockSize = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_POINT_BLOCK_SIZE, uint16bitDataTypeDisk, singleValueDataspace);
			out16bitUintDataValue[0] = spdFile->getPointBlockSize();
			datasetPointsBlockSize.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
			
            H5::DataSet datasetReceivedBlockSize = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_RECEIVED_BLOCK_SIZE, uint16bitDataTypeDisk, singleValueDataspace);
			out16bitUintDataValue[0] = spdFile->getReceivedBlockSize();
			datasetReceivedBlockSize.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
			
            H5::DataSet datasetTransmittedBlockSize = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_TRANSMITTED_BLOCK_SIZE, uint16bitDataTypeDisk, singleValueDataspace);
			out16bitUintDataValue[0] = spdFile->getTransmittedBlockSize();
			datasetTransmittedBlockSize.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
            
            H5::DataSet datasetWaveformBitRes = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_WAVEFORM_BIT_RES, uint16bitDataTypeDisk, singleValueDataspace);
            out16bitUintDataValue[0] = spdFile->getWaveformBitRes();
			datasetWaveformBitRes.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
            
            H5::DataSet datasetTemporalBinSpacing = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_TEMPORAL_BIN_SPACING, doubleDataTypeDisk, singleValueDataspace);
			outDoubleDataValue[0] = spdFile->getTemporalBinSpacing();
			datasetTemporalBinSpacing.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
			
            H5::DataSet datasetReturnNumsSynGen = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_RETURN_NUMBERS_SYN_GEN, int16bitDataTypeDisk, singleValueDataspace);
			out16bitintDataValue[0] = spdFile->getReturnNumsSynGen();
			datasetReturnNumsSynGen.write( out16bitintDataValue, H5::PredType::NATIVE_INT );
            
            H5::DataSet datasetHeightDefined = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_HEIGHT_DEFINED, int16bitDataTypeDisk, singleValueDataspace);
			out16bitintDataValue[0] = spdFile->getHeightDefined();
			datasetHeightDefined.write( out16bitintDataValue, H5::PredType::NATIVE_INT );
			
            H5::DataSet datasetSensorSpeed = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_SENSOR_SPEED, floatDataTypeDisk, singleValueDataspace);
			outFloatDataValue[0] = spdFile->getSensorSpeed();
			datasetSensorSpeed.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
			
            H5::DataSet datasetSensorScanRate = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_SENSOR_SCAN_RATE, floatDataTypeDisk, singleValueDataspace);
			outFloatDataValue[0] = spdFile->getSensorScanRate();
			datasetSensorScanRate.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
			
            H5::DataSet datasetPointDensity = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_POINT_DENSITY, floatDataTypeDisk, singleValueDataspace);
			outFloatDataValue[0] = spdFile->getPointDensity();
			datasetPointDensity.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
			
            H5::DataSet datasetPulseDensity = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_PULSE_DENSITY, floatDataTypeDisk, singleValueDataspace);
			outFloatDataValue[0] = spdFile->getPulseDensity();
			datasetPulseDensity.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
			
            H5::DataSet datasetPulseCrossTrackSpacing = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_PULSE_CROSS_TRACK_SPACING, floatDataTypeDisk, singleValueDataspace);
			outFloatDataValue[0] = spdFile->getPulseCrossTrackSpacing();
			datasetPulseCrossTrackSpacing.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
			
            H5::DataSet datasetPulseAlongTrackSpacing = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_PULSE_ALONG_TRACK_SPACING, floatDataTypeDisk, singleValueDataspace);
			outFloatDataValue[0] = spdFile->getPulseAlongTrackSpacing();
			datasetPulseAlongTrackSpacing.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
			
            H5::DataSet datasetOriginDefined = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_ORIGIN_DEFINED, int16bitDataTypeDisk, singleValueDataspace);
			out16bitintDataValue[0] = spdFile->getOriginDefined();
			datasetOriginDefined.write( out16bitintDataValue, H5::PredType::NATIVE_INT );
            
            H5::DataSet datasetPulseAngularSpacingAzimuth = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_PULSE_ANGULAR_SPACING_AZIMUTH, floatDataTypeDisk, singleValueDataspace);
			outFloatDataValue[0] = spdFile->getPulseAngularSpacingAzimuth();
			datasetPulseAngularSpacingAzimuth.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
			
            H5::DataSet datasetPulseAngularSpacingZenith = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_PULSE_ANGULAR_SPACING_ZENITH, floatDataTypeDisk, singleValueDataspace);
			outFloatDataValue[0] = spdFile->getPulseAngularSpacingZenith();
			datasetPulseAngularSpacingZenith.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            
            H5::DataSet datasetSensorApertureSize = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_SENSOR_APERTURE_SIZE, floatDataTypeDisk, singleValueDataspace);
            outFloatDataValue[0] = spdFile->getSensorApertureSize();
			datasetSensorApertureSize.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            
            H5::DataSet datasetPulseEnergy = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_PULSE_ENERGY, floatDataTypeDisk, singleValueDataspace);
            outFloatDataValue[0] = spdFile->getPulseEnergy();
			datasetPulseEnergy.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            
            H5::DataSet datasetFieldOfView = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_FIELD_OF_VIEW, floatDataTypeDisk, singleValueDataspace);
            outFloatDataValue[0] = spdFile->getFieldOfView();
			datasetFieldOfView.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            
            if(spdFile->getNumOfWavelengths() == 0)
            {
                spdFile->setNumOfWavelengths(1);
                spdFile->getWavelengths()->push_back(0);
                spdFile->getBandwidths()->push_back(0);
            }
            
            H5::DataSet datasetNumOfWavelengths = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_NUM_OF_WAVELENGTHS, uint16bitDataTypeDisk, singleValueDataspace);
            out16bitUintDataValue[0] = spdFile->getNumOfWavelengths();
			datasetNumOfWavelengths.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
            
            hsize_t dimsWavelengthsValue[1];
			dimsWavelengthsValue[0] = spdFile->getNumOfWavelengths();
			H5::DataSpace wavelengthsDataSpace(1, dimsWavelengthsValue);
            
            float *dataVals = new float[spdFile->getNumOfWavelengths()];
            for(unsigned int i = 0; i < spdFile->getNumOfWavelengths(); ++i)
            {
                dataVals[i] = spdFile->getWavelengths()->at(i);
            }
            H5::DataSet datasetWavelengths = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_WAVELENGTHS, floatDataTypeDisk, wavelengthsDataSpace );
            datasetWavelengths.write( dataVals, H5::PredType::NATIVE_FLOAT );
            
            
            for(unsigned int i = 0; i < spdFile->getNumOfWavelengths(); ++i)
            {
                dataVals[i] = spdFile->getBandwidths()->at(i);
            }
			H5::DataSet datasetBandwidths = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_BANDWIDTHS, floatDataTypeDisk, wavelengthsDataSpace );
            datasetBandwidths.write(dataVals, H5::PredType::NATIVE_FLOAT );
            delete[] dataVals;
            
            
            if(spdFile->getFileType() != SPD_UPD_TYPE)
            {
                H5::DataSet datasetBinSize = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_BIN_SIZE, floatDataTypeDisk, singleValueDataspace);
                outFloatDataValue[0] = spdFile->getBinSize();
                datasetBinSize.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
			
                H5::DataSet datasetNumberBinsX = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_NUMBER_BINS_X, uint32bitDataType, singleValueDataspace);			
                out32bitUintDataValue[0] = spdFile->getNumberBinsX();
                datasetNumberBinsX.write( out32bitUintDataValue, H5::PredType::NATIVE_ULONG );
			
                H5::DataSet datasetNumberBinsY = spdOutH5File->createDataSet( SPDFILE_DATASETNAME_NUMBER_BINS_Y, uint32bitDataType, singleValueDataspace);
                out32bitUintDataValue[0] = spdFile->getNumberBinsY();
                datasetNumberBinsY.write( out32bitUintDataValue, H5::PredType::NATIVE_ULONG );
            }
			
		}
		catch( H5::FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataTypeIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch(SPDIOException &e)
		{
			throw e;
		}
    }
    
    void SPDFileWriter::updateHeaderInfo(H5::H5File* spdH5File, SPDFile *spdFile) 
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
		
		H5::StrType strTypeAll(0, H5T_VARIABLE);
        
        try
		{
			H5::Exception::dontPrint() ;
			
			
			if((H5T_STRING!=H5Tget_class(strTypeAll.getId())) || (!H5Tis_variable_str(strTypeAll.getId())))
			{
				throw SPDIOException("The string data type defined is not variable.");
			}
            
            
            try 
            {
                H5::DataSet datasetMajorVersion = spdH5File->openDataSet( SPDFILE_DATASETNAME_MAJOR_VERSION );
                out16bitUintDataValue[0] = spdFile->getMajorSPDVersion();
                datasetMajorVersion.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("The SPD major version header value was not provided.");
            }
            
            try 
            {
                H5::DataSet datasetMinorVersion = spdH5File->openDataSet( SPDFILE_DATASETNAME_MINOR_VERSION );
                out16bitUintDataValue[0] = spdFile->getMinorSPDVersion();
                datasetMinorVersion.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("The SPD minor version header value was not provided.");
            }
            
            try 
            {
                H5::DataSet datasetPointVersion = spdH5File->openDataSet( SPDFILE_DATASETNAME_POINT_VERSION );
                out16bitUintDataValue[0] = spdFile->getPointVersion();
                datasetPointVersion.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("The SPD point version header value was not provided.");
            }
            
            try 
            {
                H5::DataSet datasetPulseVersion = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_VERSION );
                out16bitUintDataValue[0] = spdFile->getPulseVersion();
                datasetPulseVersion.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("The SPD pulse version header value was not provided.");
            }
            
            try 
            {
                H5::DataSet datasetSpatialReference = spdH5File->openDataSet( SPDFILE_DATASETNAME_SPATIAL_REFERENCE );
                wStrdata = new const char*[numLinesStr];
                wStrdata[0] = spdFile->getSpatialReference().c_str();			
                datasetSpatialReference.write((void*)wStrdata, strTypeAll);
                datasetSpatialReference.close();
                delete[] wStrdata;
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Spatial reference header value is not represent.");
            }
            
            try 
            {
                H5::DataSet datasetFileType = spdH5File->openDataSet( SPDFILE_DATASETNAME_FILE_TYPE );
                out16bitUintDataValue[0] = spdFile->getFileType();
                datasetFileType.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("File type header value not present.");
            }
            
            try 
            {
                H5::DataSet datasetIndexType = spdH5File->openDataSet( SPDFILE_DATASETNAME_INDEX_TYPE );
                out16bitUintDataValue[0] = spdFile->getIndexType();
                datasetIndexType.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Index type header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetDiscreteDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_DISCRETE_PT_DEFINED );
                out16bitintDataValue[0] = spdFile->getDiscretePtDefined();
                datasetDiscreteDefined.write( out16bitintDataValue, H5::PredType::NATIVE_INT );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Discrete Point Defined header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetDecomposedDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_DECOMPOSED_PT_DEFINED );
                out16bitintDataValue[0] = spdFile->getDecomposedPtDefined();
                datasetDecomposedDefined.write( out16bitintDataValue, H5::PredType::NATIVE_INT );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Decomposed Point Defined header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetTransWaveformDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_TRANS_WAVEFORM_DEFINED );
                out16bitintDataValue[0] = spdFile->getTransWaveformDefined();
                datasetTransWaveformDefined.write( out16bitintDataValue, H5::PredType::NATIVE_INT );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Transmitted Waveform Defined header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetReceiveWaveformDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_RECEIVE_WAVEFORM_DEFINED );
                out16bitintDataValue[0] = spdFile->getReceiveWaveformDefined();
                datasetReceiveWaveformDefined.write( out16bitintDataValue, H5::PredType::NATIVE_INT );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Received Waveform Defined header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetGeneratingSoftware = spdH5File->openDataSet( SPDFILE_DATASETNAME_GENERATING_SOFTWARE );
                wStrdata = new const char*[numLinesStr];
                wStrdata[0] = spdFile->getGeneratingSoftware().c_str();			
                datasetGeneratingSoftware.write((void*)wStrdata, strTypeAll);
                datasetGeneratingSoftware.close();
                delete[] wStrdata;
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Generating software header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetSystemIdentifier = spdH5File->openDataSet( SPDFILE_DATASETNAME_SYSTEM_IDENTIFIER );
                wStrdata = new const char*[numLinesStr];
                wStrdata[0] = spdFile->getSystemIdentifier().c_str();			
                datasetSystemIdentifier.write((void*)wStrdata, strTypeAll);
                datasetSystemIdentifier.close();
                delete[] wStrdata;
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("System identifier header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetFileSignature = spdH5File->openDataSet( SPDFILE_DATASETNAME_FILE_SIGNATURE );
                wStrdata = new const char*[numLinesStr];
                wStrdata[0] = spdFile->getFileSignature().c_str();			
                datasetFileSignature.write((void*)wStrdata, strTypeAll);
                datasetFileSignature.close();
                delete[] wStrdata;
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("File signature header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetYearOfCreation = spdH5File->openDataSet( SPDFILE_DATASETNAME_YEAR_OF_CREATION );
                H5::DataSet datasetMonthOfCreation = spdH5File->openDataSet( SPDFILE_DATASETNAME_MONTH_OF_CREATION );
                H5::DataSet datasetDayOfCreation = spdH5File->openDataSet( SPDFILE_DATASETNAME_DAY_OF_CREATION );
                H5::DataSet datasetHourOfCreation = spdH5File->openDataSet( SPDFILE_DATASETNAME_HOUR_OF_CREATION );
                H5::DataSet datasetMinuteOfCreation = spdH5File->openDataSet( SPDFILE_DATASETNAME_MINUTE_OF_CREATION );
                H5::DataSet datasetSecondOfCreation = spdH5File->openDataSet( SPDFILE_DATASETNAME_SECOND_OF_CREATION );
                
                out16bitUintDataValue[0] = spdFile->getYearOfCreation();
                datasetYearOfCreation.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
                
                out16bitUintDataValue[0] = spdFile->getMonthOfCreation();
                datasetMonthOfCreation.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
                
                out16bitUintDataValue[0] = spdFile->getDayOfCreation();
                datasetDayOfCreation.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
                
                out16bitUintDataValue[0] = spdFile->getHourOfCreation();
                datasetHourOfCreation.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
                
                out16bitUintDataValue[0] = spdFile->getMinuteOfCreation();
                datasetMinuteOfCreation.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
                
                out16bitUintDataValue[0] = spdFile->getSecondOfCreation();
                datasetSecondOfCreation.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );;
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Date of file creation header values not provided.");
            }
            
            try 
            {
                H5::DataSet datasetYearOfCapture = spdH5File->openDataSet( SPDFILE_DATASETNAME_YEAR_OF_CAPTURE );
                H5::DataSet datasetMonthOfCapture = spdH5File->openDataSet( SPDFILE_DATASETNAME_MONTH_OF_CAPTURE );
                H5::DataSet datasetDayOfCapture = spdH5File->openDataSet( SPDFILE_DATASETNAME_DAY_OF_CAPTURE );
                H5::DataSet datasetHourOfCapture = spdH5File->openDataSet( SPDFILE_DATASETNAME_HOUR_OF_CAPTURE );
                H5::DataSet datasetMinuteOfCapture = spdH5File->openDataSet( SPDFILE_DATASETNAME_MINUTE_OF_CAPTURE );
                H5::DataSet datasetSecondOfCapture = spdH5File->openDataSet( SPDFILE_DATASETNAME_SECOND_OF_CAPTURE );
                
                out16bitUintDataValue[0] = spdFile->getYearOfCapture();
                datasetYearOfCapture.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
                
                out16bitUintDataValue[0] = spdFile->getMonthOfCapture();
                datasetMonthOfCapture.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
                
                out16bitUintDataValue[0] = spdFile->getDayOfCapture();
                datasetDayOfCapture.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
                
                out16bitUintDataValue[0] = spdFile->getHourOfCapture();
                datasetHourOfCapture.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
                
                out16bitUintDataValue[0] = spdFile->getMinuteOfCapture();
                datasetMinuteOfCapture.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
                
                out16bitUintDataValue[0] = spdFile->getSecondOfCapture();
                datasetSecondOfCapture.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Date/Time of capture header values not provided.");
            }
            
            try 
            {
                H5::DataSet datasetNumberOfPoints = spdH5File->openDataSet( SPDFILE_DATASETNAME_NUMBER_OF_POINTS );
                out64bitUintDataValue[0] = spdFile->getNumberOfPoints();
                datasetNumberOfPoints.write( out64bitUintDataValue, H5::PredType::NATIVE_ULLONG );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Number of points header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetNumberOfPulses = spdH5File->openDataSet( SPDFILE_DATASETNAME_NUMBER_OF_PULSES );
                out64bitUintDataValue[0] = spdFile->getNumberOfPulses();
                datasetNumberOfPulses.write( out64bitUintDataValue, H5::PredType::NATIVE_ULLONG );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Number of pulses header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetUserMetaData = spdH5File->openDataSet( SPDFILE_DATASETNAME_USER_META_DATA );
                wStrdata = new const char*[numLinesStr];
                wStrdata[0] = spdFile->getUserMetaField().c_str();			
                datasetUserMetaData.write((void*)wStrdata, strTypeAll);
                datasetUserMetaData.close();
                delete[] wStrdata;
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("User metadata header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetXMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_X_MIN );
                H5::DataSet datasetXMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_X_MAX );
                H5::DataSet datasetYMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_Y_MIN );
                H5::DataSet datasetYMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_Y_MAX );
                H5::DataSet datasetZMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_Z_MIN );
                H5::DataSet datasetZMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_Z_MAX );
                
                outDoubleDataValue[0] = spdFile->getXMin();
                datasetXMin.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
                
                outDoubleDataValue[0] = spdFile->getXMax();
                datasetXMax.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
                
                outDoubleDataValue[0] = spdFile->getYMin();
                datasetYMin.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
                
                outDoubleDataValue[0] = spdFile->getYMax();
                datasetYMax.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
                
                outDoubleDataValue[0] = spdFile->getZMin();
                datasetZMin.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
                
                outDoubleDataValue[0] = spdFile->getZMax();
                datasetZMax.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Dataset bounding volume header values not provided.");
            }
            
            try 
            {
                H5::DataSet datasetZenithMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_ZENITH_MIN );
                H5::DataSet datasetZenithMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_ZENITH_MAX );
                H5::DataSet datasetAzimuthMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_AZIMUTH_MIN );
                H5::DataSet datasetAzimuthMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_AZIMUTH_MAX );
                H5::DataSet datasetRangeMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_RANGE_MIN );
                H5::DataSet datasetRangeMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_RANGE_MAX );
                
                outDoubleDataValue[0] = spdFile->getZenithMin();
                datasetZenithMin.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
                
                outDoubleDataValue[0] = spdFile->getZenithMax();
                datasetZenithMax.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );;
                
                outDoubleDataValue[0] = spdFile->getAzimuthMin();
                datasetAzimuthMin.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
                
                outDoubleDataValue[0] = spdFile->getAzimuthMax();
                datasetAzimuthMax.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
                
                outDoubleDataValue[0] = spdFile->getRangeMin();
                datasetRangeMin.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
                
                outDoubleDataValue[0] = spdFile->getRangeMax();
                datasetRangeMax.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Bounding spherical volume header values not provided.");
            }
            
            try 
            {
                H5::DataSet datasetScanlineMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_SCANLINE_MIN );
                H5::DataSet datasetScanlineMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_SCANLINE_MAX );
                H5::DataSet datasetScanlineIdxMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_SCANLINE_IDX_MIN );
                H5::DataSet datasetScanlineIdxMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_SCANLINE_IDX_MAX );
                
                outDoubleDataValue[0] = spdFile->getScanlineMin();
                datasetScanlineMin.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
                
                outDoubleDataValue[0] = spdFile->getScanlineMax();
                datasetScanlineMax.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );;
                
                outDoubleDataValue[0] = spdFile->getScanlineIdxMin();
                datasetScanlineIdxMin.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
                
                outDoubleDataValue[0] = spdFile->getScanlineIdxMax();
                datasetScanlineIdxMax.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
            } 
            catch (H5::Exception &e) 
            {
                H5::FloatType doubleDataTypeDisk( H5::PredType::IEEE_F64LE );
                hsize_t dimsValue[1];
                dimsValue[0] = 1;
                H5::DataSpace singleValueDataspace(1, dimsValue);
                
                H5::DataSet datasetScanlineMin = spdH5File->createDataSet( SPDFILE_DATASETNAME_SCANLINE_MIN, doubleDataTypeDisk, singleValueDataspace);
                outDoubleDataValue[0] = spdFile->getScanlineMin();
                datasetScanlineMin.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
                
                H5::DataSet datasetScanlineMax = spdH5File->createDataSet( SPDFILE_DATASETNAME_SCANLINE_MAX, doubleDataTypeDisk, singleValueDataspace);
                outDoubleDataValue[0] = spdFile->getScanlineMax();
                datasetScanlineMax.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
                
                H5::DataSet datasetScanlineIdxMin = spdH5File->createDataSet( SPDFILE_DATASETNAME_SCANLINE_IDX_MIN, doubleDataTypeDisk, singleValueDataspace);
                outDoubleDataValue[0] = spdFile->getScanlineIdxMin();
                datasetScanlineIdxMin.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
                
                H5::DataSet datasetScanlineIdxMax = spdH5File->createDataSet( SPDFILE_DATASETNAME_SCANLINE_IDX_MAX, doubleDataTypeDisk, singleValueDataspace);
                outDoubleDataValue[0] = spdFile->getScanlineIdxMax();
                datasetScanlineIdxMax.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
            }
            
            if(spdFile->getFileType() != SPD_UPD_TYPE)
            {
                try 
                {
                    H5::DataSet datasetBinSize = spdH5File->openDataSet( SPDFILE_DATASETNAME_BIN_SIZE );
                    outFloatDataValue[0] = spdFile->getBinSize();
                    datasetBinSize.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
                } 
                catch (H5::Exception &e) 
                {
                    throw SPDIOException("Bin size header value not provided.");
                }
                
                try 
                {
                    H5::DataSet datasetNumberBinsX = spdH5File->openDataSet( SPDFILE_DATASETNAME_NUMBER_BINS_X );
                    out32bitUintDataValue[0] = spdFile->getNumberBinsX();
                    datasetNumberBinsX.write( out32bitUintDataValue, H5::PredType::NATIVE_ULONG );
                } 
                catch (H5::Exception &e) 
                {
                    throw SPDIOException("Number of X bins header value not provided.");
                }
                
                try 
                {
                    H5::DataSet datasetNumberBinsY = spdH5File->openDataSet( SPDFILE_DATASETNAME_NUMBER_BINS_Y );
                    out32bitUintDataValue[0] = spdFile->getNumberBinsY();
                    datasetNumberBinsY.write( out32bitUintDataValue, H5::PredType::NATIVE_ULONG );
                } 
                catch (H5::Exception &e) 
                {
                    throw SPDIOException("Number of Y bins header value not provided.");
                }
            }
            
            try 
            {
                H5::DataSet datasetPulseRepFreq = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_REPETITION_FREQ );
                outFloatDataValue[0] = spdFile->getPulseRepetitionFreq();
                datasetPulseRepFreq.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Pulse repetition frequency header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetBeamDivergence = spdH5File->openDataSet( SPDFILE_DATASETNAME_BEAM_DIVERGENCE );
                outFloatDataValue[0] = spdFile->getBeamDivergence();
                datasetBeamDivergence.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Beam divergence header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetSensorHeight = spdH5File->openDataSet( SPDFILE_DATASETNAME_SENSOR_HEIGHT );
                outDoubleDataValue[0] = spdFile->getSensorHeight();
                datasetSensorHeight.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Sensor height header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetFootprint = spdH5File->openDataSet( SPDFILE_DATASETNAME_FOOTPRINT );
                outFloatDataValue[0] = spdFile->getFootprint();
                datasetFootprint.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Footprint header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetMaxScanAngle = spdH5File->openDataSet( SPDFILE_DATASETNAME_MAX_SCAN_ANGLE );
                outFloatDataValue[0] = spdFile->getMaxScanAngle();
                datasetMaxScanAngle.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Max scan angle header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetRGBDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_RGB_DEFINED );
                out16bitintDataValue[0] = spdFile->getRGBDefined();
                datasetRGBDefined.write( out16bitintDataValue, H5::PredType::NATIVE_INT );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("RGB defined header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetPulseBlockSize = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_BLOCK_SIZE );
                out16bitUintDataValue[0] = spdFile->getPulseBlockSize();
                datasetPulseBlockSize.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Pulse block size header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetPointsBlockSize = spdH5File->openDataSet( SPDFILE_DATASETNAME_POINT_BLOCK_SIZE );
                out16bitUintDataValue[0] = spdFile->getPointBlockSize();
                datasetPointsBlockSize.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Point block size header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetReceivedBlockSize = spdH5File->openDataSet( SPDFILE_DATASETNAME_RECEIVED_BLOCK_SIZE );
                out16bitUintDataValue[0] = spdFile->getReceivedBlockSize();
                datasetReceivedBlockSize.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Received waveform block size header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetTransmittedBlockSize = spdH5File->openDataSet( SPDFILE_DATASETNAME_TRANSMITTED_BLOCK_SIZE );
                out16bitUintDataValue[0] = spdFile->getTransmittedBlockSize();
                datasetTransmittedBlockSize.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Transmitted waveform block size header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetWaveformBitRes = spdH5File->openDataSet( SPDFILE_DATASETNAME_WAVEFORM_BIT_RES );
                out16bitUintDataValue[0] = spdFile->getWaveformBitRes();
                datasetWaveformBitRes.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Waveform bit resolution header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetTemporalBinSpacing = spdH5File->openDataSet( SPDFILE_DATASETNAME_TEMPORAL_BIN_SPACING );
                outDoubleDataValue[0] = spdFile->getTemporalBinSpacing();
                datasetTemporalBinSpacing.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Temporal bin spacing header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetReturnNumsSynGen = spdH5File->openDataSet( SPDFILE_DATASETNAME_RETURN_NUMBERS_SYN_GEN );
                out16bitintDataValue[0] = spdFile->getReturnNumsSynGen();
                datasetReturnNumsSynGen.write( out16bitintDataValue, H5::PredType::NATIVE_INT );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Return number synthetically generated header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetHeightDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_HEIGHT_DEFINED );
                out16bitintDataValue[0] = spdFile->getHeightDefined();
                datasetHeightDefined.write( out16bitintDataValue, H5::PredType::NATIVE_INT );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Height fields defined header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetSensorSpeed = spdH5File->openDataSet( SPDFILE_DATASETNAME_SENSOR_SPEED );
                outFloatDataValue[0] = spdFile->getSensorSpeed();
                datasetSensorSpeed.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Sensor speed header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetSensorScanRate = spdH5File->openDataSet( SPDFILE_DATASETNAME_SENSOR_SCAN_RATE );
                outFloatDataValue[0] = spdFile->getSensorScanRate();
                datasetSensorScanRate.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Sensor Scan Rate header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetPointDensity = spdH5File->openDataSet( SPDFILE_DATASETNAME_POINT_DENSITY );
                outFloatDataValue[0] = spdFile->getPointDensity();
                datasetPointDensity.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Point density header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetPulseDensity = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_DENSITY );
                outFloatDataValue[0] = spdFile->getPulseDensity();
                datasetPulseDensity.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Pulse density header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetPulseCrossTrackSpacing = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_CROSS_TRACK_SPACING );
                outFloatDataValue[0] = spdFile->getPulseCrossTrackSpacing();
                datasetPulseCrossTrackSpacing.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Cross track spacing header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetPulseAlongTrackSpacing = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_ALONG_TRACK_SPACING );
                outFloatDataValue[0] = spdFile->getPulseAlongTrackSpacing();
                datasetPulseAlongTrackSpacing.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Along track spacing header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetOriginDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_ORIGIN_DEFINED );
                out16bitintDataValue[0] = spdFile->getOriginDefined();
                datasetOriginDefined.write( out16bitintDataValue, H5::PredType::NATIVE_INT );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Origin defined header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetPulseAngularSpacingAzimuth = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_ANGULAR_SPACING_AZIMUTH );
                outFloatDataValue[0] = spdFile->getPulseAngularSpacingAzimuth();
                datasetPulseAngularSpacingAzimuth.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Angular azimuth spacing header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetPulseAngularSpacingZenith = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_ANGULAR_SPACING_ZENITH );
                outFloatDataValue[0] = spdFile->getPulseAngularSpacingZenith();
                datasetPulseAngularSpacingZenith.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Angular Zenith spacing header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetPulseIndexMethod = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_INDEX_METHOD );
                out16bitUintDataValue[0] = spdFile->getIndexType();
                datasetPulseIndexMethod.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
            } 
            catch (H5::Exception &e) 
            {
                throw SPDIOException("Method of indexing header value not provided.");
            }
            
            try 
            {
                H5::DataSet datasetSensorApertureSize = spdH5File->openDataSet( SPDFILE_DATASETNAME_SENSOR_APERTURE_SIZE );
                outFloatDataValue[0] = spdFile->getSensorApertureSize();
                datasetSensorApertureSize.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            } 
            catch (H5::Exception &e) 
            {
                float outFloatDataValue[1];
                H5::FloatType floatDataTypeDisk( H5::PredType::IEEE_F32LE );
                hsize_t dimsValue[1];
                dimsValue[0] = 1;
                H5::DataSpace singleValueDataspace(1, dimsValue);
                H5::DataSet datasetSensorApertureSize = spdH5File->createDataSet( SPDFILE_DATASETNAME_SENSOR_APERTURE_SIZE, floatDataTypeDisk, singleValueDataspace);
                outFloatDataValue[0] = spdFile->getSensorApertureSize();
                datasetSensorApertureSize.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            }
            
            try 
            {
                H5::DataSet datasetPulseEnergy = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_ENERGY );
                outFloatDataValue[0] = spdFile->getPulseEnergy();
                datasetPulseEnergy.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            } 
            catch (H5::Exception &e) 
            {
                float outFloatDataValue[1];
                H5::FloatType floatDataTypeDisk( H5::PredType::IEEE_F32LE );
                hsize_t dimsValue[1];
                dimsValue[0] = 1;
                H5::DataSpace singleValueDataspace(1, dimsValue);
                H5::DataSet datasetPulseEnergy = spdH5File->createDataSet( SPDFILE_DATASETNAME_PULSE_ENERGY, floatDataTypeDisk, singleValueDataspace);
                outFloatDataValue[0] = spdFile->getPulseEnergy();
                datasetPulseEnergy.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            }
            
            try 
            {
                H5::DataSet datasetFieldOfView = spdH5File->openDataSet( SPDFILE_DATASETNAME_FIELD_OF_VIEW );
                outFloatDataValue[0] = spdFile->getFieldOfView();
                datasetFieldOfView.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            } 
            catch (H5::Exception &e) 
            {
                float outFloatDataValue[1];
                H5::FloatType floatDataTypeDisk( H5::PredType::IEEE_F32LE );
                hsize_t dimsValue[1];
                dimsValue[0] = 1;
                H5::DataSpace singleValueDataspace(1, dimsValue);
                H5::DataSet datasetFieldOfView = spdH5File->createDataSet( SPDFILE_DATASETNAME_FIELD_OF_VIEW, floatDataTypeDisk, singleValueDataspace);
                outFloatDataValue[0] = spdFile->getFieldOfView();
                datasetFieldOfView.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            }
            
            try 
            {
                H5::DataSet datasetNumOfWavelengths = spdH5File->openDataSet( SPDFILE_DATASETNAME_NUM_OF_WAVELENGTHS );
                out16bitUintDataValue[0] = spdFile->getNumOfWavelengths();
                datasetNumOfWavelengths.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
                
                if(spdFile->getNumOfWavelengths() > 0)
                {
                    hsize_t dimsWavelengthsValue[1];
                    dimsWavelengthsValue[0] = spdFile->getNumOfWavelengths();
                    H5::DataSpace wavelengthsDataSpace(1, dimsWavelengthsValue);
                    
                    float *dataVals = new float[spdFile->getNumOfWavelengths()];
                    for(unsigned int i = 0; i < spdFile->getNumOfWavelengths(); ++i)
                    {
                        dataVals[i] = spdFile->getWavelengths()->at(i);
                    }
                    H5::DataSet datasetWavelengths = spdH5File->openDataSet( SPDFILE_DATASETNAME_WAVELENGTHS );
                    datasetWavelengths.write(dataVals, H5::PredType::NATIVE_FLOAT );
                    
                    for(unsigned int i = 0; i < spdFile->getNumOfWavelengths(); ++i)
                    {
                        dataVals[i] = spdFile->getBandwidths()->at(i);
                    }
                    H5::DataSet datasetBandwidths = spdH5File->openDataSet( SPDFILE_DATASETNAME_BANDWIDTHS );
                    datasetBandwidths.write(dataVals, H5::PredType::NATIVE_FLOAT );
                    delete[] dataVals;
                }
                else
                {
                    std::vector<float> wavelengths;
                    spdFile->setWavelengths(wavelengths);
                    std::vector<float> bandwidths;
                    spdFile->setBandwidths(bandwidths);
                }
                
            } 
            catch (H5::Exception &e) 
            {
                H5::DataSet datasetWavelength = spdH5File->openDataSet( SPDFILE_DATASETNAME_WAVELENGTH );
                if(spdFile->getNumOfWavelengths() > 0)
                {
                    outFloatDataValue[0] = spdFile->getWavelengths()->front();
                    datasetWavelength.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
                }
                else
                {
                    outFloatDataValue[0] = 0;
                    datasetWavelength.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
                }
                
                H5::IntType uint16bitDataTypeDisk( H5::PredType::STD_U16LE );
                hsize_t dimsValue[1];
                dimsValue[0] = 1;
                H5::DataSpace singleValueDataspace(1, dimsValue);
                H5::DataSet datasetNumOfWavelengths = spdH5File->createDataSet( SPDFILE_DATASETNAME_NUM_OF_WAVELENGTHS, uint16bitDataTypeDisk, singleValueDataspace);
                out16bitUintDataValue[0] = spdFile->getNumOfWavelengths();
                datasetNumOfWavelengths.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
                
                hsize_t dimsWavelengthsValue[1];
                dimsWavelengthsValue[0] = spdFile->getNumOfWavelengths();
                H5::DataSpace wavelengthsDataSpace(1, dimsWavelengthsValue);
                H5::FloatType floatDataTypeDisk( H5::PredType::IEEE_F32LE );
                
                float *dataVals = new float[spdFile->getNumOfWavelengths()];
                for(unsigned int i = 0; i < spdFile->getNumOfWavelengths(); ++i)
                {
                    dataVals[i] = spdFile->getWavelengths()->at(i);
                }
                H5::DataSet datasetWavelengths = spdH5File->createDataSet( SPDFILE_DATASETNAME_WAVELENGTHS, floatDataTypeDisk, wavelengthsDataSpace );
                datasetWavelengths.write(dataVals, H5::PredType::NATIVE_FLOAT );
                
                for(unsigned int i = 0; i < spdFile->getNumOfWavelengths(); ++i)
                {
                    dataVals[i] = spdFile->getBandwidths()->at(i);
                }
                H5::DataSet datasetBandwidths = spdH5File->createDataSet( SPDFILE_DATASETNAME_BANDWIDTHS, floatDataTypeDisk, wavelengthsDataSpace );
                datasetBandwidths.write(dataVals, H5::PredType::NATIVE_FLOAT );
                delete[] dataVals;
            }
			spdH5File->flush(H5F_SCOPE_GLOBAL);
			
		}
		catch( H5::FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataTypeIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
        catch( SPDIOException &e )
		{
			throw e;
		}
	}
    
    void SPDFileWriter::updateHeaderInfo(SPDFile *spdFile) 
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
		
		H5::StrType strTypeAll(0, H5T_VARIABLE);
		
		const H5std_string spdFilePath( spdFile->getFilePath() );
		try
		{
			H5::Exception::dontPrint() ;
			
			// Create File..
			H5::H5File* spdH5File= new H5::H5File( spdFilePath, H5F_ACC_RDWR );
			
			if((H5T_STRING!=H5Tget_class(strTypeAll.getId())) || (!H5Tis_variable_str(strTypeAll.getId())))
			{
				throw SPDIOException("The string data type defined is not variable.");
			}
            
            
            try
            {
                H5::DataSet datasetMajorVersion = spdH5File->openDataSet( SPDFILE_DATASETNAME_MAJOR_VERSION );
                out16bitUintDataValue[0] = spdFile->getMajorSPDVersion();
                datasetMajorVersion.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("The SPD major version header value was not provided.");
            }
            
            try
            {
                H5::DataSet datasetMinorVersion = spdH5File->openDataSet( SPDFILE_DATASETNAME_MINOR_VERSION );
                out16bitUintDataValue[0] = spdFile->getMinorSPDVersion();
                datasetMinorVersion.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("The SPD minor version header value was not provided.");
            }
            
            try
            {
                H5::DataSet datasetPointVersion = spdH5File->openDataSet( SPDFILE_DATASETNAME_POINT_VERSION );
                out16bitUintDataValue[0] = spdFile->getPointVersion();
                datasetPointVersion.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("The SPD point version header value was not provided.");
            }
            
            try
            {
                H5::DataSet datasetPulseVersion = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_VERSION );
                out16bitUintDataValue[0] = spdFile->getPulseVersion();
                datasetPulseVersion.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("The SPD pulse version header value was not provided.");
            }
            
            try
            {
                H5::DataSet datasetSpatialReference = spdH5File->openDataSet( SPDFILE_DATASETNAME_SPATIAL_REFERENCE );
                wStrdata = new const char*[numLinesStr];
                wStrdata[0] = spdFile->getSpatialReference().c_str();
                datasetSpatialReference.write((void*)wStrdata, strTypeAll);
                datasetSpatialReference.close();
                delete[] wStrdata;
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Spatial reference header value is not represent.");
            }
            
            try
            {
                H5::DataSet datasetFileType = spdH5File->openDataSet( SPDFILE_DATASETNAME_FILE_TYPE );
                out16bitUintDataValue[0] = spdFile->getFileType();
                datasetFileType.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("File type header value not present.");
            }
            
            try
            {
                H5::DataSet datasetIndexType = spdH5File->openDataSet( SPDFILE_DATASETNAME_INDEX_TYPE );
                out16bitUintDataValue[0] = spdFile->getIndexType();
                datasetIndexType.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Index type header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetDiscreteDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_DISCRETE_PT_DEFINED );
                out16bitintDataValue[0] = spdFile->getDiscretePtDefined();
                datasetDiscreteDefined.write( out16bitintDataValue, H5::PredType::NATIVE_INT );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Discrete Point Defined header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetDecomposedDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_DECOMPOSED_PT_DEFINED );
                out16bitintDataValue[0] = spdFile->getDecomposedPtDefined();
                datasetDecomposedDefined.write( out16bitintDataValue, H5::PredType::NATIVE_INT );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Decomposed Point Defined header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetTransWaveformDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_TRANS_WAVEFORM_DEFINED );
                out16bitintDataValue[0] = spdFile->getTransWaveformDefined();
                datasetTransWaveformDefined.write( out16bitintDataValue, H5::PredType::NATIVE_INT );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Transmitted Waveform Defined header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetReceiveWaveformDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_RECEIVE_WAVEFORM_DEFINED );
                out16bitintDataValue[0] = spdFile->getReceiveWaveformDefined();
                datasetReceiveWaveformDefined.write( out16bitintDataValue, H5::PredType::NATIVE_INT );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Received Waveform Defined header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetGeneratingSoftware = spdH5File->openDataSet( SPDFILE_DATASETNAME_GENERATING_SOFTWARE );
                wStrdata = new const char*[numLinesStr];
                wStrdata[0] = spdFile->getGeneratingSoftware().c_str();
                datasetGeneratingSoftware.write((void*)wStrdata, strTypeAll);
                datasetGeneratingSoftware.close();
                delete[] wStrdata;
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Generating software header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetSystemIdentifier = spdH5File->openDataSet( SPDFILE_DATASETNAME_SYSTEM_IDENTIFIER );
                wStrdata = new const char*[numLinesStr];
                wStrdata[0] = spdFile->getSystemIdentifier().c_str();
                datasetSystemIdentifier.write((void*)wStrdata, strTypeAll);
                datasetSystemIdentifier.close();
                delete[] wStrdata;
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("System identifier header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetFileSignature = spdH5File->openDataSet( SPDFILE_DATASETNAME_FILE_SIGNATURE );
                wStrdata = new const char*[numLinesStr];
                wStrdata[0] = spdFile->getFileSignature().c_str();
                datasetFileSignature.write((void*)wStrdata, strTypeAll);
                datasetFileSignature.close();
                delete[] wStrdata;
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("File signature header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetYearOfCreation = spdH5File->openDataSet( SPDFILE_DATASETNAME_YEAR_OF_CREATION );
                H5::DataSet datasetMonthOfCreation = spdH5File->openDataSet( SPDFILE_DATASETNAME_MONTH_OF_CREATION );
                H5::DataSet datasetDayOfCreation = spdH5File->openDataSet( SPDFILE_DATASETNAME_DAY_OF_CREATION );
                H5::DataSet datasetHourOfCreation = spdH5File->openDataSet( SPDFILE_DATASETNAME_HOUR_OF_CREATION );
                H5::DataSet datasetMinuteOfCreation = spdH5File->openDataSet( SPDFILE_DATASETNAME_MINUTE_OF_CREATION );
                H5::DataSet datasetSecondOfCreation = spdH5File->openDataSet( SPDFILE_DATASETNAME_SECOND_OF_CREATION );
                
                out16bitUintDataValue[0] = spdFile->getYearOfCreation();
                datasetYearOfCreation.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
                
                out16bitUintDataValue[0] = spdFile->getMonthOfCreation();
                datasetMonthOfCreation.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
                
                out16bitUintDataValue[0] = spdFile->getDayOfCreation();
                datasetDayOfCreation.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
                
                out16bitUintDataValue[0] = spdFile->getHourOfCreation();
                datasetHourOfCreation.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
                
                out16bitUintDataValue[0] = spdFile->getMinuteOfCreation();
                datasetMinuteOfCreation.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
                
                out16bitUintDataValue[0] = spdFile->getSecondOfCreation();
                datasetSecondOfCreation.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );;
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Date of file creation header values not provided.");
            }
            
            try
            {
                H5::DataSet datasetYearOfCapture = spdH5File->openDataSet( SPDFILE_DATASETNAME_YEAR_OF_CAPTURE );
                H5::DataSet datasetMonthOfCapture = spdH5File->openDataSet( SPDFILE_DATASETNAME_MONTH_OF_CAPTURE );
                H5::DataSet datasetDayOfCapture = spdH5File->openDataSet( SPDFILE_DATASETNAME_DAY_OF_CAPTURE );
                H5::DataSet datasetHourOfCapture = spdH5File->openDataSet( SPDFILE_DATASETNAME_HOUR_OF_CAPTURE );
                H5::DataSet datasetMinuteOfCapture = spdH5File->openDataSet( SPDFILE_DATASETNAME_MINUTE_OF_CAPTURE );
                H5::DataSet datasetSecondOfCapture = spdH5File->openDataSet( SPDFILE_DATASETNAME_SECOND_OF_CAPTURE );
                
                out16bitUintDataValue[0] = spdFile->getYearOfCapture();
                datasetYearOfCapture.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
                
                out16bitUintDataValue[0] = spdFile->getMonthOfCapture();
                datasetMonthOfCapture.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
                
                out16bitUintDataValue[0] = spdFile->getDayOfCapture();
                datasetDayOfCapture.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
                
                out16bitUintDataValue[0] = spdFile->getHourOfCapture();
                datasetHourOfCapture.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
                
                out16bitUintDataValue[0] = spdFile->getMinuteOfCapture();
                datasetMinuteOfCapture.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
                
                out16bitUintDataValue[0] = spdFile->getSecondOfCapture();
                datasetSecondOfCapture.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Date/Time of capture header values not provided.");
            }
            
            try
            {
                H5::DataSet datasetNumberOfPoints = spdH5File->openDataSet( SPDFILE_DATASETNAME_NUMBER_OF_POINTS );
                out64bitUintDataValue[0] = spdFile->getNumberOfPoints();
                datasetNumberOfPoints.write( out64bitUintDataValue, H5::PredType::NATIVE_ULLONG );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Number of points header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetNumberOfPulses = spdH5File->openDataSet( SPDFILE_DATASETNAME_NUMBER_OF_PULSES );
                out64bitUintDataValue[0] = spdFile->getNumberOfPulses();
                datasetNumberOfPulses.write( out64bitUintDataValue, H5::PredType::NATIVE_ULLONG );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Number of pulses header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetUserMetaData = spdH5File->openDataSet( SPDFILE_DATASETNAME_USER_META_DATA );
                wStrdata = new const char*[numLinesStr];
                wStrdata[0] = spdFile->getUserMetaField().c_str();
                datasetUserMetaData.write((void*)wStrdata, strTypeAll);
                datasetUserMetaData.close();
                delete[] wStrdata;
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("User metadata header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetXMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_X_MIN );
                H5::DataSet datasetXMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_X_MAX );
                H5::DataSet datasetYMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_Y_MIN );
                H5::DataSet datasetYMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_Y_MAX );
                H5::DataSet datasetZMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_Z_MIN );
                H5::DataSet datasetZMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_Z_MAX );
                
                outDoubleDataValue[0] = spdFile->getXMin();
                datasetXMin.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
                
                outDoubleDataValue[0] = spdFile->getXMax();
                datasetXMax.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
                
                outDoubleDataValue[0] = spdFile->getYMin();
                datasetYMin.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
                
                outDoubleDataValue[0] = spdFile->getYMax();
                datasetYMax.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
                
                outDoubleDataValue[0] = spdFile->getZMin();
                datasetZMin.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
                
                outDoubleDataValue[0] = spdFile->getZMax();
                datasetZMax.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Dataset bounding volume header values not provided.");
            }
            
            try
            {
                H5::DataSet datasetZenithMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_ZENITH_MIN );
                H5::DataSet datasetZenithMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_ZENITH_MAX );
                H5::DataSet datasetAzimuthMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_AZIMUTH_MIN );
                H5::DataSet datasetAzimuthMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_AZIMUTH_MAX );
                H5::DataSet datasetRangeMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_RANGE_MIN );
                H5::DataSet datasetRangeMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_RANGE_MAX );
                
                outDoubleDataValue[0] = spdFile->getZenithMin();
                datasetZenithMin.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
                
                outDoubleDataValue[0] = spdFile->getZenithMax();
                datasetZenithMax.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );;
                
                outDoubleDataValue[0] = spdFile->getAzimuthMin();
                datasetAzimuthMin.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
                
                outDoubleDataValue[0] = spdFile->getAzimuthMax();
                datasetAzimuthMax.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
                
                outDoubleDataValue[0] = spdFile->getRangeMin();
                datasetRangeMin.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
                
                outDoubleDataValue[0] = spdFile->getRangeMax();
                datasetRangeMax.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Bounding spherical volume header values not provided.");
            }
            
            try
            {
                H5::DataSet datasetScanlineMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_SCANLINE_MIN );
                H5::DataSet datasetScanlineMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_SCANLINE_MAX );
                H5::DataSet datasetScanlineIdxMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_SCANLINE_IDX_MIN );
                H5::DataSet datasetScanlineIdxMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_SCANLINE_IDX_MAX );
                
                outDoubleDataValue[0] = spdFile->getScanlineMin();
                datasetScanlineMin.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
                
                outDoubleDataValue[0] = spdFile->getScanlineMax();
                datasetScanlineMax.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );;
                
                outDoubleDataValue[0] = spdFile->getScanlineIdxMin();
                datasetScanlineIdxMin.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
                
                outDoubleDataValue[0] = spdFile->getScanlineIdxMax();
                datasetScanlineIdxMax.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
            }
            catch (H5::Exception &e)
            {
                H5::FloatType doubleDataTypeDisk( H5::PredType::IEEE_F64LE );
                hsize_t dimsValue[1];
                dimsValue[0] = 1;
                H5::DataSpace singleValueDataspace(1, dimsValue);
                
                H5::DataSet datasetScanlineMin = spdH5File->createDataSet( SPDFILE_DATASETNAME_SCANLINE_MIN, doubleDataTypeDisk, singleValueDataspace);
                outDoubleDataValue[0] = spdFile->getScanlineMin();
                datasetScanlineMin.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
                
                H5::DataSet datasetScanlineMax = spdH5File->createDataSet( SPDFILE_DATASETNAME_SCANLINE_MAX, doubleDataTypeDisk, singleValueDataspace);
                outDoubleDataValue[0] = spdFile->getScanlineMax();
                datasetScanlineMax.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
                
                H5::DataSet datasetScanlineIdxMin = spdH5File->createDataSet( SPDFILE_DATASETNAME_SCANLINE_IDX_MIN, doubleDataTypeDisk, singleValueDataspace);
                outDoubleDataValue[0] = spdFile->getScanlineIdxMin();
                datasetScanlineIdxMin.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
                
                H5::DataSet datasetScanlineIdxMax = spdH5File->createDataSet( SPDFILE_DATASETNAME_SCANLINE_IDX_MAX, doubleDataTypeDisk, singleValueDataspace);
                outDoubleDataValue[0] = spdFile->getScanlineIdxMax();
                datasetScanlineIdxMax.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
            }
            
            if(spdFile->getFileType() != SPD_UPD_TYPE)
            {
                try
                {
                    H5::DataSet datasetBinSize = spdH5File->openDataSet( SPDFILE_DATASETNAME_BIN_SIZE );
                    outFloatDataValue[0] = spdFile->getBinSize();
                    datasetBinSize.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
                }
                catch (H5::Exception &e)
                {
                    throw SPDIOException("Bin size header value not provided.");
                }
                
                try
                {
                    H5::DataSet datasetNumberBinsX = spdH5File->openDataSet( SPDFILE_DATASETNAME_NUMBER_BINS_X );
                    out32bitUintDataValue[0] = spdFile->getNumberBinsX();
                    datasetNumberBinsX.write( out32bitUintDataValue, H5::PredType::NATIVE_ULONG );
                }
                catch (H5::Exception &e)
                {
                    throw SPDIOException("Number of X bins header value not provided.");
                }
                
                try
                {
                    H5::DataSet datasetNumberBinsY = spdH5File->openDataSet( SPDFILE_DATASETNAME_NUMBER_BINS_Y );
                    out32bitUintDataValue[0] = spdFile->getNumberBinsY();
                    datasetNumberBinsY.write( out32bitUintDataValue, H5::PredType::NATIVE_ULONG );
                }
                catch (H5::Exception &e)
                {
                    throw SPDIOException("Number of Y bins header value not provided.");
                }
            }
            
            try
            {
                H5::DataSet datasetPulseRepFreq = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_REPETITION_FREQ );
                outFloatDataValue[0] = spdFile->getPulseRepetitionFreq();
                datasetPulseRepFreq.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Pulse repetition frequency header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetBeamDivergence = spdH5File->openDataSet( SPDFILE_DATASETNAME_BEAM_DIVERGENCE );
                outFloatDataValue[0] = spdFile->getBeamDivergence();
                datasetBeamDivergence.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Beam divergence header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetSensorHeight = spdH5File->openDataSet( SPDFILE_DATASETNAME_SENSOR_HEIGHT );
                outDoubleDataValue[0] = spdFile->getSensorHeight();
                datasetSensorHeight.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Sensor height header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetFootprint = spdH5File->openDataSet( SPDFILE_DATASETNAME_FOOTPRINT );
                outFloatDataValue[0] = spdFile->getFootprint();
                datasetFootprint.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Footprint header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetMaxScanAngle = spdH5File->openDataSet( SPDFILE_DATASETNAME_MAX_SCAN_ANGLE );
                outFloatDataValue[0] = spdFile->getMaxScanAngle();
                datasetMaxScanAngle.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Max scan angle header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetRGBDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_RGB_DEFINED );
                out16bitintDataValue[0] = spdFile->getRGBDefined();
                datasetRGBDefined.write( out16bitintDataValue, H5::PredType::NATIVE_INT );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("RGB defined header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetPulseBlockSize = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_BLOCK_SIZE );
                out16bitUintDataValue[0] = spdFile->getPulseBlockSize();
                datasetPulseBlockSize.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Pulse block size header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetPointsBlockSize = spdH5File->openDataSet( SPDFILE_DATASETNAME_POINT_BLOCK_SIZE );
                out16bitUintDataValue[0] = spdFile->getPointBlockSize();
                datasetPointsBlockSize.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Point block size header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetReceivedBlockSize = spdH5File->openDataSet( SPDFILE_DATASETNAME_RECEIVED_BLOCK_SIZE );
                out16bitUintDataValue[0] = spdFile->getReceivedBlockSize();
                datasetReceivedBlockSize.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Received waveform block size header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetTransmittedBlockSize = spdH5File->openDataSet( SPDFILE_DATASETNAME_TRANSMITTED_BLOCK_SIZE );
                out16bitUintDataValue[0] = spdFile->getTransmittedBlockSize();
                datasetTransmittedBlockSize.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Transmitted waveform block size header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetWaveformBitRes = spdH5File->openDataSet( SPDFILE_DATASETNAME_WAVEFORM_BIT_RES );
                out16bitUintDataValue[0] = spdFile->getWaveformBitRes();
                datasetWaveformBitRes.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Waveform bit resolution header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetTemporalBinSpacing = spdH5File->openDataSet( SPDFILE_DATASETNAME_TEMPORAL_BIN_SPACING );
                outDoubleDataValue[0] = spdFile->getTemporalBinSpacing();
                datasetTemporalBinSpacing.write( outDoubleDataValue, H5::PredType::NATIVE_DOUBLE );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Temporal bin spacing header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetReturnNumsSynGen = spdH5File->openDataSet( SPDFILE_DATASETNAME_RETURN_NUMBERS_SYN_GEN );
                out16bitintDataValue[0] = spdFile->getReturnNumsSynGen();
                datasetReturnNumsSynGen.write( out16bitintDataValue, H5::PredType::NATIVE_INT );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Return number synthetically generated header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetHeightDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_HEIGHT_DEFINED );
                out16bitintDataValue[0] = spdFile->getHeightDefined();
                datasetHeightDefined.write( out16bitintDataValue, H5::PredType::NATIVE_INT );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Height fields defined header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetSensorSpeed = spdH5File->openDataSet( SPDFILE_DATASETNAME_SENSOR_SPEED );
                outFloatDataValue[0] = spdFile->getSensorSpeed();
                datasetSensorSpeed.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Sensor speed header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetSensorScanRate = spdH5File->openDataSet( SPDFILE_DATASETNAME_SENSOR_SCAN_RATE );
                outFloatDataValue[0] = spdFile->getSensorScanRate();
                datasetSensorScanRate.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Sensor Scan Rate header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetPointDensity = spdH5File->openDataSet( SPDFILE_DATASETNAME_POINT_DENSITY );
                outFloatDataValue[0] = spdFile->getPointDensity();
                datasetPointDensity.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Point density header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetPulseDensity = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_DENSITY );
                outFloatDataValue[0] = spdFile->getPulseDensity();
                datasetPulseDensity.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Pulse density header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetPulseCrossTrackSpacing = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_CROSS_TRACK_SPACING );
                outFloatDataValue[0] = spdFile->getPulseCrossTrackSpacing();
                datasetPulseCrossTrackSpacing.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Cross track spacing header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetPulseAlongTrackSpacing = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_ALONG_TRACK_SPACING );
                outFloatDataValue[0] = spdFile->getPulseAlongTrackSpacing();
                datasetPulseAlongTrackSpacing.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Along track spacing header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetOriginDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_ORIGIN_DEFINED );
                out16bitintDataValue[0] = spdFile->getOriginDefined();
                datasetOriginDefined.write( out16bitintDataValue, H5::PredType::NATIVE_INT );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Origin defined header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetPulseAngularSpacingAzimuth = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_ANGULAR_SPACING_AZIMUTH );
                outFloatDataValue[0] = spdFile->getPulseAngularSpacingAzimuth();
                datasetPulseAngularSpacingAzimuth.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Angular azimuth spacing header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetPulseAngularSpacingZenith = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_ANGULAR_SPACING_ZENITH );
                outFloatDataValue[0] = spdFile->getPulseAngularSpacingZenith();
                datasetPulseAngularSpacingZenith.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Angular Zenith spacing header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetPulseIndexMethod = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_INDEX_METHOD );
                out16bitUintDataValue[0] = spdFile->getIndexType();
                datasetPulseIndexMethod.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Method of indexing header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetSensorApertureSize = spdH5File->openDataSet( SPDFILE_DATASETNAME_SENSOR_APERTURE_SIZE );
                outFloatDataValue[0] = spdFile->getSensorApertureSize();
                datasetSensorApertureSize.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            }
            catch (H5::Exception &e)
            {
                float outFloatDataValue[1];
                H5::FloatType floatDataTypeDisk( H5::PredType::IEEE_F32LE );
                hsize_t dimsValue[1];
                dimsValue[0] = 1;
                H5::DataSpace singleValueDataspace(1, dimsValue);
                H5::DataSet datasetSensorApertureSize = spdH5File->createDataSet( SPDFILE_DATASETNAME_SENSOR_APERTURE_SIZE, floatDataTypeDisk, singleValueDataspace);
                outFloatDataValue[0] = spdFile->getSensorApertureSize();
                datasetSensorApertureSize.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            }
            
            try
            {
                H5::DataSet datasetPulseEnergy = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_ENERGY );
                outFloatDataValue[0] = spdFile->getPulseEnergy();
                datasetPulseEnergy.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            }
            catch (H5::Exception &e)
            {
                float outFloatDataValue[1];
                H5::FloatType floatDataTypeDisk( H5::PredType::IEEE_F32LE );
                hsize_t dimsValue[1];
                dimsValue[0] = 1;
                H5::DataSpace singleValueDataspace(1, dimsValue);
                H5::DataSet datasetPulseEnergy = spdH5File->createDataSet( SPDFILE_DATASETNAME_PULSE_ENERGY, floatDataTypeDisk, singleValueDataspace);
                outFloatDataValue[0] = spdFile->getPulseEnergy();
                datasetPulseEnergy.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            }
            
            try
            {
                H5::DataSet datasetFieldOfView = spdH5File->openDataSet( SPDFILE_DATASETNAME_FIELD_OF_VIEW );
                outFloatDataValue[0] = spdFile->getFieldOfView();
                datasetFieldOfView.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            }
            catch (H5::Exception &e)
            {
                float outFloatDataValue[1];
                H5::FloatType floatDataTypeDisk( H5::PredType::IEEE_F32LE );
                hsize_t dimsValue[1];
                dimsValue[0] = 1;
                H5::DataSpace singleValueDataspace(1, dimsValue);
                H5::DataSet datasetFieldOfView = spdH5File->createDataSet( SPDFILE_DATASETNAME_FIELD_OF_VIEW, floatDataTypeDisk, singleValueDataspace);
                outFloatDataValue[0] = spdFile->getFieldOfView();
                datasetFieldOfView.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
            }
            
            try
            {
                H5::DataSet datasetNumOfWavelengths = spdH5File->openDataSet( SPDFILE_DATASETNAME_NUM_OF_WAVELENGTHS );
                out16bitUintDataValue[0] = spdFile->getNumOfWavelengths();
                datasetNumOfWavelengths.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
                
                if(spdFile->getNumOfWavelengths() > 0)
                {
                    hsize_t dimsWavelengthsValue[1];
                    dimsWavelengthsValue[0] = spdFile->getNumOfWavelengths();
                    H5::DataSpace wavelengthsDataSpace(1, dimsWavelengthsValue);
                    
                    float *dataVals = new float[spdFile->getNumOfWavelengths()];
                    for(unsigned int i = 0; i < spdFile->getNumOfWavelengths(); ++i)
                    {
                        dataVals[i] = spdFile->getWavelengths()->at(i);
                    }
                    H5::DataSet datasetWavelengths = spdH5File->openDataSet( SPDFILE_DATASETNAME_WAVELENGTHS );
                    datasetWavelengths.write(dataVals, H5::PredType::NATIVE_FLOAT );
                    
                    for(unsigned int i = 0; i < spdFile->getNumOfWavelengths(); ++i)
                    {
                        dataVals[i] = spdFile->getBandwidths()->at(i);
                    }
                    H5::DataSet datasetBandwidths = spdH5File->openDataSet( SPDFILE_DATASETNAME_BANDWIDTHS );
                    datasetBandwidths.write(dataVals, H5::PredType::NATIVE_FLOAT );
                    delete[] dataVals;
                }
                else
                {
                    std::vector<float> wavelengths;
                    spdFile->setWavelengths(wavelengths);
                    std::vector<float> bandwidths;
                    spdFile->setBandwidths(bandwidths);
                }
                
            }
            catch (H5::Exception &e)
            {
                H5::DataSet datasetWavelength = spdH5File->openDataSet( SPDFILE_DATASETNAME_WAVELENGTH );
                if(spdFile->getNumOfWavelengths() > 0)
                {
                    outFloatDataValue[0] = spdFile->getWavelengths()->front();
                    datasetWavelength.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
                }
                else
                {
                    outFloatDataValue[0] = 0;
                    datasetWavelength.write( outFloatDataValue, H5::PredType::NATIVE_FLOAT );
                }
                
                H5::IntType uint16bitDataTypeDisk( H5::PredType::STD_U16LE );
                hsize_t dimsValue[1];
                dimsValue[0] = 1;
                H5::DataSpace singleValueDataspace(1, dimsValue);
                H5::DataSet datasetNumOfWavelengths = spdH5File->createDataSet( SPDFILE_DATASETNAME_NUM_OF_WAVELENGTHS, uint16bitDataTypeDisk, singleValueDataspace);
                out16bitUintDataValue[0] = spdFile->getNumOfWavelengths();
                datasetNumOfWavelengths.write( out16bitUintDataValue, H5::PredType::NATIVE_UINT );
                
                hsize_t dimsWavelengthsValue[1];
                dimsWavelengthsValue[0] = spdFile->getNumOfWavelengths();
                H5::DataSpace wavelengthsDataSpace(1, dimsWavelengthsValue);
                H5::FloatType floatDataTypeDisk( H5::PredType::IEEE_F32LE );
                
                float *dataVals = new float[spdFile->getNumOfWavelengths()];
                for(unsigned int i = 0; i < spdFile->getNumOfWavelengths(); ++i)
                {
                    dataVals[i] = spdFile->getWavelengths()->at(i);
                }
                H5::DataSet datasetWavelengths = spdH5File->createDataSet( SPDFILE_DATASETNAME_WAVELENGTHS, floatDataTypeDisk, wavelengthsDataSpace );
                datasetWavelengths.write(dataVals, H5::PredType::NATIVE_FLOAT );
                
                for(unsigned int i = 0; i < spdFile->getNumOfWavelengths(); ++i)
                {
                    dataVals[i] = spdFile->getBandwidths()->at(i);
                }
                H5::DataSet datasetBandwidths = spdH5File->createDataSet( SPDFILE_DATASETNAME_BANDWIDTHS, floatDataTypeDisk, wavelengthsDataSpace );
                datasetBandwidths.write(dataVals, H5::PredType::NATIVE_FLOAT );
                delete[] dataVals;
            }
			spdH5File->flush(H5F_SCOPE_GLOBAL);
			spdH5File->close();
			delete spdH5File;
			
		}
		catch( H5::FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataTypeIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
        catch( SPDIOException &e )
		{
			throw e;
		}
	}
    
    void SPDFileWriter::readHeaderInfo(H5::H5File *spdH5File, SPDFile *spdFile) 
	{
		float inFloatDataValue[1];
		double inDoubleDataValue[1];
		int in16bitintDataValue[1];
		unsigned int in16bitUintDataValue[1];
		unsigned long in32bitUintDataValue[1];
		unsigned long long in64bitUintDataValue[1];
		
		unsigned int numLinesStr = 1;
		hid_t nativeStrType;
		char **strData = NULL;
		H5::DataType strDataType;
		
		hsize_t dimsValue[1];
		dimsValue[0] = 1;
		H5::DataSpace singleValueDataSpace(1, dimsValue);
		
		hsize_t	dims1Str[1];
		dims1Str[0] = numLinesStr;
		
		H5::StrType strTypeAll(0, H5T_VARIABLE);
		
		try
		{
			H5::Exception::dontPrint();
						
			if((H5T_STRING!=H5Tget_class(strTypeAll.getId())) || (!H5Tis_variable_str(strTypeAll.getId())))
			{
				throw SPDIOException("The string data type defined is not variable.");
			}
			
            try
            {
                H5::DataSet datasetMajorVersion = spdH5File->openDataSet( SPDFILE_DATASETNAME_MAJOR_VERSION );
                datasetMajorVersion.read(in16bitUintDataValue, H5::PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setMajorSPDVersion(in16bitUintDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("The SPD major version header value was not provided.");
            }
            
            try
            {
                H5::DataSet datasetMinorVersion = spdH5File->openDataSet( SPDFILE_DATASETNAME_MINOR_VERSION );
                datasetMinorVersion.read(in16bitUintDataValue, H5::PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setMinorSPDVersion(in16bitUintDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("The SPD minor version header value was not provided.");
            }
            
            try
            {
                H5::DataSet datasetPointVersion = spdH5File->openDataSet( SPDFILE_DATASETNAME_POINT_VERSION );
                datasetPointVersion.read(in16bitUintDataValue, H5::PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setPointVersion(in16bitUintDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("The SPD point version header value was not provided.");
            }
            
            try
            {
                H5::DataSet datasetPulseVersion = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_VERSION );
                datasetPulseVersion.read(in16bitUintDataValue, H5::PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setPulseVersion(in16bitUintDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("The SPD pulse version header value was not provided.");
            }
            
            try
            {
                H5::DataSet datasetSpatialReference = spdH5File->openDataSet( SPDFILE_DATASETNAME_SPATIAL_REFERENCE );
                strDataType = datasetSpatialReference.getDataType();
                strData = new char*[numLinesStr];
                if((nativeStrType=H5Tget_native_type(strDataType.getId(), H5T_DIR_DEFAULT))<0)
                {
                    throw SPDIOException("Could not define a native std::string type");
                }
                datasetSpatialReference.read((void*)strData, strDataType);
                spdFile->setSpatialReference(std::string(strData[0]));
                delete strData[0];
                delete[] strData;
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Spatial reference header value is not represent.");
            }
            
            try
            {
                H5::DataSet datasetFileType = spdH5File->openDataSet( SPDFILE_DATASETNAME_FILE_TYPE );
                datasetFileType.read(in16bitUintDataValue, H5::PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setFileType(in16bitUintDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                try
                {
                    H5::DataSet datasetBinSize = spdH5File->openDataSet( SPDFILE_DATASETNAME_BIN_SIZE );
                    H5::DataSet datasetNumberBinsX = spdH5File->openDataSet( SPDFILE_DATASETNAME_NUMBER_BINS_X );
                    H5::DataSet datasetNumberBinsY = spdH5File->openDataSet( SPDFILE_DATASETNAME_NUMBER_BINS_Y );
                    spdFile->setFileType(SPD_SEQ_TYPE);
                }
                catch (H5::Exception &e)
                {
                    spdFile->setFileType(SPD_UPD_TYPE);
                }
            }
            
            try
            {
                H5::DataSet datasetIndexType = spdH5File->openDataSet( SPDFILE_DATASETNAME_INDEX_TYPE );
                datasetIndexType.read(in16bitUintDataValue, H5::PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setIndexType(in16bitUintDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                spdFile->setIndexType(SPD_NO_IDX);
                std::cerr << "Warning: Index type header value not provided. Defaulting to non-indexed file.\n";
            }
            
            try
            {
                H5::DataSet datasetDiscreteDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_DISCRETE_PT_DEFINED );
                datasetDiscreteDefined.read(in16bitintDataValue, H5::PredType::NATIVE_INT, singleValueDataSpace);
                spdFile->setDiscretePtDefined(in16bitintDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Discrete Point Defined header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetDecomposedDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_DECOMPOSED_PT_DEFINED );
                datasetDecomposedDefined.read(in16bitintDataValue, H5::PredType::NATIVE_INT, singleValueDataSpace);
                spdFile->setDecomposedPtDefined(in16bitintDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Decomposed Point Defined header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetTransWaveformDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_TRANS_WAVEFORM_DEFINED );
                datasetTransWaveformDefined.read(in16bitintDataValue, H5::PredType::NATIVE_INT, singleValueDataSpace);
                spdFile->setTransWaveformDefined(in16bitintDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Transmitted Waveform Defined header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetReceiveWaveformDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_RECEIVE_WAVEFORM_DEFINED );
                datasetReceiveWaveformDefined.read(in16bitintDataValue, H5::PredType::NATIVE_INT, singleValueDataSpace);
                spdFile->setReceiveWaveformDefined(in16bitintDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Received Waveform Defined header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetGeneratingSoftware = spdH5File->openDataSet( SPDFILE_DATASETNAME_GENERATING_SOFTWARE );
                strDataType = datasetGeneratingSoftware.getDataType();
                strData = new char*[numLinesStr];
                if((nativeStrType=H5Tget_native_type(strDataType.getId(), H5T_DIR_DEFAULT))<0)
                {
                    throw SPDIOException("Could not define a native std::string type");
                }
                datasetGeneratingSoftware.read((void*)strData, strDataType);
                spdFile->setGeneratingSoftware(std::string(strData[0]));
                delete strData[0];
                delete[] strData;
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Generating software header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetSystemIdentifier = spdH5File->openDataSet( SPDFILE_DATASETNAME_SYSTEM_IDENTIFIER );
                strDataType = datasetSystemIdentifier.getDataType();
                strData = new char*[numLinesStr];
                if((nativeStrType=H5Tget_native_type(strDataType.getId(), H5T_DIR_DEFAULT))<0)
                {
                    throw SPDIOException("Could not define a native std::string type");
                }
                datasetSystemIdentifier.read((void*)strData, strDataType);
                spdFile->setSystemIdentifier(std::string(strData[0]));
                delete strData[0];
                delete[] strData;
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("System identifier header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetFileSignature = spdH5File->openDataSet( SPDFILE_DATASETNAME_FILE_SIGNATURE );
                strDataType = datasetFileSignature.getDataType();
                strData = new char*[numLinesStr];
                if((nativeStrType=H5Tget_native_type(strDataType.getId(), H5T_DIR_DEFAULT))<0)
                {
                    throw SPDIOException("Could not define a native std::string type");
                }
                datasetFileSignature.read((void*)strData, strDataType);
                spdFile->setFileSignature(std::string(strData[0]));
                delete strData[0];
                delete[] strData;
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("File signature header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetYearOfCreation = spdH5File->openDataSet( SPDFILE_DATASETNAME_YEAR_OF_CREATION );
                H5::DataSet datasetMonthOfCreation = spdH5File->openDataSet( SPDFILE_DATASETNAME_MONTH_OF_CREATION );
                H5::DataSet datasetDayOfCreation = spdH5File->openDataSet( SPDFILE_DATASETNAME_DAY_OF_CREATION );
                H5::DataSet datasetHourOfCreation = spdH5File->openDataSet( SPDFILE_DATASETNAME_HOUR_OF_CREATION );
                H5::DataSet datasetMinuteOfCreation = spdH5File->openDataSet( SPDFILE_DATASETNAME_MINUTE_OF_CREATION );
                H5::DataSet datasetSecondOfCreation = spdH5File->openDataSet( SPDFILE_DATASETNAME_SECOND_OF_CREATION );
                
                datasetYearOfCreation.read(in16bitUintDataValue, H5::PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setYearOfCreation(in16bitUintDataValue[0]);
                
                datasetMonthOfCreation.read(in16bitUintDataValue, H5::PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setMonthOfCreation(in16bitUintDataValue[0]);
                
                datasetDayOfCreation.read(in16bitUintDataValue, H5::PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setDayOfCreation(in16bitUintDataValue[0]);
                
                datasetHourOfCreation.read(in16bitUintDataValue, H5::PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setHourOfCreation(in16bitUintDataValue[0]);
                
                datasetMinuteOfCreation.read(in16bitUintDataValue, H5::PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setMinuteOfCreation(in16bitUintDataValue[0]);
                
                datasetSecondOfCreation.read(in16bitUintDataValue, H5::PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setSecondOfCreation(in16bitUintDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Date of file creation header values not provided.");
            }
            
            try
            {
                H5::DataSet datasetYearOfCapture = spdH5File->openDataSet( SPDFILE_DATASETNAME_YEAR_OF_CAPTURE );
                H5::DataSet datasetMonthOfCapture = spdH5File->openDataSet( SPDFILE_DATASETNAME_MONTH_OF_CAPTURE );
                H5::DataSet datasetDayOfCapture = spdH5File->openDataSet( SPDFILE_DATASETNAME_DAY_OF_CAPTURE );
                H5::DataSet datasetHourOfCapture = spdH5File->openDataSet( SPDFILE_DATASETNAME_HOUR_OF_CAPTURE );
                H5::DataSet datasetMinuteOfCapture = spdH5File->openDataSet( SPDFILE_DATASETNAME_MINUTE_OF_CAPTURE );
                H5::DataSet datasetSecondOfCapture = spdH5File->openDataSet( SPDFILE_DATASETNAME_SECOND_OF_CAPTURE );
                
                datasetYearOfCapture.read(in16bitUintDataValue, H5::PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setYearOfCapture(in16bitUintDataValue[0]);
                
                datasetMonthOfCapture.read(in16bitUintDataValue, H5::PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setMonthOfCapture(in16bitUintDataValue[0]);
                
                datasetDayOfCapture.read(in16bitUintDataValue, H5::PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setDayOfCapture(in16bitUintDataValue[0]);
                
                datasetHourOfCapture.read(in16bitUintDataValue, H5::PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setHourOfCapture(in16bitUintDataValue[0]);
                
                datasetMinuteOfCapture.read(in16bitUintDataValue, H5::PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setMinuteOfCapture(in16bitUintDataValue[0]);
                
                datasetSecondOfCapture.read(in16bitUintDataValue, H5::PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setSecondOfCapture(in16bitUintDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Date/Time of capture header values not provided.");
            }
            
            try
            {
                H5::DataSet datasetNumberOfPoints = spdH5File->openDataSet( SPDFILE_DATASETNAME_NUMBER_OF_POINTS );
                datasetNumberOfPoints.read(in64bitUintDataValue, H5::PredType::NATIVE_ULLONG, singleValueDataSpace);
                spdFile->setNumberOfPoints(in64bitUintDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Number of points header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetNumberOfPulses = spdH5File->openDataSet( SPDFILE_DATASETNAME_NUMBER_OF_PULSES );
                datasetNumberOfPulses.read(in64bitUintDataValue, H5::PredType::NATIVE_ULLONG, singleValueDataSpace);
                spdFile->setNumberOfPulses(in64bitUintDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Number of pulses header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetUserMetaData = spdH5File->openDataSet( SPDFILE_DATASETNAME_USER_META_DATA );
                strDataType = datasetUserMetaData.getDataType();
                strData = new char*[numLinesStr];
                if((nativeStrType=H5Tget_native_type(strDataType.getId(), H5T_DIR_DEFAULT))<0)
                {
                    throw SPDIOException("Could not define a native std::string type");
                }
                datasetUserMetaData.read((void*)strData, strDataType);
                spdFile->setUserMetaField(std::string(strData[0]));
                delete strData[0];
                delete[] strData;
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("User metadata header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetXMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_X_MIN );
                H5::DataSet datasetXMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_X_MAX );
                H5::DataSet datasetYMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_Y_MIN );
                H5::DataSet datasetYMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_Y_MAX );
                H5::DataSet datasetZMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_Z_MIN );
                H5::DataSet datasetZMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_Z_MAX );
                
                datasetXMin.read(inDoubleDataValue, H5::PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setXMin(inDoubleDataValue[0]);
                
                datasetXMax.read(inDoubleDataValue, H5::PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setXMax(inDoubleDataValue[0]);
                
                datasetYMin.read(inDoubleDataValue, H5::PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setYMin(inDoubleDataValue[0]);
                
                datasetYMax.read(inDoubleDataValue, H5::PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setYMax(inDoubleDataValue[0]);
                
                datasetZMin.read(inDoubleDataValue, H5::PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setZMin(inDoubleDataValue[0]);
                
                datasetZMax.read(inDoubleDataValue, H5::PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setZMax(inDoubleDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Dataset bounding volume header values not provided.");
            }
            
            try
            {
                H5::DataSet datasetZenithMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_ZENITH_MIN );
                H5::DataSet datasetZenithMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_ZENITH_MAX );
                H5::DataSet datasetAzimuthMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_AZIMUTH_MIN );
                H5::DataSet datasetAzimuthMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_AZIMUTH_MAX );
                H5::DataSet datasetRangeMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_RANGE_MIN );
                H5::DataSet datasetRangeMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_RANGE_MAX );
                
                datasetZenithMin.read(inDoubleDataValue, H5::PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setZenithMin(inDoubleDataValue[0]);
                
                datasetZenithMax.read(inDoubleDataValue, H5::PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setZenithMax(inDoubleDataValue[0]);
                
                datasetAzimuthMax.read(inDoubleDataValue, H5::PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setAzimuthMax(inDoubleDataValue[0]);
                
                datasetAzimuthMin.read(inDoubleDataValue, H5::PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setAzimuthMin(inDoubleDataValue[0]);
                
                datasetRangeMax.read(inDoubleDataValue, H5::PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setRangeMax(inDoubleDataValue[0]);
                
                datasetRangeMin.read(inDoubleDataValue, H5::PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setRangeMin(inDoubleDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Bounding spherical volume header values not provided.");
            }
            
            try
            {
                H5::DataSet datasetScanlineMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_SCANLINE_MIN );
                H5::DataSet datasetScanlineMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_SCANLINE_MAX );
                H5::DataSet datasetScanlineIdxMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_SCANLINE_IDX_MIN );
                H5::DataSet datasetScanlineIdxMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_SCANLINE_IDX_MAX );
                
                datasetScanlineMin.read(inDoubleDataValue, H5::PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setScanlineMin(inDoubleDataValue[0]);
                
                datasetScanlineMax.read(inDoubleDataValue, H5::PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setScanlineMax(inDoubleDataValue[0]);
                
                datasetScanlineIdxMin.read(inDoubleDataValue, H5::PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setScanlineIdxMax(inDoubleDataValue[0]);
                
                datasetScanlineIdxMax.read(inDoubleDataValue, H5::PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setScanlineIdxMin(inDoubleDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                spdFile->setScanlineMin(0);
                spdFile->setScanlineMax(0);
                spdFile->setScanlineIdxMax(0);
                spdFile->setScanlineIdxMin(0);
            }
            
            if(spdFile->getFileType() != SPD_UPD_TYPE)
            {
                try
                {
                    H5::DataSet datasetBinSize = spdH5File->openDataSet( SPDFILE_DATASETNAME_BIN_SIZE );
                    datasetBinSize.read(inFloatDataValue, H5::PredType::NATIVE_FLOAT, singleValueDataSpace);
                    spdFile->setBinSize(inFloatDataValue[0]);
                }
                catch (H5::Exception &e)
                {
                    throw SPDIOException("Bin size header value not provided.");
                }
                
                try
                {
                    H5::DataSet datasetNumberBinsX = spdH5File->openDataSet( SPDFILE_DATASETNAME_NUMBER_BINS_X );
                    datasetNumberBinsX.read(in32bitUintDataValue, H5::PredType::NATIVE_ULONG, singleValueDataSpace);
                    spdFile->setNumberBinsX(in32bitUintDataValue[0]);
                }
                catch (H5::Exception &e)
                {
                    throw SPDIOException("Number of X bins header value not provided.");
                }
                
                try
                {
                    H5::DataSet datasetNumberBinsY = spdH5File->openDataSet( SPDFILE_DATASETNAME_NUMBER_BINS_Y );
                    datasetNumberBinsY.read(in32bitUintDataValue, H5::PredType::NATIVE_ULONG, singleValueDataSpace);
                    spdFile->setNumberBinsY(in32bitUintDataValue[0]);
                }
                catch (H5::Exception &e)
                {
                    throw SPDIOException("Number of Y bins header value not provided.");
                }
            }
            
            try
            {
                H5::DataSet datasetPulseRepFreq = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_REPETITION_FREQ );
                datasetPulseRepFreq.read(inFloatDataValue, H5::PredType::NATIVE_FLOAT, singleValueDataSpace);
                spdFile->setPulseRepetitionFreq(inFloatDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Pulse repetition frequency header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetBeamDivergence = spdH5File->openDataSet( SPDFILE_DATASETNAME_BEAM_DIVERGENCE );
                datasetBeamDivergence.read(inFloatDataValue, H5::PredType::NATIVE_FLOAT, singleValueDataSpace);
                spdFile->setBeamDivergence(inFloatDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Beam divergence header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetSensorHeight = spdH5File->openDataSet( SPDFILE_DATASETNAME_SENSOR_HEIGHT );
                datasetSensorHeight.read(inDoubleDataValue, H5::PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setSensorHeight(inDoubleDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Sensor height header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetFootprint = spdH5File->openDataSet( SPDFILE_DATASETNAME_FOOTPRINT );
                datasetFootprint.read(inFloatDataValue, H5::PredType::NATIVE_FLOAT, singleValueDataSpace);
                spdFile->setFootprint(inFloatDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Footprint header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetMaxScanAngle = spdH5File->openDataSet( SPDFILE_DATASETNAME_MAX_SCAN_ANGLE );
                datasetMaxScanAngle.read(inFloatDataValue, H5::PredType::NATIVE_FLOAT, singleValueDataSpace);
                spdFile->setMaxScanAngle(inFloatDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Max scan angle header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetRGBDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_RGB_DEFINED );
                datasetRGBDefined.read(in16bitintDataValue, H5::PredType::NATIVE_INT, singleValueDataSpace);
                spdFile->setRGBDefined(in16bitintDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("RGB defined header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetPulseBlockSize = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_BLOCK_SIZE );
                datasetPulseBlockSize.read(in16bitUintDataValue, H5::PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setPulseBlockSize(in16bitUintDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Pulse block size header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetPointsBlockSize = spdH5File->openDataSet( SPDFILE_DATASETNAME_POINT_BLOCK_SIZE );
                datasetPointsBlockSize.read(in16bitUintDataValue, H5::PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setPointBlockSize(in16bitUintDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Point block size header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetReceivedBlockSize = spdH5File->openDataSet( SPDFILE_DATASETNAME_RECEIVED_BLOCK_SIZE );
                datasetReceivedBlockSize.read(in16bitUintDataValue, H5::PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setReceivedBlockSize(in16bitUintDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Received waveform block size header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetTransmittedBlockSize = spdH5File->openDataSet( SPDFILE_DATASETNAME_TRANSMITTED_BLOCK_SIZE );
                datasetTransmittedBlockSize.read(in16bitUintDataValue, H5::PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setTransmittedBlockSize(in16bitUintDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Transmitted waveform block size header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetWaveformBitRes = spdH5File->openDataSet( SPDFILE_DATASETNAME_WAVEFORM_BIT_RES );
                datasetWaveformBitRes.read(in16bitUintDataValue, H5::PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setWaveformBitRes(in16bitUintDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Waveform bit resolution header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetTemporalBinSpacing = spdH5File->openDataSet( SPDFILE_DATASETNAME_TEMPORAL_BIN_SPACING );
                datasetTemporalBinSpacing.read(inDoubleDataValue, H5::PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setTemporalBinSpacing(inDoubleDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Temporal bin spacing header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetReturnNumsSynGen = spdH5File->openDataSet( SPDFILE_DATASETNAME_RETURN_NUMBERS_SYN_GEN );
                datasetReturnNumsSynGen.read(in16bitintDataValue, H5::PredType::NATIVE_INT, singleValueDataSpace);
                spdFile->setReturnNumsSynGen(in16bitintDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Return number synthetically generated header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetHeightDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_HEIGHT_DEFINED );
                datasetHeightDefined.read(in16bitintDataValue, H5::PredType::NATIVE_INT, singleValueDataSpace);
                spdFile->setHeightDefined(in16bitintDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Height fields defined header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetSensorSpeed = spdH5File->openDataSet( SPDFILE_DATASETNAME_SENSOR_SPEED );
                datasetSensorSpeed.read(inFloatDataValue, H5::PredType::NATIVE_FLOAT, singleValueDataSpace);
                spdFile->setSensorSpeed(inFloatDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Sensor speed header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetSensorScanRate = spdH5File->openDataSet( SPDFILE_DATASETNAME_SENSOR_SCAN_RATE );
                datasetSensorScanRate.read(inFloatDataValue, H5::PredType::NATIVE_FLOAT, singleValueDataSpace);
                spdFile->setSensorScanRate(inFloatDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Sensor Scan Rate header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetPointDensity = spdH5File->openDataSet( SPDFILE_DATASETNAME_POINT_DENSITY );
                datasetPointDensity.read(inFloatDataValue, H5::PredType::NATIVE_FLOAT, singleValueDataSpace);
                spdFile->setPointDensity(inFloatDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Point density header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetPulseDensity = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_DENSITY );
                datasetPulseDensity.read(inFloatDataValue, H5::PredType::NATIVE_FLOAT, singleValueDataSpace);
                spdFile->setPulseDensity(inFloatDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Pulse density header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetPulseCrossTrackSpacing = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_CROSS_TRACK_SPACING );
                datasetPulseCrossTrackSpacing.read(inFloatDataValue, H5::PredType::NATIVE_FLOAT, singleValueDataSpace);
                spdFile->setPulseCrossTrackSpacing(inFloatDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Cross track spacing header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetPulseAlongTrackSpacing = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_ALONG_TRACK_SPACING );
                datasetPulseAlongTrackSpacing.read(inFloatDataValue, H5::PredType::NATIVE_FLOAT, singleValueDataSpace);
                spdFile->setPulseAlongTrackSpacing(inFloatDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Along track spacing header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetOriginDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_ORIGIN_DEFINED );
                datasetOriginDefined.read(in16bitintDataValue, H5::PredType::NATIVE_INT, singleValueDataSpace);
                spdFile->setOriginDefined(in16bitintDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Origin defined header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetPulseAngularSpacingAzimuth = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_ANGULAR_SPACING_AZIMUTH );
                datasetPulseAngularSpacingAzimuth.read(inFloatDataValue, H5::PredType::NATIVE_FLOAT, singleValueDataSpace);
                spdFile->setPulseAngularSpacingAzimuth(inFloatDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Angular azimuth spacing header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetPulseAngularSpacingZenith = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_ANGULAR_SPACING_ZENITH );
                datasetPulseAngularSpacingZenith.read(inFloatDataValue, H5::PredType::NATIVE_FLOAT, singleValueDataSpace);
                spdFile->setPulseAngularSpacingZenith(inFloatDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                throw SPDIOException("Angular Zenith spacing header value not provided.");
            }
            
            try
            {
                H5::DataSet datasetPulseIndexMethod = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_INDEX_METHOD );
                datasetPulseIndexMethod.read(in16bitintDataValue, H5::PredType::NATIVE_INT, singleValueDataSpace);
                spdFile->setPulseIdxMethod(in16bitintDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                spdFile->setPulseIdxMethod(SPD_FIRST_RETURN);
                std::cerr << "Method of indexing header value not provided. Default: First Return\n";
            }
            
            try
            {
                H5::DataSet datasetSensorApertureSize = spdH5File->openDataSet( SPDFILE_DATASETNAME_SENSOR_APERTURE_SIZE );
                datasetSensorApertureSize.read(inFloatDataValue, H5::PredType::NATIVE_FLOAT, singleValueDataSpace);
                spdFile->setSensorApertureSize(inFloatDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                //ignore
                spdFile->setSensorApertureSize(0);
            }
            
            try
            {
                H5::DataSet datasetPulseEnergy = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_ENERGY );
                datasetPulseEnergy.read(inFloatDataValue, H5::PredType::NATIVE_FLOAT, singleValueDataSpace);
                spdFile->setPulseEnergy(inFloatDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                //ignore
                spdFile->setPulseEnergy(0);
            }
            
            try
            {
                H5::DataSet datasetFieldOfView = spdH5File->openDataSet( SPDFILE_DATASETNAME_FIELD_OF_VIEW );
                datasetFieldOfView.read(inFloatDataValue, H5::PredType::NATIVE_FLOAT, singleValueDataSpace);
                spdFile->setFieldOfView(inFloatDataValue[0]);
            }
            catch (H5::Exception &e)
            {
                //ignore
                spdFile->setFieldOfView(0);
            }
            
            try
            {
                H5::DataSet datasetNumOfWavelengths = spdH5File->openDataSet( SPDFILE_DATASETNAME_NUM_OF_WAVELENGTHS );
                datasetNumOfWavelengths.read(in16bitUintDataValue, H5::PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setNumOfWavelengths(in16bitUintDataValue[0]);
                
                if(in16bitUintDataValue[0] > 0)
                {
                    float *inFloatDataValues = new float[in16bitUintDataValue[0]];
                    hsize_t dimsValue[1];
                    dimsValue[0] = in16bitUintDataValue[0];
                    H5::DataSpace valuesDataSpace(1, dimsValue);
                    H5::DataSet datasetWavelengths = spdH5File->openDataSet( SPDFILE_DATASETNAME_WAVELENGTHS );
                    datasetWavelengths.read(inFloatDataValues, H5::PredType::NATIVE_FLOAT, valuesDataSpace);
                    std::vector<float> wavelengths;
                    for(unsigned int i = 0; i < in16bitUintDataValue[0]; ++i)
                    {
                        wavelengths.push_back(inFloatDataValues[i]);
                    }
                    spdFile->setWavelengths(wavelengths);
                    
                    H5::DataSet datasetBandwidths = spdH5File->openDataSet( SPDFILE_DATASETNAME_BANDWIDTHS );
                    datasetWavelengths.read(inFloatDataValues, H5::PredType::NATIVE_FLOAT, valuesDataSpace);
                    std::vector<float> bandwidths;
                    for(unsigned int i = 0; i < in16bitUintDataValue[0]; ++i)
                    {
                        bandwidths.push_back(inFloatDataValues[i]);
                    }
                    spdFile->setBandwidths(bandwidths);
                    delete[] inFloatDataValues;
                }
                else
                {
                    std::vector<float> wavelengths;
                    spdFile->setWavelengths(wavelengths);
                    std::vector<float> bandwidths;
                    spdFile->setBandwidths(bandwidths);
                }
                
            }
            catch (H5::Exception &e)
            {
                H5::DataSet datasetWavelength = spdH5File->openDataSet( SPDFILE_DATASETNAME_WAVELENGTH );
                datasetWavelength.read(inFloatDataValue, H5::PredType::NATIVE_FLOAT, singleValueDataSpace);
                spdFile->setNumOfWavelengths(1);
                std::vector<float> wavelengths;
                wavelengths.push_back(inFloatDataValue[0]);
                spdFile->setWavelengths(wavelengths);
                std::vector<float> bandwidths;
                bandwidths.push_back(0);
                spdFile->setBandwidths(bandwidths);
            }
			
		}
		catch( H5::FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataTypeIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
	}
    
    
    
    
    
	SPDSeqFileWriter::SPDSeqFileWriter() : SPDDataExporter("SPD-SEQ"), spdOutH5File(NULL), pulsesDataset(NULL), spdPulseDataType(NULL), pointsDataset(NULL), spdPointDataType(NULL), datasetPlsPerBin(NULL), datasetBinsOffset(NULL), receivedDataset(NULL), transmittedDataset(NULL), datasetQuicklook(NULL), numPulses(0), numPts(0), numTransVals(0), numReceiveVals(0), firstColumn(true), nextCol(0), nextRow(0), numCols(0), numRows(0)
	{
		this->keepMinExtent = false;
	}
	
	SPDSeqFileWriter::SPDSeqFileWriter(const SPDDataExporter &dataExporter)  : SPDDataExporter(dataExporter), spdOutH5File(NULL), pulsesDataset(NULL), spdPulseDataType(NULL), pointsDataset(NULL), spdPointDataType(NULL), datasetPlsPerBin(NULL), datasetBinsOffset(NULL), receivedDataset(NULL), transmittedDataset(NULL), datasetQuicklook(NULL), numPulses(0), numPts(0), numTransVals(0), numReceiveVals(0), firstColumn(true), nextCol(0), nextRow(0), numCols(0), numRows(0)
	{
		if(fileOpened)
		{
			throw SPDException("Cannot make a copy of a file exporter when a file is open.");
		}
	}
	
	SPDSeqFileWriter& SPDSeqFileWriter::operator=(const SPDSeqFileWriter& dataExporter) 
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
	
	bool SPDSeqFileWriter::open(SPDFile *spdFile, std::string outputFile) 
	{
		SPDPulseUtils pulseUtils;
		SPDPointUtils pointUtils;
		this->spdFile = spdFile;
		this->outputFile = outputFile;
		
		const H5std_string spdFilePath( outputFile );
		
		try 
		{
			H5::Exception::dontPrint() ;
			
			// Create File..
            try
			{
				spdOutH5File = new H5::H5File( spdFilePath, H5F_ACC_TRUNC  );
			}
			catch (H5::FileIException &e)
			{
				std::string message  = std::string("Could not create SPD file: ") + spdFilePath;
				throw SPDIOException(message);
			}
			
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
			H5::DataSpace pulseDataSpace = H5::DataSpace(1, initDimsPulseDS, maxDimsPulseDS);
			
			hsize_t dimsPulseChunk[1];
			dimsPulseChunk[0] = spdFile->getPulseBlockSize();
			
			H5::DSetCreatPropList creationPulseDSPList;
			creationPulseDSPList.setChunk(1, dimsPulseChunk);
            creationPulseDSPList.setShuffle();
			creationPulseDSPList.setDeflate(SPD_DEFLATE);
            
            H5::CompType *spdPulseDataTypeDisk = NULL;
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
			pulsesDataset = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_PULSES, *spdPulseDataTypeDisk, pulseDataSpace, creationPulseDSPList));
			
			// Create DataType, DataSpace and Dataset for Points
			hsize_t initDimsPtsDS[1];
			initDimsPtsDS[0] = 0;
			hsize_t maxDimsPtsDS[1];
			maxDimsPtsDS[0] = H5S_UNLIMITED;
			H5::DataSpace ptsDataSpace = H5::DataSpace(1, initDimsPtsDS, maxDimsPtsDS);
			
			hsize_t dimsPtsChunk[1];
			dimsPtsChunk[0] = spdFile->getPointBlockSize();
			
			H5::DSetCreatPropList creationPtsDSPList;
			creationPtsDSPList.setChunk(1, dimsPtsChunk);			
			creationPtsDSPList.setShuffle();
            creationPtsDSPList.setDeflate(SPD_DEFLATE);
            
            H5::CompType *spdPointDataTypeDisk = NULL;
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
			pointsDataset = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_POINTS, *spdPointDataTypeDisk, ptsDataSpace, creationPtsDSPList));
			
			// Create transmitted and received DataSpace and Dataset
			hsize_t initDimsWaveformDS[1];
			initDimsWaveformDS[0] = 0;
			hsize_t maxDimsWaveformDS[1];
			maxDimsWaveformDS[0] = H5S_UNLIMITED;
			H5::DataSpace waveformDataSpace = H5::DataSpace(1, initDimsWaveformDS, maxDimsWaveformDS);
			
			hsize_t dimsReceivedChunk[1];
			dimsReceivedChunk[0] = spdFile->getReceivedBlockSize();
            
			hsize_t dimsTransmittedChunk[1];
			dimsTransmittedChunk[0] = spdFile->getTransmittedBlockSize();
			
            if(spdFile->getWaveformBitRes() == SPD_32_BIT_WAVE)
            {
                H5::IntType intU32DataType( H5::PredType::STD_U32LE );
                intU32DataType.setOrder( H5T_ORDER_LE );
                
                boost::uint_fast32_t fillValueUInt = 0;
                H5::DSetCreatPropList creationReceivedDSPList;
                creationReceivedDSPList.setChunk(1, dimsReceivedChunk);
                creationReceivedDSPList.setShuffle();
                creationReceivedDSPList.setDeflate(SPD_DEFLATE);
                creationReceivedDSPList.setFillValue( H5::PredType::STD_U32LE, &fillValueUInt);
                
                H5::DSetCreatPropList creationTransmittedDSPList;
                creationTransmittedDSPList.setChunk(1, dimsTransmittedChunk);			
                creationTransmittedDSPList.setShuffle();
                creationTransmittedDSPList.setDeflate(SPD_DEFLATE);
                creationTransmittedDSPList.setFillValue( H5::PredType::STD_U32LE, &fillValueUInt);
                
                receivedDataset = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_RECEIVED, intU32DataType, waveformDataSpace, creationReceivedDSPList));
                transmittedDataset = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_TRANSMITTED, intU32DataType, waveformDataSpace, creationTransmittedDSPList));
            }
            else if(spdFile->getWaveformBitRes() == SPD_16_BIT_WAVE)
            {
                H5::IntType intU16DataType( H5::PredType::STD_U16LE );
                intU16DataType.setOrder( H5T_ORDER_LE );
                
                boost::uint_fast32_t fillValueUInt = 0;
                H5::DSetCreatPropList creationReceivedDSPList;
                creationReceivedDSPList.setChunk(1, dimsReceivedChunk);			
                creationReceivedDSPList.setShuffle();
                creationReceivedDSPList.setDeflate(SPD_DEFLATE);
                creationReceivedDSPList.setFillValue( H5::PredType::STD_U16LE, &fillValueUInt);
                
                H5::DSetCreatPropList creationTransmittedDSPList;
                creationTransmittedDSPList.setChunk(1, dimsTransmittedChunk);			
                creationTransmittedDSPList.setShuffle();
                creationTransmittedDSPList.setDeflate(SPD_DEFLATE);
                creationTransmittedDSPList.setFillValue( H5::PredType::STD_U16LE, &fillValueUInt);
                
                receivedDataset = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_RECEIVED, intU16DataType, waveformDataSpace, creationReceivedDSPList));
                transmittedDataset = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_TRANSMITTED, intU16DataType, waveformDataSpace, creationTransmittedDSPList));
            }
            else if(spdFile->getWaveformBitRes() == SPD_8_BIT_WAVE)
            {
                H5::IntType intU8DataType( H5::PredType::STD_U8LE );
                intU8DataType.setOrder( H5T_ORDER_LE );
                
                boost::uint_fast32_t fillValueUInt = 0;
                H5::DSetCreatPropList creationReceivedDSPList;
                creationReceivedDSPList.setChunk(1, dimsReceivedChunk);			
                creationReceivedDSPList.setShuffle();
                creationReceivedDSPList.setDeflate(SPD_DEFLATE);
                creationReceivedDSPList.setFillValue( H5::PredType::STD_U8LE, &fillValueUInt);
                
                H5::DSetCreatPropList creationTransmittedDSPList;
                creationTransmittedDSPList.setChunk(1, dimsTransmittedChunk);			
                creationTransmittedDSPList.setShuffle();
                creationTransmittedDSPList.setDeflate(SPD_DEFLATE);
                creationTransmittedDSPList.setFillValue( H5::PredType::STD_U8LE, &fillValueUInt);
                
                receivedDataset = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_RECEIVED, intU8DataType, waveformDataSpace, creationReceivedDSPList));
                transmittedDataset = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_TRANSMITTED, intU8DataType, waveformDataSpace, creationTransmittedDSPList));
            }
            else
            {
                throw SPDIOException("Waveform bit resolution is unknown.");
            }
			
			// Create Reference datasets and dataspaces		
			H5::IntType intU64DataType( H5::PredType::STD_U64LE );
            intU64DataType.setOrder( H5T_ORDER_LE );
            H5::IntType intU32DataType( H5::PredType::STD_U32LE );
            intU32DataType.setOrder( H5T_ORDER_LE );
			
			hsize_t initDimsIndexDS[2];
			initDimsIndexDS[0] = numRows;
			initDimsIndexDS[1] = numCols;
			H5::DataSpace indexDataSpace(2, initDimsIndexDS);
			
			hsize_t dimsIndexChunk[2];
			dimsIndexChunk[0] = 1;
			dimsIndexChunk[1] = numCols;
			
			boost::uint_fast32_t fillValue32bit = 0;
			H5::DSetCreatPropList initParamsIndexPulsesPerBin;
			initParamsIndexPulsesPerBin.setChunk(2, dimsIndexChunk);			
			initParamsIndexPulsesPerBin.setShuffle();
            initParamsIndexPulsesPerBin.setDeflate(SPD_DEFLATE);
			initParamsIndexPulsesPerBin.setFillValue( H5::PredType::STD_U32LE, &fillValue32bit);
			
			boost::uint_fast64_t fillValue64bit = 0;
			H5::DSetCreatPropList initParamsIndexOffset;
			initParamsIndexOffset.setChunk(2, dimsIndexChunk);			
			initParamsIndexOffset.setShuffle();
            initParamsIndexOffset.setDeflate(SPD_DEFLATE);
			initParamsIndexOffset.setFillValue( H5::PredType::STD_U64LE, &fillValue64bit);
			
			datasetPlsPerBin = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_PLS_PER_BIN, intU32DataType, indexDataSpace, initParamsIndexPulsesPerBin ));
			datasetBinsOffset = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_BIN_OFFSETS, intU64DataType, indexDataSpace, initParamsIndexOffset ));
			
			// Created Quicklook datasets and dataspaces
			H5::FloatType floatDataType( H5::PredType::IEEE_F32LE );
			
			hsize_t initDimsQuicklookDS[2];
			initDimsQuicklookDS[0] = numRows;
			initDimsQuicklookDS[1] = numCols;
			H5::DataSpace quicklookDataSpace(2, initDimsQuicklookDS);
			
			hsize_t dimsQuicklookChunk[2];
			dimsQuicklookChunk[0] = 1;
			dimsQuicklookChunk[1] = numCols;
			
			float fillValueFloatQKL = 0;
			H5::DSetCreatPropList initParamsQuicklook;
			initParamsQuicklook.setChunk(2, dimsQuicklookChunk);			
			initParamsQuicklook.setShuffle();
            initParamsQuicklook.setDeflate(SPD_DEFLATE);
			initParamsQuicklook.setFillValue( H5::PredType::IEEE_F32LE, &fillValueFloatQKL);
			
			datasetQuicklook = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_QKLIMAGE, floatDataType, quicklookDataSpace, initParamsQuicklook ));

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
            scanlineMinWritten = 0;
            scanlineMaxWritten = 0;
            scanlineIdxMinWritten = 0;
            scanlineIdxMaxWritten = 0; 
            
			firstReturn = true;
            firstPulse = true;
            
            bufIdxCol = 0;
            bufIdxRow = 0;
            numPulsesForBuf = 0;
            
            plsBuffer = new std::vector<SPDPulse*>();
            plsBuffer->reserve(spdFile->getPulseBlockSize());
            
            qkBuffer = new float[spdFile->getNumberBinsX()];
            plsInColBuf = new unsigned long[spdFile->getNumberBinsX()];
            plsOffsetBuf = new unsigned long long[spdFile->getNumberBinsX()];
            
		}
		catch( H5::FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataTypeIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		
		return fileOpened;
	}
    
    bool SPDSeqFileWriter::reopen(SPDFile *spdFile, std::string outputFile) 
    {
        throw SPDIOException("No reopen option available.");
    }
	
	void SPDSeqFileWriter::writeDataColumn(std::list<SPDPulse*> *pls, boost::uint_fast32_t col, boost::uint_fast32_t row)
	{
		SPDPulseUtils pulseUtils;
		SPDPointUtils pointUtils;
        
		if(!fileOpened)
		{
			throw SPDIOException("SPD (HDF5) file not open, cannot finalise.");
		}
        
        if(col >= numCols)
		{
			std::cout << "Number of Columns = " << numCols << std::endl;
			std::cout << col << std::endl;
			throw SPDIOException("The column you have specified it not within the current file.");
		}
		
		if(row >= numRows)
		{
			std::cout << "Number of Columns = " << numRows << std::endl;
			std::cout << row << std::endl;
			throw SPDIOException("The row you have specified it not within the current file.");
		}
		
        if((row != this->nextRow) & (col != this->nextCol))
        {
            std::cout << "The next expected row/column was[" << nextRow << "," << nextCol << "] [" << row << "," << col << "] was provided\n";
            
            throw SPDIOException("The column and row provided were not what was expected.");
        }
		
        if(col != bufIdxCol)
        {
            throw SPDIOException("The index buffer index and the column provided are out of sync.");
        }
        
		try 
		{
			H5::Exception::dontPrint() ;
			
            std::list<SPDPulse*>::iterator iterInPls;
            float qkVal = 0;
            //boost::uint_fast16_t numVals = 0;
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
                            for(std::vector<SPDPoint*>::iterator iterPts = (*iterInPls)->pts->begin(); iterPts != (*iterInPls)->pts->end(); ++iterPts)
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
                            //qkVal += (*iterInPls)->received[i];
                            //++numVals;
                            if(qkVal < (*iterInPls)->received[i])
                            {
                                qkVal = (*iterInPls)->received[i];
                            }           
                            
                        }
                    }
                    //qkVal = qkVal/numVals;
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
					
					std::vector<SPDPulse*>::iterator iterPulses;
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
                            scanlineMinWritten = (*iterPulses)->scanline;
                            scanlineMaxWritten = (*iterPulses)->scanline;
                            scanlineIdxMinWritten = (*iterPulses)->scanlineIdx;
                            scanlineIdxMaxWritten = (*iterPulses)->scanlineIdx;                      
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
                            
                            if((*iterPulses)->scanline < scanlineMinWritten)
                            {
                                scanlineMinWritten = (*iterPulses)->scanline;
                            }
                            else if((*iterPulses)->scanline > scanlineMaxWritten) 
                            {
                                scanlineMaxWritten = (*iterPulses)->scanline;
                            }
                            
                            if((*iterPulses)->scanlineIdx < scanlineIdxMinWritten)
                            {
                                scanlineIdxMinWritten = (*iterPulses)->scanlineIdx;
                            }
                            else if((*iterPulses)->scanlineIdx > scanlineIdxMaxWritten)  
                            {
                                scanlineIdxMaxWritten = (*iterPulses)->scanlineIdx;                            
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
					
					H5::DataSpace pulseWriteDataSpace = pulsesDataset->getSpace();
					pulseWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pulseDataDims, pulseDataOffset);
					H5::DataSpace newPulsesDataspace = H5::DataSpace(1, pulseDataDims);
					
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
						
						H5::DataSpace pointWriteDataSpace = pointsDataset->getSpace();
						pointWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pointsDataDims, pointsDataOffset);
						H5::DataSpace newPointsDataspace = H5::DataSpace(1, pointsDataDims);
						
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
						
						H5::DataSpace transWriteDataSpace = transmittedDataset->getSpace();
						transWriteDataSpace.selectHyperslab(H5S_SELECT_SET, transDataDims, transDataOffset);
						H5::DataSpace newTransDataspace = H5::DataSpace(1, transDataDims);
						
						transmittedDataset->write(transmittedValues, H5::PredType::NATIVE_ULONG, newTransDataspace, transWriteDataSpace);
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
						
						H5::DataSpace receivedWriteDataSpace = receivedDataset->getSpace();
						receivedWriteDataSpace.selectHyperslab(H5S_SELECT_SET, receivedDataDims, receivedDataOffset);
						H5::DataSpace newReceivedDataspace = H5::DataSpace(1, receivedDataDims);
						
						receivedDataset->write(receivedValues, H5::PredType::NATIVE_ULONG, newReceivedDataspace, receivedWriteDataSpace);
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
                
                H5::DataSpace plsWritePlsPerBinDataSpace = datasetPlsPerBin->getSpace();
                H5::DataSpace plsWriteOffsetsDataSpace = datasetBinsOffset->getSpace();
                H5::DataSpace plsWriteQKLDataSpace = datasetQuicklook->getSpace();
                
                hsize_t dataIndexOffset[2];
                dataIndexOffset[0] = bufIdxRow;
                dataIndexOffset[1] = 0;
                hsize_t dataIndexDims[2];
                dataIndexDims[0] = 1;
                dataIndexDims[1] = spdFile->getNumberBinsX();
                H5::DataSpace newIndexDataspace = H5::DataSpace(2, dataIndexDims);
                
                plsWritePlsPerBinDataSpace.selectHyperslab( H5S_SELECT_SET, dataIndexDims, dataIndexOffset );
                plsWriteOffsetsDataSpace.selectHyperslab( H5S_SELECT_SET, dataIndexDims, dataIndexOffset );
                plsWriteQKLDataSpace.selectHyperslab( H5S_SELECT_SET, dataIndexDims, dataIndexOffset );
                
                datasetPlsPerBin->write( plsInColBuf, H5::PredType::NATIVE_ULONG, newIndexDataspace, plsWritePlsPerBinDataSpace );
                datasetBinsOffset->write( plsOffsetBuf, H5::PredType::NATIVE_ULLONG, newIndexDataspace, plsWriteOffsetsDataSpace );
                datasetQuicklook->write( qkBuffer, H5::PredType::NATIVE_FLOAT, newIndexDataspace, plsWriteQKLDataSpace );
                
                bufIdxCol = 0;
                ++bufIdxRow;
            }
		}
		catch( H5::FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataTypeIException &e )
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
	
	void SPDSeqFileWriter::writeDataColumn(std::vector<SPDPulse*> *pls, boost::uint_fast32_t col, boost::uint_fast32_t row)
	{
        SPDPulseUtils pulseUtils;
		SPDPointUtils pointUtils;
        
		if(!fileOpened)
		{
			throw SPDIOException("SPD (HDF5) file not open, cannot finalise.");
		}
        
        if(col >= numCols)
		{
			std::cout << "Number of Columns = " << numCols << std::endl;
			std::cout << col << std::endl;
			throw SPDIOException("The column you have specified it not within the current file.");
		}
		
		if(row >= numRows)
		{
			std::cout << "Number of Columns = " << numRows << std::endl;
			std::cout << row << std::endl;
			throw SPDIOException("The row you have specified it not within the current file.");
		}
		
        if((row != this->nextRow) & (col != this->nextCol))
        {
            std::cout << "The next expected row/column was[" << nextRow << "," << nextCol << "] [" << row << "," << col << "] was provided\n";
            
            throw SPDIOException("The column and row provided were not what was expected.");
        }
		
        if(col != bufIdxCol)
        {
            throw SPDIOException("The index buffer index and the column provided are out of sync.");
        }
        
		try 
		{
			H5::Exception::dontPrint();
			
            std::vector<SPDPulse*>::iterator iterInPls;
            float qkVal = 0;
            //boost::uint_fast16_t numVals = 0;
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
                            for(std::vector<SPDPoint*>::iterator iterPts = (*iterInPls)->pts->begin(); iterPts != (*iterInPls)->pts->end(); ++iterPts)
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
                            //qkVal += (*iterInPls)->received[i];
                            //++numVals;
                            if(qkVal < (*iterInPls)->received[i])
                            {
                                qkVal = (*iterInPls)->received[i];
                            } 
                        }
                    }
                    //qkVal = qkVal/numVals;
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
					
					std::vector<SPDPulse*>::iterator iterPulses;
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
                            scanlineMinWritten = (*iterPulses)->scanline;
                            scanlineMaxWritten = (*iterPulses)->scanline;
                            scanlineIdxMinWritten = (*iterPulses)->scanlineIdx;
                            scanlineIdxMaxWritten = (*iterPulses)->scanlineIdx; 
                            
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
                            
                            if((*iterPulses)->scanline < scanlineMinWritten)
                            {
                                scanlineMinWritten = (*iterPulses)->scanline;
                            }
                            else if((*iterPulses)->scanline > scanlineMaxWritten) 
                            {
                                scanlineMaxWritten = (*iterPulses)->scanline;
                            }
                            
                            if((*iterPulses)->scanlineIdx < scanlineIdxMinWritten)
                            {
                                scanlineIdxMinWritten = (*iterPulses)->scanlineIdx;
                            }
                            else if((*iterPulses)->scanlineIdx > scanlineIdxMaxWritten)  
                            {
                                scanlineIdxMaxWritten = (*iterPulses)->scanlineIdx;                            
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
					
					H5::DataSpace pulseWriteDataSpace = pulsesDataset->getSpace();
					pulseWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pulseDataDims, pulseDataOffset);
					H5::DataSpace newPulsesDataspace = H5::DataSpace(1, pulseDataDims);
					
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
						
						H5::DataSpace pointWriteDataSpace = pointsDataset->getSpace();
						pointWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pointsDataDims, pointsDataOffset);
						H5::DataSpace newPointsDataspace = H5::DataSpace(1, pointsDataDims);
						
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
						
						H5::DataSpace transWriteDataSpace = transmittedDataset->getSpace();
						transWriteDataSpace.selectHyperslab(H5S_SELECT_SET, transDataDims, transDataOffset);
						H5::DataSpace newTransDataspace = H5::DataSpace(1, transDataDims);
						
						transmittedDataset->write(transmittedValues, H5::PredType::NATIVE_ULONG, newTransDataspace, transWriteDataSpace);
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
						
						H5::DataSpace receivedWriteDataSpace = receivedDataset->getSpace();
						receivedWriteDataSpace.selectHyperslab(H5S_SELECT_SET, receivedDataDims, receivedDataOffset);
						H5::DataSpace newReceivedDataspace = H5::DataSpace(1, receivedDataDims);
						
						receivedDataset->write(receivedValues, H5::PredType::NATIVE_ULONG, newReceivedDataspace, receivedWriteDataSpace);
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
                
                H5::DataSpace plsWritePlsPerBinDataSpace = datasetPlsPerBin->getSpace();
                H5::DataSpace plsWriteOffsetsDataSpace = datasetBinsOffset->getSpace();
                H5::DataSpace plsWriteQKLDataSpace = datasetQuicklook->getSpace();
                
                hsize_t dataIndexOffset[2];
                dataIndexOffset[0] = bufIdxRow;
                dataIndexOffset[1] = 0;
                hsize_t dataIndexDims[2];
                dataIndexDims[0] = 1;
                dataIndexDims[1] = spdFile->getNumberBinsX();
                H5::DataSpace newIndexDataspace = H5::DataSpace(2, dataIndexDims);
                
                plsWritePlsPerBinDataSpace.selectHyperslab( H5S_SELECT_SET, dataIndexDims, dataIndexOffset );
                plsWriteOffsetsDataSpace.selectHyperslab( H5S_SELECT_SET, dataIndexDims, dataIndexOffset );
                plsWriteQKLDataSpace.selectHyperslab( H5S_SELECT_SET, dataIndexDims, dataIndexOffset );
                
                datasetPlsPerBin->write( plsInColBuf, H5::PredType::NATIVE_ULONG, newIndexDataspace, plsWritePlsPerBinDataSpace );
                datasetBinsOffset->write( plsOffsetBuf, H5::PredType::NATIVE_ULLONG, newIndexDataspace, plsWriteOffsetsDataSpace );
                datasetQuicklook->write( qkBuffer, H5::PredType::NATIVE_FLOAT, newIndexDataspace, plsWriteQKLDataSpace );
                
                
                bufIdxCol = 0;
                ++bufIdxRow;
            }
		}
		catch( H5::FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataTypeIException &e )
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
	
	void SPDSeqFileWriter::finaliseClose() 
	{
		if(!fileOpened)
		{
			throw SPDIOException("SPD (HDF5) file not open, cannot finalise.");
		}
        
        SPDPulseUtils pulseUtils;
		SPDPointUtils pointUtils;
		
		try 
		{
			H5::Exception::dontPrint() ;
			
			if(plsBuffer->size() > 0 )
			{
				unsigned long numPulsesInCol = plsBuffer->size();
				unsigned long numPointsInCol = 0;
				unsigned long numTransValsInCol = 0;
				unsigned long numReceiveValsInCol = 0;
				
				std::vector<SPDPulse*>::iterator iterPulses;
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
                        scanlineMinWritten = (*iterPulses)->scanline;
                        scanlineMaxWritten = (*iterPulses)->scanline;
                        scanlineIdxMinWritten = (*iterPulses)->scanlineIdx;
                        scanlineIdxMaxWritten = (*iterPulses)->scanlineIdx;                          
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
                        
                        if((*iterPulses)->scanline < scanlineMinWritten)
                        {
                            scanlineMinWritten = (*iterPulses)->scanline;
                        }
                        else if((*iterPulses)->scanline > scanlineMaxWritten) 
                        {
                            scanlineMaxWritten = (*iterPulses)->scanline;
                        }
                            
                        if((*iterPulses)->scanlineIdx < scanlineIdxMinWritten)
                        {
                            scanlineIdxMinWritten = (*iterPulses)->scanlineIdx;
                        }
                        else if((*iterPulses)->scanlineIdx > scanlineIdxMaxWritten)  
                        {
                            scanlineIdxMaxWritten = (*iterPulses)->scanlineIdx;                            
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
				
				H5::DataSpace pulseWriteDataSpace = pulsesDataset->getSpace();
				pulseWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pulseDataDims, pulseDataOffset);
				H5::DataSpace newPulsesDataspace = H5::DataSpace(1, pulseDataDims);
				
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
					
					H5::DataSpace pointWriteDataSpace = pointsDataset->getSpace();
					pointWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pointsDataDims, pointsDataOffset);
					H5::DataSpace newPointsDataspace = H5::DataSpace(1, pointsDataDims);
					
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
					
					H5::DataSpace transWriteDataSpace = transmittedDataset->getSpace();
					transWriteDataSpace.selectHyperslab(H5S_SELECT_SET, transDataDims, transDataOffset);
					H5::DataSpace newTransDataspace = H5::DataSpace(1, transDataDims);
					
					transmittedDataset->write(transmittedValues, H5::PredType::NATIVE_ULONG, newTransDataspace, transWriteDataSpace);
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
					
					H5::DataSpace receivedWriteDataSpace = receivedDataset->getSpace();
					receivedWriteDataSpace.selectHyperslab(H5S_SELECT_SET, receivedDataDims, receivedDataOffset);
					H5::DataSpace newReceivedDataspace = H5::DataSpace(1, receivedDataDims);
					
					receivedDataset->write(receivedValues, H5::PredType::NATIVE_ULONG, newReceivedDataspace, receivedWriteDataSpace);
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
		catch( H5::FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataTypeIException &e )
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
                    if(!this->keepMinExtent)
                    {
                        spdFile->setZMin(zMinWritten);
                        spdFile->setZMax(zMaxWritten);
                        spdFile->setBoundingVolume(xMinWritten, xMaxWritten, yMinWritten, yMaxWritten, zMinWritten, zMaxWritten);
                    }
                    else
                    {
                        if(xMinWritten < spdFile->getXMin())
                        {
                            spdFile->setXMin(xMinWritten);
                        }
                        if(xMaxWritten > spdFile->getXMax())
                        {
                            spdFile->setXMax(xMaxWritten);
                        }
                        if(yMinWritten < spdFile->getYMin())
                        {
                            spdFile->setYMin(yMinWritten);
                        }
                        if(yMaxWritten > spdFile->getYMax())
                        {
                            spdFile->setYMax(yMaxWritten);
                        }
                        if(zMinWritten < spdFile->getZMin())
                        {
                            spdFile->setZMin(zMinWritten);
                        }
                        if(zMaxWritten > spdFile->getZMax())
                        {
                            spdFile->setZMax(zMaxWritten);
                        }                        
                    }
                }
                else if(spdFile->getIndexType() == SPD_SPHERICAL_IDX)
                {
                    spdFile->setBoundingVolumeSpherical(zenMinWritten, zenMaxWritten, azMinWritten, azMaxWritten, ranMinWritten, ranMaxWritten);
                    spdFile->setRangeMin(ranMinWritten);
                    spdFile->setRangeMax(ranMaxWritten);
                }
                else if(spdFile->getIndexType() == SPD_SCAN_IDX)
                {
                    spdFile->setBoundingBoxScanline(scanlineMinWritten, scanlineMaxWritten, scanlineIdxMinWritten, scanlineIdxMaxWritten);
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
			H5::StrType strdatatypeLen6(H5::PredType::C_S1, 6);
			H5::StrType strdatatypeLen4(H5::PredType::C_S1, 4);
			const H5std_string strClassVal ("IMAGE");
			const H5std_string strImgVerVal ("1.2");
			
			H5::DataSpace attr_dataspace = H5::DataSpace(H5S_SCALAR);
			
			H5::Attribute classAttribute = datasetQuicklook->createAttribute(ATTRIBUTENAME_CLASS, strdatatypeLen6, attr_dataspace);
			classAttribute.write(strdatatypeLen6, strClassVal); 
			
			H5::Attribute imgVerAttribute = datasetQuicklook->createAttribute(ATTRIBUTENAME_IMAGE_VERSION, strdatatypeLen4, attr_dataspace);
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
		catch( H5::FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataTypeIException &e )
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
                std::cerr << "WARNING: " << e.what() << std::endl;
			}
		}
	}
    
    
    
    
    
    
    
    
    
    
    SPDNonSeqFileWriter::SPDNonSeqFileWriter() : SPDDataExporter("SPD-NSQ"), spdOutH5File(NULL), pulsesDataset(NULL), spdPulseDataType(NULL), pointsDataset(NULL), spdPointDataType(NULL), datasetPlsPerBin(NULL), datasetBinsOffset(NULL), receivedDataset(NULL), transmittedDataset(NULL), datasetQuicklook(NULL), numPulses(0), numPts(0), numTransVals(0), numReceiveVals(0), firstColumn(true), numCols(0), numRows(0)
	{
		this->keepMinExtent = false;
	}
	
	SPDNonSeqFileWriter::SPDNonSeqFileWriter(const SPDDataExporter &dataExporter)  : SPDDataExporter(dataExporter), spdOutH5File(NULL), pulsesDataset(NULL), spdPulseDataType(NULL), pointsDataset(NULL), spdPointDataType(NULL), datasetPlsPerBin(NULL), datasetBinsOffset(NULL), receivedDataset(NULL), transmittedDataset(NULL), datasetQuicklook(NULL), numPulses(0), numPts(0), numTransVals(0), numReceiveVals(0), firstColumn(true), numCols(0), numRows(0)
	{
		if(fileOpened)
		{
			throw SPDException("Cannot make a copy of a file exporter when a file is open.");
		}
	}
	
	SPDNonSeqFileWriter& SPDNonSeqFileWriter::operator=(const SPDNonSeqFileWriter& dataExporter) 
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
	
	bool SPDNonSeqFileWriter::open(SPDFile *spdFile, std::string outputFile) 
	{
		SPDPulseUtils pulseUtils;
		SPDPointUtils pointUtils;
		this->spdFile = spdFile;
		this->outputFile = outputFile;
		
		const H5std_string spdFilePath( outputFile );
		
		try 
		{
			H5::Exception::dontPrint() ;
			
			// Create File..
            try
			{
				spdOutH5File = new H5::H5File( spdFilePath, H5F_ACC_TRUNC  );
			}
			catch (H5::FileIException &e)
			{
				std::string message  = std::string("Could not create SPD file: ") + spdFilePath;
				throw SPDIOException(message);
			}
			
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
			H5::DataSpace pulseDataSpace = H5::DataSpace(1, initDimsPulseDS, maxDimsPulseDS);
			
			hsize_t dimsPulseChunk[1];
			dimsPulseChunk[0] = spdFile->getPulseBlockSize();
			
			H5::DSetCreatPropList creationPulseDSPList;
			creationPulseDSPList.setChunk(1, dimsPulseChunk);			
			creationPulseDSPList.setShuffle();
            creationPulseDSPList.setDeflate(SPD_DEFLATE);
            
            H5::CompType *spdPulseDataTypeDisk = NULL;
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
			pulsesDataset = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_PULSES, *spdPulseDataTypeDisk, pulseDataSpace, creationPulseDSPList));
			
			
			// Create DataType, DataSpace and Dataset for Points
			hsize_t initDimsPtsDS[1];
			initDimsPtsDS[0] = 0;
			hsize_t maxDimsPtsDS[1];
			maxDimsPtsDS[0] = H5S_UNLIMITED;
			H5::DataSpace ptsDataSpace = H5::DataSpace(1, initDimsPtsDS, maxDimsPtsDS);
			
			hsize_t dimsPtsChunk[1];
			dimsPtsChunk[0] = spdFile->getPointBlockSize();
			
			H5::DSetCreatPropList creationPtsDSPList;
			creationPtsDSPList.setChunk(1, dimsPtsChunk);			
            creationPtsDSPList.setShuffle();
			creationPtsDSPList.setDeflate(SPD_DEFLATE);
            
            H5::CompType *spdPointDataTypeDisk = NULL;
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
			pointsDataset = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_POINTS, *spdPointDataTypeDisk, ptsDataSpace, creationPtsDSPList));
			
			// Create incoming and outgoing DataSpace and Dataset
			hsize_t initDimsWaveformDS[1];
			initDimsWaveformDS[0] = 0;
			hsize_t maxDimsWaveformDS[1];
			maxDimsWaveformDS[0] = H5S_UNLIMITED;
			H5::DataSpace waveformDataSpace = H5::DataSpace(1, initDimsWaveformDS, maxDimsWaveformDS);
			
			hsize_t dimsReceivedChunk[1];
			dimsReceivedChunk[0] = spdFile->getReceivedBlockSize();
            
			hsize_t dimsTransmittedChunk[1];
			dimsTransmittedChunk[0] = spdFile->getTransmittedBlockSize();
			
            if(spdFile->getWaveformBitRes() == SPD_32_BIT_WAVE)
            {
                H5::IntType intU32DataType( H5::PredType::STD_U32LE );
                intU32DataType.setOrder( H5T_ORDER_LE );
                
                boost::uint_fast32_t fillValueUInt = 0;
                H5::DSetCreatPropList creationReceivedDSPList;
                creationReceivedDSPList.setChunk(1, dimsReceivedChunk);	
                creationReceivedDSPList.setShuffle();
                creationReceivedDSPList.setDeflate(SPD_DEFLATE);
                creationReceivedDSPList.setFillValue( H5::PredType::STD_U32LE, &fillValueUInt);
                
                H5::DSetCreatPropList creationTransmittedDSPList;
                creationTransmittedDSPList.setChunk(1, dimsTransmittedChunk);			
                creationTransmittedDSPList.setShuffle();
                creationTransmittedDSPList.setDeflate(SPD_DEFLATE);
                creationTransmittedDSPList.setFillValue( H5::PredType::STD_U32LE, &fillValueUInt);
                
                receivedDataset = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_RECEIVED, intU32DataType, waveformDataSpace, creationReceivedDSPList));
                transmittedDataset = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_TRANSMITTED, intU32DataType, waveformDataSpace, creationTransmittedDSPList));
            }
            else if(spdFile->getWaveformBitRes() == SPD_16_BIT_WAVE)
            {
                H5::IntType intU16DataType( H5::PredType::STD_U16LE );
                intU16DataType.setOrder( H5T_ORDER_LE );
                
                boost::uint_fast32_t fillValueUInt = 0;
                H5::DSetCreatPropList creationReceivedDSPList;
                creationReceivedDSPList.setChunk(1, dimsReceivedChunk);			
                creationReceivedDSPList.setShuffle();
                creationReceivedDSPList.setDeflate(SPD_DEFLATE);
                creationReceivedDSPList.setFillValue( H5::PredType::STD_U16LE, &fillValueUInt);
                
                H5::DSetCreatPropList creationTransmittedDSPList;
                creationTransmittedDSPList.setChunk(1, dimsTransmittedChunk);			
                creationTransmittedDSPList.setShuffle();
                creationTransmittedDSPList.setDeflate(SPD_DEFLATE);
                creationTransmittedDSPList.setFillValue( H5::PredType::STD_U16LE, &fillValueUInt);
                
                receivedDataset = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_RECEIVED, intU16DataType, waveformDataSpace, creationReceivedDSPList));
                transmittedDataset = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_TRANSMITTED, intU16DataType, waveformDataSpace, creationTransmittedDSPList));
            }
            else if(spdFile->getWaveformBitRes() == SPD_8_BIT_WAVE)
            {
                H5::IntType intU8DataType( H5::PredType::STD_U8LE );
                intU8DataType.setOrder( H5T_ORDER_LE );
                
                boost::uint_fast32_t fillValueUInt = 0;
                H5::DSetCreatPropList creationReceivedDSPList;
                creationReceivedDSPList.setChunk(1, dimsReceivedChunk);			
                creationReceivedDSPList.setShuffle();
                creationReceivedDSPList.setDeflate(SPD_DEFLATE);
                creationReceivedDSPList.setFillValue( H5::PredType::STD_U8LE, &fillValueUInt);
                
                H5::DSetCreatPropList creationTransmittedDSPList;
                creationTransmittedDSPList.setChunk(1, dimsTransmittedChunk);			
                creationTransmittedDSPList.setShuffle();
                creationTransmittedDSPList.setDeflate(SPD_DEFLATE);
                creationTransmittedDSPList.setFillValue( H5::PredType::STD_U8LE, &fillValueUInt);
                
                receivedDataset = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_RECEIVED, intU8DataType, waveformDataSpace, creationReceivedDSPList));
                transmittedDataset = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_TRANSMITTED, intU8DataType, waveformDataSpace, creationTransmittedDSPList));
            }
            else
            {
                throw SPDIOException("Waveform bit resolution is unknown.");
            }
			
			// Create Reference datasets and dataspaces		
			H5::IntType intU64DataType( H5::PredType::STD_U64LE );
            intU64DataType.setOrder( H5T_ORDER_LE );
            H5::IntType intU32DataType( H5::PredType::STD_U32LE );
            intU32DataType.setOrder( H5T_ORDER_LE );
			
			hsize_t initDimsIndexDS[2];
			initDimsIndexDS[0] = numRows;
			initDimsIndexDS[1] = numCols;
			H5::DataSpace indexDataSpace(2, initDimsIndexDS);
			
			hsize_t dimsIndexChunk[2];
			dimsIndexChunk[0] = 1;
			dimsIndexChunk[1] = numCols;
			
			boost::uint_fast32_t fillValue32bit = 0;
			H5::DSetCreatPropList initParamsIndexPulsesPerBin;
			initParamsIndexPulsesPerBin.setChunk(2, dimsIndexChunk);			
			initParamsIndexPulsesPerBin.setShuffle();
            initParamsIndexPulsesPerBin.setDeflate(SPD_DEFLATE);
			initParamsIndexPulsesPerBin.setFillValue( H5::PredType::STD_U32LE, &fillValue32bit);
			
			boost::uint_fast64_t fillValue64bit = 0;
			H5::DSetCreatPropList initParamsIndexOffset;
			initParamsIndexOffset.setChunk(2, dimsIndexChunk);			
			initParamsIndexOffset.setShuffle();
            initParamsIndexOffset.setDeflate(SPD_DEFLATE);
			initParamsIndexOffset.setFillValue( H5::PredType::STD_U64LE, &fillValue64bit);
			
			datasetPlsPerBin = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_PLS_PER_BIN, intU32DataType, indexDataSpace, initParamsIndexPulsesPerBin ));
			datasetBinsOffset = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_BIN_OFFSETS, intU64DataType, indexDataSpace, initParamsIndexOffset ));
			
			// Created Quicklook datasets and dataspaces
			H5::FloatType floatDataType( H5::PredType::IEEE_F32LE );
			
			hsize_t initDimsQuicklookDS[2];
			initDimsQuicklookDS[0] = numRows;
			initDimsQuicklookDS[1] = numCols;
			H5::DataSpace quicklookDataSpace(2, initDimsQuicklookDS);
			
			hsize_t dimsQuicklookChunk[2];
			dimsQuicklookChunk[0] = 1;
			dimsQuicklookChunk[1] = numCols;
			
			float fillValueFloatQKL = 0;
			H5::DSetCreatPropList initParamsQuicklook;
			initParamsQuicklook.setChunk(2, dimsQuicklookChunk);			
            initParamsQuicklook.setShuffle();
			initParamsQuicklook.setDeflate(SPD_DEFLATE);
			initParamsQuicklook.setFillValue( H5::PredType::IEEE_F32LE, &fillValueFloatQKL);
			
			datasetQuicklook = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_QKLIMAGE, floatDataType, quicklookDataSpace, initParamsQuicklook ));
            
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
            scanlineMinWritten = 0;
            scanlineMaxWritten = 0;
            scanlineIdxMinWritten = 0;
            scanlineIdxMaxWritten = 0;  
            
            
			firstReturn = true;
            firstPulse = true;
            
            plsBuffer = new std::vector<SPDPulse*>();
            plsBuffer->reserve(spdFile->getPulseBlockSize());
            plsOffset = 0;
            
		}
		catch( H5::FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataTypeIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		
		return fileOpened;
	}
    
    bool SPDNonSeqFileWriter::reopen(SPDFile *spdFile, std::string outputFile) 
    {
        throw SPDIOException("No reopen option available.");
    }
	
	void SPDNonSeqFileWriter::writeDataColumn(std::list<SPDPulse*> *pls, boost::uint_fast32_t col, boost::uint_fast32_t row)
	{
		SPDPulseUtils pulseUtils;
		SPDPointUtils pointUtils;
        
		if(!fileOpened)
		{
			throw SPDIOException("SPD (HDF5) file not open, cannot finalise.");
		}
        
        if(col >= numCols)
		{
			std::cout << "Number of Columns = " << numCols << std::endl;
			std::cout << col << std::endl;
			throw SPDIOException("The column you have specified it not within the current file.");
		}
		
		if(row >= numRows)
		{
			std::cout << "Number of Columns = " << numRows << std::endl;
			std::cout << row << std::endl;
			throw SPDIOException("The row you have specified it not within the current file.");
		}
        
		try 
		{
			H5::Exception::dontPrint() ;
			
            std::list<SPDPulse*>::iterator iterInPls;
            float qkVal = 0;
            //boost::uint_fast16_t numVals = 0;
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
                            for(std::vector<SPDPoint*>::iterator iterPts = (*iterInPls)->pts->begin(); iterPts != (*iterInPls)->pts->end(); ++iterPts)
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
                            //qkVal += (*iterInPls)->received[i];
                            //++numVals;
                            if(qkVal < (*iterInPls)->received[i])
                            {
                                qkVal = (*iterInPls)->received[i];
                            } 
                        }
                    }
                    //qkVal = qkVal/numVals;
                }
                else
                {
                    qkVal = pls->size();
                }
            }
            
            // Write the Quicklook value and index information.
            H5::DataSpace plsWritePlsPerBinDataSpace = datasetPlsPerBin->getSpace();
            H5::DataSpace plsWriteOffsetsDataSpace = datasetBinsOffset->getSpace();
            H5::DataSpace plsWriteQKLDataSpace = datasetQuicklook->getSpace();
            
            hsize_t dataIndexOffset[2];
            dataIndexOffset[0] = row;
            dataIndexOffset[1] = col;
            hsize_t dataIndexDims[2];
            dataIndexDims[0] = 1;
            dataIndexDims[1] = 1;
            H5::DataSpace newIndexDataspace = H5::DataSpace(2, dataIndexDims);
            
            plsWritePlsPerBinDataSpace.selectHyperslab( H5S_SELECT_SET, dataIndexDims, dataIndexOffset );
            plsWriteOffsetsDataSpace.selectHyperslab( H5S_SELECT_SET, dataIndexDims, dataIndexOffset );
            plsWriteQKLDataSpace.selectHyperslab( H5S_SELECT_SET, dataIndexDims, dataIndexOffset );
            
            unsigned long plsInBin = pls->size();
            
            datasetPlsPerBin->write( &plsInBin, H5::PredType::NATIVE_ULONG, newIndexDataspace, plsWritePlsPerBinDataSpace );
            datasetBinsOffset->write( &plsOffset, H5::PredType::NATIVE_ULLONG, newIndexDataspace, plsWriteOffsetsDataSpace );
            datasetQuicklook->write( &qkVal, H5::PredType::NATIVE_FLOAT, newIndexDataspace, plsWriteQKLDataSpace );
            
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
					
					std::vector<SPDPulse*>::iterator iterPulses;
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
                            
                            scanlineMinWritten = (*iterPulses)->scanline;
                            scanlineMaxWritten = (*iterPulses)->scanline;
                            scanlineIdxMinWritten = (*iterPulses)->scanlineIdx;
                            scanlineIdxMaxWritten = (*iterPulses)->scanlineIdx;  
                            
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
					
					H5::DataSpace pulseWriteDataSpace = pulsesDataset->getSpace();
					pulseWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pulseDataDims, pulseDataOffset);
					H5::DataSpace newPulsesDataspace = H5::DataSpace(1, pulseDataDims);
					
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
						
						H5::DataSpace pointWriteDataSpace = pointsDataset->getSpace();
						pointWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pointsDataDims, pointsDataOffset);
						H5::DataSpace newPointsDataspace = H5::DataSpace(1, pointsDataDims);
						
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
						
						H5::DataSpace transWriteDataSpace = transmittedDataset->getSpace();
						transWriteDataSpace.selectHyperslab(H5S_SELECT_SET, transDataDims, transDataOffset);
						H5::DataSpace newTransDataspace = H5::DataSpace(1, transDataDims);
						
						transmittedDataset->write(transmittedValues, H5::PredType::NATIVE_ULONG, newTransDataspace, transWriteDataSpace);
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
						
						H5::DataSpace receivedWriteDataSpace = receivedDataset->getSpace();
						receivedWriteDataSpace.selectHyperslab(H5S_SELECT_SET, receivedDataDims, receivedDataOffset);
						H5::DataSpace newReceivedDataspace = H5::DataSpace(1, receivedDataDims);
						
						receivedDataset->write(receivedValues, H5::PredType::NATIVE_ULONG, newReceivedDataspace, receivedWriteDataSpace);
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
		catch( H5::FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataTypeIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch(SPDIOException &e)
		{
			throw e;
		}
	}
	
	void SPDNonSeqFileWriter::writeDataColumn(std::vector<SPDPulse*> *pls, boost::uint_fast32_t col, boost::uint_fast32_t row)
	{
        SPDPulseUtils pulseUtils;
		SPDPointUtils pointUtils;
        
		if(!fileOpened)
		{
			throw SPDIOException("SPD (HDF5) file not open, cannot finalise.");
		}
        
        if(col >= numCols)
		{
			std::cout << "Number of Columns = " << numCols << std::endl;
			std::cout << col << std::endl;
			throw SPDIOException("The column you have specified it not within the current file.");
		}
		
		if(row >= numRows)
		{
			std::cout << "Number of Columns = " << numRows << std::endl;
			std::cout << row << std::endl;
			throw SPDIOException("The row you have specified it not within the current file.");
		}
        
		try 
		{
			H5::Exception::dontPrint() ;
			
            std::vector<SPDPulse*>::iterator iterInPls;
            float qkVal = 0;
            //boost::uint_fast16_t numVals = 0;
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
                            for(std::vector<SPDPoint*>::iterator iterPts = (*iterInPls)->pts->begin(); iterPts != (*iterInPls)->pts->end(); ++iterPts)
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
                            //qkVal += (*iterInPls)->received[i];
                            //++numVals;
                            if(qkVal < (*iterInPls)->received[i])
                            {
                                qkVal = (*iterInPls)->received[i];
                            } 
                        }
                    }
                    //qkVal = qkVal/numVals;
                }
                else
                {
                    qkVal = pls->size();
                }
            }
            
            // Write the Quicklook value and index information.
            H5::DataSpace plsWritePlsPerBinDataSpace = datasetPlsPerBin->getSpace();
            H5::DataSpace plsWriteOffsetsDataSpace = datasetBinsOffset->getSpace();
            H5::DataSpace plsWriteQKLDataSpace = datasetQuicklook->getSpace();
            
            hsize_t dataIndexOffset[2];
            dataIndexOffset[0] = row;
            dataIndexOffset[1] = col;
            hsize_t dataIndexDims[2];
            dataIndexDims[0] = 1;
            dataIndexDims[1] = 1;
            H5::DataSpace newIndexDataspace = H5::DataSpace(2, dataIndexDims);
            
            plsWritePlsPerBinDataSpace.selectHyperslab( H5S_SELECT_SET, dataIndexDims, dataIndexOffset );
            plsWriteOffsetsDataSpace.selectHyperslab( H5S_SELECT_SET, dataIndexDims, dataIndexOffset );
            plsWriteQKLDataSpace.selectHyperslab( H5S_SELECT_SET, dataIndexDims, dataIndexOffset );
            
            unsigned long plsInBin = pls->size();
            
            datasetPlsPerBin->write( &plsInBin, H5::PredType::NATIVE_ULONG, newIndexDataspace, plsWritePlsPerBinDataSpace );
            datasetBinsOffset->write( &plsOffset, H5::PredType::NATIVE_ULLONG, newIndexDataspace, plsWriteOffsetsDataSpace );
            datasetQuicklook->write( &qkVal, H5::PredType::NATIVE_FLOAT, newIndexDataspace, plsWriteQKLDataSpace );
            
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
					
					std::vector<SPDPulse*>::iterator iterPulses;
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
                            
                            scanlineMinWritten = (*iterPulses)->scanline;
                            scanlineMaxWritten = (*iterPulses)->scanline;
                            scanlineIdxMinWritten = (*iterPulses)->scanlineIdx;
                            scanlineIdxMaxWritten = (*iterPulses)->scanlineIdx;                          
                            
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
                            
                            if((*iterPulses)->scanline < scanlineMinWritten)
                            {
                                scanlineMinWritten = (*iterPulses)->scanline;
                            }
                            else if((*iterPulses)->scanline > scanlineMaxWritten) 
                            {
                                scanlineMaxWritten = (*iterPulses)->scanline;
                            }
                            
                            if((*iterPulses)->scanlineIdx < scanlineIdxMinWritten)
                            {
                                scanlineIdxMinWritten = (*iterPulses)->scanlineIdx;
                            }
                            else if((*iterPulses)->scanlineIdx > scanlineIdxMaxWritten)  
                            {
                                scanlineIdxMaxWritten = (*iterPulses)->scanlineIdx;                            
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
					
					H5::DataSpace pulseWriteDataSpace = pulsesDataset->getSpace();
					pulseWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pulseDataDims, pulseDataOffset);
					H5::DataSpace newPulsesDataspace = H5::DataSpace(1, pulseDataDims);
					
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
						
						H5::DataSpace pointWriteDataSpace = pointsDataset->getSpace();
						pointWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pointsDataDims, pointsDataOffset);
						H5::DataSpace newPointsDataspace = H5::DataSpace(1, pointsDataDims);
						
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
						
						H5::DataSpace transWriteDataSpace = transmittedDataset->getSpace();
						transWriteDataSpace.selectHyperslab(H5S_SELECT_SET, transDataDims, transDataOffset);
						H5::DataSpace newTransDataspace = H5::DataSpace(1, transDataDims);
						
						transmittedDataset->write(transmittedValues, H5::PredType::NATIVE_ULONG, newTransDataspace, transWriteDataSpace);
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
						
						H5::DataSpace receivedWriteDataSpace = receivedDataset->getSpace();
						receivedWriteDataSpace.selectHyperslab(H5S_SELECT_SET, receivedDataDims, receivedDataOffset);
						H5::DataSpace newReceivedDataspace = H5::DataSpace(1, receivedDataDims);
						
						receivedDataset->write(receivedValues, H5::PredType::NATIVE_ULONG, newReceivedDataspace, receivedWriteDataSpace);
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
		catch( H5::FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataTypeIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch(SPDIOException &e)
		{
			throw e;
		}
	}
	
	void SPDNonSeqFileWriter::finaliseClose() 
	{
		if(!fileOpened)
		{
			throw SPDIOException("SPD (HDF5) file not open, cannot finalise.");
		}
        
        SPDPulseUtils pulseUtils;
		SPDPointUtils pointUtils;
		
		try 
		{
			H5::Exception::dontPrint() ;
			
			if(plsBuffer->size() > 0 )
			{
				unsigned long numPulsesInCol = plsBuffer->size();
				unsigned long numPointsInCol = 0;
				unsigned long numTransValsInCol = 0;
				unsigned long numReceiveValsInCol = 0;
				
				std::vector<SPDPulse*>::iterator iterPulses;
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
                        
                        scanlineMinWritten = (*iterPulses)->scanline;
                        scanlineMaxWritten = (*iterPulses)->scanline;
                        scanlineIdxMinWritten = (*iterPulses)->scanlineIdx;
                        scanlineIdxMaxWritten = (*iterPulses)->scanlineIdx;  
                        
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
				
				H5::DataSpace pulseWriteDataSpace = pulsesDataset->getSpace();
				pulseWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pulseDataDims, pulseDataOffset);
				H5::DataSpace newPulsesDataspace = H5::DataSpace(1, pulseDataDims);
				
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
					
					H5::DataSpace pointWriteDataSpace = pointsDataset->getSpace();
					pointWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pointsDataDims, pointsDataOffset);
					H5::DataSpace newPointsDataspace = H5::DataSpace(1, pointsDataDims);
					
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
					
					H5::DataSpace transWriteDataSpace = transmittedDataset->getSpace();
					transWriteDataSpace.selectHyperslab(H5S_SELECT_SET, transDataDims, transDataOffset);
					H5::DataSpace newTransDataspace = H5::DataSpace(1, transDataDims);
					
					transmittedDataset->write(transmittedValues, H5::PredType::NATIVE_ULONG, newTransDataspace, transWriteDataSpace);
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
					
					H5::DataSpace receivedWriteDataSpace = receivedDataset->getSpace();
					receivedWriteDataSpace.selectHyperslab(H5S_SELECT_SET, receivedDataDims, receivedDataOffset);
					H5::DataSpace newReceivedDataspace = H5::DataSpace(1, receivedDataDims);
					
					receivedDataset->write(receivedValues, H5::PredType::NATIVE_ULONG, newReceivedDataspace, receivedWriteDataSpace);
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
		catch( H5::FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataTypeIException &e )
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
                    if(!this->keepMinExtent)
                    {
                        spdFile->setZMin(zMinWritten);
                        spdFile->setZMax(zMaxWritten);
                        spdFile->setBoundingVolume(xMinWritten, xMaxWritten, yMinWritten, yMaxWritten, zMinWritten, zMaxWritten);
                    }
                    else
                    {
                        if(xMinWritten < spdFile->getXMin())
                        {
                            spdFile->setXMin(xMinWritten);
                        }
                        if(xMaxWritten > spdFile->getXMax())
                        {
                            spdFile->setXMax(xMaxWritten);
                        }
                        if(yMinWritten < spdFile->getYMin())
                        {
                            spdFile->setYMin(yMinWritten);
                        }
                        if(yMaxWritten > spdFile->getYMax())
                        {
                            spdFile->setYMax(yMaxWritten);
                        }
                        if(zMinWritten < spdFile->getZMin())
                        {
                            spdFile->setZMin(zMinWritten);
                        }
                        if(zMaxWritten > spdFile->getZMax())
                        {
                            spdFile->setZMax(zMaxWritten);
                        }
                        
                    }
                }
                else if(spdFile->getIndexType() == SPD_SPHERICAL_IDX)
                {
                    spdFile->setBoundingVolumeSpherical(zenMinWritten, zenMaxWritten, azMinWritten, azMaxWritten, ranMinWritten, ranMaxWritten);
                    spdFile->setRangeMin(ranMinWritten);
                    spdFile->setRangeMax(ranMaxWritten);
                }
                else if(spdFile->getIndexType() == SPD_SCAN_IDX)
                {
                    spdFile->setBoundingBoxScanline(scanlineMinWritten, scanlineMaxWritten, scanlineIdxMinWritten, scanlineIdxMaxWritten);
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
                        
            this->writeHeaderInfo(spdOutH5File, spdFile);
            
			// Write attributes to Quicklook
			H5::StrType strdatatypeLen6(H5::PredType::C_S1, 6);
			H5::StrType strdatatypeLen4(H5::PredType::C_S1, 4);
			const H5std_string strClassVal ("IMAGE");
			const H5std_string strImgVerVal ("1.2");
			
			H5::DataSpace attr_dataspace = H5::DataSpace(H5S_SCALAR);
			
			H5::Attribute classAttribute = datasetQuicklook->createAttribute(ATTRIBUTENAME_CLASS, strdatatypeLen6, attr_dataspace);
			classAttribute.write(strdatatypeLen6, strClassVal); 
			
			H5::Attribute imgVerAttribute = datasetQuicklook->createAttribute(ATTRIBUTENAME_IMAGE_VERSION, strdatatypeLen4, attr_dataspace);
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
		catch( H5::FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataTypeIException &e )
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
                std::cerr << "WARNING: " << e.what() << std::endl;
			}
		}
	}
    
    
    
    
    
    
    
    SPDNoIdxFileWriter::SPDNoIdxFileWriter() : SPDDataExporter("UPD"), spdOutH5File(NULL), pulsesDataset(NULL), spdPulseDataType(NULL), pointsDataset(NULL), spdPointDataType(NULL), receivedDataset(NULL), transmittedDataset(NULL), numPulses(0), numPts(0), numTransVals(0), numReceiveVals(0)
	{
		reOpenedFile = false;
        this->keepMinExtent = false;
	}
    
	SPDNoIdxFileWriter::SPDNoIdxFileWriter(const SPDDataExporter &dataExporter)  : SPDDataExporter(dataExporter), spdOutH5File(NULL), pulsesDataset(NULL), spdPulseDataType(NULL), pointsDataset(NULL), spdPointDataType(NULL), receivedDataset(NULL), transmittedDataset(NULL), numPulses(0), numPts(0), numTransVals(0), numReceiveVals(0)
	{
		if(fileOpened)
		{
			throw SPDException("Cannot make a copy of a file exporter when a file is open.");
		}
	}
    
	SPDNoIdxFileWriter& SPDNoIdxFileWriter::operator=(const SPDNoIdxFileWriter& dataExporter) 
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
    
	bool SPDNoIdxFileWriter::open(SPDFile *spdFile, std::string outputFile) 
	{
		SPDPulseUtils pulseUtils;
		SPDPointUtils pointUtils;
		this->spdFile = spdFile;
		this->outputFile = outputFile;
		
		const H5std_string spdFilePath( outputFile );
		
		try 
		{
			H5::Exception::dontPrint() ;
			
			// Create File..
            try
			{
				spdOutH5File = new H5::H5File( spdFilePath, H5F_ACC_TRUNC  );
			}
			catch (H5::FileIException &e)
			{
				std::string message  = std::string("Could not create SPD file: ") + spdFilePath;
				throw SPDIOException(message);
			}
			
			// Create Groups..
			spdOutH5File->createGroup( GROUPNAME_HEADER );
			spdOutH5File->createGroup( GROUPNAME_DATA );
			
			// Create DataType, DataSpace and Dataset for Pulses
            hsize_t initDimsPulseDS[1];
			initDimsPulseDS[0] = 0;
			hsize_t maxDimsPulseDS[1];
			maxDimsPulseDS[0] = H5S_UNLIMITED;
			H5::DataSpace pulseDataSpace = H5::DataSpace(1, initDimsPulseDS, maxDimsPulseDS);
			
			hsize_t dimsPulseChunk[1];
			dimsPulseChunk[0] = spdFile->getPulseBlockSize();
			
			H5::DSetCreatPropList creationPulseDSPList;
			creationPulseDSPList.setChunk(1, dimsPulseChunk);			
			creationPulseDSPList.setShuffle();
            creationPulseDSPList.setDeflate(SPD_DEFLATE);
            
            H5::CompType *spdPulseDataTypeDisk = NULL;
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
			pulsesDataset = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_PULSES, *spdPulseDataTypeDisk, pulseDataSpace, creationPulseDSPList));

			// Create DataType, DataSpace and Dataset for Points
            hsize_t initDimsPtsDS[1];
			initDimsPtsDS[0] = 0;
			hsize_t maxDimsPtsDS[1];
			maxDimsPtsDS[0] = H5S_UNLIMITED;
			H5::DataSpace ptsDataSpace = H5::DataSpace(1, initDimsPtsDS, maxDimsPtsDS);
			
			hsize_t dimsPtsChunk[1];
			dimsPtsChunk[0] = spdFile->getPointBlockSize();
			
			H5::DSetCreatPropList creationPtsDSPList;
			creationPtsDSPList.setChunk(1, dimsPtsChunk);			
			creationPtsDSPList.setShuffle();
            creationPtsDSPList.setDeflate(SPD_DEFLATE);
            
            H5::CompType *spdPointDataTypeDisk = NULL;
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
			pointsDataset = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_POINTS, *spdPointDataTypeDisk, ptsDataSpace, creationPtsDSPList));
			
			// Create transmitted and received DataSpace and Dataset
			hsize_t initDimsWaveformDS[1];
			initDimsWaveformDS[0] = 0;
			hsize_t maxDimsWaveformDS[1];
			maxDimsWaveformDS[0] = H5S_UNLIMITED;
			H5::DataSpace waveformDataSpace = H5::DataSpace(1, initDimsWaveformDS, maxDimsWaveformDS);
			
			hsize_t dimsReceivedChunk[1];
			dimsReceivedChunk[0] = spdFile->getReceivedBlockSize();
            
			hsize_t dimsTransmittedChunk[1];
			dimsTransmittedChunk[0] = spdFile->getTransmittedBlockSize();
			
            if(spdFile->getWaveformBitRes() == SPD_32_BIT_WAVE)
            {
                H5::IntType intU32DataType( H5::PredType::STD_U32LE );
                intU32DataType.setOrder( H5T_ORDER_LE );
                
                boost::uint_fast32_t fillValueUInt = 0;
                H5::DSetCreatPropList creationReceivedDSPList;
                creationReceivedDSPList.setChunk(1, dimsReceivedChunk);			
                creationReceivedDSPList.setShuffle();
                creationReceivedDSPList.setDeflate(SPD_DEFLATE);
                creationReceivedDSPList.setFillValue( H5::PredType::STD_U32LE, &fillValueUInt);
                
                H5::DSetCreatPropList creationTransmittedDSPList;
                creationTransmittedDSPList.setChunk(1, dimsTransmittedChunk);			
                creationTransmittedDSPList.setShuffle();
                creationTransmittedDSPList.setDeflate(SPD_DEFLATE);
                creationTransmittedDSPList.setFillValue( H5::PredType::STD_U32LE, &fillValueUInt);
                
                receivedDataset = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_RECEIVED, intU32DataType, waveformDataSpace, creationReceivedDSPList));
                transmittedDataset = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_TRANSMITTED, intU32DataType, waveformDataSpace, creationTransmittedDSPList));
            }
            else if(spdFile->getWaveformBitRes() == SPD_16_BIT_WAVE)
            {
                H5::IntType intU16DataType( H5::PredType::STD_U16LE );
                intU16DataType.setOrder( H5T_ORDER_LE );
                
                boost::uint_fast32_t fillValueUInt = 0;
                H5::DSetCreatPropList creationReceivedDSPList;
                creationReceivedDSPList.setChunk(1, dimsReceivedChunk);			
                creationReceivedDSPList.setShuffle();
                creationReceivedDSPList.setDeflate(SPD_DEFLATE);
                creationReceivedDSPList.setFillValue( H5::PredType::STD_U16LE, &fillValueUInt);
                
                H5::DSetCreatPropList creationTransmittedDSPList;
                creationTransmittedDSPList.setChunk(1, dimsTransmittedChunk);			
                creationTransmittedDSPList.setShuffle();
                creationTransmittedDSPList.setDeflate(SPD_DEFLATE);
                creationTransmittedDSPList.setFillValue( H5::PredType::STD_U16LE, &fillValueUInt);
                
                receivedDataset = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_RECEIVED, intU16DataType, waveformDataSpace, creationReceivedDSPList));
                transmittedDataset = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_TRANSMITTED, intU16DataType, waveformDataSpace, creationTransmittedDSPList));
            }
            else if(spdFile->getWaveformBitRes() == SPD_8_BIT_WAVE)
            {
                H5::IntType intU8DataType( H5::PredType::STD_U8LE );
                intU8DataType.setOrder( H5T_ORDER_LE );
                
                boost::uint_fast32_t fillValueUInt = 0;
                H5::DSetCreatPropList creationReceivedDSPList;
                creationReceivedDSPList.setChunk(1, dimsReceivedChunk);			
                creationReceivedDSPList.setShuffle();
                creationReceivedDSPList.setDeflate(SPD_DEFLATE);
                creationReceivedDSPList.setFillValue( H5::PredType::STD_U8LE, &fillValueUInt);
                
                H5::DSetCreatPropList creationTransmittedDSPList;
                creationTransmittedDSPList.setChunk(1, dimsTransmittedChunk);			
                creationTransmittedDSPList.setShuffle();
                creationTransmittedDSPList.setDeflate(SPD_DEFLATE);
                creationTransmittedDSPList.setFillValue( H5::PredType::STD_U8LE, &fillValueUInt);
                
                receivedDataset = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_RECEIVED, intU8DataType, waveformDataSpace, creationReceivedDSPList));
                transmittedDataset = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_TRANSMITTED, intU8DataType, waveformDataSpace, creationTransmittedDSPList));
            }
            else
            {
                throw SPDIOException("Waveform bit resolution is unknown.");
            }
            
			
			fileOpened = true;
			
			plsBuffer = new std::vector<SPDPulse*>();
            plsBuffer->reserve(spdFile->getPulseBlockSize());
		}
		catch( H5::FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataTypeIException &e )
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
        scanlineMinWritten = 0;
        scanlineMaxWritten = 0;
        scanlineIdxMinWritten = 0;
        scanlineIdxMaxWritten = 0;       
        
        firstReturn = true;
        firstPulse = true;
        firstWaveform = true;
        
        reOpenedFile = false;
		
		return fileOpened;
	}
    
    bool SPDNoIdxFileWriter::reopen(SPDFile *spdFile, std::string outputFile) 
    {
        SPDPulseUtils pulseUtils;
		SPDPointUtils pointUtils;
		this->spdFile = spdFile;
		this->outputFile = outputFile;
		
		const H5std_string spdFilePath( outputFile );
		
		try
		{
			H5::Exception::dontPrint();
			
			// Open File..
            try
			{
				spdOutH5File = new H5::H5File( spdFilePath, H5F_ACC_RDWR,  H5::FileCreatPropList::DEFAULT );
			}
			catch (H5::FileIException &e)
			{
				std::string message  = std::string("Could not open SPD file: ") + spdFilePath;
				throw SPDIOException(message);
			}
            
            this->readHeaderInfo(spdOutH5File, spdFile);
            
            //std::cout << "SPD File: " << spdFile << std::endl;

            // Open Dataset for Pulses
            try
            {
                pulsesDataset = new H5::DataSet(spdOutH5File->openDataSet(SPDFILE_DATASETNAME_PULSES));
                
                if(spdFile->getPulseVersion() == 1)
                {
                    spdPulseDataType = pulseUtils.createSPDPulseH5V1DataTypeMemory();
                }
                else if(spdFile->getPulseVersion() == 2)
                {
                    spdPulseDataType = pulseUtils.createSPDPulseH5V2DataTypeMemory();
                }
                else
                {
                    throw SPDIOException("Did not recognise the Pulse version.");
                }
            }
            catch ( H5::Exception &e)
            {
                hsize_t initDimsPulseDS[1];
                initDimsPulseDS[0] = 0;
                hsize_t maxDimsPulseDS[1];
                maxDimsPulseDS[0] = H5S_UNLIMITED;
                H5::DataSpace pulseDataSpace = H5::DataSpace(1, initDimsPulseDS, maxDimsPulseDS);
                
                hsize_t dimsPulseChunk[1];
                dimsPulseChunk[0] = spdFile->getPulseBlockSize();
                
                H5::DSetCreatPropList creationPulseDSPList;
                creationPulseDSPList.setChunk(1, dimsPulseChunk);
                creationPulseDSPList.setShuffle();
                creationPulseDSPList.setDeflate(SPD_DEFLATE);
                
                H5::CompType *spdPulseDataTypeDisk = NULL;
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
                pulsesDataset = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_PULSES, *spdPulseDataTypeDisk, pulseDataSpace, creationPulseDSPList));
            }
            catch ( SPDIOException &e)
            {
                throw e;
            }
            catch ( std::exception &e)
            {
                throw SPDIOException(e.what());
            }
            
            
			// Open Dataset for Points
            try
            {
                pointsDataset = new H5::DataSet(spdOutH5File->openDataSet(SPDFILE_DATASETNAME_POINTS));
                                
                if(spdFile->getPointVersion() == 1)
                {
                    spdPointDataType = pointUtils.createSPDPointV1DataTypeMemory();
                }
                else if(spdFile->getPointVersion() == 2)
                {
                    spdPointDataType = pointUtils.createSPDPointV2DataTypeMemory();
                }
                else
                {
                    throw SPDIOException("Did not recognise the Point version");
                }
            }
            catch ( H5::Exception &e)
            {
                //throw SPDProcessingException(e.getCDetailMsg());
                hsize_t initDimsPtsDS[1];
                initDimsPtsDS[0] = 0;
                hsize_t maxDimsPtsDS[1];
                maxDimsPtsDS[0] = H5S_UNLIMITED;
                H5::DataSpace ptsDataSpace = H5::DataSpace(1, initDimsPtsDS, maxDimsPtsDS);
                
                hsize_t dimsPtsChunk[1];
                dimsPtsChunk[0] = spdFile->getPointBlockSize();
                
                H5::DSetCreatPropList creationPtsDSPList;
                creationPtsDSPList.setChunk(1, dimsPtsChunk);
                creationPtsDSPList.setShuffle();
                creationPtsDSPList.setDeflate(SPD_DEFLATE);
                
                H5::CompType *spdPointDataTypeDisk = NULL;
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
                pointsDataset = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_POINTS, *spdPointDataTypeDisk, ptsDataSpace, creationPtsDSPList));
            }
            catch ( SPDIOException &e)
            {
                throw e;
            }
            catch ( std::exception &e)
            {
                throw SPDIOException(e.what());
            }
			
            
             // Open transmitted Datasets
             try
             {
                 transmittedDataset = new H5::DataSet(spdOutH5File->openDataSet(SPDFILE_DATASETNAME_TRANSMITTED));
                                  
                 H5::DataSpace transSpace = transmittedDataset->getSpace();
                 int transDims = transSpace.getSimpleExtentNdims();
                 if(transDims != 1)
                 {
                     throw SPDIOException("The transmitted waveform has more than 1 dimension.");
                 }
                 hsize_t transLen = 0;
                 transSpace.getSimpleExtentDims(&transLen);
                 
                 this->numTransVals = transLen;
             }
             catch ( H5::Exception &e)
             {
                 hsize_t initDimsWaveformDS[1];
                 initDimsWaveformDS[0] = 0;
                 hsize_t maxDimsWaveformDS[1];
                 maxDimsWaveformDS[0] = H5S_UNLIMITED;
                 H5::DataSpace waveformDataSpace = H5::DataSpace(1, initDimsWaveformDS, maxDimsWaveformDS);
                 
                 hsize_t dimsTransmittedChunk[1];
                 dimsTransmittedChunk[0] = spdFile->getTransmittedBlockSize();
                 
                 if(spdFile->getWaveformBitRes() == SPD_32_BIT_WAVE)
                 {
                     H5::IntType intU32DataType( H5::PredType::STD_U32LE );
                     intU32DataType.setOrder( H5T_ORDER_LE );
                     
                     boost::uint_fast32_t fillValueUInt = 0;
                     
                     H5::DSetCreatPropList creationTransmittedDSPList;
                     creationTransmittedDSPList.setChunk(1, dimsTransmittedChunk);
                     creationTransmittedDSPList.setShuffle();
                     creationTransmittedDSPList.setDeflate(SPD_DEFLATE);
                     creationTransmittedDSPList.setFillValue( H5::PredType::STD_U32LE, &fillValueUInt);
                     
                     transmittedDataset = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_TRANSMITTED, intU32DataType, waveformDataSpace, creationTransmittedDSPList));
                 }
                 else if(spdFile->getWaveformBitRes() == SPD_16_BIT_WAVE)
                 {
                     H5::IntType intU16DataType( H5::PredType::STD_U16LE );
                     intU16DataType.setOrder( H5T_ORDER_LE );
                     
                     boost::uint_fast32_t fillValueUInt = 0;
                     
                     H5::DSetCreatPropList creationTransmittedDSPList;
                     creationTransmittedDSPList.setChunk(1, dimsTransmittedChunk);
                     creationTransmittedDSPList.setShuffle();
                     creationTransmittedDSPList.setDeflate(SPD_DEFLATE);
                     creationTransmittedDSPList.setFillValue( H5::PredType::STD_U16LE, &fillValueUInt);
                     
                     transmittedDataset = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_TRANSMITTED, intU16DataType, waveformDataSpace, creationTransmittedDSPList));
                 }
                 else if(spdFile->getWaveformBitRes() == SPD_8_BIT_WAVE)
                 {
                     H5::IntType intU8DataType( H5::PredType::STD_U8LE );
                     intU8DataType.setOrder( H5T_ORDER_LE );
                     
                     boost::uint_fast32_t fillValueUInt = 0;
                     
                     H5::DSetCreatPropList creationTransmittedDSPList;
                     creationTransmittedDSPList.setChunk(1, dimsTransmittedChunk);
                     creationTransmittedDSPList.setShuffle();
                     creationTransmittedDSPList.setDeflate(SPD_DEFLATE);
                     creationTransmittedDSPList.setFillValue( H5::PredType::STD_U8LE, &fillValueUInt);
                     
                     transmittedDataset = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_TRANSMITTED, intU8DataType, waveformDataSpace, creationTransmittedDSPList));
                 }
                 else
                 {
                     throw SPDIOException("Waveform bit resolution is unknown.");
                 }
             }
             catch ( SPDIOException &e)
             {
                 throw e;
             }
             catch ( std::exception &e)
             {
                 throw SPDIOException(e.what());
             }
             
             
            // Open received Datasets
            try
            {
                receivedDataset = new H5::DataSet(spdOutH5File->openDataSet(SPDFILE_DATASETNAME_RECEIVED));
                
                H5::DataSpace receiveSpace = receivedDataset->getSpace();
                int receiveDims = receiveSpace.getSimpleExtentNdims();
                if(receiveDims != 1)
                {
                    throw SPDIOException("The received waveform has more than 1 dimension.");
                }
                hsize_t receiveLen = 0;
                receiveSpace.getSimpleExtentDims(&receiveLen);
                
                this->numReceiveVals = receiveLen;
            }
            catch ( H5::Exception &e)
            {
                hsize_t initDimsWaveformDS[1];
                initDimsWaveformDS[0] = 0;
                hsize_t maxDimsWaveformDS[1];
                maxDimsWaveformDS[0] = H5S_UNLIMITED;
                H5::DataSpace waveformDataSpace = H5::DataSpace(1, initDimsWaveformDS, maxDimsWaveformDS);
                
                hsize_t dimsReceivedChunk[1];
                dimsReceivedChunk[0] = spdFile->getReceivedBlockSize();
                
                if(spdFile->getWaveformBitRes() == SPD_32_BIT_WAVE)
                {
                    H5::IntType intU32DataType( H5::PredType::STD_U32LE );
                    intU32DataType.setOrder( H5T_ORDER_LE );
                    
                    boost::uint_fast32_t fillValueUInt = 0;
                    H5::DSetCreatPropList creationReceivedDSPList;
                    creationReceivedDSPList.setChunk(1, dimsReceivedChunk);
                    creationReceivedDSPList.setShuffle();
                    creationReceivedDSPList.setDeflate(SPD_DEFLATE);
                    creationReceivedDSPList.setFillValue( H5::PredType::STD_U32LE, &fillValueUInt);
                    
                    receivedDataset = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_RECEIVED, intU32DataType, waveformDataSpace, creationReceivedDSPList));
                }
                else if(spdFile->getWaveformBitRes() == SPD_16_BIT_WAVE)
                {
                    H5::IntType intU16DataType( H5::PredType::STD_U16LE );
                    intU16DataType.setOrder( H5T_ORDER_LE );
                    
                    boost::uint_fast32_t fillValueUInt = 0;
                    H5::DSetCreatPropList creationReceivedDSPList;
                    creationReceivedDSPList.setChunk(1, dimsReceivedChunk);
                    creationReceivedDSPList.setShuffle();
                    creationReceivedDSPList.setDeflate(SPD_DEFLATE);
                    creationReceivedDSPList.setFillValue( H5::PredType::STD_U16LE, &fillValueUInt);
                    
                    receivedDataset = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_RECEIVED, intU16DataType, waveformDataSpace, creationReceivedDSPList));
                }
                else if(spdFile->getWaveformBitRes() == SPD_8_BIT_WAVE)
                {
                    H5::IntType intU8DataType( H5::PredType::STD_U8LE );
                    intU8DataType.setOrder( H5T_ORDER_LE );
                    
                    boost::uint_fast32_t fillValueUInt = 0;
                    H5::DSetCreatPropList creationReceivedDSPList;
                    creationReceivedDSPList.setChunk(1, dimsReceivedChunk);
                    creationReceivedDSPList.setShuffle();
                    creationReceivedDSPList.setDeflate(SPD_DEFLATE);
                    creationReceivedDSPList.setFillValue( H5::PredType::STD_U8LE, &fillValueUInt);
                    
                    receivedDataset = new H5::DataSet(spdOutH5File->createDataSet(SPDFILE_DATASETNAME_RECEIVED, intU8DataType, waveformDataSpace, creationReceivedDSPList));
                }
                else
                {
                    throw SPDIOException("Waveform bit resolution is unknown.");
                }
            }
            catch ( SPDIOException &e)
            {
                throw e;
            }
            catch ( std::exception &e)
            {
                throw SPDIOException(e.what());
            }
            
            this->numPts = spdFile->getNumberOfPoints();
            this->numPulses = spdFile->getNumberOfPulses();
            
            xMinWritten = spdFile->getXMin();
            yMinWritten = spdFile->getYMin();
            zMinWritten = spdFile->getZMin();
            xMaxWritten = spdFile->getXMax();
            yMaxWritten = spdFile->getYMax();
            zMaxWritten = spdFile->getZMax();
            azMinWritten = spdFile->getAzimuthMin();
            zenMinWritten = spdFile->getAzimuthMax();
            ranMinWritten = spdFile->getRangeMin();
            azMaxWritten = spdFile->getRangeMax();
            zenMaxWritten = spdFile->getZenithMin();
            ranMaxWritten = spdFile->getZenithMax();            
            scanlineMinWritten = spdFile->getScanlineMin();
            scanlineMaxWritten = spdFile->getScanlineMax();
            scanlineIdxMinWritten = spdFile->getScanlineIdxMin();
            scanlineIdxMaxWritten = spdFile->getScanlineIdxMax();           
            			
			fileOpened = true;
			
			plsBuffer = new std::vector<SPDPulse*>();
            plsBuffer->reserve(spdFile->getPulseBlockSize());
		}
		catch( H5::FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataTypeIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
        catch( SPDIOException &e )
		{
			throw e;
		}
        catch( std::exception &e )
		{
			throw SPDIOException(e.what());
		}
        
        firstReturn = false;
        firstPulse = false;
        firstWaveform = false;
        
        reOpenedFile = true;
        
		return fileOpened;
    }
    
	void SPDNoIdxFileWriter::writeDataColumn(std::list<SPDPulse*> *plsIn, boost::uint_fast32_t col, boost::uint_fast32_t row)
	{
		SPDPulseUtils pulseUtils;
		SPDPointUtils pointUtils;
        
		if(!fileOpened)
		{
			throw SPDIOException("SPD (HDF5) file not open, cannot finalise.");
		}
		
		try 
		{
			H5::Exception::dontPrint() ;
			
			std::list<SPDPulse*>::iterator iterInPls;
			for(iterInPls = plsIn->begin(); iterInPls != plsIn->end(); ++iterInPls)
			{
				plsBuffer->push_back(*iterInPls);
				if(plsBuffer->size() == spdFile->getPulseBlockSize() )
				{
					unsigned long numPulsesInCol = plsBuffer->size();
					unsigned long numPointsInCol = 0;
					unsigned long numTransValsInCol = 0;
					unsigned long numReceiveValsInCol = 0;
					
					std::vector<SPDPulse*>::iterator iterPulses;
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
                            
                            scanlineMinWritten = (*iterPulses)->scanline;
                            scanlineMaxWritten = (*iterPulses)->scanline;
                            scanlineIdxMinWritten = (*iterPulses)->scanlineIdx;
                            scanlineIdxMaxWritten = (*iterPulses)->scanlineIdx;  
                            
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
                            
                            if((*iterPulses)->scanline < scanlineMinWritten)
                            {
                                scanlineMinWritten = (*iterPulses)->scanline;
                            }
                            else if((*iterPulses)->scanline > scanlineMaxWritten) 
                            {
                                scanlineMaxWritten = (*iterPulses)->scanline;
                            }
                            
                            if((*iterPulses)->scanlineIdx < scanlineIdxMinWritten)
                            {
                                scanlineIdxMinWritten = (*iterPulses)->scanlineIdx;
                            }
                            else if((*iterPulses)->scanlineIdx > scanlineIdxMaxWritten)  
                            {
                                scanlineIdxMaxWritten = (*iterPulses)->scanlineIdx;                            
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
					
					H5::DataSpace pulseWriteDataSpace = pulsesDataset->getSpace();
					pulseWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pulseDataDims, pulseDataOffset);
					H5::DataSpace newPulsesDataspace = H5::DataSpace(1, pulseDataDims);
					
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
						
						H5::DataSpace pointWriteDataSpace = pointsDataset->getSpace();
						pointWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pointsDataDims, pointsDataOffset);
						H5::DataSpace newPointsDataspace = H5::DataSpace(1, pointsDataDims);
						
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
						
						H5::DataSpace transWriteDataSpace = transmittedDataset->getSpace();
						transWriteDataSpace.selectHyperslab(H5S_SELECT_SET, transDataDims, transDataOffset);
						H5::DataSpace newTransDataspace = H5::DataSpace(1, transDataDims);
						
						transmittedDataset->write(transmittedValues, H5::PredType::NATIVE_ULONG, newTransDataspace, transWriteDataSpace);
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
						
						H5::DataSpace receivedWriteDataSpace = receivedDataset->getSpace();
						receivedWriteDataSpace.selectHyperslab(H5S_SELECT_SET, receivedDataDims, receivedDataOffset);
						H5::DataSpace newReceivedDataspace = H5::DataSpace(1, receivedDataDims);
						
						receivedDataset->write(receivedValues, H5::PredType::NATIVE_ULONG, newReceivedDataspace, receivedWriteDataSpace);
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
		catch( H5::FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataTypeIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch(SPDIOException &e)
		{
			throw e;
		}
	}
	
	void SPDNoIdxFileWriter::writeDataColumn(std::vector<SPDPulse*> *plsIn, boost::uint_fast32_t col, boost::uint_fast32_t row)
	{
		SPDPulseUtils pulseUtils;
		SPDPointUtils pointUtils;
        
		if(!fileOpened)
		{
			throw SPDIOException("SPD (HDF5) file not open, cannot finalise.");
		}
		
        
		try 
		{
			H5::Exception::dontPrint() ;
			
			std::vector<SPDPulse*>::iterator iterInPls;
			for(iterInPls = plsIn->begin(); iterInPls != plsIn->end(); ++iterInPls)
			{
				plsBuffer->push_back(*iterInPls);
				if(plsBuffer->size() == spdFile->getPulseBlockSize())
				{
                    //std::cout << "Writing buffer (" << numPulses <<  " Pulses)\n";
					unsigned long numPulsesInCol = plsBuffer->size();
					unsigned long numPointsInCol = 0;
					unsigned long numTransValsInCol = 0;
					unsigned long numReceiveValsInCol = 0;
					
					std::vector<SPDPulse*>::iterator iterPulses;
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
                            
                            scanlineMinWritten = (*iterPulses)->scanline;
                            scanlineMaxWritten = (*iterPulses)->scanline;
                            scanlineIdxMinWritten = (*iterPulses)->scanlineIdx;
                            scanlineIdxMaxWritten = (*iterPulses)->scanlineIdx;  
                            
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
                            
                            if((*iterPulses)->scanline < scanlineMinWritten)
                            {
                                scanlineMinWritten = (*iterPulses)->scanline;
                            }
                            else if((*iterPulses)->scanline > scanlineMaxWritten) 
                            {
                                scanlineMaxWritten = (*iterPulses)->scanline;
                            }
                            
                            if((*iterPulses)->scanlineIdx < scanlineIdxMinWritten)
                            {
                                scanlineIdxMinWritten = (*iterPulses)->scanlineIdx;
                            }
                            else if((*iterPulses)->scanlineIdx > scanlineIdxMaxWritten)  
                            {
                                scanlineIdxMaxWritten = (*iterPulses)->scanlineIdx;                            
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
                            
                            /*std::cout << "Pulse " << ++counter << std::endl;
                            std::cout << "\txMinWritten = " << xMinWritten << std::endl;
                            std::cout << "\txMaxWritten = " << xMaxWritten << std::endl;
                            std::cout << "\tyMinWritten = " << yMinWritten << std::endl;
                            std::cout << "\tyMaxWritten = " << yMaxWritten << std::endl;
                            std::cout << "\tzMinWritten = " << zMinWritten << std::endl;
                            std::cout << "\tzMaxWritten = " << zMaxWritten << std::endl << std::endl;*/
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
					
					H5::DataSpace pulseWriteDataSpace = pulsesDataset->getSpace();
					pulseWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pulseDataDims, pulseDataOffset);
					H5::DataSpace newPulsesDataspace = H5::DataSpace(1, pulseDataDims);
					
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
						
						H5::DataSpace pointWriteDataSpace = pointsDataset->getSpace();
						pointWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pointsDataDims, pointsDataOffset);
						H5::DataSpace newPointsDataspace = H5::DataSpace(1, pointsDataDims);
						
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
						
						H5::DataSpace transWriteDataSpace = transmittedDataset->getSpace();
						transWriteDataSpace.selectHyperslab(H5S_SELECT_SET, transDataDims, transDataOffset);
						H5::DataSpace newTransDataspace = H5::DataSpace(1, transDataDims);
						
						transmittedDataset->write(transmittedValues, H5::PredType::NATIVE_ULONG, newTransDataspace, transWriteDataSpace);
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
						
						H5::DataSpace receivedWriteDataSpace = receivedDataset->getSpace();
						receivedWriteDataSpace.selectHyperslab(H5S_SELECT_SET, receivedDataDims, receivedDataOffset);
						H5::DataSpace newReceivedDataspace = H5::DataSpace(1, receivedDataDims);
						
						receivedDataset->write(receivedValues, H5::PredType::NATIVE_ULONG, newReceivedDataspace, receivedWriteDataSpace);
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
		catch( H5::FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataTypeIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch(SPDIOException &e)
		{
			throw e;
		}
        catch(std::exception &e)
		{
			throw SPDIOException(e.what());
		}
	}
    
	void SPDNoIdxFileWriter::finaliseClose() 
	{
		if(!fileOpened)
		{
			throw SPDIOException("SPD (HDF5) file not open, cannot finalise.");
		}
		
		SPDPulseUtils pulseUtils;
		SPDPointUtils pointUtils;
		
		try 
		{
			H5::Exception::dontPrint() ;
			
			if(plsBuffer->size() > 0 )
			{
				unsigned long numPulsesInCol = plsBuffer->size();
				unsigned long numPointsInCol = 0;
				unsigned long numTransValsInCol = 0;
				unsigned long numReceiveValsInCol = 0;
				
				std::vector<SPDPulse*>::iterator iterPulses;
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
                        
                        scanlineMinWritten = (*iterPulses)->scanline;
                        scanlineMaxWritten = (*iterPulses)->scanline;
                        scanlineIdxMinWritten = (*iterPulses)->scanlineIdx;
                        scanlineIdxMaxWritten = (*iterPulses)->scanlineIdx;  
                        
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
                        
                        if((*iterPulses)->scanline < scanlineMinWritten)
                        {
                            scanlineMinWritten = (*iterPulses)->scanline;
                        }
                        else if((*iterPulses)->scanline > scanlineMaxWritten) 
                        {
                            scanlineMaxWritten = (*iterPulses)->scanline;
                        }
                            
                        if((*iterPulses)->scanlineIdx < scanlineIdxMinWritten)
                        {
                            scanlineIdxMinWritten = (*iterPulses)->scanlineIdx;
                        }
                        else if((*iterPulses)->scanlineIdx > scanlineIdxMaxWritten)  
                        {
                            scanlineIdxMaxWritten = (*iterPulses)->scanlineIdx;                            
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
				
				H5::DataSpace pulseWriteDataSpace = pulsesDataset->getSpace();
				pulseWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pulseDataDims, pulseDataOffset);
				H5::DataSpace newPulsesDataspace = H5::DataSpace(1, pulseDataDims);
				
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
					
					H5::DataSpace pointWriteDataSpace = pointsDataset->getSpace();
					pointWriteDataSpace.selectHyperslab(H5S_SELECT_SET, pointsDataDims, pointsDataOffset);
					H5::DataSpace newPointsDataspace = H5::DataSpace(1, pointsDataDims);
					
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
					
					H5::DataSpace transWriteDataSpace = transmittedDataset->getSpace();
					transWriteDataSpace.selectHyperslab(H5S_SELECT_SET, transDataDims, transDataOffset);
					H5::DataSpace newTransDataspace = H5::DataSpace(1, transDataDims);
					
					transmittedDataset->write(transmittedValues, H5::PredType::NATIVE_ULONG, newTransDataspace, transWriteDataSpace);
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
					
					H5::DataSpace receivedWriteDataSpace = receivedDataset->getSpace();
					receivedWriteDataSpace.selectHyperslab(H5S_SELECT_SET, receivedDataDims, receivedDataOffset);
					H5::DataSpace newReceivedDataspace = H5::DataSpace(1, receivedDataDims);
					
					receivedDataset->write(receivedValues, H5::PredType::NATIVE_ULONG, newReceivedDataspace, receivedWriteDataSpace);
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
		catch( H5::FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataTypeIException &e )
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
                
                spdFile->setScanlineMin(scanlineMinWritten);
                spdFile->setScanlineMax(scanlineMaxWritten);
                spdFile->setScanlineIdxMin(scanlineIdxMinWritten);
                spdFile->setScanlineIdxMax(scanlineIdxMaxWritten);  
                
            }
            
            if(!firstReturn)
            {
                //spdFile->setBoundingVolume(xMinWritten, xMaxWritten, yMinWritten, yMaxWritten, zMinWritten, zMaxWritten);
                if(!this->keepMinExtent)
                {
                    spdFile->setBoundingVolume(xMinWritten, xMaxWritten, yMinWritten, yMaxWritten, zMinWritten, zMaxWritten);
                }
                else
                {
                    if(xMinWritten < spdFile->getXMin())
                    {
                        spdFile->setXMin(xMinWritten);
                    }
                    if(xMaxWritten > spdFile->getXMax())
                    {
                        spdFile->setXMax(xMaxWritten);
                    }
                    if(yMinWritten < spdFile->getYMin())
                    {
                        spdFile->setYMin(yMinWritten);
                    }
                    if(yMaxWritten > spdFile->getYMax())
                    {
                        spdFile->setYMax(yMaxWritten);
                    }
                    if(zMinWritten < spdFile->getZMin())
                    {
                        spdFile->setZMin(zMinWritten);
                    }
                    if(zMaxWritten > spdFile->getZMax())
                    {
                        spdFile->setZMax(zMaxWritten);
                    }
                    
                }
                spdFile->setBoundingVolumeSpherical(zenMinWritten, zenMaxWritten, azMinWritten, azMaxWritten, ranMinWritten, ranMaxWritten);
                spdFile->setBoundingBoxScanline(scanlineMinWritten, scanlineMaxWritten, scanlineIdxMinWritten, scanlineIdxMaxWritten);
            }
            
            spdFile->setNumberOfPoints(numPts);
            spdFile->setNumberOfPulses(numPulses);
            
            spdFile->setFileType(SPD_UPD_TYPE);
            
            spdFile->setIndexType(SPD_NO_IDX);
            
            //std::cout << "spdFile:\n" << spdFile << std::endl;
            
            if(reOpenedFile)
            {
                this->updateHeaderInfo(spdOutH5File, spdFile);
            }
            else
            {
                this->writeHeaderInfo(spdOutH5File, spdFile);
            }
            reOpenedFile = false;
            
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
		catch( H5::FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( H5::DataTypeIException &e )
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
                std::cerr << "WARNING: " << e.what() << std::endl;
			}
		}
	}
}


