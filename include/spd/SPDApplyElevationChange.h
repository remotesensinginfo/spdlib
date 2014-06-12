/*
 *  SPDApplyElevationChange.h
 *  SPDLIB
 *
 *  Created by Pete Bunting on 11/02/2011.
 *  Copyright 2011 SPDLib. All rights reserved.
 *
 *  This file is part of SPDLib.
 *
 *  SPDLib is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  SPDLib is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with SPDLib.  If not, see <http://www.gnu.org/licenses/>.
 *
 */


#ifndef SPDApplyElevationChange_H
#define SPDApplyElevationChange_H

#include <iostream>
#include <fstream>
#include <string>
#include <list>

#include <boost/cstdint.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

#include "spd/SPDFile.h"
#include "spd/SPDPoint.h"
#include "spd/SPDPulse.h"
#include "spd/SPDImageUtils.h"
#include "spd/SPDFileWriter.h"
#include "spd/SPDFileReader.h"
#include "spd/SPDProcessPulses.h"
#include "spd/SPDPulseProcessor.h"
#include "spd/SPDProcessingException.h"

namespace spdlib
{	
	class SPDApplyElevationChange
	{
	public:
		SPDApplyElevationChange();
		void applyConstantElevationChangeUnsorted(std::string inputFile, std::string outputFile, double elevConstant, bool addOffset) throw(SPDException);
		void applyConstantElevationChangeSPD(std::string inputSPDFile, std::string outputSPDFile, double elevConstant, bool addOffset, boost::uint_fast32_t blockXSize, boost::uint_fast32_t blockYSize) throw(SPDException);
		void applyVariableElevationChangeUnsorted(std::string inputFile, std::string outputFile, std::string elevImage, bool addOffset) throw(SPDException);
		void applyVariableElevationChangeSPD(std::string inputSPDFile, std::string outputSPDFile, std::string elevImage, bool addOffset, boost::uint_fast32_t blockXSize, boost::uint_fast32_t blockYSize) throw(SPDException);
		~SPDApplyElevationChange();
	};
	
	class SPDApplyUnsortedElevChangeConstant : public SPDImporterProcessor
	{
	public:
		SPDApplyUnsortedElevChangeConstant(double elevConstant, bool addOffset, SPDDataExporter *exporter, SPDFile *spdFileOut) throw(SPDException);
		void processImportedPulse(SPDFile *spdFile, SPDPulse *pulse) throw(SPDIOException);
		void completeFileAndClose(SPDFile *spdFile)throw(SPDIOException);
		~SPDApplyUnsortedElevChangeConstant();
	private:
		double elevConstant;
		bool addOffset;
		SPDDataExporter *exporter;
		SPDFile *spdFileOut;
		std::list<SPDPulse*> *pulses;
	};
	
	class SPDApplyUnsortedElevChangeVariable : public SPDImporterProcessor
	{
	public:
		SPDApplyUnsortedElevChangeVariable(GDALDataset *elevImage, bool addOffset, SPDDataExporter *exporter, SPDFile *spdFileOut) throw(SPDException);
		void processImportedPulse(SPDFile *spdFile, SPDPulse *pulse) throw(SPDIOException);
		void completeFileAndClose(SPDFile *spdFile)throw(SPDIOException);
		~SPDApplyUnsortedElevChangeVariable();
	private:
		GDALDataset *elevImage;
		bool addOffset;
		SPDDataExporter *exporter;
		SPDFile *spdFileOut;
		std::list<SPDPulse*> *pulses;
		float **pxlVals;
	boost::int_fast32_t *prevImgX;
	boost::int_fast32_t *prevImgY;
		bool first;
	};
	
    
    
    class SPDApplySPDElevChangeConstant : public SPDPulseProcessor
	{
	public:
        SPDApplySPDElevChangeConstant(double elevConstant, bool addOffset);
        
        void processDataColumnImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float *imageData, SPDXYPoint *cenPts, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing is not implemented for processDataColumnImage().");};
        
		void processDataColumn(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, SPDXYPoint *cenPts) throw(SPDProcessingException);
        
        void processDataWindowImage(SPDFile *inSPDFile, bool **validBins, std::vector<SPDPulse*> ***pulses, float ***imageData, SPDXYPoint ***cenPts, boost::uint_fast32_t numImgBands, float binSize, boost::uint_fast16_t winSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing using a window is not implemented.");};
		void processDataWindow(SPDFile *inSPDFile, bool **validBins, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast16_t winSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing using a window is not implemented.");};
        
        std::vector<std::string> getImageBandDescriptions() throw(SPDProcessingException)
        {
            return std::vector<std::string>();
        };
        void setHeaderValues(SPDFile *spdFile) throw(SPDProcessingException){};
        
        ~SPDApplySPDElevChangeConstant();
    private:
		double elevConstant;
		bool addOffset;
	};
    
    class SPDApplySPDElevChangeVariable : public SPDPulseProcessor
	{
	public:
        SPDApplySPDElevChangeVariable(GDALDataset *elevImage, bool addOffset);
        
        void processDataColumnImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float *imageData, SPDXYPoint *cenPts, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing is not implemented for processDataColumnImage().");};
        
		void processDataColumn(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, SPDXYPoint *cenPts) throw(SPDProcessingException);
        
        void processDataWindowImage(SPDFile *inSPDFile, bool **validBins, std::vector<SPDPulse*> ***pulses, float ***imageData, SPDXYPoint ***cenPts, boost::uint_fast32_t numImgBands, float binSize, boost::uint_fast16_t winSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing using a window is not implemented.");};
		void processDataWindow(SPDFile *inSPDFile, bool **validBins, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast16_t winSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing using a window is not implemented.");};
        
        std::vector<std::string> getImageBandDescriptions() throw(SPDProcessingException)
        {
            return std::vector<std::string>();
        };
        void setHeaderValues(SPDFile *spdFile) throw(SPDProcessingException){};
        
        ~SPDApplySPDElevChangeVariable();
    private:
		GDALDataset *elevImage;
		bool addOffset;
		float **pxlVals;
        boost::int_fast32_t *prevImgX;
        boost::int_fast32_t *prevImgY;
		bool first;
	};
    
    
    /*
	class SPDApplySPDElevChangeConstant : public SPDPulsesProcessor
	{
	public:
		SPDApplySPDElevChangeConstant(double elevConstant, bool addOffset);
		
		bool processPulsesInputImage(SPDFile*, std::vector<SPDPulse*>*, float*, unsigned int) throw(SPDProcessingException){throw SPDProcessingException("Not Implemented");};
		bool processPulsesInputImageCenPxl(SPDFile*, std::vector<SPDPulse*>*, float*, unsigned int, SPDPulse*, float) throw(SPDProcessingException){throw SPDProcessingException("Not Implemented");};
		
		void processPulsesOutputImage(SPDFile*, std::vector<SPDPulse*>*, float*, unsigned int) throw(SPDProcessingException){throw SPDProcessingException("Not Implemented");};
		void processPulsesOutputImageCenPxl(SPDFile*, std::vector<SPDPulse*>*, float*, unsigned int, SPDPulse*, float) throw(SPDProcessingException){throw SPDProcessingException("Not Implemented");};
		
		bool processPulses(SPDFile *spdFile, std::vector<SPDPulse*> *pulses) throw(SPDProcessingException);
		bool processPulsesCenPxl(SPDFile*, std::vector<SPDPulse*>*, SPDPulse*, float) throw(SPDProcessingException){throw SPDProcessingException("Not Implemented");};
		
		bool processPulsesWindowInputImage(SPDFile*, std::list<SPDPulse*>***, unsigned int, float***, unsigned int) throw(SPDProcessingException){throw SPDProcessingException("Not Implemented");};
		bool processPulsesWindowInputImageCenPxl(SPDFile*, std::list<SPDPulse*>***, unsigned int, float***, unsigned int, SPDPulse*, float) throw(SPDProcessingException){throw SPDProcessingException("Not Implemented");};
		
		void processPulsesWindowOutputImage(SPDFile *spdFile, std::list<SPDPulse*>***, unsigned int, float*, unsigned int) throw(SPDProcessingException){throw SPDProcessingException("Not Implemented");};
		void processPulsesWindowOutputImageCenPxl(SPDFile*, std::list<SPDPulse*>***, unsigned int, float*, unsigned int, SPDPulse*, float) throw(SPDProcessingException){throw SPDProcessingException("Not Implemented");};
		
		bool processPulsesWindow(SPDFile*, std::list<SPDPulse*>***, unsigned int) throw(SPDProcessingException){throw SPDProcessingException("Not Implemented");};
		bool processPulsesWindowCenPxl(SPDFile*, std::list<SPDPulse*>***, unsigned int, SPDPulse*, float) throw(SPDProcessingException){throw SPDProcessingException("Not Implemented");};
		
        std::vector<std::string> getImageBandDescriptions() throw(SPDProcessingException){throw SPDProcessingException("Not Implemented");}
        
		~SPDApplySPDElevChangeConstant();
	private:
		double elevConstant;
		bool addOffset;
	};
	
	class SPDApplySPDElevChangeVariable : public SPDPulsesProcessor
	{
	public:
		SPDApplySPDElevChangeVariable(GDALDataset *elevImage, bool addOffset);
		
		bool processPulsesInputImage(SPDFile*, std::vector<SPDPulse*>*, float*, unsigned int) throw(SPDProcessingException){throw SPDProcessingException("Not Implemented");};
		bool processPulsesInputImageCenPxl(SPDFile*, std::vector<SPDPulse*>*, float*, unsigned int, SPDPulse*, float) throw(SPDProcessingException){throw SPDProcessingException("Not Implemented");};
		
		void processPulsesOutputImage(SPDFile*, std::vector<SPDPulse*>*, float*, unsigned int) throw(SPDProcessingException){throw SPDProcessingException("Not Implemented");};
		void processPulsesOutputImageCenPxl(SPDFile*, std::vector<SPDPulse*>*, float*, unsigned int, SPDPulse*, float) throw(SPDProcessingException){throw SPDProcessingException("Not Implemented");};
		
		bool processPulses(SPDFile *spdFile, std::vector<SPDPulse*> *pulses) throw(SPDProcessingException);
		bool processPulsesCenPxl(SPDFile*, std::vector<SPDPulse*>*, SPDPulse*, float) throw(SPDProcessingException){throw SPDProcessingException("Not Implemented");};
		
		bool processPulsesWindowInputImage(SPDFile*, std::list<SPDPulse*>***, unsigned int, float***, unsigned int) throw(SPDProcessingException){throw SPDProcessingException("Not Implemented");};
		bool processPulsesWindowInputImageCenPxl(SPDFile*, std::list<SPDPulse*>***, unsigned int, float***, unsigned int, SPDPulse*, float) throw(SPDProcessingException){throw SPDProcessingException("Not Implemented");};
		
		void processPulsesWindowOutputImage(SPDFile *spdFile, std::list<SPDPulse*>***, unsigned int, float*, unsigned int) throw(SPDProcessingException){throw SPDProcessingException("Not Implemented");};
		void processPulsesWindowOutputImageCenPxl(SPDFile*, std::list<SPDPulse*>***, unsigned int, float*, unsigned int, SPDPulse*, float) throw(SPDProcessingException){throw SPDProcessingException("Not Implemented");};
		
		bool processPulsesWindow(SPDFile*, std::list<SPDPulse*>***, unsigned int) throw(SPDProcessingException){throw SPDProcessingException("Not Implemented");};
		bool processPulsesWindowCenPxl(SPDFile*, std::list<SPDPulse*>***, unsigned int, SPDPulse*, float) throw(SPDProcessingException){throw SPDProcessingException("Not Implemented");};
		
        std::vector<std::string> getImageBandDescriptions() throw(SPDProcessingException){throw SPDProcessingException("Not Implemented");}
        
		~SPDApplySPDElevChangeVariable();
	private:
		GDALDataset *elevImage;
		bool addOffset;
		float **pxlVals;
        boost::int_fast32_t *prevImgX;
        boost::int_fast32_t *prevImgY;
		bool first;
	};
	*/
}

#endif
