/*
 *  SPDCalcMetrics.h
 *  SPDLIB
 *
 *  Created by Pete Bunting on 17/03/2011.
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
 */

#ifndef SPDCalcMetrics_H
#define SPDCalcMetrics_H

#include <iostream>
#include <string>
#include <list>
#include <vector>
#include <sys/stat.h>

#include <gsl/gsl_sort.h>
#include <gsl/gsl_statistics.h>

#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>

#include <boost/math/special_functions/fpclassify.hpp>

#include "spd/SPDFile.h"
#include "spd/SPDPoint.h"
#include "spd/SPDPulse.h"
#include "spd/SPDMetrics.h"
#include "spd/SPDTextFileUtilities.h"
#include "spd/SPDProcessPulses.h"
#include "spd/SPDPulseProcessor.h"
#include "spd/SPDProcessingException.h"
#include "spd/SPDPolygonProcessor.h"
#include "spd/SPDSetupProcessPolygons.h"
#include "spd/SPDProcessPolygons.h"

namespace spdlib
{
	
	class DllExport SPDCalcMetrics
	{
	public:
		SPDCalcMetrics();
		void calcMetricToImage(std::string inXMLFilePath, std::string inputSPDFile, std::string outputImage, boost::uint_fast32_t blockXSize=250, boost::uint_fast32_t blockYSize=250, float processingResolution=0, std::string gdalFormat="ENVI") throw (SPDProcessingException);
        void calcMetricToVectorShp(std::string inXMLFilePath, std::string inputSPDFile, std::string inputVectorShp, std::string outputVectorShp, bool deleteOutShp, bool copyAttributes) throw (SPDProcessingException);
        void calcMetricForVector2ASCII(std::string inXMLFilePath, std::string inputSPDFile, std::string inputVectorShp, std::string outputASCII) throw (SPDProcessingException);
		~SPDCalcMetrics();
    protected:
        void parseMetricsXML(std::string inXMLFilePath, std::vector<SPDMetric*> *metrics, std::vector<std::string> *fieldNames) throw(SPDProcessingException);
        SPDMetric* createMetric(xercesc::DOMElement *metricElement) throw(SPDProcessingException);
	};
    

    
    class DllExport SPDCalcPolyMetrics : public SPDPolygonProcessor
	{
	public:
		SPDCalcPolyMetrics(std::vector<SPDMetric*> *metrics, std::vector<std::string> *fieldNames);
		void processFeature(OGRFeature *inFeature, OGRFeature *outFeature,boost::uint_fast64_t fid, std::vector<SPDPulse*> *pulses, SPDFile *spdFile) throw(SPDProcessingException);
        void processFeature(OGRFeature *inFeature, std::ofstream *outASCIIFile, boost::uint_fast64_t fid, std::vector<SPDPulse*> *pulses, SPDFile *spdFile) throw(SPDProcessingException);
        void createOutputLayerDefinition(OGRLayer *outputLayer, OGRFeatureDefn *inFeatureDefn) throw(SPDProcessingException);
        void writeASCIIHeader(std::ofstream *outASCIIFile) throw(SPDProcessingException);
		~SPDCalcPolyMetrics();
    private:
        std::vector<SPDMetric*> *metrics;
        std::vector<std::string> *fieldNames;
	};
    
    
    class DllExport SPDCalcImageMetrics : public SPDPulseProcessor
	{
	public:
        SPDCalcImageMetrics(std::vector<SPDMetric*> *metrics, std::vector<std::string> *fieldNames);
        
        void processDataColumnImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float *imageData, SPDXYPoint *cenPts, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException);

		void processDataColumn(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, SPDXYPoint *cenPts) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing must output an image. therefore function is not implemented.");};
        void processDataWindowImage(SPDFile *inSPDFile, bool **validBins, std::vector<SPDPulse*> ***pulses, float ***imageData, SPDXYPoint ***cenPts, boost::uint_fast32_t numImgBands, float binSize, boost::uint_fast16_t winSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing using a window is not implemented.");};
		void processDataWindow(SPDFile *inSPDFile, bool **validBins, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast16_t winSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing using a window is not implemented.");};
        
        std::vector<std::string> getImageBandDescriptions() throw(SPDProcessingException);
        void setHeaderValues(SPDFile *spdFile) throw(SPDProcessingException);
        
        ~SPDCalcImageMetrics();
    private:
        std::vector<SPDMetric*> *metrics;
        std::vector<std::string> *fieldNames;
	};
    
    
    
    class DllExport SPDCalcZMedianVal : public SPDPulseProcessor
	{
	public:
        SPDCalcZMedianVal();
        
        void processDataColumnImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float *imageData, SPDXYPoint *cenPts, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing is not implemented for processDataColumn().");};
        
		void processDataColumn(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, SPDXYPoint *cenPts) throw(SPDProcessingException);
        
        void processDataWindowImage(SPDFile *inSPDFile, bool **validBins, std::vector<SPDPulse*> ***pulses, float ***imageData, SPDXYPoint ***cenPts, boost::uint_fast32_t numImgBands, float binSize, boost::uint_fast16_t winSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing using a window is not implemented.");};
		void processDataWindow(SPDFile *inSPDFile, bool **validBins, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast16_t winSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing using a window is not implemented.");};
        
        std::vector<std::string> getImageBandDescriptions() throw(SPDProcessingException){return std::vector<std::string>();};
        void setHeaderValues(SPDFile *spdFile) throw(SPDProcessingException){};
        
        double getMedianMedianVal();
        
        ~SPDCalcZMedianVal();
    protected:
        std::vector<double> *colMedianVals;
	};
}

#endif
