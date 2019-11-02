/*
 *  SPDMultiscaleCurvatureGrdClassification.h
 *
 *  Created by Pete Bunting on 03/03/2012.
 *  Copyright 2012 SPDLib. All rights reserved.
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

#ifndef SPDMultiscaleCurvatureGrdClassification_H
#define SPDMultiscaleCurvatureGrdClassification_H

#include <iostream>
#include <string>
#include <list>
#include <algorithm>
#include "math.h"

#include <boost/cstdint.hpp>

#include "spd/SPDCommon.h"
#include "spd/SPDFile.h"
#include "spd/SPDPoint.h"
#include "spd/SPDPulse.h"
#include "spd/SPDProcessingException.h"
#include "spd/SPDDataBlockProcessor.h"
#include "spd/SPDPointGridIndex.h"

#include "spd/tps/spline.h"
#include "spd/tps/linalg3d.h"
#include "spd/tps/ludecomposition.h"

#include <boost/math/special_functions/fpclassify.hpp>


namespace spdlib
{    
    enum SPDSmoothFilterType
    {
        meanFilter = 0,
        medianFilter = 1
    };
    
    class DllExport SPDMultiscaleCurvatureGrdClassification : public SPDDataBlockProcessor
	{
	public:
        SPDMultiscaleCurvatureGrdClassification(float initScale,boost::uint_fast16_t numOfScalesAbove,boost::uint_fast16_t numOfScalesBelow, float scaleGaps, float initCurveTolerance, float minCurveTolerance, float stepCurveTolerance, float interpMaxRadius,boost::uint_fast16_t interpNumPoints, SPDSmoothFilterType filterType,boost::uint_fast16_t smoothFilterHSize, float thresOfChange, bool multiReturnPulsesOnly,boost::uint_fast16_t classParameters);
        void processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands, float binSize) 
		{throw SPDProcessingException("SPDMultiscaleCurvatureGrdClassification cannot output an image layer.");};
        
        void processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binSize) ;
        
        void processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands) 
		{throw SPDProcessingException("SPDMultiscaleCurvatureGrdClassification requires processing with a grid.");};
        
        void processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses) 
        {throw SPDProcessingException("SPDMultiscaleCurvatureGrdClassification requires processing with a grid.");};
        
        std::vector<std::string> getImageBandDescriptions() 
        {
            std::vector<std::string> bandNames;
            bandNames.push_back("MCC Surface");
            return bandNames;
        }
        void setHeaderValues(SPDFile *spdFile) 
        {
            // Nothing to do...
        }

        ~SPDMultiscaleCurvatureGrdClassification();
        
    protected:
        std::pair<double*,size_t> findDataExtentAndClassifyAllPtsAsGrd(std::vector<SPDPulse*> ***pulses,boost::uint_fast32_t xSizePulses,boost::uint_fast32_t ySizePulses) ;
        void assignToUnclassified(std::vector<SPDPulse*> ***pulses,boost::uint_fast32_t xSizePulses,boost::uint_fast32_t ySizePulses) ;
        float** createElevationRaster(double *bbox, float rasterScale,boost::uint_fast32_t *xSizeRaster,boost::uint_fast32_t *ySizeRaster, std::vector<SPDPulse*> ***pulses,boost::uint_fast32_t xSizePulses,boost::uint_fast32_t ySizePulses) ;
        void smoothMeanRaster(float **raster,boost::uint_fast32_t xSizeRaster,boost::uint_fast32_t ySizeRasterr,boost::uint_fast16_t filterHSize) ;
        void smoothMedianRaster(float **raster,boost::uint_fast32_t xSizeRaster,boost::uint_fast32_t ySizeRasterr,boost::uint_fast16_t filterHSize) ;
        float classifyNonGrdPoints(float curveTolerance, double *bbox, float rasterScale, float **raster,boost::uint_fast32_t xSizeRaster,boost::uint_fast32_t ySizeRaster, std::vector<SPDPulse*> ***pulses,boost::uint_fast32_t xSizePulses,boost::uint_fast32_t ySizePulses) ;
        float initScale;
        boost::uint_fast16_t numOfScalesAbove;
        boost::uint_fast16_t numOfScalesBelow;
        float scaleGaps;
        float initCurveTolerance;
        float minCurveTolerance;
        float stepCurveTolerance;
        float interpMaxRadius;
        boost::uint_fast16_t interpNumPoints;
        SPDSmoothFilterType filterType;
        boost::uint_fast16_t smoothFilterHSize;
        float thresOfChange;
        bool multiReturnPulsesOnly;
        boost::uint_fast16_t classParameters;
	};
    
    
    
    class DllExport SPDTPSNumPtsUseAvThinInterpolator
	{
	public:
		SPDTPSNumPtsUseAvThinInterpolator(float radius,boost::uint_fast16_t numPoints,boost::uint_fast16_t elevVal, double gridResolution, bool thinGrid);
		void initInterpolator(std::list<SPDPulse*> ***pulses,boost::uint_fast32_t numXBins,boost::uint_fast32_t numYBins,boost::uint_fast16_t ptClass) ;
		void initInterpolator(std::vector<SPDPulse*> ***pulses,boost::uint_fast32_t numXBins,boost::uint_fast32_t numYBins,boost::uint_fast16_t ptClass) ;
		void initInterpolator(std::list<SPDPulse*> *pulses,boost::uint_fast16_t ptClass) ;
		void initInterpolator(std::vector<SPDPulse*> *pulses,boost::uint_fast16_t ptClass) ;
		float getValue(double eastings, double northings) ;
		void resetInterpolator() ;
		~SPDTPSNumPtsUseAvThinInterpolator();
	protected:
        bool initialised;
		SPDPointGridIndex *idx;
		std::vector<SPDPoint*> *pts;
		double gridResolution;
        bool thinGrid;
        float radius;
        boost::uint_fast16_t numPoints;
        boost::uint_fast16_t elevVal;
	};
    
    
}



#endif



