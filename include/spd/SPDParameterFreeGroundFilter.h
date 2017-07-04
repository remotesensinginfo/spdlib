/*
 *  SPDParameterFreeGroundFilter.h
 *
 *  Created by Pete Bunting on 04/05/2012.
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

#ifndef SPDParameterFreeGroundFilter_H
#define SPDParameterFreeGroundFilter_H

#include <iostream>
#include <string>
#include <list>

#include <boost/cstdint.hpp>

#include "spd/SPDFile.h"
#include "spd/SPDPoint.h"
#include "spd/SPDPulse.h"
#include "spd/SPDProcessingException.h"
#include "spd/SPDDataBlockProcessor.h"

#include "boost/math/special_functions/fpclassify.hpp"

#include "spd/tps/spline.h"
#include "spd/tps/linalg3d.h"
#include "spd/tps/ludecomposition.h"

#define CLOSING_WINDOW_SIZE 9
#define OPENING_WINDOW_SIZE 11
#define MORPH_MIN_THRESHOLD 1.0f

// mark all exported classes/functions with DllExport to have
// them exported by Visual Studio
#undef DllExport
#ifdef _MSC_VER
    #ifdef libspd_EXPORTS
        #define DllExport   __declspec( dllexport )
    #else
        #define DllExport   __declspec( dllimport )
    #endif
#else
    #define DllExport
#endif

namespace spdlib
{
    
    struct SPDPFFProcessLevel
    {
        float **data;
        boost::uint_fast32_t xSize;
        boost::uint_fast32_t ySize;
        float pxlRes;
    };
        
    class DllExport SPDParameterFreeGroundFilter : public SPDDataBlockProcessor
	{
	public:
        // constructor
        SPDParameterFreeGroundFilter(float grdPtDev, boost::uint_fast16_t classParameters, bool checkForFalseMinma, boost::uint_fast32_t kValue, boost::uint_fast32_t classifyDevThresh, boost::uint_fast32_t topHatStart, bool topHatScales, boost::uint_fast32_t topHatFactor, boost::uint_fast16_t minPointDensity);
        
        // public functions
        void processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException);
		void processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binSize) throw(SPDProcessingException);
        void processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands) throw(SPDProcessingException)
		{throw SPDProcessingException("SPDProgressiveMophologicalGrdFilter requires processing with a grid.");};
        void processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses) throw(SPDProcessingException)
        {throw SPDProcessingException("SPDProgressiveMophologicalGrdFilter requires processing with a grid.");};
        
        boost::uint_fast16_t** generateHoldingElement(boost::uint_fast16_t elSize);
        void deleteHoldingElement(boost::uint_fast16_t** toDelete, boost::uint_fast16_t elSize);
        
        std::vector<std::string> getImageBandDescriptions() throw(SPDProcessingException)
        {
            std::vector<std::string> bandNames;
            bandNames.push_back("PFF Surface");
            return bandNames;
        }
        void setHeaderValues(SPDFile *spdFile) throw(SPDProcessingException)
        {
            // Nothing to do...
        }
        
        ~SPDParameterFreeGroundFilter();
    protected:
        // members
        float grdPtDev;
        boost::uint_fast16_t classParameters;
        bool checkForFalseMinma;
        float k;
        boost::uint_fast32_t classDevThresh;
        boost::uint_fast32_t thSize;
        bool thScales;
        boost::uint_fast32_t thFac;
        boost::uint_fast16_t mpd;
        // functions
        void findMinSurface(std::vector<SPDPulse*> ***pulses, float **elev, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize);
        void performErosion(float **elev, float **elevErode, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast16_t filterHSize, boost::uint_fast16_t **element);
		void performDialation(float **elev, float **elevDialate, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast16_t filterHSize, boost::uint_fast16_t **element);
        void performOpenning(float **elev, float **elevOpen, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast16_t filterHSize, boost::uint_fast16_t **element);
        void performClosing(float **elev, float **elevClose, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast16_t filterHSize, boost::uint_fast16_t **element);
        void performWhiteTopHat(float **elev, float **elevTH, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast16_t filterHSize, boost::uint_fast16_t **element);
        void createStructuringElement(boost::uint_fast16_t **element, boost::uint_fast16_t filterHSize);
        void freeHierarchy(std::vector<SPDPFFProcessLevel*> *levels);
        void freeLevel(SPDPFFProcessLevel *level);
        void getSingleCellThreshold(std::vector<SPDPulse*> *pulses, float dtmHeight, float *outMedian, float *outStdDev);
        void filterPoints(std::vector<float> *allPoints, std::vector<float> *filteredPoints);
        void getMedianAndStdDev(boost::uint_fast32_t x, boost::uint_fast32_t y, float **data, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast16_t filterHSize, boost::uint_fast16_t **element, float* results);
        void calcResidualMedianStdDev(std::vector<float> *residuals, float *outMed, float *outStdDev);

        float getThreshold(float mean, float stdDev);
        float getClassificationThreshold(float mean, float stdDev);
        std::vector<SPDPFFProcessLevel*>* generateHierarchy(float **initElev, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float pxlRes);
        SPDPFFProcessLevel* interpLevel(SPDPFFProcessLevel *cLevel, SPDPFFProcessLevel *pLevel, double tlY, double tlX);
        SPDPFFProcessLevel* runSurfaceEstimation(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binSize) throw(SPDProcessingException);

	};
    
    
    
    
    class DllExport SPDTPSPFFGrdFilteringInterpolator
	{
	public:
		SPDTPSPFFGrdFilteringInterpolator(float radius);
		void initInterpolator(float **data, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, double tlEastings, double tlNorthings, float binSize) throw(SPDProcessingException);
		float getValue(double eastings, double northings) throw(SPDProcessingException);
		void resetInterpolator() throw(SPDProcessingException);
		~SPDTPSPFFGrdFilteringInterpolator();
	private:
        inline double distance(double eastings, double northings, double cEastings, double cNorthings);
        bool initialised;
        float **data;
        boost::uint_fast32_t xSize;
        boost::uint_fast32_t ySize;
        double tlEastings;
        double tlNorthings;
        float binSize;
        float radius;
	};
     
    
}



#endif



