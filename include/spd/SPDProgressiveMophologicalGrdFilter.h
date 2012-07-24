/*
 *  SPDProgressiveMophologicalGrdFilter.h
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

#ifndef SPDProgressiveMophologicalGrdFilter_H
#define SPDProgressiveMophologicalGrdFilter_H

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

namespace spdlib
{

    class SPDProgressiveMophologicalGrdFilter : public SPDDataBlockProcessor
	{
	public:
        SPDProgressiveMophologicalGrdFilter(boost::uint_fast16_t initFilterHSize, boost::uint_fast16_t maxFilterHSize, float terrainSlope, float initElevDiff, float maxElevDiff, float grdPtDev, bool medianFilter,boost::uint_fast16_t medianFilterHSize,boost::uint_fast16_t classParameters);
        void processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException);
		void processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binSize) throw(SPDProcessingException);
        
        void processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands) throw(SPDProcessingException)
		{throw SPDProcessingException("SPDProgressiveMophologicalGrdFilter requires processing with a grid.");};
        
        void processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses) throw(SPDProcessingException)
        {throw SPDProcessingException("SPDProgressiveMophologicalGrdFilter requires processing with a grid.");};
        
        std::vector<std::string> getImageBandDescriptions() throw(SPDProcessingException)
        {
            std::vector<std::string> bandNames;
            bandNames.push_back("PMF Surface");
            return bandNames;
        }
        void setHeaderValues(SPDFile *spdFile) throw(SPDProcessingException)
        {
            // Nothing to do...
        }
        
        ~SPDProgressiveMophologicalGrdFilter();
    protected:
        void performErosion(float **elev, float **elevErode, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast16_t filterHSize, boost::uint_fast16_t **element);
		void performDialation(float **elev, float **elevDialate, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast16_t filterHSize, boost::uint_fast16_t **element);
		void findMinSurface(std::vector<SPDPulse*> ***pulses, float **elev, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize);
		void applyMedianFilter(float **elev, float **elevMedian, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast16_t filterHSize, boost::uint_fast16_t **element);
		void fillHolesInMinSurface(float **elev, float **elevOut, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast16_t filterHSize);
		void createStructuringElement(boost::uint_fast16_t **element, boost::uint_fast16_t filterHSize);
        boost::uint_fast16_t initFilterHSize;
        boost::uint_fast16_t maxFilterHSize;
        float terrainSlope;
        float initElevDiff;
        float maxElevDiff;
        float grdPtDev;
        bool medianFilter;
        boost::uint_fast16_t medianFilterHSize;
        boost::uint_fast16_t classParameters;
	};

}



#endif



