/*
 *  SPDDefinePulseHeights.h
 *
 *  Created by Pete Bunting on 06/03/2012.
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

#ifndef SPDDefinePulseHeights_H
#define SPDDefinePulseHeights_H

#include <iostream>
#include <string>
#include <list>

#include <boost/cstdint.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

#include "spd/SPDCommon.h"
#include "spd/SPDFile.h"
#include "spd/SPDPoint.h"
#include "spd/SPDPulse.h"
#include "spd/SPDProcessingException.h"
#include "spd/SPDDataBlockProcessor.h"
#include "spd/SPDPointInterpolation.h"

#include "boost/math/special_functions/fpclassify.hpp"

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
    
    class DllExport SPDDefinePulseHeights : public SPDDataBlockProcessor
	{
	public:
        SPDDefinePulseHeights(SPDPointInterpolator *interpolator);
        
        void processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands, float binSize) ;
        void processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binSize) ;
        void processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands) 
		{throw SPDProcessingException("SPDDTMInterpolation requires processing with a grid.");};
        
        void processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses) 
        {throw SPDProcessingException("SPDDTMInterpolation requires processing with a grid.");};
        
        std::vector<std::string> getImageBandDescriptions() 
        {
            std::vector<std::string> bandNames;
            return bandNames;
        }
        void setHeaderValues(SPDFile *spdFile) 
        {
            spdFile->setHeightDefined(SPD_TRUE);
        }
        
        ~SPDDefinePulseHeights();
    protected:
        SPDPointInterpolator *interpolator;
	};
    
}

#endif



