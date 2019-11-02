/*
 *  SPDClassifyPts.h
 *
 *  Created by Pete Bunting on 30/04/2014.
 *  Copyright 2014 SPDLib. All rights reserved.
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

#ifndef SPDClassifyPts_H
#define SPDClassifyPts_H

#include <iostream>
#include <string>
#include <list>
#include "math.h"

#include "spd/SPDFile.h"
#include "spd/SPDPoint.h"
#include "spd/SPDPulse.h"
#include "spd/SPDCommon.h"
#include "spd/SPDProcessPulses.h"
#include "spd/SPDPulseProcessor.h"
#include "spd/SPDProcessingException.h"

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
	   
    class DllExport SPDClassifyPtsNumReturns : public SPDDataBlockProcessor
	{
	public:

        SPDClassifyPtsNumReturns();
        
        void processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands, float binSize) 
		{throw SPDProcessingException("SPDClassifyPts processDataBlockImage not implemented.");};
        void processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binSize) ;
        void processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands) 
		{throw SPDProcessingException("SPDClassifyPts requires processing with a grid.");};
        void processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses) 
        {throw SPDProcessingException("SPDClassifyPts requires processing with a grid.");};
        
        std::vector<std::string> getImageBandDescriptions() 
        {
            std::vector<std::string> bandNames;
            return bandNames;
        }
        void setHeaderValues(SPDFile *spdFile) 
        {
            // Nothing to do...
        }
        
        ~SPDClassifyPtsNumReturns();
	};
    
    
    
    
}


#endif


