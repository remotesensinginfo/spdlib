/*
 *  SPDPulseProcessor.h
 *
 *  Created by Pete Bunting on 27/03/2012.
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

#ifndef SPDPulseProcessor_H
#define SPDPulseProcessor_H

#include <iostream>
#include <string>
#include <vector>

#include <boost/cstdint.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

#include "spd/SPDFile.h"
#include "spd/SPDPoint.h"
#include "spd/SPDPulse.h"
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
	
	class DllExport SPDPulseProcessor
	{
	public:
        SPDPulseProcessor(){};
        
        virtual void processDataColumnImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float *imageData, SPDXYPoint *cenPts, boost::uint_fast32_t numImgBands, float binSize) =0;
		virtual void processDataColumn(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, SPDXYPoint *cenPts) =0;
        
        virtual void processDataWindowImage(SPDFile *inSPDFile, bool **validBins, std::vector<SPDPulse*> ***pulses, float ***imageData, SPDXYPoint ***cenPts, boost::uint_fast32_t numImgBands, float binSize, boost::uint_fast16_t winSize) =0;
		virtual void processDataWindow(SPDFile *inSPDFile, bool **validBins, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast16_t winSize) =0;
        
        virtual std::vector<std::string> getImageBandDescriptions()  = 0;
        virtual void setHeaderValues(SPDFile *spdFile)  = 0;
                
        virtual ~SPDPulseProcessor(){};
	};

}

#endif



