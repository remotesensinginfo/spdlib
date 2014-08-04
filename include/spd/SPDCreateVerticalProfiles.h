/*
 *  SPDCreateVerticalProfiles.h
 *  SPDLIB
 *
 *  Created by Pete Bunting on 31/01/2013.
 *  Copyright 2013 SPDLib. All rights reserved.
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


#ifndef SPDCreateVerticalProfiles_H
#define SPDCreateVerticalProfiles_H

#include <iostream>
#include <string>
#include <vector>

#include <boost/math/special_functions/fpclassify.hpp>

#include "spd/SPDFile.h"
#include "spd/SPDPulse.h"
#include "spd/SPDProcessPulses.h"
#include "spd/SPDPulseProcessor.h"
#include "spd/SPDProcessingException.h"
#include "spd/SPDMathsUtils.h"

namespace spdlib
{
	
    class DllExport SPDCreateVerticalProfiles : public SPDPulseProcessor
	{
	public:
        SPDCreateVerticalProfiles(bool useSmoothing, boost::uint_fast32_t smoothWindowSize, boost::uint_fast32_t smoothPolyOrder, boost::uint_fast32_t maxProfileHeight, boost::uint_fast32_t numOfBins, float minPtHeight);
        
        void processDataColumnImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float *imageData, SPDXYPoint *cenPts, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException);
		void processDataColumn(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, SPDXYPoint *cenPts) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing is not implemented for processDataColumn().");};
        void processDataWindowImage(SPDFile *inSPDFile, bool **validBins, std::vector<SPDPulse*> ***pulses, float ***imageData, SPDXYPoint ***cenPts, boost::uint_fast32_t numImgBands, float binSize, boost::uint_fast16_t winSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing using a window is not implemented.");};
		void processDataWindow(SPDFile *inSPDFile, bool **validBins, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast16_t winSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing using a window is not implemented.");};
        
        std::vector<std::string> getImageBandDescriptions() throw(SPDProcessingException)
        {return std::vector<std::string>();};
        void setHeaderValues(SPDFile *spdFile) throw(SPDProcessingException){};
        
        ~SPDCreateVerticalProfiles();
    protected:
        bool useSmoothing;
        boost::uint_fast32_t smoothWindowSize;
        boost::uint_fast32_t smoothPolyOrder;
        boost::uint_fast32_t maxProfileHeight;
        boost::uint_fast32_t numOfBins;
        float binWidth;
        float *binHeightValues;
        float minPtHeight;
	};
}

#endif




