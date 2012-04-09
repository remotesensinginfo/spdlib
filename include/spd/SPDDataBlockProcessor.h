/*
 *  SPDDataBlockProcessor.h
 *
 *  Created by Pete Bunting on 11/03/2012.
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

#ifndef SPDDataBlockProcessor_H
#define SPDDataBlockProcessor_H

#include <iostream>
#include <string>
#include <list>

#include "spd/SPDFile.h"
#include "spd/SPDPoint.h"
#include "spd/SPDPulse.h"
#include "spd/SPDProcessingException.h"

using namespace std;

namespace spdlib
{
    struct SPDXYPoint
    {
        SPDXYPoint()
        {
            x = 0;
            y = 0;
        };
        SPDXYPoint(double x, double y)
        {
            this->x = x;
            this->y = y;
        };
        double x;
        double y;
    };
	
	class SPDDataBlockProcessor
	{
	public:
        virtual void processDataBlockImage(SPDFile *inSPDFile, vector<SPDPulse*> ***pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException)=0;
		virtual void processDataBlock(SPDFile *inSPDFile, vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize) throw(SPDProcessingException)=0;
        
        virtual void processDataBlockImage(SPDFile *inSPDFile, vector<SPDPulse*> *pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands) throw(SPDProcessingException)=0;
		virtual void processDataBlock(SPDFile *inSPDFile, vector<SPDPulse*> *pulses) throw(SPDProcessingException)=0;
        
        virtual vector<string> getImageBandDescriptions() throw(SPDProcessingException) = 0;
        virtual void setHeaderValues(SPDFile *spdFile) throw(SPDProcessingException) = 0;
        virtual ~SPDDataBlockProcessor(){};
	};
    
    class SPDDataBlockProcessorBlank : public SPDDataBlockProcessor
	{
	public:
        SPDDataBlockProcessorBlank():SPDDataBlockProcessor(){};
        void processDataBlockImage(SPDFile *inSPDFile, vector<SPDPulse*> ***pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException){};
		void processDataBlock(SPDFile *inSPDFile, vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize) throw(SPDProcessingException){};
        
        void processDataBlockImage(SPDFile *inSPDFile, vector<SPDPulse*> *pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands) throw(SPDProcessingException){};
		void processDataBlock(SPDFile *inSPDFile, vector<SPDPulse*> *pulses) throw(SPDProcessingException){};
        
        vector<string> getImageBandDescriptions() throw(SPDProcessingException){return vector<string>();};
        void setHeaderValues(SPDFile *spdFile) throw(SPDProcessingException){};
        ~SPDDataBlockProcessorBlank(){};
	};
}

#endif



