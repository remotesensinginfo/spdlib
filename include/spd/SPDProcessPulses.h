/*
 *  SPDProcessPulses.h
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

#ifndef SPDProcessPulses_H
#define SPDProcessPulses_H

#include <iostream>
#include <string>
#include <vector>

#include "spd/SPDFile.h"
#include "spd/SPDPoint.h"
#include "spd/SPDPulse.h"
#include "spd/SPDProcessDataBlocks.h"
#include "spd/SPDDataBlockProcessor.h"
#include "spd/SPDProcessingException.h"
#include "spd/SPDPulseProcessor.h"

namespace spdlib
{
    class SPDProcessPulses : public SPDDataBlockProcessor
	{
	public:
        SPDProcessPulses(SPDPulseProcessor *pulseProcessor, bool usingWindow, boost::uint_fast16_t winHSize);
        
        void processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException);
		void processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binSize) throw(SPDProcessingException);
        
        void processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands) throw(SPDProcessingException);
		void processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses) throw(SPDProcessingException);
        
        std::vector<std::string> getImageBandDescriptions() throw(SPDProcessingException);
        void setHeaderValues(SPDFile *spdFile) throw(SPDProcessingException);
        
        ~SPDProcessPulses();
    protected:
        SPDPulseProcessor *pulseProcessor;
        bool usingWindow;
        boost::uint_fast16_t winHSize;
	};
    
    class SPDSetupProcessPulses
    {
    public:
        SPDSetupProcessPulses(boost::uint_fast32_t blockXSize=250, boost::uint_fast32_t blockYSize=250, bool printProgress=true);
        void processPulsesWithInputImage(SPDPulseProcessor *pulseProcessor, SPDFile *spdInFile, std::string outFile, std::string imageFilePath, bool usingWindow=false, boost::uint_fast16_t winHSize=0) throw(SPDProcessingException);
        void processPulsesWithOutputImage(SPDPulseProcessor *pulseProcessor, SPDFile *spdInFile, std::string outImagePath, boost::uint_fast16_t numImgBands, float processingResolution=0, std::string gdalFormat="ENVI", bool usingWindow=false, boost::uint_fast16_t winHSize=0) throw(SPDProcessingException);
        void processPulsesWithOutputSPD(SPDPulseProcessor *pulseProcessor, SPDFile *spdInFile, std::string outFile, float processingResolution=0, bool usingWindow=false, boost::uint_fast16_t winHSize=0) throw(SPDProcessingException);
        void processPulses(SPDPulseProcessor *pulseProcessor, SPDFile *spdInFile, float processingResolution=0, bool usingWindow=false, boost::uint_fast16_t winHSize=0) throw(SPDProcessingException);
        ~SPDSetupProcessPulses();
    protected:
        boost::uint_fast32_t blockXSize;
        boost::uint_fast32_t blockYSize;
        bool printProgress;
    };    
}

#endif



