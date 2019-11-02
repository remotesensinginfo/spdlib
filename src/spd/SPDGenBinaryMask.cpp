/*
 *  SPDGenBinaryMask.cpp
 *  SPDLIB
 *
 *  Created by Pete Bunting on 28/03/2012.
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

#include "spd/SPDGenBinaryMask.h"


namespace spdlib
{	
    
    SPDGenBinaryMask::SPDGenBinaryMask()
    {
        
    }
    
    void SPDGenBinaryMask::generateBinaryMask(boost::uint_fast32_t numPulses, std::string inputSPDFile, std::string outputImageFile, boost::uint_fast32_t blockXSize, boost::uint_fast32_t blockYSize, float processingResolution, std::string gdalFormat) 
    {
        try 
        {
            SPDFile *spdInFile = new SPDFile(inputSPDFile);
            SPDPulseProcessor *pulseStatsProcessor = new SPDPulseProcessorCalcMask(numPulses);            
            SPDSetupProcessPulses processPulses = SPDSetupProcessPulses(blockXSize, blockYSize, true);
            processPulses.processPulsesWithOutputImage(pulseStatsProcessor, spdInFile, outputImageFile, 1, processingResolution, gdalFormat, false, 0);
            
            delete spdInFile;
            delete pulseStatsProcessor;
        }
        catch (SPDProcessingException &e) 
        {
            throw e;
        }
    }
    
    SPDGenBinaryMask::~SPDGenBinaryMask()
    {
        
    }
    
    
    
    
    SPDPulseProcessorCalcMask::SPDPulseProcessorCalcMask(boost::uint_fast32_t numPulses):SPDPulseProcessor()
    {
        this->numPulses = numPulses;
    }
    
    void SPDPulseProcessorCalcMask::processDataColumnImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float *imageData, SPDXYPoint *cenPts, boost::uint_fast32_t numImgBands, float binSize) 
    {
        try
        {
            if(numImgBands < 1)
            {
                throw SPDProcessingException("Processing requires at least 1 image band.");
            }
            
            if(pulses->size() >= this->numPulses)
            {
                imageData[0] = 1;
            }
            else
            {
                imageData[0] = 0;
            }
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
    }
    
    std::vector<std::string> SPDPulseProcessorCalcMask::getImageBandDescriptions() 
    {
        std::vector<std::string> bandNames;
        bandNames.push_back("Mask");
        
        return bandNames;
    }
    
    void SPDPulseProcessorCalcMask::setHeaderValues(SPDFile *spdFile) 
    {
        // NOTHING TO DO HERE...
    }
    
    SPDPulseProcessorCalcMask::~SPDPulseProcessorCalcMask()
    {
        
    }    
    
}




