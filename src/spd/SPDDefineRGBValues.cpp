/*
 *  SPDDefineRGBValues.cpp
 *  SPDLIB
 *
 *  Created by Pete Bunting on 10/03/2011.
 *  Copyright 2010 SPDLib. All rights reserved.
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

#include "spd/SPDDefineRGBValues.h"


namespace spdlib
{


    SPDDefineRGBValues::SPDDefineRGBValues(boost::uint_fast16_t redBand, boost::uint_fast16_t greenBand, boost::uint_fast16_t blueBand)
    {
        this->redBand = redBand;
        this->greenBand = greenBand;
        this->blueBand = blueBand;
    }
        
    void SPDDefineRGBValues::processDataColumnImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float *imageData, SPDXYPoint *cenPts, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException)
    {
        if((inSPDFile->getDecomposedPtDefined() == SPD_TRUE) | (inSPDFile->getDiscretePtDefined() == SPD_TRUE))
        {
            if(redBand >= numImgBands)
            {
                throw SPDProcessingException("Defined Red band is not in the dataset");
            }
            if(greenBand >= numImgBands)
            {
                throw SPDProcessingException("Defined Green band is not in the dataset");
            }
            if(blueBand >= numImgBands)
            {
                throw SPDProcessingException("Defined Blue band is not in the dataset");
            }
            
            std::vector<SPDPulse*>::iterator iterPulses;
            std::vector<SPDPoint*>::iterator iterPoints;
            for(iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
            {
                if((*iterPulses)->numberOfReturns > 0)
                {
                    for(iterPoints = (*iterPulses)->pts->begin(); iterPoints != (*iterPulses)->pts->end(); ++iterPoints)
                    {
                        (*iterPoints)->red = imageData[redBand];
                        (*iterPoints)->green = imageData[greenBand];
                        (*iterPoints)->blue = imageData[blueBand];
                    }
                }
            }
        }
        else
        {
            throw SPDProcessingException("You can only define RGB values on to points (i.e., not waveform data) decompose the data first.");
        }
    }
		        
    SPDDefineRGBValues::~SPDDefineRGBValues()
    {
        
    }

}


