/*
 *  SPDDefineClassesFromImg.cpp
 *  SPDLIB
 *
 *  Created by Pete Bunting on 03/11/2016.
 *  Copyright 2016 SPDLib. All rights reserved.
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

#include "spd/SPDDefineClassesFromImg.h"


namespace spdlib
{


    SPDDefineClassesFromImg::SPDDefineClassesFromImg(boost::uint_fast16_t classBand)
    {
        this->classBand = classBand;
    }

    void SPDDefineClassesFromImg::processDataColumnImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float *imageData, SPDXYPoint *cenPts, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException)
    {
        if((inSPDFile->getDecomposedPtDefined() == SPD_TRUE) | (inSPDFile->getDiscretePtDefined() == SPD_TRUE))
        {
            if(classBand >= numImgBands)
            {
                throw SPDProcessingException("Defined classes band is not in the dataset");
            }

            if(imageData[classBand] > 0)
            {
                std::vector<SPDPulse*>::iterator iterPulses;
                std::vector<SPDPoint*>::iterator iterPoints;
                for(iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
                {
                    if((*iterPulses)->numberOfReturns > 0)
                    {
                        for(iterPoints = (*iterPulses)->pts->begin(); iterPoints != (*iterPulses)->pts->end(); ++iterPoints)
                        {
                            (*iterPoints)->classification = boost::numeric_cast<boost::uint_fast16_t>(imageData[classBand]);
                        }
                    }
                }
            }
        }
        else
        {
            throw SPDProcessingException("You can only define a classification on to points (i.e., not waveform data), decompose the data first.");
        }
    }
		
    SPDDefineClassesFromImg::~SPDDefineClassesFromImg()
    {

    }

}


