/*
 *  SPDTidyGroundReturn.cpp
 *
 *  Created by Pete Bunting on 05/11/2013.
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

#include "spd/SPDTidyGroundReturn.h"

namespace spdlib
{
    
    SPDTidyGroundReturnNegativeHeights::SPDTidyGroundReturnNegativeHeights() : SPDDataBlockProcessor()
    {

    }
    
    void SPDTidyGroundReturnNegativeHeights::processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binSize) throw(SPDProcessingException)
    {
        try
		{
            int feedback = ySize/10.0;
            int feedbackCounter = 0;
            std::cout << "Started" << std::flush;
            for(boost::uint_fast32_t i = 0; i < ySize; ++i)
            {
                if(ySize < 10)
                {
                    std::cout << "." << i << "." << std::flush;
                }
                else if((feedback != 0) && ((i % feedback) == 0))
                {
                    std::cout << "." << feedbackCounter << "." << std::flush;
                    feedbackCounter = feedbackCounter + 10;
                }
                
                std::vector<SPDPoint*> *grdPts = new std::vector<SPDPoint*>();
                
                for(boost::uint_fast32_t j = 0; j < xSize; ++j)
                {
                    for(std::vector<SPDPulse*>::iterator iterPulses = pulses[i][j]->begin(); iterPulses != pulses[i][j]->end(); ++iterPulses)
                    {
                        if((*iterPulses)->numberOfReturns > 0)
                        {
                            for(std::vector<SPDPoint*>::iterator iterPts = (*iterPulses)->pts->begin(); iterPts != (*iterPulses)->pts->end(); ++iterPts)
                            {
                                if((*iterPts)->height < 0)
                                {
                                    (*iterPts)->classification = SPD_GROUND;
                                }
                                
                                if((*iterPts)->classification == SPD_GROUND)
                                {
                                    grdPts->push_back((*iterPts));
                                }
                            }
                        }
                    }
                }
                
                std::cout << "There are " << grdPts->size() << " ground returns.\n";
                
                delete grdPts;
                
            }
            std::cout << " Complete.\n";
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
    }
    
    SPDTidyGroundReturnNegativeHeights::~SPDTidyGroundReturnNegativeHeights()
    {
        
    }
    
}






