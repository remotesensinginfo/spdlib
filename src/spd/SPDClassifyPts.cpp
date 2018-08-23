/*
 *  SPDClassifyPts.cpp
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

#include "spd/SPDClassifyPts.h"


namespace spdlib
{
    
    SPDClassifyPtsNumReturns::SPDClassifyPtsNumReturns(): SPDDataBlockProcessor()
    {
        
    }
        
    void SPDClassifyPtsNumReturns::processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binSize) throw(SPDProcessingException)
    {
        try
        {
            int feedback = ySize/10.0;
            int feedbackCounter = 0;
            std::cout << "Started" << std::flush;
            boost::uint_fast32_t sumNumReturns = 0;
            boost::uint_fast32_t numPulses = 0;
            float meanNumReturns = 0.0;
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
                
                for(boost::uint_fast32_t j = 0; j < xSize; ++j)
                {
                    sumNumReturns = 0;
                    numPulses = 0;
                    for(std::vector<SPDPulse*>::iterator iterPulses = pulses[i][j]->begin(); iterPulses != pulses[i][j]->end(); ++iterPulses)
                    {
                        sumNumReturns += (*iterPulses)->numberOfReturns;
                        ++numPulses;
                    }
                    if(numPulses > 0)
                    {
                        meanNumReturns = (float)sumNumReturns / (float)numPulses;
                        //std::cout << "MeanNumReturns = " << meanNumReturns << std::endl;
                        
                        for(std::vector<SPDPulse*>::iterator iterPulses = pulses[i][j]->begin(); iterPulses != pulses[i][j]->end(); ++iterPulses)
                        {
                            if((*iterPulses)->numberOfReturns > 0)
                            {
                                for(std::vector<SPDPoint*>::iterator iterPts = (*iterPulses)->pts->begin(); iterPts != (*iterPulses)->pts->end(); ++iterPts)
                                {
                                    if(((*iterPts)->height > 0.25) & ((*iterPts)->classification != SPD_GROUND))
                                    {
                                        if(meanNumReturns > 1)
                                        {
                                            // Then vegetation.
                                            if((*iterPts)->height > 20)
                                            {
                                                (*iterPts)->classification = SPD_HIGH_VEGETATION;
                                            }
                                            else if((*iterPts)->height > 5)
                                            {
                                                (*iterPts)->classification = SPD_MEDIUM_VEGETATION;
                                            }
                                            else
                                            {
                                                (*iterPts)->classification = SPD_LOW_VEGETATION;
                                            }
                                        }
                                        else if((*iterPts)->height > 1) // Create than 1 m to try and avoid grass etc..
                                        {
                                            // If above ground it will be a hard surface - building.
                                            (*iterPts)->classification = SPD_BUILDING;
                                        }
                                    }
                                }
                            }
                            
                            
                            
                        }
                    }
                }
            }
            std::cout << " Complete.\n";
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
    }
    
    SPDClassifyPtsNumReturns::~SPDClassifyPtsNumReturns()
    {
        
    }
    
    
}



