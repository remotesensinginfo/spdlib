/*
 *  SPDDefinePulseHeights.cpp
 *
 *  Created by Pete Bunting on 06/03/2012.
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

#include "spd/SPDDefinePulseHeights.h"

namespace spdlib
{

    SPDDefinePulseHeights::SPDDefinePulseHeights(SPDPointInterpolator *interpolator) : SPDDataBlockProcessor()
    {
        this->interpolator = interpolator;
    }
        
    void SPDDefinePulseHeights::processDataBlockImage(SPDFile *inSPDFile, vector<SPDPulse*> ***pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException)
    {
        if(numImgBands > 0)
        {
            throw SPDProcessingException("The input image needs to have at least 1 image band.");
        }
        
        for(boost::uint_fast32_t i = 0; i < ySize; ++i)
        {
            for(boost::uint_fast32_t j = 0; j < xSize; ++j)
            {
                for(vector<SPDPulse*>::iterator iterPulses = pulses[i][j]->begin(); iterPulses != pulses[i][j]->end(); ++iterPulses)
                {
                    (*iterPulses)->h0 = (*iterPulses)->z0 - imageDataBlock[i][j][0];
                    for(vector<SPDPoint*>::iterator iterPts = (*iterPulses)->pts->begin(); iterPts != (*iterPulses)->pts->end(); ++iterPts)
                    {
                        (*iterPts)->height = (*iterPts)->z - imageDataBlock[i][j][0];
                    }
                }
            }
        }
    }
		
    void SPDDefinePulseHeights::processDataBlock(SPDFile *inSPDFile, vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binSize) throw(SPDProcessingException)
    {
        try 
		{
            bool ptsAvail = true;
            try 
            {
                interpolator->initInterpolator(pulses, xSize, ySize, SPD_GROUND);
            }
            catch (SPDException &e) 
            {
                ptsAvail = false;
            }
            
            if(ptsAvail)
            {
                double elevVal = 0;
                
                for(boost::uint_fast32_t i = 0; i < ySize; ++i)
                {
                    for(boost::uint_fast32_t j = 0; j < xSize; ++j)
                    {
                        for(vector<SPDPulse*>::iterator iterPulses = pulses[i][j]->begin(); iterPulses != pulses[i][j]->end(); ++iterPulses)
                        {
                            elevVal = interpolator->getValue((*iterPulses)->x0, (*iterPulses)->y0);
                            (*iterPulses)->h0 = (*iterPulses)->z0 - elevVal;
                            for(vector<SPDPoint*>::iterator iterPts = (*iterPulses)->pts->begin(); iterPts != (*iterPulses)->pts->end(); ++iterPts)
                            {
                                elevVal = interpolator->getValue((*iterPts)->x, (*iterPts)->y);
                                (*iterPts)->height = (*iterPts)->z - elevVal;
                            }
                        }
                    }
                }
            }
            
            interpolator->resetInterpolator();
        }
        catch (SPDProcessingException &e) 
        {
            throw e;
        }
    }
             
    SPDDefinePulseHeights::~SPDDefinePulseHeights()
    {
        
    }
    
}






