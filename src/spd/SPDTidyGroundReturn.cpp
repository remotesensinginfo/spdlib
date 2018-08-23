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
    
    
    
    SPDTidyGroundReturnsPlaneFitting::SPDTidyGroundReturnsPlaneFitting() : SPDPulseProcessor()
    {
        
    }
    
    void SPDTidyGroundReturnsPlaneFitting::processDataWindow(SPDFile *inSPDFile, bool **validBins, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast16_t winSize) throw(SPDProcessingException)
    {
        try
        {
            SPDMathsUtils mathUtils;
            
            boost::uint_fast16_t winHSize = (winSize-1)/2;
            
            std::vector<SPDPoint*> *grdPts = new std::vector<SPDPoint*>();
            for(boost::uint_fast16_t i = 0; i < winSize; ++i)
            {
                for(boost::uint_fast16_t j = 0; j < winSize; ++j)
                {
                    if(validBins[i][j])
                    {
                        for(std::vector<SPDPulse*>::iterator iterPls = pulses[i][j]->begin(); iterPls != pulses[i][j]->end(); ++iterPls)
                        {
                            if((*iterPls)->numberOfReturns > 0)
                            {
                                for(std::vector<SPDPoint*>::iterator iterPts = (*iterPls)->pts->begin(); iterPts != (*iterPls)->pts->end(); ++iterPts)
                                {
                                    if((*iterPts)->classification == SPD_GROUND)
                                    {
                                        grdPts->push_back((*iterPts));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            boost::uint_fast32_t numGrdPts = grdPts->size();
            if(numGrdPts > 3)
            {
                SPDMathsUtils mathUtils;
                double *x = new double[numGrdPts];
                double *y = new double[numGrdPts];
                double *z = new double[numGrdPts];
                
                double normX = cenPts[winHSize][winHSize]->x;
                double normY = cenPts[winHSize][winHSize]->y;
                
                double a = 0;
                double b = 0;
                double c = 0;
                
                for(boost::uint_fast32_t i = 0; i < numGrdPts; ++i)
                {
                    x[i] = grdPts->at(i)->x;
                    y[i] = grdPts->at(i)->y;
                    z[i] = grdPts->at(i)->z;
                }
                mathUtils.fitPlane(x, y, z, numGrdPts, normX, normY, &a, &b, &c);
                std::cout << "Plane: a = " << a << " b = " << b << " c = " << c << std::endl;
                mathUtils.devFromPlane(x, y, z, numGrdPts, normX, normY, a, b, c);
                
                std::cout << "WARNING: SPDTidyGroundReturnsPlaneFitting implementation is not COMPLETE!!!!!!!!\n";
                
                delete[] x;
                delete[] y;
                delete[] z;
                
            }
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
    }
    
    SPDTidyGroundReturnsPlaneFitting::~SPDTidyGroundReturnsPlaneFitting()
    {
        
    }
    
}






