/*
 *  SPDRasterInterpolation.cpp
 *
 *  Created by Pete Bunting on 05/03/2012.
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

#include "spd/SPDRasterInterpolation.h"



namespace spdlib
{

    SPDDTMInterpolation::SPDDTMInterpolation(SPDPointInterpolator *interpolator)
    {
        this->interpolator = interpolator;
    }
        
    void SPDDTMInterpolation::processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException)
    {        
        try 
		{   
			if(numImgBands <= 0)
			{
				throw SPDProcessingException("The output image needs to have at least 1 image band.");
			}
            
            bool ptsAvail = true;
            try 
            {
                interpolator->initInterpolator(pulses, xSize, ySize, SPD_GROUND);
            }
            catch (SPDException &e) 
            {
                ptsAvail = false;
            }
            
            int feedback = ySize/10.0;
            int feedbackCounter = 0;
            std::cout << "Started" << std::flush;
            if(ptsAvail)
            {                
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
                        imageDataBlock[i][j][0] = interpolator->getValue(cenPts[i][j]->x, cenPts[i][j]->y);
                    }
                }
                
            }
            else
            {
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
                        imageDataBlock[i][j][0] = std::numeric_limits<float>::signaling_NaN();
                    }
                }
            }
			std::cout << " Complete.\n";
			interpolator->resetInterpolator();
            
		}
		catch (SPDProcessingException &e) 
		{
			throw e;
		}
        
    }
		
    SPDDTMInterpolation::~SPDDTMInterpolation()
    {
        
    }
    
    
    
    
    SPDDSMInterpolation::SPDDSMInterpolation(SPDPointInterpolator *interpolator)
    {
        this->interpolator = interpolator;
    }
    
    void SPDDSMInterpolation::processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException)
    {
        try 
		{            
			if(numImgBands <= 0)
			{
				throw SPDProcessingException("The output image needs to have at least 1 image band.");
			}
			
            bool ptsAvail = true;
            try
            {
                interpolator->initInterpolator(pulses, xSize, ySize, SPD_ALL_CLASSES_TOP);
            }
            catch (SPDException &e) 
            {
                ptsAvail = false;
            }
			
            int feedback = ySize/10.0;
            int feedbackCounter = 0;
            std::cout << "Started" << std::flush;
            
            if(ptsAvail)
            {
                
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
                        imageDataBlock[i][j][0] = interpolator->getValue(cenPts[i][j]->x, cenPts[i][j]->y);
                    }
                }
            }
            else
            {
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
                        imageDataBlock[i][j][0] = std::numeric_limits<float>::signaling_NaN();
                    }
                }
            }
            std::cout << " Complete.\n";
			
			interpolator->resetInterpolator();
		}
		catch (SPDProcessingException &e) 
		{
			throw e;
		}
    }
    
    SPDDSMInterpolation::~SPDDSMInterpolation()
    {
        
    }
    
    
    
    
    
    SPDCHMInterpolation::SPDCHMInterpolation(SPDPointInterpolator *interpolator)
    {
        this->interpolator = interpolator;
    }
    
    void SPDCHMInterpolation::processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException)
    {
        try 
		{            
			if(numImgBands <= 0)
			{
				throw SPDProcessingException("The output image needs to have at least 1 image band.");
			}
			
            bool ptsAvail = true;
            try
            {
                interpolator->initInterpolator(pulses, xSize, ySize, SPD_VEGETATION_TOP);
			}
            catch (SPDException &e) 
            {
                ptsAvail = false;
            }
			
            int feedback = ySize/10.0;
            int feedbackCounter = 0;
            std::cout << "Started" << std::flush;
            if(ptsAvail)
            {
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
                        imageDataBlock[i][j][0] = interpolator->getValue(cenPts[i][j]->x, cenPts[i][j]->y);
                    }
                }
            }
            else
            {
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
                        imageDataBlock[i][j][0] = std::numeric_limits<float>::signaling_NaN();
                    }
                }
            }
            std::cout << " Complete.\n";
			
			interpolator->resetInterpolator();
		}
		catch (SPDProcessingException &e) 
		{
			throw e;
		}
    }
    
    SPDCHMInterpolation::~SPDCHMInterpolation()
    {
        
    }
    
    
    SPDRangeInterpolation::SPDRangeInterpolation(SPDPointInterpolator *interpolator)
    {
        
    }
        
    void SPDRangeInterpolation::processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException)
    {
        try 
		{
			if(numImgBands != 1)
			{
				throw SPDProcessingException("The output image needs to have at least 1 image band.");
			}
            
            bool ptsAvail = true;
            try
            {
                interpolator->initInterpolator(pulses, xSize, ySize, SPD_ALL_CLASSES);
            }
            catch (SPDException &e) 
            {
                ptsAvail = false;
            }
			
            int feedback = ySize/10.0;
            int feedbackCounter = 0;
            std::cout << "Started" << std::flush;
            if(ptsAvail)
            {
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
                        imageDataBlock[i][j][0] = interpolator->getValue(cenPts[i][j]->x, cenPts[i][j]->y);
                    }
                }
            }
            else
            {
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
                        imageDataBlock[i][j][0] = std::numeric_limits<float>::signaling_NaN();
                    }
                }
            }
			std::cout << " Complete.\n";
            
			interpolator->resetInterpolator();
		}
		catch (SPDProcessingException &e) 
		{
			throw e;
		}
    }
    
    SPDRangeInterpolation::~SPDRangeInterpolation()
    {
        
    }

    
}





