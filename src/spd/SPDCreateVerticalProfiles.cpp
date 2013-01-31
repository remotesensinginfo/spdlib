/*
 *  SPDCreateVerticalProfiles.cpp
 *  SPDLIB
 *
 *  Created by Pete Bunting on 31/01/2013.
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


#include "spd/SPDCreateVerticalProfiles.h"

namespace spdlib
{
	

    SPDCreateVerticalProfiles::SPDCreateVerticalProfiles(bool useSmoothing, boost::uint_fast32_t smoothWindowSize, boost::uint_fast32_t smoothPolyOrder, boost::uint_fast32_t maxProfileHeight, boost::uint_fast32_t numOfBins, float minPtHeight)
    {
        this->useSmoothing = useSmoothing;
        this->smoothWindowSize = smoothWindowSize;
        this->smoothPolyOrder = smoothPolyOrder;
        this->maxProfileHeight = maxProfileHeight;
        this->numOfBins = numOfBins;
        this->minPtHeight = minPtHeight;
        this->binWidth = ((double)maxProfileHeight)/((double)numOfBins);
        this->binHeightValues = new float[numOfBins];
        float binHalfWidth = binWidth/2;
        for(boost::uint_fast32_t i = 0; i < numOfBins; ++i)
        {
            this->binHeightValues[i] = (i*binWidth)+binHalfWidth;
        }
    }
        
    void SPDCreateVerticalProfiles::processDataColumnImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float *imageData, SPDXYPoint *cenPts, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException)
    {
        if(this->numOfBins != numImgBands)
        {
            throw SPDProcessingException("The number of images bands is not equal to the number of required bins.");
        }
        
        for(boost::uint_fast32_t i = 0; i < numImgBands; ++i)
        {
            imageData[i] = 0.0;
        }
        
        
        if(pulses->size() > 0)
        {
            
            boost::int_fast32_t binIdx = 0;
            size_t numReturns = 0;
            
            std::vector<SPDPulse*>::iterator iterPulses;
            std::vector<SPDPoint*>::iterator iterPoints;

            for(iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
            {
                if((*iterPulses)->numberOfReturns > 0)
                {
                    for(iterPoints = (*iterPulses)->pts->begin(); iterPoints != (*iterPulses)->pts->end(); ++iterPoints)
                    {
                        // Check point is above the min height
                        if((*iterPoints)->height > minPtHeight)
                        {
                            // Identify the bin.
                            binIdx = floor((*iterPoints)->height/this->binWidth);
                            
                            if((binIdx >= 0) && (binIdx < this->numOfBins))
                            {
                                // Add 1 to the bin.
                                imageData[binIdx] += 1;
                                ++numReturns;
                            }
                        }
                    }
                }
            }
            
            if((numReturns > 0) && useSmoothing)
            {
                SPDMathsUtils mathUtils;
                mathUtils.applySavitzkyGolaySmoothing(imageData, this->binHeightValues, this->numOfBins, this->smoothWindowSize, this->smoothPolyOrder, true);
            }
        }
        
        
    }
		  
    SPDCreateVerticalProfiles::~SPDCreateVerticalProfiles()
    {
        delete[] this->binHeightValues;
    }

}




