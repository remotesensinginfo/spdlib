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

    void SPDCreateVerticalProfiles::processDataColumnImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float *imageData, SPDXYPoint *cenPts, boost::uint_fast32_t numImgBands, float binSize) 
    {
        if(this->numOfBins != numImgBands)
        {
            throw SPDProcessingException("The number of images bands is not equal to the number of required bins.");
        }

        for(boost::uint_fast32_t i = 0; i < numImgBands; ++i)
        {
            imageData[i] = 0.0;
        }

        boost::int_fast32_t binIdx = 0;
        size_t numReturns = 0;

        std::vector<SPDPulse*>::iterator iterPulses;
        std::vector<SPDPoint*>::iterator iterPoints;

        if(inSPDFile->getReceiveWaveformDefined() == SPD_TRUE)
        {
            // If the recieved waveform is defined use this to generate the profiles
			double tmpX = 0;
			double tmpY = 0;
			double tmpH = 0;

            for(iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
            {
                for(unsigned int s = 0; s < (*iterPulses)->numOfReceivedBins; s++)
                {
                    // Get the DN of the digitised value and check
                    // this is above the noise threshold.
                    double pulseDNVal = (*iterPulses)->received[s];

                    if(pulseDNVal > (*iterPulses)->receiveWaveNoiseThreshold)
                    {
                        // Get the time of the digitised value, relative to the origin
                        double timeOffset = s * inSPDFile->getTemporalBinSpacing();

                        // Get the height of the digitised value using the height of the origin
                        // and the time offset within the pulse.
                        SPDConvertToCartesian((*iterPulses)->zenith, (*iterPulses)->azimuth, 
                                            (SPD_SPEED_OF_LIGHT_NS * timeOffset), (*iterPulses)->x0, (*iterPulses)->y0, (*iterPulses)->h0, &tmpX, &tmpY, &tmpH);

                        // Identify the bin.
                        binIdx = floor(tmpH/this->binWidth);

                        if((binIdx >= 0) && (binIdx < this->numOfBins))
                        {
                            // Add pulse amplitude value to the total for the bin.
                            imageData[binIdx] += pulseDNVal;
                            ++numReturns;
                        }
                    }
                }

            }
        }
        // If no waveform is defined use the points within each pulse
        else if(pulses->size() > 0)
        {
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
                                // Add return amplitude value to the total for the bin.
                                imageData[binIdx] += (*iterPoints)->amplitudeReturn;
                                ++numReturns;
                            }
                        }
                    }
                }
            }
        }

        // If more than one return (or waveform sample) and useSmooting
        // apply Savitzky-Golay filter
        if((numReturns > 0) && useSmoothing)
        {
            SPDMathsUtils mathUtils;
            mathUtils.applySavitzkyGolaySmoothing(imageData, this->binHeightValues, this->numOfBins, this->smoothWindowSize, this->smoothPolyOrder, true);
        }

    }

    std::vector<std::string> SPDCreateVerticalProfiles::getImageBandDescriptions() 
    {
        // Set the band names to the height in the middle of the bin, equal to (upper+lower) / 2
        std::vector<std::string> bandNames;
        for(boost::uint_fast32_t i = 0; i < numOfBins; ++i)
        {
            bandNames.push_back(boost::lexical_cast<std::string>(this->binHeightValues[i]) + " m");
        }
        return bandNames;
    }

    SPDCreateVerticalProfiles::~SPDCreateVerticalProfiles()
    {
        delete[] this->binHeightValues;
    }

}




