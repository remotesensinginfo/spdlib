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
    
    
    
    SPDFindRGBValuesStats::SPDFindRGBValuesStats()
    {
        this->redMean = 0;
        this->redStdDev = 0;
        this->redMin = 0;
        this->redMax = 0;
        
        this->greenMean = 0;
        this->greenStdDev = 0;
        this->greenMin = 0;
        this->greenMax = 0;
        
        this->blueMean = 0;
        this->blueStdDev = 0;
        this->blueMin = 0;
        this->blueMax = 0;
        
        this->calcStdDev = false;
        this->first = true;
        this->countPts = 0;
    }
    
    void SPDFindRGBValuesStats::processDataColumn(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, SPDXYPoint *cenPts) throw(SPDProcessingException)
    {
        for(std::vector<SPDPulse*>::iterator iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
        {
            if((*iterPulses)->numberOfReturns > 0)
            {
                for(std::vector<SPDPoint*>::iterator iterPts = (*iterPulses)->pts->begin(); iterPts != (*iterPulses)->pts->end(); ++iterPts)
                {
                    if(first)
                    {
                        if(!calcStdDev)
                        {
                            this->redMean = (*iterPts)->red;
                            this->redMin = (*iterPts)->red;
                            this->redMax = (*iterPts)->red;
                            
                            this->greenMean = (*iterPts)->green;
                            this->greenMin = (*iterPts)->green;
                            this->greenMax = (*iterPts)->green;
                            
                            this->blueMean = (*iterPts)->blue;
                            this->blueMin = (*iterPts)->blue;
                            this->blueMax = (*iterPts)->blue;
                        }
                        else
                        {
                            this->redStdDev = pow((*iterPts)->red - this->redMean, 2);
                            this->greenStdDev = pow((*iterPts)->green - this->greenMean, 2);
                            this->blueStdDev = pow((*iterPts)->blue - this->blueMean, 2);
                        }
                        first = false;
                    }
                    else
                    {
                        if(!calcStdDev)
                        {
                            this->redMean += (*iterPts)->red;
                            if((*iterPts)->red < this->redMin)
                            {
                                this->redMin = (*iterPts)->red;
                            }
                            else if((*iterPts)->red > this->redMax)
                            {
                                this->redMax = (*iterPts)->red;
                            }
                            
                            this->greenMean += (*iterPts)->green;
                            if((*iterPts)->green < this->greenMin)
                            {
                                this->greenMin = (*iterPts)->green;
                            }
                            else if((*iterPts)->green > this->greenMin)
                            {
                                this->greenMax = (*iterPts)->green;
                            }
                            
                            this->blueMean += (*iterPts)->blue;
                            if((*iterPts)->blue < this->blueMin)
                            {
                                this->blueMin = (*iterPts)->blue;
                            }
                            else if((*iterPts)->blue > this->blueMax)
                            {
                                this->blueMax = (*iterPts)->blue;
                            }
                        }
                        else
                        {
                            this->redStdDev += pow((*iterPts)->red - this->redMean, 2);
                            this->greenStdDev += pow((*iterPts)->green - this->greenMean, 2);
                            this->blueStdDev += pow((*iterPts)->blue - this->blueMean, 2);
                        }
                    }
                    
                    ++this->countPts;
                }
            }
            
            
        }
    }

    SPDFindRGBValuesStats::~SPDFindRGBValuesStats()
    {
        
    }
    
    
    
    SPDLinearStretchRGBValues::SPDLinearStretchRGBValues(float redMin, float redMax, float greenMin, float greenMax, float blueMin, float blueMax)
    {
        this->redMin = redMin;
        this->redMax = redMax;
        this->greenMin = greenMin;
        this->greenMax = greenMax;
        this->blueMin = blueMin;
        this->blueMax = blueMax;
        
        this->redRange = redMax - redMin;
        this->greenRange = greenMax - greenMin;
        this->blueRange = blueMax - blueMin;
    }
    
    void SPDLinearStretchRGBValues::processDataColumn(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, SPDXYPoint *cenPts) throw(SPDProcessingException)
    {
        for(std::vector<SPDPulse*>::iterator iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
        {
            if((*iterPulses)->numberOfReturns > 0)
            {
                for(std::vector<SPDPoint*>::iterator iterPts = (*iterPulses)->pts->begin(); iterPts != (*iterPulses)->pts->end(); ++iterPts)
                {
                    if((*iterPts)->red < redMin)
                    {
                        (*iterPts)->red = 0;
                    }
                    else if((*iterPts)->red > redMax)
                    {
                        (*iterPts)->red = 255;
                    }
                    else
                    {
                        (*iterPts)->red = (((*iterPts)->red-redMin)/redRange)*255;
                    }
                    
                    if((*iterPts)->green < greenMin)
                    {
                        (*iterPts)->green = 0;
                    }
                    else if((*iterPts)->green > greenMax)
                    {
                        (*iterPts)->green = 255;
                    }
                    else
                    {
                        (*iterPts)->green = (((*iterPts)->green-greenMin)/greenRange)*255;
                    }
                    
                    if((*iterPts)->blue < blueMin)
                    {
                        (*iterPts)->blue = 0;
                    }
                    else if((*iterPts)->blue > blueMax)
                    {
                        (*iterPts)->blue = 255;
                    }
                    else
                    {
                        (*iterPts)->blue = (((*iterPts)->blue-blueMin)/blueRange)*255;
                    }                    
                }
            }
            
            
        }
    }
    
        
    SPDLinearStretchRGBValues::~SPDLinearStretchRGBValues()
    {
        
    }

}


