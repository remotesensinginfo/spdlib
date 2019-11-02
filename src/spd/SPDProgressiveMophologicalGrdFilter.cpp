/*
 *  SPDProgressiveMophologicalGrdFilter.cpp
 *
 *  Created by Pete Bunting on 03/03/2012.
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

#include "spd/SPDProgressiveMophologicalGrdFilter.h"

namespace spdlib
{
    
    SPDProgressiveMophologicalGrdFilter::SPDProgressiveMophologicalGrdFilter(boost::uint_fast16_t initFilterHSize, boost::uint_fast16_t maxFilterHSize, float terrainSlope, float initElevDiff, float maxElevDiff, float grdPtDev, bool medianFilter, boost::uint_fast16_t medianFilterHSize, boost::uint_fast16_t classParameters)
    {
        this->initFilterHSize = initFilterHSize;
        this->maxFilterHSize = maxFilterHSize;
        this->terrainSlope = terrainSlope;
        this->initElevDiff = initElevDiff;
        this->maxElevDiff = maxElevDiff;
        this->grdPtDev = grdPtDev;
        this->medianFilter = medianFilter;
        this->medianFilterHSize = medianFilterHSize;
        this->classParameters = classParameters;
    }
    
    void SPDProgressiveMophologicalGrdFilter::processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands, float binSize) 
    {
        if(numImgBands < 1)
		{
			throw SPDProcessingException("The number of output image bands has to be at least 1.");
		}
		
		float **elev = new float*[ySize];
		float **elevErode = new float*[ySize];
		float **elevDialate = new float*[ySize];
		boost::uint_fast16_t ***changeFlag = new boost::uint_fast16_t**[ySize];
		
		// Allocate Memory...
		for(boost::uint_fast32_t i = 0; i < ySize; ++i)
		{
			elev[i] = new float[xSize];
			elevErode[i] = new float[xSize];
			elevDialate[i] = new float[xSize];
			changeFlag[i] = new boost::uint_fast16_t*[xSize];
			for(boost::uint_fast32_t j = 0; j < xSize; ++j)
			{
				elev[i][j] = std::numeric_limits<float>::signaling_NaN();
				elevErode[i][j] = std::numeric_limits<float>::signaling_NaN();
				elevDialate[i][j] = std::numeric_limits<float>::signaling_NaN();
				imageDataBlock[0][i][j] = std::numeric_limits<float>::signaling_NaN();
				changeFlag[i][j] = new boost::uint_fast16_t[2];
				changeFlag[i][j][0] = 0;
				changeFlag[i][j][1] = 0;
			}
		}		
		
		// Define the minimum surface grid
		this->findMinSurface(pulses, elev, xSize, ySize);
		//this->fillHolesInMinSurface(elevTmp, elev, xSize, ySize, 2);
		
		for(boost::uint_fast32_t i = 0; i < ySize; ++i)
		{
			for(boost::uint_fast32_t j = 0; j < xSize; ++j)
			{
				imageDataBlock[0][i][j] = elev[i][j];
			}
		}
        
		float elevDiffThreshold = initElevDiff;
		
		boost::uint_fast16_t **element = NULL;
        boost::int_fast32_t filterSize = 0;
		
		// Apply the progressive morphological filter
		for(boost::uint_fast16_t filterHSize = initFilterHSize; filterHSize <= maxFilterHSize; ++filterHSize)
		{
			filterSize = (filterHSize * 2)+1;
			//std::cout << "Filter Size: " << filterHSize << " threshold = " << elevDiffThreshold << std::endl;
			
			element = new boost::uint_fast16_t*[filterSize];
			for(boost::uint_fast16_t i = 0; i < filterSize; ++i)
			{
				element[i] = new boost::uint_fast16_t[filterSize];
			}
			
			this->createStructuringElement(element, filterHSize);
			
			// Perform Erosion
			this->performErosion(elev, elevErode, xSize, ySize, filterHSize, element);
			
			// Perform Dialation
			this->performDialation(elevErode, elevDialate, xSize, ySize, filterHSize, element);
			
			// Decide whether to keep value...
			for(boost::uint_fast32_t i = 0; i < ySize; ++i)
			{
				for(boost::uint_fast32_t j = 0; j < xSize; ++j)
				{
					if((imageDataBlock[0][i][j] - elevDialate[i][j]) > elevDiffThreshold)
					{
						imageDataBlock[0][i][j] = elevDialate[i][j];
						changeFlag[i][j][0] = 1;
						changeFlag[i][j][1] = filterHSize;
					}
					elev[i][j] = elevDialate[i][j];
				}
			}
			
			elevDiffThreshold = (terrainSlope * filterHSize * binSize) + initElevDiff;
			if(elevDiffThreshold > maxElevDiff)
			{
				elevDiffThreshold = maxElevDiff;
			}			
			
			for(boost::uint_fast16_t i = 0; i < filterSize; ++i)
			{
				delete[] element[i];
			}
			delete[] element;
		}
		
		if(medianFilter)
		{
			for(boost::uint_fast32_t i = 0; i < ySize; ++i)
			{
				for(boost::uint_fast32_t j = 0; j < xSize; ++j)
				{
					elev[i][j] = imageDataBlock[0][i][j];
				}
			}
			
			filterSize = (medianFilterHSize * 2)+1;
			element = new boost::uint_fast16_t*[filterSize];
			for(boost::uint_fast16_t i = 0; i < filterSize; ++i)
			{
				element[i] = new boost::uint_fast16_t[filterSize];
			}
			this->createStructuringElement(element, medianFilterHSize);
			
			this->applyMedianFilter(elev, imageDataBlock[0], xSize, ySize, medianFilterHSize, element);
			
			for(boost::uint_fast16_t i = 0; i < filterSize; ++i)
			{
				delete[] element[i];
			}
			delete[] element;
		}
        
		// Clean up memory
		for(boost::uint_fast32_t i = 0; i < ySize; ++i)
		{
			delete[] elev[i];
			delete[] elevErode[i];
			delete[] elevDialate[i];
			for(boost::uint_fast32_t j = 0; j < xSize; ++j)
			{
				delete[] changeFlag[i][j];
			}
			delete[] changeFlag[i];
		}
		delete[] elev;
		delete[] elevErode;
		delete[] elevDialate;
		delete[] changeFlag;
    }
    
    void SPDProgressiveMophologicalGrdFilter::processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binSize) 
    {
		float **elev = new float*[ySize];
		float **elevFinal = new float*[ySize];
		float **elevErode = new float*[ySize];
		float **elevDialate = new float*[ySize];
		boost::uint_fast16_t ***changeFlag = new boost::uint_fast16_t**[ySize];
		
		// Allocate Memory...
		for(boost::uint_fast32_t i = 0; i < ySize; ++i)
		{
			elev[i] = new float[xSize];
			elevErode[i] = new float[xSize];
			elevDialate[i] = new float[xSize];
			changeFlag[i] = new boost::uint_fast16_t*[xSize];
			elevFinal[i] = new float[xSize];
			for(boost::uint_fast32_t j = 0; j < xSize; ++j)
			{
				elev[i][j] = std::numeric_limits<float>::signaling_NaN();
				elevFinal[i][j] = std::numeric_limits<float>::signaling_NaN();
				elevErode[i][j] = std::numeric_limits<float>::signaling_NaN();
				elevDialate[i][j] = std::numeric_limits<float>::signaling_NaN();
				changeFlag[i][j] = new boost::uint_fast16_t[2];
				changeFlag[i][j][0] = 0;
				changeFlag[i][j][1] = 0;
			}
		}		
		
		// Define the minimum surface grid
		this->findMinSurface(pulses, elev, xSize, ySize);
		//this->fillHolesInMinSurface(elevTmp, elev, xSize, ySize, 2);
		
		for(boost::uint_fast32_t i = 0; i < ySize; ++i)
		{
			for(boost::uint_fast32_t j = 0; j < xSize; ++j)
			{
				elevFinal[i][j] = elev[i][j];
			}
		}
		
		float elevDiffThreshold = initElevDiff;
		
		boost::uint_fast16_t **element = NULL;
        boost::int_fast32_t filterSize = 0;
		
		// Apply the progressive morphological filter
		for(boost::uint_fast16_t filterHSize = initFilterHSize; filterHSize <= maxFilterHSize; ++filterHSize)
		{
			filterSize = (filterHSize * 2)+1;
			//std::cout << "Filter Size: " << filterHSize << " threshold = " << elevDiffThreshold << std::endl;
			
			element = new boost::uint_fast16_t*[filterSize];
			for(boost::uint_fast16_t i = 0; i < filterSize; ++i)
			{
				element[i] = new boost::uint_fast16_t[filterSize];
			}
			
			this->createStructuringElement(element, filterHSize);
			
			// Perform Erosion
			this->performErosion(elev, elevErode, xSize, ySize, filterHSize, element);
			
			// Perform Dialation
			this->performDialation(elevErode, elevDialate, xSize, ySize, filterHSize, element);
			
			// Decide whether to keep value...
			for(boost::uint_fast32_t i = 0; i < ySize; ++i)
			{
				for(boost::uint_fast32_t j = 0; j < xSize; ++j)
				{
					if((elevFinal[i][j] - elevDialate[i][j]) > elevDiffThreshold)
					{
						elevFinal[i][j] = elevDialate[i][j];
						changeFlag[i][j][0] = 1;
						changeFlag[i][j][1] = filterHSize;
					}
					elev[i][j] = elevDialate[i][j];
				}
			}
			
			elevDiffThreshold = (terrainSlope * filterHSize * binSize) + initElevDiff;
			if(elevDiffThreshold > maxElevDiff)
			{
				elevDiffThreshold = maxElevDiff;
			}			
			
			for(boost::uint_fast16_t i = 0; i < filterSize; ++i)
			{
				delete[] element[i];
			}
			delete[] element;
		}
		
		if(medianFilter)
		{
			for(boost::uint_fast32_t i = 0; i < ySize; ++i)
			{
				for(boost::uint_fast32_t j = 0; j < xSize; ++j)
				{
					elev[i][j] = elevFinal[i][j];
				}
			}
			
			filterSize = (medianFilterHSize * 2)+1;
			element = new boost::uint_fast16_t*[filterSize];
			for(boost::uint_fast16_t i = 0; i < filterSize; ++i)
			{
				element[i] = new boost::uint_fast16_t[filterSize];
			}
			this->createStructuringElement(element, medianFilterHSize);
			
			this->applyMedianFilter(elev, elevFinal, xSize, ySize, medianFilterHSize, element);
			
			for(boost::uint_fast16_t i = 0; i < filterSize; ++i)
			{
				delete[] element[i];
			}
			delete[] element;
		}
		
		std::vector<SPDPulse*>::iterator iterPulses;
		std::vector<SPDPoint*>::iterator iterPoints;
		
		// Classifiy Ground returns
		for(boost::uint_fast32_t i = 0; i < ySize; ++i)
		{
			for(boost::uint_fast32_t j = 0; j < xSize; ++j)
			{
				if(pulses[i][j]->size() > 0)
				{
					for(iterPulses = pulses[i][j]->begin(); iterPulses != pulses[i][j]->end(); ++iterPulses)
					{
						if((*iterPulses)->numberOfReturns > 0)
						{
							for(iterPoints = (*iterPulses)->pts->begin(); iterPoints != (*iterPulses)->pts->end(); ++iterPoints)
							{
                                if((*iterPoints)->classification == SPD_GROUND)
                                {
                                    (*iterPoints)->classification = SPD_UNCLASSIFIED;
                                }
                                
								if(fabs((*iterPoints)->z - elevFinal[i][j]) < grdPtDev)
								{
									(*iterPoints)->classification = SPD_GROUND;
								}
							}
						}
					}
				}
			}
		}
		
		// Clean up memory
		for(boost::uint_fast32_t i = 0; i < ySize; ++i)
		{
			delete[] elev[i];
			delete[] elevErode[i];
			delete[] elevDialate[i];
			delete[] elevFinal[i];
			for(boost::uint_fast32_t j = 0; j < xSize; ++j)
			{
				delete[] changeFlag[i][j];
			}
			delete[] changeFlag[i];
		}
		delete[] elev;
		delete[] elevErode;
		delete[] elevDialate;
		delete[] elevFinal;
		delete[] changeFlag;
    }
    
	void SPDProgressiveMophologicalGrdFilter::performErosion(float **elev, float **elevErode, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast16_t filterHSize, boost::uint_fast16_t **element)
	{
		boost::uint_fast32_t filterPxlStartX = 0;
		boost::uint_fast32_t filterPxlEndX = 0;
		boost::uint_fast32_t filterPxlStartY = 0;
		boost::uint_fast32_t filterPxlEndY = 0;
		
		boost::uint_fast32_t elementStartX = 0;
		boost::uint_fast32_t elementEndX = 0;
		boost::uint_fast32_t elementStartY = 0;
		boost::uint_fast32_t elementEndY = 0;
		
        boost::int_fast32_t filterSize = (filterHSize * 2)+1;
		boost::uint_fast32_t maxNumFilterValues = ((filterHSize * 2)+1)*((filterHSize * 2)+1);
		bool first = true;
		float minVal = 0;
		
		std::vector<float> *elevValues = new std::vector<float>();
		std::vector<float>::iterator iterVals;
		elevValues->reserve(maxNumFilterValues);
		
		for(boost::uint_fast32_t i = 0; i < ySize; ++i)
		{
			for(boost::uint_fast32_t j = 0; j < xSize; ++j)
			{
				if(!boost::math::isnan(elev[i][j]))
				{
					if((((int_fast64_t)i) - ((int_fast64_t)filterHSize)) < 0)
					{
						filterPxlStartY = 0;
						filterPxlEndY = i + filterHSize;
						elementStartY = filterHSize - i;
						elementEndY = filterSize;
					}
					else if((i+filterHSize) >= ySize)
					{
						filterPxlStartY = i - filterHSize;
						filterPxlEndY = (ySize-1);
						elementStartY = 0;
						elementEndY = filterSize - ((i+filterHSize)-ySize);
					}
					else
					{
						filterPxlStartY = i - filterHSize;
						filterPxlEndY = i + filterHSize;
						elementStartY = 0;
						elementEndY = filterSize;
					}
					
					if((((int_fast64_t)j) - ((int_fast64_t)filterHSize)) < 0)
					{
						filterPxlStartX = 0;
						filterPxlEndX = j + filterHSize;
						elementStartX = filterHSize - j;
						elementEndX = filterSize;
						
					}
					else if((j+filterHSize) >= xSize)
					{
						filterPxlStartX = j - filterHSize;
						filterPxlEndX = xSize-1;
						elementStartX = 0;
						elementEndX = filterSize - ((j+filterHSize)-xSize);
					}
					else
					{
						filterPxlStartX = j - filterHSize;
						filterPxlEndX = j + filterHSize;
						elementStartX = 0;
						elementEndX = filterSize;
					}
					
					//std::cout << "Filter [" << j << "," << i << "] [" << filterPxlStartX << "," << filterPxlEndX << "][" << filterPxlStartY << "," << filterPxlEndY << "]\n\n";
					
					elevValues->clear();
					
					for(boost::uint_fast32_t n = filterPxlStartY, eY = elementStartY; n <= filterPxlEndY; ++n, ++eY)
					{
						for(boost::uint_fast32_t m = filterPxlStartX, eX = elementStartX; m <= filterPxlEndX; ++m, ++eX)
						{
							if((element[eY][eX] == 1) & (!boost::math::isnan(elev[n][m])))
							{
								elevValues->push_back(elev[n][m]);
							}
						}
					}
					
					if(elevValues->size() > 1)
					{
						first = true;
						for(iterVals = elevValues->begin(); iterVals != elevValues->end(); ++iterVals)
						{
							if(first)
							{
								minVal = *iterVals;
								first = false;
							}
							else if((*iterVals) < minVal)
							{
								minVal = *iterVals;
							}
						}
						elevErode[i][j] = minVal;
					}
					else if(elevValues->size() == 1)
					{
						elevErode[i][j] = elevValues->front();
					}
					else 
					{
						elevErode[i][j] = std::numeric_limits<float>::signaling_NaN();
					}
				}
				else 
				{
					elevErode[i][j] = std::numeric_limits<float>::signaling_NaN();
				}
                
				
			}
		}
		delete elevValues;
	}
	
	void SPDProgressiveMophologicalGrdFilter::performDialation(float **elev, float **elevDialate, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast16_t filterHSize, boost::uint_fast16_t **element)
	{
		boost::uint_fast32_t filterPxlStartX = 0;
		boost::uint_fast32_t filterPxlEndX = 0;
		boost::uint_fast32_t filterPxlStartY = 0;
		boost::uint_fast32_t filterPxlEndY = 0;
		
		boost::uint_fast32_t elementStartX = 0;
		boost::uint_fast32_t elementEndX = 0;
		boost::uint_fast32_t elementStartY = 0;
		boost::uint_fast32_t elementEndY = 0;
		
        boost::int_fast32_t filterSize = (filterHSize * 2)+1;
		boost::uint_fast32_t maxNumFilterValues = ((filterHSize * 2)+1)*((filterHSize * 2)+1);
		bool first = true;
		float maxVal = 0;
		
		std::vector<float> *elevValues = new std::vector<float>();
		std::vector<float>::iterator iterVals;
		elevValues->reserve(maxNumFilterValues);
		
		for(boost::uint_fast32_t i = 0; i < ySize; ++i)
		{
			for(boost::uint_fast32_t j = 0; j < xSize; ++j)
			{
				if(!boost::math::isnan(elev[i][j]))
				{
					if((((int_fast64_t)i) - ((int_fast64_t)filterHSize)) < 0)
					{
						filterPxlStartY = 0;
						filterPxlEndY = i + filterHSize;
						elementStartY = filterHSize - i;
						elementEndY = filterSize;
					}
					else if((i+filterHSize) >= ySize)
					{
						filterPxlStartY = i - filterHSize;
						filterPxlEndY = (ySize-1);
						elementStartY = 0;
						elementEndY = filterSize - ((i+filterHSize)-ySize);
					}
					else
					{
						filterPxlStartY = i - filterHSize;
						filterPxlEndY = i + filterHSize;
						elementStartY = 0;
						elementEndY = filterSize;
					}
					
					if((((int_fast64_t)j) - ((int_fast64_t)filterHSize)) < 0)
					{
						filterPxlStartX = 0;
						filterPxlEndX = j + filterHSize;
						elementStartX = filterHSize - j;
						elementEndX = filterSize;
						
					}
					else if((j+filterHSize) >= xSize)
					{
						filterPxlStartX = j - filterHSize;
						filterPxlEndX = xSize-1;
						elementStartX = 0;
						elementEndX = filterSize - ((j+filterHSize)-xSize);
					}
					else
					{
						filterPxlStartX = j - filterHSize;
						filterPxlEndX = j + filterHSize;
						elementStartX = 0;
						elementEndX = filterSize;
					}
					
					//std::cout << "Filter [" << j << "," << i << "] [" << filterPxlStartX << "," << filterPxlEndX << "][" << filterPxlStartY << "," << filterPxlEndY << "]\n\n";
					
					elevValues->clear();
					
					for(boost::uint_fast32_t n = filterPxlStartY, eY = elementStartY; n <= filterPxlEndY; ++n, ++eY)
					{
						for(boost::uint_fast32_t m = filterPxlStartX, eX = elementStartX; m <= filterPxlEndX; ++m, ++eX)
						{
							if((element[eY][eX] == 1) & (!boost::math::isnan(elev[n][m])))
							{
								elevValues->push_back(elev[n][m]);
							}
						}
					}
					
					if(elevValues->size() > 1)
					{
						first = true;
						for(iterVals = elevValues->begin(); iterVals != elevValues->end(); ++iterVals)
						{
							if(first)
							{
								maxVal = *iterVals;
								first = false;
							}
							else if((*iterVals) > maxVal)
							{
								maxVal = *iterVals;
							}
						}
						elevDialate[i][j] = maxVal;
					}
					else if(elevValues->size() == 1)
					{
						elevDialate[i][j] = elevValues->front();
					}
					else 
					{
						elevDialate[i][j] = std::numeric_limits<float>::signaling_NaN();
					}
				}
				else 
				{
					elevDialate[i][j] = std::numeric_limits<float>::signaling_NaN();
				}
			}
		}
		delete elevValues;
	}
	
	void SPDProgressiveMophologicalGrdFilter::findMinSurface(std::vector<SPDPulse*> ***pulses, float **elev, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize)
	{
		std::vector<SPDPulse*>::iterator iterPulses;
		std::vector<SPDPoint*>::iterator iterPoints;
		SPDPoint *pt = NULL;
		bool firstPls = true;
		bool firstPts = true;
		
		for(boost::uint_fast32_t i = 0; i < ySize; ++i)
		{
			for(boost::uint_fast32_t j = 0; j < xSize; ++j)
			{
				firstPls = true;
				if(pulses[i][j]->size() > 0)
				{
					//std::cout << "\nBlock [" << i << "," << j << "] has " << pulses[i][j]->size() << " pulses\n";
					for(iterPulses = pulses[i][j]->begin(); iterPulses != pulses[i][j]->end(); ++iterPulses)
					{
						if((*iterPulses)->numberOfReturns > 0)
						{
							firstPts = true;
							pt = NULL;
							for(iterPoints = (*iterPulses)->pts->begin(); iterPoints != (*iterPulses)->pts->end(); ++iterPoints)
							{
								//std::cout << (*iterPoints)->z << std::endl;
                                if(classParameters == SPD_ALL_CLASSES)
                                {
                                    if(firstPts)
                                    {
                                        pt = *iterPoints;
                                        firstPts = false;
                                    }
                                    else if((*iterPoints)->z < pt->z)
                                    {
                                        pt = *iterPoints;
                                    }
                                }
								else
                                {
                                    if((*iterPoints)->classification == classParameters)
                                    {
                                        if(firstPts)
                                        {
                                            pt = *iterPoints;
                                            firstPts = false;
                                        }
                                        else if((*iterPoints)->z < pt->z)
                                        {
                                            pt = *iterPoints;
                                        }
                                    }
                                }
							}
							if(!firstPts)
                            {
                                if(firstPls)
                                {
                                    elev[i][j] = pt->z;
                                    firstPls = false;
                                }
                                else if(pt->z < elev[i][j])
                                {
                                    elev[i][j] = pt->z;
                                }
                            }
						}
					}
					//std::cout << "Min = " << elev[i][j] << std::endl;
				}
				else
				{
					elev[i][j] = std::numeric_limits<float>::signaling_NaN();
				}
			}
		}
	}
	
	void SPDProgressiveMophologicalGrdFilter::applyMedianFilter(float **elev, float **elevMedian, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast16_t filterHSize, boost::uint_fast16_t **element)
	{
		boost::uint_fast32_t filterPxlStartX = 0;
		boost::uint_fast32_t filterPxlEndX = 0;
		boost::uint_fast32_t filterPxlStartY = 0;
		boost::uint_fast32_t filterPxlEndY = 0;
		
		boost::uint_fast32_t elementStartX = 0;
		boost::uint_fast32_t elementEndX = 0;
		boost::uint_fast32_t elementStartY = 0;
		boost::uint_fast32_t elementEndY = 0;
		
        boost::int_fast32_t filterSize = (filterHSize * 2)+1;
		boost::uint_fast32_t maxNumFilterValues = ((filterHSize * 2)+1)*((filterHSize * 2)+1);
		
		std::vector<float> *elevValues = new std::vector<float>();
		elevValues->reserve(maxNumFilterValues);
		for(boost::uint_fast32_t i = 0; i < ySize; ++i)
		{
			for(boost::uint_fast32_t j = 0; j < xSize; ++j)
			{
				if((((int_fast64_t)i) - ((int_fast64_t)filterHSize)) < 0)
				{
					filterPxlStartY = 0;
					filterPxlEndY = i + filterHSize;
					elementStartY = filterHSize - i;
					elementEndY = filterSize;
				}
				else if((i+filterHSize) >= ySize)
				{
					filterPxlStartY = i - filterHSize;
					filterPxlEndY = (ySize-1);
					elementStartY = 0;
					elementEndY = filterSize - ((i+filterHSize)-ySize);
				}
				else
				{
					filterPxlStartY = i - filterHSize;
					filterPxlEndY = i + filterHSize;
					elementStartY = 0;
					elementEndY = filterSize;
				}
				
				if((((int_fast64_t)j) - ((int_fast64_t)filterHSize)) < 0)
				{
					filterPxlStartX = 0;
					filterPxlEndX = j + filterHSize;
					elementStartX = filterHSize - j;
					elementEndX = filterSize;
					
				}
				else if((j+filterHSize) >= xSize)
				{
					filterPxlStartX = j - filterHSize;
					filterPxlEndX = xSize-1;
					elementStartX = 0;
					elementEndX = filterSize - ((j+filterHSize)-xSize);
				}
				else
				{
					filterPxlStartX = j - filterHSize;
					filterPxlEndX = j + filterHSize;
					elementStartX = 0;
					elementEndX = filterSize;
				}
				
				//std::cout << "Filter [" << j << "," << i << "] [" << filterPxlStartX << "," << filterPxlEndX << "][" << filterPxlStartY << "," << filterPxlEndY << "]\n\n";
				
				elevValues->clear();
				
				for(boost::uint_fast32_t n = filterPxlStartY, eY = elementStartY; n <= filterPxlEndY; ++n, ++eY)
				{
					for(boost::uint_fast32_t m = filterPxlStartX, eX = elementStartX; m <= filterPxlEndX; ++m, ++eX)
					{
						if((element[eY][eX] == 1) & (!boost::math::isnan(elev[n][m])))
						{
							elevValues->push_back(elev[n][m]);
						}
					}
				}
				
				if(elevValues->size() == 1)
				{
					elevMedian[i][j] = elevValues->at(0);
				}
				else if(elevValues->size() > 1)
				{
                    std::sort(elevValues->begin(), elevValues->end());
					elevMedian[i][j] = elevValues->at(elevValues->size()/2);
				}
				else
				{
					elevMedian[i][j] = std::numeric_limits<float>::signaling_NaN();
				}
			}
		}
		delete elevValues;
	}
	
	void SPDProgressiveMophologicalGrdFilter::fillHolesInMinSurface(float **elev, float **elevOut, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast16_t filterHSize)
	{
		boost::uint_fast32_t filterPxlStartX = 0;
		boost::uint_fast32_t filterPxlEndX = 0;
		boost::uint_fast32_t filterPxlStartY = 0;
		boost::uint_fast32_t filterPxlEndY = 0;
		boost::uint_fast32_t maxNumFilterValues = ((filterHSize * 2)+1)*((filterHSize * 2)+1);
		bool first = true;
		float minVal = 0;
		
		std::vector<float> *elevValues = new std::vector<float>();
		std::vector<float>::iterator iterVals;
		elevValues->reserve(maxNumFilterValues);
		
		for(boost::uint_fast32_t i = 0; i < ySize; ++i)
		{
			for(boost::uint_fast32_t j = 0; j < xSize; ++j)
			{
				if(boost::math::isnan(elev[i][j]))
				{
					if((((int_fast64_t)i) - ((int_fast64_t)filterHSize)) < 0)
					{
						filterPxlStartY = 0;
						filterPxlEndY = i + filterHSize;
					}
					else if((i+filterHSize) >= ySize)
					{
						filterPxlStartY = i - filterHSize;
						filterPxlEndY = (ySize-1);
					}
					else
					{
						filterPxlStartY = i - filterHSize;
						filterPxlEndY = i + filterHSize;						
					}
					
					if((((int_fast64_t)j) - ((int_fast64_t)filterHSize)) < 0)
					{
						filterPxlStartX = 0;
						filterPxlEndX = j + filterHSize;
					}
					else if((j+filterHSize) >= xSize)
					{
						filterPxlStartX = j - filterHSize;
						filterPxlEndX = xSize-1;
					}
					else
					{
						filterPxlStartX = j - filterHSize;
						filterPxlEndX = j + filterHSize;
					}
					
					elevValues->clear();
					
					for(boost::uint_fast32_t n = filterPxlStartY; n < filterPxlEndY; ++n)
					{
						for(boost::uint_fast32_t m = filterPxlStartX; m < filterPxlEndX; ++m)
						{
							if(!boost::math::isnan(elev[n][m]))
							{
								elevValues->push_back(elev[n][m]);
							}
						}
					}
					
					if(elevValues->size() > 1)
					{
						first = true;
						for(iterVals = elevValues->begin(); iterVals != elevValues->end(); ++iterVals)
						{
							if(first)
							{
								minVal = *iterVals;
								first = false;
							}
							else if((*iterVals) < minVal)
							{
								minVal = *iterVals;
							}
						}
						elevOut[i][j] = minVal;
					}
					else if(elevValues->size() > 1)
					{
						elevOut[i][j] = elevValues->front();
					}
					else 
					{
						elevOut[i][j] = std::numeric_limits<float>::signaling_NaN();
					}
				}
				else 
				{
					elevOut[i][j] = elev[i][j];
				}
                
				
				
			}
		}
		delete elevValues;
	}
	
	void SPDProgressiveMophologicalGrdFilter::createStructuringElement(boost::uint_fast16_t **element, boost::uint_fast16_t filterHSize)
	{
        boost::int_fast32_t filterSize = (filterHSize * 2)+1;
		float xdiff = 0;
		float ydiff = 0;
		
		for(int_fast32_t i = 0; i < filterSize; ++i)
		{
			for(int_fast32_t j = 0; j < filterSize; ++j)
			{
				xdiff = pow(((double)j-filterHSize), 2);
				ydiff = pow(((double)i-filterHSize), 2);
				//std::cout << "radius [" << i << "," << j << "] = " << pow((xdiff + ydiff),2) << "\n";
				if(sqrt(xdiff + ydiff) <= filterHSize)
				{
					element[i][j] = 1;
				}
				else 
				{
					element[i][j] = 0;
				}
                
			}
		}
	}

    SPDProgressiveMophologicalGrdFilter::~SPDProgressiveMophologicalGrdFilter()
    {
        
    }
    
}




