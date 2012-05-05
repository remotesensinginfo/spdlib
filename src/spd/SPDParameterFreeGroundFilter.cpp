/*
 *  SPDParameterFreeGroundFilter.cpp
 *
 *  Created by Pete Bunting on 04/05/2012.
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

#include "spd/SPDParameterFreeGroundFilter.h"


namespace spdlib
{

    SPDParameterFreeGroundFilter::SPDParameterFreeGroundFilter(float grdPtDev, boost::uint_fast16_t classParameters, bool checkForFalseMinma)
    {
        this->grdPtDev = grdPtDev;
        this->classParameters = classParameters;
        this->checkForFalseMinma = checkForFalseMinma;
        this->k = 3;
    }
    
    void SPDParameterFreeGroundFilter::processDataBlockImage(SPDFile *inSPDFile, vector<SPDPulse*> ***pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException)
    {
        // Allocate Memory...
        float **elev = new float*[ySize];
		for(boost::uint_fast32_t i = 0; i < ySize; ++i)
		{
			elev[i] = new float[xSize];
			for(boost::uint_fast32_t j = 0; j < xSize; ++j)
			{
				elev[i][j] = numeric_limits<float>::signaling_NaN();
			}
		}		
		
		// Define the inital minimum surface grid
		this->findMinSurface(pulses, elev, xSize, ySize);
        
        if(checkForFalseMinma)
        {
            float **elevOpen = new float*[ySize];
            float **elevClose = new float*[ySize];
            for(boost::uint_fast32_t i = 0; i < ySize; ++i)
            {
                elevOpen[i] = new float[xSize];
                elevClose[i] = new float[xSize];
                for(boost::uint_fast32_t j = 0; j < xSize; ++j)
                {
                    elevOpen[i][j] = numeric_limits<float>::signaling_NaN();
                    elevClose[i][j] = numeric_limits<float>::signaling_NaN();
                }
            }
            
            
            boost::uint_fast16_t **openElement9 = new boost::uint_fast16_t*[9];
            for(boost::uint_fast32_t i = 0; i < 9; ++i)
            {
                openElement9[i] = new boost::uint_fast16_t[9];
            }
            this->createStructuringElement(openElement9, 4);
            this->performOpenning(elev, elevOpen, xSize, ySize, 4, openElement9);
            for(boost::uint_fast32_t i = 0; i < 9; ++i)
            {
                delete[] openElement9[i];
            }
            delete[] openElement9;
            
            boost::uint_fast16_t **closeElement11 = new boost::uint_fast16_t*[11];
            for(boost::uint_fast32_t i = 0; i < 11; ++i)
            {
                closeElement11[i] = new boost::uint_fast16_t[11];
            }
            this->createStructuringElement(closeElement11, 5);
            this->performClosing(elevOpen, elevClose, xSize, ySize, 5, closeElement11);
            for(boost::uint_fast32_t i = 0; i < 11; ++i)
            {
                delete[] closeElement11[i];
            }
            delete[] closeElement11;
            
            for(boost::uint_fast32_t i = 0; i < ySize; ++i)
            {
                for(boost::uint_fast32_t j = 0; j < xSize; ++j)
                {
                    if((elevClose[i][j] - elev[i][j]) >= 1)
                    {
                        elev[i][j] = elevClose[i][j];
                    }
                }
            }
            
            for(boost::uint_fast32_t i = 0; i < ySize; ++i)
            {
                delete[] elevOpen[i];
                delete[] elevClose[i];
            }
            delete[] elevOpen;
            delete[] elevClose;
        }
        
        // Generate resolution hierarchy...
        vector<SPDPFFProcessLevel*> *elevLevels = this->generateHierarchy(elev, xSize, ySize, binSize);
        
        /*
        for(boost::int_fast16_t i = elevLevels->size()-1; i >= 0; --i)
        {
            SPDPFFProcessLevel *level = elevLevels->at(i);
            cout << "\n\nLevel  " << i << endl;
            for(boost::uint_fast32_t i = 0; i < level->ySize; ++i)
            {
                for(boost::uint_fast32_t j = 0; j < level->xSize; ++j)
                {
                    if(j == 0)
                    {
                        cout << level->data[i][j];
                    }
                    else 
                    {
                        cout << "," << level->data[i][j];
                    }
                }
                cout << endl;
            }
        }
        */
        
        
        // Prepare first level
        SPDPFFProcessLevel *cLevel = NULL;
        SPDPFFProcessLevel *prevLevel = NULL;
        SPDPFFProcessLevel *interpdLevel = NULL;
        
        cout << "Filtering Level " << elevLevels->size()-2 << endl;
        prevLevel = elevLevels->at(elevLevels->size()-1);
        cLevel = elevLevels->at(elevLevels->size()-2);
        double tlX = cenPts[0][0]->x - (binSize/2);
        double tlY = cenPts[0][0]->y + (binSize/2);
        // Interpolate level values
        interpdLevel = this->interpLevel(prevLevel, cLevel, tlY, tlX);
        
        // Decide on values which are to be taken forward...
        float **elevRes = new float*[cLevel->ySize];
        float **elevTH = new float*[cLevel->ySize];
        for(boost::uint_fast32_t i = 0; i < cLevel->ySize; ++i)
        {
            elevRes[i] = new float[cLevel->ySize];
            elevTH[i] = new float[cLevel->ySize];
            for(boost::uint_fast32_t j = 0; j < cLevel->xSize; ++j)
            {
                elevRes[i][j] = cLevel->data[i][j] - interpdLevel->data[i][j];
                elevTH[i][j] = numeric_limits<float>::signaling_NaN();
            }
        }
        boost::uint_fast16_t **wTHElem = new boost::uint_fast16_t*[3];
        for(boost::uint_fast32_t i = 0; i < 3; ++i)
        {
            wTHElem[i] = new boost::uint_fast16_t[3];
        }
        this->createStructuringElement(wTHElem, 1);
        
        this->performWhiteTopHat(elevRes, elevTH, cLevel->xSize, cLevel->ySize, 1, wTHElem);
                
        /*for(boost::uint_fast32_t i = 0; i < cLevel->ySize; ++i)
        {
            for(boost::uint_fast32_t j = 0; j < cLevel->xSize; ++j)
            {
                if(j == 0)
                {
                    cout << "[" << cLevel->data[i][j] << "," << interpdLevel->data[i][j] << "]";
                }
                else 
                {
                    cout << "," << "[" << cLevel->data[i][j] << "," << interpdLevel->data[i][j] << "]";
                }
            }
            cout << endl;
        }
        for(boost::uint_fast32_t i = 0; i < cLevel->ySize; ++i)
        {
            for(boost::uint_fast32_t j = 0; j < cLevel->xSize; ++j)
            {
                if(j == 0)
                {
                    cout << "[" << elevRes[i][j] << "," << elevTH[i][j] << "]";
                }
                else 
                {
                    cout << "," << "[" << elevRes[i][j] << "," << elevTH[i][j] << "]";
                }
            }
            cout << endl;
        }*/
        float threshold = 0;
        for(boost::uint_fast32_t i = 0; i < cLevel->ySize; ++i)
        {
            for(boost::uint_fast32_t j = 0; j < cLevel->xSize; ++j)
            {
                threshold = this->getThreshold(j, i, elevTH, cLevel->xSize, cLevel->ySize, 1, wTHElem);
                if((boost::math::isnan(cLevel->data[i][j]) | boost::math::isnan(threshold)) | (elevTH[i][j] < threshold))
                {
                    interpdLevel->data[i][j] = cLevel->data[i][j];
                }
                /*if(j == 0)
                {
                    cout << threshold;
                }
                else 
                {
                    cout << "," << threshold;
                }*/
            }
            //cout << endl;
        }
        for(boost::uint_fast32_t i = 0; i < 3; ++i)
        {
            delete[] wTHElem[i];
        }
        delete[] wTHElem;
        for(boost::uint_fast32_t i = 0; i < cLevel->ySize; ++i)
		{
			delete[] elevRes[i];
            delete[] elevTH[i];
		}
		delete[] elevRes;
        delete[] elevTH;
        
        // Copy to previous level variable for next level
        prevLevel = interpdLevel;
        
        if(elevLevels->size() > 2)
        {
            // Iterate through remaining levels
            boost::int_fast16_t numLevels = elevLevels->size();
            for(boost::int_fast16_t i = numLevels-3; i >= 0; --i)
            {
                cout << "Filtering Level " << i << endl;
                // Interpolate level values
                cLevel = elevLevels->at(i);
                interpdLevel = this->interpLevel(prevLevel, cLevel, tlY, tlX);
                
                // Decide on values which are to be taken forward...
                float **elevRes = new float*[cLevel->ySize];
                float **elevTH = new float*[cLevel->ySize];
                for(boost::uint_fast32_t i = 0; i < cLevel->ySize; ++i)
                {
                    elevRes[i] = new float[cLevel->ySize];
                    elevTH[i] = new float[cLevel->ySize];
                    for(boost::uint_fast32_t j = 0; j < cLevel->xSize; ++j)
                    {
                        elevRes[i][j] = cLevel->data[i][j] - interpdLevel->data[i][j];
                        elevTH[i][j] = numeric_limits<float>::signaling_NaN();
                    }
                }
                boost::uint_fast16_t **wTHElem = new boost::uint_fast16_t*[7];
                for(boost::uint_fast32_t i = 0; i < 7; ++i)
                {
                    wTHElem[i] = new boost::uint_fast16_t[7];
                }
                this->createStructuringElement(wTHElem, 3);
                
                this->performWhiteTopHat(elevRes, elevTH, cLevel->xSize, cLevel->ySize, 3, wTHElem);
                /*
                for(boost::uint_fast32_t i = 0; i < cLevel->ySize; ++i)
                {
                    for(boost::uint_fast32_t j = 0; j < cLevel->xSize; ++j)
                    {
                        if(j == 0)
                        {
                            cout << "[" << cLevel->data[i][j] << "," << interpdLevel->data[i][j] << "]";
                        }
                        else 
                        {
                            cout << "," << "[" << cLevel->data[i][j] << "," << interpdLevel->data[i][j] << "]";
                        }
                    }
                    cout << endl;
                }
                for(boost::uint_fast32_t i = 0; i < cLevel->ySize; ++i)
                {
                    for(boost::uint_fast32_t j = 0; j < cLevel->xSize; ++j)
                    {
                        if(j == 0)
                        {
                            cout << "[" << elevRes[i][j] << "," << elevTH[i][j] << "]";
                        }
                        else 
                        {
                            cout << "," << "[" << elevRes[i][j] << "," << elevTH[i][j] << "]";
                        }
                    }
                    cout << endl;
                }*/
                for(boost::uint_fast32_t i = 0; i < cLevel->ySize; ++i)
                {
                    for(boost::uint_fast32_t j = 0; j < cLevel->xSize; ++j)
                    {
                        threshold = this->getThreshold(j, i, elevTH, cLevel->xSize, cLevel->ySize, 3, wTHElem);
                        if((boost::math::isnan(cLevel->data[i][j]) | boost::math::isnan(threshold)) | (elevTH[i][j] < threshold))
                        {
                            interpdLevel->data[i][j] = cLevel->data[i][j];
                        }
                        /*if(j == 0)
                        {
                            cout << threshold;
                        }
                        else 
                        {
                            cout << "," << threshold;
                        }*/
                    }
                    //cout << endl;
                }
                for(boost::uint_fast32_t i = 0; i < 7; ++i)
                {
                    delete[] wTHElem[i];
                }
                delete[] wTHElem;
                for(boost::uint_fast32_t i = 0; i < cLevel->ySize; ++i)
                {
                    delete[] elevRes[i];
                    delete[] elevTH[i];
                }
                delete[] elevRes;
                delete[] elevTH;
                
                // Copy to previous level variable for next level and free previous level
                this->freeLevel(prevLevel);
                prevLevel = interpdLevel;
            }
        }
        
        for(boost::uint_fast32_t i = 0; i < ySize; ++i)
        {
            for(boost::uint_fast32_t j = 0; j < xSize; ++j)
            {
                if((i < prevLevel->ySize) & (j < prevLevel->xSize))
                {
                    imageDataBlock[i][j][0] = prevLevel->data[i][j];
                }
                else 
                {
                    imageDataBlock[i][j][0] = numeric_limits<float>::signaling_NaN();
                }
            }
        }
        this->freeLevel(prevLevel);
        
        // Clean up memory
        this->freeHierarchy(elevLevels);
		for(boost::uint_fast32_t i = 0; i < ySize; ++i)
		{
			delete[] elev[i];
		}
		delete[] elev;
    }
		
    void SPDParameterFreeGroundFilter::processDataBlock(SPDFile *inSPDFile, vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize) throw(SPDProcessingException)
    {
        
    }
    
    
    void SPDParameterFreeGroundFilter::findMinSurface(vector<SPDPulse*> ***pulses, float **elev, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize)
	{
		vector<SPDPulse*>::iterator iterPulses;
		vector<SPDPoint*>::iterator iterPoints;
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
					//cout << "\nBlock [" << i << "," << j << "] has " << pulses[i][j]->size() << " pulses\n";
					for(iterPulses = pulses[i][j]->begin(); iterPulses != pulses[i][j]->end(); ++iterPulses)
					{
						if((*iterPulses)->numberOfReturns > 0)
						{
							firstPts = true;
							pt = NULL;
							for(iterPoints = (*iterPulses)->pts->begin(); iterPoints != (*iterPulses)->pts->end(); ++iterPoints)
							{
								//cout << (*iterPoints)->z << endl;
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
					//cout << "Min = " << elev[i][j] << endl;
				}
				else
				{
					elev[i][j] = numeric_limits<float>::signaling_NaN();
				}
			}
		}
	}
    
    void SPDParameterFreeGroundFilter::performErosion(float **elev, float **elevErode, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast16_t filterHSize, boost::uint_fast16_t **element)
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
		
		vector<float> *elevValues = new vector<float>();
		vector<float>::iterator iterVals;
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
					
					//cout << "Filter [" << j << "," << i << "] [" << filterPxlStartX << "," << filterPxlEndX << "][" << filterPxlStartY << "," << filterPxlEndY << "]\n\n";
					
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
						elevErode[i][j] = numeric_limits<float>::signaling_NaN();
					}
				}
				else 
				{
					elevErode[i][j] = numeric_limits<float>::signaling_NaN();
				}
                
				
			}
		}
		delete elevValues;
	}
	
	void SPDParameterFreeGroundFilter::performDialation(float **elev, float **elevDialate, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast16_t filterHSize, boost::uint_fast16_t **element)
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
		
		vector<float> *elevValues = new vector<float>();
		vector<float>::iterator iterVals;
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
					
					//cout << "Filter [" << j << "," << i << "] [" << filterPxlStartX << "," << filterPxlEndX << "][" << filterPxlStartY << "," << filterPxlEndY << "]\n\n";
					
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
						elevDialate[i][j] = numeric_limits<float>::signaling_NaN();
					}
				}
				else 
				{
					elevDialate[i][j] = numeric_limits<float>::signaling_NaN();
				}
			}
		}
		delete elevValues;
	}
    
    void SPDParameterFreeGroundFilter::performOpenning(float **elev, float **elevOpen, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast16_t filterHSize, boost::uint_fast16_t **element)
    {
		// Allocate Memory...
        float **tmpElev = new float*[ySize];
		for(boost::uint_fast32_t i = 0; i < ySize; ++i)
		{
			tmpElev[i] = new float[xSize];
			for(boost::uint_fast32_t j = 0; j < xSize; ++j)
			{
				tmpElev[i][j] = numeric_limits<float>::signaling_NaN();
			}
		}
        
        this->performErosion(elev, tmpElev, xSize, ySize, filterHSize, element);
        this->performDialation(tmpElev, elevOpen, xSize, ySize, filterHSize, element);
        
        // Clean up memory
		for(boost::uint_fast32_t i = 0; i < ySize; ++i)
		{
			delete[] tmpElev[i];
		}
		delete[] tmpElev;
    }
    
    void SPDParameterFreeGroundFilter::performClosing(float **elev, float **elevClose, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast16_t filterHSize, boost::uint_fast16_t **element)
    {		
		// Allocate Memory...
        float **tmpElev = new float*[ySize];
		for(boost::uint_fast32_t i = 0; i < ySize; ++i)
		{
			tmpElev[i] = new float[xSize];
			for(boost::uint_fast32_t j = 0; j < xSize; ++j)
			{
				tmpElev[i][j] = numeric_limits<float>::signaling_NaN();
			}
		}
        
        this->performDialation(elev, tmpElev, xSize, ySize, filterHSize, element);
        this->performErosion(tmpElev, elevClose, xSize, ySize, filterHSize, element);
        
        // Clean up memory
		for(boost::uint_fast32_t i = 0; i < ySize; ++i)
		{
			delete[] tmpElev[i];
		}
		delete[] tmpElev;
    }
    
    void SPDParameterFreeGroundFilter::performWhiteTopHat(float **elev, float **elevTH, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast16_t filterHSize, boost::uint_fast16_t **element)
    {
        float **tmpElev = new float*[ySize];
        float **tmpOpenElev = new float*[ySize];
		for(boost::uint_fast32_t i = 0; i < ySize; ++i)
		{
			tmpElev[i] = new float[xSize];
            tmpOpenElev[i] = new float[xSize];
			for(boost::uint_fast32_t j = 0; j < xSize; ++j)
			{
				tmpElev[i][j] = numeric_limits<float>::signaling_NaN();
                tmpOpenElev[i][j] = numeric_limits<float>::signaling_NaN();
			}
		}
        
        this->performErosion(elev, tmpElev, xSize, ySize, filterHSize, element);
        this->performDialation(tmpElev, tmpOpenElev, xSize, ySize, filterHSize, element);
        
        for(boost::uint_fast32_t i = 0; i < ySize; ++i)
		{
			for(boost::uint_fast32_t j = 0; j < xSize; ++j)
			{
				elevTH[i][j] = elev[i][j] - tmpOpenElev[i][j];
			}
		}
        
        // Clean up memory
		for(boost::uint_fast32_t i = 0; i < ySize; ++i)
		{
			delete[] tmpElev[i];
            delete[] tmpOpenElev[i];
		}
		delete[] tmpElev;
        delete[] tmpOpenElev;
    }
    
    void SPDParameterFreeGroundFilter::createStructuringElement(boost::uint_fast16_t **element, boost::uint_fast16_t filterHSize)
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
				//cout << "radius [" << i << "," << j << "] = " << pow((xdiff + ydiff),2) << "\n";
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
    
    float SPDParameterFreeGroundFilter::getThreshold(boost::uint_fast32_t x, boost::uint_fast32_t y, float **data, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast16_t filterHSize, boost::uint_fast16_t **element)
    {
        float threVal = 0;
        
        boost::uint_fast32_t filterPxlStartX = 0;
		boost::uint_fast32_t filterPxlEndX = 0;
		boost::uint_fast32_t filterPxlStartY = 0;
		boost::uint_fast32_t filterPxlEndY = 0;
		
		boost::uint_fast32_t elementStartX = 0;
		boost::uint_fast32_t elementEndX = 0;
		boost::uint_fast32_t elementStartY = 0;
		boost::uint_fast32_t elementEndY = 0;
        
        boost::int_fast32_t filterSize = (filterHSize * 2)+1;        
        
        if((((int_fast64_t)y) - ((int_fast64_t)filterHSize)) < 0)
        {
            filterPxlStartY = 0;
            filterPxlEndY = y + filterHSize;
            elementStartY = filterHSize - y;
            elementEndY = filterSize;
        }
        else if((y+filterHSize) >= ySize)
        {
            filterPxlStartY = y - filterHSize;
            filterPxlEndY = (ySize-1);
            elementStartY = 0;
            elementEndY = filterSize - ((y+filterHSize)-ySize);
        }
        else
        {
            filterPxlStartY = y - filterHSize;
            filterPxlEndY = y + filterHSize;
            elementStartY = 0;
            elementEndY = filterSize;
        }
        
        if((((int_fast64_t)x) - ((int_fast64_t)filterHSize)) < 0)
        {
            filterPxlStartX = 0;
            filterPxlEndX = x + filterHSize;
            elementStartX = filterHSize - x;
            elementEndX = filterSize;
            
        }
        else if((x+filterHSize) >= xSize)
        {
            filterPxlStartX = x - filterHSize;
            filterPxlEndX = xSize-1;
            elementStartX = 0;
            elementEndX = filterSize - ((x+filterHSize)-xSize);
        }
        else
        {
            filterPxlStartX = x - filterHSize;
            filterPxlEndX = x + filterHSize;
            elementStartX = 0;
            elementEndX = filterSize;
        }
        
        boost::uint_fast32_t valCount = 0;
        double sum = 0;
        
        for(boost::uint_fast32_t n = filterPxlStartY, eY = elementStartY; n <= filterPxlEndY; ++n, ++eY)
        {
            for(boost::uint_fast32_t m = filterPxlStartX, eX = elementStartX; m <= filterPxlEndX; ++m, ++eX)
            {
                if((element[eY][eX] == 1) & (!boost::math::isnan(data[n][m])))
                {
                    sum += data[n][m];
                    ++valCount;
                }
            }
        }
        
        if(valCount > 0)
        {
            float mean = sum/valCount;
            
            for(boost::uint_fast32_t n = filterPxlStartY, eY = elementStartY; n <= filterPxlEndY; ++n, ++eY)
            {
                for(boost::uint_fast32_t m = filterPxlStartX, eX = elementStartX; m <= filterPxlEndX; ++m, ++eX)
                {
                    if((element[eY][eX] == 1) & (!boost::math::isnan(data[n][m])))
                    {
                        sum += pow(mean-data[n][m],2);
                    }
                }
            }
            
            float stdDev = sqrt(sum/valCount);
            
            
            threVal = mean + (k * stdDev);
        }
        else
        {
            threVal = numeric_limits<float>::signaling_NaN();
        }
        

        
        return threVal;
    }

    vector<SPDPFFProcessLevel*>* SPDParameterFreeGroundFilter::generateHierarchy(float **initElev, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float pxlRes)
    {
        vector<SPDPFFProcessLevel*> *processingLevels = new vector<SPDPFFProcessLevel*>();
        
        boost::uint_fast32_t numXPxls = xSize;
        boost::uint_fast32_t numYPxls = ySize;
        SPDPFFProcessLevel *level = NULL;
        SPDPFFProcessLevel *prevLevel = new SPDPFFProcessLevel();
        prevLevel->xSize = numXPxls;
        prevLevel->ySize = numYPxls;
        prevLevel->pxlRes = pxlRes;
        prevLevel->data = new float*[numYPxls];
        for(boost::uint_fast32_t i = 0; i < numYPxls; ++i)
        {
            prevLevel->data[i] = new float[numXPxls];
            for(boost::uint_fast32_t j = 0; j < numXPxls; ++j)
            {
                prevLevel->data[i][j] = initElev[i][j];
            }
        }
        processingLevels->push_back(prevLevel);
        
        numXPxls = (numXPxls/2);
        numYPxls = (numYPxls/2);
        float minVal = 0;
        bool firstMin = true;
        
        boost::uint_fast32_t lX = 0;
        boost::uint_fast32_t tY = 0;
        boost::uint_fast32_t rX = 0;
        boost::uint_fast32_t bY = 0;
        
        while((numXPxls > 1) & (numYPxls > 1))
        {
            cout << "Generating level : " << processingLevels->size() << " with [" << numXPxls << "," << numYPxls << "] pixels\n";
            level = new SPDPFFProcessLevel();
            level->xSize = numXPxls;
            level->ySize = numYPxls;
            level->pxlRes = prevLevel->pxlRes*2;
            level->data = new float*[numYPxls];
            for(boost::uint_fast32_t y = 0; y < numYPxls; ++y)
            {
                tY = y * 2;
                bY = tY + 1;
                if(bY >= prevLevel->ySize)
                {
                    bY = tY;
                }
                //cout << y << ": tY = " << tY <<  " bY = " << bY << endl;
                
                level->data[y] = new float[numXPxls];
                for(boost::uint_fast32_t x = 0; x < numXPxls; ++x)
                {
                    firstMin = true;
                    lX = x * 2;
                    rX = lX + 1;
                    if(rX >= prevLevel->xSize)
                    {
                        rX = lX;
                    }
                    
                    //cout << "\t" << x << ": lX = " << lX << " rX = " << rX << endl;
                    
                    
                    if(!boost::math::isnan(prevLevel->data[tY][lX]))
                    {
                        minVal = prevLevel->data[tY][lX];
                        firstMin = false;
                    }
                    if(!boost::math::isnan(prevLevel->data[tY][rX]))
                    {
                        if(firstMin)
                        {
                            minVal = prevLevel->data[tY][rX];
                            firstMin = false;
                        }
                        else if(prevLevel->data[tY][rX] < minVal)
                        {
                            minVal = prevLevel->data[tY][rX];
                        }
                    }
                    if(!boost::math::isnan(prevLevel->data[bY][lX]))
                    {
                        if(firstMin)
                        {
                            minVal = prevLevel->data[bY][lX];
                            firstMin = false;
                        }
                        else if(prevLevel->data[bY][lX] < minVal)
                        {
                            minVal = prevLevel->data[bY][lX];
                        }
                    }
                    if(!boost::math::isnan(prevLevel->data[bY][rX]))
                    {
                        if(firstMin)
                        {
                            minVal = prevLevel->data[bY][rX];
                            firstMin = false;
                        }
                        else if(prevLevel->data[bY][rX] < minVal)
                        {
                            minVal = prevLevel->data[bY][rX];
                        }
                    }
                    
                    if(firstMin)
                    {
                        level->data[y][x] = numeric_limits<float>::signaling_NaN();
                    }
                    else
                    {
                        level->data[y][x] = minVal;
                    }
                }
                //cout << endl;
            }
            processingLevels->push_back(level);
            prevLevel = level;
            numXPxls = (numXPxls/2);
            numYPxls = (numYPxls/2);
        }
        
        return processingLevels;
    }
    
    void SPDParameterFreeGroundFilter::freeHierarchy(vector<SPDPFFProcessLevel*> *levels)
    {
        for(vector<SPDPFFProcessLevel*>::iterator iterLevels = levels->begin(); iterLevels != levels->end(); ++iterLevels)
        {
            for(boost::uint_fast32_t i = 0; i < (*iterLevels)->ySize; ++i)
            {
                delete[] (*iterLevels)->data[i];
            }
            delete[] (*iterLevels)->data;
            delete (*iterLevels);
        }
        levels->clear();
    }
    
    void SPDParameterFreeGroundFilter::freeLevel(SPDPFFProcessLevel *level)
    {
        for(boost::uint_fast32_t i = 0; i < level->ySize; ++i)
        {
            delete[] level->data[i];
        }
        delete[] level->data;
        delete level;
    }
    
    SPDPFFProcessLevel* SPDParameterFreeGroundFilter::interpLevel(SPDPFFProcessLevel *cLevel, SPDPFFProcessLevel *processLevel, double tlY, double tlX)
    {
        // Generate interpolator...
        double eastings = tlX + (cLevel->pxlRes/2);
        double northings = tlY - (cLevel->pxlRes/2);        

        SPDTPSPFFGrdFilteringInterpolator interpolator = SPDTPSPFFGrdFilteringInterpolator(cLevel->pxlRes*4);
        interpolator.initInterpolator(cLevel->data, cLevel->xSize, cLevel->ySize, eastings, northings, cLevel->pxlRes);
        
        eastings = tlX + (processLevel->pxlRes/2);
        northings = tlY - (processLevel->pxlRes/2);
        SPDPFFProcessLevel *interpdLevel = new SPDPFFProcessLevel();
        interpdLevel->xSize = processLevel->xSize;
        interpdLevel->ySize = processLevel->ySize;
        interpdLevel->pxlRes = processLevel->pxlRes;
        interpdLevel->data = new float*[interpdLevel->ySize];
        for(boost::uint_fast32_t i = 0; i < interpdLevel->ySize; ++i)
        {
            interpdLevel->data[i] = new float[interpdLevel->xSize];
            eastings = tlX + (processLevel->pxlRes/2);
            for(boost::uint_fast32_t j = 0; j < interpdLevel->xSize; ++j)
            {
                try
                {
                    interpdLevel->data[i][j] = interpolator.getValue(eastings, northings); // Interpolate value for eastings, northings
                }
                catch(SPDException &e)
                {
                    interpdLevel->data[i][j] = numeric_limits<float>::signaling_NaN();
                }
                eastings += (processLevel->pxlRes/2);
            }
            northings -= (processLevel->pxlRes/2);
        }
        interpolator.resetInterpolator();
        
        return interpdLevel;
    }
    
    SPDParameterFreeGroundFilter::~SPDParameterFreeGroundFilter()
    {
        
    }
    
    
    
    
    
    
    SPDTPSPFFGrdFilteringInterpolator::SPDTPSPFFGrdFilteringInterpolator(float radius)
	{
        this->initialised = false;
        this->radius = radius;
	}
    
    void SPDTPSPFFGrdFilteringInterpolator::initInterpolator(float **data, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, double tlEastings, double tlNorthings, float binSize) throw(SPDProcessingException)
    {
        this->data = data;
        this->xSize = xSize;
        this->ySize = ySize;
        this->tlEastings = tlEastings;
        this->tlNorthings = tlNorthings;
        this->binSize = binSize;
        this->initialised = true;
    }
    
	void SPDTPSPFFGrdFilteringInterpolator::resetInterpolator() throw(SPDProcessingException)
	{
		if(initialised)
		{
			this->data = NULL;
            this->initialised = false;
		}
	}
	
	float SPDTPSPFFGrdFilteringInterpolator::getValue(double eastings, double northings) throw(SPDProcessingException)
	{
        float newZValue = 0;
        if(initialised)
		{
            try 
            {
                //cout.precision(12);
                //cout << "\nPos: [" << eastings << "," << northings << "]\n";
                //cout << "Size: [" << xSize << "," << ySize << "]\n";
                //cout << "Radius: " << radius << endl;
                //cout << "Bin Size: " << binSize << endl;
                
                
                vector<Vec> cntrlPts = vector<Vec>();
                boost::int_fast32_t radiusInPxl = numeric_cast<boost::uint_fast32_t>(radius/binSize)+1;
                boost::uint_fast16_t elemSize = (radiusInPxl*2)+1;
                
                //cout << "Pixels in Radius: " << radiusInPxl << endl;
                //cout << "Element Size: " << elemSize << endl;
                
                if((xSize < elemSize) & (ySize < elemSize))
                {
                    double cEast = tlEastings;
                    double cNorth = tlNorthings;
                    for(boost::uint_fast32_t i = 0; i < ySize; ++i)
                    {
                        cEast = eastings;
                        for(boost::uint_fast32_t j = 0; j < xSize; ++j)
                        {
                            //cout << "Data [" << i << "][" << j << "] = " << data[i][j] << endl;
                            if(!boost::math::isnan(data[i][j]))
                            {
                                cntrlPts.push_back(Vec(cEast, data[i][j], cNorth));
                            }
                            cEast += this->binSize;
                        }
                        cNorth -= this->binSize;
                    }
                }
                else
                {
                    double diffEast = eastings - tlEastings;
                    double diffNorth = tlNorthings - northings;
                    
                    //cout << "Spatial Diff: [" << diffEast << "," << diffNorth << "]\n";
                    
                    boost::int_fast32_t xPxl = 0;
                    boost::int_fast32_t yPxl = 0;
                    boost::uint_fast32_t tlXPxl = 0;
                    boost::uint_fast32_t tlYPxl = 0;
                    boost::uint_fast32_t brXPxl = 0;
                    boost::uint_fast32_t brYPxl = 0;
                    boost::uint_fast32_t widthPxls = 0;
                    boost::uint_fast32_t heightPxls = 0;
                    
                    if(diffEast < 0)
                    {
                        tlXPxl = 0;
                        brXPxl = tlXPxl + radiusInPxl;
                    }
                    else
                    {
                        xPxl = numeric_cast<boost::uint_fast32_t>(diffEast/binSize);
                        //cout << "xPxl: " << xPxl << endl;
                        if((xPxl - radiusInPxl) < 0)
                        {
                            tlXPxl = 0;
                        }
                        else
                        {
                            tlXPxl = xPxl-radiusInPxl;
                        }
                        brXPxl = tlXPxl + elemSize;
                    }
                    
                    if(diffNorth < 0)
                    {
                        tlYPxl = 0;
                        brYPxl = tlYPxl + radiusInPxl;
                    }
                    else
                    {
                        yPxl = numeric_cast<boost::uint_fast32_t>(diffNorth/binSize);
                        //cout << "yPxl: " << yPxl << endl;
                        if((yPxl - radiusInPxl) < 0)
                        {
                            tlYPxl = 0;
                        }
                        else
                        {
                            tlYPxl = yPxl-radiusInPxl;
                        }
                        
                        brYPxl = tlYPxl + elemSize;
                    }
                    
                    if(brXPxl > xSize)
                    {
                        brXPxl = xSize;
                    }
                    
                    if(brYPxl > ySize)
                    {
                        brYPxl = ySize;
                    }
                    
                    widthPxls = brXPxl - tlXPxl;
                    heightPxls = brYPxl - tlYPxl;

                    
                    //cout << "TL Spatial: [" << tlEastings << "," << tlNorthings << "]\n";
                    //cout << "TL Pxl: [" << tlXPxl << "," << tlYPxl << "]\n";
                    //cout << "BR Pxl: [" << brXPxl << "," << brYPxl << "]\n";
                    //cout << "Size Pxl: [" << widthPxls << "," << heightPxls << "]\n";
                    
                    double pxlTLEast = tlEastings + ((tlXPxl * this->binSize) + this->binSize/2);
                    double pxlTLNorth = tlNorthings - ((tlYPxl * this->binSize) - this->binSize/2);
                    
                    //cout << "Pxl TL Spatial: [" << pxlTLEast << "," << pxlTLNorth << "]\n";
                    
                    double cEast = pxlTLEast;
                    double cNorth = pxlTLNorth;
                    
                    for(boost::uint_fast32_t i = tlYPxl; i < brYPxl; ++i)
                    {
                        cEast = pxlTLEast;
                        for(boost::uint_fast32_t j = tlXPxl; j < brXPxl; ++j)
                        {
                            //cout << "Data [" << i << "][" << j << "] = " << data[i][j] << endl;
                            if(!boost::math::isnan(data[i][j]))
                            {
                                if(this->distance(eastings, northings, cEast, cNorth) < radius)
                                {
                                    cntrlPts.push_back(Vec(cEast, data[i][j], cNorth));
                                }
                            }
                            cEast += this->binSize;
                            cEast += this->binSize;
                        }
                        cNorth -= this->binSize;
                    }
                }
                
                //cout << "Num cntrlPts = " << cntrlPts.size() << endl;
                
                if(cntrlPts.size() == 0)
                {
                    newZValue = numeric_limits<float>::signaling_NaN();
                }
                else if(cntrlPts.size() == 1)
                {
                    newZValue = cntrlPts.front().y;
                }
                else if(cntrlPts.size() == 2)
                {
                    newZValue = (cntrlPts.at(0).y + cntrlPts.at(1).y)/2;
                }
                else
                {
                    Spline splineFunc = Spline(cntrlPts, 0.0);
                    newZValue = splineFunc.interpolate_height(eastings, northings);
                }
            }
            catch(negative_overflow& e) 
            {
                newZValue = numeric_limits<float>::signaling_NaN();
            }
            catch(positive_overflow& e) 
            {
                newZValue = numeric_limits<float>::signaling_NaN();
            }
            catch(bad_numeric_cast& e) 
            {
                newZValue = numeric_limits<float>::signaling_NaN();
            }
            catch(SingularMatrixError &e)
            {
                newZValue = numeric_limits<float>::signaling_NaN();
            }
            catch (SPDProcessingException &e) 
            {
                newZValue = numeric_limits<float>::signaling_NaN();
            }
            catch(exception &e)
            {
                newZValue = numeric_limits<float>::signaling_NaN();
            }
        }
        else
        {
            throw SPDProcessingException("Interpolator has not been initialised.");
        }
        return newZValue;
	}
    
    double SPDTPSPFFGrdFilteringInterpolator::distance(double eastings, double northings, double cEastings, double cNorthings)
    {
        return sqrt((pow(eastings-cEastings,2)+pow(northings-cNorthings,2))/2);
    }
	
	SPDTPSPFFGrdFilteringInterpolator::~SPDTPSPFFGrdFilteringInterpolator()
	{
		if(initialised)
		{
			this->data = NULL;
            this->initialised = false;
		}
	}
    
    
}



