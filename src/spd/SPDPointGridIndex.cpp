/*
 *  SPDPointGridIndex.cpp
 *  SPDLIB
 *
 *  Created by Pete Bunting on 08/03/2011.
 *  Copyright 2011 SPDLib. All rights reserved.
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
 */

#include "spd/SPDPointGridIndex.h"

namespace spdlib
{

	SPDPointGridIndex::SPDPointGridIndex():ptGrid(NULL), tlX(0), tlY(0), brX(0), brY(0), binSize(0), xBins(0), yBins(0), deletePtsInBins(false)
	{
		deletePtsInBins = false;
	}
	
    void SPDPointGridIndex::buildIndex(std::vector<SPDPoint*> *pts, double resolution, OGREnvelope *env) throw(SPDProcessingException)
	{
		if(pts->size() > 0)
		{
            double minX = 0;
			double maxX = 0;
			double minY = 0;
			double maxY = 0;
			bool first = true;
			std::vector<SPDPoint*>::iterator iterPts;
			for(iterPts = pts->begin(); iterPts != pts->end(); ++iterPts)
			{
				if(first)
				{
					minX = (*iterPts)->x;
					maxX = (*iterPts)->x;
					minY = (*iterPts)->y;
					maxY = (*iterPts)->y;
					first = false;
				}
				else 
				{
					if((*iterPts)->x < minX)
					{
						minX = (*iterPts)->x;
					}
					else if((*iterPts)->x > maxX)
					{
						maxX = (*iterPts)->x;
					}
					
					if((*iterPts)->y < minY)
					{
						minY = (*iterPts)->y;
					}
					else if((*iterPts)->y > maxY)
					{
						maxY = (*iterPts)->y;
					}
				}
			}
            
            if(env->MinX < minX)
            {
                this->tlX = env->MinX;
            }
            else 
            {
                this->tlX = minX;
            }
            if(env->MaxY > maxY)
            {
                this->tlY = env->MaxY;
            }
            else
            {
                this->tlY = maxY;
            }
            if(env->MaxX > maxX)
            {
                this->brX = env->MaxX;
            }
            else
            {
                this->brX = maxX;
            }
            if(env->MinY < minY)
            {
                this->brY = env->MinY;
            }
            else
            {
                this->brY = minY;
            }
			this->binSize = resolution;
			
			boost::uint_fast32_t roundingAddition = 0;
			try
			{
				if(this->binSize < 1)
				{
					roundingAddition = boost::numeric_cast<boost::uint_fast32_t>(1/this->binSize);
				}
				else 
				{
					roundingAddition = 1;
				}
				
				xBins = boost::numeric_cast<boost::uint_fast32_t>(((this->brX-this->tlX)/this->binSize)+roundingAddition);
				yBins = boost::numeric_cast<boost::uint_fast32_t>(((this->tlY-this->brY)/this->binSize)+roundingAddition);
			}
			catch(boost::numeric::negative_overflow& e) 
			{
				throw SPDProcessingException(e.what());
			}
			catch(boost::numeric::positive_overflow& e) 
			{
				throw SPDProcessingException(e.what());
			}
			catch(boost::numeric::bad_numeric_cast& e) 
			{
				throw SPDProcessingException(e.what());
			}
			
			if((xBins < 1) | (yBins < 1))
			{
				throw SPDProcessingException("There insufficent number of bins for binning (try reducing resolution).");
			}
			
			this->ptGrid = new std::vector<SPDPoint*>**[yBins];
			for(boost::uint_fast32_t i = 0; i < yBins; ++i)
			{
				this->ptGrid[i] = new std::vector<SPDPoint*>*[xBins];
				for(boost::uint_fast32_t j = 0; j < xBins; ++j)
				{
					this->ptGrid[i][j] = new std::vector<SPDPoint*>();
				}
			}
			
			try 
			{	
				double xDiff = 0;
				double yDiff = 0;
				boost::uint_fast32_t xIdx = 0;
				boost::uint_fast32_t yIdx = 0;
				
				SPDPoint *pt = NULL;
                std::vector<SPDPoint*>::iterator iterPts;
				for(iterPts = pts->begin(); iterPts != pts->end(); ++iterPts)
				{
					pt = *iterPts;
					
					xDiff = (pt->x - this->tlX)/this->binSize;
					yDiff = (this->tlY - pt->y)/this->binSize;				
					
					try 
					{
						xIdx = boost::numeric_cast<boost::uint_fast32_t>(xDiff);
						yIdx = boost::numeric_cast<boost::uint_fast32_t>(yDiff);
					}
					catch(boost::numeric::negative_overflow& e) 
					{
						throw SPDProcessingException(e.what());
					}
					catch(boost::numeric::positive_overflow& e) 
					{
						throw SPDProcessingException(e.what());
					}
					catch(boost::numeric::bad_numeric_cast& e) 
					{
						throw SPDProcessingException(e.what());
					}
					
					if(xIdx > (this->xBins-1))
					{
						std::cout << "Point: [" << pt->x << "," << pt->y << "]\n";
						std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
						std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
						std::cout << "Size [" << xBins << "," << yBins << "]\n";
						throw SPDProcessingException("Did not find x index within range.");
					}
					
					if(yIdx > (this->yBins-1))
					{
						std::cout << "Point: [" << pt->x << "," << pt->y << "]\n";
						std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
						std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
						std::cout << "Size [" << xBins << "," << yBins << "]\n";
						throw SPDProcessingException("Did not find y index within range.");
					}
					
					this->ptGrid[yIdx][xIdx]->push_back(pt);
				}
			}
			catch (SPDProcessingException &e) 
			{
				for(boost::uint_fast32_t i = 0; i < yBins; ++i)
				{
					for(boost::uint_fast32_t j = 0; j < xBins; ++j)
					{
						delete this->ptGrid[i][j];
					}
					delete[] this->ptGrid[i];
				}
				delete[] this->ptGrid;
				
				throw e;
			}
			
		}
		else 
		{
			throw SPDProcessingException("Inputted list of points was empty.");
		}
        deletePtsInBins = false;
	}
    
	void SPDPointGridIndex::buildIndex(std::vector<SPDPoint*> *pts, double resolution) throw(SPDProcessingException)
	{
		if(pts->size() > 0)
		{
			double minX = 0;
			double maxX = 0;
			double minY = 0;
			double maxY = 0;
			bool first = true;
			std::vector<SPDPoint*>::iterator iterPts;
			for(iterPts = pts->begin(); iterPts != pts->end(); ++iterPts)
			{
				if(first)
				{
					minX = (*iterPts)->x;
					maxX = (*iterPts)->x;
					minY = (*iterPts)->y;
					maxY = (*iterPts)->y;
					first = false;
				}
				else 
				{
					if((*iterPts)->x < minX)
					{
						minX = (*iterPts)->x;
					}
					else if((*iterPts)->x > maxX)
					{
						maxX = (*iterPts)->x;
					}
					
					if((*iterPts)->y < minY)
					{
						minY = (*iterPts)->y;
					}
					else if((*iterPts)->y > maxY)
					{
						maxY = (*iterPts)->y;
					}
				}
			}
			
			this->tlX = minX;
			this->tlY = maxY;
			this->brX = maxX;
			this->brY = minY;
			this->binSize = resolution;
            
            //std::cout << "Index TL [" << this->tlX << "," << this->tlY << "] BR [" << this->brX << "," << this->brY << "]\n";
			
			boost::uint_fast32_t roundingAddition = 0;
			try
			{
				if(this->binSize < 1)
				{
					roundingAddition = 4;
				}
				else 
				{
					roundingAddition = 1;
				}
				
				xBins = boost::numeric_cast<boost::uint_fast32_t>(((maxX-minX)/this->binSize)+roundingAddition);
				yBins = boost::numeric_cast<boost::uint_fast32_t>(((maxY-minY)/this->binSize)+roundingAddition);
			}
			catch(boost::numeric::negative_overflow& e) 
			{
				throw SPDProcessingException(e.what());
			}
			catch(boost::numeric::positive_overflow& e) 
			{
				throw SPDProcessingException(e.what());
			}
			catch(boost::numeric::bad_numeric_cast& e) 
			{
				throw SPDProcessingException(e.what());
			}
			
			if((xBins < 1) | (yBins < 1))
			{
				throw SPDProcessingException("There insufficent number of bins for binning (try reducing resolution).");
			}
			
			this->ptGrid = new std::vector<SPDPoint*>**[yBins];
			for(boost::uint_fast32_t i = 0; i < yBins; ++i)
			{
				this->ptGrid[i] = new std::vector<SPDPoint*>*[xBins];
				for(boost::uint_fast32_t j = 0; j < xBins; ++j)
				{
					this->ptGrid[i][j] = new std::vector<SPDPoint*>();
				}
			}
			
			try 
			{	
				double xDiff = 0;
				double yDiff = 0;
				boost::uint_fast32_t xIdx = 0;
				boost::uint_fast32_t yIdx = 0;
				
				SPDPoint *pt = NULL;
				for(iterPts = pts->begin(); iterPts != pts->end(); ++iterPts)
				{
					pt = *iterPts;
					
					xDiff = (pt->x - this->tlX)/this->binSize;
					yDiff = (this->tlY - pt->y)/this->binSize;				
					
					try 
					{
						xIdx = boost::numeric_cast<boost::uint_fast32_t>(xDiff);
						yIdx = boost::numeric_cast<boost::uint_fast32_t>(yDiff);
					}
					catch(boost::numeric::negative_overflow& e) 
					{
						throw SPDProcessingException(e.what());
					}
					catch(boost::numeric::positive_overflow& e) 
					{
						throw SPDProcessingException(e.what());
					}
					catch(boost::numeric::bad_numeric_cast& e) 
					{
						throw SPDProcessingException(e.what());
					}
					
					if(xIdx > (this->xBins-1))
					{
						std::cout << "Point: [" << pt->x << "," << pt->y << "]\n";
						std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
						std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
						std::cout << "Size [" << xBins << "," << yBins << "]\n";
						throw SPDProcessingException("Did not find x index within range.");
					}
					
					if(yIdx > (this->yBins-1))
					{
						std::cout << "Point: [" << pt->x << "," << pt->y << "]\n";
						std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
						std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
						std::cout << "Size [" << xBins << "," << yBins << "]\n";
						throw SPDProcessingException("Did not find y index within range.");
					}
					                    
					this->ptGrid[yIdx][xIdx]->push_back(pt);
				}
			}
			catch (SPDProcessingException &e) 
			{
				for(boost::uint_fast32_t i = 0; i < yBins; ++i)
				{
					for(boost::uint_fast32_t j = 0; j < xBins; ++j)
					{
						delete this->ptGrid[i][j];
					}
					delete[] this->ptGrid[i];
				}
				delete[] this->ptGrid;
				
				throw e;
			}
			
		}
		else 
		{
			throw SPDProcessingException("Inputted list of points was empty.");
		}
		deletePtsInBins = false;
	}
	
	bool SPDPointGridIndex::getPointsInRadius(std::vector<SPDPoint*> *pts, double eastings, double northings, double radius) throw(SPDProcessingException)
	{
		if((eastings < tlX) | 
		   (eastings > brX) |
		   (northings < brY) |
		   (northings > tlY))
		{
			return false;
		}
		
		if(radius <= 0)
		{
			throw SPDProcessingException("Radius is less than or equal to 0.");
		}
		
		bool returnVal = false;
		try 
		{
			boost::uint_fast32_t xIdx = 0;
			boost::uint_fast32_t yIdx = 0;
			boost::uint_fast32_t radiusInBins = 0;
			
			double xDiff = (eastings - this->tlX)/this->binSize;
			double yDiff = (this->tlY - northings)/this->binSize;
			double radiusInBinsFl = radius / this->binSize;
			try 
			{
				xIdx = boost::numeric_cast<boost::uint_fast32_t>(xDiff);
				yIdx = boost::numeric_cast<boost::uint_fast32_t>(yDiff);
				radiusInBins = boost::numeric_cast<boost::uint_fast32_t>(ceil(radiusInBinsFl));
			}
			catch(boost::numeric::negative_overflow& e) 
			{
				throw SPDProcessingException(e.what());
			}
			catch(boost::numeric::positive_overflow& e) 
			{
				throw SPDProcessingException(e.what());
			}
			catch(boost::numeric::bad_numeric_cast& e) 
			{
				throw SPDProcessingException(e.what());
			}
			
			if(xIdx > (this->xBins-1))
			{
				std::cout << "Point: [" << eastings << "," << northings << "]\n";
				std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
				std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
				std::cout << "Size [" << xBins << "," << yBins << "]\n";
				return false;
			}
			
			if(yIdx > (this->yBins-1))
			{
				std::cout << "Point: [" << eastings << "," << northings << "]\n";
				std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
				std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
				std::cout << "Size [" << xBins << "," << yBins << "]\n";
				return false;
			}
			
			boost::uint_fast32_t radiusStartX = 0;
			boost::uint_fast32_t radiusEndX = 0;
			boost::uint_fast32_t radiusStartY = 0;
			boost::uint_fast32_t radiusEndY = 0;
			
			if((((int_fast64_t)xIdx) - ((int_fast64_t)radiusInBins)) < 0)
			{
				radiusStartX = 0;
				radiusEndX = xIdx + radiusInBins;
			}
			else if((xIdx+radiusInBins) >= xBins)
			{
				radiusStartX = xIdx - radiusInBins;
				radiusEndX = xBins-1;
			}
			else
			{
				radiusStartX = xIdx - radiusInBins;
				radiusEndX = xIdx + radiusInBins;
			}
			
			if((((int_fast64_t)yIdx) - ((int_fast64_t)radiusInBins)) < 0)
			{
				radiusStartY = 0;
				radiusEndY = yIdx + radiusInBins;
			}
			else if((yIdx+radiusInBins) >= yBins)
			{
				radiusStartY = yIdx - radiusInBins;
				radiusEndY = yBins-1;
			}
			else
			{
				radiusStartY = yIdx - radiusInBins;
				radiusEndY = yIdx + radiusInBins;
			}
			
			SPDPointUtils ptUtils;
			std::vector<SPDPoint*>::iterator iterPts;
			for(boost::uint_fast32_t i = radiusStartY; i < radiusEndY; ++i)
			{
				for(boost::uint_fast32_t j = radiusStartX; j < radiusEndX; ++j)
				{
					for(iterPts = ptGrid[i][j]->begin(); iterPts != ptGrid[i][j]->end(); ++iterPts)
                    {
                        if(ptUtils.distanceXY(eastings, northings, (*iterPts)) <= radius)
                        {
                            pts->push_back(*iterPts);
                        }
                    }
				}
			}
			returnVal = true;
		}
		catch (SPDProcessingException &e) 
		{
			throw e;
		}
		
		return returnVal;
	}
	
    bool SPDPointGridIndex::getSetNumOfPoints(std::vector<SPDPoint*> *pts, double eastings, double northings, boost::uint_fast16_t numPts, double maxRadius) throw(SPDProcessingException)
    {
        //std::cout << "PT: [" << eastings << "," << northings << "]\n";
        
        if((eastings < tlX) | 
		   (eastings > brX) |
		   (northings < brY) |
		   (northings > tlY))
		{
			return false;
		}
		
		if(maxRadius <= 0)
		{
			throw SPDProcessingException("Radius is less than or equal to 0.");
		}
        
        bool returnVal = false;
		try 
		{
            currentCmpEastings = eastings;
            currentCmpNorthings = northings;
            
			boost::uint_fast32_t xIdx = 0;
			boost::uint_fast32_t yIdx = 0;
			
			double xDiff = (eastings - this->tlX)/this->binSize;
			double yDiff = (this->tlY - northings)/this->binSize;
			try 
			{
				xIdx = boost::numeric_cast<boost::uint_fast32_t>(xDiff);
				yIdx = boost::numeric_cast<boost::uint_fast32_t>(yDiff);
			}
			catch(boost::numeric::negative_overflow& e) 
			{
				throw SPDProcessingException(e.what());
			}
			catch(boost::numeric::positive_overflow& e) 
			{
				throw SPDProcessingException(e.what());
			}
			catch(boost::numeric::bad_numeric_cast& e) 
			{
				throw SPDProcessingException(e.what());
			}
			
			if(xIdx > (this->xBins-1))
			{
				std::cout << "Point: [" << eastings << "," << northings << "]\n";
				std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
				std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
				std::cout << "Size [" << xBins << "," << yBins << "]\n";
				return false;
			}
			
			if(yIdx > (this->yBins-1))
			{
				std::cout << "Point: [" << eastings << "," << northings << "]\n";
				std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
				std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
				std::cout << "Size [" << xBins << "," << yBins << "]\n";
				return false;
			}
            
            //std::cout << "PT: [yIdx][xIdx] = [" << yIdx<< "][" << xIdx << "]" << std::endl;
            
            if(ptGrid[yIdx][xIdx]->size() >= numPts)
            {
                std::sort(ptGrid[yIdx][xIdx]->begin(), ptGrid[yIdx][xIdx]->end(), compareFuncSortByDistanceTo);
                
                for(boost::uint_fast16_t i = 0; i < numPts; ++i)
                {
                    pts->push_back(ptGrid[yIdx][xIdx]->at(i));
                }
                returnVal = true;
            }
            else
            {
                bool foundSufficientPoints = false;
                float currentRadius = binSize;
                boost::uint_fast32_t currRadiusInBins = 0;
                boost::uint_fast32_t radiusStartX = 0;
                boost::uint_fast32_t radiusEndX = 0;
                boost::uint_fast32_t radiusStartY = 0;
                boost::uint_fast32_t radiusEndY = 0;
                boost::uint_fast32_t totalPtsCount = 0;
                
                do
                {
                    try 
                    {
                        currRadiusInBins = boost::numeric_cast<boost::uint_fast32_t>(ceil(currentRadius));
                    }
                    catch(boost::numeric::negative_overflow& e) 
                    {
                        throw SPDProcessingException(e.what());
                    }
                    catch(boost::numeric::positive_overflow& e) 
                    {
                        throw SPDProcessingException(e.what());
                    }
                    catch(boost::numeric::bad_numeric_cast& e) 
                    {
                        throw SPDProcessingException(e.what());
                    }
                                       
                    if((((int_fast64_t)xIdx) - ((int_fast64_t)currRadiusInBins)) < 0)
                    {
                        radiusStartX = 0;
                        radiusEndX = xIdx + currRadiusInBins;
                    }
                    else if((xIdx+currRadiusInBins) >= xBins)
                    {
                        radiusStartX = xIdx - currRadiusInBins;
                        radiusEndX = xBins-1;
                    }
                    else
                    {
                        radiusStartX = xIdx - currRadiusInBins;
                        radiusEndX = xIdx + currRadiusInBins;
                    }
                    
                    if((((int_fast64_t)yIdx) - ((int_fast64_t)currRadiusInBins)) < 0)
                    {
                        radiusStartY = 0;
                        radiusEndY = yIdx + currRadiusInBins;
                    }
                    else if((yIdx+currRadiusInBins) >= yBins)
                    {
                        radiusStartY = yIdx - currRadiusInBins;
                        radiusEndY = yBins-1;
                    }
                    else
                    {
                        radiusStartY = yIdx - currRadiusInBins;
                        radiusEndY = yIdx + currRadiusInBins;
                    }
                    
                    if(radiusEndX > this->xBins)
                    {
                        radiusEndX = this->xBins;
                    }
                    
                    if(radiusEndY > this->yBins)
                    {
                        radiusEndY = this->yBins;
                    }
                    
                    //std::cout << "radiusStartY = " << radiusStartY << std::endl;
                    //std::cout << "radiusEndY = " << radiusEndY << std::endl;
                    //std::cout << "radiusStartX = " << radiusStartX << std::endl;
                    //std::cout << "radiusEndX = " << radiusEndX << std::endl;
                    
                    totalPtsCount = 0;
                    for(boost::uint_fast32_t i = radiusStartY; i < radiusEndY; ++i)
                    {
                        for(boost::uint_fast32_t j = radiusStartX; j < radiusEndX; ++j)
                        {
                            //std::cout << "[" << i << "][" << j << "]\n";
                            totalPtsCount += ptGrid[i][j]->size();
                        }
                    }
                    
                    //std::cout << "CurrentRadius = " << currentRadius << std::endl;
                    //std::cout << "totalPtsCount = " << totalPtsCount <<std::endl;
                    //std::cout << "Number of Points Required = " << numPts <<std::endl << std::endl;
                    
                    if(totalPtsCount >= numPts)
                    {
                        foundSufficientPoints = true;
                    }
                    
                    currentRadius += binSize;
                }while((currentRadius < maxRadius) & !foundSufficientPoints);
                
                if(foundSufficientPoints)
                {
                    //std::cout << "X idxs [" << radiusStartX << "," << radiusEndX << "]\n";
                    //std::cout << "Y idxs [" << radiusStartY << "," << radiusEndY << "]\n";
                    
                    std::vector<SPDPoint*> *possPoints = new std::vector<SPDPoint*>();
                    
                    std::vector<SPDPoint*>::iterator iterPts;
                    for(boost::uint_fast32_t i = radiusStartY; i < radiusEndY; ++i)
                    {
                        for(boost::uint_fast32_t j = radiusStartX; j < radiusEndX; ++j)
                        {
                            for(iterPts = ptGrid[i][j]->begin(); iterPts != ptGrid[i][j]->end(); ++iterPts)
                            {
                                possPoints->push_back(*iterPts);
                            }
                        }
                    }
                    //std::cout << "num poss points (1) = " << possPoints->size() << std::endl;
                    
                    std::sort(possPoints->begin(), possPoints->end(), compareFuncSortByDistanceTo);
                    
                    //std::cout << "num poss points (2) = " << possPoints->size() << std::endl;
                    
                    for(boost::uint_fast16_t i = 0; i < numPts; ++i)
                    {
                        pts->push_back(possPoints->at(i));
                    }
                    
                    delete possPoints;
                    returnVal = true;
                }
                else
                {
                    returnVal = false;
                }
            }
            
		}
		catch (SPDProcessingException &e) 
		{
			throw e;
		}
		
		return returnVal;
    }
    
    void SPDPointGridIndex::thinPtsInBins(boost::uint_fast16_t elevVal, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin) throw(SPDProcessingException)
    {
        boost::uint_fast32_t numPtsToRemove = 0;
        for(boost::uint_fast32_t i = 0; i < yBins; ++i)
        {
            for(boost::uint_fast32_t j = 0; j < xBins; ++j)
            {
                if(ptGrid[i][j]->size() > maxNumPtsPerBin)
                {
                    if(elevVal == SPD_USE_Z)
                    {
                        if(selectHighOrLow == SPD_SELECT_LOWEST)
                        {
                            std::sort(ptGrid[i][j]->begin(), ptGrid[i][j]->end(), compareFuncSortByZSmallestFirst);
                        }
                        else if(selectHighOrLow == SPD_SELECT_HIGHEST)
                        {
                            std::sort(ptGrid[i][j]->begin(), ptGrid[i][j]->end(), compareFuncSortByZLargestFirst);
                        }
                        else
                        {
                            throw SPDProcessingException("Do not recognise point selection type (needs to be either highest or lowest).");
                        }
                    }
                    else if(elevVal == SPD_USE_HEIGHT)
                    {
                        if(selectHighOrLow == SPD_SELECT_LOWEST)
                        {
                            std::sort(ptGrid[i][j]->begin(), ptGrid[i][j]->end(), compareFuncSortByHeightSmallestFirst);
                        }
                        else if(selectHighOrLow == SPD_SELECT_HIGHEST)
                        {
                            std::sort(ptGrid[i][j]->begin(), ptGrid[i][j]->end(), compareFuncSortByHeightLargestFirst);
                        }
                        else
                        {
                            throw SPDProcessingException("Do not recognise point selection type (needs to be either highest or lowest).");
                        }
                    }
                    else
                    {
                        throw SPDProcessingException("Do not recognise elevation type (needs to be height or Z).");
                    }
                    
                    numPtsToRemove = ptGrid[i][j]->size() - maxNumPtsPerBin;
                    for(boost::uint_fast32_t n = 0; n < numPtsToRemove; ++n)
                    {
                        ptGrid[i][j]->pop_back();
                    }
                }
            }
        }
    }
    
    /*
    void SPDPointGridIndex::thinPtsInBinsWithDelete(boost::uint_fast16_t elevVal, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin) throw(SPDProcessingException)
    {
        boost::uint_fast32_t numPtsToRemove = 0;
        for(boost::uint_fast32_t i = 0; i < yBins; ++i)
        {
            for(boost::uint_fast32_t j = 0; j < xBins; ++j)
            {
                if(ptGrid[i][j]->size() > maxNumPtsPerBin)
                {
                    if(elevVal == SPD_USE_Z)
                    {
                        if(selectHighOrLow == SPD_SELECT_LOWEST)
                        {
                            std::sort(ptGrid[i][j]->begin(), ptGrid[i][j]->end(), compareFuncSortByZSmallestFirst);
                        }
                        else if(selectHighOrLow == SPD_SELECT_HIGHEST)
                        {
                            std::sort(ptGrid[i][j]->begin(), ptGrid[i][j]->end(), compareFuncSortByZLargestFirst);
                        }
                        else
                        {
                            throw SPDProcessingException("Do not recognise point selection type (needs to be either highest or lowest).");
                        }
                    }
                    else if(elevVal == SPD_USE_HEIGHT)
                    {
                        if(selectHighOrLow == SPD_SELECT_LOWEST)
                        {
                            std::sort(ptGrid[i][j]->begin(), ptGrid[i][j]->end(), compareFuncSortByHeightSmallestFirst);
                        }
                        else if(selectHighOrLow == SPD_SELECT_HIGHEST)
                        {
                            std::sort(ptGrid[i][j]->begin(), ptGrid[i][j]->end(), compareFuncSortByHeightLargestFirst);
                        }
                        else
                        {
                            throw SPDProcessingException("Do not recognise point selection type (needs to be either highest or lowest).");
                        }
                    }
                    else
                    {
                        throw SPDProcessingException("Do not recognise elevation type (needs to be height or Z).");
                    }
                    
                    numPtsToRemove = ptGrid[i][j]->size() - maxNumPtsPerBin;
                    for(boost::uint_fast32_t n = 0; n < numPtsToRemove; ++n)
                    {
                        delete ptGrid[i][j]->back();
                        ptGrid[i][j]->pop_back();
                    }
                }
            }
        }
    }
    */
    
    void SPDPointGridIndex::thinPtsWithAvZ(boost::uint_fast16_t elevVal) throw(SPDProcessingException)
    {
        std::vector<SPDPoint*>::iterator iterPts;
        SPDPoint *pt = NULL;
        SPDPointUtils ptUtils;
        double currentX = tlX + (binSize/2);
        double currentY = tlY - (binSize/2);
        float zVal = 0;
        for(boost::uint_fast32_t i = 0; i < yBins; ++i)
        {
            currentX = tlX + (binSize/2);
            for(boost::uint_fast32_t j = 0; j < xBins; ++j)
            {
                if(ptGrid[i][j]->size() > 0)
                {
                    pt = new SPDPoint();
                    ptUtils.initSPDPoint(pt);
                    pt->x = currentX;
                    pt->y = currentY;
                    zVal = 0;
                    
                    if(elevVal == SPD_USE_Z)
                    {
                        for(iterPts = ptGrid[i][j]->begin(); iterPts != ptGrid[i][j]->end(); ++iterPts)
                        {
                            zVal += (*iterPts)->z;
                        }
                        pt->z = zVal/ptGrid[i][j]->size();
                    }
                    else if(elevVal == SPD_USE_HEIGHT)
                    {
                        for(iterPts = ptGrid[i][j]->begin(); iterPts != ptGrid[i][j]->end(); ++iterPts)
                        {
                            zVal += (*iterPts)->height;
                        }
                        pt->height = zVal/ptGrid[i][j]->size();
                    }
                    else
                    {
                        throw SPDProcessingException("Do not recognise elevation type (needs to be height or Z).");
                    }
                    
                    //std::cout << "ptGrid[" << i << "][" << j << "] = {" << pt->x << "," << pt->y << "}\n";
                    
                    ptGrid[i][j]->clear();
                    ptGrid[i][j]->push_back(pt);
                }
                else
                {
                    ptGrid[i][j]->clear();
                }
                currentX = currentX + binSize;
            }
            currentY = currentY - binSize;
            //std::cout << std::endl;
        }
        
        deletePtsInBins = true;
    }
    
    void SPDPointGridIndex::getAllPointsInGrid(std::vector<SPDPoint*> *pts) throw(SPDProcessingException)
    {
        std::vector<SPDPoint*>::iterator iterPts;
        for(boost::uint_fast32_t i = 0; i < yBins; ++i)
        {
            for(boost::uint_fast32_t j = 0; j < xBins; ++j)
            {
                for(iterPts = ptGrid[i][j]->begin(); iterPts != ptGrid[i][j]->end(); ++iterPts)
                {
                    pts->push_back(*iterPts);
                }
            }
        }
    }
    
	SPDPointGridIndex::~SPDPointGridIndex()
	{
		if(this->ptGrid != NULL)
        {
            for(boost::uint_fast32_t i = 0; i < this->yBins; ++i)
            {
                for(boost::uint_fast32_t j = 0; j < this->xBins; ++j)
                {
                    if(deletePtsInBins)
                    {
                        for(std::vector<SPDPoint*>::iterator iterPts = this->ptGrid[i][j]->begin(); iterPts != this->ptGrid[i][j]->end(); ++iterPts)
                        {
                            delete *iterPts;
                        }
                    }
                    delete this->ptGrid[i][j];
                }
                delete[] this->ptGrid[i];
            }
            delete[] this->ptGrid;
        }
	}
}
