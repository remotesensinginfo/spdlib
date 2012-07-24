/*
 *  SPDMultiscaleCurvatureGrdClassification.cpp
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

#include "spd/SPDMultiscaleCurvatureGrdClassification.h"

namespace spdlib
{
    

    SPDMultiscaleCurvatureGrdClassification::SPDMultiscaleCurvatureGrdClassification(float initScale,boost::uint_fast16_t numOfScalesAbove,boost::uint_fast16_t numOfScalesBelow, float scaleGaps, float initCurveTolerance, float minCurveTolerance, float stepCurveTolerance, float interpMaxRadius,boost::uint_fast16_t interpNumPoints, SPDSmoothFilterType filterType,boost::uint_fast16_t smoothFilterHSize, float thresOfChange, bool multiReturnPulsesOnly,boost::uint_fast16_t classParameters)
    {
        this->initScale = initScale;
        this->numOfScalesAbove = numOfScalesAbove;
        this->numOfScalesBelow = numOfScalesBelow;
        this->scaleGaps = scaleGaps;
        this->initCurveTolerance = initCurveTolerance;
        this->minCurveTolerance = minCurveTolerance;
        this->stepCurveTolerance = stepCurveTolerance;
        this->interpMaxRadius = interpMaxRadius; 
        this->interpNumPoints = interpNumPoints;
        this->filterType = filterType;
        this->smoothFilterHSize = smoothFilterHSize;
        this->thresOfChange = thresOfChange;
        this->multiReturnPulsesOnly = multiReturnPulsesOnly;
        this->classParameters = classParameters;
    }
    
    void SPDMultiscaleCurvatureGrdClassification::processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binSize) throw(SPDProcessingException)
    {
        try
        {
            boost::uint_fast32_t numScales = numOfScalesBelow + 1 + numOfScalesAbove;
            float scale = initScale + (numOfScalesAbove * scaleGaps); // Start at lowest resolution...
            std::cout << "scale = " << scale << std::endl;
            // Initialise point classification (all classified to ground) and find point cloud dimensions
            double *bbox = this->findDataExtentAndClassifyAllPtsAsGrd(pulses, xSize, ySize);
            
            std::cout << "BBOX: [" << bbox[0] << "," << bbox[1] << "," << bbox[2] << "," << bbox[3] << "]\n";
            
            boost::uint_fast32_t xSizeRaster = 0;
            boost::uint_fast32_t ySizeRaster = 0;
            float **raster = NULL;
            float proportionOfChange = 1;
            float curveTolerance = this->initCurveTolerance;
            
            for(boost::uint_fast32_t scaleCounter = 0; scaleCounter < numScales; ++scaleCounter)
            {
                std::cout << "Scale (" << scale << ") " << scaleCounter+1 << " of " << numScales << std::endl;
                proportionOfChange = 1;
                do
                {
                    raster = this->createElevationRaster(bbox, scale, &xSizeRaster, &ySizeRaster, pulses, xSize, ySize);
                    try
                    {
                        if(this->filterType == meanFilter)
                        {
                            this->smoothMeanRaster(raster, xSizeRaster, ySizeRaster, this->smoothFilterHSize);
                        }
                        else if(this->filterType == medianFilter)
                        {
                            this->smoothMedianRaster(raster, xSizeRaster, ySizeRaster, this->smoothFilterHSize);
                        }
                        else
                        {
                            throw SPDProcessingException("Filter type is not recognised.");
                        }
                        
                        proportionOfChange = this->classifyNonGrdPoints(curveTolerance, bbox, scale, raster, xSizeRaster, ySizeRaster, pulses, xSize, ySize);
                        
                        for(boost::uint_fast32_t i = 0; i < ySizeRaster; ++i)
                        {
                            delete[] raster[i];
                        }
                        delete[] raster;
                    }
                    catch(SPDProcessingException &e)
                    {
                        for(boost::uint_fast32_t i = 0; i < ySizeRaster; ++i)
                        {
                            delete[] raster[i];
                        }
                        delete[] raster;
                        
                        throw e;
                    }
                    std::cout << "Change:\t" << proportionOfChange << std::endl;
                }while(proportionOfChange > thresOfChange);
                
                scale -= scaleGaps; // decrement the scale parameter
                curveTolerance -= stepCurveTolerance;
                if(curveTolerance < minCurveTolerance)
                {
                    curveTolerance = minCurveTolerance;
                }
            }
            
            delete[] bbox;
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
    }
    
    
    double* SPDMultiscaleCurvatureGrdClassification::findDataExtentAndClassifyAllPtsAsGrd(std::vector<SPDPulse*> ***pulses, boost::uint_fast32_t xSizePulses, boost::uint_fast32_t ySizePulses) throw(SPDProcessingException)
    {
        // bbox - TLX TLY BRX BRY
        double *bbox = new double[4];
        try
        {
            bool first = true;
            
            std::vector<SPDPulse*>::iterator iterPulses;
            std::vector<SPDPoint*>::iterator iterPoints;
            for(boost::uint_fast32_t i = 0; i < ySizePulses; ++i)
            {
                for(boost::uint_fast32_t j = 0; j < xSizePulses; ++j)
                {
                    for(iterPulses = pulses[i][j]->begin(); iterPulses != pulses[i][j]->end(); ++iterPulses)
                    {
                        if((*iterPulses)->numberOfReturns > 0)
                        {
                            for(iterPoints = (*iterPulses)->pts->begin(); iterPoints != (*iterPulses)->pts->end(); ++iterPoints)
                            {
                                if(first)
                                {
                                    bbox[0] = (*iterPoints)->x;
                                    bbox[1] = (*iterPoints)->y;
                                    bbox[2] = (*iterPoints)->x;
                                    bbox[3] = (*iterPoints)->y;
                                    first = false;
                                }
                                else
                                {
                                    if((*iterPoints)->x < bbox[0])
                                    {
                                        bbox[0] = (*iterPoints)->x; 
                                    }
                                    else if((*iterPoints)->x > bbox[2])
                                    {
                                        bbox[2] = (*iterPoints)->x; 
                                    }
                                    
                                    if((*iterPoints)->y > bbox[1])
                                    {
                                        bbox[1] = (*iterPoints)->y; 
                                    }
                                    else if((*iterPoints)->y < bbox[3])
                                    {
                                        bbox[3] = (*iterPoints)->y; 
                                    }
                                }
                                
                                if((classParameters != SPD_GROUND) && ((*iterPoints)->classification == SPD_GROUND))
                                {
                                    (*iterPoints)->classification = SPD_UNCLASSIFIED;
                                }
                                
                                if(classParameters == SPD_ALL_CLASSES)
                                {
                                    (*iterPoints)->classification = SPD_GROUND;
                                }
                                else if(classParameters == (*iterPoints)->classification)
                                {
                                    (*iterPoints)->classification = SPD_GROUND;
                                }
                            }
                        }
                    }
                }
            }
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
        
        return bbox;
    }
    
    float** SPDMultiscaleCurvatureGrdClassification::createElevationRaster(double *bbox, float rasterScale, boost::uint_fast32_t *xSizeRaster, boost::uint_fast32_t *ySizeRaster, std::vector<SPDPulse*> ***pulses, boost::uint_fast32_t xSizePulses, boost::uint_fast32_t ySizePulses) throw(SPDProcessingException)
    {
        float **raster = NULL;
        try 
        {
            boost::uint_fast32_t roundingAddition = 0;
            try
            {
                if(rasterScale < 1)
                {
                    roundingAddition = boost::numeric_cast<boost::uint_fast32_t>(1/rasterScale);
                }
                else 
                {
                    roundingAddition = 1;
                }
                
                *xSizeRaster = boost::numeric_cast<boost::uint_fast32_t>(((bbox[2] - bbox[0])/rasterScale)+roundingAddition);
                *ySizeRaster = boost::numeric_cast<boost::uint_fast32_t>(((bbox[1] - bbox[3])/rasterScale)+roundingAddition);
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
            
            if((*xSizeRaster < 1) | (*ySizeRaster < 1))
            {
                throw SPDProcessingException("The raster has a size less than 1 'pixel' in one of the axis' (try reducing resolution).");
            }
            
            //std::cout << "Size Raster [" << *xSizeRaster << "," << *ySizeRaster << "]\n";
            
            SPDTPSNumPtsUseAvThinInterpolator *interpolator = new SPDTPSNumPtsUseAvThinInterpolator((interpMaxRadius*rasterScale), interpNumPoints, SPD_USE_Z, rasterScale, true);
            interpolator->initInterpolator(pulses, xSizePulses, ySizePulses, SPD_GROUND);
            
            double currentX = bbox[0] + (rasterScale/2);
            double currentY = bbox[1] - (rasterScale/2);
            raster = new float*[(*ySizeRaster)];
            
            boost::uint_fast32_t feedback = (*ySizeRaster)/10;
			boost::uint_fast32_t feedbackCounter = 0;
			std::cout << "Started " << std::flush;
            for(boost::uint_fast32_t i = 0; i < (*ySizeRaster); ++i)
            {
                if(((*ySizeRaster) >= 10) && ((i % feedback) == 0))
				{
					std::cout << "." << feedbackCounter << "." << std::flush;
					feedbackCounter = feedbackCounter + 10;
				}
                currentX = bbox[0] + (rasterScale/2);
                raster[i] = new float[(*xSizeRaster)];
                for(boost::uint_fast32_t j = 0; j < (*xSizeRaster); ++j)
                {
                    //std::cout << "pxl [" << currentX << "," << currentY << "]\n";
                    raster[i][j] = interpolator->getValue(currentX, currentY);
                    currentX = currentX + rasterScale;
                }
                currentY = currentY - rasterScale;
            }
            std::cout << " Complete.\n";
            
            delete interpolator;
        } 
        catch (SPDProcessingException &e) 
        {
            throw e;
        }
        
        return raster;
    }
    
    void SPDMultiscaleCurvatureGrdClassification::smoothMeanRaster(float **raster, boost::uint_fast32_t xSizeRaster, boost::uint_fast32_t ySizeRaster, boost::uint_fast16_t filterHSize) throw(SPDProcessingException)
    {
        try
        {
            boost::uint_fast32_t filterPxlStartX = 0;
            boost::uint_fast32_t filterPxlEndX = 0;
            boost::uint_fast32_t filterPxlStartY = 0;
            boost::uint_fast32_t filterPxlEndY = 0;
            
            float **rasterMedian = new float*[ySizeRaster];
            for(boost::uint_fast32_t i = 0; i < ySizeRaster; ++i)
            {
                rasterMedian[i] = new float[xSizeRaster];
                for(boost::uint_fast32_t j = 0; j < xSizeRaster; ++j)
                {
                    rasterMedian[i][j] = 0;
                }
            }
            
            double sumValues = 0;
            boost::uint_fast32_t valCount = 0;
            
            for(boost::uint_fast32_t i = 0; i < ySizeRaster; ++i)
            {
                for(boost::uint_fast32_t j = 0; j < xSizeRaster; ++j)
                {
                    if((((int_fast64_t)i) - ((int_fast64_t)filterHSize)) < 0)
                    {
                        filterPxlStartY = 0;
                        filterPxlEndY = i + filterHSize;
                    }
                    else if((i+filterHSize) >= ySizeRaster)
                    {
                        filterPxlStartY = i - filterHSize;
                        filterPxlEndY = (ySizeRaster-1);
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
                    else if((j+filterHSize) >= xSizeRaster)
                    {
                        filterPxlStartX = j - filterHSize;
                        filterPxlEndX = xSizeRaster-1;
                    }
                    else
                    {
                        filterPxlStartX = j - filterHSize;
                        filterPxlEndX = j + filterHSize;
                    }
                    
                    sumValues = 0;
                    valCount = 0;
                    
                    for(boost::uint_fast32_t n = filterPxlStartY; n <= filterPxlEndY; ++n)
                    {
                        for(boost::uint_fast32_t m = filterPxlStartX; m <= filterPxlEndX; ++m)
                        {
                            if(!std::isnan(raster[n][m]))
                            {
                                sumValues += raster[n][m];
                                ++valCount;
                            }
                        }
                    }
                    
                    if(valCount == 1)
                    {
                        rasterMedian[i][j] = sumValues;
                    }
                    else if(valCount > 1)
                    {
                        rasterMedian[i][j] = sumValues/valCount;
                    }
                    else
                    {
                        rasterMedian[i][j] = std::numeric_limits<float>::signaling_NaN();
                    }
                }
            }
            
            for(boost::uint_fast32_t i = 0; i < ySizeRaster; ++i)
            {
                for(boost::uint_fast32_t j = 0; j < xSizeRaster; ++j)
                {
                    raster[i][j] = rasterMedian[i][j];
                }
                delete[] rasterMedian[i];
            }
            delete[] rasterMedian;
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
    }
    
    void SPDMultiscaleCurvatureGrdClassification::smoothMedianRaster(float **raster, boost::uint_fast32_t xSizeRaster, boost::uint_fast32_t ySizeRaster, boost::uint_fast16_t filterHSize) throw(SPDProcessingException)
    {
        try
        {
            boost::uint_fast32_t filterPxlStartX = 0;
            boost::uint_fast32_t filterPxlEndX = 0;
            boost::uint_fast32_t filterPxlStartY = 0;
            boost::uint_fast32_t filterPxlEndY = 0;
            
            boost::uint_fast32_t maxNumFilterValues = ((filterHSize * 2)+1)*((filterHSize * 2)+1);
            
            float **rasterMedian = new float*[ySizeRaster];
            for(boost::uint_fast32_t i = 0; i < ySizeRaster; ++i)
            {
                rasterMedian[i] = new float[xSizeRaster];
                for(boost::uint_fast32_t j = 0; j < xSizeRaster; ++j)
                {
                    rasterMedian[i][j] = 0;
                }
            }
            
            std::vector<float> *elevValues = new std::vector<float>();
            elevValues->reserve(maxNumFilterValues);
            for(boost::uint_fast32_t i = 0; i < ySizeRaster; ++i)
            {
                for(boost::uint_fast32_t j = 0; j < xSizeRaster; ++j)
                {
                    if((((int_fast64_t)i) - ((int_fast64_t)filterHSize)) < 0)
                    {
                        filterPxlStartY = 0;
                        filterPxlEndY = i + filterHSize;
                    }
                    else if((i+filterHSize) >= ySizeRaster)
                    {
                        filterPxlStartY = i - filterHSize;
                        filterPxlEndY = (ySizeRaster-1);
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
                    else if((j+filterHSize) >= xSizeRaster)
                    {
                        filterPxlStartX = j - filterHSize;
                        filterPxlEndX = xSizeRaster-1;
                    }
                    else
                    {
                        filterPxlStartX = j - filterHSize;
                        filterPxlEndX = j + filterHSize;
                    }
                    
                    elevValues->clear();
                    
                    for(boost::uint_fast32_t n = filterPxlStartY; n <= filterPxlEndY; ++n)
                    {
                        for(boost::uint_fast32_t m = filterPxlStartX; m <= filterPxlEndX; ++m)
                        {
                            if(!std::isnan(raster[n][m]))
                            {
                                elevValues->push_back(raster[n][m]);
                            }
                        }
                    }
                    
                    if(elevValues->size() == 1)
                    {
                        rasterMedian[i][j] = elevValues->at(0);
                    }
                    else if(elevValues->size() > 1)
                    {
                        std::sort(elevValues->begin(), elevValues->end());
                        rasterMedian[i][j] = elevValues->at(elevValues->size()/2);
                    }
                    else
                    {
                        rasterMedian[i][j] = std::numeric_limits<float>::signaling_NaN();
                    }
                }
            }
            
            for(boost::uint_fast32_t i = 0; i < ySizeRaster; ++i)
            {
                for(boost::uint_fast32_t j = 0; j < xSizeRaster; ++j)
                {
                    raster[i][j] = rasterMedian[i][j];
                }
                delete[] rasterMedian[i];
            }
            delete[] rasterMedian;
            
            delete elevValues;
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
    }
    
    float SPDMultiscaleCurvatureGrdClassification::classifyNonGrdPoints(float curveTolerance, double *bbox, float rasterScale, float **raster, boost::uint_fast32_t xSizeRaster, boost::uint_fast32_t ySizeRaster, std::vector<SPDPulse*> ***pulses, boost::uint_fast32_t xSizePulses, boost::uint_fast32_t ySizePulses) throw(SPDProcessingException)
    {
        boost::uint_fast64_t totalNumPoints = 0;
        boost::uint_fast64_t numPointsChanged = 0;
        
        try 
		{
            boost::uint_fast32_t xIdx = 0;
			boost::uint_fast32_t yIdx = 0;
            double xDiff = 0;
            double yDiff = 0;
            std::vector<SPDPulse*>::iterator iterPulses;
            std::vector<SPDPoint*>::iterator iterPoints;
            for(boost::uint_fast32_t i = 0; i < ySizePulses; ++i)
            {
                for(boost::uint_fast32_t j = 0; j < xSizePulses; ++j)
                {
                    for(iterPulses = pulses[i][j]->begin(); iterPulses != pulses[i][j]->end(); ++iterPulses)
                    {
                        if((*iterPulses)->numberOfReturns > 0)
                        {
                            for(iterPoints = (*iterPulses)->pts->begin(); iterPoints != (*iterPulses)->pts->end(); ++iterPoints)
                            {
                                xDiff = ((*iterPoints)->x - bbox[0])/rasterScale;
                                yDiff = (bbox[1] - (*iterPoints)->y)/rasterScale;
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
                                
                                if(xIdx > (xSizeRaster-1))
                                {
                                    std::cout << "Point: [" << (*iterPoints)->x << "," << (*iterPoints)->y << "]\n";
                                    std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
                                    std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
                                    std::cout << "Size [" << xSizeRaster << "," << ySizeRaster << "]\n";
                                    throw SPDProcessingException("Point did not fit within raster (X Axis).");
                                }
                                
                                if(yIdx > (ySizeRaster-1))
                                {
                                    std::cout << "Point: [" << (*iterPoints)->x << "," << (*iterPoints)->y << "]\n";
                                    std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
                                    std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
                                    std::cout << "Size [" << xSizeRaster << "," << ySizeRaster << "]\n";
                                    throw SPDProcessingException("Point did not fit within raster (Y Axis).");
                                }
                                
                                if(((*iterPoints)->z > (raster[yIdx][xIdx] + curveTolerance)) & ((*iterPoints)->classification == SPD_GROUND))
                                {
                                    (*iterPoints)->classification = SPD_UNCLASSIFIED;
                                    if(multiReturnPulsesOnly)
                                    {
                                        if((*iterPulses)->numberOfReturns > 1)
                                        {
                                            ++numPointsChanged;
                                        }
                                    }
                                    else
                                    {
                                        ++numPointsChanged;
                                    }
                                }
                                if(multiReturnPulsesOnly)
                                {
                                    if((*iterPulses)->numberOfReturns > 1)
                                    {
                                        ++totalNumPoints;
                                    }
                                }
                                else
                                {
                                    ++totalNumPoints;
                                }
                            }
                        }
                    }
                }
            }
        }
        catch (SPDProcessingException &e) 
        {
            throw e;
        }
        
        return ((double)numPointsChanged)/((double)totalNumPoints);
    }
    
    
    SPDMultiscaleCurvatureGrdClassification::~SPDMultiscaleCurvatureGrdClassification()
    {
        
    }
    
    
    
    
    
    
    SPDTPSNumPtsUseAvThinInterpolator::SPDTPSNumPtsUseAvThinInterpolator(float radius, boost::uint_fast16_t numPoints, boost::uint_fast16_t elevVal, double gridResolution, bool thinGrid): initialised(false), idx(NULL), pts(NULL), gridResolution(0), thinGrid(false), radius(0), numPoints(12), elevVal(0)
	{
		this->radius = radius;
        this->numPoints = numPoints;
        this->elevVal = elevVal;
        this->gridResolution = gridResolution;
        this->thinGrid = thinGrid;
	}
    
    void SPDTPSNumPtsUseAvThinInterpolator::initInterpolator(std::list<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException)
	{
		try 
		{
			pts = new std::vector<SPDPoint*>();
			std::list<SPDPulse*>::iterator iterPulses;
			std::vector<SPDPoint*>::iterator iterPts;
            SPDPoint *pt = NULL;
			for(boost::uint_fast32_t i = 0; i < numYBins; ++i)
			{
				for(boost::uint_fast32_t j = 0; j < numXBins; ++j)
				{
					for(iterPulses = pulses[i][j]->begin(); iterPulses != pulses[i][j]->end(); ++iterPulses)
					{
						if((*iterPulses)->numberOfReturns > 0)
						{
							if(ptClass == SPD_VEGETATION_TOP)
                            {
                                pt = (*iterPulses)->pts->front();
                                if((pt->classification == SPD_HIGH_VEGETATION) |
                                   (pt->classification == SPD_MEDIUM_VEGETATION) |
                                   (pt->classification == SPD_LOW_VEGETATION))
                                {
                                    pts->push_back(pt);
                                }
                            }
                            else if(ptClass == SPD_ALL_CLASSES_TOP)
                            {
                                pts->push_back((*iterPulses)->pts->front());
                            }
                            else
                            {
                                for(iterPts = (*iterPulses)->pts->begin(); iterPts != (*iterPulses)->pts->end(); ++iterPts)
                                {
                                    if(ptClass == SPD_ALL_CLASSES)
                                    {
                                        pts->push_back(*iterPts);
                                    }
                                    else if((*iterPts)->classification == ptClass)
                                    {
                                        pts->push_back(*iterPts);
                                    }
                                }
                            }
						}
					}
				}
			}
			
			idx = new SPDPointGridIndex();
			idx->buildIndex(pts, this->gridResolution);
            
            if(thinGrid)
            {
                idx->thinPtsWithAvZ(this->elevVal);
            }
            
            initialised = true;
		}
		catch (SPDProcessingException &e) 
		{
			throw e;
		}
	}
	
	void SPDTPSNumPtsUseAvThinInterpolator::initInterpolator(std::vector<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException)
	{
		try 
		{
			pts = new std::vector<SPDPoint*>();
			std::vector<SPDPulse*>::iterator iterPulses;
			std::vector<SPDPoint*>::iterator iterPts;
            SPDPoint *pt = NULL;
			for(boost::uint_fast32_t i = 0; i < numYBins; ++i)
			{
				for(boost::uint_fast32_t j = 0; j < numXBins; ++j)
				{
					for(iterPulses = pulses[i][j]->begin(); iterPulses != pulses[i][j]->end(); ++iterPulses)
					{
						if((*iterPulses)->numberOfReturns > 0)
						{
							if(ptClass == SPD_VEGETATION_TOP)
                            {
                                pt = (*iterPulses)->pts->front();
                                if((pt->classification == SPD_HIGH_VEGETATION) |
                                   (pt->classification == SPD_MEDIUM_VEGETATION) |
                                   (pt->classification == SPD_LOW_VEGETATION))
                                {
                                    pts->push_back(pt);
                                }
                            }
                            else if(ptClass == SPD_ALL_CLASSES_TOP)
                            {
                                pts->push_back((*iterPulses)->pts->front());
                            }
                            else
                            {
                                for(iterPts = (*iterPulses)->pts->begin(); iterPts != (*iterPulses)->pts->end(); ++iterPts)
                                {
                                    if(ptClass == SPD_ALL_CLASSES)
                                    {
                                        pts->push_back(*iterPts);
                                    }
                                    else if((*iterPts)->classification == ptClass)
                                    {
                                        pts->push_back(*iterPts);
                                    }
                                }
                            }
						}
					}
				}
			}
			
			idx = new SPDPointGridIndex();
			idx->buildIndex(pts, this->gridResolution);
            
            if(thinGrid)
            {
                idx->thinPtsWithAvZ(this->elevVal);
            }
            
            initialised = true;
		}
		catch (SPDProcessingException &e) 
		{
			throw e;
		}
	}
	
	void SPDTPSNumPtsUseAvThinInterpolator::initInterpolator(std::list<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException)
	{
        try 
        {
            pts = new std::vector<SPDPoint*>();
            
            std::list<SPDPulse*>::iterator iterPulses;
            std::vector<SPDPoint*>::iterator iterPts;
            SPDPoint *pt = NULL;
            for(iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
            {
                if((*iterPulses)->numberOfReturns > 0)
                {
                    if(ptClass == SPD_VEGETATION_TOP)
                    {
                        pt = (*iterPulses)->pts->front();
                        if((pt->classification == SPD_HIGH_VEGETATION) |
                           (pt->classification == SPD_MEDIUM_VEGETATION) |
                           (pt->classification == SPD_LOW_VEGETATION))
                        {
                            pts->push_back(pt);
                        }
                    }
                    else if(ptClass == SPD_ALL_CLASSES_TOP)
                    {
                        pts->push_back((*iterPulses)->pts->front());
                    }
                    else
                    {
                        for(iterPts = (*iterPulses)->pts->begin(); iterPts != (*iterPulses)->pts->end(); ++iterPts)
                        {
                            if(ptClass == SPD_ALL_CLASSES)
                            {
                                pts->push_back(*iterPts);
                            }
                            else if((*iterPts)->classification == ptClass)
                            {
                                pts->push_back(*iterPts);
                            }
                        }
                    }
                }
            }
            
            idx = new SPDPointGridIndex();
            idx->buildIndex(pts, this->gridResolution);
            
            if(thinGrid)
            {
                idx->thinPtsWithAvZ(this->elevVal);
            }
            
            initialised = true;
        } 
        catch (SPDProcessingException &e) 
        {
            throw e;
        }
	}
	
	void SPDTPSNumPtsUseAvThinInterpolator::initInterpolator(std::vector<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException)
	{
        try
        {
            pts = new std::vector<SPDPoint*>();
            
            std::vector<SPDPulse*>::iterator iterPulses;
            std::vector<SPDPoint*>::iterator iterPts;
            SPDPoint *pt = NULL;
            for(iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
            {
                if((*iterPulses)->numberOfReturns > 0)
                {
                    if(ptClass == SPD_VEGETATION_TOP)
                    {
                        pt = (*iterPulses)->pts->front();
                        if((pt->classification == SPD_HIGH_VEGETATION) |
                           (pt->classification == SPD_MEDIUM_VEGETATION) |
                           (pt->classification == SPD_LOW_VEGETATION))
                        {
                            pts->push_back(pt);
                        }
                    }
                    else if(ptClass == SPD_ALL_CLASSES_TOP)
                    {
                        pts->push_back((*iterPulses)->pts->front());
                    }
                    else
                    {
                        for(iterPts = (*iterPulses)->pts->begin(); iterPts != (*iterPulses)->pts->end(); ++iterPts)
                        {
                            if(ptClass == SPD_ALL_CLASSES)
                            {
                                pts->push_back(*iterPts);
                            }
                            else if((*iterPts)->classification == ptClass)
                            {
                                pts->push_back(*iterPts);
                            }
                        }
                    }
                }
            }
            
            idx = new SPDPointGridIndex();
            idx->buildIndex(pts, this->gridResolution);
            
            if(thinGrid)
            {
                idx->thinPtsWithAvZ(this->elevVal);
            }
            
            initialised = true;
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
	}
    
	void SPDTPSNumPtsUseAvThinInterpolator::resetInterpolator() throw(SPDProcessingException)
	{
		if(initialised)
		{
			delete idx;
			pts->clear();
			delete pts;
		}
	}
	
	float SPDTPSNumPtsUseAvThinInterpolator::getValue(double eastings, double northings) throw(SPDProcessingException)
	{
        float newZValue = 0;
        if(initialised)
		{
            std::vector<SPDPoint*> *ptsInRadius = new std::vector<SPDPoint*>();
            try 
            {
                if(idx->getSetNumOfPoints(ptsInRadius, eastings, northings, numPoints, radius))
                {
                    std::vector<spdlib::tps::Vec> cntrlPts(ptsInRadius->size());
                    int ptIdx = 0;
                    for(std::vector<SPDPoint*>::iterator iterPts = ptsInRadius->begin(); iterPts != ptsInRadius->end(); ++iterPts)
                    {
                        // Please note that Z and Y and been switch around as the TPS code (tpsdemo) 
                        // interpolates for Y rather than Z.
                        if(elevVal == SPD_USE_Z)
                        {
                            cntrlPts[ptIdx++] = spdlib::tps::Vec((*iterPts)->x, (*iterPts)->z, (*iterPts)->y);
                        }
                        else if(elevVal == SPD_USE_HEIGHT)
                        {
                            cntrlPts[ptIdx++] = spdlib::tps::Vec((*iterPts)->x, (*iterPts)->height, (*iterPts)->y);
                        }
                        else
                        {
                            throw SPDProcessingException("Elevation type not recognised.");
                        }
                    }
                    
                    //std::cout.precision(12);
                    //for(unsigned int i = 0; i < cntrlPts.size(); ++i)
                    //{
                    //    std::cout << "pt[" << i << "]:\t" << cntrlPts[i].x << ", " << cntrlPts[i].y << ", " << cntrlPts[i].z << std::endl;
                    //}
                    
                    spdlib::tps::Spline splineFunc = spdlib::tps::Spline(cntrlPts, 0.0);
                    newZValue = splineFunc.interpolate_height(eastings, northings);
                    //std::cout << "New Value = " << newZValue << std::endl;
                    //std::cout << std::endl << std::endl;
                }
                else
                {
                    newZValue = std::numeric_limits<float>::signaling_NaN();
                }
            }
            catch(spdlib::tps::SingularMatrixError &e)
            {
                //throw SPDProcessingException(e.what()); // Ignore and pass an NAN value back.
                newZValue = std::numeric_limits<float>::signaling_NaN();
            }
            catch (SPDProcessingException &e) 
            {
                throw e;
            }
            delete ptsInRadius;
        }
        else
        {
            throw SPDProcessingException("Interpolator has not been initialised.");
        }
        
        return newZValue;
	}
	
	SPDTPSNumPtsUseAvThinInterpolator::~SPDTPSNumPtsUseAvThinInterpolator()
	{
		if(initialised)
		{
			delete idx;
			pts->clear();
			delete pts;
		}
	}
    
}




