/*
 *  SPDMetrics.cpp
 *  SPDLIB
 *
 *  Created by Pete Bunting on 26/02/2011.
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
 *
 */

#include "spd/SPDMetrics.h"


namespace spdlib{
    
    
    /*
     * Metric's with are neither height, Amplitude or range
     */
    
    double SPDMetricCalcNumPulses::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        boost::uint_fast64_t numPulses = 0;
        if(minNumReturns == 0)
        {
            numPulses = pulses->size();
        }
        else
        {
            if((spdFile->getDecomposedPtDefined() == SPD_TRUE) | (spdFile->getDiscretePtDefined() == SPD_TRUE))
            {
                for(vector<SPDPulse*>::iterator iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
                {
                    if((*iterPulses)->numberOfReturns >= minNumReturns)
                    {
                        ++numPulses;
                    }
                }
            }
            else if(spdFile->getReceiveWaveformDefined() == SPD_TRUE)
            {
                for(vector<SPDPulse*>::iterator iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
                {
                    if((*iterPulses)->numOfReceivedBins >= minNumReturns)
                    {
                        ++numPulses;
                    }
                }
            }
            else
            {
                throw SPDProcessingException("Neither waveform or point returns have been defind.");
            }
        }
        return numPulses;
    }
    
    
    double SPDMetricCalcCanopyCover::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        double canopyCover = 0;
        try
        {
            if(pulses->size() > 0)
            {
                // 1) Get points associated with parameters provided.
                vector<SPDPoint*> *points = this->getPointsWithinHeightParameters(pulses, spdFile, geom);
                
                // 2) Check points > 0
                if(points->size() > 0)
                {
                    // 3) Get envelope of the geom
                    OGREnvelope *env = new OGREnvelope();
                    geom->getEnvelope(env);
                    
                    // 4) Create grid index of points.
                    SPDPointGridIndex *gridIdx = new SPDPointGridIndex();
                    gridIdx->buildIndex(points, resolution, env);
                    
                    // 5) Iterate through cell centres.
                    OGRPoint *pt = new OGRPoint();
                    double cellX = env->MinX + (resolution/2);
                    double cellY = env->MaxY - (resolution/2);
                    boost::uint_fast32_t xBins = gridIdx->getXBins();
                    boost::uint_fast32_t yBins = gridIdx->getYBins();
                    
                    vector<SPDPoint*> *ptsInRadius = new vector<SPDPoint*>();
                    vector<SPDPoint*>::iterator iterPts;
                    boost::uint_fast64_t allCellCount = 0;
                    boost::uint_fast64_t canopyCellCount = 0;
                    for(boost::uint_fast32_t i = 0; i < yBins; ++i)
                    {
                        cellX = env->MinX + (resolution/2);
                        for(boost::uint_fast32_t j = 0; j < xBins; ++j)
                        {
                            // 6) Check point is within geometry
                            pt->setX(cellX);
                            pt->setY(cellY);
                            if(geom->Contains(pt))
                            {
                                ++allCellCount;
                                
                                // 7) Get value for each cell and if value > 0 then increment cover counter
                                gridIdx->getPointsInRadius(ptsInRadius, cellX, cellY, radius);
                                if(ptsInRadius->size() > 0)
                                {
                                    ++canopyCellCount;
                                }
                                ptsInRadius->clear();
                            }
                            cellX += resolution;
                        }
                        cellY -= resolution;
                    }
                    delete pt;
                    delete ptsInRadius;
                    
                    // 8) Output canopy cover. (in metre squared).
                    canopyCover = ((double)canopyCellCount) * resolution;
                    
                    delete gridIdx;
                }
                else
                {
                    canopyCover = 0;
                }
                
                delete points;
            }
            else 
            {
                canopyCover = 0;
            }
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
        return canopyCover;
    }
    
    double SPDMetricCalcCanopyCoverPercent::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        double canopyCover = 0;
        try
        {
            if(pulses->size() > 0)
            {
                // 1) Get points associated with parameters provided.
                vector<SPDPoint*> *points = this->getPointsWithinHeightParameters(pulses, spdFile, geom);
                
                // 2) Check points > 0
                if(points->size() > 0)
                {
                    // 3) Get envelope of the geom
                    OGREnvelope *env = new OGREnvelope();
                    geom->getEnvelope(env);
                    
                    // 4) Create grid index of points.
                    SPDPointGridIndex *gridIdx = new SPDPointGridIndex();
                    gridIdx->buildIndex(points, resolution, env);
                    
                    // 5) Iterate through cell centres.
                    OGRPoint *pt = new OGRPoint();
                    double cellX = env->MinX + (resolution/2);
                    double cellY = env->MaxY - (resolution/2);
                    boost::uint_fast32_t xBins = gridIdx->getXBins();
                    boost::uint_fast32_t yBins = gridIdx->getYBins();
                    
                    vector<SPDPoint*> *ptsInRadius = new vector<SPDPoint*>();
                    //vector<SPDPoint*>::iterator iterPts;
                    boost::uint_fast64_t allCellCount = 0;
                    boost::uint_fast64_t canopyCellCount = 0;
                    for(boost::uint_fast32_t i = 0; i < yBins; ++i)
                    {
                        cellX = env->MinX + (resolution/2);
                        for(boost::uint_fast32_t j = 0; j < xBins; ++j)
                        {
                            // 6) Check point is within geometry
                            pt->setX(cellX);
                            pt->setY(cellY);
                            //cout << "Cell [" << i << "," << j << "]\t[" << cellX << "," << cellY << "] - ";
                            if(geom->Contains(pt))
                            {
                                //cout << "CONTAINED\n";
                                ++allCellCount;
                                
                                // 7) Get value for each cell and if value > 0 then increment cover counter
                                gridIdx->getPointsInRadius(ptsInRadius, cellX, cellY, radius);
                                if(ptsInRadius->size() > 0)
                                {
                                    ++canopyCellCount;
                                }
                                ptsInRadius->clear();
                            }
                                // else
                                // {
                                //    cout << "NOT CONTAINED\n";
                                //}
                            cellX += resolution;
                        }
                        //cout << endl << endl;
                        cellY -= resolution;
                    }
                    delete pt;
                    delete ptsInRadius;
                    
                    //cout << "Number of Cells = "  << allCellCount << " of which " << canopyCellCount << " are canopy\n";
                    
                    // 8) Output canopy cover. (as percentage).
                    canopyCover = (((double)canopyCellCount) / ((double)allCellCount)) * 100;
                    
                    delete gridIdx;
                }
                else
                {
                    canopyCover = 0;
                }
                
                delete points;
            }
            else 
            {
                canopyCover = 0;
            }
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
        
        //cout << "Returning = " << canopyCover << endl;
        return canopyCover;
    }
    
    
    
    
    /*
     * Metric's for height
     */
    
    double SPDMetricCalcLeeOpennessHeight::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        double openness = 0;
        
        if(ptVals->size() > 0)
		{
            double max = gsl_stats_max (&(*ptVals)[0], 1, ptVals->size());
            boost::uint_fast32_t nBins = ceil(max/vRes)+1;
            
            //cout << "\nNumber of bins = " << nBins << endl;
            if(nBins > 0)
            {
                double *bins = new double[nBins];
                bool *binsFirst = new bool[nBins];
                for(boost::uint_fast32_t i = 0; i < nBins; ++i)
                {
                    bins[i] = 0;
                    binsFirst[i] = true;
                }
                
                boost::uint_fast32_t idx = 0;
                for(vector<double>::iterator iterVals = ptVals->begin(); iterVals != ptVals->end(); ++iterVals)
                {
                    try 
                    {
                        if((*iterVals) > 0)
                        {
                            idx = numeric_cast<boost::uint_fast32_t>((*iterVals)/vRes);
                        }
                    }
                    catch(negative_overflow& e) 
                    {
                        cout << "(*iterVals) = " << (*iterVals) << endl;
                        cout << "vRes = " << vRes << endl;
                        throw SPDProcessingException(e.what());
                    }
                    catch(positive_overflow& e) 
                    {
                        cout << "(*iterVals) = " << (*iterVals) << endl;
                        cout << "vRes = " << vRes << endl;
                        throw SPDProcessingException(e.what());
                    }
                    catch(bad_numeric_cast& e) 
                    {
                        cout << "(*iterVals) = " << (*iterVals) << endl;
                        cout << "vRes = " << vRes << endl;
                        throw SPDProcessingException(e.what());
                    }
                    
                    if(idx >= nBins)
                    {
                        cout << "Value = " << *iterVals << endl;
                        cout << "Max Value = " << max << endl;
                        cout << "idx = " << idx << endl;
                        cout << "nBins = " << nBins << endl;
                        cout << "vRes = " << vRes << endl;
                        
                        throw SPDProcessingException("Index is not within list.");
                    }
                    
                    if(binsFirst[idx])
                    {
                        bins[idx] = *iterVals;
                        binsFirst[idx] = false;
                    }
                    else if((*iterVals) > bins[idx])
                    {
                        bins[idx] = *iterVals;
                    }                
                }
                
                openness = 0;
                boost::uint_fast32_t numVoxels = 0;
                for(boost::uint_fast32_t i = 0; i < nBins; ++i)
                {
                    if(!binsFirst[i])
                    {
                        ++numVoxels;
                    }
                }
                //cout << "Number of voxels = " << numVoxels << endl;
                
                for(boost::uint_fast32_t i = 0; i < nBins; ++i)
                {
                    if(!binsFirst[i])
                    {
                        //cout << bins[i] << ",";
                        openness += (((max-bins[i])/max) / numVoxels);
                    }
                }
                //cout << endl;
                //cout << "openness = " << openness << endl;
                openness = openness * 100;
                //cout << "openness = " << openness << endl;
            }
            else
            {
                openness = NAN;
            }
        }
        else
        {
            openness = NAN;
        }
        delete ptVals;
        return openness;
    }
    
    double SPDMetricCalcNumReturnsHeight::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        boost::uint_fast64_t numReturns = ptVals->size();
        delete ptVals;
        return numReturns;
    }
    
    double SPDMetricCalcSumHeight::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        double sum = 0;
        for(vector<double>::iterator iterVals = ptVals->begin(); iterVals != ptVals->end(); ++iterVals)
        {
            sum += (*iterVals);
        }
        delete ptVals;
        return sum;
    }
    
    double SPDMetricCalcMeanHeight::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        double mean = 0;
        if(ptVals->size() > 0)
		{
            mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            mean = NAN;
        }
        delete ptVals;
        return mean;
    }

    double SPDMetricCalcMedianHeight::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        double median = 0;
        if(ptVals->size() > 0)
		{
            sort(ptVals->begin(), ptVals->end());
            /*for(unsigned int i = 0; i < ptVals->size(); ++i)
            {
                if( i == 0 )
                {
                    cout << &(*ptVals)[0][i];
                }
                else
                {
                    cout << ", " << &(*ptVals)[0][i];
                }
            }
            cout << endl << endl;*/
            median = gsl_stats_median_from_sorted_data(&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            median = NAN;
        }
        delete ptVals;
        return median;
    }

    double SPDMetricCalcModeHeight::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        double mode = 0;
        if(ptVals->size() > 0)
        {
            mode = this->calcBinnedMode(ptVals, resolution);
        }
        else
        {
            mode = NAN;
        }
        
        delete ptVals;
        return mode;
    }

    double SPDMetricCalcMinHeight::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        double min = 0;
        if(ptVals->size() > 0)
		{
            min = gsl_stats_min (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            min = NAN;
        }
        delete ptVals;
        return min;
    }

    double SPDMetricCalcMaxHeight::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        double max = 0;
        if(ptVals->size() > 0)
		{
            max = gsl_stats_max (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            max = NAN;
        }
        delete ptVals;
        return max;
    }
    
    
    double SPDMetricCalcDominantHeight::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        double dominantHeight = 0;
        try
        {
            if(pulses->size() > 0)
            {               
                double xMin = 0;
                double yMin = 0;
                double xMax = 0;
                double yMax = 0;
                bool first = true;
                
                for(vector<SPDPulse*>::iterator iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
                {
                    if(first)
                    {
                        xMin = (*iterPulses)->xIdx;
                        xMax = (*iterPulses)->xIdx;
                        yMin = (*iterPulses)->yIdx;
                        yMax = (*iterPulses)->yIdx;
                        first = false;
                    }
                    else
                    {
                        if((*iterPulses)->xIdx < xMin)
                        {
                            xMin = (*iterPulses)->xIdx;
                        }
                        else if((*iterPulses)->xIdx > xMax)
                        {
                            xMax = (*iterPulses)->xIdx;
                        }
                        
                        if((*iterPulses)->yIdx < yMin)
                        {
                            yMin = (*iterPulses)->yIdx;
                        }
                        else if((*iterPulses)->yIdx > yMax)
                        {
                            yMax = (*iterPulses)->yIdx;
                        }
                    }
                    
                    if((*iterPulses)->numberOfReturns > 0)
                    {
                        for(vector<SPDPoint*>::iterator iterPoints = (*iterPulses)->pts->begin(); iterPoints != (*iterPulses)->pts->end(); ++iterPoints)
                        {
                            if(first)
                            {
                                xMin = (*iterPoints)->x;
                                xMax = (*iterPoints)->x;
                                yMin = (*iterPoints)->y;
                                yMax = (*iterPoints)->y;
                                first = false;
                            }
                            else
                            {
                                if((*iterPoints)->x < xMin)
                                {
                                    xMin = (*iterPoints)->x;
                                }
                                else if((*iterPoints)->x > xMax)
                                {
                                    xMax = (*iterPoints)->x;
                                }
                                
                                if((*iterPoints)->y < yMin)
                                {
                                    yMin = (*iterPoints)->y;
                                }
                                else if((*iterPoints)->y > yMax)
                                {
                                    yMax = (*iterPoints)->y;
                                }
                            }
                        }
                    }
                }
                
                
                boost::uint_fast32_t roundingAddition = 0;
                if(spdFile->getBinSize() < 1)
                {
                    roundingAddition = numeric_cast<boost::uint_fast32_t>(1/resolution);
                }
                else 
                {
                    roundingAddition = 2;
                }
                
                double width = xMax - xMin;
                double height = yMax - yMin;
                
                boost::uint_fast32_t xBins = 0;
                boost::uint_fast32_t yBins = 0;
                
                try 
				{
					xBins = numeric_cast<boost::uint_fast32_t>((width/resolution))+roundingAddition;
                    yBins = numeric_cast<boost::uint_fast32_t>((height/resolution))+roundingAddition;
				}
				catch(negative_overflow& e) 
				{
					throw SPDProcessingException(e.what());
				}
				catch(positive_overflow& e) 
				{
					throw SPDProcessingException(e.what());
				}
				catch(bad_numeric_cast& e) 
				{
					throw SPDProcessingException(e.what());
				}
                
                if((xBins < 1) | (yBins < 1))
                {
                    throw SPDProcessingException("There insufficent number of bins for binning (try reducing resolution).");
                }
                
                vector<SPDPulse*> ***plsGrd = new vector<SPDPulse*>**[yBins];
                for(boost::uint_fast32_t i = 0; i < yBins; ++i)
                {
                    plsGrd[i] = new vector<SPDPulse*>*[xBins];
                    for(boost::uint_fast32_t j = 0; j < xBins; ++j)
                    {
                        plsGrd[i][j] = new vector<SPDPulse*>();
                    }
                }
                
                double xDiff = 0;
                double yDiff = 0;
                boost::uint_fast32_t xIdx = 0;
                boost::uint_fast32_t yIdx = 0;
                                
                for(vector<SPDPulse*>::iterator iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
                {
                    xDiff = ((*iterPulses)->xIdx - xMin)/resolution;
                    yDiff = (yMax - (*iterPulses)->yIdx)/resolution;				
                    
                    try 
                    {
                        xIdx = numeric_cast<boost::uint_fast32_t>(xDiff);
                        yIdx = numeric_cast<boost::uint_fast32_t>(yDiff);
                    }
                    catch(negative_overflow& e) 
                    {
                        throw SPDProcessingException(e.what());
                    }
                    catch(positive_overflow& e) 
                    {
                        throw SPDProcessingException(e.what());
                    }
                    catch(bad_numeric_cast& e) 
                    {
                        throw SPDProcessingException(e.what());
                    }
                    
                    if(xIdx > ((xBins)-1))
                    {
                        cout << "Point: [" << (*iterPulses)->xIdx << "," << (*iterPulses)->yIdx << "]\n";
                        cout << "Diff [" << xDiff << "," << yDiff << "]\n";
                        cout << "Index [" << xIdx << "," << yIdx << "]\n";
                        cout << "Size [" << xBins << "," << yBins << "]\n";
                        throw SPDProcessingException("Did not find x index within range.");
                    }
                    
                    if(yIdx > ((yBins)-1))
                    {
                        cout << "Point: [" << (*iterPulses)->xIdx << "," << (*iterPulses)->yIdx << "]\n";
                        cout << "Diff [" << xDiff << "," << yDiff << "]\n";
                        cout << "Index [" << xIdx << "," << yIdx << "]\n";
                        cout << "Size [" << xBins << "," << yBins << "]\n";
                        throw SPDProcessingException("Did not find y index within range.");
                    }
                    
                    plsGrd[yIdx][xIdx]->push_back((*iterPulses));
                }
                
                double heightSum = 0;
                boost::uint_fast32_t cellCount = 0;
                for(boost::uint_fast32_t i = 0; i < yBins; ++i)
                {
                    for(boost::uint_fast32_t j = 0; j < xBins; ++j)
                    {
                        if(plsGrd[i][j]->size() > 0)
                        {
                            vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(plsGrd[i][j], spdFile, geom);
                            if(ptVals->size() > 0)
                            {
                                heightSum += gsl_stats_max (&(*ptVals)[0], 1, ptVals->size());
                                ++cellCount;
                            }
                            delete ptVals;
                            delete plsGrd[i][j];
                        }
                    }
                    delete[] plsGrd[i];
                }
                delete[] plsGrd;
                
                if(cellCount == 0)
                {
                    dominantHeight = NAN;
                }
                else
                {
                    dominantHeight = heightSum/cellCount;
                }
            }
            else
            {
                dominantHeight = NAN;
            }
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
        return dominantHeight;
    }

    double SPDMetricCalcStdDevHeight::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        double stddev = 0;
        if(ptVals->size() > 0)
		{
            stddev = gsl_stats_sd (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            stddev = NAN;
        }
        delete ptVals;
        return stddev;
    }

    double SPDMetricCalcVarianceHeight::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        double variance = 0;
        if(ptVals->size() > 0)
		{
            variance = gsl_stats_variance (&(*ptVals)[0], 1, ptVals->size());
        }
        else 
        {
            variance = NAN;
        }
        delete ptVals;
        return variance;
    }

    double SPDMetricCalcAbsDeviationHeight::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        double absdev = 0;
        if(ptVals->size() > 0)
		{
            absdev = gsl_stats_absdev (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            absdev = NAN;
        }
        delete ptVals;
        return absdev;
    }

    double SPDMetricCalcCoefficientOfVariationHeight::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        double cv = 0;
        if(ptVals->size() > 0)
		{
            double sumSq = 0;
            double mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
            for(vector<double>::iterator iterVals; iterVals != ptVals->end(); ++iterVals)
            {
                sumSq += pow(((*iterVals) - mean),2);
            }
            cv = sqrt(sumSq/ptVals->size())/mean;
        }
        else 
        {
            cv = NAN;
        }

        delete ptVals;
        return cv;
    }

    double SPDMetricCalcPercentileHeight::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        /*
        cout << endl << endl << endl;
        vector<SPDPulse*>::iterator iterPulses;
        for(iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
        {
            if((*iterPulses)->numberOfReturns > 0)
            {
                for(vector<SPDPoint*>::iterator iterPts = (*iterPulses)->pts->begin(); iterPts != (*iterPulses)->pts->end(); ++iterPts)
                {
                    cout << (*iterPulses)->pulseID << "," << (*iterPts)->x << "," << (*iterPts)->y << "," << (*iterPts)->z << "," << (*iterPts)->height << "," << (*iterPts)->classification << endl;
                }
            }
        }
        cout << endl << endl << endl;
        */
        vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        double percentileVal = 0;
        if(ptVals->size() > 0)
		{
            double quatFrac = ((double)percentile)/100;
            sort(ptVals->begin(), ptVals->end());
            /*cout << "Calc Percentile " << quatFrac << endl;
            for(unsigned int i = 0; i < ptVals->size(); ++i)
            {
                if( i == 0 )
                {
                    cout << &(*ptVals)[0][i];
                }
                else
                {
                    cout << ", " << &(*ptVals)[0][i];
                }
            }
            cout << endl;*/
            percentileVal = gsl_stats_quantile_from_sorted_data(&(*ptVals)[0], 1, ptVals->size(), quatFrac);
            //cout << "Percentile = " << percentileVal << endl << endl;
        }
        else 
        {
            percentileVal = NAN;
        }
        delete ptVals;
        return percentileVal;
    }

    double SPDMetricCalcSkewnessHeight::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        double skew = 0;
        if(ptVals->size() > 0)
		{
            skew = gsl_stats_skew (&(*ptVals)[0], 1, ptVals->size());
        }
        else 
        {
            skew = NAN;
        }
        delete ptVals;
        return skew;
    }

    double SPDMetricCalcPersonModeSkewnessHeight::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        double personModeSkew = 0;
        if(ptVals->size() > 0)
		{
            sort(ptVals->begin(), ptVals->end());
            double mode = this->calcBinnedMode(ptVals, resolution);
            double mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
            double stddev = gsl_stats_sd (&(*ptVals)[0], 1, ptVals->size());
            
            personModeSkew = (mean - mode)/stddev;
        }
        else
        {
            personModeSkew = NAN;
        }
        delete ptVals;
        return personModeSkew;
    }

    double SPDMetricCalcPersonMedianSkewnessHeight::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        double personMedianSkew = 0;
        if(ptVals->size() > 0)
		{
            sort(ptVals->begin(), ptVals->end());
            double median = gsl_stats_median_from_sorted_data(&(*ptVals)[0], 1, ptVals->size());
            double mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
            double stddev = gsl_stats_sd (&(*ptVals)[0], 1, ptVals->size());
            
            personMedianSkew = (mean - median)/stddev;
        }
        else
        {
            personMedianSkew = NAN;
        }
        delete ptVals;
        return personMedianSkew;
    }

    double SPDMetricCalcKurtosisHeight::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        double kurtosis = 0;
        if(ptVals->size() > 0)
		{
            kurtosis = gsl_stats_kurtosis (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            kurtosis = NAN;
        }
        delete ptVals;
        return kurtosis;
    }
    
    double SPDMetricCalcNumReturnsAboveMetricHeight::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        boost::uint_fast64_t valCount = 0;
        if(ptVals->size() > 0)
		{
            double thresValue = metric->calcValue(pulses, spdFile, geom);
            for(vector<double>::iterator iterVals; iterVals != ptVals->end(); ++iterVals)
            {
                if((*iterVals) > thresValue)
                {
                    ++valCount;
                }
            }
        }
        delete ptVals;
        return valCount;
    }
    
    double SPDMetricCalcNumReturnsBelowMetricHeight::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        boost::uint_fast64_t valCount = 0;
        if(ptVals->size() > 0)
		{
            double thresValue = metric->calcValue(pulses, spdFile, geom);
            for(vector<double>::iterator iterVals; iterVals != ptVals->end(); ++iterVals)
            {
                if((*iterVals) < thresValue)
                {
                    ++valCount;
                }
            }
        }
        delete ptVals;
        return valCount;
    }
    
    double SPDMetricCalcWeibullAlphaHeight::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        double weibullAlpha = 0;
        try 
        {
            if(pulses->size() > 0)
            {
                vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
                if(ptVals->size() > 0)
                {
                    double minH = 0; 
                    double maxH = 0;
                    gsl_stats_minmax(&minH, &maxH, &(*ptVals)[0], 1, ptVals->size());
                    if(minH > 0)
                    {
                        minH = 0;
                    }
                    
                    boost::uint_fast32_t numBins = 0;
                    double *bins = this->binData(ptVals, this->resolution, &numBins, minH, maxH);
                    size_t maxIdx =  gsl_stats_max_index (bins, 1, numBins);
                    //double maxHBins = minH + (resolution * numBins);
                    
                    WeibullFitVals *fitData = new WeibullFitVals();
                    fitData->heights = new double[numBins];
                    fitData->binVals = new double[numBins];
                    fitData->error = new double[numBins];
                    double binSums = 0;
                    cout << "Bin Heights:\t";
                    for(boost::uint_fast32_t i = 0; i < numBins; ++i)
                    {
                        if(i == 0)
                        {
                            cout << (minH + (resolution * i));
                        }
                        else
                        {
                            cout << "," << (minH + (resolution * i));
                        }
                    }
                    cout << endl;
                    cout << "Bins:\t";
                    for(boost::uint_fast32_t i = 0; i < numBins; ++i)
                    {
                        binSums += bins[i];
                        fitData->heights[i] = minH + (resolution/2) + (resolution * ((double)i));
                        fitData->error[i] = 1;
                        if(i == 0)
                        {
                            cout << bins[i];
                        }
                        else
                        {
                            cout << "," << bins[i];
                        }
                    }
                    cout << endl;
                    // Make area == 1
                    cout << "Bins (Area == 1):\t";
                    for(boost::uint_fast32_t i = 0; i < numBins; ++i)
                    {
                        fitData->binVals[i] = bins[i]/binSums;
                        if(i == 0)
                        {
                            cout << fitData->binVals[i];
                        }
                        else
                        {
                            cout << "," << fitData->binVals[i];
                        }
                    }
                    cout << endl;
                    
                    // parameters[0] - Alpha
                    // parameters[1] - Beta
                    double *parameters = new double[2];                    
                    mp_par *paramConstraints = new mp_par[2];
                    
                    parameters[0] = fitData->binVals[maxIdx]; // Alpha
                    cout << "init alpha = " << parameters[0] << endl; 
                    paramConstraints[0].fixed = false;
					paramConstraints[0].limited[0] = true;
					paramConstraints[0].limited[1] = false;
					paramConstraints[0].limits[0] = 0;
					paramConstraints[0].limits[1] = 0;
                    //cout << "Alpha constraint = [" << paramConstraints[0].limits[0] << ", " << paramConstraints[0].limits[1] << "]\n";
					paramConstraints[0].parname = const_cast<char*>(string("Alpha").c_str());;
					paramConstraints[0].step = 0;
					paramConstraints[0].relstep = 0;
					paramConstraints[0].side = 0;
					paramConstraints[0].deriv_debug = 0;
					
                    //double percent20height = (maxH - minH) * 0.2;
					parameters[1] = minH + (((double)maxIdx) * resolution); // Beta
                    cout << "init beta = " << parameters[1] << endl;
					paramConstraints[1].fixed = false;
					paramConstraints[1].limited[0] = false;
					paramConstraints[1].limited[1] = false;
					paramConstraints[1].limits[0] = 0;//parameters[1] - percent20height;
                                                      //if(paramConstraints[1].limits[0] < minH)
                                                      //{
                                                      //    paramConstraints[1].limits[0] = minH;
                                                      // }
					paramConstraints[1].limits[1] = 0;//parameters[1] + percent20height;
                                                      //if(paramConstraints[1].limits[1] > maxHBins)
                                                      //{
                                                      //    paramConstraints[1].limits[1] = maxHBins;
                                                      //}
                                                      //cout << "Beta constraint = [" << paramConstraints[1].limits[0] << ", " << paramConstraints[1].limits[1] << "]\n";
					paramConstraints[1].parname = const_cast<char*>(string("Beta").c_str());;
					paramConstraints[1].step = 0;
					paramConstraints[1].relstep = 0;
					paramConstraints[1].side = 0;
					paramConstraints[1].deriv_debug = 0;
                    
                    mpResultsValues->bestnorm = 0;
					mpResultsValues->orignorm = 0;
					mpResultsValues->niter = 0;
					mpResultsValues->nfev = 0;
					mpResultsValues->status = 0;
					mpResultsValues->npar = 0;
					mpResultsValues->nfree = 0;
					mpResultsValues->npegged = 0;
					mpResultsValues->nfunc = 0;
					mpResultsValues->resid = 0;
					mpResultsValues->xerror = 0;
					mpResultsValues->covar = 0; // Not being retrieved
                    
                    /*
					 * int m     - number of data points
					 * int npar  - number of parameters
					 * double *xall - parameters values (initial values and then best fit values)
					 * mp_par *pars - Constrains
					 * mp_config *config - Configuration parameters
					 * void *private_data - Waveform data structure
					 * mp_result *result - diagnostic info from function
					 */
					int returnCode = mpfit(weibullFit, numBins, 2, parameters, paramConstraints, mpConfigValues, fitData, mpResultsValues);
					if((returnCode == MP_OK_CHI) | (returnCode == MP_OK_PAR) |
					   (returnCode == MP_OK_BOTH) | (returnCode == MP_OK_DIR) |
					   (returnCode == MP_MAXITER) | (returnCode == MP_FTOL)
					   | (returnCode == MP_XTOL) | (returnCode == MP_XTOL))
					{
						// MP Fit completed.. On on debug_info for more information. 
					}
					else if(returnCode == MP_ERR_INPUT)
					{
						throw SPDProcessingException("mpfit - Check inputs.");
					}
					else if(returnCode == MP_ERR_NAN)
					{
						throw SPDProcessingException("mpfit - Weibull fit function produced NaN value.");
					}
					else if(returnCode == MP_ERR_FUNC)
					{
						throw SPDProcessingException("mpfit - No Weibull fit function was supplied.");
					}
					else if(returnCode == MP_ERR_NPOINTS)
					{
						throw SPDProcessingException("mpfit - No data points were supplied.");
					}
					else if(returnCode == MP_ERR_NFREE)
					{
						throw SPDProcessingException("mpfit - No parameters are free - i.e., nothing to optimise!");
					}
					else if(returnCode == MP_ERR_MEMORY)
					{
						throw SPDProcessingException("mpfit - memory allocation error - may have run out!");
					}
					else if(returnCode == MP_ERR_INITBOUNDS)
					{
						throw SPDProcessingException("mpfit - Initial parameter values inconsistant with constraints.");
					}
					else if(returnCode == MP_ERR_PARAM)
					{
						throw SPDProcessingException("mpfit - An error has occur with an input parameter.");
					}
					else if(returnCode == MP_ERR_DOF)
					{
						throw SPDProcessingException("mpfit - Not enough degrees of freedom.");
					}
					else 
					{
						cout << "Return code is :" << returnCode << " - this can not been defined!\n";
					}
                    
                    cout << "final alpha = " << parameters[0] << endl;
                    cout << "final beta = " << parameters[1] << endl << endl;
                    
                    weibullAlpha = parameters[0];
                    
                    delete[] parameters;                    
                    delete[] paramConstraints;
                    delete[] bins;
                }
                else
                {
                    weibullAlpha = NAN;
                }
                
            }
            else
            {
                weibullAlpha = NAN;
            }
            
        } 
        catch (SPDProcessingException &e) 
        {
            throw e;
        }
        
        return weibullAlpha;
    }
    
    double SPDMetricCalcWeibullBetaHeight::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        double weibullBeta = 0;
        try 
        {
            if(pulses->size() > 0)
            {
                vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
                if(ptVals->size() > 0)
                {
                    double minH = 0; 
                    double maxH = 0;
                    gsl_stats_minmax(&minH, &maxH, &(*ptVals)[0], 1, ptVals->size());
                    if(minH > 0)
                    {
                        minH = 0;
                    }
                    
                    boost::uint_fast32_t numBins = 0;
                    double *bins = this->binData(ptVals, this->resolution, &numBins, minH, maxH);
                    size_t maxIdx =  gsl_stats_max_index (bins, 1, numBins);
                    
                    WeibullFitVals *fitData = new WeibullFitVals();
                    fitData->heights = new double[numBins];
                    fitData->binVals = new double[numBins];
                    fitData->error = new double[numBins];
                    double binSums = 0;
                    for(boost::uint_fast32_t i = 0; i < numBins; ++i)
                    {
                        binSums += bins[i];
                        fitData->heights[i] = minH + (resolution/2) + (resolution * ((double)i));
                        fitData->error[i] = 1;
                    }
                    
                    // Make area == 1
                    for(boost::uint_fast32_t i = 0; i < numBins; ++i)
                    {
                        fitData->binVals[i] = bins[i]/binSums;
                    }
                    
                    // parameters[0] - Alpha
                    // parameters[1] - Beta
                    double *parameters = new double[2];                    
                    mp_par *paramConstraints = new mp_par[2];
                    
                    parameters[0] = fitData->binVals[maxIdx]; // Alpha
                    paramConstraints[0].fixed = false;
					paramConstraints[0].limited[0] = true;
					paramConstraints[0].limited[1] = true;
					paramConstraints[0].limits[0] = 0.00001;
					paramConstraints[0].limits[1] = 0.99999;
					paramConstraints[0].parname = const_cast<char*>(string("Alpha").c_str());;
					paramConstraints[0].step = 0;
					paramConstraints[0].relstep = 0;
					paramConstraints[0].side = 0;
					paramConstraints[0].deriv_debug = 0;
					
                    double percent20height = (maxH - minH) * 0.2;
					parameters[1] = minH + (((double)maxIdx) * resolution); // Beta
					paramConstraints[1].fixed = false;
					paramConstraints[1].limited[0] = true;
					paramConstraints[1].limited[1] = true;
					paramConstraints[1].limits[0] = parameters[1] - percent20height;
                    if(paramConstraints[1].limits[0] < minH)
                    {
                        paramConstraints[1].limits[0] = minH;
                    }
					paramConstraints[1].limits[1] = parameters[1] + percent20height;
                    if(paramConstraints[1].limits[1] > maxH)
                    {
                        paramConstraints[1].limits[1] = maxH;
                    }
					paramConstraints[1].parname = const_cast<char*>(string("Beta").c_str());;
					paramConstraints[1].step = 0;
					paramConstraints[1].relstep = 0;
					paramConstraints[1].side = 0;
					paramConstraints[1].deriv_debug = 0;
                    
                    mpResultsValues->bestnorm = 0;
					mpResultsValues->orignorm = 0;
					mpResultsValues->niter = 0;
					mpResultsValues->nfev = 0;
					mpResultsValues->status = 0;
					mpResultsValues->npar = 0;
					mpResultsValues->nfree = 0;
					mpResultsValues->npegged = 0;
					mpResultsValues->nfunc = 0;
					mpResultsValues->resid = 0;
					mpResultsValues->xerror = 0;
					mpResultsValues->covar = 0; // Not being retrieved
                    
                    /*
					 * int m     - number of data points
					 * int npar  - number of parameters
					 * double *xall - parameters values (initial values and then best fit values)
					 * mp_par *pars - Constrains
					 * mp_config *config - Configuration parameters
					 * void *private_data - Waveform data structure
					 * mp_result *result - diagnostic info from function
					 */
					int returnCode = mpfit(weibullFit, numBins, 2, parameters, paramConstraints, mpConfigValues, fitData, mpResultsValues);
					if((returnCode == MP_OK_CHI) | (returnCode == MP_OK_PAR) |
					   (returnCode == MP_OK_BOTH) | (returnCode == MP_OK_DIR) |
					   (returnCode == MP_MAXITER) | (returnCode == MP_FTOL)
					   | (returnCode == MP_XTOL) | (returnCode == MP_XTOL))
					{
						// MP Fit completed.. On on debug_info for more information. 
					}
					else if(returnCode == MP_ERR_INPUT)
					{
						throw SPDProcessingException("mpfit - Check inputs.");
					}
					else if(returnCode == MP_ERR_NAN)
					{
						throw SPDProcessingException("mpfit - Weibull fit function produced NaN value.");
					}
					else if(returnCode == MP_ERR_FUNC)
					{
						throw SPDProcessingException("mpfit - No Weibull fit function was supplied.");
					}
					else if(returnCode == MP_ERR_NPOINTS)
					{
						throw SPDProcessingException("mpfit - No data points were supplied.");
					}
					else if(returnCode == MP_ERR_NFREE)
					{
						throw SPDProcessingException("mpfit - No parameters are free - i.e., nothing to optimise!");
					}
					else if(returnCode == MP_ERR_MEMORY)
					{
						throw SPDProcessingException("mpfit - memory allocation error - may have run out!");
					}
					else if(returnCode == MP_ERR_INITBOUNDS)
					{
						throw SPDProcessingException("mpfit - Initial parameter values inconsistant with constraints.");
					}
					else if(returnCode == MP_ERR_PARAM)
					{
						throw SPDProcessingException("mpfit - An error has occur with an input parameter.");
					}
					else if(returnCode == MP_ERR_DOF)
					{
						throw SPDProcessingException("mpfit - Not enough degrees of freedom.");
					}
					else 
					{
						cout << "Return code is :" << returnCode << " - this can not been defined!\n";
					}
                    
                    weibullBeta = parameters[1];
                    
                    delete[] parameters;                    
                    delete[] paramConstraints;
                    delete[] bins;
                }
                else
                {
                    weibullBeta = NAN;
                }
                
            }
            else
            {
                weibullBeta = NAN;
            }
            
        } 
        catch (SPDProcessingException &e) 
        {
            throw e;
        }
        
        return weibullBeta;
    }
    
    double SPDMetricCalcWeibullQuantileRangeHeight::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        double weibullQuantileRange = 0;
        try 
        {
            if(pulses->size() > 0)
            {
                vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
                if(ptVals->size() > 0)
                {
                    double minH = 0; 
                    double maxH = 0;
                    gsl_stats_minmax(&minH, &maxH, &(*ptVals)[0], 1, ptVals->size());
                    if(minH > 0)
                    {
                        minH = 0;
                    }
                    
                    boost::uint_fast32_t numBins = 0;
                    double *bins = this->binData(ptVals, this->resolution, &numBins, minH, maxH);
                    size_t maxIdx =  gsl_stats_max_index (bins, 1, numBins);
                    
                    WeibullFitVals *fitData = new WeibullFitVals();
                    fitData->heights = new double[numBins];
                    fitData->binVals = new double[numBins];
                    fitData->error = new double[numBins];
                    double binSums = 0;
                    for(boost::uint_fast32_t i = 0; i < numBins; ++i)
                    {
                        binSums += bins[i];
                        fitData->heights[i] = minH + (resolution/2) + (resolution * ((double)i));
                        fitData->error[i] = 1;
                    }
                    
                    // Make area == 1
                    for(boost::uint_fast32_t i = 0; i < numBins; ++i)
                    {
                        fitData->binVals[i] = bins[i]/binSums;
                    }
                    
                    // parameters[0] - Alpha
                    // parameters[1] - Beta
                    double *parameters = new double[2];                    
                    mp_par *paramConstraints = new mp_par[2];
                    
                    parameters[0] = fitData->binVals[maxIdx]; // Alpha
                    paramConstraints[0].fixed = false;
					paramConstraints[0].limited[0] = true;
					paramConstraints[0].limited[1] = true;
					paramConstraints[0].limits[0] = 0.00001;
					paramConstraints[0].limits[1] = 0.99999;
					paramConstraints[0].parname = const_cast<char*>(string("Alpha").c_str());;
					paramConstraints[0].step = 0;
					paramConstraints[0].relstep = 0;
					paramConstraints[0].side = 0;
					paramConstraints[0].deriv_debug = 0;
					
                    double percent20height = (maxH - minH) * 0.2;
					parameters[1] = minH + (((double)maxIdx) * resolution); // Beta
					paramConstraints[1].fixed = false;
					paramConstraints[1].limited[0] = true;
					paramConstraints[1].limited[1] = true;
					paramConstraints[1].limits[0] = parameters[1] - percent20height;
                    if(paramConstraints[1].limits[0] < minH)
                    {
                        paramConstraints[1].limits[0] = minH;
                    }
					paramConstraints[1].limits[1] = parameters[1] + percent20height;
                    if(paramConstraints[1].limits[1] > maxH)
                    {
                        paramConstraints[1].limits[1] = maxH;
                    }
					paramConstraints[1].parname = const_cast<char*>(string("Beta").c_str());;
					paramConstraints[1].step = 0;
					paramConstraints[1].relstep = 0;
					paramConstraints[1].side = 0;
					paramConstraints[1].deriv_debug = 0;
                    
                    mpResultsValues->bestnorm = 0;
					mpResultsValues->orignorm = 0;
					mpResultsValues->niter = 0;
					mpResultsValues->nfev = 0;
					mpResultsValues->status = 0;
					mpResultsValues->npar = 0;
					mpResultsValues->nfree = 0;
					mpResultsValues->npegged = 0;
					mpResultsValues->nfunc = 0;
					mpResultsValues->resid = 0;
					mpResultsValues->xerror = 0;
					mpResultsValues->covar = 0; // Not being retrieved
                    
                    /*
					 * int m     - number of data points
					 * int npar  - number of parameters
					 * double *xall - parameters values (initial values and then best fit values)
					 * mp_par *pars - Constrains
					 * mp_config *config - Configuration parameters
					 * void *private_data - Waveform data structure
					 * mp_result *result - diagnostic info from function
					 */
					int returnCode = mpfit(weibullFit, numBins, 2, parameters, paramConstraints, mpConfigValues, fitData, mpResultsValues);
					if((returnCode == MP_OK_CHI) | (returnCode == MP_OK_PAR) |
					   (returnCode == MP_OK_BOTH) | (returnCode == MP_OK_DIR) |
					   (returnCode == MP_MAXITER) | (returnCode == MP_FTOL)
					   | (returnCode == MP_XTOL) | (returnCode == MP_XTOL))
					{
						// MP Fit completed.. On on debug_info for more information. 
					}
					else if(returnCode == MP_ERR_INPUT)
					{
						throw SPDProcessingException("mpfit - Check inputs.");
					}
					else if(returnCode == MP_ERR_NAN)
					{
						throw SPDProcessingException("mpfit - Weibull fit function produced NaN value.");
					}
					else if(returnCode == MP_ERR_FUNC)
					{
						throw SPDProcessingException("mpfit - No Weibull fit function was supplied.");
					}
					else if(returnCode == MP_ERR_NPOINTS)
					{
						throw SPDProcessingException("mpfit - No data points were supplied.");
					}
					else if(returnCode == MP_ERR_NFREE)
					{
						throw SPDProcessingException("mpfit - No parameters are free - i.e., nothing to optimise!");
					}
					else if(returnCode == MP_ERR_MEMORY)
					{
						throw SPDProcessingException("mpfit - memory allocation error - may have run out!");
					}
					else if(returnCode == MP_ERR_INITBOUNDS)
					{
						throw SPDProcessingException("mpfit - Initial parameter values inconsistant with constraints.");
					}
					else if(returnCode == MP_ERR_PARAM)
					{
						throw SPDProcessingException("mpfit - An error has occur with an input parameter.");
					}
					else if(returnCode == MP_ERR_DOF)
					{
						throw SPDProcessingException("mpfit - Not enough degrees of freedom.");
					}
					else 
					{
						cout << "Return code is :" << returnCode << " - this can not been defined!\n";
					}
                    
                    weibullQuantileRange = 0; // Need to calculate... TODO
                    
                    delete[] parameters;                    
                    delete[] paramConstraints;
                    delete[] bins;
                }
                else
                {
                    weibullQuantileRange = NAN;
                }
                
            }
            else
            {
                weibullQuantileRange = NAN;
            }
            
        } 
        catch (SPDProcessingException &e) 
        {
            throw e;
        }
        
        return weibullQuantileRange;
    }
    
    
    /*
     * Metric's for Z
     */
    
    
    double SPDMetricCalcNumReturnsZ::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        boost::uint_fast64_t numReturns = ptVals->size();
        delete ptVals;
        return numReturns;
    }
    
    double SPDMetricCalcSumZ::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        double sum = 0;
        for(vector<double>::iterator iterVals = ptVals->begin(); iterVals != ptVals->end(); ++iterVals)
        {
            sum += (*iterVals);
        }
        delete ptVals;
        return sum;
    }    
    
    double SPDMetricCalcMeanZ::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        double mean = 0;
        if(ptVals->size() > 0)
		{
            mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            mean = NAN;
        }
        delete ptVals;
        return mean;
    }
    
    double SPDMetricCalcMedianZ::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        double median = 0;
        if(ptVals->size() > 0)
		{
            sort(ptVals->begin(), ptVals->end());
            /*for(unsigned int i = 0; i < ptVals->size(); ++i)
             {
             if( i == 0 )
             {
             cout << &(*ptVals)[0][i];
             }
             else
             {
             cout << ", " << &(*ptVals)[0][i];
             }
             }
             cout << endl << endl;*/
            median = gsl_stats_median_from_sorted_data(&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            median = NAN;
        }
        delete ptVals;
        return median;
    }
    
    double SPDMetricCalcModeZ::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        double mode = 0;
        if(ptVals->size() > 0)
        {
            mode = this->calcBinnedMode(ptVals, resolution);
        }
        else
        {
            mode = NAN;
        }
        
        delete ptVals;
        return mode;
    }
    
    double SPDMetricCalcMinZ::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        double min = 0;
        if(ptVals->size() > 0)
		{
            min = gsl_stats_min (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            min = NAN;
        }
        delete ptVals;
        return min;
    }
    
    double SPDMetricCalcMaxZ::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        double max = 0;
        if(ptVals->size() > 0)
		{
            max = gsl_stats_max (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            max = NAN;
        }
        delete ptVals;
        return max;
    }
    
    double SPDMetricCalcStdDevZ::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        double stddev = 0;
        if(ptVals->size() > 0)
		{
            stddev = gsl_stats_sd (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            stddev = NAN;
        }
        delete ptVals;
        return stddev;
    }
    
    double SPDMetricCalcVarianceZ::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        double variance = 0;
        if(ptVals->size() > 0)
		{
            variance = gsl_stats_variance (&(*ptVals)[0], 1, ptVals->size());
        }
        else 
        {
            variance = NAN;
        }
        delete ptVals;
        return variance;
    }
    
    double SPDMetricCalcAbsDeviationZ::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        double absdev = 0;
        if(ptVals->size() > 0)
		{
            absdev = gsl_stats_absdev (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            absdev = NAN;
        }
        delete ptVals;
        return absdev;
    }
    
    double SPDMetricCalcCoefficientOfVariationZ::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        double cv = 0;
        if(ptVals->size() > 0)
		{
            double sumSq = 0;
            double mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
            for(vector<double>::iterator iterVals; iterVals != ptVals->end(); ++iterVals)
            {
                sumSq += pow(((*iterVals) - mean),2);
            }
            cv = sqrt(sumSq/ptVals->size())/mean;
        }
        else 
        {
            cv = NAN;
        }
        
        delete ptVals;
        return cv;
    }
    
    double SPDMetricCalcPercentileZ::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        /*
         cout << endl << endl << endl;
         vector<SPDPulse*>::iterator iterPulses;
         for(iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
         {
         if((*iterPulses)->numberOfReturns > 0)
         {
         for(vector<SPDPoint*>::iterator iterPts = (*iterPulses)->pts->begin(); iterPts != (*iterPulses)->pts->end(); ++iterPts)
         {
         cout << (*iterPulses)->pulseID << "," << (*iterPts)->x << "," << (*iterPts)->y << "," << (*iterPts)->z << "," << (*iterPts)->z << "," << (*iterPts)->classification << endl;
         }
         }
         }
         cout << endl << endl << endl;
         */
        vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        double percentileVal = 0;
        if(ptVals->size() > 0)
		{
            double quatFrac = ((double)percentile)/100;
            sort(ptVals->begin(), ptVals->end());
            /*cout << "Calc Percentile " << quatFrac << endl;
             for(unsigned int i = 0; i < ptVals->size(); ++i)
             {
             if( i == 0 )
             {
             cout << &(*ptVals)[0][i];
             }
             else
             {
             cout << ", " << &(*ptVals)[0][i];
             }
             }
             cout << endl;*/
            percentileVal = gsl_stats_quantile_from_sorted_data(&(*ptVals)[0], 1, ptVals->size(), quatFrac);
            //cout << "Percentile = " << percentileVal << endl << endl;
        }
        else 
        {
            percentileVal = NAN;
        }
        delete ptVals;
        return percentileVal;
    }
    
    double SPDMetricCalcSkewnessZ::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        double skew = 0;
        if(ptVals->size() > 0)
		{
            skew = gsl_stats_skew (&(*ptVals)[0], 1, ptVals->size());
        }
        else 
        {
            skew = NAN;
        }
        delete ptVals;
        return skew;
    }
    
    double SPDMetricCalcPersonModeSkewnessZ::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        double personModeSkew = 0;
        if(ptVals->size() > 0)
		{
            sort(ptVals->begin(), ptVals->end());
            double mode = this->calcBinnedMode(ptVals, resolution);
            double mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
            double stddev = gsl_stats_sd (&(*ptVals)[0], 1, ptVals->size());
            
            personModeSkew = (mean - mode)/stddev;
        }
        else
        {
            personModeSkew = NAN;
        }
        delete ptVals;
        return personModeSkew;
    }
    
    double SPDMetricCalcPersonMedianSkewnessZ::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        double personMedianSkew = 0;
        if(ptVals->size() > 0)
		{
            sort(ptVals->begin(), ptVals->end());
            double median = gsl_stats_median_from_sorted_data(&(*ptVals)[0], 1, ptVals->size());
            double mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
            double stddev = gsl_stats_sd (&(*ptVals)[0], 1, ptVals->size());
            
            personMedianSkew = (mean - median)/stddev;
        }
        else
        {
            personMedianSkew = NAN;
        }
        delete ptVals;
        return personMedianSkew;
    }
    
    double SPDMetricCalcKurtosisZ::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        double kurtosis = 0;
        if(ptVals->size() > 0)
		{
            kurtosis = gsl_stats_kurtosis (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            kurtosis = NAN;
        }
        delete ptVals;
        return kurtosis;
    }
    
    double SPDMetricCalcNumReturnsAboveMetricZ::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        boost::uint_fast64_t valCount = 0;
        if(ptVals->size() > 0)
		{
            double thresValue = metric->calcValue(pulses, spdFile, geom);
            for(vector<double>::iterator iterVals; iterVals != ptVals->end(); ++iterVals)
            {
                if((*iterVals) > thresValue)
                {
                    ++valCount;
                }
            }
        }
        delete ptVals;
        return valCount;
    }
    
    double SPDMetricCalcNumReturnsBelowMetricZ::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        boost::uint_fast64_t valCount = 0;
        if(ptVals->size() > 0)
		{
            double thresValue = metric->calcValue(pulses, spdFile, geom);
            for(vector<double>::iterator iterVals; iterVals != ptVals->end(); ++iterVals)
            {
                if((*iterVals) < thresValue)
                {
                    ++valCount;
                }
            }
        }
        delete ptVals;
        return valCount;
    }
    
    double SPDMetricCalcWeibullAlphaZ::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        double weibullAlpha = 0;
        try 
        {
            if(pulses->size() > 0)
            {
                vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
                if(ptVals->size() > 0)
                {
                    double minH = 0; 
                    double maxH = 0;
                    gsl_stats_minmax(&minH, &maxH, &(*ptVals)[0], 1, ptVals->size());
                    if(minH > 0)
                    {
                        minH = 0;
                    }
                    
                    boost::uint_fast32_t numBins = 0;
                    double *bins = this->binData(ptVals, this->resolution, &numBins, minH, maxH);
                    size_t maxIdx =  gsl_stats_max_index (bins, 1, numBins);
                    
                    WeibullFitVals *fitData = new WeibullFitVals();
                    fitData->heights = new double[numBins];
                    fitData->binVals = new double[numBins];
                    fitData->error = new double[numBins];
                    double binSums = 0;
                    for(boost::uint_fast32_t i = 0; i < numBins; ++i)
                    {
                        binSums += bins[i];
                        fitData->heights[i] = minH + (resolution/2) + (resolution * ((double)i));
                        fitData->error[i] = 1;
                    }
                    
                    // Make area == 1
                    for(boost::uint_fast32_t i = 0; i < numBins; ++i)
                    {
                        fitData->binVals[i] = bins[i]/binSums;
                    }
                    
                    // parameters[0] - Alpha
                    // parameters[1] - Beta
                    double *parameters = new double[2];                    
                    mp_par *paramConstraints = new mp_par[2];
                    
                    parameters[0] = fitData->binVals[maxIdx]; // Alpha
                    paramConstraints[0].fixed = false;
					paramConstraints[0].limited[0] = true;
					paramConstraints[0].limited[1] = true;
					paramConstraints[0].limits[0] = 0.00001;
					paramConstraints[0].limits[1] = 0.99999;
					paramConstraints[0].parname = const_cast<char*>(string("Alpha").c_str());;
					paramConstraints[0].step = 0;
					paramConstraints[0].relstep = 0;
					paramConstraints[0].side = 0;
					paramConstraints[0].deriv_debug = 0;
					
                    double percent20z = (maxH - minH) * 0.2;
					parameters[1] = minH + (((double)maxIdx) * resolution); // Beta
					paramConstraints[1].fixed = false;
					paramConstraints[1].limited[0] = true;
					paramConstraints[1].limited[1] = true;
					paramConstraints[1].limits[0] = parameters[1] - percent20z;
                    if(paramConstraints[1].limits[0] < minH)
                    {
                        paramConstraints[1].limits[0] = minH;
                    }
					paramConstraints[1].limits[1] = parameters[1] + percent20z;
                    if(paramConstraints[1].limits[1] > maxH)
                    {
                        paramConstraints[1].limits[1] = maxH;
                    }
					paramConstraints[1].parname = const_cast<char*>(string("Beta").c_str());;
					paramConstraints[1].step = 0;
					paramConstraints[1].relstep = 0;
					paramConstraints[1].side = 0;
					paramConstraints[1].deriv_debug = 0;
                    
                    mpResultsValues->bestnorm = 0;
					mpResultsValues->orignorm = 0;
					mpResultsValues->niter = 0;
					mpResultsValues->nfev = 0;
					mpResultsValues->status = 0;
					mpResultsValues->npar = 0;
					mpResultsValues->nfree = 0;
					mpResultsValues->npegged = 0;
					mpResultsValues->nfunc = 0;
					mpResultsValues->resid = 0;
					mpResultsValues->xerror = 0;
					mpResultsValues->covar = 0; // Not being retrieved
                    
                    /*
					 * int m     - number of data points
					 * int npar  - number of parameters
					 * double *xall - parameters values (initial values and then best fit values)
					 * mp_par *pars - Constrains
					 * mp_config *config - Configuration parameters
					 * void *private_data - Waveform data structure
					 * mp_result *result - diagnostic info from function
					 */
					int returnCode = mpfit(weibullFit, numBins, 2, parameters, paramConstraints, mpConfigValues, fitData, mpResultsValues);
					if((returnCode == MP_OK_CHI) | (returnCode == MP_OK_PAR) |
					   (returnCode == MP_OK_BOTH) | (returnCode == MP_OK_DIR) |
					   (returnCode == MP_MAXITER) | (returnCode == MP_FTOL)
					   | (returnCode == MP_XTOL) | (returnCode == MP_XTOL))
					{
						// MP Fit completed.. On on debug_info for more information. 
					}
					else if(returnCode == MP_ERR_INPUT)
					{
						throw SPDException("mpfit - Check inputs.");
					}
					else if(returnCode == MP_ERR_NAN)
					{
						throw SPDException("mpfit - Weibull fit function produced NaN value.");
					}
					else if(returnCode == MP_ERR_FUNC)
					{
						throw SPDException("mpfit - No Weibull fit function was supplied.");
					}
					else if(returnCode == MP_ERR_NPOINTS)
					{
						throw SPDException("mpfit - No data points were supplied.");
					}
					else if(returnCode == MP_ERR_NFREE)
					{
						throw SPDException("mpfit - No parameters are free - i.e., nothing to optimise!");
					}
					else if(returnCode == MP_ERR_MEMORY)
					{
						throw SPDException("mpfit - memory allocation error - may have run out!");
					}
					else if(returnCode == MP_ERR_INITBOUNDS)
					{
						throw SPDException("mpfit - Initial parameter values inconsistant with constraints.");
					}
					else if(returnCode == MP_ERR_PARAM)
					{
						throw SPDException("mpfit - An error has occur with an input parameter.");
					}
					else if(returnCode == MP_ERR_DOF)
					{
						throw SPDException("mpfit - Not enough degrees of freedom.");
					}
					else 
					{
						cout << "Return code is :" << returnCode << " - this can not been defined!\n";
					}
                    
                    weibullAlpha = parameters[0];
                    
                    delete[] parameters;                    
                    delete[] paramConstraints;
                    delete[] bins;
                }
                else
                {
                    weibullAlpha = NAN;
                }
                
            }
            else
            {
                weibullAlpha = NAN;
            }
            
        } 
        catch (SPDProcessingException &e) 
        {
            throw e;
        }
        
        return weibullAlpha;
    }
    
    double SPDMetricCalcWeibullBetaZ::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        double weibullBeta = 0;
        try 
        {
            if(pulses->size() > 0)
            {
                vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
                if(ptVals->size() > 0)
                {
                    double minH = 0; 
                    double maxH = 0;
                    gsl_stats_minmax(&minH, &maxH, &(*ptVals)[0], 1, ptVals->size());
                    if(minH > 0)
                    {
                        minH = 0;
                    }
                    
                    boost::uint_fast32_t numBins = 0;
                    double *bins = this->binData(ptVals, this->resolution, &numBins, minH, maxH);
                    size_t maxIdx =  gsl_stats_max_index (bins, 1, numBins);
                    
                    WeibullFitVals *fitData = new WeibullFitVals();
                    fitData->heights = new double[numBins];
                    fitData->binVals = new double[numBins];
                    fitData->error = new double[numBins];
                    double binSums = 0;
                    for(boost::uint_fast32_t i = 0; i < numBins; ++i)
                    {
                        binSums += bins[i];
                        fitData->heights[i] = minH + (resolution/2) + (resolution * ((double)i));
                        fitData->error[i] = 1;
                    }
                    
                    // Make area == 1
                    for(boost::uint_fast32_t i = 0; i < numBins; ++i)
                    {
                        fitData->binVals[i] = bins[i]/binSums;
                    }
                    
                    // parameters[0] - Alpha
                    // parameters[1] - Beta
                    double *parameters = new double[2];                    
                    mp_par *paramConstraints = new mp_par[2];
                    
                    parameters[0] = fitData->binVals[maxIdx]; // Alpha
                    paramConstraints[0].fixed = false;
					paramConstraints[0].limited[0] = true;
					paramConstraints[0].limited[1] = true;
					paramConstraints[0].limits[0] = 0.00001;
					paramConstraints[0].limits[1] = 0.99999;
					paramConstraints[0].parname = const_cast<char*>(string("Alpha").c_str());;
					paramConstraints[0].step = 0;
					paramConstraints[0].relstep = 0;
					paramConstraints[0].side = 0;
					paramConstraints[0].deriv_debug = 0;
					
                    double percent20z = (maxH - minH) * 0.2;
					parameters[1] = minH + (((double)maxIdx) * resolution); // Beta
					paramConstraints[1].fixed = false;
					paramConstraints[1].limited[0] = true;
					paramConstraints[1].limited[1] = true;
					paramConstraints[1].limits[0] = parameters[1] - percent20z;
                    if(paramConstraints[1].limits[0] < minH)
                    {
                        paramConstraints[1].limits[0] = minH;
                    }
					paramConstraints[1].limits[1] = parameters[1] + percent20z;
                    if(paramConstraints[1].limits[1] > maxH)
                    {
                        paramConstraints[1].limits[1] = maxH;
                    }
					paramConstraints[1].parname = const_cast<char*>(string("Beta").c_str());;
					paramConstraints[1].step = 0;
					paramConstraints[1].relstep = 0;
					paramConstraints[1].side = 0;
					paramConstraints[1].deriv_debug = 0;
                    
                    mpResultsValues->bestnorm = 0;
					mpResultsValues->orignorm = 0;
					mpResultsValues->niter = 0;
					mpResultsValues->nfev = 0;
					mpResultsValues->status = 0;
					mpResultsValues->npar = 0;
					mpResultsValues->nfree = 0;
					mpResultsValues->npegged = 0;
					mpResultsValues->nfunc = 0;
					mpResultsValues->resid = 0;
					mpResultsValues->xerror = 0;
					mpResultsValues->covar = 0; // Not being retrieved
                    
                    /*
					 * int m     - number of data points
					 * int npar  - number of parameters
					 * double *xall - parameters values (initial values and then best fit values)
					 * mp_par *pars - Constrains
					 * mp_config *config - Configuration parameters
					 * void *private_data - Waveform data structure
					 * mp_result *result - diagnostic info from function
					 */
					int returnCode = mpfit(weibullFit, numBins, 2, parameters, paramConstraints, mpConfigValues, fitData, mpResultsValues);
					if((returnCode == MP_OK_CHI) | (returnCode == MP_OK_PAR) |
					   (returnCode == MP_OK_BOTH) | (returnCode == MP_OK_DIR) |
					   (returnCode == MP_MAXITER) | (returnCode == MP_FTOL)
					   | (returnCode == MP_XTOL) | (returnCode == MP_XTOL))
					{
						// MP Fit completed.. On on debug_info for more information. 
					}
					else if(returnCode == MP_ERR_INPUT)
					{
						throw SPDException("mpfit - Check inputs.");
					}
					else if(returnCode == MP_ERR_NAN)
					{
						throw SPDException("mpfit - Weibull fit function produced NaN value.");
					}
					else if(returnCode == MP_ERR_FUNC)
					{
						throw SPDException("mpfit - No Weibull fit function was supplied.");
					}
					else if(returnCode == MP_ERR_NPOINTS)
					{
						throw SPDException("mpfit - No data points were supplied.");
					}
					else if(returnCode == MP_ERR_NFREE)
					{
						throw SPDException("mpfit - No parameters are free - i.e., nothing to optimise!");
					}
					else if(returnCode == MP_ERR_MEMORY)
					{
						throw SPDException("mpfit - memory allocation error - may have run out!");
					}
					else if(returnCode == MP_ERR_INITBOUNDS)
					{
						throw SPDException("mpfit - Initial parameter values inconsistant with constraints.");
					}
					else if(returnCode == MP_ERR_PARAM)
					{
						throw SPDException("mpfit - An error has occur with an input parameter.");
					}
					else if(returnCode == MP_ERR_DOF)
					{
						throw SPDException("mpfit - Not enough degrees of freedom.");
					}
					else 
					{
						cout << "Return code is :" << returnCode << " - this can not been defined!\n";
					}
                    
                    weibullBeta = parameters[1];
                    
                    delete[] parameters;                    
                    delete[] paramConstraints;
                    delete[] bins;
                }
                else
                {
                    weibullBeta = NAN;
                }
                
            }
            else
            {
                weibullBeta = NAN;
            }
            
        } 
        catch (SPDProcessingException &e) 
        {
            throw e;
        }
        
        return weibullBeta;
    }
    
    double SPDMetricCalcWeibullQuantileRangeZ::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        double weibullQuantileRange = 0;
        try 
        {
            if(pulses->size() > 0)
            {
                vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
                if(ptVals->size() > 0)
                {
                    double minH = 0; 
                    double maxH = 0;
                    gsl_stats_minmax(&minH, &maxH, &(*ptVals)[0], 1, ptVals->size());
                    if(minH > 0)
                    {
                        minH = 0;
                    }
                    
                    boost::uint_fast32_t numBins = 0;
                    double *bins = this->binData(ptVals, this->resolution, &numBins, minH, maxH);
                    size_t maxIdx =  gsl_stats_max_index (bins, 1, numBins);
                    
                    WeibullFitVals *fitData = new WeibullFitVals();
                    fitData->heights = new double[numBins];
                    fitData->binVals = new double[numBins];
                    fitData->error = new double[numBins];
                    double binSums = 0;
                    for(boost::uint_fast32_t i = 0; i < numBins; ++i)
                    {
                        binSums += bins[i];
                        fitData->heights[i] = minH + (resolution/2) + (resolution * ((double)i));
                        fitData->error[i] = 1;
                    }
                    
                    // Make area == 1
                    for(boost::uint_fast32_t i = 0; i < numBins; ++i)
                    {
                        fitData->binVals[i] = bins[i]/binSums;
                    }
                    
                    // parameters[0] - Alpha
                    // parameters[1] - Beta
                    double *parameters = new double[2];                    
                    mp_par *paramConstraints = new mp_par[2];
                    
                    parameters[0] = fitData->binVals[maxIdx]; // Alpha
                    paramConstraints[0].fixed = false;
					paramConstraints[0].limited[0] = true;
					paramConstraints[0].limited[1] = true;
					paramConstraints[0].limits[0] = 0.00001;
					paramConstraints[0].limits[1] = 0.99999;
					paramConstraints[0].parname = const_cast<char*>(string("Alpha").c_str());;
					paramConstraints[0].step = 0;
					paramConstraints[0].relstep = 0;
					paramConstraints[0].side = 0;
					paramConstraints[0].deriv_debug = 0;
					
                    double percent20z = (maxH - minH) * 0.2;
					parameters[1] = minH + (((double)maxIdx) * resolution); // Beta
					paramConstraints[1].fixed = false;
					paramConstraints[1].limited[0] = true;
					paramConstraints[1].limited[1] = true;
					paramConstraints[1].limits[0] = parameters[1] - percent20z;
                    if(paramConstraints[1].limits[0] < minH)
                    {
                        paramConstraints[1].limits[0] = minH;
                    }
					paramConstraints[1].limits[1] = parameters[1] + percent20z;
                    if(paramConstraints[1].limits[1] > maxH)
                    {
                        paramConstraints[1].limits[1] = maxH;
                    }
					paramConstraints[1].parname = const_cast<char*>(string("Beta").c_str());;
					paramConstraints[1].step = 0;
					paramConstraints[1].relstep = 0;
					paramConstraints[1].side = 0;
					paramConstraints[1].deriv_debug = 0;
                    
                    mpResultsValues->bestnorm = 0;
					mpResultsValues->orignorm = 0;
					mpResultsValues->niter = 0;
					mpResultsValues->nfev = 0;
					mpResultsValues->status = 0;
					mpResultsValues->npar = 0;
					mpResultsValues->nfree = 0;
					mpResultsValues->npegged = 0;
					mpResultsValues->nfunc = 0;
					mpResultsValues->resid = 0;
					mpResultsValues->xerror = 0;
					mpResultsValues->covar = 0; // Not being retrieved
                    
                    /*
					 * int m     - number of data points
					 * int npar  - number of parameters
					 * double *xall - parameters values (initial values and then best fit values)
					 * mp_par *pars - Constrains
					 * mp_config *config - Configuration parameters
					 * void *private_data - Waveform data structure
					 * mp_result *result - diagnostic info from function
					 */
					int returnCode = mpfit(weibullFit, numBins, 2, parameters, paramConstraints, mpConfigValues, fitData, mpResultsValues);
					if((returnCode == MP_OK_CHI) | (returnCode == MP_OK_PAR) |
					   (returnCode == MP_OK_BOTH) | (returnCode == MP_OK_DIR) |
					   (returnCode == MP_MAXITER) | (returnCode == MP_FTOL)
					   | (returnCode == MP_XTOL) | (returnCode == MP_XTOL))
					{
						// MP Fit completed.. On on debug_info for more information. 
					}
					else if(returnCode == MP_ERR_INPUT)
					{
						throw SPDException("mpfit - Check inputs.");
					}
					else if(returnCode == MP_ERR_NAN)
					{
						throw SPDException("mpfit - Weibull fit function produced NaN value.");
					}
					else if(returnCode == MP_ERR_FUNC)
					{
						throw SPDException("mpfit - No Weibull fit function was supplied.");
					}
					else if(returnCode == MP_ERR_NPOINTS)
					{
						throw SPDException("mpfit - No data points were supplied.");
					}
					else if(returnCode == MP_ERR_NFREE)
					{
						throw SPDException("mpfit - No parameters are free - i.e., nothing to optimise!");
					}
					else if(returnCode == MP_ERR_MEMORY)
					{
						throw SPDException("mpfit - memory allocation error - may have run out!");
					}
					else if(returnCode == MP_ERR_INITBOUNDS)
					{
						throw SPDException("mpfit - Initial parameter values inconsistant with constraints.");
					}
					else if(returnCode == MP_ERR_PARAM)
					{
						throw SPDException("mpfit - An error has occur with an input parameter.");
					}
					else if(returnCode == MP_ERR_DOF)
					{
						throw SPDException("mpfit - Not enough degrees of freedom.");
					}
					else 
					{
						cout << "Return code is :" << returnCode << " - this can not been defined!\n";
					}
                    
                    weibullQuantileRange = 0; // Need to calculate... TODO
                    
                    delete[] parameters;                    
                    delete[] paramConstraints;
                    delete[] bins;
                }
                else
                {
                    weibullQuantileRange = NAN;
                }
                
            }
            else
            {
                weibullQuantileRange = NAN;
            }
            
        } 
        catch (SPDProcessingException &e) 
        {
            throw e;
        }
        
        return weibullQuantileRange;
    }
    
    
    
    
    /*
     * Metric's for Amplitude
     */

    double SPDMetricCalcNumReturnsAmplitude::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);
        boost::uint_fast64_t numReturns = ptVals->size();
        delete ptVals;
        return numReturns;
    }
    
    double SPDMetricCalcSumAmplitude::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);
        double sum = 0;
        for(vector<double>::iterator iterVals = ptVals->begin(); iterVals != ptVals->end(); ++iterVals)
        {
            sum += (*iterVals);
        }
        delete ptVals;
        return sum;
    }
    
    double SPDMetricCalcMeanAmplitude::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);
        double mean = 0;
        if(ptVals->size() > 0)
		{
            mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            mean = NAN;
        }
        delete ptVals;
        return mean;
    }
    
    double SPDMetricCalcMedianAmplitude::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);
        double median = 0;
        if(ptVals->size() > 0)
		{
            sort(ptVals->begin(), ptVals->end());
            median = gsl_stats_median_from_sorted_data(&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            median = NAN;
        }
        delete ptVals;
        return median;
    }
    
    double SPDMetricCalcModeAmplitude::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);
        double mode = 0;
        if(ptVals->size() > 0)
        {
            mode = this->calcBinnedMode(ptVals, resolution);
        }
        else
        {
            mode = NAN;
        }
        
        delete ptVals;
        return mode;
    }
    
    double SPDMetricCalcMinAmplitude::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);
        double min = 0;
        if(ptVals->size() > 0)
		{
            min = gsl_stats_min (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            min = NAN;
        }
        delete ptVals;
        return min;
    }
    
    double SPDMetricCalcMaxAmplitude::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);
        double max = 0;
        if(ptVals->size() > 0)
		{
            max = gsl_stats_max (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            max = NAN;
        }
        delete ptVals;
        return max;
    }
    
    double SPDMetricCalcStdDevAmplitude::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);
        double stddev = 0;
        if(ptVals->size() > 0)
		{
            stddev = gsl_stats_sd (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            stddev = NAN;
        }
        delete ptVals;
        return stddev;
    }
    
    double SPDMetricCalcVarianceAmplitude::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);
        double variance = 0;
        if(ptVals->size() > 0)
		{
            variance = gsl_stats_variance (&(*ptVals)[0], 1, ptVals->size());
        }
        else 
        {
            variance = NAN;
        }
        delete ptVals;
        return variance;
    }
    
    double SPDMetricCalcAbsDeviationAmplitude::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);
        double absdev = 0;
        if(ptVals->size() > 0)
		{
            absdev = gsl_stats_absdev (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            absdev = NAN;
        }
        delete ptVals;
        return absdev;
    }
    
    double SPDMetricCalcCoefficientOfVariationAmplitude::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);
        double cv = 0;
        if(ptVals->size() > 0)
		{
            double sumSq = 0;
            double mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
            for(vector<double>::iterator iterVals; iterVals != ptVals->end(); ++iterVals)
            {
                sumSq += pow(((*iterVals) - mean),2);
            }
            cv = sqrt(sumSq/ptVals->size())/mean;
        }
        else 
        {
            cv = NAN;
        }
        
        delete ptVals;
        return cv;
    }
    
    double SPDMetricCalcPercentileAmplitude::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);
        double quantile = 0;
        if(ptVals->size() > 0)
		{
            double quatFrac = ((float)percentile)/100;
            sort(ptVals->begin(), ptVals->end());
            quantile = gsl_stats_quantile_from_sorted_data(&(*ptVals)[0], 1, ptVals->size(), quatFrac);
        }
        else 
        {
            quantile = NAN;
        }
        delete ptVals;
        return quantile;
    }
    
    double SPDMetricCalcSkewnessAmplitude::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);
        double skew = 0;
        if(ptVals->size() > 0)
		{
            skew = gsl_stats_skew (&(*ptVals)[0], 1, ptVals->size());
        }
        else 
        {
            skew = NAN;
        }
        delete ptVals;
        return skew;
    }
    
    double SPDMetricCalcPersonModeSkewnessAmplitude::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);
        double personModeSkew = 0;
        if(ptVals->size() > 0)
		{
            sort(ptVals->begin(), ptVals->end());
            double mode = this->calcBinnedMode(ptVals, resolution);
            double mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
            double stddev = gsl_stats_sd (&(*ptVals)[0], 1, ptVals->size());
            
            personModeSkew = (mean - mode)/stddev;
        }
        else
        {
            personModeSkew = NAN;
        }
        delete ptVals;
        return personModeSkew;
    }
    
    double SPDMetricCalcPersonMedianSkewnessAmplitude::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);
        double personMedianSkew = 0;
        if(ptVals->size() > 0)
		{
            sort(ptVals->begin(), ptVals->end());
            double median = gsl_stats_median_from_sorted_data(&(*ptVals)[0], 1, ptVals->size());
            double mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
            double stddev = gsl_stats_sd (&(*ptVals)[0], 1, ptVals->size());
            
            personMedianSkew = (mean - median)/stddev;
        }
        else
        {
            personMedianSkew = NAN;
        }
        delete ptVals;
        return personMedianSkew;
    }
    
    double SPDMetricCalcKurtosisAmplitude::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);
        double kurtosis = 0;
        if(ptVals->size() > 0)
		{
            kurtosis = gsl_stats_kurtosis (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            kurtosis = NAN;
        }
        delete ptVals;
        return kurtosis;
    }
    
    double SPDMetricCalcNumReturnsAboveMetricAmplitude::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);
        boost::uint_fast64_t valCount = 0;
        if(ptVals->size() > 0)
		{
            double thresValue = metric->calcValue(pulses, spdFile, geom);
            for(vector<double>::iterator iterVals; iterVals != ptVals->end(); ++iterVals)
            {
                if((*iterVals) > thresValue)
                {
                    ++valCount;
                }
            }
        }
        delete ptVals;
        return valCount;
    }
    
    double SPDMetricCalcNumReturnsBelowMetricAmplitude::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);
        boost::uint_fast64_t valCount = 0;
        if(ptVals->size() > 0)
		{
            double thresValue = metric->calcValue(pulses, spdFile, geom);
            for(vector<double>::iterator iterVals; iterVals != ptVals->end(); ++iterVals)
            {
                if((*iterVals) < thresValue)
                {
                    ++valCount;
                }
            }
        }
        delete ptVals;
        return valCount;
    }
    
    
    
    
    /*
     * Metric's for Range (Spherical Coordinates)
     */
    
    double SPDMetricCalcNumReturnsRange::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        boost::uint_fast64_t numReturns = ptVals->size();
        delete ptVals;
        return numReturns;
    }
    
    double SPDMetricCalcSumRange::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        double sum = 0;
        for(vector<double>::iterator iterVals = ptVals->begin(); iterVals != ptVals->end(); ++iterVals)
        {
            sum += (*iterVals);
        }
        delete ptVals;
        return sum;
    }
    
    double SPDMetricCalcMeanRange::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        double mean = 0;
        if(ptVals->size() > 0)
		{
            mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            mean = NAN;
        }
        delete ptVals;
        return mean;
    }
    
    double SPDMetricCalcMedianRange::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        double median = 0;
        if(ptVals->size() > 0)
		{
            sort(ptVals->begin(), ptVals->end());
            median = gsl_stats_median_from_sorted_data(&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            median = NAN;
        }
        delete ptVals;
        return median;
    }
    
    double SPDMetricCalcModeRange::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        double mode = 0;
        if(ptVals->size() > 0)
        {
            mode = this->calcBinnedMode(ptVals, resolution);
        }
        else
        {
            mode = NAN;
        }
        
        delete ptVals;
        return mode;
    }
    
    double SPDMetricCalcMinRange::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        double min = 0;
        if(ptVals->size() > 0)
		{
            min = gsl_stats_min (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            min = NAN;
        }
        delete ptVals;
        return min;
    }
    
    double SPDMetricCalcMaxRange::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        double max = 0;
        if(ptVals->size() > 0)
		{
            max = gsl_stats_max (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            max = NAN;
        }
        delete ptVals;
        return max;
    }
    
    double SPDMetricCalcStdDevRange::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        double stddev = 0;
        if(ptVals->size() > 0)
		{
            stddev = gsl_stats_sd (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            stddev = NAN;
        }
        delete ptVals;
        return stddev;
    }
    
    double SPDMetricCalcVarianceRange::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        double variance = 0;
        if(ptVals->size() > 0)
		{
            variance = gsl_stats_variance (&(*ptVals)[0], 1, ptVals->size());
        }
        else 
        {
            variance = NAN;
        }
        delete ptVals;
        return variance;
    }
    
    double SPDMetricCalcAbsDeviationRange::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        double absdev = 0;
        if(ptVals->size() > 0)
		{
            absdev = gsl_stats_absdev (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            absdev = NAN;
        }
        delete ptVals;
        return absdev;
    }
    
    double SPDMetricCalcCoefficientOfVariationRange::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        double cv = 0;
        if(ptVals->size() > 0)
		{
            double sumSq = 0;
            double mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
            for(vector<double>::iterator iterVals; iterVals != ptVals->end(); ++iterVals)
            {
                sumSq += pow(((*iterVals) - mean),2);
            }
            cv = sqrt(sumSq/ptVals->size())/mean;
        }
        else 
        {
            cv = NAN;
        }
        
        delete ptVals;
        return cv;
    }
    
    double SPDMetricCalcPercentileRange::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        double quantile = 0;
        if(ptVals->size() > 0)
		{
            double quatFrac = ((float)percentile)/100;
            sort(ptVals->begin(), ptVals->end());
            quantile = gsl_stats_quantile_from_sorted_data(&(*ptVals)[0], 1, ptVals->size(), quatFrac);
        }
        else 
        {
            quantile = NAN;
        }
        delete ptVals;
        return quantile;
    }
    
    double SPDMetricCalcSkewnessRange::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        double skew = 0;
        if(ptVals->size() > 0)
		{
            skew = gsl_stats_skew (&(*ptVals)[0], 1, ptVals->size());
        }
        else 
        {
            skew = NAN;
        }
        delete ptVals;
        return skew;
    }
    
    double SPDMetricCalcPersonModeSkewnessRange::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        double personModeSkew = 0;
        if(ptVals->size() > 0)
		{
            sort(ptVals->begin(), ptVals->end());
            double mode = this->calcBinnedMode(ptVals, resolution);
            double mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
            double stddev = gsl_stats_sd (&(*ptVals)[0], 1, ptVals->size());
            
            personModeSkew = (mean - mode)/stddev;
        }
        else
        {
            personModeSkew = NAN;
        }
        delete ptVals;
        return personModeSkew;
    }
    
    double SPDMetricCalcPersonMedianSkewnessRange::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        double personMedianSkew = 0;
        if(ptVals->size() > 0)
		{
            sort(ptVals->begin(), ptVals->end());
            double median = gsl_stats_median_from_sorted_data(&(*ptVals)[0], 1, ptVals->size());
            double mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
            double stddev = gsl_stats_sd (&(*ptVals)[0], 1, ptVals->size());
            
            personMedianSkew = (mean - median)/stddev;
        }
        else
        {
            personMedianSkew = NAN;
        }
        delete ptVals;
        return personMedianSkew;
    }
    
    double SPDMetricCalcKurtosisRange::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        double kurtosis = 0;
        if(ptVals->size() > 0)
		{
            kurtosis = gsl_stats_kurtosis (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            kurtosis = NAN;
        }
        delete ptVals;
        return kurtosis;
    }
    
    double SPDMetricCalcNumReturnsAboveMetricRange::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        boost::uint_fast64_t valCount = 0;
        if(ptVals->size() > 0)
		{
            double thresValue = metric->calcValue(pulses, spdFile, geom);
            for(vector<double>::iterator iterVals; iterVals != ptVals->end(); ++iterVals)
            {
                if((*iterVals) > thresValue)
                {
                    ++valCount;
                }
            }
        }
        delete ptVals;
        return valCount;
    }
    
    double SPDMetricCalcNumReturnsBelowMetricRange::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        boost::uint_fast64_t valCount = 0;
        if(ptVals->size() > 0)
		{
            double thresValue = metric->calcValue(pulses, spdFile, geom);
            for(vector<double>::iterator iterVals; iterVals != ptVals->end(); ++iterVals)
            {
                if((*iterVals) < thresValue)
                {
                    ++valCount;
                }
            }
        }
        delete ptVals;
        return valCount;
    }
    
    double SPDMetricCalcWeibullAlphaRange::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        double weibullAlpha = 0;
        try 
        {
            if(pulses->size() > 0)
            {
                vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
                if(ptVals->size() > 0)
                {
                    double minH = 0; 
                    double maxH = 0;
                    gsl_stats_minmax(&minH, &maxH, &(*ptVals)[0], 1, ptVals->size());
                    if(minH > 0)
                    {
                        minH = 0;
                    }
                    
                    boost::uint_fast32_t numBins = 0;
                    double *bins = this->binData(ptVals, this->resolution, &numBins, minH, maxH);
                    size_t maxIdx =  gsl_stats_max_index (bins, 1, numBins);
                    
                    WeibullFitVals *fitData = new WeibullFitVals();
                    fitData->heights = new double[numBins];
                    fitData->binVals = new double[numBins];
                    fitData->error = new double[numBins];
                    double binSums = 0;
                    for(boost::uint_fast32_t i = 0; i < numBins; ++i)
                    {
                        binSums += bins[i];
                        fitData->heights[i] = minH + (resolution/2) + (resolution * ((double)i));
                        fitData->error[i] = 1;
                    }
                    
                    // Make area == 1
                    for(boost::uint_fast32_t i = 0; i < numBins; ++i)
                    {
                        fitData->binVals[i] = bins[i]/binSums;
                    }
                    
                    // parameters[0] - Alpha
                    // parameters[1] - Beta
                    double *parameters = new double[2];                    
                    mp_par *paramConstraints = new mp_par[2];
                    
                    parameters[0] = fitData->binVals[maxIdx]; // Alpha
                    paramConstraints[0].fixed = false;
					paramConstraints[0].limited[0] = true;
					paramConstraints[0].limited[1] = true;
					paramConstraints[0].limits[0] = 0.00001;
					paramConstraints[0].limits[1] = 0.99999;
					paramConstraints[0].parname = const_cast<char*>(string("Alpha").c_str());;
					paramConstraints[0].step = 0;
					paramConstraints[0].relstep = 0;
					paramConstraints[0].side = 0;
					paramConstraints[0].deriv_debug = 0;
					
                    double percent20height = (maxH - minH) * 0.2;
					parameters[1] = minH + (((double)maxIdx) * resolution); // Beta
					paramConstraints[1].fixed = false;
					paramConstraints[1].limited[0] = true;
					paramConstraints[1].limited[1] = true;
					paramConstraints[1].limits[0] = parameters[1] - percent20height;
                    if(paramConstraints[1].limits[0] < minH)
                    {
                        paramConstraints[1].limits[0] = minH;
                    }
					paramConstraints[1].limits[1] = parameters[1] + percent20height;
                    if(paramConstraints[1].limits[1] > maxH)
                    {
                        paramConstraints[1].limits[1] = maxH;
                    }
					paramConstraints[1].parname = const_cast<char*>(string("Beta").c_str());;
					paramConstraints[1].step = 0;
					paramConstraints[1].relstep = 0;
					paramConstraints[1].side = 0;
					paramConstraints[1].deriv_debug = 0;
                    
                    mpResultsValues->bestnorm = 0;
					mpResultsValues->orignorm = 0;
					mpResultsValues->niter = 0;
					mpResultsValues->nfev = 0;
					mpResultsValues->status = 0;
					mpResultsValues->npar = 0;
					mpResultsValues->nfree = 0;
					mpResultsValues->npegged = 0;
					mpResultsValues->nfunc = 0;
					mpResultsValues->resid = 0;
					mpResultsValues->xerror = 0;
					mpResultsValues->covar = 0; // Not being retrieved
                    
                    /*
					 * int m     - number of data points
					 * int npar  - number of parameters
					 * double *xall - parameters values (initial values and then best fit values)
					 * mp_par *pars - Constrains
					 * mp_config *config - Configuration parameters
					 * void *private_data - Waveform data structure
					 * mp_result *result - diagnostic info from function
					 */
					int returnCode = mpfit(weibullFit, numBins, 2, parameters, paramConstraints, mpConfigValues, fitData, mpResultsValues);
					if((returnCode == MP_OK_CHI) | (returnCode == MP_OK_PAR) |
					   (returnCode == MP_OK_BOTH) | (returnCode == MP_OK_DIR) |
					   (returnCode == MP_MAXITER) | (returnCode == MP_FTOL)
					   | (returnCode == MP_XTOL) | (returnCode == MP_XTOL))
					{
						// MP Fit completed.. On on debug_info for more information. 
					}
					else if(returnCode == MP_ERR_INPUT)
					{
						throw SPDProcessingException("mpfit - Check inputs.");
					}
					else if(returnCode == MP_ERR_NAN)
					{
						throw SPDProcessingException("mpfit - Weibull fit function produced NaN value.");
					}
					else if(returnCode == MP_ERR_FUNC)
					{
						throw SPDProcessingException("mpfit - No Weibull fit function was supplied.");
					}
					else if(returnCode == MP_ERR_NPOINTS)
					{
						throw SPDProcessingException("mpfit - No data points were supplied.");
					}
					else if(returnCode == MP_ERR_NFREE)
					{
						throw SPDProcessingException("mpfit - No parameters are free - i.e., nothing to optimise!");
					}
					else if(returnCode == MP_ERR_MEMORY)
					{
						throw SPDProcessingException("mpfit - memory allocation error - may have run out!");
					}
					else if(returnCode == MP_ERR_INITBOUNDS)
					{
						throw SPDProcessingException("mpfit - Initial parameter values inconsistant with constraints.");
					}
					else if(returnCode == MP_ERR_PARAM)
					{
						throw SPDProcessingException("mpfit - An error has occur with an input parameter.");
					}
					else if(returnCode == MP_ERR_DOF)
					{
						throw SPDProcessingException("mpfit - Not enough degrees of freedom.");
					}
					else 
					{
						cout << "Return code is :" << returnCode << " - this can not been defined!\n";
					}
                    
                    weibullAlpha = parameters[0];
                    
                    delete[] parameters;                    
                    delete[] paramConstraints;
                    delete[] bins;
                }
                else
                {
                    weibullAlpha = NAN;
                }
                
            }
            else
            {
                weibullAlpha = NAN;
            }
            
        } 
        catch (SPDProcessingException &e) 
        {
            throw e;
        }
        
        return weibullAlpha;
    }
    
    double SPDMetricCalcWeibullBetaRange::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        double weibullBeta = 0;
        try 
        {
            if(pulses->size() > 0)
            {
                vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
                if(ptVals->size() > 0)
                {
                    double minH = 0; 
                    double maxH = 0;
                    gsl_stats_minmax(&minH, &maxH, &(*ptVals)[0], 1, ptVals->size());
                    if(minH > 0)
                    {
                        minH = 0;
                    }
                    
                    boost::uint_fast32_t numBins = 0;
                    double *bins = this->binData(ptVals, this->resolution, &numBins, minH, maxH);
                    size_t maxIdx =  gsl_stats_max_index (bins, 1, numBins);
                    
                    WeibullFitVals *fitData = new WeibullFitVals();
                    fitData->heights = new double[numBins];
                    fitData->binVals = new double[numBins];
                    fitData->error = new double[numBins];
                    double binSums = 0;
                    for(boost::uint_fast32_t i = 0; i < numBins; ++i)
                    {
                        binSums += bins[i];
                        fitData->heights[i] = minH + (resolution/2) + (resolution * ((double)i));
                        fitData->error[i] = 1;
                    }
                    
                    // Make area == 1
                    for(boost::uint_fast32_t i = 0; i < numBins; ++i)
                    {
                        fitData->binVals[i] = bins[i]/binSums;
                    }
                    
                    // parameters[0] - Alpha
                    // parameters[1] - Beta
                    double *parameters = new double[2];                    
                    mp_par *paramConstraints = new mp_par[2];
                    
                    parameters[0] = fitData->binVals[maxIdx]; // Alpha
                    paramConstraints[0].fixed = false;
					paramConstraints[0].limited[0] = true;
					paramConstraints[0].limited[1] = true;
					paramConstraints[0].limits[0] = 0.00001;
					paramConstraints[0].limits[1] = 0.99999;
					paramConstraints[0].parname = const_cast<char*>(string("Alpha").c_str());;
					paramConstraints[0].step = 0;
					paramConstraints[0].relstep = 0;
					paramConstraints[0].side = 0;
					paramConstraints[0].deriv_debug = 0;
					
                    double percent20height = (maxH - minH) * 0.2;
					parameters[1] = minH + (((double)maxIdx) * resolution); // Beta
					paramConstraints[1].fixed = false;
					paramConstraints[1].limited[0] = true;
					paramConstraints[1].limited[1] = true;
					paramConstraints[1].limits[0] = parameters[1] - percent20height;
                    if(paramConstraints[1].limits[0] < minH)
                    {
                        paramConstraints[1].limits[0] = minH;
                    }
					paramConstraints[1].limits[1] = parameters[1] + percent20height;
                    if(paramConstraints[1].limits[1] > maxH)
                    {
                        paramConstraints[1].limits[1] = maxH;
                    }
					paramConstraints[1].parname = const_cast<char*>(string("Beta").c_str());;
					paramConstraints[1].step = 0;
					paramConstraints[1].relstep = 0;
					paramConstraints[1].side = 0;
					paramConstraints[1].deriv_debug = 0;
                    
                    mpResultsValues->bestnorm = 0;
					mpResultsValues->orignorm = 0;
					mpResultsValues->niter = 0;
					mpResultsValues->nfev = 0;
					mpResultsValues->status = 0;
					mpResultsValues->npar = 0;
					mpResultsValues->nfree = 0;
					mpResultsValues->npegged = 0;
					mpResultsValues->nfunc = 0;
					mpResultsValues->resid = 0;
					mpResultsValues->xerror = 0;
					mpResultsValues->covar = 0; // Not being retrieved
                    
                    /*
					 * int m     - number of data points
					 * int npar  - number of parameters
					 * double *xall - parameters values (initial values and then best fit values)
					 * mp_par *pars - Constrains
					 * mp_config *config - Configuration parameters
					 * void *private_data - Waveform data structure
					 * mp_result *result - diagnostic info from function
					 */
					int returnCode = mpfit(weibullFit, numBins, 2, parameters, paramConstraints, mpConfigValues, fitData, mpResultsValues);
					if((returnCode == MP_OK_CHI) | (returnCode == MP_OK_PAR) |
					   (returnCode == MP_OK_BOTH) | (returnCode == MP_OK_DIR) |
					   (returnCode == MP_MAXITER) | (returnCode == MP_FTOL)
					   | (returnCode == MP_XTOL) | (returnCode == MP_XTOL))
					{
						// MP Fit completed.. On on debug_info for more information. 
					}
					else if(returnCode == MP_ERR_INPUT)
					{
						throw SPDProcessingException("mpfit - Check inputs.");
					}
					else if(returnCode == MP_ERR_NAN)
					{
						throw SPDProcessingException("mpfit - Weibull fit function produced NaN value.");
					}
					else if(returnCode == MP_ERR_FUNC)
					{
						throw SPDProcessingException("mpfit - No Weibull fit function was supplied.");
					}
					else if(returnCode == MP_ERR_NPOINTS)
					{
						throw SPDProcessingException("mpfit - No data points were supplied.");
					}
					else if(returnCode == MP_ERR_NFREE)
					{
						throw SPDProcessingException("mpfit - No parameters are free - i.e., nothing to optimise!");
					}
					else if(returnCode == MP_ERR_MEMORY)
					{
						throw SPDProcessingException("mpfit - memory allocation error - may have run out!");
					}
					else if(returnCode == MP_ERR_INITBOUNDS)
					{
						throw SPDProcessingException("mpfit - Initial parameter values inconsistant with constraints.");
					}
					else if(returnCode == MP_ERR_PARAM)
					{
						throw SPDProcessingException("mpfit - An error has occur with an input parameter.");
					}
					else if(returnCode == MP_ERR_DOF)
					{
						throw SPDProcessingException("mpfit - Not enough degrees of freedom.");
					}
					else 
					{
						cout << "Return code is :" << returnCode << " - this can not been defined!\n";
					}
                    
                    weibullBeta = parameters[1];
                    
                    delete[] parameters;                    
                    delete[] paramConstraints;
                    delete[] bins;
                }
                else
                {
                    weibullBeta = NAN;
                }
                
            }
            else
            {
                weibullBeta = NAN;
            }
            
        } 
        catch (SPDProcessingException &e) 
        {
            throw e;
        }
        
        return weibullBeta;
    }
    
    double SPDMetricCalcWeibullQuantileRangeRange::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        double weibullQuantileRange = 0;
        try 
        {
            if(pulses->size() > 0)
            {
                vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
                if(ptVals->size() > 0)
                {
                    double minH = 0; 
                    double maxH = 0;
                    gsl_stats_minmax(&minH, &maxH, &(*ptVals)[0], 1, ptVals->size());
                    if(minH > 0)
                    {
                        minH = 0;
                    }
                    
                    boost::uint_fast32_t numBins = 0;
                    double *bins = this->binData(ptVals, this->resolution, &numBins, minH, maxH);
                    size_t maxIdx =  gsl_stats_max_index (bins, 1, numBins);
                    
                    WeibullFitVals *fitData = new WeibullFitVals();
                    fitData->heights = new double[numBins];
                    fitData->binVals = new double[numBins];
                    fitData->error = new double[numBins];
                    double binSums = 0;
                    for(boost::uint_fast32_t i = 0; i < numBins; ++i)
                    {
                        binSums += bins[i];
                        fitData->heights[i] = minH + (resolution/2) + (resolution * ((double)i));
                        fitData->error[i] = 1;
                    }
                    
                    // Make area == 1
                    for(boost::uint_fast32_t i = 0; i < numBins; ++i)
                    {
                        fitData->binVals[i] = bins[i]/binSums;
                    }
                    
                    // parameters[0] - Alpha
                    // parameters[1] - Beta
                    double *parameters = new double[2];                    
                    mp_par *paramConstraints = new mp_par[2];
                    
                    parameters[0] = fitData->binVals[maxIdx]; // Alpha
                    paramConstraints[0].fixed = false;
					paramConstraints[0].limited[0] = true;
					paramConstraints[0].limited[1] = true;
					paramConstraints[0].limits[0] = 0.00001;
					paramConstraints[0].limits[1] = 0.99999;
					paramConstraints[0].parname = const_cast<char*>(string("Alpha").c_str());;
					paramConstraints[0].step = 0;
					paramConstraints[0].relstep = 0;
					paramConstraints[0].side = 0;
					paramConstraints[0].deriv_debug = 0;
					
                    double percent20height = (maxH - minH) * 0.2;
					parameters[1] = minH + (((double)maxIdx) * resolution); // Beta
					paramConstraints[1].fixed = false;
					paramConstraints[1].limited[0] = true;
					paramConstraints[1].limited[1] = true;
					paramConstraints[1].limits[0] = parameters[1] - percent20height;
                    if(paramConstraints[1].limits[0] < minH)
                    {
                        paramConstraints[1].limits[0] = minH;
                    }
					paramConstraints[1].limits[1] = parameters[1] + percent20height;
                    if(paramConstraints[1].limits[1] > maxH)
                    {
                        paramConstraints[1].limits[1] = maxH;
                    }
					paramConstraints[1].parname = const_cast<char*>(string("Beta").c_str());;
					paramConstraints[1].step = 0;
					paramConstraints[1].relstep = 0;
					paramConstraints[1].side = 0;
					paramConstraints[1].deriv_debug = 0;
                    
                    mpResultsValues->bestnorm = 0;
					mpResultsValues->orignorm = 0;
					mpResultsValues->niter = 0;
					mpResultsValues->nfev = 0;
					mpResultsValues->status = 0;
					mpResultsValues->npar = 0;
					mpResultsValues->nfree = 0;
					mpResultsValues->npegged = 0;
					mpResultsValues->nfunc = 0;
					mpResultsValues->resid = 0;
					mpResultsValues->xerror = 0;
					mpResultsValues->covar = 0; // Not being retrieved
                    
                    /*
					 * int m     - number of data points
					 * int npar  - number of parameters
					 * double *xall - parameters values (initial values and then best fit values)
					 * mp_par *pars - Constrains
					 * mp_config *config - Configuration parameters
					 * void *private_data - Waveform data structure
					 * mp_result *result - diagnostic info from function
					 */
					int returnCode = mpfit(weibullFit, numBins, 2, parameters, paramConstraints, mpConfigValues, fitData, mpResultsValues);
					if((returnCode == MP_OK_CHI) | (returnCode == MP_OK_PAR) |
					   (returnCode == MP_OK_BOTH) | (returnCode == MP_OK_DIR) |
					   (returnCode == MP_MAXITER) | (returnCode == MP_FTOL)
					   | (returnCode == MP_XTOL) | (returnCode == MP_XTOL))
					{
						// MP Fit completed.. On on debug_info for more information. 
					}
					else if(returnCode == MP_ERR_INPUT)
					{
						throw SPDProcessingException("mpfit - Check inputs.");
					}
					else if(returnCode == MP_ERR_NAN)
					{
						throw SPDProcessingException("mpfit - Weibull fit function produced NaN value.");
					}
					else if(returnCode == MP_ERR_FUNC)
					{
						throw SPDProcessingException("mpfit - No Weibull fit function was supplied.");
					}
					else if(returnCode == MP_ERR_NPOINTS)
					{
						throw SPDProcessingException("mpfit - No data points were supplied.");
					}
					else if(returnCode == MP_ERR_NFREE)
					{
						throw SPDProcessingException("mpfit - No parameters are free - i.e., nothing to optimise!");
					}
					else if(returnCode == MP_ERR_MEMORY)
					{
						throw SPDProcessingException("mpfit - memory allocation error - may have run out!");
					}
					else if(returnCode == MP_ERR_INITBOUNDS)
					{
						throw SPDProcessingException("mpfit - Initial parameter values inconsistant with constraints.");
					}
					else if(returnCode == MP_ERR_PARAM)
					{
						throw SPDProcessingException("mpfit - An error has occur with an input parameter.");
					}
					else if(returnCode == MP_ERR_DOF)
					{
						throw SPDProcessingException("mpfit - Not enough degrees of freedom.");
					}
					else 
					{
						cout << "Return code is :" << returnCode << " - this can not been defined!\n";
					}
                    
                    weibullQuantileRange = 0; // Need to calculate... TODO
                    
                    delete[] parameters;                    
                    delete[] paramConstraints;
                    delete[] bins;
                }
                else
                {
                    weibullQuantileRange = NAN;
                }
                
            }
            else
            {
                weibullQuantileRange = NAN;
            }
            
        } 
        catch (SPDProcessingException &e) 
        {
            throw e;
        }
        
        return weibullQuantileRange;
    }
    
    
    
    /*
     * Metric's for width
     */
    
    double SPDMetricCalcNumReturnsWidth::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        boost::uint_fast64_t numReturns = ptVals->size();
        delete ptVals;
        return numReturns;
    }
    
    double SPDMetricCalcSumWidth::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        double sum = 0;
        for(vector<double>::iterator iterVals = ptVals->begin(); iterVals != ptVals->end(); ++iterVals)
        {
            sum += (*iterVals);
        }
        delete ptVals;
        return sum;
    }
    
    double SPDMetricCalcMeanWidth::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        double mean = 0;
        if(ptVals->size() > 0)
		{
            mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            mean = NAN;
        }
        delete ptVals;
        return mean;
    }
    
    double SPDMetricCalcMedianWidth::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        double median = 0;
        if(ptVals->size() > 0)
		{
            sort(ptVals->begin(), ptVals->end());
            median = gsl_stats_median_from_sorted_data(&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            median = NAN;
        }
        delete ptVals;
        return median;
    }
    
    double SPDMetricCalcModeWidth::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        double mode = 0;
        if(ptVals->size() > 0)
        {
            mode = this->calcBinnedMode(ptVals, resolution);
        }
        else
        {
            mode = NAN;
        }
        
        delete ptVals;
        return mode;
    }
    
    double SPDMetricCalcMinWidth::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        double min = 0;
        if(ptVals->size() > 0)
		{
            min = gsl_stats_min (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            min = NAN;
        }
        delete ptVals;
        return min;
    }
    
    double SPDMetricCalcMaxWidth::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        double max = 0;
        if(ptVals->size() > 0)
		{
            max = gsl_stats_max (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            max = NAN;
        }
        delete ptVals;
        return max;
    }
    
    double SPDMetricCalcStdDevWidth::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        double stddev = 0;
        if(ptVals->size() > 0)
		{
            stddev = gsl_stats_sd (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            stddev = NAN;
        }
        delete ptVals;
        return stddev;
    }
    
    double SPDMetricCalcVarianceWidth::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        double variance = 0;
        if(ptVals->size() > 0)
		{
            variance = gsl_stats_variance (&(*ptVals)[0], 1, ptVals->size());
        }
        else 
        {
            variance = NAN;
        }
        delete ptVals;
        return variance;
    }
    
    double SPDMetricCalcAbsDeviationWidth::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        double absdev = 0;
        if(ptVals->size() > 0)
		{
            absdev = gsl_stats_absdev (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            absdev = NAN;
        }
        delete ptVals;
        return absdev;
    }
    
    double SPDMetricCalcCoefficientOfVariationWidth::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        double cv = 0;
        if(ptVals->size() > 0)
		{
            double sumSq = 0;
            double mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
            for(vector<double>::iterator iterVals; iterVals != ptVals->end(); ++iterVals)
            {
                sumSq += pow(((*iterVals) - mean),2);
            }
            cv = sqrt(sumSq/ptVals->size())/mean;
        }
        else 
        {
            cv = NAN;
        }
        
        delete ptVals;
        return cv;
    }
    
    double SPDMetricCalcPercentileWidth::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        double quantile = 0;
        if(ptVals->size() > 0)
		{
            double quatFrac = ((float)percentile)/100;
            sort(ptVals->begin(), ptVals->end());
            quantile = gsl_stats_quantile_from_sorted_data(&(*ptVals)[0], 1, ptVals->size(), quatFrac);
        }
        else 
        {
            quantile = NAN;
        }
        delete ptVals;
        return quantile;
    }
    
    double SPDMetricCalcSkewnessWidth::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        double skew = 0;
        if(ptVals->size() > 0)
		{
            skew = gsl_stats_skew (&(*ptVals)[0], 1, ptVals->size());
        }
        else 
        {
            skew = NAN;
        }
        delete ptVals;
        return skew;
    }
    
    double SPDMetricCalcPersonModeSkewnessWidth::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        double personModeSkew = 0;
        if(ptVals->size() > 0)
		{
            sort(ptVals->begin(), ptVals->end());
            double mode = this->calcBinnedMode(ptVals, resolution);
            double mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
            double stddev = gsl_stats_sd (&(*ptVals)[0], 1, ptVals->size());
            
            personModeSkew = (mean - mode)/stddev;
        }
        else
        {
            personModeSkew = NAN;
        }
        delete ptVals;
        return personModeSkew;
    }
    
    double SPDMetricCalcPersonMedianSkewnessWidth::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        double personMedianSkew = 0;
        if(ptVals->size() > 0)
		{
            sort(ptVals->begin(), ptVals->end());
            double median = gsl_stats_median_from_sorted_data(&(*ptVals)[0], 1, ptVals->size());
            double mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
            double stddev = gsl_stats_sd (&(*ptVals)[0], 1, ptVals->size());
            
            personMedianSkew = (mean - median)/stddev;
        }
        else
        {
            personMedianSkew = NAN;
        }
        delete ptVals;
        return personMedianSkew;
    }
    
    double SPDMetricCalcKurtosisWidth::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        double kurtosis = 0;
        if(ptVals->size() > 0)
		{
            kurtosis = gsl_stats_kurtosis (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            kurtosis = NAN;
        }
        delete ptVals;
        return kurtosis;
    }
    
    double SPDMetricCalcNumReturnsAboveMetricWidth::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        boost::uint_fast64_t valCount = 0;
        if(ptVals->size() > 0)
		{
            double thresValue = metric->calcValue(pulses, spdFile, geom);
            for(vector<double>::iterator iterVals; iterVals != ptVals->end(); ++iterVals)
            {
                if((*iterVals) > thresValue)
                {
                    ++valCount;
                }
            }
        }
        delete ptVals;
        return valCount;
    }
    
    double SPDMetricCalcNumReturnsBelowMetricWidth::calcValue(vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) throw(SPDProcessingException)
    {
        vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        boost::uint_fast64_t valCount = 0;
        if(ptVals->size() > 0)
		{
            double thresValue = metric->calcValue(pulses, spdFile, geom);
            for(vector<double>::iterator iterVals; iterVals != ptVals->end(); ++iterVals)
            {
                if((*iterVals) < thresValue)
                {
                    ++valCount;
                }
            }
        }
        delete ptVals;
        return valCount;
    }

}
