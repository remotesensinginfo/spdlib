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

    double SPDMetricCalcNumPulses::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
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
                for(std::vector<SPDPulse*>::iterator iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
                {
                    if((*iterPulses)->numberOfReturns >= minNumReturns)
                    {
                        ++numPulses;
                    }
                }
            }
            else if(spdFile->getReceiveWaveformDefined() == SPD_TRUE)
            {
                for(std::vector<SPDPulse*>::iterator iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
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


    double SPDMetricCalcCanopyCover::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        double canopyCover = 0;
        try
        {
            if(pulses->size() > 0)
            {
                // 1) Get points associated with parameters provided.
                std::vector<SPDPoint*> *points = this->getPointsWithinHeightParameters(pulses, spdFile, geom);

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

                    std::vector<SPDPoint*> *ptsInRadius = new std::vector<SPDPoint*>();
                    std::vector<SPDPoint*>::iterator iterPts;
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

    double SPDMetricCalcCanopyCoverPercent::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        double canopyCover = 0;
        try
        {
            if(pulses->size() > 0)
            {
                // 1) Get points associated with parameters provided.
                std::vector<SPDPoint*> *points = this->getPointsWithinHeightParameters(pulses, spdFile, geom);

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

                    std::vector<SPDPoint*> *ptsInRadius = new std::vector<SPDPoint*>();
                    //std::vector<SPDPoint*>::iterator iterPts;
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
                            //std::cout << "Cell [" << i << "," << j << "]\t[" << cellX << "," << cellY << "] - ";
                            if(geom->Contains(pt))
                            {
                                //std::cout << "CONTAINED\n";
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
                                //    std::cout << "NOT CONTAINED\n";
                                //}
                            cellX += resolution;
                        }
                        //std::cout << std::endl << std::endl;
                        cellY -= resolution;
                    }
                    delete pt;
                    delete ptsInRadius;

                    //std::cout << "Number of Cells = "  << allCellCount << " of which " << canopyCellCount << " are canopy\n";

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

        //std::cout << "Returning = " << canopyCover << std::endl;
        return canopyCover;
    }




    /*
     * Metric's for height
     */

    double SPDMetricCalcLeeOpennessHeight::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        double openness = 0;

        if(ptVals->size() > 0)
		{
            double max = gsl_stats_max (&(*ptVals)[0], 1, ptVals->size());
            if(!boost::math::isnan(max))
            {
                boost::uint_fast32_t nBins = ceil(((double)max/vRes))+1;

                //std::cout << "\nNumber of bins = " << nBins << std::endl;
                if((nBins > 0) & (nBins < 1000))
                {
                    double *bins = new double[nBins];
                    bool *binsFirst = new bool[nBins];
                    for(boost::uint_fast32_t i = 0; i < nBins; ++i)
                    {
                        bins[i] = 0;
                        binsFirst[i] = true;
                    }

                    boost::uint_fast32_t idx = 0;
                    for(std::vector<double>::iterator iterVals = ptVals->begin(); iterVals != ptVals->end(); ++iterVals)
                    {
                        try
                        {
                            if((*iterVals) > 0)
                            {
                                idx = boost::numeric_cast<boost::uint_fast32_t>((*iterVals)/vRes);
                            }
                        }
                        catch(boost::numeric::negative_overflow& e)
                        {
                            std::cout << "(*iterVals) = " << (*iterVals) << std::endl;
                            std::cout << "vRes = " << vRes << std::endl;
                            throw SPDProcessingException(e.what());
                        }
                        catch(boost::numeric::positive_overflow& e)
                        {
                            std::cout << "(*iterVals) = " << (*iterVals) << std::endl;
                            std::cout << "vRes = " << vRes << std::endl;
                            throw SPDProcessingException(e.what());
                        }
                        catch(boost::numeric::bad_numeric_cast& e)
                        {
                            std::cout << "(*iterVals) = " << (*iterVals) << std::endl;
                            std::cout << "vRes = " << vRes << std::endl;
                            throw SPDProcessingException(e.what());
                        }

                        if(idx >= nBins)
                        {
                            std::cout << "Value = " << *iterVals << std::endl;
                            std::cout << "Max Value = " << max << std::endl;
                            std::cout << "idx = " << idx << std::endl;
                            std::cout << "nBins = " << nBins << std::endl;
                            std::cout << "vRes = " << vRes << std::endl;

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
                    //std::cout << "Number of voxels = " << numVoxels << std::endl;

                    for(boost::uint_fast32_t i = 0; i < nBins; ++i)
                    {
                        if(!binsFirst[i])
                        {
                            //std::cout << bins[i] << ",";
                            openness += (((max-bins[i])/max) / numVoxels);
                        }
                    }
                    //std::cout << std::endl;
                    //std::cout << "openness = " << openness << std::endl;
                    openness = openness * 100;
                    //std::cout << "openness = " << openness << std::endl;
                }
                else
                {
                    openness = std::numeric_limits<float>::signaling_NaN();
                }
            }
            else
            {
                openness = std::numeric_limits<float>::signaling_NaN();
            }
        }
        else
        {
            openness = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return openness;
    }

    double SPDMetricCalcNumReturnsHeight::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        boost::uint_fast64_t numReturns = ptVals->size();
        delete ptVals;
        return numReturns;
    }

    double SPDMetricCalcSumHeight::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        double sum = 0;
        for(std::vector<double>::iterator iterVals = ptVals->begin(); iterVals != ptVals->end(); ++iterVals)
        {
            sum += (*iterVals);
        }
        delete ptVals;
        return sum;
    }

    double SPDMetricCalcMeanHeight::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        double mean = 0;
        if(ptVals->size() > 0)
		{
            mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            mean = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return mean;
    }

    double SPDMetricCalcMedianHeight::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        double median = 0;
        if(ptVals->size() > 0)
		{
            std::sort(ptVals->begin(), ptVals->end());
            median = gsl_stats_median_from_sorted_data(&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            median = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return median;
    }

    double SPDMetricCalcModeHeight::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        double mode = 0;
        if(ptVals->size() > 0)
        {
            mode = this->calcBinnedMode(ptVals, resolution);
        }
        else
        {
            mode = std::numeric_limits<float>::signaling_NaN();
        }

        delete ptVals;
        return mode;
    }

    double SPDMetricCalcMinHeight::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        double min = 0;
        if(ptVals->size() > 0)
		{
            min = gsl_stats_min (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            min = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return min;
    }

    double SPDMetricCalcMaxHeight::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        double max = 0;
        if(ptVals->size() > 0)
		{
            max = gsl_stats_max (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            max = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return max;
    }


    double SPDMetricCalcDominantHeight::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
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

                for(std::vector<SPDPulse*>::iterator iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
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
                        for(std::vector<SPDPoint*>::iterator iterPoints = (*iterPulses)->pts->begin(); iterPoints != (*iterPulses)->pts->end(); ++iterPoints)
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
                    roundingAddition = boost::numeric_cast<boost::uint_fast32_t>(1/resolution);
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
					xBins = boost::numeric_cast<boost::uint_fast32_t>((width/resolution))+roundingAddition;
                    yBins = boost::numeric_cast<boost::uint_fast32_t>((height/resolution))+roundingAddition;
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

                std::vector<SPDPulse*> ***plsGrd = new std::vector<SPDPulse*>**[yBins];
                for(boost::uint_fast32_t i = 0; i < yBins; ++i)
                {
                    plsGrd[i] = new std::vector<SPDPulse*>*[xBins];
                    for(boost::uint_fast32_t j = 0; j < xBins; ++j)
                    {
                        plsGrd[i][j] = new std::vector<SPDPulse*>();
                    }
                }

                double xDiff = 0;
                double yDiff = 0;
                boost::uint_fast32_t xIdx = 0;
                boost::uint_fast32_t yIdx = 0;

                for(std::vector<SPDPulse*>::iterator iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
                {
                    xDiff = ((*iterPulses)->xIdx - xMin)/resolution;
                    yDiff = (yMax - (*iterPulses)->yIdx)/resolution;

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

                    if(xIdx > ((xBins)-1))
                    {
                        std::cout << "Point: [" << (*iterPulses)->xIdx << "," << (*iterPulses)->yIdx << "]\n";
                        std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
                        std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
                        std::cout << "Size [" << xBins << "," << yBins << "]\n";
                        throw SPDProcessingException("Did not find x index within range.");
                    }

                    if(yIdx > ((yBins)-1))
                    {
                        std::cout << "Point: [" << (*iterPulses)->xIdx << "," << (*iterPulses)->yIdx << "]\n";
                        std::cout << "Diff [" << xDiff << "," << yDiff << "]\n";
                        std::cout << "Index [" << xIdx << "," << yIdx << "]\n";
                        std::cout << "Size [" << xBins << "," << yBins << "]\n";
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
                            std::vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(plsGrd[i][j], spdFile, geom);
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
                    dominantHeight = std::numeric_limits<float>::signaling_NaN();
                }
                else
                {
                    dominantHeight = heightSum/cellCount;
                }
            }
            else
            {
                dominantHeight = std::numeric_limits<float>::signaling_NaN();
            }
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
        return dominantHeight;
    }

    double SPDMetricCalcStdDevHeight::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        double stddev = 0;
        if(ptVals->size() > 0)
		{
            stddev = gsl_stats_sd (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            stddev = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return stddev;
    }

    double SPDMetricCalcVarianceHeight::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        double variance = 0;
        if(ptVals->size() > 0)
		{
            variance = gsl_stats_variance (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            variance = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return variance;
    }

    double SPDMetricCalcAbsDeviationHeight::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        double absdev = 0;
        if(ptVals->size() > 0)
		{
            absdev = gsl_stats_absdev (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            absdev = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return absdev;
    }

    double SPDMetricCalcCoefficientOfVariationHeight::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        double cv = 0;
        if(ptVals->size() > 0)
		{
            double sumSq = 0;
            double mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
            for(std::vector<double>::iterator iterVals; iterVals != ptVals->end(); ++iterVals)
            {
                sumSq += pow(((*iterVals) - mean),2);
            }
            cv = sqrt(sumSq/ptVals->size())/mean;
        }
        else
        {
            cv = std::numeric_limits<float>::signaling_NaN();
        }

        delete ptVals;
        return cv;
    }

    double SPDMetricCalcPercentileHeight::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        double percentileVal = 0;
        if(ptVals->size() > 0)
		{
            double quatFrac = ((double)percentile)/100;
            std::sort(ptVals->begin(), ptVals->end());

            percentileVal = gsl_stats_quantile_from_sorted_data(&(*ptVals)[0], 1, ptVals->size(), quatFrac);
        }
        else
        {
            percentileVal = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return percentileVal;
    }

    double SPDMetricCalcSkewnessHeight::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        double skew = 0;
        if(ptVals->size() > 0)
		{
            skew = gsl_stats_skew (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            skew = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return skew;
    }

    double SPDMetricCalcPersonModeSkewnessHeight::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        double personModeSkew = 0;
        if(ptVals->size() > 0)
		{
            std::sort(ptVals->begin(), ptVals->end());
            double mode = this->calcBinnedMode(ptVals, resolution);
            double mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
            double stddev = gsl_stats_sd (&(*ptVals)[0], 1, ptVals->size());

            personModeSkew = (mean - mode)/stddev;
        }
        else
        {
            personModeSkew = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return personModeSkew;
    }

    double SPDMetricCalcPersonMedianSkewnessHeight::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        double personMedianSkew = 0;
        if(ptVals->size() > 0)
		{
            std::sort(ptVals->begin(), ptVals->end());
            double median = gsl_stats_median_from_sorted_data(&(*ptVals)[0], 1, ptVals->size());
            double mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
            double stddev = gsl_stats_sd (&(*ptVals)[0], 1, ptVals->size());

            personMedianSkew = (mean - median)/stddev;
        }
        else
        {
            personMedianSkew = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return personMedianSkew;
    }

    double SPDMetricCalcKurtosisHeight::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        double kurtosis = 0;
        if(ptVals->size() > 0)
		{
            kurtosis = gsl_stats_kurtosis (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            kurtosis = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return kurtosis;
    }

    double SPDMetricCalcNumReturnsAboveMetricHeight::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        boost::uint_fast64_t valCount = 0;
        if(ptVals->size() > 0)
        {
            double thresValue = this->metric->calcValue(pulses, spdFile, geom);

            for(std::vector<double>::iterator iterVals = ptVals->begin(); iterVals != ptVals->end(); ++iterVals)
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

    double SPDMetricCalcNumReturnsBelowMetricHeight::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
        boost::uint_fast64_t valCount = 0;
        if(ptVals->size() > 0)
        {
            double thresValue = this->metric->calcValue(pulses, spdFile, geom);
            for(std::vector<double>::iterator iterVals = ptVals->begin(); iterVals != ptVals->end(); ++iterVals)
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

    double SPDMetricCalcWeibullAlphaHeight::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        double weibullAlpha = 0;
        try
        {
            if(pulses->size() > 0)
            {
                std::vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
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
                    std::cout << "Bin Heights:\t";
                    for(boost::uint_fast32_t i = 0; i < numBins; ++i)
                    {
                        if(i == 0)
                        {
                            std::cout << (minH + (resolution * i));
                        }
                        else
                        {
                            std::cout << "," << (minH + (resolution * i));
                        }
                    }
                    std::cout << std::endl;
                    std::cout << "Bins:\t";
                    for(boost::uint_fast32_t i = 0; i < numBins; ++i)
                    {
                        binSums += bins[i];
                        fitData->heights[i] = minH + (resolution/2) + (resolution * ((double)i));
                        fitData->error[i] = 1;
                        if(i == 0)
                        {
                            std::cout << bins[i];
                        }
                        else
                        {
                            std::cout << "," << bins[i];
                        }
                    }
                    std::cout << std::endl;
                    // Make area == 1
                    std::cout << "Bins (Area == 1):\t";
                    for(boost::uint_fast32_t i = 0; i < numBins; ++i)
                    {
                        fitData->binVals[i] = bins[i]/binSums;
                        if(i == 0)
                        {
                            std::cout << fitData->binVals[i];
                        }
                        else
                        {
                            std::cout << "," << fitData->binVals[i];
                        }
                    }
                    std::cout << std::endl;

                    // parameters[0] - Alpha
                    // parameters[1] - Beta
                    double *parameters = new double[2];
                    mp_par *paramConstraints = new mp_par[2];

                    parameters[0] = fitData->binVals[maxIdx]; // Alpha
                    std::cout << "init alpha = " << parameters[0] << std::endl;
                    paramConstraints[0].fixed = false;
					paramConstraints[0].limited[0] = true;
					paramConstraints[0].limited[1] = false;
					paramConstraints[0].limits[0] = 0;
					paramConstraints[0].limits[1] = 0;
                    //std::cout << "Alpha constraint = [" << paramConstraints[0].limits[0] << ", " << paramConstraints[0].limits[1] << "]\n";
					paramConstraints[0].parname = const_cast<char*>(std::string("Alpha").c_str());;
					paramConstraints[0].step = 0;
					paramConstraints[0].relstep = 0;
					paramConstraints[0].side = 0;
					paramConstraints[0].deriv_debug = 0;

                    //double percent20height = (maxH - minH) * 0.2;
					parameters[1] = minH + (((double)maxIdx) * resolution); // Beta
                    std::cout << "init beta = " << parameters[1] << std::endl;
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
                                                      //std::cout << "Beta constraint = [" << paramConstraints[1].limits[0] << ", " << paramConstraints[1].limits[1] << "]\n";
					paramConstraints[1].parname = const_cast<char*>(std::string("Beta").c_str());;
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
						std::cout << "Return code is :" << returnCode << " - this can not been defined!\n";
					}

                    std::cout << "final alpha = " << parameters[0] << std::endl;
                    std::cout << "final beta = " << parameters[1] << std::endl << std::endl;

                    weibullAlpha = parameters[0];

                    delete[] parameters;
                    delete[] paramConstraints;
                    delete[] bins;
                }
                else
                {
                    weibullAlpha = std::numeric_limits<float>::signaling_NaN();
                }

            }
            else
            {
                weibullAlpha = std::numeric_limits<float>::signaling_NaN();
            }

        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }

        return weibullAlpha;
    }

    double SPDMetricCalcWeibullBetaHeight::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        double weibullBeta = 0;
        try
        {
            if(pulses->size() > 0)
            {
                std::vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
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
					paramConstraints[0].parname = const_cast<char*>(std::string("Alpha").c_str());;
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
					paramConstraints[1].parname = const_cast<char*>(std::string("Beta").c_str());;
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
						std::cout << "Return code is :" << returnCode << " - this can not been defined!\n";
					}

                    weibullBeta = parameters[1];

                    delete[] parameters;
                    delete[] paramConstraints;
                    delete[] bins;
                }
                else
                {
                    weibullBeta = std::numeric_limits<float>::signaling_NaN();
                }

            }
            else
            {
                weibullBeta = std::numeric_limits<float>::signaling_NaN();
            }

        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }

        return weibullBeta;
    }

    double SPDMetricCalcWeibullQuantileRangeHeight::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        double weibullQuantileRange = 0;
        try
        {
            if(pulses->size() > 0)
            {
                std::vector<double> *ptVals = this->getPointsValuesWithinHeightParameters(pulses, spdFile, geom);
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
					paramConstraints[0].parname = const_cast<char*>(std::string("Alpha").c_str());;
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
					paramConstraints[1].parname = const_cast<char*>(std::string("Beta").c_str());;
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
						std::cout << "Return code is :" << returnCode << " - this can not been defined!\n";
					}

                    weibullQuantileRange = 0; // Need to calculate... TODO

                    delete[] parameters;
                    delete[] paramConstraints;
                    delete[] bins;
                }
                else
                {
                    weibullQuantileRange = std::numeric_limits<float>::signaling_NaN();
                }

            }
            else
            {
                weibullQuantileRange = std::numeric_limits<float>::signaling_NaN();
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


    double SPDMetricCalcNumReturnsZ::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        boost::uint_fast64_t numReturns = ptVals->size();
        delete ptVals;
        return numReturns;
    }

    double SPDMetricCalcSumZ::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        double sum = 0;
        for(std::vector<double>::iterator iterVals = ptVals->begin(); iterVals != ptVals->end(); ++iterVals)
        {
            sum += (*iterVals);
        }
        delete ptVals;
        return sum;
    }

    double SPDMetricCalcMeanZ::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        double mean = 0;
        if(ptVals->size() > 0)
		{
            mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            mean = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return mean;
    }

    double SPDMetricCalcMedianZ::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        double median = 0;
        if(ptVals->size() > 0)
		{
            std::sort(ptVals->begin(), ptVals->end());
            /*for(unsigned int i = 0; i < ptVals->size(); ++i)
             {
             if( i == 0 )
             {
             std::cout << &(*ptVals)[0][i];
             }
             else
             {
             std::cout << ", " << &(*ptVals)[0][i];
             }
             }
             std::cout << std::endl << std::endl;*/
            median = gsl_stats_median_from_sorted_data(&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            median = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return median;
    }

    double SPDMetricCalcModeZ::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        double mode = 0;
        if(ptVals->size() > 0)
        {
            mode = this->calcBinnedMode(ptVals, resolution);
        }
        else
        {
            mode = std::numeric_limits<float>::signaling_NaN();
        }

        delete ptVals;
        return mode;
    }

    double SPDMetricCalcMinZ::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        double min = 0;
        if(ptVals->size() > 0)
		{
            min = gsl_stats_min (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            min = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return min;
    }

    double SPDMetricCalcMaxZ::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        double max = 0;
        if(ptVals->size() > 0)
		{
            max = gsl_stats_max (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            max = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return max;
    }

    double SPDMetricCalcStdDevZ::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        double stddev = 0;
        if(ptVals->size() > 0)
		{
            stddev = gsl_stats_sd (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            stddev = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return stddev;
    }

    double SPDMetricCalcVarianceZ::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        double variance = 0;
        if(ptVals->size() > 0)
		{
            variance = gsl_stats_variance (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            variance = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return variance;
    }

    double SPDMetricCalcAbsDeviationZ::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        double absdev = 0;
        if(ptVals->size() > 0)
		{
            absdev = gsl_stats_absdev (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            absdev = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return absdev;
    }

    double SPDMetricCalcCoefficientOfVariationZ::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        double cv = 0;
        if(ptVals->size() > 0)
		{
            double sumSq = 0;
            double mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
            for(std::vector<double>::iterator iterVals; iterVals != ptVals->end(); ++iterVals)
            {
                sumSq += pow(((*iterVals) - mean),2);
            }
            cv = sqrt(sumSq/ptVals->size())/mean;
        }
        else
        {
            cv = std::numeric_limits<float>::signaling_NaN();
        }

        delete ptVals;
        return cv;
    }

    double SPDMetricCalcPercentileZ::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        double percentileVal = 0;
        if(ptVals->size() > 0)
		{
            double quatFrac = ((double)percentile)/100;
            std::sort(ptVals->begin(), ptVals->end());

            percentileVal = gsl_stats_quantile_from_sorted_data(&(*ptVals)[0], 1, ptVals->size(), quatFrac);
        }
        else
        {
            percentileVal = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return percentileVal;
    }

    double SPDMetricCalcSkewnessZ::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        double skew = 0;
        if(ptVals->size() > 0)
		{
            skew = gsl_stats_skew (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            skew = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return skew;
    }

    double SPDMetricCalcPersonModeSkewnessZ::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        double personModeSkew = 0;
        if(ptVals->size() > 0)
		{
            std::sort(ptVals->begin(), ptVals->end());
            double mode = this->calcBinnedMode(ptVals, resolution);
            double mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
            double stddev = gsl_stats_sd (&(*ptVals)[0], 1, ptVals->size());

            personModeSkew = (mean - mode)/stddev;
        }
        else
        {
            personModeSkew = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return personModeSkew;
    }

    double SPDMetricCalcPersonMedianSkewnessZ::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        double personMedianSkew = 0;
        if(ptVals->size() > 0)
		{
            std::sort(ptVals->begin(), ptVals->end());
            double median = gsl_stats_median_from_sorted_data(&(*ptVals)[0], 1, ptVals->size());
            double mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
            double stddev = gsl_stats_sd (&(*ptVals)[0], 1, ptVals->size());

            personMedianSkew = (mean - median)/stddev;
        }
        else
        {
            personMedianSkew = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return personMedianSkew;
    }

    double SPDMetricCalcKurtosisZ::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        double kurtosis = 0;
        if(ptVals->size() > 0)
		{
            kurtosis = gsl_stats_kurtosis (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            kurtosis = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return kurtosis;
    }

    double SPDMetricCalcNumReturnsAboveMetricZ::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        boost::uint_fast64_t valCount = 0;
        if(ptVals->size() > 0)
		{
            double thresValue = this->metric->calcValue(pulses, spdFile, geom);
            for(std::vector<double>::iterator iterVals = ptVals->begin(); iterVals != ptVals->end(); ++iterVals)
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

    double SPDMetricCalcNumReturnsBelowMetricZ::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
        boost::uint_fast64_t valCount = 0;
        if(ptVals->size() > 0)
		{
            double thresValue = this->metric->calcValue(pulses, spdFile, geom);
            for(std::vector<double>::iterator iterVals = ptVals->begin(); iterVals != ptVals->end(); ++iterVals)
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

    double SPDMetricCalcWeibullAlphaZ::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        double weibullAlpha = 0;
        try
        {
            if(pulses->size() > 0)
            {
                std::vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
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
					paramConstraints[0].parname = const_cast<char*>(std::string("Alpha").c_str());;
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
					paramConstraints[1].parname = const_cast<char*>(std::string("Beta").c_str());;
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
						std::cout << "Return code is :" << returnCode << " - this can not been defined!\n";
					}

                    weibullAlpha = parameters[0];

                    delete[] parameters;
                    delete[] paramConstraints;
                    delete[] bins;
                }
                else
                {
                    weibullAlpha = std::numeric_limits<float>::signaling_NaN();
                }

            }
            else
            {
                weibullAlpha = std::numeric_limits<float>::signaling_NaN();
            }

        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }

        return weibullAlpha;
    }

    double SPDMetricCalcWeibullBetaZ::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        double weibullBeta = 0;
        try
        {
            if(pulses->size() > 0)
            {
                std::vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
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
					paramConstraints[0].parname = const_cast<char*>(std::string("Alpha").c_str());;
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
					paramConstraints[1].parname = const_cast<char*>(std::string("Beta").c_str());;
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
						std::cout << "Return code is :" << returnCode << " - this can not been defined!\n";
					}

                    weibullBeta = parameters[1];

                    delete[] parameters;
                    delete[] paramConstraints;
                    delete[] bins;
                }
                else
                {
                    weibullBeta = std::numeric_limits<float>::signaling_NaN();
                }

            }
            else
            {
                weibullBeta = std::numeric_limits<float>::signaling_NaN();
            }

        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }

        return weibullBeta;
    }

    double SPDMetricCalcWeibullQuantileRangeZ::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        double weibullQuantileRange = 0;
        try
        {
            if(pulses->size() > 0)
            {
                std::vector<double> *ptVals = this->getPointsValuesWithinZParameters(pulses, spdFile, geom);
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
					paramConstraints[0].parname = const_cast<char*>(std::string("Alpha").c_str());;
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
					paramConstraints[1].parname = const_cast<char*>(std::string("Beta").c_str());;
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
						std::cout << "Return code is :" << returnCode << " - this can not been defined!\n";
					}

                    weibullQuantileRange = 0; // Need to calculate... TODO

                    delete[] parameters;
                    delete[] paramConstraints;
                    delete[] bins;
                }
                else
                {
                    weibullQuantileRange = std::numeric_limits<float>::signaling_NaN();
                }

            }
            else
            {
                weibullQuantileRange = std::numeric_limits<float>::signaling_NaN();
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

    double SPDMetricCalcNumReturnsAmplitude::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);
        boost::uint_fast64_t numReturns = ptVals->size();
        delete ptVals;
        return numReturns;
    }

    double SPDMetricCalcSumAmplitude::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals;
        ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);

        double sum = 0;
        for(std::vector<double>::iterator iterVals = ptVals->begin(); iterVals != ptVals->end(); ++iterVals)
        {
            sum += (*iterVals);
        }
        delete ptVals;
        return sum;
    }

    double SPDMetricCalcMeanAmplitude::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        double mean = 0;

        std::vector<double> *ptVals;
        ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);

        if(ptVals->size() > 0)
		{
            mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            mean = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;

        return mean;
    }

    double SPDMetricCalcMedianAmplitude::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals;
        ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);

        double median = 0;
        if(ptVals->size() > 0)
		{
            std::sort(ptVals->begin(), ptVals->end());
            median = gsl_stats_median_from_sorted_data(&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            median = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return median;
    }

    double SPDMetricCalcModeAmplitude::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals;
        ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);

        double mode = 0;
        if(ptVals->size() > 0)
        {
            mode = this->calcBinnedMode(ptVals, resolution);
        }
        else
        {
            mode = std::numeric_limits<float>::signaling_NaN();
        }

        delete ptVals;
        return mode;
    }

    double SPDMetricCalcMinAmplitude::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals;
        ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);

        double min = 0;
        if(ptVals->size() > 0)
		{
            min = gsl_stats_min (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            min = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return min;
    }

    double SPDMetricCalcMaxAmplitude::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals;
        ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);

        double max = 0;
        if(ptVals->size() > 0)
		{
            max = gsl_stats_max (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            max = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return max;
    }

    double SPDMetricCalcStdDevAmplitude::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals;
        ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);

        double stddev = 0;
        if(ptVals->size() > 0)
		{
            stddev = gsl_stats_sd (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            stddev = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return stddev;
    }

    double SPDMetricCalcVarianceAmplitude::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals;
        ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);

        double variance = 0;
        if(ptVals->size() > 0)
		{
            variance = gsl_stats_variance (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            variance = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return variance;
    }

    double SPDMetricCalcAbsDeviationAmplitude::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals;
        ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);

        double absdev = 0;
        if(ptVals->size() > 0)
		{
            absdev = gsl_stats_absdev (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            absdev = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return absdev;
    }

    double SPDMetricCalcCoefficientOfVariationAmplitude::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals;
        ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);

        double cv = 0;
        if(ptVals->size() > 0)
		{
            double sumSq = 0;
            double mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
            for(std::vector<double>::iterator iterVals=ptVals->begin(); iterVals != ptVals->end(); ++iterVals)
            {
                sumSq += pow(((*iterVals) - mean),2);
            }
            cv = sqrt(sumSq/ptVals->size())/mean;
        }
        else
        {
            cv = std::numeric_limits<float>::signaling_NaN();
        }

        delete ptVals;
        return cv;
    }

    double SPDMetricCalcPercentileAmplitude::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals;
        ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);

        double quantile = 0;
        if(ptVals->size() > 0)
		{
            double quatFrac = ((float)percentile)/100;
            std::sort(ptVals->begin(), ptVals->end());
            quantile = gsl_stats_quantile_from_sorted_data(&(*ptVals)[0], 1, ptVals->size(), quatFrac);
        }
        else
        {
            quantile = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return quantile;
    }

    double SPDMetricCalcSkewnessAmplitude::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals;
        ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);

        double skew = 0;
        if(ptVals->size() > 0)
		{
            skew = gsl_stats_skew (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            skew = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return skew;
    }

    double SPDMetricCalcPersonModeSkewnessAmplitude::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals;
        ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);

        double personModeSkew = 0;
        if(ptVals->size() > 0)
		{
            std::sort(ptVals->begin(), ptVals->end());
            double mode = this->calcBinnedMode(ptVals, resolution);
            double mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
            double stddev = gsl_stats_sd (&(*ptVals)[0], 1, ptVals->size());

            personModeSkew = (mean - mode)/stddev;
        }
        else
        {
            personModeSkew = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return personModeSkew;
    }

    double SPDMetricCalcPersonMedianSkewnessAmplitude::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals;
        ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);

        double personMedianSkew = 0;
        if(ptVals->size() > 0)
		{
            std::sort(ptVals->begin(), ptVals->end());
            double median = gsl_stats_median_from_sorted_data(&(*ptVals)[0], 1, ptVals->size());
            double mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
            double stddev = gsl_stats_sd (&(*ptVals)[0], 1, ptVals->size());

            personMedianSkew = (mean - median)/stddev;
        }
        else
        {
            personMedianSkew = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return personMedianSkew;
    }

    double SPDMetricCalcKurtosisAmplitude::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals;
        ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);

        double kurtosis = 0;
        if(ptVals->size() > 0)
		{
            kurtosis = gsl_stats_kurtosis (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            kurtosis = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return kurtosis;
    }

    double SPDMetricCalcNumReturnsAboveMetricAmplitude::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);
        boost::uint_fast64_t valCount = 0;
        if(ptVals->size() > 0)
		{
            double thresValue = this->metric->calcValue(pulses, spdFile, geom);
            for(std::vector<double>::iterator iterVals = ptVals->begin(); iterVals != ptVals->end(); ++iterVals)
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

    double SPDMetricCalcNumReturnsBelowMetricAmplitude::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinAmplitudeParameters(pulses, spdFile, geom);
        boost::uint_fast64_t valCount = 0;
        if(ptVals->size() > 0)
		{
            double thresValue = this->metric->calcValue(pulses, spdFile, geom);
            for(std::vector<double>::iterator iterVals = ptVals->begin(); iterVals != ptVals->end(); ++iterVals)
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

    double SPDMetricCalcNumReturnsRange::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        boost::uint_fast64_t numReturns = ptVals->size();
        delete ptVals;
        return numReturns;
    }

    double SPDMetricCalcSumRange::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        double sum = 0;
        for(std::vector<double>::iterator iterVals = ptVals->begin(); iterVals != ptVals->end(); ++iterVals)
        {
            sum += (*iterVals);
        }
        delete ptVals;
        return sum;
    }

    double SPDMetricCalcMeanRange::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        double mean = 0;
        if(ptVals->size() > 0)
		{
            mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            mean = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return mean;
    }

    double SPDMetricCalcMedianRange::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        double median = 0;
        if(ptVals->size() > 0)
		{
            std::sort(ptVals->begin(), ptVals->end());
            median = gsl_stats_median_from_sorted_data(&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            median = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return median;
    }

    double SPDMetricCalcModeRange::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        double mode = 0;
        if(ptVals->size() > 0)
        {
            mode = this->calcBinnedMode(ptVals, resolution);
        }
        else
        {
            mode = std::numeric_limits<float>::signaling_NaN();
        }

        delete ptVals;
        return mode;
    }

    double SPDMetricCalcMinRange::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        double min = 0;
        if(ptVals->size() > 0)
		{
            min = gsl_stats_min (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            min = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return min;
    }

    double SPDMetricCalcMaxRange::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        double max = 0;
        if(ptVals->size() > 0)
		{
            max = gsl_stats_max (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            max = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return max;
    }

    double SPDMetricCalcStdDevRange::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        double stddev = 0;
        if(ptVals->size() > 0)
		{
            stddev = gsl_stats_sd (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            stddev = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return stddev;
    }

    double SPDMetricCalcVarianceRange::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        double variance = 0;
        if(ptVals->size() > 0)
		{
            variance = gsl_stats_variance (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            variance = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return variance;
    }

    double SPDMetricCalcAbsDeviationRange::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        double absdev = 0;
        if(ptVals->size() > 0)
		{
            absdev = gsl_stats_absdev (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            absdev = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return absdev;
    }

    double SPDMetricCalcCoefficientOfVariationRange::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        double cv = 0;
        if(ptVals->size() > 0)
		{
            double sumSq = 0;
            double mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
            for(std::vector<double>::iterator iterVals; iterVals != ptVals->end(); ++iterVals)
            {
                sumSq += pow(((*iterVals) - mean),2);
            }
            cv = sqrt(sumSq/ptVals->size())/mean;
        }
        else
        {
            cv = std::numeric_limits<float>::signaling_NaN();
        }

        delete ptVals;
        return cv;
    }

    double SPDMetricCalcPercentileRange::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        double quantile = 0;
        if(ptVals->size() > 0)
		{
            double quatFrac = ((float)percentile)/100;
            std::sort(ptVals->begin(), ptVals->end());
            quantile = gsl_stats_quantile_from_sorted_data(&(*ptVals)[0], 1, ptVals->size(), quatFrac);
        }
        else
        {
            quantile = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return quantile;
    }

    double SPDMetricCalcSkewnessRange::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        double skew = 0;
        if(ptVals->size() > 0)
		{
            skew = gsl_stats_skew (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            skew = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return skew;
    }

    double SPDMetricCalcPersonModeSkewnessRange::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        double personModeSkew = 0;
        if(ptVals->size() > 0)
		{
            std::sort(ptVals->begin(), ptVals->end());
            double mode = this->calcBinnedMode(ptVals, resolution);
            double mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
            double stddev = gsl_stats_sd (&(*ptVals)[0], 1, ptVals->size());

            personModeSkew = (mean - mode)/stddev;
        }
        else
        {
            personModeSkew = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return personModeSkew;
    }

    double SPDMetricCalcPersonMedianSkewnessRange::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        double personMedianSkew = 0;
        if(ptVals->size() > 0)
		{
            std::sort(ptVals->begin(), ptVals->end());
            double median = gsl_stats_median_from_sorted_data(&(*ptVals)[0], 1, ptVals->size());
            double mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
            double stddev = gsl_stats_sd (&(*ptVals)[0], 1, ptVals->size());

            personMedianSkew = (mean - median)/stddev;
        }
        else
        {
            personMedianSkew = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return personMedianSkew;
    }

    double SPDMetricCalcKurtosisRange::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        double kurtosis = 0;
        if(ptVals->size() > 0)
		{
            kurtosis = gsl_stats_kurtosis (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            kurtosis = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return kurtosis;
    }

    double SPDMetricCalcNumReturnsAboveMetricRange::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        boost::uint_fast64_t valCount = 0;
        if(ptVals->size() > 0)
		{
            double thresValue = this->metric->calcValue(pulses, spdFile, geom);
            for(std::vector<double>::iterator iterVals = ptVals->begin(); iterVals != ptVals->end(); ++iterVals)
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

    double SPDMetricCalcNumReturnsBelowMetricRange::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
        boost::uint_fast64_t valCount = 0;
        if(ptVals->size() > 0)
		{
            double thresValue = this->metric->calcValue(pulses, spdFile, geom);
            for(std::vector<double>::iterator iterVals = ptVals->begin(); iterVals != ptVals->end(); ++iterVals)
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

    double SPDMetricCalcWeibullAlphaRange::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        double weibullAlpha = 0;
        try
        {
            if(pulses->size() > 0)
            {
                std::vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
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
					paramConstraints[0].parname = const_cast<char*>(std::string("Alpha").c_str());;
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
					paramConstraints[1].parname = const_cast<char*>(std::string("Beta").c_str());;
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
						std::cout << "Return code is :" << returnCode << " - this can not been defined!\n";
					}

                    weibullAlpha = parameters[0];

                    delete[] parameters;
                    delete[] paramConstraints;
                    delete[] bins;
                }
                else
                {
                    weibullAlpha = std::numeric_limits<float>::signaling_NaN();
                }

            }
            else
            {
                weibullAlpha = std::numeric_limits<float>::signaling_NaN();
            }

        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }

        return weibullAlpha;
    }

    double SPDMetricCalcWeibullBetaRange::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        double weibullBeta = 0;
        try
        {
            if(pulses->size() > 0)
            {
                std::vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
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
					paramConstraints[0].parname = const_cast<char*>(std::string("Alpha").c_str());;
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
					paramConstraints[1].parname = const_cast<char*>(std::string("Beta").c_str());;
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
						std::cout << "Return code is :" << returnCode << " - this can not been defined!\n";
					}

                    weibullBeta = parameters[1];

                    delete[] parameters;
                    delete[] paramConstraints;
                    delete[] bins;
                }
                else
                {
                    weibullBeta = std::numeric_limits<float>::signaling_NaN();
                }

            }
            else
            {
                weibullBeta = std::numeric_limits<float>::signaling_NaN();
            }

        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }

        return weibullBeta;
    }

    double SPDMetricCalcWeibullQuantileRangeRange::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        double weibullQuantileRange = 0;
        try
        {
            if(pulses->size() > 0)
            {
                std::vector<double> *ptVals = this->getPointsValuesWithinRangeParameters(pulses, spdFile, geom);
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
					paramConstraints[0].parname = const_cast<char*>(std::string("Alpha").c_str());;
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
					paramConstraints[1].parname = const_cast<char*>(std::string("Beta").c_str());;
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
						std::cout << "Return code is :" << returnCode << " - this can not been defined!\n";
					}

                    weibullQuantileRange = 0; // Need to calculate... TODO

                    delete[] parameters;
                    delete[] paramConstraints;
                    delete[] bins;
                }
                else
                {
                    weibullQuantileRange = std::numeric_limits<float>::signaling_NaN();
                }

            }
            else
            {
                weibullQuantileRange = std::numeric_limits<float>::signaling_NaN();
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

    double SPDMetricCalcNumReturnsWidth::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        boost::uint_fast64_t numReturns = ptVals->size();
        delete ptVals;
        return numReturns;
    }

    double SPDMetricCalcSumWidth::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        double sum = 0;
        for(std::vector<double>::iterator iterVals = ptVals->begin(); iterVals != ptVals->end(); ++iterVals)
        {
            sum += (*iterVals);
        }
        delete ptVals;
        return sum;
    }

    double SPDMetricCalcMeanWidth::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        double mean = 0;
        if(ptVals->size() > 0)
		{
            mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            mean = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return mean;
    }

    double SPDMetricCalcMedianWidth::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        double median = 0;
        if(ptVals->size() > 0)
		{
            std::sort(ptVals->begin(), ptVals->end());
            median = gsl_stats_median_from_sorted_data(&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            median = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return median;
    }

    double SPDMetricCalcModeWidth::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        double mode = 0;
        if(ptVals->size() > 0)
        {
            mode = this->calcBinnedMode(ptVals, resolution);
        }
        else
        {
            mode = std::numeric_limits<float>::signaling_NaN();
        }

        delete ptVals;
        return mode;
    }

    double SPDMetricCalcMinWidth::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        double min = 0;
        if(ptVals->size() > 0)
		{
            min = gsl_stats_min (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            min = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return min;
    }

    double SPDMetricCalcMaxWidth::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        double max = 0;
        if(ptVals->size() > 0)
		{
            max = gsl_stats_max (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            max = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return max;
    }

    double SPDMetricCalcStdDevWidth::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        double stddev = 0;
        if(ptVals->size() > 0)
		{
            stddev = gsl_stats_sd (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            stddev = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return stddev;
    }

    double SPDMetricCalcVarianceWidth::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        double variance = 0;
        if(ptVals->size() > 0)
		{
            variance = gsl_stats_variance (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            variance = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return variance;
    }

    double SPDMetricCalcAbsDeviationWidth::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        double absdev = 0;
        if(ptVals->size() > 0)
		{
            absdev = gsl_stats_absdev (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            absdev = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return absdev;
    }

    double SPDMetricCalcCoefficientOfVariationWidth::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        double cv = 0;
        if(ptVals->size() > 0)
		{
            double sumSq = 0;
            double mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
            for(std::vector<double>::iterator iterVals; iterVals != ptVals->end(); ++iterVals)
            {
                sumSq += pow(((*iterVals) - mean),2);
            }
            cv = sqrt(sumSq/ptVals->size())/mean;
        }
        else
        {
            cv = std::numeric_limits<float>::signaling_NaN();
        }

        delete ptVals;
        return cv;
    }

    double SPDMetricCalcPercentileWidth::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        double quantile = 0;
        if(ptVals->size() > 0)
		{
            double quatFrac = ((float)percentile)/100;
            std::sort(ptVals->begin(), ptVals->end());
            quantile = gsl_stats_quantile_from_sorted_data(&(*ptVals)[0], 1, ptVals->size(), quatFrac);
        }
        else
        {
            quantile = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return quantile;
    }

    double SPDMetricCalcSkewnessWidth::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        double skew = 0;
        if(ptVals->size() > 0)
		{
            skew = gsl_stats_skew (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            skew = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return skew;
    }

    double SPDMetricCalcPersonModeSkewnessWidth::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        double personModeSkew = 0;
        if(ptVals->size() > 0)
		{
            std::sort(ptVals->begin(), ptVals->end());
            double mode = this->calcBinnedMode(ptVals, resolution);
            double mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
            double stddev = gsl_stats_sd (&(*ptVals)[0], 1, ptVals->size());

            personModeSkew = (mean - mode)/stddev;
        }
        else
        {
            personModeSkew = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return personModeSkew;
    }

    double SPDMetricCalcPersonMedianSkewnessWidth::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        double personMedianSkew = 0;
        if(ptVals->size() > 0)
		{
            std::sort(ptVals->begin(), ptVals->end());
            double median = gsl_stats_median_from_sorted_data(&(*ptVals)[0], 1, ptVals->size());
            double mean = gsl_stats_mean (&(*ptVals)[0], 1, ptVals->size());
            double stddev = gsl_stats_sd (&(*ptVals)[0], 1, ptVals->size());

            personMedianSkew = (mean - median)/stddev;
        }
        else
        {
            personMedianSkew = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return personMedianSkew;
    }

    double SPDMetricCalcKurtosisWidth::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        double kurtosis = 0;
        if(ptVals->size() > 0)
		{
            kurtosis = gsl_stats_kurtosis (&(*ptVals)[0], 1, ptVals->size());
        }
        else
        {
            kurtosis = std::numeric_limits<float>::signaling_NaN();
        }
        delete ptVals;
        return kurtosis;
    }

    double SPDMetricCalcNumReturnsAboveMetricWidth::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        boost::uint_fast64_t valCount = 0;
        if(ptVals->size() > 0)
		{
            double thresValue = this->metric->calcValue(pulses, spdFile, geom);
            for(std::vector<double>::iterator iterVals = ptVals->begin(); iterVals != ptVals->end(); ++iterVals)
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

    double SPDMetricCalcNumReturnsBelowMetricWidth::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        std::vector<double> *ptVals = this->getPointsValuesWithinWidthParameters(pulses, spdFile, geom);
        boost::uint_fast64_t valCount = 0;
        if(ptVals->size() > 0)
		{
            double thresValue = this->metric->calcValue(pulses, spdFile, geom);
            for(std::vector<double>::iterator iterVals = ptVals->begin(); iterVals != ptVals->end(); ++iterVals)
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

    /**
    Waveform-only metrics
    */
    double SPDMetricCalcHeightOfMedianEnergy::calcValue(std::vector<SPDPulse*> *pulses, SPDFile *spdFile, OGRGeometry *geom) 
    {
        /*
        Calculate Height of Median Energy (HOME)
        */
        std::vector<double> *pulseVals = this->getPulseExpandedHistWithinHeightParameters(pulses, spdFile, geom);
        double median = 0;
        if(pulseVals->size() > 0)
		{
            std::sort(pulseVals->begin(), pulseVals->end());
            median = gsl_stats_median_from_sorted_data(&(*pulseVals)[0], 1, pulseVals->size());
        }
        else
        {
            median = std::numeric_limits<float>::signaling_NaN();
        }
        delete pulseVals;
        return median;
    }

}
