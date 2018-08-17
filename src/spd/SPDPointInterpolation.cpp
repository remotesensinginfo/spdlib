/*
 *  SPDPointInterpolation.cpp
 *  SPDLIB
 *
 *  Created by Pete Bunting on 02/03/2011.
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

#include "spd/SPDPointInterpolation.h"

namespace spdlib
{	
	
	SPDPointInterpolator::SPDPointInterpolator(boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin): initialised(false), elevVal(0), thinGridRes(0.5), thinData(false), selectHighOrLow(0), maxNumPtsPerBin(0), totalNumPoints(0)
	{
        this->elevVal = elevVal;
        this->thinGridRes = thinGridRes;
        this->thinData = thinData;
        this->selectHighOrLow = selectHighOrLow;
        this->maxNumPtsPerBin = maxNumPtsPerBin;
	}

    std::vector<SPDPoint*>* SPDPointInterpolator::findPoints(std::list<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException)
    {
        std::vector<SPDPoint*> *points = new std::vector<SPDPoint*>();
        std::list<SPDPulse*>::iterator iterPulses;
        std::vector<SPDPoint*>::iterator iterPts;

        bool firstPt = true;
        double maxZ = 0;
        SPDPoint *maxPt = NULL;

        if((ptClass == SPD_VEGETATION_TOP) || (ptClass == SPD_ALL_CLASSES_TOP))
        {
            for(boost::uint_fast32_t i = 0; i < numYBins; ++i)
            {
                for(boost::uint_fast32_t j = 0; j < numXBins; ++j)
                {
                    firstPt = true;
                    maxPt = NULL;
                    for(iterPulses = pulses[i][j]->begin(); iterPulses != pulses[i][j]->end(); ++iterPulses)
                    {
                        if((*iterPulses)->numberOfReturns > 0)
                        {
                            for(iterPts = (*iterPulses)->pts->begin(); iterPts != (*iterPulses)->pts->end(); ++iterPts)
                            {
                                if(ptClass == SPD_VEGETATION_TOP)
                                {
                                    if(((*iterPts)->classification == SPD_HIGH_VEGETATION) |
                                       ((*iterPts)->classification == SPD_MEDIUM_VEGETATION) |
                                       ((*iterPts)->classification == SPD_LOW_VEGETATION))
                                    {
                                        if(firstPt)
                                        {
                                            maxPt = (*iterPts);
                                            if(elevVal == SPD_USE_Z)
                                            {
                                                maxZ = (*iterPts)->z;
                                            }
                                            else if(elevVal == SPD_USE_HEIGHT)
                                            {
                                                maxZ = (*iterPts)->height;
                                            }
                                            firstPt = false;
                                        }
                                        else if((elevVal == SPD_USE_Z) && ((*iterPts)->z > maxZ))
                                        {
                                            maxZ = (*iterPts)->z;
                                        }
                                        else if((elevVal == SPD_USE_HEIGHT) && ((*iterPts)->height > maxZ))
                                        {
                                            maxZ = (*iterPts)->height;
                                        }
                                    }
                                }
                                else if(ptClass == SPD_ALL_CLASSES_TOP)
                                {
                                    if(firstPt)
                                    {
                                        maxPt = (*iterPts);
                                        if(elevVal == SPD_USE_Z)
                                        {
                                            maxZ = (*iterPts)->z;
                                        }
                                        else if(elevVal == SPD_USE_HEIGHT)
                                        {
                                            maxZ = (*iterPts)->height;
                                        }
                                        firstPt = false;
                                    }
                                    else if((elevVal == SPD_USE_Z) && ((*iterPts)->z > maxZ))
                                    {
                                        maxPt = (*iterPts);
                                        maxZ = (*iterPts)->z;
                                    }
                                    else if((elevVal == SPD_USE_HEIGHT) && ((*iterPts)->height > maxZ))
                                    {
                                        maxPt = (*iterPts);
                                        maxZ = (*iterPts)->height;
                                    }
                                }
                            }
                        }
                    }
                    if(!firstPt)
                    {
                        points->push_back(maxPt);
                    }
                }
            }
        }
        else
        {
            for(boost::uint_fast32_t i = 0; i < numYBins; ++i)
            {
                for(boost::uint_fast32_t j = 0; j < numXBins; ++j)
                {
                    for(iterPulses = pulses[i][j]->begin(); iterPulses != pulses[i][j]->end(); ++iterPulses)
                    {
                        if((*iterPulses)->numberOfReturns > 0)
                        {
                            for(iterPts = (*iterPulses)->pts->begin(); iterPts != (*iterPulses)->pts->end(); ++iterPts)
                            {
                                if(ptClass == SPD_ALL_CLASSES)
                                {
                                    points->push_back(*iterPts);
                                }
                                else if((*iterPts)->classification == ptClass)
                                {
                                    points->push_back(*iterPts);
                                }
                            }
                        }
                    }
                }
            }
        }


        totalNumPoints = points->size();

        if(totalNumPoints < 1)
        {
            throw SPDProcessingException("Not enough points for interpolation.");
        }

        return points;
    }

    std::vector<SPDPoint*>* SPDPointInterpolator::findPoints(std::vector<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException)
    {
        std::vector<SPDPoint*> *points = new std::vector<SPDPoint*>();
        std::vector<SPDPulse*>::iterator iterPulses;
        std::vector<SPDPoint*>::iterator iterPts;

        if((ptClass == SPD_VEGETATION_TOP) || (ptClass == SPD_ALL_CLASSES_TOP))
        {
            bool firstPt = true;
            double maxZ = 0;
            SPDPoint *maxPt = NULL;

            for(boost::uint_fast32_t i = 0; i < numYBins; ++i)
            {
                for(boost::uint_fast32_t j = 0; j < numXBins; ++j)
                {
                    firstPt = true;
                    maxPt = NULL;
                    for(iterPulses = pulses[i][j]->begin(); iterPulses != pulses[i][j]->end(); ++iterPulses)
                    {
                        if((*iterPulses)->numberOfReturns > 0)
                        {
                            for(iterPts = (*iterPulses)->pts->begin(); iterPts != (*iterPulses)->pts->end(); ++iterPts)
                            {
                                if(ptClass == SPD_VEGETATION_TOP)
                                {
                                    if(((*iterPts)->classification == SPD_HIGH_VEGETATION) |
                                       ((*iterPts)->classification == SPD_MEDIUM_VEGETATION) |
                                       ((*iterPts)->classification == SPD_LOW_VEGETATION))
                                    {
                                        if(firstPt)
                                        {
                                            maxPt = (*iterPts);
                                            if(elevVal == SPD_USE_Z)
                                            {
                                                maxZ = (*iterPts)->z;
                                            }
                                            else if(elevVal == SPD_USE_HEIGHT)
                                            {
                                                maxZ = (*iterPts)->height;
                                            }
                                            firstPt = false;
                                        }
                                        else if((elevVal == SPD_USE_Z) && ((*iterPts)->z > maxZ))
                                        {
                                            maxZ = (*iterPts)->z;
                                        }
                                        else if((elevVal == SPD_USE_HEIGHT) && ((*iterPts)->height > maxZ))
                                        {
                                            maxZ = (*iterPts)->height;
                                        }
                                    }
                                }
                                else if(ptClass == SPD_ALL_CLASSES_TOP)
                                {
                                    if(firstPt)
                                    {
                                        maxPt = (*iterPts);
                                        if(elevVal == SPD_USE_Z)
                                        {
                                            maxZ = (*iterPts)->z;
                                        }
                                        else if(elevVal == SPD_USE_HEIGHT)
                                        {
                                            maxZ = (*iterPts)->height;
                                        }
                                        firstPt = false;
                                    }
                                    else if((elevVal == SPD_USE_Z) && ((*iterPts)->z > maxZ))
                                    {
                                        maxPt = (*iterPts);
                                        maxZ = (*iterPts)->z;
                                    }
                                    else if((elevVal == SPD_USE_HEIGHT) && ((*iterPts)->height > maxZ))
                                    {
                                        maxPt = (*iterPts);
                                        maxZ = (*iterPts)->height;
                                    }
                                }
                            }
                        }
                    }
                    if(!firstPt)
                    {
                        points->push_back(maxPt);
                    }
                }
            }
        }
        else
        {
            for(boost::uint_fast32_t i = 0; i < numYBins; ++i)
            {
                for(boost::uint_fast32_t j = 0; j < numXBins; ++j)
                {
                    for(iterPulses = pulses[i][j]->begin(); iterPulses != pulses[i][j]->end(); ++iterPulses)
                    {
                        if((*iterPulses)->numberOfReturns > 0)
                        {
                            for(iterPts = (*iterPulses)->pts->begin(); iterPts != (*iterPulses)->pts->end(); ++iterPts)
                            {
                                if(ptClass == SPD_ALL_CLASSES)
                                {
                                    points->push_back(*iterPts);
                                }
                                else if((*iterPts)->classification == ptClass)
                                {
                                    points->push_back(*iterPts);
                                }
                            }
                        }
                    }
                }
            }
        }

        totalNumPoints = points->size();

        //std::cout << "totalNumPoints = " << totalNumPoints << std::endl;

        if(totalNumPoints < 1)
        {
            throw SPDProcessingException("Not enough points for interpolation.");
        }

        return points;
    }

    std::vector<SPDPoint*>* SPDPointInterpolator::findPoints(std::list<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException)
    {
        std::vector<SPDPoint*> *points = new std::vector<SPDPoint*>();
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
                        points->push_back(pt);
                    }
                }
                else if(ptClass == SPD_ALL_CLASSES_TOP)
                {
                    points->push_back((*iterPulses)->pts->front());
                }
                else
                {
                    for(iterPts = (*iterPulses)->pts->begin(); iterPts != (*iterPulses)->pts->end(); ++iterPts)
                    {
                        if(ptClass == SPD_ALL_CLASSES)
                        {
                            points->push_back(*iterPts);
                        }
                        else if((*iterPts)->classification == ptClass)
                        {
                            points->push_back(*iterPts);
                        }
                    }
                }
			}
		}
        totalNumPoints = points->size();

        if(totalNumPoints < 1)
        {
            throw SPDProcessingException("Not enough points for interpolation.");
        }

        return points;
    }

    std::vector<SPDPoint*>* SPDPointInterpolator::findPoints(std::vector<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException)
    {
        std::vector<SPDPoint*> *points = new std::vector<SPDPoint*>();
        std::vector<SPDPulse*>::iterator iterPulses;
		std::vector<SPDPoint*>::iterator iterPts;
        SPDPoint *pt = NULL;
		for(iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
		{
            //std::cout << "Pulse " << (*iterPulses)->pulseID << std::endl;
			if((*iterPulses)->numberOfReturns > 0)
			{
				if(ptClass == SPD_VEGETATION_TOP)
                {
                    pt = (*iterPulses)->pts->front();
                    if((pt->classification == SPD_HIGH_VEGETATION) |
                       (pt->classification == SPD_MEDIUM_VEGETATION) |
                       (pt->classification == SPD_LOW_VEGETATION))
                    {
                        points->push_back(pt);
                    }
                }
                else if(ptClass == SPD_ALL_CLASSES_TOP)
                {
                    points->push_back((*iterPulses)->pts->at(0));
                }
                else
                {
                    for(iterPts = (*iterPulses)->pts->begin(); iterPts != (*iterPulses)->pts->end(); ++iterPts)
                    {
                        if(ptClass == SPD_ALL_CLASSES)
                        {
                            points->push_back(*iterPts);
                        }
                        else if((*iterPts)->classification == ptClass)
                        {
                            points->push_back(*iterPts);
                        }
                    }
                }
			}
		}
        totalNumPoints = points->size();

        if(totalNumPoints < 1)
        {
            throw SPDProcessingException("Not enough points for interpolation.");
        }

        return points;
    }

    void SPDPointInterpolator::thinPoints(std::vector<SPDPoint*> *points) throw(SPDProcessingException)
    {
        try
        {
            if(thinData)
            {
                SPDPointGridIndex ptIdx;
                ptIdx.buildIndex(points, thinGridRes);
                ptIdx.thinPtsInBins(elevVal, selectHighOrLow, maxNumPtsPerBin);
                points->clear();
                ptIdx.getAllPointsInGrid(points);
                totalNumPoints = points->size();
            }
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
    }
	
	
	SPDTriangulationPointInterpolator::SPDTriangulationPointInterpolator(boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin): SPDPointInterpolator(elevVal, thinGridRes, thinData, selectHighOrLow, maxNumPtsPerBin)
	{

	}
	
    void SPDTriangulationPointInterpolator::initInterpolator(std::list<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException)
    {
        try
        {
            std::vector<SPDPoint*> *points = this->findPoints(pulses, numXBins, numYBins, ptClass);
            if(points->size() < 3)
            {
                delete points;
                throw SPDProcessingException("Not enough points, need at least 3.");
            }
            else if(points->size() < 100)
            {
                double meanX = 0;
                double meanY = 0;

                double varX = 0;
                double varY = 0;

                for(std::vector<SPDPoint*>::iterator iterPts = points->begin(); iterPts != points->end(); ++iterPts)
                {
                    meanX += (*iterPts)->x;
                    meanY += (*iterPts)->y;
                }

                meanX = meanX / points->size();
                meanY = meanY / points->size();

                //std::cout << "meanX = " << meanX << std::endl;
                //std::cout << "meanY = " << meanY << std::endl;

                for(std::vector<SPDPoint*>::iterator iterPts = points->begin(); iterPts != points->end(); ++iterPts)
                {
                    varX += (*iterPts)->x - meanX;
                    varY += (*iterPts)->y - meanY;
                }

                varX = fabs(varX / points->size());
                varY = fabs(varY / points->size());

                //std::cout << "varX = " << varX << std::endl;
                //std::cout << "varY = " << varX << std::endl;

                if((varX < 4) | (varY < 4))
                {
                    delete points;
                    throw SPDProcessingException("Points are all within a line.");
                }
            }
            if(thinData)
            {
                this->thinPoints(points);
            }

            dt = new DelaunayTriangulation();
            values = new PointValueMap();
            fh = Face_handle();

            std::vector<SPDPoint*>::iterator iterPts;
            for(iterPts = points->begin(); iterPts != points->end(); ++iterPts)
            {
                if((*iterPts) != NULL)
                {
                    K::Point_2 cgalPt((*iterPts)->x,(*iterPts)->y);
                    dt->insert(cgalPt);
                    if(elevVal == SPD_USE_Z)
                    {
                        CGALCoordType value = (*iterPts)->z;
                        values->insert(std::make_pair(cgalPt, value));
                    }
                    else if(elevVal == SPD_USE_HEIGHT)
                    {
                        CGALCoordType value = (*iterPts)->height;
                        values->insert(std::make_pair(cgalPt, value));
                    }
                    else if(elevVal == SPD_USE_AMP)
                    {
                        CGALCoordType value = (*iterPts)->amplitudeReturn;
                        values->insert(std::make_pair(cgalPt, value));
                    }
                    else
                    {
                        throw SPDProcessingException("Elevation type not recognised.");
                    }
                }
            }
            delete points;
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
        initialised = true;
    }

    void SPDTriangulationPointInterpolator::initInterpolator(std::vector<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException)
    {
        try
        {
            std::vector<SPDPoint*> *points = this->findPoints(pulses, numXBins, numYBins, ptClass);
            if(points->size() < 3)
            {
                delete points;
                throw SPDProcessingException("Not enough points, need at least 3.");
            }
            else if(points->size() < 100)
            {
                double meanX = 0;
                double meanY = 0;

                double varX = 0;
                double varY = 0;

                for(std::vector<SPDPoint*>::iterator iterPts = points->begin(); iterPts != points->end(); ++iterPts)
                {
                    meanX += (*iterPts)->x;
                    meanY += (*iterPts)->y;
                }

                meanX = meanX / points->size();
                meanY = meanY / points->size();

                //std::cout << "meanX = " << meanX << std::endl;
                //std::cout << "meanY = " << meanY << std::endl;

                for(std::vector<SPDPoint*>::iterator iterPts = points->begin(); iterPts != points->end(); ++iterPts)
                {
                    varX += (*iterPts)->x - meanX;
                    varY += (*iterPts)->y - meanY;
                }

                varX = fabs(varX / points->size());
                varY = fabs(varY / points->size());

                //std::cout << "varX = " << varX << std::endl;
                //std::cout << "varY = " << varX << std::endl;

                if((varX < 4) | (varY < 4))
                {
                    delete points;
                    throw SPDProcessingException("Points are all within a line.");
                }
            }
            if(thinData)
            {
                this->thinPoints(points);
            }

            dt = new DelaunayTriangulation();
            values = new PointValueMap();
            fh = Face_handle();

            std::vector<SPDPoint*>::iterator iterPts;
            for(iterPts = points->begin(); iterPts != points->end(); ++iterPts)
            {
                if((*iterPts) != NULL)
                {
                    K::Point_2 cgalPt((*iterPts)->x,(*iterPts)->y);
                    dt->insert(cgalPt);
                    if(elevVal == SPD_USE_Z)
                    {
                        CGALCoordType value = (*iterPts)->z;
                        values->insert(std::make_pair(cgalPt, value));
                    }
                    else if(elevVal == SPD_USE_HEIGHT)
                    {
                        CGALCoordType value = (*iterPts)->height;
                        values->insert(std::make_pair(cgalPt, value));
                    }
                    else if(elevVal == SPD_USE_AMP)
                    {
                        CGALCoordType value = (*iterPts)->amplitudeReturn;
                        values->insert(std::make_pair(cgalPt, value));
                    }
                    else
                    {
                        throw SPDProcessingException("Elevation type not recognised.");
                    }
                }
            }
            delete points;
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
        initialised = true;
    }

    void SPDTriangulationPointInterpolator::initInterpolator(std::list<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException)
    {
        try
        {
            std::vector<SPDPoint*> *points = this->findPoints(pulses, ptClass);
            if(points->size() < 3)
            {
                delete points;
                throw SPDProcessingException("Not enough points, need at least 3.");
            }
            else if(points->size() < 100)
            {
                double meanX = 0;
                double meanY = 0;

                double varX = 0;
                double varY = 0;

                for(std::vector<SPDPoint*>::iterator iterPts = points->begin(); iterPts != points->end(); ++iterPts)
                {
                    meanX += (*iterPts)->x;
                    meanY += (*iterPts)->y;
                }

                meanX = meanX / points->size();
                meanY = meanY / points->size();

                //std::cout << "meanX = " << meanX << std::endl;
                //std::cout << "meanY = " << meanY << std::endl;

                for(std::vector<SPDPoint*>::iterator iterPts = points->begin(); iterPts != points->end(); ++iterPts)
                {
                    varX += (*iterPts)->x - meanX;
                    varY += (*iterPts)->y - meanY;
                }

                varX = fabs(varX / points->size());
                varY = fabs(varY / points->size());

                //std::cout << "varX = " << varX << std::endl;
                //std::cout << "varY = " << varX << std::endl;

                if((varX < 4) | (varY < 4))
                {
                    delete points;
                    throw SPDProcessingException("Points are all within a line.");
                }
            }
            if(thinData)
            {
                this->thinPoints(points);
            }

            dt = new DelaunayTriangulation();
            values = new PointValueMap();
            fh = Face_handle();

            std::vector<SPDPoint*>::iterator iterPts;
            for(iterPts = points->begin(); iterPts != points->end(); ++iterPts)
            {
                if((*iterPts) != NULL)
                {
                    K::Point_2 cgalPt((*iterPts)->x,(*iterPts)->y);
                    dt->insert(cgalPt);
                    if(elevVal == SPD_USE_Z)
                    {
                        CGALCoordType value = (*iterPts)->z;
                        values->insert(std::make_pair(cgalPt, value));
                    }
                    else if(elevVal == SPD_USE_HEIGHT)
                    {
                        CGALCoordType value = (*iterPts)->height;
                        values->insert(std::make_pair(cgalPt, value));
                    }
                    else if(elevVal == SPD_USE_AMP)
                    {
                        CGALCoordType value = (*iterPts)->amplitudeReturn;
                        values->insert(std::make_pair(cgalPt, value));
                    }
                    else
                    {
                        throw SPDProcessingException("Elevation type not recognised.");
                    }
                }
            }
            delete points;
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
        initialised = true;
    }

    void SPDTriangulationPointInterpolator::initInterpolator(std::vector<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException)
    {
        try
        {
            std::vector<SPDPoint*> *points = this->findPoints(pulses, ptClass);
            if(points->size() < 3)
            {
                delete points;
                throw SPDProcessingException("Not enough points, need at least 3.");
            }
            else if(points->size() < 100)
            {
                double meanX = 0;
                double meanY = 0;

                double varX = 0;
                double varY = 0;

                for(std::vector<SPDPoint*>::iterator iterPts = points->begin(); iterPts != points->end(); ++iterPts)
                {
                    meanX += (*iterPts)->x;
                    meanY += (*iterPts)->y;
                }

                meanX = meanX / points->size();
                meanY = meanY / points->size();

                //std::cout << "meanX = " << meanX << std::endl;
                //std::cout << "meanY = " << meanY << std::endl;

                for(std::vector<SPDPoint*>::iterator iterPts = points->begin(); iterPts != points->end(); ++iterPts)
                {
                    varX += (*iterPts)->x - meanX;
                    varY += (*iterPts)->y - meanY;
                }

                varX = fabs(varX / points->size());
                varY = fabs(varY / points->size());

                //std::cout << "varX = " << varX << std::endl;
                //std::cout << "varY = " << varX << std::endl;

                if((varX < 4) | (varY < 4))
                {
                    delete points;
                    throw SPDProcessingException("Points are all within a line.");
                }
            }
            if(thinData)
            {
                this->thinPoints(points);
            }

            dt = new DelaunayTriangulation();
            values = new PointValueMap();
            fh = Face_handle();

            std::vector<SPDPoint*>::iterator iterPts;
            for(iterPts = points->begin(); iterPts != points->end(); ++iterPts)
            {
                if((*iterPts) != NULL)
                {
                    K::Point_2 cgalPt((*iterPts)->x,(*iterPts)->y);
                    dt->insert(cgalPt);
                    if(elevVal == SPD_USE_Z)
                    {
                        CGALCoordType value = (*iterPts)->z;
                        values->insert(std::make_pair(cgalPt, value));
                    }
                    else if(elevVal == SPD_USE_HEIGHT)
                    {
                        CGALCoordType value = (*iterPts)->height;
                        values->insert(std::make_pair(cgalPt, value));
                    }
                    else if(elevVal == SPD_USE_AMP)
                    {
                        CGALCoordType value = (*iterPts)->amplitudeReturn;
                        values->insert(std::make_pair(cgalPt, value));
                    }
                    else
                    {
                        throw SPDProcessingException("Elevation type not recognised.");
                    }
                }
            }
            delete points;
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
        initialised = true;
    }
	
	void SPDTriangulationPointInterpolator::resetInterpolator() throw(SPDProcessingException)
	{
        totalNumPoints = 0;
        if(initialised)
        {
            delete dt;
            delete values;
            fh = NULL;
        }
		initialised = false;
	}
	
	SPDTriangulationPointInterpolator::~SPDTriangulationPointInterpolator()
	{
        if(initialised)
        {
            delete dt;
            delete values;
        }
		initialised = false;
	}




	
	

	SPDGridIndexPointInterpolator::SPDGridIndexPointInterpolator(double gridResolution, boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin): SPDPointInterpolator(elevVal, thinGridRes, thinData, selectHighOrLow, maxNumPtsPerBin), idx(NULL), gridResolution(0)
	{
		this->gridResolution = gridResolution;
	}
	
	void SPDGridIndexPointInterpolator::initInterpolator(std::list<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException)
	{
        try
        {
            std::vector<SPDPoint*> *points = this->findPoints(pulses, numXBins, numYBins, ptClass);
            if(thinData)
            {
                this->thinPoints(points);
            }

            idx = new SPDPointGridIndex();
			idx->buildIndex(points, this->gridResolution);
            delete points;
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
        initialised = true;
	}
	
	void SPDGridIndexPointInterpolator::initInterpolator(std::vector<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException)
	{
        try
        {
            std::vector<SPDPoint*> *points = this->findPoints(pulses, numXBins, numYBins, ptClass);
            if(thinData)
            {
                this->thinPoints(points);
            }

            idx = new SPDPointGridIndex();
			idx->buildIndex(points, this->gridResolution);
            delete points;
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
        initialised = true;
	}
	
	void SPDGridIndexPointInterpolator::initInterpolator(std::list<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException)
	{
        try
        {
            std::vector<SPDPoint*> *points = this->findPoints(pulses, ptClass);
            if(thinData)
            {
                this->thinPoints(points);
            }

            idx = new SPDPointGridIndex();
			idx->buildIndex(points, this->gridResolution);
            delete points;
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
        initialised = true;
	}
	
	void SPDGridIndexPointInterpolator::initInterpolator(std::vector<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException)
	{
        try
        {
            std::vector<SPDPoint*> *points = this->findPoints(pulses, ptClass);
            if(thinData)
            {
                this->thinPoints(points);
            }

            idx = new SPDPointGridIndex();
			idx->buildIndex(points, this->gridResolution);
            delete points;
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
        initialised = true;
	}

	void SPDGridIndexPointInterpolator::resetInterpolator() throw(SPDProcessingException)
	{
		if(initialised)
		{
			delete idx;
            totalNumPoints = 0;
            initialised = false;
		}
	}
	
	SPDGridIndexPointInterpolator::~SPDGridIndexPointInterpolator()
	{
		if(initialised)
		{
			delete idx;
            totalNumPoints = 0;
            initialised = false;
		}
	}
	
	
	
	

    SPDRFBPointInterpolator::SPDRFBPointInterpolator(double radius, boost::uint_fast16_t numLayers, boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin):SPDPointInterpolator(elevVal, thinGridRes, thinData, selectHighOrLow, maxNumPtsPerBin)
    {
        this->radius = radius;
        this->numLayers = numLayers;
    }
		
    void SPDRFBPointInterpolator::initInterpolator(std::list<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException)
    {
        try
        {
            std::vector<SPDPoint*> *points = this->findPoints(pulses, numXBins, numYBins, ptClass);
            if(thinData)
            {
                this->thinPoints(points);
            }

            throw SPDProcessingException("RFB Interpolator has not been initialised - list grid.");

            delete points;
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
        initialised = true;
    }

    void SPDRFBPointInterpolator::initInterpolator(std::vector<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException)
    {
        try
        {
            SPDTextFileUtilities txtUtils;
            std::vector<SPDPoint*> *points = this->findPoints(pulses, numXBins, numYBins, ptClass);
            if(thinData)
            {
                std::cout << "thinning that data\n";
                this->thinPoints(points);
            }

            rbfModel = new spd_alglib::rbfmodel();
            spd_alglib::rbfcreate(2, 1, *rbfModel);

            size_t numPts = points->size();
            std::cout << numPts << " are being used for the interpolation." << std::endl;

            double minX = 0;
            double minY = 0;
            this->minZ = 0;
            double maxX = 0;
            double maxY = 0;
            bool first = true;
            for(std::vector<SPDPoint*>::iterator iterPts = points->begin(); iterPts != points->end(); ++iterPts)
            {
                if(first)
                {
                    minX = (*iterPts)->x;
                    maxX = (*iterPts)->x;
                    minY = (*iterPts)->y;
                    maxY = (*iterPts)->y;

                    if(elevVal == SPD_USE_Z)
                    {
                        minZ = (*iterPts)->z;
                    }
                    else if(elevVal == SPD_USE_HEIGHT)
                    {
                        minZ = (*iterPts)->height;
                    }
                    else
                    {
                        throw SPDProcessingException("Elevation type not recognised.");
                    }

                    first = false;
                }
                else
                {
                    if((*iterPts)->x < minX)
                    {
                        minX = (*iterPts)->x;
                    }
                    if((*iterPts)->x > maxX)
                    {
                        maxX = (*iterPts)->x;
                    }
                    if((*iterPts)->y < minY)
                    {
                        minY = (*iterPts)->y;
                    }
                    if((*iterPts)->y > maxY)
                    {
                        maxY = (*iterPts)->y;
                    }

                    if(elevVal == SPD_USE_Z)
                    {
                        if((*iterPts)->z < minZ)
                        {
                            minZ = (*iterPts)->z;
                        }
                    }
                    else if(elevVal == SPD_USE_HEIGHT)
                    {
                        if((*iterPts)->height < minZ)
                        {
                            minZ = (*iterPts)->height;
                        }
                    }
                    else
                    {
                        throw SPDProcessingException("Elevation type not recognised.");
                    }


                }
            }
            this->midX = minX + ((maxX-minX)/2);
            this->midY = minY + ((maxY-minY)/2);

            double xVal = 0;
            double yVal = 0;
            double zVal = 0;
            spd_alglib::ae_int_t row = 0;

            spd_alglib::real_2d_array dataPts;
            dataPts.setlength(numPts, 3);

            for(std::vector<SPDPoint*>::iterator iterPts = points->begin(); iterPts != points->end(); ++iterPts)
            {
                xVal = round(((*iterPts)->x - midX));
                dataPts(row,0) = xVal;
                yVal = round(((*iterPts)->y - midY));
                dataPts(row,1) = yVal;

                if(elevVal == SPD_USE_Z)
                {
                    zVal = (*iterPts)->z - minZ;
                }
                else if(elevVal == SPD_USE_HEIGHT)
                {
                    zVal = (*iterPts)->height - minZ;
                }
                else
                {
                    throw SPDProcessingException("Elevation type not recognised.");
                }
                dataPts(row,2) = zVal;

                ++row;
            }

            //std::cout << "Data:\n" << dataPts.tostring(3) << std::endl;
            spd_alglib::rbfsetpoints(*rbfModel, dataPts);
            spd_alglib::rbfsetalgomultilayer(*rbfModel, radius, numLayers);
            std::cout << "Building Model\n";
            spd_alglib::rbfreport rep;
            spd_alglib::rbfbuildmodel(*rbfModel, rep);

            std::cout << "Initialised...\n";
            delete points;
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
        catch(std::exception &e)
        {
            std::cout << "Error: " << e.what() << std::endl;
            throw SPDProcessingException(e.what());
        }
        initialised = true;
    }

	void SPDRFBPointInterpolator::initInterpolator(std::list<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException)
    {
        try
        {
            std::vector<SPDPoint*> *points = this->findPoints(pulses, ptClass);
            if(thinData)
            {
                this->thinPoints(points);
            }




            throw SPDProcessingException("RFB Interpolator has not been initialised - list.");

            delete points;
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
        initialised = true;
    }

	void SPDRFBPointInterpolator::initInterpolator(std::vector<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException)
    {
        try
        {
            std::vector<SPDPoint*> *points = this->findPoints(pulses, ptClass);
            if(thinData)
            {
                this->thinPoints(points);
            }

            throw SPDProcessingException("RFB Interpolator has not been initialised - vector.");

            delete points;
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
        initialised = true;
    }

	float SPDRFBPointInterpolator::getValue(double eastings, double northings) throw(SPDProcessingException)
    {
        float val = 0.0;
        if(initialised)
		{
            val = spd_alglib::rbfcalc2(*rbfModel, round((eastings - midX)), round((northings - midY))) + minZ;
        }
        else
        {
            throw SPDProcessingException("RFB Interpolator has not been initialised.");
        }
        return val;
    }

	void SPDRFBPointInterpolator::resetInterpolator() throw(SPDProcessingException)
    {
        if(initialised)
		{
            if(rbfModel != NULL)
            {
                delete rbfModel;
            }
            initialised = false;
		}
    }

    SPDRFBPointInterpolator::~SPDRFBPointInterpolator()
    {
        if(initialised)
		{
            if(rbfModel != NULL)
            {
                delete rbfModel;
            }
            initialised = false;
		}
    }




	
	
	
	
	
	
	
	
	SPDNearestNeighbourInterpolator::SPDNearestNeighbourInterpolator(boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin):SPDTriangulationPointInterpolator(elevVal, thinGridRes, thinData, selectHighOrLow, maxNumPtsPerBin)
	{
		
	}
	
	float SPDNearestNeighbourInterpolator::getValue(double eastings, double northings) throw(SPDProcessingException)
	{
		double outElevation = std::numeric_limits<float>::signaling_NaN();
		if(initialised)
		{
            CGALPoint p(eastings, northings);
            Vertex_handle vh = dt->nearest_vertex(p);
            CGALPoint nearestPt = vh->point();
            PointValueMap::iterator iterVal = values->find(nearestPt);
            outElevation = (*iterVal).second;
		}
		else
		{
			throw SPDProcessingException("Interpolated needs to be initialised before values can be retrieved.");
		}
		return outElevation;
	
    }
	
	SPDNearestNeighbourInterpolator::~SPDNearestNeighbourInterpolator()
	{
		
	}
	
	
	SPDTINPlaneFitInterpolator::SPDTINPlaneFitInterpolator(boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin):SPDTriangulationPointInterpolator(elevVal, thinGridRes, thinData, selectHighOrLow, maxNumPtsPerBin)
	{
		mathUtils = new SPDMathsUtils();
        aPt = new SPD3DDataPt();
        bPt = new SPD3DDataPt();
        cPt = new SPD3DDataPt();
	}
	
	float SPDTINPlaneFitInterpolator::getValue(double eastings, double northings) throw(SPDProcessingException)
	{
		double outElevation = 0;
		if(initialised)
		{
            //std::cout << "[" << eastings << "," << northings << "]\n";
            CGALPoint p(eastings, northings);
            Vertex_handle vh = dt->nearest_vertex(p);
            CGALPoint nearestPt = vh->point();
            PointValueMap::iterator iterVal;
            fh = dt->locate(nearestPt, fh);
            Vertex_handle pt1Vh = fh->vertex(0);
            iterVal = values->find(pt1Vh->point());
            aPt->x = (*iterVal).first.x();
            aPt->y = (*iterVal).first.y();
            aPt->z = (*iterVal).second;
            //std::cout << "A [" << aPt->x << ", " << aPt->y << "] = " << aPt->z << std::endl;
            Vertex_handle pt2Vh = fh->vertex(1);
            iterVal = values->find(pt2Vh->point());
            bPt->x = (*iterVal).first.x();
            bPt->y = (*iterVal).first.y();
            bPt->z = (*iterVal).second;
            //std::cout << "B [" << bPt->x << ", " << bPt->y << "] = " << bPt->z << std::endl;
            Vertex_handle pt3Vh = fh->vertex(2);
            iterVal = values->find(pt3Vh->point());
            cPt->x = (*iterVal).first.x();
            cPt->y = (*iterVal).first.y();
            cPt->z = (*iterVal).second;
            //std::cout << "C [" << cPt->x << ", " << cPt->y << "] = " << cPt->z << std::endl;

            try
            {
                /*if((eastings > aPt->x) & (eastings > bPt->x) & (eastings > cPt->x))
                {
                    outElevation = std::numeric_limits<float>::signaling_NaN();
                }
                else if((eastings < aPt->x) & (eastings < bPt->x) & (eastings < cPt->x))
                {
                    outElevation = std::numeric_limits<float>::signaling_NaN();
                }
                else if((northings > aPt->y) & (northings > bPt->y) & (northings > cPt->y))
                {
                    outElevation = std::numeric_limits<float>::signaling_NaN();
                }
                else if((northings < aPt->y) & (northings < bPt->y) & (northings < cPt->y))
                {
                    outElevation = std::numeric_limits<float>::signaling_NaN();
                */
                if(false)
                {

                }
                else
                {
                    //std::cout << "[" << eastings << "," << northings << "]\n";
                    //std::cout << "A [" << aPt->x << ", " << aPt->y << "] = " << aPt->z << std::endl;
                    //std::cout << "B [" << bPt->x << ", " << bPt->y << "] = " << bPt->z << std::endl;
                    //std::cout << "C [" << cPt->x << ", " << cPt->y << "] = " << cPt->z << std::endl;
                    outElevation = mathUtils->calcValueViaPlaneFitting(aPt, bPt, cPt, eastings, northings);
                    //std::cout << "Z = " << outElevation << std::endl << std::endl;
                }
            }
            catch (SPDProcessingException &e)
            {
                outElevation = std::numeric_limits<float>::signaling_NaN();
            }
		}
		else
		{
			throw SPDProcessingException("Interpolated needs to be initialised before values can be retrieved.");
		}
		return outElevation;
	}

	
	SPDTINPlaneFitInterpolator::~SPDTINPlaneFitInterpolator()
	{
		delete mathUtils;
        delete aPt;
        delete bPt;
        delete cPt;
	}
	
	
	
	
	
	
	SPDStdDevFilterInterpolator::SPDStdDevFilterInterpolator(float stdDevThreshold, float lowDist, float highDist, float stdDevDist, double gridResolution, boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin):SPDGridIndexPointInterpolator(gridResolution, elevVal, thinGridRes, thinData, selectHighOrLow, maxNumPtsPerBin), stdDevThreshold(0), lowDist(0), highDist(0), stdDevDist(0)
	{
		this->stdDevThreshold = stdDevThreshold;
		this->lowDist = lowDist;
		this->highDist = highDist;
		this->stdDevDist = stdDevDist;
	}
	
	float SPDStdDevFilterInterpolator::getValue(double eastings, double northings) throw(SPDProcessingException)
	{
		float returnZVal = std::numeric_limits<float>::signaling_NaN();
		try
		{
			std::vector<SPDPoint*> *pts = new std::vector<SPDPoint*>();
			if(idx->getPointsInRadius(pts, eastings, northings, stdDevDist))
			{
                if( selectHighOrLow == SPD_SELECT_LOWEST)
                {
                    double minLow = 0;
                    double minHigh = 0;
                    double sum = 0;
                    bool firstLow = true;
                    bool firstHigh = true;
                    double dist = 0;
                    double elev = 0;

                    SPDPointUtils ptUtils;

                    std::vector<SPDPoint*>::iterator iterPts;
                    for(iterPts = pts->begin(); iterPts != pts->end(); ++iterPts)
                    {
                        if(elevVal == SPD_USE_Z)
                        {
                            elev = (*iterPts)->z;
                        }
                        else if(elevVal == SPD_USE_HEIGHT)
                        {
                            elev = (*iterPts)->height;
                        }
                        else
                        {
                            throw SPDProcessingException("Elevation type not recognised.");
                        }
                        sum += elev;
                        dist = ptUtils.distanceXY(eastings, northings, *iterPts);
                        if(dist < this->highDist)
                        {
                            if(firstHigh)
                            {
                                minHigh = elev;
                                firstHigh = false;
                            }
                            else if(elev < minHigh)
                            {
                                minHigh = elev;
                            }

                            if(dist < this->lowDist)
                            {
                                if(firstLow)
                                {
                                    minLow = elev;
                                    firstLow = false;
                                }
                                else if(elev < minLow)
                                {
                                    minLow = elev;
                                }
                            }
                        }
                    }

                    double mean = sum / pts->size();
                    double sumSq = 0;

                    for(iterPts = pts->begin(); iterPts != pts->end(); ++iterPts)
                    {
                        if(elevVal == SPD_USE_Z)
                        {
                            elev = (*iterPts)->z;
                        }
                        else if(elevVal == SPD_USE_HEIGHT)
                        {
                            elev = (*iterPts)->height;
                        }
                        else
                        {
                            throw SPDProcessingException("Elevation type not recognised.");
                        }

                        sumSq += (elev - mean) * (elev - mean);
                    }

                    double stdDev = sqrt(sumSq);

                    if(stdDev < this->stdDevThreshold)
                    {
                        returnZVal = minHigh;
                    }
                    else
                    {
                        if(firstLow)
                        {
                            returnZVal = minHigh;
                        }
                        else
                        {
                            returnZVal = minLow;
                        }

                    }
                }
                else if( selectHighOrLow == SPD_SELECT_HIGHEST)
                {
                    double maxLow = 0;
                    double maxHigh = 0;
                    double sum = 0;
                    bool firstLow = true;
                    bool firstHigh = true;
                    double dist = 0;
                    double elev = 0;

                    SPDPointUtils ptUtils;

                    std::vector<SPDPoint*>::iterator iterPts;
                    for(iterPts = pts->begin(); iterPts != pts->end(); ++iterPts)
                    {
                        if(elevVal == SPD_USE_Z)
                        {
                            elev = (*iterPts)->z;
                        }
                        else if(elevVal == SPD_USE_HEIGHT)
                        {
                            elev = (*iterPts)->height;
                        }
                        else
                        {
                            throw SPDProcessingException("Elevation type not recognised.");
                        }
                        sum += elev;
                        dist = ptUtils.distanceXY(eastings, northings, *iterPts);
                        if(dist < this->highDist)
                        {
                            if(firstHigh)
                            {
                                maxHigh = elev;
                                firstHigh = false;
                            }
                            else if(elev > maxHigh)
                            {
                                maxHigh = elev;
                            }

                            if(dist < this->lowDist)
                            {
                                if(firstLow)
                                {
                                    maxLow = elev;
                                    firstLow = false;
                                }
                                else if(elev > maxLow)
                                {
                                    maxLow = elev;
                                }
                            }
                        }
                    }

                    double mean = sum / pts->size();
                    double sumSq = 0;

                    for(iterPts = pts->begin(); iterPts != pts->end(); ++iterPts)
                    {
                        if(elevVal == SPD_USE_Z)
                        {
                            elev = (*iterPts)->z;
                        }
                        else if(elevVal == SPD_USE_HEIGHT)
                        {
                            elev = (*iterPts)->height;
                        }
                        else
                        {
                            throw SPDProcessingException("Elevation type not recognised.");
                        }

                        sumSq += (elev - mean) * (elev - mean);
                    }

                    double stdDev = sqrt(sumSq);

                    if(stdDev < this->stdDevThreshold)
                    {
                        returnZVal = maxHigh;
                    }
                    else
                    {
                        if(firstLow)
                        {
                            returnZVal = maxHigh;
                        }
                        else
                        {
                            returnZVal = maxLow;
                        }

                    }
                }
			}
			else
			{
				returnZVal = std::numeric_limits<float>::signaling_NaN();
			}
			
			delete pts;
		}
		catch (SPDProcessingException &e)
		{
			throw e;
		}

        //std::cout << "returnZVal " << returnZVal << std::endl;
		return returnZVal;
	}
	
	SPDStdDevFilterInterpolator::~SPDStdDevFilterInterpolator()
	{
		
	}

	
	SPDTPSRadiusInterpolator::SPDTPSRadiusInterpolator(float radius, boost::uint_fast16_t minNumPoints, double gridResolution, boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin):SPDGridIndexPointInterpolator(gridResolution, elevVal, thinGridRes, thinData, selectHighOrLow, maxNumPtsPerBin),radius(0), minNumPoints(12)
	{
		this->radius = radius;
        this->minNumPoints = minNumPoints;
	}
	
	float SPDTPSRadiusInterpolator::getValue(double eastings, double northings) throw(SPDProcessingException)
	{
        float newZValue = std::numeric_limits<float>::signaling_NaN();
        std::vector<SPDPoint*> *splinePts = new std::vector<SPDPoint*>();
		try
        {
            if(idx->getPointsInRadius(splinePts, eastings, northings, radius))
            {
                if(splinePts->size() < minNumPoints)
                {
                    newZValue = std::numeric_limits<float>::signaling_NaN();
                }
                else
                {
                    std::vector<spdlib::tps::Vec> cntrlPts(splinePts->size());
                    int ptIdx = 0;
                    for(std::vector<SPDPoint*>::iterator iterPts = splinePts->begin(); iterPts != splinePts->end(); ++iterPts)
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

                    spdlib::tps::Spline splineFunc = spdlib::tps::Spline(cntrlPts, 0.0);
                    newZValue = splineFunc.interpolate_height(eastings, northings);
                }
            }
            else
            {
                newZValue = std::numeric_limits<float>::signaling_NaN();
            }
        }
        catch(spdlib::tps::SingularMatrixError &e)
        {
            //throw SPDProcessingException(e.what());
            newZValue = std::numeric_limits<float>::signaling_NaN();
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
        delete splinePts;

        return newZValue;
	}
	
	SPDTPSRadiusInterpolator::~SPDTPSRadiusInterpolator()
	{
		
	}


    SPDTPSNumPtsInterpolator::SPDTPSNumPtsInterpolator(float radius, boost::uint_fast16_t numPoints, double gridResolution, boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin):SPDGridIndexPointInterpolator(gridResolution, elevVal, thinGridRes, thinData, selectHighOrLow, maxNumPtsPerBin),radius(0), numPoints(12)
	{
		this->radius = radius;
        this->numPoints = numPoints;
	}
	
	float SPDTPSNumPtsInterpolator::getValue(double eastings, double northings) throw(SPDProcessingException)
	{
        float newZValue = std::numeric_limits<float>::signaling_NaN();
        std::vector<SPDPoint*> *splinePts = new std::vector<SPDPoint*>();
		try
        {
            if(idx->getSetNumOfPoints(splinePts, eastings, northings, numPoints, radius))
            {
                std::vector<spdlib::tps::Vec> cntrlPts(splinePts->size());
                int ptIdx = 0;
                for(std::vector<SPDPoint*>::iterator iterPts = splinePts->begin(); iterPts != splinePts->end(); ++iterPts)
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
                spdlib::tps::Spline splineFunc = spdlib::tps::Spline(cntrlPts, 0.0);
                newZValue = splineFunc.interpolate_height(eastings, northings);
            }
            else
            {
                newZValue = std::numeric_limits<float>::signaling_NaN();
            }
        }
        catch(spdlib::tps::SingularMatrixError &e)
        {
            //throw SPDProcessingException(e.what());
            newZValue = std::numeric_limits<float>::signaling_NaN();
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
        delete splinePts;

        return newZValue;
	}
	
	SPDTPSNumPtsInterpolator::~SPDTPSNumPtsInterpolator()
	{
		
	}


    SPDNaturalNeighborCGALPointInterpolator::SPDNaturalNeighborCGALPointInterpolator(boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin):SPDTriangulationPointInterpolator(elevVal, thinGridRes, thinData, selectHighOrLow, maxNumPtsPerBin)
    {

    }

    float SPDNaturalNeighborCGALPointInterpolator::getValue(double eastings, double northings) throw(SPDProcessingException)
    {
        float newZValue = std::numeric_limits<float>::signaling_NaN();
        if(initialised)
        {
            try
            {
                K::Point_2 p(eastings, northings);
                CoordinateVector coords;
                // Locate point using previous triangle as a hint and use this as starting point
                // As going through locations within a block this approach should be valid and makes faster.
                // from http://stackoverflow.com/questions/30354284/cgal-natural-neighbor-interpolation
                fh = dt->locate(p, fh);
                CGAL::Triple<std::back_insert_iterator<CoordinateVector>, K::FT, bool> result = CGAL::natural_neighbor_coordinates_2(*dt, p, std::back_inserter(coords), fh);
                if(!result.third)
                {
                    newZValue = std::numeric_limits<float>::signaling_NaN();
                }
                else
                {
                    CGALCoordType norm = result.second;

                    CGALCoordType outValue = CGAL::linear_interpolation(coords.begin(), coords.end(), norm, CGAL::Data_access<PointValueMap>(*this->values));

                    newZValue = outValue;
                }
            }
            catch(SPDProcessingException &e)
            {
                throw e;
            }
        }
        return newZValue;
    }

    SPDNaturalNeighborCGALPointInterpolator::~SPDNaturalNeighborCGALPointInterpolator()
    {

    }




    SPDSphericalPointInterpolator::SPDSphericalPointInterpolator(boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin): initialised(false), elevVal(0), thinGridRes(0.5), thinData(false), selectHighOrLow(0), maxNumPtsPerBin(0), totalNumPoints(0)
	{
        this->elevVal = elevVal;
        this->thinGridRes = thinGridRes;
        this->thinData = thinData;
        this->selectHighOrLow = selectHighOrLow;
        this->maxNumPtsPerBin = maxNumPtsPerBin;
	}

    std::vector<SPDPoint*>* SPDSphericalPointInterpolator::findPoints(std::list<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException)
    {
        std::vector<SPDPoint*> *points = new std::vector<SPDPoint*>();
        std::list<SPDPulse*>::iterator iterPulses;
        std::vector<SPDPoint*>::iterator iterPts;
        SPDPoint *pt = NULL;
        SPDPoint *newPt = NULL;
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
                                newPt = new SPDPoint();
                                newPt->x = (*iterPulses)->azimuth;
                                newPt->y = (*iterPulses)->zenith;
                                newPt->z = pt->range;
                                points->push_back(newPt);
                            }
                        }
                        else if(ptClass == SPD_ALL_CLASSES_TOP)
                        {
                            pt = (*iterPulses)->pts->front();
                            newPt = new SPDPoint();
                            newPt->x = (*iterPulses)->azimuth;
                            newPt->y = (*iterPulses)->zenith;
                            newPt->z = pt->range;
                            points->push_back(newPt);
                        }
                        else
                        {
                            for(iterPts = (*iterPulses)->pts->begin(); iterPts != (*iterPulses)->pts->end(); ++iterPts)
                            {
                                if(ptClass == SPD_ALL_CLASSES)
                                {
                                    newPt = new SPDPoint();
                                    newPt->x = (*iterPulses)->azimuth;
                                    newPt->y = (*iterPulses)->zenith;
                                    newPt->z = (*iterPts)->range;
                                    points->push_back(newPt);
                                }
                                else if((*iterPts)->classification == ptClass)
                                {
                                    newPt = new SPDPoint();
                                    newPt->x = (*iterPulses)->azimuth;
                                    newPt->y = (*iterPulses)->zenith;
                                    newPt->z = (*iterPts)->range;
                                    points->push_back(newPt);
                                }
                            }
                        }
                    }
                }
            }
        }
        totalNumPoints = points->size();

        return points;
    }

    std::vector<SPDPoint*>* SPDSphericalPointInterpolator::findPoints(std::vector<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException)
    {
        std::vector<SPDPoint*> *points = new std::vector<SPDPoint*>();
        std::vector<SPDPulse*>::iterator iterPulses;
        std::vector<SPDPoint*>::iterator iterPts;
        SPDPoint *pt = NULL;
        SPDPoint *newPt = NULL;
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
                                newPt = new SPDPoint();
                                newPt->x = (*iterPulses)->azimuth;
                                newPt->y = (*iterPulses)->zenith;
                                newPt->z = pt->range;
                                points->push_back(newPt);
                            }
                        }
                        else if(ptClass == SPD_ALL_CLASSES_TOP)
                        {
                            pt = (*iterPulses)->pts->front();
                            newPt = new SPDPoint();
                            newPt->x = (*iterPulses)->azimuth;
                            newPt->y = (*iterPulses)->zenith;
                            newPt->z = pt->range;
                            points->push_back(newPt);
                        }
                        else
                        {
                            for(iterPts = (*iterPulses)->pts->begin(); iterPts != (*iterPulses)->pts->end(); ++iterPts)
                            {
                                if(ptClass == SPD_ALL_CLASSES)
                                {
                                    newPt = new SPDPoint();
                                    newPt->x = (*iterPulses)->azimuth;
                                    newPt->y = (*iterPulses)->zenith;
                                    newPt->z = (*iterPts)->range;
                                    points->push_back(newPt);
                                }
                                else if((*iterPts)->classification == ptClass)
                                {
                                    newPt = new SPDPoint();
                                    newPt->x = (*iterPulses)->azimuth;
                                    newPt->y = (*iterPulses)->zenith;
                                    newPt->z = (*iterPts)->range;
                                    points->push_back(newPt);
                                }
                            }
                        }
                    }
                }
            }
        }
        totalNumPoints = points->size();

        return points;
    }

    std::vector<SPDPoint*>* SPDSphericalPointInterpolator::findPoints(std::list<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException)
    {
        std::vector<SPDPoint*> *points = new std::vector<SPDPoint*>();
        std::list<SPDPulse*>::iterator iterPulses;
		std::vector<SPDPoint*>::iterator iterPts;
        SPDPoint *pt = NULL;
        SPDPoint *newPt = NULL;
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
                        newPt = new SPDPoint();
                        newPt->x = (*iterPulses)->azimuth;
                        newPt->y = (*iterPulses)->zenith;
                        newPt->z = pt->range;
                        points->push_back(newPt);
                    }
                }
                else if(ptClass == SPD_ALL_CLASSES_TOP)
                {
                    pt = (*iterPulses)->pts->front();
                    newPt = new SPDPoint();
                    newPt->x = (*iterPulses)->azimuth;
                    newPt->y = (*iterPulses)->zenith;
                    newPt->z = pt->range;
                    points->push_back(newPt);
                }
                else
                {
                    for(iterPts = (*iterPulses)->pts->begin(); iterPts != (*iterPulses)->pts->end(); ++iterPts)
                    {
                        if(ptClass == SPD_ALL_CLASSES)
                        {
                            newPt = new SPDPoint();
                            newPt->x = (*iterPulses)->azimuth;
                            newPt->y = (*iterPulses)->zenith;
                            newPt->z = (*iterPts)->range;
                            points->push_back(newPt);
                        }
                        else if((*iterPts)->classification == ptClass)
                        {
                            newPt = new SPDPoint();
                            newPt->x = (*iterPulses)->azimuth;
                            newPt->y = (*iterPulses)->zenith;
                            newPt->z = (*iterPts)->range;
                            points->push_back(newPt);
                        }
                    }
                }
			}
		}
        totalNumPoints = points->size();

        return points;
    }

    std::vector<SPDPoint*>* SPDSphericalPointInterpolator::findPoints(std::vector<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException)
    {
        std::vector<SPDPoint*> *points = new std::vector<SPDPoint*>();
        std::vector<SPDPulse*>::iterator iterPulses;
		std::vector<SPDPoint*>::iterator iterPts;
        SPDPoint *pt = NULL;
        SPDPoint *newPt = NULL;
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
                        newPt = new SPDPoint();
                        newPt->x = (*iterPulses)->azimuth;
                        newPt->y = (*iterPulses)->zenith;
                        newPt->z = pt->range;
                        points->push_back(newPt);
                    }
                }
                else if(ptClass == SPD_ALL_CLASSES_TOP)
                {
                    pt = (*iterPulses)->pts->front();
                    newPt = new SPDPoint();
                    newPt->x = (*iterPulses)->azimuth;
                    newPt->y = (*iterPulses)->zenith;
                    newPt->z = pt->range;
                    points->push_back(newPt);
                }
                else
                {
                    for(iterPts = (*iterPulses)->pts->begin(); iterPts != (*iterPulses)->pts->end(); ++iterPts)
                    {
                        if(ptClass == SPD_ALL_CLASSES)
                        {
                            newPt = new SPDPoint();
                            newPt->x = (*iterPulses)->azimuth;
                            newPt->y = (*iterPulses)->zenith;
                            newPt->z = (*iterPts)->range;
                            points->push_back(newPt);
                        }
                        else if((*iterPts)->classification == ptClass)
                        {
                            newPt = new SPDPoint();
                            newPt->x = (*iterPulses)->azimuth;
                            newPt->y = (*iterPulses)->zenith;
                            newPt->z = (*iterPts)->range;
                            points->push_back(newPt);
                        }
                    }
                }
			}
		}
        totalNumPoints = points->size();

        return points;
    }

    void SPDSphericalPointInterpolator::thinPoints(std::vector<SPDPoint*> *points) throw(SPDProcessingException)
    {
        try
        {
            if(thinData)
            {
                SPDPointGridIndex ptIdx;
                ptIdx.buildIndex(points, thinGridRes);
                ptIdx.thinPtsInBins(elevVal, selectHighOrLow, maxNumPtsPerBin);
                points->clear();
                ptIdx.getAllPointsInGrid(points);
                totalNumPoints = points->size();
            }
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
    }





    SPDTriangulationSphericalPointInterpolator::SPDTriangulationSphericalPointInterpolator(boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin): SPDSphericalPointInterpolator(elevVal, thinGridRes, thinData, selectHighOrLow, maxNumPtsPerBin)
	{
        returnNaNValue = false;
	}
	
    void SPDTriangulationSphericalPointInterpolator::initInterpolator(std::list<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException)
    {
        try
        {
            points = this->findPoints(pulses, numXBins, numYBins, ptClass);
            if(thinData & (points->size() > 2))
            {
                this->thinPoints(points);
            }

            dt = new DelaunayTriangulation();
            values = new PointValueMap();
            if(points->size() > 2)
            {
                returnNaNValue = false;
                std::vector<SPDPoint*>::iterator iterPts;
                for(iterPts = points->begin(); iterPts != points->end(); ++iterPts)
                {
                    CGALPoint cgalPt((*iterPts)->x, (*iterPts)->y);
                    dt->insert(cgalPt);
                    CGALCoordType value = (*iterPts)->z;
                    values->insert(std::make_pair(cgalPt, value));
                }
            }
            else
            {
                returnNaNValue = true;
            }
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
        initialised = true;
    }

    void SPDTriangulationSphericalPointInterpolator::initInterpolator(std::vector<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException)
    {
        try
        {
            points = this->findPoints(pulses, numXBins, numYBins, ptClass);

            if(thinData & (points->size() > 2))
            {
                this->thinPoints(points);
            }


            dt = new DelaunayTriangulation();
            values = new PointValueMap();

            if(points->size() > 2)
            {
                returnNaNValue = false;
                std::vector<SPDPoint*>::iterator iterPts;
                for(iterPts = points->begin(); iterPts != points->end(); ++iterPts)
                {
                    CGALPoint cgalPt((*iterPts)->x, (*iterPts)->y);
                    dt->insert(cgalPt);
                    CGALCoordType value = (*iterPts)->z;
                    values->insert(std::make_pair(cgalPt, value));
                }
            }
            else
            {
                returnNaNValue = true;
            }

        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
        initialised = true;
    }

    void SPDTriangulationSphericalPointInterpolator::initInterpolator(std::list<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException)
    {
        try
        {
            points = this->findPoints(pulses, ptClass);

            if(thinData & (points->size() > 2))
            {
                this->thinPoints(points);
            }


            dt = new DelaunayTriangulation();
            values = new PointValueMap();

            if(points->size() > 2)
            {
                returnNaNValue = false;
                std::vector<SPDPoint*>::iterator iterPts;
                for(iterPts = points->begin(); iterPts != points->end(); ++iterPts)
                {
                    CGALPoint cgalPt((*iterPts)->x, (*iterPts)->y);
                    dt->insert(cgalPt);
                    CGALCoordType value = (*iterPts)->z;
                    values->insert(std::make_pair(cgalPt, value));
                }
            }
            else
            {
                returnNaNValue = true;
            }
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
        initialised = true;
    }

    void SPDTriangulationSphericalPointInterpolator::initInterpolator(std::vector<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException)
    {
        try
        {
            points = this->findPoints(pulses, ptClass);

            if(thinData & (points->size() > 2))
            {
                this->thinPoints(points);
            }


            dt = new DelaunayTriangulation();
            values = new PointValueMap();

            if(points->size() > 2)
            {
                returnNaNValue = false;
                std::vector<SPDPoint*>::iterator iterPts;
                for(iterPts = points->begin(); iterPts != points->end(); ++iterPts)
                {
                    CGALPoint cgalPt((*iterPts)->x, (*iterPts)->y);
                    dt->insert(cgalPt);
                    CGALCoordType value = (*iterPts)->z;
                    values->insert(std::make_pair(cgalPt, value));
                }
            }
            else
            {
                returnNaNValue = true;
            }
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
        initialised = true;
    }
	
	void SPDTriangulationSphericalPointInterpolator::resetInterpolator() throw(SPDProcessingException)
	{
        if(initialised)
        {
            delete dt;
            delete values;
            for(std::vector<SPDPoint*>::iterator iterPts = points->begin(); iterPts != points->end(); )
            {
                delete *iterPts;
                iterPts = points->erase(iterPts);
            }
            delete points;
            totalNumPoints = 0;
        }
		initialised = false;
        returnNaNValue = false;
	}
	
	SPDTriangulationSphericalPointInterpolator::~SPDTriangulationSphericalPointInterpolator()
	{
        if(initialised)
        {
            delete dt;
            delete values;
            for(std::vector<SPDPoint*>::iterator iterPts = points->begin(); iterPts != points->end(); )
            {
                delete *iterPts;
                iterPts = points->erase(iterPts);
            }
            delete points;
        }
		initialised = false;
        returnNaNValue = false;
	}







	SPDGridIndexSphericalPointInterpolator::SPDGridIndexSphericalPointInterpolator(double gridResolution, boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin): SPDSphericalPointInterpolator(elevVal, thinGridRes, thinData, selectHighOrLow, maxNumPtsPerBin), idx(NULL), gridResolution(0)
	{
		this->gridResolution = gridResolution;
	}
	
	void SPDGridIndexSphericalPointInterpolator::initInterpolator(std::list<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException)
	{
        try
        {
            points = this->findPoints(pulses, numXBins, numYBins, ptClass);
            if(thinData & (points->size() > 2))
            {
                this->thinPoints(points);
            }

            idx = new SPDPointGridIndex();
			idx->buildIndex(points, this->gridResolution);
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
        initialised = true;
	}
	
	void SPDGridIndexSphericalPointInterpolator::initInterpolator(std::vector<SPDPulse*> ***pulses, boost::uint_fast32_t numXBins, boost::uint_fast32_t numYBins, boost::uint_fast16_t ptClass) throw(SPDProcessingException)
	{
        try
        {
            points = this->findPoints(pulses, numXBins, numYBins, ptClass);

            if(thinData & (points->size() > 2))
            {
                this->thinPoints(points);
            }

            idx = new SPDPointGridIndex();
			idx->buildIndex(points, this->gridResolution);
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
        initialised = true;
	}
	
	void SPDGridIndexSphericalPointInterpolator::initInterpolator(std::list<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException)
	{
        try
        {
            points = this->findPoints(pulses, ptClass);
            if(thinData & (points->size() > 2))
            {
                this->thinPoints(points);
            }

            idx = new SPDPointGridIndex();
			idx->buildIndex(points, this->gridResolution);
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
        initialised = true;
	}
	
	void SPDGridIndexSphericalPointInterpolator::initInterpolator(std::vector<SPDPulse*> *pulses, boost::uint_fast16_t ptClass) throw(SPDProcessingException)
	{
        try
        {
            points = this->findPoints(pulses, ptClass);
            if(thinData & (points->size() > 2))
            {
                this->thinPoints(points);
            }

            idx = new SPDPointGridIndex();
			idx->buildIndex(points, this->gridResolution);
        }
        catch(SPDProcessingException &e)
        {
            throw e;
        }
        initialised = true;
	}

	void SPDGridIndexSphericalPointInterpolator::resetInterpolator() throw(SPDProcessingException)
	{
		if(initialised)
		{
			delete idx;
            totalNumPoints = 0;
            for(std::vector<SPDPoint*>::iterator iterPts = points->begin(); iterPts != points->end(); )
            {
                delete *iterPts;
                iterPts = points->erase(iterPts);
            }
            delete points;
		}
        initialised = false;
	}
	
	SPDGridIndexSphericalPointInterpolator::~SPDGridIndexSphericalPointInterpolator()
	{
		if(initialised)
		{
			delete idx;
            for(std::vector<SPDPoint*>::iterator iterPts = points->begin(); iterPts != points->end(); )
            {
                delete *iterPts;
                iterPts = points->erase(iterPts);
            }
            delete points;
		}
        initialised = false;
	}
	



    SPDNaturalNeighborSphericalPointInterpolator::SPDNaturalNeighborSphericalPointInterpolator(boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin):SPDTriangulationSphericalPointInterpolator(elevVal, thinGridRes, thinData, selectHighOrLow, maxNumPtsPerBin)
    {

    }

    float SPDNaturalNeighborSphericalPointInterpolator::getValue(double azimuth, double zenith) throw(SPDProcessingException)
    {
        float newRangeValue = std::numeric_limits<float>::signaling_NaN();
        if(initialised)
        {
            if(!returnNaNValue)
            {
                try
                {
                    CGALPoint p(azimuth, zenith);
                    CoordinateVector coords;
                    CGAL::Triple<std::back_insert_iterator<CoordinateVector>, K::FT, bool> result = CGAL::natural_neighbor_coordinates_2(*dt, p, std::back_inserter(coords));
                    if(!result.third)
                    {
                        newRangeValue = std::numeric_limits<float>::signaling_NaN();
                    }
                    else
                    {
                        CGALCoordType norm = result.second;

                        CGALCoordType outValue = CGAL::linear_interpolation(coords.begin(), coords.end(), norm, CGAL::Data_access<PointValueMap>(*this->values));

                        newRangeValue = outValue;
                    }
                }
                catch(SPDProcessingException &e)
                {
                    throw e;
                }
            }
        }
        return newRangeValue;
    }

    SPDNaturalNeighborSphericalPointInterpolator::~SPDNaturalNeighborSphericalPointInterpolator()
    {

    }



    SPDNearestNeighborSphericalPointInterpolator::SPDNearestNeighborSphericalPointInterpolator(boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin, float distanceThreshold):SPDTriangulationSphericalPointInterpolator(elevVal, thinGridRes, thinData, selectHighOrLow, maxNumPtsPerBin)
	{
		this->distanceThreshold = distanceThreshold;
	}
	
	float SPDNearestNeighborSphericalPointInterpolator::getValue(double azimuth, double zenith) throw(SPDProcessingException)
	{
		double newRangeValue = std::numeric_limits<float>::signaling_NaN();
		if(initialised)
		{
            if(!returnNaNValue)
            {
                CGALPoint p(azimuth, zenith);
                Vertex_handle vh = dt->nearest_vertex(p);
                CGALPoint nearestPt = vh->point();

                double distance = CGAL::squared_distance(p, nearestPt);

                if(distance < distanceThreshold)
                {
                    PointValueMap::iterator iterVal = values->find(nearestPt);
                    newRangeValue = (*iterVal).second;
                }
                else
                {
                    newRangeValue = std::numeric_limits<float>::signaling_NaN();
                }
            }
		}
		else
		{
			throw SPDProcessingException("Interpolated needs to be initialised before values can be retrieved.");
		}
		return newRangeValue;

    }
	
	SPDNearestNeighborSphericalPointInterpolator::~SPDNearestNeighborSphericalPointInterpolator()
	{
		
	}



    SPDTPSRadiusSphericalInterpolator::SPDTPSRadiusSphericalInterpolator(float radius, boost::uint_fast16_t minNumPoints, double gridResolution, boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin):SPDGridIndexSphericalPointInterpolator(gridResolution, elevVal, thinGridRes, thinData, selectHighOrLow, maxNumPtsPerBin),radius(0), minNumPoints(12)
	{
		this->radius = radius;
        this->minNumPoints = minNumPoints;
	}
	
	float SPDTPSRadiusSphericalInterpolator::getValue(double azimuth, double zenith) throw(SPDProcessingException)
	{
        float newRangeValue = std::numeric_limits<float>::signaling_NaN();
        std::vector<SPDPoint*> *splinePts = new std::vector<SPDPoint*>();
		try
        {
            if(idx->getPointsInRadius(splinePts, azimuth, zenith, radius))
            {
                if(splinePts->size() < minNumPoints)
                {
                    newRangeValue = std::numeric_limits<float>::signaling_NaN();
                }
                else
                {
                    std::vector<spdlib::tps::Vec> cntrlPts(splinePts->size());
                    int ptIdx = 0;
                    for(std::vector<SPDPoint*>::iterator iterPts = splinePts->begin(); iterPts != splinePts->end(); ++iterPts)
                    {
                        // Please note that Z and Y and been switch around as the TPS code (tpsdemo)
                        // interpolates for Y rather than Z.
                        cntrlPts[ptIdx++] = spdlib::tps::Vec((*iterPts)->x, (*iterPts)->z, (*iterPts)->y);
                    }

                    spdlib::tps::Spline splineFunc = spdlib::tps::Spline(cntrlPts, 0.0);
                    newRangeValue = splineFunc.interpolate_height(azimuth, zenith);
                }
            }
            else
            {
                newRangeValue = std::numeric_limits<float>::signaling_NaN();
            }
        }
        catch(spdlib::tps::SingularMatrixError &e)
        {
            //throw SPDProcessingException(e.what());
            newRangeValue = std::numeric_limits<float>::signaling_NaN();
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
        delete splinePts;

        return newRangeValue;
	}
	
	SPDTPSRadiusSphericalInterpolator::~SPDTPSRadiusSphericalInterpolator()
	{
		
	}



    SPDTPSNumPtsSphericalInterpolator::SPDTPSNumPtsSphericalInterpolator(float radius, boost::uint_fast16_t numPoints, double gridResolution, boost::uint_fast16_t elevVal, float thinGridRes, bool thinData, boost::uint_fast16_t selectHighOrLow, boost::uint_fast16_t maxNumPtsPerBin):SPDGridIndexSphericalPointInterpolator(gridResolution, elevVal, thinGridRes, thinData, selectHighOrLow, maxNumPtsPerBin),radius(0), numPoints(12)
	{
		this->radius = radius;
        this->numPoints = numPoints;
	}
	
	float SPDTPSNumPtsSphericalInterpolator::getValue(double azimuth, double zenith) throw(SPDProcessingException)
	{
        float newRangeValue = std::numeric_limits<float>::signaling_NaN();
        std::vector<SPDPoint*> *splinePts = new std::vector<SPDPoint*>();
		try
        {
            if(idx->getSetNumOfPoints(splinePts, azimuth, zenith, numPoints, radius))
            {
                std::vector<spdlib::tps::Vec> cntrlPts(splinePts->size());
                int ptIdx = 0;
                for(std::vector<SPDPoint*>::iterator iterPts = splinePts->begin(); iterPts != splinePts->end(); ++iterPts)
                {
                    // Please note that Z and Y and been switch around as the TPS code (tpsdemo)
                    // interpolates for Y rather than Z.
                    cntrlPts[ptIdx++] = spdlib::tps::Vec((*iterPts)->x, (*iterPts)->z, (*iterPts)->y);
                }
                spdlib::tps::Spline splineFunc = spdlib::tps::Spline(cntrlPts, 0.0);
                newRangeValue = splineFunc.interpolate_height(azimuth, zenith);
            }
            else
            {
                newRangeValue = std::numeric_limits<float>::signaling_NaN();
            }
        }
        catch(spdlib::tps::SingularMatrixError &e)
        {
            //throw SPDProcessingException(e.what());
            newRangeValue = std::numeric_limits<float>::signaling_NaN();
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
        delete splinePts;

        return newRangeValue;
	}
	
	SPDTPSNumPtsSphericalInterpolator::~SPDTPSNumPtsSphericalInterpolator()
	{
		
	}

}


