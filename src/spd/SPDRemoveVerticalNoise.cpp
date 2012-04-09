/*
 *  SPDRemoveVerticalNoise.cpp
 *  SPDLIB
 *
 *  Created by Pete Bunting on 07/06/2011.
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

#include "spd/SPDRemoveVerticalNoise.h"

namespace spdlib
{

    SPDRemoveVerticalNoise::SPDRemoveVerticalNoise(bool absUpSet, bool absLowSet, bool relUpSet, bool relLowSet, float absUpThres, float absLowThres, float relUpThres, float relLowThres)
    {
        this->absUpSet = absUpSet;
        this->absLowSet = absLowSet;
        this->relUpSet = relUpSet;
        this->relLowSet = relLowSet;
        this->absUpThres = absUpThres;
        this->absLowThres = absLowThres;
        this->relUpThres = relUpThres;
        this->relLowThres = relLowThres;
    }

    void SPDRemoveVerticalNoise::processDataColumn(SPDFile *inSPDFile, vector<SPDPulse*> *pulses, SPDXYPoint *cenPts) throw(SPDProcessingException)
    {
        if(pulses->size() > 0)
        {
            vector<SPDPulse*>::iterator iterPulses;
            vector<SPDPoint*>::iterator iterPoints;
            vector<double> ptVals;
            bool firstPt = true;
            bool removedFirst = false;
            
            for(iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
            {
                if((*iterPulses)->numberOfReturns > 0)
                {
                    firstPt = true;
                    removedFirst = false;
                    for(iterPoints = (*iterPulses)->pts->begin(); iterPoints != (*iterPulses)->pts->end(); )
                    {
                        if(removedFirst && (inSPDFile->getIndexType() == SPD_FIRST_RETURN))
                        {
                            (*iterPulses)->xIdx = (*iterPoints)->x;
                            (*iterPulses)->yIdx = (*iterPoints)->y;
                            removedFirst = false;
                        }
                        
                        if(absUpSet && ((*iterPoints)->z > absUpThres))
                        {
                            delete *iterPoints;
                            iterPoints = (*iterPulses)->pts->erase(iterPoints);
                            (*iterPulses)->numberOfReturns -= 1;
                            if(firstPt)
                            {
                                removedFirst = true;
                            }
                        }
                        else if(absLowSet && ((*iterPoints)->z < absLowThres))
                        {
                            delete *iterPoints;
                            iterPoints = (*iterPulses)->pts->erase(iterPoints);
                            (*iterPulses)->numberOfReturns -= 1;
                            if(firstPt)
                            {
                                removedFirst = true;
                            }
                        }
                        else
                        {
                            ptVals.push_back((*iterPoints)->z);
                            ++iterPoints;
                            firstPt = false;
                        }
                    }
                }
            }
            
            if(ptVals.size() > 0)
            {
                if(relUpSet | relLowSet)
                {
                    double min, max = 0;
                    gsl_stats_minmax(&min, &max, &ptVals[0], 1, ptVals.size());
                    double upThres = min + relUpThres;
                    double lowThres = max - relLowThres;
                    
                    //cout << "Thresholds: " << lowThres << ", " << upThres << endl;
                    
                    for(iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
                    {
                        if((*iterPulses)->numberOfReturns > 0)
                        {
                            firstPt = true;
                            removedFirst = false;
                            for(iterPoints = (*iterPulses)->pts->begin(); iterPoints != (*iterPulses)->pts->end(); )
                            {
                                if(removedFirst && (inSPDFile->getIndexType() == SPD_FIRST_RETURN))
                                {
                                    (*iterPulses)->xIdx = (*iterPoints)->x;
                                    (*iterPulses)->yIdx = (*iterPoints)->y;
                                    removedFirst = false;
                                }
                                
                                if(relUpSet && ((*iterPoints)->z > upThres))
                                {
                                    delete *iterPoints;
                                    iterPoints = (*iterPulses)->pts->erase(iterPoints);
                                    (*iterPulses)->numberOfReturns -= 1;
                                    if(firstPt)
                                    {
                                        removedFirst = true;
                                    }
                                }
                                else if(relLowSet && ((*iterPoints)->z < lowThres))
                                {
                                    delete *iterPoints;
                                    iterPoints = (*iterPulses)->pts->erase(iterPoints);
                                    (*iterPulses)->numberOfReturns -= 1;
                                    if(firstPt)
                                    {
                                        removedFirst = true;
                                    }
                                }
                                else
                                {
                                    ++iterPoints;
                                    firstPt = false;
                                }
                            }
                        }
                    }
                    
                }
            }
        }
    }
        
    SPDRemoveVerticalNoise::~SPDRemoveVerticalNoise()
    {
        
    }
    
}


