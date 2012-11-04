/*
 *  SPDThinPulses.cpp
 *  SPDLIB
 *
 *  Created by Pete Bunting on 04/11/2012.
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

#include "spd/SPDThinPulses.h"

namespace spdlib
{
    
    SPDThinPulses::SPDThinPulses(boost::uint_fast16_t numPulses)
    {
        this->numPulses = numPulses;
    }
    
    void SPDThinPulses::processDataColumn(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, SPDXYPoint *cenPts) throw(SPDProcessingException)
    {
        try
        {
            //std::cout << "pulses->size() = " << pulses->size();
            
            if(pulses->size() > numPulses)
            {
                std::vector<boost::uint_fast16_t> idxes;
                boost::uint_fast16_t idx = 0;
                bool notFoundIdx = false;
                bool found = false;
                
                for(boost::uint_fast16_t i = 0; i < numPulses; ++i)
                {
                    try
                    {
                        notFoundIdx = true;
                        while(notFoundIdx)
                        {                            
                            idx = boost::numeric_cast<boost::uint_fast16_t>(rand() % pulses->size());                            
                            
                            found = false;
                            for(std::vector<boost::uint_fast16_t>::iterator iterIdxes = idxes.begin(); iterIdxes != idxes.end(); ++iterIdxes)
                            {
                                if((*iterIdxes) == idx)
                                {
                                    found = true;
                                    break;
                                }
                            }
                            
                            if(!found)
                            {
                                notFoundIdx = false;
                                break;
                            }
                        }
                                                
                        idxes.push_back(idx);
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
                }
                
                SPDPulseUtils plsUtils;
                //for(std::vector<SPDPulse*>::iterator iterPulse = pulses->begin(); iterPulse != pulses->end(); )
                for(size_t i = 0; i < pulses->size(); ++i)
                {
                    found = false;
                    for(std::vector<boost::uint_fast16_t>::iterator iterIdxes = idxes.begin(); iterIdxes != idxes.end(); ++iterIdxes)
                    {
                        if((*iterIdxes) == i)
                        {
                            found = true;
                            break;
                        }
                    }
                    
                    if(!found)
                    {
                        //std::cout << "Remove idx " << i << std::endl;
                        //plsUtils.deleteSPDPulse(pulses->at(i));
                        pulses->at(i) = NULL;
                    }
                    
                    ++i;
                }
                
            }
            
            //std::cout << " = " << pulses->size() << std::endl;
        }
        catch (SPDProcessingException &e)
        {
            throw e;
        }
        catch(std::exception &e)
        {
            throw SPDProcessingException(e.what());
        }
        
    }
    
    SPDThinPulses::~SPDThinPulses()
    {
        
    }
    
}


