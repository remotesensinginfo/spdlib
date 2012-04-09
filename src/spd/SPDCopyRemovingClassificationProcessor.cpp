/*
 *  SPDCopyRemovingClassificationProcessor.cpp
 *  SPDLIB
 *
 *  Created by Pete Bunting on 28/12/2010.
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

#include "spd/SPDCopyRemovingClassificationProcessor.h"


namespace spdlib
{
    
    SPDCopyRemovingClassificationProcessor::SPDCopyRemovingClassificationProcessor()
    {

    }
    
    void SPDCopyRemovingClassificationProcessor::processDataColumn(SPDFile *inSPDFile, vector<SPDPulse*> *pulses, SPDXYPoint *cenPts) throw(SPDProcessingException)
    {
        vector<SPDPoint*>::iterator iterPoints;
		for(vector<SPDPulse*>::iterator iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
		{
			if((*iterPulses)->numberOfReturns > 0)
            {
                for(iterPoints = (*iterPulses)->pts->begin(); iterPoints != (*iterPulses)->pts->end(); ++iterPoints)
                {
                    (*iterPoints)->classification = SPD_UNCLASSIFIED;
                }
            }
		}
    }
    
    SPDCopyRemovingClassificationProcessor::~SPDCopyRemovingClassificationProcessor()
    {
        
    }
    
}


