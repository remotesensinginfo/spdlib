/*
 *  SPDSplit.cpp
 *  SPDLIB
 *
 *  Created by John Armston on 20/05/2014.
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

#include "spd/SPDSplit.h"

namespace spdlib
{

    SPDSplit::SPDSplit(std::string outputFilePath, boost::uint_fast16_t sourceID)  : SPDImporterProcessor()
    {
        try
        {
            this->outSPDFile = new SPDFile(outputFilePath);
            this->exporter = new SPDNoIdxFileWriter();
            
            this->exporter->open(this->outSPDFile, outputFilePath);
            
            this->sourceID = sourceID;
            this->pulseCount = 0;
            this->pulseCountOut = 0;
            this->pulses = new std::vector<SPDPulse*>();
        }
        catch (SPDException &e)
        {
            throw e;
        }
        
    }
    
    void SPDSplit::processImportedPulse(SPDFile *spdFile, SPDPulse *pulse) 
    {
        //std::cout << "pulseCount = " << pulseCount << std::endl;
        //std::cout << "sourceID = " << sourceID << std::endl;
        try
		{
            if(pulse->sourceID == sourceID)
            {
                pulses->push_back(pulse);
                this->exporter->writeDataColumn(pulses, 0, 0);
                ++pulseCountOut;
            }
            else
            {
                SPDPulseUtils plsUtils;
                plsUtils.deleteSPDPulse(pulse);
            }
            
            ++pulseCount;
		}
		catch (SPDIOException &e)
		{
			throw e;
		}
    }
    
    void SPDSplit::completeFileAndClose()
    {
        try
		{
            std::cout << "Number of pulses output = " << pulseCountOut << std::endl;
			this->exporter->finaliseClose();
            delete pulses;
		}
		catch (SPDIOException &e)
		{
			throw e;
		}
    }
    
    SPDSplit::~SPDSplit()
    {
        
    }
    
}

