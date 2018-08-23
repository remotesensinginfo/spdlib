/*
 *  SPDSampleInTime.cpp
 *  SPDLIB
 *
 *  Created by Pete Bunting on 31/10/2012.
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

#include "spd/SPDSampleInTime.h"

namespace spdlib
{

    SPDSampleInTime::SPDSampleInTime(std::string outputFilePath, boost::uint_fast16_t tSample) throw(SPDException) : SPDImporterProcessor()
    {
        try
        {
            this->outSPDFile = new SPDFile(outputFilePath);
            this->exporter = new SPDNoIdxFileWriter();
            
            this->exporter->open(this->outSPDFile, outputFilePath);
            
            this->tSample = tSample;
            this->pulseCount = 0;
            this->pulseCountOut = 0;
            this->pulses = new std::vector<SPDPulse*>();
        }
        catch (SPDException &e)
        {
            throw e;
        }
        
    }
    
    void SPDSampleInTime::processImportedPulse(SPDFile *spdFile, SPDPulse *pulse) throw(SPDIOException)
    {
        //std::cout << "pulseCount = " << pulseCount << std::endl;
        //std::cout << "tSample = " << tSample << std::endl;
        //std::cout << "(pulseCount % tSample) = " << (pulseCount % tSample) << std::endl;
        try
		{
            if((pulseCount % tSample) == 0)
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
    
    void SPDSampleInTime::completeFileAndClose()throw(SPDIOException)
    {
        try
		{
            std::cout << "Number of Pulses Outputed = " << pulseCountOut << std::endl;
			this->exporter->finaliseClose();
            delete pulses;
		}
		catch (SPDIOException &e)
		{
			throw e;
		}
    }
    
    SPDSampleInTime::~SPDSampleInTime()
    {
        
    }
    
}

