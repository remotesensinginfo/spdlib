/*
 *  SPDExportAsReadUnGridded.cpp
 *  spdlib
 *
 *  Created by Pete Bunting on 04/12/2010.
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
 */

#include "spd/SPDExportAsReadUnGridded.h"

namespace spdlib
{

	SPDExportAsReadUnGridded::SPDExportAsReadUnGridded(SPDDataExporter *exporter, SPDFile *spdFileOut, bool defineSource, boost::uint_fast16_t sourceID, bool defineReturnID, boost::uint_fast16_t returnID, bool defineClasses, boost::uint_fast16_t classValue) throw(SPDException): SPDImporterProcessor(), exporter(NULL), spdFileOut(NULL), fileOpen(false), pulses(NULL)
	{
		this->exporter = exporter;
		this->spdFileOut = spdFileOut;
        this->defineSource = defineSource;
        this->sourceID = sourceID;
        this->defineReturnID = defineReturnID;
        this->returnID = returnID;
        this->defineClasses = defineClasses;
        this->classValue = classValue;
		
		if(exporter->requireGrid())
		{
			throw SPDException("This class does not support the export of gridded formats.");
		}
		
		try
		{
			this->exporter->open(this->spdFileOut, this->spdFileOut->getFilePath());
		}
		catch (SPDException &e)
		{
			throw e;
		}
		this->fileOpen = true;
		this->pulses = new std::vector<SPDPulse*>();
        this->pulses->reserve(1000);
		
	}
		
	void SPDExportAsReadUnGridded::processImportedPulse(SPDFile *spdFile, SPDPulse *pulse) throw(SPDIOException)
	{
		try
		{
            if(defineSource)
            {
                pulse->sourceID = sourceID;
            }
            if(defineReturnID)
            {
                if(pulse->numberOfReturns > 0)
                {
                    std::vector<SPDPoint*>::iterator iterPts;
                    for(iterPts = pulse->pts->begin(); iterPts != pulse->pts->end(); ++iterPts)
                    {
                        (*iterPts)->returnID = returnID;
                    }
                }
            }
            if(defineClasses)
            {
                if(pulse->numberOfReturns > 0)
                {
                    std::vector<SPDPoint*>::iterator iterPts;
                    for(iterPts = pulse->pts->begin(); iterPts != pulse->pts->end(); ++iterPts)
                    {
                        (*iterPts)->classification = classValue;
                    }
                }
            }
			this->pulses->push_back(pulse);
			this->exporter->writeDataColumn(pulses, 0, 0);
		}
		catch (SPDIOException &e)
		{
			throw e;
		}
	}
		
	void SPDExportAsReadUnGridded::completeFileAndClose(SPDFile *spdFile)throw(SPDIOException)
	{
		try
		{
			spdFileOut->copyAttributesFrom(spdFile);
			exporter->finaliseClose();
		}
		catch (SPDIOException &e)
		{
			throw e;
		}
	}

    void SPDExportAsReadUnGridded::setSourceID(boost::uint_fast16_t sourceID)
    {
        this->sourceID = sourceID;
    }

    void SPDExportAsReadUnGridded::setReturnID(boost::uint_fast16_t returnID)
    {
        this->returnID = returnID;
    }

    void SPDExportAsReadUnGridded::setClassValue(boost::uint_fast16_t classValue)
    {
        this->classValue = classValue;
    }
	
	SPDExportAsReadUnGridded::~SPDExportAsReadUnGridded()
	{
		delete pulses;
	}
	
}

