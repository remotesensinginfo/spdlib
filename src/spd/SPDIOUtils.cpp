/*
 *  SPDIOUtils.cpp
 *  SPDLIB
 *
 *  Created by Pete Bunting on 20/07/2011.
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

#include "spd/SPDIOUtils.h"

namespace spdlib
{	

    SPDIOUtils::SPDIOUtils()
    {

    }

	void SPDIOUtils::gridAndWriteData(SPDDataExporter *exporter, std::list<SPDPulse*> *pls, SPDFile *spdFile, std::string outputFile)throw(SPDIOException)
	{
		try
		{
			SPDGridData gridData;
			std::list<SPDPulse*> ***griddedPts = gridData.gridData(pls, spdFile);
			
			exporter->open(spdFile, outputFile);
			
			boost::uint_fast32_t xSize = spdFile->getNumberBinsX();
			boost::uint_fast32_t ySize = spdFile->getNumberBinsY();
			
			boost::uint_fast32_t feedback = ySize/10;
			int feedbackCounter = 0;
			
			std::cout << "Started (Write Data) ." << std::flush;
			for(boost::uint_fast32_t i = 0; i < ySize; ++i)
			{
				if((feedback > 10) && ((i % feedback) == 0))
				{
					std::cout << "." << feedbackCounter << "." << std::flush;
					feedbackCounter += 10;
				}
				
				for(boost::uint_fast32_t j = 0; j < xSize; ++j)
				{
					exporter->writeDataColumn(griddedPts[i][j], j, i);
                    delete griddedPts[i][j];
				}
                delete[] griddedPts[i];
			}
			std::cout << ".Complete\n";
            delete [] griddedPts;
			
			exporter->finaliseClose();
		}
		catch (SPDProcessingException &e)
		{
			throw SPDIOException(e.what());
		}
		catch (SPDIOException &e)
		{
			throw e;
		}
	}
	
	void SPDIOUtils::gridAndWriteData(SPDDataExporter *exporter, std::vector<SPDPulse*> *pls, SPDFile *spdFile, std::string outputFile)throw(SPDIOException)
	{
		try
		{
			SPDGridData gridData;
			std::list<SPDPulse*> ***griddedPts = gridData.gridData(pls, spdFile);
			
			exporter->open(spdFile, outputFile);
			
			boost::uint_fast32_t xSize = spdFile->getNumberBinsX();
			boost::uint_fast32_t ySize = spdFile->getNumberBinsY();
			
			boost::uint_fast32_t feedback = ySize/10;
			int feedbackCounter = 0;
			
			std::cout << "Started (Write Data) ." << std::flush;
			for(boost::uint_fast32_t i = 0; i < ySize; ++i)
			{
				if((feedback > 10) && ((i % feedback) == 0))
				{
					std::cout << "." << feedbackCounter << "." << std::flush;
					feedbackCounter += 10;
				}
				
				for(boost::uint_fast32_t j = 0; j < xSize; ++j)
				{
					exporter->writeDataColumn(griddedPts[i][j], j, i);
                    delete griddedPts[i][j];
				}
                delete[] griddedPts[i];
			}
			std::cout << ".Complete\n";
            delete [] griddedPts;
			
			exporter->finaliseClose();
		}
		catch (SPDProcessingException &e)
		{
			throw SPDIOException(e.what());
		}
		catch (SPDIOException &e)
		{
			throw e;
		}
	}

    SPDIOUtils::~SPDIOUtils()
    {

    }
}

