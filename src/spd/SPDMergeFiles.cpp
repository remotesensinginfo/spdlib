/*
 *  SPDMergeFiles.cpp
 *  SPDLIB
 *
 *  Created by Pete Bunting on 19/12/2010.
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

#include "spd/SPDMergeFiles.h"

namespace spdlib
{
	
	SPDMergeFiles::SPDMergeFiles()
	{
		
	}
	
	void SPDMergeFiles::mergeToUPD(std::vector<std::string> inputFiles, std::string output, std::string inFormat,  std::string schema, std::string inSpatialRef, bool convertCoords, std::string outputProj4, boost::uint_fast16_t indexCoords, bool setSourceID, bool setReturnIDs, std::vector<boost::uint_fast16_t> returnID, bool setClasses, std::vector<boost::uint_fast16_t> classValues, bool ignoreChecks, boost::uint_fast16_t waveBinRes, bool keepMinExtent) throw(SPDException)
	{
		try 
		{
			SPDIOFactory ioFactory;
			
			SPDDataImporter *importer = ioFactory.getImporter(inFormat, convertCoords, outputProj4, schema, indexCoords);
			SPDDataExporter *exporter = ioFactory.getExporter("UPD");
            exporter->setKeepMinExtent(keepMinExtent);
			
			SPDFile *spdFileMerged = new SPDFile("");
			bool first = true;
			SPDFile *spdFileOut = new SPDFile(output);
            spdFileOut->setWaveformBitRes(waveBinRes);
			SPDExportAsReadUnGridded *exportAsRead = new SPDExportAsReadUnGridded(exporter, spdFileOut, setSourceID, 0, setReturnIDs, 0, setClasses, 0);
			
            boost::uint_fast16_t sourceID = 0;
            boost::uint_fast16_t fileCount = 0;
            
			for(std::vector<std::string>::iterator iterInFiles = inputFiles.begin(); iterInFiles != inputFiles.end(); ++iterInFiles)
			{
				SPDFile *spdFile = new SPDFile(*iterInFiles);
				spdFile->setSpatialReference(inSpatialRef);
                if(setSourceID)
                {
                    exportAsRead->setSourceID(sourceID++);
                }
                if(setReturnIDs)
                {
                    exportAsRead->setReturnID(returnID.at(fileCount));
                }
                if(setClasses)
                {
                    exportAsRead->setClassValue(classValues.at(fileCount));
                }
				importer->readAndProcessAllData(*iterInFiles, spdFile, exportAsRead);
				if(first)
				{
					spdFileMerged->copyAttributesFrom(spdFile);
					first = false;
				}
				else 
				{
					if(ignoreChecks)
                    {
                        spdFileMerged->expandExtent(spdFile);
                    }
                    else if(!spdFileMerged->checkCompatibilityGeneralCheckExpandExtent(spdFile))
					{
						std::string message = (*iterInFiles) + std::string(" was not compatiable with the file(s) previously read.");
						throw SPDException(message);
					}
				}
				delete spdFile;
                ++fileCount;
			}
			exportAsRead->completeFileAndClose(spdFileMerged);
			delete spdFileOut;
			delete exportAsRead;
		}
		catch (SPDException &e) 
		{
			throw e;
		}
	}
	
	SPDMergeFiles::~SPDMergeFiles()
	{
		
	}
	
}


