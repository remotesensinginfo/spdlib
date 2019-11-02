/*
 *  SPDDecomposedDatFileImporter.h
 *  spdlib
 *
 *  Created by Pete Bunting on 02/12/2010.
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

#ifndef SPDDecomposedDatFileImporter_H
#define SPDDecomposedDatFileImporter_H

#include <list>
#include <stdexcept>

#include <boost/cstdint.hpp>

#include "ogrsf_frmts.h"

#include "spd/SPDCommon.h"
#include "spd/SPDFile.h"
#include "spd/SPDPulse.h"
#include "spd/SPDIOException.h"
#include "spd/SPDTextFileLineReader.h"
#include "spd/SPDTextFileUtilities.h"
#include "spd/SPDDataImporter.h"

// mark all exported classes/functions with DllExport to have
// them exported by Visual Studio
#undef DllExport
#ifdef _MSC_VER
    #ifdef libspd_EXPORTS
        #define DllExport   __declspec( dllexport )
    #else
        #define DllExport   __declspec( dllimport )
    #endif
#else
    #define DllExport
#endif

namespace spdlib
{
	
	class DllExport SPDDecomposedDatFileImporter : public SPDDataImporter
	{
	public:
		SPDDecomposedDatFileImporter(bool convertCoords=false, std::string outputProjWKT="", std::string schema="", boost::uint_fast16_t indexCoords=SPD_FIRST_RETURN, bool defineOrigin=false, double originX=0, double originY=0, float originZ=0, float waveNoiseThreshold=0);
		SPDDataImporter* getInstance(bool convertCoords, std::string outputProjWKT, std::string schema, boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold);
        std::list<SPDPulse*>* readAllDataToList(std::string inputFile, SPDFile *spdFile);
		std::vector<SPDPulse*>* readAllDataToVector(std::string inputFile, SPDFile *spdFile);
		void readAndProcessAllData(std::string inputFile, SPDFile *spdFile, SPDImporterProcessor *processor) ;
		bool isFileType(std::string fileType);
        void readHeaderInfo(std::string inputFile, SPDFile *spdFile) ;
		~SPDDecomposedDatFileImporter();
	private:
		SPDPoint* createSPDPoint(std::string pointLine);
		bool classWarningGiven;
		
	};
}

#endif


