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

using namespace std;

namespace spdlib
{
	
	class SPDDecomposedDatFileImporter : public SPDDataImporter
	{
	public:
		SPDDecomposedDatFileImporter(bool convertCoords=false, string outputProjWKT="", string schema="", boost::uint_fast16_t indexCoords=SPD_FIRST_RETURN, bool defineOrigin=false, double originX=0, double originY=0, float originZ=0, float waveNoiseThreshold=0);
		SPDDataImporter* getInstance(bool convertCoords, string outputProjWKT, string schema, boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold);
        list<SPDPulse*>* readAllDataToList(string inputFile, SPDFile *spdFile)throw(SPDIOException);
		vector<SPDPulse*>* readAllDataToVector(string inputFile, SPDFile *spdFile)throw(SPDIOException);
		void readAndProcessAllData(string inputFile, SPDFile *spdFile, SPDImporterProcessor *processor) throw(SPDIOException);
		bool isFileType(string fileType);
        void readHeaderInfo(string inputFile, SPDFile *spdFile) throw(SPDIOException);
		~SPDDecomposedDatFileImporter();
	private:
		SPDPoint* createSPDPoint(string pointLine);
		bool classWarningGiven;
		
	};
}

#endif


