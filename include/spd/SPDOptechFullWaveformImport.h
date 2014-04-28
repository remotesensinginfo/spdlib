/*
 *  SPDOptechFullWaveformImport.h
 *  spdlib
 *
 *  Created by Pete Bunting on 28/04/2014.
 *  Copyright 2014 SPDLib. All rights reserved.
 *
 *  Code within this file has been provided by
 *  Steven Hancock for reading the SALCA data
 *  and sorting out the geometry. This has been
 *  adapted and brought across into the SPD
 *  importer interface.
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

#ifndef SPDOptechFullWaveformImport_H
#define SPDOptechFullWaveformImport_H

#include <list>
#include <vector>

#include <boost/cstdint.hpp>
#include "boost/filesystem.hpp"
#include <boost/algorithm/string/trim.hpp>

#include "ogrsf_frmts.h"

#include "spd/SPDFile.h"
#include "spd/SPDPulse.h"
#include "spd/SPDIOException.h"
#include "spd/SPDDataExporter.h"
#include "spd/SPDDataImporter.h"
#include "spd/SPDTextFileLineReader.h"
#include "spd/SPDTextFileUtilities.h"

namespace spdlib
{
	
    class SPDOptechFullWaveformASCIIImport : public SPDDataImporter
	{
	public:
		SPDOptechFullWaveformASCIIImport(bool convertCoords=false, std::string outputProjWKT="", std::string schema="", boost::uint_fast16_t indexCoords=SPD_FIRST_RETURN, bool defineOrigin=false, double originX=0, double originY=0, float originZ=0, float waveNoiseThreshold=0);
		SPDDataImporter* getInstance(bool convertCoords, std::string outputProjWKT,std::string schema,boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold);
        std::list<SPDPulse*>* readAllDataToList(std::string, SPDFile *spdFile)throw(SPDIOException);
		std::vector<SPDPulse*>* readAllDataToVector(std::string inputFile, SPDFile *spdFile)throw(SPDIOException);
		void readAndProcessAllData(std::string inputFile, SPDFile *spdFile, SPDImporterProcessor *processor) throw(SPDIOException);
		bool isFileType(std::string fileType);
        void readHeaderInfo(std::string inputFile, SPDFile *spdFile) throw(SPDIOException);
		~SPDOptechFullWaveformASCIIImport();
    protected:
        void readSPDOPTHeader(std::string inputHDRFile, SPDFile *spdFile, std::string *sensorFile, std::string *waveformsFile)throw(SPDIOException);
        float transWaveformThershold;
        float transWaveformGain;
        float transWaveformOffset;
        float receivedWaveformThershold;
        float receivedWaveformGain;
        float receivedWaveformOffset;
        float laserWavelength;
	};
    
    
}

#endif

