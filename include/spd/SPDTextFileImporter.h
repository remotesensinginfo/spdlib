 /*
  *  SPDTextFileImporter.h
  *  spdlib
  *
  *  Created by Pete Bunting on 28/11/2010.
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

#ifndef SPDTextFileImporter_H
#define SPDTextFileImporter_H

#include <string>
#include <iostream>
#include <fstream>
#include <list>

#include <boost/algorithm/string/trim.hpp>
#include <boost/cstdint.hpp>

#include "ogrsf_frmts.h"

#include "spd/SPDFile.h"
#include "spd/SPDPulse.h"
#include "spd/SPDIOException.h"
#include "spd/SPDDataImporter.h"
#include "spd/SPDTextFileUtilities.h"

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
	class DllExport SPDTextLineProcessor
	{
	public:
		SPDTextLineProcessor():headerRead(false){};
		virtual bool haveReadheader()=0;
		virtual void parseHeader(std::string strLine) =0;
		virtual bool parseLine(std::string line, SPDPulse *pl,boost::uint_fast16_t indexCoords) =0;
		virtual bool isFileType(std::string fileType)=0;
		virtual void saveHeaderValues(SPDFile *spdFile)=0;
		virtual void reset()=0;
        virtual void parseSchema(std::string schema)=0;
		virtual ~SPDTextLineProcessor(){};
	protected:
		bool headerRead;
	};
	
	
	class DllExport SPDTextFileImporter : public SPDDataImporter
	{
	public:
		SPDTextFileImporter(SPDTextLineProcessor *lineParser, bool convertCoords=false, std::string outputProjWKT="", std::string schema="", boost::uint_fast16_t indexCoords=SPD_FIRST_RETURN, bool defineOrigin=false, double originX=0, double originY=0, float originZ=0, float waveNoiseThreshold=0);
		SPDTextFileImporter(const SPDTextFileImporter &textFileImporter);
		SPDDataImporter* getInstance(bool convertCoords, std::string outputProjWKT, std::string schema, boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold);
        std::list<SPDPulse*>* readAllDataToList(std::string, SPDFile *spdFile);
		std::vector<SPDPulse*>* readAllDataToVector(std::string inputFile, SPDFile *spdFile);
		void readAndProcessAllData(std::string inputFile, SPDFile *spdFile, SPDImporterProcessor *processor) ;
        void readHeaderInfo(std::string inputFile, SPDFile *spdFile) ;
        void readSchema();
		bool isFileType(std::string fileType);
		SPDTextFileImporter& operator=(const SPDTextFileImporter& textFileImporter);
		~SPDTextFileImporter();
	private:
		SPDTextLineProcessor *lineParser;
	};
}

#endif

