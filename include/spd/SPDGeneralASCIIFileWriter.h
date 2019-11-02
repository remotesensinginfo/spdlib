/*
 *  SPDGeneralASCIIFileWriter.h
 *  SPDLIB
 *
 *  Created by Pete Bunting on 24/01/2011.
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
 *
 */


#ifndef SPDGeneralASCIIFileWriter_H
#define SPDGeneralASCIIFileWriter_H

#include <iostream>
#include <string>
#include <list>
#include <fstream>
#include <stdexcept>

#include "spd/SPDFile.h"
#include "spd/SPDPoint.h"
#include "spd/SPDPulse.h"
#include "spd/SPDIOException.h"
#include "spd/SPDDataExporter.h"
#include "spd/SPDCommon.h"

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
	class DllExport SPDGeneralASCIIFileWriter : public SPDDataExporter
	{
	public:
		SPDGeneralASCIIFileWriter();
		SPDGeneralASCIIFileWriter(const SPDDataExporter &dataExporter) ;
		SPDGeneralASCIIFileWriter(const SPDGeneralASCIIFileWriter &dataExporter) ;
        SPDDataExporter* getInstance();
		bool open(SPDFile *spdFile, std::string outputFile) ;
        bool reopen(SPDFile *spdFile, std::string outputFile) ;
		void writeDataColumn(std::list<SPDPulse*> *pls,boost::uint_fast32_t col,boost::uint_fast32_t row);
		void writeDataColumn(std::vector<SPDPulse*> *pls,boost::uint_fast32_t col,boost::uint_fast32_t row);
		void finaliseClose() ;
		bool requireGrid();
		bool needNumOutPts();
		SPDGeneralASCIIFileWriter& operator=(const SPDGeneralASCIIFileWriter& dataExporter) ;
		~SPDGeneralASCIIFileWriter();
	private:
        std::ofstream *outASCIIFile;
        boost::uint_fast16_t fileType;
	};
}

#endif





