/*
 *  SPDLASFileExporter.h
 *  SPDLIB
 *
 *  Created by Pete Bunting on 17/02/2011.
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


#ifndef SPDLASFileExporter_H
#define SPDLASFileExporter_H

#include <fstream>
#include <iostream>
#include <string>
#include <list>

#include <liblas/liblas.hpp>

#include "spd/SPDFile.h"
#include "spd/SPDPoint.h"
#include "spd/SPDPulse.h"
#include "spd/SPDIOException.h"
#include "spd/SPDDataExporter.h"
#include "spd/SPDCommon.h"

namespace spdlib
{
	class SPDLASFileExporter : public SPDDataExporter
	{
	public:
		SPDLASFileExporter();
		SPDLASFileExporter(const SPDDataExporter &dataExporter) throw(SPDException);
		SPDLASFileExporter(const SPDLASFileExporter &dataExporter) throw(SPDException);
        SPDDataExporter* getInstance();
		bool open(SPDFile *spdFile, std::string outputFile) throw(SPDIOException);
		void writeDataColumn(std::list<SPDPulse*> *pls,boost::uint_fast32_t col,boost::uint_fast32_t row)throw(SPDIOException);
		void writeDataColumn(std::vector<SPDPulse*> *pls,boost::uint_fast32_t col,boost::uint_fast32_t row)throw(SPDIOException);
		void finaliseClose() throw(SPDIOException);
		bool requireGrid();
		bool needNumOutPts();
		SPDLASFileExporter& operator=(const SPDLASFileExporter& dataExporter) throw(SPDException);
		~SPDLASFileExporter();
	private:
		std::fstream *outDataStream;
		liblas::Writer *lasWriter;
        bool finalisedClosed;
	};
}

#endif



