/*
 *  SPDIOUtils.h
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


#ifndef SPDIOUtils_H
#define SPDIOUtils_H

#include <iostream>
#include <string>
#include <list>

#include "spd/SPDPulse.h"
#include "spd/SPDPoint.h"
#include "spd/SPDFile.h"
#include "spd/SPDIOException.h"
#include "spd/SPDDataExporter.h"
#include "spd/SPDGridData.h"

namespace spdlib
{	
	class DllExport SPDIOUtils
	{
	public:
		SPDIOUtils();
		void gridAndWriteData(SPDDataExporter *exporter, std::list<SPDPulse*> *pls, SPDFile *spdFile, std::string outputFile)throw(SPDIOException);
		void gridAndWriteData(SPDDataExporter *exporter, std::vector<SPDPulse*> *pls, SPDFile *spdFile, std::string outputFile)throw(SPDIOException);
		~SPDIOUtils();
	};
}

#endif

