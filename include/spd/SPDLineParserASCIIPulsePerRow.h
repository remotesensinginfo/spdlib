 /*
  *  SPDLineParserASCIIPulsePerRow.h
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

#ifndef SPDLineParserASCIIPulsePerRow_H
#define SPDLineParserASCIIPulsePerRow_H

#include <string>
#include <iostream>
#include <list>

#include <boost/cstdint.hpp>

#include "spd/SPDFile.h"
#include "spd/SPDPulse.h"
#include "spd/SPDIOException.h"
#include "spd/SPDTextFileImporter.h"
#include "spd/SPDTextFileUtilities.h"
#include "spd/SPDTextFileException.h"

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
	class DllExport SPDLineParserASCIIPulsePerRow : public SPDTextLineProcessor
	{
	public:
		SPDLineParserASCIIPulsePerRow();
		bool haveReadheader();
		void parseHeader(std::string) ;
		bool parseLine(std::string line, SPDPulse *pl,boost::uint_fast16_t indexCoords) ;
		bool isFileType(std::string fileType);
		void saveHeaderValues(SPDFile *spdFile);
		void reset();
        void parseSchema(std::string schema){};
		~SPDLineParserASCIIPulsePerRow();
	};
	
}

#endif



