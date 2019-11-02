/*
 *  SPDTextFileLineReader.h
 *  spdlib
 *
 *  Created by Pete Bunting on 01/12/2010.
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

#ifndef SPDTextFileLineReader_H
#define SPDTextFileLineReader_H

#include <iostream>
#include <fstream>
#include <string>
#include <list>

#include <boost/algorithm/string/trim.hpp>

#include "spd/SPDIOException.h"

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
	class DllExport SPDTextFileLineReader
	{
	public:
		SPDTextFileLineReader();
		void openFile(std::string filepath);
		bool endOfFile();
		std::string readLine();
		void closeFile();
		~SPDTextFileLineReader();	
	private:
		std::ifstream inputFileStream;
		bool fileOpened;
	};
}

#endif



