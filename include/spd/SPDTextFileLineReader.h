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

using namespace std;

namespace spdlib
{
	class SPDTextFileLineReader
	{
	public:
		SPDTextFileLineReader();
		void openFile(string filepath)throw(SPDIOException);
		bool endOfFile();
		string readLine()throw(SPDIOException);
		void closeFile()throw(SPDIOException);
		~SPDTextFileLineReader();	
	private:
		ifstream inputFileStream;
		bool fileOpened;
	};
}

#endif



