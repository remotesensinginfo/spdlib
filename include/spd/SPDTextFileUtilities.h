/*
 *  SPDTextFileUtilities.h
 *  spdlib_prj
 *
 *  Created by Pete Bunting on 09/10/2009.
 *  Copyright 2009 SPDLib. All rights reserved.
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

#ifndef SPDTextFileUtilities_H
#define SPDTextFileUtilities_H

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/cstdint.hpp>
#include <boost/algorithm/string/trim.hpp>

#include "spd/SPDTextFileException.h"
#include "spd/SPDIOException.h"

using boost::lexical_cast;
using boost::bad_lexical_cast;

using namespace std;

namespace spdlib{
	
	class SPDTextFileUtilities
	{
	public:
		SPDTextFileUtilities();
        boost::uint_fast64_t countLines(string input) throw(SPDIOException);
		bool lineStart(string line, char token);
		bool blankline(string line);
		string removeWhiteSpace(string line);
		void tokenizeString(string line, char token, vector<string> *tokens, bool ignoreDuplicateTokens=true);
		string readFileToString(string input) throw(SPDIOException);
		
		double strtodouble(string inValue)throw(SPDTextFileException);
		float strtofloat(string inValue)throw(SPDTextFileException);
		
        boost::uint_fast8_t strto8bitUInt(string inValue)throw(SPDTextFileException);
        boost::uint_fast16_t strto16bitUInt(string inValue)throw(SPDTextFileException);
        boost::uint_fast32_t strto32bitUInt(string inValue)throw(SPDTextFileException);
        boost::uint_fast64_t strto64bitUInt(string inValue)throw(SPDTextFileException);
            
        boost::int_fast8_t strto8bitInt(string inValue)throw(SPDTextFileException);
        boost::int_fast16_t strto16bitInt(string inValue)throw(SPDTextFileException);
        boost::int_fast32_t strto32bitInt(string inValue)throw(SPDTextFileException);
        boost::int_fast64_t strto64bitInt(string inValue)throw(SPDTextFileException);
		
		string doubletostring(double number)throw(SPDTextFileException);
		string floattostring(float number)throw(SPDTextFileException);
		
		string uInt8bittostring(boost::uint_fast8_t number)throw(SPDTextFileException);
		string uInt16bittostring(boost::uint_fast16_t number)throw(SPDTextFileException);
		string uInt32bittostring(boost::uint_fast32_t number)throw(SPDTextFileException);
		string uInt64bittostring(boost::uint_fast64_t number)throw(SPDTextFileException);
		
		string int8bittostring(boost::int_fast8_t number)throw(SPDTextFileException);
		string int16bittostring(boost::int_fast16_t number)throw(SPDTextFileException);
		string int32bittostring(boost::int_fast32_t number)throw(SPDTextFileException);
		string int64bittostring(boost::int_fast64_t number)throw(SPDTextFileException);
		
		~SPDTextFileUtilities();
	};
}

#endif


