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
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/algorithm/string/trim.hpp>

#include "spd/SPDTextFileException.h"
#include "spd/SPDIOException.h"

namespace spdlib{
	
	class SPDTextFileUtilities
	{
	public:
		SPDTextFileUtilities();
        boost::uint_fast64_t countLines(std::string input) throw(SPDIOException);
		bool lineStart(std::string line, char token);
		bool blankline(std::string line);
		std::string removeWhiteSpace(std::string line);
        std::string removeChar(std::string line, char val);
		void tokenizeString(std::string line, char token, std::vector<std::string> *tokens, bool ignoreDuplicateTokens=true);
		std::string readFileToString(std::string input) throw(SPDIOException);
        std::vector<std::string> readFileLinesToVector(std::string input) throw(SPDIOException);
        bool isNumber(char val);
        bool lineStartWithHash(std::string line) throw(SPDIOException);
        bool lineContainsChar(std::string line, char val) throw(SPDIOException);
		
		double strtodouble(std::string inValue)throw(SPDTextFileException);
		float strtofloat(std::string inValue)throw(SPDTextFileException);
		
        boost::uint_fast8_t strto8bitUInt(std::string inValue)throw(SPDTextFileException);
        boost::uint_fast16_t strto16bitUInt(std::string inValue)throw(SPDTextFileException);
        boost::uint_fast32_t strto32bitUInt(std::string inValue)throw(SPDTextFileException);
        boost::uint_fast64_t strto64bitUInt(std::string inValue)throw(SPDTextFileException);
            
        boost::int_fast8_t strto8bitInt(std::string inValue)throw(SPDTextFileException);
        boost::int_fast16_t strto16bitInt(std::string inValue)throw(SPDTextFileException);
        boost::int_fast32_t strto32bitInt(std::string inValue)throw(SPDTextFileException);
        boost::int_fast64_t strto64bitInt(std::string inValue)throw(SPDTextFileException);
		
		std::string doubletostring(double number)throw(SPDTextFileException);
		std::string floattostring(float number)throw(SPDTextFileException);
		
		std::string uInt8bittostring(boost::uint_fast8_t number)throw(SPDTextFileException);
		std::string uInt16bittostring(boost::uint_fast16_t number)throw(SPDTextFileException);
		std::string uInt32bittostring(boost::uint_fast32_t number)throw(SPDTextFileException);
		std::string uInt64bittostring(boost::uint_fast64_t number)throw(SPDTextFileException);
		
		std::string int8bittostring(boost::int_fast8_t number)throw(SPDTextFileException);
		std::string int16bittostring(boost::int_fast16_t number)throw(SPDTextFileException);
		std::string int32bittostring(boost::int_fast32_t number)throw(SPDTextFileException);
		std::string int64bittostring(boost::int_fast64_t number)throw(SPDTextFileException);
		
		~SPDTextFileUtilities();
	};
}

#endif


