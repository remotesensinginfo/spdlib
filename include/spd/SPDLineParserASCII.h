/*
 *  SPDLineParserASCII.h
 *
 *  Created by Pete Bunting on 21/03/2012.
 *  Copyright 2012 SPDLib. All rights reserved.
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

#ifndef SPDLineParserASCII_H
#define SPDLineParserASCII_H

#include <string>
#include <iostream>
#include <list>
#include <math.h>
#include <stdexcept>
#include <vector>

#include <boost/cstdint.hpp>

#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>

#include "spd/SPDCommon.h"
#include "spd/SPDFile.h"
#include "spd/SPDPulse.h"
#include "spd/SPDIOException.h"
#include "spd/SPDTextFileImporter.h"
#include "spd/SPDTextFileUtilities.h"
#include "spd/SPDTextFileException.h"

namespace spdlib
{
	class DllExport SPDLineParserASCII : public SPDTextLineProcessor
	{
        struct ASCIIField
        {
            ASCIIField(){};
            ASCIIField(std::string name, SPDDataType dataType, boost::uint_fast16_t idx)
            {
                this->name = name;
                this->dataType = dataType;
                this->idx = idx;
            };
            std::string name;
            SPDDataType dataType;
            boost::uint_fast16_t idx;
        };
        
        
	public:
		SPDLineParserASCII();
		bool haveReadheader();
		void parseHeader(std::string) throw(SPDIOException);
		bool parseLine(std::string line, SPDPulse *pl,boost::uint_fast16_t) throw(SPDIOException);
		bool isFileType(std::string fileType);
		void saveHeaderValues(SPDFile *spdFile);
		void reset();
        void parseSchema(std::string schema)throw(SPDIOException);
		~SPDLineParserASCII();
	private:
        boost::uint_fast16_t numLinesIgnore;
        char commentChar;
        char delimiter; 
        std::vector<ASCIIField> fields;
        boost::uint_fast16_t sourceID;
        boost::uint_fast64_t ptCount;
        boost::uint_fast64_t lineCount;
        bool rgbValuesFound;
	};
	
}

#endif

