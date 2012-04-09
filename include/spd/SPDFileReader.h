/*
 *  SPDFileReader.h
 *  spdlib_prj
 *
 *  Created by Pete Bunting on 14/09/2009.
 *  Copyright 2009 SPDLib. All rights reserved.
 *
 *  This file is part of SPDLib.
 *
 *  Permission is hereby granted, free of charge, to any person 
 *  obtaining a copy of this software and associated documentation 
 *  files (the "Software"), to deal in the Software without restriction, 
 *  including without limitation the rights to use, copy, modify, 
 *  merge, publish, distribute, sublicense, and/or sell copies of the 
 *  Software, and to permit persons to whom the Software is furnished 
 *  to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be 
 *  included in all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
 *  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
 *  OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
 *  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR 
 *  ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF 
 *  CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION 
 *  WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#ifndef SPDFileReader_H
#define SPDFileReader_H

#include <iostream>
#include <string>
#include <list>

#include <boost/cstdint.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include "spd/SPDFile.h"
#include "spd/SPDDataImporter.h"
#include "spd/SPDPoint.h"
#include "spd/SPDPulse.h"
#include "spd/SPDIOException.h"

using namespace std;
using boost::numeric_cast;
using boost::numeric::bad_numeric_cast;
using boost::numeric::positive_overflow;
using boost::numeric::negative_overflow;

namespace spdlib
{
	class SPDFileReader : public SPDDataImporter
	{
	public:
		SPDFileReader(bool convertCoords=false, string outputProjWKT="", string schema="", boost::uint_fast16_t indexCoords=SPD_FIRST_RETURN, bool defineOrigin=false, double originX=0, double originY=0, float originZ=0, float waveNoiseThreshold=0);
		SPDDataImporter* getInstance(bool convertCoords, string outputProjWKT,string schema, boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold);
        list<SPDPulse*>* readAllDataToList(string, SPDFile *spdFile)throw(SPDIOException);
		vector<SPDPulse*>* readAllDataToVector(string, SPDFile *spdFile)throw(SPDIOException);
		void readAndProcessAllData(string inputFile, SPDFile *spdFile, SPDImporterProcessor *processor) throw(SPDIOException);
        bool isFileType(string fileType);
		void readHeaderInfo(string inputFile, SPDFile *spdFile) throw(SPDIOException);
		~SPDFileReader();	
	private:
		void readRefHeaderRow(H5File *spdInFile,boost::uint_fast32_t row, unsigned long long *binOffsets, unsigned long *numPtsInBin,boost::uint_fast32_t numXBins) throw(SPDIOException);
	};
}

#endif



