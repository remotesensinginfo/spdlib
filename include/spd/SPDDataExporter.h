/*
 *  SPDDataExporter.h
 *  spdlib_prj
 *
 *  Created by Pete Bunting on 28/09/2009.
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
 */


#ifndef SPDDataExporter_H
#define SPDDataExporter_H

#include <iostream>
#include <fstream>
#include <string>
#include <list>

#include <boost/cstdint.hpp>

#include "spd/SPDFile.h"
#include "spd/SPDPulse.h"
#include "spd/SPDIOException.h"
#include "spd/SPDDataImporter.h"

using namespace std;

namespace spdlib
{
	class SPDDataExporter
	{
	public:
		SPDDataExporter(string filetype);
		SPDDataExporter(const SPDDataExporter &dataExporter) throw(SPDException);
        virtual SPDDataExporter* getInstance()=0;
		virtual bool open(SPDFile *spdFile, string outputFile) throw(SPDIOException) = 0;
		virtual void writeDataColumn(list<SPDPulse*> *pls, boost::uint_fast32_t col, boost::uint_fast32_t row)throw(SPDIOException) = 0;
		virtual void writeDataColumn(vector<SPDPulse*> *pls, boost::uint_fast32_t col, boost::uint_fast32_t row)throw(SPDIOException) = 0;
		virtual void writeData(list<SPDPulse*> ***griddedPls, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize)throw(SPDIOException);
		virtual void writeData(vector<SPDPulse*> ***griddedPls, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize)throw(SPDIOException);
        virtual void writeData(list<SPDPulse*> ***griddedPls, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t startBinX, boost::uint_fast32_t startBinY, boost::uint_fast32_t startIdxX, boost::uint_fast32_t startIdxY)throw(SPDIOException);
		virtual void writeData(vector<SPDPulse*> ***griddedPls, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t startBinX, boost::uint_fast32_t startBinY, boost::uint_fast32_t startIdxX, boost::uint_fast32_t startIdxY)throw(SPDIOException);
		virtual void finaliseClose() throw(SPDIOException) = 0;
		virtual bool needNumOutPts()=0;
		virtual void setNumOutPts(boost::uint_fast64_t numOutPts);
		virtual bool isFileType(string filetype);
		virtual bool requireGrid()=0;
		SPDDataExporter& operator=(const SPDDataExporter& dataExporter) throw(SPDException);
		virtual bool opened();
		virtual ~SPDDataExporter();
	protected:
		SPDFile *spdFile;
		string outputFile;
		bool fileOpened;
		string filetype;
	boost::uint_fast64_t numOutPts;
		bool numOutPtsDefined;
	};
}

#endif



