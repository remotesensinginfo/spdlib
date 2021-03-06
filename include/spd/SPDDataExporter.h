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
#include <vector>

#include <boost/cstdint.hpp>

#include "spd/SPDFile.h"
#include "spd/SPDPulse.h"
#include "spd/SPDIOException.h"
#include "spd/SPDDataImporter.h"

// mark all exported classes/functions with DllExport to have
// them exported by Visual Studio
#undef DllExport
#ifdef _MSC_VER
    #ifdef libspdio_EXPORTS
        #define DllExport   __declspec( dllexport )
    #else
        #define DllExport   __declspec( dllimport )
    #endif
#else
    #define DllExport
#endif

namespace spdlib
{
	class DllExport SPDDataExporter
	{
	public:
		SPDDataExporter(std::string filetype);
		SPDDataExporter(const SPDDataExporter &dataExporter) ;
        virtual SPDDataExporter* getInstance()=0;
		virtual bool open(SPDFile *spdFile, std::string outputFile)  = 0;
        virtual bool reopen(SPDFile *spdFile, std::string outputFile)  = 0;
		virtual void writeDataColumn(std::list<SPDPulse*> *pls, boost::uint_fast32_t col, boost::uint_fast32_t row) = 0;
		virtual void writeDataColumn(std::vector<SPDPulse*> *pls, boost::uint_fast32_t col, boost::uint_fast32_t row) = 0;
		virtual void writeData(std::list<SPDPulse*> ***griddedPls, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize);
		virtual void writeData(std::vector<SPDPulse*> ***griddedPls, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize);
        virtual void writeData(std::list<SPDPulse*> ***griddedPls, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t startBinX, boost::uint_fast32_t startBinY, boost::uint_fast32_t startIdxX, boost::uint_fast32_t startIdxY);
		virtual void writeData(std::vector<SPDPulse*> ***griddedPls, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t startBinX, boost::uint_fast32_t startBinY, boost::uint_fast32_t startIdxX, boost::uint_fast32_t startIdxY);
		virtual void finaliseClose()  = 0;
		virtual bool needNumOutPts()=0;
        virtual void setKeepMinExtent(bool keepMinExtent){this->keepMinExtent = keepMinExtent;};
		virtual void setNumOutPts(boost::uint_fast64_t numOutPts);
		virtual bool isFileType(std::string filetype);
		virtual bool requireGrid()=0;
        virtual void setExportZasH(bool exportZasH){this->exportZasH = exportZasH;};
        SPDDataExporter& operator=(const SPDDataExporter& dataExporter) ;
		virtual bool opened();
		virtual ~SPDDataExporter();
	protected:
		SPDFile *spdFile;
		std::string outputFile;
		bool fileOpened;
		std::string filetype;
        boost::uint_fast64_t numOutPts;
		bool numOutPtsDefined;
        bool keepMinExtent;
        bool exportZasH;
	};
}

#endif



