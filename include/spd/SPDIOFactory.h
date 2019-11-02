/*
 *  SPDIOFactory.h
 *  spdlib_prj
 *
 *  Created by Pete Bunting on 13/10/2009.
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


#ifndef SPDIOFactory_H
#define SPDIOFactory_H

#include <iostream>
#include <string>
#include <list>

#include "spd/SPDIOException.h"
#include "spd/SPDDataExporter.h"
#include "spd/SPDDataImporter.h"

#include "spd/SPDTextFileImporter.h"
#include "spd/SPDLineParserASCIIPulsePerRow.h"
#include "spd/SPDFileReader.h"
#include "spd/SPDLineParserASCII.h"
#include "spd/SPDFullWaveformDatFileImporter.h"
#include "spd/SPDDecomposedDatFileImporter.h"
#include "spd/SPDARADecomposedDatFileImporter.h"
#include "spd/SPDLASFileImporter.h"
#include "spd/SPDDecomposedCOOFileImporter.h"
#include "spd/SPDASCIIMultiLineReader.h"
#include "spd/SPDImportSALCAData2SPD.h"
#include "spd/SPDOptechFullWaveformImport.h"

#include "spd/SPDFileWriter.h"
#include "spd/SPDGeneralASCIIFileWriter.h"
#include "spd/SPDLASFileExporter.h"

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
	class DllExport SPDIOFactory
	{
	public:
		SPDIOFactory();
		SPDIOFactory(const SPDIOFactory &ioFactory);
		SPDDataExporter* getExporter(std::string filetype, bool exportZasH) ;
		SPDDataImporter* getImporter(std::string filetype, bool convertCoords=false, std::string outputProj4="", std::string schema="", boost::uint_fast16_t indexCoords=SPD_FIRST_RETURN, bool defineOrigin=false, double originX=0, double originY=0, float originZ=0, float waveNoiseThreshold=0) ;
		void registerExporter(SPDDataExporter *exporter);
		void registerImporter(SPDDataImporter *importer);
		SPDIOFactory& operator=(const SPDIOFactory& ioFactory);
		~SPDIOFactory();
	private:
		void registerAll();
		std::list<SPDDataExporter*> *exporters;
		std::list<SPDDataImporter*> *importers;
	};
}

#endif




