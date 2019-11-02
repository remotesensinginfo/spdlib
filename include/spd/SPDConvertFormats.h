/*
 *  SPDConvertFormats.h
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


#ifndef SPDConvertFormats_H
#define SPDConvertFormats_H

#include <iostream>
#include <fstream>
#include <string>
#include <list>

#include "spd/SPDFile.h"
#include "spd/SPDPoint.h"
#include "spd/SPDPulse.h"

#include "spd/SPDGridData.h"
#include "spd/SPDDataImporter.h"
#include "spd/SPDDataExporter.h"
#include "spd/SPDIOUtils.h"
#include "spd/SPDIOFactory.h"
#include "spd/SPDFileIncrementalReader.h"
#include "spd/SPDExportAsReadUnGridded.h"
#include "spd/SPDExportAsTiles.h"

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
	class DllExport SPDConvertFormats
	{
	public:
		SPDConvertFormats();
		void convertInMemory(std::string input, std::string output, std::string inFormat, std::string schema, std::string outFormat, float binsize, std::string inSpatialRef, bool convertCoords, std::string outputProj4,boost::uint_fast16_t indexCoords, bool defineTL, double tlX, double tlY, bool defineOrigin, double originX, double originY, float originZ, bool useSphericIdx, bool usePolarIdx, bool useScanIdx, float waveNoiseThreshold,boost::uint_fast16_t waveformBitRes, boost::uint_fast16_t pointVersion, boost::uint_fast16_t pulseVersion, bool keepInMinExtent, bool exportZasH) ;
		void convertToSPDUsingRowTiles(std::string input, std::string output, std::string inFormat, std::string schema, float binsize, std::string inSpatialRef, bool convertCoords, std::string outputProjWKT,boost::uint_fast16_t indexCoords, std::string tempdir,boost::uint_fast16_t numRowsInTile, bool defineTL, double tlX, double tlY, bool defineOrigin, double originX, double originY, float originZ, bool useSphericIdx, bool usePolarIdx, bool useScanIdx, float waveNoiseThreshold,boost::uint_fast16_t waveformBitRes, bool keepTmpFiles, boost::uint_fast16_t pointVersion, boost::uint_fast16_t pulseVersion, bool keepInMinExtent) ;
		void convertToSPDUsingBlockTiles(std::string input, std::string output, std::string inFormat, std::string schema, float binsize, std::string inSpatialRef, bool convertCoords, std::string outputProjWKT,boost::uint_fast16_t indexCoords, std::string tempdir,boost::uint_fast16_t numRowsInTile, boost::uint_fast16_t numColsInTile, bool defineTL, double tlX, double tlY, bool defineOrigin, double originX, double originY, float originZ, bool useSphericIdx, bool usePolarIdx, bool useScanIdx, float waveNoiseThreshold,boost::uint_fast16_t waveformBitRes, bool keepTmpFiles, boost::uint_fast16_t pointVersion, boost::uint_fast16_t pulseVersion, bool keepInMinExtent) ;
		void copySPD2SPD(SPDFile *inSPDFile, SPDFile *outSPDFile) ;
        ~SPDConvertFormats();
	};
}

#endif

