 /*
  *  SPDExportAsTiles.h
  *  spdlib
  *
  *  Created by Pete Bunting on 04/12/2010.
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

#ifndef SPDExportAsTiles_H
#define SPDExportAsTiles_H

#include <list>

#include <boost/cstdint.hpp>

#include "ogrsf_frmts.h"

#include "spd/SPDFile.h"
#include "spd/SPDPulse.h"
#include "spd/SPDIOException.h"
#include "spd/SPDDataExporter.h"
#include "spd/SPDDataImporter.h"

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
	struct DllExport PointDataTileFile
	{
		SPDDataExporter *exporter;
        std::list<SPDPulse*> *pulses;
		OGREnvelope *env;
		SPDFile *spdFile;
	};
	
	class DllExport SPDExportAsRowTiles : public SPDImporterProcessor
	{
	public:
		SPDExportAsRowTiles(PointDataTileFile *tiles,boost::uint_fast32_t numOfTiles, SPDFile *overallSPD, double tileHeight, bool useSphericIdx, bool useScanIdx) throw(SPDException);
		void processImportedPulse(SPDFile *spdFile, SPDPulse *pulse) throw(SPDIOException);
		void completeFileAndClose()throw(SPDIOException);
		~SPDExportAsRowTiles();
	private:
		PointDataTileFile *tiles;
        boost::uint_fast32_t numOfTiles;
		SPDFile *overallSPD;
		double tileHeight;
        bool useSphericIdx;
        bool useScanIdx;
		bool filesOpen;
	};


    class DllExport SPDExportAsBlockTiles : public SPDImporterProcessor
	{
	public:
		SPDExportAsBlockTiles(PointDataTileFile *tiles, boost::uint_fast32_t numOfTiles, boost::uint_fast32_t numOfXTiles, boost::uint_fast32_t numOfYTiles, SPDFile *overallSPD, double tileHeight, double tileWidth, bool useSphericIdx, bool useScanIdx) throw(SPDException);
		void processImportedPulse(SPDFile *spdFile, SPDPulse *pulse) throw(SPDIOException);
		void completeFileAndClose()throw(SPDIOException);
		~SPDExportAsBlockTiles();
	private:
		PointDataTileFile *tiles;
        boost::uint_fast32_t numOfTiles;
        boost::uint_fast32_t numOfXTiles;
        boost::uint_fast32_t numOfYTiles;
		SPDFile *overallSPD;
		double tileHeight;
        double tileWidth;
        bool useSphericIdx;
        bool useScanIdx;
		bool filesOpen;
	};
}

#endif


