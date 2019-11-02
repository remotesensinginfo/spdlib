/*
 *  SPDExportAsReadUnGridded.h
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
 */

#ifndef SPDExportAsReadUnGridded_H
#define SPDExportAsReadUnGridded_H

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
	class DllExport SPDExportAsReadUnGridded : public SPDImporterProcessor
	{
	public:
		SPDExportAsReadUnGridded(SPDDataExporter *exporter, SPDFile *spdFileOut, bool defineSource=false, boost::uint_fast16_t sourceID=0, bool defineReturnID=false, boost::uint_fast16_t returnID=0, bool defineClasses=false, boost::uint_fast16_t classValue=0) ;
		void processImportedPulse(SPDFile *spdFile, SPDPulse *pulse) ;
		void completeFileAndClose(SPDFile *spdFile);
        void setSourceID(boost::uint_fast16_t sourceID);
        void setReturnID(boost::uint_fast16_t returnID);
        void setClassValue(boost::uint_fast16_t classValue);
		~SPDExportAsReadUnGridded();
	private:
		SPDDataExporter *exporter;
		SPDFile *spdFileOut;
        bool defineSource;
        boost::uint_fast16_t sourceID;
        bool defineReturnID;
        boost::uint_fast16_t returnID;
        bool defineClasses;
        boost::uint_fast16_t classValue;
		bool fileOpen;
		std::vector<SPDPulse*> *pulses;
	};
}

#endif

