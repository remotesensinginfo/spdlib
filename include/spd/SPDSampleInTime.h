/*
 *  SPDSampleInTime.h
 *  SPDLIB
 *
 *  Created by Pete Bunting on 31/10/2012.
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


#ifndef SPDSampleInTime_H
#define SPDSampleInTime_H

#include <boost/cstdint.hpp>

#include "ogrsf_frmts.h"

#include "spd/SPDFile.h"
#include "spd/SPDPulse.h"
#include "spd/SPDIOException.h"
#include "spd/SPDDataExporter.h"
#include "spd/SPDFileWriter.h"

namespace spdlib
{
	class SPDSampleInTime : public SPDImporterProcessor
	{
	public:
		SPDSampleInTime(std::string outputFilePath, boost::uint_fast16_t tSample) throw(SPDException);
		void processImportedPulse(SPDFile *spdFile, SPDPulse *pulse) throw(SPDIOException);
		void completeFileAndClose()throw(SPDIOException);
		~SPDSampleInTime();
	private:
        SPDDataExporter *exporter;
        SPDFile *outSPDFile;
        boost::uint_fast16_t tSample;
        boost::uint_fast64_t pulseCount;
        boost::uint_fast64_t pulseCountOut;
        std::vector<SPDPulse*> *pulses;
	};

}

#endif


