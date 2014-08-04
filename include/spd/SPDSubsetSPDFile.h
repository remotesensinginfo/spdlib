 /*
  *  SPDSubsetSPDFile.h
  *  SPDLIB
  *
  *  Created by Pete Bunting on 28/12/2010.
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


#ifndef SPDSubsetSPDFile_H
#define SPDSubsetSPDFile_H

#include <iostream>
#include <fstream>
#include <string>
#include <list>

#include "ogrsf_frmts.h"
#include "ogr_api.h"

#include <boost/cstdint.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include "spd/SPDFile.h"
#include "spd/SPDPoint.h"
#include "spd/SPDPulse.h"

#include "spd/SPDFileReader.h"
#include "spd/SPDFileWriter.h"
#include "spd/SPDFileIncrementalReader.h"

#include "spd/SPDVectorUtils.h"

namespace spdlib
{	
	class DllExport SPDSubsetSPDFile
	{
	public:
		SPDSubsetSPDFile();
		void subsetSPDFile(std::string inputFile, std::string outputFile, double *bbox, bool *bboxDefined) throw(SPDException);
        void subsetSPDFile(std::string inputFile, std::string outputFile, std::string shapefile) throw(SPDException);
        void subsetSPDFileHeightOnly(std::string inputFile, std::string outputFile, double lowHeight, double upperHeight) throw(SPDException);
        void subsetSphericalSPDFile(std::string inputFile, std::string outputFile, double *bbox, bool *bboxDefined) throw(SPDException);
        void subsetScanSPDFile(std::string inputFile, std::string outputFile, double *bbox, bool *bboxDefined) throw(SPDException);
		~SPDSubsetSPDFile();
	};
    
    class DllExport SPDUPDPulseSubset
	{
	public:
		SPDUPDPulseSubset();
		void subsetUPD(std::string inputFile, std::string outputFile, boost::uint_fast32_t startPulse, boost::uint_fast32_t numOfPulses)throw(SPDIOException);
		~SPDUPDPulseSubset();
	};
}

#endif
