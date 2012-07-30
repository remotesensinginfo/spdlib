 /*
  *  SPDExportProcessorSubset.h
  *  SPDLIB
  *
  *  Created by Pete Bunting on 19/12/2010.
  *  Copyright 2010 RSGISLib. All rights reserved.
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

#ifndef SPDExportProcessorSubset_H
#define SPDExportProcessorSubset_H

#include <list>

#include <boost/cstdint.hpp>

#include "ogrsf_frmts.h"

#include "spd/SPDFile.h"
#include "spd/SPDPulse.h"
#include "spd/SPDIOException.h"
#include "spd/SPDDataExporter.h"
#include "spd/SPDFileWriter.h"
#include "spd/SPDFileReader.h"

namespace spdlib
{
	class SPDExportProcessorSubset : public SPDImporterProcessor
	{
	public:
		SPDExportProcessorSubset(SPDDataExporter *exporter, SPDFile *spdFileOut, double *bbox) throw(SPDException);
		void processImportedPulse(SPDFile *spdFile, SPDPulse *pulse) throw(SPDIOException);
		void completeFileAndClose(SPDFile *spdFile)throw(SPDIOException);
		~SPDExportProcessorSubset();
	private:
		SPDDataExporter *exporter;
		SPDFile *spdFileOut;
		bool fileOpen;
		std::list<SPDPulse*> *pulses;
		double *bbox;
		double xMin;
		double xMax;
		double yMin;
		double yMax;
		double zMin;
		double zMax;
		bool first;
	};

	class SPDExportProcessorSubsetSpherical : public SPDImporterProcessor
	{
	public:
		SPDExportProcessorSubsetSpherical(SPDDataExporter *exporter, SPDFile *spdFileOut, double *bbox) throw(SPDException);
		void processImportedPulse(SPDFile *spdFile, SPDPulse *pulse) throw(SPDIOException);
		void completeFileAndClose(SPDFile *spdFile)throw(SPDIOException);
		~SPDExportProcessorSubsetSpherical();
	private:
		SPDDataExporter *exporter;
		SPDFile *spdFileOut;
		bool fileOpen;
		std::list<SPDPulse*> *pulses;
		double *bbox;
	};

    
	class SPDSubsetNonGriddedFile
	{
	public:
		SPDSubsetNonGriddedFile();
		void subsetCartesian(std::string input, std::string output, double *bbox, bool *bboxDefined) throw(SPDException);
        void subsetSpherical(std::string input, std::string output, double *bbox, bool *bboxDefined) throw(SPDException);
		~SPDSubsetNonGriddedFile();
	};
}

#endif



