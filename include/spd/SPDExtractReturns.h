/*
 *  SPDExtractReturns.h
 *  SPDLIB
 *
 *  Created by Pete Bunting on 15/11/2012.
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

#ifndef SPDExtractReturns_H
#define SPDExtractReturns_H

#include <boost/cstdint.hpp>

#include "ogrsf_frmts.h"

#include "spd/SPDFile.h"
#include "spd/SPDPulse.h"
#include "spd/SPDIOException.h"
#include "spd/SPDDataExporter.h"
#include "spd/SPDFileWriter.h"
#include "spd/SPDProcessPulses.h"
#include "spd/SPDPulseProcessor.h"
#include "spd/SPDProcessingException.h"

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
	class DllExport SPDExtractReturnsImportProcess : public SPDImporterProcessor
	{
	public:
		SPDExtractReturnsImportProcess(std::string outputFilePath, bool classValSet, boost::uint_fast16_t classID, bool returnValSet, boost::uint_fast16_t returnVal) throw(SPDException);
		void processImportedPulse(SPDFile *spdFile, SPDPulse *pulse) throw(SPDIOException);
		void completeFileAndClose(SPDFile *spdFile)throw(SPDIOException);
		~SPDExtractReturnsImportProcess();
	private:
        SPDDataExporter *exporter;
        SPDFile *outSPDFile;
        bool classValSet;
        boost::uint_fast16_t classID;
        bool returnValSet;
        boost::uint_fast16_t returnVal;
        std::vector<SPDPulse*> *pulses;
	};


    class DllExport SPDExtractReturnsBlockProcess : public SPDPulseProcessor
	{
	public:
        SPDExtractReturnsBlockProcess(bool classValSet, boost::uint_fast16_t classID, bool returnValSet, boost::uint_fast16_t returnVal, bool minMaxSet, boost::uint_fast16_t highOrLow);

        void processDataColumnImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float *imageData, SPDXYPoint *cenPts, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing is not implemented for processDataColumn().");};

		void processDataColumn(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, SPDXYPoint *cenPts) throw(SPDProcessingException);

        void processDataWindowImage(SPDFile *inSPDFile, bool **validBins, std::vector<SPDPulse*> ***pulses, float ***imageData, SPDXYPoint ***cenPts, boost::uint_fast32_t numImgBands, float binSize, boost::uint_fast16_t winSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing using a window is not implemented.");};
		void processDataWindow(SPDFile *inSPDFile, bool **validBins, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast16_t winSize) throw(SPDProcessingException)
        {throw SPDProcessingException("Processing using a window is not implemented.");};

        std::vector<std::string> getImageBandDescriptions() throw(SPDProcessingException){return std::vector<std::string>();};
        void setHeaderValues(SPDFile *spdFile) throw(SPDProcessingException){};

        ~SPDExtractReturnsBlockProcess();
    protected:
        bool classValSet;
        boost::uint_fast16_t classID;
        bool returnValSet;
        boost::uint_fast16_t returnVal;
        bool minMaxSet;
        boost::uint_fast16_t highOrLow;
	};

}

#endif







