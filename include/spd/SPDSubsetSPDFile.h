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

#include "spd/SPDProcessDataBlocks.h"
#include "spd/SPDDataBlockProcessor.h"

#include "spd/SPDVectorUtils.h"

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
	class DllExport SPDSubsetSPDFile
	{
	public:
		SPDSubsetSPDFile();
		void subsetSPDFile(std::string inputFile, std::string outputFile, double *bbox, bool *bboxDefined) throw(SPDException);
        void subsetSPDFile2Shp(std::string inputFile, std::string outputFile, std::string shapefile) throw(SPDException);
        void subsetSPDFile2Img(std::string inputFile, std::string outputFile, std::string imgfile, boost::uint_fast32_t blockXSize, boost::uint_fast32_t blockYSize) throw(SPDException);
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
    
    
    
    class DllExport SPDDataSubsetWithImgBlockProcessor: public SPDDataBlockProcessor
	{
	public:
        SPDDataSubsetWithImgBlockProcessor();
        void processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands, float binSize) throw(SPDProcessingException);
		void processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> ***pulses, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, float binSize) throw(SPDProcessingException){throw SPDProcessingException("Not Implemented.");};
        
        void processDataBlockImage(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, float ***imageDataBlock, SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands) throw(SPDProcessingException){throw SPDProcessingException("Not Implemented.");};
		void processDataBlock(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses) throw(SPDProcessingException){throw SPDProcessingException("Not Implemented.");};
        
        std::vector<std::string> getImageBandDescriptions() throw(SPDProcessingException){return std::vector<std::string>();};
        void setHeaderValues(SPDFile *spdFile) throw(SPDProcessingException){};
        ~SPDDataSubsetWithImgBlockProcessor();
	};
}

#endif
