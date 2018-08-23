/*
 *  SPDProcessDataBlocks.h
 *
 *  Created by Pete Bunting on 11/03/2012.
 *  Copyright 2012 RSGISLib. All rights reserved.
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

#ifndef SPDProcessDataBlocks_H
#define SPDProcessDataBlocks_H

#include <iostream>
#include <string>
#include <vector>
#include <math.h>

#include "gdal_priv.h"
#include "ogrsf_frmts.h"
#include "ogr_api.h"

#include <boost/cstdint.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include "spd/SPDFile.h"
#include "spd/SPDPulse.h"
#include "spd/SPDPoint.h"
#include "spd/SPDFileReader.h"
#include "spd/SPDFileIncrementalReader.h"
#include "spd/SPDProcessingException.h"
#include "spd/SPDDataBlockProcessor.h"
#include "spd/SPDFileWriter.h"
#include "spd/SPDGridData.h"

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
	class DllExport SPDProcessDataBlocks
	{
	public:		
		SPDProcessDataBlocks(SPDDataBlockProcessor *dataBlockProcessor, boost::uint_fast32_t overlap=25, boost::uint_fast32_t blockXSize=250, boost::uint_fast32_t blockYSize=250, bool printProgress=true, bool keepMinExtent=true);
		SPDProcessDataBlocks(const SPDProcessDataBlocks &processDataBlock);
		
        void processDataBlocksGridPulsesInputImage(SPDFile *spdInFile, std::string outFile, std::string imageFilePath) throw(SPDProcessingException);
        void processDataBlocksGridPulsesOutputImage(SPDFile *spdInFile, std::string outImagePath, float processingResolution, boost::uint_fast16_t numImgBands, std::string gdalFormat) throw(SPDProcessingException);
        void processDataBlocksGridPulsesOutputImage(SPDFile *spdInFile, GDALDataset *outImageDS) throw(SPDProcessingException);
        void processDataBlocksGridPulsesOutputSPD(SPDFile *spdInFile, std::string outFile, float processingResolution) throw(SPDProcessingException);
        void processDataBlocksGridPulses(SPDFile *spdInFile, float processingResolution) throw(SPDProcessingException);
        
        void processDataBlocksOutputImage(SPDFile *spdInFile, std::string outImagePath, float processingResolution, boost::uint_fast16_t numImgBands, std::string gdalFormat) throw(SPDProcessingException);
        void processDataBlocks(SPDFile *spdInFile) throw(SPDProcessingException);
        
        
		SPDProcessDataBlocks& operator=(const SPDProcessDataBlocks& processDataBlock);
		~SPDProcessDataBlocks();
	protected:
        void removeNullPulses(std::vector<SPDPulse*> ***pulses, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize);
        void clearPulses(std::vector<SPDPulse*> ***pulses, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize);
        void clearPulses(std::vector<SPDPulse*> *pulses);
        void clearPulsesNoDelete(std::vector<SPDPulse*> ***pulses, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize);
        void populateCentrePoints(SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, double xOrigin, double yOrigin, float binRes);
		void populateFromImage(float ***imageDataBlock, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast16_t numImgBands, GDALRasterBand **imgBands, double imgXOrigin, double imgYOrigin, float imgRes, double blockXOrigin, double blockYOrigin)throw(SPDProcessingException);
        void resetImageBlock2Zeros(float ***imageDataBlock, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast16_t numImgBands);
        void writeImageData(GDALRasterBand **imageBands, float ***imageDataBlock, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, boost::uint_fast16_t numImgBands, boost::uint_fast32_t startBinX, boost::uint_fast32_t startBinY, boost::uint_fast32_t startIdxX, boost::uint_fast32_t startIdxY)throw(SPDProcessingException);
        
        SPDDataBlockProcessor *dataBlockProcessor;
        boost::uint_fast32_t overlap;
        boost::uint_fast32_t blockXSize;
        boost::uint_fast32_t blockYSize;
		bool printProgress;
        bool keepMinExtent;
	};
}

#endif

