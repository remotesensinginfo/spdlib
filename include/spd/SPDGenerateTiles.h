/*
 *  SPDGenerateTiles.h
 *  SPDLIB
 *
 *  Created by Pete Bunting on 18/04/2013.
 *  Copyright 2013 SPDLib. All rights reserved.
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

#ifndef SPDGenerateTiles_H
#define SPDGenerateTiles_H

#include <boost/cstdint.hpp>
#include "boost/filesystem.hpp"

#include "ogrsf_frmts.h"
#include "gdal_priv.h"
#include "gdal_rat.h"

#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/util/OutOfMemoryException.hpp>
#include <xercesc/sax/HandlerBase.hpp>

#include "spd/SPDFile.h"
#include "spd/SPDPulse.h"
#include "spd/SPDIOException.h"
#include "spd/SPDDataExporter.h"
#include "spd/SPDFileWriter.h"
#include "spd/SPDProcessPulses.h"
#include "spd/SPDPulseProcessor.h"
#include "spd/SPDProcessingException.h"
#include "spd/SPDTextFileUtilities.h"
#include "spd/SPDMathsUtils.h"
#include "spd/SPDImageUtils.h"
#include "spd/SPDFileUtilities.h"
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
    struct DllExport SPDTile
    {
        boost::uint_fast16_t row;
        boost::uint_fast16_t col;
        double xMinCore;
        double xMaxCore;
        double yMinCore;
        double yMaxCore;
        double xMin;
        double xMax;
        double yMin;
        double yMax;
        std::string outFileName;
        SPDFile *spdFile;
        SPDNoIdxFileWriter *writer;
        bool writerOpen;
    };
    
    
	class DllExport SPDTilesUtils
	{
	public:
		SPDTilesUtils() ;
        void calcFileExtent(std::vector<std::string> inputFiles, double *xMin, double *xMax, double *yMin, double *yMax) ;
        std::vector<SPDTile*>* createTiles(double xSize, double ySize, double overlap, double xMin, double xMax, double yMin, double yMax, boost::uint_fast32_t *rows, boost::uint_fast32_t *cols);
		void exportTiles2XML(std::string outputFile, std::vector<SPDTile*> *tiles, double xSize, double ySize, double overlap, double xMin, double xMax, double yMin, double yMax, boost::uint_fast32_t rows, boost::uint_fast32_t cols);
        std::vector<SPDTile*>* importTilesFromXML(std::string inputFile, boost::uint_fast32_t *rows, boost::uint_fast32_t *cols, double *xSize, double *ySize, double *overlap, double *xMin, double *xMax, double *yMin, double *yMax);
        void findNumRowsColsFromXML(std::string inputFile, boost::uint_fast32_t *rows, boost::uint_fast32_t *cols);
        void deleteTiles(std::vector<SPDTile*> *tiles);
        void printTiles2Console(std::vector<SPDTile*> *tiles);
        void createTileSPDFiles(std::vector<SPDTile*> *tiles, SPDFile *templateSPDFile, std::string outputBase, double xSize, double ySize, double overlap, double xMin, double xMax, double yMin, double yMax, boost::uint_fast32_t rows, boost::uint_fast32_t cols, bool checkTemplateSPDExtent=false) ;
        void createTileSPDFilesDirStruct(std::vector<SPDTile*> *tiles, SPDFile *templateSPDFile, std::string outputBase, bool usePrefix, std::string prefix, double xSize, double ySize, double overlap, double xMin, double xMax, double yMin, double yMax, boost::uint_fast32_t rows, boost::uint_fast32_t cols, bool checkTemplateSPDExtent=false) ;
        void populateTilesWithData(std::vector<SPDTile*> *tiles, std::string inputSPDFile) ;
        void populateTilesWithData(std::vector<SPDTile*> *tiles, std::vector<std::string> inputFiles) ;
        void populateTileWithData(SPDTile *tile, std::vector<std::string> inputFiles) ;
        void deleteTilesWithNoPulses(std::vector<SPDTile*> *tiles) ;
        void deleteTileIfNoPulses(std::vector<SPDTile*> *tiles, boost::uint_fast32_t row, boost::uint_fast32_t col) ;
        GDALDataset* createNewImageFile(std::string imageFile, std::string format, GDALDataType dataType, std::string wktFile, double xRes, double yRes, double tlX, double tlY, boost::uint_fast32_t xImgSize, boost::uint_fast32_t yImgSize, boost::uint_fast32_t numBands, double backgroundVal) ;
        void addImageTiles(GDALDataset *image, std::vector<SPDTile*> *tiles, std::vector<std::string> inputImageFiles) ;
        void addImageTilesParseFileName(GDALDataset *image, std::vector<SPDTile*> *tiles, std::vector<std::string> inputImageFiles) ;
        void addTiles2ClumpImage(GDALDataset *image, std::vector<SPDTile*> *tiles) ;
        void extractRowColFromFileName(std::string filePathStr, boost::uint_fast32_t *row, boost::uint_fast32_t *col) ;
        void extractTileCore(std::string inputSPDFile, std::string outputSPDFile, boost::uint_fast32_t row, boost::uint_fast32_t col, std::vector<SPDTile*> *tiles) ;
        void buildDirectoryStruct(std::string outputBase, boost::uint_fast32_t rows, boost::uint_fast32_t cols) ;
        void removeDirectoriesWithNoTiles(std::vector<SPDTile*> *tiles, std::string outputBase, boost::uint_fast32_t rows, boost::uint_fast32_t cols) ;
        void removeTilesNotInFilesList(std::vector<SPDTile*> *tiles, std::vector<std::string> inputFiles) ;
        void generateTileCoresShpFile(std::vector<SPDTile*> *tiles, std::string shpFile, std::string projWKTStr, bool deleteShp) ;
        ~SPDTilesUtils();
	};
    
    
    
    class DllExport SPDWrite2OverlapTiles : public SPDImporterProcessor
	{
	public:
		SPDWrite2OverlapTiles(std::vector<SPDTile*> *tiles) ;
		void processImportedPulse(SPDFile *spdFile, SPDPulse *pulse) ;
		void completeFileAndClose();
		void setTiles(std::vector<SPDTile*> *tiles);
        ~SPDWrite2OverlapTiles();
	private:
        inline bool ptWithinTile(double x, double y, double xMin, double xMax, double yMin, double yMax);
        std::vector<SPDTile*> *tiles;
        std::vector<SPDPulse*> *pls;
        boost::uint_fast16_t numOpenFiles;
	};
    
    
    class DllExport SPDWrite2TilesCore : public SPDImporterProcessor
	{
	public:
		SPDWrite2TilesCore(std::vector<SPDTile*> *tiles) ;
		void processImportedPulse(SPDFile *spdFile, SPDPulse *pulse) ;
		void completeFileAndClose();
		void setTiles(std::vector<SPDTile*> *tiles);
        ~SPDWrite2TilesCore();
	private:
        inline bool ptWithinTile(double x, double y, double xMin, double xMax, double yMin, double yMax);
        std::vector<SPDTile*> *tiles;
        std::vector<SPDPulse*> *pls;
	};
    
    
}

#endif







