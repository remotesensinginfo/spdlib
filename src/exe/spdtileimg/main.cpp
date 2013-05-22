/*
 *  main.cpp
 *  spdlib
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

#include <string>
#include <iostream>
#include <algorithm>

#include <spd/tclap/CmdLine.h>

#include "spd/SPDTextFileUtilities.h"
#include "spd/SPDException.h"

#include "spd/SPDGenerateTiles.h"

#include "spd/spd-config.h"

int main (int argc, char * const argv[])
{
    std::cout.precision(12);
    
    std::cout << "spdtileimg " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	std::cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	std::cout << "and you are welcome to redistribute it under certain conditions; See\n";
	std::cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	std::cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << std::endl;
	
	try
	{
        TCLAP::CmdLine cmd("Tools for mosaicing raster results following tiling: spdtileimg", ' ', "1.0.0");
        
        TCLAP::SwitchArg mosaicSwitch("m","mosaic","Mosaic the images (within the input list) together.", false);
		TCLAP::SwitchArg clumpSwitch("c","clump","Create a clumps image specifying the location of the tiles.", false);
        
        std::vector<TCLAP::Arg*> arguments;
        arguments.push_back(&mosaicSwitch);
        arguments.push_back(&clumpSwitch);
        cmd.xorAdd(arguments);
        
        TCLAP::ValueArg<std::string> formatArg("f","format","The output image format.",false,"KEA","String");
		cmd.add( formatArg );
        
        TCLAP::ValueArg<double> resolutionArg("r","resolution","The output image pixel size (--clump only).",false,10,"double");
		cmd.add( resolutionArg );
        
        TCLAP::ValueArg<std::string> wktFileArg("w","wkt","A file containing the WKT string representing the projection (--clump only).",false,"","String");
		cmd.add( wktFileArg );
        
        TCLAP::ValueArg<std::string> outputFileArg("o","output","The output image.",false,"","String");
		cmd.add( outputFileArg );
        
        TCLAP::ValueArg<std::string> tilesFileArg("t","tiles","The input XML file defining the tiles.",false,"","String");
		cmd.add( tilesFileArg );
        
        TCLAP::ValueArg<std::string> inputFilesArg("i","input","The text file with a list of input files.",false,"","String");
		cmd.add( inputFilesArg );
		        
		cmd.parse( argc, argv );
        
        if(tilesFileArg.getValue() == "")
        {
            throw TCLAP::ArgException("Tiling file path needs to have a value.");
        }
        
        if(formatArg.getValue() == "")
        {
            throw TCLAP::ArgException("Image file format needs to have a value.");
        }
        
        if(outputFileArg.getValue() == "")
        {
            throw TCLAP::ArgException("Output file path needs to have a value.");
        }
        
        GDALAllRegister();
        
        if(mosaicSwitch.getValue())
        {
            if(inputFilesArg.getValue() == "")
            {
                throw TCLAP::ArgException("The input files path needs to have a value.");
            }
            
            double xTileSize = 0;
            double yTileSize = 0;
            double overlap = 0;
            
            double xMin = 0;
            double xMax = 0;
            double yMin = 0;
            double yMax = 0;
            
            boost::uint_fast32_t rows = 0;
            boost::uint_fast32_t cols = 0;
            
            std::cout.precision(12);
            
            std::cout << "Reading tile XML\n";
            spdlib::SPDTilesUtils tileUtils;
            std::vector<spdlib::SPDTile*> *tiles = tileUtils.importTilesFromXML(tilesFileArg.getValue(), &rows, &cols, &xTileSize, &yTileSize, &overlap, &xMin, &xMax, &yMin, &yMax);
            
            std::cout << "Number of rows: " << rows << std::endl;
            std::cout << "Number of cols: " << cols << std::endl;
            
            std::cout << "Tile Size: [" << xTileSize << "," << yTileSize << "] Overlap: " << overlap << std::endl;
            
            double totalSizeX = cols * xTileSize;
            double totalSizeY = rows * yTileSize;
            
            xMax = xMin + totalSizeX;
            yMax = yMin + totalSizeY;
            
            std::cout << "Full Area: [" << xMin << "," << xMax << "][" << yMin << "," << yMax << "]\n";
            
            spdlib::SPDTextFileUtilities txtUtils;
            std::vector<std::string> inputFiles = txtUtils.readFileLinesToVector(inputFilesArg.getValue());            
            
            std::string tmpImgFile = inputFiles.at(0);
            
            GDALDataset *tmpDataset = NULL;
            tmpDataset = (GDALDataset *) GDALOpen(tmpImgFile.c_str(), GA_ReadOnly);
            if(tmpDataset == NULL)
            {
                std::string message = std::string("Could not open image ") + tmpImgFile;
                throw spdlib::SPDException(message.c_str());
            }
            
            boost::uint_fast32_t numOfImgBands = tmpDataset->GetRasterCount();
            double *trans = new double[6];
            tmpDataset->GetGeoTransform(trans);
            
            double xRes = trans[1];
            double yRes = trans[5];
            double pYRes = yRes;
            if(yRes < 0)
            {
                pYRes = yRes * (-1);
            }
            delete[] trans;
            
            std::string wktStr = std::string(tmpDataset->GetProjectionRef());
            GDALDataType dataType = tmpDataset->GetRasterBand(1)->GetRasterDataType();
            
            boost::uint_fast32_t xImgSize = floor(((xMax - xMin)/xRes)+0.5);
            boost::uint_fast32_t yImgSize = floor(((yMax - yMin)/pYRes)+0.5);
            GDALClose(tmpDataset);
            
            std::cout << "Create blank image\n";
            GDALDataset *outDataset = tileUtils.createNewImageFile(outputFileArg.getValue(), formatArg.getValue(), dataType, wktStr, xRes, yRes, xMin, yMax, xImgSize, yImgSize, numOfImgBands);
            
            std::cout << "Add tiles to output image\n";
            tileUtils.addImageTilesParseFileName(outDataset, tiles, inputFiles);
            
            GDALClose(outDataset);
        }
        else if(clumpSwitch.getValue())
        {
            if(wktFileArg.getValue() == "")
            {
                throw TCLAP::ArgException("The WKT file path needs to have a value.");
            }
            
            double xRes = resolutionArg.getValue();
            double yRes = resolutionArg.getValue()*(-1);
            
            double xTileSize = 0;
            double yTileSize = 0;
            double overlap = 0;
            
            double xMin = 0;
            double xMax = 0;
            double yMin = 0;
            double yMax = 0;
            
            boost::uint_fast32_t rows = 0;
            boost::uint_fast32_t cols = 0;
            
            std::cout.precision(12);
            
            std::cout << "Reading tile XML\n";
            spdlib::SPDTilesUtils tileUtils;
            std::vector<spdlib::SPDTile*> *tiles = tileUtils.importTilesFromXML(tilesFileArg.getValue(), &rows, &cols, &xTileSize, &yTileSize, &overlap, &xMin, &xMax, &yMin, &yMax);
            
            std::cout << "Number of rows: " << rows << std::endl;
            std::cout << "Number of cols: " << cols << std::endl;
            
            std::cout << "Tile Size: [" << xTileSize << "," << yTileSize << "] Overlap: " << overlap << std::endl;
            
            double totalSizeX = cols * xTileSize;
            double totalSizeY = rows * yTileSize;

            xMax = xMin + totalSizeX;
            yMax = yMin + totalSizeY;
            
            std::cout << "Full Area: [" << xMin << "," << xMax << "][" << yMin << "," << yMax << "]\n";
            
            spdlib::SPDTextFileUtilities txtUtils;
            std::string wktStr = txtUtils.readFileToString(wktFileArg.getValue());
            
            boost::uint_fast32_t xImgSize = floor(((xMax - xMin)/xRes)+0.5);
            boost::uint_fast32_t yImgSize = floor(((yMax - yMin)/xRes)+0.5);
            
            std::cout << "Output Image Size = [" << xImgSize << ", " << yImgSize << "]\n";
            
            std::cout << "Create blank image\n";
            GDALDataset *outDataset = tileUtils.createNewImageFile(outputFileArg.getValue(), formatArg.getValue(), GDT_UInt32, wktStr, xRes, yRes, xMin, yMax, xImgSize, yImgSize, 1);
            outDataset->GetRasterBand(1)->SetMetadataItem("LAYER_TYPE", "thematic");
            
            std::cout << "Populate Clumps Image File\n";
            tileUtils.addTiles2ClumpImage(outDataset, tiles);
            
            GDALClose(outDataset);
        }
        else
        {
            throw TCLAP::ArgException("Either the mosaic or clump switch need to be provided.");
        }
        
        GDALDestroyDriverManager();
	}
	catch (TCLAP::ArgException &e)
	{
		std::cerr << "Parse Error: " << e.what() << std::endl;
	}
	catch(spdlib::SPDException &e)
	{
		std::cerr << "Error: " << e.what() << std::endl;
	}
    
    std::cout << "spdtileimg - end\n";
}

