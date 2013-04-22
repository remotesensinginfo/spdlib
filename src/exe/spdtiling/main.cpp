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
    std::cout << "spdtiling " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	std::cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	std::cout << "and you are welcome to redistribute it under certain conditions; See\n";
	std::cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	std::cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << std::endl;
	
	try
	{
        TCLAP::CmdLine cmd("Tools for tiling a set of SPD files using predefined tile areas: spdtiling", ' ', "1.0.0");
				
        TCLAP::ValueArg<std::string> tilesFileArg("t","tiles","XML file defining the tile regions",true,"","String");
		cmd.add( tilesFileArg );
        
        TCLAP::ValueArg<std::string> outputBaseArg("o","output","The base path for the tiles.",true,"","String");
		cmd.add( outputBaseArg );
        
        TCLAP::ValueArg<std::string> inputFilesArg("i","input","A text file with a list of input files, one per line.",true,"","String");
		cmd.add( inputFilesArg );
        
        TCLAP::SwitchArg rmEmptyTilesSwitch("r","rmempty","Remove tiles which have no data.", false);
        cmd.add( rmEmptyTilesSwitch );
        
        TCLAP::SwitchArg updateXMLSwitch("u","updatexml","Update the tiles XML file.", false);
        cmd.add( updateXMLSwitch );
        
		cmd.parse( argc, argv );
        
        if(tilesFileArg.getValue() == "")
        {
            throw TCLAP::ArgException("Tiles file path should not be blank.");
        }
        
        if(outputBaseArg.getValue() == "")
        {
            throw TCLAP::ArgException("Output files path should not be blank.");
        }
        
        if(inputFilesArg.getValue() == "")
        {
            throw TCLAP::ArgException("Input file path should not be blank.");
        }
        
        std::cout.precision(12);
        std::cout << "Tiles XML file: " << tilesFileArg.getValue() << std::endl;
        std::cout << "Input SPD file list: " << inputFilesArg.getValue() << std::endl;
        std::cout << "Output Tiles base: " << outputBaseArg.getValue() << std::endl;
        
        double xSize = 0;
        double ySize = 0;
        double overlap = 0;
        
        double xMin = 0;
        double xMax = 0;
        double yMin = 0;
        double yMax = 0;
        
        boost::uint_fast32_t rows = 0;
        boost::uint_fast32_t cols = 0;
        
        std::cout << "Reading tile XML\n";
        spdlib::SPDTilesUtils tileUtils;
        std::vector<spdlib::SPDTile*> *tiles = tileUtils.importTilesFromXML(tilesFileArg.getValue(), &rows, &cols, &xSize, &ySize, &overlap, &xMin, &xMax, &yMin, &yMax);
        
        std::cout << "Number of rows: " << rows << std::endl;
        std::cout << "Number of cols: " << cols << std::endl;
        
        std::cout << "Tile Size: [" << xSize << "," << ySize << "] Overlap: " << overlap << std::endl;
        
        std::cout << "Full Area: [" << xMin << "," << xMax << "][" << yMin << "," << yMax << "]\n";
        
        //tileUtils.printTiles2Console(tiles);
        
        
        
        
        spdlib::SPDTextFileUtilities txtUtils;        
        std::vector<std::string> inputFiles = txtUtils.readFileLinesToVector(inputFilesArg.getValue());
        /*
        for(std::vector<std::string>::iterator iterFiles = inputFiles.begin(); iterFiles != inputFiles.end(); ++iterFiles)
        {
            std::cout << "\'" << *iterFiles << "\'" << std::endl;
        }
        */
        spdlib::SPDFile *inSPDFile = new spdlib::SPDFile(inputFiles.front());
        
        spdlib::SPDFileReader reader;
        reader.readHeaderInfo(inputFiles.front(), inSPDFile);
        
        std::cout << "Opening and creating output files.\n";
        tileUtils.createTileSPDFiles(tiles, inSPDFile, outputBaseArg.getValue(), xSize, ySize, overlap, xMin, xMax, yMin, yMax, rows, cols);
        
        std::cout << "Populate the tiles with data\n.";
        tileUtils.populateTileWithData(tiles, inputFiles);
        
        if(rmEmptyTilesSwitch.getValue())
        {
            std::cout << "Remove empty tiles\n";
            tileUtils.deleteTilesWithNoPulses(tiles);
        }
        
        if(updateXMLSwitch.getValue())
        {
            std::cout << "Export updated tiles XML.";
            tileUtils.exportTiles2XML(tilesFileArg.getValue(), tiles, xSize, ySize, overlap, xMin, xMax, yMin, yMax, rows, cols);
        }
        
	}
	catch (TCLAP::ArgException &e)
	{
		std::cerr << "Parse Error: " << e.what() << std::endl;
	}
	catch(spdlib::SPDException &e)
	{
		std::cerr << "Error: " << e.what() << std::endl;
	}
}

