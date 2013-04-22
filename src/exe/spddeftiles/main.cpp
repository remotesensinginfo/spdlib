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
    std::cout << "spddeftiles " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	std::cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	std::cout << "and you are welcome to redistribute it under certain conditions; See\n";
	std::cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	std::cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << std::endl;
	
	try
	{
        TCLAP::CmdLine cmd("Tools for defining a set of tiles: spddeftiles", ' ', "1.0.0");
		
        TCLAP::SwitchArg defTilesSwitch("t","tiles","Define a set of tiles for a region.", false);
		TCLAP::SwitchArg filesExtentSwitch("e","extent","Calculate the extent of a set of files.", false);
        
        std::vector<TCLAP::Arg*> arguments;
        arguments.push_back(&defTilesSwitch);
        arguments.push_back(&filesExtentSwitch);
        cmd.xorAdd(arguments);
        
        TCLAP::ValueArg<double> xSizeArg("","xsize","X size (in units of coordinate systems) of the tiles (Default 1000) (--tiles).",false,1000,"double");
		cmd.add( xSizeArg );
        
        TCLAP::ValueArg<double> ySizeArg("","ysize","Y size (in units of coordinate systems) of the tiles (Default 1000) (--tiles).",false,1000,"double");
		cmd.add( ySizeArg );
        
        TCLAP::ValueArg<double> overlapArg("","overlap","Size (in units of coordinate systems) of the overlap for tiles (Default 100) (--tiles).",false,100,"double");
		cmd.add( overlapArg );
        
        TCLAP::ValueArg<double> xMinArg("","xmin","X min (in units of coordinate systems) of the region to be tiled (--tiles).",false,0,"double");
		cmd.add( xMinArg );
        
        TCLAP::ValueArg<double> yMinArg("","ymin","Y min (in units of coordinate systems) of the region to be tiled (--tiles).",false,0,"double");
		cmd.add( yMinArg );
        
        TCLAP::ValueArg<double> xMaxArg("","xmax","X max (in units of coordinate systems) of the region to be tiled (--tiles).",false,0,"double");
		cmd.add( xMaxArg );
        
        TCLAP::ValueArg<double> yMaxArg("","ymax","Y max (in units of coordinate systems) of the region to be tiled (--tiles).",false,0,"double");
		cmd.add( yMaxArg );
        
        TCLAP::ValueArg<std::string> outputArg("o","output","Output XML file defining the tiles (--tiles).",false,"","String");
		cmd.add( outputArg );
        
        TCLAP::ValueArg<std::string> inputArg("i","input","Input file listing the set of input files (--extent).",false,"","String");
		cmd.add( inputArg );
        
		cmd.parse( argc, argv );
    
        if(defTilesSwitch.getValue())
        {
            if( outputArg.getValue() != "" )
            {
                if(xSizeArg.getValue() <= 0)
                {
                    throw TCLAP::ArgException("X tile size must be greater than zero.");
                }
                else if(ySizeArg.getValue() <= 0)
                {
                    throw TCLAP::ArgException("Y tile size must be greater than zero.");
                }
                
                
                double xSize = xSizeArg.getValue();
                double ySize = ySizeArg.getValue();
                double overlap = overlapArg.getValue();
                
                double xMin = xMinArg.getValue();
                double xMax = xMaxArg.getValue();
                double yMin = yMinArg.getValue();
                double yMax = yMaxArg.getValue();
                
                boost::uint_fast32_t rows = 0;
                boost::uint_fast32_t cols = 0;
                
                if(xMax <= xMin)
                {
                    throw TCLAP::ArgException("xMax must be larger than xMin.");
                }
                else if(yMax <= yMin)
                {
                    throw TCLAP::ArgException("yMax must be larger than yMin.");
                }
                
                spdlib::SPDTilesUtils tileUtils;
                std::vector<spdlib::SPDTile*> *tiles = tileUtils.createTiles(xSize, ySize, overlap, xMin, xMax, yMin, yMax, &rows, &cols);
                tileUtils.printTiles2Console(tiles);
                std::cout << "Exporting to XML: " << outputArg.getValue() << std::endl;
                tileUtils.exportTiles2XML(outputArg.getValue(), tiles, xSize, ySize, overlap, xMin, xMax, yMin, yMax, rows, cols);
                tileUtils.deleteTiles(tiles);
            }
            else
            {
                throw TCLAP::ArgException("Output argument must not be a blank string.");
            }
        }
        else if (filesExtentSwitch.getValue())
        {
            if( inputArg.getValue() != "" )
            {                
                double xMin = 0;
                double xMax = 0;
                double yMin = 0;
                double yMax = 0;
                
                spdlib::SPDTextFileUtilities txtUtils;
                std::vector<std::string> inputFiles = txtUtils.readFileLinesToVector(inputArg.getValue());
                
                spdlib::SPDTilesUtils tileUtils;
                tileUtils.calcFileExtent(inputFiles, &xMin, &xMax, &yMin, &yMax);
                
                std::cout.precision(12);
                std::cout << "Extent [xMin, xMax, yMin, yMax]: [" << xMin << ", " << xMax << ", " << yMin << ", " << yMax << "]\n";
            }
            else
            {
                throw TCLAP::ArgException("Input argument must not be a blank string.");
            }
        }
        else
        {
            throw TCLAP::ArgException("Either the define tiles or calculate extent options must be provided.");
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

