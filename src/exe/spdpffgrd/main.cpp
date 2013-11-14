/*
 *  main.cpp
 *  spdlib
 *
 *  Created by Pete Bunting on 04/05/2012.
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

#include <string>
#include <iostream>
#include <algorithm>

#include <spd/tclap/CmdLine.h>

#include "spd/SPDException.h"
#include "spd/SPDFile.h"

#include "spd/SPDProcessDataBlocks.h"
#include "spd/SPDDataBlockProcessor.h"
#include "spd/SPDParameterFreeGroundFilter.h"

#include "spd/spd-config.h"

int main (int argc, char * const argv[]) 
{
    std::cout.precision(12);
    
	std::cout << "spdpffgrd " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	std::cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	std::cout << "and you are welcome to redistribute it under certain conditions; See\n";
	std::cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	std::cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << std::endl;
	
	try 
	{
        
        TCLAP::CmdLine cmd("Classifies the ground returns using a parameter-free filtering algorithm: spdpffgrd", ' ', "1.0.0");
		
        TCLAP::ValueArg<boost::uint_fast32_t> numOfRowsBlockArg("r","blockrows","Number of rows within a block (Default 100)",false,100,"unsigned int");
		cmd.add( numOfRowsBlockArg );
        
        TCLAP::ValueArg<boost::uint_fast32_t> numOfColsBlockArg("c","blockcols","Number of columns within a block (Default 0) - Note values greater than 1 result in a non-sequencial SPD file.",false,0,"unsigned int");
		cmd.add( numOfColsBlockArg );
        
        TCLAP::ValueArg<float> binSizeArg("b","binsize","Bin size for processing and output image (Default 0) - Note 0 will use the native SPD file bin size.",false,0,"float");
		cmd.add( binSizeArg );
        
        TCLAP::ValueArg<uint_fast16_t> overlapArg("","overlap","Size (in bins) of the overlap between processing blocks (Default 10)",false,10,"uint_fast16_t");
		cmd.add( overlapArg );
		
        // add below to constructor
        
		TCLAP::ValueArg<float> grdPtDevThresArg("","grd","Threshold for deviation from identified ground surface for classifying the ground returns (Default 0.3)",false,0.3,"float");
		cmd.add( grdPtDevThresArg );
        
        TCLAP::ValueArg<boost::uint_fast32_t> kValue("k","kvalue","Number of stddevs used for control point filtering - default 3",false,3,"uint_fast32_t");
		cmd.add( kValue );
        
        TCLAP::ValueArg<boost::uint_fast32_t> classifyDevThresh("s","stddev","Number of standard deviations used in classification threshold - default 3",false,3,"uint_fast32_t");
		cmd.add( classifyDevThresh );
        
        TCLAP::ValueArg<boost::uint_fast32_t> topHatStart("t","tophatstart","Starting window size (actually second, first is always 1) for tophat transforms, must be >= 2, setting this too big can cause segfault! - default 4",false,4,"uint_fast32_t");
		cmd.add( topHatStart );
        
        TCLAP::ValueArg<bool> topHatScales("","tophatscales","Whether the tophat window size decreases through the resolutions - default true",false,true,"bool");
		cmd.add( topHatScales );
        
        TCLAP::ValueArg<boost::uint_fast32_t> topHatFactor("f","tophatfactor","How quickly the tophat window reduces through the resolution, higher numbers reduce size quicker - default 2",false,2,"uint_fast32_t");
		cmd.add( topHatFactor );
        
        TCLAP::ValueArg<uint_fast16_t> minPointDensity("m","mpd","Minimum point density in block to use for surface estimation - default 40",false,40,"uint_fast16_t");
        cmd.add( minPointDensity );

        
        // stop adding to constructor
		
		TCLAP::SwitchArg imageSwitch("","image","If set an image of the output surface will be generated rather than classifying the points (useful for debugging and parameter selection)", false);
		cmd.add( imageSwitch );
        
        TCLAP::SwitchArg morphMinSwitch("","morphmin","Apply morphological opening and closing to remove multiple path returns (note this can remove real ground returns).", false);
		cmd.add( morphMinSwitch );
        
        TCLAP::ValueArg<uint_fast16_t> usePointsofClassArg("","class","Only use points of particular class",false,spdlib::SPD_ALL_CLASSES,"uint_fast16_t");
        cmd.add( usePointsofClassArg );
        
        TCLAP::ValueArg<std::string> gdalDriverArg("","gdal","Provide the GDAL driver format (Default ENVI), Erdas Imagine is HFA, KEA is KEA",false,"ENVI","string");
		cmd.add( gdalDriverArg );
        
		TCLAP::ValueArg<std::string> inputFileArg("i","input","The input SPD file.",true,"","String");
		cmd.add( inputFileArg );
        
        TCLAP::ValueArg<std::string> outputFileArg("o","output","The output file.",true,"","String");
		cmd.add( outputFileArg );
                
		cmd.parse( argc, argv );
		
		std::string inSPDFilePath = inputFileArg.getValue();
        std::string outFilePath = outputFileArg.getValue();
        
        spdlib::SPDFile *spdInFile = new spdlib::SPDFile(inSPDFilePath);
        
        spdlib::SPDParameterFreeGroundFilter *blockProcessor = new spdlib::SPDParameterFreeGroundFilter(grdPtDevThresArg.getValue(), usePointsofClassArg.getValue(), morphMinSwitch.getValue(),
                                                                                                        kValue.getValue(), classifyDevThresh.getValue(), topHatStart.getValue(),
                                                                                                        topHatScales.getValue(), topHatFactor.getValue(), minPointDensity.getValue());
        
        spdlib::SPDProcessDataBlocks processBlocks = spdlib::SPDProcessDataBlocks(blockProcessor, overlapArg.getValue(), numOfColsBlockArg.getValue(), numOfRowsBlockArg.getValue(), true);
        
        if(imageSwitch.getValue())
        {
            processBlocks.processDataBlocksGridPulsesOutputImage(spdInFile, outFilePath, binSizeArg.getValue(), 1, gdalDriverArg.getValue());
        }
        else
        {
            processBlocks.processDataBlocksGridPulsesOutputSPD(spdInFile, outFilePath, binSizeArg.getValue());
        }
        
        delete blockProcessor;
        delete spdInFile;
		
	}
	catch (TCLAP::ArgException &e)
	{
		std::cerr << "Parse Error: " << e.what() << std::endl;
	}
	catch(spdlib::SPDException &e)
	{
		std::cerr << "Error: " << e.what() << std::endl;
	}
    
    std::cout << "spdpffgrd - end\n";
}

