/*
 *  main.cpp
 *  spdlib
 *
 *  Created by Pete Bunting on 04/03/2012.
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
#include "spd/SPDPolyGroundFilter.h"

#include "spd/spd-config.h"

int main (int argc, char * const argv[]) 
{
    std::cout.precision(12);
    
	std::cout << "spdpolygrd " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	std::cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	std::cout << "and you are welcome to redistribute it under certain conditions; See\n";
	std::cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	std::cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << std::endl;
	
	try 
	{
        TCLAP::CmdLine cmd("Classify ground returns using a surface fitting algorithm: spdpolygrd", ' ', "1.0.0");
		
        TCLAP::SwitchArg globalSwitch("","global","Classify negative height as ground", false);
        TCLAP::SwitchArg localSwitch("","local","Remove falsely classified ground returns using plane fitting", false);
        
        std::vector<TCLAP::Arg*> arguments;
        arguments.push_back(&globalSwitch);
        arguments.push_back(&localSwitch);
        cmd.xorAdd(arguments);
        
        TCLAP::ValueArg<boost::uint_fast32_t> numOfRowsBlockArg("r","blockrows","Number of rows within a block (Default 100)",false,100,"unsigned int");
		cmd.add( numOfRowsBlockArg );
        
        TCLAP::ValueArg<boost::uint_fast32_t> numOfColsBlockArg("c","blockcols","Number of columns within a block (Default 0) - Note values greater than 1 result in a non-sequencial SPD file.",false,0,"unsigned int");
		cmd.add( numOfColsBlockArg );
        
        TCLAP::ValueArg<uint_fast16_t> overlapArg("","overlap","Size (in bins) of the overlap between processing blocks (Default 10)",false,10,"uint_fast16_t");
		cmd.add( overlapArg );
        
        TCLAP::ValueArg<float> binSizeArg("b","binsize","Bin size for processing and output image (Default 0) - Note 0 will use the native SPD file bin size.",false,0,"float");
		cmd.add( binSizeArg );
        
        TCLAP::ValueArg<float> grdClassThresArg("","grdthres","Threshold for how far above the interpolated ground surface a return can be and be reclassified as ground (Default = 0.25).",false,0.25,"float");
		cmd.add( grdClassThresArg );
		
		TCLAP::ValueArg<unsigned int> degreeArg("","degree","Order of polynomial surface (Default = 1).",false,1,"int");
		cmd.add( degreeArg );
		
		TCLAP::ValueArg<unsigned int> numItersArg("","iters","Number of iterations for polynomial surface to converge on ground (Default = 2).",false,2,"int");
		cmd.add( numItersArg );
        
        TCLAP::ValueArg<uint_fast16_t> usePointsofClassArg("","class","Only use points of particular class (Ground is class == 3, Default is All classes)",false,spdlib::SPD_ALL_CLASSES,"uint_fast16_t");
        cmd.add( usePointsofClassArg );
        
		TCLAP::ValueArg<std::string> inputFileArg("i","input","The input SPD file.",true,"","String");
		cmd.add( inputFileArg );
        
        TCLAP::ValueArg<std::string> outputFileArg("o","output","The output file.",true,"","String");
		cmd.add( outputFileArg );
        
		cmd.parse( argc, argv );
        
        spdlib::SPDPolyFitGroundFilter grdFilter;
        if(globalSwitch.getValue())
        {
            grdFilter.applyGlobalPolyFitGroundFilter(inputFileArg.getValue(), outputFileArg.getValue(), grdClassThresArg.getValue(), degreeArg.getValue(), numItersArg.getValue(), numOfColsBlockArg.getValue(), numOfRowsBlockArg.getValue(), binSizeArg.getValue(), usePointsofClassArg.getValue());
        }
        else if(localSwitch.getValue())
        {
            grdFilter.applyLocalPolyFitGroundFilter(inputFileArg.getValue(), outputFileArg.getValue(), grdClassThresArg.getValue(), degreeArg.getValue(), numItersArg.getValue(), numOfColsBlockArg.getValue(), numOfRowsBlockArg.getValue(), overlapArg.getValue(), binSizeArg.getValue(), usePointsofClassArg.getValue());
        }
        else
        {
            throw spdlib::SPDException("Need to define whether you require a local (--local) or global (--global) filter.");
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
    std::cout << "spdpolygrd - end\n";
}

