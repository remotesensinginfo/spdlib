/*
 *  main.cpp
 *  spdlib
 *
 *  Created by Pete Bunting on 03/11/2016.
 *  Copyright 2016 SPDLib. All rights reserved.
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

#include "spd/SPDDefineClassesFromImg.h"

#include "spd/spd-config.h"

int main (int argc, char * const argv[])
{
    std::cout.precision(12);

    std::cout << "spdclassimg " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	std::cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	std::cout << "and you are welcome to redistribute it under certain conditions; See\n";
	std::cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	std::cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << std::endl;
	
	try
	{
        TCLAP::CmdLine cmd("Define point class from an image file: spdclassimg", ' ', "1.0.0");

        TCLAP::ValueArg<boost::uint_fast32_t> numOfRowsBlockArg("r","blockrows","Number of rows within a block (Default 100)",false,100,"unsigned int");
		cmd.add( numOfRowsBlockArg );

        TCLAP::ValueArg<boost::uint_fast32_t> numOfColsBlockArg("c","blockcols","Number of columns within a block (Default 0) - Note values greater than 1 result in a non-sequencial SPD file.",false,0,"unsigned int");
		cmd.add( numOfColsBlockArg );
		
		TCLAP::ValueArg<uint_fast16_t> classBandArg("","band","Image band for class image band",false,1,"uint_fast16_t");
		cmd.add( classBandArg );

		TCLAP::ValueArg<std::string> inputFileArg("i","input","The input SPD file.",true,"","String");
		cmd.add( inputFileArg );

        TCLAP::ValueArg<std::string> imageFileArg("","image","The input image file.",false,"","String");
		cmd.add( imageFileArg );

        TCLAP::ValueArg<std::string> outputFileArg("o","output","The output SPD file.",true,"","String");
		cmd.add( outputFileArg );

		cmd.parse( argc, argv );
		
        std::cout << "Use image values (0 ignored) to reclassify points intersecting with a image pixel.\n";

        std::string inputSPDFile = inputFileArg.getValue();
        std::string inputImageFile = imageFileArg.getValue();
        std::string outputSPDFile = outputFileArg.getValue();

        uint_fast16_t classBand = classBandArg.getValue()-1;

        spdlib::SPDFile *spdInFile = new spdlib::SPDFile(inputSPDFile);
        spdlib::SPDPulseProcessor *pulseProcessor = new spdlib::SPDDefineClassesFromImg(classBand);
        spdlib::SPDSetupProcessPulses processPulses = spdlib::SPDSetupProcessPulses(numOfColsBlockArg.getValue(), numOfRowsBlockArg.getValue(), true);
        processPulses.processPulsesWithInputImage(pulseProcessor, spdInFile, outputSPDFile, inputImageFile);

        delete spdInFile;
        delete pulseProcessor;
	}
	catch (TCLAP::ArgException &e)
	{
		std::cerr << "Parse Error: " << e.what() << std::endl;
	}
	catch(spdlib::SPDException &e)
	{
		std::cerr << "Error: " << e.what() << std::endl;
	}
    std::cout << "spddefrgb - end\n";
}

