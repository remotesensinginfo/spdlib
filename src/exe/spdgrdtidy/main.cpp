/*
 *  main.cpp
 *  spdlib
 *
 *  Created by Pete Bunting on 05/11/2013.
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

#include "spd/SPDException.h"
#include "spd/SPDFile.h"
#include "spd/SPDFileReader.h"
#include "spd/SPDTidyGroundReturn.h"
#include "spd/SPDProcessDataBlocks.h"
#include "spd/SPDDataBlockProcessor.h"
#include "spd/SPDProcessPulses.h"

#include "spd/spd-config.h"

int main (int argc, char * const argv[])
{
    std::cout.precision(12);

    std::cout << "spdgrdtidy " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	std::cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	std::cout << "and you are welcome to redistribute it under certain conditions; See\n";
	std::cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	std::cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << std::endl;
	
	try
	{
        TCLAP::CmdLine cmd("Attempt to tidy up the ground return classification: spdgrdtidy", ' ', "1.0.0");
		
        TCLAP::SwitchArg negHeightSwitch("","negheights","Classify negative height as ground", false);
        TCLAP::SwitchArg planeFitSwitch("","planefit","Remove falsely classified ground returns using plane fitting", false);

        std::vector<TCLAP::Arg*> arguments;
        arguments.push_back(&negHeightSwitch);
        arguments.push_back(&planeFitSwitch);
        cmd.xorAdd(arguments);

        TCLAP::ValueArg<boost::uint_fast32_t> numOfRowsBlockArg("r","blockrows","Number of rows within a block (Default 100)",false,100,"unsigned int");
		cmd.add( numOfRowsBlockArg );

        TCLAP::ValueArg<boost::uint_fast32_t> numOfColsBlockArg("c","blockcols","Number of columns within a block (Default 0) - Note values greater than 1 result in a non-sequencial SPD file.",false,0,"unsigned int");
		cmd.add( numOfColsBlockArg );

        TCLAP::ValueArg<float> binSizeArg("b","binsize","Bin size for processing and output image (Default 0) - Note 0 will use the native SPD file bin size.",false,0,"float");
		cmd.add( binSizeArg );

        TCLAP::ValueArg<boost::uint_fast16_t> winHSizeArg("w","winhsize","Half the window size for the processing (Default 3) - Note 3 means a 7 x 7 filter window size (in units of the bin size).",false,3,"int");
		cmd.add( winHSizeArg );

        TCLAP::ValueArg<std::string> inputFileArg("i","input","The input SPD file.",true,"","String");
		cmd.add( inputFileArg );

        TCLAP::ValueArg<std::string> outputFileArg("o","output","The output SPD file.",true,"","String");
		cmd.add( outputFileArg );

		cmd.parse( argc, argv );
		
		std::string inSPDFilePath = inputFileArg.getValue();
        std::string outFilePath = outputFileArg.getValue();

        if(negHeightSwitch.getValue())
        {
            spdlib::SPDFile *spdInFile = new spdlib::SPDFile(inSPDFilePath);
            spdlib::SPDFileReader reader;
            reader.readHeaderInfo(inSPDFilePath, spdInFile);

            if(spdInFile->getHeightDefined() == spdlib::SPD_TRUE)
            {
                std::cout << "Ground tidying running...\n";
                spdlib::SPDTidyGroundReturnNegativeHeights *blockProcessorGrdTidy = new spdlib::SPDTidyGroundReturnNegativeHeights();

                spdlib::SPDProcessDataBlocks processBlocks = spdlib::SPDProcessDataBlocks(blockProcessorGrdTidy, 0, numOfColsBlockArg.getValue(), numOfRowsBlockArg.getValue(), true);
                processBlocks.processDataBlocksGridPulsesOutputSPD(spdInFile, outFilePath, 0);

                delete blockProcessorGrdTidy;
            }
            else
            {
                throw spdlib::SPDException("The height field must be defined; use spddefheight.");
            }
            delete spdInFile;
        }
        else if(planeFitSwitch.getValue())
        {
            spdlib::SPDFile *spdInFile = new spdlib::SPDFile(inSPDFilePath);
            spdlib::SPDPulseProcessor *pulseProcessor = new spdlib::SPDTidyGroundReturnsPlaneFitting();
            spdlib::SPDSetupProcessPulses processPulses = spdlib::SPDSetupProcessPulses(numOfColsBlockArg.getValue(), numOfRowsBlockArg.getValue(), true);
            processPulses.processPulsesWithOutputSPD(pulseProcessor, spdInFile, outFilePath, binSizeArg.getValue(), true, winHSizeArg.getValue());

            delete pulseProcessor;
            delete spdInFile;
        }
        else
        {
            throw spdlib::SPDException("Did not recognised tidy option.");
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

    std::cout << "spdgrdtidy - end\n";
}

