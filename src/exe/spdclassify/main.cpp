/*
 *  main.cpp
 *  spdlib
 *
 *  Created by Pete Bunting on 30/04/2014.
 *  Copyright 2014 SPDLib. All rights reserved.
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

#include "spd/SPDCommon.h"
#include "spd/SPDException.h"
#include "spd/SPDFile.h"

#include "spd/SPDClassifyPts.h"
#include "spd/SPDProcessDataBlocks.h"
#include "spd/SPDDataBlockProcessor.h"

#include "spd/spd-config.h"

int main (int argc, char * const argv[])
{
    std::cout.precision(12);
    
    std::cout << "spdclassify " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	std::cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	std::cout << "and you are welcome to redistribute it under certain conditions; See\n";
	std::cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	std::cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << std::endl;
	
	try
	{
        TCLAP::CmdLine cmd("Attempts to classify returns into hard standing (buildings) and vegetation: spdclassify", ' ', "1.0.0");
		
        TCLAP::ValueArg<boost::uint_fast32_t> numOfRowsBlockArg("r","blockrows","Number of rows within a block (Default 100)",false,25,"unsigned int");
		cmd.add( numOfRowsBlockArg );
        
        TCLAP::ValueArg<boost::uint_fast32_t> numOfColsBlockArg("c","blockcols","Number of columns within a block (Default 0) - Note values greater than 1 result in a non-sequencial SPD file.",false,0,"unsigned int");
		cmd.add( numOfColsBlockArg );
        
        TCLAP::ValueArg<float> binSizeArg("b","binsize","Bin size for processing and output image (Default 0) - Note 0 will use the native SPD file bin size.",false,0,"float");
		cmd.add( binSizeArg );
        
		TCLAP::ValueArg<std::string> inputFileArg("i","input","The input file.",true,"","String");
		cmd.add( inputFileArg );
        
        TCLAP::ValueArg<std::string> outputFileArg("o","output","The output file.",true,"","String");
		cmd.add( outputFileArg );
        
		cmd.parse( argc, argv );
		
        
		std::string inSPDFilePath = inputFileArg.getValue();
        std::string outSPDFilePath = outputFileArg.getValue();
        
        spdlib::SPDFile *spdInFile = new spdlib::SPDFile(inSPDFilePath);
        spdlib::SPDFileReader reader;
        reader.readHeaderInfo(spdInFile->getFilePath(), spdInFile);
        
        if(spdInFile->getHeightDefined() == spdlib::SPD_TRUE)
        {
            spdlib::SPDClassifyPtsNumReturns *classifyReturnsProcessor = new spdlib::SPDClassifyPtsNumReturns();
            spdlib::SPDProcessDataBlocks processBlocks = spdlib::SPDProcessDataBlocks(classifyReturnsProcessor, 0, numOfColsBlockArg.getValue(), numOfRowsBlockArg.getValue(), true);
            processBlocks.processDataBlocksGridPulsesOutputSPD(spdInFile, outSPDFilePath, binSizeArg.getValue());
            delete classifyReturnsProcessor;
        }
        else
        {
            throw spdlib::SPDException("The height field needs to be defined before spdclassify can be used.");
        }
        
        
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
	std::cout << "spdclassify - end\n";
}

