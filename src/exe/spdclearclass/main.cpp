/*
 *  main.cpp
 *  spdlib
 *
 *  Created by Pete Bunting on 30/11/2010.
 *  Copyright 2010 SPDLib. All rights reserved.
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

#include "spd/SPDCopyRemovingClassificationProcessor.h"
#include "spd/SPDException.h"

#include "spd/spd-config.h"

using namespace std;
using namespace spdlib;
using namespace TCLAP;

int main (int argc, char * const argv[]) 
{
	cout << "spdclearclass " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	cout << "and you are welcome to redistribute it under certain conditions; See\n";
	cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << endl;
	
	try 
	{
		CmdLine cmd("Clear the classification of an SPD file: spdclearclass", ' ', "1.0.0");
		
        ValueArg<boost::uint_fast32_t> numOfRowsBlockArg("r","blockrows","Number of rows within a block (Default 100)",false,100,"unsigned int");
		cmd.add( numOfRowsBlockArg );
        
        ValueArg<boost::uint_fast32_t> numOfColsBlockArg("c","blockcols","Number of columns within a block (Default 0) - Note values greater than 1 result in a non-sequencial SPD file.",false,0,"unsigned int");
		cmd.add( numOfColsBlockArg );
        
		UnlabeledMultiArg<string> multiFileNames("Files", "File names for the input and output files", true, "string");
		cmd.add( multiFileNames );
		cmd.parse( argc, argv );
		
		vector<string> fileNames = multiFileNames.getValue();		
		if(fileNames.size() != 2)
		{
			throw SPDException("Two file paths should have been specified (e.g., Input and Output)");
		}
		
		string inputFile = fileNames.at(0);
		string outputFile = fileNames.at(1);
        
        SPDFile *spdInFile = new SPDFile(inputFile);
        SPDPulseProcessor *pulseProcessor = new SPDCopyRemovingClassificationProcessor();            
        SPDSetupProcessPulses processPulses = SPDSetupProcessPulses(numOfColsBlockArg.getValue(), numOfRowsBlockArg.getValue(), true);
        processPulses.processPulsesWithOutputSPD(pulseProcessor, spdInFile, outputFile);
		delete pulseProcessor;
        delete spdInFile;
		
	}
	catch (ArgException &e) 
	{
		cerr << "Parse Error: " << e.what() << endl;
	}
	catch(SPDException &e)
	{
		cerr << "Error: " << e.what() << endl;
	}
    
    std::cout << "spdclearclass - end\n";
}

