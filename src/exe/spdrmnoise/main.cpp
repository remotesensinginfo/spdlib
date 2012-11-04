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

#include "spd/SPDTextFileUtilities.h"
#include "spd/SPDRemoveVerticalNoise.h"
#include "spd/SPDException.h"
#include "spd/SPDProcessPulses.h"
#include "spd/SPDProcessingException.h"

#include "spd/spd-config.h"

using namespace std;
using namespace spdlib;
using namespace TCLAP;

int main (int argc, char * const argv[]) 
{
    cout << "spdrmnoise " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	cout << "and you are welcome to redistribute it under certain conditions; See\n";
	cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << endl;
	
	try 
	{
		CmdLine cmd("Remove vertical noise from LiDAR datasets: spdrmnoise", ' ', "1.0.0");
        
        ValueArg<boost::uint_fast32_t> numOfRowsBlockArg("r","blockrows","Number of rows within a block (Default 100)",false,100,"unsigned int");
		cmd.add( numOfRowsBlockArg );
        
        ValueArg<boost::uint_fast32_t> numOfColsBlockArg("c","blockcols","Number of columns within a block (Default 0) - Note values greater than 1 result in a non-sequencial SPD file.",false,0,"unsigned int");
		cmd.add( numOfColsBlockArg );
        /*
        ValueArg<float> binSizeArg("b","binsize","Bin size for processing and output image (Default 0) - Note 0 will use the native SPD file bin size.",false,0,"float");
		cmd.add( binSizeArg );
        */
        ValueArg<float> absUpperArg("","absup","Absolute upper threshold for returns which are to be removed.",false,0,"float");
        cmd.add( absUpperArg );
        
        ValueArg<float> absLowerArg("","abslow","Absolute lower threshold for returns which are to be removed.",false,0,"float");
        cmd.add( absLowerArg );
        
        ValueArg<float> relUpperArg("","relup","Relative (to median) upper threshold for returns which are to be removed.",false,0,"float");
        cmd.add( relUpperArg );
        
        ValueArg<float> relLowerArg("","rellow","Relative (to median) lower threshold for returns which are to be removed.",false,0,"float");
        cmd.add( relLowerArg );
        
		UnlabeledMultiArg<string> multiFileNames("Files", "File names for the input and output files", true, "string");
		cmd.add( multiFileNames );
		cmd.parse( argc, argv );
		
		vector<string> fileNames = multiFileNames.getValue();		
		if(fileNames.size() != 2)
		{
			for(unsigned int i = 0; i < fileNames.size(); ++i)
			{
				cout << i << ": " << fileNames.at(i) << endl;
			}
			
			SPDTextFileUtilities textUtils;
			string message = string("Two file paths should have been specified (e.g., Input and Output). ") + textUtils.uInt32bittostring(fileNames.size()) + string(" were provided.");
			throw SPDException(message);
		}
				
		string inputFile = fileNames.at(0);
		string outputFile = fileNames.at(1);
        
        bool absUpSet = absUpperArg.isSet();
        bool absLowSet = absLowerArg.isSet();
        bool relUpSet = relUpperArg.isSet();
        bool relLowSet = relLowerArg.isSet();
        
        SPDFile *spdInFile = new SPDFile(inputFile);
        SPDPulseProcessor *pulseProcessor = new SPDRemoveVerticalNoise(absUpSet, absLowSet, relUpSet, relLowSet, absUpperArg.getValue(), absLowerArg.getValue(), relUpperArg.getValue(), relLowerArg.getValue());            
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
	
}

