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

#include "spd/SPDFileReader.h"
#include "spd/SPDFile.h"
#include "spd/SPDTextFileUtilities.h"
#include "spd/SPDException.h"
#include "spd/SPDApplyElevationChange.h"

#include "spd/spd-config.h"

using namespace std;
using namespace spdlib;
using namespace TCLAP;

int main (int argc, char * const argv[]) 
{
	cout << "spdelevation " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	cout << "and you are welcome to redistribute it under certain conditions; See\n";
	cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << endl;
	
	try 
	{
		CmdLine cmd("Alter the elevation of the pulses: spdelevation", ' ', "1.0.0");
		
		ValueArg<double> constantArg("","constant","Alter pulse elevation by a constant amount",false,0,"double");
		ValueArg<string> variableArg("","variable","Alter pulse elevation by a variable amount defined using an image",false,"","string");
		cmd.xorAdd(constantArg, variableArg);
        
		SwitchArg addSwitch("","add","Add offset", false);		
		SwitchArg minusSwitch("","minus","Remove offset", false);		
		cmd.xorAdd(addSwitch, minusSwitch);
        
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
			for(unsigned int i = 0; i < fileNames.size(); ++i)
			{
				cout << i << ": " << fileNames.at(i) << endl;
			}
			
			SPDTextFileUtilities textUtils;
			string message = string("Two file paths should have been specified (e.g., Input and Output). ") + textUtils.uInt32bittostring(fileNames.size()) + string(" were provided.");
			throw SPDException(message);
		}
		
		bool constantElevationChange = false;
		double elevConstant = 0;
		string elevImagePath = "";
		if(constantArg.isSet())
		{
			constantElevationChange = true;
			elevConstant = constantArg.getValue();
		}
		else if(variableArg.isSet())
		{
			constantElevationChange = false;
			elevImagePath = variableArg.getValue();
		}
		else 
		{
			throw SPDException("Either a constant needs to be defined or an image with the values provided.");
		}
		
		bool addOffset = true;
		if(addSwitch.getValue())
		{
			addOffset = true;
		}
		else if(minusSwitch.getValue())
		{
			addOffset = false;
		}
		else
		{
			throw SPDException("You need to specify whether the offset is to be added or removed.");
		}
		
		string inputFile = fileNames.at(0);
        string outputFile = fileNames.at(1);
        
        SPDFile *inSPDFile = new SPDFile(inputFile);
        SPDFileReader spdReader = SPDFileReader();
        spdReader.readHeaderInfo(inputFile, inSPDFile);
        
        bool indexedSPDFile = true;
        if(inSPDFile->getFileType() == SPD_UPD_TYPE)
        {
            indexedSPDFile = false;
        }
		
		SPDApplyElevationChange applyElevChange;
		if(indexedSPDFile)
		{
			if(constantElevationChange)
			{
				applyElevChange.applyConstantElevationChangeSPD(inputFile, outputFile, elevConstant, addOffset, numOfColsBlockArg.getValue(), numOfRowsBlockArg.getValue());
			}
			else 
			{
				applyElevChange.applyVariableElevationChangeSPD(inputFile, outputFile, elevImagePath, addOffset, numOfColsBlockArg.getValue(), numOfRowsBlockArg.getValue());
			}
		}
		else
		{
			if(constantElevationChange)
			{
				applyElevChange.applyConstantElevationChangeUnsorted(inputFile, outputFile, elevConstant, addOffset);
			}
			else 
			{
				applyElevChange.applyVariableElevationChangeUnsorted(inputFile, outputFile, elevImagePath, addOffset);
			}
		}
	}
	catch (ArgException &e) 
	{
		cerr << "Parse Error: " << e.what() << endl;
	}
	catch(SPDException &e)
	{
		cerr << "Error: " << e.what() << endl;
	}
	std::cout << "spdelevation - end\n";
}

