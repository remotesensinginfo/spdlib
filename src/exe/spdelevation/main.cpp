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

int main (int argc, char * const argv[])
{
    std::cout.precision(12);

    std::cout << "spdelevation " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	std::cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	std::cout << "and you are welcome to redistribute it under certain conditions; See\n";
	std::cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	std::cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << std::endl;
	
	try
	{
        TCLAP::CmdLine cmd("Alter the elevation of the pulses: spdelevation", ' ', "1.0.0");
		
		TCLAP::ValueArg<double> constantArg("","constant","Alter pulse elevation by a constant amount",false,0,"double");
		TCLAP::ValueArg<std::string> variableArg("","variable","Alter pulse elevation by a variable amount defined using an image",false,"","string");
		cmd.xorAdd(constantArg, variableArg);

		TCLAP::SwitchArg addSwitch("","add","Add offset", false);		
		TCLAP::SwitchArg minusSwitch("","minus","Remove offset", false);		
		cmd.xorAdd(addSwitch, minusSwitch);

        TCLAP::ValueArg<boost::uint_fast32_t> numOfRowsBlockArg("r","blockrows","Number of rows within a block (Default 100)",false,100,"unsigned int");
		cmd.add( numOfRowsBlockArg );

        TCLAP::ValueArg<boost::uint_fast32_t> numOfColsBlockArg("c","blockcols","Number of columns within a block (Default 0) - Note values greater than 1 result in a non-sequencial SPD file.",false,0,"unsigned int");
		cmd.add( numOfColsBlockArg );
		
		TCLAP::ValueArg<std::string> inputFileArg("i","input","The input SPD file.",true,"","String");
		cmd.add( inputFileArg );

        TCLAP::ValueArg<std::string> outputFileArg("o","output","The output SPD file.",true,"","String");
		cmd.add( outputFileArg );

		cmd.parse( argc, argv );
		
		bool constantElevationChange = false;
		double elevConstant = 0;
		std::string elevImagePath = "";
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
			throw spdlib::SPDException("Either a constant needs to be defined or an image with the values provided.");
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
			throw spdlib::SPDException("You need to specify whether the offset is to be added or removed.");
		}
		
		std::string inputFile = inputFileArg.getValue();
        std::string outputFile = outputFileArg.getValue();

        spdlib::SPDFile *inSPDFile = new spdlib::SPDFile(inputFile);
        spdlib::SPDFileReader spdReader = spdlib::SPDFileReader();
        spdReader.readHeaderInfo(inputFile, inSPDFile);

        bool indexedSPDFile = true;
        if(inSPDFile->getFileType() == spdlib::SPD_UPD_TYPE)
        {
            indexedSPDFile = false;
        }
		
		spdlib::SPDApplyElevationChange applyElevChange;
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
	catch(TCLAP::ArgException &e)
	{
		std::cerr << "Parse Error: " << e.what() << std::endl;
	}
	catch(spdlib::SPDException &e)
	{
		std::cerr << "Error: " << e.what() << std::endl;
	}
	std::cout << "spdelevation - end\n";
}

