/*
 *  main.cpp
 *  spdlib
 *
 *  Created by Pete Bunting on 15/11/2012.
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

#include "spd/SPDTextFileUtilities.h"
#include "spd/SPDException.h"
#include "spd/SPDProcessPulses.h"
#include "spd/SPDProcessingException.h"
#include "spd/SPDExtractReturns.h"
#include "spd/SPDFileReader.h"
#include "spd/SPDCommon.h"

#include "spd/spd-config.h"

int main (int argc, char * const argv[])
{
    std::cout.precision(12);
    
    std::cout << "spdextract " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	std::cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	std::cout << "and you are welcome to redistribute it under certain conditions; See\n";
	std::cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	std::cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << std::endl;
	
	try
	{
        TCLAP::CmdLine cmd("Extract returns and pulses which meet a set of criteria: spdextract", ' ', "1.0.0");
        
        TCLAP::ValueArg<boost::uint_fast32_t> numOfRowsBlockArg("r","blockrows","Number of rows within a block (Default 100)",false,100,"unsigned int");
		cmd.add( numOfRowsBlockArg );
        
        TCLAP::ValueArg<boost::uint_fast32_t> numOfColsBlockArg("c","blockcols","Number of columns within a block (Default 0) - Note values greater than 1 result in a non-sequencial SPD file.",false,0,"unsigned int");
		cmd.add( numOfColsBlockArg );
        
        TCLAP::ValueArg<float> binSizeArg("b","binsize","Bin size for processing and output image (Default 0) - Note 0 will use the native SPD file bin size.",false,0,"float");
		cmd.add( binSizeArg );
        
        TCLAP::ValueArg<boost::uint_fast16_t> classOfInterestArg("", "class", "Class of interest (Ground == 3; Not Ground == 104)", false, spdlib::SPD_ALL_CLASSES, "unsigned int");
		cmd.add( classOfInterestArg );
        
        std::vector<std::string> allowedReturns;
		allowedReturns.push_back("ALL");
		allowedReturns.push_back("FIRST");
        allowedReturns.push_back("LAST");
        allowedReturns.push_back("NOTFIRST");
        allowedReturns.push_back("FIRSTLAST");
		TCLAP::ValuesConstraint<std::string> allowedReturnVals( allowedReturns );
		
		TCLAP::ValueArg<std::string> returnOfInterestArg("", "return", "The return(s) of interest", false, "ALL", &allowedReturnVals);
		cmd.add( returnOfInterestArg );
        
        TCLAP::SwitchArg minSwitch("","min","Extract only the minimum returns (within the bin and therefore only available for SPD file, not UPD).", false);
        cmd.add( minSwitch );
        
        TCLAP::SwitchArg maxSwitch("","max","Extract only the maximum returns (within the bin and therefore only available for SPD file, not UPD).", false);
        cmd.add( maxSwitch );
        
		TCLAP::ValueArg<std::string> inputFileArg("i","input","The input SPD file.",true,"","String");
		cmd.add( inputFileArg );
        
        TCLAP::ValueArg<std::string> outputFileArg("o","output","The output SPD file.",true,"","String");
		cmd.add( outputFileArg );
        
		cmd.parse( argc, argv );
        
		std::string inputFile = inputFileArg.getValue();
		std::string outputFile = outputFileArg.getValue();
        
        bool classValSet = false;
        boost::uint_fast16_t classVal = 0;
        if(classOfInterestArg.isSet())
        {
            classValSet = true;
            classVal = classOfInterestArg.getValue();
        }

        bool returnValSet = false;
        boost::uint_fast16_t returnVal = 0;
        if(returnOfInterestArg.isSet())
        {
            returnValSet = true;
            
            if(returnOfInterestArg.getValue() == "ALL")
            {
                returnVal = spdlib::SPD_ALL_RETURNS;
            }
            else if(returnOfInterestArg.getValue() == "FIRST")
            {
                returnVal = spdlib::SPD_FIRST_RETURNS;
            }
            else if(returnOfInterestArg.getValue() == "LAST")
            {
                returnVal = spdlib::SPD_LAST_RETURNS;
            }
            else if(returnOfInterestArg.getValue() == "NOTFIRST")
            {
                returnVal = spdlib::SPD_NOTFIRST_RETURNS;
            }
            else if(returnOfInterestArg.getValue() == "FIRSTLAST")
            {
                returnVal = spdlib::SPD_FIRSTLAST_RETURNS;
            }
            else
            {
                std::cout << "WARNING: Did not recognise return type so defaulting to ALL.\n";
                returnVal = spdlib::SPD_ALL_RETURNS;
            }
        }
        
        if(minSwitch.getValue() & maxSwitch.getValue())
        {
            throw spdlib::SPDException("The --min and --max option cannot both be selected.");
        }
        
        bool minMaxSet = false;
        boost::uint_fast16_t highOrLow = spdlib::SPD_SELECT_LOWEST;
        if(minSwitch.getValue())
        {
            minMaxSet = true;
            highOrLow = spdlib::SPD_SELECT_LOWEST;
        }
        else if(maxSwitch.getValue())
        {
            minMaxSet = true;
            highOrLow = spdlib::SPD_SELECT_HIGHEST;
        }
        
        
        spdlib::SPDFile *spdInFile = new spdlib::SPDFile(inputFile);
        spdlib::SPDFileReader reader;
        reader.readHeaderInfo(spdInFile->getFilePath(), spdInFile);
        
        if(spdInFile->getFileType() == spdlib::SPD_UPD_TYPE)
        {
            spdlib::SPDExtractReturnsImportProcess *importProcessor = new spdlib::SPDExtractReturnsImportProcess(outputFile, classValSet, classVal, returnValSet, returnVal);
            
            reader.readAndProcessAllData(spdInFile->getFilePath(), spdInFile, importProcessor);
            importProcessor->completeFileAndClose(spdInFile);
            delete importProcessor;
        }
        else
        {
            spdlib::SPDPulseProcessor *pulseProcessor = new spdlib::SPDExtractReturnsBlockProcess(classValSet, classVal, returnValSet, returnVal, minMaxSet, highOrLow);
            spdlib::SPDSetupProcessPulses processPulses = spdlib::SPDSetupProcessPulses(numOfColsBlockArg.getValue(), numOfRowsBlockArg.getValue(), true);
            processPulses.processPulsesWithOutputSPD(pulseProcessor, spdInFile, outputFile, binSizeArg.getValue());
            delete pulseProcessor;
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
	std::cout << "spdextract - end\n";
}

