/*
 *  main.cpp
 *  spdlib
 *
 *  Created by Pete Bunting on 30/11/2010.
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
#include <vector>

#include <spd/tclap/CmdLine.h>

#include "spd/SPDTextFileUtilities.h"
#include "spd/SPDMergeFiles.h"
#include "spd/SPDException.h"

#include "spd/spd-config.h"

using namespace spdlib;
using namespace TCLAP;

int main (int argc, char * const argv[]) 
{
	std::cout << "spdmerge " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	std::cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	std::cout << "and you are welcome to redistribute it under certain conditions; See\n";
	std::cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	std::cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << std::endl;
	
	try 
	{
		CmdLine cmd("Merge compatable files into a single non-indexed SPD file: spdmerge", ' ', "1.0.0");
		
		std::vector<std::string> allowedInFormats;
		allowedInFormats.push_back("SPD");
		allowedInFormats.push_back("ASCIIPULSEROW");
		allowedInFormats.push_back("ASCII");
		allowedInFormats.push_back("FWF_DAT");
		allowedInFormats.push_back("DECOMPOSED_DAT");
		allowedInFormats.push_back("LAS");
        allowedInFormats.push_back("ASCIIMULTILINE");
		ValuesConstraint<std::string> allowedInFormatsVals( allowedInFormats );
		
		ValueArg<std::string> inFormatArg("i","inputformat","Format of the input file",true,"SPD", &allowedInFormatsVals);
		cmd.add( inFormatArg );
		
		std::vector<std::string> allowedIndexTypes;
		allowedIndexTypes.push_back("FIRST_RETURN");
		allowedIndexTypes.push_back("LAST_RETURN");
		allowedIndexTypes.push_back("START_WAVEFORM");
		allowedIndexTypes.push_back("END_WAVEFORM");
		allowedIndexTypes.push_back("ORIGIN");
		ValuesConstraint<std::string> allowedIndexTypeVals( allowedIndexTypes );
		
		ValueArg<std::string> indexTypeArg("x","indexfield","The location used to index the pulses",false,"FIRST_RETURN", &allowedIndexTypeVals);
		cmd.add( indexTypeArg );
        
        std::vector<std::string> allowedWaveformBitResTypes;
		allowedWaveformBitResTypes.push_back("8BIT");
		allowedWaveformBitResTypes.push_back("16BIT");
		allowedWaveformBitResTypes.push_back("32BIT");
		ValuesConstraint<std::string> allowedWaveformBitResTypesVals( allowedWaveformBitResTypes );
		
		ValueArg<std::string> waveBitResArg("","wavebitres","The bit resolution used for storing the waveform data (Default: 32BIT)",false,"32BIT", &allowedWaveformBitResTypesVals);
		cmd.add( waveBitResArg );
				
		ValueArg<std::string> spatialInArg("p","input_proj","WKT std::string representing the projection of the input file",false,"","std::string");
		cmd.add( spatialInArg );
		
		ValueArg<std::string> spatialOutArg("r","output_proj","WKT std::string representing the projection of the output file",false,"","std::string");
		cmd.add( spatialOutArg );
		
		SwitchArg convertProjSwitch("c","convert_proj","Convert file buffering to disk", false);
		cmd.add( convertProjSwitch );
        
        SwitchArg ignoreChecksSwitch("","ignorechecks","Ignore checks between input files to ensure compatibility", false);
		cmd.add( ignoreChecksSwitch );
        
        SwitchArg sourceIDSwitch("","source","Set source ID for each input file", false);
		cmd.add( sourceIDSwitch );
        
        MultiArg<boost::uint_fast16_t> returnIDsArg("","returnIDs", "Lists the return IDs for the files listed.", false, "uint_fast16_t");
        cmd.add(returnIDsArg);
        
        MultiArg<boost::uint_fast16_t> classesArg("","classes", "Lists the classes for the files listed.", false, "uint_fast16_t");
        cmd.add(classesArg);
        
        ValueArg<std::string> schemaArg("s","schema","A schema for the format of the file being imported (Note, most importers do not require a schema)",false,"", "std::string");
		cmd.add( schemaArg );
		
		UnlabeledMultiArg<std::string> multiFileNames("Files", "File names for the output file and input files", true, "std::string");
		cmd.add( multiFileNames );
		cmd.parse( argc, argv );
		
		std::vector<std::string> fileNames = multiFileNames.getValue();		
		if(fileNames.size() < 2)
		{			
			SPDTextFileUtilities textUtils;
			std::string message = std::string("Two file paths should have been specified (e.g., Input and Output). ") + textUtils.uInt32bittostring(fileNames.size()) + std::string(" were provided.");
			throw SPDException(message);
		}
		
        boost::uint_fast16_t indexType = SPD_FIRST_RETURN;
		if(indexTypeArg.getValue() == "FIRST_RETURN")
		{
			indexType = SPD_FIRST_RETURN;
		}
		else if(indexTypeArg.getValue() == "LAST_RETURN")
		{
			indexType = SPD_FIRST_RETURN;
		}
		else if(indexTypeArg.getValue() == "START_WAVEFORM")
		{
			indexType = SPD_START_OF_RECEIVED_WAVEFORM;
		}
		else if(indexTypeArg.getValue() == "END_WAVEFORM")
		{
			indexType = SPD_END_OF_RECEIVED_WAVEFORM;
		}
		else if(indexTypeArg.getValue() == "ORIGIN")
		{
			indexType = SPD_ORIGIN;
		}
		else 
		{
			throw SPDException("Index type from not recognised.");
		}
		
		std::string outputFile = fileNames.at(0);
		std::cout << "Output File: " << outputFile << std::endl;
		std::cout << "Merging:\n";
		std::vector<std::string> inputFiles;
		for(unsigned int i = 1; i < fileNames.size(); ++i)
		{
			inputFiles.push_back(fileNames.at(i));
			std::cout << fileNames.at(i) << std::endl;
		}
		std::string inputFormat = inFormatArg.getValue();
		std::string inProjFile = spatialInArg.getValue();
		bool convertCoords = convertProjSwitch.getValue();
		std::string outProjFile  = spatialOutArg.getValue();
        
        std::string inProjWKT = "";
		std::string outProjWKT = "";
		
		SPDTextFileUtilities textFileUtils;
		if(inProjFile != "")
		{
			inProjWKT = textFileUtils.readFileToString(inProjFile);
		}
		
		if(outProjFile != "")
		{
			outProjWKT = textFileUtils.readFileToString(outProjFile);
		}
        
		
        std::vector<boost::uint_fast16_t> returnIds;
        bool returnIDsSet = false;
        if(returnIDsArg.isSet())
        {
            returnIDsSet = true;
            returnIds = returnIDsArg.getValue();
            if(returnIds.size() != (fileNames.size()-1))
            {
                throw SPDException("The number of inputted return IDs needs to equal the number of input files.");
            }
        }

        std::vector<boost::uint_fast16_t> classesValues;
        bool classesSet = false;
        if(classesArg.isSet())
        {
            classesSet = true;
            classesValues = classesArg.getValue();
            if(classesValues.size() != (fileNames.size()-1))
            {
                throw SPDException("The number of inputted classes needs to equal the number of input files.");
            }
        }
        
        boost::uint_fast16_t waveBitRes = SPD_32_BIT_WAVE;
		if(waveBitResArg.isSet())
		{
			if(waveBitResArg.getValue() == "8BIT")
			{
				waveBitRes = SPD_8_BIT_WAVE;
			}
			else if(waveBitResArg.getValue() == "16BIT")
			{
				waveBitRes = SPD_16_BIT_WAVE;
			}
			else if(waveBitResArg.getValue() == "32BIT")
			{
				waveBitRes = SPD_32_BIT_WAVE;
			}
			else 
			{
				throw SPDException("Waveform bit resolution option was not recognised.");
			}
		}		

        SPDMergeFiles merge;
        merge.mergeToUPD(inputFiles, outputFile, inputFormat, schemaArg.getValue(), inProjWKT, convertCoords, outProjWKT, indexType, sourceIDSwitch.getValue(), returnIDsSet, returnIds, classesSet, classesValues, ignoreChecksSwitch.getValue(), waveBitRes);
	}
	catch (ArgException &e) 
	{
        std::cerr << "Parse Error: " << e.what() << std::endl;
	}
	catch(SPDException &e)
	{
        std::cerr << "Error: " << e.what() << std::endl;
	}
	
}

