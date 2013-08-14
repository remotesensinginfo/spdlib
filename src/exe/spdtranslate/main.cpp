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
#include "spd/SPDConvertFormats.h"
#include "spd/SPDException.h"

#include "spd/spd-config.h"

int main (int argc, char * const argv[]) 
{
    std::cout.precision(12);
    
    std::cout << "spdtranslate " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	std::cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	std::cout << "and you are welcome to redistribute it under certain conditions; See\n";
	std::cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	std::cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << std::endl;
	
	try 
	{
        TCLAP::CmdLine cmd("Convert between file formats: spdtranslate", ' ', "1.0.0");
		
		std::vector<std::string> allowedInFormats;
		allowedInFormats.push_back("SPD");
		allowedInFormats.push_back("ASCIIPULSEROW");
        allowedInFormats.push_back("ASCII");
		allowedInFormats.push_back("FWF_DAT");
		allowedInFormats.push_back("DECOMPOSED_DAT");
		allowedInFormats.push_back("LAS");
        allowedInFormats.push_back("LASNP");
        allowedInFormats.push_back("LASSTRICT");
		allowedInFormats.push_back("DECOMPOSED_COO");
        allowedInFormats.push_back("ASCIIMULTILINE");
		TCLAP::ValuesConstraint<std::string> allowedInFormatsVals( allowedInFormats );
		
		TCLAP::ValueArg<std::string> inFormatArg("","if","Format of the input file (Default SPD)",true,"SPD", &allowedInFormatsVals);
		cmd.add( inFormatArg );
		
		std::vector<std::string> allowedOutFormats;
		allowedOutFormats.push_back("SPD");
		allowedOutFormats.push_back("UPD");
		allowedOutFormats.push_back("ASCII");
		allowedOutFormats.push_back("LAS");
        allowedOutFormats.push_back("LAZ");
		TCLAP::ValuesConstraint<std::string> allowedOutFormatsVals( allowedOutFormats );
		
		TCLAP::ValueArg<std::string> outFormatArg("","of","Format of the output file (Default SPD)",true,"SPD", &allowedOutFormatsVals);
		cmd.add( outFormatArg );
		
		std::vector<std::string> allowedIndexTypes;
		allowedIndexTypes.push_back("FIRST_RETURN");
		allowedIndexTypes.push_back("LAST_RETURN");
		allowedIndexTypes.push_back("START_WAVEFORM");
		allowedIndexTypes.push_back("END_WAVEFORM");
		allowedIndexTypes.push_back("ORIGIN");
		allowedIndexTypes.push_back("MAX_INTENSITY");
		allowedIndexTypes.push_back("UNCHANGED");
		TCLAP::ValuesConstraint<std::string> allowedIndexTypeVals( allowedIndexTypes );
		
		TCLAP::ValueArg<std::string> indexTypeArg("x","indexfield","The location used to index the pulses (Default: UNCHANGED)",false,"UNCHANGED", &allowedIndexTypeVals);
		cmd.add( indexTypeArg );

        std::vector<std::string> allowedWaveformBitResTypes;
		allowedWaveformBitResTypes.push_back("8BIT");
		allowedWaveformBitResTypes.push_back("16BIT");
		allowedWaveformBitResTypes.push_back("32BIT");
		TCLAP::ValuesConstraint<std::string> allowedWaveformBitResTypesVals( allowedWaveformBitResTypes );
		
		TCLAP::ValueArg<std::string> waveBitResArg("","wavebitres","The bit resolution used for storing the waveform data (Default: 32BIT)",false,"32BIT", &allowedWaveformBitResTypesVals);
		cmd.add( waveBitResArg );
        
        TCLAP::SwitchArg defineSphericalSwitch("","spherical","Index the pulses using a spherical coordinate system", false);
        cmd.add( defineSphericalSwitch );
        TCLAP::SwitchArg definePolarSwitch("","polar","Index the pulses using a polar coordinate system", false);
        cmd.add( definePolarSwitch );
        TCLAP::SwitchArg defineScanSwitch("","scan","Index the pulses using a scan coordinate system", false);
        cmd.add( defineScanSwitch );
		
		TCLAP::ValueArg<std::string> tempDirPathArg("t","temppath","A path were temporary files can be written too",false,"", "string");
		cmd.add( tempDirPathArg );
        
        TCLAP::ValueArg<std::string> schemaArg("s","schema","A schema for the format of the file being imported (Note, most importers do not require a schema)",false,"", "string");
		cmd.add( schemaArg );
		
		TCLAP::ValueArg<boost::uint_fast16_t> numOfRowsInTileArg("r","numofrows","Number of rows within a tile (Default 25)",false,25,"unsigned int");
		cmd.add( numOfRowsInTileArg );
        
        TCLAP::ValueArg<boost::uint_fast16_t> numOfColsInTileArg("c","numofcols","Number of columns within a tile (Default 0), using this option generats a non-sequencial SPD file.",false,0,"unsigned int");
		cmd.add( numOfColsInTileArg );
		
		TCLAP::ValueArg<float> binSizeArg("b","binsize","Bin size for SPD file index (Default 1)",false,1,"float");
		cmd.add( binSizeArg );
		
		TCLAP::ValueArg<std::string> spatialInArg("","input_proj","WKT string representing the projection of the input file",false,"","string");
		cmd.add( spatialInArg );
		
		TCLAP::ValueArg<std::string> spatialOutArg("","output_proj","WKT string representing the projection of the output file",false,"","string");
		cmd.add( spatialOutArg );
		
		TCLAP::SwitchArg convertProjSwitch("","convert_proj","Convert file buffering to disk", false);
		cmd.add( convertProjSwitch );
        
        TCLAP::SwitchArg keepTmpFilesSwitch("","keeptemp","Keep the tempory files generated during the conversion.", false);
		cmd.add( keepTmpFilesSwitch );
        
        TCLAP::SwitchArg defineTLSwitch("","defineTL","Define the top left (TL) coordinate for the SPD file index", false);
		cmd.add( defineTLSwitch );
        
        TCLAP::ValueArg<double> tlXArg("","tlx","Top left X coordinate for defining the SPD file index.",false,0,"double");
		cmd.add( tlXArg );
        
        TCLAP::ValueArg<double> tlYArg("","tly","Top left Y coordinate for defining the SPD file index.",false,0,"double");
		cmd.add( tlYArg );

		TCLAP::SwitchArg defineOriginSwitch("","defineOrigin","Define the origin coordinate for the SPD.", false);
		cmd.add( defineOriginSwitch );
        
        TCLAP::ValueArg<double> originXArg("","Ox","Origin X coordinate.",false,0,"double");
		cmd.add( originXArg );
        
        TCLAP::ValueArg<double> originYArg("","Oy","Origin Y coordinate",false,0,"double");
		cmd.add( originYArg );
        
        TCLAP::ValueArg<float> originZArg("","Oz","Origin Z coordinate",false,0,"float");
		cmd.add( originZArg );
        
        TCLAP::ValueArg<float> waveNoiseThresholdArg("","wavenoise","Waveform noise threshold (Default 0)",false,0,"float");
		cmd.add( waveNoiseThresholdArg );
        
		TCLAP::ValueArg<boost::uint_fast16_t> pointVersionArg("","pointversion","Specify the point version to be used within the SPD file (Default: 2)",false ,2 ,"unsigned int");
		cmd.add( pointVersionArg );
        
        TCLAP::ValueArg<boost::uint_fast16_t> pulseVersionArg("","pulseversion","Specify the pulse version to be used within the SPD file (Default: 2)",false ,2 ,"unsigned int");
		cmd.add( pulseVersionArg );
		
		//TCLAP::SwitchArg diskTempFilesSwitch("f","usetmp","Convert file buffering to disk using temporary files", false);
		//cmd.add( diskTempFilesSwitch );
		
		TCLAP::ValueArg<std::string> inputFileArg("i","input","The input file.",true,"","String");
		cmd.add( inputFileArg );
        
        TCLAP::ValueArg<std::string> outputFileArg("o","output","The output file.",true,"","String");
		cmd.add( outputFileArg );
        
		cmd.parse( argc, argv );
		
		boost::uint_fast16_t indexType = spdlib::SPD_IDX_UNCHANGED;
		if(indexTypeArg.isSet())
		{
			if(indexTypeArg.getValue() == "FIRST_RETURN")
			{
				indexType = spdlib::SPD_FIRST_RETURN;
			}
			else if(indexTypeArg.getValue() == "LAST_RETURN")
			{
				indexType = spdlib::SPD_LAST_RETURN;
			}
			else if(indexTypeArg.getValue() == "START_WAVEFORM")
			{
				indexType = spdlib::SPD_START_OF_RECEIVED_WAVEFORM;
			}
			else if(indexTypeArg.getValue() == "END_WAVEFORM")
			{
				indexType = spdlib::SPD_END_OF_RECEIVED_WAVEFORM;
			}
			else if(indexTypeArg.getValue() == "ORIGIN")
			{
				indexType = spdlib::SPD_ORIGIN;
			}
			else if(indexTypeArg.getValue() == "MAX_INTENSITY")
			{
				indexType = spdlib::SPD_MAX_INTENSITY;
			}
			else if(indexTypeArg.getValue() == "UNCHANGED")
			{
				indexType = spdlib::SPD_IDX_UNCHANGED;
			}
			else 
			{
				throw spdlib::SPDException("Index type from not recognised.");
			}
		}
        
        boost::uint_fast16_t waveBitRes = spdlib::SPD_32_BIT_WAVE;
		if(waveBitResArg.isSet())
		{
			if(waveBitResArg.getValue() == "8BIT")
			{
				waveBitRes = spdlib::SPD_8_BIT_WAVE;
			}
			else if(waveBitResArg.getValue() == "16BIT")
			{
				waveBitRes = spdlib::SPD_16_BIT_WAVE;
			}
			else if(waveBitResArg.getValue() == "32BIT")
			{
				waveBitRes = spdlib::SPD_32_BIT_WAVE;
			}
			else 
			{
				throw spdlib::SPDException("Waveform bit resolution option was not recognised.");
			}
		}		
		
		std::string inputFile = inputFileArg.getValue();
		std::string outputFile = outputFileArg.getValue();
		std::string inputFormat = inFormatArg.getValue();
		std::string outputFormat = outFormatArg.getValue();
		float binSize = binSizeArg.getValue();
		std::string inProjFile = spatialInArg.getValue();
		bool convertCoords = convertProjSwitch.getValue();
		std::string outProjFile  = spatialOutArg.getValue();
		boost::uint_fast16_t numOfRows = numOfRowsInTileArg.getValue();
        boost::uint_fast16_t numOfCols = numOfColsInTileArg.getValue();
		std::string tempdir = tempDirPathArg.getValue();
        std::string schema = schemaArg.getValue();
		bool keepTempFiles = keepTmpFilesSwitch.getValue();
        boost::uint_fast16_t pointVersion = pointVersionArg.getValue();
        boost::uint_fast16_t pulseVersion = pulseVersionArg.getValue();
        
        if((pointVersion == 0) | (pointVersion > 2))
        {
            throw spdlib::SPDException("Point version can only have a value of 1 or 2.");
        }
        
        if((pulseVersion == 0) | (pulseVersion > 2))
        {
            throw spdlib::SPDException("Pulse version can only have a value of 1 or 2.");
        }
        
		std::string inProjWKT = "";
		std::string outProjWKT = "";
		
        spdlib::SPDTextFileUtilities textFileUtils;
		if(inProjFile != "")
		{
			inProjWKT = textFileUtils.readFileToString(inProjFile);
		}
		
		if(outProjFile != "")
		{
			outProjWKT = textFileUtils.readFileToString(outProjFile);
		}
		/*		
		std::cout << "inProjFile = " << inProjFile << std::endl;
		std::cout << "outProjFile = " << outProjFile << std::endl;
		std::cout << "inProjWKT = " << inProjWKT << std::endl;
		std::cout << "outProjWKT = " << outProjWKT << std::endl;
		*/
		spdlib::SPDConvertFormats convert;
		if(tempDirPathArg.isSet())
		{
            if(outputFormat != "SPD")
			{
				throw spdlib::SPDException("The temporary outputs option (i.e., --temppath) is only required when converting to an indexed format (i.e., SPD). Either select SPD as your output format or remove these options.");
			}
			
			if(tempdir == "")
			{
				throw spdlib::SPDException("A temporary path needs to be provided.");
			}
            
            if(numOfRows == 0)
			{
				throw spdlib::SPDException("The number of rows within a tile needs to greater than zero.");
			}
            
            if(numOfCols > 0)
            {
                convert.convertToSPDUsingBlockTiles(inputFile, outputFile, inputFormat, schema, binSize, inProjWKT, convertCoords, outProjWKT, indexType, tempdir, numOfRows, numOfCols, defineTLSwitch.getValue(), tlXArg.getValue(), tlYArg.getValue(), defineOriginSwitch.getValue(), originXArg.getValue(), originYArg.getValue(), originZArg.getValue(), defineSphericalSwitch.getValue(), definePolarSwitch.getValue(), defineScanSwitch.getValue(), waveNoiseThresholdArg.getValue(), waveBitRes, keepTempFiles, pointVersion, pulseVersion);
            }
            else
            {
                convert.convertToSPDUsingRowTiles(inputFile, outputFile, inputFormat, schema, binSize, inProjWKT, convertCoords, outProjWKT, indexType, tempdir, numOfRows, defineTLSwitch.getValue(), tlXArg.getValue(), tlYArg.getValue(), defineOriginSwitch.getValue(), originXArg.getValue(), originYArg.getValue(), originZArg.getValue(), defineSphericalSwitch.getValue(), definePolarSwitch.getValue(), defineScanSwitch.getValue(), waveNoiseThresholdArg.getValue(), waveBitRes, keepTempFiles, pointVersion, pulseVersion);
            }

		}
		else 
		{
            if(outputFormat == "SPD")
            {
                outputFormat = "SPD-SEQ";
            }
			convert.convertInMemory(inputFile, outputFile, inputFormat, schema, outputFormat, binSize, inProjWKT, convertCoords, outProjWKT, indexType, defineTLSwitch.getValue(), tlXArg.getValue(), tlYArg.getValue(), defineOriginSwitch.getValue(), originXArg.getValue(), originYArg.getValue(), originZArg.getValue(), defineSphericalSwitch.getValue(), definePolarSwitch.getValue(), defineScanSwitch.getValue(), waveNoiseThresholdArg.getValue(), waveBitRes, pointVersion, pulseVersion);
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
    std::cout << "spdtranslate - end\n";
}

