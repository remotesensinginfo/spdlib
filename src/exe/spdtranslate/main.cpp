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

using namespace std;
using namespace spdlib;
using namespace TCLAP;

int main (int argc, char * const argv[]) 
{
	cout << "spdtranslate " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	cout << "and you are welcome to redistribute it under certain conditions; See\n";
	cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << endl;
	
	try 
	{
		CmdLine cmd("Convert between file formats: spdtranslate", ' ', "1.0.0");
		
		vector<string> allowedInFormats;
		allowedInFormats.push_back("SPD");
		allowedInFormats.push_back("ASCIIPULSEROW");
        allowedInFormats.push_back("ASCII");
		allowedInFormats.push_back("FWF_DAT");
		allowedInFormats.push_back("DECOMPOSED_DAT");
		allowedInFormats.push_back("LAS");
		allowedInFormats.push_back("DECOMPOSED_COO");
        allowedInFormats.push_back("ASCIIMULTILINE");
		ValuesConstraint<string> allowedInFormatsVals( allowedInFormats );
		
		ValueArg<string> inFormatArg("i","inputformat","Format of the input file (Default SPD)",true,"SUPD", &allowedInFormatsVals);
		cmd.add( inFormatArg );
		
		vector<string> allowedOutFormats;
		allowedOutFormats.push_back("SPD");
		allowedOutFormats.push_back("UPD");
		allowedOutFormats.push_back("ASCII");
		allowedOutFormats.push_back("LAS");
		ValuesConstraint<string> allowedOutFormatsVals( allowedOutFormats );
		
		ValueArg<string> outFormatArg("o","outputformat","Format of the output file (Default SPD)",true,"SPD", &allowedOutFormatsVals);
		cmd.add( outFormatArg );
		
		vector<string> allowedIndexTypes;
		allowedIndexTypes.push_back("FIRST_RETURN");
		allowedIndexTypes.push_back("LAST_RETURN");
		allowedIndexTypes.push_back("START_WAVEFORM");
		allowedIndexTypes.push_back("END_WAVEFORM");
		allowedIndexTypes.push_back("ORIGIN");
		allowedIndexTypes.push_back("MAX_INTENSITY");
		allowedIndexTypes.push_back("UNCHANGED");
		ValuesConstraint<string> allowedIndexTypeVals( allowedIndexTypes );
		
		ValueArg<string> indexTypeArg("x","indexfield","The location used to index the pulses (Default: UNCHANGED)",false,"UNCHANGED", &allowedIndexTypeVals);
		cmd.add( indexTypeArg );

        vector<string> allowedWaveformBitResTypes;
		allowedWaveformBitResTypes.push_back("8BIT");
		allowedWaveformBitResTypes.push_back("16BIT");
		allowedWaveformBitResTypes.push_back("32BIT");
		ValuesConstraint<string> allowedWaveformBitResTypesVals( allowedWaveformBitResTypes );
		
		ValueArg<string> waveBitResArg("","wavebitres","The bit resolution used for storing the waveform data (Default: 32BIT)",false,"32BIT", &allowedWaveformBitResTypesVals);
		cmd.add( waveBitResArg );
        
        SwitchArg defineSphericalSwitch("","spherical","Index the pulses using a spherical coordinate system", false);
        cmd.add( defineSphericalSwitch );
        SwitchArg definePolarSwitch("","polar","Index the pulses using a polar coordinate system", false);
        cmd.add( definePolarSwitch );
        SwitchArg defineScanSwitch("","scan","Index the pulses using a scan coordinate system", false);
        cmd.add( defineScanSwitch );
		
		ValueArg<string> tempDirPathArg("t","temppath","A path were temporary files can be written too",false,"", "string");
		cmd.add( tempDirPathArg );
        
        ValueArg<string> schemaArg("s","schema","A schema for the format of the file being imported (Note, most importers do not require a schema)",false,"", "string");
		cmd.add( schemaArg );
		
		ValueArg<boost::uint_fast16_t> numOfRowsInTileArg("r","numofrows","Number of rows within a tile (Default 25)",false,25,"unsigned int");
		cmd.add( numOfRowsInTileArg );
        
        ValueArg<boost::uint_fast16_t> numOfColsInTileArg("c","numofcols","Number of columns within a tile (Default 0), using this option generats a non-sequencial SPD file.",false,0,"unsigned int");
		cmd.add( numOfColsInTileArg );
		
		ValueArg<float> binSizeArg("b","binsize","Bin size for SPD file index (Default 1)",false,1,"float");
		cmd.add( binSizeArg );
		
		ValueArg<string> spatialInArg("","input_proj","WKT string representing the projection of the input file",false,"","string");
		cmd.add( spatialInArg );
		
		ValueArg<string> spatialOutArg("","output_proj","WKT string representing the projection of the output file",false,"","string");
		cmd.add( spatialOutArg );
		
		SwitchArg convertProjSwitch("","convert_proj","Convert file buffering to disk", false);
		cmd.add( convertProjSwitch );
        
        SwitchArg keepTmpFilesSwitch("","keeptemp","Keep the tempory files generated during the conversion.", false);
		cmd.add( keepTmpFilesSwitch );
        
        SwitchArg defineTLSwitch("","defineTL","Define the top left (TL) coordinate for the SPD file index", false);
		cmd.add( defineTLSwitch );
        
        ValueArg<double> tlXArg("","tlx","Top left X coordinate for defining the SPD file index.",false,0,"double");
		cmd.add( tlXArg );
        
        ValueArg<double> tlYArg("","tly","Top left Y coordinate for defining the SPD file index.",false,0,"double");
		cmd.add( tlYArg );

		SwitchArg defineOriginSwitch("","defineOrigin","Define the origin coordinate for the SPD.", false);
		cmd.add( defineOriginSwitch );
        
        ValueArg<double> originXArg("","Ox","Origin X coordinate.",false,0,"double");
		cmd.add( originXArg );
        
        ValueArg<double> originYArg("","Oy","Origin Y coordinate",false,0,"double");
		cmd.add( originYArg );
        
        ValueArg<float> originZArg("","Oz","Origin Z coordinate",false,0,"float");
		cmd.add( originZArg );
        
        ValueArg<float> waveNoiseThresholdArg("","wavenoise","Waveform noise threshold (Default 0)",false,0,"float");
		cmd.add( waveNoiseThresholdArg );
        
		ValueArg<boost::uint_fast16_t> pointVersionArg("","pointversion","Specify the point version to be used within the SPD file (Default: 2)",false ,2 ,"unsigned int");
		cmd.add( pointVersionArg );
        
        ValueArg<boost::uint_fast16_t> pulseVersionArg("","pulseversion","Specify the pulse version to be used within the SPD file (Default: 2)",false ,2 ,"unsigned int");
		cmd.add( pulseVersionArg );
		
		SwitchArg diskTempFilesSwitch("f","usetmp","Convert file buffering to disk using temporary files", false);
		cmd.add( diskTempFilesSwitch );
		
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
		
		boost::uint_fast16_t indexType = SPD_IDX_UNCHANGED;
		if(indexTypeArg.isSet())
		{
			if(indexTypeArg.getValue() == "FIRST_RETURN")
			{
				indexType = SPD_FIRST_RETURN;
			}
			else if(indexTypeArg.getValue() == "LAST_RETURN")
			{
				indexType = SPD_LAST_RETURN;
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
			else if(indexTypeArg.getValue() == "MAX_INTENSITY")
			{
				indexType = SPD_MAX_INTENSITY;
			}
			else if(indexTypeArg.getValue() == "UNCHANGED")
			{
				indexType = SPD_IDX_UNCHANGED;
			}
			else 
			{
				throw SPDException("Index type from not recognised.");
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
		
		string inputFile = fileNames.at(0);
		string outputFile = fileNames.at(1);
		string inputFormat = inFormatArg.getValue();
		string outputFormat = outFormatArg.getValue();
		float binSize = binSizeArg.getValue();
		string inProjFile = spatialInArg.getValue();
		bool convertCoords = convertProjSwitch.getValue();
		string outProjFile  = spatialOutArg.getValue();
		boost::uint_fast16_t numOfRows = numOfRowsInTileArg.getValue();
        boost::uint_fast16_t numOfCols = numOfColsInTileArg.getValue();
		string tempdir = tempDirPathArg.getValue();
        string schema = schemaArg.getValue();
		bool keepTempFiles = keepTmpFilesSwitch.getValue();
        boost::uint_fast16_t pointVersion = pointVersionArg.getValue();
        boost::uint_fast16_t pulseVersion = pulseVersionArg.getValue();
        
        if((pointVersion == 0) | (pointVersion > 2))
        {
            throw SPDException("Point version can only have a value of 1 or 2.");
        }
        
        if((pulseVersion == 0) | (pulseVersion > 2))
        {
            throw SPDException("Pulse version can only have a value of 1 or 2.");
        }
        
		string inProjWKT = "";
		string outProjWKT = "";
		
		SPDTextFileUtilities textFileUtils;
		if(inProjFile != "")
		{
			inProjWKT = textFileUtils.readFileToString(inProjFile);
		}
		
		if(outProjFile != "")
		{
			outProjWKT = textFileUtils.readFileToString(outProjFile);
		}
		/*		
		cout << "inProjFile = " << inProjFile << endl;
		cout << "outProjFile = " << outProjFile << endl;
		cout << "inProjWKT = " << inProjWKT << endl;
		cout << "outProjWKT = " << outProjWKT << endl;
		*/
		SPDConvertFormats convert;
		if(diskTempFilesSwitch.getValue())
		{
            if(outputFormat != "SPD")
			{
				throw SPDException("This converter only supports conversion to the SPD format.");
			}
			
			if(tempdir == "")
			{
				throw SPDException("A temporary path needs to be provided.");
			}
            
            if(numOfRows == 0)
			{
				throw SPDException("The number of rows within a tile needs to greater than zero.");
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
	catch (ArgException &e) 
	{
		cerr << "Parse Error: " << e.what() << endl;
	}
	catch(SPDException &e)
	{
		cerr << "Error: " << e.what() << endl;
	}
	
}

