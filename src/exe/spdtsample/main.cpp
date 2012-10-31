/*
 *  main.cpp
 *  spdlib
 *
 *  Created by Pete Bunting on 31/10/2012.
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
#include "spd/SPDSampleInTime.h"
#include "spd/SPDException.h"
#include "spd/SPDFileReader.h"

#include "spd/spd-config.h"

using namespace std;
using namespace spdlib;
using namespace TCLAP;

int main (int argc, char * const argv[]) 
{
	cout << "spdsubset " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	cout << "and you are welcome to redistribute it under certain conditions; See\n";
	cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << endl;
	
	try 
	{
		CmdLine cmd("Sample the pulses using time, i.e., to thin the pulses: spdtsample", ' ', "1.0.0");

		ValueArg<boost::uint_fast16_t> sampleArg("s","sample","Take every \'s\' pulse",false,1,"boost::uint_fast16_t");
		cmd.add( sampleArg );		
		
		UnlabeledMultiArg<string> multiFileNames("Files", "File names for the input and output files", true, "string");
		cmd.add( multiFileNames );
		cmd.parse( argc, argv );
		
		vector<string> fileNames = multiFileNames.getValue();		
		if(fileNames.size() != 2)
		{
			SPDTextFileUtilities textUtils;
			string message = string("Two file paths should have been specified (e.g., Input and Output). ") + textUtils.uInt32bittostring(fileNames.size()) + string(" were provided.");
			throw SPDException(message);
		}
        
        boost::uint_fast16_t tSample = sampleArg.getValue();
        if(tSample == 0)
        {
            tSample = 1;
        }
        
        string inputFile = fileNames.at(0);
        string outputFile = fileNames.at(1);
        
        SPDFile *inSPDFile = new SPDFile(inputFile);
        SPDFileReader spdReader = SPDFileReader();
        spdReader.readHeaderInfo(inputFile, inSPDFile);
        
        if(inSPDFile->getFileType() != SPD_UPD_TYPE)
        {
            throw SPDException("The specified input file must be of time UPD in time order.");
        }
        
        SPDSampleInTime *processor = new SPDSampleInTime(outputFile, tSample);
        
        SPDFile *spdFile = new SPDFile(inputFile);
        spdReader.readAndProcessAllData(inputFile, spdFile, processor);
        processor->completeFileAndClose();
        
        delete spdFile;
        delete processor;
        
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

