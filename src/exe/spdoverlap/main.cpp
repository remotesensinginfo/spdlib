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
#include <fstream>
#include <stdexcept>

#include <spd/tclap/CmdLine.h>

#include "spd/SPDTextFileUtilities.h"
#include "spd/SPDFile.h"
#include "spd/SPDFileReader.h"
#include "spd/SPDException.h"

#include "spd/spd-config.h"

using namespace std;
using namespace spdlib;
using namespace TCLAP;

int main (int argc, char * const argv[]) 
{    
    cout << "spdoverlap " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	cout << "and you are welcome to redistribute it under certain conditions; See\n";
	cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << endl;
	
	try 
	{
		CmdLine cmd("Calculate the overlap between UPD and SPD files: spdoverlap", ' ', "1.0.0");
		
        SwitchArg cartesianSwitch("c","cartesian","Find cartesian overlap.", false);
        SwitchArg sphericalSwitch("s","spherical","Find spherical overlap.", false);
        vector<Arg*> arguments;
        arguments.push_back(&cartesianSwitch);
        arguments.push_back(&sphericalSwitch);
        cmd.xorAdd(arguments);
        
        SwitchArg print2ConsoleSwitch("p","print","Print overlapping information to console.", false);
        cmd.add( print2ConsoleSwitch );
        
		UnlabeledMultiArg<string> multiFileNames("Files", "File names for the output (if required) and input files", true, "string");
		cmd.add( multiFileNames );
		cmd.parse( argc, argv );
		
		vector<string> fileNames = multiFileNames.getValue();		
		if(fileNames.size() < 2)
		{
            for(unsigned int i = 0; i < fileNames.size(); ++i)
            {
                cout << i << ": " << fileNames.at(i) << endl;
            }
            
			SPDTextFileUtilities textUtils;
			string message = string("At least two file paths should have been specified. ") + textUtils.uInt32bittostring(fileNames.size()) + string(" were provided.");
			throw SPDException(message);
		}
		
        bool outToConsole = print2ConsoleSwitch.getValue();
        string outputTextFile = "";
        uint_fast16_t numOfFiles = fileNames.size();
        uint_fast16_t startIdx = 0;
        if(!outToConsole)
        {
            outputTextFile = fileNames.at(0);
            ++startIdx;
            --numOfFiles;
        }
        
        SPDFileReader spdReader;
        
        SPDFile **spdFiles = new SPDFile*[numOfFiles];
        uint_fast16_t outIdx = 0;
        for(uint_fast16_t i = startIdx; i < fileNames.size(); ++i)
        {
            spdFiles[outIdx] = new SPDFile(fileNames.at(i));
            spdReader.readHeaderInfo(fileNames.at(i), spdFiles[outIdx]);
            ++outIdx;
        }
        
        double *overlap = NULL;
        
        SPDFileProcessingUtilities spdUtils;
        if(cartesianSwitch.getValue())
        {
            overlap = spdUtils.calcCartesainOverlap(spdFiles, numOfFiles);
        }
        else if(sphericalSwitch.getValue())
        {
            overlap = spdUtils.calcSphericalOverlap(spdFiles, numOfFiles);
        }
        else
        {
            throw SPDProcessingException("Option was not recognised, need to seleted either spherical or cartesian.");
        }
        
        for(uint_fast16_t i = 0; i < numOfFiles; ++i)
        {
            delete spdFiles[i];
        }
        delete[] spdFiles;
        
        // Output:
        /*
         * overlap[0] = min X
         * overlap[1] = max X
         * overlap[2] = min Y
         * overlap[3] = max Y
         * overlap[4] = min Z
         * overlap[5] = max Z
         */
        //OR
        /*
         * overlap[0] = min Azimuth
         * overlap[1] = max Azimuth
         * overlap[2] = min Zenith
         * overlap[3] = max Zenith
         * overlap[4] = min Range
         * overlap[5] = max Range
         */
        if(outToConsole)
        {
            if(cartesianSwitch.getValue())
            {
                cout << "X: [" << overlap[0] << "," << overlap[1] << "]\n";
                cout << "Y: [" << overlap[2] << "," << overlap[3] << "]\n";
                cout << "Z: [" << overlap[4] << "," << overlap[5] << "]\n";
            }
            else if(sphericalSwitch.getValue())
            {
                cout << "Azimuth: [" << overlap[0] << "," << overlap[1] << "]\n";
                cout << "Zenith: [" << overlap[2] << "," << overlap[3] << "]\n";
                cout << "Range: [" << overlap[4] << "," << overlap[5] << "]\n";
            }
        }
        else
        {
            ofstream outASCIIFile;
            outASCIIFile.open(outputTextFile.c_str(), ios::out | ios::trunc);
            
            if(!outASCIIFile.is_open())
            {               
                string message = string("Could not open file ") + outputTextFile;
                throw SPDException(message);
            }            
            outASCIIFile.precision(12);
            
            if(cartesianSwitch.getValue())
            {
                outASCIIFile << "#Cartesian" << endl;
            }
            else if(sphericalSwitch.getValue())
            {
                outASCIIFile << "#Spherical" << endl;
            }
            outASCIIFile << overlap[0] << endl;
            outASCIIFile << overlap[1] << endl;
            outASCIIFile << overlap[2] << endl;
            outASCIIFile << overlap[3] << endl;
            outASCIIFile << overlap[4] << endl;
            outASCIIFile << overlap[5] << endl;
            outASCIIFile.flush();
            outASCIIFile.close();
        }
        delete[] overlap;
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

