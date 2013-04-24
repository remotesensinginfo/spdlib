/*
 *  main.cpp
 *  spdlib
 *
 *  Created by Pete Bunting on 27/03/2012.
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

#include "spd/SPDException.h"
#include "spd/SPDCalcMetrics.h"

#include "spd/spd-config.h"

using namespace std;
using namespace spdlib;
using namespace TCLAP;

int main (int argc, char * const argv[]) 
{
	cout << "spdmetrics " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	cout << "and you are welcome to redistribute it under certain conditions; See\n";
	cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << endl;
	
	try 
	{
		CmdLine cmd("Calculate metrics : spdmetrics", ' ', "1.0.0");
		
        SwitchArg imageStatsSwitch("i","image","Run metrics with image output", false);		
		SwitchArg vectorStatsSwitch("v","vector","Run metrics with vector output", false);
        SwitchArg asciiStatsSwitch("a","ascii","Run metrics with ASCII output", false);
        
        vector<Arg*> arguments;
        arguments.push_back(&imageStatsSwitch);
        arguments.push_back(&vectorStatsSwitch);
        arguments.push_back(&asciiStatsSwitch);
        
        cmd.xorAdd(arguments);
        
        ValueArg<boost::uint_fast32_t> numOfRowsBlockArg("r","blockrows","Number of rows within a block (Default 100)",false,100,"unsigned int");
		cmd.add( numOfRowsBlockArg );
        
        ValueArg<boost::uint_fast32_t> numOfColsBlockArg("c","blockcols","Number of columns within a block (Default 0) - Note values greater than 1 result in a non-sequencial SPD file.",false,0,"unsigned int");
		cmd.add( numOfColsBlockArg );
        
        ValueArg<float> binSizeArg("b","binsize","Bin size for processing and output image (Default 0) - Note 0 will use the native SPD file bin size.",false,0,"float");
		cmd.add( binSizeArg );
        
        ValueArg<string> imgFormatArg("f","format","Image format (GDAL driver string), Default is ENVI.",false,"ENVI","string");
		cmd.add( imgFormatArg );
        
		UnlabeledMultiArg<string> multiFileNames("File", "File names for the input files", false, "string");
		cmd.add( multiFileNames );
		cmd.parse( argc, argv );
		
		vector<string> fileNames = multiFileNames.getValue();		
		if((fileNames.size() == 3) & imageStatsSwitch.getValue())
		{
            string inXMLFilePath = fileNames.at(0);
            string inSPDFilePath = fileNames.at(1);
            string outFilePath = fileNames.at(2);
            
            SPDCalcMetrics calcMetrics;
            calcMetrics.calcMetricToImage(inXMLFilePath, inSPDFilePath, outFilePath, numOfColsBlockArg.getValue(), numOfRowsBlockArg.getValue(), binSizeArg.getValue(), imgFormatArg.getValue()); 
		}
        else if(fileNames.size() == 4)
		{
            string inXMLFilePath = fileNames.at(0);
            string inSPDFilePath = fileNames.at(1);
            string inVectorFile = fileNames.at(2);
            string outFilePath = fileNames.at(3);
            
            if(vectorStatsSwitch.getValue())
            {
                SPDCalcMetrics calcMetrics;
                calcMetrics.calcMetricToVectorShp(inXMLFilePath, inSPDFilePath, inVectorFile, outFilePath, true, true);
            }
            else if(asciiStatsSwitch.getValue())
            {
                SPDCalcMetrics calcMetrics;
                calcMetrics.calcMetricForVector2ASCII(inXMLFilePath, inSPDFilePath, inVectorFile, outFilePath);
            }
            else
            {
                throw SPDException("Option not recognised. Must select image, vector or ASCII noting that image only accepts 3 input files.");
            }
            
		}
        else
        {
            cout << "ERROR: Only 3 files can be provided for image output while 4 files are required for vector or ASCII output\n";
            for(unsigned int i = 0; i < fileNames.size(); ++i)
			{
                cout << i << ":\t" << fileNames.at(i) << endl;
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
	std::cout << "spdmetrics - end\n";
}

