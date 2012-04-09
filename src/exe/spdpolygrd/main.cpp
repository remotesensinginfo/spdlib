/*
 *  main.cpp
 *  spdlib
 *
 *  Created by Pete Bunting on 04/03/2012.
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
#include "spd/SPDFile.h"

#include "spd/SPDProcessDataBlocks.h"
#include "spd/SPDDataBlockProcessor.h"
#include "spd/SPDPolyGroundFilter.h"

#include "spd/spd-config.h"

using namespace std;
using namespace spdlib;
using namespace TCLAP;

int main (int argc, char * const argv[]) 
{
	cout << "spdpolygrd " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	cout << "and you are welcome to redistribute it under certain conditions; See\n";
	cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << endl;
	
	try 
	{
		CmdLine cmd("Classify ground returns using a surface fitting algorithm: spdpolygrd", ' ', "1.0.0");
		
        ValueArg<boost::uint_fast32_t> numOfRowsBlockArg("r","blockrows","Number of rows within a block (Default 100)",false,100,"unsigned int");
		cmd.add( numOfRowsBlockArg );
        
        ValueArg<boost::uint_fast32_t> numOfColsBlockArg("c","blockcols","Number of columns within a block (Default 0) - Note values greater than 1 result in a non-sequencial SPD file.",false,0,"unsigned int");
		cmd.add( numOfColsBlockArg );
        
        ValueArg<float> binSizeArg("b","binsize","Bin size for processing and output image (Default 0) - Note 0 will use the native SPD file bin size.",false,0,"float");
		cmd.add( binSizeArg );
        
        ValueArg<float> grdClassThresArg("","grdthres","Threshold for how far above the interpolated ground surface a return can be and be reclassified as ground (Default = 0.25).",false,0.25,"float");
		cmd.add( grdClassThresArg );
		
		ValueArg<unsigned int> degreeArg("","degree","Order of polynomial surface (Default = 1).",false,1,"int");
		cmd.add( degreeArg );
		
		ValueArg<unsigned int> numItersArg("","iters","Number of iterations for polynomial surface to converge on ground (Default = 2).",false,2,"int");
		cmd.add( numItersArg );
        
        //ValueArg<uint_fast16_t> usePointsofClassArg("","class","Only use points of particular class",false,SPD_ALL_CLASSES,"uint_fast16_t");
        //cmd.add( usePointsofClassArg );
        
		UnlabeledMultiArg<string> multiFileNames("File", "File names for the input files", false, "string");
		cmd.add( multiFileNames );
		cmd.parse( argc, argv );
		
		vector<string> fileNames = multiFileNames.getValue();		
        cout << "fileNames.size() = " << fileNames.size() << endl;
		if(fileNames.size() == 2)
		{
            string inSPDFilePath = fileNames.at(0);
            string outFilePath = fileNames.at(1);
                        
            SPDPolyFitGroundFilter grdFilter;
            grdFilter.applyPolyFitGroundFilter(inSPDFilePath, outFilePath, grdClassThresArg.getValue(), degreeArg.getValue(), numItersArg.getValue(), numOfColsBlockArg.getValue(), numOfRowsBlockArg.getValue(), binSizeArg.getValue());
		}
        else
        {
            cout << "ERROR: Only 2 files can be provided\n";
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
}

