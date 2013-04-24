/*
 *  main.cpp
 *  spdlib
 *
 *  Created by Pete Bunting on 03/03/2012.
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
#include "spd/SPDProgressiveMophologicalGrdFilter.h"

#include "spd/spd-config.h"

using namespace std;
using namespace spdlib;
using namespace TCLAP;

int main (int argc, char * const argv[]) 
{
	cout << "spdpmfgrd " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	cout << "and you are welcome to redistribute it under certain conditions; See\n";
	cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << endl;
	
	try 
	{
		CmdLine cmd("Classifies the ground returns using the progressive morphology algorithm: spdpmfgrd", ' ', "1.0.0");
		
        ValueArg<boost::uint_fast32_t> numOfRowsBlockArg("r","blockrows","Number of rows within a block (Default 100)",false,100,"unsigned int");
		cmd.add( numOfRowsBlockArg );
        
        ValueArg<boost::uint_fast32_t> numOfColsBlockArg("c","blockcols","Number of columns within a block (Default 0) - Note values greater than 1 result in a non-sequencial SPD file.",false,0,"unsigned int");
		cmd.add( numOfColsBlockArg );
        
        ValueArg<float> binSizeArg("b","binsize","Bin size for processing and output image (Default 0) - Note 0 will use the native SPD file bin size.",false,0,"float");
		cmd.add( binSizeArg );
        
        ValueArg<uint_fast16_t> overlapArg("","overlap","Size (in bins) of the overlap between processing blocks (Default 10)",false,10,"uint_fast16_t");
		cmd.add( overlapArg );
		
		ValueArg<uint_fast16_t> initFilterSizeArg("","initfilter","Initial size of the filter (note this is half the filter size so a 3x3 will be 1 and 5x5 will be 2) (Default 1)",false,1,"uint_fast16_t");
		cmd.add( initFilterSizeArg );
		
		ValueArg<uint_fast16_t> maxFilterSizeArg("","maxfilter","Maximum size of the filter (Default 7)",false,7,"uint_fast16_t");
		cmd.add( maxFilterSizeArg );
		
		ValueArg<float> slopeArg("","slope","Slope parameter related to terrain (Default 0.3)",false,0.3,"float");
		cmd.add( slopeArg );
		
		ValueArg<float> initElevThresArg("","initelev","Initial elevation difference threshold (Default 0.3)",false,0.3,"float");
		cmd.add( initElevThresArg );
		
		ValueArg<float> maxElevThresArg("","maxelev","Maximum elevation difference threshold (Default 5)",false,5,"float");
		cmd.add( maxElevThresArg );
		
		ValueArg<float> grdPtDevThresArg("","grd","Threshold for deviation from identified ground surface for classifying the ground returns (Default 0.3)",false,0.3,"float");
		cmd.add( grdPtDevThresArg );
		
		SwitchArg noMedianSwitch("","nomedian","Do not run a median filter on generated surface (before classifying ground point or export)", false);
		cmd.add( noMedianSwitch );
		
		ValueArg<uint_fast16_t> medianFilterArg("","medianfilter","Size of the median filter (half size i.e., 3x3 is 1) (Default 2)",false,2,"uint_fast16_t");
		cmd.add( medianFilterArg );
		
		SwitchArg imageSwitch("","image","If set an image of the output surface will be generated rather than classifying the points (useful for debugging and parameter selection)", false);
		cmd.add( imageSwitch );
        
        ValueArg<uint_fast16_t> usePointsofClassArg("","class","Only use points of particular class",false,SPD_ALL_CLASSES,"uint_fast16_t");
        cmd.add( usePointsofClassArg );
        
        ValueArg<string> gdalDriverArg("","gdal","Provide the GDAL driver format (Default ENVI), Erdas Imagine is HFA",false,"ENVI","string");
		cmd.add( gdalDriverArg );
        
		UnlabeledMultiArg<string> multiFileNames("File", "File names for the input files", false, "string");
		cmd.add( multiFileNames );
		cmd.parse( argc, argv );
		
		vector<string> fileNames = multiFileNames.getValue();		
        cout << "fileNames.size() = " << fileNames.size() << endl;
		if(fileNames.size() == 2)
		{
            string inSPDFilePath = fileNames.at(0);
            string outFilePath = fileNames.at(1);
            
            SPDFile *spdInFile = new SPDFile(inSPDFilePath);
            
            bool medianFilter = true;
            if(noMedianSwitch.getValue())
            {
                medianFilter = false;
            }
            
            SPDProgressiveMophologicalGrdFilter *blockProcessor = new SPDProgressiveMophologicalGrdFilter(initFilterSizeArg.getValue(), maxFilterSizeArg.getValue(), slopeArg.getValue(), initElevThresArg.getValue(), maxElevThresArg.getValue(), grdPtDevThresArg.getValue(), medianFilter, medianFilterArg.getValue(), usePointsofClassArg.getValue());
            SPDProcessDataBlocks processBlocks = SPDProcessDataBlocks(blockProcessor, overlapArg.getValue(), numOfColsBlockArg.getValue(), numOfRowsBlockArg.getValue(), true);
            
            if(imageSwitch.getValue())
            {
                processBlocks.processDataBlocksGridPulsesOutputImage(spdInFile, outFilePath, binSizeArg.getValue(), 1, gdalDriverArg.getValue());
            }
            else
            {
                processBlocks.processDataBlocksGridPulsesOutputSPD(spdInFile, outFilePath, binSizeArg.getValue());
            }
            
            delete blockProcessor;
            delete spdInFile;            
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
    std::cout << "spdpmfgrd - end\n";
}

