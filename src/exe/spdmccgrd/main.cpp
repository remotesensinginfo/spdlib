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
#include "spd/SPDMultiscaleCurvatureGrdClassification.h"


#include "spd/spd-config.h"

using namespace std;
using namespace spdlib;
using namespace TCLAP;

int main (int argc, char * const argv[]) 
{
	cout << "spdmccgrd " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	cout << "and you are welcome to redistribute it under certain conditions; See\n";
	cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << endl;
	
	try 
	{
		CmdLine cmd("Classifies the ground returns using the multiscale curvature algorithm: spdmccgrd", ' ', "1.0.0");
		
        ValueArg<boost::uint_fast32_t> numOfRowsBlockArg("r","blockrows","Number of rows within a block (Default 100)",false,100,"unsigned int");
		cmd.add( numOfRowsBlockArg );
        
        ValueArg<boost::uint_fast32_t> numOfColsBlockArg("c","blockcols","Number of columns within a block (Default 0) - Note values greater than 1 result in a non-sequencial SPD file.",false,0,"unsigned int");
		cmd.add( numOfColsBlockArg );
        
        ValueArg<float> binSizeArg("b","binsize","Bin size for processing and output image (Default 0) - Note 0 will use the native SPD file bin size.",false,0,"float");
		cmd.add( binSizeArg );
        
        ValueArg<uint_fast16_t> overlapArg("","overlap","Size (in bins) of the overlap between processing blocks (Default 10)",false,10,"uint_fast16_t");
		cmd.add( overlapArg );
        
        ValueArg<float> initScaleArg("","initscale","Initial processing scale, this is usually the native resolution of the data.",false,1,"float");
        cmd.add( initScaleArg );
        
        ValueArg<uint_fast16_t> numOfScalesAboveArg("","numofscalesabove","The number of scales above the init scale to be used (Default = 1)",false,1,"uint_fast16_t");
        cmd.add( numOfScalesAboveArg );
        
        ValueArg<uint_fast16_t> numOfScalesBelowArg("","numofscalesbelow","The number of scales below the init scale to be used (Default = 1)",false,1,"uint_fast16_t");
        cmd.add( numOfScalesBelowArg );
        
        ValueArg<float> scaleGapsArg("","scalegaps","Gap between increments in scale (Default = 0.5)",false,0.5,"float");
        cmd.add( scaleGapsArg );
        
        ValueArg<float> initCurveToleranceArg("","initcurvetol","Initial curveture tolerance parameter (Default = 1)",false,1,"float");
        cmd.add( initCurveToleranceArg );
        
        ValueArg<float> minCurveToleranceArg("","mincurvetol","Minimum curveture tolerance parameter (Default = 0.1)",false,0.1,"float");
        cmd.add( minCurveToleranceArg );
        
        ValueArg<float> stepCurveToleranceArg("","stepcurvetol","Iteration step curveture tolerance parameter (Default = 0.5)",false,0.5,"float");
        cmd.add( stepCurveToleranceArg );
        
        ValueArg<float> interpMaxRadiusArg("","interpmaxradius","Maximum search radius for the TPS interpolation (Default = 20)",false,20,"float");
        cmd.add( interpMaxRadiusArg );
        
        ValueArg<uint_fast16_t> interpNumPointsArg("","interpnumpts","The number of points used for the TPS interpolation (Default = 16)",false,16,"uint_fast16_t");
        cmd.add( interpNumPointsArg );
        
        ValueArg<uint_fast16_t> smoothFilterHSizeArg("","filtersize","The size of the smoothing filter (half size i.e., 3x3 is 1; Default = 1).",false,1,"uint_fast16_t");
        cmd.add( smoothFilterHSizeArg );
        
        ValueArg<float> thresOfChangeArg("","thresofchange","The threshold for the  (Default = 0.1)",false,0.1,"float");
        cmd.add( thresOfChangeArg );
        
        SwitchArg medianFilterSwitch("","median","Use a median filter to smooth the generated raster instead of a (mean) averaging filter.", false);		
        cmd.add(medianFilterSwitch);
        
        ValueArg<uint_fast16_t> usePointsofClassArg("","class","Only use points of particular class",false,SPD_ALL_CLASSES,"uint_fast16_t");
        cmd.add( usePointsofClassArg );
        
        SwitchArg calcThresOfChangeWithMultiPulsesSwitch("","thresofchangemultireturn","Use only multiple return pulses to calculate the amount of change between iterations.", false);		
        cmd.add(calcThresOfChangeWithMultiPulsesSwitch);
        
		UnlabeledMultiArg<string> multiFileNames("File", "File names for the input files", false, "string");
		cmd.add( multiFileNames );
		cmd.parse( argc, argv );
		
		vector<string> fileNames = multiFileNames.getValue();		
        cout << "fileNames.size() = " << fileNames.size() << endl;
		if(fileNames.size() == 2)
		{
            string inSPDFilePath = fileNames.at(0);
            string outFilePath = fileNames.at(1);
            
            SPDSmoothFilterType filterType = meanFilter;
            if(medianFilterSwitch.getValue())
            {
                filterType = medianFilter;
            }
            
            SPDFile *spdInFile = new SPDFile(inSPDFilePath);
            
            SPDMultiscaleCurvatureGrdClassification *blockProcessor = new SPDMultiscaleCurvatureGrdClassification(initScaleArg.getValue(), numOfScalesAboveArg.getValue(), numOfScalesBelowArg.getValue(), scaleGapsArg.getValue(), initCurveToleranceArg.getValue(), minCurveToleranceArg.getValue(), stepCurveToleranceArg.getValue(), interpMaxRadiusArg.getValue(), interpNumPointsArg.getValue(), filterType, smoothFilterHSizeArg.getValue(), thresOfChangeArg.getValue(), calcThresOfChangeWithMultiPulsesSwitch.getValue(), usePointsofClassArg.getValue());
            SPDProcessDataBlocks processBlocks = SPDProcessDataBlocks(blockProcessor, overlapArg.getValue(), numOfColsBlockArg.getValue(), numOfRowsBlockArg.getValue(), true);
            processBlocks.processDataBlocksGridPulsesOutputSPD(spdInFile, outFilePath, binSizeArg.getValue());
            
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
    std::cout << "spdmccgrd - end\n";
}

