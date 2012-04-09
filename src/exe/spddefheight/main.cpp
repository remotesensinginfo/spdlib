/*
 *  main.cpp
 *  spdlib
 *
 *  Created by Pete Bunting on 05/03/2012.
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
#include "spd/SPDPointInterpolation.h"
#include "spd/SPDDefinePulseHeights.h"

#include "spd/spd-config.h"

using namespace std;
using namespace spdlib;
using namespace TCLAP;

int main (int argc, char * const argv[]) 
{
	cout << "spddefheight " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	cout << "and you are welcome to redistribute it under certain conditions; See\n";
	cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << endl;
	
	try 
	{
		CmdLine cmd("Define the height field within pulses: spddefheight", ' ', "1.0.0");
		
        ValueArg<boost::uint_fast32_t> numOfRowsBlockArg("r","blockrows","Number of rows within a block (Default 100)",false,100,"unsigned int");
		cmd.add( numOfRowsBlockArg );
        
        ValueArg<boost::uint_fast32_t> numOfColsBlockArg("c","blockcols","Number of columns within a block (Default 0) - Note values greater than 1 result in a non-sequencial SPD file.",false,0,"unsigned int");
		cmd.add( numOfColsBlockArg );
        
        ValueArg<uint_fast16_t> overlapArg("","overlap","Size (in bins) of the overlap between processing blocks (Default 10)",false,10,"uint_fast16_t");
		cmd.add( overlapArg );
        
        ValueArg<float> binSizeArg("b","binsize","Bin size for processing and output image (Default 0) - Note 0 will use the native SPD file bin size.",false,0,"float");
		cmd.add( binSizeArg );
        
        
        SwitchArg interpSwitch("","interp","Use interpolation of the ground returns to calculate ground elevation", false);		
		SwitchArg imageSwitch("","image","Use an image which defines the ground elevation.", false);
        
        vector<Arg*> argumentsElev;
        argumentsElev.push_back(&interpSwitch);
        argumentsElev.push_back(&imageSwitch);
        cmd.xorAdd(argumentsElev);
        
		vector<string> interpolators;
		interpolators.push_back("TIN_PLANE");
		interpolators.push_back("NEAREST_NEIGHBOR");
		interpolators.push_back("NATURAL_NEIGHBOR");
		interpolators.push_back("STDEV_MULTISCALE");
		interpolators.push_back("TPS_RAD");
        interpolators.push_back("TPS_PTNO");
		ValuesConstraint<string> allowedInterpolatorVals( interpolators );
		
		ValueArg<string> interpolatorsArg("i","interpolator","The interpolator to be used.",false,"NATURAL_NEIGHBOR", &allowedInterpolatorVals);
		cmd.add( interpolatorsArg );
		
		ValueArg<float> stddevThresholdArg("","stddevThreshold","STDEV_MULTISCALE: Standard Deviation threshold",false,3,"float");
		cmd.add( stddevThresholdArg );
		
		ValueArg<float> smallRadiusArg("","smallRadius","STDEV_MULTISCALE: Smaller radius to be used when standard deviation is high",false,1,"float");
		cmd.add( smallRadiusArg );
		
		ValueArg<float> largeRadiusArg("","largeRadius","STDEV_MULTISCALE: Large radius to be used when standard deviation is low",false,5,"float");
		cmd.add( largeRadiusArg );
		
		ValueArg<float> stdDevRadiusArg("","stdDevRadius","STDEV_MULTISCALE: Radius used to calculate the standard deviation",false,5,"float");
		cmd.add( stdDevRadiusArg );
		
		ValueArg<float> tpsRadiusArg("","tpsRadius","TPS: (TPS_PTNO - maximum) Radius used to retrieve data in TPS algorithm",false,1,"float");
		cmd.add( tpsRadiusArg );
        
        ValueArg<uint_fast16_t> tpsNoPtsArg("","tpsnopts","TPS: (TPS_RAD - minimum) Number of points to be used by TPS algorithm",false,12,"uint_fast16_t");
		cmd.add( tpsNoPtsArg );
        
        SwitchArg thinSwitch("","thin","Thin the point cloud when interpolating", false);		
        cmd.add( thinSwitch );
        
        ValueArg<uint_fast16_t> noPtsPerBinArg("","ptsperbin","The number of point allowed within a grid cell following thinning",false,1,"uint_fast16_t");
		cmd.add( noPtsPerBinArg );
        
        ValueArg<float> thinGridResArg("","thinres","Resolution of the grid used to thin the point cloud",false,0.5,"float");
		cmd.add( thinGridResArg );
        
        ValueArg<float> gridIdxResolutionArg("","idxres","Resolution of the grid index used for some interpolates",false,0.5,"float");
		cmd.add( gridIdxResolutionArg );
        
		UnlabeledMultiArg<string> multiFileNames("File", "File names for the input files", false, "string");
		cmd.add( multiFileNames );
		cmd.parse( argc, argv );
		
		vector<string> fileNames = multiFileNames.getValue();		
        cout << "fileNames.size() = " << fileNames.size() << endl;
		if((fileNames.size() == 2) & interpSwitch.getValue())
		{
            string inSPDFilePath = fileNames.at(0);
            string outFilePath = fileNames.at(1);
                        
            float stdDevThreshold = stddevThresholdArg.getValue();
            float lowDist = smallRadiusArg.getValue();
            float highDist = largeRadiusArg.getValue();
            float stdDevDist = stdDevRadiusArg.getValue();	
            
            string interpolatorStr = interpolatorsArg.getValue();
            SPDPointInterpolator *interpolator = NULL;
            if(interpolatorStr == "NEAREST_NEIGHBOR")
            {
                interpolator = new SPDNearestNeighbourInterpolator(SPD_USE_Z, thinGridResArg.getValue(), thinSwitch.getValue(), SPD_SELECT_LOWEST, noPtsPerBinArg.getValue());
            }
            else if(interpolatorStr == "TIN_PLANE")
            {
                interpolator = new SPDTINPlaneFitInterpolator(SPD_USE_Z, thinGridResArg.getValue(), thinSwitch.getValue(), SPD_SELECT_LOWEST, noPtsPerBinArg.getValue());
            }
            else if(interpolatorStr == "NATURAL_NEIGHBOR")
            {
                interpolator = new SPDNaturalNeighborPointInterpolator(SPD_USE_Z, thinGridResArg.getValue(), thinSwitch.getValue(), SPD_SELECT_LOWEST, noPtsPerBinArg.getValue());
            }
            else if(interpolatorStr == "STDEV_MULTISCALE")
            {
                interpolator = new SPDStdDevFilterInterpolator(stdDevThreshold, lowDist, highDist, stdDevDist, gridIdxResolutionArg.getValue(), SPD_USE_Z, thinGridResArg.getValue(), thinSwitch.getValue(), SPD_SELECT_LOWEST, noPtsPerBinArg.getValue());
            }
            else if(interpolatorStr == "TPS_RAD")
            {
                interpolator = new SPDTPSRadiusInterpolator( tpsRadiusArg.getValue(), tpsNoPtsArg.getValue(), gridIdxResolutionArg.getValue(), SPD_USE_Z, thinGridResArg.getValue(), thinSwitch.getValue(), SPD_SELECT_LOWEST, noPtsPerBinArg.getValue()); 
            }
            else if(interpolatorStr == "TPS_PTNO")
            {
                interpolator = new SPDTPSNumPtsInterpolator( tpsRadiusArg.getValue(), tpsNoPtsArg.getValue(), gridIdxResolutionArg.getValue(), SPD_USE_Z, thinGridResArg.getValue(), thinSwitch.getValue(), SPD_SELECT_LOWEST, noPtsPerBinArg.getValue());  
            }
            else 
            {
                throw SPDException("Interpolator was not recognised.");
            }
            
            SPDFile *spdInFile = new SPDFile(inSPDFilePath);
            
            SPDDataBlockProcessor *blockProcessor = new SPDDefinePulseHeights(interpolator);
            SPDProcessDataBlocks processBlocks = SPDProcessDataBlocks(blockProcessor, overlapArg.getValue(), numOfColsBlockArg.getValue(), numOfRowsBlockArg.getValue(), true);
            processBlocks.processDataBlocksGridPulsesOutputSPD(spdInFile, outFilePath, binSizeArg.getValue());
            
            delete blockProcessor;
            delete interpolator;
            delete spdInFile; 
        }
        else if((fileNames.size() == 3) & imageSwitch.getValue())
        {
            string inSPDFilePath = fileNames.at(0);
            string inElevRasterPath = fileNames.at(1);
            string outFilePath = fileNames.at(2);
            
            SPDFile *spdInFile = new SPDFile(inSPDFilePath);
            
            SPDDataBlockProcessor *blockProcessor = new SPDDefinePulseHeights(NULL);
            SPDProcessDataBlocks processBlocks = SPDProcessDataBlocks(blockProcessor, 0, numOfColsBlockArg.getValue(), numOfRowsBlockArg.getValue(), true);
            processBlocks.processDataBlocksGridPulsesInputImage(spdInFile, outFilePath, inElevRasterPath);
            
            delete blockProcessor;
            delete spdInFile; 
        }
        else
        {
            cout << "ERROR: for interpolation only 2 files can be inputted while for using an elevation image 3 are required.\n";
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

