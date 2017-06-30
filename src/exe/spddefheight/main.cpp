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

int main (int argc, char * const argv[]) 
{
    std::cout.precision(12);
    
    std::cout << "spddefheight " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	std::cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	std::cout << "and you are welcome to redistribute it under certain conditions; See\n";
	std::cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	std::cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << std::endl;
	
	try 
	{
        TCLAP::CmdLine cmd("Define the height field within pulses and points: spddefheight", ' ', "1.1.0");
		
        TCLAP::ValueArg<boost::uint_fast32_t> numOfRowsBlockArg("r","blockrows","Number of rows within a block (Default 100)",false,100,"unsigned int");
		cmd.add( numOfRowsBlockArg );
        
        TCLAP::ValueArg<boost::uint_fast32_t> numOfColsBlockArg("c","blockcols","Number of columns within a block (Default 0) - Note values greater than 1 result in a non-sequencial SPD file.",false,0,"unsigned int");
		cmd.add( numOfColsBlockArg );
        
        TCLAP::ValueArg<uint_fast16_t> overlapArg("","overlap","Size (in bins) of the overlap between processing blocks (Default 10)",false,10,"uint_fast16_t");
		cmd.add( overlapArg );
        
        TCLAP::ValueArg<float> binSizeArg("b","binsize","Bin size for processing and output image (Default 0) - Note 0 will use the native SPD file bin size.",false,0,"float");
		cmd.add( binSizeArg );
        
        
        TCLAP::SwitchArg interpSwitch("","interp","Use interpolation of the ground returns to calculate ground elevation", false);		
		TCLAP::SwitchArg imageSwitch("","image","Use an image which defines the ground elevation.", false);
        
        std::vector<TCLAP::Arg*> argumentsElev;
        argumentsElev.push_back(&interpSwitch);
        argumentsElev.push_back(&imageSwitch);
        cmd.xorAdd(argumentsElev);
        
		std::vector<std::string> interpolators;
		interpolators.push_back("TIN_PLANE");
		interpolators.push_back("NEAREST_NEIGHBOR");
		interpolators.push_back("NATURAL_NEIGHBOR_CGAL");
        interpolators.push_back("NATURAL_NEIGHBOR");
		interpolators.push_back("STDEV_MULTISCALE");
		interpolators.push_back("TPS_RAD");
        interpolators.push_back("TPS_PTNO");
		TCLAP::ValuesConstraint<std::string> allowedInterpolatorVals( interpolators );
		
		TCLAP::ValueArg<std::string> interpolatorsArg("","in","The interpolator to be used.",false,"NATURAL_NEIGHBOR", &allowedInterpolatorVals);
		cmd.add( interpolatorsArg );
		
		TCLAP::ValueArg<float> stddevThresholdArg("","stddevThreshold","STDEV_MULTISCALE: Standard Deviation threshold",false,3,"float");
		cmd.add( stddevThresholdArg );
		
		TCLAP::ValueArg<float> smallRadiusArg("","smallRadius","STDEV_MULTISCALE: Smaller radius to be used when standard deviation is high",false,1,"float");
		cmd.add( smallRadiusArg );
		
		TCLAP::ValueArg<float> largeRadiusArg("","largeRadius","STDEV_MULTISCALE: Large radius to be used when standard deviation is low",false,5,"float");
		cmd.add( largeRadiusArg );
		
		TCLAP::ValueArg<float> stdDevRadiusArg("","stdDevRadius","STDEV_MULTISCALE: Radius used to calculate the standard deviation",false,5,"float");
		cmd.add( stdDevRadiusArg );
		
		TCLAP::ValueArg<float> tpsRadiusArg("","tpsRadius","TPS: (TPS_PTNO - maximum) Radius used to retrieve data in TPS algorithm",false,1,"float");
		cmd.add( tpsRadiusArg );
        
        TCLAP::ValueArg<uint_fast16_t> tpsNoPtsArg("","tpsnopts","TPS: (TPS_RAD - minimum) Number of points to be used by TPS algorithm",false,12,"uint_fast16_t");
		cmd.add( tpsNoPtsArg );
        
        TCLAP::SwitchArg thinSwitch("","thin","Thin the point cloud when interpolating", false);		
        cmd.add( thinSwitch );
        
        TCLAP::ValueArg<uint_fast16_t> noPtsPerBinArg("","ptsperbin","The number of point allowed within a grid cell following thinning",false,1,"uint_fast16_t");
		cmd.add( noPtsPerBinArg );
        
        TCLAP::ValueArg<float> thinGridResArg("","thinres","Resolution of the grid used to thin the point cloud",false,0.5,"float");
		cmd.add( thinGridResArg );
        
        TCLAP::ValueArg<float> gridIdxResolutionArg("","idxres","Resolution of the grid index used for some interpolates",false,0.5,"float");
		cmd.add( gridIdxResolutionArg );
		
        TCLAP::ValueArg<std::string> inputFileArg("i","input","The input SPD file.",true,"","String");
		cmd.add( inputFileArg );
        
        TCLAP::ValueArg<std::string> elevationFileArg("e","elevation","The input elevation image.",false,"","String");
		cmd.add( elevationFileArg );
        
        TCLAP::ValueArg<std::string> outputFileArg("o","output","The output file.",true,"","String");
		cmd.add( outputFileArg );
        
		cmd.parse( argc, argv );
		
        std::cout.precision(12);
		if(interpSwitch.getValue())
		{
            std::string inSPDFilePath = inputFileArg.getValue();
            std::string outFilePath = outputFileArg.getValue();
                        
            float stdDevThreshold = stddevThresholdArg.getValue();
            float lowDist = smallRadiusArg.getValue();
            float highDist = largeRadiusArg.getValue();
            float stdDevDist = stdDevRadiusArg.getValue();	
            
            std::string interpolatorStr = interpolatorsArg.getValue();
            spdlib::SPDPointInterpolator *interpolator = NULL;
            if(interpolatorStr == "NEAREST_NEIGHBOR")
            {
                interpolator = new spdlib::SPDNearestNeighbourInterpolator(spdlib::SPD_USE_Z, thinGridResArg.getValue(), thinSwitch.getValue(), spdlib::SPD_SELECT_LOWEST, noPtsPerBinArg.getValue());
            }
            else if(interpolatorStr == "TIN_PLANE")
            {
                interpolator = new spdlib::SPDTINPlaneFitInterpolator(spdlib::SPD_USE_Z, thinGridResArg.getValue(), thinSwitch.getValue(), spdlib::SPD_SELECT_LOWEST, noPtsPerBinArg.getValue());
            }
            else if(interpolatorStr == "NATURAL_NEIGHBOR_CGAL")
            {
                interpolator = new spdlib::SPDNaturalNeighborCGALPointInterpolator(spdlib::SPD_USE_Z, thinGridResArg.getValue(), thinSwitch.getValue(), spdlib::SPD_SELECT_LOWEST, noPtsPerBinArg.getValue());
            }
            else if(interpolatorStr == "NATURAL_NEIGHBOR")
            {
                //interpolator = new spdlib::SPDNaturalNeighborCGALPointInterpolator(spdlib::SPD_USE_Z, thinGridResArg.getValue(), thinSwitch.getValue(), spdlib::SPD_SELECT_LOWEST, noPtsPerBinArg.getValue());
            }
            else if(interpolatorStr == "STDEV_MULTISCALE")
            {
                interpolator = new spdlib::SPDStdDevFilterInterpolator(stdDevThreshold, lowDist, highDist, stdDevDist, gridIdxResolutionArg.getValue(), spdlib::SPD_USE_Z, thinGridResArg.getValue(), thinSwitch.getValue(), spdlib::SPD_SELECT_LOWEST, noPtsPerBinArg.getValue());
            }
            else if(interpolatorStr == "TPS_RAD")
            {
                interpolator = new spdlib::SPDTPSRadiusInterpolator( tpsRadiusArg.getValue(), tpsNoPtsArg.getValue(), gridIdxResolutionArg.getValue(), spdlib::SPD_USE_Z, thinGridResArg.getValue(), thinSwitch.getValue(), spdlib::SPD_SELECT_LOWEST, noPtsPerBinArg.getValue()); 
            }
            else if(interpolatorStr == "TPS_PTNO")
            {
                interpolator = new spdlib::SPDTPSNumPtsInterpolator( tpsRadiusArg.getValue(), tpsNoPtsArg.getValue(), gridIdxResolutionArg.getValue(), spdlib::SPD_USE_Z, thinGridResArg.getValue(), thinSwitch.getValue(), spdlib::SPD_SELECT_LOWEST, noPtsPerBinArg.getValue());  
            }
            else 
            {
                throw spdlib::SPDException("Interpolator was not recognised.");
            }
            
            spdlib::SPDFile *spdInFile = new spdlib::SPDFile(inSPDFilePath);
            
            spdlib::SPDDataBlockProcessor *blockProcessor = new spdlib::SPDDefinePulseHeights(interpolator);
            spdlib::SPDProcessDataBlocks processBlocks = spdlib::SPDProcessDataBlocks(blockProcessor, overlapArg.getValue(), numOfColsBlockArg.getValue(), numOfRowsBlockArg.getValue(), true);
            processBlocks.processDataBlocksGridPulsesOutputSPD(spdInFile, outFilePath, binSizeArg.getValue());
            
            delete blockProcessor;
            delete interpolator;
            delete spdInFile; 
        }
        else if(imageSwitch.getValue())
        {
            std::string inSPDFilePath = inputFileArg.getValue();
            std::string inElevRasterPath = elevationFileArg.getValue();
            std::string outFilePath = outputFileArg.getValue();
            
            spdlib::SPDFile *spdInFile = new spdlib::SPDFile(inSPDFilePath);
            
            spdlib::SPDDataBlockProcessor *blockProcessor = new spdlib::SPDDefinePulseHeights(NULL);
            spdlib::SPDProcessDataBlocks processBlocks = spdlib::SPDProcessDataBlocks(blockProcessor, 0, numOfColsBlockArg.getValue(), numOfRowsBlockArg.getValue(), true);
            processBlocks.processDataBlocksGridPulsesInputImage(spdInFile, outFilePath, inElevRasterPath);
            
            delete blockProcessor;
            delete spdInFile; 
        }
        else
        {
            throw TCLAP::ArgException("Need to select either --image or --interp option.");
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
    std::cout << "spddefheight - end\n";
}

