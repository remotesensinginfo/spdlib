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
#include "spd/SPDRasterInterpolation.h"

#include "spd/spd-config.h"

int main (int argc, char * const argv[]) 
{
    std::cout.precision(12);
    
    std::cout << "spdinterp " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	std::cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	std::cout << "and you are welcome to redistribute it under certain conditions; See\n";
	std::cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	std::cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << std::endl;
	
	try 
	{
        TCLAP::CmdLine cmd("Interpolate a raster elevation surface: spdinterp", ' ', "1.0.0");
		
        TCLAP::ValueArg<boost::uint_fast32_t> numOfRowsBlockArg("r","blockrows","Number of rows within a block (Default 100)",false,100,"unsigned int");
		cmd.add( numOfRowsBlockArg );
        
        TCLAP::ValueArg<boost::uint_fast32_t> numOfColsBlockArg("c","blockcols","Number of columns within a block (Default 0) - Note values greater than 1 result in a non-sequencial SPD file.",false,0,"unsigned int");
		cmd.add( numOfColsBlockArg );
        
        TCLAP::ValueArg<uint_fast16_t> overlapArg("","overlap","Size (in bins) of the overlap between processing blocks (Default 10)",false,10,"uint_fast16_t");
		cmd.add( overlapArg );
        
        TCLAP::ValueArg<float> binSizeArg("b","binsize","Bin size for processing and output image (Default 0) - Note 0 will use the native SPD file bin size.",false,0,"float");
		cmd.add( binSizeArg );
        
        TCLAP::ValueArg<std::string> imgFormatArg("f","format","Image format (GDAL driver string), Default is ENVI.",false,"ENVI","string");
		cmd.add( imgFormatArg );
        
        TCLAP::SwitchArg dtmSwitch("","dtm","Interpolate a DTM image", false);		
		TCLAP::SwitchArg chmSwitch("","chm","Interpolate a CHM image.", false);
        TCLAP::SwitchArg dsmSwitch("","dsm","Interpolate a DSM image.", false);
        
        std::vector<TCLAP::Arg*> arguments;
        arguments.push_back(&dtmSwitch);
        arguments.push_back(&chmSwitch);
        arguments.push_back(&dsmSwitch);
        cmd.xorAdd(arguments);
        
        TCLAP::SwitchArg zSwitch("","topo","Use topographic elevation", false);		
		TCLAP::SwitchArg hSwitch("","height","Use height above ground elevation.", false);
        
        std::vector<TCLAP::Arg*> argumentsElev;
        argumentsElev.push_back(&zSwitch);
        argumentsElev.push_back(&hSwitch);
        cmd.xorAdd(argumentsElev);
        
		std::vector<std::string> interpolators;
		interpolators.push_back("TIN_PLANE");
		interpolators.push_back("NEAREST_NEIGHBOR");
		interpolators.push_back("NATURAL_NEIGHBOR");
		interpolators.push_back("STDEV_MULTISCALE");
		interpolators.push_back("TPS_RAD");
        interpolators.push_back("TPS_PTNO");
		TCLAP::ValuesConstraint<std::string> allowedInterpolatorVals( interpolators );
		
		TCLAP::ValueArg<std::string> interpolatorsArg("","interpolator","The interpolator to be used.",false,"NATURAL_NEIGHBOR", &allowedInterpolatorVals);
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
        
        TCLAP::ValueArg<std::string> outputFileArg("o","output","The output SPD file.",true,"","String");
		cmd.add( outputFileArg );
        
		cmd.parse( argc, argv );
		
		std::string inSPDFilePath = inputFileArg.getValue();
        std::string outFilePath = outputFileArg.getValue();
        
        std::string interpolatorStr = interpolatorsArg.getValue();
        
        float stdDevThreshold = stddevThresholdArg.getValue();
        float lowDist = smallRadiusArg.getValue();
        float highDist = largeRadiusArg.getValue();
        float stdDevDist = stdDevRadiusArg.getValue();
        
        uint_fast16_t elevVal = spdlib::SPD_USE_Z;
        
        if(zSwitch.getValue())
        {
            elevVal = spdlib::SPD_USE_Z;
        }
        else if(hSwitch.getValue())
        {
            elevVal = spdlib::SPD_USE_HEIGHT;
        }
        else
        {
            throw spdlib::SPDException("Elevation (height or Z) parameter has not been defined.");
        }
        
        uint_fast16_t thinPtSelectLowHigh = spdlib::SPD_SELECT_LOWEST;
        if(dtmSwitch.getValue())
        {
            thinPtSelectLowHigh = spdlib::SPD_SELECT_LOWEST;
        }
        else if(chmSwitch.getValue())
        {
            thinPtSelectLowHigh = spdlib::SPD_SELECT_HIGHEST;
        }
        else if(dsmSwitch.getValue())
        {
            thinPtSelectLowHigh = spdlib::SPD_SELECT_HIGHEST;
        }
        else
        {
            throw spdlib::SPDException("Error do not know whether to generate a CHM or DTM.");
        }
        
        spdlib::SPDPointInterpolator *interpolator = NULL;
        if(interpolatorStr == "NEAREST_NEIGHBOR")
        {
            interpolator = new spdlib::SPDNearestNeighbourInterpolator(elevVal, thinGridResArg.getValue(), thinSwitch.getValue(), thinPtSelectLowHigh, noPtsPerBinArg.getValue());
        }
        else if(interpolatorStr == "TIN_PLANE")
        {
            interpolator = new spdlib::SPDTINPlaneFitInterpolator(elevVal, thinGridResArg.getValue(), thinSwitch.getValue(), thinPtSelectLowHigh, noPtsPerBinArg.getValue());
        }
        else if(interpolatorStr == "NATURAL_NEIGHBOR")
        {
            interpolator = new spdlib::SPDNaturalNeighborPointInterpolator(elevVal, thinGridResArg.getValue(), thinSwitch.getValue(), thinPtSelectLowHigh, noPtsPerBinArg.getValue());
        }
        else if(interpolatorStr == "STDEV_MULTISCALE")
        {
            interpolator = new spdlib::SPDStdDevFilterInterpolator(stdDevThreshold, lowDist, highDist, stdDevDist, gridIdxResolutionArg.getValue(), elevVal, thinGridResArg.getValue(), thinSwitch.getValue(), thinPtSelectLowHigh, noPtsPerBinArg.getValue());
        }
        else if(interpolatorStr == "TPS_RAD")
        {
            interpolator = new spdlib::SPDTPSRadiusInterpolator( tpsRadiusArg.getValue(), tpsNoPtsArg.getValue(), gridIdxResolutionArg.getValue(), elevVal, thinGridResArg.getValue(), thinSwitch.getValue(), thinPtSelectLowHigh, noPtsPerBinArg.getValue());
        }
        else if(interpolatorStr == "TPS_PTNO")
        {
            interpolator = new spdlib::SPDTPSNumPtsInterpolator( tpsRadiusArg.getValue(), tpsNoPtsArg.getValue(), gridIdxResolutionArg.getValue(), elevVal, thinGridResArg.getValue(), thinSwitch.getValue(), thinPtSelectLowHigh, noPtsPerBinArg.getValue());
        }
        else
        {
            throw spdlib::SPDException("Interpolator was not recognised.");
        }
        
        spdlib::SPDDataBlockProcessor *blockProcessor = NULL;
        if(dtmSwitch.getValue())
        {
            blockProcessor = new spdlib::SPDDTMInterpolation(interpolator);
        }
        else if(chmSwitch.getValue())
        {
            blockProcessor = new spdlib::SPDCHMInterpolation(interpolator);
        }
        else if(dsmSwitch.getValue())
        {
            blockProcessor = new spdlib::SPDDSMInterpolation(interpolator);
        }
        else
        {
            throw spdlib::SPDException("Error do not know whether to generate a CHM or DTM.");
        }
        
        spdlib::SPDFile *spdInFile = new spdlib::SPDFile(inSPDFilePath);
        spdlib::SPDProcessDataBlocks processBlocks = spdlib::SPDProcessDataBlocks(blockProcessor, overlapArg.getValue(), numOfColsBlockArg.getValue(), numOfRowsBlockArg.getValue(), true);
        processBlocks.processDataBlocksGridPulsesOutputImage(spdInFile, outFilePath, binSizeArg.getValue(), 1, imgFormatArg.getValue());
        
        delete spdInFile;
        delete blockProcessor;
        delete interpolator;
		
	}
	catch (TCLAP::ArgException &e)
	{
		std::cerr << "Parse Error: " << e.what() << std::endl;
	}
	catch(spdlib::SPDException &e)
	{
		std::cerr << "Error: " << e.what() << std::endl;
	}
    
    std::cout << "spdinterp - end\n";
}

