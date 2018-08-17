/*
 *  main.cpp
 *  spdlib
 *
 *  Created by Pete Bunting on 21/01/2013.
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
#include "spd/SPDWarpData.h"

#include "spd/spd-config.h"

int main (int argc, char * const argv[])
{
    std::cout.precision(12);

	std::cout << "spdwarp " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	std::cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	std::cout << "and you are welcome to redistribute it under certain conditions; See\n";
	std::cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	std::cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << std::endl;
	
	try
	{
        TCLAP::CmdLine cmd("Interpolate a raster elevation surface: spdwarp", ' ', "1.0.0");
		
        TCLAP::SwitchArg shiftSwitch("","shift","Apply a linear shift to the SPD file.", false);
		TCLAP::SwitchArg warpSwitch("","warp","Apply a nonlinear warp to the SPD file defined by a set of GCPs.", false);

        std::vector<TCLAP::Arg*> arguments;
        arguments.push_back(&shiftSwitch);
        arguments.push_back(&warpSwitch);
        cmd.xorAdd(arguments);

		std::vector<std::string> transformations;
		transformations.push_back("POLYNOMIAL");
		transformations.push_back("NEAREST_NEIGHBOR");
		transformations.push_back("TRIANGULATION");
		TCLAP::ValuesConstraint<std::string> allowedTransformVals( transformations );
		
		TCLAP::ValueArg<std::string> transformationsArg("t","transform","WARP (Default=POLYNOMIAL): The transformation model to be fitted to the GPCs and used to warp the data.",false,"POLYNOMIAL", &allowedTransformVals);
		cmd.add( transformationsArg );


        std::vector<std::string> pulseWarpOptions;
		transformations.push_back("ALL_RETURNS");
		transformations.push_back("PULSE_IDX");
		transformations.push_back("PULSE_ORIGIN");
		TCLAP::ValuesConstraint<std::string> allowedPulseWarpVals( pulseWarpOptions );
		
		TCLAP::ValueArg<std::string> pulseWarpArg("p","pulsewarp","WARP (Default=PULSE_IDX): The eastings and northings used to calculate the warp. ALL_RETURNS recalculates the offsets for each X,Y while PULSE_IDX and PULSE_ORIGIN use a single offset for the whole pulse.",false,"PULSE_IDX", &allowedPulseWarpVals);
		cmd.add( pulseWarpArg );

		
		TCLAP::ValueArg<boost::uint_fast32_t> polyOrderArg("","order","POLY TRANSFORM (Default=3): The order of the polynomial fitted.",false,3,"unsigned int");
		cmd.add( polyOrderArg );

        TCLAP::ValueArg<std::string> gcpsFileArg("g","gcps","WARP: The path and file name of the gcps file.",false,"","std::string");
		cmd.add( gcpsFileArg );

        TCLAP::ValueArg<float> xShiftArg("x","xshift","SHIFT: The x shift in the units of the dataset (probably metres).",false,0,"float");
		cmd.add( xShiftArg );
		
		TCLAP::ValueArg<float> yShiftArg("y","yshift","SHIFT: The y shift in the units of the dataset (probably metres).",false,0,"float");
		cmd.add( yShiftArg );

        TCLAP::ValueArg<boost::uint_fast32_t> numOfRowsBlockArg("r","blockrows","Number of rows within a block (Default 100)",false,100,"unsigned int");
		cmd.add( numOfRowsBlockArg );

        TCLAP::ValueArg<boost::uint_fast32_t> numOfColsBlockArg("c","blockcols","Number of columns within a block (Default 0) - Note values greater than 1 result in a non-sequencial SPD file.",false,0,"unsigned int");
		cmd.add( numOfColsBlockArg );

        TCLAP::ValueArg<std::string> inputFileArg("i","input","The input SPD file.",true,"","String");
		cmd.add( inputFileArg );

        TCLAP::ValueArg<std::string> outputFileArg("o","output","The output SPD file.",true,"","String");
		cmd.add( outputFileArg );

		cmd.parse( argc, argv );
		
        std::string inSPDFilePath = inputFileArg.getValue();
        std::string outFilePath = outputFileArg.getValue();

        boost::uint_fast32_t blockXSize = numOfColsBlockArg.getValue();
        boost::uint_fast32_t blockYSize = numOfRowsBlockArg.getValue();

        if(shiftSwitch.getValue())
        {
            float xShift = xShiftArg.getValue();
            float yShift = yShiftArg.getValue();

            spdlib::SPDFile *spdInFile = new spdlib::SPDFile(inSPDFilePath);

            spdlib::SPDShiftData *shiftData = new spdlib::SPDShiftData(xShift, yShift);
            spdlib::SPDProcessDataBlocks processBlocks = spdlib::SPDProcessDataBlocks(shiftData, 0, blockXSize, blockYSize, true);
            processBlocks.processDataBlocksGridPulsesOutputSPD(spdInFile, outFilePath, 0);

            delete spdInFile;
            delete shiftData;
        }
        else if(warpSwitch.getValue())
        {
            std::string warpLocStr = pulseWarpArg.getValue();
            spdlib::SPDWarpLocation warpLoc = spdlib::spdwarppulseidx;

            if(warpLocStr == "ALL_RETURNS")
            {
                warpLoc = spdlib::spdwarpfromall;
            }
            else if (warpLocStr == "PULSE_IDX")
            {
                warpLoc = spdlib::spdwarppulseidx;
            }
            else if (warpLocStr == "PULSE_ORIGIN")
            {
                warpLoc = spdlib::spdwarppulseorigin;
            }
            else
            {
                throw spdlib::SPDException("The warp location has not been recognised.");
            }

            std::string transformationStr = transformationsArg.getValue();
            spdlib::SPDWarpPointData *warpData = NULL;

            if(transformationStr == "POLYNOMIAL")
            {
                float polyOrder = polyOrderArg.getValue();
                warpData = new spdlib::SPDPolynomialWarp(polyOrder);
            }
            else if (transformationStr == "NEAREST_NEIGHBOR")
            {
                warpData = new spdlib::SPDNearestNeighbourWarp();
            }
            else if (transformationStr == "TRIANGULATION")
            {
                warpData = new spdlib::SPDTriangulationPlaneFittingWarp();
            }
            else
            {
                throw spdlib::SPDException("The transformation has not been recognised.");
            }

            std::string gcpsFile = gcpsFileArg.getValue();
            warpData->initWarp(gcpsFile);

            spdlib::SPDFile *inSPDFile = new spdlib::SPDFile(inSPDFilePath);
            spdlib::SPDFile *spdFileOut = new spdlib::SPDFile(outFilePath);
            spdlib::SPDDataExporter *exporter = new spdlib::SPDNoIdxFileWriter();

            spdlib::SPDNonLinearWarp *warpProcessor = new spdlib::SPDNonLinearWarp(exporter, spdFileOut, warpData, warpLoc);

            spdlib::SPDFileReader spdReader;
            spdReader.readAndProcessAllData(inSPDFilePath, inSPDFile, warpProcessor);
            warpProcessor->completeFileAndClose(inSPDFile);

            delete exporter;
            delete warpProcessor;
            delete spdFileOut;
            delete inSPDFile;
        }
        else
        {
            throw spdlib::SPDException("Either the Shift or Warp option needs to be specified.");
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

    std::cout << "spdwarp - end\n";
}

