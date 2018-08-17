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

int main (int argc, char * const argv[])
{
    std::cout.precision(12);

    std::cout << "spdmetrics " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	std::cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	std::cout << "and you are welcome to redistribute it under certain conditions; See\n";
	std::cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	std::cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << std::endl;
	
	try
	{
        TCLAP::CmdLine cmd("Calculate metrics : spdmetrics", ' ', "1.0.0");
		
        TCLAP::SwitchArg imageStatsSwitch("","image","Run metrics with image output", false);
		TCLAP::SwitchArg vectorStatsSwitch("","vector","Run metrics with vector output", false);
        TCLAP::SwitchArg asciiStatsSwitch("","ascii","Run metrics with ASCII output", false);

        std::vector<TCLAP::Arg*> arguments;
        arguments.push_back(&imageStatsSwitch);
        arguments.push_back(&vectorStatsSwitch);
        arguments.push_back(&asciiStatsSwitch);

        cmd.xorAdd(arguments);

        TCLAP::ValueArg<boost::uint_fast32_t> numOfRowsBlockArg("r","blockrows","Number of rows within a block (Default 100)",false,100,"unsigned int");
		cmd.add( numOfRowsBlockArg );

        TCLAP::ValueArg<boost::uint_fast32_t> numOfColsBlockArg("c","blockcols","Number of columns within a block (Default 0) - Note values greater than 1 result in a non-sequencial SPD file.",false,0,"unsigned int");
		cmd.add( numOfColsBlockArg );

        TCLAP::ValueArg<float> binSizeArg("b","binsize","Bin size for processing and output image (Default 0) - Note 0 will use the native SPD file bin size.",false,0,"float");
		cmd.add( binSizeArg );

        TCLAP::ValueArg<std::string> imgFormatArg("f","format","Image format (GDAL driver string), Default is ENVI.",false,"ENVI","string");
		cmd.add( imgFormatArg );

        TCLAP::ValueArg<std::string> inputFileArg("i","input","The input SPD file.",true,"","String");
		cmd.add( inputFileArg );

        TCLAP::ValueArg<std::string> outputFileArg("o","output","The output file.",true,"","String");
		cmd.add( outputFileArg );

		TCLAP::ValueArg<std::string> xmlFileArg("m","metricsxml","The output SPD file.",true,"","String");
		cmd.add( xmlFileArg );

        TCLAP::ValueArg<std::string> vectorFileArg("v","vectorfile","The input vector file.",false,"","String");
		cmd.add( vectorFileArg );
		
        cmd.parse( argc, argv );
		
		if(imageStatsSwitch.getValue())
		{
            std::string inXMLFilePath = xmlFileArg.getValue();
            std::string inSPDFilePath = inputFileArg.getValue();
            std::string outFilePath = outputFileArg.getValue();

            spdlib::SPDCalcMetrics calcMetrics;
            calcMetrics.calcMetricToImage(inXMLFilePath, inSPDFilePath, outFilePath, numOfColsBlockArg.getValue(), numOfRowsBlockArg.getValue(), binSizeArg.getValue(), imgFormatArg.getValue());
		}
        else if(vectorStatsSwitch.getValue())
		{
            std::string inXMLFilePath = xmlFileArg.getValue();
            std::string inSPDFilePath = inputFileArg.getValue();
            std::string inVectorFile = vectorFileArg.getValue();
            std::string outFilePath = outputFileArg.getValue();

            spdlib::SPDCalcMetrics calcMetrics;
            calcMetrics.calcMetricToVectorShp(inXMLFilePath, inSPDFilePath, inVectorFile, outFilePath, true, true);
        }
        else if(asciiStatsSwitch.getValue())
        {
            std::string inXMLFilePath = xmlFileArg.getValue();
            std::string inSPDFilePath = inputFileArg.getValue();
            std::string inVectorFile = vectorFileArg.getValue();
            std::string outFilePath = outputFileArg.getValue();

            spdlib::SPDCalcMetrics calcMetrics;
            calcMetrics.calcMetricForVector2ASCII(inXMLFilePath, inSPDFilePath, inVectorFile, outFilePath);
        }
        else
        {
            throw spdlib::SPDException("Option not recognised. Must select image, vector or ASCII.");
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
	std::cout << "spdmetrics - end\n";
}

