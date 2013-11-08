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

#include <spd/tclap/CmdLine.h>

#include "spd/SPDTextFileUtilities.h"
#include "spd/SPDRemoveVerticalNoise.h"
#include "spd/SPDException.h"
#include "spd/SPDProcessPulses.h"
#include "spd/SPDProcessingException.h"
#include "spd/SPDCalcMetrics.h"

#include "spd/spd-config.h"

int main (int argc, char * const argv[]) 
{
    std::cout.precision(12);
    
    std::cout << "spdrmnoise " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	std::cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	std::cout << "and you are welcome to redistribute it under certain conditions; See\n";
	std::cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	std::cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << std::endl;
	
	try 
	{
        TCLAP::CmdLine cmd("Remove vertical noise from LiDAR datasets: spdrmnoise", ' ', "1.0.0");
        
        TCLAP::ValueArg<boost::uint_fast32_t> numOfRowsBlockArg("r","blockrows","Number of rows within a block (Default 100)",false,100,"unsigned int");
		cmd.add( numOfRowsBlockArg );
        
        TCLAP::ValueArg<boost::uint_fast32_t> numOfColsBlockArg("c","blockcols","Number of columns within a block (Default 0) - Note values greater than 1 result in a non-sequencial SPD file.",false,0,"unsigned int");
		cmd.add( numOfColsBlockArg );

        TCLAP::ValueArg<float> absUpperArg("","absup","Absolute upper threshold for returns which are to be removed.",false,0,"float");
        cmd.add( absUpperArg );
        
        TCLAP::ValueArg<float> absLowerArg("","abslow","Absolute lower threshold for returns which are to be removed.",false,0,"float");
        cmd.add( absLowerArg );
        
        TCLAP::ValueArg<float> relUpperArg("","relup","Relative (to median) upper threshold for returns which are to be removed.",false,0,"float");
        cmd.add( relUpperArg );
        
        TCLAP::ValueArg<float> relLowerArg("","rellow","Relative (to median) lower threshold for returns which are to be removed.",false,0,"float");
        cmd.add( relLowerArg );
        
        TCLAP::ValueArg<float> gRelUpperArg("","grelup","Global relative (to median) upper threshold for returns which are to be removed.",false,0,"float");
        cmd.add( gRelUpperArg );
        
        TCLAP::ValueArg<float> gRelLowerArg("","grellow","Global relative (to median) lower threshold for returns which are to be removed.",false,0,"float");
        cmd.add( gRelLowerArg );
        
		TCLAP::ValueArg<std::string> inputFileArg("i","input","The input SPD file.",true,"","String");
		cmd.add( inputFileArg );
        
        TCLAP::ValueArg<std::string> outputFileArg("o","output","The output SPD file.",true,"","String");
		cmd.add( outputFileArg );
        
		cmd.parse( argc, argv );
				
		std::string inputFile = inputFileArg.getValue();
		std::string outputFile = outputFileArg.getValue();
        
        bool absUpSet = absUpperArg.isSet();
        bool absLowSet = absLowerArg.isSet();
        bool relUpSet = relUpperArg.isSet();
        bool relLowSet = relLowerArg.isSet();
        bool gRelUpSet = gRelUpperArg.isSet();
        bool gRelLowSet = gRelLowerArg.isSet();
        
        if( (absUpSet | absLowSet) & (gRelUpSet | gRelLowSet))
        {
            throw spdlib::SPDException("Either global relative or absolute thresholding can be used.");
        }
        
        spdlib::SPDFile *spdInFile = new spdlib::SPDFile(inputFile);
        spdlib::SPDSetupProcessPulses processPulses = spdlib::SPDSetupProcessPulses(numOfColsBlockArg.getValue(), numOfRowsBlockArg.getValue(), true);
        
        double gMedianZ = 0;
        if(gRelUpSet | gRelLowSet)
        {
            std::cout << "Calculating Global Median\n";
            spdlib::SPDCalcZMedianVal *pulseProcessorCalcMedian = new spdlib::SPDCalcZMedianVal();
            processPulses.processPulses(pulseProcessorCalcMedian, spdInFile);
            gMedianZ = pulseProcessorCalcMedian->getMedianMedianVal();
            std::cout << "Global Median = " << gMedianZ << std::endl;
        }
        
        float absUpThres = 0;
        if(absUpSet)
        {
            absUpThres = absUpperArg.getValue();
        }
        else if(gRelUpSet)
        {
            absUpThres = gMedianZ + gRelUpperArg.getValue();
            absUpSet = true;
        }
        
        float absLowThres = 0;
        if(absLowSet)
        {
            absLowThres = absLowerArg.getValue();
        }
        else if(gRelLowSet)
        {
            absLowThres = gMedianZ - gRelLowerArg.getValue();
            absLowSet = true;
        }
        
        spdlib::SPDPulseProcessor *pulseProcessor = new spdlib::SPDRemoveVerticalNoise(absUpSet, absLowSet, relUpSet, relLowSet, absUpThres, absLowerArg.getValue(), relUpperArg.getValue(), relLowerArg.getValue());
        processPulses.processPulsesWithOutputSPD(pulseProcessor, spdInFile, outputFile);
        
        delete pulseProcessor;
        delete spdInFile;
	}
	catch (TCLAP::ArgException &e) 
	{
		std::cerr << "Parse Error: " << e.what() << std::endl;
	}
	catch(spdlib::SPDException &e)
	{
		std::cerr << "Error: " << e.what() << std::endl;
	}
	std::cout << "spdrmnoise - end\n";
}

