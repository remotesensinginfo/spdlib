/*
 *  main.cpp
 *  spdlib
 *
 *  Created by Pete Bunting on 30/11/2010.
 *  Copyright 2011 SPDLib. All rights reserved.
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
#include "spd/SPDException.h"

#include "spd/SPDDefineRGBValues.h"

#include "spd/spd-config.h"

int main (int argc, char * const argv[])
{
    std::cout.precision(12);

    std::cout << "spddefrgb " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	std::cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	std::cout << "and you are welcome to redistribute it under certain conditions; See\n";
	std::cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	std::cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << std::endl;
	
	try
	{
        TCLAP::CmdLine cmd("Define the RGB values on the SPDFile: spddefrgb", ' ', "1.0.0");
		
        TCLAP::SwitchArg defRGBSwitch("","define","Define the RGB values on an SPD file from an input image.", false);
		TCLAP::SwitchArg stretchRGBSwitch("","stretch","Stretch existing RGB values to a range of 0 to 255.", false);

        std::vector<TCLAP::Arg*> arguments;
        arguments.push_back(&defRGBSwitch);
        arguments.push_back(&stretchRGBSwitch);
        cmd.xorAdd(arguments);

        TCLAP::SwitchArg linearStretchSwitch("","linear","Use a linear stretch between the min and max values.", false);
		TCLAP::SwitchArg stddevRGBSwitch("","stddev","Use a linear 2 standard deviation stretch.", false);
        cmd.add(linearStretchSwitch);
        cmd.add(stddevRGBSwitch);

        TCLAP::ValueArg<boost::uint_fast32_t> numOfRowsBlockArg("r","blockrows","Number of rows within a block (Default 100)",false,100,"unsigned int");
		cmd.add( numOfRowsBlockArg );

        TCLAP::ValueArg<boost::uint_fast32_t> numOfColsBlockArg("c","blockcols","Number of columns within a block (Default 0) - Note values greater than 1 result in a non-sequencial SPD file.",false,0,"unsigned int");
		cmd.add( numOfColsBlockArg );
		
		TCLAP::ValueArg<uint_fast16_t> redBandArg("","red","Image band for red channel",false,1,"uint_fast16_t");
		cmd.add( redBandArg );
		
		TCLAP::ValueArg<uint_fast16_t> greenBandArg("","green","Image band for green channel",false,2,"uint_fast16_t");
		cmd.add( greenBandArg );
		
		TCLAP::ValueArg<uint_fast16_t> blueBandArg("","blue","Image band for blue channel",false,3,"uint_fast16_t");
		cmd.add( blueBandArg );

        TCLAP::ValueArg<float> stretchCoeffrg("","coef","The coefficient for the standard deviation stretch (Default is 2)",false,2,"float");
		cmd.add( stretchCoeffrg );

        TCLAP::SwitchArg stretchIndependSwitch("","independ","Stretch the RGB values independently.", false);
		cmd.add( stretchIndependSwitch );

		TCLAP::ValueArg<std::string> inputFileArg("i","input","The input SPD file.",true,"","String");
		cmd.add( inputFileArg );

        TCLAP::ValueArg<std::string> imageFileArg("","image","The input image file.",false,"","String");
		cmd.add( imageFileArg );

        TCLAP::ValueArg<std::string> outputFileArg("o","output","The output SPD file.",true,"","String");
		cmd.add( outputFileArg );

		cmd.parse( argc, argv );
		
        if(defRGBSwitch.getValue())
        {
            std::cout << "Defining the RGB values from an input image.\n";

            std::string inputSPDFile = inputFileArg.getValue();
            std::string inputImageFile = imageFileArg.getValue();
            std::string outputSPDFile = outputFileArg.getValue();

            uint_fast16_t redBand = redBandArg.getValue()-1;	
            uint_fast16_t greenBand = greenBandArg.getValue()-1;	
            uint_fast16_t blueBand = blueBandArg.getValue()-1;

            spdlib::SPDFile *spdInFile = new spdlib::SPDFile(inputSPDFile);
            spdlib::SPDPulseProcessor *pulseProcessor = new spdlib::SPDDefineRGBValues(redBand, greenBand, blueBand);
            spdlib::SPDSetupProcessPulses processPulses = spdlib::SPDSetupProcessPulses(numOfColsBlockArg.getValue(), numOfRowsBlockArg.getValue(), true);
            processPulses.processPulsesWithInputImage(pulseProcessor, spdInFile, outputSPDFile, inputImageFile);

            delete spdInFile;
            delete pulseProcessor;
        }
        else if(stretchRGBSwitch.getValue())
        {
            std::cout << "Scaling the RGB values within an SPD file.\n";

            std::string inputSPDFile = inputFileArg.getValue();
            std::string outputSPDFile = outputFileArg.getValue();

            spdlib::SPDFile *spdInFile = new spdlib::SPDFile(inputSPDFile);
            spdlib::SPDFindRGBValuesStats *pulseProcessorStats = new spdlib::SPDFindRGBValuesStats();
            spdlib::SPDSetupProcessPulses processPulses = spdlib::SPDSetupProcessPulses(numOfColsBlockArg.getValue(), numOfRowsBlockArg.getValue(), true);
            std::cout << "Calculating Statistics (Mean, Min Max).\n";
            processPulses.processPulses(pulseProcessorStats, spdInFile);

            float redStretchMin = 0;
            float redStretchMax = 0;
            float greenStretchMin = 0;
            float greenStretchMax = 0;
            float blueStretchMin = 0;
            float blueStretchMax = 0;

            if(linearStretchSwitch.getValue())
            {
                redStretchMin = pulseProcessorStats->getRedMin();
                redStretchMax = pulseProcessorStats->getRedMax();
                greenStretchMin = pulseProcessorStats->getGreenMin();
                greenStretchMax = pulseProcessorStats->getGreenMax();
                blueStretchMin = pulseProcessorStats->getBlueMin();
                blueStretchMax = pulseProcessorStats->getBlueMax();
            }
            else if(stddevRGBSwitch.getValue())
            {
                float redMean = pulseProcessorStats->getRedMean();
                float greenMean = pulseProcessorStats->getGreenMean();
                float blueMean = pulseProcessorStats->getBlueMean();
                boost::uint_least16_t redMin = pulseProcessorStats->getRedMin();
                boost::uint_least16_t redMax = pulseProcessorStats->getRedMax();
                boost::uint_least16_t greenMin = pulseProcessorStats->getGreenMin();
                boost::uint_least16_t greenMax = pulseProcessorStats->getGreenMax();
                boost::uint_least16_t blueMin = pulseProcessorStats->getBlueMin();
                boost::uint_least16_t blueMax = pulseProcessorStats->getBlueMax();

                std::cout << "Red Mean = " << redMean << std::endl;
                std::cout << "Green Mean = " << greenMean << std::endl;
                std::cout << "Blue Mean = " << blueMean << std::endl;

                pulseProcessorStats->setCalcStdDev(true, redMean, greenMean, blueMean);
                std::cout << "Calculating Statistics (Std Dev).\n";
                processPulses.processPulses(pulseProcessorStats, spdInFile);

                float redStdDev = pulseProcessorStats->getRedStdDev();
                float greenStdDev = pulseProcessorStats->getGreenStdDev();
                float blueStdDev = pulseProcessorStats->getBlueStdDev();

                std::cout << "Red Standard Deviation = " << redStdDev << std::endl;
                std::cout << "Green Standard Deviation = " << greenStdDev << std::endl;
                std::cout << "Blue Standard Deviation = " << blueStdDev << std::endl;

                redStretchMin = redMean - (redStdDev*stretchCoeffrg.getValue());
                redStretchMax = redMean + (redStdDev*stretchCoeffrg.getValue());
                if(redStretchMin < redMin)
                {
                    redStretchMin = redMin;
                }
                if(redStretchMax > redMax)
                {
                    redStretchMax = redMax;
                }

                greenStretchMin = greenMean - (greenStdDev*stretchCoeffrg.getValue());
                greenStretchMax = greenMean + (greenStdDev*stretchCoeffrg.getValue());
                if(greenStretchMin < greenMin)
                {
                    greenStretchMin = greenMin;
                }
                if(greenStretchMax > greenMax)
                {
                    greenStretchMax = greenMax;
                }

                blueStretchMin = blueMean - (blueStdDev*stretchCoeffrg.getValue());
                blueStretchMax = blueMean + (blueStdDev*stretchCoeffrg.getValue());
                if(blueStretchMin < blueMin)
                {
                    blueStretchMin = blueMin;
                }
                if(blueStretchMax > blueMax)
                {
                    blueStretchMax = blueMax;
                }
            }
            else
            {
                throw spdlib::SPDException("A stretch switch needs to be defined (linear or stddev).");
            }

            std::cout << "Linear stretch red between " << redStretchMin << " and " << redStretchMax << std::endl;
            std::cout << "Linear stretch green between " << greenStretchMin << " and " << greenStretchMax << std::endl;
            std::cout << "Linear stretch blue between " << blueStretchMin << " and " << blueStretchMax << std::endl;

            std::cout << "\nStretching RGB values and creating new output file.\n";
            spdlib::SPDLinearStretchRGBValues *pulseProcessorStretch = new spdlib::SPDLinearStretchRGBValues(redStretchMin, redStretchMax, greenStretchMin, greenStretchMax, blueStretchMin, blueStretchMax, stretchIndependSwitch.getValue());
            processPulses.processPulsesWithOutputSPD(pulseProcessorStretch, spdInFile, outputSPDFile);

            delete spdInFile;
            delete pulseProcessorStats;
            delete pulseProcessorStretch;
        }
        else
        {
            throw spdlib::SPDException("Either the stretch or define switches are required to define the functionality of the command.");
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
    std::cout << "spddefrgb - end\n";
}

