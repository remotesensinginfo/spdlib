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
#include "spd/SPDDecomposeWaveforms.h"

#include "spd/spd-config.h"

int main (int argc, char * const argv[])
{
    std::cout.precision(12);

    std::cout << "spddecomp " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	std::cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	std::cout << "and you are welcome to redistribute it under certain conditions; See\n";
	std::cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	std::cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << std::endl;

	try
	{
        TCLAP::CmdLine cmd("Decompose full waveform data to create discrete points: spddecomp", ' ', "1.0.0");

        TCLAP::ValueArg<boost::uint_fast32_t> numOfRowsBlockArg("r","blockrows","Number of rows within a block (Default 100)",false,100,"unsigned int");
		cmd.add( numOfRowsBlockArg );

        TCLAP::ValueArg<boost::uint_fast32_t> numOfColsBlockArg("c","blockcols","Number of columns within a block (Default 0) - Note values greater than 1 result in a non-sequencial SPD file.",false,0,"unsigned int");
		cmd.add( numOfColsBlockArg );

        TCLAP::ValueArg<uint_fast32_t> intThresholdArg("t","threshold","Noise threshold below which peaks are ignored (Default: Value in pulse->waveNoiseThreshold)",false,0, "uint_fast32_t");
		cmd.add( intThresholdArg );

		TCLAP::SwitchArg noiseSetSwitch("n","noise","Estimate noise. Only applicable when --all is set. Note an initial estimate is required for peak detection (see -t)", false);
		cmd.add( noiseSetSwitch );

        TCLAP::ValueArg<uint_fast32_t> intDecayThresholdArg("e","decaythres","Intensity threshold above which a decay function is used (Default 100)",false,100, "uint_fast32_t");
		cmd.add( intDecayThresholdArg );

		TCLAP::ValueArg<uint_fast32_t> windowThresholdArg("w","window","Window for the values taken either side of the peak for fitting (Default 5)",false,5, "uint_fast32_t");
		cmd.add( windowThresholdArg );

		TCLAP::ValueArg<float> decayArg("d","decay","Decay value for ignoring ringing artifacts (Default 5)",false,5, "float");
		cmd.add( decayArg );

		TCLAP::SwitchArg allGausSwitch("a","all","Fit all Gaussian at once", false);
		cmd.add( allGausSwitch );

        TCLAP::ValueArg<std::string> inputFileArg("i","input","The input file.",true,"","String");
		cmd.add( inputFileArg );

        TCLAP::ValueArg<std::string> outputFileArg("o","output","The output file.",true,"","String");
		cmd.add( outputFileArg );

		cmd.parse( argc, argv );

		std::string inSPDFilePath = inputFileArg.getValue();
        std::string outFilePath = outputFileArg.getValue();

        uint_fast32_t intThreshold = intThresholdArg.getValue();
        bool noiseSet = noiseSetSwitch.getValue();
        uint_fast32_t decayThres = intDecayThresholdArg.getValue();
        uint_fast32_t window = windowThresholdArg.getValue();
        float decayVal = decayArg.getValue();

        spdlib::SPDDecompOption decompOption = spdlib::spd_decomp_indvid;
        if(allGausSwitch.getValue())
        {
            decompOption = spdlib::spd_decomp_all;
        }

        bool thresholdSet = false;
        if(intThresholdArg.isSet())
        {
            thresholdSet = true;
        }

        spdlib::SPDDecomposeWaveforms decompSPDFile;
        decompSPDFile.decomposeWaveforms(inSPDFilePath, outFilePath, numOfColsBlockArg.getValue(), numOfRowsBlockArg.getValue(), decompOption, intThreshold, thresholdSet, noiseSet, window, decayThres, decayVal);

	}
	catch (TCLAP::ArgException &e)
	{
		std::cerr << "Parse Error: " << e.what() << std::endl;
	}
	catch(spdlib::SPDException &e)
	{
		std::cerr << "Error: " << e.what() << std::endl;
	}

    std::cout << "spddecomp - end\n";
}

