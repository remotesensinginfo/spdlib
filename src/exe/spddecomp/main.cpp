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

using namespace std;
using namespace spdlib;
using namespace TCLAP;

int main (int argc, char * const argv[])
{
	cout << "spddecomp " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	cout << "and you are welcome to redistribute it under certain conditions; See\n";
	cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << endl;

	try
	{
		CmdLine cmd("Decompose full waveform data to create discrete points: spddecomp", ' ', "1.0.0");

        ValueArg<boost::uint_fast32_t> numOfRowsBlockArg("r","blockrows","Number of rows within a block (Default 100)",false,100,"unsigned int");
		cmd.add( numOfRowsBlockArg );

        ValueArg<boost::uint_fast32_t> numOfColsBlockArg("c","blockcols","Number of columns within a block (Default 0) - Note values greater than 1 result in a non-sequencial SPD file.",false,0,"unsigned int");
		cmd.add( numOfColsBlockArg );


        ValueArg<uint_fast32_t> intThresholdArg("t","threshold","Noise threshold below which peaks are ignored (Default: Value in pulse->waveNoiseThreshold)",false,0, "uint_fast32_t");
		cmd.add( intThresholdArg );

		SwitchArg noiseSetSwitch("n","noise","Estimate noise. Only applicable when --all is set. Note an initial estimate is required for peak detection (see -t)", false);
		cmd.add( noiseSetSwitch );

        ValueArg<uint_fast32_t> intDecayThresholdArg("e","decaythres","Intensity threshold above which a decay function is used (Default 100)",false,100, "uint_fast32_t");
		cmd.add( intDecayThresholdArg );

		ValueArg<uint_fast32_t> windowThresholdArg("w","window","Window for the values taken either side of the peak for fitting (Default 5)",false,5, "uint_fast32_t");
		cmd.add( windowThresholdArg );

		ValueArg<float> decayArg("d","decay","Decay value for ignoring ringing artifacts (Default 5)",false,5, "float");
		cmd.add( decayArg );

		SwitchArg allGausSwitch("a","all","Fit all Gaussian at once", false);
		cmd.add( allGausSwitch );

		UnlabeledMultiArg<string> multiFileNames("File", "File names for the input files", false, "string");
		cmd.add( multiFileNames );
		cmd.parse( argc, argv );

		vector<string> fileNames = multiFileNames.getValue();
        cout << "fileNames.size() = " << fileNames.size() << endl;
		if(fileNames.size() == 2)
		{
            string inSPDFilePath = fileNames.at(0);
            string outFilePath = fileNames.at(1);

            uint_fast32_t intThreshold = intThresholdArg.getValue();
            bool noiseSet = noiseSetSwitch.getValue();
            uint_fast32_t decayThres = intDecayThresholdArg.getValue();
            uint_fast32_t window = windowThresholdArg.getValue();
            float decayVal = decayArg.getValue();

            SPDDecompOption decompOption = spd_decomp_indvid;
            if(allGausSwitch.getValue())
            {
                decompOption = spd_decomp_all;
            }

            bool thresholdSet = false;
            if(intThresholdArg.isSet())
            {
                thresholdSet = true;
            }


            //string inFilePath, string outFilePath, boost::uint_fast32_t intThreshold, bool thresholdSet, bool noiseSet
            SPDDecomposeWaveforms decompSPDFile;
            decompSPDFile.decomposeWaveforms(inSPDFilePath, outFilePath, numOfColsBlockArg.getValue(), numOfRowsBlockArg.getValue(), decompOption, intThreshold, thresholdSet, noiseSet, window, decayThres, decayVal);

        }
        else
        {
            cout << "ERROR: only 2 files can be inputted.\n";
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
    
    std::cout << "spddecomp - end\n";
}

