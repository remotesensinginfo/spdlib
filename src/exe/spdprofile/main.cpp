/*
 *  main.cpp
 *  spdlib
 *
 *  Created by Pete Bunting on 31/01/2013.
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
#include "spd/SPDCreateVerticalProfiles.h"

#include "spd/spd-config.h"

int main (int argc, char * const argv[])
{
	std::cout << "spdprofile " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	std::cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	std::cout << "and you are welcome to redistribute it under certain conditions; See\n";
	std::cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	std::cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << std::endl;
	
	try
	{
        TCLAP::CmdLine cmd("Generate vertical profiles: spdprofile", ' ', "1.0.0");
        
        TCLAP::SwitchArg smoothSwitch("","smooth","Apply a Savitzky Golay smoothing to the profiles.", false);
		cmd.add( smoothSwitch );
        
		TCLAP::ValueArg<boost::uint_fast32_t> smoothPolyOrderArg("o","order","The order of the polynomial used to smooth the profile (Default: 3).",false,3,"unsigned int");
		cmd.add( smoothPolyOrderArg );
		
		TCLAP::ValueArg<boost::uint_fast32_t> smoothWindowArg("w","window","The window size ((w*2)+1) used for the smoothing filter (Default: 3).",false,3,"unsigned int");
		cmd.add( smoothWindowArg );
        
        TCLAP::ValueArg<float> minHeightArg("m","minheight","The the height below which points are ignored (Default: 0).",false,0,"float");
		cmd.add( minHeightArg );
        
        TCLAP::ValueArg<boost::uint_fast32_t> maxHeightArg("t","topheight","The highest bin of the profile (Default: 40).",false,40,"unsigned int");
		cmd.add( maxHeightArg );
        
        TCLAP::ValueArg<boost::uint_fast32_t> numOfBinsArg("n","numbins","The number of bins within the profile (Default: 20).",false,20,"unsigned int");
		cmd.add( numOfBinsArg );        
        
        TCLAP::ValueArg<boost::uint_fast32_t> numOfRowsBlockArg("r","blockrows","Number of rows within a block (Default 100)",false,100,"unsigned int");
		cmd.add( numOfRowsBlockArg );
        
        TCLAP::ValueArg<boost::uint_fast32_t> numOfColsBlockArg("c","blockcols","Number of columns within a block (Default 0) - Note values greater than 1 result in a non-sequencial SPD file.",false,0,"unsigned int");
		cmd.add( numOfColsBlockArg );
        
        TCLAP::ValueArg<float> binSizeArg("b","binsize","Bin size for processing and output image (Default 0) - Note 0 will use the native SPD file bin size.",false,0,"float");
		cmd.add( binSizeArg );
        
        TCLAP::ValueArg<std::string> imgFormatArg("f","format","Image format (GDAL driver string), Default is ENVI.",false,"ENVI","std::string");
		cmd.add( imgFormatArg );
        
		TCLAP::UnlabeledMultiArg<std::string> multiFileNames("File", "File names for the files (input, output)", false, "std::string");
		cmd.add( multiFileNames );
		cmd.parse( argc, argv );
		
		std::vector<std::string> fileNames = multiFileNames.getValue();
		if(fileNames.size() == 2)
		{
            std::cout.precision(12);
            std::string inSPDFilePath = fileNames.at(0);
            std::string outFilePath = fileNames.at(1);
            
            bool useSmoothing = false;
            boost::uint_fast32_t smoothWindowSize = 0;
            boost::uint_fast32_t smoothPolyOrder = 0;
            
            boost::uint_fast32_t maxProfileHeight = maxHeightArg.getValue();
            boost::uint_fast32_t numOfBins = numOfBinsArg.getValue();
            float minPtHeight = minHeightArg.getValue();
            
            if(smoothSwitch.getValue())
            {
                useSmoothing = true;
                smoothWindowSize = smoothWindowArg.getValue();
                smoothPolyOrder = smoothPolyOrderArg.getValue();
                
                if(smoothPolyOrder > smoothWindowSize)
                {
                    throw spdlib::SPDException("The smoothing window size must be larger or equal to the polynomial order.");
                }
            }
            
            spdlib::SPDFile *spdInFile = new spdlib::SPDFile(inSPDFilePath);
            
            spdlib::SPDCreateVerticalProfiles *createProfiles = new spdlib::SPDCreateVerticalProfiles(useSmoothing, smoothWindowSize, smoothPolyOrder, maxProfileHeight, numOfBins, minPtHeight);
            spdlib::SPDSetupProcessPulses processPulses = spdlib::SPDSetupProcessPulses(numOfColsBlockArg.getValue(), numOfRowsBlockArg.getValue(), true);
            processPulses.processPulsesWithOutputImage(createProfiles, spdInFile, outFilePath, numOfBins, binSizeArg.getValue(), imgFormatArg.getValue());
            
            delete spdInFile;
            delete createProfiles;
            
		}
        else
        {
            for(unsigned int i = 0; i < fileNames.size(); ++i)
			{
                std::cout << i << ":\t" << fileNames.at(i) << std::endl;
            }
            throw spdlib::SPDException("Only 2 files can be provided");
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
}

