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
#include <fstream>
#include <stdexcept>

#include <spd/tclap/CmdLine.h>

#include "spd/SPDTextFileUtilities.h"
#include "spd/SPDFile.h"
#include "spd/SPDFileReader.h"
#include "spd/SPDException.h"

#include "spd/spd-config.h"

int main (int argc, char * const argv[]) 
{
    std::cout.precision(12);
    
    std::cout << "spdoverlap " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	std::cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	std::cout << "and you are welcome to redistribute it under certain conditions; See\n";
	std::cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	std::cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << std::endl;
	
	try 
	{
        TCLAP::CmdLine cmd("Calculate the overlap between UPD and SPD files: spdoverlap", ' ', "1.0.0");
		
        TCLAP::SwitchArg cartesianSwitch("c","cartesian","Find cartesian overlap.", false);
        TCLAP::SwitchArg sphericalSwitch("s","spherical","Find spherical overlap.", false);
        std::vector<TCLAP::Arg*> arguments;
        arguments.push_back(&cartesianSwitch);
        arguments.push_back(&sphericalSwitch);
        cmd.xorAdd(arguments);
        
        TCLAP::ValueArg<std::string> outputFileArg("o","output","The output file.",false,"","String");
		cmd.add( outputFileArg );
        
		TCLAP::UnlabeledMultiArg<std::string> multiFileNames("Files", "File names for the output (if required) and input files", true, "string");
		cmd.add( multiFileNames );
		cmd.parse( argc, argv );
		
        bool outToConsole = false;
        if(outputFileArg.getValue() == "")
        {
            outToConsole = true;
        }
        std::string outputTextFile = outputFileArg.getValue();
        std::vector<std::string> fileNames = multiFileNames.getValue();
        boost::uint_fast16_t numOfFiles = fileNames.size();
        
        spdlib::SPDFileReader spdReader;
        
        spdlib::SPDFile **spdFiles = new spdlib::SPDFile*[numOfFiles];
        for(boost::uint_fast16_t i = 0; i < fileNames.size(); ++i)
        {
            spdFiles[i] = new spdlib::SPDFile(fileNames.at(i));
            spdReader.readHeaderInfo(fileNames.at(i), spdFiles[i]);
        }
        
        double *overlap = NULL;
        
        spdlib::SPDFileProcessingUtilities spdUtils;
        if(cartesianSwitch.getValue())
        {
            overlap = spdUtils.calcCartesainOverlap(spdFiles, numOfFiles);
        }
        else if(sphericalSwitch.getValue())
        {
            overlap = spdUtils.calcSphericalOverlap(spdFiles, numOfFiles);
        }
        else
        {
            throw spdlib::SPDProcessingException("Option was not recognised, need to seleted either spherical or cartesian.");
        }
        
        for(boost::uint_fast16_t i = 0; i < numOfFiles; ++i)
        {
            delete spdFiles[i];
        }
        delete[] spdFiles;
        
        // Output:
        /*
         * overlap[0] = min X
         * overlap[1] = max X
         * overlap[2] = min Y
         * overlap[3] = max Y
         * overlap[4] = min Z
         * overlap[5] = max Z
         */
        //OR
        /*
         * overlap[0] = min Azimuth
         * overlap[1] = max Azimuth
         * overlap[2] = min Zenith
         * overlap[3] = max Zenith
         * overlap[4] = min Range
         * overlap[5] = max Range
         */
        if(outToConsole)
        {
            if(cartesianSwitch.getValue())
            {
                std::cout << "X: [" << overlap[0] << "," << overlap[1] << "]\n";
                std::cout << "Y: [" << overlap[2] << "," << overlap[3] << "]\n";
                std::cout << "Z: [" << overlap[4] << "," << overlap[5] << "]\n";
            }
            else if(sphericalSwitch.getValue())
            {
                std::cout << "Azimuth: [" << overlap[0] << "," << overlap[1] << "]\n";
                std::cout << "Zenith: [" << overlap[2] << "," << overlap[3] << "]\n";
                std::cout << "Range: [" << overlap[4] << "," << overlap[5] << "]\n";
            }
        }
        else
        {
            std::ofstream outASCIIFile;
            outASCIIFile.open(outputTextFile.c_str(), std::ios::out | std::ios::trunc);
            
            if(!outASCIIFile.is_open())
            {               
                std::string message = std::string("Could not open file ") + outputTextFile;
                throw spdlib::SPDException(message);
            }            
            outASCIIFile.precision(12);
            
            if(cartesianSwitch.getValue())
            {
                outASCIIFile << "#Cartesian" << std::endl;
            }
            else if(sphericalSwitch.getValue())
            {
                outASCIIFile << "#Spherical" << std::endl;
            }
            outASCIIFile << overlap[0] << std::endl;
            outASCIIFile << overlap[1] << std::endl;
            outASCIIFile << overlap[2] << std::endl;
            outASCIIFile << overlap[3] << std::endl;
            outASCIIFile << overlap[4] << std::endl;
            outASCIIFile << overlap[5] << std::endl;
            outASCIIFile.flush();
            outASCIIFile.close();
        }
        delete[] overlap;
	}
	catch (TCLAP::ArgException &e) 
	{
		std::cerr << "Parse Error: " << e.what() << std::endl;
	}
	catch(spdlib::SPDException &e)
	{
		std::cerr << "Error: " << e.what() << std::endl;
	}
	std::cout << "spdoverlap - end\n";
}

