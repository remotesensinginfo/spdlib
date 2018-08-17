/*
 *  main.cpp
 *  spdlib
 *
 *  Created by Pete Bunting on 21/03/2012.
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

#include "spd/SPDFile.h"
#include "spd/SPDFileReader.h"
#include "spd/SPDException.h"

#include "spd/spd-config.h"

int main (int argc, char * const argv[])
{
    std::cout.precision(12);

    std::cout << "spdversion " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	std::cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	std::cout << "and you are welcome to redistribute it under certain conditions; See\n";
	std::cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	std::cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << std::endl;
	
	try
	{
        TCLAP::CmdLine cmd("Prints version information: spdversion", ' ', "1.0.0");
		
		TCLAP::UnlabeledMultiArg<std::string> multiFileNames("File", "File names for the input files", false, "string");
		cmd.add( multiFileNames );
		cmd.parse( argc, argv );
		
		std::vector<std::string> fileNames = multiFileNames.getValue();		
		if(fileNames.size() > 0)
		{
            spdlib::SPDFileReader spdReader;
            spdlib::SPDFile *spdFile;
			for(unsigned int i = 0; i < fileNames.size(); ++i)
			{
                spdFile = new spdlib::SPDFile(fileNames.at(i));
                spdReader.readHeaderInfo(fileNames.at(i), spdFile);
				std::cout << fileNames.at(i) << std::endl;
                std::cout << "SPD Pulse Version: " << spdFile->getPulseVersion() << std::endl;
                std::cout << "SPD Point Version: " << spdFile->getPulseVersion() << std::endl;
                delete spdFile;
			}
		}
        else
        {
            std::cout << "Mercurial Version: " << SPDLIB_REPO_VERSION << std::endl;
            std::cout << "SPD IO Library Version: " << SPDLIB_IO_VERSION << std::endl;
            std::cout << "SPD Library Version: " << SPDLIB_VERSION << std::endl;
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
	
    std::cout << "spdversion - end\n";
}

