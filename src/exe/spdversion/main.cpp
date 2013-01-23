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

using namespace std;
using namespace spdlib;
using namespace TCLAP;

int main (int argc, char * const argv[]) 
{
	cout << "spdversion " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	cout << "and you are welcome to redistribute it under certain conditions; See\n";
	cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << endl;
	
	try 
	{
		CmdLine cmd("Prints version information: spdversion", ' ', "1.0.0");
		
		UnlabeledMultiArg<string> multiFileNames("File", "File names for the input files", false, "string");
		cmd.add( multiFileNames );
		cmd.parse( argc, argv );
		
		vector<string> fileNames = multiFileNames.getValue();		
		if(fileNames.size() > 0)
		{
            SPDFileReader spdReader;
            SPDFile *spdFile; 
			for(unsigned int i = 0; i < fileNames.size(); ++i)
			{
                spdFile = new SPDFile(fileNames.at(i));
                spdReader.readHeaderInfo(fileNames.at(i), spdFile);
				cout << fileNames.at(i) << endl;
                cout << "SPD Pulse Version: " << spdFile->getPulseVersion() << endl;
                cout << "SPD Point Version: " << spdFile->getPulseVersion() << endl;
                delete spdFile;
			}
		}
        else
        {
            cout << "Mercurial Version: " << SPDLIB_REPO_VERSION << endl;
            cout << "SPD IO Library Version: " << SPDLIB_IO_VERSION << endl;
            cout << "SPD Library Version: " << SPDLIB_VERSION << endl;
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
	
}

