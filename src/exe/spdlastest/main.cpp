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

#include <boost/cstdint.hpp>

#include <spd/tclap/CmdLine.h>

#include <liblas/liblas.hpp>

#include "spd/SPDException.h"
#include "spd/SPDTextFileUtilities.h"

#include "spd/spd-config.h"

using namespace std;
using namespace spdlib;
using namespace TCLAP;

namespace spdlib
{
	class SPDTestLasFile
	{
	public:
		SPDTestLasFile(){};
        
        void printPulsesFromLASFile(string inputFile, uint_fast32_t startIdx, uint_fast16_t numPulses) throw(SPDException)
        {
            try
            {
                cout.precision(10);
                ifstream ifs;
                ifs.open(inputFile.c_str(), ios::in | ios::binary);
                if(ifs.is_open())
                {
                    liblas::ReaderFactory lasReaderFactory;
                    liblas::Reader reader = lasReaderFactory.CreateWithStream(ifs);
                    std::cout << "here\n";
                    liblas::Header const& header = reader.GetHeader();
                    cout << "Number of Points in LAS file: " <<  header.GetPointRecordsCount() << endl;
                    
                    uint_fast16_t numOfReturnsInPulse = 0;
                    uint_fast64_t numOfPoints = 0;
                    uint_fast64_t numOfPulses = 0;
                    uint_fast64_t numOfPrintedPulses = 0;
                    
                    while (reader.ReadNextPoint())
                    {
                        liblas::Point const& p = reader.GetPoint();
                        ++numOfPoints;
                        numOfReturnsInPulse = p.GetNumberOfReturns();
                        if(numOfPulses >= startIdx)
                        {
                            cout << "###################################\n";
                            cout << "Pulse " << numOfPulses << endl;
                            cout << "Number of Returns: " << numOfReturnsInPulse << endl;
                            cout << "Return: " << p.GetReturnNumber() << endl;
                            cout << "[X,Y,Z]: [" << p.GetX() << "," << p.GetY() << "," << p.GetZ() << "]\n";
                            cout << "*********************************\n";
                        }
                        
                        for(boost::uint_fast16_t i = 0; i < (numOfReturnsInPulse-1); ++i)
                        {
                            if(reader.ReadNextPoint())
                            {
                                if(numOfPulses >= startIdx)
                                {
                                    liblas::Point const& pt = reader.GetPoint();
                                    cout << "Return: " << pt.GetReturnNumber() << endl;
                                    cout << "[X,Y,Z]: [" << pt.GetX() << "," << pt.GetY() << "," << pt.GetZ() << "]\n";
                                    cout << "*********************************\n";
                                }
                                ++numOfPoints;
                            }
                            else
                            {
                                cerr << "\nWarning: The file ended unexpectedly.\n";
                                cerr << "Expected " << numOfReturnsInPulse << " but only found " << i + 1 << " returns" << endl;
                            }
                        }
                        ++numOfPulses;
                        
                        if(numOfPulses >= startIdx)
                        {
                            cout << "###################################\n";
                            ++numOfPrintedPulses;
                            if(numOfPrintedPulses >= numPulses)
                            {
                                break;
                            }
                        }
                    }
                    
                    cout << "Number of Points counted: " << numOfPoints << endl;
                    cout << "Number of Pulses counted: " << numOfPulses << endl;
                }
                else
                {
                    throw SPDException("Couldn't open input file.");
                }
            }
            catch(SPDException &e)
            {
                throw e;
            }
            catch(std::exception &e)
            {
                throw SPDException(e.what());
            }
        };
        
        void printPulsesFromLASFileNotFirstReturnsStartPulse(string inputFile) throw(SPDException)
        {
            cout.precision(10);
            ifstream ifs;
			ifs.open(inputFile.c_str(), ios::in | ios::binary);
			if(ifs.is_open())
			{
                liblas::ReaderFactory lasReaderFactory;
				liblas::Reader reader = lasReaderFactory.CreateWithStream(ifs);
				liblas::Header const& header = reader.GetHeader();
                cout << "Number of Points in LAS file: " <<  header.GetPointRecordsCount() << endl;
                
                uint_fast16_t numOfReturnsInPulse = 0;
                uint_fast64_t numOfPoints = 0;
                uint_fast64_t numOfPulses = 0;
                uint_fast64_t numOfPrintedPoints = 0;
                
                while (reader.ReadNextPoint())
				{
                    liblas::Point const& p = reader.GetPoint();
                    ++numOfPoints;
                    numOfReturnsInPulse = p.GetNumberOfReturns();
                    if(p.GetReturnNumber() > 1)
                    {
                        cout << "###################################\n";
                        cout << "Pulse " << numOfPulses << endl;
                        cout << "Number of Returns: " << numOfReturnsInPulse << endl;
                        cout << "Return: " << p.GetReturnNumber() << endl;
                        cout << "[X,Y,Z]: [" << p.GetX() << "," << p.GetY() << "," << p.GetZ() << "]\n";
                        cout << "###################################\n";
                        ++numOfPrintedPoints;
                    }
                    
                    for(boost::uint_fast16_t i = 0; i < (numOfReturnsInPulse-1); ++i)
					{
						if(reader.ReadNextPoint())
						{
							++numOfPoints;
						}
						else
						{
							cerr << "\nWarning: The file ended unexpectedly.\n";
                            cerr << "Expected " << numOfReturnsInPulse << " but only found " << i + 1 << " returns" << endl;
						}
					}
                    ++numOfPulses;
                }
                
                cout << "Number of Points counted: " << numOfPoints << endl;
                cout << "Number of Pulses counted: " << numOfPulses << endl;
            }
            else
            {
                throw SPDException("Couldn't open input file.");
            }
        };
        
        void countNumPulses(string inputFile) throw(SPDException)
        {
            ifstream ifs;
			ifs.open(inputFile.c_str(), ios::in | ios::binary);
			if(ifs.is_open())
			{
                liblas::ReaderFactory lasReaderFactory;
				liblas::Reader reader = lasReaderFactory.CreateWithStream(ifs);
				liblas::Header const& header = reader.GetHeader();
                cout << "Number of Points in LAS file: " <<  header.GetPointRecordsCount() << endl;
                
                uint_fast16_t numOfReturnsInPulse = 0;
                uint_fast64_t numOfPoints = 0;
                uint_fast64_t numOfPulses = 0;
                
                while (reader.ReadNextPoint())
				{
                    liblas::Point const& p = reader.GetPoint();
                    ++numOfPoints;
                    numOfReturnsInPulse = p.GetNumberOfReturns();
                    for(boost::uint_fast16_t i = 0; i < (numOfReturnsInPulse-1); ++i)
					{
						if(reader.ReadNextPoint())
						{
							//liblas::Point const& p = reader.GetPoint();
							++numOfPoints;
						}
						else
						{
							cerr << "\nWarning: The file ended unexpectedly.\n";
                            cerr << "Expected " << numOfReturnsInPulse << " but only found " << i + 1 << " returns" << endl;
						}
					}
					++numOfPulses;
                }
                
                cout << "Number of Points counted: " << numOfPoints << endl;
                cout << "Number of Pulses counted: " << numOfPulses << endl;
            }
            else
            {
                throw SPDException("Couldn't open input file.");
            }
        };
        
		~SPDTestLasFile(){};
	};
}


int main (int argc, char * const argv[]) 
{
    cout << "spdlastest " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	cout << "and you are welcome to redistribute it under certain conditions; See\n";
	cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << endl;
	
	try 
	{
		CmdLine cmd("Print data pulses from a LAS file - for debugging: spdlastest", ' ', "1.0.0");
		
		ValueArg<uint_fast32_t> startPulseArg("s","start","Starting pulse index (Default 0)",false,0,"unsigned int");
		cmd.add( startPulseArg );
        
        ValueArg<uint_fast16_t> numPulseArg("n","number","Number of pulses to be printed out (Default 10)",false,10,"unsigned int");
		cmd.add( numPulseArg );
		
        SwitchArg printPulsesArg("p","print","Print a selction of pulses from LAS file.", false);		
		SwitchArg countPulsesArg("c","count","Count the number of pulses in LAS file.", false);
        SwitchArg notFirstStartPulseArg("f","notfirst","Print the returns which start a pulse with point IDs greater than 1", false);
        
        vector<Arg*> arguments;
        arguments.push_back(&printPulsesArg);
        arguments.push_back(&countPulsesArg);
        arguments.push_back(&notFirstStartPulseArg);
        
        cmd.xorAdd(arguments);
        
		UnlabeledMultiArg<string> inputFileList("Files", "File name of the input file", true, "string");
		cmd.add( inputFileList );
		cmd.parse( argc, argv );
		
		vector<string> fileNames = inputFileList.getValue();		
		if(fileNames.size() != 1)
		{
			for(unsigned int i = 0; i < fileNames.size(); ++i)
			{
				cout << i << ": " << fileNames.at(i) << endl;
			}
			
			SPDTextFileUtilities textUtils;
			string message = string("Only 1 input file path should have been specified, ") + textUtils.uInt32bittostring(fileNames.size()) + string(" were provided.");
			throw SPDException(message);
		}
		
		string inputFile = fileNames.at(0);        
        if(printPulsesArg.isSet())
        {
            uint_fast32_t startIdx = startPulseArg.getValue();
            uint_fast16_t numPulses = numPulseArg.getValue();
            SPDTestLasFile testLASFile;
            testLASFile.printPulsesFromLASFile(inputFile, startIdx, numPulses);
        }
        else if(countPulsesArg.isSet())
        {
            SPDTestLasFile testLASFile;
            testLASFile.countNumPulses(inputFile);
        }
        else if(notFirstStartPulseArg.isSet())
        {
            SPDTestLasFile testLASFile;
            testLASFile.printPulsesFromLASFileNotFirstReturnsStartPulse(inputFile);
        }
        else
        {
            cout << "No option (e.g., print or count) provided\n";
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
	std::cout << "spdlastest - end\n";
}

