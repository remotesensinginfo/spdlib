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

#include <lasreader.hpp>

#include "spd/SPDException.h"
#include "spd/SPDTextFileUtilities.h"

#include "spd/spd-config.h"

namespace spdlib
{
	class SPDTestLasFile
	{
	public:
		SPDTestLasFile(){};

        void printPulsesFromLASFile(std::string inputFile, uint_fast32_t startIdx, uint_fast16_t numPulses) throw(SPDException)
        {
            try
            {
                // Open LAS file
                LASreadOpener lasreadopener;
                LASreader* lasreader = lasreadopener.open(inputFile.c_str());

                if(lasreader != 0)
                {
                    // Get header
                    LASheader *header = &lasreader->header;

                    std::cout << "Number of Points in LAS file: " << header->number_of_point_records << std::endl;

                    uint_fast16_t numOfReturnsInPulse = 0;
                    uint_fast64_t numOfPoints = 0;
                    uint_fast64_t numOfPulses = 0;
                    uint_fast64_t numOfPrintedPulses = 0;

                    while (lasreader->read_point())
                    {
                        ++numOfPoints;
                        numOfReturnsInPulse = lasreader->point.get_number_of_returns();
                        if(numOfPulses >= startIdx)
                        {
                            std::cout << "###################################\n";
                            std::cout << "Pulse " << numOfPulses << std::endl;
                            std::cout << "Number of Returns: " << numOfReturnsInPulse << std::endl;
                            std::cout << "Return: " << lasreader->point.get_return_number()<< std::endl;
                            std::cout << "[X,Y,Z]: [" << lasreader->point.get_X() << "," << lasreader->point.get_Y() << "," << lasreader->point.get_Z() << "]\n";
                            std::cout << "*********************************\n";
                        }

                        for(boost::uint_fast16_t i = 0; i < (numOfReturnsInPulse-1); ++i)
                        {
                            if(lasreader->read_point())
                            {
                                if(numOfPulses >= startIdx)
                                {
                                    std::cout << "Return: " << lasreader->point.get_return_number()<< std::endl;
                                    std::cout << "[X,Y,Z]: [" << lasreader->point.get_X() << "," << lasreader->point.get_Y() << "," << lasreader->point.get_Z() << "]\n";
                                    std::cout << "*********************************\n";
                                }
                                ++numOfPoints;
                            }
                            else
                            {
                                std::cerr << "\nWarning: The file ended unexpectedly.\n";
                                std::cerr << "Expected " << numOfReturnsInPulse << " but only found " << i + 1 << " returns" << std::endl;
                            }
                        }
                        ++numOfPulses;

                        if(numOfPulses >= startIdx)
                        {
                            std::cout << "###################################\n";
                            ++numOfPrintedPulses;
                            if(numOfPrintedPulses >= numPulses)
                            {
                                break;
                            }
                        }
                    }

                    std::cout << "Number of Points counted: " << numOfPoints << std::endl;
                    std::cout << "Number of Pulses counted: " << numOfPulses << std::endl;
                }
                else
                {
                    throw spdlib::SPDException("Couldn't open input file.");
                }
            }
            catch(spdlib::SPDException &e)
            {
                throw e;
            }
            catch(std::exception &e)
            {
                throw spdlib::SPDException(e.what());
            }
        };

        void printPulsesFromLASFileNotFirstReturnsStartPulse(std::string inputFile) throw(SPDException)
        {
            // Open LAS file
            LASreadOpener lasreadopener;
            LASreader* lasreader = lasreadopener.open(inputFile.c_str());

            if(lasreader != 0)
            {
                // Get header
                LASheader *header = &lasreader->header;

                std::cout << "Number of Points in LAS file: " << header->number_of_point_records << std::endl;

                uint_fast16_t numOfReturnsInPulse = 0;
                uint_fast64_t numOfPoints = 0;
                uint_fast64_t numOfPulses = 0;
                uint_fast64_t numOfPrintedPoints = 0;

                while (lasreader->read_point())
				{
                    ++numOfPoints;
                    numOfReturnsInPulse = lasreader->point.get_number_of_returns();
                    if(lasreader->point.get_return_number() > 1)
                    {

                        std::cout << "###################################\n";
                        std::cout << "Pulse " << numOfPulses << std::endl;
                        std::cout << "Number of Returns: " << numOfReturnsInPulse << std::endl;
                        std::cout << "Return: " << lasreader->point.get_return_number()<< std::endl;
                        std::cout << "[X,Y,Z]: [" << lasreader->point.get_X() << "," << lasreader->point.get_Y() << "," << lasreader->point.get_Z() << "]\n";
                        std::cout << "###################################\n";
                        ++numOfPrintedPoints;
                    }

                    for(boost::uint_fast16_t i = 0; i < (numOfReturnsInPulse-1); ++i)
					{
						if(lasreader->read_point())
						{
							++numOfPoints;
						}
						else
						{
							std::cerr << "\nWarning: The file ended unexpectedly.\n";
                            std::cerr << "Expected " << numOfReturnsInPulse << " but only found " << i + 1 << " returns" << std::endl;
						}
					}
                    ++numOfPulses;
                }

                std::cout << "Number of Points counted: " << numOfPoints << std::endl;
                std::cout << "Number of Pulses counted: " << numOfPulses << std::endl;
            }
            else
            {
                throw spdlib::SPDException("Couldn't open input file.");
            }
        };

        void countNumPulses(std::string inputFile) throw(SPDException)
        {
            // Open LAS file
            LASreadOpener lasreadopener;
            LASreader* lasreader = lasreadopener.open(inputFile.c_str());

            if(lasreader != 0)
            {
                // Get header
                LASheader *header = &lasreader->header;

                std::cout << "Number of Points in LAS file: " << header->number_of_point_records << std::endl;

                uint_fast16_t numOfReturnsInPulse = 0;
                uint_fast64_t numOfPoints = 0;
                uint_fast64_t numOfPulses = 0;

                while (lasreader->read_point())
				{
                    ++numOfPoints;
                    numOfReturnsInPulse = lasreader->point.get_number_of_returns();
                    for(boost::uint_fast16_t i = 0; i < (numOfReturnsInPulse-1); ++i)
					{
						if(lasreader->read_point())
						{
							++numOfPoints;
						}
						else
						{
							std::cerr << "\nWarning: The file ended unexpectedly.\n";
                            std::cerr << "Expected " << numOfReturnsInPulse << " but only found " << i + 1 << " returns" << std::endl;
						}
					}
					++numOfPulses;
                }

                std::cout << "Number of Points counted: " << numOfPoints << std::endl;
                std::cout << "Number of Pulses counted: " << numOfPulses << std::endl;
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
    std::cout.precision(12);

    std::cout << "spdlastest " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	std::cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	std::cout << "and you are welcome to redistribute it under certain conditions; See\n";
	std::cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	std::cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << std::endl;
	
	try
	{
        TCLAP::CmdLine cmd("Print data pulses from a LAS file - for debugging: spdlastest", ' ', "1.0.0");
		
		TCLAP::ValueArg<uint_fast32_t> startPulseArg("s","start","Starting pulse index (Default 0)",false,0,"unsigned int");
		cmd.add( startPulseArg );

        TCLAP::ValueArg<uint_fast16_t> numPulseArg("n","number","Number of pulses to be printed out (Default 10)",false,10,"unsigned int");
		cmd.add( numPulseArg );
		
        TCLAP::SwitchArg printPulsesArg("p","print","Print a selction of pulses from LAS file.", false);		
		TCLAP::SwitchArg countPulsesArg("c","count","Count the number of pulses in LAS file.", false);
        TCLAP::SwitchArg notFirstStartPulseArg("f","notfirst","Print the returns which start a pulse with point IDs greater than 1", false);

        std::vector<TCLAP::Arg*> arguments;
        arguments.push_back(&printPulsesArg);
        arguments.push_back(&countPulsesArg);
        arguments.push_back(&notFirstStartPulseArg);

        cmd.xorAdd(arguments);

		TCLAP::ValueArg<std::string> inputFileArg("i","input","The input SPD file.",true,"","String");
		cmd.add( inputFileArg );

		cmd.parse( argc, argv );
				
		std::string inputFile = inputFileArg.getValue();
        if(printPulsesArg.isSet())
        {
            uint_fast32_t startIdx = startPulseArg.getValue();
            uint_fast16_t numPulses = numPulseArg.getValue();
            spdlib::SPDTestLasFile testLASFile;
            testLASFile.printPulsesFromLASFile(inputFile, startIdx, numPulses);
        }
        else if(countPulsesArg.isSet())
        {
            spdlib::SPDTestLasFile testLASFile;
            testLASFile.countNumPulses(inputFile);
        }
        else if(notFirstStartPulseArg.isSet())
        {
            spdlib::SPDTestLasFile testLASFile;
            testLASFile.printPulsesFromLASFileNotFirstReturnsStartPulse(inputFile);
        }
        else
        {
            std::cout << "No option (e.g., print or count) provided\n";
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
	std::cout << "spdlastest - end\n";
}

