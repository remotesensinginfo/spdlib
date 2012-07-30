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
#include "spd/SPDExportProcessorSubset.h"
#include "spd/SPDSubsetSPDFile.h"
#include "spd/SPDException.h"
#include "spd/SPDTextFileLineReader.h"
#include "spd/SPDFileReader.h"

#include "spd/spd-config.h"

using namespace std;
using namespace spdlib;
using namespace TCLAP;

int main (int argc, char * const argv[]) 
{
	cout << "spdsubset " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	cout << "and you are welcome to redistribute it under certain conditions; See\n";
	cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << endl;
	
	try 
	{
		CmdLine cmd("Subset point cloud data: spdsubset", ' ', "1.0.0");

		ValueArg<double> xMinArg("","xmin","Minimum X threshold",false,NAN,"double");
		cmd.add( xMinArg );
		
		ValueArg<double> xMaxArg("","xmax","Maximum X threshold",false,NAN,"double");
		cmd.add( xMaxArg );
		
		ValueArg<double> yMinArg("","ymin","Minimum Y threshold",false,NAN,"double");
		cmd.add( yMinArg );
		
		ValueArg<double> yMaxArg("","ymax","Maximum Y threshold",false,NAN,"double");
		cmd.add( yMaxArg );
		
		ValueArg<double> zMinArg("","zmin","Minimum Z threshold",false,NAN,"double");
		cmd.add( zMinArg );
		
		ValueArg<double> zMaxArg("","zmax","Maximum Z threshold",false,NAN,"double");
		cmd.add( zMaxArg );
        
        ValueArg<double> hMinArg("","hmin","Minimum Height threshold",false,NAN,"double");
		cmd.add( hMinArg );
		
		ValueArg<double> hMaxArg("","hmax","Maximum Height threshold",false,NAN,"double");
		cmd.add( hMaxArg );
        
        ValueArg<double> azMinArg("","azmin","Minimum azimuth threshold",false,NAN,"double");
		cmd.add( azMinArg );
		
		ValueArg<double> azMaxArg("","azmax","Maximum azmuth threshold",false,NAN,"double");
		cmd.add( azMaxArg );
		
		ValueArg<double> zenMinArg("","zenmin","Minimum zenith threshold",false,NAN,"double");
		cmd.add( zenMinArg );
		
		ValueArg<double> zenMaxArg("","zenmax","Maximum zenith threshold",false,NAN,"double");
		cmd.add( zenMaxArg );
		
		ValueArg<double> ranMinArg("","ranmin","Minimum range threshold",false,NAN,"double");
		cmd.add( ranMinArg );
		
		ValueArg<double> ranMaxArg("","ranmax","Maximum range threshold",false,NAN,"double");
		cmd.add( ranMaxArg );
        
        SwitchArg heightSwitch("","height","Threshold the height of each pulse (currently only valid with SPD to SPD subsetting)", false);
		cmd.add( heightSwitch );
        
        SwitchArg sphericalSwitch("","spherical","Subset a spherically indexed SPD file.", false);
        cmd.add( sphericalSwitch );

        ValueArg<string> textfileArg("","txtfile","A text containing the extent to which the file should be cut to.",false,"","string");
		cmd.add( textfileArg );
        
        SwitchArg ignoreRangeSwitch("","ignorerange","Defining that range should be ignored when subsetting using a text file.", false);
		cmd.add( ignoreRangeSwitch );
        
        SwitchArg ignoreZSwitch("","ignorez","Defining that Z should be ignored when subsetting using a text file.", false);
		cmd.add( ignoreZSwitch );
        
        ValueArg<string> shapefileArg("","shpfile","A shapefile to which the dataset should be subsetted to",false,"","string");
		cmd.add( shapefileArg );
        
        ValueArg<uint_fast32_t> startArg("","start","First pulse in the block",false,0,"uint_fast32_t");
		cmd.add( startArg );
		
		ValueArg<uint_fast32_t> numArg("","num","Number of pulses to be exported",false,0,"uint_fast32_t");
		cmd.add( numArg );
		
		UnlabeledMultiArg<string> multiFileNames("Files", "File names for the input and output files", true, "string");
		cmd.add( multiFileNames );
		cmd.parse( argc, argv );
		
		vector<string> fileNames = multiFileNames.getValue();		
		if(fileNames.size() != 2)
		{
			SPDTextFileUtilities textUtils;
			string message = string("Two file paths should have been specified (e.g., Input and Output). ") + textUtils.uInt32bittostring(fileNames.size()) + string(" were provided.");
			throw SPDException(message);
		}
		
        
        string inputFile = fileNames.at(0);
        string outputFile = fileNames.at(1);
        
        SPDFile *inSPDFile = new SPDFile(inputFile);
        SPDFileReader spdReader = SPDFileReader();
        spdReader.readHeaderInfo(inputFile, inSPDFile);
        
        bool indexedSPDFile = true;
        if(inSPDFile->getFileType() == SPD_UPD_TYPE)
        {
            indexedSPDFile = false;
        }
        
        if(startArg.isSet() & !indexedSPDFile)
        {
            SPDUPDPulseSubset updSubset;
            updSubset.subsetUPD(inputFile, outputFile, startArg.getValue(), numArg.getValue());
        }
        else
        {
            bool ignoreZ = ignoreZSwitch.getValue();
            bool ignoreRange = ignoreRangeSwitch.getValue();
            double *bbox = new double[6];
            bool *bboxDefined = new bool[6];
            bool sphericalCoords = sphericalSwitch.getValue();
            if(textfileArg.isSet())
            {
                SPDTextFileLineReader lineReader;
                SPDTextFileUtilities textFileUtils;
                lineReader.openFile(textfileArg.getValue());
                bool first = true;
                uint_fast16_t dataLineCount = 0;
                string line = "";
                while(!lineReader.endOfFile())
                {
                    line = lineReader.readLine();
                    //cout << "Line = " << line << endl;
                    if(!textFileUtils.blankline(line))
                    {
                        if(first)
                        {
                            if(line == "#Cartesian")
                            {
                                sphericalCoords = false;
                            }
                            else if(line == "#Spherical")
                            {
                                sphericalCoords = true;
                            }
                            first = false;
                        }
                        else
                        {
                            if(dataLineCount > 5)
                            {
                                throw SPDException("The number of data lines should be either 4 or 6. Gone over.");
                            }
                            bbox[dataLineCount++] = textFileUtils.strtodouble(line);
                            //cout << "dataLineCount = " << dataLineCount << endl;
                        }
                    }
                }
                lineReader.closeFile();
                if(dataLineCount == 4)
                {
                    if(sphericalCoords)
                    {
                        ignoreRange = true;
                    }
                    else
                    {
                        ignoreZ = true;
                    }
                    bbox[4] = NAN;
                    bbox[5] = NAN;
                }
                else if(dataLineCount != 6)
                {
                    cout << "dataLineCount = " << dataLineCount << endl;
                    throw SPDException("The number of data lines should be either 4 or 6.");
                }
            }
            else
            {
                if(sphericalCoords)
                {
                    bbox[0] = azMinArg.getValue();
                    bbox[1] = azMaxArg.getValue();
                    bbox[2] = zenMinArg.getValue();
                    bbox[3] = zenMaxArg.getValue();
                    bbox[4] = ranMinArg.getValue();
                    bbox[5] = ranMaxArg.getValue();
                }
                else
                {
                    bbox[0] = xMinArg.getValue();
                    bbox[1] = xMaxArg.getValue();
                    bbox[2] = yMinArg.getValue();
                    bbox[3] = yMaxArg.getValue();
                    bbox[4] = zMinArg.getValue();
                    bbox[5] = zMaxArg.getValue();
                }
            }
            
            if(ignoreRange | ignoreZ)
            {
                bbox[4] = NAN;
                bbox[5] = NAN;
            }
                
            if(boost::math::isnan(bbox[0]))
            {
                bboxDefined[0] = false;
            }
            else
            {
                bboxDefined[0] = true;
            }
            if(boost::math::isnan(bbox[1]))
            {
                bboxDefined[1] = false;
            }
            else
            {
                bboxDefined[1] = true;
            }
            if(boost::math::isnan(bbox[2]))
            {
                bboxDefined[2] = false;
            }
            else
            {
                bboxDefined[2] = true;
            }
            if(boost::math::isnan(bbox[3]))
            {
                bboxDefined[3] = false;
            }
            else
            {
                bboxDefined[3] = true;
            }
            if(boost::math::isnan(bbox[4]))
            {
                bboxDefined[4] = false;
            }
            else
            {
                bboxDefined[4] = true;
            }
            if(boost::math::isnan(bbox[5]))
            {
                bboxDefined[5] = false;
            }
            else
            {
                bboxDefined[5] = true;
            }
            
            if(ignoreRange | ignoreZ)
            {
                bboxDefined[4] = false;
                bboxDefined[5] = false;
            }
            
            if(sphericalCoords)
            {
                cout << "Azimuth Min:" << bbox[0] << endl;
                cout << "Azimuth Max:" << bbox[1] << endl;
                cout << "Zenith Min:" << bbox[2] << endl;
                cout << "Zenith Max:" << bbox[3] << endl;
                cout << "Range Min:" << bbox[4] << endl;
                cout << "Range Max:" << bbox[5] << endl;
            }
            else
            {
                cout << "X Min:" << bbox[0] << endl;
                cout << "X Max:" << bbox[1] << endl;
                cout << "Y Min:" << bbox[2] << endl;
                cout << "Y Max:" << bbox[3] << endl;
                cout << "Z Min:" << bbox[4] << endl;
                cout << "Z Max:" << bbox[5] << endl;
            }
            
            if(sphericalCoords)
            {
                if(indexedSPDFile)
                {
                    SPDSubsetSPDFile subset;
                    subset.subsetSphericalSPDFile(inputFile, outputFile, bbox, bboxDefined);
                }
                else
                {
                    SPDSubsetNonGriddedFile subset;
                    subset.subsetSpherical(inputFile, outputFile, bbox, bboxDefined);
                }
            }
            else
            {
                if(indexedSPDFile & (shapefileArg.isSet()))
                {
                    SPDSubsetSPDFile subset;
                    subset.subsetSPDFile(inputFile, outputFile, shapefileArg.getValue());
                }
                else if(indexedSPDFile & (heightSwitch.getValue()))
                {
                    SPDSubsetSPDFile subset;
                    subset.subsetSPDFileHeightOnly(inputFile, outputFile, hMinArg.getValue(), hMaxArg.getValue());
                }
                else if(indexedSPDFile)
                {
                    SPDSubsetSPDFile subset;
                    subset.subsetSPDFile(inputFile, outputFile, bbox, bboxDefined);
                }
                else
                {
                    SPDSubsetNonGriddedFile subset;
                    subset.subsetCartesian(inputFile, outputFile, bbox, bboxDefined);
                }
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
	
}

