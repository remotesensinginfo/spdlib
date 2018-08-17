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

int main (int argc, char * const argv[])
{
    std::cout.precision(12);

    std::cout << "spdsubset " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	std::cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	std::cout << "and you are welcome to redistribute it under certain conditions; See\n";
	std::cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	std::cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << std::endl;
	
	try
	{
        TCLAP::CmdLine cmd("Subset point cloud data: spdsubset", ' ', "1.0.0");

		TCLAP::ValueArg<double> xMinArg("","xmin","Minimum X threshold",false,NAN,"double");
		cmd.add( xMinArg );
		
		TCLAP::ValueArg<double> xMaxArg("","xmax","Maximum X threshold",false,NAN,"double");
		cmd.add( xMaxArg );
		
		TCLAP::ValueArg<double> yMinArg("","ymin","Minimum Y threshold",false,NAN,"double");
		cmd.add( yMinArg );
		
		TCLAP::ValueArg<double> yMaxArg("","ymax","Maximum Y threshold",false,NAN,"double");
		cmd.add( yMaxArg );
		
		TCLAP::ValueArg<double> zMinArg("","zmin","Minimum Z threshold",false,NAN,"double");
		cmd.add( zMinArg );
		
		TCLAP::ValueArg<double> zMaxArg("","zmax","Maximum Z threshold",false,NAN,"double");
		cmd.add( zMaxArg );

        TCLAP::ValueArg<double> hMinArg("","hmin","Minimum Height threshold",false,NAN,"double");
		cmd.add( hMinArg );
		
		TCLAP::ValueArg<double> hMaxArg("","hmax","Maximum Height threshold",false,NAN,"double");
		cmd.add( hMaxArg );

        TCLAP::ValueArg<double> azMinArg("","azmin","Minimum azimuth threshold",false,NAN,"double");
		cmd.add( azMinArg );
		
		TCLAP::ValueArg<double> azMaxArg("","azmax","Maximum azmuth threshold",false,NAN,"double");
		cmd.add( azMaxArg );
		
		TCLAP::ValueArg<double> zenMinArg("","zenmin","Minimum zenith threshold",false,NAN,"double");
		cmd.add( zenMinArg );
		
		TCLAP::ValueArg<double> zenMaxArg("","zenmax","Maximum zenith threshold",false,NAN,"double");
		cmd.add( zenMaxArg );
		
		TCLAP::ValueArg<double> ranMinArg("","ranmin","Minimum range threshold",false,NAN,"double");
		cmd.add( ranMinArg );
		
		TCLAP::ValueArg<double> ranMaxArg("","ranmax","Maximum range threshold",false,NAN,"double");
		cmd.add( ranMaxArg );

        TCLAP::ValueArg<double> sliMinArg("","slimin","Minimum scanline index threshold",false,NAN,"uint_fast32_t");
		cmd.add( sliMinArg );
		
		TCLAP::ValueArg<double> sliMaxArg("","slimax","Maximum scanline index threshold",false,NAN,"uint_fast32_t");
		cmd.add( sliMaxArg );
		
		TCLAP::ValueArg<double> slMinArg("","slmin","Minimum scanline threshold",false,NAN,"uint_fast32_t");
		cmd.add( slMinArg );
		
		TCLAP::ValueArg<double> slMaxArg("","slmax","Maximum scanline threshold",false,NAN,"uint_fast32_t");
		cmd.add( slMaxArg );

        TCLAP::SwitchArg heightSwitch("","height","Threshold the height of each pulse (currently only valid with SPD to SPD subsetting)", false);
		cmd.add( heightSwitch );

        TCLAP::SwitchArg sphericalSwitch("","spherical","Subset a spherically indexed SPD file.", false);
        cmd.add( sphericalSwitch );

        TCLAP::SwitchArg scanSwitch("","scan","Subset a scan indexed SPD file.", false);
        cmd.add( scanSwitch );

        TCLAP::ValueArg<std::string> textfileArg("","txtfile","A text containing the extent to which the file should be cut to.",false,"","string");
		cmd.add( textfileArg );

        TCLAP::SwitchArg ignoreRangeSwitch("","ignorerange","Defining that range should be ignored when subsetting using a text file.", false);
		cmd.add( ignoreRangeSwitch );

        TCLAP::SwitchArg ignoreZSwitch("","ignorez","Defining that Z should be ignored when subsetting using a text file.", false);
		cmd.add( ignoreZSwitch );

        TCLAP::ValueArg<std::string> shapefileArg("","shpfile","A shapefile to which the dataset should be subsetted to.",false,"","string");
		cmd.add( shapefileArg );

        TCLAP::ValueArg<std::string> imgfileArg("","imgfile","A binary image to which the dataset should be subsetted to (pixel values of 1 define ROI).",false,"","string");
		cmd.add( imgfileArg );

        TCLAP::ValueArg<uint_fast32_t> startArg("","start","First pulse in the block",false,0,"uint_fast32_t");
		cmd.add( startArg );
		
		TCLAP::ValueArg<uint_fast32_t> numArg("","num","Number of pulses to be exported",false,0,"uint_fast32_t");
		cmd.add( numArg );

        TCLAP::ValueArg<boost::uint_fast32_t> numOfRowsBlockArg("r","blockrows","Number of rows within a block (Default 100)",false,100,"unsigned int");
		cmd.add( numOfRowsBlockArg );

        TCLAP::ValueArg<boost::uint_fast32_t> numOfColsBlockArg("c","blockcols","Number of columns within a block (Default 0) - Note values greater than 1 result in a non-sequencial SPD file.",false,0,"unsigned int");
		cmd.add( numOfColsBlockArg );
		
		TCLAP::ValueArg<std::string> inputFileArg("i","input","The input SPD file.",true,"","String");
		cmd.add( inputFileArg );

        TCLAP::ValueArg<std::string> outputFileArg("o","output","The output SPD file.",true,"","String");
		cmd.add( outputFileArg );

		cmd.parse( argc, argv );
		
        std::string inputFile = inputFileArg.getValue();
        std::string outputFile = outputFileArg.getValue();

        spdlib::SPDFile *inSPDFile = new spdlib::SPDFile(inputFile);
        spdlib::SPDFileReader spdReader = spdlib::SPDFileReader();
        spdReader.readHeaderInfo(inputFile, inSPDFile);

        bool indexedSPDFile = true;
        if(inSPDFile->getFileType() == spdlib::SPD_UPD_TYPE)
        {
            indexedSPDFile = false;
        }

        if(startArg.isSet() & !indexedSPDFile)
        {
            spdlib::SPDUPDPulseSubset updSubset;
            updSubset.subsetUPD(inputFile, outputFile, startArg.getValue(), numArg.getValue());
        }
        else
        {
            bool ignoreZ = ignoreZSwitch.getValue();
            bool ignoreRange = ignoreRangeSwitch.getValue();
            double *bbox = new double[6];
            bool *bboxDefined = new bool[6];
            bool sphericalCoords = sphericalSwitch.getValue();
            bool scanCoords = scanSwitch.getValue();
            if(textfileArg.isSet())
            {
                spdlib::SPDTextFileLineReader lineReader;
                spdlib::SPDTextFileUtilities textFileUtils;
                lineReader.openFile(textfileArg.getValue());
                bool first = true;
                uint_fast16_t dataLineCount = 0;
                std::string line = "";
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
                            else if(line == "#Scan")
                            {
                                scanCoords = true;
                            }
                            first = false;
                        }
                        else
                        {
                            if(dataLineCount > 5)
                            {
                                throw spdlib::SPDException("The number of data lines should be either 4 or 6. Gone over.");
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
                    else if(scanCoords)
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
                    std::cout << "dataLineCount = " << dataLineCount << std::endl;
                    throw spdlib::SPDException("The number of data lines should be either 4 or 6.");
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
                else if(scanCoords)
                {
                    bbox[0] = sliMinArg.getValue();
                    bbox[1] = sliMaxArg.getValue();
                    bbox[2] = slMinArg.getValue();
                    bbox[3] = slMaxArg.getValue();
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
                std::cout << "Azimuth Min:" << bbox[0] << std::endl;
                std::cout << "Azimuth Max:" << bbox[1] << std::endl;
                std::cout << "Zenith Min:" << bbox[2] << std::endl;
                std::cout << "Zenith Max:" << bbox[3] << std::endl;
                std::cout << "Range Min:" << bbox[4] << std::endl;
                std::cout << "Range Max:" << bbox[5] << std::endl;
            }
            else if(scanCoords)
            {
                std::cout << "ScanlineIdx Min:" << bbox[0] << std::endl;
                std::cout << "ScanlineIdx Max:" << bbox[1] << std::endl;
                std::cout << "Scanline Min:" << bbox[2] << std::endl;
                std::cout << "Scanline Max:" << bbox[3] << std::endl;
                std::cout << "Range Min:" << bbox[4] << std::endl;
                std::cout << "Range Max:" << bbox[5] << std::endl;
            }
            else
            {
                std::cout << "X Min:" << bbox[0] << std::endl;
                std::cout << "X Max:" << bbox[1] << std::endl;
                std::cout << "Y Min:" << bbox[2] << std::endl;
                std::cout << "Y Max:" << bbox[3] << std::endl;
                std::cout << "Z Min:" << bbox[4] << std::endl;
                std::cout << "Z Max:" << bbox[5] << std::endl;
            }

            if(sphericalCoords)
            {
                if(indexedSPDFile)
                {
                    spdlib::SPDSubsetSPDFile subset;
                    subset.subsetSphericalSPDFile(inputFile, outputFile, bbox, bboxDefined);
                }
                else
                {
                    spdlib::SPDSubsetNonGriddedFile subset;
                    subset.subsetSpherical(inputFile, outputFile, bbox, bboxDefined);
                }
            }
            else if(scanCoords)
            {
                if(indexedSPDFile)
                {
                    spdlib::SPDSubsetSPDFile subset;
                    subset.subsetScanSPDFile(inputFile, outputFile, bbox, bboxDefined);
                }
                else
                {
                    spdlib::SPDSubsetNonGriddedFile subset;
                    subset.subsetScan(inputFile, outputFile, bbox, bboxDefined);
                }
            }
            else
            {
                if(indexedSPDFile & (shapefileArg.isSet()))
                {
                    spdlib::SPDSubsetSPDFile subset;
                    subset.subsetSPDFile2Shp(inputFile, outputFile, shapefileArg.getValue());
                }
                if(indexedSPDFile & (imgfileArg.isSet()))
                {
                    spdlib::SPDSubsetSPDFile subset;
                    subset.subsetSPDFile2Img(inputFile, outputFile, imgfileArg.getValue(), numOfColsBlockArg.getValue(), numOfRowsBlockArg.getValue());
                }
                else if(indexedSPDFile & (heightSwitch.getValue()))
                {
                    spdlib::SPDSubsetSPDFile subset;
                    subset.subsetSPDFileHeightOnly(inputFile, outputFile, hMinArg.getValue(), hMaxArg.getValue());
                }
                else if(indexedSPDFile)
                {
                    spdlib::SPDSubsetSPDFile subset;
                    subset.subsetSPDFile(inputFile, outputFile, bbox, bboxDefined);
                }
                else
                {
                    spdlib::SPDSubsetNonGriddedFile subset;
                    subset.subsetCartesian(inputFile, outputFile, bbox, bboxDefined);
                }
            }
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
	std::cout << "spdsubset - end\n";
}

