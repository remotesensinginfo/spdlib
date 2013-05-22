/*
 *  main.cpp
 *  spdlib
 *
 *  Created by Pete Bunting on 18/04/2013.
 *  Copyright 2013 SPDLib. All rights reserved.
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
#include "spd/SPDException.h"

#include "spd/SPDGenerateTiles.h"

#include "spd/spd-config.h"

int main (int argc, char * const argv[])
{
    std::cout.precision(12);
    
    std::cout << "spdtiling " << SPDLIB_PACKAGE_STRING << ", Copyright (C) " << SPDLIB_COPYRIGHT_YEAR << " Sorted Pulse Library (SPD)\n";
	std::cout << "This program comes with ABSOLUTELY NO WARRANTY. This is free software,\n";
	std::cout << "and you are welcome to redistribute it under certain conditions; See\n";
	std::cout << "website (http://www.spdlib.org). Bugs are to be reported on the trac\n";
	std::cout << "or directly to " << SPDLIB_PACKAGE_BUGREPORT << std::endl;
	
	try
	{
        TCLAP::CmdLine cmd("Tools for tiling a set of SPD files using predefined tile areas: spdtiling", ' ', "1.0.0");
        
        TCLAP::SwitchArg createAllSwitch("","all","Create all tiles.", false);
		TCLAP::SwitchArg extractIndividualSwitch("","extract","Extract an individual tile as specified in the XML file.", false);
        TCLAP::SwitchArg extractCoreSwitch("","extractcore","Extract the core of a tile as specified in the XML file.", false);
        
        std::vector<TCLAP::Arg*> arguments;
        arguments.push_back(&createAllSwitch);
        arguments.push_back(&extractIndividualSwitch);
        arguments.push_back(&extractCoreSwitch);
        cmd.xorAdd(arguments);
        
        TCLAP::ValueArg<std::string> tilesFileArg("t","tiles","XML file defining the tile regions",true,"","String");
		cmd.add( tilesFileArg );
        
        TCLAP::ValueArg<std::string> outputBaseArg("o","output","The base path for the tiles. (--extractcore expects a single output SPD file)",true,"","String");
		cmd.add( outputBaseArg );
        
        TCLAP::ValueArg<std::string> inputFilesArg("i","input","A text file with a list of input files, one per line. (--extractcore expects a single input SPD file)",true,"","String");
		cmd.add( inputFilesArg );
        
        TCLAP::ValueArg<unsigned int> extractRowArg("r","row","The row of the tile to be extracted (--extract).",false,0,"Unsigned int");
		cmd.add( extractRowArg );
        
        TCLAP::ValueArg<unsigned int> extractColArg("c","col","The column of the tile to be extracted (--extract).",false,0,"Unsigned int");
		cmd.add( extractColArg );
        
        TCLAP::SwitchArg rmEmptyTilesSwitch("d","deltiles","Remove tiles which have no data.", false);
        cmd.add( rmEmptyTilesSwitch );
        
        TCLAP::SwitchArg updateXMLSwitch("u","updatexml","Update the tiles XML file.", false);
        cmd.add( updateXMLSwitch );
        
		cmd.parse( argc, argv );
        
        if(tilesFileArg.getValue() == "")
        {
            throw TCLAP::ArgException("Tiles file path should not be blank.");
        }
        
        if(inputFilesArg.getValue() == "")
        {
            throw TCLAP::ArgException("Input file path should not be blank.");
        }
        
        if(outputBaseArg.getValue() == "")
        {
            throw TCLAP::ArgException("Output files path should not be blank.");
        }
        
        std::cout.precision(12);
        
        
        if(createAllSwitch.getValue())
        {
            std::cout << "Tiles XML file: " << tilesFileArg.getValue() << std::endl;
            std::cout << "Input SPD file list: " << inputFilesArg.getValue() << std::endl;
            std::cout << "Output Tiles base: " << outputBaseArg.getValue() << std::endl;
            
            double xSize = 0;
            double ySize = 0;
            double overlap = 0;
            
            double xMin = 0;
            double xMax = 0;
            double yMin = 0;
            double yMax = 0;
            
            boost::uint_fast32_t rows = 0;
            boost::uint_fast32_t cols = 0;
            
            std::cout << "Reading tile XML\n";
            spdlib::SPDTilesUtils tileUtils;
            std::vector<spdlib::SPDTile*> *tiles = tileUtils.importTilesFromXML(tilesFileArg.getValue(), &rows, &cols, &xSize, &ySize, &overlap, &xMin, &xMax, &yMin, &yMax);
            
            std::cout << "Number of rows: " << rows << std::endl;
            std::cout << "Number of cols: " << cols << std::endl;
            
            std::cout << "Tile Size: [" << xSize << "," << ySize << "] Overlap: " << overlap << std::endl;
            
            std::cout << "Full Area: [" << xMin << "," << xMax << "][" << yMin << "," << yMax << "]\n";
            
            //tileUtils.printTiles2Console(tiles);
            
            
            
            
            spdlib::SPDTextFileUtilities txtUtils;        
            std::vector<std::string> inputFiles = txtUtils.readFileLinesToVector(inputFilesArg.getValue());
            /*
            for(std::vector<std::string>::iterator iterFiles = inputFiles.begin(); iterFiles != inputFiles.end(); ++iterFiles)
            {
                std::cout << "\'" << *iterFiles << "\'" << std::endl;
            }
            */
            spdlib::SPDFile *inSPDFile = new spdlib::SPDFile(inputFiles.front());
            
            spdlib::SPDFileReader reader;
            reader.readHeaderInfo(inputFiles.front(), inSPDFile);
            
            std::cout << "Opening and creating output files.\n";
            tileUtils.createTileSPDFiles(tiles, inSPDFile, outputBaseArg.getValue(), xSize, ySize, overlap, xMin, xMax, yMin, yMax, rows, cols);
            
            std::cout << "Populate the tiles with data\n.";
            tileUtils.populateTilesWithData(tiles, inputFiles);
            
            if(rmEmptyTilesSwitch.getValue())
            {
                std::cout << "Remove empty tiles\n";
                tileUtils.deleteTilesWithNoPulses(tiles);
            }
            
            if(updateXMLSwitch.getValue())
            {
                std::cout << "Export updated tiles XML.";
                tileUtils.exportTiles2XML(tilesFileArg.getValue(), tiles, xSize, ySize, overlap, xMin, xMax, yMin, yMax, rows, cols);
            }
            
            tileUtils.deleteTiles(tiles);
        }
        else if(extractIndividualSwitch.getValue())
        {
            boost::uint_fast32_t row = extractRowArg.getValue();
            boost::uint_fast32_t col = extractColArg.getValue();
            
            std::cout << "Tiles XML file: " << tilesFileArg.getValue() << std::endl;
            std::cout << "Input SPD file list: " << inputFilesArg.getValue() << std::endl;
            std::cout << "Output Tiles base: " << outputBaseArg.getValue() << std::endl;
            
            std::cout << "Tile to be extracted: [" << row << ", " << col << "]" << std::endl;
            
            double xSize = 0;
            double ySize = 0;
            double overlap = 0;
            
            double xMin = 0;
            double xMax = 0;
            double yMin = 0;
            double yMax = 0;
            
            boost::uint_fast32_t numRows = 0;
            boost::uint_fast32_t numCols = 0;
            
            std::cout << "Reading tile XML\n";
            spdlib::SPDTilesUtils tileUtils;
            std::vector<spdlib::SPDTile*> *tiles = tileUtils.importTilesFromXML(tilesFileArg.getValue(), &numRows, &numCols, &xSize, &ySize, &overlap, &xMin, &xMax, &yMin, &yMax);
            
            std::cout << "Number of rows: " << numRows << std::endl;
            std::cout << "Number of cols: " << numCols << std::endl;
            
            std::cout << "Tile Size: [" << xSize << "," << ySize << "] Overlap: " << overlap << std::endl;
            
            std::cout << "Full Area: [" << xMin << "," << xMax << "][" << yMin << "," << yMax << "]\n";
            
            if((row > 0) && (row <= numRows) & (col > 0) && (col <= numCols))
            {
                spdlib::SPDTextFileUtilities txtUtils;
                std::vector<std::string> inputFiles = txtUtils.readFileLinesToVector(inputFilesArg.getValue());
                
                spdlib::SPDFile *inSPDFile = new spdlib::SPDFile(inputFiles.front());
                
                spdlib::SPDFileReader reader;
                reader.readHeaderInfo(inputFiles.front(), inSPDFile);
                                
                spdlib::SPDTile *tileToProcess = NULL;
                
                for(std::vector<spdlib::SPDTile*>::iterator iterTiles = tiles->begin(); iterTiles != tiles->end(); ++iterTiles)
                {
                    if(((*iterTiles)->row == row) & ((*iterTiles)->col == col))
                    {
                        if((*iterTiles)->outFileName == "")
                        {
                            (*iterTiles)->outFileName = outputBaseArg.getValue() + std::string("_row") + txtUtils.uInt32bittostring((*iterTiles)->row) + std::string("col") + txtUtils.uInt32bittostring((*iterTiles)->col) + std::string(".spd");
                            std::cout << "Creating File: " << (*iterTiles)->outFileName << std::endl;
                            (*iterTiles)->spdFile = new spdlib::SPDFile((*iterTiles)->outFileName);
                            (*iterTiles)->spdFile->copyAttributesFromTemplate(inSPDFile);
                            (*iterTiles)->spdFile->setBoundingBox((*iterTiles)->xMin, (*iterTiles)->xMax, (*iterTiles)->yMin, (*iterTiles)->yMax);
                            (*iterTiles)->writer = new spdlib::SPDNoIdxFileWriter();
                            (*iterTiles)->writer->open((*iterTiles)->spdFile, (*iterTiles)->outFileName);
                            (*iterTiles)->writer->finaliseClose();
                            delete (*iterTiles)->writer;
                            (*iterTiles)->writerOpen = false;
                        }
                        else
                        {
                            std::cout << "Opening File: " << (*iterTiles)->outFileName << std::endl;
                            (*iterTiles)->spdFile = new spdlib::SPDFile((*iterTiles)->outFileName);
                            (*iterTiles)->writer = new spdlib::SPDNoIdxFileWriter();
                            (*iterTiles)->writer->reopen((*iterTiles)->spdFile, (*iterTiles)->outFileName);
                            (*iterTiles)->writer->finaliseClose();
                            delete (*iterTiles)->writer;
                            (*iterTiles)->writerOpen = false;
                        }
                        
                        tileToProcess = (*iterTiles);
                        break;
                    }
                }
                
                
                try
                {
                    tileUtils.populateTileWithData(tileToProcess, inputFiles);
                }
                catch(spdlib::SPDException &e)
                {
                    tileUtils.deleteTileIfNoPulses(tiles, row, col);
                    throw e;
                }
                                
                if(rmEmptyTilesSwitch.getValue())
                {
                    std::cout << "Remove if tile empty.\n";
                    tileUtils.deleteTileIfNoPulses(tiles, row, col);
                }
                
                if(updateXMLSwitch.getValue())
                {
                    std::cout << "Export updated tiles XML.";
                    tileUtils.exportTiles2XML(tilesFileArg.getValue(), tiles, xSize, ySize, overlap, xMin, xMax, yMin, yMax, numRows, numCols);
                }
                
                delete inSPDFile;
                tileUtils.deleteTiles(tiles);
            }
            else
            {
                throw TCLAP::ArgException("The tile specified is not with the list of tiles.");
            }            
        }
        else if(extractCoreSwitch.getValue())
        {
            std::cout << "Tiles XML file: " << tilesFileArg.getValue() << std::endl;
            std::cout << "Input SPD file: " << inputFilesArg.getValue() << std::endl;
            std::cout << "Output SPD file: " << outputBaseArg.getValue() << std::endl;
            
            try
            {
                double xSize = 0;
                double ySize = 0;
                double overlap = 0;
                
                double xMin = 0;
                double xMax = 0;
                double yMin = 0;
                double yMax = 0;
                
                boost::uint_fast32_t numRows = 0;
                boost::uint_fast32_t numCols = 0;
                
                std::cout << "Reading tile XML\n";
                spdlib::SPDTilesUtils tileUtils;
                std::vector<spdlib::SPDTile*> *tiles = tileUtils.importTilesFromXML(tilesFileArg.getValue(), &numRows, &numCols, &xSize, &ySize, &overlap, &xMin, &xMax, &yMin, &yMax);
                
                boost::uint_fast32_t row;
                boost::uint_fast32_t col;
                
                tileUtils.extractRowColFromFileName(inputFilesArg.getValue(), &row, &col);
                
                std::cout << "Extracting Tile: [" << row << ", " << col << "]\n";
                
                tileUtils.extractTileCore(inputFilesArg.getValue(), outputBaseArg.getValue(), row, col, tiles);
                
                tileUtils.deleteTiles(tiles);
            }
            catch (spdlib::SPDProcessingException &e)
            {
                throw spdlib::SPDException(e.what());
            }
        }
        else
        {
            throw TCLAP::ArgException("Only the --all, --extract and --extractcore options are known and one must be specified.");
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
    std::cout << "spdtiling - end\n";
}

