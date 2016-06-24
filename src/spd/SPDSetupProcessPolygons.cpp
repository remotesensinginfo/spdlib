/*
 *  SPDSetupProcessPolygons.cpp
 *  SPDLIB
 *
 *  Created by Pete Bunting on 06/04/20121.
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
 */

#include "spd/SPDSetupProcessPolygons.h"

namespace spdlib
{

	std::string SPDSetupProcessPolygonsAbstract::getLayerName(std::string filepath)
	{
		int strSize = filepath.size();
		int lastSlash = 0;
		for(int i = 0; i < strSize; i++)
		{
			if(filepath.at(i) == '/')
			{
				lastSlash = i;
			}
		}
		std::string filename = filepath.substr(lastSlash+1);
		
		strSize = filename.size();
		int lastpt = 0;
		for(int i = 0; i < strSize; i++)
		{
			if(filename.at(i) == '.')
			{
				lastpt = i;
			}
		}
		
		std::string layerName = filename.substr(0, lastpt);
		return layerName;		
	}
	
	

	SPDSetupProcessShapefilePolygons::SPDSetupProcessShapefilePolygons() : SPDSetupProcessPolygonsAbstract()
	{
		
	}
		
	void SPDSetupProcessShapefilePolygons::processPolygons(std::string spdInputFile, std::string inputLayer, std::string outputLayer, bool deleteOutShpIfExists, bool copyAttributes, SPDPolygonProcessor *processor) throw(SPDProcessingException)
	{
		OGRRegisterAll();
		
		std::string SHPFileInLayer = this->getLayerName(inputLayer);
		std::string SHPFileOutLayer = this->getLayerName(outputLayer);
		
		GDALDataset *inputSHPDS = NULL;
		OGRLayer *inputSHPLayer = NULL;
		GDALDriver *shpFiledriver = NULL;
		GDALDataset *outputSHPDS = NULL;
		OGRLayer *outputSHPLayer = NULL;
		
		try
		{
            
            std::string outputDIR = SPDFileUtilities::getFileDirectoryPath(outputLayer);
            
            if(SPDFileUtilities::checkDIR4SHP(outputDIR, SHPFileOutLayer))
            {
                if(deleteOutShpIfExists)
                {
                    SPDFileUtilities::deleteSHP(outputDIR, SHPFileOutLayer);
                }
                else
                {
                    throw SPDProcessingException("Shapefile already exists, either delete or select force.");
                }
            }
            
			/////////////////////////////////////
			//
			// Open Input Shapfile.
			//
			/////////////////////////////////////
			inputSHPDS = (GDALDataset*) GDALOpenEx(inputLayer.c_str(), GDAL_OF_VECTOR, NULL, NULL, NULL);
			if(inputSHPDS == NULL)
			{
				std::string message = std::string("Could not open vector file ") + inputLayer;
				throw SPDProcessingException(message.c_str());
			}
			inputSHPLayer = inputSHPDS->GetLayerByName(SHPFileInLayer.c_str());
			if(inputSHPLayer == NULL)
			{
				std::string message = std::string("Could not open vector layer ") + SHPFileInLayer;
				throw SPDProcessingException(message.c_str());
			}
			
			/////////////////////////////////////
			//
			// Create Output Shapfile.
			//
			/////////////////////////////////////
			const char *pszDriverName = "ESRI Shapefile";
			shpFiledriver = GetGDALDriverManager()->GetDriverByName(pszDriverName );
			if( shpFiledriver == NULL )
			{
				throw SPDProcessingException("SHP driver not available.");
			}
			outputSHPDS = shpFiledriver->Create(outputLayer.c_str(), 0, 0, 0, GDT_Unknown, NULL);
			if( outputSHPDS == NULL )
			{
				std::string message = std::string("Could not create vector file ") + outputLayer;
				throw SPDProcessingException(message.c_str());
			}
			outputSHPLayer = outputSHPDS->CreateLayer(SHPFileOutLayer.c_str(), NULL, wkbPolygon, NULL );
			if( outputSHPLayer == NULL )
			{
				std::string message = std::string("Could not create vector layer ") + SHPFileOutLayer;
				throw SPDProcessingException(message.c_str());
			}
			
			SPDFile *spdFile = new SPDFile(spdInputFile);
			SPDFileIncrementalReader *spdReader = new SPDFileIncrementalReader();
			spdReader->open(spdFile);
			
			SPDProcessPolygons processPolys(processor);
			processPolys.processPolygons(spdFile, spdReader, inputSHPLayer, outputSHPLayer, copyAttributes);
			
			spdReader->close();
			delete spdReader;
			delete spdFile;
		}
		catch(SPDProcessingException &e)
		{
			throw e;
		}
		
		GDALClose(inputSHPDS);
		GDALClose(outputSHPDS);
		
	}
	
	void SPDSetupProcessShapefilePolygons::processPolygons(std::string spdInputFile, std::string inputLayer, std::string outputASCII, SPDPolygonProcessor *processor) throw(SPDProcessingException)
	{
		OGRRegisterAll();
		
		std::string SHPFileInLayer = this->getLayerName(inputLayer);
		
		GDALDataset *inputSHPDS = NULL;
		OGRLayer *inputSHPLayer = NULL;
		
		try
		{
			/////////////////////////////////////
			//
			// Open Input Shapfile.
			//
			/////////////////////////////////////
			inputSHPDS = (GDALDataset*) GDALOpenEx(inputLayer.c_str(), GDAL_OF_VECTOR, NULL, NULL, NULL);
			if(inputSHPDS == NULL)
			{
				std::string message = std::string("Could not open vector file ") + inputLayer;
				throw SPDProcessingException(message.c_str());
			}
			inputSHPLayer = inputSHPDS->GetLayerByName(SHPFileInLayer.c_str());
			if(inputSHPLayer == NULL)
			{
				std::string message = std::string("Could not open vector layer ") + SHPFileInLayer;
				throw SPDProcessingException(message.c_str());
			}
			
            std::ofstream *outASCIIFile = new std::ofstream();
            outASCIIFile->open(outputASCII.c_str(), std::ios::out | std::ios::trunc);
            
            if(!outASCIIFile->is_open())
            {                
                std::string message = std::string("Could not open file ") + outputASCII;
                throw SPDProcessingException(message);
            }

            
			SPDFile *spdFile = new SPDFile(spdInputFile);
			SPDFileIncrementalReader *spdReader = new SPDFileIncrementalReader();
			spdReader->open(spdFile);
			
			SPDProcessPolygons processPolys(processor);
			processPolys.processPolygons(spdFile, spdReader, inputSHPLayer, outASCIIFile);
			
			outASCIIFile->flush();
            outASCIIFile->close();
            delete outASCIIFile;
            
            spdReader->close();
			delete spdReader;
			delete spdFile;
		}
		catch(SPDProcessingException &e)
		{
			throw e;
		}
		
		GDALClose(inputSHPDS);
		
	}
	
	SPDSetupProcessShapefilePolygons::~SPDSetupProcessShapefilePolygons()
	{
		
	}
}


