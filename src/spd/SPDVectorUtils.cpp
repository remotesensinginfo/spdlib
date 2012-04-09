/*
 *  SPDVectorUtils.cpp
 *  SPDLIB
 *
 *  Created by Pete Bunting on 12/03/2011.
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

#include "spd/SPDVectorUtils.h"

namespace spdlib
{	

    SPDVectorUtils::SPDVectorUtils()
    {
        
    }
    
	OGRGeometryCollection* SPDVectorUtils::getGeometryCollection(string inputVector) throw(SPDIOException)
    {
        OGRRegisterAll();
		
		string SHPFileInLayer = this->getLayerName(inputVector);
		
		OGRDataSource *inputSHPDS = NULL;
		OGRLayer *inputSHPLayer = NULL;
		
        OGRGeometryCollection *geomCollection = new OGRGeometryCollection();
        OGRGeometry *geometry = NULL;
        OGRFeature *inFeature = NULL;
        
		try
		{
			/////////////////////////////////////
			//
			// Open Input Shapfile.
			//
			/////////////////////////////////////
			inputSHPDS = OGRSFDriverRegistrar::Open(inputVector.c_str(), FALSE);
			if(inputSHPDS == NULL)
			{
				string message = string("Could not open vector file ") + inputVector;
				throw SPDIOException(message.c_str());
			}
			inputSHPLayer = inputSHPDS->GetLayerByName(SHPFileInLayer.c_str());
			if(inputSHPLayer == NULL)
			{
				string message = string("Could not open vector layer ") + SHPFileInLayer;
				throw SPDIOException(message.c_str());
			}
			
            inputSHPLayer->ResetReading();
			while( (inFeature = inputSHPLayer->GetNextFeature()) != NULL )
			{
                geometry = inFeature->GetGeometryRef();
                if(geometry != NULL)
                {
                   geomCollection->addGeometryDirectly(geometry); 
                }
            }
        }
		catch(SPDIOException &e)
		{
			throw e;
		}
		
		OGRDataSource::DestroyDataSource(inputSHPDS);
		
		OGRCleanupAll();
        
        return geomCollection;
    }
    
    string SPDVectorUtils::getLayerName(string filepath)
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
		string filename = filepath.substr(lastSlash+1);
		
		strSize = filename.size();
		int lastpt = 0;
		for(int i = 0; i < strSize; i++)
		{
			if(filename.at(i) == '.')
			{
				lastpt = i;
			}
		}
		
		string layerName = filename.substr(0, lastpt);
		return layerName;		
	}
    
    
    SPDVectorUtils::~SPDVectorUtils()
    {
        
    }
}

