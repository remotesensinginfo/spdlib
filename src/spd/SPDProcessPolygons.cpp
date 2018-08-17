/*
 *  SPDProcessPolygons.cpp
 *  SPDLIB
 *
 *  Created by Pete Bunting on 09/03/2011.
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
 */

#include "spd/SPDProcessPolygons.h"

namespace spdlib
{	
	SPDProcessPolygons::SPDProcessPolygons(SPDPolygonProcessor *processor): processor(NULL)
	{
		this->processor = processor;
	}
	
	void SPDProcessPolygons::processPolygons(SPDFile *spdFile, SPDFileIncrementalReader *spdReader, OGRLayer *inputLayer, OGRLayer *outputLayer, bool copyAttributes)throw(SPDProcessingException)
	{
		OGRGeometry *geometry = NULL;
		
		OGRFeature *inFeature = NULL;
		OGRFeature *outFeature = NULL;
		
		OGRFeatureDefn *inFeatureDefn = NULL;
		OGRFeatureDefn *outFeatureDefn = NULL;
		
		boost::uint_fast64_t fid = 0;
		
		std::vector<SPDPulse*> *pulses = new std::vector<SPDPulse*>();
		std::vector<SPDPulse*>::iterator iterPulses;

		try
		{
			inFeatureDefn = inputLayer->GetLayerDefn();
			
			if(copyAttributes)
			{
				this->copyFeatureDefn(outputLayer, inFeatureDefn);
			}
			this->processor->createOutputLayerDefinition(outputLayer, inFeatureDefn);
			
			outFeatureDefn = outputLayer->GetLayerDefn();
			
			int numFeatures = inputLayer->GetFeatureCount(TRUE);
			
			bool nullGeometry = false;
			
			boost::uint_fast32_t feedback = numFeatures/10;
			boost::uint_fast32_t feedbackCounter = 0;
			boost::uint_fast32_t i = 0;
			
			std::cout << "Started " << std::flush;	
			
			inputLayer->ResetReading();
			while( (inFeature = inputLayer->GetNextFeature()) != NULL )
			{
                fid = inFeature->GetFID();

				if((numFeatures > 10) && (i % feedback) == 0)
				{
					std::cout << ".." << feedbackCounter << ".." << std::flush;
					feedbackCounter = feedbackCounter + 10;
				}
                else if(numFeatures <= 10)
                {
                    std::cout << ".." << fid << ".." << std::flush;
                }				
				
				outFeature = OGRFeature::CreateFeature(outFeatureDefn);
				
				// Get Geometry.
				nullGeometry = false;
				geometry = inFeature->GetGeometryRef();
				if( geometry != NULL && wkbFlatten(geometry->getGeometryType()) == wkbPolygon )
				{
					OGRPolygon *polygon = (OGRPolygon *) geometry;
					outFeature->SetGeometry(polygon);
				}
				else if( geometry != NULL && wkbFlatten(geometry->getGeometryType()) == wkbMultiPolygon )
				{
					OGRMultiPolygon *multiPolygon = (OGRMultiPolygon *) geometry;
					outFeature->SetGeometry(multiPolygon);
				}
				else if( geometry != NULL && wkbFlatten(geometry->getGeometryType()) == wkbPoint )
				{
					throw SPDProcessingException("Processor will only handle Polygon data not Point.");
				}	
				else if( geometry != NULL && wkbFlatten(geometry->getGeometryType()) == wkbLineString )
				{
					throw SPDProcessingException("Processor will only handle Polygon data not Line.");
				}
				else if(geometry != NULL)
				{
					std::string message = std::string("Unsupport data type: ") + std::string(geometry->getGeometryName());
					throw SPDProcessingException(message);
				}
				else
				{
					nullGeometry = true;
					std::cerr << "WARNING: NULL Geometry Present within input file - IGNORED\n";
				}
				
				if(!nullGeometry)
				{
					// Get Pulses within a Geometry..
                    spdReader->readPulseDataInGeom(pulses, geometry);

					// Process pulses to attribute feature
					processor->processFeature(inFeature, outFeature, fid, pulses, spdFile);
										
					outFeature->SetFID(fid);
					
					if(copyAttributes)
					{
						this->copyFeatureData(inFeature, outFeature, inFeatureDefn, outFeatureDefn);
					}
					
					if( outputLayer->CreateFeature(outFeature) != OGRERR_NONE )
					{
						throw SPDProcessingException("Failed to write feature to the output shapefile.");
					}
					
					OGRFeature::DestroyFeature(outFeature);

                    for(iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
                    {
                        SPDPulseUtils::deleteSPDPulse(*iterPulses);
                    }
                    pulses->clear();
				}
				OGRFeature::DestroyFeature(inFeature);
				i++;
			}
			std::cout << " Complete.\n";
		}
		catch(SPDProcessingException& e)
		{
			throw e;
		}
	}

    void SPDProcessPolygons::processPolygons(SPDFile *spdFile, SPDFileIncrementalReader *spdReader, OGRLayer *inputLayer, std::ofstream *outASCIIFile)throw(SPDProcessingException)
	{
		OGRGeometry *geometry = NULL;
		
		OGRFeature *inFeature = NULL;
		OGRFeatureDefn *inFeatureDefn = NULL;
		
		boost::uint_fast64_t fid = 0;
		
		std::vector<SPDPulse*> *pulses = new std::vector<SPDPulse*>();
		std::vector<SPDPulse*>::iterator iterPulses;

		try
		{
			inFeatureDefn = inputLayer->GetLayerDefn();
			
			this->processor->writeASCIIHeader(outASCIIFile);
			
            int numFeatures = inputLayer->GetFeatureCount(TRUE);
			
			bool nullGeometry = false;
			
			boost::uint_fast32_t feedback = numFeatures/10;
			boost::uint_fast32_t feedbackCounter = 0;
			boost::uint_fast32_t i = 0;
			
			std::cout << "Started " << std::flush;	
			
			inputLayer->ResetReading();
			while( (inFeature = inputLayer->GetNextFeature()) != NULL )
			{
                fid = inFeature->GetFID();

				if((numFeatures > 10) && (i % feedback) == 0)
				{
					std::cout << ".." << feedbackCounter << ".." << std::flush;
					feedbackCounter = feedbackCounter + 10;
				}
                else if(numFeatures <= 10)
                {
                    std::cout << ".." << fid << ".." << std::flush;
                }				
							
				// Get Geometry.
				nullGeometry = false;
				geometry = inFeature->GetGeometryRef();
				if( (geometry != NULL) && ((wkbFlatten(geometry->getGeometryType()) == wkbPolygon) | (wkbFlatten(geometry->getGeometryType()) == wkbMultiPolygon)))
				{
					
                    // Get Pulses within a Geometry..
                    spdReader->readPulseDataInGeom(pulses, geometry);

					// Process pulses to attribute feature
					processor->processFeature(inFeature, outASCIIFile, fid, pulses, spdFile);

                    for(iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
                    {
                        SPDPulseUtils::deleteSPDPulse(*iterPulses);
                    }
                    pulses->clear();

                }
				else if( geometry != NULL && wkbFlatten(geometry->getGeometryType()) == wkbPoint )
				{
					throw SPDProcessingException("Processor will only handle Polygon data not Point.");
				}	
				else if( geometry != NULL && wkbFlatten(geometry->getGeometryType()) == wkbLineString )
				{
					throw SPDProcessingException("Processor will only handle Polygon data not Line.");
				}
				else if(geometry != NULL)
				{
					std::string message = std::string("Unsupport data type: ") + std::string(geometry->getGeometryName());
					throw SPDProcessingException(message);
				}
				else
				{
					nullGeometry = true;
					std::cerr << "WARNING: NULL Geometry Present within input file - IGNORED\n";
				}

				OGRFeature::DestroyFeature(inFeature);
				i++;
			}
			std::cout << " Complete.\n";
		}
		catch(SPDProcessingException& e)
		{
			throw e;
		}
	}
	
	void SPDProcessPolygons::copyFeatureDefn(OGRLayer *outputSHPLayer, OGRFeatureDefn *inFeatureDefn) throw(SPDProcessingException)
	{
		boost::uint_fast32_t fieldCount = inFeatureDefn->GetFieldCount();
		for(boost::uint_fast32_t i = 0; i < fieldCount; ++i)
		{
			if( outputSHPLayer->CreateField( inFeatureDefn->GetFieldDefn(i) ) != OGRERR_NONE )
			{
				std::string message = std::string("Creating ") + std::string(inFeatureDefn->GetFieldDefn(i)->GetNameRef()) + std::string(" field has failed.");
				throw SPDProcessingException(message.c_str());
			}
		}
	}
	
	void SPDProcessPolygons::copyFeatureData(OGRFeature *inFeature, OGRFeature *outFeature, OGRFeatureDefn *inFeatureDefn, OGRFeatureDefn *outFeatureDefn) throw(SPDProcessingException)
	{
		boost::uint_fast32_t fieldCount = inFeatureDefn->GetFieldCount();
		for(boost::uint_fast32_t i = 0; i < fieldCount; ++i)
		{
			outFeature->SetField(outFeatureDefn->GetFieldIndex(inFeatureDefn->GetFieldDefn(i)->GetNameRef()), inFeature->GetRawFieldRef(inFeatureDefn->GetFieldIndex(inFeatureDefn->GetFieldDefn(i)->GetNameRef())));
		}
		
	}
	
	SPDProcessPolygons::~SPDProcessPolygons()
	{
		
	}
}


