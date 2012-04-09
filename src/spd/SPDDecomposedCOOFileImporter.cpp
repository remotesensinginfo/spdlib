/*
 *  SPDDecomposedCOOFileImporter.cpp
 *  SPDLIB
 *
 *  Created by Pete Bunting on 03/01/2011.
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

#include "spd/SPDDecomposedCOOFileImporter.h"


namespace spdlib
{
	
	SPDDecomposedCOOFileImporter::SPDDecomposedCOOFileImporter(bool convertCoords, string outputProjWKT, string schema, boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold) : SPDDataImporter(convertCoords, outputProjWKT, schema, indexCoords, defineOrigin, originX, originY, originZ, waveNoiseThreshold), countIgnoredPulses(0)
	{
		
	}
	
    SPDDataImporter* SPDDecomposedCOOFileImporter::getInstance(bool convertCoords, string outputProjWKT, string schema, boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold)
    {
        return new SPDDecomposedCOOFileImporter(convertCoords, outputProjWKT, schema, indexCoords, defineOrigin, originX, originY, originZ, waveNoiseThreshold);
    }
    
	list<SPDPulse*>* SPDDecomposedCOOFileImporter::readAllDataToList(string inputFile, SPDFile *spdFile)throw(SPDIOException)
	{
		SPDTextFileUtilities textFileUtils;
		SPDTextFileLineReader lineReader;
		SPDPulseUtils pulseUtils;
		
		double xMin = 0;
		double xMax = 0;
		double yMin = 0;
		double yMax = 0;
		double zMin = 0;
		double zMax = 0;
		double azimuthMin = 0;
		double azimuthMax = 0;
		double zenithMin = 0;
		double zenithMax = 0;
		bool first = true;
		bool firstZ = true;
		string pointLine = "";
		
		boost::uint_fast64_t numPulses = 0;
		boost::uint_fast64_t totalNumPoints = 0;
		list<SPDPulse*> *pulses = new list<SPDPulse*>();
		try 
		{
			if(convertCoords)
			{
				this->initCoordinateSystemTransformation(spdFile);
			}
			
			SPDPulse *pulse = NULL;
			SPDPoint *point = NULL;
			bool incompletePulse = false;
			
			vector<string> *lineTokens = new vector<string>();
			
			lineReader.openFile(inputFile);
			cout << "Read ." << flush;
			while(!lineReader.endOfFile())
			{
				if((numPulses % 10000) == 0)
				{
					cout << "." << numPulses << "." << flush;
				}
				
				pointLine = lineReader.readLine();
				
				if(!textFileUtils.blankline(pointLine))
				{
					textFileUtils.tokenizeString(pointLine, ' ', lineTokens);
					incompletePulse = false;
					pulse = new SPDPulse();
					pulseUtils.initSPDPulse(pulse);
					pulse->numberOfReturns = textFileUtils.strto16bitUInt(lineTokens->at(9));
					pulse->gpsTime = (textFileUtils.strtodouble(lineTokens->at(1))*1000000000);
					
					double x = textFileUtils.strtodouble(lineTokens->at(17));
					double y = textFileUtils.strtodouble(lineTokens->at(18));
					double z = textFileUtils.strtodouble(lineTokens->at(19));
					if(convertCoords)
					{
						this->transformCoordinateSystem(&x, &y, &z);
					}
					pulse->x0 = x;
					pulse->y0 = y;
					pulse->z0 = z;
					
					point = this->createSPDPoint(pointLine, pulse);
					
					if(point != NULL)
					{
						if(firstZ)
						{
							zMin = point->z;
							zMax = point->z;
							firstZ = false;
						}
						else
						{
							if(point->z < zMin)
							{
								zMin = point->z;
							}
							else if(point->z > zMax)
							{
								zMax = point->z;
							}
						}
						
						pulse->pts->push_back(point);
						for(boost::uint_fast16_t i = 0; i < (pulse->numberOfReturns-1); ++i)
						{
							if(!lineReader.endOfFile())
							{
								pointLine = lineReader.readLine();
								if(!textFileUtils.blankline(pointLine))
								{
									point = this->createSPDPoint(pointLine, pulse);
									if(point != NULL)
									{
										if(firstZ)
										{
											zMin = point->z;
											zMax = point->z;
											firstZ = false;
										}
										else
										{
											if(point->z < zMin)
											{
												zMin = point->z;
											}
											else if(point->z > zMax)
											{
												zMax = point->z;
											}
										}
										pulse->pts->push_back(point);
									}
									else 
									{
										cout << "\'" << pointLine << "\'\n";
										cout << "Warning: Could not create a point from line.\n";
										incompletePulse = true;
									}
								}
								else 
								{
									throw SPDIOException("Blank line found when expecting point.");
								}
								
							}
							else 
							{
								throw SPDIOException("Unexpected end to the file.");
							}
						}
						
						if(!incompletePulse)
						{
							if(indexCoords == SPD_FIRST_RETURN)
							{
								pulse->xIdx = pulse->pts->front()->x;
								pulse->yIdx = pulse->pts->front()->y;
							}
							else if(indexCoords == SPD_LAST_RETURN)
							{
								pulse->xIdx = pulse->pts->back()->x;
								pulse->yIdx = pulse->pts->back()->y;
							}
							else if(indexCoords == SPD_MAX_INTENSITY)
							{
								unsigned int maxIdx = 0;
								double maxVal = 0;
								bool first = true;
								for(unsigned int i = 0; i < pulse->pts->size(); ++i)
								{
									if(first)
									{
										maxIdx = i;
										maxVal = pulse->pts->at(i)->amplitudeReturn;
										first = false;
									}
									else if(pulse->pts->at(i)->amplitudeReturn > maxVal)
									{
										maxIdx = i;
										maxVal = pulse->pts->at(i)->amplitudeReturn;
									}
								}
								
								pulse->xIdx = pulse->pts->at(maxIdx)->x;
								pulse->yIdx = pulse->pts->at(maxIdx)->y;
							}
							else
							{
								throw SPDIOException("Indexing type unsupported");
							}
							
							if(first)
							{
								xMin = pulse->xIdx;
								xMax = pulse->xIdx;
								yMin = pulse->yIdx;
								yMax = pulse->yIdx;
								azimuthMin = pulse->azimuth;
								azimuthMax = pulse->azimuth;
								zenithMin = pulse->zenith;
								zenithMax = pulse->zenith;
								first = false;
							}
							else
							{
								if(pulse->xIdx < xMin)
								{
									xMin = pulse->xIdx;
								}
								else if(pulse->xIdx > xMax)
								{
									xMax = pulse->xIdx;
								}
								
								if(pulse->yIdx < yMin)
								{
									yMin = pulse->yIdx;
								}
								else if(pulse->yIdx > yMax)
								{
									yMax = pulse->yIdx;
								}
								
								if(pulse->azimuth < azimuthMin)
								{
									azimuthMin = pulse->azimuth;
								}
								else if(pulse->azimuth > azimuthMax)
								{
									azimuthMax = pulse->azimuth;
								}
								
								if(pulse->zenith < zenithMin)
								{
									zenithMin = pulse->zenith;
								}
								else if(pulse->zenith > zenithMax)
								{
									zenithMax = pulse->zenith;
								}
							}
							
							totalNumPoints += pulse->numberOfReturns;
							pulse->pulseID = numPulses++;
							pulses->push_back(pulse);
						}
						else 
						{
							++countIgnoredPulses;
							++numPulses;
							SPDPulseUtils::deleteSPDPulse(pulse);
						}						
					}
					else
					{
						for(boost::uint_fast16_t i = 0; i < (pulse->numberOfReturns-1); ++i)
						{
							if(!lineReader.endOfFile())
							{
								pointLine = lineReader.readLine();
							}
						}
						++countIgnoredPulses;
						++numPulses;
						SPDPulseUtils::deleteSPDPulse(pulse);
					}
					
					lineTokens->clear();
				}
			}
			spdFile->setBoundingVolume(xMin, xMax, yMin, yMax, zMin, zMax);
			spdFile->setBoundingBoxSpherical(zenithMin, zenithMax, azimuthMin, azimuthMax);
			if(convertCoords)
			{
				spdFile->setSpatialReference(outputProjWKT);
			}
			spdFile->setNumberOfPulses(numPulses);
			spdFile->setNumberOfPoints(totalNumPoints);
			spdFile->setOriginDefined(SPD_FALSE);
			spdFile->setDiscretePtDefined(SPD_TRUE);
			spdFile->setDecomposedPtDefined(SPD_TRUE);
			spdFile->setTransWaveformDefined(SPD_FALSE);
            spdFile->setReceiveWaveformDefined(SPD_FALSE);
			lineReader.closeFile();
			delete lineTokens;
			cout << "." << numPulses << ".Pulses\n";
			if(countIgnoredPulses > 0)
			{
				cout << countIgnoredPulses << " pulses were ignored due to errors\n";
			}
		}
		catch(std::out_of_range &e)
		{
			cout << "ERROR (finding pulse): " << e.what() << endl;
			cout << "\'" << pointLine << "\'\n";
			throw SPDIOException(e.what());
		}
		catch(SPDTextFileException &e)
		{
			cout << "\'" << pointLine << "\'\n";
			throw SPDIOException(e.what());
		}
		catch (SPDIOException &e) 
		{
			throw e;
		}
		
		return pulses;
	}
	
	vector<SPDPulse*>* SPDDecomposedCOOFileImporter::readAllDataToVector(string inputFile, SPDFile *spdFile)throw(SPDIOException)
	{
		SPDTextFileUtilities textFileUtils;
		SPDTextFileLineReader lineReader;
		SPDPulseUtils pulseUtils;
		
		double xMin = 0;
		double xMax = 0;
		double yMin = 0;
		double yMax = 0;
		double zMin = 0;
		double zMax = 0;
		double azimuthMin = 0;
		double azimuthMax = 0;
		double zenithMin = 0;
		double zenithMax = 0;
		bool first = true;
		bool firstZ = true;
		string pointLine = "";
		
		boost::uint_fast64_t numPulses = 0;
		boost::uint_fast64_t totalNumPoints = 0;
		vector<SPDPulse*> *pulses = new vector<SPDPulse*>();
		try 
		{
			if(convertCoords)
			{
				this->initCoordinateSystemTransformation(spdFile);
			}
			
			SPDPulse *pulse = NULL;
			SPDPoint *point = NULL;
			bool incompletePulse = false;
			
			vector<string> *lineTokens = new vector<string>();
			
			lineReader.openFile(inputFile);
			cout << "Read ." << flush;
			while(!lineReader.endOfFile())
			{
				if((numPulses % 10000) == 0)
				{
					cout << "." << numPulses << "." << flush;
				}
				
				pointLine = lineReader.readLine();
				
				if(!textFileUtils.blankline(pointLine))
				{
					textFileUtils.tokenizeString(pointLine, ' ', lineTokens);
					incompletePulse = false;
					pulse = new SPDPulse();
					pulseUtils.initSPDPulse(pulse);
					pulse->numberOfReturns = textFileUtils.strto16bitUInt(lineTokens->at(9));
					pulse->gpsTime = (textFileUtils.strtodouble(lineTokens->at(1))*1000000000);
					
					double x = textFileUtils.strtodouble(lineTokens->at(17));
					double y = textFileUtils.strtodouble(lineTokens->at(18));
					double z = textFileUtils.strtodouble(lineTokens->at(19));
					if(convertCoords)
					{
						this->transformCoordinateSystem(&x, &y, &z);
					}
					pulse->x0 = x;
					pulse->y0 = y;
					pulse->z0 = z;
					
					point = this->createSPDPoint(pointLine, pulse);
					
					if(point != NULL)
					{
						if(firstZ)
						{
							zMin = point->z;
							zMax = point->z;
							firstZ = false;
						}
						else
						{
							if(point->z < zMin)
							{
								zMin = point->z;
							}
							else if(point->z > zMax)
							{
								zMax = point->z;
							}
						}
						
						pulse->pts->push_back(point);
						for(boost::uint_fast16_t i = 0; i < (pulse->numberOfReturns-1); ++i)
						{
							if(!lineReader.endOfFile())
							{
								pointLine = lineReader.readLine();
								if(!textFileUtils.blankline(pointLine))
								{
									point = this->createSPDPoint(pointLine, pulse);
									if(point != NULL)
									{
										if(firstZ)
										{
											zMin = point->z;
											zMax = point->z;
											firstZ = false;
										}
										else
										{
											if(point->z < zMin)
											{
												zMin = point->z;
											}
											else if(point->z > zMax)
											{
												zMax = point->z;
											}
										}
										pulse->pts->push_back(point);
									}
									else 
									{
										cout << "\'" << pointLine << "\'\n";
										cout << "Warning: Could not create a point from line.\n";
										incompletePulse = true;
									}
								}
								else 
								{
									throw SPDIOException("Blank line found when expecting point.");
								}
								
							}
							else 
							{
								throw SPDIOException("Unexpected end to the file.");
							}
						}
						
						if(!incompletePulse)
						{
							if(indexCoords == SPD_FIRST_RETURN)
							{
								pulse->xIdx = pulse->pts->front()->x;
								pulse->yIdx = pulse->pts->front()->y;
							}
							else if(indexCoords == SPD_LAST_RETURN)
							{
								pulse->xIdx = pulse->pts->back()->x;
								pulse->yIdx = pulse->pts->back()->y;
							}
							else if(indexCoords == SPD_MAX_INTENSITY)
							{
								unsigned int maxIdx = 0;
								double maxVal = 0;
								bool first = true;
								for(unsigned int i = 0; i < pulse->pts->size(); ++i)
								{
									if(first)
									{
										maxIdx = i;
										maxVal = pulse->pts->at(i)->amplitudeReturn;
										first = false;
									}
									else if(pulse->pts->at(i)->amplitudeReturn > maxVal)
									{
										maxIdx = i;
										maxVal = pulse->pts->at(i)->amplitudeReturn;
									}
								}
								
								pulse->xIdx = pulse->pts->at(maxIdx)->x;
								pulse->yIdx = pulse->pts->at(maxIdx)->y;
							}
							else
							{
								throw SPDIOException("Indexing type unsupported");
							}
							
							if(first)
							{
								xMin = pulse->xIdx;
								xMax = pulse->xIdx;
								yMin = pulse->yIdx;
								yMax = pulse->yIdx;
								azimuthMin = pulse->azimuth;
								azimuthMax = pulse->azimuth;
								zenithMin = pulse->zenith;
								zenithMax = pulse->zenith;
								first = false;
							}
							else
							{
								if(pulse->xIdx < xMin)
								{
									xMin = pulse->xIdx;
								}
								else if(pulse->xIdx > xMax)
								{
									xMax = pulse->xIdx;
								}
								
								if(pulse->yIdx < yMin)
								{
									yMin = pulse->yIdx;
								}
								else if(pulse->yIdx > yMax)
								{
									yMax = pulse->yIdx;
								}
								
								if(pulse->azimuth < azimuthMin)
								{
									azimuthMin = pulse->azimuth;
								}
								else if(pulse->azimuth > azimuthMax)
								{
									azimuthMax = pulse->azimuth;
								}
								
								if(pulse->zenith < zenithMin)
								{
									zenithMin = pulse->zenith;
								}
								else if(pulse->zenith > zenithMax)
								{
									zenithMax = pulse->zenith;
								}
							}
							
							totalNumPoints += pulse->numberOfReturns;
							pulse->pulseID = numPulses++;
							pulses->push_back(pulse);
						}
						else 
						{
							++countIgnoredPulses;
							++numPulses;
							SPDPulseUtils::deleteSPDPulse(pulse);
						}						
					}
					else
					{
						for(boost::uint_fast16_t i = 0; i < (pulse->numberOfReturns-1); ++i)
						{
							if(!lineReader.endOfFile())
							{
								pointLine = lineReader.readLine();
							}
						}
						++countIgnoredPulses;
						++numPulses;
						SPDPulseUtils::deleteSPDPulse(pulse);
					}
					
					lineTokens->clear();
				}
			}
			spdFile->setBoundingVolume(xMin, xMax, yMin, yMax, zMin, zMax);
			spdFile->setBoundingBoxSpherical(zenithMin, zenithMax, azimuthMin, azimuthMax);
			if(convertCoords)
			{
				spdFile->setSpatialReference(outputProjWKT);
			}
			spdFile->setNumberOfPulses(numPulses);
			spdFile->setNumberOfPoints(totalNumPoints);
			spdFile->setOriginDefined(SPD_FALSE);
			spdFile->setDiscretePtDefined(SPD_TRUE);
			spdFile->setDecomposedPtDefined(SPD_TRUE);
			spdFile->setTransWaveformDefined(SPD_FALSE);
            spdFile->setReceiveWaveformDefined(SPD_FALSE);
			lineReader.closeFile();
			delete lineTokens;
			cout << "." << numPulses << ".Pulses\n";
			if(countIgnoredPulses > 0)
			{
				cout << countIgnoredPulses << " pulses were ignored due to errors\n";
			}
		}
		catch(std::out_of_range &e)
		{
			cout << "ERROR (finding pulse): " << e.what() << endl;
			cout << "\'" << pointLine << "\'\n";
			throw SPDIOException(e.what());
		}
		catch(SPDTextFileException &e)
		{
			cout << "\'" << pointLine << "\'\n";
			throw SPDIOException(e.what());
		}
		catch (SPDIOException &e) 
		{
			throw e;
		}
		
		return pulses;
	}
	
	void SPDDecomposedCOOFileImporter::readAndProcessAllData(string inputFile, SPDFile *spdFile, SPDImporterProcessor *processor)throw(SPDIOException)
	{
		SPDTextFileUtilities textFileUtils;
		SPDTextFileLineReader lineReader;
		SPDPulseUtils pulseUtils;
		
		double xMin = 0;
		double xMax = 0;
		double yMin = 0;
		double yMax = 0;
		double zMin = 0;
		double zMax = 0;
		double azimuthMin = 0;
		double azimuthMax = 0;
		double zenithMin = 0;
		double zenithMax = 0;
		bool first = true;
		bool firstZ = true;
		string pointLine = "";
		
		boost::uint_fast64_t numPulses = 0;
		boost::uint_fast64_t totalNumPoints = 0;
		try 
		{
			if(convertCoords)
			{
				this->initCoordinateSystemTransformation(spdFile);
			}
			
			SPDPulse *pulse = NULL;
			SPDPoint *point = NULL;
			bool incompletePulse = false;
			
			vector<string> *lineTokens = new vector<string>();
			
			lineReader.openFile(inputFile);
			cout << "Read ." << flush;
			while(!lineReader.endOfFile())
			{
				if((numPulses % 10000) == 0)
				{
					cout << "." << numPulses << "." << flush;
				}
				
				pointLine = lineReader.readLine();
				
				if(!textFileUtils.blankline(pointLine))
				{
					textFileUtils.tokenizeString(pointLine, ' ', lineTokens);
					incompletePulse = false;
					pulse = new SPDPulse();
					pulseUtils.initSPDPulse(pulse);
					pulse->numberOfReturns = textFileUtils.strto16bitUInt(lineTokens->at(9));
					pulse->gpsTime = (textFileUtils.strtodouble(lineTokens->at(1))*1000000000);
					
					double x = textFileUtils.strtodouble(lineTokens->at(17));
					double y = textFileUtils.strtodouble(lineTokens->at(18));
					double z = textFileUtils.strtodouble(lineTokens->at(19));
					if(convertCoords)
					{
						this->transformCoordinateSystem(&x, &y, &z);
					}
					pulse->x0 = x;
					pulse->y0 = y;
					pulse->z0 = z;
					
					point = this->createSPDPoint(pointLine, pulse);
					
					if(point != NULL)
					{
						if(firstZ)
						{
							zMin = point->z;
							zMax = point->z;
							firstZ = false;
						}
						else
						{
							if(point->z < zMin)
							{
								zMin = point->z;
							}
							else if(point->z > zMax)
							{
								zMax = point->z;
							}
						}
						
						pulse->pts->push_back(point);
						for(boost::uint_fast16_t i = 0; i < (pulse->numberOfReturns-1); ++i)
						{
							if(!lineReader.endOfFile())
							{
								pointLine = lineReader.readLine();
								if(!textFileUtils.blankline(pointLine))
								{
									point = this->createSPDPoint(pointLine, pulse);
									if(point != NULL)
									{
										if(firstZ)
										{
											zMin = point->z;
											zMax = point->z;
											firstZ = false;
										}
										else
										{
											if(point->z < zMin)
											{
												zMin = point->z;
											}
											else if(point->z > zMax)
											{
												zMax = point->z;
											}
										}
										pulse->pts->push_back(point);
									}
									else 
									{
										cout << "\'" << pointLine << "\'\n";
										cout << "Warning: Could not create a point from line.\n";
										incompletePulse = true;
									}
								}
								else 
								{
									throw SPDIOException("Blank line found when expecting point.");
								}
								
							}
							else 
							{
								throw SPDIOException("Unexpected end to the file.");
							}
						}
						
						if(!incompletePulse)
						{
							if(indexCoords == SPD_FIRST_RETURN)
							{
								pulse->xIdx = pulse->pts->front()->x;
								pulse->yIdx = pulse->pts->front()->y;
							}
							else if(indexCoords == SPD_LAST_RETURN)
							{
								pulse->xIdx = pulse->pts->back()->x;
								pulse->yIdx = pulse->pts->back()->y;
							}
							else if(indexCoords == SPD_MAX_INTENSITY)
							{
								unsigned int maxIdx = 0;
								double maxVal = 0;
								bool first = true;
								for(unsigned int i = 0; i < pulse->pts->size(); ++i)
								{
									if(first)
									{
										maxIdx = i;
										maxVal = pulse->pts->at(i)->amplitudeReturn;
										first = false;
									}
									else if(pulse->pts->at(i)->amplitudeReturn > maxVal)
									{
										maxIdx = i;
										maxVal = pulse->pts->at(i)->amplitudeReturn;
									}
								}
								
								pulse->xIdx = pulse->pts->at(maxIdx)->x;
								pulse->yIdx = pulse->pts->at(maxIdx)->y;
							}
							else
							{
								throw SPDIOException("Indexing type unsupported");
							}
							
							if(first)
							{
								xMin = pulse->xIdx;
								xMax = pulse->xIdx;
								yMin = pulse->yIdx;
								yMax = pulse->yIdx;
								azimuthMin = pulse->azimuth;
								azimuthMax = pulse->azimuth;
								zenithMin = pulse->zenith;
								zenithMax = pulse->zenith;
								first = false;
							}
							else
							{
								if(pulse->xIdx < xMin)
								{
									xMin = pulse->xIdx;
								}
								else if(pulse->xIdx > xMax)
								{
									xMax = pulse->xIdx;
								}
								
								if(pulse->yIdx < yMin)
								{
									yMin = pulse->yIdx;
								}
								else if(pulse->yIdx > yMax)
								{
									yMax = pulse->yIdx;
								}
								
								if(pulse->azimuth < azimuthMin)
								{
									azimuthMin = pulse->azimuth;
								}
								else if(pulse->azimuth > azimuthMax)
								{
									azimuthMax = pulse->azimuth;
								}
								
								if(pulse->zenith < zenithMin)
								{
									zenithMin = pulse->zenith;
								}
								else if(pulse->zenith > zenithMax)
								{
									zenithMax = pulse->zenith;
								}
							}
							
							totalNumPoints += pulse->numberOfReturns;
							pulse->pulseID = numPulses++;
							processor->processImportedPulse(spdFile, pulse);
						}
						else 
						{
							++countIgnoredPulses;
							++numPulses;
							SPDPulseUtils::deleteSPDPulse(pulse);
						}						
					}
					else
					{
						for(boost::uint_fast16_t i = 0; i < (pulse->numberOfReturns-1); ++i)
						{
							if(!lineReader.endOfFile())
							{
								pointLine = lineReader.readLine();
							}
						}
						++countIgnoredPulses;
						++numPulses;
						SPDPulseUtils::deleteSPDPulse(pulse);
					}
					
					lineTokens->clear();
				}
			}
			spdFile->setBoundingVolume(xMin, xMax, yMin, yMax, zMin, zMax);
			spdFile->setBoundingBoxSpherical(zenithMin, zenithMax, azimuthMin, azimuthMax);
			if(convertCoords)
			{
				spdFile->setSpatialReference(outputProjWKT);
			}
			spdFile->setNumberOfPulses(numPulses);
			spdFile->setNumberOfPoints(totalNumPoints);
			spdFile->setOriginDefined(SPD_FALSE);
			spdFile->setDiscretePtDefined(SPD_TRUE);
			spdFile->setDecomposedPtDefined(SPD_TRUE);
			spdFile->setTransWaveformDefined(SPD_FALSE);
            spdFile->setReceiveWaveformDefined(SPD_FALSE);
			lineReader.closeFile();
			delete lineTokens;
			cout << "." << numPulses << ".Pulses\n";
			if(countIgnoredPulses > 0)
			{
				cout << countIgnoredPulses << " pulses were ignored due to errors\n";
			}
		}
		catch(std::out_of_range &e)
		{
			cout << "ERROR (finding pulse): " << e.what() << endl;
			cout << "\'" << pointLine << "\'\n";
			throw SPDIOException(e.what());
		}
		catch(SPDTextFileException &e)
		{
			cout << "\'" << pointLine << "\'\n";
			throw SPDIOException(e.what());
		}
		catch (SPDIOException &e) 
		{
			throw e;
		}
	}
	
	bool SPDDecomposedCOOFileImporter::isFileType(string fileType)
	{
		if(fileType == "DECOMPOSED_COO")
		{
			return true;
		}
		return false;
	}
    
    void SPDDecomposedCOOFileImporter::readHeaderInfo(string, SPDFile*) throw(SPDIOException)
    {
        // No Header to Read..
    }
	
	SPDPoint* SPDDecomposedCOOFileImporter::createSPDPoint(string pointLine, SPDPulse *pulse)
	{
		SPDTextFileUtilities textFileUtils;
		SPDPointUtils pointUtils;
		SPDPoint *point = new SPDPoint();
		try 
		{
			pointUtils.initSPDPoint(point);
			vector<string> *lineTokens = new vector<string>();
			textFileUtils.tokenizeString(pointLine, ' ', lineTokens);
			
			double x = textFileUtils.strtodouble(lineTokens->at(2));
			double y = textFileUtils.strtodouble(lineTokens->at(3));
			double z = textFileUtils.strtodouble(lineTokens->at(4));
			if(convertCoords)
			{
				this->transformCoordinateSystem(&x, &y, &z);
			}
			
			point->x = x;
			point->y = y;
			point->z = z;
			
			double zenith = 0;
			double azimuth = 0;
			double range = 0;
			
			SPDConvertToSpherical(pulse->x0, pulse->y0, pulse->z0, point->x, point->y, point->z, &zenith, &azimuth, &range);
			pulse->zenith = zenith;
			pulse->azimuth = azimuth;
			point->range = range;
			
			point->gpsTime = textFileUtils.strtodouble(lineTokens->at(1))*1000000000;
			try 
			{
				point->user = textFileUtils.strtofloat(lineTokens->at(5)); // Range corrected amplitude
				point->amplitudeReturn = textFileUtils.strtofloat(lineTokens->at(6));
			}
			catch(SPDTextFileException &e)
			{
				cout << "WARNING: " << e.what() << endl;
				cout << "Line: \'" << pointLine << "\'\n";
				delete point;
				return NULL;
			}
			point->returnID = textFileUtils.strto16bitUInt(lineTokens->at(8));
			point->widthReturn = textFileUtils.strtofloat(lineTokens->at(7));
			point->classification = SPD_UNCLASSIFIED;
			
			delete lineTokens;
		}
		catch(SPDTextFileException &e)
		{
			cout << "ERROR (creating point): " << e.what() << endl;
			cout << "\'" << pointLine << "\'\n";
			throw e;
		}
		catch(std::out_of_range &e)
		{
			cout << "ERROR (creating point): " << e.what() << endl;
			cout << "\'" << pointLine << "\'\n";
			throw e;
		}
		
		
		return point;
	}
	
	SPDDecomposedCOOFileImporter::~SPDDecomposedCOOFileImporter()
	{
		
	}
}





