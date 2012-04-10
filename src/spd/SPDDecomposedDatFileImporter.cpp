/*
 *  SPDDecomposedDatFileImporter.cpp
 *  spdlib
 *
 *  Created by Pete Bunting on 02/12/2010.
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

#include "spd/SPDDecomposedDatFileImporter.h"


namespace spdlib
{
	
	SPDDecomposedDatFileImporter::SPDDecomposedDatFileImporter(bool convertCoords, string outputProjWKT, string schema, boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold) : SPDDataImporter(convertCoords, outputProjWKT, schema, indexCoords, defineOrigin, originX, originY, originZ, waveNoiseThreshold), classWarningGiven(false)
	{
		
	}
    
    SPDDataImporter* SPDDecomposedDatFileImporter::getInstance(bool convertCoords, string outputProjWKT, string schema, boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold)
    {
        return new SPDDecomposedDatFileImporter(convertCoords, outputProjWKT, schema, indexCoords, defineOrigin, originX, originY, originZ, waveNoiseThreshold);
    }
	
	list<SPDPulse*>* SPDDecomposedDatFileImporter::readAllDataToList(string inputFile, SPDFile *spdFile)throw(SPDIOException)
	{
		SPDTextFileUtilities textFileUtils;
		SPDTextFileLineReader lineReader;
		SPDPulseUtils pulseUtils;
		classWarningGiven = false;
		
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
        double zenith = 0;
        double gpsTime = 0;
        double rangeTime = 0;
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
			
			vector<string> *lineTokens = new vector<string>();
			
			lineReader.openFile(inputFile);
			cout << "Read ." << flush;
			while(!lineReader.endOfFile())
			{
				if((numPulses % 100000) == 0)
				{
					cout << "." << numPulses << "." << flush;
				}

				pointLine = lineReader.readLine();

				if(!textFileUtils.blankline(pointLine))
				{
					textFileUtils.tokenizeString(pointLine, ' ', lineTokens);
					pulse = new SPDPulse();
					pulseUtils.initSPDPulse(pulse);
					pulse->numberOfReturns = textFileUtils.strto16bitUInt(lineTokens->at(10));
                    gpsTime = (textFileUtils.strtodouble(lineTokens->at(5))*1000000);
					rangeTime = (textFileUtils.strtodouble(lineTokens->at(13))/SPD_SPEED_OF_LIGHT_NS)*2;
                    pulse->gpsTime = gpsTime - rangeTime;

                    /* Retain the info on scan direction */
                    zenith = textFileUtils.strtodouble(lineTokens->at(12));
                    pulse->zenith = (180.0 - abs(zenith-90.0)) * M_PI / 180.0;
                    if(zenith < 90)
                    {
                        pulse->user = 0;
                    }
                    else
                    {
                        pulse->user = 1;                     
                    }
                                 
                    point = this->createSPDPoint(pointLine);
					
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
								point = this->createSPDPoint(pointLine);
								pulse->pts->push_back(point);
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
					
					totalNumPoints += pulse->numberOfReturns;
					pulse->pulseID = numPulses++;
					pulses->push_back(pulse);
					
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
			spdFile->setDiscretePtDefined(SPD_FALSE);
			spdFile->setDecomposedPtDefined(SPD_TRUE);
			spdFile->setTransWaveformDefined(SPD_FALSE);
            spdFile->setReceiveWaveformDefined(SPD_FALSE);
			lineReader.closeFile();
			delete lineTokens;
			cout << "." << numPulses << ".Pulses\n";
		}
		catch(std::out_of_range &e)
		{
			cout << "ERROR (finding pulse): " << e.what() << endl;
			cout << "\'" << pointLine << "\'\n";
			throw SPDIOException(e.what());
		}
		catch(SPDTextFileException &e)
		{
			throw SPDIOException(e.what());
		}
		catch (SPDIOException &e) 
		{
			throw e;
		}
		
		return pulses;
	}
	
	vector<SPDPulse*>* SPDDecomposedDatFileImporter::readAllDataToVector(string inputFile, SPDFile *spdFile)throw(SPDIOException)
	{
		SPDTextFileUtilities textFileUtils;
		SPDTextFileLineReader lineReader;
		SPDPulseUtils pulseUtils;
		classWarningGiven = false;
		
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
        double zenith = 0;
        double gpsTime = 0;
        double rangeTime = 0;
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
					pulse = new SPDPulse();
					pulseUtils.initSPDPulse(pulse);
					pulse->numberOfReturns = textFileUtils.strto16bitUInt(lineTokens->at(10));
                    gpsTime = (textFileUtils.strtodouble(lineTokens->at(5))*1000000);
					rangeTime = (textFileUtils.strtodouble(lineTokens->at(13))/SPD_SPEED_OF_LIGHT_NS)*2;
                    pulse->gpsTime = gpsTime - rangeTime;
                          
                    /* Retain the info on scan direction */
                    zenith = textFileUtils.strtodouble(lineTokens->at(12));
                    pulse->zenith = (180.0 - abs(zenith-90.0)) * M_PI / 180.0;
                    if(zenith < 90)
                    {
                        pulse->user = 0;
                    }
                    else
                    {
                        pulse->user = 1;                     
                    }
                    
                    point = this->createSPDPoint(pointLine);
					
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
								point = this->createSPDPoint(pointLine);
								pulse->pts->push_back(point);
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
					
					totalNumPoints += pulse->numberOfReturns;
					pulse->pulseID = numPulses++;
					pulses->push_back(pulse);
					
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
			spdFile->setDiscretePtDefined(SPD_FALSE);
			spdFile->setDecomposedPtDefined(SPD_TRUE);
			spdFile->setTransWaveformDefined(SPD_FALSE);
            spdFile->setReceiveWaveformDefined(SPD_FALSE);
			lineReader.closeFile();
			delete lineTokens;
			cout << "." << numPulses << ".Pulses\n";
		}
		catch(std::out_of_range &e)
		{
			cout << "ERROR (finding pulse): " << e.what() << endl;
			cout << "\'" << pointLine << "\'\n";
			throw SPDIOException(e.what());
		}
		catch(SPDTextFileException &e)
		{
			throw SPDIOException(e.what());
		}
		catch (SPDIOException &e) 
		{
			throw e;
		}
		
		return pulses;
	}
	
	void SPDDecomposedDatFileImporter::readAndProcessAllData(string inputFile, SPDFile *spdFile, SPDImporterProcessor *processor)throw(SPDIOException)
	{
		SPDTextFileUtilities textFileUtils;
		SPDTextFileLineReader lineReader;
		SPDPulseUtils pulseUtils;
		classWarningGiven = false;
		
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
        double zenith = 0;
        double gpsTime = 0;
        double rangeTime = 0;
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
					pulse = new SPDPulse();
					pulseUtils.initSPDPulse(pulse);
					pulse->numberOfReturns = textFileUtils.strto16bitUInt(lineTokens->at(10));
                    gpsTime = (textFileUtils.strtodouble(lineTokens->at(5))*1000000);
					rangeTime = (textFileUtils.strtodouble(lineTokens->at(13))/SPD_SPEED_OF_LIGHT_NS)*2;
                    pulse->gpsTime = gpsTime - rangeTime;
                    
                    /* Retain the info on scan direction */
                    zenith = textFileUtils.strtodouble(lineTokens->at(12));
                    pulse->zenith = (180.0 - abs(zenith-90.0)) * M_PI / 180.0;
                    if(zenith < 90)
                    {
                        pulse->user = 0;
                    }
                    else
                    {
                        pulse->user = 1;                     
                    }

					point = this->createSPDPoint(pointLine);
					
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
								point = this->createSPDPoint(pointLine);
								pulse->pts->push_back(point);
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
			spdFile->setDiscretePtDefined(SPD_FALSE);
			spdFile->setDecomposedPtDefined(SPD_TRUE);
			spdFile->setTransWaveformDefined(SPD_FALSE);
            spdFile->setReceiveWaveformDefined(SPD_FALSE);
			lineReader.closeFile();
			delete lineTokens;
			cout << "." << numPulses << ".Pulses\n";
		}
		catch(std::out_of_range &e)
		{
			cout << "ERROR (finding pulse): " << e.what() << endl;
			cout << "\'" << pointLine << "\'\n";
			throw SPDIOException(e.what());
		}
		catch(SPDTextFileException &e)
		{
			throw SPDIOException(e.what());
		}
		catch (SPDIOException &e) 
		{
			throw e;
		}
	}
	
	bool SPDDecomposedDatFileImporter::isFileType(string fileType)
	{
		if(fileType == "DECOMPOSED_DAT")
		{
			return true;
		}
		return false;
	}
    
    void SPDDecomposedDatFileImporter::readHeaderInfo(string, SPDFile*) throw(SPDIOException)
    {
        // No Header to Read..
    }
	
	SPDPoint* SPDDecomposedDatFileImporter::createSPDPoint(string pointLine)
	{
		SPDTextFileUtilities textFileUtils;
		SPDPointUtils pointUtils;
		SPDPoint *point = new SPDPoint();
		try 
		{
			pointUtils.initSPDPoint(point);
			vector<string> *lineTokens = new vector<string>();
			textFileUtils.tokenizeString(pointLine, ' ', lineTokens);
			
			double x = textFileUtils.strtodouble(lineTokens->at(0));
			double y = textFileUtils.strtodouble(lineTokens->at(1));
			double z = textFileUtils.strtodouble(lineTokens->at(2));
			if(convertCoords)
			{
				this->transformCoordinateSystem(&x, &y, &z);
			}
			
			point->x = x;
			point->y = y;
			point->z = z;
			point->range = textFileUtils.strtofloat(lineTokens->at(13));
			point->gpsTime = textFileUtils.strtodouble(lineTokens->at(5))*1000000;
			point->amplitudeReturn = textFileUtils.strtofloat(lineTokens->at(6));
			point->returnID = textFileUtils.strto16bitUInt(lineTokens->at(8));
			point->widthReturn = textFileUtils.strtofloat(lineTokens->at(11));
			point->classification = textFileUtils.strto16bitUInt(lineTokens->at(7));
			
			if(point->classification == 0)
			{
				point->classification = SPD_CREATED;
			}
			else if(point->classification == 1)
			{
				point->classification = SPD_UNCLASSIFIED;
			}
			else if(point->classification == 2)
			{
				point->classification = SPD_GROUND;
			}
			else if(point->classification == 3)
			{
				point->classification = SPD_LOW_VEGETATION;
			}
			else if(point->classification == 4)
			{
				point->classification = SPD_MEDIUM_VEGETATION;
			}
			else if(point->classification == 5)
			{
				point->classification = SPD_HIGH_VEGETATION;
			}
			else if(point->classification == 6)
			{
				point->classification = SPD_BUILDING;
			}
			else if(point->classification == 7)
			{
				point->classification = SPD_CREATED;
				point->modelKeyPoint = SPD_TRUE;
			}
			else if(point->classification == 8)
			{
				point->classification = SPD_CREATED;
				point->lowPoint = SPD_TRUE;
			}
			else if(point->classification == 9)
			{
				point->classification = SPD_WATER;
			}
			else if(point->classification == 12)
			{
				point->classification = SPD_CREATED;
				point->overlap = SPD_TRUE;
			}
			else 
			{
				if(!classWarningGiven)
				{
					cerr << "WARNING: The class ID was not recognised - check the classes points were allocated too.";
					classWarningGiven = true;
				}
				point->classification = SPD_CREATED;
			}
			delete lineTokens;
		}
		catch(std::out_of_range &e)
		{
			cout << "ERROR (creating point): " << e.what() << endl;
			cout << "\'" << pointLine << "\'\n";
			throw e;
		}
		
		return point;
	}
	
	SPDDecomposedDatFileImporter::~SPDDecomposedDatFileImporter()
	{
		
	}
}





