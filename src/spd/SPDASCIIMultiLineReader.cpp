/*
 *  SPDASCIIMultiLineReader.cpp
 *  SPDLIB
 *
 *  Created by Pete Bunting on 28/04/2011.
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

#include "spd/SPDASCIIMultiLineReader.h"

namespace spdlib
{
	

	SPDASCIIMultiLineReader::SPDASCIIMultiLineReader(bool convertCoords, std::string outputProjWKT, std::string schema, boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold):SPDDataImporter(convertCoords, outputProjWKT, schema, indexCoords, defineOrigin, originX, originY, originZ, waveNoiseThreshold)
	{
		
	}

    SPDDataImporter* SPDASCIIMultiLineReader::getInstance(bool convertCoords, std::string outputProjWKT, std::string schema, boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold)
    {
        return new SPDASCIIMultiLineReader(convertCoords, outputProjWKT, schema, indexCoords, defineOrigin, originX, originY, originZ, waveNoiseThreshold);
    }
	
	std::list<SPDPulse*>* SPDASCIIMultiLineReader::readAllDataToList(std::string inputFile, SPDFile *spdFile)throw(SPDIOException)
	{
		SPDPulseUtils pulseUtils;
        SPDTextFileUtilities textFileUtils;
		SPDTextFileLineReader lineReader;
		std::list<SPDPulse*> *pulses = new std::list<SPDPulse*>();
		boost::uint_fast64_t numPulses = 0;
		boost::uint_fast64_t totalNumPoints = 0;
        std::string pointLine = "";
		
		double xMin = 0;
		double xMax = 0;
		double yMin = 0;
		double yMax = 0;
		double zMin = 0;
		double zMax = 0;
		bool first = true;
		bool firstZ = true;
        bool firstXY = true;
		
		classWarningGiven = false;
		
		try
		{
            if(convertCoords)
			{
				this->initCoordinateSystemTransformation(spdFile);
			}

            SPDPulse *pulse = NULL;
			SPDPoint *point = NULL;
			
			std::vector<std::string> *lineTokens = new std::vector<std::string>();
			
			lineReader.openFile(inputFile);
			std::cout << "Read ." << std::flush;
            first = true;
			while(!lineReader.endOfFile())
			{
				if((numPulses % 10000) == 0)
				{
					std::cout << "." << numPulses << "." << std::flush;
				}

                pointLine = lineReader.readLine();
				
				if(!textFileUtils.blankline(pointLine))
				{
                    lineTokens->clear();
					textFileUtils.tokenizeString(pointLine, ' ', lineTokens);
                    if(lineTokens->size() != 8)
                    {
                        std::cout << "Line: " << pointLine << std::endl;
                        throw SPDIOException("Expected 8 tokens in line.");
                    }

                    point = this->convertLineToPoint(lineTokens);
                    ++totalNumPoints;

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

                    if(point->returnID == 1)
                    {
                        if(!first)
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
                            else
                            {
                                throw SPDIOException("Indexing type unsupported");
                            }

                            if(firstXY)
                            {
                                xMin = pulse->xIdx;
                                xMax = pulse->xIdx;
                                yMin = pulse->yIdx;
                                yMax = pulse->yIdx;
                                firstXY = false;
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
                            }
                            pulses->push_back(pulse);
                        }
                        else
                        {
                            first = false;
                        }
                        pulse = new SPDPulse();
                        pulse->pulseID = numPulses++;
                        pulseUtils.initSPDPulse(pulse);
                        pulse->gpsTime = (textFileUtils.strtodouble(lineTokens->at(0))*1000000000);
                        pulse->scanAngleRank = textFileUtils.strtofloat(lineTokens->at(7));
                        pulse->pts->push_back(point);
                        pulse->numberOfReturns = 1;
                        if(pulse->numberOfReturns > 0)
                        {
                            pulse->pts->reserve(pulse->numberOfReturns);
                        }
                    }
                    else
                    {
                        pulse->pts->push_back(point);
                        pulse->numberOfReturns += 1;
                    }

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
            else
            {
                throw SPDIOException("Indexing type unsupported");
            }

            if(firstXY)
            {
                xMin = pulse->xIdx;
                xMax = pulse->xIdx;
                yMin = pulse->yIdx;
                yMax = pulse->yIdx;
                firstXY = false;
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
            }
            pulses->push_back(pulse);
            std::cout << ". Complete\n";
				
            spdFile->setBoundingVolume(xMin, xMax, yMin, yMax, zMin, zMax);
            if(convertCoords)
            {
                spdFile->setSpatialReference(outputProjWKT);
            }
            spdFile->setNumberOfPulses(numPulses);
            spdFile->setNumberOfPoints(totalNumPoints);
            spdFile->setOriginDefined(SPD_FALSE);
            spdFile->setDiscretePtDefined(SPD_TRUE);
            spdFile->setDecomposedPtDefined(SPD_FALSE);
            spdFile->setTransWaveformDefined(SPD_FALSE);
            spdFile->setReceiveWaveformDefined(SPD_FALSE);
		}
		catch (SPDIOException &e)
		{
			throw e;
		}
		catch(std::invalid_argument &e)
		{
			throw SPDIOException(e.what());
		}
		catch(std::runtime_error &e)
		{
			throw SPDIOException(e.what());
		}
		
		return pulses;
	}
	
	std::vector<SPDPulse*>* SPDASCIIMultiLineReader::readAllDataToVector(std::string inputFile, SPDFile *spdFile)throw(SPDIOException)
	{
		SPDPulseUtils pulseUtils;
        SPDTextFileUtilities textFileUtils;
		SPDTextFileLineReader lineReader;
		std::vector<SPDPulse*> *pulses = new std::vector<SPDPulse*>();
		boost::uint_fast64_t numPulses = 0;
		boost::uint_fast64_t totalNumPoints = 0;
        std::string pointLine = "";
		
		double xMin = 0;
		double xMax = 0;
		double yMin = 0;
		double yMax = 0;
		double zMin = 0;
		double zMax = 0;
		bool first = true;
		bool firstZ = true;
        bool firstXY = true;
		
		classWarningGiven = false;
		
		try
		{
            if(convertCoords)
			{
				this->initCoordinateSystemTransformation(spdFile);
			}

            SPDPulse *pulse = NULL;
			SPDPoint *point = NULL;
			
			std::vector<std::string> *lineTokens = new std::vector<std::string>();
			
			lineReader.openFile(inputFile);
			std::cout << "Read ." << std::flush;
            first = true;
			while(!lineReader.endOfFile())
			{
				if((numPulses % 10000) == 0)
				{
					std::cout << "." << numPulses << "." << std::flush;
				}

                pointLine = lineReader.readLine();
				
				if(!textFileUtils.blankline(pointLine))
				{
                    lineTokens->clear();
					textFileUtils.tokenizeString(pointLine, ' ', lineTokens);
                    if(lineTokens->size() != 8)
                    {
                        std::cout << "Line: " << pointLine << std::endl;
                        throw SPDIOException("Expected 8 tokens in line.");
                    }

                    point = this->convertLineToPoint(lineTokens);
                    ++totalNumPoints;

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

                    if(point->returnID == 1)
                    {
                        if(!first)
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
                            else
                            {
                                throw SPDIOException("Indexing type unsupported");
                            }

                            if(firstXY)
                            {
                                xMin = pulse->xIdx;
                                xMax = pulse->xIdx;
                                yMin = pulse->yIdx;
                                yMax = pulse->yIdx;
                                firstXY = false;
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
                            }
                            pulses->push_back(pulse);
                        }
                        else
                        {
                            first = false;
                        }
                        pulse = new SPDPulse();
                        pulse->pulseID = numPulses++;
                        pulseUtils.initSPDPulse(pulse);
                        pulse->gpsTime = (textFileUtils.strtodouble(lineTokens->at(0))*1000000000);
                        pulse->scanAngleRank = textFileUtils.strtofloat(lineTokens->at(7));
                        pulse->pts->push_back(point);
                        pulse->numberOfReturns = 1;
                        if(pulse->numberOfReturns > 0)
                        {
                            pulse->pts->reserve(pulse->numberOfReturns);
                        }
                    }
                    else
                    {
                        pulse->pts->push_back(point);
                        pulse->numberOfReturns += 1;
                    }

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
            else
            {
                throw SPDIOException("Indexing type unsupported");
            }

            if(firstXY)
            {
                xMin = pulse->xIdx;
                xMax = pulse->xIdx;
                yMin = pulse->yIdx;
                yMax = pulse->yIdx;
                firstXY = false;
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
            }
            pulses->push_back(pulse);
            std::cout << ". Complete\n";

            spdFile->setBoundingVolume(xMin, xMax, yMin, yMax, zMin, zMax);
            if(convertCoords)
            {
                spdFile->setSpatialReference(outputProjWKT);
            }
            spdFile->setNumberOfPulses(numPulses);
            spdFile->setNumberOfPoints(totalNumPoints);
            spdFile->setOriginDefined(SPD_FALSE);
            spdFile->setDiscretePtDefined(SPD_TRUE);
            spdFile->setDecomposedPtDefined(SPD_FALSE);
            spdFile->setTransWaveformDefined(SPD_FALSE);
            spdFile->setReceiveWaveformDefined(SPD_FALSE);
		}
		catch (SPDIOException &e)
		{
			throw e;
		}
		catch(std::invalid_argument &e)
		{
			throw SPDIOException(e.what());
		}
		catch(std::runtime_error &e)
		{
			throw SPDIOException(e.what());
		}
		
		return pulses;
	}
	
	void SPDASCIIMultiLineReader::readAndProcessAllData(std::string inputFile, SPDFile *spdFile, SPDImporterProcessor *processor)throw(SPDIOException)
	{
		SPDPulseUtils pulseUtils;
        SPDTextFileUtilities textFileUtils;
		SPDTextFileLineReader lineReader;
		boost::uint_fast64_t numPulses = 0;
		boost::uint_fast64_t totalNumPoints = 0;
        std::string pointLine = "";
		
		double xMin = 0;
		double xMax = 0;
		double yMin = 0;
		double yMax = 0;
		double zMin = 0;
		double zMax = 0;
		bool first = true;
		bool firstZ = true;
        bool firstXY = true;
		
		classWarningGiven = false;
		
		try
		{
            if(convertCoords)
			{
				this->initCoordinateSystemTransformation(spdFile);
			}

            SPDPulse *pulse = NULL;
			SPDPoint *point = NULL;
			
			std::vector<std::string> *lineTokens = new std::vector<std::string>();
			
			lineReader.openFile(inputFile);
			std::cout << "Read ." << std::flush;
            first = true;
			while(!lineReader.endOfFile())
			{
				if((numPulses % 10000) == 0)
				{
					std::cout << "." << numPulses << "." << std::flush;
				}

                pointLine = lineReader.readLine();
				
				if(!textFileUtils.blankline(pointLine))
				{
                    lineTokens->clear();
					textFileUtils.tokenizeString(pointLine, ' ', lineTokens);
                    if(lineTokens->size() != 8)
                    {
                        std::cout << "lineTokens->size() = " << lineTokens->size() << std::endl;
                        std::cout << "Line: " << pointLine << std::endl;
                        throw SPDIOException("Expected 8 tokens in line.");
                    }

                    point = this->convertLineToPoint(lineTokens);
                    ++totalNumPoints;

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

                    if(point->returnID == 1)
                    {
                        if(!first)
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
                            else
                            {
                                throw SPDIOException("Indexing type unsupported");
                            }

                            if(firstXY)
                            {
                                xMin = pulse->xIdx;
                                xMax = pulse->xIdx;
                                yMin = pulse->yIdx;
                                yMax = pulse->yIdx;
                                firstXY = false;
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
                            }
                            processor->processImportedPulse(spdFile, pulse);
                        }
                        else
                        {
                            first = false;
                        }
                        pulse = new SPDPulse();
                        pulse->pulseID = numPulses++;
                        pulseUtils.initSPDPulse(pulse);
                        pulse->gpsTime = (textFileUtils.strtodouble(lineTokens->at(0))*1000000000);
                        pulse->scanAngleRank = textFileUtils.strtofloat(lineTokens->at(7));
                        pulse->pts->push_back(point);
                        pulse->numberOfReturns = 1;
                        if(pulse->numberOfReturns > 0)
                        {
                            pulse->pts->reserve(pulse->numberOfReturns);
                        }
                    }
                    else
                    {
                        pulse->pts->push_back(point);
                        pulse->numberOfReturns += 1;
                    }

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
            else
            {
                throw SPDIOException("Indexing type unsupported");
            }

            if(firstXY)
            {
                xMin = pulse->xIdx;
                xMax = pulse->xIdx;
                yMin = pulse->yIdx;
                yMax = pulse->yIdx;
                firstXY = false;
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
            }
            processor->processImportedPulse(spdFile, pulse);
            std::cout << ". Complete\n";

            spdFile->setBoundingVolume(xMin, xMax, yMin, yMax, zMin, zMax);
            if(convertCoords)
            {
                spdFile->setSpatialReference(outputProjWKT);
            }
            spdFile->setNumberOfPulses(numPulses);
            spdFile->setNumberOfPoints(totalNumPoints);
            spdFile->setOriginDefined(SPD_FALSE);
            spdFile->setDiscretePtDefined(SPD_TRUE);
            spdFile->setDecomposedPtDefined(SPD_FALSE);
            spdFile->setTransWaveformDefined(SPD_FALSE);
            spdFile->setReceiveWaveformDefined(SPD_FALSE);
		}
		catch (SPDIOException &e)
		{
			throw e;
		}
		catch(std::invalid_argument &e)
		{
			throw SPDIOException(e.what());
		}
		catch(std::runtime_error &e)
		{
			throw SPDIOException(e.what());
		}
	}
	
	bool SPDASCIIMultiLineReader::isFileType(std::string fileType)
	{
		if(fileType == "ASCIIMULTILINE")
		{
			return true;
		}
		return false;
	}

    void SPDASCIIMultiLineReader::readHeaderInfo(std::string, SPDFile*) throw(SPDIOException)
    {
        // No Header to Read..
    }
	
	SPDPoint* SPDASCIIMultiLineReader::convertLineToPoint(std::vector<std::string> *lineTokens)throw(SPDIOException)
	{
		try
		{
            SPDTextFileUtilities textFileUtils;
			SPDPointUtils spdPtUtils;
			SPDPoint *spdPt = new SPDPoint();
			spdPtUtils.initSPDPoint(spdPt);
			
            double x = textFileUtils.strtodouble(lineTokens->at(1));
			double y = textFileUtils.strtodouble(lineTokens->at(2));
			double z = textFileUtils.strtodouble(lineTokens->at(3));
			
			if(convertCoords)
			{
				this->transformCoordinateSystem(&x, &y, &z);
			}
			
			spdPt->x = x;
			spdPt->y = y;
			spdPt->z = z;
			spdPt->amplitudeReturn = textFileUtils.strtofloat(lineTokens->at(4));
			spdPt->returnID = textFileUtils.strto16bitUInt(lineTokens->at(6));
            spdPt->gpsTime = (textFileUtils.strtodouble(lineTokens->at(0))*1000000000);
            spdPt->classification = textFileUtils.strto16bitUInt(lineTokens->at(5));

			return spdPt;
		}
		catch (SPDIOException &e)
		{
			throw e;
		}
		catch(std::invalid_argument &e)
		{
			throw SPDIOException(e.what());
		}
		catch(std::runtime_error &e)
		{
			throw SPDIOException(e.what());
		}
	}
	
	SPDASCIIMultiLineReader::~SPDASCIIMultiLineReader()
	{
		
	}	
}

