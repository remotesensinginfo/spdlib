/*
 *  SPDFullWaveformDatFileImporter.cpp
 *  spdlib
 *
 *  Created by Pete Bunting on 01/12/2010.
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

#include "spd/SPDFullWaveformDatFileImporter.h"


namespace spdlib
{

	SPDFullWaveformDatFileImporter::SPDFullWaveformDatFileImporter(bool convertCoords, std::string outputProjWKT, std::string schema, boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold) : SPDDataImporter(convertCoords, outputProjWKT, schema, indexCoords, defineOrigin, originX, originY, originZ, waveNoiseThreshold)
	{
        mathUtils = new SPDMathsUtils();
	}
	
    SPDDataImporter* SPDFullWaveformDatFileImporter::getInstance(bool convertCoords, std::string outputProjWKT, std::string schema, boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold)
    {
        return new SPDFullWaveformDatFileImporter(convertCoords, outputProjWKT, schema, indexCoords, defineOrigin, originX, originY, originZ, waveNoiseThreshold);
    }
    
	std::list<SPDPulse*>* SPDFullWaveformDatFileImporter::readAllDataToList(std::string inputFile, SPDFile *spdFile)throw(SPDIOException)
	{
		SPDTextFileUtilities textFileUtils;
		SPDTextFileLineReader lineReader;
		
        std::list<SPDPulse*> *pulses = new std::list<SPDPulse*>();
        
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
		
		SPDPulse *pulse = NULL;
		bool firstLine = true;
		bool foundTransmittedLine = true;
		bool foundReceivedLine = true;
		bool onlyTransmitted = false;
		std::string lineTrans = "";
		std::string lineReceived = "";
		std::string lineReceivedExtra = "";
		std::vector<std::string> *extralinesRec = new std::vector<std::string>();
		std::vector<std::string> *extralinesTrans = new std::vector<std::string>();
		std::vector<std::string> *tokensTrans = new std::vector<std::string>();
		std::vector<std::string> *tokensReceived = new std::vector<std::string>();
		std::vector<std::string> *tokensReceivedExtra = new std::vector<std::string>();
		
		boost::uint_fast64_t numPulses = 0;
		try 
		{
			if(convertCoords)
			{
				this->initCoordinateSystemTransformation(spdFile);
			}
			
			lineReader.openFile(inputFile);
			std::cout << "Read ." << std::flush;
			while(!lineReader.endOfFile())
			{
				onlyTransmitted = false;
				if((numPulses % 10000) == 0)
				{
					std::cout << "." << numPulses << "." << std::flush;
				}
				
				if(firstLine)
				{
					lineTrans = lineReader.readLine();
					if(!isdigit(lineTrans.at(0)))
					{
						lineTrans = lineReader.readLine();
					}
					firstLine = false;
                    boost::trim(lineTrans);
					if(textFileUtils.blankline(lineTrans))
					{
						throw SPDIOException("The first line is blank please check your file.");
					}
				}
				else 
				{
					lineTrans = lineReceivedExtra;
				}
				lineReceived = lineReader.readLine();
				boost::trim(lineReceived);
				if(textFileUtils.blankline(lineReceived))
				{
					throw SPDIOException("There is no received line due to blank line, please check you input file.");
				}
				
				textFileUtils.tokenizeString(lineTrans, ',', tokensTrans);
				textFileUtils.tokenizeString(lineReceived, ',', tokensReceived);
				
				if(!((tokensTrans->at(0) == tokensReceived->at(0)) &&
					 (tokensTrans->at(1) == tokensReceived->at(1)) &&
					 (tokensTrans->at(2) == tokensReceived->at(2)) &&
					 (tokensTrans->at(3) == tokensReceived->at(3)) &&
					 (tokensTrans->at(4) == tokensReceived->at(4)) &&
					 (tokensTrans->at(5) == tokensReceived->at(5)) &&
					 (tokensTrans->at(6) == tokensReceived->at(6))))
				{
					if((textFileUtils.strtodouble(tokensTrans->at(7)) < 500) && 
					   (textFileUtils.strtodouble(tokensReceived->at(7)) < 500))
					{
						lineReceivedExtra = lineReceived;
						onlyTransmitted = true;
					}
					else 
					{
						std::cout << "\nTransmitted Line: " << lineTrans << std::endl;
						std::cout << "Received Line: " << lineReceived << std::endl;
						throw SPDIOException("Something has gone wrong reading the input file: the trasmitted and received origin and point are different.");
					}					
				}
				
				
				if(!onlyTransmitted)
				{
					foundTransmittedLine = false;
					if((textFileUtils.strtodouble(tokensTrans->at(7)) < 500) && 
					   (textFileUtils.strtodouble(tokensReceived->at(7)) < 500))
					{
						extralinesTrans->push_back(lineReceived);
						tokensReceived->clear();
						foundTransmittedLine = true;
					}
					while(foundTransmittedLine)
					{
						if(!lineReader.endOfFile())
						{
							lineReceived = lineReader.readLine();
							boost::trim(lineReceivedExtra);
							if(!textFileUtils.blankline(lineReceivedExtra))
							{
								textFileUtils.tokenizeString(lineReceived, ',', tokensReceived);
								
								if(!((tokensTrans->at(0) == tokensReceived->at(0)) &&
									 (tokensTrans->at(1) == tokensReceived->at(1)) &&
									 (tokensTrans->at(2) == tokensReceived->at(2)) &&
									 (tokensTrans->at(3) == tokensReceived->at(3)) &&
									 (tokensTrans->at(4) == tokensReceived->at(4)) &&
									 (tokensTrans->at(5) == tokensReceived->at(5)) &&
									 (tokensTrans->at(6) == tokensReceived->at(6))))
								{
									foundTransmittedLine = false;
									onlyTransmitted = true;
									lineReceivedExtra = lineReceived;
								}
								else 
								{
									if((textFileUtils.strtodouble(tokensTrans->at(7)) < 500) && 
									   (textFileUtils.strtodouble(tokensReceived->at(7)) < 500))
									{
										extralinesTrans->push_back(lineReceived);
										tokensReceived->clear();
										foundTransmittedLine = true;
									}
									else 
									{
										foundTransmittedLine = false;
									}
								}
							}
							else 
							{
								foundTransmittedLine = false;
							}
						}
					}
				}
				
				if(!onlyTransmitted)
				{
					foundReceivedLine = true;
					while (foundReceivedLine)
					{
						if(!lineReader.endOfFile())
						{
							lineReceivedExtra = lineReader.readLine();
							boost::trim(lineReceivedExtra);
							if(!textFileUtils.blankline(lineReceivedExtra))
							{
								textFileUtils.tokenizeString(lineReceivedExtra, ',', tokensReceivedExtra);
								
								if((tokensReceived->at(0) == tokensReceivedExtra->at(0)) &&
								   (tokensReceived->at(1) == tokensReceivedExtra->at(1)) &&
								   (tokensReceived->at(2) == tokensReceivedExtra->at(2)) &&
								   (tokensReceived->at(3) == tokensReceivedExtra->at(3)) &&
								   (tokensReceived->at(4) == tokensReceivedExtra->at(4)) &&
								   (tokensReceived->at(5) == tokensReceivedExtra->at(5)) &&
								   (tokensReceived->at(6) == tokensReceivedExtra->at(6)))
								{
									extralinesRec->push_back(lineReceivedExtra);
									foundReceivedLine = true;
								}
								else
								{
									foundReceivedLine = false;
								}
								tokensReceivedExtra->clear();
							}
						}
						else 
						{
							foundReceivedLine = false;
						}
					}
				}
				
				if(onlyTransmitted)
				{
					pulse = this->createPulse(tokensTrans, extralinesTrans);
				}
				else
				{
					pulse = this->createPulse(tokensTrans, extralinesTrans, tokensReceived, extralinesRec);
				}
				
				if(pulse != NULL)
				{
					pulse->gpsTime = textFileUtils.strtodouble(tokensTrans->at(0)) + (textFileUtils.strtodouble(tokensTrans->at(7))*0.001);
					pulse->pulseID = numPulses++;
                    pulse->receiveWaveNoiseThreshold = this->waveNoiseThreshold;
					
					if(first)
					{
						xMin = pulse->xIdx;
						xMax = pulse->xIdx;
						yMin = pulse->yIdx;
						yMax = pulse->yIdx;
						zMin = pulse->z0;
						zMax = pulse->z0;
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
						
						if(pulse->numOfReceivedBins > 0)
						{
							double tempX = 0;
							double tempY = 0;
							double tempZ = 0;
							
							SPDConvertToCartesian(pulse->zenith, pulse->azimuth, pulse->rangeToWaveformStart, pulse->x0, pulse->y0, pulse->z0, &tempX, &tempY, &tempZ);
							zMin = tempZ;
							SPDConvertToCartesian(pulse->zenith, pulse->azimuth, (pulse->rangeToWaveformStart+(((pulse->numOfReceivedBins-1)*SPD_SPEED_OF_LIGHT_NS))/2), pulse->x0, pulse->y0, pulse->z0, &tempX, &tempY, &tempZ);
							zMax = tempZ;
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
					
                    pulses->push_back(pulse);
				}
				
				tokensTrans->clear();
				tokensReceived->clear();
				tokensReceivedExtra->clear();
				extralinesRec->clear();
				extralinesTrans->clear();
			}
			spdFile->setBoundingVolume(xMin, xMax, yMin, yMax, zMin, zMax);
			spdFile->setBoundingBoxSpherical(zenithMin, zenithMax, azimuthMin, azimuthMax);
			spdFile->setSpatialReference(outputProjWKT);
			spdFile->setNumberOfPulses(numPulses);
			spdFile->setOriginDefined(SPD_TRUE);
			spdFile->setDiscretePtDefined(SPD_FALSE);
			spdFile->setDecomposedPtDefined(SPD_FALSE);
            spdFile->setTransWaveformDefined(SPD_TRUE);
            spdFile->setReceiveWaveformDefined(SPD_TRUE);
			spdFile->setTemporalBinSpacing(1);
			lineReader.closeFile();
			std::cout << "." << numPulses << ".Pulses\n";
		}
		catch (SPDIOException &e) 
		{
			std::cout << "Pulse = " << numPulses << std::endl;
			if(onlyTransmitted)
			{
				std::cout << "Only Transmitted\n";
			}
			else if(extralinesRec->size() == 0)
			{
				std::cout << "Transmitted and Single line of Recieved\n";
			}
			else
			{
				std::cout << "Extra recieved lines, an extra " <<  extralinesRec->size() << " lines\n";
			}
			
			std::cout << "Transmitted: " << lineTrans << std::endl;
			std::cout << "Recieved: " << lineReceived << std::endl;
			std::cout << "Extra Lines: " << std::endl;
			std::vector<std::string>::iterator iterLines;
			for(iterLines = extralinesRec->begin(); iterLines != extralinesRec->end(); ++iterLines)
			{
				std::cout << *iterLines << std::endl;
			}
			throw e;
		}			
		return pulses;
	}
	
	std::vector<SPDPulse*>* SPDFullWaveformDatFileImporter::readAllDataToVector(std::string inputFile, SPDFile *spdFile)throw(SPDIOException)
	{
		SPDTextFileUtilities textFileUtils;
		SPDTextFileLineReader lineReader;
		
        std::vector<SPDPulse*> *pulses = new std::vector<SPDPulse*>();
        
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
		
		SPDPulse *pulse = NULL;
		bool firstLine = true;
		bool foundTransmittedLine = true;
		bool foundReceivedLine = true;
		bool onlyTransmitted = false;
		std::string lineTrans = "";
		std::string lineReceived = "";
		std::string lineReceivedExtra = "";
		std::vector<std::string> *extralinesRec = new std::vector<std::string>();
		std::vector<std::string> *extralinesTrans = new std::vector<std::string>();
		std::vector<std::string> *tokensTrans = new std::vector<std::string>();
		std::vector<std::string> *tokensReceived = new std::vector<std::string>();
		std::vector<std::string> *tokensReceivedExtra = new std::vector<std::string>();
		
		boost::uint_fast64_t numPulses = 0;
		try 
		{
			if(convertCoords)
			{
				this->initCoordinateSystemTransformation(spdFile);
			}
			
			lineReader.openFile(inputFile);
			std::cout << "Read ." << std::flush;
			while(!lineReader.endOfFile())
			{
				onlyTransmitted = false;
				if((numPulses % 10000) == 0)
				{
					std::cout << "." << numPulses << "." << std::flush;
				}
				
				if(firstLine)
				{
					lineTrans = lineReader.readLine();
					if(!isdigit(lineTrans.at(0)))
					{
						lineTrans = lineReader.readLine();
					}
					firstLine = false;
					boost::trim(lineTrans);
					if(textFileUtils.blankline(lineTrans))
					{
						throw SPDIOException("The first line is blank please check your file.");
					}
				}
				else 
				{
					lineTrans = lineReceivedExtra;
				}
				lineReceived = lineReader.readLine();
				boost::trim(lineReceived);
				if(textFileUtils.blankline(lineReceived))
				{
					throw SPDIOException("There is no received line due to blank line, please check you input file.");
				}
				
				textFileUtils.tokenizeString(lineTrans, ',', tokensTrans);
				textFileUtils.tokenizeString(lineReceived, ',', tokensReceived);
				
				if(!((tokensTrans->at(0) == tokensReceived->at(0)) &&
					 (tokensTrans->at(1) == tokensReceived->at(1)) &&
					 (tokensTrans->at(2) == tokensReceived->at(2)) &&
					 (tokensTrans->at(3) == tokensReceived->at(3)) &&
					 (tokensTrans->at(4) == tokensReceived->at(4)) &&
					 (tokensTrans->at(5) == tokensReceived->at(5)) &&
					 (tokensTrans->at(6) == tokensReceived->at(6))))
				{
					if((textFileUtils.strtodouble(tokensTrans->at(7)) < 500) && 
					   (textFileUtils.strtodouble(tokensReceived->at(7)) < 500))
					{
						lineReceivedExtra = lineReceived;
						onlyTransmitted = true;
					}
					else 
					{
						std::cout << "\nTransmitted Line: " << lineTrans << std::endl;
						std::cout << "Received Line: " << lineReceived << std::endl;
						throw SPDIOException("Something has gone wrong reading the input file: the trasmitted and received origin and point are different.");
					}					
				}
				
				
				if(!onlyTransmitted)
				{
					foundTransmittedLine = false;
					if((textFileUtils.strtodouble(tokensTrans->at(7)) < 500) && 
					   (textFileUtils.strtodouble(tokensReceived->at(7)) < 500))
					{
						extralinesTrans->push_back(lineReceived);
						tokensReceived->clear();
						foundTransmittedLine = true;
					}
					while(foundTransmittedLine)
					{
						if(!lineReader.endOfFile())
						{
							lineReceived = lineReader.readLine();
							boost::trim(lineReceivedExtra);
							if(!textFileUtils.blankline(lineReceivedExtra))
							{
								textFileUtils.tokenizeString(lineReceived, ',', tokensReceived);
								
								if(!((tokensTrans->at(0) == tokensReceived->at(0)) &&
									 (tokensTrans->at(1) == tokensReceived->at(1)) &&
									 (tokensTrans->at(2) == tokensReceived->at(2)) &&
									 (tokensTrans->at(3) == tokensReceived->at(3)) &&
									 (tokensTrans->at(4) == tokensReceived->at(4)) &&
									 (tokensTrans->at(5) == tokensReceived->at(5)) &&
									 (tokensTrans->at(6) == tokensReceived->at(6))))
								{
									foundTransmittedLine = false;
									onlyTransmitted = true;
									lineReceivedExtra = lineReceived;
								}
								else 
								{
									if((textFileUtils.strtodouble(tokensTrans->at(7)) < 500) && 
									   (textFileUtils.strtodouble(tokensReceived->at(7)) < 500))
									{
										extralinesTrans->push_back(lineReceived);
										tokensReceived->clear();
										foundTransmittedLine = true;
									}
									else 
									{
										foundTransmittedLine = false;
									}
								}
							}
							else 
							{
								foundTransmittedLine = false;
							}
						}
					}
				}
				
				if(!onlyTransmitted)
				{
					foundReceivedLine = true;
					while (foundReceivedLine)
					{
						if(!lineReader.endOfFile())
						{
							lineReceivedExtra = lineReader.readLine();
							boost::trim(lineReceivedExtra);
							if(!textFileUtils.blankline(lineReceivedExtra))
							{
								textFileUtils.tokenizeString(lineReceivedExtra, ',', tokensReceivedExtra);
								
								if((tokensReceived->at(0) == tokensReceivedExtra->at(0)) &&
								   (tokensReceived->at(1) == tokensReceivedExtra->at(1)) &&
								   (tokensReceived->at(2) == tokensReceivedExtra->at(2)) &&
								   (tokensReceived->at(3) == tokensReceivedExtra->at(3)) &&
								   (tokensReceived->at(4) == tokensReceivedExtra->at(4)) &&
								   (tokensReceived->at(5) == tokensReceivedExtra->at(5)) &&
								   (tokensReceived->at(6) == tokensReceivedExtra->at(6)))
								{
									extralinesRec->push_back(lineReceivedExtra);
									foundReceivedLine = true;
								}
								else
								{
									foundReceivedLine = false;
								}
								tokensReceivedExtra->clear();
							}
						}
						else 
						{
							foundReceivedLine = false;
						}
					}
				}
				
				if(onlyTransmitted)
				{
					pulse = this->createPulse(tokensTrans, extralinesTrans);
				}
				else
				{
					pulse = this->createPulse(tokensTrans, extralinesTrans, tokensReceived, extralinesRec);
				}
				
				if(pulse != NULL)
				{
					pulse->gpsTime = textFileUtils.strtodouble(tokensTrans->at(0)) + (textFileUtils.strtodouble(tokensTrans->at(7))*0.001);
					pulse->pulseID = numPulses++;
                    pulse->receiveWaveNoiseThreshold = this->waveNoiseThreshold;
					
					if(first)
					{
						xMin = pulse->xIdx;
						xMax = pulse->xIdx;
						yMin = pulse->yIdx;
						yMax = pulse->yIdx;
						zMin = pulse->z0;
						zMax = pulse->z0;
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
						
						if(pulse->numOfReceivedBins > 0)
						{
							double tempX = 0;
							double tempY = 0;
							double tempZ = 0;
							
							SPDConvertToCartesian(pulse->zenith, pulse->azimuth, pulse->rangeToWaveformStart, pulse->x0, pulse->y0, pulse->z0, &tempX, &tempY, &tempZ);
							zMin = tempZ;
							SPDConvertToCartesian(pulse->zenith, pulse->azimuth, (pulse->rangeToWaveformStart+(((pulse->numOfReceivedBins-1)*SPD_SPEED_OF_LIGHT_NS))/2), pulse->x0, pulse->y0, pulse->z0, &tempX, &tempY, &tempZ);
							zMax = tempZ;
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
					
                    pulses->push_back(pulse);
				}
				
				tokensTrans->clear();
				tokensReceived->clear();
				tokensReceivedExtra->clear();
				extralinesRec->clear();
				extralinesTrans->clear();
			}
			spdFile->setBoundingVolume(xMin, xMax, yMin, yMax, zMin, zMax);
			spdFile->setBoundingBoxSpherical(zenithMin, zenithMax, azimuthMin, azimuthMax);
			spdFile->setSpatialReference(outputProjWKT);
			spdFile->setNumberOfPulses(numPulses);
			spdFile->setOriginDefined(SPD_TRUE);
			spdFile->setDiscretePtDefined(SPD_FALSE);
			spdFile->setDecomposedPtDefined(SPD_FALSE);
			spdFile->setTransWaveformDefined(SPD_TRUE);
            spdFile->setReceiveWaveformDefined(SPD_TRUE);
			spdFile->setTemporalBinSpacing(1);
			lineReader.closeFile();
			std::cout << "." << numPulses << ".Pulses\n";
		}
		catch (SPDIOException &e) 
		{
			std::cout << "Pulse = " << numPulses << std::endl;
			if(onlyTransmitted)
			{
				std::cout << "Only Transmitted\n";
			}
			else if(extralinesRec->size() == 0)
			{
				std::cout << "Transmitted and Single line of Recieved\n";
			}
			else
			{
				std::cout << "Extra recieved lines, an extra " <<  extralinesRec->size() << " lines\n";
			}
			
			std::cout << "Transmitted: " << lineTrans << std::endl;
			std::cout << "Recieved: " << lineReceived << std::endl;
			std::cout << "Extra Lines: " << std::endl;
			std::vector<std::string>::iterator iterLines;
			for(iterLines = extralinesRec->begin(); iterLines != extralinesRec->end(); ++iterLines)
			{
				std::cout << *iterLines << std::endl;
			}
			throw e;
		}			
		return pulses;
	}
	
	void SPDFullWaveformDatFileImporter::readAndProcessAllData(std::string inputFile, SPDFile *spdFile, SPDImporterProcessor *processor)throw(SPDIOException)
	{
		SPDTextFileUtilities textFileUtils;
		SPDTextFileLineReader lineReader;
		
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
		
		SPDPulse *pulse = NULL;
		bool firstLine = true;
		bool foundTransmittedLine = true;
		bool foundReceivedLine = true;
		bool onlyTransmitted = false;
		std::string lineTrans = "";
		std::string lineReceived = "";
		std::string lineReceivedExtra = "";
		std::vector<std::string> *extralinesRec = new std::vector<std::string>();
		std::vector<std::string> *extralinesTrans = new std::vector<std::string>();
		std::vector<std::string> *tokensTrans = new std::vector<std::string>();
		std::vector<std::string> *tokensReceived = new std::vector<std::string>();
		std::vector<std::string> *tokensReceivedExtra = new std::vector<std::string>();
		
		boost::uint_fast64_t numPulses = 0;
		try 
		{
			if(convertCoords)
			{
				this->initCoordinateSystemTransformation(spdFile);
			}
			
			lineReader.openFile(inputFile);
			std::cout << "Read ." << std::flush;
			while(!lineReader.endOfFile())
			{
				onlyTransmitted = false;
				if((numPulses % 10000) == 0)
				{
					std::cout << "." << numPulses << "." << std::flush;
				}
				
				if(firstLine)
				{
					lineTrans = lineReader.readLine();
					if(!isdigit(lineTrans.at(0)))
					{
						lineTrans = lineReader.readLine();
					}
					firstLine = false;
					boost::trim(lineTrans);
					if(textFileUtils.blankline(lineTrans))
					{
						throw SPDIOException("The first line is blank please check your file.");
					}
				}
				else 
				{
					lineTrans = lineReceivedExtra;
				}
				lineReceived = lineReader.readLine();
				boost::trim(lineReceived);
				if(textFileUtils.blankline(lineReceived))
				{
					throw SPDIOException("There is no received line due to blank line, please check you input file.");
				}
				
				textFileUtils.tokenizeString(lineTrans, ',', tokensTrans);
				textFileUtils.tokenizeString(lineReceived, ',', tokensReceived);
				
				if(!((tokensTrans->at(0) == tokensReceived->at(0)) &&
					 (tokensTrans->at(1) == tokensReceived->at(1)) &&
					 (tokensTrans->at(2) == tokensReceived->at(2)) &&
					 (tokensTrans->at(3) == tokensReceived->at(3)) &&
					 (tokensTrans->at(4) == tokensReceived->at(4)) &&
					 (tokensTrans->at(5) == tokensReceived->at(5)) &&
					 (tokensTrans->at(6) == tokensReceived->at(6))))
				{
					if((textFileUtils.strtodouble(tokensTrans->at(7)) < 500) && 
					   (textFileUtils.strtodouble(tokensReceived->at(7)) < 500))
					{
						lineReceivedExtra = lineReceived;
						onlyTransmitted = true;
					}
					else 
					{
						std::cout << "\nTransmitted Line: " << lineTrans << std::endl;
						std::cout << "Received Line: " << lineReceived << std::endl;
						throw SPDIOException("Something has gone wrong reading the input file: the trasmitted and received origin and point are different.");
					}					
				}
				
				
				if(!onlyTransmitted)
				{
					foundTransmittedLine = false;
					if((textFileUtils.strtodouble(tokensTrans->at(7)) < 500) && 
					   (textFileUtils.strtodouble(tokensReceived->at(7)) < 500))
					{
						extralinesTrans->push_back(lineReceived);
						tokensReceived->clear();
						foundTransmittedLine = true;
					}
					while(foundTransmittedLine)
					{
						if(!lineReader.endOfFile())
						{
							lineReceived = lineReader.readLine();
							boost::trim(lineReceivedExtra);
							if(!textFileUtils.blankline(lineReceivedExtra))
							{
								textFileUtils.tokenizeString(lineReceived, ',', tokensReceived);
								
								if(!((tokensTrans->at(0) == tokensReceived->at(0)) &&
									 (tokensTrans->at(1) == tokensReceived->at(1)) &&
									 (tokensTrans->at(2) == tokensReceived->at(2)) &&
									 (tokensTrans->at(3) == tokensReceived->at(3)) &&
									 (tokensTrans->at(4) == tokensReceived->at(4)) &&
									 (tokensTrans->at(5) == tokensReceived->at(5)) &&
									 (tokensTrans->at(6) == tokensReceived->at(6))))
								{
									foundTransmittedLine = false;
									onlyTransmitted = true;
									lineReceivedExtra = lineReceived;
								}
								else 
								{
									if((textFileUtils.strtodouble(tokensTrans->at(7)) < 500) && 
									   (textFileUtils.strtodouble(tokensReceived->at(7)) < 500))
									{
										extralinesTrans->push_back(lineReceived);
										tokensReceived->clear();
										foundTransmittedLine = true;
									}
									else 
									{
										foundTransmittedLine = false;
									}
								}
							}
							else 
							{
								foundTransmittedLine = false;
							}
						}
					}
				}
				
				if(!onlyTransmitted)
				{
					foundReceivedLine = true;
					while (foundReceivedLine)
					{
						if(!lineReader.endOfFile())
						{
							lineReceivedExtra = lineReader.readLine();
							boost::trim(lineReceivedExtra);
							if(!textFileUtils.blankline(lineReceivedExtra))
							{
								textFileUtils.tokenizeString(lineReceivedExtra, ',', tokensReceivedExtra);
								
								if((tokensReceived->at(0) == tokensReceivedExtra->at(0)) &&
								   (tokensReceived->at(1) == tokensReceivedExtra->at(1)) &&
								   (tokensReceived->at(2) == tokensReceivedExtra->at(2)) &&
								   (tokensReceived->at(3) == tokensReceivedExtra->at(3)) &&
								   (tokensReceived->at(4) == tokensReceivedExtra->at(4)) &&
								   (tokensReceived->at(5) == tokensReceivedExtra->at(5)) &&
								   (tokensReceived->at(6) == tokensReceivedExtra->at(6)))
								{
									extralinesRec->push_back(lineReceivedExtra);
									foundReceivedLine = true;
								}
								else
								{
									foundReceivedLine = false;
								}
								tokensReceivedExtra->clear();
							}
						}
						else 
						{
							foundReceivedLine = false;
						}
					}
				}
				
				if(onlyTransmitted)
				{
					pulse = this->createPulse(tokensTrans, extralinesTrans);
				}
				else
				{
					pulse = this->createPulse(tokensTrans, extralinesTrans, tokensReceived, extralinesRec);
				}
				
				if(pulse != NULL)
				{
					pulse->gpsTime = textFileUtils.strtodouble(tokensTrans->at(0)) + (textFileUtils.strtodouble(tokensTrans->at(7))*0.001);
					pulse->pulseID = numPulses++;
                    pulse->receiveWaveNoiseThreshold = this->waveNoiseThreshold;
					
					if(first)
					{
						xMin = pulse->xIdx;
						xMax = pulse->xIdx;
						yMin = pulse->yIdx;
						yMax = pulse->yIdx;
						zMin = pulse->z0;
						zMax = pulse->z0;
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
						
						if(pulse->numOfReceivedBins > 0)
						{
							double tempX = 0;
							double tempY = 0;
							double tempZ = 0;
							
							SPDConvertToCartesian(pulse->zenith, pulse->azimuth, pulse->rangeToWaveformStart, pulse->x0, pulse->y0, pulse->z0, &tempX, &tempY, &tempZ);
							zMin = tempZ;
							SPDConvertToCartesian(pulse->zenith, pulse->azimuth, (pulse->rangeToWaveformStart+(((pulse->numOfReceivedBins-1)*SPD_SPEED_OF_LIGHT_NS))/2), pulse->x0, pulse->y0, pulse->z0, &tempX, &tempY, &tempZ);
							zMax = tempZ;
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
					
					processor->processImportedPulse(spdFile, pulse);
				}
				
				tokensTrans->clear();
				tokensReceived->clear();
				tokensReceivedExtra->clear();
				extralinesRec->clear();
				extralinesTrans->clear();
			}
			spdFile->setBoundingVolume(xMin, xMax, yMin, yMax, zMin, zMax);
			spdFile->setBoundingBoxSpherical(zenithMin, zenithMax, azimuthMin, azimuthMax);
			spdFile->setSpatialReference(outputProjWKT);
			spdFile->setNumberOfPulses(numPulses);
			spdFile->setOriginDefined(SPD_TRUE);
			spdFile->setDiscretePtDefined(SPD_FALSE);
			spdFile->setDecomposedPtDefined(SPD_FALSE);
			spdFile->setTransWaveformDefined(SPD_TRUE);
            spdFile->setReceiveWaveformDefined(SPD_TRUE);
			spdFile->setTemporalBinSpacing(1);
			lineReader.closeFile();
			std::cout << "." << numPulses << ".Pulses\n";
		}
		catch (SPDIOException &e) 
		{
			std::cout << "Pulse = " << numPulses << std::endl;
			if(onlyTransmitted)
			{
				std::cout << "Only Transmitted\n";
			}
			else if(extralinesRec->size() == 0)
			{
				std::cout << "Transmitted and Single line of Recieved\n";
			}
			else
			{
				std::cout << "Extra recieved lines, an extra " <<  extralinesRec->size() << " lines\n";
			}
			
			std::cout << "Transmitted: " << lineTrans << std::endl;
			std::cout << "Recieved: " << lineReceived << std::endl;
			std::cout << "Extra Lines: " << std::endl;
			std::vector<std::string>::iterator iterLines;
			for(iterLines = extralinesRec->begin(); iterLines != extralinesRec->end(); ++iterLines)
			{
				std::cout << *iterLines << std::endl;
			}
			throw e;
		}		
	}
	
	bool SPDFullWaveformDatFileImporter::isFileType(std::string fileType)
	{
		if(fileType == "FWF_DAT")
		{
			return true;
		}
		return false;
	}
    
    void SPDFullWaveformDatFileImporter::readHeaderInfo(std::string, SPDFile*) throw(SPDIOException)
    {
        // No Header to Read..
    }
	
	SPDPulse* SPDFullWaveformDatFileImporter::createPulse(std::vector<std::string> *transTokens, std::vector<std::string> *extraTransLines) throw(SPDIOException)
	{
		SPDTextFileUtilities textFileUtils;
		SPDPulseUtils pulseUtils;
		SPDPulse *pulse = new SPDPulse();
		pulseUtils.initSPDPulse(pulse);
		
		try 
		{
			double originX = textFileUtils.strtodouble(transTokens->at(2));
			double originY = textFileUtils.strtodouble(transTokens->at(1));
			double originZ = textFileUtils.strtodouble(transTokens->at(3));
			if(convertCoords)
			{
				this->transformCoordinateSystem(&originX, &originY, &originZ);
			}
			
			double arbX = textFileUtils.strtodouble(transTokens->at(5));
			double arbY = textFileUtils.strtodouble(transTokens->at(4));
			double arbZ = textFileUtils.strtodouble(transTokens->at(6));
			if(convertCoords)
			{
				this->transformCoordinateSystem(&arbX, &arbY, &arbZ);
			}
			
			pulse->x0 = originX;
			pulse->y0 = originY;
			pulse->z0 = originZ;
			
			double range = 0;
			double zenith = 0;
			double azimuth = 0;
			
			SPDConvertToSpherical(originX, originY, originZ, arbX, arbY, arbZ, &zenith, &azimuth, &range);
			
			pulse->azimuth = azimuth;
			pulse->zenith = zenith;
			
			pulse->numOfTransmittedBins = transTokens->size() - 9;
			
			boost::uint_fast32_t nextSampleTime = textFileUtils.strto32bitUInt(transTokens->at(7)) + pulse->numOfTransmittedBins;
			
			std::vector<std::string>::iterator iterLines;
			std::vector<std::string> *tokensTransExtra = new std::vector<std::string>();
			bool pulseInvalid = false;
			for(iterLines = extraTransLines->begin(); iterLines != extraTransLines->end(); ++iterLines)
			{
				textFileUtils.tokenizeString(*iterLines, ',', tokensTransExtra);
				if(textFileUtils.strto32bitUInt(transTokens->at(7)) != nextSampleTime)
				{
					pulseInvalid = true;
				}
				
				pulse->numOfTransmittedBins += (tokensTransExtra->size()-9);
				nextSampleTime += (tokensTransExtra->size()-9);
				tokensTransExtra->clear();
			}
			
			if(pulseInvalid)
			{
				SPDPulseUtils::deleteSPDPulse(pulse);
				return NULL;
			}
			
			pulse->transmitted = new boost::uint_fast32_t[pulse->numOfTransmittedBins];
			boost::uint_fast16_t transIdx = 0;
			for(boost::uint_fast16_t i = 0; i < (transTokens->size() - 9); ++i)
			{
				pulse->transmitted[transIdx] = textFileUtils.strto32bitUInt(transTokens->at(i+9));
                transIdx++;
			}
						
			for(iterLines = extraTransLines->begin(); iterLines != extraTransLines->end(); ++iterLines)
			{
				textFileUtils.tokenizeString(*iterLines, ',', tokensTransExtra);
				
				for(boost::uint_fast16_t i = 0; i < (tokensTransExtra->size() - 9); ++i)
				{
					pulse->transmitted[transIdx++] = textFileUtils.strto32bitUInt(tokensTransExtra->at(i+9));
				}
				tokensTransExtra->clear();
			}
			
			float transAmp = 0;
			float transWidth = 0;
			float transPeakTime = 0;
			
			mathUtils->decomposeSingleGaussian(pulse->transmitted, pulse->numOfTransmittedBins, 5, 1, &transAmp, &transWidth, &transPeakTime);

			pulse->rangeToWaveformStart = 0;
			pulse->amplitudePulse = transAmp;
			pulse->widthPulse = transWidth;
			
			
			
			pulse->numOfReceivedBins = 0;
			pulse->received = NULL;
			
			if((indexCoords == SPD_START_OF_RECEIVED_WAVEFORM) | (indexCoords == SPD_FIRST_RETURN))
			{
				pulse->xIdx = pulse->x0;
				pulse->yIdx = pulse->y0;
			}
			else if(indexCoords == SPD_ORIGIN)
			{
				pulse->xIdx = originX;
				pulse->yIdx = originY;
			}
			else if((indexCoords == SPD_LAST_RETURN) | (indexCoords == SPD_END_OF_RECEIVED_WAVEFORM))
			{
				pulse->xIdx = pulse->x0;
				pulse->yIdx = pulse->y0;
			}
			else if(indexCoords == SPD_MAX_INTENSITY)
			{
				pulse->xIdx = originX;
				pulse->yIdx = originY;
			}
			else 
			{
				throw SPDIOException("Unknown method of calculating the index X and Y coords.");
			}
			
		}
		catch (SPDIOException &e) 
		{
			SPDPulseUtils::deleteSPDPulse(pulse);
			throw e;
		}
        catch (SPDProcessingException &e) 
		{
			SPDPulseUtils::deleteSPDPulse(pulse);
			throw SPDIOException(e.what());
		}
		catch(SPDTextFileException &e)
		{
			SPDPulseUtils::deleteSPDPulse(pulse);
			throw SPDIOException(e.what());
		}
		
		return pulse;
	}
	
	SPDPulse* SPDFullWaveformDatFileImporter::createPulse(std::vector<std::string> *transTokens, std::vector<std::string> *extraTransLines, std::vector<std::string> *receivedTokens, std::vector<std::string> *receivedExtraLines) throw(SPDIOException)
	{
		SPDTextFileUtilities textFileUtils;
		SPDPulseUtils pulseUtils;
		SPDPulse *pulse = new SPDPulse();
		pulseUtils.initSPDPulse(pulse);
		
		try 
		{
			double originX = textFileUtils.strtodouble(transTokens->at(2));
			double originY = textFileUtils.strtodouble(transTokens->at(1));
			double originZ = textFileUtils.strtodouble(transTokens->at(3));
			if(convertCoords)
			{
				this->transformCoordinateSystem(&originX, &originY, &originZ);
			}
			
			double arbX = textFileUtils.strtodouble(transTokens->at(5));
			double arbY = textFileUtils.strtodouble(transTokens->at(4));
			double arbZ = textFileUtils.strtodouble(transTokens->at(6));
			if(convertCoords)
			{
				this->transformCoordinateSystem(&arbX, &arbY, &arbZ);
			}
			
			pulse->x0 = originX;
			pulse->y0 = originY;
			pulse->z0 = originZ;
						
			double range = 0;
			double zenith = 0;
			double azimuth = 0;
			
			SPDConvertToSpherical(originX, originY, originZ, arbX, arbY, arbZ, &zenith, &azimuth, &range);
			
			pulse->azimuth = azimuth;
			pulse->zenith = zenith;
			
			pulse->numOfTransmittedBins = transTokens->size() - 9;
			boost::uint_fast32_t nextSampleTime = textFileUtils.strto32bitUInt(transTokens->at(7)) + pulse->numOfTransmittedBins;
			
			std::vector<std::string>::iterator iterLines;
			std::vector<std::string> *tokensTransExtra = new std::vector<std::string>();
			bool pulseInvalid = false;
			for(iterLines = extraTransLines->begin(); iterLines != extraTransLines->end(); ++iterLines)
			{
				textFileUtils.tokenizeString(*iterLines, ',', tokensTransExtra);
				if(textFileUtils.strto32bitUInt(transTokens->at(7)) != nextSampleTime)
				{
					pulseInvalid = true;
				}
				
				pulse->numOfTransmittedBins += (tokensTransExtra->size()-9);
				nextSampleTime += (tokensTransExtra->size()-9);
				tokensTransExtra->clear();
			}
			
			if(pulseInvalid)
			{
				SPDPulseUtils::deleteSPDPulse(pulse);
				return NULL;
			}
			
			pulse->transmitted = new boost::uint_fast32_t[pulse->numOfTransmittedBins];
			boost::uint_fast16_t transIdx = 0;
			for(boost::uint_fast16_t i = 0; i < (transTokens->size() - 9); ++i)
			{
				pulse->transmitted[transIdx] = textFileUtils.strto32bitUInt(transTokens->at(i+9));
                transIdx++;
			}
			
			for(iterLines = extraTransLines->begin(); iterLines != extraTransLines->end(); ++iterLines)
			{
				textFileUtils.tokenizeString(*iterLines, ',', tokensTransExtra);
				
				for(boost::uint_fast16_t i = 0; i < (tokensTransExtra->size() - 9); ++i)
				{
					pulse->transmitted[transIdx++] = textFileUtils.strto32bitUInt(tokensTransExtra->at(i+9));
				}
				tokensTransExtra->clear();
			}
			
			float transAmp = 0;
			float transWidth = 0;
			float transPeakTime = 0;
			
			mathUtils->decomposeSingleGaussian(pulse->transmitted, pulse->numOfTransmittedBins, 5, 1, &transAmp, &transWidth, &transPeakTime);
			
			double timeToWaveformStart = (textFileUtils.strtodouble(receivedTokens->at(7)) - (textFileUtils.strtodouble(transTokens->at(7))));
			pulse->rangeToWaveformStart = (SPD_SPEED_OF_LIGHT_NS*(timeToWaveformStart/2));
			pulse->amplitudePulse = transAmp;
			pulse->widthPulse = transWidth;
			
			std::vector<std::string> *tokensReceivedExtra = new std::vector<std::string>();
			for(iterLines = receivedExtraLines->begin(); iterLines != receivedExtraLines->end(); )
			{
				textFileUtils.tokenizeString(*iterLines, ',', tokensReceivedExtra);
				if(textFileUtils.strto16bitUInt(tokensReceivedExtra->at(8)) == 1)
				{
					iterLines = receivedExtraLines->erase(iterLines);
				}
				else 
				{
					++iterLines;
				}
				tokensReceivedExtra->clear();
			}
			
			boost::uint_fast16_t totalNumReceivedBins = receivedTokens->size() - 9;
			boost::uint_fast16_t timeGap = 0;
			boost::uint_fast16_t numBins = 0;
			for(iterLines = receivedExtraLines->begin(); iterLines != receivedExtraLines->end(); ++iterLines)
			{
				textFileUtils.tokenizeString(*iterLines, ',', tokensReceivedExtra);
				numBins = tokensReceivedExtra->size() - 9;
				timeGap = textFileUtils.strtodouble(tokensReceivedExtra->at(7)) - textFileUtils.strtodouble(receivedTokens->at(7));
				timeGap -= totalNumReceivedBins;
				totalNumReceivedBins += numBins;
				totalNumReceivedBins += timeGap;
				tokensReceivedExtra->clear();
			}
			
			pulse->numOfReceivedBins = totalNumReceivedBins;
			pulse->received = new boost::uint_fast32_t[pulse->numOfReceivedBins];
			
			boost::uint_fast16_t binIdx = 0;
			for(boost::uint_fast16_t i = 9; i < receivedTokens->size(); ++i)
			{
				pulse->received[binIdx] = textFileUtils.strto32bitUInt(receivedTokens->at(i));
                binIdx++;                
			}
			boost::uint_fast16_t tmpNumReceivedBins = receivedTokens->size() - 9;
			for(iterLines = receivedExtraLines->begin(); iterLines != receivedExtraLines->end(); ++iterLines)
			{
				textFileUtils.tokenizeString(*iterLines, ',', tokensReceivedExtra);
				numBins = tokensReceivedExtra->size() - 9;
				timeGap = textFileUtils.strtodouble(tokensReceivedExtra->at(7)) - textFileUtils.strtodouble(receivedTokens->at(7));
				timeGap -= tmpNumReceivedBins;
				tmpNumReceivedBins += numBins;
				tmpNumReceivedBins += timeGap;
				
				for(boost::uint_fast16_t i = 0; i < timeGap; ++i)
				{
					pulse->received[binIdx++] = 0;
				}
				
				for(boost::uint_fast16_t i = 9; i < tokensReceivedExtra->size(); ++i)
				{
					pulse->received[binIdx++] = textFileUtils.strto32bitUInt(tokensReceivedExtra->at(i));
				}
				tokensReceivedExtra->clear();
			}
			
			if(binIdx != pulse->numOfReceivedBins)
			{
				std::cout << "binIdx = " << binIdx << std::endl;
				std::cout << "pulse->numOfReceivedBins = " << pulse->numOfReceivedBins << std::endl;
				throw SPDIOException("The number of values read is not equal to the number expected.");
			}
			
			if((indexCoords == SPD_START_OF_RECEIVED_WAVEFORM) | (indexCoords == SPD_FIRST_RETURN))
			{
				double tempX = 0;
				double tempY = 0;
				double tempZ = 0;
				
				SPDConvertToCartesian(pulse->zenith, pulse->azimuth, pulse->rangeToWaveformStart, pulse->x0, pulse->y0, pulse->z0, &tempX, &tempY, &tempZ);
				
				pulse->xIdx = tempX;
				pulse->yIdx = tempY;
			}
			else if(indexCoords == SPD_ORIGIN)
			{
				pulse->xIdx = originX;
				pulse->yIdx = originY;
			}
			else if((indexCoords == SPD_LAST_RETURN) | (indexCoords == SPD_END_OF_RECEIVED_WAVEFORM))
			{
				double tempX = 0;
				double tempY = 0;
				double tempZ = 0;
				
				SPDConvertToCartesian(pulse->zenith, pulse->azimuth, (pulse->rangeToWaveformStart+((((float)pulse->numOfReceivedBins)/2)*SPD_SPEED_OF_LIGHT_NS)), pulse->x0, pulse->y0, pulse->z0, &tempX, &tempY, &tempZ);
				
				pulse->xIdx = tempX;
				pulse->yIdx = tempY;
			}
			else if(indexCoords == SPD_MAX_INTENSITY)
			{
				double tempX = 0;
				double tempY = 0;
				double tempZ = 0;
				
				unsigned int maxIdx = 0;
				double maxVal = 0;
				bool first = true;
				for(unsigned int i = 0; i < pulse->numOfReceivedBins; ++i)
				{
					if(first)
					{
						maxIdx = i;
						maxVal = pulse->received[i];
						first = false;
					}
					else if(pulse->received[i] > maxVal)
					{
						maxIdx = i;
						maxVal = pulse->received[i];
					}
				}
				
				SPDConvertToCartesian(pulse->zenith, pulse->azimuth, (pulse->rangeToWaveformStart+((((float)maxIdx)/2)*SPD_SPEED_OF_LIGHT_NS)), pulse->x0, pulse->y0, pulse->z0, &tempX, &tempY, &tempZ);
				
				pulse->xIdx = tempX;
				pulse->yIdx = tempY;
			}
			else 
			{
				throw SPDIOException("Unknown method of calculating the index X and Y coords.");
			}			

			delete tokensReceivedExtra;
		}
		catch (SPDIOException &e) 
		{
			SPDPulseUtils::deleteSPDPulse(pulse);
			throw e;
		}
        catch (SPDProcessingException &e) 
		{
			SPDPulseUtils::deleteSPDPulse(pulse);
			throw SPDIOException(e.what());
		}
		catch(SPDTextFileException &e)
		{
			SPDPulseUtils::deleteSPDPulse(pulse);
			throw SPDIOException(e.what());
		}
		return pulse;
	}
	

	
	SPDFullWaveformDatFileImporter::~SPDFullWaveformDatFileImporter()
	{
		delete mathUtils;
	}
}


