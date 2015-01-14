/*
 *  SPDLASFileImporter.cpp
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

#include "spd/SPDLASFileImporter.h"

namespace spdlib
{
	

	SPDLASFileImporter::SPDLASFileImporter(bool convertCoords, std::string outputProjWKT, std::string schema, boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold):SPDDataImporter(convertCoords, outputProjWKT, schema, indexCoords, defineOrigin, originX, originY, originZ, waveNoiseThreshold)
	{
		
	}
    
    SPDDataImporter* SPDLASFileImporter::getInstance(bool convertCoords, std::string outputProjWKT, std::string schema, boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold)
    {
        return new SPDLASFileImporter(convertCoords, outputProjWKT, schema, indexCoords, defineOrigin, originX, originY, originZ, waveNoiseThreshold);
    }
	
	std::list<SPDPulse*>* SPDLASFileImporter::readAllDataToList(std::string inputFile, SPDFile *spdFile)throw(SPDIOException)
	{
		SPDPulseUtils pulseUtils;
		std::list<SPDPulse*> *pulses = new std::list<SPDPulse*>();
		boost::uint_fast64_t numOfPulses = 0;
		boost::uint_fast64_t numOfPoints = 0;
		
		double xMin = 0;
		double xMax = 0;
		double yMin = 0;
		double yMax = 0;
		double zMin = 0;
		double zMax = 0;
		bool first = true;
		bool firstZ = true;
		
		classWarningGiven = false;
		
		try 
		{
			std::ifstream ifs;
			ifs.open(inputFile.c_str(), std::ios::in | std::ios::binary);
			if(ifs.is_open())
			{
				liblas::ReaderFactory lasReaderFactory;
				liblas::Reader reader = lasReaderFactory.CreateWithStream(ifs);
				liblas::Header const& header = reader.GetHeader();
				
				spdFile->setFileSignature(header.GetFileSignature());
				spdFile->setSystemIdentifier(header.GetSystemId());
				
				if(spdFile->getSpatialReference() == "")
				{
					liblas::SpatialReference const &lasSpatial = header.GetSRS();
					std::string spatialRefProjWKT = lasSpatial.GetWKT();
					spdFile->setSpatialReference(spatialRefProjWKT);
				}
				
				if(convertCoords)
				{
					this->initCoordinateSystemTransformation(spdFile);
				}
				
				boost::uint_fast64_t reportedNumOfPts = header.GetPointRecordsCount();
				boost::uint_fast64_t feedback = reportedNumOfPts/10;
				unsigned int feedbackCounter = 0;
				
				SPDPoint *spdPt = NULL;
				SPDPulse *spdPulse = NULL;
                
                boost::uint_fast16_t nPtIdx = 0;
                double x0 = 0.0;
                double y0 = 0.0;
                double z0 = 0.0;
                double x1 = 0.0;
                double y1 = 0.0;
                double z1 = 0.0;                
                double range = 0.0;
				
				std::cout << "Started (Read Data) ." << std::flush;
				while (reader.ReadNextPoint())
				{
					//std::cout << numOfPoints << std::endl;
					if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
					{
						std::cout << "." << feedbackCounter << "." << std::flush;
						feedbackCounter += 10;
					}
					
					liblas::Point const& p = reader.GetPoint();
					if(p.GetReturnNumber() <= 1)
                    {
                        spdPt = this->createSPDPoint(p);
                        ++numOfPoints;
                        
                        if(firstZ)
                        {
                            zMin = spdPt->z;
                            zMax = spdPt->z;
                            firstZ = false;
                        }
                        else
                        {
                            if(spdPt->z < zMin)
                            {
                                zMin = spdPt->z;
                            }
                            else if(spdPt->z > zMax)
                            {
                                zMax = spdPt->z;
                            }
                        }
                        
                        x0 = spdPt->x;
                        y0 = spdPt->y;
                        z0 = spdPt->z;
                            
                        spdPulse = new SPDPulse();
                        pulseUtils.initSPDPulse(spdPulse);
                        spdPulse->pulseID = numOfPulses;
                        spdPulse->numberOfReturns = p.GetNumberOfReturns();
                        if(spdPulse->numberOfReturns == 0)
                        {
                            spdPulse->numberOfReturns = 1;
                        }
                        if(spdPulse->numberOfReturns > 0)
                        {
                            spdPulse->pts->reserve(spdPulse->numberOfReturns);
                        }
                        
                        //std::cout << "Start Pulse (Num Returns = " << p.GetNumberOfReturns() << "): p.GetReturnNumber() = " << p.GetReturnNumber() << std::endl;
                        
                        if(p.GetReturnNumber() == 0)
                        {
                            nPtIdx = 1;
                        }
                        else
                        {
                            nPtIdx = 2;
                        }
                        
                        
                        spdPulse->pts->push_back(spdPt);
                        for(boost::uint_fast16_t i = 0; i < (spdPulse->numberOfReturns-1); ++i)
                        {
                            if(reader.ReadNextPoint())
                            {
                                if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
                                {
                                    std::cout << "." << feedbackCounter << "." << std::flush;
                                    feedbackCounter += 10;
                                }
                                liblas::Point const& pt = reader.GetPoint();
                                
                                //std::cout << "\tIn Pulse (Num Returns = " << pt.GetNumberOfReturns() << "): pt.GetReturnNumber() = " << pt.GetReturnNumber() << std::endl;
                                
                                if(nPtIdx != pt.GetReturnNumber())
                                {
                                    std::cerr << "Start Pulse (Num Returns = " << spdPulse->numberOfReturns << "): p.GetReturnNumber() = " << p.GetReturnNumber() << std::endl;
                                    std::cerr << "\tIn Pulse (Num Returns = " << pt.GetNumberOfReturns() << "): pt.GetReturnNumber() = " << pt.GetReturnNumber() << std::endl;
                                    std::cerr << "The return number was: " << pt.GetReturnNumber() << std::endl;
                                    std::cerr << "The next return number should have been: " << nPtIdx << std::endl;
                                    std::cerr << "WARNING: Pulse was written as incompleted pulse.\n";
                                    spdPulse->numberOfReturns = i+1;
                                    //throw SPDIOException("Error in point numbering when building pulses.");
                                    break;
                                }
                                
                                spdPt = this->createSPDPoint(pt);
                                
                                if(spdPt->z < zMin)
                                {
                                    zMin = spdPt->z;
                                }
                                else if(spdPt->z > zMax)
                                {
                                    zMax = spdPt->z;
                                }
                                
                                x1 = spdPt->x;
                                y1 = spdPt->y;
                                z1 = spdPt->z;                                
                                
                                spdPulse->pts->push_back(spdPt);
                                ++numOfPoints;
                                ++nPtIdx;
                            }
                            else
                            {
                                std::cerr << "\nWarning: The file ended unexpectedly.\n";
                                std::cerr << "Expected " << spdPulse->numberOfReturns << " but only found " << i + 1 << " returns" << std::endl;
                                spdPulse->numberOfReturns = i+1;
                                break;
                                //throw SPDIOException("Unexpected end to the file.");
                            }
                        }
                        ++numOfPulses;
                        
                        if(p.GetFlightLineEdge() == 1)
                        {
                            spdPulse->edgeFlightLineFlag = SPD_WITH_SCAN;
                        }
                        else
                        {
                            spdPulse->edgeFlightLineFlag = SPD_SCAN_END;
                        }
                        
                        if(p.GetScanDirection() == 1)
                        {
                            spdPulse->scanDirectionFlag = SPD_POSITIVE;
                        }
                        else
                        {
                            spdPulse->scanDirectionFlag = SPD_NEGATIVE;
                        }
                                                
                        if(spdPulse->numberOfReturns > 1)
                        {
                            range = std::sqrt(std::pow(x1-x0,2) + std::pow(y1-y0,2) + std::pow(z1-z0,2));
                            spdPulse->zenith = std::acos((z1-z0) / range);
                            spdPulse->azimuth = std::atan((x1-x0)/(y1-y0));
                            if(spdPulse->azimuth < 0)
                            {
                                spdPulse->azimuth = spdPulse->azimuth + M_PI * 2;
                            }
                        }
					    else
                        {
                            spdPulse->zenith = 0.0;
                            spdPulse->azimuth = 0.0;
                        }
                        spdPulse->user = p.GetScanAngleRank() + 90;
                        
                        spdPulse->sourceID = p.GetPointSourceID();
                        
                        if(indexCoords == SPD_FIRST_RETURN)
                        {
                            spdPulse->xIdx = spdPulse->pts->front()->x;
                            spdPulse->yIdx = spdPulse->pts->front()->y;
                        }
                        else if(indexCoords == SPD_LAST_RETURN)
                        {
                            spdPulse->xIdx = spdPulse->pts->back()->x;
                            spdPulse->yIdx = spdPulse->pts->back()->y;
                        }
                        else
                        {
                            throw SPDIOException("Indexing type unsupported");
                        }
                        
                        if(first)
                        {
                            xMin = spdPulse->xIdx;
                            xMax = spdPulse->xIdx;
                            yMin = spdPulse->yIdx;
                            yMax = spdPulse->yIdx;
                            first = false;
                        }
                        else
                        {
                            if(spdPulse->xIdx < xMin)
                            {
                                xMin = spdPulse->xIdx;
                            }
                            else if(spdPulse->xIdx > xMax)
                            {
                                xMax = spdPulse->xIdx;
                            }
                            
                            if(spdPulse->yIdx < yMin)
                            {
                                yMin = spdPulse->yIdx;
                            }
                            else if(spdPulse->yIdx > yMax)
                            {
                                yMax = spdPulse->yIdx;
                            }
                        }
                        
                        pulses->push_back(spdPulse);
					}
                    else
                    {
                        std::cerr << "p.GetReturnNumber() = " << p.GetReturnNumber() << std::endl;
                        std::cerr << "p.GetNumberOfReturns() = " << p.GetNumberOfReturns() << std::endl;
                        std::cerr << "Warning: Point ignored. It is the first in pulse but has a return number greater than 1.\n";
                    }
				}
				
				ifs.close();
				std::cout << ". Complete\n";
				spdFile->setBoundingVolume(xMin, xMax, yMin, yMax, zMin, zMax);
				if(convertCoords)
				{
					spdFile->setSpatialReference(outputProjWKT);
				}
				spdFile->setNumberOfPulses(numOfPulses);
				spdFile->setNumberOfPoints(numOfPoints);
				spdFile->setOriginDefined(SPD_FALSE);
				spdFile->setDiscretePtDefined(SPD_TRUE);
				spdFile->setDecomposedPtDefined(SPD_FALSE);
				spdFile->setTransWaveformDefined(SPD_FALSE);
                spdFile->setReceiveWaveformDefined(SPD_FALSE);
			}
			else 
			{
				throw SPDIOException("LAS file could not be opened.");
			}
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
	
	std::vector<SPDPulse*>* SPDLASFileImporter::readAllDataToVector(std::string inputFile, SPDFile *spdFile)throw(SPDIOException)
	{
		SPDPulseUtils pulseUtils;
		std::vector<SPDPulse*> *pulses = new std::vector<SPDPulse*>();
		boost::uint_fast64_t numOfPulses = 0;
		boost::uint_fast64_t numOfPoints = 0;
		
		double xMin = 0;
		double xMax = 0;
		double yMin = 0;
		double yMax = 0;
		double zMin = 0;
		double zMax = 0;
		bool first = true;
		bool firstZ = true;
		
		classWarningGiven = false;
		
		try 
		{
			std::ifstream ifs;
			ifs.open(inputFile.c_str(), std::ios::in | std::ios::binary);
			if(ifs.is_open())
			{
				liblas::ReaderFactory lasReaderFactory;
				liblas::Reader reader = lasReaderFactory.CreateWithStream(ifs);
				liblas::Header const& header = reader.GetHeader();
				
				spdFile->setFileSignature(header.GetFileSignature());
				spdFile->setSystemIdentifier(header.GetSystemId());
				
				if(spdFile->getSpatialReference() == "")
				{
					liblas::SpatialReference const &lasSpatial = header.GetSRS();
					std::string spatialRefProjWKT = lasSpatial.GetWKT();
					spdFile->setSpatialReference(spatialRefProjWKT);
				}
				
				if(convertCoords)
				{
					this->initCoordinateSystemTransformation(spdFile);
				}
				
				pulses->reserve(header.GetPointRecordsCount());
				
				boost::uint_fast64_t reportedNumOfPts = header.GetPointRecordsCount();
				boost::uint_fast64_t feedback = reportedNumOfPts/10;
				unsigned int feedbackCounter = 0;
				
				SPDPoint *spdPt = NULL;
				SPDPulse *spdPulse = NULL;
                
                boost::uint_fast16_t nPtIdx = 0;
                double x0 = 0.0;
                double y0 = 0.0;
                double z0 = 0.0;
                double x1 = 0.0;
                double y1 = 0.0;
                double z1 = 0.0;                
                double range = 0.0;
				
				std::cout << "Started (Read Data) ." << std::flush;
				while (reader.ReadNextPoint())
				{
					//std::cout << numOfPoints << std::endl;
					if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
					{
						std::cout << "." << feedbackCounter << "." << std::flush;
						feedbackCounter += 10;
					}
					
					liblas::Point const& p = reader.GetPoint();
					if(p.GetReturnNumber() <= 1)
                    {
                        spdPt = this->createSPDPoint(p);
                        ++numOfPoints;
                        
                        if(firstZ)
                        {
                            zMin = spdPt->z;
                            zMax = spdPt->z;
                            firstZ = false;
                        }
                        else
                        {
                            if(spdPt->z < zMin)
                            {
                                zMin = spdPt->z;
                            }
                            else if(spdPt->z > zMax)
                            {
                                zMax = spdPt->z;
                            }
                        }
                        
                        x0 = spdPt->x;
                        y0 = spdPt->y;
                        z0 = spdPt->z;
                            
                        spdPulse = new SPDPulse();
                        pulseUtils.initSPDPulse(spdPulse);
                        spdPulse->pulseID = numOfPulses;
                        spdPulse->numberOfReturns = p.GetNumberOfReturns();
                        if(spdPulse->numberOfReturns == 0)
                        {
                            spdPulse->numberOfReturns = 1;
                        }
                        if(spdPulse->numberOfReturns > 0)
                        {
                            spdPulse->pts->reserve(spdPulse->numberOfReturns);
                        }
                        
                        //std::cout << "Start Pulse (Num Returns = " << p.GetNumberOfReturns() << "): p.GetReturnNumber() = " << p.GetReturnNumber() << std::endl;
                        
                        if(p.GetReturnNumber() == 0)
                        {
                            nPtIdx = 1;
                        }
                        else
                        {
                            nPtIdx = 2;
                        }
                        
                        
                        spdPulse->pts->push_back(spdPt);
                        for(boost::uint_fast16_t i = 0; i < (spdPulse->numberOfReturns-1); ++i)
                        {
                            if(reader.ReadNextPoint())
                            {
                                if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
                                {
                                    std::cout << "." << feedbackCounter << "." << std::flush;
                                    feedbackCounter += 10;
                                }
                                liblas::Point const& pt = reader.GetPoint();
                                
                                //std::cout << "\tIn Pulse (Num Returns = " << pt.GetNumberOfReturns() << "): pt.GetReturnNumber() = " << pt.GetReturnNumber() << std::endl;
                                
                                if(nPtIdx != pt.GetReturnNumber())
                                {
                                    std::cerr << "Start Pulse (Num Returns = " << spdPulse->numberOfReturns << "): p.GetReturnNumber() = " << p.GetReturnNumber() << std::endl;
                                    std::cerr << "\tIn Pulse (Num Returns = " << pt.GetNumberOfReturns() << "): pt.GetReturnNumber() = " << pt.GetReturnNumber() << std::endl;
                                    std::cerr << "The return number was: " << pt.GetReturnNumber() << std::endl;
                                    std::cerr << "The next return number should have been: " << nPtIdx << std::endl;
                                    std::cerr << "WARNING: Pulse was written as incompleted pulse.\n";
                                    spdPulse->numberOfReturns = i+1;
                                    //throw SPDIOException("Error in point numbering when building pulses.");
                                    break;
                                }
                                
                                spdPt = this->createSPDPoint(pt);
                                
                                if(spdPt->z < zMin)
                                {
                                    zMin = spdPt->z;
                                }
                                else if(spdPt->z > zMax)
                                {
                                    zMax = spdPt->z;
                                }
                        
                                x1 = spdPt->x;
                                y1 = spdPt->y;
                                z1 = spdPt->z;
                                                        
                                spdPulse->pts->push_back(spdPt);
                                ++numOfPoints;
                                ++nPtIdx;
                            }
                            else
                            {
                                std::cerr << "\nWarning: The file ended unexpectedly.\n";
                                std::cerr << "Expected " << spdPulse->numberOfReturns << " but only found " << i + 1 << " returns" << std::endl;
                                spdPulse->numberOfReturns = i+1;
                                break;
                                //throw SPDIOException("Unexpected end to the file.");
                            }
                        }
                        ++numOfPulses;
                        
                        if(p.GetFlightLineEdge() == 1)
                        {
                            spdPulse->edgeFlightLineFlag = SPD_WITH_SCAN;
                        }
                        else
                        {
                            spdPulse->edgeFlightLineFlag = SPD_SCAN_END;
                        }
                        
                        if(p.GetScanDirection() == 1)
                        {
                            spdPulse->scanDirectionFlag = SPD_POSITIVE;
                        }
                        else
                        {
                            spdPulse->scanDirectionFlag = SPD_NEGATIVE;
                        }
                        
                        if(spdPulse->numberOfReturns > 1)
                        {
                            range = std::sqrt(std::pow(x1-x0,2) + std::pow(y1-y0,2) + std::pow(z1-z0,2));
                            spdPulse->zenith = std::acos((z1-z0) / range);
                            spdPulse->azimuth = std::atan((x1-x0)/(y1-y0));
                            if(spdPulse->azimuth < 0)
                            {
                                spdPulse->azimuth = spdPulse->azimuth + M_PI * 2;
                            }                            
                        }
					    else
                        {
                            spdPulse->zenith = 0.0;
                            spdPulse->azimuth = 0.0;
                        }
                        spdPulse->user = p.GetScanAngleRank() + 90;
                                                
                        spdPulse->sourceID = p.GetPointSourceID();
                        
                        if(indexCoords == SPD_FIRST_RETURN)
                        {
                            spdPulse->xIdx = spdPulse->pts->front()->x;
                            spdPulse->yIdx = spdPulse->pts->front()->y;
                        }
                        else if(indexCoords == SPD_LAST_RETURN)
                        {
                            spdPulse->xIdx = spdPulse->pts->back()->x;
                            spdPulse->yIdx = spdPulse->pts->back()->y;
                        }
                        else
                        {
                            throw SPDIOException("Indexing type unsupported");
                        }
                        
                        if(first)
                        {
                            xMin = spdPulse->xIdx;
                            xMax = spdPulse->xIdx;
                            yMin = spdPulse->yIdx;
                            yMax = spdPulse->yIdx;
                            first = false;
                        }
                        else
                        {
                            if(spdPulse->xIdx < xMin)
                            {
                                xMin = spdPulse->xIdx;
                            }
                            else if(spdPulse->xIdx > xMax)
                            {
                                xMax = spdPulse->xIdx;
                            }
                            
                            if(spdPulse->yIdx < yMin)
                            {
                                yMin = spdPulse->yIdx;
                            }
                            else if(spdPulse->yIdx > yMax)
                            {
                                yMax = spdPulse->yIdx;
                            }
                        }
                        
                        pulses->push_back(spdPulse);
					}
                    else
                    {
                        std::cerr << "p.GetReturnNumber() = " << p.GetReturnNumber() << std::endl;
                        std::cerr << "p.GetNumberOfReturns() = " << p.GetNumberOfReturns() << std::endl;
                        std::cerr << "Warning: Point ignored. It is the first in pulse but has a return number greater than 1.\n";
                    }					
				}
				
				ifs.close();
				std::cout << ". Complete\n";
				spdFile->setBoundingVolume(xMin, xMax, yMin, yMax, zMin, zMax);
				if(convertCoords)
				{
					spdFile->setSpatialReference(outputProjWKT);
				}
				spdFile->setNumberOfPulses(numOfPulses);
				spdFile->setNumberOfPoints(numOfPoints);
				spdFile->setOriginDefined(SPD_FALSE);
				spdFile->setDiscretePtDefined(SPD_TRUE);
				spdFile->setDecomposedPtDefined(SPD_FALSE);
				spdFile->setTransWaveformDefined(SPD_FALSE);
                spdFile->setReceiveWaveformDefined(SPD_FALSE);
			}
			else 
			{
				throw SPDIOException("LAS file could not be opened.");
			}
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
	
	void SPDLASFileImporter::readAndProcessAllData(std::string inputFile, SPDFile *spdFile, SPDImporterProcessor *processor)throw(SPDIOException)
	{
		SPDPulseUtils pulseUtils;
		boost::uint_fast64_t numOfPulses = 0;
		boost::uint_fast64_t numOfPoints = 0;
		
		double xMin = 0;
		double xMax = 0;
		double yMin = 0;
		double yMax = 0;
		double zMin = 0;
		double zMax = 0;
		bool first = true;
		bool firstZ = true;
		
		classWarningGiven = false;
		
		try 
		{
			std::ifstream ifs;
			ifs.open(inputFile.c_str(), std::ios::in | std::ios::binary);
			if(ifs.is_open())
			{
				liblas::ReaderFactory lasReaderFactory;
				liblas::Reader reader = lasReaderFactory.CreateWithStream(ifs);
				liblas::Header const& header = reader.GetHeader();
				
				spdFile->setFileSignature(header.GetFileSignature());
				spdFile->setSystemIdentifier(header.GetSystemId());
				
				if(spdFile->getSpatialReference() == "")
				{
					liblas::SpatialReference const &lasSpatial = header.GetSRS();
					std::string spatialRefProjWKT = lasSpatial.GetWKT();
					spdFile->setSpatialReference(spatialRefProjWKT);
				}
				
				if(convertCoords)
				{
					this->initCoordinateSystemTransformation(spdFile);
				}
				
				boost::uint_fast64_t reportedNumOfPts = header.GetPointRecordsCount();
				boost::uint_fast64_t feedback = reportedNumOfPts/10;
				unsigned int feedbackCounter = 0;
				
				SPDPoint *spdPt = NULL;
				SPDPulse *spdPulse = NULL;
                
                boost::uint_fast16_t nPtIdx = 0;
                double x0 = 0.0;
                double y0 = 0.0;
                double z0 = 0.0;
                double x1 = 0.0;
                double y1 = 0.0;
                double z1 = 0.0;                
                double range = 0.0;
                				
				std::cout << "Started (Read Data) ." << std::flush;
				while (reader.ReadNextPoint())
				{
					if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
					{
						std::cout << "." << feedbackCounter << "." << std::flush;
						feedbackCounter += 10;
					}
					
					liblas::Point const& p = reader.GetPoint();
					if(p.GetReturnNumber() <= 1)
                    {
                        spdPt = this->createSPDPoint(p);
                        ++numOfPoints;
                        
                        if(firstZ)
                        {
                            zMin = spdPt->z;
                            zMax = spdPt->z;
                            firstZ = false;
                        }
                        else
                        {
                            if(spdPt->z < zMin)
                            {
                                zMin = spdPt->z;
                            }
                            else if(spdPt->z > zMax)
                            {
                                zMax = spdPt->z;
                            }
                        }
                        
                        x0 = spdPt->x;
                        y0 = spdPt->y;
                        z0 = spdPt->z;
                        
                        spdPulse = new SPDPulse();
                        pulseUtils.initSPDPulse(spdPulse);
                        spdPulse->pulseID = numOfPulses;
                        spdPulse->numberOfReturns = p.GetNumberOfReturns();
                        if(spdPulse->numberOfReturns == 0)
                        {
                            spdPulse->numberOfReturns = 1;
                        }
                        if(spdPulse->numberOfReturns > 0)
                        {
                            spdPulse->pts->reserve(spdPulse->numberOfReturns);
                        }
                        
                        //std::cout << "Start Pulse (Num Returns = " << p.GetNumberOfReturns() << "): p.GetReturnNumber() = " << p.GetReturnNumber() << std::endl;
                        
                        if(p.GetReturnNumber() == 0)
                        {
                            nPtIdx = 1;
                        }
                        else
                        {
                            nPtIdx = 2;
                        }
                        
                        
                        spdPulse->pts->push_back(spdPt);
                        for(boost::uint_fast16_t i = 0; i < (spdPulse->numberOfReturns-1); ++i)
                        {
                            if(reader.ReadNextPoint())
                            {
                                if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
                                {
                                    std::cout << "." << feedbackCounter << "." << std::flush;
                                    feedbackCounter += 10;
                                }
                                liblas::Point const& pt = reader.GetPoint();
                                
                                //std::cout << "\tIn Pulse (Num Returns = " << pt.GetNumberOfReturns() << "): pt.GetReturnNumber() = " << pt.GetReturnNumber() << std::endl;
                                
                                if(nPtIdx != pt.GetReturnNumber())
                                {
                                    std::cerr << "Start Pulse (Num Returns = " << spdPulse->numberOfReturns << "): p.GetReturnNumber() = " << p.GetReturnNumber() << std::endl;
                                    std::cerr << "\tIn Pulse (Num Returns = " << pt.GetNumberOfReturns() << "): pt.GetReturnNumber() = " << pt.GetReturnNumber() << std::endl;
                                    std::cerr << "The return number was: " << pt.GetReturnNumber() << std::endl;
                                    std::cerr << "The next return number should have been: " << nPtIdx << std::endl;
                                    std::cerr << "WARNING: Pulse was written as incompleted pulse.\n";
                                    spdPulse->numberOfReturns = i+1;
                                    //throw SPDIOException("Error in point numbering when building pulses.");
                                    break;
                                }
                                
                                spdPt = this->createSPDPoint(pt);
                                
                                if(spdPt->z < zMin)
                                {
                                    zMin = spdPt->z;
                                }
                                else if(spdPt->z > zMax)
                                {
                                    zMax = spdPt->z;
                                }
                                
                                x1 = spdPt->x;
                                y1 = spdPt->y;
                                z1 = spdPt->z;
                                
                                spdPulse->pts->push_back(spdPt);
                                ++numOfPoints;
                                ++nPtIdx;
                            }
                            else
                            {
                                std::cerr << "\nWarning: The file ended unexpectedly.\n";
                                std::cerr << "Expected " << spdPulse->numberOfReturns << " but only found " << i + 1 << " returns" << std::endl;
                                spdPulse->numberOfReturns = i+1;
                                break;
                                //throw SPDIOException("Unexpected end to the file.");
                            }
                        }
                        ++numOfPulses;
                        
                        if(p.GetFlightLineEdge() == 1)
                        {
                            spdPulse->edgeFlightLineFlag = SPD_WITH_SCAN;
                        }
                        else 
                        {
                            spdPulse->edgeFlightLineFlag = SPD_SCAN_END;
                        }
                        
                        if(p.GetScanDirection() == 1)
                        {
                            spdPulse->scanDirectionFlag = SPD_POSITIVE;
                        }
                        else 
                        {
                            spdPulse->scanDirectionFlag = SPD_NEGATIVE;
                        }
                        
                        if(spdPulse->numberOfReturns > 1)
                        {
                            range = std::sqrt(std::pow(x1-x0,2) + std::pow(y1-y0,2) + std::pow(z1-z0,2));
                            spdPulse->zenith = std::acos((z1-z0) / range);
                            spdPulse->azimuth = std::atan((x1-x0)/(y1-y0));
                            if(spdPulse->azimuth < 0)
                            {
                                spdPulse->azimuth = spdPulse->azimuth + M_PI * 2;
                            }                            
                        }
					    else
                        {
                            spdPulse->zenith = 0.0;
                            spdPulse->azimuth = 0.0;
                        }
                        spdPulse->user = p.GetScanAngleRank() + 90;                        
                        
                        spdPulse->sourceID = p.GetPointSourceID();
                        
                        if(indexCoords == SPD_FIRST_RETURN)
                        {
                            spdPulse->xIdx = spdPulse->pts->front()->x;
                            spdPulse->yIdx = spdPulse->pts->front()->y;
                        }
                        else if(indexCoords == SPD_LAST_RETURN)
                        {
                            spdPulse->xIdx = spdPulse->pts->back()->x;
                            spdPulse->yIdx = spdPulse->pts->back()->y;
                        }
                        else
                        {
                            throw SPDIOException("Indexing type unsupported");
                        }
                        
                        if(first)
                        {
                            xMin = spdPulse->xIdx;
                            xMax = spdPulse->xIdx;
                            yMin = spdPulse->yIdx;
                            yMax = spdPulse->yIdx;
                            first = false;
                        }
                        else
                        {
                            if(spdPulse->xIdx < xMin)
                            {
                                xMin = spdPulse->xIdx;
                            }
                            else if(spdPulse->xIdx > xMax)
                            {
                                xMax = spdPulse->xIdx;
                            }
                            
                            if(spdPulse->yIdx < yMin)
                            {
                                yMin = spdPulse->yIdx;
                            }
                            else if(spdPulse->yIdx > yMax)
                            {
                                yMax = spdPulse->yIdx;
                            }
                        }
                        
                        processor->processImportedPulse(spdFile, spdPulse);
					}
                    else
                    {
                        std::cerr << "p.GetReturnNumber() = " << p.GetReturnNumber() << std::endl;
                        std::cerr << "p.GetNumberOfReturns() = " << p.GetNumberOfReturns() << std::endl;
                        std::cerr << "Warning: Point ignored. It is the first in pulse but has a return number greater than 1.\n";
                    }
					
				}
				
				ifs.close();
				std::cout << ". Complete\n";
				spdFile->setBoundingVolume(xMin, xMax, yMin, yMax, zMin, zMax);
				if(convertCoords)
				{
					spdFile->setSpatialReference(outputProjWKT);
				}
				spdFile->setNumberOfPulses(numOfPulses);
				spdFile->setNumberOfPoints(numOfPoints);
				spdFile->setOriginDefined(SPD_FALSE);
				spdFile->setDiscretePtDefined(SPD_TRUE);
				spdFile->setDecomposedPtDefined(SPD_FALSE);
				spdFile->setTransWaveformDefined(SPD_FALSE);
                spdFile->setReceiveWaveformDefined(SPD_FALSE);
			}
			else 
			{
				throw SPDIOException("LAS file could not be opened.");
			}
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
	
	bool SPDLASFileImporter::isFileType(std::string fileType)
	{
		if(fileType == "LAS")
		{
			return true;
		}
		return false;
	}
	
    void SPDLASFileImporter::readHeaderInfo(std::string, SPDFile*) throw(SPDIOException)
    {
        // No Header to Read..
    }
    
	SPDPoint* SPDLASFileImporter::createSPDPoint(liblas::Point const& pt)throw(SPDIOException)
	{
		try 
		{
			SPDPointUtils spdPtUtils;
			SPDPoint *spdPt = new SPDPoint();
			spdPtUtils.initSPDPoint(spdPt);
			double x = pt.GetX();
			double y = pt.GetY();
			double z = pt.GetZ();
			
			if(convertCoords)
			{
				this->transformCoordinateSystem(&x, &y, &z);
			}
			
			spdPt->x = x;
			spdPt->y = y;
			spdPt->z = z;
			spdPt->amplitudeReturn = pt.GetIntensity();
			spdPt->user = pt.GetUserData();
			
			liblas::Classification lasClass = pt.GetClassification();
			
			if(lasClass.GetClass() == pt.eCreated)
			{
				spdPt->classification = SPD_CREATED;
			}
			else if(lasClass.GetClass() == pt.eUnclassified)
			{
				spdPt->classification = SPD_UNCLASSIFIED;
			}
			else if(lasClass.GetClass() == pt.eGround)
			{
				spdPt->classification = SPD_GROUND;
			}
			else if(lasClass.GetClass() == pt.eLowVegetation)
			{
				spdPt->classification = SPD_LOW_VEGETATION;
			}
			else if(lasClass.GetClass() == pt.eMediumVegetation)
			{
				spdPt->classification = SPD_MEDIUM_VEGETATION;
			}
			else if(lasClass.GetClass() == pt.eHighVegetation)
			{
				spdPt->classification = SPD_HIGH_VEGETATION;
			}
			else if(lasClass.GetClass() == pt.eBuilding)
			{
				spdPt->classification = SPD_BUILDING;
			}
			else if(lasClass.GetClass() == pt.eLowPoint)
			{
				spdPt->classification = SPD_CREATED;
				spdPt->lowPoint = SPD_TRUE;
			}
			else if(lasClass.GetClass() == pt.eModelKeyPoint)
			{
				spdPt->classification = SPD_CREATED;
				spdPt->modelKeyPoint = SPD_TRUE;
			}
			else if(lasClass.GetClass() == pt.eWater)
			{
				spdPt->classification = SPD_WATER;
			}
			else if(lasClass.GetClass() == pt.eOverlapPoints)
			{
				spdPt->classification = SPD_CREATED;
				spdPt->overlap = SPD_TRUE;
			}
			else
			{
				spdPt->classification = SPD_CREATED;
				if(!classWarningGiven)
				{
					std::cerr << "WARNING: The class ID was not recognised - check the classes points were allocated too.";
					classWarningGiven = true;
				}
			}
			
			liblas::Color const &lasColor = pt.GetColor();
			spdPt->red = lasColor.GetRed();
			spdPt->green = lasColor.GetGreen();
			spdPt->blue = lasColor.GetBlue();
			
			spdPt->returnID = pt.GetReturnNumber();
			spdPt->gpsTime = pt.GetTime();
			
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
	
	SPDLASFileImporter::~SPDLASFileImporter()
	{
		
	}
    
    
    
    
    
    
    
    
    
    
    
    SPDLASFileImporterStrictPulses::SPDLASFileImporterStrictPulses(bool convertCoords, std::string outputProjWKT, std::string schema, boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold):SPDDataImporter(convertCoords, outputProjWKT, schema, indexCoords, defineOrigin, originX, originY, originZ, waveNoiseThreshold)
	{
		
	}
    
    SPDDataImporter* SPDLASFileImporterStrictPulses::getInstance(bool convertCoords, std::string outputProjWKT, std::string schema, boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold)
    {
        return new SPDLASFileImporterStrictPulses(convertCoords, outputProjWKT, schema, indexCoords, defineOrigin, originX, originY, originZ, waveNoiseThreshold);
    }
	
	std::list<SPDPulse*>* SPDLASFileImporterStrictPulses::readAllDataToList(std::string inputFile, SPDFile *spdFile)throw(SPDIOException)
	{
		SPDPulseUtils pulseUtils;
		std::list<SPDPulse*> *pulses = new std::list<SPDPulse*>();
		boost::uint_fast64_t numOfPulses = 0;
		boost::uint_fast64_t numOfPoints = 0;
		
		double xMin = 0;
		double xMax = 0;
		double yMin = 0;
		double yMax = 0;
		double zMin = 0;
		double zMax = 0;
		bool first = true;
		bool firstZ = true;
		
		classWarningGiven = false;
		
		try
		{
			std::ifstream ifs;
			ifs.open(inputFile.c_str(), std::ios::in | std::ios::binary);
			if(ifs.is_open())
			{
				liblas::ReaderFactory lasReaderFactory;
				liblas::Reader reader = lasReaderFactory.CreateWithStream(ifs);
				liblas::Header const& header = reader.GetHeader();
				
				spdFile->setFileSignature(header.GetFileSignature());
				spdFile->setSystemIdentifier(header.GetSystemId());
				
				if(spdFile->getSpatialReference() == "")
				{
					liblas::SpatialReference const &lasSpatial = header.GetSRS();
					std::string spatialRefProjWKT = lasSpatial.GetWKT();
					spdFile->setSpatialReference(spatialRefProjWKT);
				}
				
				if(convertCoords)
				{
					this->initCoordinateSystemTransformation(spdFile);
				}
				
				boost::uint_fast64_t reportedNumOfPts = header.GetPointRecordsCount();
				boost::uint_fast64_t feedback = reportedNumOfPts/10;
				unsigned int feedbackCounter = 0;
				
				SPDPoint *spdPt = NULL;
				SPDPulse *spdPulse = NULL;
                
                boost::uint_fast16_t nPtIdx = 0;
                double x0 = 0.0;
                double y0 = 0.0;
                double z0 = 0.0;
                double x1 = 0.0;
                double y1 = 0.0;
                double z1 = 0.0;                
                double range = 0.0;
				
				std::cout << "Started (Read Data) ." << std::flush;
				while (reader.ReadNextPoint())
				{
					//std::cout << numOfPoints << std::endl;
					if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
					{
						std::cout << "." << feedbackCounter << "." << std::flush;
						feedbackCounter += 10;
					}
					
					liblas::Point const& p = reader.GetPoint();
					spdPt = this->createSPDPoint(p);
					++numOfPoints;
					
					if(firstZ)
					{
						zMin = spdPt->z;
						zMax = spdPt->z;
						firstZ = false;
					}
					else
					{
						if(spdPt->z < zMin)
						{
							zMin = spdPt->z;
						}
						else if(spdPt->z > zMax)
						{
							zMax = spdPt->z;
						}
					}
					
                    x0 = spdPt->x;
                    y0 = spdPt->y;
                    z0 = spdPt->z;
                    
					spdPulse = new SPDPulse();
					pulseUtils.initSPDPulse(spdPulse);
                    spdPulse->pulseID = numOfPulses;
					//std::cout << "Pulse size " << sizeof(SPDPulse) << std::endl;
					//std::cout << "Point size " << sizeof(SPDPoint) << std::endl;
					//std::cout << "Points capacity (1) " << spdPulse->pts.capacity() << std::endl;
					spdPulse->numberOfReturns = p.GetNumberOfReturns();
                    if(spdPulse->numberOfReturns == 0)
                    {
                        spdPulse->numberOfReturns = 1;
                    }
                    if(spdPulse->numberOfReturns > 0)
                    {
                        spdPulse->pts->reserve(spdPulse->numberOfReturns);
                    }
                    
                    //std::cout << "Start Pulse (Num Returns = " << p.GetNumberOfReturns() << "): p.GetReturnNumber() = " << p.GetReturnNumber() << std::endl;
                    
                    if(p.GetReturnNumber() == 0)
                    {
                        nPtIdx = 1;
                    }
                    else
                    {
                        nPtIdx = 2;
                    }
                    
					spdPulse->pts->push_back(spdPt);
					for(boost::uint_fast16_t i = 0; i < (spdPulse->numberOfReturns-1); ++i)
					{
						if(reader.ReadNextPoint())
						{
							if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
							{
								std::cout << "." << feedbackCounter << "." << std::flush;
								feedbackCounter += 10;
							}
							liblas::Point const& pt = reader.GetPoint();
                            
                            //std::cout << "\tIn Pulse (Num Returns = " << pt.GetNumberOfReturns() << "): pt.GetReturnNumber() = " << pt.GetReturnNumber() << std::endl;
                            
                            if(nPtIdx != pt.GetReturnNumber())
                            {
                                std::cerr << "Start Pulse (Num Returns = " << p.GetNumberOfReturns() << "): p.GetReturnNumber() = " << p.GetReturnNumber() << std::endl;
                                std::cerr << "\tIn Pulse (Num Returns = " << pt.GetNumberOfReturns() << "): pt.GetReturnNumber() = " << pt.GetReturnNumber() << std::endl;
                                std::cerr << "The return number was: " << pt.GetReturnNumber() << std::endl;
                                std::cerr << "The next return number should have been: " << nPtIdx << std::endl;
                                throw SPDIOException("Error in point numbering when building pulses.");
                            }
                            
							spdPt = this->createSPDPoint(pt);
							
							if(spdPt->z < zMin)
							{
								zMin = spdPt->z;
							}
							else if(spdPt->z > zMax)
							{
								zMax = spdPt->z;
							}
                            
                            x1 = spdPt->x;
                            y1 = spdPt->y;
                            z1 = spdPt->z;                            
							
							spdPulse->pts->push_back(spdPt);
							++numOfPoints;
						}
						else
						{
                            std::cerr << "\nWarning: The file ended unexpectedly.\n";
                            std::cerr << "Expected " << spdPulse->numberOfReturns << " but only found " << i + 1 << " returns" << std::endl;
                            spdPulse->numberOfReturns = i+1;
							throw SPDIOException("Unexpected end to the file.");
						}
					}
					++numOfPulses;
					//std::cout << "Points capacity (2) " << spdPulse->pts.capacity() << std::endl << std::endl;
					if(p.GetFlightLineEdge() == 1)
					{
						spdPulse->edgeFlightLineFlag = SPD_WITH_SCAN;
					}
					else
					{
						spdPulse->edgeFlightLineFlag = SPD_SCAN_END;
					}
					
					if(p.GetScanDirection() == 1)
					{
						spdPulse->scanDirectionFlag = SPD_POSITIVE;
					}
					else
					{
						spdPulse->scanDirectionFlag = SPD_NEGATIVE;
					}
                    
                    if(p.GetNumberOfReturns() > 1)
                    {
                        range = std::sqrt(std::pow(x1-x0,2) + std::pow(y1-y0,2) + std::pow(z1-z0,2));
                        spdPulse->zenith = std::acos((z1-z0) / range);
                        spdPulse->azimuth = std::atan((x1-x0)/(y1-y0));
                        if(spdPulse->azimuth < 0)
                        {
                            spdPulse->azimuth = spdPulse->azimuth + M_PI * 2;
                        }                        
                    }
					else
                    {
                        spdPulse->zenith = 0.0;
                        spdPulse->azimuth = 0.0;
                    }
                    spdPulse->user = p.GetScanAngleRank() + 90;
                    
					spdPulse->sourceID = p.GetPointSourceID();
					
					if(indexCoords == SPD_FIRST_RETURN)
					{
						spdPulse->xIdx = spdPulse->pts->front()->x;
						spdPulse->yIdx = spdPulse->pts->front()->y;
					}
					else if(indexCoords == SPD_LAST_RETURN)
					{
						spdPulse->xIdx = spdPulse->pts->back()->x;
						spdPulse->yIdx = spdPulse->pts->back()->y;
					}
					else
					{
						throw SPDIOException("Indexing type unsupported");
					}
					
					if(first)
					{
						xMin = spdPulse->xIdx;
						xMax = spdPulse->xIdx;
						yMin = spdPulse->yIdx;
						yMax = spdPulse->yIdx;
						first = false;
					}
					else
					{
						if(spdPulse->xIdx < xMin)
						{
							xMin = spdPulse->xIdx;
						}
						else if(spdPulse->xIdx > xMax)
						{
							xMax = spdPulse->xIdx;
						}
						
						if(spdPulse->yIdx < yMin)
						{
							yMin = spdPulse->yIdx;
						}
						else if(spdPulse->yIdx > yMax)
						{
							yMax = spdPulse->yIdx;
						}
					}
					
					pulses->push_back(spdPulse);
				}
				
				ifs.close();
				std::cout << ". Complete\n";
				spdFile->setBoundingVolume(xMin, xMax, yMin, yMax, zMin, zMax);
				if(convertCoords)
				{
					spdFile->setSpatialReference(outputProjWKT);
				}
				spdFile->setNumberOfPulses(numOfPulses);
				spdFile->setNumberOfPoints(numOfPoints);
				spdFile->setOriginDefined(SPD_FALSE);
				spdFile->setDiscretePtDefined(SPD_TRUE);
				spdFile->setDecomposedPtDefined(SPD_FALSE);
				spdFile->setTransWaveformDefined(SPD_FALSE);
                spdFile->setReceiveWaveformDefined(SPD_FALSE);
			}
			else
			{
				throw SPDIOException("LAS file could not be opened.");
			}
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
	
	std::vector<SPDPulse*>* SPDLASFileImporterStrictPulses::readAllDataToVector(std::string inputFile, SPDFile *spdFile)throw(SPDIOException)
	{
		SPDPulseUtils pulseUtils;
		std::vector<SPDPulse*> *pulses = new std::vector<SPDPulse*>();
		boost::uint_fast64_t numOfPulses = 0;
		boost::uint_fast64_t numOfPoints = 0;
		
		double xMin = 0;
		double xMax = 0;
		double yMin = 0;
		double yMax = 0;
		double zMin = 0;
		double zMax = 0;
		bool first = true;
		bool firstZ = true;
		
		classWarningGiven = false;
		
		try
		{
			std::ifstream ifs;
			ifs.open(inputFile.c_str(), std::ios::in | std::ios::binary);
			if(ifs.is_open())
			{
				liblas::ReaderFactory lasReaderFactory;
				liblas::Reader reader = lasReaderFactory.CreateWithStream(ifs);
				liblas::Header const& header = reader.GetHeader();
				
				spdFile->setFileSignature(header.GetFileSignature());
				spdFile->setSystemIdentifier(header.GetSystemId());
				
				if(spdFile->getSpatialReference() == "")
				{
					liblas::SpatialReference const &lasSpatial = header.GetSRS();
					std::string spatialRefProjWKT = lasSpatial.GetWKT();
					spdFile->setSpatialReference(spatialRefProjWKT);
				}
				
				if(convertCoords)
				{
					this->initCoordinateSystemTransformation(spdFile);
				}
				
				pulses->reserve(header.GetPointRecordsCount());
				
				boost::uint_fast64_t reportedNumOfPts = header.GetPointRecordsCount();
				boost::uint_fast64_t feedback = reportedNumOfPts/10;
				unsigned int feedbackCounter = 0;
				
				SPDPoint *spdPt = NULL;
				SPDPulse *spdPulse = NULL;
                
                boost::uint_fast16_t nPtIdx = 0;
                double x0 = 0.0;
                double y0 = 0.0;
                double z0 = 0.0;
                double x1 = 0.0;
                double y1 = 0.0;
                double z1 = 0.0;                
                double range = 0.0;
				
				std::cout << "Started (Read Data) ." << std::flush;
				while (reader.ReadNextPoint())
				{
					//std::cout << numOfPoints << std::endl;
					if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
					{
						std::cout << "." << feedbackCounter << "." << std::flush;
						feedbackCounter += 10;
					}
					
					liblas::Point const& p = reader.GetPoint();
					spdPt = this->createSPDPoint(p);
					++numOfPoints;
					
					if(firstZ)
					{
						zMin = spdPt->z;
						zMax = spdPt->z;
						firstZ = false;
					}
					else
					{
						if(spdPt->z < zMin)
						{
							zMin = spdPt->z;
						}
						else if(spdPt->z > zMax)
						{
							zMax = spdPt->z;
						}
					}
                    
                    x0 = spdPt->x;
                    y0 = spdPt->y;
                    z0 = spdPt->z;
					
					spdPulse = new SPDPulse();
					pulseUtils.initSPDPulse(spdPulse);
                    spdPulse->pulseID = numOfPulses;
					//std::cout << "Pulse size " << sizeof(SPDPulse) << std::endl;
					//std::cout << "Point size " << sizeof(SPDPoint) << std::endl;
					//std::cout << "Points capacity (1) " << spdPulse->pts.capacity() << std::endl;
					spdPulse->numberOfReturns = p.GetNumberOfReturns();
                    if(spdPulse->numberOfReturns == 0)
                    {
                        spdPulse->numberOfReturns = 1;
                    }
                    if(spdPulse->numberOfReturns > 0)
                    {
                        spdPulse->pts->reserve(spdPulse->numberOfReturns);
                    }
                    
                    //std::cout << "Start Pulse (Num Returns = " << p.GetNumberOfReturns() << "): p.GetReturnNumber() = " << p.GetReturnNumber() << std::endl;
                    
                    if(p.GetReturnNumber() == 0)
                    {
                        nPtIdx = 1;
                    }
                    else
                    {
                        nPtIdx = 2;
                    }
                    
					spdPulse->pts->push_back(spdPt);
					for(boost::uint_fast16_t i = 0; i < (spdPulse->numberOfReturns-1); ++i)
					{
						if(reader.ReadNextPoint())
						{
							if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
							{
								std::cout << "." << feedbackCounter << "." << std::flush;
								feedbackCounter += 10;
							}
							liblas::Point const& pt = reader.GetPoint();
                            
                            //std::cout << "\tIn Pulse (Num Returns = " << pt.GetNumberOfReturns() << "): pt.GetReturnNumber() = " << pt.GetReturnNumber() << std::endl;
                            
                            if(nPtIdx != pt.GetReturnNumber())
                            {
                                std::cerr << "Start Pulse (Num Returns = " << p.GetNumberOfReturns() << "): p.GetReturnNumber() = " << p.GetReturnNumber() << std::endl;
                                std::cerr << "\tIn Pulse (Num Returns = " << pt.GetNumberOfReturns() << "): pt.GetReturnNumber() = " << pt.GetReturnNumber() << std::endl;
                                std::cerr << "The return number was: " << pt.GetReturnNumber() << std::endl;
                                std::cerr << "The next return number should have been: " << nPtIdx << std::endl;
                                throw SPDIOException("Error in point numbering when building pulses.");
                            }
                            
							spdPt = this->createSPDPoint(pt);
							
							if(spdPt->z < zMin)
							{
								zMin = spdPt->z;
							}
							else if(spdPt->z > zMax)
							{
								zMax = spdPt->z;
							}
                            
                            x1 = spdPt->x;
                            y1 = spdPt->y;
                            z1 = spdPt->z;							
							
                            spdPulse->pts->push_back(spdPt);
							++numOfPoints;
						}
						else
						{
							std::cerr << "\nWarning: The file ended unexpectedly.\n";
                            std::cerr << "Expected " << spdPulse->numberOfReturns << " but only found " << i + 1 << " returns" << std::endl;
							spdPulse->numberOfReturns = i+1;
							throw SPDIOException("Unexpected end to the file.");
						}
					}
					++numOfPulses;
					//std::cout << "Points capacity (2) " << spdPulse->pts.capacity() << std::endl << std::endl;
					if(p.GetFlightLineEdge() == 1)
					{
						spdPulse->edgeFlightLineFlag = SPD_WITH_SCAN;
					}
					else
					{
						spdPulse->edgeFlightLineFlag = SPD_SCAN_END;
					}
					
					if(p.GetScanDirection() == 1)
					{
						spdPulse->scanDirectionFlag = SPD_POSITIVE;
					}
					else
					{
						spdPulse->scanDirectionFlag = SPD_NEGATIVE;
					}
					
					if(p.GetNumberOfReturns() > 1)
                    {
                        range = std::sqrt(std::pow(x1-x0,2) + std::pow(y1-y0,2) + std::pow(z1-z0,2));
                        spdPulse->zenith = std::acos((z1-z0) / range);
                        spdPulse->azimuth = std::atan((x1-x0)/(y1-y0));
                        if(spdPulse->azimuth < 0)
                        {
                            spdPulse->azimuth = spdPulse->azimuth + M_PI * 2;
                        }
                    }
					else
                    {
                        spdPulse->zenith = 0.0;
                        spdPulse->azimuth = 0.0;
                    }
                    spdPulse->user = p.GetScanAngleRank() + 90;
                    
					spdPulse->sourceID = p.GetPointSourceID();
                    
					if(indexCoords == SPD_FIRST_RETURN)
					{
						spdPulse->xIdx = spdPulse->pts->front()->x;
						spdPulse->yIdx = spdPulse->pts->front()->y;
					}
					else if(indexCoords == SPD_LAST_RETURN)
					{
						spdPulse->xIdx = spdPulse->pts->back()->x;
						spdPulse->yIdx = spdPulse->pts->back()->y;
					}
					else
					{
						throw SPDIOException("Indexing type unsupported");
					}
					
					if(first)
					{
						xMin = spdPulse->xIdx;
						xMax = spdPulse->xIdx;
						yMin = spdPulse->yIdx;
						yMax = spdPulse->yIdx;
						first = false;
					}
					else
					{
						if(spdPulse->xIdx < xMin)
						{
							xMin = spdPulse->xIdx;
						}
						else if(spdPulse->xIdx > xMax)
						{
							xMax = spdPulse->xIdx;
						}
						
						if(spdPulse->yIdx < yMin)
						{
							yMin = spdPulse->yIdx;
						}
						else if(spdPulse->yIdx > yMax)
						{
							yMax = spdPulse->yIdx;
						}
					}
					
					pulses->push_back(spdPulse);
				}
				
				ifs.close();
				std::cout << ". Complete\n";
				spdFile->setBoundingVolume(xMin, xMax, yMin, yMax, zMin, zMax);
				if(convertCoords)
				{
					spdFile->setSpatialReference(outputProjWKT);
				}
				spdFile->setNumberOfPulses(numOfPulses);
				spdFile->setNumberOfPoints(numOfPoints);
				spdFile->setOriginDefined(SPD_FALSE);
				spdFile->setDiscretePtDefined(SPD_TRUE);
				spdFile->setDecomposedPtDefined(SPD_FALSE);
				spdFile->setTransWaveformDefined(SPD_FALSE);
                spdFile->setReceiveWaveformDefined(SPD_FALSE);
			}
			else
			{
				throw SPDIOException("LAS file could not be opened.");
			}
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
	
	void SPDLASFileImporterStrictPulses::readAndProcessAllData(std::string inputFile, SPDFile *spdFile, SPDImporterProcessor *processor)throw(SPDIOException)
	{
		SPDPulseUtils pulseUtils;
		boost::uint_fast64_t numOfPulses = 0;
		boost::uint_fast64_t numOfPoints = 0;
		
		double xMin = 0;
		double xMax = 0;
		double yMin = 0;
		double yMax = 0;
		double zMin = 0;
		double zMax = 0;
		bool first = true;
		bool firstZ = true;
		
		classWarningGiven = false;
		
		try
		{
			std::ifstream ifs;
			ifs.open(inputFile.c_str(), std::ios::in | std::ios::binary);
			if(ifs.is_open())
			{
				liblas::ReaderFactory lasReaderFactory;
				liblas::Reader reader = lasReaderFactory.CreateWithStream(ifs);
				liblas::Header const& header = reader.GetHeader();
				
				spdFile->setFileSignature(header.GetFileSignature());
				spdFile->setSystemIdentifier(header.GetSystemId());
				
				if(spdFile->getSpatialReference() == "")
				{
					liblas::SpatialReference const &lasSpatial = header.GetSRS();
					std::string spatialRefProjWKT = lasSpatial.GetWKT();
					spdFile->setSpatialReference(spatialRefProjWKT);
				}
				
				if(convertCoords)
				{
					this->initCoordinateSystemTransformation(spdFile);
				}
				
				boost::uint_fast64_t reportedNumOfPts = header.GetPointRecordsCount();
				boost::uint_fast64_t feedback = reportedNumOfPts/10;
				unsigned int feedbackCounter = 0;
				
				SPDPoint *spdPt = NULL;
				SPDPulse *spdPulse = NULL;
                
                boost::uint_fast16_t nPtIdx = 0;
                double x0 = 0.0;
                double y0 = 0.0;
                double z0 = 0.0;
                double x1 = 0.0;
                double y1 = 0.0;
                double z1 = 0.0;                
                double range = 0.0;
                
				std::cout << "Started (Read Data) ." << std::flush;
				while (reader.ReadNextPoint())
				{
					if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
					{
						std::cout << "." << feedbackCounter << "." << std::flush;
						feedbackCounter += 10;
					}
					
					liblas::Point const& p = reader.GetPoint();
					spdPt = this->createSPDPoint(p);
					++numOfPoints;
					
					if(firstZ)
					{
						zMin = spdPt->z;
						zMax = spdPt->z;
						firstZ = false;
					}
					else
					{
						if(spdPt->z < zMin)
						{
							zMin = spdPt->z;
						}
						else if(spdPt->z > zMax)
						{
							zMax = spdPt->z;
						}
					}
                    
                    x0 = spdPt->x;
                    y0 = spdPt->y;
                    z0 = spdPt->z;
					
					spdPulse = new SPDPulse();
					pulseUtils.initSPDPulse(spdPulse);
                    spdPulse->pulseID = numOfPulses;
					spdPulse->numberOfReturns = p.GetNumberOfReturns();
                    if(spdPulse->numberOfReturns == 0)
                    {
                        spdPulse->numberOfReturns = 1;
                    }
                    if(spdPulse->numberOfReturns > 0)
                    {
                        spdPulse->pts->reserve(spdPulse->numberOfReturns);
                    }
                    
                    //std::cout << "Start Pulse (Num Returns = " << p.GetNumberOfReturns() << "): p.GetReturnNumber() = " << p.GetReturnNumber() << std::endl;
                    
                    if(p.GetReturnNumber() == 0)
                    {
                        nPtIdx = 1;
                    }
                    else
                    {
                        nPtIdx = 2;
                    }
                    
                    
					spdPulse->pts->push_back(spdPt);
					for(boost::uint_fast16_t i = 0; i < (spdPulse->numberOfReturns-1); ++i)
					{
						if(reader.ReadNextPoint())
						{
							if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
							{
								std::cout << "." << feedbackCounter << "." << std::flush;
								feedbackCounter += 10;
							}
							liblas::Point const& pt = reader.GetPoint();
                            
                            //std::cout << "\tIn Pulse (Num Returns = " << pt.GetNumberOfReturns() << "): pt.GetReturnNumber() = " << pt.GetReturnNumber() << std::endl;
                            
                            if(nPtIdx != pt.GetReturnNumber())
                            {
                                std::cerr << "Start Pulse (Num Returns = " << spdPulse->numberOfReturns << "): p.GetReturnNumber() = " << p.GetReturnNumber() << std::endl;
                                std::cerr << "\tIn Pulse (Num Returns = " << pt.GetNumberOfReturns() << "): pt.GetReturnNumber() = " << pt.GetReturnNumber() << std::endl;
                                std::cerr << "The return number was: " << pt.GetReturnNumber() << std::endl;
                                std::cerr << "The next return number should have been: " << nPtIdx << std::endl;
                                throw SPDIOException("Error in point numbering when building pulses.");
                            }
                            
							spdPt = this->createSPDPoint(pt);
							
							if(spdPt->z < zMin)
							{
								zMin = spdPt->z;
							}
							else if(spdPt->z > zMax)
							{
								zMax = spdPt->z;
							}
							
                            x1 = spdPt->x;
                            y1 = spdPt->y;
                            z1 = spdPt->z;                            
                            
							spdPulse->pts->push_back(spdPt);
							++numOfPoints;
                            ++nPtIdx;
						}
						else
						{
							std::cerr << "\nWarning: The file ended unexpectedly.\n";
                            std::cerr << "Expected " << spdPulse->numberOfReturns << " but only found " << i + 1 << " returns" << std::endl;
							spdPulse->numberOfReturns = i+1;
							throw SPDIOException("Unexpected end to the file.");
						}
					}
					++numOfPulses;
					
					if(p.GetFlightLineEdge() == 1)
					{
						spdPulse->edgeFlightLineFlag = SPD_WITH_SCAN;
					}
					else
					{
						spdPulse->edgeFlightLineFlag = SPD_SCAN_END;
					}
					
					if(p.GetScanDirection() == 1)
					{
						spdPulse->scanDirectionFlag = SPD_POSITIVE;
					}
					else
					{
						spdPulse->scanDirectionFlag = SPD_NEGATIVE;
					}
				    
                    if(p.GetNumberOfReturns() > 1)
                    {
                        range = std::sqrt(std::pow(x1-x0,2) + std::pow(y1-y0,2) + std::pow(z1-z0,2));
                        spdPulse->zenith = std::acos((z1-z0) / range);
                        spdPulse->azimuth = std::atan((x1-x0)/(y1-y0));
                        if(spdPulse->azimuth < 0)
                        {
                            spdPulse->azimuth = spdPulse->azimuth + M_PI * 2;
                        }
                    }
					else
                    {
                        spdPulse->zenith = 0.0;
                        spdPulse->azimuth = 0.0;
                    }
                    spdPulse->user = p.GetScanAngleRank() + 90;
                    
                    spdPulse->sourceID = p.GetPointSourceID();
					
					if(indexCoords == SPD_FIRST_RETURN)
					{
						spdPulse->xIdx = spdPulse->pts->front()->x;
						spdPulse->yIdx = spdPulse->pts->front()->y;
					}
					else if(indexCoords == SPD_LAST_RETURN)
					{
						spdPulse->xIdx = spdPulse->pts->back()->x;
						spdPulse->yIdx = spdPulse->pts->back()->y;
					}
					else
					{
						throw SPDIOException("Indexing type unsupported");
					}
					
					if(first)
					{
						xMin = spdPulse->xIdx;
						xMax = spdPulse->xIdx;
						yMin = spdPulse->yIdx;
						yMax = spdPulse->yIdx;
						first = false;
					}
					else
					{
						if(spdPulse->xIdx < xMin)
						{
							xMin = spdPulse->xIdx;
						}
						else if(spdPulse->xIdx > xMax)
						{
							xMax = spdPulse->xIdx;
						}
						
						if(spdPulse->yIdx < yMin)
						{
							yMin = spdPulse->yIdx;
						}
						else if(spdPulse->yIdx > yMax)
						{
							yMax = spdPulse->yIdx;
						}
					}
					
					processor->processImportedPulse(spdFile, spdPulse);
				}
				
				ifs.close();
				std::cout << ". Complete\n";
				spdFile->setBoundingVolume(xMin, xMax, yMin, yMax, zMin, zMax);
				if(convertCoords)
				{
					spdFile->setSpatialReference(outputProjWKT);
				}
				spdFile->setNumberOfPulses(numOfPulses);
				spdFile->setNumberOfPoints(numOfPoints);
				spdFile->setOriginDefined(SPD_FALSE);
				spdFile->setDiscretePtDefined(SPD_TRUE);
				spdFile->setDecomposedPtDefined(SPD_FALSE);
				spdFile->setTransWaveformDefined(SPD_FALSE);
                spdFile->setReceiveWaveformDefined(SPD_FALSE);
			}
			else
			{
				throw SPDIOException("LAS file could not be opened.");
			}
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
	
	bool SPDLASFileImporterStrictPulses::isFileType(std::string fileType)
	{
		if(fileType == "LASSTRICT")
		{
			return true;
		}
		return false;
	}
	
    void SPDLASFileImporterStrictPulses::readHeaderInfo(std::string, SPDFile*) throw(SPDIOException)
    {
        // No Header to Read..
    }
    
	SPDPoint* SPDLASFileImporterStrictPulses::createSPDPoint(liblas::Point const& pt)throw(SPDIOException)
	{
		try
		{
			SPDPointUtils spdPtUtils;
			SPDPoint *spdPt = new SPDPoint();
			spdPtUtils.initSPDPoint(spdPt);
			double x = pt.GetX();
			double y = pt.GetY();
			double z = pt.GetZ();
			
			if(convertCoords)
			{
				this->transformCoordinateSystem(&x, &y, &z);
			}
			
			spdPt->x = x;
			spdPt->y = y;
			spdPt->z = z;
			spdPt->amplitudeReturn = pt.GetIntensity();
			spdPt->user = pt.GetUserData();
			
			liblas::Classification lasClass = pt.GetClassification();
			
			if(lasClass.GetClass() == pt.eCreated)
			{
				spdPt->classification = SPD_CREATED;
			}
			else if(lasClass.GetClass() == pt.eUnclassified)
			{
				spdPt->classification = SPD_UNCLASSIFIED;
			}
			else if(lasClass.GetClass() == pt.eGround)
			{
				spdPt->classification = SPD_GROUND;
			}
			else if(lasClass.GetClass() == pt.eLowVegetation)
			{
				spdPt->classification = SPD_LOW_VEGETATION;
			}
			else if(lasClass.GetClass() == pt.eMediumVegetation)
			{
				spdPt->classification = SPD_MEDIUM_VEGETATION;
			}
			else if(lasClass.GetClass() == pt.eHighVegetation)
			{
				spdPt->classification = SPD_HIGH_VEGETATION;
			}
			else if(lasClass.GetClass() == pt.eBuilding)
			{
				spdPt->classification = SPD_BUILDING;
			}
			else if(lasClass.GetClass() == pt.eLowPoint)
			{
				spdPt->classification = SPD_CREATED;
				spdPt->lowPoint = SPD_TRUE;
			}
			else if(lasClass.GetClass() == pt.eModelKeyPoint)
			{
				spdPt->classification = SPD_CREATED;
				spdPt->modelKeyPoint = SPD_TRUE;
			}
			else if(lasClass.GetClass() == pt.eWater)
			{
				spdPt->classification = SPD_WATER;
			}
			else if(lasClass.GetClass() == pt.eOverlapPoints)
			{
				spdPt->classification = SPD_CREATED;
				spdPt->overlap = SPD_TRUE;
			}
			else
			{
				spdPt->classification = SPD_CREATED;
				if(!classWarningGiven)
				{
					std::cerr << "WARNING: The class ID was not recognised - check the classes points were allocated too.";
					classWarningGiven = true;
				}
			}
			
			liblas::Color const &lasColor = pt.GetColor();
			spdPt->red = lasColor.GetRed();
			spdPt->green = lasColor.GetGreen();
			spdPt->blue = lasColor.GetBlue();
			
			spdPt->returnID = pt.GetReturnNumber();
			spdPt->gpsTime = pt.GetTime();
			
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
	
	SPDLASFileImporterStrictPulses::~SPDLASFileImporterStrictPulses()
	{
		
	}
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    SPDLASFileNoPulsesImporter::SPDLASFileNoPulsesImporter(bool convertCoords, std::string outputProjWKT, std::string schema, boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold):SPDDataImporter(convertCoords, outputProjWKT, schema, indexCoords, defineOrigin, originX, originY, originZ, waveNoiseThreshold)
	{
		
	}
    
    SPDDataImporter* SPDLASFileNoPulsesImporter::getInstance(bool convertCoords, std::string outputProjWKT, std::string schema, boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold)
    {
        return new SPDLASFileNoPulsesImporter(convertCoords, outputProjWKT, schema, indexCoords, defineOrigin, originX, originY, originZ, waveNoiseThreshold);
    }
	
	std::list<SPDPulse*>* SPDLASFileNoPulsesImporter::readAllDataToList(std::string inputFile, SPDFile *spdFile)throw(SPDIOException)
	{
		SPDPulseUtils pulseUtils;
		std::list<SPDPulse*> *pulses = new std::list<SPDPulse*>();
		boost::uint_fast64_t numOfPulses = 0;
		boost::uint_fast64_t numOfPoints = 0;
		
		double xMin = 0;
		double xMax = 0;
		double yMin = 0;
		double yMax = 0;
		double zMin = 0;
		double zMax = 0;
		bool first = true;
		bool firstZ = true;
		
		classWarningGiven = false;
		
		try
		{
			std::ifstream ifs;
			ifs.open(inputFile.c_str(), std::ios::in | std::ios::binary);
			if(ifs.is_open())
			{
				liblas::ReaderFactory lasReaderFactory;
				liblas::Reader reader = lasReaderFactory.CreateWithStream(ifs);
				liblas::Header const& header = reader.GetHeader();
				
				spdFile->setFileSignature(header.GetFileSignature());
				spdFile->setSystemIdentifier(header.GetSystemId());
				
				if(spdFile->getSpatialReference() == "")
				{
					liblas::SpatialReference const &lasSpatial = header.GetSRS();
					std::string spatialRefProjWKT = lasSpatial.GetWKT();
					spdFile->setSpatialReference(spatialRefProjWKT);
				}
				
				if(convertCoords)
				{
					this->initCoordinateSystemTransformation(spdFile);
				}
				
				boost::uint_fast64_t reportedNumOfPts = header.GetPointRecordsCount();
				boost::uint_fast64_t feedback = reportedNumOfPts/10;
				unsigned int feedbackCounter = 0;
				
				SPDPoint *spdPt = NULL;
				SPDPulse *spdPulse = NULL;               
				
				std::cout << "Started (Read Data) ." << std::flush;
				while (reader.ReadNextPoint())
				{
					//std::cout << numOfPoints << std::endl;
					if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
					{
						std::cout << "." << feedbackCounter << "." << std::flush;
						feedbackCounter += 10;
					}
					
					liblas::Point const& p = reader.GetPoint();
					spdPt = this->createSPDPoint(p);
					++numOfPoints;
					
					if(firstZ)
					{
						zMin = spdPt->z;
						zMax = spdPt->z;
						firstZ = false;
					}
					else
					{
						if(spdPt->z < zMin)
						{
							zMin = spdPt->z;
						}
						else if(spdPt->z > zMax)
						{
							zMax = spdPt->z;
						}
					}
					
					spdPulse = new SPDPulse();
					pulseUtils.initSPDPulse(spdPulse);
                    spdPulse->pulseID = numOfPulses;
					//std::cout << "Pulse size " << sizeof(SPDPulse) << std::endl;
					//std::cout << "Point size " << sizeof(SPDPoint) << std::endl;
					//std::cout << "Points capacity (1) " << spdPulse->pts.capacity() << std::endl;
					spdPulse->numberOfReturns = p.GetNumberOfReturns();
                    if(spdPulse->numberOfReturns == 0)
                    {
                        spdPulse->numberOfReturns = 1;
                    }
                    if(spdPulse->numberOfReturns > 0)
                    {
                        spdPulse->pts->reserve(spdPulse->numberOfReturns);
                    }
                    
					spdPulse->pts->push_back(spdPt);

					++numOfPulses;
					//std::cout << "Points capacity (2) " << spdPulse->pts.capacity() << std::endl << std::endl;
					if(p.GetFlightLineEdge() == 1)
					{
						spdPulse->edgeFlightLineFlag = SPD_WITH_SCAN;
					}
					else
					{
						spdPulse->edgeFlightLineFlag = SPD_SCAN_END;
					}
					
					if(p.GetScanDirection() == 1)
					{
						spdPulse->scanDirectionFlag = SPD_POSITIVE;
					}
					else
					{
						spdPulse->scanDirectionFlag = SPD_NEGATIVE;
					}
					
                    spdPulse->zenith = 0.0;
                    spdPulse->azimuth = 0.0;
                    spdPulse->user = p.GetScanAngleRank() + 90;

					spdPulse->sourceID = p.GetPointSourceID();
					
					if(indexCoords == SPD_FIRST_RETURN)
					{
						spdPulse->xIdx = spdPulse->pts->front()->x;
						spdPulse->yIdx = spdPulse->pts->front()->y;
					}
					else if(indexCoords == SPD_LAST_RETURN)
					{
						spdPulse->xIdx = spdPulse->pts->back()->x;
						spdPulse->yIdx = spdPulse->pts->back()->y;
					}
					else
					{
						throw SPDIOException("Indexing type unsupported");
					}
					
					if(first)
					{
						xMin = spdPulse->xIdx;
						xMax = spdPulse->xIdx;
						yMin = spdPulse->yIdx;
						yMax = spdPulse->yIdx;
						first = false;
					}
					else
					{
						if(spdPulse->xIdx < xMin)
						{
							xMin = spdPulse->xIdx;
						}
						else if(spdPulse->xIdx > xMax)
						{
							xMax = spdPulse->xIdx;
						}
						
						if(spdPulse->yIdx < yMin)
						{
							yMin = spdPulse->yIdx;
						}
						else if(spdPulse->yIdx > yMax)
						{
							yMax = spdPulse->yIdx;
						}
					}
					
					pulses->push_back(spdPulse);
				}
				
				ifs.close();
				std::cout << ". Complete\n";
				spdFile->setBoundingVolume(xMin, xMax, yMin, yMax, zMin, zMax);
				if(convertCoords)
				{
					spdFile->setSpatialReference(outputProjWKT);
				}
				spdFile->setNumberOfPulses(numOfPulses);
				spdFile->setNumberOfPoints(numOfPoints);
				spdFile->setOriginDefined(SPD_FALSE);
				spdFile->setDiscretePtDefined(SPD_TRUE);
				spdFile->setDecomposedPtDefined(SPD_FALSE);
				spdFile->setTransWaveformDefined(SPD_FALSE);
                spdFile->setReceiveWaveformDefined(SPD_FALSE);
			}
			else
			{
				throw SPDIOException("LAS file could not be opened.");
			}
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
	
	std::vector<SPDPulse*>* SPDLASFileNoPulsesImporter::readAllDataToVector(std::string inputFile, SPDFile *spdFile)throw(SPDIOException)
	{
		SPDPulseUtils pulseUtils;
		std::vector<SPDPulse*> *pulses = new std::vector<SPDPulse*>();
		boost::uint_fast64_t numOfPulses = 0;
		boost::uint_fast64_t numOfPoints = 0;
		
		double xMin = 0;
		double xMax = 0;
		double yMin = 0;
		double yMax = 0;
		double zMin = 0;
		double zMax = 0;
		bool first = true;
		bool firstZ = true;
		
		classWarningGiven = false;
		
		try
		{
			std::ifstream ifs;
			ifs.open(inputFile.c_str(), std::ios::in | std::ios::binary);
			if(ifs.is_open())
			{
				liblas::ReaderFactory lasReaderFactory;
				liblas::Reader reader = lasReaderFactory.CreateWithStream(ifs);
				liblas::Header const& header = reader.GetHeader();
				
				spdFile->setFileSignature(header.GetFileSignature());
				spdFile->setSystemIdentifier(header.GetSystemId());
				
				if(spdFile->getSpatialReference() == "")
				{
					liblas::SpatialReference const &lasSpatial = header.GetSRS();
					std::string spatialRefProjWKT = lasSpatial.GetWKT();
					spdFile->setSpatialReference(spatialRefProjWKT);
				}
				
				if(convertCoords)
				{
					this->initCoordinateSystemTransformation(spdFile);
				}
				
				pulses->reserve(header.GetPointRecordsCount());
				
				boost::uint_fast64_t reportedNumOfPts = header.GetPointRecordsCount();
				boost::uint_fast64_t feedback = reportedNumOfPts/10;
				unsigned int feedbackCounter = 0;
				
				SPDPoint *spdPt = NULL;
				SPDPulse *spdPulse = NULL;
				
				std::cout << "Started (Read Data) ." << std::flush;
				while (reader.ReadNextPoint())
				{
					//std::cout << numOfPoints << std::endl;
					if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
					{
						std::cout << "." << feedbackCounter << "." << std::flush;
						feedbackCounter += 10;
					}
					
					liblas::Point const& p = reader.GetPoint();
					spdPt = this->createSPDPoint(p);
					++numOfPoints;
					
					if(firstZ)
					{
						zMin = spdPt->z;
						zMax = spdPt->z;
						firstZ = false;
					}
					else
					{
						if(spdPt->z < zMin)
						{
							zMin = spdPt->z;
						}
						else if(spdPt->z > zMax)
						{
							zMax = spdPt->z;
						}
					}
                    
					spdPulse = new SPDPulse();
					pulseUtils.initSPDPulse(spdPulse);
                    spdPulse->pulseID = numOfPulses;
					//std::cout << "Pulse size " << sizeof(SPDPulse) << std::endl;
					//std::cout << "Point size " << sizeof(SPDPoint) << std::endl;
					//std::cout << "Points capacity (1) " << spdPulse->pts.capacity() << std::endl;
					spdPulse->numberOfReturns = p.GetNumberOfReturns();
                    if(spdPulse->numberOfReturns == 0)
                    {
                        spdPulse->numberOfReturns = 1;
                    }
                    if(spdPulse->numberOfReturns > 0)
                    {
                        spdPulse->pts->reserve(spdPulse->numberOfReturns);
                    }
                    
					spdPulse->pts->push_back(spdPt);

					++numOfPulses;
					//std::cout << "Points capacity (2) " << spdPulse->pts.capacity() << std::endl << std::endl;
					if(p.GetFlightLineEdge() == 1)
					{
						spdPulse->edgeFlightLineFlag = SPD_WITH_SCAN;
					}
					else
					{
						spdPulse->edgeFlightLineFlag = SPD_SCAN_END;
					}
					
					if(p.GetScanDirection() == 1)
					{
						spdPulse->scanDirectionFlag = SPD_POSITIVE;
					}
					else
					{
						spdPulse->scanDirectionFlag = SPD_NEGATIVE;
					}
                                        
                    spdPulse->zenith = 0.0;
                    spdPulse->azimuth = 0.0;
                    spdPulse->user = p.GetScanAngleRank() + 90;

					spdPulse->sourceID = p.GetPointSourceID();
                    
					if(indexCoords == SPD_FIRST_RETURN)
					{
						spdPulse->xIdx = spdPulse->pts->front()->x;
						spdPulse->yIdx = spdPulse->pts->front()->y;
					}
					else if(indexCoords == SPD_LAST_RETURN)
					{
						spdPulse->xIdx = spdPulse->pts->back()->x;
						spdPulse->yIdx = spdPulse->pts->back()->y;
					}
					else
					{
						throw SPDIOException("Indexing type unsupported");
					}
					
					if(first)
					{
						xMin = spdPulse->xIdx;
						xMax = spdPulse->xIdx;
						yMin = spdPulse->yIdx;
						yMax = spdPulse->yIdx;
						first = false;
					}
					else
					{
						if(spdPulse->xIdx < xMin)
						{
							xMin = spdPulse->xIdx;
						}
						else if(spdPulse->xIdx > xMax)
						{
							xMax = spdPulse->xIdx;
						}
						
						if(spdPulse->yIdx < yMin)
						{
							yMin = spdPulse->yIdx;
						}
						else if(spdPulse->yIdx > yMax)
						{
							yMax = spdPulse->yIdx;
						}
					}
					
					pulses->push_back(spdPulse);
				}
				
				ifs.close();
				std::cout << ". Complete\n";
				spdFile->setBoundingVolume(xMin, xMax, yMin, yMax, zMin, zMax);
				if(convertCoords)
				{
					spdFile->setSpatialReference(outputProjWKT);
				}
				spdFile->setNumberOfPulses(numOfPulses);
				spdFile->setNumberOfPoints(numOfPoints);
				spdFile->setOriginDefined(SPD_FALSE);
				spdFile->setDiscretePtDefined(SPD_TRUE);
				spdFile->setDecomposedPtDefined(SPD_FALSE);
				spdFile->setTransWaveformDefined(SPD_FALSE);
                spdFile->setReceiveWaveformDefined(SPD_FALSE);
			}
			else
			{
				throw SPDIOException("LAS file could not be opened.");
			}
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
	
	void SPDLASFileNoPulsesImporter::readAndProcessAllData(std::string inputFile, SPDFile *spdFile, SPDImporterProcessor *processor)throw(SPDIOException)
	{
		SPDPulseUtils pulseUtils;
		boost::uint_fast64_t numOfPulses = 0;
		boost::uint_fast64_t numOfPoints = 0;
		
		double xMin = 0;
		double xMax = 0;
		double yMin = 0;
		double yMax = 0;
		double zMin = 0;
		double zMax = 0;
		bool first = true;
		bool firstZ = true;
		
		classWarningGiven = false;
		
		try
		{
			std::ifstream ifs;
			ifs.open(inputFile.c_str(), std::ios::in | std::ios::binary);
			if(ifs.is_open())
			{
				liblas::ReaderFactory lasReaderFactory;
				liblas::Reader reader = lasReaderFactory.CreateWithStream(ifs);
				liblas::Header const& header = reader.GetHeader();
				
				spdFile->setFileSignature(header.GetFileSignature());
				spdFile->setSystemIdentifier(header.GetSystemId());
				
				if(spdFile->getSpatialReference() == "")
				{
					liblas::SpatialReference const &lasSpatial = header.GetSRS();
					std::string spatialRefProjWKT = lasSpatial.GetWKT();
					spdFile->setSpatialReference(spatialRefProjWKT);
				}
				
				if(convertCoords)
				{
					this->initCoordinateSystemTransformation(spdFile);
				}
				
				boost::uint_fast64_t reportedNumOfPts = header.GetPointRecordsCount();
				boost::uint_fast64_t feedback = reportedNumOfPts/10;
				unsigned int feedbackCounter = 0;
				
				SPDPoint *spdPt = NULL;
				SPDPulse *spdPulse = NULL;
                                
				std::cout << "Started (Read Data) ." << std::flush;
				while (reader.ReadNextPoint())
				{
					if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
					{
						std::cout << "." << feedbackCounter << "." << std::flush;
						feedbackCounter += 10;
					}
					
					liblas::Point const& p = reader.GetPoint();
					spdPt = this->createSPDPoint(p);
					++numOfPoints;
					
					if(firstZ)
					{
						zMin = spdPt->z;
						zMax = spdPt->z;
						firstZ = false;
					}
					else
					{
						if(spdPt->z < zMin)
						{
							zMin = spdPt->z;
						}
						else if(spdPt->z > zMax)
						{
							zMax = spdPt->z;
						}
					}
                    
					spdPulse = new SPDPulse();
					pulseUtils.initSPDPulse(spdPulse);
                    spdPulse->pulseID = numOfPulses;
					spdPulse->numberOfReturns = 1;
                    
					spdPulse->pts->push_back(spdPt);
					
					++numOfPulses;
					
					if(p.GetFlightLineEdge() == 1)
					{
						spdPulse->edgeFlightLineFlag = SPD_WITH_SCAN;
					}
					else
					{
						spdPulse->edgeFlightLineFlag = SPD_SCAN_END;
					}
					
					if(p.GetScanDirection() == 1)
					{
						spdPulse->scanDirectionFlag = SPD_POSITIVE;
					}
					else
					{
						spdPulse->scanDirectionFlag = SPD_NEGATIVE;
					}
                    
                    spdPulse->zenith = 0.0;
                    spdPulse->azimuth = 0.0;
                    spdPulse->user = p.GetScanAngleRank() + 90;                    
                    
					spdPulse->sourceID = p.GetPointSourceID();
					
					if(indexCoords == SPD_FIRST_RETURN)
					{
						spdPulse->xIdx = spdPulse->pts->front()->x;
						spdPulse->yIdx = spdPulse->pts->front()->y;
					}
					else if(indexCoords == SPD_LAST_RETURN)
					{
						spdPulse->xIdx = spdPulse->pts->back()->x;
						spdPulse->yIdx = spdPulse->pts->back()->y;
					}
					else
					{
						throw SPDIOException("Indexing type unsupported");
					}
					
					if(first)
					{
						xMin = spdPulse->xIdx;
						xMax = spdPulse->xIdx;
						yMin = spdPulse->yIdx;
						yMax = spdPulse->yIdx;
						first = false;
					}
					else
					{
						if(spdPulse->xIdx < xMin)
						{
							xMin = spdPulse->xIdx;
						}
						else if(spdPulse->xIdx > xMax)
						{
							xMax = spdPulse->xIdx;
						}
						
						if(spdPulse->yIdx < yMin)
						{
							yMin = spdPulse->yIdx;
						}
						else if(spdPulse->yIdx > yMax)
						{
							yMax = spdPulse->yIdx;
						}
					}
					
					processor->processImportedPulse(spdFile, spdPulse);
				}
				
				ifs.close();
				std::cout << ". Complete\n";
				spdFile->setBoundingVolume(xMin, xMax, yMin, yMax, zMin, zMax);
				if(convertCoords)
				{
					spdFile->setSpatialReference(outputProjWKT);
				}
				spdFile->setNumberOfPulses(numOfPulses);
				spdFile->setNumberOfPoints(numOfPoints);
				spdFile->setOriginDefined(SPD_FALSE);
				spdFile->setDiscretePtDefined(SPD_TRUE);
				spdFile->setDecomposedPtDefined(SPD_FALSE);
				spdFile->setTransWaveformDefined(SPD_FALSE);
                spdFile->setReceiveWaveformDefined(SPD_FALSE);
			}
			else
			{
				throw SPDIOException("LAS file could not be opened.");
			}
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
	
	bool SPDLASFileNoPulsesImporter::isFileType(std::string fileType)
	{
		if(fileType == "LASNP")
		{
			return true;
		}
		return false;
	}
	
    void SPDLASFileNoPulsesImporter::readHeaderInfo(std::string, SPDFile*) throw(SPDIOException)
    {
        // No Header to Read..
    }
    
	SPDPoint* SPDLASFileNoPulsesImporter::createSPDPoint(liblas::Point const& pt)throw(SPDIOException)
	{
		try
		{
			SPDPointUtils spdPtUtils;
			SPDPoint *spdPt = new SPDPoint();
			spdPtUtils.initSPDPoint(spdPt);
			double x = pt.GetX();
			double y = pt.GetY();
			double z = pt.GetZ();
			
			if(convertCoords)
			{
				this->transformCoordinateSystem(&x, &y, &z);
			}
			
			spdPt->x = x;
			spdPt->y = y;
			spdPt->z = z;
			spdPt->amplitudeReturn = pt.GetIntensity();
			spdPt->user = pt.GetUserData();
			
			liblas::Classification lasClass = pt.GetClassification();
			
			if(lasClass.GetClass() == pt.eCreated)
			{
				spdPt->classification = SPD_CREATED;
			}
			else if(lasClass.GetClass() == pt.eUnclassified)
			{
				spdPt->classification = SPD_UNCLASSIFIED;
			}
			else if(lasClass.GetClass() == pt.eGround)
			{
				spdPt->classification = SPD_GROUND;
			}
			else if(lasClass.GetClass() == pt.eLowVegetation)
			{
				spdPt->classification = SPD_LOW_VEGETATION;
			}
			else if(lasClass.GetClass() == pt.eMediumVegetation)
			{
				spdPt->classification = SPD_MEDIUM_VEGETATION;
			}
			else if(lasClass.GetClass() == pt.eHighVegetation)
			{
				spdPt->classification = SPD_HIGH_VEGETATION;
			}
			else if(lasClass.GetClass() == pt.eBuilding)
			{
				spdPt->classification = SPD_BUILDING;
			}
			else if(lasClass.GetClass() == pt.eLowPoint)
			{
				spdPt->classification = SPD_CREATED;
				spdPt->lowPoint = SPD_TRUE;
			}
			else if(lasClass.GetClass() == pt.eModelKeyPoint)
			{
				spdPt->classification = SPD_CREATED;
				spdPt->modelKeyPoint = SPD_TRUE;
			}
			else if(lasClass.GetClass() == pt.eWater)
			{
				spdPt->classification = SPD_WATER;
			}
			else if(lasClass.GetClass() == pt.eOverlapPoints)
			{
				spdPt->classification = SPD_CREATED;
				spdPt->overlap = SPD_TRUE;
			}
			else
			{
				spdPt->classification = SPD_CREATED;
				if(!classWarningGiven)
				{
					std::cerr << "WARNING: The class ID was not recognised - check the classes points were allocated too.";
					classWarningGiven = true;
				}
			}
			
			liblas::Color const &lasColor = pt.GetColor();
			spdPt->red = lasColor.GetRed();
			spdPt->green = lasColor.GetGreen();
			spdPt->blue = lasColor.GetBlue();
			
			spdPt->returnID = pt.GetReturnNumber();
			spdPt->gpsTime = pt.GetTime();
			
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
	
	SPDLASFileNoPulsesImporter::~SPDLASFileNoPulsesImporter()
	{
		
	}
    
    
}
