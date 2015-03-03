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
            // Open LAS file
            LASreadOpener lasreadopener;
            LASreader* lasreader = lasreadopener.open(inputFile.c_str());
            
            if(lasreader != 0)
			{
                // Get header
                LASheader *header = &lasreader->header;
                
				spdFile->setFileSignature(header->file_signature);
				spdFile->setSystemIdentifier(header->system_identifier);
				
				if(spdFile->getSpatialReference() == "")
				{
					//FIXME: Need to get spatial information from LAS and convert to WKT (don't think LASlib does this)
                    /*liblas::SpatialReference const &lasSpatial = header.GetSRS();
					std::string spatialRefProjWKT = lasSpatial.GetWKT();
					spdFile->setSpatialReference(spatialRefProjWKT);*/
				}
				
				if(convertCoords)
				{
					this->initCoordinateSystemTransformation(spdFile);
				}
				
				boost::uint_fast64_t reportedNumOfPts = header->number_of_point_records;
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
				while (lasreader->read_point())
				{
					//std::cout << numOfPoints << std::endl;
					if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
					{
						std::cout << "." << feedbackCounter << "." << std::flush;
						feedbackCounter += 10;
					}
					
					if(lasreader->point.get_return_number() <= 1)
                    {
                        spdPt = this->createSPDPoint(lasreader->point);
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
                        spdPulse->numberOfReturns = lasreader->point.get_number_of_returns();
                        if(spdPulse->numberOfReturns == 0)
                        {
                            spdPulse->numberOfReturns = 1;
                        }
                        if(spdPulse->numberOfReturns > 0)
                        {
                            spdPulse->pts->reserve(spdPulse->numberOfReturns);
                        }
                        
                        //std::cout << "Start Pulse (Num Returns = " << p.GetNumberOfReturns() << "): p.GetReturnNumber() = " << p.GetReturnNumber() << std::endl;
                        
                        if(lasreader->point.get_return_number() == 0)
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
                            if(lasreader->read_point())
                            {
                                if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
                                {
                                    std::cout << "." << feedbackCounter << "." << std::flush;
                                    feedbackCounter += 10;
                                }
                                
                                
                                //std::cout << "\tIn Pulse (Num Returns = " << pt.get_NumberOfReturns() << "): pt.get_ReturnNumber() = " << pt.get_ReturnNumber() << std::endl;
                                
                                if(nPtIdx != lasreader->point.get_return_number())
                                {
                                    // FIXME: Could this error could be tidied up. Get it a lot with our ALSPP produced LAS files
                                    /*std::cerr << "Start Pulse (Num Returns = " << spdPulse->numberOfReturns << "): p.GetReturnNumber() = " << p.GetReturnNumber() << std::endl;
                                    std::cerr << "\tIn Pulse (Num Returns = " << pt.get_NumberOfReturns() << "): pt.get_ReturnNumber() = " << pt.get_ReturnNumber() << std::endl;
                                    std::cerr << "The return number was: " << pt.get_ReturnNumber() << std::endl;
                                    std::cerr << "The next return number should have been: " << nPtIdx << std::endl;*/
                                    std::cerr << "WARNING: Pulse was written as incompleted pulse.\n";
                                    spdPulse->numberOfReturns = i+1;
                                    //throw SPDIOException("Error in point numbering when building pulses.");
                                    break;
                                }
                                
                                spdPt = this->createSPDPoint(lasreader->point);
                                
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
                        
                        if(lasreader->point.get_edge_of_flight_line() == 1)
                        {
                            spdPulse->edgeFlightLineFlag = SPD_WITH_SCAN;
                        }
                        else
                        {
                            spdPulse->edgeFlightLineFlag = SPD_SCAN_END;
                        }
                        
                        if(lasreader->point.get_scan_direction_flag() == 1)
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
                        spdPulse->user = lasreader->point.get_scan_angle_rank() + 90;
                        
                        spdPulse->sourceID = lasreader->point.get_point_source_ID();
                        
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
                        //std::cerr << "p.GetReturnNumber() = " << p.GetReturnNumber() << std::endl;
                        //std::cerr << "p.GetNumberOfReturns() = " << p.GetNumberOfReturns() << std::endl;
                        std::cerr << "Warning: Point ignored. It is the first in pulse but has a return number greater than 1.\n";
                    }
				}
				
				lasreader->close();
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
            // Open LAS file
            LASreadOpener lasreadopener;
            LASreader* lasreader = lasreadopener.open(inputFile.c_str());
            
            if(lasreader != 0)
            {
                // Get header
                LASheader *header = &lasreader->header;
                
                spdFile->setFileSignature(header->file_signature);
                spdFile->setSystemIdentifier(header->system_identifier);
            				
				if(spdFile->getSpatialReference() == "")
				{
					//FIXME: Need to get spatial information from LAS and convert to WKT (don't think LASlib does this)
                    /*liblas::SpatialReference const &lasSpatial = header.GetSRS();
					std::string spatialRefProjWKT = lasSpatial.GetWKT();
					spdFile->setSpatialReference(spatialRefProjWKT);*/
				}
				
				if(convertCoords)
				{
					this->initCoordinateSystemTransformation(spdFile);
				}
				
				pulses->reserve(header->number_of_point_records);
				
				boost::uint_fast64_t reportedNumOfPts = header->number_of_point_records;
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
                while (lasreader->read_point())
				{
					//std::cout << numOfPoints << std::endl;
					if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
					{
						std::cout << "." << feedbackCounter << "." << std::flush;
						feedbackCounter += 10;
					}
					
                    if(lasreader->point.get_return_number() <= 1)
                    {
                        spdPt = this->createSPDPoint(lasreader->point);
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
                        spdPulse->numberOfReturns = lasreader->point.get_number_of_returns();
                        if(spdPulse->numberOfReturns == 0)
                        {
                            spdPulse->numberOfReturns = 1;
                        }
                        if(spdPulse->numberOfReturns > 0)
                        {
                            spdPulse->pts->reserve(spdPulse->numberOfReturns);
                        }
                        
                        //std::cout << "Start Pulse (Num Returns = " << p.GetNumberOfReturns() << "): p.GetReturnNumber() = " << p.GetReturnNumber() << std::endl;
                        
                        if(lasreader->point.get_return_number() == 0)
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
                            if(lasreader->read_point())
                            {
                                if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
                                {
                                    std::cout << "." << feedbackCounter << "." << std::flush;
                                    feedbackCounter += 10;
                                }
                                
                                //std::cout << "\tIn Pulse (Num Returns = " << pt.get_NumberOfReturns() << "): pt.get_ReturnNumber() = " << pt.get_ReturnNumber() << std::endl;
                                
                                if(nPtIdx != lasreader->point.get_return_number())
                                {
                                    /*std::cerr << "Start Pulse (Num Returns = " << spdPulse->numberOfReturns << "): p.GetReturnNumber() = " << p.GetReturnNumber() << std::endl;
                                    std::cerr << "\tIn Pulse (Num Returns = " << pt.get_NumberOfReturns() << "): pt.get_ReturnNumber() = " << pt.get_ReturnNumber() << std::endl;
                                    std::cerr << "The return number was: " << pt.get_ReturnNumber() << std::endl;
                                    std::cerr << "The next return number should have been: " << nPtIdx << std::endl;*/
                                    std::cerr << "WARNING: Pulse was written as incompleted pulse.\n";
                                    spdPulse->numberOfReturns = i+1;
                                    //throw SPDIOException("Error in point numbering when building pulses.");
                                    break;
                                }
                                
                                spdPt = this->createSPDPoint(lasreader->point);
                                
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
                        
                        if(lasreader->point.get_edge_of_flight_line() == 1)
                        {
                            spdPulse->edgeFlightLineFlag = SPD_WITH_SCAN;
                        }
                        else
                        {
                            spdPulse->edgeFlightLineFlag = SPD_SCAN_END;
                        }
                        
                        if(lasreader->point.get_scan_direction_flag() == 1)
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
                        spdPulse->user = lasreader->point.get_scan_angle_rank() + 90;
                                                
                        spdPulse->sourceID = lasreader->point.get_point_source_ID();
                        
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
                        //std::cerr << "p.GetReturnNumber() = " << p.GetReturnNumber() << std::endl;
                        //std::cerr << "p.GetNumberOfReturns() = " << p.GetNumberOfReturns() << std::endl;
                        std::cerr << "Warning: Point ignored. It is the first in pulse but has a return number greater than 1.\n";
                    }					
				}
				
				lasreader->close();
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
            // Open LAS file
            LASreadOpener lasreadopener;
            LASreader* lasreader = lasreadopener.open(inputFile.c_str());
            
            if(lasreader != 0)
            {
                // Get header
                LASheader *header = &lasreader->header;
                
                spdFile->setFileSignature(header->file_signature);
                spdFile->setSystemIdentifier(header->system_identifier);
				
				if(spdFile->getSpatialReference() == "")
				{
                    //FIXME: Need to get spatial information from LAS and convert to WKT (don't think LASlib does this)
                    /*liblas::SpatialReference const &lasSpatial = header.GetSRS();
                     std::string spatialRefProjWKT = lasSpatial.GetWKT();
                     spdFile->setSpatialReference(spatialRefProjWKT);*/
				}
				
				if(convertCoords)
				{
					this->initCoordinateSystemTransformation(spdFile);
				}
				
				boost::uint_fast64_t reportedNumOfPts = header->number_of_point_records;
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
				while (lasreader->read_point())
				{
					if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
					{
						std::cout << "." << feedbackCounter << "." << std::flush;
						feedbackCounter += 10;
					}
					
                    if(lasreader->point.get_return_number() <= 1)
                    {
                        spdPt = this->createSPDPoint(lasreader->point);
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
                        spdPulse->numberOfReturns = lasreader->point.get_number_of_returns();
                        if(spdPulse->numberOfReturns == 0)
                        {
                            spdPulse->numberOfReturns = 1;
                        }
                        if(spdPulse->numberOfReturns > 0)
                        {
                            spdPulse->pts->reserve(spdPulse->numberOfReturns);
                        }
                        
                        //std::cout << "Start Pulse (Num Returns = " << p.GetNumberOfReturns() << "): p.GetReturnNumber() = " << p.GetReturnNumber() << std::endl;
                        
                        if(lasreader->point.get_return_number() == 0)
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
                            if(lasreader->read_point())
                            {
                                if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
                                {
                                    std::cout << "." << feedbackCounter << "." << std::flush;
                                    feedbackCounter += 10;
                                }
                                
                                //std::cout << "\tIn Pulse (Num Returns = " << pt.get_NumberOfReturns() << "): pt.get_ReturnNumber() = " << pt.get_ReturnNumber() << std::endl;
                                
                                if(nPtIdx != lasreader->point.get_return_number())
                                {
                                    // FIXME: Could this error could be tidied up. Get it a lot with our ALSPP produced LAS files
                                    /*std::cerr << "Start Pulse (Num Returns = " << spdPulse->numberOfReturns << "): p.GetReturnNumber() = " << p.GetReturnNumber() << std::endl;
                                     std::cerr << "\tIn Pulse (Num Returns = " << pt.get_NumberOfReturns() << "): pt.get_ReturnNumber() = " << pt.get_ReturnNumber() << std::endl;
                                     std::cerr << "The return number was: " << pt.get_ReturnNumber() << std::endl;
                                     std::cerr << "The next return number should have been: " << nPtIdx << std::endl;*/
                                    spdPulse->numberOfReturns = i+1;
                                    //throw SPDIOException("Error in point numbering when building pulses.");
                                    break;
                                }
                                
                                spdPt = this->createSPDPoint(lasreader->point);

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
                        
                        if(lasreader->point.get_edge_of_flight_line() == 1)
                        {
                            spdPulse->edgeFlightLineFlag = SPD_WITH_SCAN;
                        }
                        else
                        {
                            spdPulse->edgeFlightLineFlag = SPD_SCAN_END;
                        }
                        
                        if(lasreader->point.get_scan_direction_flag() == 1)
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
                        spdPulse->user = lasreader->point.get_scan_angle_rank() + 90;
                        
                        spdPulse->sourceID = lasreader->point.get_point_source_ID();
                        
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
                        //std::cerr << "p.GetReturnNumber() = " << p.GetReturnNumber() << std::endl;
                        //std::cerr << "p.GetNumberOfReturns() = " << p.GetNumberOfReturns() << std::endl;
                        std::cerr << "Warning: Point ignored. It is the first in pulse but has a return number greater than 1.\n";
                    }
					
				}
				
                lasreader->close();
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
    
	SPDPoint* SPDLASFileImporter::createSPDPoint(LASpoint const& pt)throw(SPDIOException)
	{
		try 
		{
			SPDPointUtils spdPtUtils;
			SPDPoint *spdPt = new SPDPoint();
			spdPtUtils.initSPDPoint(spdPt);
            double x = pt.get_X();
            double y = pt.get_Y();
            double z = pt.get_Z();
            
            if(convertCoords)
            {
                this->transformCoordinateSystem(&x, &y, &z);
            }
            
            spdPt->x = x;
            spdPt->y = y;
            spdPt->z = z;
            spdPt->amplitudeReturn = pt.get_intensity();
            spdPt->user = pt.get_user_data();
			
            unsigned int lasClass = pt.get_classification();
            
            switch (lasClass)
            {
                case 0:
                    spdPt->classification = SPD_CREATED;
                    break;
                case 1:
                    spdPt->classification = SPD_UNCLASSIFIED;
                    break;
                case 2:
                    spdPt->classification = SPD_GROUND;
                    break;
                case 3:
                    spdPt->classification = SPD_LOW_VEGETATION;
                    break;
                case 4:
                    spdPt->classification = SPD_MEDIUM_VEGETATION;
                    break;
                case 5:
                    spdPt->classification = SPD_HIGH_VEGETATION;
                    break;
                case 6:
                    spdPt->classification = SPD_BUILDING;
                    break;
                case 7:
                    spdPt->classification = SPD_CREATED;
                    spdPt->lowPoint = SPD_TRUE;
                    break;
                case 8:
                    spdPt->classification = SPD_CREATED;
                    spdPt->modelKeyPoint = SPD_TRUE;
                    break;
                case 9:
                    spdPt->classification = SPD_WATER;
                    break;
                case 12:
                    spdPt->classification = SPD_CREATED;
                    spdPt->overlap = SPD_TRUE;
                    break;
                default:
                    spdPt->classification = SPD_CREATED;
                    if(!classWarningGiven)
                    {
                        std::cerr << "\nWARNING: The class ID " << lasClass<< " was not recognised - check the classes points were allocated too." << std::endl;
                        classWarningGiven = true;
                    }
                    break;
            }
			
            // Get array of RBG values (of type U16 in LASlib typedef)
            const unsigned short *rgb = pt.get_rgb();
            
            spdPt->red = rgb[0];
            spdPt->green = rgb[1];
            spdPt->blue = rgb[2];
            
            spdPt->returnID = pt.get_return_number();
            spdPt->gpsTime = pt.get_gps_time();
			
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
            // Open LAS file
            LASreadOpener lasreadopener;
            LASreader* lasreader = lasreadopener.open(inputFile.c_str());
            
            if(lasreader != 0)
			{
                // Get header
                LASheader *header = &lasreader->header;
                
                spdFile->setFileSignature(header->file_signature);
                spdFile->setSystemIdentifier(header->system_identifier);
                
                if(spdFile->getSpatialReference() == "")
                {
                    //FIXME: Need to get spatial information from LAS and convert to WKT (don't think LASlib does this)
                    /*liblas::SpatialReference const &lasSpatial = header.GetSRS();
                     std::string spatialRefProjWKT = lasSpatial.GetWKT();
                     spdFile->setSpatialReference(spatialRefProjWKT);*/
                }
                
                if(convertCoords)
                {
                    this->initCoordinateSystemTransformation(spdFile);
                }
                
                boost::uint_fast64_t reportedNumOfPts = header->number_of_point_records;
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
                while (lasreader->read_point())
				{
					//std::cout << numOfPoints << std::endl;
					if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
					{
						std::cout << "." << feedbackCounter << "." << std::flush;
						feedbackCounter += 10;
					}
					
					spdPt = this->createSPDPoint(lasreader->point);
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
					spdPulse->numberOfReturns = lasreader->point.get_number_of_returns();
                    if(spdPulse->numberOfReturns == 0)
                    {
                        spdPulse->numberOfReturns = 1;
                    }
                    if(spdPulse->numberOfReturns > 0)
                    {
                        spdPulse->pts->reserve(spdPulse->numberOfReturns);
                    }
                    
                    //std::cout << "Start Pulse (Num Returns = " << p.GetNumberOfReturns() << "): p.GetReturnNumber() = " << p.GetReturnNumber() << std::endl;
                    
                    if(lasreader->point.get_return_number() == 0)
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
						if(lasreader->read_point())
						{
							if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
							{
								std::cout << "." << feedbackCounter << "." << std::flush;
								feedbackCounter += 10;
							}
                            
                            //std::cout << "\tIn Pulse (Num Returns = " << pt.get_NumberOfReturns() << "): pt.get_ReturnNumber() = " << pt.get_ReturnNumber() << std::endl;
                            
                            if(nPtIdx != lasreader->point.get_return_number())
                            {
                                // FIXME: Could this error could be tidied up. Get it a lot with our ALSPP produced LAS files
                                /*std::cerr << "Start Pulse (Num Returns = " << p.GetNumberOfReturns() << "): p.GetReturnNumber() = " << p.GetReturnNumber() << std::endl;
                                std::cerr << "\tIn Pulse (Num Returns = " << pt.get_NumberOfReturns() << "): pt.get_ReturnNumber() = " << pt.get_ReturnNumber() << std::endl;
                                std::cerr << "The return number was: " << pt.get_ReturnNumber() << std::endl;
                                std::cerr << "The next return number should have been: " << nPtIdx << std::endl;*/
                                throw SPDIOException("Error in point numbering when building pulses.");
                            }
                            
							spdPt = this->createSPDPoint(lasreader->point);
							
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
                    if(lasreader->point.get_edge_of_flight_line() == 1)
                    {
                        spdPulse->edgeFlightLineFlag = SPD_WITH_SCAN;
                    }
                    else
                    {
                        spdPulse->edgeFlightLineFlag = SPD_SCAN_END;
                    }
                    
                    if(lasreader->point.get_scan_direction_flag() == 1)
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
                    spdPulse->user = lasreader->point.get_scan_angle_rank() + 90;
                    
                    spdPulse->sourceID = lasreader->point.get_point_source_ID();
					
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
				
                lasreader->close();
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
            // Open LAS file
            LASreadOpener lasreadopener;
            LASreader* lasreader = lasreadopener.open(inputFile.c_str());
            
            if(lasreader != 0)
            {
                // Get header
                LASheader *header = &lasreader->header;
                
                spdFile->setFileSignature(header->file_signature);
                spdFile->setSystemIdentifier(header->system_identifier);
            				
                if(spdFile->getSpatialReference() == "")
                {
                    //FIXME: Need to get spatial information from LAS and convert to WKT (don't think LASlib does this)
                    /*liblas::SpatialReference const &lasSpatial = header.GetSRS();
                     std::string spatialRefProjWKT = lasSpatial.GetWKT();
                     spdFile->setSpatialReference(spatialRefProjWKT);*/
                }
                
                if(convertCoords)
                {
                    this->initCoordinateSystemTransformation(spdFile);
                }
                
                pulses->reserve(header->number_of_point_records);
                
                boost::uint_fast64_t reportedNumOfPts = header->number_of_point_records;
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
				while (lasreader->read_point())
				{
					//std::cout << numOfPoints << std::endl;
					if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
					{
						std::cout << "." << feedbackCounter << "." << std::flush;
						feedbackCounter += 10;
					}
					
                    spdPt = this->createSPDPoint(lasreader->point);
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
					spdPulse->numberOfReturns = lasreader->point.get_number_of_returns();
                    if(spdPulse->numberOfReturns == 0)
                    {
                        spdPulse->numberOfReturns = 1;
                    }
                    if(spdPulse->numberOfReturns > 0)
                    {
                        spdPulse->pts->reserve(spdPulse->numberOfReturns);
                    }
                    
                    //std::cout << "Start Pulse (Num Returns = " << p.GetNumberOfReturns() << "): p.GetReturnNumber() = " << p.GetReturnNumber() << std::endl;
                    
                    if(lasreader->point.get_return_number() == 0)
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
						if(lasreader->read_point())
						{
							if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
							{
								std::cout << "." << feedbackCounter << "." << std::flush;
								feedbackCounter += 10;
							}
                            
                            //std::cout << "\tIn Pulse (Num Returns = " << pt.get_NumberOfReturns() << "): pt.get_ReturnNumber() = " << pt.get_ReturnNumber() << std::endl;
                            
                            if(nPtIdx != lasreader->point.get_return_number())
                            {
                                /*std::cerr << "Start Pulse (Num Returns = " << p.GetNumberOfReturns() << "): p.GetReturnNumber() = " << p.GetReturnNumber() << std::endl;
                                std::cerr << "\tIn Pulse (Num Returns = " << pt.get_NumberOfReturns() << "): pt.get_ReturnNumber() = " << pt.get_ReturnNumber() << std::endl;
                                std::cerr << "The return number was: " << pt.get_ReturnNumber() << std::endl;
                                std::cerr << "The next return number should have been: " << nPtIdx << std::endl;*/
                                throw SPDIOException("Error in point numbering when building pulses.");
                            }
                            
							spdPt = this->createSPDPoint(lasreader->point);
							
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

                    if(lasreader->point.get_edge_of_flight_line() == 1)
                    {
                        spdPulse->edgeFlightLineFlag = SPD_WITH_SCAN;
                    }
                    else
                    {
                        spdPulse->edgeFlightLineFlag = SPD_SCAN_END;
                    }
                    
                    if(lasreader->point.get_scan_direction_flag() == 1)
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
                    spdPulse->user = lasreader->point.get_scan_angle_rank() + 90;
                    
                    spdPulse->sourceID = lasreader->point.get_point_source_ID();
                    
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
				
				lasreader->close();
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
            // Open LAS file
            LASreadOpener lasreadopener;
            LASreader* lasreader = lasreadopener.open(inputFile.c_str());
            
            if(lasreader != 0)
			{
                // Get header
                LASheader *header = &lasreader->header;
                
                spdFile->setFileSignature(header->file_signature);
                spdFile->setSystemIdentifier(header->system_identifier);
            				
                if(spdFile->getSpatialReference() == "")
                {
                    //FIXME: Need to get spatial information from LAS and convert to WKT (don't think LASlib does this)
                    /*liblas::SpatialReference const &lasSpatial = header.GetSRS();
                     std::string spatialRefProjWKT = lasSpatial.GetWKT();
                     spdFile->setSpatialReference(spatialRefProjWKT);*/
                }
                
                if(convertCoords)
                {
                    this->initCoordinateSystemTransformation(spdFile);
                }
				
				boost::uint_fast64_t reportedNumOfPts = header->number_of_point_records;
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
				while (lasreader->read_point())
				{
					if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
					{
						std::cout << "." << feedbackCounter << "." << std::flush;
						feedbackCounter += 10;
					}
					
                    spdPt = this->createSPDPoint(lasreader->point);
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
					spdPulse->numberOfReturns = lasreader->point.get_number_of_returns();
                    if(spdPulse->numberOfReturns == 0)
                    {
                        spdPulse->numberOfReturns = 1;
                    }
                    if(spdPulse->numberOfReturns > 0)
                    {
                        spdPulse->pts->reserve(spdPulse->numberOfReturns);
                    }
                    
                    //std::cout << "Start Pulse (Num Returns = " << p.GetNumberOfReturns() << "): p.GetReturnNumber() = " << p.GetReturnNumber() << std::endl;
                    
                    if(lasreader->point.get_return_number() == 0)
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
						if(lasreader->read_point())
						{
							if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
							{
								std::cout << "." << feedbackCounter << "." << std::flush;
								feedbackCounter += 10;
							}
                            
                            //std::cout << "\tIn Pulse (Num Returns = " << pt.get_NumberOfReturns() << "): pt.get_ReturnNumber() = " << pt.get_ReturnNumber() << std::endl;
                            
                            if(nPtIdx != lasreader->point.get_return_number())
                            {
                                /*std::cerr << "Start Pulse (Num Returns = " << spdPulse->numberOfReturns << "): p.GetReturnNumber() = " << p.GetReturnNumber() << std::endl;
                                std::cerr << "\tIn Pulse (Num Returns = " << pt.get_NumberOfReturns() << "): pt.get_ReturnNumber() = " << pt.get_ReturnNumber() << std::endl;
                                std::cerr << "The return number was: " << pt.get_ReturnNumber() << std::endl;
                                std::cerr << "The next return number should have been: " << nPtIdx << std::endl;*/
                                throw SPDIOException("Error in point numbering when building pulses.");
                            }
                            
							spdPt = this->createSPDPoint(lasreader->point);
							
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
					
                    if(lasreader->point.get_edge_of_flight_line() == 1)
                    {
                        spdPulse->edgeFlightLineFlag = SPD_WITH_SCAN;
                    }
                    else
                    {
                        spdPulse->edgeFlightLineFlag = SPD_SCAN_END;
                    }
                    
                    if(lasreader->point.get_scan_direction_flag() == 1)
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
                    spdPulse->user = lasreader->point.get_scan_angle_rank() + 90;
                    
                    spdPulse->sourceID = lasreader->point.get_point_source_ID();
					
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
				
				lasreader->close();
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
    
	SPDPoint* SPDLASFileImporterStrictPulses::createSPDPoint(LASpoint const& pt)throw(SPDIOException)
	{
		try
		{
			SPDPointUtils spdPtUtils;
			SPDPoint *spdPt = new SPDPoint();
			spdPtUtils.initSPDPoint(spdPt);
			double x = pt.get_X();
			double y = pt.get_Y();
			double z = pt.get_Z();
			
			if(convertCoords)
			{
				this->transformCoordinateSystem(&x, &y, &z);
			}
			
			spdPt->x = x;
			spdPt->y = y;
			spdPt->z = z;
			spdPt->amplitudeReturn = pt.get_intensity();
			spdPt->user = pt.get_user_data();
			
			unsigned int lasClass = pt.get_classification();
            
            switch (lasClass)
            {
                case 0:
                    spdPt->classification = SPD_CREATED;
                    break;
                case 1:
                    spdPt->classification = SPD_UNCLASSIFIED;
                    break;
                case 2:
                    spdPt->classification = SPD_GROUND;
                    break;
                case 3:
                    spdPt->classification = SPD_LOW_VEGETATION;
                    break;
                case 4:
                    spdPt->classification = SPD_MEDIUM_VEGETATION;
                    break;
                case 5:
                    spdPt->classification = SPD_HIGH_VEGETATION;
                    break;
                case 6:
                    spdPt->classification = SPD_BUILDING;
                    break;
                case 7:
                    spdPt->classification = SPD_CREATED;
                    spdPt->lowPoint = SPD_TRUE;
                    break;
                case 8:
                    spdPt->classification = SPD_CREATED;
                    spdPt->modelKeyPoint = SPD_TRUE;
                    break;
                case 9:
                    spdPt->classification = SPD_WATER;
                    break;
                case 12:
                    spdPt->classification = SPD_CREATED;
                    spdPt->overlap = SPD_TRUE;
                    break;
                default:
                    spdPt->classification = SPD_CREATED;
                    if(!classWarningGiven)
                    {
                        std::cerr << "\nWARNING: The class ID " << lasClass<< " was not recognised - check the classes points were allocated too." << std::endl;
                        classWarningGiven = true;
                    }
                    break;
            }
            // Get array of RBG values (of type U16 in LASlib typedef)
            const unsigned short *rgb = pt.get_rgb();
            
			spdPt->red = rgb[0];
			spdPt->green = rgb[1];
			spdPt->blue = rgb[2];
			
			spdPt->returnID = pt.get_return_number();
			spdPt->gpsTime = pt.get_gps_time();
			
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
            // Open LAS file
            LASreadOpener lasreadopener;
            LASreader* lasreader = lasreadopener.open(inputFile.c_str());
            
            if(lasreader != 0)
			{
                // Get header
                LASheader *header = &lasreader->header;
                
                spdFile->setFileSignature(header->file_signature);
                spdFile->setSystemIdentifier(header->system_identifier);
            				
                if(spdFile->getSpatialReference() == "")
                {
                    //FIXME: Need to get spatial information from LAS and convert to WKT (don't think LASlib does this)
                    /*liblas::SpatialReference const &lasSpatial = header.GetSRS();
                     std::string spatialRefProjWKT = lasSpatial.GetWKT();
                     spdFile->setSpatialReference(spatialRefProjWKT);*/
                }
                
                if(convertCoords)
                {
                    this->initCoordinateSystemTransformation(spdFile);
                }
				
				boost::uint_fast64_t reportedNumOfPts = header->number_of_point_records;
				boost::uint_fast64_t feedback = reportedNumOfPts/10;
				unsigned int feedbackCounter = 0;
				
				SPDPoint *spdPt = NULL;
				SPDPulse *spdPulse = NULL;               
				
				std::cout << "Started (Read Data) ." << std::flush;
				while (lasreader->read_point())
				{
					//std::cout << numOfPoints << std::endl;
					if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
					{
						std::cout << "." << feedbackCounter << "." << std::flush;
						feedbackCounter += 10;
					}
					
					spdPt = this->createSPDPoint(lasreader->point);
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
					spdPulse->numberOfReturns = lasreader->point.get_number_of_returns();
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

                    if(lasreader->point.get_edge_of_flight_line() == 1)
                    {
                        spdPulse->edgeFlightLineFlag = SPD_WITH_SCAN;
                    }
                    else
                    {
                        spdPulse->edgeFlightLineFlag = SPD_SCAN_END;
                    }
                    
                    if(lasreader->point.get_scan_direction_flag() == 1)
                    {
                        spdPulse->scanDirectionFlag = SPD_POSITIVE;
                    }
                    else
                    {
                        spdPulse->scanDirectionFlag = SPD_NEGATIVE;
                    }
                    
                    spdPulse->zenith = 0.0;
                    spdPulse->azimuth = 0.0;
                    spdPulse->user = lasreader->point.get_scan_angle_rank() + 90;
                    
                    spdPulse->sourceID = lasreader->point.get_point_source_ID();
                    
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
				
				lasreader->close();
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
            // Open LAS file
            LASreadOpener lasreadopener;
            LASreader* lasreader = lasreadopener.open(inputFile.c_str());
            
            if(lasreader != 0)
            {
                // Get header
                LASheader *header = &lasreader->header;
                
                spdFile->setFileSignature(header->file_signature);
                spdFile->setSystemIdentifier(header->system_identifier);
                
                if(spdFile->getSpatialReference() == "")
                {
                    //FIXME: Need to get spatial information from LAS and convert to WKT (don't think LASlib does this)
                    /*liblas::SpatialReference const &lasSpatial = header.GetSRS();
                     std::string spatialRefProjWKT = lasSpatial.GetWKT();
                     spdFile->setSpatialReference(spatialRefProjWKT);*/
                }
                
                if(convertCoords)
                {
                    this->initCoordinateSystemTransformation(spdFile);
                }
				
				boost::uint_fast64_t reportedNumOfPts = header->number_of_point_records;
				boost::uint_fast64_t feedback = reportedNumOfPts/10;
				unsigned int feedbackCounter = 0;
				
				SPDPoint *spdPt = NULL;
				SPDPulse *spdPulse = NULL;
				
				std::cout << "Started (Read Data) ." << std::flush;
				while (lasreader->read_point())
				{
					//std::cout << numOfPoints << std::endl;
					if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
					{
						std::cout << "." << feedbackCounter << "." << std::flush;
						feedbackCounter += 10;
					}
					
					spdPt = this->createSPDPoint(lasreader->point);
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
					spdPulse->numberOfReturns = lasreader->point.get_number_of_returns();
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
                    
                    if(lasreader->point.get_edge_of_flight_line() == 1)
                    {
                        spdPulse->edgeFlightLineFlag = SPD_WITH_SCAN;
                    }
                    else
                    {
                        spdPulse->edgeFlightLineFlag = SPD_SCAN_END;
                    }
                    
                    if(lasreader->point.get_scan_direction_flag() == 1)
                    {
                        spdPulse->scanDirectionFlag = SPD_POSITIVE;
                    }
                    else
                    {
                        spdPulse->scanDirectionFlag = SPD_NEGATIVE;
                    }
                    
                    spdPulse->zenith = 0.0;
                    spdPulse->azimuth = 0.0;
                    spdPulse->user = lasreader->point.get_scan_angle_rank() + 90;

					spdPulse->sourceID = lasreader->point.get_point_source_ID();
                    
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
				
				lasreader->close();
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
            // Open LAS file
            LASreadOpener lasreadopener;
            LASreader* lasreader = lasreadopener.open(inputFile.c_str());
            
            if(lasreader != 0)
            {
                // Get header
                LASheader *header = &lasreader->header;
                
                spdFile->setFileSignature(header->file_signature);
                spdFile->setSystemIdentifier(header->system_identifier);
                
                if(spdFile->getSpatialReference() == "")
                {
                    //FIXME: Need to get spatial information from LAS and convert to WKT (don't think LASlib does this)
                    /*liblas::SpatialReference const &lasSpatial = header.GetSRS();
                     std::string spatialRefProjWKT = lasSpatial.GetWKT();
                     spdFile->setSpatialReference(spatialRefProjWKT);*/
                }
                
                if(convertCoords)
                {
                    this->initCoordinateSystemTransformation(spdFile);
                }
				
				boost::uint_fast64_t reportedNumOfPts = header->number_of_point_records;
				boost::uint_fast64_t feedback = reportedNumOfPts/10;
				unsigned int feedbackCounter = 0;
				
				SPDPoint *spdPt = NULL;
				SPDPulse *spdPulse = NULL;
                                
				std::cout << "Started (Read Data) ." << std::flush;
				while (lasreader->read_point())
				{
					if((reportedNumOfPts > 10) && ((numOfPoints % feedback) == 0))
					{
						std::cout << "." << feedbackCounter << "." << std::flush;
						feedbackCounter += 10;
					}
					
					spdPt = this->createSPDPoint(lasreader->point);
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
					
                    if(lasreader->point.get_edge_of_flight_line() == 1)
                    {
                        spdPulse->edgeFlightLineFlag = SPD_WITH_SCAN;
                    }
                    else
                    {
                        spdPulse->edgeFlightLineFlag = SPD_SCAN_END;
                    }
                    
                    if(lasreader->point.get_scan_direction_flag() == 1)
                    {
                        spdPulse->scanDirectionFlag = SPD_POSITIVE;
                    }
                    else
                    {
                        spdPulse->scanDirectionFlag = SPD_NEGATIVE;
                    }
                    
                    spdPulse->zenith = 0.0;
                    spdPulse->azimuth = 0.0;
                    spdPulse->user = lasreader->point.get_scan_angle_rank() + 90;
                    
                    spdPulse->sourceID = lasreader->point.get_point_source_ID();
					
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
				
                lasreader->close();
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
    
	SPDPoint* SPDLASFileNoPulsesImporter::createSPDPoint(LASpoint const& pt)throw(SPDIOException)
	{
		try
		{
			SPDPointUtils spdPtUtils;
			SPDPoint *spdPt = new SPDPoint();
			spdPtUtils.initSPDPoint(spdPt);
			double x = pt.get_X();
			double y = pt.get_Y();
			double z = pt.get_Z();
			
			if(convertCoords)
			{
				this->transformCoordinateSystem(&x, &y, &z);
			}
			
			spdPt->x = x;
			spdPt->y = y;
			spdPt->z = z;
			spdPt->amplitudeReturn = pt.get_intensity();
			spdPt->user = pt.get_user_data();
			
            spdPt->x = x;
            spdPt->y = y;
            spdPt->z = z;
            spdPt->amplitudeReturn = pt.get_intensity();
            spdPt->user = pt.get_user_data();
            
            unsigned int lasClass = pt.get_classification();
            
            switch (lasClass)
            {
                case 0:
                    spdPt->classification = SPD_CREATED;
                    break;
                case 1:
                    spdPt->classification = SPD_UNCLASSIFIED;
                    break;
                case 2:
                    spdPt->classification = SPD_GROUND;
                    break;
                case 3:
                    spdPt->classification = SPD_LOW_VEGETATION;
                    break;
                case 4:
                    spdPt->classification = SPD_MEDIUM_VEGETATION;
                    break;
                case 5:
                    spdPt->classification = SPD_HIGH_VEGETATION;
                    break;
                case 6:
                    spdPt->classification = SPD_BUILDING;
                    break;
                case 7:
                    spdPt->classification = SPD_CREATED;
                    spdPt->lowPoint = SPD_TRUE;
                    break;
                case 8:
                    spdPt->classification = SPD_CREATED;
                    spdPt->modelKeyPoint = SPD_TRUE;
                    break;
                case 9:
                    spdPt->classification = SPD_WATER;
                    break;
                case 12:
                    spdPt->classification = SPD_CREATED;
                    spdPt->overlap = SPD_TRUE;
                    break;
                default:
                    spdPt->classification = SPD_CREATED;
                    if(!classWarningGiven)
                    {
                        std::cerr << "\nWARNING: The class ID " << lasClass<< " was not recognised - check the classes points were allocated too." << std::endl;
                        classWarningGiven = true;
                    }
                    break;
            }
			
            // Get array of RBG values (of type U16 in LASlib typedef)
            const unsigned short *rgb = pt.get_rgb();
            
            spdPt->red = rgb[0];
            spdPt->green = rgb[1];
            spdPt->blue = rgb[2];
            
            spdPt->returnID = pt.get_return_number();
            spdPt->gpsTime = pt.get_gps_time();
			
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
