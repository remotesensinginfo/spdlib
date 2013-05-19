/*
 *  SPDLASFileExporter.cpp
 *  SPDLIB
 *
 *  Created by Pete Bunting on 17/02/2011.
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

#include "spd/SPDLASFileExporter.h"
#include "spd/spd-config.h"


namespace spdlib
{
	SPDLASFileExporter::SPDLASFileExporter() : SPDDataExporter("LAS")
	{
		
	}
	
	SPDLASFileExporter::SPDLASFileExporter(const SPDDataExporter &dataExporter) throw(SPDException) : SPDDataExporter(dataExporter)
	{
		if(fileOpened)
		{
			throw SPDException("Cannot make a copy of a file exporter when a file is open.");
		}
	}
	
	SPDLASFileExporter& SPDLASFileExporter::operator=(const SPDLASFileExporter& dataExporter) throw(SPDException)
	{
		if(fileOpened)
		{
			throw SPDException("Cannot make a copy of a file exporter when a file is open.");
		}
		
		this->spdFile = dataExporter.spdFile;
		this->outputFile = dataExporter.outputFile;
		return *this;
	}
    
    SPDDataExporter* SPDLASFileExporter::getInstance()
    {
        return new SPDLASFileExporter();
    }
	
	bool SPDLASFileExporter::open(SPDFile *spdFile, std::string outputFile) throw(SPDIOException)
	{
		try 
		{
            if (spdFile->getDecomposedPtDefined() == SPD_TRUE)
            {
                std::cout << "Decomposed Point data found - Note. Widths are not stored.\n";
            }
			else if(spdFile->getDiscretePtDefined() == SPD_TRUE)
			{
				std::cout << "Point data found\n";
			}
            else
            {
                throw SPDIOException("This writer can only export point data.");
            }
            
            std::cout << "Outputting to " << outputFile << std::endl;
            
			outDataStream = new std::fstream();
			outDataStream->open(outputFile.c_str(), std::ios::out | std::ios::binary);

			liblas::Header lasFileHeader;
            lasFileHeader.SetDataFormatId(liblas::ePointFormat3);
            
			if(spdFile->getSpatialReference() != "")
			{
				liblas::SpatialReference lasSpatRef;
				lasSpatRef.SetWKT(spdFile->getSpatialReference());
				lasFileHeader.SetSRS(lasSpatRef);
			}
			lasFileHeader.SetCompressed(false);
            lasFileHeader.SetScale(0.01,0.01,0.01);
            lasFileHeader.SetMin(spdFile->getXMin(),spdFile->getYMin(),spdFile->getZMin());
            lasFileHeader.SetMax(spdFile->getXMax(),spdFile->getYMax(),spdFile->getZMax());
            lasFileHeader.SetSoftwareId(SPDLIB_PACKAGE_STRING);
            lasFileHeader.SetSystemId("EXPORT");
			lasWriter = new liblas::Writer(*outDataStream, lasFileHeader);
			
			fileOpened = true;
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
        catch (std::exception const& e)
        {
            throw SPDIOException(e.what());
        }
		
        finalisedClosed = false;
        
		return fileOpened;
	}
    
    bool SPDLASFileExporter::reopen(SPDFile *spdFile, std::string outputFile) throw(SPDIOException)
    {
        throw SPDIOException("No reopen option available.");
    }
	
	void SPDLASFileExporter::writeDataColumn(std::list<SPDPulse*> *plsIn, boost::uint_fast32_t col, boost::uint_fast32_t row)throw(SPDIOException)
	{
		SPDPulseUtils pulseUtils;
		SPDPointUtils pointUtils;
		
		if(!fileOpened)
		{
			throw SPDIOException("Output file not open, cannot write to the file.");
		}
		
		try 
		{			
			std::vector<SPDPoint*>::iterator iterPts;
			std::list<SPDPulse*>::iterator iterInPls;
            if(plsIn->size() > 0)
            {
                for(iterInPls = plsIn->begin(); iterInPls != plsIn->end(); ++iterInPls)
                {
                    if((*iterInPls)->numberOfReturns)
                    {
                        for(iterPts = (*iterInPls)->pts->begin(); iterPts != (*iterInPls)->pts->end(); ++iterPts)
                        {
                            liblas::Point point;
                            //cout << "PT (list): [" << (*iterPts)->x << ", " << (*iterPts)->y << ", " << (*iterPts)->z << "]\n";
                            point.SetCoordinates((*iterPts)->x/0.01, (*iterPts)->y/0.01, (*iterPts)->z/0.01);
                            point.SetIntensity((*iterPts)->amplitudeReturn);
                            point.SetReturnNumber((*iterPts)->returnID);                            
                            point.SetNumberOfReturns((*iterInPls)->numberOfReturns);
                            point.SetPointSourceID((*iterInPls)->sourceID);
                            point.SetScanAngleRank((*iterInPls)->zenith*180.0/3.141592653589793+0.5-180.0);
                            point.SetTime((*iterPts)->gpsTime/1000000.0);
                            point.SetUserData((*iterPts)->widthReturn);
                            point.SetColor(liblas::Color ((*iterPts)->red, (*iterPts)->blue, (*iterPts)->green));
                            
                            liblas::Classification lasClass;
                            if((*iterPts)->classification == SPD_CREATED)
                            {
                                lasClass.SetClass(point.eCreated);
                            }
                            else if((*iterPts)->classification == SPD_UNCLASSIFIED)
                            {
                                lasClass.SetClass(point.eUnclassified);
                            }
                            else if((*iterPts)->classification == SPD_GROUND)
                            {
                                lasClass.SetClass(point.eGround);
                            }
                            else if((*iterPts)->classification == SPD_LOW_VEGETATION)
                            {
                                lasClass.SetClass(point.eLowVegetation);
                            }
                            else if((*iterPts)->classification == SPD_MEDIUM_VEGETATION)
                            {
                                lasClass.SetClass(point.eMediumVegetation);
                            }
                            else if((*iterPts)->classification == SPD_HIGH_VEGETATION)
                            {
                                lasClass.SetClass(point.eHighVegetation);
                            }
                            else if((*iterPts)->classification == SPD_BUILDING)
                            {
                                lasClass.SetClass(point.eBuilding);
                            }
                            else if((*iterPts)->classification == SPD_WATER)
                            {
                                lasClass.SetClass(point.eWater);
                            }
                            else if((*iterPts)->lowPoint == SPD_TRUE)
                            {
                                lasClass.SetClass(point.eLowPoint);
                            }
                            else if((*iterPts)->modelKeyPoint == SPD_CREATED)
                            {
                                lasClass.SetClass(point.eModelKeyPoint);
                            }
                            else if((*iterPts)->overlap == SPD_CREATED)
                            {
                                lasClass.SetClass(point.eOverlapPoints);
                            }
                            point.SetClassification(lasClass);
                            
                            lasWriter->WritePoint(point);
                        }
                    }
                    SPDPulseUtils::deleteSPDPulse(*iterInPls);
                }
                plsIn->clear();
            }
			
		}
		catch(SPDIOException &e)
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
	
	void SPDLASFileExporter::writeDataColumn(std::vector<SPDPulse*> *plsIn, boost::uint_fast32_t col, boost::uint_fast32_t row)throw(SPDIOException)
	{
		SPDPulseUtils pulseUtils;
		SPDPointUtils pointUtils;
		
		if(!fileOpened)
		{
			throw SPDIOException("Output file not open, cannot write to the file.");
		}
		
		try 
		{			
			std::vector<SPDPoint*>::iterator iterPts;
			std::vector<SPDPulse*>::iterator iterInPls;
			if(plsIn->size() > 0)
            {
                for(iterInPls = plsIn->begin(); iterInPls != plsIn->end(); ++iterInPls)
                {
                    if((*iterInPls)->numberOfReturns)
                    {
                        for(iterPts = (*iterInPls)->pts->begin(); iterPts != (*iterInPls)->pts->end(); ++iterPts)
                        {
                            liblas::Point point;
                            //cout << "PT (list): [" << (*iterPts)->x << ", " << (*iterPts)->y << ", " << (*iterPts)->z << "]\n";
                            point.SetCoordinates((*iterPts)->x/0.01, (*iterPts)->y/0.01, (*iterPts)->z/0.01);
                            point.SetIntensity((*iterPts)->amplitudeReturn);
                            point.SetReturnNumber((*iterPts)->returnID);
                            point.SetNumberOfReturns((*iterInPls)->numberOfReturns);
                            point.SetPointSourceID((*iterInPls)->sourceID);
                            point.SetScanAngleRank((*iterInPls)->zenith*180.0/3.141592653589793+0.5-180.0);
                            point.SetTime((*iterPts)->gpsTime/1000000.0);
                            point.SetUserData((*iterPts)->widthReturn);
                            point.SetColor(liblas::Color ((*iterPts)->red, (*iterPts)->blue, (*iterPts)->green));
                            
                            
                            liblas::Classification lasClass;
                            if((*iterPts)->classification == SPD_CREATED)
                            {
                                lasClass.SetClass(point.eCreated);
                            }
                            else if((*iterPts)->classification == SPD_UNCLASSIFIED)
                            {
                                lasClass.SetClass(point.eUnclassified);
                            }
                            else if((*iterPts)->classification == SPD_GROUND)
                            {
                                lasClass.SetClass(point.eGround);
                            }
                            else if((*iterPts)->classification == SPD_LOW_VEGETATION)
                            {
                                lasClass.SetClass(point.eLowVegetation);
                            }
                            else if((*iterPts)->classification == SPD_MEDIUM_VEGETATION)
                            {
                                lasClass.SetClass(point.eMediumVegetation);
                            }
                            else if((*iterPts)->classification == SPD_HIGH_VEGETATION)
                            {
                                lasClass.SetClass(point.eHighVegetation);
                            }
                            else if((*iterPts)->classification == SPD_BUILDING)
                            {
                                lasClass.SetClass(point.eBuilding);
                            }
                            else if((*iterPts)->classification == SPD_WATER)
                            {
                                lasClass.SetClass(point.eWater);
                            }
                            else if((*iterPts)->lowPoint == SPD_TRUE)
                            {
                                lasClass.SetClass(point.eLowPoint);
                            }
                            else if((*iterPts)->modelKeyPoint == SPD_CREATED)
                            {
                                lasClass.SetClass(point.eModelKeyPoint);
                            }
                            else if((*iterPts)->overlap == SPD_CREATED)
                            {
                                lasClass.SetClass(point.eOverlapPoints);
                            }
                            point.SetClassification(lasClass);
                            
                            lasWriter->WritePoint(point);
                        }
                    }
                    SPDPulseUtils::deleteSPDPulse(*iterInPls);
                }
                plsIn->clear();
            }
			
		}
		catch(SPDIOException &e)
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
	
	void SPDLASFileExporter::finaliseClose() throw(SPDIOException)
	{
		if(!fileOpened)
		{
			throw SPDIOException("Output file not open, cannot finalise.");
		}
        
        if(!finalisedClosed)
        {
            delete lasWriter;
            
            outDataStream->flush();
            outDataStream->close();
        }
        finalisedClosed = true;
	}
	
	bool SPDLASFileExporter::requireGrid()
	{
		return false;
	}
	
	bool SPDLASFileExporter::needNumOutPts()
	{
		return false;
	}
	
	SPDLASFileExporter::~SPDLASFileExporter()
	{
		if(fileOpened)
		{
			try 
			{
				this->finaliseClose();
			}
			catch (SPDIOException &e) 
			{
				std::cerr << "WARNING: " << e.what() << std::endl;
			}
		}
	}
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    SPDLAZFileExporter::SPDLAZFileExporter() : SPDDataExporter("LAZ")
	{
		
	}
	
	SPDLAZFileExporter::SPDLAZFileExporter(const SPDDataExporter &dataExporter) throw(SPDException) : SPDDataExporter(dataExporter)
	{
		if(fileOpened)
		{
			throw SPDException("Cannot make a copy of a file exporter when a file is open.");
		}
	}
	
	SPDLAZFileExporter& SPDLAZFileExporter::operator=(const SPDLAZFileExporter& dataExporter) throw(SPDException)
	{
		if(fileOpened)
		{
			throw SPDException("Cannot make a copy of a file exporter when a file is open.");
		}
		
		this->spdFile = dataExporter.spdFile;
		this->outputFile = dataExporter.outputFile;
		return *this;
	}
    
    SPDDataExporter* SPDLAZFileExporter::getInstance()
    {
        return new SPDLAZFileExporter();
    }
	
	bool SPDLAZFileExporter::open(SPDFile *spdFile, std::string outputFile) throw(SPDIOException)
	{
		try
		{
            if (spdFile->getDecomposedPtDefined() == SPD_TRUE)
            {
                std::cout << "Decomposed Point data found - Note. Widths are not stored.\n";
            }
			else if(spdFile->getDiscretePtDefined() == SPD_TRUE)
			{
				std::cout << "Point data found\n";
			}
            else
            {
                throw SPDIOException("This writer can only export point data.");
            }
            
            std::cout << "Outputting to " << outputFile << std::endl;
            
			outDataStream = new std::fstream();
			outDataStream->open(outputFile.c_str(), std::ios::out | std::ios::binary);
            
			liblas::Header lasFileHeader;
            lasFileHeader.SetDataFormatId(liblas::ePointFormat3);
            
			if(spdFile->getSpatialReference() != "")
			{
				liblas::SpatialReference lasSpatRef;
				lasSpatRef.SetWKT(spdFile->getSpatialReference());
				lasFileHeader.SetSRS(lasSpatRef);
			}
			lasFileHeader.SetCompressed(true);
            lasFileHeader.SetScale(0.01,0.01,0.01);
            lasFileHeader.SetMin(spdFile->getXMin(),spdFile->getYMin(),spdFile->getZMin());
            lasFileHeader.SetMax(spdFile->getXMax(),spdFile->getYMax(),spdFile->getZMax());
            lasFileHeader.SetSoftwareId(SPDLIB_PACKAGE_STRING);
            lasFileHeader.SetSystemId("EXPORT");
			lasWriter = new liblas::Writer(*outDataStream, lasFileHeader);
			
			fileOpened = true;
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
        catch (std::exception const& e)
        {
            throw SPDIOException(e.what());
        }
		
        finalisedClosed = false;
        
		return fileOpened;
	}
    
    bool SPDLAZFileExporter::reopen(SPDFile *spdFile, std::string outputFile) throw(SPDIOException)
    {
        throw SPDIOException("No reopen option available.");
    }
	
	void SPDLAZFileExporter::writeDataColumn(std::list<SPDPulse*> *plsIn, boost::uint_fast32_t col, boost::uint_fast32_t row)throw(SPDIOException)
	{
		SPDPulseUtils pulseUtils;
		SPDPointUtils pointUtils;
		
		if(!fileOpened)
		{
			throw SPDIOException("Output file not open, cannot write to the file.");
		}
		
		try
		{
			std::vector<SPDPoint*>::iterator iterPts;
			std::list<SPDPulse*>::iterator iterInPls;
            if(plsIn->size() > 0)
            {
                for(iterInPls = plsIn->begin(); iterInPls != plsIn->end(); ++iterInPls)
                {
                    if((*iterInPls)->numberOfReturns)
                    {
                        for(iterPts = (*iterInPls)->pts->begin(); iterPts != (*iterInPls)->pts->end(); ++iterPts)
                        {
                            liblas::Point point;
                            //cout << "PT (list): [" << (*iterPts)->x << ", " << (*iterPts)->y << ", " << (*iterPts)->z << "]\n";
                            point.SetCoordinates((*iterPts)->x/0.01, (*iterPts)->y/0.01, (*iterPts)->z/0.01);
                            point.SetIntensity((*iterPts)->amplitudeReturn);
                            point.SetReturnNumber((*iterPts)->returnID);
                            point.SetNumberOfReturns((*iterInPls)->numberOfReturns);
                            point.SetPointSourceID((*iterInPls)->sourceID);
                            point.SetScanAngleRank((*iterInPls)->zenith*180.0/3.141592653589793+0.5-180.0);
                            point.SetTime((*iterPts)->gpsTime/1000000.0);
                            point.SetUserData((*iterPts)->widthReturn);
                            point.SetColor(liblas::Color ((*iterPts)->red, (*iterPts)->blue, (*iterPts)->green));
                            
                            liblas::Classification lasClass;
                            if((*iterPts)->classification == SPD_CREATED)
                            {
                                lasClass.SetClass(point.eCreated);
                            }
                            else if((*iterPts)->classification == SPD_UNCLASSIFIED)
                            {
                                lasClass.SetClass(point.eUnclassified);
                            }
                            else if((*iterPts)->classification == SPD_GROUND)
                            {
                                lasClass.SetClass(point.eGround);
                            }
                            else if((*iterPts)->classification == SPD_LOW_VEGETATION)
                            {
                                lasClass.SetClass(point.eLowVegetation);
                            }
                            else if((*iterPts)->classification == SPD_MEDIUM_VEGETATION)
                            {
                                lasClass.SetClass(point.eMediumVegetation);
                            }
                            else if((*iterPts)->classification == SPD_HIGH_VEGETATION)
                            {
                                lasClass.SetClass(point.eHighVegetation);
                            }
                            else if((*iterPts)->classification == SPD_BUILDING)
                            {
                                lasClass.SetClass(point.eBuilding);
                            }
                            else if((*iterPts)->classification == SPD_WATER)
                            {
                                lasClass.SetClass(point.eWater);
                            }
                            else if((*iterPts)->lowPoint == SPD_TRUE)
                            {
                                lasClass.SetClass(point.eLowPoint);
                            }
                            else if((*iterPts)->modelKeyPoint == SPD_CREATED)
                            {
                                lasClass.SetClass(point.eModelKeyPoint);
                            }
                            else if((*iterPts)->overlap == SPD_CREATED)
                            {
                                lasClass.SetClass(point.eOverlapPoints);
                            }
                            point.SetClassification(lasClass);
                            
                            lasWriter->WritePoint(point);
                        }
                    }
                    SPDPulseUtils::deleteSPDPulse(*iterInPls);
                }
                plsIn->clear();
            }
			
		}
		catch(SPDIOException &e)
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
	
	void SPDLAZFileExporter::writeDataColumn(std::vector<SPDPulse*> *plsIn, boost::uint_fast32_t col, boost::uint_fast32_t row)throw(SPDIOException)
	{
		SPDPulseUtils pulseUtils;
		SPDPointUtils pointUtils;
		
		if(!fileOpened)
		{
			throw SPDIOException("Output file not open, cannot write to the file.");
		}
		
		try
		{
			std::vector<SPDPoint*>::iterator iterPts;
			std::vector<SPDPulse*>::iterator iterInPls;
			if(plsIn->size() > 0)
            {
                for(iterInPls = plsIn->begin(); iterInPls != plsIn->end(); ++iterInPls)
                {
                    if((*iterInPls)->numberOfReturns)
                    {
                        for(iterPts = (*iterInPls)->pts->begin(); iterPts != (*iterInPls)->pts->end(); ++iterPts)
                        {
                            liblas::Point point;
                            //cout << "PT (list): [" << (*iterPts)->x << ", " << (*iterPts)->y << ", " << (*iterPts)->z << "]\n";
                            point.SetCoordinates((*iterPts)->x/0.01, (*iterPts)->y/0.01, (*iterPts)->z/0.01);
                            point.SetIntensity((*iterPts)->amplitudeReturn);
                            point.SetReturnNumber((*iterPts)->returnID);
                            point.SetNumberOfReturns((*iterInPls)->numberOfReturns);
                            point.SetPointSourceID((*iterInPls)->sourceID);
                            point.SetScanAngleRank((*iterInPls)->zenith*180.0/3.141592653589793+0.5-180.0);
                            point.SetTime((*iterPts)->gpsTime/1000000.0);
                            point.SetUserData((*iterPts)->widthReturn);
                            point.SetColor(liblas::Color ((*iterPts)->red, (*iterPts)->blue, (*iterPts)->green));
                            
                            
                            liblas::Classification lasClass;
                            if((*iterPts)->classification == SPD_CREATED)
                            {
                                lasClass.SetClass(point.eCreated);
                            }
                            else if((*iterPts)->classification == SPD_UNCLASSIFIED)
                            {
                                lasClass.SetClass(point.eUnclassified);
                            }
                            else if((*iterPts)->classification == SPD_GROUND)
                            {
                                lasClass.SetClass(point.eGround);
                            }
                            else if((*iterPts)->classification == SPD_LOW_VEGETATION)
                            {
                                lasClass.SetClass(point.eLowVegetation);
                            }
                            else if((*iterPts)->classification == SPD_MEDIUM_VEGETATION)
                            {
                                lasClass.SetClass(point.eMediumVegetation);
                            }
                            else if((*iterPts)->classification == SPD_HIGH_VEGETATION)
                            {
                                lasClass.SetClass(point.eHighVegetation);
                            }
                            else if((*iterPts)->classification == SPD_BUILDING)
                            {
                                lasClass.SetClass(point.eBuilding);
                            }
                            else if((*iterPts)->classification == SPD_WATER)
                            {
                                lasClass.SetClass(point.eWater);
                            }
                            else if((*iterPts)->lowPoint == SPD_TRUE)
                            {
                                lasClass.SetClass(point.eLowPoint);
                            }
                            else if((*iterPts)->modelKeyPoint == SPD_CREATED)
                            {
                                lasClass.SetClass(point.eModelKeyPoint);
                            }
                            else if((*iterPts)->overlap == SPD_CREATED)
                            {
                                lasClass.SetClass(point.eOverlapPoints);
                            }
                            point.SetClassification(lasClass);
                            
                            lasWriter->WritePoint(point);
                        }
                    }
                    SPDPulseUtils::deleteSPDPulse(*iterInPls);
                }
                plsIn->clear();
            }
			
		}
		catch(SPDIOException &e)
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
	
	void SPDLAZFileExporter::finaliseClose() throw(SPDIOException)
	{
		if(!fileOpened)
		{
			throw SPDIOException("Output file not open, cannot finalise.");
		}
        
        if(!finalisedClosed)
        {
            delete lasWriter;
            
            outDataStream->flush();
            outDataStream->close();
        }
        finalisedClosed = true;
	}
	
	bool SPDLAZFileExporter::requireGrid()
	{
		return false;
	}
	
	bool SPDLAZFileExporter::needNumOutPts()
	{
		return false;
	}
	
	SPDLAZFileExporter::~SPDLAZFileExporter()
	{
		if(fileOpened)
		{
			try
			{
				this->finaliseClose();
			}
			catch (SPDIOException &e)
			{
				std::cerr << "WARNING: " << e.what() << std::endl;
			}
		}
	}
    
    
    
    
    
    
    
    
}
