/*
 *  SPDConvertFormats.cpp
 *  spdlib_prj
 *
 *  Created by Pete Bunting on 13/10/2009.
 *  Copyright 2009 SPDLib. All rights reserved.
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

#include "spd/SPDConvertFormats.h"

namespace spdlib
{
	
	SPDConvertFormats::SPDConvertFormats()
	{
		
	}
	
	void SPDConvertFormats::convertInMemory(std::string input, std::string output, std::string inFormat, std::string schema, std::string outFormat, float binsize, std::string inSpatialRef, bool convertCoords, std::string outputProjWKT, boost::uint_fast16_t indexCoords, bool defineTL, double tlX, double tlY, bool defineOrigin, double originX, double originY, float originZ, bool useSphericIdx, bool usePolarIdx, bool useScanIdx, float waveNoiseThreshold, boost::uint_fast16_t waveformBitRes, boost::uint_fast16_t pointVersion, boost::uint_fast16_t pulseVersion, bool keepInMinExtent, bool exportZasH) 
	{        
		try 
		{
            std::cout.precision(12);
            
			SPDIOFactory ioFactory;
			
			SPDDataImporter *importer = ioFactory.getImporter(inFormat, convertCoords, outputProjWKT, schema, indexCoords, defineOrigin, originX, originY, originZ, waveNoiseThreshold);
			SPDDataExporter *exporter = ioFactory.getExporter(outFormat, exportZasH);
            exporter->setKeepMinExtent(keepInMinExtent);
			
			SPDFile *spdFile = new SPDFile(input);
			spdFile->setSpatialReference(inSpatialRef);
            spdFile->setPulseVersion(pulseVersion);
            spdFile->setPointVersion(pointVersion);
			
			if(exporter->requireGrid())
			{
                bool copySPDFile = true;
                if((inFormat == "SPD") & (outFormat == "SPD-SEQ"))
				{                    
					SPDFileReader *spdReader = (SPDFileReader*)importer;
					spdReader->readHeaderInfo(spdFile->getFilePath(), spdFile);
                    
                    if((binsize != 0) & (binsize != spdFile->getBinSize()))
                    {
                        copySPDFile = false;
                    }
                    else if(convertCoords)
                    {
                        copySPDFile = false;
                    }
                    else if(defineTL)
                    {
                        copySPDFile = false;
                    }
                    else if(defineOrigin)
                    {
                        copySPDFile = false;
                    }
                    else if(pointVersion != spdFile->getPointVersion())
                    {
                        copySPDFile = false;
                    }
                    else if(pulseVersion != spdFile->getPulseVersion())
                    {
                        copySPDFile = false;
                    }
				}
                else
                {
                    copySPDFile = false;
                }
                
                if(copySPDFile)
                {
                    SPDFile *spdFileOut = new SPDFile(output);
                    spdFileOut->copyAttributesFrom(spdFile);
                    this->copySPD2SPD(spdFile, spdFileOut);
                }
                else
                {
                    std::vector<SPDPulse*> *pulses = importer->readAllDataToVector(input, spdFile);
                    if(useSphericIdx)
                    {
                        spdFile->setIndexType(spdlib::SPD_SPHERICAL_IDX);
                    }
                    else if(usePolarIdx)
                    {
                        spdFile->setIndexType(spdlib::SPD_POLAR_IDX);
                    }
                    else if(useScanIdx)
                    {
                        spdFile->setIndexType(spdlib::SPD_SCAN_IDX);
                    }
                    else
                    {
                        spdFile->setIndexType(spdlib::SPD_CARTESIAN_IDX);
                    }
                    if(binsize > 0)
                    {
                        spdFile->setBinSize(binsize);
                    }
                    else if(compare_double(spdFile->getBinSize(), 0) & compare_double(binsize, 0))
                    {
                        throw SPDException("Bin size needs to be specified.");
                    }
                    
                    if((inFormat != "SPD") & (inFormat != "UPD"))
                    {
                        spdFile->setWaveformBitRes(waveformBitRes);
                    }
                    
                    if(defineTL & (spdFile->getIndexType() == spdlib::SPD_CARTESIAN_IDX))
                    {
                        if(tlX > spdFile->getXMin())
                        {
                            throw SPDException("Defined TL corner (X) needs to be outside the range of the LiDAR data.");
                        }
                        spdFile->setXMin(tlX);
                        
                        if(tlY < spdFile->getYMax())
                        {
                            throw SPDException("Defined TL corner (Y) needs to be outside the range of the LiDAR data."); 
                        }
                        spdFile->setYMax(tlY);
                    }
                    else if(defineTL & (spdFile->getIndexType() == spdlib::SPD_SPHERICAL_IDX))
                    {
                        if(tlX > spdFile->getAzimuthMin())
                        {
                            std::cout << "spdFile->getAzimuthMin() = " << spdFile->getAzimuthMin() << std::endl;
                            std::cout << "tlX = " << tlX << std::endl;
                            throw SPDException("Defined TL corner (Azimuth) needs to be outside the range of the LiDAR data.");
                        }
                        spdFile->setAzimuthMin(tlX);
                        
                        if(tlY > spdFile->getZenithMin())
                        {
                            std::cout << "spdFile->getZenithMin() = " << spdFile->getZenithMin() << std::endl;
                            std::cout << "tlY = " << tlY << std::endl;
                            throw SPDException("Defined TL corner (Zenith) needs to be outside the range of the LiDAR data."); 
                        }
                        spdFile->setZenithMin(tlY);

                    }
                    else if(defineTL & (spdFile->getIndexType() == spdlib::SPD_SCAN_IDX))
                    {
                        if(tlX > spdFile->getScanlineIdxMin())
                        {
                            std::cout << "spdFile->getScanlineIdxMin() = " << spdFile->getScanlineIdxMin() << std::endl;
                            std::cout << "tlX = " << tlX << std::endl;
                            throw SPDException("Defined TL corner (scanlineIdx) needs to be outside the range of the LiDAR data.");
                        }
                        spdFile->setScanlineIdxMin(tlX);
                        
                        if(tlY > spdFile->getScanlineMin())
                        {
                            std::cout << "spdFile->getScanlineMin() = " << spdFile->getScanlineMin() << std::endl;
                            std::cout << "tlY = " << tlY << std::endl;
                            throw SPDException("Defined TL corner (Scanline) needs to be outside the range of the LiDAR data."); 
                        }
                        spdFile->setScanlineMin(tlY);

                    }
                    
                    SPDIOUtils ioUtils;
                    ioUtils.gridAndWriteData(exporter, pulses, spdFile, output);
                    
                    delete pulses;
                }
			}
			else if(exporter->needNumOutPts())
			{
				std::list<SPDPulse*> *pulses = importer->readAllDataToList(input, spdFile);
				exporter->setNumOutPts(spdFile->getNumberOfPoints());
				exporter->open(spdFile, output);
				exporter->writeDataColumn(pulses, 0, 0);
				exporter->finaliseClose();
				
				std::list<SPDPulse*>::iterator iterPulses;
				for(iterPulses = pulses->begin(); iterPulses != pulses->end(); )
				{
					SPDPulseUtils::deleteSPDPulse(*iterPulses);
					pulses->erase(iterPulses++);
				}
				delete pulses;
			}
			else 
			{
				SPDFile *spdFileOut = new SPDFile(output);
				if(importer->isFileType("SPD"))
				{
					SPDFileReader *spdReader = (SPDFileReader*)importer;
					spdReader->readHeaderInfo(spdFile->getFilePath(), spdFile);
					spdFileOut->copyAttributesFrom(spdFile);
				}
                spdFileOut->setWaveformBitRes(waveformBitRes);
				SPDExportAsReadUnGridded *exportAsRead = new SPDExportAsReadUnGridded(exporter, spdFileOut, false, 0, false, 0, false, 0);
				importer->readAndProcessAllData(input, spdFile, exportAsRead);
                spdFile->setWaveformBitRes(waveformBitRes);
				exportAsRead->completeFileAndClose(spdFile);
				delete spdFileOut;
				delete exportAsRead;
			}

			delete spdFile;
		}
		catch (SPDException &e) 
		{
			throw e;
		}
	}
	
	void SPDConvertFormats::convertToSPDUsingRowTiles(std::string input, std::string output, std::string inFormat, std::string schema, float binsize, std::string inSpatialRef, bool convertCoords, std::string outputProjWKT, boost::uint_fast16_t indexCoords, std::string tempdir, boost::uint_fast16_t numRowsInTile, bool defineTL, double tlX, double tlY,  bool defineOrigin, double originX, double originY, float originZ, bool useSphericIdx, bool usePolarIdx, bool useScanIdx, float waveNoiseThreshold, boost::uint_fast16_t waveformBitRes, bool keepTmpFiles, boost::uint_fast16_t pointVersion, boost::uint_fast16_t pulseVersion, bool keepInMinExtent) 
	{
        std::cout.precision(12);
        if(usePolarIdx)
        {
            throw SPDException("Gridding data using a hemispherical coordinate index is not currently supported while generating SPD file using a temporary directory.");
        }
        
		SPDTextFileUtilities txtUtils;
		SPDFile *spdFileIn = NULL;
		SPDFile *spdFileAllIn = NULL;
        SPDFile *spdFileAllOutVals = NULL;
		SPDFile *spdFileFinalOut = NULL;
		SPDIOFactory ioFactory;
		SPDDataImporter *importer = NULL;
		SPDDataExporter *exporterSPD = NULL;
		SPDDataExporter *exporterUPD = NULL;
		SPDGridData gridData;
		
		double tempFileYSize = binsize * numRowsInTile;
		boost::uint_fast32_t numOfTiles = 0;
		
		// Convert input file to temp UPD file and calc dimensions
		std::cout << "Calculate Data dimensions and convert to SPD File\n";
		std::string filePathAllData = "";
		try 
		{
			if(inFormat == "SPD")
			{
				spdFileAllIn = new SPDFile(input);
                spdFileAllIn->setSpatialReference(inSpatialRef);
                spdFileAllIn->setPulseVersion(pulseVersion);
                spdFileAllIn->setPointVersion(pointVersion);
				SPDFileReader *tmpSPDImport = new SPDFileReader(false, "", schema, indexCoords, defineOrigin, originX, originY, originZ);
				tmpSPDImport->readHeaderInfo(spdFileAllIn->getFilePath(), spdFileAllIn);
				delete tmpSPDImport;
			}
			else 
			{
				importer = ioFactory.getImporter(inFormat, convertCoords, outputProjWKT, schema, indexCoords, defineOrigin, originX, originY, originZ, waveNoiseThreshold);
				spdFileIn = new SPDFile(input);
				spdFileIn->setSpatialReference(inSpatialRef);
                spdFileIn->setPulseVersion(pulseVersion);
                spdFileIn->setPointVersion(pointVersion);
				
				exporterUPD = ioFactory.getExporter("UPD", false);
                exporterUPD->setKeepMinExtent(keepInMinExtent);
				filePathAllData = tempdir + "alldata.spd";
				spdFileAllIn = new SPDFile(filePathAllData);
                spdFileAllIn->setWaveformBitRes(waveformBitRes);
				SPDExportAsReadUnGridded *exportAsRead = new SPDExportAsReadUnGridded(exporterUPD, spdFileAllIn, false, 0, false, 0, false, 0);
				importer->readAndProcessAllData(input, spdFileIn, exportAsRead);
				exportAsRead->completeFileAndClose(spdFileIn);
				delete exportAsRead;
			}
            
            if(useSphericIdx)
            {
                std::cout << "Converting into spherical coordinate system index\n";
                spdFileAllIn->setIndexType(spdlib::SPD_SPHERICAL_IDX);
            }
            else if(useScanIdx)
            {
                std::cout << "Converting into scanner coordinate system index\n";
                spdFileAllIn->setIndexType(spdlib::SPD_SCAN_IDX);
            }
            else if(usePolarIdx)
            {
                std::cout << "Converting into polar coordinate system index\n";
                spdFileAllIn->setIndexType(spdlib::SPD_POLAR_IDX);
            }
            else
            {
                std::cout << "Converting into cartisian coordinate system index\n";
                spdFileAllIn->setIndexType(spdlib::SPD_CARTESIAN_IDX);
            }
            
            if(binsize > 0)
            {
                spdFileAllIn->setBinSize(binsize);
            }
            else if(compare_double(spdFileAllIn->getBinSize(), 0) & compare_double(binsize, 0))
            {
                throw SPDException("Bin size needs to be specified.");
            }
            
            if(defineTL & (spdFileAllIn->getIndexType() == spdlib::SPD_CARTESIAN_IDX))
            {
                if(tlX > spdFileAllIn->getXMin())
                {
                    throw SPDException("Defined TL corner (X) needs to be outside the range of the LiDAR data.");
                }
                spdFileAllIn->setXMin(tlX);
                
                if(tlY < spdFileAllIn->getYMax())
                {
                    throw SPDException("Defined TL corner (Y) needs to be outside the range of the LiDAR data."); 
                }
                spdFileAllIn->setYMax(tlY);
            }
            else if(defineTL & (spdFileAllIn->getIndexType() == spdlib::SPD_SPHERICAL_IDX))
            {
                if(tlX > spdFileAllIn->getAzimuthMin())
                {
                    std::cout << "spdFileAllIn->getAzimuthMin() = " << spdFileAllIn->getAzimuthMin() << std::endl;
                    std::cout << "tlX = " << tlX << std::endl;
                    throw SPDException("Defined TL corner (Azimuth) needs to be outside the range of the LiDAR data.");
                }
                spdFileAllIn->setAzimuthMin(tlX);
                
                if(tlY > spdFileAllIn->getZenithMin())
                {
                    std::cout << "spdFileAllIn->getZenithMin() = " << spdFileAllIn->getZenithMin() << std::endl;
                    std::cout << "tlY = " << tlY << std::endl;
                    throw SPDException("Defined TL corner (Zenith) needs to be outside the range of the LiDAR data."); 
                }
                spdFileAllIn->setZenithMin(tlY);

            }
            else if(defineTL & (spdFileAllIn->getIndexType() == spdlib::SPD_SCAN_IDX))
            {
                if(tlX > spdFileAllIn->getScanlineIdxMin())
                {
                    std::cout << "spdFileAllIn->getScanlineIdxMin() = " << spdFileAllIn->getScanlineIdxMin() << std::endl;
                    std::cout << "tlX = " << tlX << std::endl;
                    throw SPDException("Defined TL corner (scanlineIdx) needs to be outside the range of the LiDAR data.");
                }
                spdFileAllIn->setScanlineIdxMin(tlX);
                
                if(tlY > spdFileAllIn->getScanlineMin())
                {
                    std::cout << "spdFileAllIn->getScanlineMin() = " << spdFileAllIn->getScanlineMin() << std::endl;
                    std::cout << "tlY = " << tlY << std::endl;
                    throw SPDException("Defined TL corner (Scanline) needs to be outside the range of the LiDAR data."); 
                }
                spdFileAllIn->setScanlineMin(tlY);

            }
            
            if(useSphericIdx)
            {
                numOfTiles = boost::numeric_cast<boost::uint_fast32_t>(((spdFileAllIn->getZenithMax() - spdFileAllIn->getZenithMin()) / tempFileYSize))+1;  
            }
            else if(useScanIdx)
            {
                numOfTiles = boost::numeric_cast<boost::uint_fast32_t>(((spdFileAllIn->getScanlineIdxMax() - spdFileAllIn->getScanlineIdxMin()) / tempFileYSize))+1;
            }
            else
            {
                numOfTiles = boost::numeric_cast<boost::uint_fast32_t>(((spdFileAllIn->getYMax() - spdFileAllIn->getYMin()) / tempFileYSize))+1;
            }
			std::cout << "Number of Tiles = " << numOfTiles << std::endl;
		}
		catch(boost::numeric::negative_overflow& e) 
		{
			throw SPDException(e.what());
		}
		catch(boost::numeric::positive_overflow& e) 
		{
			throw SPDException(e.what());
		}
		catch(boost::numeric::bad_numeric_cast& e) 
		{
			throw SPDException(e.what());
		}
		catch (SPDException &e) 
		{
			throw e;
		}
		
		PointDataTileFile *tiles = new PointDataTileFile[numOfTiles];
		std::string filePath = "";
        
        double yMax = 0.0;
        double yMin = 0.0;
        if(useSphericIdx)
        {
		     
            yMin = spdFileAllIn->getZenithMin(); 
            yMax = yMin + tempFileYSize;

		    // Create list of tiles.
		    for(boost::uint_fast32_t i = 0; i < numOfTiles; ++i)
		    {
                filePath = tempdir + txtUtils.uInt32bittostring(i) + std::string(".spd");
			    //std::cout << "File: " << filePath << std::endl;
			    //std::cout << "File: " << yMax << std::endl;
                //std::cout << "File: " <<  yMin << std::endl;		
                tiles[i].exporter = new SPDNoIdxFileWriter();
			    tiles[i].pulses = new std::list<SPDPulse*>();
			    tiles[i].env = new OGREnvelope();
			    tiles[i].spdFile = new SPDFile(filePath);
			    tiles[i].spdFile->copyAttributesFrom(spdFileAllIn);
                tiles[i].env->MinY = yMin;
			    tiles[i].env->MaxY = yMax;            
			    tiles[i].env->MinX = spdFileAllIn->getAzimuthMin();
                tiles[i].env->MaxX = spdFileAllIn->getAzimuthMax();
                tiles[i].spdFile->setAzimuthMin(tiles[i].env->MinX);
                tiles[i].spdFile->setAzimuthMax(tiles[i].env->MaxX);
                tiles[i].spdFile->setZenithMin(tiles[i].env->MinY);
                tiles[i].spdFile->setZenithMax(tiles[i].env->MaxY);
                
			    yMin = yMax;
			    if(i < numOfTiles-2)
			    {
				    yMax = yMin + tempFileYSize;
			    }
			    else
			    {
			        yMax = spdFileAllIn->getZenithMax();
                }
		    }          
        }
        else if(useScanIdx)
        {
		     
            yMin = spdFileAllIn->getScanlineMin(); 
            yMax = yMin + tempFileYSize;

		    // Create list of tiles.
		    for(boost::uint_fast32_t i = 0; i < numOfTiles; ++i)
		    {
                filePath = tempdir + txtUtils.uInt32bittostring(i) + std::string(".spd");
			    //std::cout << "File: " << filePath << std::endl;
			    //std::cout << "File: " << yMax << std::endl;
                //std::cout << "File: " <<  yMin << std::endl;		
                tiles[i].exporter = new SPDNoIdxFileWriter();
			    tiles[i].pulses = new std::list<SPDPulse*>();
			    tiles[i].env = new OGREnvelope();
			    tiles[i].spdFile = new SPDFile(filePath);
			    tiles[i].spdFile->copyAttributesFrom(spdFileAllIn);
                tiles[i].env->MinY = yMin;
			    tiles[i].env->MaxY = yMax;            
			    tiles[i].env->MinX = spdFileAllIn->getScanlineIdxMin();
                tiles[i].env->MaxX = spdFileAllIn->getScanlineIdxMax();
                tiles[i].spdFile->setScanlineIdxMin(tiles[i].env->MinX);
                tiles[i].spdFile->setScanlineIdxMax(tiles[i].env->MaxX);
                tiles[i].spdFile->setScanlineMin(tiles[i].env->MinY);
                tiles[i].spdFile->setScanlineMax(tiles[i].env->MaxY);
                
			    yMin = yMax;
			    if(i < numOfTiles-2)
			    {
				    yMax = yMin + tempFileYSize;
			    }
			    else
			    {
			        yMax = spdFileAllIn->getScanlineMax();
                }
		    }
        }
        else
        {
		    yMax = spdFileAllIn->getYMax();
            yMin = yMax - tempFileYSize; 
            
		    // Create list of tiles.
		    for(boost::uint_fast32_t i = 0; i < numOfTiles; ++i)
		    {
                filePath = tempdir + txtUtils.uInt32bittostring(i) + std::string(".spd");
			    //std::cout << "File: " << filePath << std::endl;
			    //std::cout << "File: " << yMax << std::endl;
                //std::cout << "File: " <<  yMin << std::endl;		
                tiles[i].exporter = new SPDNoIdxFileWriter();
			    tiles[i].pulses = new std::list<SPDPulse*>();
			    tiles[i].env = new OGREnvelope();
			    tiles[i].spdFile = new SPDFile(filePath);
			    tiles[i].spdFile->copyAttributesFrom(spdFileAllIn);
                tiles[i].env->MinY = yMin;
			    tiles[i].env->MaxY = yMax;
			    tiles[i].env->MinX = spdFileAllIn->getXMin();
                tiles[i].env->MaxX = spdFileAllIn->getXMax();
                tiles[i].spdFile->setXMin(tiles[i].env->MinX);
                tiles[i].spdFile->setXMax(tiles[i].env->MaxX);
                tiles[i].spdFile->setYMin(tiles[i].env->MinY);
                tiles[i].spdFile->setYMax(tiles[i].env->MaxY);
                
			    yMax = yMin;
			    if(i < numOfTiles-2)
			    {
				    yMin = yMax - tempFileYSize;
			    }
			    else
			    {
			        yMin = spdFileAllIn->getYMin();
                }
		    }
        }
        
		
		std::cout << "Create Individual tiles\n";
		SPDDataImporter *dataImporter = NULL;
		try 
		{
            spdFileAllOutVals = new SPDFile("");
            spdFileAllIn->copyAttributesTo(spdFileAllOutVals);
            if(useSphericIdx)
            {
                spdFileAllOutVals->setIndexType(spdlib::SPD_SPHERICAL_IDX);
            }
            else if(useScanIdx)
            {
                spdFileAllOutVals->setIndexType(spdlib::SPD_SCAN_IDX);
            }
            else
            {
                spdFileAllOutVals->setIndexType(spdlib::SPD_CARTESIAN_IDX);
            }
			dataImporter = new SPDFileReader(false, "", "", indexCoords);
			SPDExportAsRowTiles *createTiles = new SPDExportAsRowTiles(tiles, numOfTiles, spdFileAllOutVals, tempFileYSize, useSphericIdx, useScanIdx);
			dataImporter->readAndProcessAllData(spdFileAllIn->getFilePath(), spdFileAllIn, createTiles);
			createTiles->completeFileAndClose();
			delete createTiles;
			delete dataImporter;
		}
		catch (SPDIOException &e) 
		{
			throw e;
		}
		
        
        if(defineTL & (spdFileAllIn->getIndexType() == spdlib::SPD_CARTESIAN_IDX))
        {
            if(tlX > spdFileAllIn->getXMin())
            {
                throw SPDException("Defined TL corner (X) needs to be outside the range of the LiDAR data.");
            }
            spdFileAllIn->setXMin(tlX);
            
            if(tlY < spdFileAllIn->getYMax())
            {
                throw SPDException("Defined TL corner (Y) needs to be outside the range of the LiDAR data."); 
            }
            spdFileAllIn->setYMax(tlY);
        }
        else if(defineTL & (spdFileAllIn->getIndexType() == spdlib::SPD_SPHERICAL_IDX))
        {
            if(tlX > spdFileAllIn->getAzimuthMin())
            {
                std::cout << "spdFileAllIn->getAzimuthMin() = " << spdFileAllIn->getAzimuthMin() << std::endl;
                std::cout << "tlX = " << tlX << std::endl;
                throw SPDException("Defined TL corner (Azimuth) needs to be outside the range of the LiDAR data.");
            }
            spdFileAllIn->setAzimuthMin(tlX);
            
            if(tlY > spdFileAllIn->getZenithMin())
            {
                std::cout << "spdFileAllIn->getZenithMin() = " << spdFileAllIn->getZenithMin() << std::endl;
                std::cout << "tlY = " << tlY << std::endl;
                throw SPDException("Defined TL corner (Zenith) needs to be outside the range of the LiDAR data."); 
            }
            spdFileAllIn->setZenithMin(tlY);

        }        
        else if(defineTL & (spdFileAllIn->getIndexType() == spdlib::SPD_SCAN_IDX))
        {
            if(tlX > spdFileAllIn->getScanlineIdxMin())
            {
                std::cout << "spdFileAllIn->getScanlineIdxMin() = " << spdFileAllIn->getScanlineIdxMin() << std::endl;
                std::cout << "tlX = " << tlX << std::endl;
                throw SPDException("Defined TL corner (ScanlineIdx) needs to be outside the range of the LiDAR data.");
            }
            spdFileAllIn->setScanlineIdxMin(tlX);
            
            if(tlY > spdFileAllIn->getScanlineMin())
            {
                std::cout << "spdFileAllIn->getScanlineMin() = " << spdFileAllIn->getScanlineMin() << std::endl;
                std::cout << "tlY = " << tlY << std::endl;
                throw SPDException("Defined TL corner (Scanline) needs to be outside the range of the LiDAR data."); 
            }
            spdFileAllIn->setScanlineMin(tlY);
            
        }
        
      
		std::cout << "Create Final SPD file\n";
		spdFileFinalOut = new SPDFile(output);
		spdFileFinalOut->copyAttributesFrom(spdFileAllOutVals);
		spdFileFinalOut->setBinSize(binsize);
        
        if(useSphericIdx)
        {
            spdFileFinalOut->setIndexType(spdlib::SPD_SPHERICAL_IDX);
        }
        else if(useScanIdx)
        {
            spdFileFinalOut->setIndexType(spdlib::SPD_SCAN_IDX);
        }
        else
        {
            spdFileFinalOut->setIndexType(spdlib::SPD_CARTESIAN_IDX);
        }
        
		boost::uint_fast32_t xSize = 0;
		boost::uint_fast32_t ySize = 0;
		boost::uint_fast32_t roundingAddition = 0;
		try
		{
			if(spdFileFinalOut->getIndexType() == spdlib::SPD_SPHERICAL_IDX)
            {
				if(spdFileFinalOut->getBinSize() < 1)
			    {
				    roundingAddition = 2;//boost::numeric_cast<boost::uint_fast32_t>(1/spdFile->getBinSize());
			    }
			    else 
			    {
				    roundingAddition = 1;
			    }
                                
                xSize = boost::numeric_cast<boost::uint_fast32_t>((((spdFileFinalOut->getAzimuthMax()-spdFileFinalOut->getAzimuthMin())/spdFileFinalOut->getBinSize())+roundingAddition)+0.5);
			    ySize = boost::numeric_cast<boost::uint_fast32_t>((((spdFileFinalOut->getZenithMax()-spdFileFinalOut->getZenithMin())/spdFileFinalOut->getBinSize())+roundingAddition)+0.5);                
			}
 			else if(spdFileFinalOut->getIndexType() == spdlib::SPD_SCAN_IDX)
            {
				roundingAddition = 1;
                xSize = boost::numeric_cast<boost::uint_fast32_t>(((spdFileFinalOut->getScanlineIdxMax()-spdFileFinalOut->getScanlineIdxMin())/spdFileFinalOut->getBinSize())+roundingAddition);
			    ySize = boost::numeric_cast<boost::uint_fast32_t>(((spdFileFinalOut->getScanlineMax()-spdFileFinalOut->getScanlineMin())/spdFileFinalOut->getBinSize())+roundingAddition);                
            }
            else
            {
                if(spdFileFinalOut->getBinSize() < 1)
                {
                    roundingAddition = boost::numeric_cast<boost::uint_fast32_t>(1/spdFileFinalOut->getBinSize());
                }
                else 
                {
                    roundingAddition = 1;
                }
                xSize = boost::numeric_cast<boost::uint_fast32_t>((((spdFileFinalOut->getXMax()-spdFileFinalOut->getXMin())/spdFileFinalOut->getBinSize())+roundingAddition)+0.5);
                ySize = boost::numeric_cast<boost::uint_fast32_t>((((spdFileFinalOut->getYMax()-spdFileFinalOut->getYMin())/spdFileFinalOut->getBinSize())+roundingAddition)+0.5);			
            }
            
			spdFileFinalOut->setNumberBinsX(xSize);
			spdFileFinalOut->setNumberBinsY(ySize);
            std::cout << "Number of Bins in final output SPD file: [" << xSize << "," << ySize << "]" << std::endl;
            
		}
		catch(boost::numeric::negative_overflow& e) 
		{
			throw SPDProcessingException(e.what());
		}
		catch(boost::numeric::positive_overflow& e) 
		{
			throw SPDProcessingException(e.what());
		}
		catch(boost::numeric::bad_numeric_cast& e) 
		{
			throw SPDProcessingException(e.what());
		}
		
		try 
		{
			exporterSPD = new SPDSeqFileWriter();
			exporterSPD->open(spdFileFinalOut, spdFileFinalOut->getFilePath());
            exporterSPD->setKeepMinExtent(keepInMinExtent);
			dataImporter = new SPDFileReader(false, "", "", indexCoords);
			std::vector<SPDPulse*> *tilePulses = NULL;
			boost::uint_fast32_t totalNumRows = 0;
			boost::uint_fast32_t numRows = 0;
			std::list<SPDPulse*> ***griddedPls = NULL;
			std::list<SPDPulse*>::iterator iterPulses;
			for(boost::uint_fast32_t i = 0; i < numOfTiles; ++i)
			{
				if(i == (numOfTiles-1))
				{
					numRows = ySize - totalNumRows;//boost::numeric_cast<boost::uint_fast32_t>(((tiles[i].env->MaxY-tiles[i].env->MinY)/spdFileFinalOut->getBinSize())+1);
				}
				else 
				{
					numRows = boost::numeric_cast<boost::uint_fast32_t>(((tiles[i].env->MaxY-tiles[i].env->MinY)/spdFileFinalOut->getBinSize())+0.5);
				}
				std::cout << "Tile " << i+1 << " of " << numOfTiles << " covers " << numRows << " rows\n";
				
                griddedPls = new std::list<SPDPulse*>**[numRows];
                for(boost::uint_fast32_t n = 0; n < numRows; ++n)
				{
                    griddedPls[n] = new std::list<SPDPulse*>*[xSize];
					for(boost::uint_fast32_t m = 0; m < xSize; ++m)
					{
						griddedPls[n][m] =  new std::list<SPDPulse*>();
					}
				}
                
				// Read in pulses from tile
				tilePulses = dataImporter->readAllDataToVector(tiles[i].spdFile->getFilePath(), tiles[i].spdFile);
				
				// Grid tile
                if(tilePulses->size() > 0)
                {
                    gridData.gridData(tilePulses, spdFileFinalOut, griddedPls, tiles[i].env, xSize, numRows, spdFileFinalOut->getBinSize());
                }
				
				// Write to SPD File and remove pulses from memory.
				for(boost::uint_fast32_t n = 0; n < numRows; ++n)
				{
					for(boost::uint_fast32_t m = 0; m < xSize; ++m)
					{
						exporterSPD->writeDataColumn(griddedPls[n][m], m, totalNumRows);
						for(iterPulses = griddedPls[n][m]->begin(); iterPulses != griddedPls[n][m]->end(); )
						{
							iterPulses = griddedPls[n][m]->erase(iterPulses);
						}
						delete griddedPls[n][m];
					}
					delete[] griddedPls[n];
					++totalNumRows;
				}
				delete[] griddedPls;
				delete tilePulses;
				
				// Delete tile
				delete tiles[i].exporter;
				delete tiles[i].pulses;
				delete tiles[i].env;
				delete tiles[i].spdFile;
			}
			exporterSPD->finaliseClose();
			delete exporterSPD;
			delete dataImporter;
			delete[] tiles;

			if(totalNumRows != ySize)
			{
				throw SPDException("The number of used row and the number of rows in the file is different.");
			}
		}
		catch(boost::numeric::negative_overflow& e) 
		{
			throw SPDProcessingException(e.what());
		}
		catch(boost::numeric::positive_overflow& e) 
		{
			throw SPDProcessingException(e.what());
		}
		catch(boost::numeric::bad_numeric_cast& e) 
		{
			throw SPDProcessingException(e.what());
		}		
		catch (SPDIOException &e) 
		{
			throw e;
		}
		catch (SPDException &e) 
		{
			throw e;
		}
		
		// Clean up memory.
		delete spdFileIn;
		delete spdFileAllIn;
        delete spdFileAllOutVals;
        
        std::cout << "Complete - Sequencial indexed SPD file created\n";

	}
	
    void SPDConvertFormats::convertToSPDUsingBlockTiles(std::string input, std::string output, std::string inFormat, std::string schema, float binsize, std::string inSpatialRef, bool convertCoords, std::string outputProjWKT, boost::uint_fast16_t indexCoords, std::string tempdir,boost::uint_fast16_t numRowsInTile, boost::uint_fast16_t numColsInTile, bool defineTL, double tlX, double tlY, bool defineOrigin, double originX, double originY, float originZ, bool useSphericIdx, bool usePolarIdx, bool useScanIdx, float waveNoiseThreshold,boost::uint_fast16_t waveformBitRes, bool keepTmpFiles, boost::uint_fast16_t pointVersion, boost::uint_fast16_t pulseVersion, bool keepInMinExtent) 
    {
        //std::cout.precision(10);
        if(usePolarIdx)
        {
            throw SPDException("Gridding data using a polar coordinate index is not currently supported while generating SPD file using a temporary directory.");
        }
        if(useSphericIdx)
        {
            std::cout << "Gridding data using a spherical coordinate index is not currently tested while generating SPD file using a temporary directory.\n";
        }
        if(useScanIdx)
        {
            std::cout << "Gridding data using a scan coordinate index is not currently tested while generating SPD file using a temporary directory.\n";
        }
        
		SPDTextFileUtilities txtUtils;
		SPDFile *spdFileIn = NULL;
		SPDFile *spdFileAllIn = NULL;
        SPDFile *spdFileAllOutVals = NULL;
		SPDFile *spdFileFinalOut = NULL;
		SPDIOFactory ioFactory;
		SPDDataImporter *importer = NULL;
		SPDDataExporter *exporterSPD = NULL;
		SPDDataExporter *exporterUPD = NULL;
		SPDGridData gridData;
		
		double tempFileXSize = binsize * numColsInTile;
        double tempFileYSize = binsize * numRowsInTile;
		boost::uint_fast32_t numOfXTiles = 0;
        boost::uint_fast32_t numOfYTiles = 0;
        boost::uint_fast32_t numOfTiles = 0;
		
		// Convert input file to temp UPD file and calc dimensions
		std::cout << "Calculate Data dimensions and convert to UPD File\n";
		std::string filePathAllData = "";
		try 
		{
			if(inFormat == "SPD")
			{
				spdFileAllIn = new SPDFile(input);
                spdFileAllIn->setSpatialReference(inSpatialRef);
                spdFileAllIn->setPulseVersion(pulseVersion);
                spdFileAllIn->setPointVersion(pointVersion);
				SPDFileReader *tmpSPDImport = new SPDFileReader(false, "", schema, indexCoords, defineOrigin, originX, originY, originZ);
				tmpSPDImport->readHeaderInfo(spdFileAllIn->getFilePath(), spdFileAllIn);
				delete tmpSPDImport;
			}
			else 
			{
				importer = ioFactory.getImporter(inFormat, convertCoords, outputProjWKT, schema, indexCoords, defineOrigin, originX, originY, originZ, waveNoiseThreshold);
				spdFileIn = new SPDFile(input);
				spdFileIn->setSpatialReference(inSpatialRef);
                spdFileIn->setPulseVersion(pulseVersion);
                spdFileIn->setPointVersion(pointVersion);
				
				exporterUPD = ioFactory.getExporter("UPD", false);
                exporterUPD->setKeepMinExtent(keepInMinExtent);
				filePathAllData = tempdir + "alldata.spd";
				spdFileAllIn = new SPDFile(filePathAllData);
                spdFileAllIn->setWaveformBitRes(waveformBitRes);
				SPDExportAsReadUnGridded *exportAsRead = new SPDExportAsReadUnGridded(exporterUPD, spdFileAllIn, false, 0, false, 0, false, 0);
				importer->readAndProcessAllData(input, spdFileIn, exportAsRead);
				exportAsRead->completeFileAndClose(spdFileIn);
				delete exportAsRead;
			}
            
            if(useSphericIdx)
            {
                spdFileAllIn->setIndexType(spdlib::SPD_SPHERICAL_IDX);
            }
            else if(useScanIdx)
            {
                spdFileAllIn->setIndexType(spdlib::SPD_SCAN_IDX);
            }
            else
            {
                spdFileAllIn->setIndexType(spdlib::SPD_CARTESIAN_IDX);
            }
            
            if(binsize > 0)
            {
                spdFileAllIn->setBinSize(binsize);
            }
            else if(compare_double(spdFileAllIn->getBinSize(), 0) & compare_double(binsize, 0))
            {
                throw SPDException("Bin size needs to be specified.");
            }
            
            if(defineTL & (spdFileAllIn->getIndexType() == spdlib::SPD_CARTESIAN_IDX))
            {
                if(tlX > spdFileAllIn->getXMin())
                {
                    throw SPDException("Defined TL corner (X) needs to be outside the range of the LiDAR data.");
                }
                spdFileAllIn->setXMin(tlX);
                
                if(tlY < spdFileAllIn->getYMax())
                {
                    throw SPDException("Defined TL corner (Y) needs to be outside the range of the LiDAR data."); 
                }
                spdFileAllIn->setYMax(tlY);
            }
            else if(defineTL & (spdFileAllIn->getIndexType() == spdlib::SPD_SPHERICAL_IDX))
            {
                if(tlX > spdFileAllIn->getAzimuthMin())
                {
                    std::cout << "spdFileAllIn->getAzimuthMin() = " << spdFileAllIn->getAzimuthMin() << std::endl;
                    std::cout << "tlX = " << tlX << std::endl;
                    throw SPDException("Defined TL corner (Azimuth) needs to be outside the range of the LiDAR data.");
                }
                spdFileAllIn->setAzimuthMin(tlX);
                
                if(tlY > spdFileAllIn->getZenithMin())
                {
                    std::cout << "spdFileAllIn->getZenithMin() = " << spdFileAllIn->getZenithMin() << std::endl;
                    std::cout << "tlY = " << tlY << std::endl;
                    throw SPDException("Defined TL corner (Zenith) needs to be outside the range of the LiDAR data."); 
                }
                spdFileAllIn->setZenithMin(tlY);
                
            }
            else if(defineTL & (spdFileAllIn->getIndexType() == spdlib::SPD_SCAN_IDX))
            {
                if(tlX > spdFileAllIn->getScanlineIdxMin())
                {
                    std::cout << "spdFileAllIn->getScanlineIdxMin() = " << spdFileAllIn->getScanlineIdxMin() << std::endl;
                    std::cout << "tlX = " << tlX << std::endl;
                    throw SPDException("Defined TL corner (ScanlineIdx) needs to be outside the range of the LiDAR data.");
                }
                spdFileAllIn->setScanlineIdxMin(tlX);
                
                if(tlY > spdFileAllIn->getScanlineMin())
                {
                    std::cout << "spdFileAllIn->getScanlineMin() = " << spdFileAllIn->getScanlineMin() << std::endl;
                    std::cout << "tlY = " << tlY << std::endl;
                    throw SPDException("Defined TL corner (Scanline) needs to be outside the range of the LiDAR data."); 
                }
                spdFileAllIn->setScanlineMin(tlY);
                
            }
            
			//std::cout << "Y DIMS: [" << spdFileAllInUPD->getYMax() << ", " << spdFileAllInUPD->getYMin() << "]\n";
            if(useSphericIdx)
            {
                numOfXTiles = boost::numeric_cast<boost::uint_fast32_t>(((spdFileAllIn->getAzimuthMax() - spdFileAllIn->getAzimuthMin()) / tempFileXSize))+1;  
                numOfYTiles = boost::numeric_cast<boost::uint_fast32_t>(((spdFileAllIn->getZenithMax() - spdFileAllIn->getZenithMin()) / tempFileYSize))+1;  
            }
            else if(useScanIdx)
            {
                numOfXTiles = boost::numeric_cast<boost::uint_fast32_t>(((spdFileAllIn->getScanlineIdxMax() - spdFileAllIn->getScanlineIdxMin()) / tempFileXSize))+1;  
                numOfYTiles = boost::numeric_cast<boost::uint_fast32_t>(((spdFileAllIn->getScanlineMax() - spdFileAllIn->getScanlineMin()) / tempFileYSize))+1;  
            }
            else
            {
                numOfXTiles = boost::numeric_cast<boost::uint_fast32_t>(((spdFileAllIn->getXMax() - spdFileAllIn->getXMin()) / tempFileXSize))+1; 
                numOfYTiles = boost::numeric_cast<boost::uint_fast32_t>(((spdFileAllIn->getYMax() - spdFileAllIn->getYMin()) / tempFileYSize))+1;  
            }
            
            numOfTiles = numOfXTiles * numOfYTiles;
            
            std::cout << "Number of X Blocks = " << numOfXTiles << std::endl;
			std::cout << "Number of Y Blocks = " << numOfYTiles << std::endl;
            std::cout << "Total of Tiles = " << numOfTiles << std::endl;
		}
		catch(boost::numeric::negative_overflow& e) 
		{
			throw SPDException(e.what());
		}
		catch(boost::numeric::positive_overflow& e) 
		{
			throw SPDException(e.what());
		}
		catch(boost::numeric::bad_numeric_cast& e) 
		{
			throw SPDException(e.what());
		}
		catch (SPDException &e) 
		{
			throw e;
		}
        
        numOfTiles = numOfXTiles * numOfYTiles;
        
        PointDataTileFile *tiles = new PointDataTileFile[numOfTiles];
		std::string filePath = "";
        
        double yMax = 0.0;
        double yMin = 0.0;
        double xMax = 0.0;
        double xMin = 0.0;
        
        boost::uint_fast32_t tileCounter = 0;
        
        if(useSphericIdx)
        {
            yMin = spdFileAllIn->getZenithMin(); 
            yMax = yMin + tempFileYSize;
            
		    // Create list of tiles.
		    for(boost::uint_fast32_t i = 0; i < numOfYTiles; ++i)
		    {
                xMin = spdFileAllIn->getAzimuthMin();
                xMax = xMin + tempFileXSize; 
                for(boost::uint_fast32_t j = 0; j < numOfXTiles; ++j)
                {
                    filePath = tempdir + txtUtils.uInt32bittostring(i) + std::string("_") + txtUtils.uInt32bittostring(j) + std::string(".spd");
                    //std::cout << "File: " << filePath << std::endl;
                    //std::cout << "xMax: " << xMax << std::endl;
                    //std::cout << "xMin: " << xMin << std::endl;
                    //std::cout << "yMax: " << yMax << std::endl;
                    //std::cout << "yMin: " << yMin << std::endl;
                    tiles[tileCounter].exporter = new SPDNoIdxFileWriter();
                    tiles[tileCounter].pulses = new std::list<SPDPulse*>();
                    tiles[tileCounter].env = new OGREnvelope();
                    tiles[tileCounter].spdFile = new SPDFile(filePath);
                    tiles[tileCounter].spdFile->copyAttributesFrom(spdFileAllIn);
                    tiles[tileCounter].env->MinY = yMin;
                    tiles[tileCounter].env->MaxY = yMax;
                    tiles[tileCounter].env->MinX = xMin;
                    tiles[tileCounter].env->MaxX = xMax;
                    tiles[tileCounter].spdFile->setAzimuthMin(tiles[i].env->MinX);
                    tiles[tileCounter].spdFile->setAzimuthMax(tiles[i].env->MaxX);
                    tiles[tileCounter].spdFile->setZenithMin(tiles[i].env->MinY);
                    tiles[tileCounter].spdFile->setZenithMax(tiles[i].env->MaxY);
                    ++tileCounter;
                    xMin = xMax;
                    if(j < numOfXTiles-2)
                    {
                        xMax = xMin + tempFileXSize;
                    }
                    else
                    {
                        xMax = spdFileAllIn->getXMax();
                    }
                }
                
			    yMin = yMax;
			    if(i < numOfTiles-2)
			    {
				    yMax = yMin + tempFileYSize;
			    }
			    else
			    {
			        yMax = spdFileAllIn->getZenithMax();
                }
		    }  
        }
        else if(useScanIdx)
        {
            yMin = spdFileAllIn->getScanlineMin(); 
            yMax = yMin + tempFileYSize;
            
		    // Create list of tiles.
		    for(boost::uint_fast32_t i = 0; i < numOfYTiles; ++i)
		    {
                xMin = spdFileAllIn->getScanlineIdxMin();
                xMax = xMin + tempFileXSize; 
                for(boost::uint_fast32_t j = 0; j < numOfXTiles; ++j)
                {
                    filePath = tempdir + txtUtils.uInt32bittostring(i) + std::string("_") + txtUtils.uInt32bittostring(j) + std::string(".spd");
                    //std::cout << "File: " << filePath << std::endl;
                    //std::cout << "xMax: " << xMax << std::endl;
                    //std::cout << "xMin: " << xMin << std::endl;
                    //std::cout << "yMax: " << yMax << std::endl;
                    //std::cout << "yMin: " << yMin << std::endl;		
                    tiles[tileCounter].exporter = new SPDNoIdxFileWriter();
                    tiles[tileCounter].pulses = new std::list<SPDPulse*>();
                    tiles[tileCounter].env = new OGREnvelope();
                    tiles[tileCounter].spdFile = new SPDFile(filePath);
                    tiles[tileCounter].spdFile->copyAttributesFrom(spdFileAllIn);
                    tiles[tileCounter].env->MinY = yMin;
                    tiles[tileCounter].env->MaxY = yMax;
                    tiles[tileCounter].env->MinX = xMin;
                    tiles[tileCounter].env->MaxX = xMax;
                    tiles[tileCounter].spdFile->setScanlineIdxMin(tiles[i].env->MinX);
                    tiles[tileCounter].spdFile->setScanlineIdxMax(tiles[i].env->MaxX);
                    tiles[tileCounter].spdFile->setScanlineMin(tiles[i].env->MinY);
                    tiles[tileCounter].spdFile->setScanlineMax(tiles[i].env->MaxY);
                    ++tileCounter;
                    xMin = xMax;
                    if(j < numOfXTiles-2)
                    {
                        xMax = xMin + tempFileXSize;
                    }
                    else
                    {
                        xMax = spdFileAllIn->getXMax();
                    }
                }
                
			    yMin = yMax;
			    if(i < numOfTiles-2)
			    {
				    yMax = yMin + tempFileYSize;
			    }
			    else
			    {
			        yMax = spdFileAllIn->getScanlineMax();
                }
		    }  
        }
        else
        {
		    yMax = spdFileAllIn->getYMax();
            yMin = yMax - tempFileYSize; 
            
		    // Create list of tiles.
		    for(boost::uint_fast32_t i = 0; i < numOfYTiles; ++i)
		    {
                xMin = spdFileAllIn->getXMin();
                xMax = xMin + tempFileXSize; 
                
                for(boost::uint_fast32_t j = 0; j < numOfXTiles; ++j)
                {
                    filePath = tempdir + txtUtils.uInt32bittostring(i) + std::string("_") + txtUtils.uInt32bittostring(j) + std::string(".spd");
                    //std::cout << "File: " << filePath << std::endl;
                    //std::cout << "xMax: " << xMax << std::endl;
                    //std::cout << "xMin: " << xMin << std::endl;
                    //std::cout << "yMax: " << yMax << std::endl;
                    //std::cout << "yMin: " << yMin << std::endl;		
                    tiles[tileCounter].exporter = new SPDNoIdxFileWriter();
                    tiles[tileCounter].pulses = new std::list<SPDPulse*>();
                    tiles[tileCounter].env = new OGREnvelope();
                    tiles[tileCounter].spdFile = new SPDFile(filePath);
                    tiles[tileCounter].spdFile->copyAttributesFrom(spdFileAllIn);
                    tiles[tileCounter].env->MinY = yMin;
                    tiles[tileCounter].env->MaxY = yMax;
                    tiles[tileCounter].env->MinX = xMin;
                    tiles[tileCounter].env->MaxX = xMax;
                    tiles[tileCounter].spdFile->setXMin(tiles[i].env->MinX);
                    tiles[tileCounter].spdFile->setXMax(tiles[i].env->MaxX);
                    tiles[tileCounter].spdFile->setYMin(tiles[i].env->MinY);
                    tiles[tileCounter].spdFile->setYMax(tiles[i].env->MaxY); 
                    ++tileCounter;
                    
                    xMin = xMax;
                    if(j < numOfXTiles-2)
                    {
                        xMax = xMin + tempFileXSize;
                    }
                    else
                    {
                        xMax = spdFileAllIn->getXMax();
                    }
                }
                
			    yMax = yMin;
			    if(i < numOfYTiles-2)
			    {
				    yMin = yMax - tempFileYSize;
			    }
			    else
			    {
			        yMin = spdFileAllIn->getYMin();
                }
		    }
        }
        
        std::cout << "Create Individual tile\n";
		SPDDataImporter *dataImporter = NULL;
		try 
		{
            spdFileAllOutVals = new SPDFile("");
            spdFileAllIn->copyAttributesTo(spdFileAllOutVals);
            if(useSphericIdx)
            {
                spdFileAllOutVals->setIndexType(spdlib::SPD_SPHERICAL_IDX);
            }
            if(useScanIdx)
            {
                spdFileAllOutVals->setIndexType(spdlib::SPD_SCAN_IDX);
            }
            else
            {
                spdFileAllOutVals->setIndexType(spdlib::SPD_CARTESIAN_IDX);
            }
			dataImporter = new SPDFileReader(false, "", "", indexCoords);
			SPDExportAsBlockTiles *createTiles = new SPDExportAsBlockTiles(tiles, numOfTiles, numOfXTiles, numOfYTiles, spdFileAllOutVals, tempFileYSize, tempFileXSize, useSphericIdx, useScanIdx);
			dataImporter->readAndProcessAllData(spdFileAllIn->getFilePath(), spdFileAllIn, createTiles);
			createTiles->completeFileAndClose();
			delete createTiles;
			delete dataImporter;
		}
		catch (SPDIOException &e) 
		{
			throw e;
		}
        
        
        std::cout << "Create Final SPD file\n";
		spdFileFinalOut = new SPDFile(output);
		spdFileFinalOut->copyAttributesFrom(spdFileAllOutVals);
		spdFileFinalOut->setBinSize(binsize);
        
        if(useSphericIdx)
        {
            spdFileAllOutVals->setIndexType(spdlib::SPD_SPHERICAL_IDX);
        }
		
		boost::uint_fast32_t xSize = 0;
		boost::uint_fast32_t ySize = 0;
		boost::uint_fast32_t roundingAddition = 0;
		try
		{
			if(spdFileFinalOut->getIndexType() == spdlib::SPD_SPHERICAL_IDX)
            {
				if(spdFileFinalOut->getBinSize() < 1)
			    {
				    roundingAddition = 2;//boost::numeric_cast<boost::uint_fast32_t>(1/spdFile->getBinSize());
			    }
			    else 
			    {
				    roundingAddition = 1;
			    }
                xSize = boost::numeric_cast<boost::uint_fast32_t>((((spdFileFinalOut->getAzimuthMax()-spdFileFinalOut->getAzimuthMin())/spdFileFinalOut->getBinSize())+roundingAddition)+0.5)+1;
			    ySize = boost::numeric_cast<boost::uint_fast32_t>((((spdFileFinalOut->getZenithMax()-spdFileFinalOut->getZenithMin())/spdFileFinalOut->getBinSize())+roundingAddition)+0.5)+1;                
			}
			else if(spdFileFinalOut->getIndexType() == spdlib::SPD_SCAN_IDX)
            {
				roundingAddition = 1;
                xSize = boost::numeric_cast<boost::uint_fast32_t>(((spdFileFinalOut->getScanlineIdxMax()-spdFileFinalOut->getScanlineIdxMin())/spdFileFinalOut->getBinSize())+roundingAddition);
			    ySize = boost::numeric_cast<boost::uint_fast32_t>(((spdFileFinalOut->getScanlineMax()-spdFileFinalOut->getScanlineMin())/spdFileFinalOut->getBinSize())+roundingAddition);                
			}
            else
            {
                if(spdFileFinalOut->getBinSize() < 1)
			    {
				    roundingAddition = boost::numeric_cast<boost::uint_fast32_t>(1/spdFileFinalOut->getBinSize());
			    }
			    else 
			    {
				    roundingAddition = 1;
			    }
			    xSize = boost::numeric_cast<boost::uint_fast32_t>((((spdFileFinalOut->getXMax()-spdFileFinalOut->getXMin())/spdFileFinalOut->getBinSize())+roundingAddition)+0.5)+1;
			    ySize = boost::numeric_cast<boost::uint_fast32_t>((((spdFileFinalOut->getYMax()-spdFileFinalOut->getYMin())/spdFileFinalOut->getBinSize())+roundingAddition)+0.5)+1;			
            }
            
			spdFileFinalOut->setNumberBinsX(xSize);
			spdFileFinalOut->setNumberBinsY(ySize);            
            std::cout << "Number of Bins in final output SPD file: [" << xSize << "," << ySize << "]" << std::endl;
            
		}
		catch(boost::numeric::negative_overflow& e) 
		{
			throw SPDProcessingException(e.what());
		}
		catch(boost::numeric::positive_overflow& e) 
		{
			throw SPDProcessingException(e.what());
		}
		catch(boost::numeric::bad_numeric_cast& e) 
		{
			throw SPDProcessingException(e.what());
		}
        
        
        
        try 
		{
			exporterSPD = new SPDNonSeqFileWriter();
			exporterSPD->open(spdFileFinalOut, spdFileFinalOut->getFilePath());
            exporterSPD->setKeepMinExtent(keepInMinExtent);
			dataImporter = new SPDFileReader(false, "", "", indexCoords);
			std::vector<SPDPulse*> *tilePulses = NULL;
			boost::uint_fast32_t totalNumRows = 0;
            boost::uint_fast32_t totalNumCols = 0;
			boost::uint_fast32_t numRows = 0;
            boost::uint_fast32_t numCols = 0;
            boost::uint_fast32_t cRow = 0;
            boost::uint_fast32_t cCol = 0;
			std::list<SPDPulse*> ***griddedPls = NULL;
			std::list<SPDPulse*>::iterator iterPulses;
            tileCounter = 0;
			for(boost::uint_fast32_t i = 0; i < numOfYTiles; ++i)
			{
				if(i == (numOfYTiles-1))
				{
					numRows = (ySize - totalNumRows)-1;
				}
				else 
				{
					numRows = boost::numeric_cast<boost::uint_fast32_t>(((tiles[tileCounter].env->MaxY-tiles[tileCounter].env->MinY)/spdFileFinalOut->getBinSize())+0.5);
                }
                totalNumCols = 0;
                
                for(boost::uint_fast32_t j = 0; j < numOfXTiles; ++j)
                {
                    if(j == (numOfXTiles-1))
                    {
                        numCols = (xSize - totalNumCols);
                    }
                    else 
                    {
                        numCols = boost::numeric_cast<boost::uint_fast32_t>(((tiles[tileCounter].env->MaxX-tiles[tileCounter].env->MinX)/spdFileFinalOut->getBinSize())+0.5);
                    }
                    
                    
                    std::cout << "Tile [" << i+1 << "," << j+1 << "] of [" << numOfYTiles << "," << numOfXTiles << "] covers " << numRows << " rows and " << numCols << " columns.\n";
				
                    // Create Grid Data Structure
                    griddedPls = new std::list<SPDPulse*>**[numRows];
                    for(boost::uint_fast32_t n = 0; n < numRows; ++n)
                    {
                        griddedPls[n] = new std::list<SPDPulse*>*[numCols];
                        for(boost::uint_fast32_t m = 0; m < numCols; ++m)
                        {
                            griddedPls[n][m] = new std::list<SPDPulse*>();
                        }
                    }
                    
                    // Read in pulses from tile
                    tilePulses = dataImporter->readAllDataToVector(tiles[tileCounter].spdFile->getFilePath(), tiles[tileCounter].spdFile);
                                        
                    // Grid tile
                    if(tilePulses->size() > 0)
                    {
                        gridData.gridData(tilePulses, spdFileFinalOut, griddedPls, tiles[tileCounter].env, numCols, numRows, spdFileFinalOut->getBinSize());
                    }
				
                    cRow = totalNumRows;                    
                    // Write to SPD File and remove pulses from memory.
                    for(boost::uint_fast32_t n = 0; n < numRows; ++n)
                    {
                        cCol = totalNumCols;
                        for(boost::uint_fast32_t m = 0; m < numCols; ++m)
                        {
                            exporterSPD->writeDataColumn(griddedPls[n][m], cCol, cRow);
                            /*for(iterPulses = griddedPls[n][m]->begin(); iterPulses != griddedPls[n][m]->end(); )
                            {
                                iterPulses = griddedPls[n][m]->erase(iterPulses);
                            }*/
                            delete griddedPls[n][m];
                            ++cCol;
                        }
                        delete[] griddedPls[n];
                        ++cRow;
                    }
                    delete[] griddedPls;
                    delete tilePulses;
                     
				
                    // Delete tile
                    delete tiles[tileCounter].exporter;
                    delete tiles[tileCounter].pulses;
                    delete tiles[tileCounter].env;
                    delete tiles[tileCounter].spdFile;
                    ++tileCounter;
                    
                    totalNumCols += numCols;
                }
                totalNumRows += numRows;
                
                if(totalNumCols != xSize)
                {
                    std::cout << "Total Num Cols = " << totalNumCols << std::endl;
                    std::cout << "Actual Num Cols = " << xSize << std::endl;
                    throw SPDException("The number of used cols and the number of cols in the file is different.");
                }
			}
			exporterSPD->finaliseClose();
			delete exporterSPD;
			delete dataImporter;
			delete[] tiles;
			
			if(totalNumRows != (ySize-1))
			{
                std::cout << "Total Num Rows = " << totalNumRows << std::endl;
                std::cout << "Actual Num Rows = " << ySize << std::endl;
				throw SPDException("The number of used row and the number of rows in the file is different.");
			}
		}
		catch(boost::numeric::negative_overflow& e) 
		{
			throw SPDProcessingException(e.what());
		}
		catch(boost::numeric::positive_overflow& e) 
		{
			throw SPDProcessingException(e.what());
		}
		catch(boost::numeric::bad_numeric_cast& e) 
		{
			throw SPDProcessingException(e.what());
		}		
		catch (SPDIOException &e) 
		{
			throw e;
		}
		catch (SPDException &e) 
		{
			throw e;
		}
        
        // Clean up memory.
		delete spdFileIn;
		delete spdFileAllIn;
        delete spdFileAllOutVals;
        
        std::cout << "Complete - Non-Sequencial indexed SPD file created\n";
    }

    
    void SPDConvertFormats::copySPD2SPD(SPDFile *inSPDFile, SPDFile *outSPDFile)
    {
        try
        {
            SPDSeqFileWriter spdWriter;
            spdWriter.open(outSPDFile, outSPDFile->getFilePath());
            SPDFileIncrementalReader incReader;
            incReader.open(inSPDFile);
            
            std::list<SPDPulse*> **pulses = new std::list<SPDPulse*>*[inSPDFile->getNumberBinsX()];
            for(unsigned int j = 0; j < inSPDFile->getNumberBinsX(); ++j)
            {
                pulses[j] = new std::list<SPDPulse*>();
            }
            
            boost::uint_fast32_t feedback = inSPDFile->getNumberBinsY()/10;
			boost::uint_fast32_t feedbackCounter = 0;
            std::cout << "Started ." << std::flush;
            for(unsigned int i = 0; i < inSPDFile->getNumberBinsY(); ++i)
            {
                if((inSPDFile->getNumberBinsY() > 10) && (i % feedback == 0))
				{
					std::cout << "." << feedbackCounter << "." << std::flush;
					feedbackCounter += 10;
				}
                
                incReader.readPulseDataRow(i, pulses);
                for(int j = 0; j < inSPDFile->getNumberBinsX(); ++j)
                {
                    spdWriter.writeDataColumn(pulses[j], j, i);
                }
            }
            spdWriter.finaliseClose();
            std::cout << " Complete.\n";
            
            incReader.close();
            
            for(unsigned int j = 0; j < inSPDFile->getNumberBinsX(); ++j)
            {
                delete pulses[j];
            }
            delete[] pulses;
        }
        catch(SPDException &e)
        {
            throw e;
        }

    }
    
    
	SPDConvertFormats::~SPDConvertFormats()
	{
		
	}

}


