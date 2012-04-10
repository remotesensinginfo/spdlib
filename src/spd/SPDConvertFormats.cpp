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
	
	void SPDConvertFormats::convertInMemory(string input, string output, string inFormat, string schema, string outFormat, float binsize, string inSpatialRef, bool convertCoords, string outputProjWKT, boost::uint_fast16_t indexCoords, bool defineTL, double tlX, double tlY, bool defineOrigin, double originX, double originY, float originZ, bool useSphericIdx, bool usePolarIdx, bool useScanIdx, float waveNoiseThreshold, boost::uint_fast16_t waveformBitRes, boost::uint_fast16_t pointVersion, boost::uint_fast16_t pulseVersion) throw(SPDException)
	{        
		try 
		{
			SPDIOFactory ioFactory;
			
			SPDDataImporter *importer = ioFactory.getImporter(inFormat, convertCoords, outputProjWKT, schema, indexCoords, defineOrigin, originX, originY, originZ, waveNoiseThreshold);
			SPDDataExporter *exporter = ioFactory.getExporter(outFormat);
			
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
                    vector<SPDPulse*> *pulses = importer->readAllDataToVector(input, spdFile);
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
                            cout << "spdFile->getAzimuthMin() = " << spdFile->getAzimuthMin() << endl;
                            cout << "tlX = " << tlX << endl;
                            throw SPDException("Defined TL corner (Azimuth) needs to be outside the range of the LiDAR data.");
                        }
                        spdFile->setAzimuthMin(tlX);
                        
                        if(tlY > spdFile->getZenithMin())
                        {
                            cout << "spdFile->getZenithMin() = " << spdFile->getZenithMin() << endl;
                            cout << "tlY = " << tlY << endl;
                            throw SPDException("Defined TL corner (Zenith) needs to be outside the range of the LiDAR data."); 
                        }
                        spdFile->setZenithMin(tlY);

                    }
                    else if(defineTL & (spdFile->getIndexType() == spdlib::SPD_SCAN_IDX))
                    {
                        if(tlX > spdFile->getScanlineIdxMin())
                        {
                            cout << "spdFile->getScanlineIdxMin() = " << spdFile->getScanlineIdxMin() << endl;
                            cout << "tlX = " << tlX << endl;
                            throw SPDException("Defined TL corner (scanlineIdx) needs to be outside the range of the LiDAR data.");
                        }
                        spdFile->setScanlineIdxMin(tlX);
                        
                        if(tlY > spdFile->getScanlineMin())
                        {
                            cout << "spdFile->getScanlineMin() = " << spdFile->getScanlineMin() << endl;
                            cout << "tlY = " << tlY << endl;
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
				list<SPDPulse*> *pulses = importer->readAllDataToList(input, spdFile);
				exporter->setNumOutPts(spdFile->getNumberOfPoints());
				exporter->open(spdFile, output);
				exporter->writeDataColumn(pulses, 0, 0);
				exporter->finaliseClose();
				
				list<SPDPulse*>::iterator iterPulses;
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
	
	void SPDConvertFormats::convertToSPDUsingRowTiles(string input, string output, string inFormat, string schema, float binsize, string inSpatialRef, bool convertCoords, string outputProjWKT, boost::uint_fast16_t indexCoords, string tempdir, boost::uint_fast16_t numRowsInTile, bool defineTL, double tlX, double tlY,  bool defineOrigin, double originX, double originY, float originZ, bool useSphericIdx, bool usePolarIdx, bool useScanIdx, float waveNoiseThreshold, boost::uint_fast16_t waveformBitRes, bool keepTmpFiles, boost::uint_fast16_t pointVersion, boost::uint_fast16_t pulseVersion) throw(SPDException)
	{
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
		cout << "Calculate Data dimensions and convert to SPD File\n";
		string filePathAllData = "";
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
				
				exporterUPD = ioFactory.getExporter("UPD");
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
            else if(usePolarIdx)
            {
                spdFileAllIn->setIndexType(spdlib::SPD_POLAR_IDX);
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
                    cout << "spdFileAllIn->getAzimuthMin() = " << spdFileAllIn->getAzimuthMin() << endl;
                    cout << "tlX = " << tlX << endl;
                    throw SPDException("Defined TL corner (Azimuth) needs to be outside the range of the LiDAR data.");
                }
                spdFileAllIn->setAzimuthMin(tlX);
                
                if(tlY > spdFileAllIn->getZenithMin())
                {
                    cout << "spdFileAllIn->getZenithMin() = " << spdFileAllIn->getZenithMin() << endl;
                    cout << "tlY = " << tlY << endl;
                    throw SPDException("Defined TL corner (Zenith) needs to be outside the range of the LiDAR data."); 
                }
                spdFileAllIn->setZenithMin(tlY);

            }
            else if(defineTL & (spdFileAllIn->getIndexType() == spdlib::SPD_SCAN_IDX))
            {
                if(tlX > spdFileAllIn->getScanlineIdxMin())
                {
                    cout << "spdFileAllIn->getScanlineIdxMin() = " << spdFileAllIn->getScanlineIdxMin() << endl;
                    cout << "tlX = " << tlX << endl;
                    throw SPDException("Defined TL corner (scanlineIdx) needs to be outside the range of the LiDAR data.");
                }
                spdFileAllIn->setScanlineIdxMin(tlX);
                
                if(tlY > spdFileAllIn->getScanlineMin())
                {
                    cout << "spdFileAllIn->getScanlineMin() = " << spdFileAllIn->getScanlineMin() << endl;
                    cout << "tlY = " << tlY << endl;
                    throw SPDException("Defined TL corner (Scanline) needs to be outside the range of the LiDAR data."); 
                }
                spdFileAllIn->setScanlineMin(tlY);

            }
            
            if(useSphericIdx)
            {
                numOfTiles = numeric_cast<boost::uint_fast32_t>(((spdFileAllIn->getZenithMax() - spdFileAllIn->getZenithMin()) / tempFileYSize)+1);  
            }
            else if(useScanIdx)
            {
                numOfTiles = numeric_cast<boost::uint_fast32_t>(((spdFileAllIn->getScanlineIdxMax() - spdFileAllIn->getScanlineIdxMin()) / tempFileYSize)+1);  
            }
            else
            {
                numOfTiles = numeric_cast<boost::uint_fast32_t>(((spdFileAllIn->getYMax() - spdFileAllIn->getYMin()) / tempFileYSize)+1);  
            }
			cout << "Number of Tiles = " << numOfTiles << endl;
		}
		catch(negative_overflow& e) 
		{
			throw SPDException(e.what());
		}
		catch(positive_overflow& e) 
		{
			throw SPDException(e.what());
		}
		catch(bad_numeric_cast& e) 
		{
			throw SPDException(e.what());
		}
		catch (SPDException &e) 
		{
			throw e;
		}
		
		PointDataTileFile *tiles = new PointDataTileFile[numOfTiles];
		string filePath = "";
        
        double yMax = 0.0;
        double yMin = 0.0;
        if(useSphericIdx)
        {
		     
            yMin = spdFileAllIn->getZenithMin(); 
            yMax = yMin + tempFileYSize;

		    // Create list of tiles.
		    for(boost::uint_fast32_t i = 0; i < numOfTiles; ++i)
		    {
                filePath = tempdir + txtUtils.uInt32bittostring(i) + string(".spd");
			    //cout << "File: " << filePath << endl;
			    //cout << "File: " << yMax << endl;
                //cout << "File: " <<  yMin << endl;		
                tiles[i].exporter = new SPDNoIdxFileWriter();
			    tiles[i].pulses = new list<SPDPulse*>();
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
                filePath = tempdir + txtUtils.uInt32bittostring(i) + string(".spd");
			    //cout << "File: " << filePath << endl;
			    //cout << "File: " << yMax << endl;
                //cout << "File: " <<  yMin << endl;		
                tiles[i].exporter = new SPDNoIdxFileWriter();
			    tiles[i].pulses = new list<SPDPulse*>();
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
                filePath = tempdir + txtUtils.uInt32bittostring(i) + string(".spd");
			    //cout << "File: " << filePath << endl;
			    //cout << "File: " << yMax << endl;
                //cout << "File: " <<  yMin << endl;		
                tiles[i].exporter = new SPDNoIdxFileWriter();
			    tiles[i].pulses = new list<SPDPulse*>();
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
        
		
		cout << "Create Individual tiles\n";
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
                cout << "spdFileAllIn->getAzimuthMin() = " << spdFileAllIn->getAzimuthMin() << endl;
                cout << "tlX = " << tlX << endl;
                throw SPDException("Defined TL corner (Azimuth) needs to be outside the range of the LiDAR data.");
            }
            spdFileAllIn->setAzimuthMin(tlX);
            
            if(tlY > spdFileAllIn->getZenithMin())
            {
                cout << "spdFileAllIn->getZenithMin() = " << spdFileAllIn->getZenithMin() << endl;
                cout << "tlY = " << tlY << endl;
                throw SPDException("Defined TL corner (Zenith) needs to be outside the range of the LiDAR data."); 
            }
            spdFileAllIn->setZenithMin(tlY);

        }        
        else if(defineTL & (spdFileAllIn->getIndexType() == spdlib::SPD_SCAN_IDX))
        {
            if(tlX > spdFileAllIn->getScanlineIdxMin())
            {
                cout << "spdFileAllIn->getScanlineIdxMin() = " << spdFileAllIn->getScanlineIdxMin() << endl;
                cout << "tlX = " << tlX << endl;
                throw SPDException("Defined TL corner (ScanlineIdx) needs to be outside the range of the LiDAR data.");
            }
            spdFileAllIn->setScanlineIdxMin(tlX);
            
            if(tlY > spdFileAllIn->getScanlineMin())
            {
                cout << "spdFileAllIn->getScanlineMin() = " << spdFileAllIn->getScanlineMin() << endl;
                cout << "tlY = " << tlY << endl;
                throw SPDException("Defined TL corner (Scanline) needs to be outside the range of the LiDAR data."); 
            }
            spdFileAllIn->setScanlineMin(tlY);
            
        }
        
      
		cout << "Create Final SPD file\n";
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
				    roundingAddition = 2;//numeric_cast<boost::uint_fast32_t>(1/spdFile->getBinSize());
			    }
			    else 
			    {
				    roundingAddition = 1;
			    }
                xSize = numeric_cast<boost::uint_fast32_t>((((spdFileFinalOut->getAzimuthMax()-spdFileFinalOut->getAzimuthMin())/spdFileFinalOut->getBinSize())+roundingAddition)+0.5);
			    ySize = numeric_cast<boost::uint_fast32_t>((((spdFileFinalOut->getZenithMax()-spdFileFinalOut->getZenithMin())/spdFileFinalOut->getBinSize())+roundingAddition)+0.5);                
			}
 			if(spdFileFinalOut->getIndexType() == spdlib::SPD_SCAN_IDX)
            {
				if(spdFileFinalOut->getBinSize() < 1)
			    {
				    roundingAddition = 2;//numeric_cast<boost::uint_fast32_t>(1/spdFile->getBinSize());
			    }
			    else 
			    {
				    roundingAddition = 1;
			    }
                xSize = numeric_cast<boost::uint_fast32_t>((((spdFileFinalOut->getScanlineIdxMax()-spdFileFinalOut->getScanlineIdxMin())/spdFileFinalOut->getBinSize())+roundingAddition)+0.5);
			    ySize = numeric_cast<boost::uint_fast32_t>((((spdFileFinalOut->getScanlineMax()-spdFileFinalOut->getScanlineMin())/spdFileFinalOut->getBinSize())+roundingAddition)+0.5);                
			}
           else
            {
                if(spdFileFinalOut->getBinSize() < 1)
			    {
				    roundingAddition = numeric_cast<boost::uint_fast32_t>(1/spdFileFinalOut->getBinSize());
			    }
			    else 
			    {
				    roundingAddition = 1;
			    }
			    xSize = numeric_cast<boost::uint_fast32_t>((((spdFileFinalOut->getXMax()-spdFileFinalOut->getXMin())/spdFileFinalOut->getBinSize())+roundingAddition)+0.5);
			    ySize = numeric_cast<boost::uint_fast32_t>((((spdFileFinalOut->getYMax()-spdFileFinalOut->getYMin())/spdFileFinalOut->getBinSize())+roundingAddition)+0.5);			
            }
            
			spdFileFinalOut->setNumberBinsX(xSize);
			spdFileFinalOut->setNumberBinsY(ySize);
            cout << "Number of Bins in final output SPD file: [" << xSize << "," << ySize << "]" << endl;
            
		}
		catch(negative_overflow& e) 
		{
			throw SPDProcessingException(e.what());
		}
		catch(positive_overflow& e) 
		{
			throw SPDProcessingException(e.what());
		}
		catch(bad_numeric_cast& e) 
		{
			throw SPDProcessingException(e.what());
		}
		
		try 
		{
			exporterSPD = new SPDSeqFileWriter();
			exporterSPD->open(spdFileFinalOut, spdFileFinalOut->getFilePath());
			dataImporter = new SPDFileReader(false, "", "", indexCoords);
			vector<SPDPulse*> *tilePulses = NULL;
			boost::uint_fast32_t totalNumRows = 0;
			boost::uint_fast32_t numRows = 0;
			list<SPDPulse*> ***griddedPls = NULL;
			list<SPDPulse*>::iterator iterPulses;
			for(boost::uint_fast32_t i = 0; i < numOfTiles; ++i)
			{
				if(i == (numOfTiles-1))
				{
					numRows = ySize - totalNumRows;//numeric_cast<boost::uint_fast32_t>(((tiles[i].env->MaxY-tiles[i].env->MinY)/spdFileFinalOut->getBinSize())+1);
				}
				else 
				{
					numRows = numeric_cast<boost::uint_fast32_t>(((tiles[i].env->MaxY-tiles[i].env->MinY)/spdFileFinalOut->getBinSize())+0.5);
				}
				cout << "Tile " << i+1 << " of " << numOfTiles << " covers " << numRows << " rows\n";
				
                griddedPls = new list<SPDPulse*>**[numRows];
                for(boost::uint_fast32_t n = 0; n < numRows; ++n)
				{
                    griddedPls[n] = new list<SPDPulse*>*[xSize];
					for(boost::uint_fast32_t m = 0; m < xSize; ++m)
					{
						griddedPls[n][m] =  new list<SPDPulse*>();
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
		catch(negative_overflow& e) 
		{
			throw SPDProcessingException(e.what());
		}
		catch(positive_overflow& e) 
		{
			throw SPDProcessingException(e.what());
		}
		catch(bad_numeric_cast& e) 
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
        
        cout << "Complete - Sequencial indexed SPD file created\n";

	}
	
    void SPDConvertFormats::convertToSPDUsingBlockTiles(string input, string output, string inFormat, string schema, float binsize, string inSpatialRef, bool convertCoords, string outputProjWKT, boost::uint_fast16_t indexCoords, string tempdir,boost::uint_fast16_t numRowsInTile, boost::uint_fast16_t numColsInTile, bool defineTL, double tlX, double tlY, bool defineOrigin, double originX, double originY, float originZ, bool useSphericIdx, bool usePolarIdx, bool useScanIdx, float waveNoiseThreshold,boost::uint_fast16_t waveformBitRes, bool keepTmpFiles, boost::uint_fast16_t pointVersion, boost::uint_fast16_t pulseVersion) throw(SPDException)
    {
        //cout.precision(10);
        if(usePolarIdx)
        {
            throw SPDException("Gridding data using a polar coordinate index is not currently supported while generating SPD file using a temporary directory.");
        }
        if(useSphericIdx)
        {
            cout << "Gridding data using a spherical coordinate index is not currently tested while generating SPD file using a temporary directory.\n";
        }
        if(useScanIdx)
        {
            cout << "Gridding data using a scan coordinate index is not currently tested while generating SPD file using a temporary directory.\n";
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
		cout << "Calculate Data dimensions and convert to UPD File\n";
		string filePathAllData = "";
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
				
				exporterUPD = ioFactory.getExporter("UPD");
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
                    cout << "spdFileAllIn->getAzimuthMin() = " << spdFileAllIn->getAzimuthMin() << endl;
                    cout << "tlX = " << tlX << endl;
                    throw SPDException("Defined TL corner (Azimuth) needs to be outside the range of the LiDAR data.");
                }
                spdFileAllIn->setAzimuthMin(tlX);
                
                if(tlY > spdFileAllIn->getZenithMin())
                {
                    cout << "spdFileAllIn->getZenithMin() = " << spdFileAllIn->getZenithMin() << endl;
                    cout << "tlY = " << tlY << endl;
                    throw SPDException("Defined TL corner (Zenith) needs to be outside the range of the LiDAR data."); 
                }
                spdFileAllIn->setZenithMin(tlY);
                
            }
            else if(defineTL & (spdFileAllIn->getIndexType() == spdlib::SPD_SCAN_IDX))
            {
                if(tlX > spdFileAllIn->getScanlineIdxMin())
                {
                    cout << "spdFileAllIn->getScanlineIdxMin() = " << spdFileAllIn->getScanlineIdxMin() << endl;
                    cout << "tlX = " << tlX << endl;
                    throw SPDException("Defined TL corner (ScanlineIdx) needs to be outside the range of the LiDAR data.");
                }
                spdFileAllIn->setScanlineIdxMin(tlX);
                
                if(tlY > spdFileAllIn->getScanlineMin())
                {
                    cout << "spdFileAllIn->getScanlineMin() = " << spdFileAllIn->getScanlineMin() << endl;
                    cout << "tlY = " << tlY << endl;
                    throw SPDException("Defined TL corner (Scanline) needs to be outside the range of the LiDAR data."); 
                }
                spdFileAllIn->setScanlineMin(tlY);
                
            }
            
			//cout << "Y DIMS: [" << spdFileAllInUPD->getYMax() << ", " << spdFileAllInUPD->getYMin() << "]\n";
            if(useSphericIdx)
            {
                numOfXTiles = numeric_cast<boost::uint_fast32_t>(((spdFileAllIn->getAzimuthMax() - spdFileAllIn->getAzimuthMin()) / tempFileXSize)+1);  
                numOfYTiles = numeric_cast<boost::uint_fast32_t>(((spdFileAllIn->getZenithMax() - spdFileAllIn->getZenithMin()) / tempFileYSize)+1);  
            }
            else if(useScanIdx)
            {
                numOfXTiles = numeric_cast<boost::uint_fast32_t>(((spdFileAllIn->getScanlineIdxMax() - spdFileAllIn->getScanlineIdxMin()) / tempFileXSize)+1);  
                numOfYTiles = numeric_cast<boost::uint_fast32_t>(((spdFileAllIn->getScanlineMax() - spdFileAllIn->getScanlineMin()) / tempFileYSize)+1);  
            }
            else
            {
                numOfXTiles = numeric_cast<boost::uint_fast32_t>(((spdFileAllIn->getXMax() - spdFileAllIn->getXMin()) / tempFileXSize)+1); 
                numOfYTiles = numeric_cast<boost::uint_fast32_t>(((spdFileAllIn->getYMax() - spdFileAllIn->getYMin()) / tempFileYSize)+1);  
            }
            
            numOfTiles = numOfXTiles * numOfYTiles;
            
            cout << "Number of X Blocks = " << numOfXTiles << endl;
			cout << "Number of Y Blocks = " << numOfYTiles << endl;
            cout << "Total of Tiles = " << numOfTiles << endl;
		}
		catch(negative_overflow& e) 
		{
			throw SPDException(e.what());
		}
		catch(positive_overflow& e) 
		{
			throw SPDException(e.what());
		}
		catch(bad_numeric_cast& e) 
		{
			throw SPDException(e.what());
		}
		catch (SPDException &e) 
		{
			throw e;
		}
        
        numOfTiles = numOfXTiles * numOfYTiles;
        
        PointDataTileFile *tiles = new PointDataTileFile[numOfTiles];
		string filePath = "";
        
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
                    filePath = tempdir + txtUtils.uInt32bittostring(i) + string("_") + txtUtils.uInt32bittostring(j) + string(".spd");
                    //cout << "File: " << filePath << endl;
                    //cout << "xMax: " << xMax << endl;
                    //cout << "xMin: " << xMin << endl;
                    //cout << "yMax: " << yMax << endl;
                    //cout << "yMin: " << yMin << endl;
                    tiles[tileCounter].exporter = new SPDNoIdxFileWriter();
                    tiles[tileCounter].pulses = new list<SPDPulse*>();
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
                    filePath = tempdir + txtUtils.uInt32bittostring(i) + string("_") + txtUtils.uInt32bittostring(j) + string(".spd");
                    //cout << "File: " << filePath << endl;
                    //cout << "xMax: " << xMax << endl;
                    //cout << "xMin: " << xMin << endl;
                    //cout << "yMax: " << yMax << endl;
                    //cout << "yMin: " << yMin << endl;		
                    tiles[tileCounter].exporter = new SPDNoIdxFileWriter();
                    tiles[tileCounter].pulses = new list<SPDPulse*>();
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
                    filePath = tempdir + txtUtils.uInt32bittostring(i) + string("_") + txtUtils.uInt32bittostring(j) + string(".spd");
                    //cout << "File: " << filePath << endl;
                    //cout << "xMax: " << xMax << endl;
                    //cout << "xMin: " << xMin << endl;
                    //cout << "yMax: " << yMax << endl;
                    //cout << "yMin: " << yMin << endl;		
                    tiles[tileCounter].exporter = new SPDNoIdxFileWriter();
                    tiles[tileCounter].pulses = new list<SPDPulse*>();
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
        
        cout << "Create Individual tile\n";
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
        
        
        cout << "Create Final SPD file\n";
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
				    roundingAddition = 2;//numeric_cast<boost::uint_fast32_t>(1/spdFile->getBinSize());
			    }
			    else 
			    {
				    roundingAddition = 1;
			    }
                xSize = numeric_cast<boost::uint_fast32_t>((((spdFileFinalOut->getAzimuthMax()-spdFileFinalOut->getAzimuthMin())/spdFileFinalOut->getBinSize())+roundingAddition)+0.5)+1;
			    ySize = numeric_cast<boost::uint_fast32_t>((((spdFileFinalOut->getZenithMax()-spdFileFinalOut->getZenithMin())/spdFileFinalOut->getBinSize())+roundingAddition)+0.5)+1;                
			}
			else if(spdFileFinalOut->getIndexType() == spdlib::SPD_SCAN_IDX)
            {
				if(spdFileFinalOut->getBinSize() < 1)
			    {
				    roundingAddition = 2;//numeric_cast<boost::uint_fast32_t>(1/spdFile->getBinSize());
			    }
			    else 
			    {
				    roundingAddition = 1;
			    }
                xSize = numeric_cast<boost::uint_fast32_t>((((spdFileFinalOut->getScanlineIdxMax()-spdFileFinalOut->getScanlineIdxMin())/spdFileFinalOut->getBinSize())+roundingAddition)+0.5)+1;
			    ySize = numeric_cast<boost::uint_fast32_t>((((spdFileFinalOut->getScanlineMax()-spdFileFinalOut->getScanlineMin())/spdFileFinalOut->getBinSize())+roundingAddition)+0.5)+1;                
			}
            else
            {
                if(spdFileFinalOut->getBinSize() < 1)
			    {
				    roundingAddition = numeric_cast<boost::uint_fast32_t>(1/spdFileFinalOut->getBinSize());
			    }
			    else 
			    {
				    roundingAddition = 1;
			    }
			    xSize = numeric_cast<boost::uint_fast32_t>((((spdFileFinalOut->getXMax()-spdFileFinalOut->getXMin())/spdFileFinalOut->getBinSize())+roundingAddition)+0.5)+1;
			    ySize = numeric_cast<boost::uint_fast32_t>((((spdFileFinalOut->getYMax()-spdFileFinalOut->getYMin())/spdFileFinalOut->getBinSize())+roundingAddition)+0.5)+1;			
            }
            
			spdFileFinalOut->setNumberBinsX(xSize);
			spdFileFinalOut->setNumberBinsY(ySize);            
            cout << "Number of Bins in final output SPD file: [" << xSize << "," << ySize << "]" << endl;
            
		}
		catch(negative_overflow& e) 
		{
			throw SPDProcessingException(e.what());
		}
		catch(positive_overflow& e) 
		{
			throw SPDProcessingException(e.what());
		}
		catch(bad_numeric_cast& e) 
		{
			throw SPDProcessingException(e.what());
		}
        
        
        
        try 
		{
			exporterSPD = new SPDNonSeqFileWriter();
			exporterSPD->open(spdFileFinalOut, spdFileFinalOut->getFilePath());
			dataImporter = new SPDFileReader(false, "", "", indexCoords);
			vector<SPDPulse*> *tilePulses = NULL;
			boost::uint_fast32_t totalNumRows = 0;
            boost::uint_fast32_t totalNumCols = 0;
			boost::uint_fast32_t numRows = 0;
            boost::uint_fast32_t numCols = 0;
            boost::uint_fast32_t cRow = 0;
            boost::uint_fast32_t cCol = 0;
			list<SPDPulse*> ***griddedPls = NULL;
			list<SPDPulse*>::iterator iterPulses;
            tileCounter = 0;
			for(boost::uint_fast32_t i = 0; i < numOfYTiles; ++i)
			{
				if(i == (numOfYTiles-1))
				{
					numRows = (ySize - totalNumRows)-1;
				}
				else 
				{
					numRows = numeric_cast<boost::uint_fast32_t>(((tiles[tileCounter].env->MaxY-tiles[tileCounter].env->MinY)/spdFileFinalOut->getBinSize())+0.5);
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
                        numCols = numeric_cast<boost::uint_fast32_t>(((tiles[tileCounter].env->MaxX-tiles[tileCounter].env->MinX)/spdFileFinalOut->getBinSize())+0.5);
                    }
                    
                    
                    cout << "Tile [" << i+1 << "," << j+1 << "] of [" << numOfYTiles << "," << numOfXTiles << "] covers " << numRows << " rows and " << numCols << " columns.\n";
				
                    // Create Grid Data Structure
                    griddedPls = new list<SPDPulse*>**[numRows];
                    for(boost::uint_fast32_t n = 0; n < numRows; ++n)
                    {
                        griddedPls[n] = new list<SPDPulse*>*[numCols];
                        for(boost::uint_fast32_t m = 0; m < numCols; ++m)
                        {
                            griddedPls[n][m] = new list<SPDPulse*>();
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
                    cout << "Total Num Cols = " << totalNumCols << endl;
                    cout << "Actual Num Cols = " << xSize << endl;
                    throw SPDException("The number of used cols and the number of cols in the file is different.");
                }
			}
			exporterSPD->finaliseClose();
			delete exporterSPD;
			delete dataImporter;
			delete[] tiles;
			
			if(totalNumRows != (ySize-1))
			{
                cout << "Total Num Rows = " << totalNumRows << endl;
                cout << "Actual Num Rows = " << ySize << endl;
				throw SPDException("The number of used row and the number of rows in the file is different.");
			}
		}
		catch(negative_overflow& e) 
		{
			throw SPDProcessingException(e.what());
		}
		catch(positive_overflow& e) 
		{
			throw SPDProcessingException(e.what());
		}
		catch(bad_numeric_cast& e) 
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
        
        cout << "Complete - Non-Sequencial indexed SPD file created\n";
    }

    
    void SPDConvertFormats::copySPD2SPD(SPDFile *inSPDFile, SPDFile *outSPDFile)throw(SPDException)
    {
        try
        {
            SPDSeqFileWriter spdWriter;
            spdWriter.open(outSPDFile, outSPDFile->getFilePath());
            SPDFileIncrementalReader incReader;
            incReader.open(inSPDFile);
            
            list<SPDPulse*> **pulses = new list<SPDPulse*>*[inSPDFile->getNumberBinsX()];
            for(unsigned int j = 0; j < inSPDFile->getNumberBinsX(); ++j)
            {
                pulses[j] = new list<SPDPulse*>();
            }
            
            boost::uint_fast32_t feedback = inSPDFile->getNumberBinsY()/10;
			boost::uint_fast32_t feedbackCounter = 0;
            cout << "Started ." << flush;
            for(unsigned int i = 0; i < inSPDFile->getNumberBinsY(); ++i)
            {
                if((inSPDFile->getNumberBinsY() > 10) && (i % feedback == 0))
				{
					cout << "." << feedbackCounter << "." << flush;
					feedbackCounter += 10;
				}
                
                incReader.readPulseDataRow(i, pulses);
                for(int j = 0; j < inSPDFile->getNumberBinsX(); ++j)
                {
                    spdWriter.writeDataColumn(pulses[j], j, i);
                }
            }
            spdWriter.finaliseClose();
            cout << " Complete.\n";
            
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


