/*
 *  SPDExportProcessorSubset.cpp
 *  SPDLIB
 *
 *  Created by Pete Bunting on 19/12/2010.
 *  Copyright 2010 RSGISLib. All rights reserved.
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

#include "spd/SPDExportProcessorSubset.h"

namespace spdlib
{
	SPDExportProcessorSubset::SPDExportProcessorSubset(SPDDataExporter *exporter, SPDFile *spdFileOut, double *bbox)  : SPDImporterProcessor()
	{
		this->exporter = exporter;
		this->spdFileOut = spdFileOut;
		this->bbox = bbox;
		if(exporter->requireGrid())
		{
			throw SPDException("This class does not support the export of gridded formats.");
		}
		
		try 
		{
			this->exporter->open(this->spdFileOut, this->spdFileOut->getFilePath());
		}
		catch (SPDException &e) 
		{
			throw e;
		}
		this->fileOpen = true;
		this->pulses = new std::list<SPDPulse*>();
		
		this->xMin = 0;
		this->xMax = 0;
		this->yMin = 0;
		this->yMax = 0;
		this->zMin = 0;
		this->zMax = 0;
		this->first = true;
	}
	
	void SPDExportProcessorSubset::processImportedPulse(SPDFile *spdFile, SPDPulse *pulse) 
	{
		try
		{
			if((pulse->xIdx > bbox[0]) & (pulse->xIdx < bbox[1]) &
			   (pulse->yIdx > bbox[2]) & (pulse->yIdx < bbox[3]))
			{
				bool zWithBBox = true;
				if((pulse->pts != NULL) && (pulse->numberOfReturns > 0))
				{
					if((pulse->pts->front()->z < bbox[4]) |
					   (pulse->pts->front()->z > bbox[5]))
					{
						zWithBBox = false;
					}
					
					if((pulse->pts->back()->z < bbox[4]) |
					   (pulse->pts->back()->z > bbox[5]))
					{
						zWithBBox = false;
					}
				}

				if(pulse->numOfReceivedBins > 0)
				{
					double tempX = 0;
					double tempY = 0;
					double tempZ = 0;
					
					SPDConvertToCartesian(pulse->zenith, pulse->azimuth, pulse->rangeToWaveformStart, pulse->x0, pulse->y0, pulse->z0, &tempX, &tempY, &tempZ);
					if((tempZ < bbox[4]) |
					   (tempZ > bbox[5]))
					{
						zWithBBox = false;
					}
					
					SPDConvertToCartesian(pulse->zenith, pulse->azimuth, (pulse->rangeToWaveformStart+(((pulse->numOfReceivedBins-1)*SPD_SPEED_OF_LIGHT_NS))/2), pulse->x0, pulse->y0, pulse->z0, &tempX, &tempY, &tempZ);
					if((tempZ < bbox[4]) |
					   (tempZ > bbox[5]))
					{
						zWithBBox = false;
					}
				}
				
				if(zWithBBox)
				{
					if(first)
					{
						this->xMin = pulse->xIdx;
						this->xMax = pulse->xIdx;
						this->yMin = pulse->yIdx;
						this->yMax = pulse->yIdx;
						
						if(pulse->numOfReceivedBins > 0)
						{
							double tempX = 0;
							double tempY = 0;
							double tempZ = 0;
							
							SPDConvertToCartesian(pulse->zenith, pulse->azimuth, pulse->rangeToWaveformStart, pulse->x0, pulse->y0, pulse->z0, &tempX, &tempY, &tempZ);
							this->zMin = tempZ;
							SPDConvertToCartesian(pulse->zenith, pulse->azimuth, (pulse->rangeToWaveformStart+(((pulse->numOfReceivedBins-1)*SPD_SPEED_OF_LIGHT_NS))/2), pulse->x0, pulse->y0, pulse->z0, &tempX, &tempY, &tempZ);
							this->zMax = tempZ;
						}
						else if((pulse->pts != NULL) && (pulse->numberOfReturns > 0))
						{
							this->zMin = pulse->pts->front()->z;
							this->zMax = pulse->pts->back()->z;
						}
						first = false;
					}
					else 
					{
						if(pulse->xIdx < this->xMin)
						{
							this->xMin = pulse->xIdx;
						}
						else if(pulse->xIdx > this->xMax)
						{
							this->xMax = pulse->xIdx;
						}
						
						if(pulse->yIdx < this->yMin)
						{
							this->yMin = pulse->yIdx;
						}
						else if(pulse->yIdx > this->yMax)
						{
							this->yMax = pulse->yIdx;
						}
						
						if(pulse->numOfReceivedBins > 0)
						{
							double tempX = 0;
							double tempY = 0;
							double tempZ_min = 0;
							double tempZ_max = 0;
							
							SPDConvertToCartesian(pulse->zenith, pulse->azimuth, pulse->rangeToWaveformStart, pulse->x0, pulse->y0, pulse->z0, &tempX, &tempY, &tempZ_min);
							SPDConvertToCartesian(pulse->zenith, pulse->azimuth, (pulse->rangeToWaveformStart+(((pulse->numOfReceivedBins-1)*SPD_SPEED_OF_LIGHT_NS))/2), pulse->x0, pulse->y0, pulse->z0, &tempX, &tempY, &tempZ_max);
							if(tempZ_min < this->zMin)
							{
								this->zMin = tempZ_min;
							}
							else if(tempZ_max > this->zMax)
							{
								this->zMax = tempZ_max;
							}
						}
						else if((pulse->pts != NULL) && (pulse->numberOfReturns > 0))
						{
							if(pulse->pts->front()->z < this->zMin)
							{
								this->zMin = pulse->pts->front()->z;
							}
							else if(pulse->pts->back()->z > this->zMax)
							{
								this->zMax = pulse->pts->back()->z;
							}
						}
					}
					
					this->pulses->push_back(pulse);
					this->exporter->writeDataColumn(pulses, 0, 0);
				}
				else
				{
					SPDPulseUtils::deleteSPDPulse(pulse);
				}
			}
			else 
			{
				SPDPulseUtils::deleteSPDPulse(pulse);
			}

		}
		catch (SPDIOException &e) 
		{
			throw e;
		}
	}
	
	void SPDExportProcessorSubset::completeFileAndClose(SPDFile *spdFile)
	{
		try 
		{
			spdFileOut->copyAttributesFrom(spdFile);
			spdFileOut->setBoundingVolume(xMin, xMax, yMin, yMax, zMin, zMax);
			exporter->finaliseClose();
		}
		catch (SPDIOException &e) 
		{
			throw e;
		}
	}
	
	SPDExportProcessorSubset::~SPDExportProcessorSubset()
	{
		delete pulses;
	}
    
    SPDExportProcessorSubsetSpherical::SPDExportProcessorSubsetSpherical(SPDDataExporter *exporter, SPDFile *spdFileOut, double *bbox)  : SPDImporterProcessor()
	{
		this->exporter = exporter;
		this->spdFileOut = spdFileOut;
		this->bbox = bbox;
		if(exporter->requireGrid())
		{
			throw SPDException("This class does not support the export of gridded formats.");
		}
		
		try 
		{
			this->exporter->open(this->spdFileOut, this->spdFileOut->getFilePath());
		}
		catch (SPDException &e) 
		{
			throw e;
		}
		this->fileOpen = true;
		this->pulses = new std::list<SPDPulse*>();
	}
	
	void SPDExportProcessorSubsetSpherical::processImportedPulse(SPDFile *spdFile, SPDPulse *pulse) 
	{
		try
		{
			if((pulse->azimuth > bbox[0]) & (pulse->azimuth < bbox[1]) &
			   (pulse->zenith > bbox[2]) & (pulse->zenith < bbox[3]))
			{
				bool rangeWithBBox = true;
				if((pulse->pts != NULL) && (pulse->numberOfReturns > 0))
				{
					if((pulse->pts->front()->range < bbox[4]) |
					   (pulse->pts->front()->range > bbox[5]))
					{
						rangeWithBBox = false;
					}
					
					if((pulse->pts->back()->range < bbox[4]) |
					   (pulse->pts->back()->range > bbox[5]))
					{
						rangeWithBBox = false;
					}
				}
				
				if(rangeWithBBox)
				{
					this->pulses->push_back(pulse);
					this->exporter->writeDataColumn(pulses, 0, 0);
				}
				else
				{
					SPDPulseUtils::deleteSPDPulse(pulse);
				}
			}
			else 
			{
				SPDPulseUtils::deleteSPDPulse(pulse);
			}
            
		}
		catch (SPDIOException &e) 
		{
			throw e;
		}
	}
	
	void SPDExportProcessorSubsetSpherical::completeFileAndClose(SPDFile *spdFile)
	{
		try 
		{
			spdFileOut->copyAttributesFrom(spdFile);
			exporter->finaliseClose();
		}
		catch (SPDIOException &e) 
		{
			throw e;
		}
	}
	
	SPDExportProcessorSubsetSpherical::~SPDExportProcessorSubsetSpherical()
	{
		delete pulses;
	}
	
    SPDExportProcessorSubsetScan::SPDExportProcessorSubsetScan(SPDDataExporter *exporter, SPDFile *spdFileOut, double *bbox)  : SPDImporterProcessor()
	{
		this->exporter = exporter;
		this->spdFileOut = spdFileOut;
		this->bbox = bbox;
		if(exporter->requireGrid())
		{
			throw SPDException("This class does not support the export of gridded formats.");
		}
		
		try 
		{
			this->exporter->open(this->spdFileOut, this->spdFileOut->getFilePath());
		}
		catch (SPDException &e) 
		{
			throw e;
		}
		this->fileOpen = true;
		this->pulses = new std::list<SPDPulse*>();
	}
	
	void SPDExportProcessorSubsetScan::processImportedPulse(SPDFile *spdFile, SPDPulse *pulse) 
	{
		try
		{
			if((pulse->scanlineIdx > bbox[0]) & (pulse->scanlineIdx < bbox[1]) &
			   (pulse->scanline > bbox[2]) & (pulse->scanline < bbox[3]))
			{
				bool rangeWithBBox = true;
				if((pulse->pts != NULL) && (pulse->numberOfReturns > 0))
				{
					if((pulse->pts->front()->range < bbox[4]) |
					   (pulse->pts->front()->range > bbox[5]))
					{
						rangeWithBBox = false;
					}
					
					if((pulse->pts->back()->range < bbox[4]) |
					   (pulse->pts->back()->range > bbox[5]))
					{
						rangeWithBBox = false;
					}
				}
				
				if(rangeWithBBox)
				{
					this->pulses->push_back(pulse);
					this->exporter->writeDataColumn(pulses, 0, 0);
				}
				else
				{
					SPDPulseUtils::deleteSPDPulse(pulse);
				}
			}
			else 
			{
				SPDPulseUtils::deleteSPDPulse(pulse);
			}
            
		}
		catch (SPDIOException &e) 
		{
			throw e;
		}
	}
	
	void SPDExportProcessorSubsetScan::completeFileAndClose(SPDFile *spdFile)
	{
		try 
		{
			spdFileOut->copyAttributesFrom(spdFile);
			exporter->finaliseClose();
		}
		catch (SPDIOException &e) 
		{
			throw e;
		}
	}
	
	SPDExportProcessorSubsetScan::~SPDExportProcessorSubsetScan()
	{
		delete pulses;
	}
	
	
	SPDSubsetNonGriddedFile::SPDSubsetNonGriddedFile()
	{
		
	}
	
	void SPDSubsetNonGriddedFile::subsetCartesian(std::string input, std::string output, double *bbox, bool *bboxDefined) 
	{
        try
        {
            SPDFile *inSPDFile = new SPDFile(input);
            SPDFile *outSPDFile = new SPDFile(output);
            
            SPDFileReader reader;
            reader.readHeaderInfo(input, inSPDFile);		
                        
            if(!bboxDefined[0] |
               !bboxDefined[1] |
               !bboxDefined[2] |
               !bboxDefined[3] |
               !bboxDefined[4] |
               !bboxDefined[5])
            {
                
                if(!bboxDefined[0])
                {
                    bbox[0] = inSPDFile->getXMin();
                }
                
                if(!bboxDefined[1])
                {
                    bbox[1] = inSPDFile->getXMax();
                }
                
                if(!bboxDefined[2])
                {
                    bbox[2] = inSPDFile->getYMin();
                }
                
                if(!bboxDefined[3])
                {
                    bbox[3] = inSPDFile->getYMax();
                }
                
                if(!bboxDefined[4])
                {
                    bbox[4] = inSPDFile->getZMin();
                }
                
                if(!bboxDefined[5])
                {
                    bbox[5] = inSPDFile->getZMax();
                }
            }
            
            std::cout << "BBOX: [" << bbox[0] << "," << bbox[1] << "][" << bbox[2] << "," << bbox[3] << "][" << bbox[4] << "," << bbox[5] << "]\n";
            
            SPDDataExporter *exporter = new SPDNoIdxFileWriter();
            
            SPDExportProcessorSubset *exportAsRead = new SPDExportProcessorSubset(exporter, outSPDFile, bbox);
            reader.readAndProcessAllData(input, inSPDFile, exportAsRead);
            exportAsRead->completeFileAndClose(inSPDFile);
            delete outSPDFile;
            delete exportAsRead;
            delete inSPDFile;
            delete exporter;
        }
        catch(SPDException &e)
        {
            throw e;
        }
	}
    
    void SPDSubsetNonGriddedFile::subsetSpherical(std::string input, std::string output, double *bbox, bool *bboxDefined) 
    {
        try
        {                        
            SPDFile *inSPDFile = new SPDFile(input);
            SPDFile *outSPDFile = new SPDFile(output);
            
            SPDFileReader reader;
            reader.readHeaderInfo(input, inSPDFile);
            
            if(!bboxDefined[0] |
               !bboxDefined[1] |
               !bboxDefined[2] |
               !bboxDefined[3] |
               !bboxDefined[4] |
               !bboxDefined[5])
            {                
                if(!bboxDefined[0])
                {
                    bbox[0] = inSPDFile->getAzimuthMin();
                }
                
                if(!bboxDefined[1])
                {
                    bbox[1] = inSPDFile->getAzimuthMax();
                }
                
                if(!bboxDefined[2])
                {
                    bbox[2] = inSPDFile->getZenithMin();
                }
                
                if(!bboxDefined[3])
                {
                    bbox[3] = inSPDFile->getZenithMax();
                }
                
                if(!bboxDefined[4])
                {
                    bbox[4] = inSPDFile->getRangeMin();
                }
                
                if(!bboxDefined[5])
                {
                    bbox[5] = inSPDFile->getRangeMax();
                }
            }
            
            std::cout << "BBOX: [" << bbox[0] << "," << bbox[1] << "][" << bbox[2] << "," << bbox[3] << "][" << bbox[4] << "," << bbox[5] << "]\n";
            
            SPDDataExporter *exporter = new SPDNoIdxFileWriter();
            
            SPDExportProcessorSubsetSpherical *exportAsRead = new SPDExportProcessorSubsetSpherical(exporter, outSPDFile, bbox);
            reader.readAndProcessAllData(input, inSPDFile, exportAsRead);
            exportAsRead->completeFileAndClose(inSPDFile);
            delete outSPDFile;
            delete exportAsRead;
            delete inSPDFile;
            delete exporter;
        }
        catch(SPDException &e)
        {
            throw e;
        }
    }

    void SPDSubsetNonGriddedFile::subsetScan(std::string input, std::string output, double *bbox, bool *bboxDefined) 
    {
        try
        {                        
            SPDFile *inSPDFile = new SPDFile(input);
            SPDFile *outSPDFile = new SPDFile(output);
            
            SPDFileReader reader;
            reader.readHeaderInfo(input, inSPDFile);
            
            if(!bboxDefined[0] |
               !bboxDefined[1] |
               !bboxDefined[2] |
               !bboxDefined[3] |
               !bboxDefined[4] |
               !bboxDefined[5])
            {                
                if(!bboxDefined[0])
                {
                    bbox[0] = inSPDFile->getScanlineIdxMin();
                }
                
                if(!bboxDefined[1])
                {
                    bbox[1] = inSPDFile->getScanlineIdxMax();
                }
                
                if(!bboxDefined[2])
                {
                    bbox[2] = inSPDFile->getScanlineMin();
                }
                
                if(!bboxDefined[3])
                {
                    bbox[3] = inSPDFile->getScanlineMax();
                }
                
                if(!bboxDefined[4])
                {
                    bbox[4] = inSPDFile->getRangeMin();
                }
                
                if(!bboxDefined[5])
                {
                    bbox[5] = inSPDFile->getRangeMax();
                }
            }
            
            std::cout << "BBOX: [" << bbox[0] << "," << bbox[1] << "][" << bbox[2] << "," << bbox[3] << "][" << bbox[4] << "," << bbox[5] << "]\n";
            
            SPDDataExporter *exporter = new SPDNoIdxFileWriter();
            
            SPDExportProcessorSubsetScan *exportAsRead = new SPDExportProcessorSubsetScan(exporter, outSPDFile, bbox);
            reader.readAndProcessAllData(input, inSPDFile, exportAsRead);
            exportAsRead->completeFileAndClose(inSPDFile);
            delete outSPDFile;
            delete exportAsRead;
            delete inSPDFile;
            delete exporter;
        }
        catch(SPDException &e)
        {
            throw e;
        }
    }
	
	SPDSubsetNonGriddedFile::~SPDSubsetNonGriddedFile()
	{
		
	}
}


