/*
 *  SPDApplyElevationChange.cpp
 *  SPDLIB
 *
 *  Created by Pete Bunting on 11/02/2011.
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

#include "spd/SPDApplyElevationChange.h"


namespace spdlib
{	

	SPDApplyElevationChange::SPDApplyElevationChange()
	{
		
	}
	
	void SPDApplyElevationChange::applyConstantElevationChangeUnsorted(std::string inputFile, std::string outputFile, double elevConstant, bool addOffset) throw(SPDException)
	{
		try 
		{
            SPDFileReader spdReader = SPDFileReader();            
			SPDDataExporter *exporter = new SPDNoIdxFileWriter();
			
			SPDFile *spdInFile = new SPDFile(inputFile);
			SPDFile *spdOutFile = new SPDFile(outputFile);

			SPDApplyUnsortedElevChangeConstant *processData = new SPDApplyUnsortedElevChangeConstant(elevConstant, addOffset, exporter, spdOutFile);
			spdReader.readAndProcessAllData(inputFile, spdInFile, processData);
			processData->completeFileAndClose(spdInFile);
            
            delete exporter;
            delete spdInFile;
            delete spdOutFile;
            delete processData;
		}
		catch (SPDException &e) 
		{
			throw e;
		}
	}
	
	void SPDApplyElevationChange::applyConstantElevationChangeSPD(std::string inputSPDFile, std::string outputSPDFile, double elevConstant, bool addOffset, boost::uint_fast32_t blockXSize, boost::uint_fast32_t blockYSize) throw(SPDException)
	{
		try 
		{
            
            SPDFile *spdInFile = new SPDFile(inputSPDFile);
            SPDPulseProcessor *pulseStatsProcessor = new SPDApplySPDElevChangeConstant(elevConstant, addOffset);            
            SPDSetupProcessPulses processPulses = SPDSetupProcessPulses(blockXSize, blockYSize, true);
            processPulses.processPulsesWithOutputSPD(pulseStatsProcessor, spdInFile, outputSPDFile);
            
            delete spdInFile;
            delete pulseStatsProcessor;
		}
		catch (SPDException &e) 
		{
			throw e;
		}
	}
	
	void SPDApplyElevationChange::applyVariableElevationChangeUnsorted(std::string inputFile, std::string outputFile, std::string elevImage, bool addOffset) throw(SPDException)
	{
		try 
		{
			GDALAllRegister();
			
			GDALDataset *inGDALImage = (GDALDataset *) GDALOpenShared(elevImage.c_str(), GA_ReadOnly);
			
			if(inGDALImage == NULL)
			{
				GDALDestroyDriverManager();
				throw SPDException("Image could not be openned.");
			}
			
			SPDFileReader spdReader = SPDFileReader();            
			SPDDataExporter *exporter = new SPDNoIdxFileWriter();
			
			SPDFile *spdInFile = new SPDFile(inputFile);
			SPDFile *spdOutFile = new SPDFile(outputFile);
			
			SPDApplyUnsortedElevChangeVariable *processData = new SPDApplyUnsortedElevChangeVariable(inGDALImage, addOffset, exporter, spdOutFile);
			spdReader.readAndProcessAllData(inputFile, spdInFile, processData);
			processData->completeFileAndClose(spdInFile);
			
            delete exporter;
            delete spdInFile;
            delete spdOutFile;
            delete processData;
			GDALClose(inGDALImage);
			GDALDestroyDriverManager();
		}
		catch (SPDException &e) 
		{
			throw e;
		}
	}
	
	void SPDApplyElevationChange::applyVariableElevationChangeSPD(std::string inputSPDFile, std::string outputSPDFile, std::string elevImage, bool addOffset, boost::uint_fast32_t blockXSize, boost::uint_fast32_t blockYSize) throw(SPDException)
	{
		try 
		{
			GDALAllRegister();
			
			GDALDataset *inGDALImage = (GDALDataset *) GDALOpenShared(elevImage.c_str(), GA_ReadOnly);
			
			if(inGDALImage == NULL)
			{
				GDALDestroyDriverManager();
				throw SPDException("Image could not be openned.");
			}
			
            
            SPDFile *spdInFile = new SPDFile(inputSPDFile);
            SPDPulseProcessor *pulseStatsProcessor = new SPDApplySPDElevChangeVariable(inGDALImage, addOffset);            
            SPDSetupProcessPulses processPulses = SPDSetupProcessPulses(blockXSize, blockYSize, true);
            processPulses.processPulsesWithOutputSPD(pulseStatsProcessor, spdInFile, outputSPDFile);
            
            delete spdInFile;
            delete pulseStatsProcessor;
            GDALClose(inGDALImage);
			GDALDestroyDriverManager();
		}
		catch (SPDException &e) 
		{
			throw e;
		}
	}
	
	SPDApplyElevationChange::~SPDApplyElevationChange()
	{
		
	}
	
	
	
	
	
	

	SPDApplyUnsortedElevChangeConstant::SPDApplyUnsortedElevChangeConstant(double elevConstant, bool addOffset, SPDDataExporter *exporter, SPDFile *spdFileOut) throw(SPDException)
	{
		this->elevConstant = elevConstant;
		this->addOffset = addOffset;
		this->exporter = exporter;
		this->spdFileOut = spdFileOut;
		
		this->pulses = new std::list<SPDPulse*>();
	}
	
	void SPDApplyUnsortedElevChangeConstant::processImportedPulse(SPDFile *spdFile, SPDPulse *pulse) throw(SPDIOException)
	{
		try 
		{
			// Edit pulse - Using constant elevation value
			if(spdFile->getOriginDefined() == SPD_TRUE)
			{
				if(addOffset)
				{
					pulse->z0 += this->elevConstant;
				}
				else 
				{
					pulse->z0 -= this->elevConstant;
				}
			}
			
			if(pulse->numberOfReturns > 0)
			{
				for(std::vector<SPDPoint*>::iterator iterPts = pulse->pts->begin(); iterPts != pulse->pts->end(); ++iterPts)
				{
					if(addOffset)
					{
						(*iterPts)->z += this->elevConstant;
					}
					else 
					{
						(*iterPts)->z -= this->elevConstant;
					}
				}
			}
		
			// Write to disk
			this->pulses->push_back(pulse);
			this->exporter->writeDataColumn(pulses, 0, 0);
		}
		catch (SPDException &e) 
		{
			throw SPDIOException(e.what());
		}
	}
	
	void SPDApplyUnsortedElevChangeConstant::completeFileAndClose(SPDFile *spdFile)throw(SPDIOException)
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
	
	SPDApplyUnsortedElevChangeConstant::~SPDApplyUnsortedElevChangeConstant()
	{
		delete pulses;
	}
	
	
	
	SPDApplyUnsortedElevChangeVariable::SPDApplyUnsortedElevChangeVariable(GDALDataset *elevImage, bool addOffset, SPDDataExporter *exporter, SPDFile *spdFileOut) throw(SPDException)
	{
		this->elevImage = elevImage;
		this->addOffset = addOffset;
		this->exporter = exporter;
		this->spdFileOut = spdFileOut;
		
		this->pulses = new std::list<SPDPulse*>();
		
		this->pxlVals = new float*[4];
		for(unsigned int i = 0; i < 4; ++i)
		{
			this->pxlVals[i] = new float[4];
		}
		prevImgX = new boost::int_fast32_t[4];
		prevImgY = new boost::int_fast32_t[4];
		first = true;
	}
	
	void SPDApplyUnsortedElevChangeVariable::processImportedPulse(SPDFile *spdFile, SPDPulse *pulse) throw(SPDIOException)
	{
		try 
		{
			SPDImageUtils imgUtils;
			// Get elevation Value
			double elevationVal = 0;
            boost::int_fast32_t *imgX = new boost::int_fast32_t[4];
            boost::int_fast32_t *imgY = new boost::int_fast32_t[4];
			float xOff = 0;
			float yOff = 0;
			imgUtils.getPixelPointLocations(elevImage, pulse->xIdx, pulse->yIdx, spdFile->getSpatialReference(), imgX, imgY, &xOff, &yOff);
			if(this->first | 
			   (prevImgX[0] != imgX[0]) | (prevImgY[0] != imgY[0]) |
			   (prevImgX[1] != imgX[1]) | (prevImgY[1] != imgY[1]) |
			   (prevImgX[2] != imgX[2]) | (prevImgY[2] != imgY[2]) |
			   (prevImgX[3] != imgX[3]) | (prevImgY[3] != imgY[3]))
			{
				imgUtils.getImagePixelPtValues(elevImage, imgX, imgY, pxlVals, 2, 1);
				prevImgX[0] = imgX[0];
				prevImgY[0] = imgY[0];
				prevImgX[1] = imgX[1];
				prevImgY[1] = imgY[1];
				prevImgX[2] = imgX[2];
				prevImgY[2] = imgY[2];
				prevImgX[3] = imgX[3];
				prevImgY[3] = imgY[3];
				this->first = false;
			}
			
			elevationVal = imgUtils.cubicInterpValue(xOff, yOff, pxlVals, 4);
			
			// Edit pulse
			if(spdFile->getOriginDefined() == SPD_TRUE)
			{
				if(addOffset)
				{
					pulse->z0 += elevationVal;
				}
				else 
				{
					pulse->z0 -= elevationVal;
				}
			}
			
			if(pulse->numberOfReturns > 0)
			{
				for(std::vector<SPDPoint*>::iterator iterPts = pulse->pts->begin(); iterPts != pulse->pts->end(); ++iterPts)
				{
					if(addOffset)
					{
						(*iterPts)->z += elevationVal;
					}
					else 
					{
						(*iterPts)->z -= elevationVal;
					}
				}
			}
			
			// Write to disk
			this->pulses->push_back(pulse);
			this->exporter->writeDataColumn(pulses, 0, 0);
		}
		catch (SPDException &e) 
		{
			throw SPDIOException(e.what());
		}
	}
	
	void SPDApplyUnsortedElevChangeVariable::completeFileAndClose(SPDFile *spdFile)throw(SPDIOException)
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
	
	SPDApplyUnsortedElevChangeVariable::~SPDApplyUnsortedElevChangeVariable()
	{
		delete pulses;
		for(unsigned int i = 0; i < 4; ++i)
		{
			delete[] this->pxlVals[i];
		}
		delete[] this->pxlVals;
	}
	
	
	
    SPDApplySPDElevChangeConstant::SPDApplySPDElevChangeConstant(double elevConstant, bool addOffset)
    {
        this->elevConstant = elevConstant;
        this->addOffset = addOffset;
    }
        
    void SPDApplySPDElevChangeConstant::processDataColumn(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, SPDXYPoint *cenPts) throw(SPDProcessingException)
    {
        try 
		{
			for(std::vector<SPDPulse*>::iterator iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
			{
				// Edit pulse - Using constant elevation value
				if(inSPDFile->getOriginDefined() == SPD_TRUE)
				{
					if(addOffset)
					{
						(*iterPulses)->z0 += this->elevConstant;
					}
					else 
					{
						(*iterPulses)->z0 -= this->elevConstant;
					}
				}
				
				if((*iterPulses)->numberOfReturns > 0)
				{
					for(std::vector<SPDPoint*>::iterator iterPts = (*iterPulses)->pts->begin(); iterPts != (*iterPulses)->pts->end(); ++iterPts)
					{
						if(addOffset)
						{
							(*iterPts)->z += this->elevConstant;
						}
						else 
						{
							(*iterPts)->z -= this->elevConstant;
						}
					}
				}
			}
			
		}
		catch (SPDException &e) 
		{
			throw SPDProcessingException(e.what());
		}

    }
        
    SPDApplySPDElevChangeConstant::~SPDApplySPDElevChangeConstant()
    {
        
    }
	
    
	
    SPDApplySPDElevChangeVariable::SPDApplySPDElevChangeVariable(GDALDataset *elevImage, bool addOffset)
    {
        this->elevImage = elevImage;
		this->addOffset = addOffset;
		this->pxlVals = new float*[4];
		for(unsigned int i = 0; i < 4; ++i)
		{
			this->pxlVals[i] = new float[4];
		}
		prevImgX = new boost::int_fast32_t[4];
		prevImgY = new boost::int_fast32_t[4];
		first = true;
    }
  
    void SPDApplySPDElevChangeVariable::processDataColumn(SPDFile *inSPDFile, std::vector<SPDPulse*> *pulses, SPDXYPoint *cenPts) throw(SPDProcessingException)
    {
        SPDImageUtils imgUtils;
		try 
		{
			double elevationVal = 0;
			for(std::vector<SPDPulse*>::iterator iterPulses = pulses->begin(); iterPulses != pulses->end(); ++iterPulses)
			{
				// Get elevation Value			
				elevationVal = 0;
                boost::int_fast32_t *imgX = new boost::int_fast32_t[4];
                boost::int_fast32_t *imgY = new boost::int_fast32_t[4];
				float xOff = 0;
				float yOff = 0;
				imgUtils.getPixelPointLocations(elevImage, (*iterPulses)->xIdx, (*iterPulses)->yIdx, inSPDFile->getSpatialReference(), imgX, imgY, &xOff, &yOff);
				
				if(this->first | 
				   (prevImgX[0] != imgX[0]) | (prevImgY[0] != imgY[0]) |
				   (prevImgX[1] != imgX[1]) | (prevImgY[1] != imgY[1]) |
				   (prevImgX[2] != imgX[2]) | (prevImgY[2] != imgY[2]) |
				   (prevImgX[3] != imgX[3]) | (prevImgY[3] != imgY[3]))
				{
					imgUtils.getImagePixelPtValues(elevImage, imgX, imgY, pxlVals, 2, 1);
					prevImgX[0] = imgX[0];
					prevImgY[0] = imgY[0];
					prevImgX[1] = imgX[1];
					prevImgY[1] = imgY[1];
					prevImgX[2] = imgX[2];
					prevImgY[2] = imgY[2];
					prevImgX[3] = imgX[3];
					prevImgY[3] = imgY[3];
					this->first = false;
				}
				
				elevationVal = imgUtils.cubicInterpValue(xOff, yOff, pxlVals, 4);
				
				// Edit pulse
				if(inSPDFile->getOriginDefined() == SPD_TRUE)
				{
					if(addOffset)
					{
						(*iterPulses)->z0 += elevationVal;
					}
					else 
					{
						(*iterPulses)->z0 -= elevationVal;
					}
				}
				
				if((*iterPulses)->numberOfReturns > 0)
				{
					for(std::vector<SPDPoint*>::iterator iterPts = (*iterPulses)->pts->begin(); iterPts != (*iterPulses)->pts->end(); ++iterPts)
					{
						if(addOffset)
						{
							(*iterPts)->z += elevationVal;
						}
						else 
						{
							(*iterPts)->z -= elevationVal;
						}
					}
				}
			}
			
		}
		catch (SPDException &e) 
		{
			throw SPDProcessingException(e.what());
		}
    }

    SPDApplySPDElevChangeVariable::~SPDApplySPDElevChangeVariable()
    {
        delete[] prevImgX;
		delete[] prevImgY;
		
		for(unsigned int i = 0; i < 4; ++i)
		{
			delete[] this->pxlVals[i];
		}
		delete[] this->pxlVals;
    }

}




