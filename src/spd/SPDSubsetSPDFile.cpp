/*
 *  SPDSubsetSPDFile.cpp
 *  SPDLIB
 *
 *  Created by Pete Bunting on 28/12/2010.
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

#include "spd/SPDSubsetSPDFile.h"


namespace spdlib
{	
	SPDSubsetSPDFile::SPDSubsetSPDFile()
	{
		
	}
	
	void SPDSubsetSPDFile::subsetSPDFile(string inputFile, string outputFile, double *bbox, bool *bboxDefined) throw(SPDException)
	{
		try 
		{
			SPDFile *spdFile = new SPDFile(inputFile);
			SPDFile *spdOutFile = new SPDFile(outputFile);
			
			SPDFileReader reader;
			reader.readHeaderInfo(spdFile->getFilePath(), spdFile);
			spdOutFile->copyAttributesFrom(spdFile);
			
			if(!bboxDefined[0] |
			   !bboxDefined[1] |
			   !bboxDefined[2] |
			   !bboxDefined[3] |
			   !bboxDefined[4] |
			   !bboxDefined[5])
			{
				if(!bboxDefined[0])
				{
					bbox[0] = spdFile->getXMin();
				}
				
				if(!bboxDefined[1])
				{
					bbox[1] = spdFile->getXMax();
				}
				
				if(!bboxDefined[2])
				{
					bbox[2] = spdFile->getYMin();
				}
				
				if(!bboxDefined[3])
				{
					bbox[3] = spdFile->getYMax();
				}
				
				if(!bboxDefined[4])
				{
					bbox[4] = spdFile->getZMin();
				}
				
				if(!bboxDefined[5])
				{
					bbox[5] = spdFile->getZMax();
				}
			}
			
			boost::uint_fast32_t startCol = 0;
			boost::uint_fast32_t endCol = 0;
			boost::uint_fast32_t startRow = 0;
			boost::uint_fast32_t endRow = 0;
			
			double tmpDist = 0;
			boost::uint_fast32_t numCols = 0;
			boost::uint_fast32_t numRows = 0;
			try
			{
				// Define Starting Column
				tmpDist = bbox[0] - spdFile->getXMin();
				if(tmpDist < 0)
				{
					startCol = 0;
				}
				else
				{
					numCols = numeric_cast<boost::uint_fast32_t>(tmpDist/spdFile->getBinSize());
					startCol = numCols;
					spdOutFile->setXMin(spdFile->getXMin() + (((float)numCols) * spdFile->getBinSize()));
				}
				
				// Define End Column
				tmpDist = spdFile->getXMax() - bbox[1];
				if(tmpDist < 0)
				{
					endCol = spdFile->getNumberBinsX();
				}
				else
				{
					numCols = numeric_cast<boost::uint_fast32_t>(tmpDist/spdFile->getBinSize());
					endCol = spdFile->getNumberBinsX() - numCols;
					spdOutFile->setXMax(spdFile->getXMax() - (((float)numCols) * spdFile->getBinSize()));
				}
				
				// Define Starting Row
				tmpDist = spdFile->getYMax() - bbox[3];
				if(tmpDist < 0)
				{
					startRow = 0;
				}
				else
				{
					numRows = numeric_cast<boost::uint_fast32_t>(tmpDist/spdFile->getBinSize());
					startRow = numRows;
					spdOutFile->setYMax(spdFile->getYMax() - (((float)numRows) * spdFile->getBinSize()));
				}
				
				// Define End Row
				tmpDist = bbox[2] - spdFile->getYMin();
				if(tmpDist < 0)
				{
					endRow = spdFile->getNumberBinsY();
				}
				else
				{
					numRows = numeric_cast<boost::uint_fast32_t>(tmpDist/spdFile->getBinSize());
					endRow = spdFile->getNumberBinsY() - numRows;
					spdOutFile->setYMin(spdFile->getYMin() + (((float)numRows) * spdFile->getBinSize()));
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
			
			if(endCol <= startCol)
			{
				throw SPDProcessingException("Define subset is not within the input file (X Axis).");
			}
			
			if(endRow <= startRow)
			{
				throw SPDProcessingException("Define subset is not within the input file (Y Axis).");
			}
			
			spdOutFile->setNumberBinsX(endCol - startCol);
			spdOutFile->setNumberBinsY(endRow - startRow);
			
			cout << "Subset To: X[" << bbox[0] << "," << bbox[1] << "] Y[" << bbox[2] << "," << bbox[3] << "] Z[" << bbox[4] << "," << bbox[5] << "]\n";
			cout << "New SPD file has " << spdOutFile->getNumberBinsX() << " columns and " << spdOutFile->getNumberBinsY() << " rows\n";
			
			list<SPDPulse*> **pulses = new list<SPDPulse*>*[spdFile->getNumberBinsX()];
			for(boost::uint_fast32_t i = 0; i < spdFile->getNumberBinsX(); ++i)
			{
				pulses[i] = new list<SPDPulse*>();
			}
			
			SPDSeqFileWriter spdWriter;
			spdWriter.open(spdOutFile, outputFile);
			SPDFileIncrementalReader spdIncReader;
			spdIncReader.open(spdFile);
			
			int feedback = spdOutFile->getNumberBinsY()/10;
			int feedbackCounter = 0;
			
			numRows = 0;
			cout << "Started (Write Data) .";
			for(boost::uint_fast32_t rows = startRow; rows < endRow; ++rows)
			{
				if((spdOutFile->getNumberBinsY() > 10) && (numRows % feedback == 0))
				{
					cout << "." << feedbackCounter << "." << flush;
					feedbackCounter += 10;
				}
				
				spdIncReader.readPulseDataRow(rows, pulses);
				
				numCols = 0;
				bool zWithBBox = true;
				for(boost::uint_fast32_t cols = startCol; cols < endCol; ++cols)
				{
					if(bboxDefined[4] | bboxDefined[5])
					{
						for(list<SPDPulse*>::iterator iterPulses = pulses[cols]->begin(); iterPulses != pulses[cols]->end(); )
						{
							zWithBBox = true;
							if(((*iterPulses)->pts != NULL) && ((*iterPulses)->numberOfReturns > 0))
							{
								if(((*iterPulses)->pts->front()->z < bbox[4]) |
								   ((*iterPulses)->pts->front()->z > bbox[5]))
								{
									zWithBBox = false;
								}
								
								if(((*iterPulses)->pts->back()->z < bbox[4]) |
								   ((*iterPulses)->pts->back()->z > bbox[5]))
								{
									zWithBBox = false;
								}
							}
							
							if((*iterPulses)->numOfReceivedBins > 0)
							{
								double tempX = 0;
								double tempY = 0;
								double tempZ = 0;
								
								SPDConvertToCartesian((*iterPulses)->zenith, (*iterPulses)->azimuth, (*iterPulses)->rangeToWaveformStart, (*iterPulses)->x0, (*iterPulses)->y0, (*iterPulses)->z0, &tempX, &tempY, &tempZ);
								if((tempZ < bbox[4]) |
								   (tempZ > bbox[5]))
								{
									zWithBBox = false;
								}
								
								SPDConvertToCartesian((*iterPulses)->zenith, (*iterPulses)->azimuth, ((*iterPulses)->rangeToWaveformStart+((((*iterPulses)->numOfReceivedBins-1)*SPD_SPEED_OF_LIGHT_NS))/2), (*iterPulses)->x0, (*iterPulses)->y0, (*iterPulses)->z0, &tempX, &tempY, &tempZ);
								if((tempZ < bbox[4]) |
								   (tempZ > bbox[5]))
								{
									zWithBBox = false;
								}
							}
							
							if(zWithBBox)
							{
								++iterPulses; // Next Pulse...
							}
							else 
							{
								SPDPulseUtils::deleteSPDPulse(*iterPulses);
								iterPulses = pulses[cols]->erase(iterPulses);
							}
						}
					}
					
					spdWriter.writeDataColumn(pulses[cols], numCols, numRows);
					++numCols;
				}
				
				for(boost::uint_fast32_t i = 0; i < spdFile->getNumberBinsX(); ++i)
				{
					for(list<SPDPulse*>::iterator iterPulses = pulses[i]->begin(); iterPulses != pulses[i]->end(); )
					{
						SPDPulseUtils::deleteSPDPulse(*iterPulses);
						iterPulses = pulses[i]->erase(iterPulses);
					}
				}
				++numRows;
			}
			cout << ". Complete.\n";
			spdIncReader.close();
			spdWriter.finaliseClose();
		}
		catch (SPDException *e) 
		{
			throw e;
		}
	}
	
    void SPDSubsetSPDFile::subsetSPDFile(string inputFile, string outputFile, string shapefile) throw(SPDException)
    {
        try
        {
            // Get Vector Geometry
            SPDVectorUtils vecUtils; 
            OGRGeometryCollection *geomCollect = vecUtils.getGeometryCollection(shapefile);
            OGREnvelope geomEnv;
            geomCollect->getEnvelope(&geomEnv);
            
            SPDFile *spdFile = new SPDFile(inputFile);
			SPDFile *spdOutFile = new SPDFile(outputFile);
			
			SPDFileReader reader;
			reader.readHeaderInfo(spdFile->getFilePath(), spdFile);
			spdOutFile->copyAttributesFrom(spdFile);
            
            boost::uint_fast32_t startCol = 0;
			boost::uint_fast32_t endCol = 0;
			boost::uint_fast32_t startRow = 0;
			boost::uint_fast32_t endRow = 0;
			
			double tmpDist = 0;
			boost::uint_fast32_t numCols = 0;
			boost::uint_fast32_t numRows = 0;
			try
			{
				// Define Starting Column
				tmpDist = geomEnv.MinX - spdFile->getXMin();
				if(tmpDist < 0)
				{
					startCol = 0;
				}
				else
				{
					numCols = numeric_cast<boost::uint_fast32_t>(tmpDist/spdFile->getBinSize());
					startCol = numCols;
					spdOutFile->setXMin(spdFile->getXMin() + (((float)numCols) * spdFile->getBinSize()));
				}
				
				// Define End Column
				tmpDist = spdFile->getXMax() - geomEnv.MaxX;
				if(tmpDist < 0)
				{
					endCol = spdFile->getNumberBinsX();
				}
				else
				{
					numCols = numeric_cast<boost::uint_fast32_t>(tmpDist/spdFile->getBinSize());
					endCol = spdFile->getNumberBinsX() - numCols;
					spdOutFile->setXMax(spdFile->getXMax() - (((float)numCols) * spdFile->getBinSize()));
				}
				
				// Define Starting Row
				tmpDist = spdFile->getYMax() - geomEnv.MaxY;
				if(tmpDist < 0)
				{
					startRow = 0;
				}
				else
				{
					numRows = numeric_cast<boost::uint_fast32_t>(tmpDist/spdFile->getBinSize());
					startRow = numRows;
					spdOutFile->setYMax(spdFile->getYMax() - (((float)numRows) * spdFile->getBinSize()));
				}
				
				// Define End Row
				tmpDist = geomEnv.MinY - spdFile->getYMin();
				if(tmpDist < 0)
				{
					endRow = spdFile->getNumberBinsY();
				}
				else
				{
					numRows = numeric_cast<boost::uint_fast32_t>(tmpDist/spdFile->getBinSize());
					endRow = spdFile->getNumberBinsY() - numRows;
					spdOutFile->setYMin(spdFile->getYMin() + (((float)numRows) * spdFile->getBinSize()));
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
			
			if(endCol <= startCol)
			{
				throw SPDProcessingException("Define subset is not within the input file (X Axis).");
			}
			
			if(endRow <= startRow)
			{
				throw SPDProcessingException("Define subset is not within the input file (Y Axis).");
			}
			
			spdOutFile->setNumberBinsX(endCol - startCol);
			spdOutFile->setNumberBinsY(endRow - startRow);
			
			cout << "Subset To: X[" << geomEnv.MinX << "," << geomEnv.MaxX << "] Y[" << geomEnv.MinY << "," <<geomEnv.MaxY << "]\n";
			cout << "New SPD file has " << spdOutFile->getNumberBinsX() << " columns and " << spdOutFile->getNumberBinsY() << " rows\n";
            
            list<SPDPulse*> **pulses = new list<SPDPulse*>*[spdFile->getNumberBinsX()];
			for(boost::uint_fast32_t i = 0; i < spdFile->getNumberBinsX(); ++i)
			{
				pulses[i] = new list<SPDPulse*>();
			}
			
			SPDSeqFileWriter spdWriter;
			spdWriter.open(spdOutFile, outputFile);
			SPDFileIncrementalReader spdIncReader;
			spdIncReader.open(spdFile);
			
			int feedback = spdOutFile->getNumberBinsY()/10;
			int feedbackCounter = 0;
            
            OGRPoint *pt = new OGRPoint();
			
			numRows = 0;
			cout << "Started (Write Data) .";
			for(boost::uint_fast32_t rows = startRow; rows < endRow; ++rows)
			{
				if((spdOutFile->getNumberBinsY() > 10) && (numRows % feedback == 0))
				{
					cout << "." << feedbackCounter << "." << flush;
					feedbackCounter += 10;
				}
				
				spdIncReader.readPulseDataRow(rows, pulses);
				
				numCols = 0;
				for(boost::uint_fast32_t cols = startCol; cols < endCol; ++cols)
				{
                    for(list<SPDPulse*>::iterator iterPulses = pulses[cols]->begin(); iterPulses != pulses[cols]->end(); )
                    {
                        pt->setX((*iterPulses)->xIdx);
                        pt->setY((*iterPulses)->yIdx);
                        
                        if(geomCollect->Contains(pt))
                        {
                            ++iterPulses; // Next Pulse...
                        }
                        else 
                        {
                            SPDPulseUtils::deleteSPDPulse(*iterPulses);
                            iterPulses = pulses[cols]->erase(iterPulses);
                        }
                    }
                                        
					spdWriter.writeDataColumn(pulses[cols], numCols, numRows);
					++numCols;
				}
				
				for(boost::uint_fast32_t i = 0; i < spdFile->getNumberBinsX(); ++i)
				{
					for(list<SPDPulse*>::iterator iterPulses = pulses[i]->begin(); iterPulses != pulses[i]->end(); )
					{
						SPDPulseUtils::deleteSPDPulse(*iterPulses);
						iterPulses = pulses[i]->erase(iterPulses);
					}
				}
				++numRows;
			}
            delete pt;
            
			cout << ". Complete.\n";
			spdIncReader.close();
			spdWriter.finaliseClose();
            
        }
        catch(SPDException &e)
        {
            throw e;
        }
    }
    
    void SPDSubsetSPDFile::subsetSPDFileHeightOnly(string inputFile, string outputFile, double lowHeight, double upperHeight) throw(SPDException)
    {
        try 
		{
			SPDFile *spdFile = new SPDFile(inputFile);
			SPDFile *spdOutFile = new SPDFile(outputFile);
			
			SPDFileReader reader;
			reader.readHeaderInfo(spdFile->getFilePath(), spdFile);
			spdOutFile->copyAttributesFrom(spdFile);
			
            vector<SPDPulse*>::iterator iterPulses;
            vector<SPDPoint*>::iterator iterPts;
			vector<SPDPulse*> **pulses = new vector<SPDPulse*>*[spdFile->getNumberBinsX()];
			for(boost::uint_fast32_t i = 0; i < spdFile->getNumberBinsX(); ++i)
			{
				pulses[i] = new vector<SPDPulse*>();
			}
			
			SPDSeqFileWriter spdWriter;
			spdWriter.open(spdOutFile, outputFile);
			SPDFileIncrementalReader spdIncReader;
			spdIncReader.open(spdFile);
			
			int feedback = spdOutFile->getNumberBinsY()/10;
			int feedbackCounter = 0;
			
			boost::uint_fast32_t numRows = 0;
			cout << "Started (Write Data) .";
			for(boost::uint_fast32_t rows = 0; rows < spdOutFile->getNumberBinsY(); ++rows)
			{
				if((spdOutFile->getNumberBinsY() > 10) && (numRows % feedback == 0))
				{
					cout << "." << feedbackCounter << "." << flush;
					feedbackCounter += 10;
				}
				
				spdIncReader.readPulseDataRow(rows, pulses);
				
				boost::uint_fast32_t numCols = 0;
				for(boost::uint_fast32_t cols = 0; cols < spdOutFile->getNumberBinsX(); ++cols)
				{
					for(iterPulses = pulses[cols]->begin(); iterPulses != pulses[cols]->end(); ++iterPulses)
                    {
                        if((*iterPulses)->numberOfReturns > 0)
                        {
                            for(iterPts = (*iterPulses)->pts->begin(); iterPts != (*iterPulses)->pts->end(); )
                            {
                                if(((*iterPts)->height < lowHeight) | ((*iterPts)->height > upperHeight))
                                {
                                    delete *iterPts;
                                    iterPts = (*iterPulses)->pts->erase(iterPts);
                                    --(*iterPulses)->numberOfReturns;
                                }
                                else
                                {
                                    ++iterPts;
                                }
                            }
                        }
                    }
					
					spdWriter.writeDataColumn(pulses[cols], numCols, numRows);
					++numCols;
				}
				
				for(boost::uint_fast32_t i = 0; i < spdFile->getNumberBinsX(); ++i)
				{
					for(iterPulses = pulses[i]->begin(); iterPulses != pulses[i]->end(); )
					{
						SPDPulseUtils::deleteSPDPulse(*iterPulses);
						iterPulses = pulses[i]->erase(iterPulses);
					}
				}
				++numRows;
			}
			cout << ". Complete.\n";
			spdIncReader.close();
			spdWriter.finaliseClose();
		}
		catch (SPDException *e) 
		{
			throw e;
		}
    }
    
    void SPDSubsetSPDFile::subsetSphericalSPDFile(string inputFile, string outputFile, double *bbox, bool *bboxDefined) throw(SPDException)
    {
        try 
		{
			SPDFile *spdFile = new SPDFile(inputFile);
			SPDFile *spdOutFile = new SPDFile(outputFile);
			
			SPDFileReader reader;
			reader.readHeaderInfo(spdFile->getFilePath(), spdFile);
            
            if(spdFile->getIndexType() != SPD_SPHERICAL_IDX)
            {
                throw SPDException("This function only supports subsetting spherically indexed SPD files.");
            }
            
			spdOutFile->copyAttributesFrom(spdFile);
			
			if(!bboxDefined[0] |
			   !bboxDefined[1] |
			   !bboxDefined[2] |
			   !bboxDefined[3] |
			   !bboxDefined[4] |
			   !bboxDefined[5])
			{
				if(!bboxDefined[0])
				{
					bbox[0] = spdFile->getAzimuthMin();
				}
				
				if(!bboxDefined[1])
				{
					bbox[1] = spdFile->getAzimuthMax();
				}
				
				if(!bboxDefined[2])
				{
                    bbox[2] = spdFile->getZenithMin();
				}
				
				if(!bboxDefined[3])
				{
                    bbox[3] = spdFile->getZenithMax();
				}
				
				if(!bboxDefined[4])
				{
					bbox[4] = spdFile->getRangeMin();
				}
				
				if(!bboxDefined[5])
				{
					bbox[5] = spdFile->getRangeMax();
				}
			}
			
            cout << "Input SPD File has " << spdFile->getNumberBinsX() << " columns and " << spdFile->getNumberBinsY() << " rows\n";
            
			boost::uint_fast32_t startCol = 0;
			boost::uint_fast32_t endCol = 0;
			boost::uint_fast32_t startRow = 0;
			boost::uint_fast32_t endRow = 0;
			
			double tmpDist = 0;
			boost::uint_fast32_t numCols = 0;
			boost::uint_fast32_t numRows = 0;
			try
			{
				// Define Starting Column
				tmpDist = bbox[0] - spdFile->getAzimuthMin();
				if(tmpDist < 0)
				{
					startCol = 0;
				}
				else
				{
					numCols = numeric_cast<boost::uint_fast32_t>(tmpDist/spdFile->getBinSize());
					startCol = numCols;
					spdOutFile->setAzimuthMin(spdFile->getAzimuthMin() + (((float)numCols) * spdFile->getBinSize()));
				}
				
				// Define End Column
				tmpDist = bbox[1] - bbox[0];
				numCols = numeric_cast<boost::uint_fast32_t>(tmpDist/spdFile->getBinSize());
                endCol = startCol + numCols;
                
                spdOutFile->setAzimuthMax(spdFile->getAzimuthMin() + (((float)endCol) * spdFile->getBinSize()));
				
				// Define Starting Row
				tmpDist = bbox[2] - spdFile->getZenithMin();
				if(tmpDist < 0)
				{
					startRow = 0;
				}
				else
				{
					numRows = numeric_cast<boost::uint_fast32_t>(tmpDist/spdFile->getBinSize());
					startRow = numRows;
					spdOutFile->setZenithMin(spdFile->getZenithMin() + (((float)numRows) * spdFile->getBinSize()));
				}
				
				// Define End Row
				tmpDist = bbox[3] - bbox[2];
				numRows = numeric_cast<boost::uint_fast32_t>(tmpDist/spdFile->getBinSize());
                endRow = startRow + numRows;
                
                if(endRow > spdFile->getNumberBinsY())
                {
                    endRow = spdFile->getNumberBinsY();
                }
                
                spdOutFile->setZenithMax(spdFile->getZenithMin() + (((float)endRow) * spdFile->getBinSize()));
                
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
			
            //cout << "Cols: [" << startCol << "," << endCol << "]\n";
            //cout << "Rows: [" << startRow << "," << endRow << "]\n";
            
			if(endCol <= startCol)
			{
				throw SPDProcessingException("Define subset is not within the input file (X Axis).");
			}
			
			if(endRow <= startRow)
			{
				throw SPDProcessingException("Define subset is not within the input file (Y Axis).");
			}
			
			spdOutFile->setNumberBinsX(endCol - startCol);
			spdOutFile->setNumberBinsY(endRow - startRow);
			
			cout << "Subset To: Azimuth[" << bbox[0] << "," << bbox[1] << "] Zenith[" << bbox[2] << "," << bbox[3] << "] Range[" << bbox[4] << "," << bbox[5] << "]\n";
			cout << "New SPD file has " << spdOutFile->getNumberBinsX() << " columns and " << spdOutFile->getNumberBinsY() << " rows\n";
			
			list<SPDPulse*> **pulses = new list<SPDPulse*>*[spdFile->getNumberBinsX()];
			for(boost::uint_fast32_t i = 0; i < spdFile->getNumberBinsX(); ++i)
			{
				pulses[i] = new list<SPDPulse*>();
			}
			
			SPDSeqFileWriter spdWriter;
			spdWriter.open(spdOutFile, outputFile);
			SPDFileIncrementalReader spdIncReader;
			spdIncReader.open(spdFile);
			
			int feedback = spdOutFile->getNumberBinsY()/10;
			int feedbackCounter = 0;
			
			numRows = 0;
			cout << "Started (Write Data) .";
			for(boost::uint_fast32_t rows = startRow; rows < endRow; ++rows)
			{
				if((spdOutFile->getNumberBinsY() > 10) && (numRows % feedback == 0))
				{
					cout << "." << feedbackCounter << "." << flush;
					feedbackCounter += 10;
				}
				
				spdIncReader.readPulseDataRow(rows, pulses);
				
				numCols = 0;
				bool zWithBBox = true;
				for(boost::uint_fast32_t cols = startCol; cols < endCol; ++cols)
				{
					if(bboxDefined[4] | bboxDefined[5])
					{
						for(list<SPDPulse*>::iterator iterPulses = pulses[cols]->begin(); iterPulses != pulses[cols]->end(); )
						{
							zWithBBox = true;
							if(((*iterPulses)->pts != NULL) && ((*iterPulses)->numberOfReturns > 0))
							{
								if(((*iterPulses)->pts->front()->range < bbox[4]) |
								   ((*iterPulses)->pts->front()->range > bbox[5]))
								{
									zWithBBox = false;
								}
								
								if(((*iterPulses)->pts->back()->range < bbox[4]) |
								   ((*iterPulses)->pts->back()->range > bbox[5]))
								{
									zWithBBox = false;
								}
							}
							
							if(zWithBBox)
							{
								++iterPulses; // Next Pulse...
							}
							else 
							{
								SPDPulseUtils::deleteSPDPulse(*iterPulses);
								iterPulses = pulses[cols]->erase(iterPulses);
							}
						}
					}
					
					spdWriter.writeDataColumn(pulses[cols], numCols, numRows);
					++numCols;
				}
				
				for(boost::uint_fast32_t i = 0; i < spdFile->getNumberBinsX(); ++i)
				{
					for(list<SPDPulse*>::iterator iterPulses = pulses[i]->begin(); iterPulses != pulses[i]->end(); )
					{
						SPDPulseUtils::deleteSPDPulse(*iterPulses);
						iterPulses = pulses[i]->erase(iterPulses);
					}
				}
				++numRows;
			}
			cout << ". Complete.\n";
			spdIncReader.close();
			spdWriter.finaliseClose();
		}
		catch (SPDException *e) 
		{
			throw e;
		}
    }
    
	SPDSubsetSPDFile::~SPDSubsetSPDFile()
	{
		
	}
    
    
    
    SPDUPDPulseSubset::SPDUPDPulseSubset()
	{
		
	}
	
	void SPDUPDPulseSubset::subsetUPD(string inputFile, string outputFile, boost::uint_fast32_t startPulse, boost::uint_fast32_t numOfPulses)throw(SPDIOException)
	{
		try 
		{
			list<SPDPulse*> *pulses = new list<SPDPulse*>();
			
			SPDFile *spdInFile = new SPDFile(inputFile);
			SPDFile *spdOutFile = new SPDFile(outputFile);
			
			SPDFileIncrementalReader spdIncReader;
			spdIncReader.open(spdInFile);
			spdIncReader.readPulseData(pulses, startPulse, numOfPulses);
			spdIncReader.close();
			
			spdOutFile->copyAttributesFrom(spdInFile);
			SPDNoIdxFileWriter updWriter;
			updWriter.open(spdOutFile, outputFile);
			updWriter.writeDataColumn(pulses, 0, 0);
			updWriter.finaliseClose();
			
		}
		catch (SPDIOException &e) 
		{
			throw e;
		}
		
	}
	
	SPDUPDPulseSubset::~SPDUPDPulseSubset()
	{
		
	}
}
