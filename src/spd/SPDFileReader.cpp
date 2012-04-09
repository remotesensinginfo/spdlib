/*
 *  SPDFileReader.cpp
 *  spdlib_prj
 *
 *  Created by Pete Bunting on 14/09/2009.
 *  Copyright 2009 SPDLib. All rights reserved.
 *
 *  This file is part of SPDLib.
 *
 *  Permission is hereby granted, free of charge, to any person 
 *  obtaining a copy of this software and associated documentation 
 *  files (the "Software"), to deal in the Software without restriction, 
 *  including without limitation the rights to use, copy, modify, 
 *  merge, publish, distribute, sublicense, and/or sell copies of the 
 *  Software, and to permit persons to whom the Software is furnished 
 *  to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be 
 *  included in all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
 *  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
 *  OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
 *  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR 
 *  ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF 
 *  CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION 
 *  WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#include "spd/SPDFileReader.h"

namespace spdlib
{
	
	SPDFileReader::SPDFileReader(bool convertCoords, string outputProjWKT, string schema, boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold) : SPDDataImporter(convertCoords, outputProjWKT, schema, indexCoords, defineOrigin, originX, originY, originZ, waveNoiseThreshold)
	{
		
	}
    
    SPDDataImporter* SPDFileReader::getInstance(bool convertCoords, string outputProjWKT, string schema, boost::uint_fast16_t indexCoords, bool defineOrigin, double originX, double originY, float originZ, float waveNoiseThreshold)
    {
        return new SPDFileReader(convertCoords, outputProjWKT, schema, indexCoords, defineOrigin, originX, originY, originZ, waveNoiseThreshold);
    }
	
	list<SPDPulse*>* SPDFileReader::readAllDataToList(string inputFile, SPDFile *spdFile)throw(SPDIOException)
	{
		SPDPointUtils ptsUtils;
		SPDPulseUtils pulseUtils;
		list<SPDPulse*> *pulses = new list<SPDPulse*>();
		try 
		{
			Exception::dontPrint();
			H5File *spdInFile = NULL;
            
			this->readHeaderInfo(inputFile, spdFile);
			
			if(convertCoords)
			{
				this->initCoordinateSystemTransformation(spdFile);
				
				spdFile->setSpatialReference(outputProjWKT);
			}
			bool changeIdxMethod = false;
			if((spdFile->getPulseIdxMethod() == indexCoords) | (indexCoords == SPD_IDX_UNCHANGED))
			{
				changeIdxMethod = false;
			}
			else 
			{
				spdFile->setPulseIdxMethod(indexCoords);
				changeIdxMethod = true;
			}
			
			try 
			{
				spdInFile = new H5File( spdFile->getFilePath(), H5F_ACC_RDONLY );
			}
			catch (FileIException &e) 
			{
				string message  = string("Could not open SPD file: ") + spdFile->getFilePath();
				throw SPDIOException(message);
			}
			
			double xMin = 0;
			double yMin = 0;
			double xMax = 0;
			double yMax = 0;
			bool first = true;
            
			boost::uint_fast64_t multipleOfBlocks = spdFile->getNumberOfPulses()/spdFile->getPulseBlockSize();
			boost::uint_fast64_t numOfPulsesInBlocks = spdFile->getPulseBlockSize() * multipleOfBlocks;
			boost::uint_fast64_t remainingPulses = spdFile->getNumberOfPulses() - numOfPulsesInBlocks;
			
			DataSet pulsesDataset = spdInFile->openDataSet( SPDFILE_DATASETNAME_PULSES );
			DataSpace pulsesDataspace = pulsesDataset.getSpace();
			
			DataSet pointsDataset = spdInFile->openDataSet( SPDFILE_DATASETNAME_POINTS );
			DataSpace pointsDataspace = pointsDataset.getSpace();
			
			DataSet transmittedDataset = spdInFile->openDataSet( SPDFILE_DATASETNAME_TRANSMITTED );
			DataSpace transmittedDataspace = transmittedDataset.getSpace();
			
			DataSet receivedDataset = spdInFile->openDataSet( SPDFILE_DATASETNAME_RECEIVED );
			DataSpace receivedDataspace = receivedDataset.getSpace();
			
			int rank = 1;
			// START: Variables for Pulse //
			hsize_t pulseOffset[1];
			pulseOffset[0] = 0;
			hsize_t pulseCount[1];
			pulseCount[0]  = spdFile->getPulseBlockSize();
			pulsesDataspace.selectHyperslab( H5S_SELECT_SET, pulseCount, pulseOffset );
			
			hsize_t pulseDims[1]; 
			pulseDims[0] = spdFile->getPulseBlockSize();
			DataSpace pulseMemspace( rank, pulseDims );
			
			hsize_t pulseOffset_out[1];
			hsize_t pulseCount_out[1];
			pulseOffset_out[0] = 0;
			pulseCount_out[0]  = spdFile->getPulseBlockSize();
			pulseMemspace.selectHyperslab( H5S_SELECT_SET, pulseCount_out, pulseOffset_out );
			// END: Variables for Pulse //
			
			// START: Variables for Point //
			hsize_t pointOffset[1];
			hsize_t pointCount[1];
			
			hsize_t pointDims[1]; 
			DataSpace pointMemspace;
			
			hsize_t pointOffset_out[1];
			hsize_t pointCount_out[1];
			// END: Variables for Point //
			
			// START: Variables for Transmitted //
			hsize_t transOffset[1];
			hsize_t transCount[1];
			
			hsize_t transDims[1]; 
			DataSpace transMemspace;
			
			hsize_t transOffset_out[1];
			hsize_t transCount_out[1];
			// END: Variables for Transmitted //
			
			// START: Variables for Received //
			hsize_t receivedOffset[1];
			hsize_t receivedCount[1];
			
			hsize_t receivedDims[1]; 
			DataSpace receivedMemspace;
			
			hsize_t receivedOffset_out[1];
			hsize_t receivedCount_out[1];
			// END: Variables for Received //
            
			
			void *pulseArray = NULL;
            void *pointsArray = NULL;
            CompType *pulseType = NULL;
            CompType *pointType = NULL;
            
            unsigned long *transmittedArray = NULL;
			unsigned long *receivedArray = NULL;
            
            if(spdFile->getPulseVersion() == 1)
            {
                pulseArray = new SPDPulseH5V1[spdFile->getPulseBlockSize()];
                pulseType = pulseUtils.createSPDPulseH5V1DataTypeMemory();
            }
            else if(spdFile->getPulseVersion() == 2)
            {
                pulseArray = new SPDPulseH5V2[spdFile->getPulseBlockSize()];
                pulseType = pulseUtils.createSPDPulseH5V2DataTypeMemory();
            }
            else
            {
                throw SPDIOException("SPD Pulse version was not recognised.");
            }
            
			if(spdFile->getPointVersion() == 1)
            {
                pointType = ptsUtils.createSPDPointV1DataTypeMemory();
            }
            else if(spdFile->getPointVersion() == 2)
            {
                pointType = ptsUtils.createSPDPointV2DataTypeMemory();
            }
            else
            {
                throw SPDIOException("SPD Point version was not recognised.");
            }
			
			boost::uint_fast64_t ptStartIdx = 0;
			boost::uint_fast64_t numOfPoints = 0;
			boost::uint_fast64_t transStartIdx = 0;
			boost::uint_fast64_t numOfTransVals = 0;
			boost::uint_fast64_t receivedStartIdx = 0;
			boost::uint_fast64_t numOfReceivedVals = 0;
			
			boost::uint_fast64_t ptIdx = 0;
			boost::uint_fast64_t transIdx = 0;
			boost::uint_fast64_t receivedIdx = 0;
			
			SPDPulse *pulse = NULL;
			SPDPoint *point = NULL;
            
			boost::uint_fast32_t feedback = multipleOfBlocks/10;
			boost::uint_fast16_t feedbackCounter = 0;
			
			cout << "Started (Read Data) ." << flush;
			for( boost::uint_fast64_t i = 0; i < multipleOfBlocks; ++i)
			{
				if((feedback > 10) && ((i % feedback) == 0))
				{
					cout << "." << feedbackCounter << "." << flush;
					feedbackCounter += 10;
				}
				pulseOffset[0] = i * spdFile->getPulseBlockSize();
				pulsesDataspace.selectHyperslab(H5S_SELECT_SET, pulseCount, pulseOffset);
				pulsesDataset.read(pulseArray, *pulseType, pulseMemspace, pulsesDataspace);
				ptStartIdx = 0;
				numOfPoints = 0;
				transStartIdx = 0;
				numOfTransVals = 0;
				receivedStartIdx = 0;
				numOfReceivedVals = 0;
				
                if(spdFile->getPulseVersion() == 1)
                {
                    SPDPulseH5V1 *pulseObj = NULL;
                    for( boost::uint_fast32_t j = 0; j < spdFile->getPulseBlockSize(); ++j)
                    {
                        pulseObj = &((SPDPulseH5V1 *)pulseArray)[j];
                        if(j == 0)
                        {
                            ptStartIdx = pulseObj->ptsStartIdx;
                            transStartIdx = pulseObj->transmittedStartIdx;
                            receivedStartIdx = pulseObj->receivedStartIdx;
                        }
                        numOfPoints += pulseObj->numberOfReturns;
                        numOfTransVals += pulseObj->numOfTransmittedBins;
                        numOfReceivedVals += pulseObj->numOfReceivedBins;
                    }
                }
                else if(spdFile->getPulseVersion() == 2)
                {
                    SPDPulseH5V2 *pulseObj = NULL;
                    for( boost::uint_fast32_t j = 0; j < spdFile->getPulseBlockSize(); ++j)
                    {
                        pulseObj = &((SPDPulseH5V2 *)pulseArray)[j];
                        if(j == 0)
                        {
                            ptStartIdx = pulseObj->ptsStartIdx;
                            transStartIdx = pulseObj->transmittedStartIdx;
                            receivedStartIdx = pulseObj->receivedStartIdx;
                        }
                        numOfPoints += pulseObj->numberOfReturns;
                        numOfTransVals += pulseObj->numOfTransmittedBins;
                        numOfReceivedVals += pulseObj->numOfReceivedBins;
                    }
                }
                
				if(numOfPoints > 0)
				{
					// Read Points.
                    if(spdFile->getPointVersion() == 1)
                    {
                        pointsArray = new SPDPointH5V1[numOfPoints];
                    }
                    else if(spdFile->getPointVersion() == 2)
                    {
                        pointsArray = new SPDPointH5V2[numOfPoints];
                    }
                    
					pointOffset[0] = ptStartIdx;
					pointCount[0] = numOfPoints;
					pointsDataspace.selectHyperslab(H5S_SELECT_SET, pointCount, pointOffset);
					pointDims[0] = numOfPoints;
					pointMemspace = DataSpace(rank, pointDims);
					pointOffset_out[0] = 0;
					pointCount_out[0] = numOfPoints;
					pointMemspace.selectHyperslab( H5S_SELECT_SET, pointCount_out, pointOffset_out );
					pointsDataset.read(pointsArray, *pointType, pointMemspace, pointsDataspace);
				}
				
				if(numOfTransVals > 0)
				{
					// Read Transmitted Vals.
					transmittedArray = new unsigned long[numOfTransVals];
					transOffset[0] = transStartIdx;
					transCount[0] = numOfTransVals;
					transmittedDataspace.selectHyperslab(H5S_SELECT_SET, transCount, transOffset);
					transDims[0] = numOfTransVals;
					transMemspace = DataSpace(rank, transDims);
					transOffset_out[0] = 0;
					transCount_out[0] = numOfTransVals;
					transMemspace.selectHyperslab( H5S_SELECT_SET, transCount_out, transOffset_out );
					transmittedDataset.read(transmittedArray, PredType::NATIVE_ULONG, transMemspace, transmittedDataspace);
				}
				
				if(numOfReceivedVals > 0)
				{
					// Read Received Vals.
					receivedArray = new unsigned long[numOfReceivedVals];
					receivedOffset[0] = receivedStartIdx;
					receivedCount[0] = numOfReceivedVals;
					receivedDataspace.selectHyperslab(H5S_SELECT_SET, receivedCount, receivedOffset);
					receivedDims[0] = numOfReceivedVals;
					receivedMemspace = DataSpace(rank, receivedDims);
					receivedOffset_out[0] = 0;
					receivedCount_out[0] = numOfReceivedVals;
					receivedMemspace.selectHyperslab( H5S_SELECT_SET, receivedCount_out, receivedOffset_out );
					receivedDataset.read(receivedArray, PredType::NATIVE_ULONG, receivedMemspace, receivedDataspace);
				}
				
				ptIdx = 0;
				transIdx = 0;
				receivedIdx = 0;
				
				for( boost::uint_fast32_t j = 0; j < spdFile->getPulseBlockSize(); ++j)
				{
                    if(spdFile->getPulseVersion() == 1)
                    {
                        pulse = pulseUtils.createSPDPulseCopyFromH5(&((SPDPulseH5V1 *)pulseArray)[j]);
                    }
                    else if(spdFile->getPulseVersion() == 2)
                    {
                        pulse = pulseUtils.createSPDPulseCopyFromH5(&((SPDPulseH5V2 *)pulseArray)[j]);
                    }
                    
					if(pulse->numberOfReturns > 0)
					{
						pulse->pts = new vector<SPDPoint*>();
                        
                        for(boost::uint_fast16_t n = 0; n < pulse->numberOfReturns; ++n)
                        {
                            if(spdFile->getPointVersion() == 1)
                            {
                                point = ptsUtils.createSPDPointCopy(&((SPDPointH5V1 *)pointsArray)[ptIdx++]);
                            }
                            else if(spdFile->getPointVersion() == 2)
                            {
                                point = ptsUtils.createSPDPointCopy(&((SPDPointH5V2 *)pointsArray)[ptIdx++]);
                            }
                            pulse->pts->push_back(point);
                        }
					}
					
					if(pulse->numOfTransmittedBins > 0)
					{
						pulse->transmitted = new boost::uint_fast32_t[pulse->numOfTransmittedBins];
						for(boost::uint_fast16_t n = 0; n < pulse->numOfTransmittedBins; ++n)
						{
							pulse->transmitted[n] = transmittedArray[transIdx++];
						}
					}
					
					if(pulse->numOfReceivedBins > 0)
					{
						pulse->received = new boost::uint_fast32_t[pulse->numOfReceivedBins];
						for(boost::uint_fast16_t n = 0; n < pulse->numOfReceivedBins; ++n)
						{
							pulse->received[n] = receivedArray[receivedIdx++];
						}
					}
					
					if(changeIdxMethod)
					{
						this->defineIdxCoords(spdFile, pulse);
						
						if(first)
						{
							xMin = pulse->xIdx;
							xMax = pulse->xIdx;
							yMin = pulse->yIdx;
							yMax = pulse->yIdx;
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
						}
					}
					
					if(convertCoords)
					{
						this->transformPulseCoords(spdFile, pulse);
						
						if(first)
						{
							xMin = pulse->xIdx;
							xMax = pulse->xIdx;
							yMin = pulse->yIdx;
							yMax = pulse->yIdx;
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
						}
					}
					
					pulses->push_back(pulse);
				}
				
				if(numOfPoints > 0)
				{
                    if(spdFile->getPointVersion() == 1)
                    {
                        delete[] reinterpret_cast<SPDPointH5V1*>(pointsArray);
                    }
                    else if(spdFile->getPointVersion() == 2)
                    {
                        delete[] reinterpret_cast<SPDPointH5V2*>(pointsArray);
                    }
				}
				
				if(numOfTransVals > 0)
				{
					delete[] transmittedArray;
				}
				
				if(numOfReceivedVals > 0)
				{
					delete[] receivedArray;
				}
			}
            
            if(spdFile->getPulseVersion() == 1)
            {
                delete[] reinterpret_cast<SPDPulseH5V1*>(pulseArray);
            }
            else if(spdFile->getPulseVersion() == 2)
            {
                delete[] reinterpret_cast<SPDPulseH5V2*>(pulseArray);
            }
			
			if(remainingPulses > 0)
			{
				pulseOffset[0] = numOfPulsesInBlocks;
				pulseCount[0]  = remainingPulses;
				pulsesDataspace.selectHyperslab( H5S_SELECT_SET, pulseCount, pulseOffset );
				
				pulseDims[0] = remainingPulses;
				pulseMemspace = DataSpace( rank, pulseDims );
				
				pulseOffset_out[0] = 0;
				pulseCount_out[0]  = remainingPulses;
				pulseMemspace.selectHyperslab( H5S_SELECT_SET, pulseCount_out, pulseOffset_out );
                
                if(spdFile->getPulseVersion() == 1)
                {
                    pulseArray = new SPDPulseH5V1[remainingPulses];
                }
                else if(spdFile->getPulseVersion() == 2)
                {
                    pulseArray = new SPDPulseH5V2[remainingPulses];
                }
                else
                {
                    throw SPDIOException("SPD Pulse version was not recognised.");
                }
                
				pulsesDataset.read(pulseArray, *pulseType, pulseMemspace, pulsesDataspace );
				
				
				ptStartIdx = 0;
				numOfPoints = 0;
				transStartIdx = 0;
				numOfTransVals = 0;
				receivedStartIdx = 0;
				numOfReceivedVals = 0;
				
                if(spdFile->getPulseVersion() == 1)
                {
                    SPDPulseH5V1 *pulseObj = NULL;
                    for( boost::uint_fast32_t j = 0; j < remainingPulses; ++j)
                    {
                        pulseObj = &((SPDPulseH5V1 *)pulseArray)[j];
                        if(j == 0)
                        {
                            ptStartIdx = pulseObj->ptsStartIdx;
                            transStartIdx = pulseObj->transmittedStartIdx;
                            receivedStartIdx = pulseObj->receivedStartIdx;
                        }
                        numOfPoints += pulseObj->numberOfReturns;
                        numOfTransVals += pulseObj->numOfTransmittedBins;
                        numOfReceivedVals += pulseObj->numOfReceivedBins;
                    }
                }
                else if(spdFile->getPulseVersion() == 2)
                {
                    SPDPulseH5V2 *pulseObj = NULL;
                    for( boost::uint_fast32_t j = 0; j < remainingPulses; ++j)
                    {
                        pulseObj = &((SPDPulseH5V2 *)pulseArray)[j];
                        if(j == 0)
                        {
                            ptStartIdx = pulseObj->ptsStartIdx;
                            transStartIdx = pulseObj->transmittedStartIdx;
                            receivedStartIdx = pulseObj->receivedStartIdx;
                        }
                        numOfPoints += pulseObj->numberOfReturns;
                        numOfTransVals += pulseObj->numOfTransmittedBins;
                        numOfReceivedVals += pulseObj->numOfReceivedBins;
                    }
                }
                
				
				if(numOfPoints > 0)
				{
					// Read Points.
					if(spdFile->getPointVersion() == 1)
                    {
                        pointsArray = new SPDPointH5V1[numOfPoints];
                    }
                    else if(spdFile->getPointVersion() == 2)
                    {
                        pointsArray = new SPDPointH5V2[numOfPoints];
                    }
                    
					pointOffset[0] = ptStartIdx;
					pointCount[0] = numOfPoints;
					pointsDataspace.selectHyperslab(H5S_SELECT_SET, pointCount, pointOffset);
					pointDims[0] = numOfPoints;
					pointMemspace = DataSpace(rank, pointDims);
					pointOffset_out[0] = 0;
					pointCount_out[0] = numOfPoints;
					pointMemspace.selectHyperslab( H5S_SELECT_SET, pointCount_out, pointOffset_out );
					pointsDataset.read(pointsArray, *pointType, pointMemspace, pointsDataspace);
				}
				
				if(numOfTransVals > 0)
				{
					// Read Transmitted Vals.
					transmittedArray = new unsigned long[numOfTransVals];
					transOffset[0] = transStartIdx;
					transCount[0] = numOfTransVals;
					transmittedDataspace.selectHyperslab(H5S_SELECT_SET, transCount, transOffset);
					transDims[0] = numOfTransVals;
					transMemspace = DataSpace(rank, transDims);
					transOffset_out[0] = 0;
					transCount_out[0] = numOfTransVals;
					transMemspace.selectHyperslab( H5S_SELECT_SET, transCount_out, transOffset_out );
					transmittedDataset.read(transmittedArray, PredType::NATIVE_ULONG, transMemspace, transmittedDataspace);
				}
				
				if(numOfReceivedVals > 0)
				{
					// Read Received Vals.
					receivedArray = new unsigned long[numOfReceivedVals];
					receivedOffset[0] = receivedStartIdx;
					receivedCount[0] = numOfReceivedVals;
					receivedDataspace.selectHyperslab(H5S_SELECT_SET, receivedCount, receivedOffset);
					receivedDims[0] = numOfReceivedVals;
					receivedMemspace = DataSpace(rank, receivedDims);
					receivedOffset_out[0] = 0;
					receivedCount_out[0] = numOfReceivedVals;
					receivedMemspace.selectHyperslab( H5S_SELECT_SET, receivedCount_out, receivedOffset_out );
					receivedDataset.read(receivedArray, PredType::NATIVE_ULONG, receivedMemspace, receivedDataspace);
				}
				
				ptIdx = 0;
				transIdx = 0;
				receivedIdx = 0;
				
				for( boost::uint_fast32_t j = 0; j < remainingPulses; ++j)
				{
					if(spdFile->getPulseVersion() == 1)
                    {
                        pulse = pulseUtils.createSPDPulseCopyFromH5(&((SPDPulseH5V1 *)pulseArray)[j]);
                    }
                    else if(spdFile->getPulseVersion() == 2)
                    {
                        pulse = pulseUtils.createSPDPulseCopyFromH5(&((SPDPulseH5V2 *)pulseArray)[j]);
                    }
					
					if(pulse->numberOfReturns > 0)
					{
						pulse->pts = new vector<SPDPoint*>();
                        
                        for(boost::uint_fast16_t n = 0; n < pulse->numberOfReturns; ++n)
                        {
                            if(spdFile->getPointVersion() == 1)
                            {
                                point = ptsUtils.createSPDPointCopy(&((SPDPointH5V1 *)pointsArray)[ptIdx++]);
                            }
                            else if(spdFile->getPointVersion() == 2)
                            {
                                point = ptsUtils.createSPDPointCopy(&((SPDPointH5V2 *)pointsArray)[ptIdx++]);
                            }
                            pulse->pts->push_back(point);
                        }
					}
					
					if(pulse->numOfTransmittedBins > 0)
					{
						pulse->transmitted = new boost::uint_fast32_t[pulse->numOfTransmittedBins];
						for(boost::uint_fast16_t n = 0; n < pulse->numOfTransmittedBins; ++n)
						{
							pulse->transmitted[n] = transmittedArray[transIdx++];
						}
					}
					
					if(pulse->numOfReceivedBins > 0)
					{
						pulse->received = new boost::uint_fast32_t[pulse->numOfReceivedBins];
						for(boost::uint_fast16_t n = 0; n < pulse->numOfReceivedBins; ++n)
						{
							pulse->received[n] = receivedArray[receivedIdx++];
						}
					}
					
					if(changeIdxMethod)
					{
						this->defineIdxCoords(spdFile, pulse);
						
						if(first)
						{
							xMin = pulse->xIdx;
							xMax = pulse->xIdx;
							yMin = pulse->yIdx;
							yMax = pulse->yIdx;
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
						}
					}
					
					if(convertCoords)
					{
						this->transformPulseCoords(spdFile, pulse);
						
						if(first)
						{
							xMin = pulse->xIdx;
							xMax = pulse->xIdx;
							yMin = pulse->yIdx;
							yMax = pulse->yIdx;
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
						}
					}
					pulses->push_back(pulse);
				}
				
				if(numOfPoints > 0)
				{
                    if(spdFile->getPointVersion() == 1)
                    {
                        delete[] reinterpret_cast<SPDPointH5V1*>(pointsArray);
                    }
                    else if(spdFile->getPointVersion() == 2)
                    {
                        delete[] reinterpret_cast<SPDPointH5V2*>(pointsArray);
                    }
				}
				
				if(numOfTransVals > 0)
				{
					delete[] transmittedArray;
				}
				
				if(numOfReceivedVals > 0)
				{
					delete[] receivedArray;
				}
                
                if(spdFile->getPulseVersion() == 1)
                {
                    delete[] reinterpret_cast<SPDPulseH5V1*>(pulseArray);
                }
                else if(spdFile->getPulseVersion() == 2)
                {
                    delete[] reinterpret_cast<SPDPulseH5V2*>(pulseArray);
                }
            }
			delete pulseType;
			delete pointType;
            
			cout << ".Complete\n";
			
			if(convertCoords | changeIdxMethod)
			{
				spdFile->setBoundingBox(xMin, xMax, yMin, yMax);
			}
			
			spdInFile->close();
		}
		catch( FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataTypeIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch(SPDIOException &e)
		{
			throw e;
		}		
		
		return pulses;
	}
	
	vector<SPDPulse*>* SPDFileReader::readAllDataToVector(string inputFile, SPDFile *spdFile)throw(SPDIOException)
	{
		SPDPointUtils ptsUtils;
		SPDPulseUtils pulseUtils;
		vector<SPDPulse*> *pulses = new vector<SPDPulse*>();
		try 
		{
			Exception::dontPrint();
			H5File *spdInFile = NULL;
            
			this->readHeaderInfo(inputFile, spdFile);
			
			if(convertCoords)
			{
				this->initCoordinateSystemTransformation(spdFile);
				
				spdFile->setSpatialReference(outputProjWKT);
			}
			bool changeIdxMethod = false;
			if((spdFile->getPulseIdxMethod() == indexCoords) | (indexCoords == SPD_IDX_UNCHANGED))
			{
				changeIdxMethod = false;
			}
			else 
			{
				spdFile->setPulseIdxMethod(indexCoords);
				changeIdxMethod = true;
			}
			
			try 
			{
				spdInFile = new H5File( spdFile->getFilePath(), H5F_ACC_RDONLY );
			}
			catch (FileIException &e) 
			{
				string message  = string("Could not open SPD file: ") + spdFile->getFilePath();
				throw SPDIOException(message);
			}
			
			double xMin = 0;
			double yMin = 0;
			double xMax = 0;
			double yMax = 0;
			bool first = true;
            
			boost::uint_fast64_t multipleOfBlocks = spdFile->getNumberOfPulses()/spdFile->getPulseBlockSize();
			boost::uint_fast64_t numOfPulsesInBlocks = spdFile->getPulseBlockSize() * multipleOfBlocks;
			boost::uint_fast64_t remainingPulses = spdFile->getNumberOfPulses() - numOfPulsesInBlocks;
			
			DataSet pulsesDataset = spdInFile->openDataSet( SPDFILE_DATASETNAME_PULSES );
			DataSpace pulsesDataspace = pulsesDataset.getSpace();
			
			DataSet pointsDataset = spdInFile->openDataSet( SPDFILE_DATASETNAME_POINTS );
			DataSpace pointsDataspace = pointsDataset.getSpace();
			
			DataSet transmittedDataset = spdInFile->openDataSet( SPDFILE_DATASETNAME_TRANSMITTED );
			DataSpace transmittedDataspace = transmittedDataset.getSpace();
			
			DataSet receivedDataset = spdInFile->openDataSet( SPDFILE_DATASETNAME_RECEIVED );
			DataSpace receivedDataspace = receivedDataset.getSpace();
			
			int rank = 1;
			// START: Variables for Pulse //
			hsize_t pulseOffset[1];
			pulseOffset[0] = 0;
			hsize_t pulseCount[1];
			pulseCount[0]  = spdFile->getPulseBlockSize();
			pulsesDataspace.selectHyperslab( H5S_SELECT_SET, pulseCount, pulseOffset );
			
			hsize_t pulseDims[1]; 
			pulseDims[0] = spdFile->getPulseBlockSize();
			DataSpace pulseMemspace( rank, pulseDims );
			
			hsize_t pulseOffset_out[1];
			hsize_t pulseCount_out[1];
			pulseOffset_out[0] = 0;
			pulseCount_out[0]  = spdFile->getPulseBlockSize();
			pulseMemspace.selectHyperslab( H5S_SELECT_SET, pulseCount_out, pulseOffset_out );
			// END: Variables for Pulse //
			
			// START: Variables for Point //
			hsize_t pointOffset[1];
			hsize_t pointCount[1];
			
			hsize_t pointDims[1]; 
			DataSpace pointMemspace;
			
			hsize_t pointOffset_out[1];
			hsize_t pointCount_out[1];
			// END: Variables for Point //
			
			// START: Variables for Transmitted //
			hsize_t transOffset[1];
			hsize_t transCount[1];
			
			hsize_t transDims[1]; 
			DataSpace transMemspace;
			
			hsize_t transOffset_out[1];
			hsize_t transCount_out[1];
			// END: Variables for Transmitted //
			
			// START: Variables for Received //
			hsize_t receivedOffset[1];
			hsize_t receivedCount[1];
			
			hsize_t receivedDims[1]; 
			DataSpace receivedMemspace;
			
			hsize_t receivedOffset_out[1];
			hsize_t receivedCount_out[1];
			// END: Variables for Received //
            
			
			void *pulseArray = NULL;
            void *pointsArray = NULL;
            CompType *pulseType = NULL;
            CompType *pointType = NULL;
            
            unsigned long *transmittedArray = NULL;
			unsigned long *receivedArray = NULL;
            
            if(spdFile->getPulseVersion() == 1)
            {
                pulseArray = new SPDPulseH5V1[spdFile->getPulseBlockSize()];
                pulseType = pulseUtils.createSPDPulseH5V1DataTypeMemory();
            }
            else if(spdFile->getPulseVersion() == 2)
            {
                pulseArray = new SPDPulseH5V2[spdFile->getPulseBlockSize()];
                pulseType = pulseUtils.createSPDPulseH5V2DataTypeMemory();
            }
            else
            {
                throw SPDIOException("SPD Pulse version was not recognised.");
            }
            
			if(spdFile->getPointVersion() == 1)
            {
                pointType = ptsUtils.createSPDPointV1DataTypeMemory();
            }
            else if(spdFile->getPointVersion() == 2)
            {
                pointType = ptsUtils.createSPDPointV2DataTypeMemory();
            }
            else
            {
                throw SPDIOException("SPD Point version was not recognised.");
            }
			
			boost::uint_fast64_t ptStartIdx = 0;
			boost::uint_fast64_t numOfPoints = 0;
			boost::uint_fast64_t transStartIdx = 0;
			boost::uint_fast64_t numOfTransVals = 0;
			boost::uint_fast64_t receivedStartIdx = 0;
			boost::uint_fast64_t numOfReceivedVals = 0;
			
			boost::uint_fast64_t ptIdx = 0;
			boost::uint_fast64_t transIdx = 0;
			boost::uint_fast64_t receivedIdx = 0;
			
			SPDPulse *pulse = NULL;
			SPDPoint *point = NULL;
            
			boost::uint_fast32_t feedback = multipleOfBlocks/10;
			boost::uint_fast16_t feedbackCounter = 0;
			
			cout << "Started (Read Data) ." << flush;
			for( boost::uint_fast64_t i = 0; i < multipleOfBlocks; ++i)
			{
				if((feedback > 10) && ((i % feedback) == 0))
				{
					cout << "." << feedbackCounter << "." << flush;
					feedbackCounter += 10;
				}
				pulseOffset[0] = i * spdFile->getPulseBlockSize();
				pulsesDataspace.selectHyperslab(H5S_SELECT_SET, pulseCount, pulseOffset);
				pulsesDataset.read(pulseArray, *pulseType, pulseMemspace, pulsesDataspace);
				ptStartIdx = 0;
				numOfPoints = 0;
				transStartIdx = 0;
				numOfTransVals = 0;
				receivedStartIdx = 0;
				numOfReceivedVals = 0;
				
                if(spdFile->getPulseVersion() == 1)
                {
                    SPDPulseH5V1 *pulseObj = NULL;
                    for( boost::uint_fast32_t j = 0; j < spdFile->getPulseBlockSize(); ++j)
                    {
                        pulseObj = &((SPDPulseH5V1 *)pulseArray)[j];
                        if(j == 0)
                        {
                            ptStartIdx = pulseObj->ptsStartIdx;
                            transStartIdx = pulseObj->transmittedStartIdx;
                            receivedStartIdx = pulseObj->receivedStartIdx;
                        }
                        numOfPoints += pulseObj->numberOfReturns;
                        numOfTransVals += pulseObj->numOfTransmittedBins;
                        numOfReceivedVals += pulseObj->numOfReceivedBins;
                    }
                }
                else if(spdFile->getPulseVersion() == 2)
                {
                    SPDPulseH5V2 *pulseObj = NULL;
                    for( boost::uint_fast32_t j = 0; j < spdFile->getPulseBlockSize(); ++j)
                    {
                        pulseObj = &((SPDPulseH5V2 *)pulseArray)[j];
                        if(j == 0)
                        {
                            ptStartIdx = pulseObj->ptsStartIdx;
                            transStartIdx = pulseObj->transmittedStartIdx;
                            receivedStartIdx = pulseObj->receivedStartIdx;
                        }
                        numOfPoints += pulseObj->numberOfReturns;
                        numOfTransVals += pulseObj->numOfTransmittedBins;
                        numOfReceivedVals += pulseObj->numOfReceivedBins;
                    }
                }
                
				if(numOfPoints > 0)
				{
					// Read Points.
                    if(spdFile->getPointVersion() == 1)
                    {
                        pointsArray = new SPDPointH5V1[numOfPoints];
                    }
                    else if(spdFile->getPointVersion() == 2)
                    {
                        pointsArray = new SPDPointH5V2[numOfPoints];
                    }
                    
					pointOffset[0] = ptStartIdx;
					pointCount[0] = numOfPoints;
					pointsDataspace.selectHyperslab(H5S_SELECT_SET, pointCount, pointOffset);
					pointDims[0] = numOfPoints;
					pointMemspace = DataSpace(rank, pointDims);
					pointOffset_out[0] = 0;
					pointCount_out[0] = numOfPoints;
					pointMemspace.selectHyperslab( H5S_SELECT_SET, pointCount_out, pointOffset_out );
					pointsDataset.read(pointsArray, *pointType, pointMemspace, pointsDataspace);
				}
				
				if(numOfTransVals > 0)
				{
					// Read Transmitted Vals.
					transmittedArray = new unsigned long[numOfTransVals];
					transOffset[0] = transStartIdx;
					transCount[0] = numOfTransVals;
					transmittedDataspace.selectHyperslab(H5S_SELECT_SET, transCount, transOffset);
					transDims[0] = numOfTransVals;
					transMemspace = DataSpace(rank, transDims);
					transOffset_out[0] = 0;
					transCount_out[0] = numOfTransVals;
					transMemspace.selectHyperslab( H5S_SELECT_SET, transCount_out, transOffset_out );
					transmittedDataset.read(transmittedArray, PredType::NATIVE_ULONG, transMemspace, transmittedDataspace);
				}
				
				if(numOfReceivedVals > 0)
				{
					// Read Received Vals.
					receivedArray = new unsigned long[numOfReceivedVals];
					receivedOffset[0] = receivedStartIdx;
					receivedCount[0] = numOfReceivedVals;
					receivedDataspace.selectHyperslab(H5S_SELECT_SET, receivedCount, receivedOffset);
					receivedDims[0] = numOfReceivedVals;
					receivedMemspace = DataSpace(rank, receivedDims);
					receivedOffset_out[0] = 0;
					receivedCount_out[0] = numOfReceivedVals;
					receivedMemspace.selectHyperslab( H5S_SELECT_SET, receivedCount_out, receivedOffset_out );
					receivedDataset.read(receivedArray, PredType::NATIVE_ULONG, receivedMemspace, receivedDataspace);
				}
				
				ptIdx = 0;
				transIdx = 0;
				receivedIdx = 0;
				
				for( boost::uint_fast32_t j = 0; j < spdFile->getPulseBlockSize(); ++j)
				{
                    if(spdFile->getPulseVersion() == 1)
                    {
                        pulse = pulseUtils.createSPDPulseCopyFromH5(&((SPDPulseH5V1 *)pulseArray)[j]);
                    }
                    else if(spdFile->getPulseVersion() == 2)
                    {
                        pulse = pulseUtils.createSPDPulseCopyFromH5(&((SPDPulseH5V2 *)pulseArray)[j]);
                    }
					                    
					if(pulse->numberOfReturns > 0)
					{
						pulse->pts = new vector<SPDPoint*>();
                        
                        for(boost::uint_fast16_t n = 0; n < pulse->numberOfReturns; ++n)
                        {
                            if(spdFile->getPointVersion() == 1)
                            {
                                point = ptsUtils.createSPDPointCopy(&((SPDPointH5V1 *)pointsArray)[ptIdx++]);
                            }
                            else if(spdFile->getPointVersion() == 2)
                            {
                                point = ptsUtils.createSPDPointCopy(&((SPDPointH5V2 *)pointsArray)[ptIdx++]);
                            }
                            pulse->pts->push_back(point);
                        }
					}
					
					if(pulse->numOfTransmittedBins > 0)
					{
						pulse->transmitted = new boost::uint_fast32_t[pulse->numOfTransmittedBins];
						for(boost::uint_fast16_t n = 0; n < pulse->numOfTransmittedBins; ++n)
						{
							pulse->transmitted[n] = transmittedArray[transIdx++];
						}
					}
					
					if(pulse->numOfReceivedBins > 0)
					{
						pulse->received = new boost::uint_fast32_t[pulse->numOfReceivedBins];
						for(boost::uint_fast16_t n = 0; n < pulse->numOfReceivedBins; ++n)
						{
							pulse->received[n] = receivedArray[receivedIdx++];
						}
					}
					
					if(changeIdxMethod)
					{
						this->defineIdxCoords(spdFile, pulse);
						
						if(first)
						{
							xMin = pulse->xIdx;
							xMax = pulse->xIdx;
							yMin = pulse->yIdx;
							yMax = pulse->yIdx;
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
						}
					}
					
					if(convertCoords)
					{
						this->transformPulseCoords(spdFile, pulse);
						
						if(first)
						{
							xMin = pulse->xIdx;
							xMax = pulse->xIdx;
							yMin = pulse->yIdx;
							yMax = pulse->yIdx;
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
						}
					}
					
					pulses->push_back(pulse);
				}
				
				if(numOfPoints > 0)
				{
                    if(spdFile->getPointVersion() == 1)
                    {
                        delete[] reinterpret_cast<SPDPointH5V1*>(pointsArray);
                    }
                    else if(spdFile->getPointVersion() == 2)
                    {
                        delete[] reinterpret_cast<SPDPointH5V2*>(pointsArray);
                    }
				}
				
				if(numOfTransVals > 0)
				{
					delete[] transmittedArray;
				}
				
				if(numOfReceivedVals > 0)
				{
					delete[] receivedArray;
				}
			}
            
            if(spdFile->getPulseVersion() == 1)
            {
                delete[] reinterpret_cast<SPDPulseH5V1*>(pulseArray);
            }
            else if(spdFile->getPulseVersion() == 2)
            {
                delete[] reinterpret_cast<SPDPulseH5V2*>(pulseArray);
            }
			
			if(remainingPulses > 0)
			{
				pulseOffset[0] = numOfPulsesInBlocks;
				pulseCount[0]  = remainingPulses;
				pulsesDataspace.selectHyperslab( H5S_SELECT_SET, pulseCount, pulseOffset );
				
				pulseDims[0] = remainingPulses;
				pulseMemspace = DataSpace( rank, pulseDims );
				
				pulseOffset_out[0] = 0;
				pulseCount_out[0]  = remainingPulses;
				pulseMemspace.selectHyperslab( H5S_SELECT_SET, pulseCount_out, pulseOffset_out );
                
                if(spdFile->getPulseVersion() == 1)
                {
                    pulseArray = new SPDPulseH5V1[remainingPulses];
                }
                else if(spdFile->getPulseVersion() == 2)
                {
                    pulseArray = new SPDPulseH5V2[remainingPulses];
                }
                else
                {
                    throw SPDIOException("SPD Pulse version was not recognised.");
                }
                
				pulsesDataset.read(pulseArray, *pulseType, pulseMemspace, pulsesDataspace );
				
				
				ptStartIdx = 0;
				numOfPoints = 0;
				transStartIdx = 0;
				numOfTransVals = 0;
				receivedStartIdx = 0;
				numOfReceivedVals = 0;
				
                if(spdFile->getPulseVersion() == 1)
                {
                    SPDPulseH5V1 *pulseObj = NULL;
                    for( boost::uint_fast32_t j = 0; j < remainingPulses; ++j)
                    {
                        pulseObj = &((SPDPulseH5V1 *)pulseArray)[j];
                        if(j == 0)
                        {
                            ptStartIdx = pulseObj->ptsStartIdx;
                            transStartIdx = pulseObj->transmittedStartIdx;
                            receivedStartIdx = pulseObj->receivedStartIdx;
                        }
                        numOfPoints += pulseObj->numberOfReturns;
                        numOfTransVals += pulseObj->numOfTransmittedBins;
                        numOfReceivedVals += pulseObj->numOfReceivedBins;
                    }
                }
                else if(spdFile->getPulseVersion() == 2)
                {
                    SPDPulseH5V2 *pulseObj = NULL;
                    for( boost::uint_fast32_t j = 0; j < remainingPulses; ++j)
                    {
                        pulseObj = &((SPDPulseH5V2 *)pulseArray)[j];
                        if(j == 0)
                        {
                            ptStartIdx = pulseObj->ptsStartIdx;
                            transStartIdx = pulseObj->transmittedStartIdx;
                            receivedStartIdx = pulseObj->receivedStartIdx;
                        }
                        numOfPoints += pulseObj->numberOfReturns;
                        numOfTransVals += pulseObj->numOfTransmittedBins;
                        numOfReceivedVals += pulseObj->numOfReceivedBins;
                    }
                }
                
				
				if(numOfPoints > 0)
				{
					// Read Points.
					if(spdFile->getPointVersion() == 1)
                    {
                        pointsArray = new SPDPointH5V1[numOfPoints];
                    }
                    else if(spdFile->getPointVersion() == 2)
                    {
                        pointsArray = new SPDPointH5V2[numOfPoints];
                    }
                    
					pointOffset[0] = ptStartIdx;
					pointCount[0] = numOfPoints;
					pointsDataspace.selectHyperslab(H5S_SELECT_SET, pointCount, pointOffset);
					pointDims[0] = numOfPoints;
					pointMemspace = DataSpace(rank, pointDims);
					pointOffset_out[0] = 0;
					pointCount_out[0] = numOfPoints;
					pointMemspace.selectHyperslab( H5S_SELECT_SET, pointCount_out, pointOffset_out );
					pointsDataset.read(pointsArray, *pointType, pointMemspace, pointsDataspace);
				}
				
				if(numOfTransVals > 0)
				{
					// Read Transmitted Vals.
					transmittedArray = new unsigned long[numOfTransVals];
					transOffset[0] = transStartIdx;
					transCount[0] = numOfTransVals;
					transmittedDataspace.selectHyperslab(H5S_SELECT_SET, transCount, transOffset);
					transDims[0] = numOfTransVals;
					transMemspace = DataSpace(rank, transDims);
					transOffset_out[0] = 0;
					transCount_out[0] = numOfTransVals;
					transMemspace.selectHyperslab( H5S_SELECT_SET, transCount_out, transOffset_out );
					transmittedDataset.read(transmittedArray, PredType::NATIVE_ULONG, transMemspace, transmittedDataspace);
				}
				
				if(numOfReceivedVals > 0)
				{
					// Read Received Vals.
					receivedArray = new unsigned long[numOfReceivedVals];
					receivedOffset[0] = receivedStartIdx;
					receivedCount[0] = numOfReceivedVals;
					receivedDataspace.selectHyperslab(H5S_SELECT_SET, receivedCount, receivedOffset);
					receivedDims[0] = numOfReceivedVals;
					receivedMemspace = DataSpace(rank, receivedDims);
					receivedOffset_out[0] = 0;
					receivedCount_out[0] = numOfReceivedVals;
					receivedMemspace.selectHyperslab( H5S_SELECT_SET, receivedCount_out, receivedOffset_out );
					receivedDataset.read(receivedArray, PredType::NATIVE_ULONG, receivedMemspace, receivedDataspace);
				}
				
				ptIdx = 0;
				transIdx = 0;
				receivedIdx = 0;
				
				for( boost::uint_fast32_t j = 0; j < remainingPulses; ++j)
				{
					if(spdFile->getPulseVersion() == 1)
                    {
                        pulse = pulseUtils.createSPDPulseCopyFromH5(&((SPDPulseH5V1 *)pulseArray)[j]);
                    }
                    else if(spdFile->getPulseVersion() == 2)
                    {
                        pulse = pulseUtils.createSPDPulseCopyFromH5(&((SPDPulseH5V2 *)pulseArray)[j]);
                    }
					
					if(pulse->numberOfReturns > 0)
					{
						pulse->pts = new vector<SPDPoint*>();
                        
                        for(boost::uint_fast16_t n = 0; n < pulse->numberOfReturns; ++n)
                        {
                            if(spdFile->getPointVersion() == 1)
                            {
                                point = ptsUtils.createSPDPointCopy(&((SPDPointH5V1 *)pointsArray)[ptIdx++]);
                            }
                            else if(spdFile->getPointVersion() == 2)
                            {
                                point = ptsUtils.createSPDPointCopy(&((SPDPointH5V2 *)pointsArray)[ptIdx++]);
                            }
                            pulse->pts->push_back(point);
                        }
					}
					
					if(pulse->numOfTransmittedBins > 0)
					{
						pulse->transmitted = new boost::uint_fast32_t[pulse->numOfTransmittedBins];
						for(boost::uint_fast16_t n = 0; n < pulse->numOfTransmittedBins; ++n)
						{
							pulse->transmitted[n] = transmittedArray[transIdx++];
						}
					}
					
					if(pulse->numOfReceivedBins > 0)
					{
						pulse->received = new boost::uint_fast32_t[pulse->numOfReceivedBins];
						for(boost::uint_fast16_t n = 0; n < pulse->numOfReceivedBins; ++n)
						{
							pulse->received[n] = receivedArray[receivedIdx++];
						}
					}
					
					if(changeIdxMethod)
					{
						this->defineIdxCoords(spdFile, pulse);
						
						if(first)
						{
							xMin = pulse->xIdx;
							xMax = pulse->xIdx;
							yMin = pulse->yIdx;
							yMax = pulse->yIdx;
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
						}
					}
					
					if(convertCoords)
					{
						this->transformPulseCoords(spdFile, pulse);
						
						if(first)
						{
							xMin = pulse->xIdx;
							xMax = pulse->xIdx;
							yMin = pulse->yIdx;
							yMax = pulse->yIdx;
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
						}
					}
					pulses->push_back(pulse);
				}
				
				if(numOfPoints > 0)
				{
                    if(spdFile->getPointVersion() == 1)
                    {
                        delete[] reinterpret_cast<SPDPointH5V1*>(pointsArray);
                    }
                    else if(spdFile->getPointVersion() == 2)
                    {
                        delete[] reinterpret_cast<SPDPointH5V2*>(pointsArray);
                    }
				}
				
				if(numOfTransVals > 0)
				{
					delete[] transmittedArray;
				}
				
				if(numOfReceivedVals > 0)
				{
					delete[] receivedArray;
				}
                
                if(spdFile->getPulseVersion() == 1)
                {
                    delete[] reinterpret_cast<SPDPulseH5V1*>(pulseArray);
                }
                else if(spdFile->getPulseVersion() == 2)
                {
                    delete[] reinterpret_cast<SPDPulseH5V2*>(pulseArray);
                }
            }
			delete pulseType;
			delete pointType;
            
			cout << ".Complete\n";
			
			if(convertCoords | changeIdxMethod)
			{
				spdFile->setBoundingBox(xMin, xMax, yMin, yMax);
			}
			
			spdInFile->close();
		}
		catch( FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataTypeIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch(SPDIOException &e)
		{
			throw e;
		}		
		
		return pulses;
	}
	
	void SPDFileReader::readAndProcessAllData(string inputFile, SPDFile *spdFile, SPDImporterProcessor *processor) throw(SPDIOException)
	{
		SPDPointUtils ptsUtils;
		SPDPulseUtils pulseUtils;
		try 
		{
			Exception::dontPrint();
			H5File *spdInFile = NULL;
            
			this->readHeaderInfo(inputFile, spdFile);
			
			if(convertCoords)
			{
				this->initCoordinateSystemTransformation(spdFile);
				
				spdFile->setSpatialReference(outputProjWKT);
			}
			bool changeIdxMethod = false;
			if((spdFile->getPulseIdxMethod() == indexCoords) | (indexCoords == SPD_IDX_UNCHANGED))
			{
				changeIdxMethod = false;
			}
			else 
			{
				spdFile->setPulseIdxMethod(indexCoords);
				changeIdxMethod = true;
			}
			
			try 
			{
				spdInFile = new H5File( spdFile->getFilePath(), H5F_ACC_RDONLY );
			}
			catch (FileIException &e) 
			{
				string message  = string("Could not open SPD file: ") + spdFile->getFilePath();
				throw SPDIOException(message);
			}
			
			double xMin = 0;
			double yMin = 0;
			double xMax = 0;
			double yMax = 0;
			bool first = true;
            
			boost::uint_fast64_t multipleOfBlocks = spdFile->getNumberOfPulses()/spdFile->getPulseBlockSize();
			boost::uint_fast64_t numOfPulsesInBlocks = spdFile->getPulseBlockSize() * multipleOfBlocks;
			boost::uint_fast64_t remainingPulses = spdFile->getNumberOfPulses() - numOfPulsesInBlocks;
			
			DataSet pulsesDataset = spdInFile->openDataSet( SPDFILE_DATASETNAME_PULSES );
			DataSpace pulsesDataspace = pulsesDataset.getSpace();
			
			DataSet pointsDataset = spdInFile->openDataSet( SPDFILE_DATASETNAME_POINTS );
			DataSpace pointsDataspace = pointsDataset.getSpace();
			
			DataSet transmittedDataset = spdInFile->openDataSet( SPDFILE_DATASETNAME_TRANSMITTED );
			DataSpace transmittedDataspace = transmittedDataset.getSpace();
			
			DataSet receivedDataset = spdInFile->openDataSet( SPDFILE_DATASETNAME_RECEIVED );
			DataSpace receivedDataspace = receivedDataset.getSpace();
			
			int rank = 1;
			// START: Variables for Pulse //
			hsize_t pulseOffset[1];
			pulseOffset[0] = 0;
			hsize_t pulseCount[1];
			pulseCount[0]  = spdFile->getPulseBlockSize();
			pulsesDataspace.selectHyperslab( H5S_SELECT_SET, pulseCount, pulseOffset );
			
			hsize_t pulseDims[1]; 
			pulseDims[0] = spdFile->getPulseBlockSize();
			DataSpace pulseMemspace( rank, pulseDims );
			
			hsize_t pulseOffset_out[1];
			hsize_t pulseCount_out[1];
			pulseOffset_out[0] = 0;
			pulseCount_out[0]  = spdFile->getPulseBlockSize();
			pulseMemspace.selectHyperslab( H5S_SELECT_SET, pulseCount_out, pulseOffset_out );
			// END: Variables for Pulse //
			
			// START: Variables for Point //
			hsize_t pointOffset[1];
			hsize_t pointCount[1];
			
			hsize_t pointDims[1]; 
			DataSpace pointMemspace;
			
			hsize_t pointOffset_out[1];
			hsize_t pointCount_out[1];
			// END: Variables for Point //
			
			// START: Variables for Transmitted //
			hsize_t transOffset[1];
			hsize_t transCount[1];
			
			hsize_t transDims[1]; 
			DataSpace transMemspace;
			
			hsize_t transOffset_out[1];
			hsize_t transCount_out[1];
			// END: Variables for Transmitted //
			
			// START: Variables for Received //
			hsize_t receivedOffset[1];
			hsize_t receivedCount[1];
			
			hsize_t receivedDims[1]; 
			DataSpace receivedMemspace;
			
			hsize_t receivedOffset_out[1];
			hsize_t receivedCount_out[1];
			// END: Variables for Received //
            
			
			void *pulseArray = NULL;
            void *pointsArray = NULL;
            CompType *pulseType = NULL;
            CompType *pointType = NULL;
            
            unsigned long *transmittedArray = NULL;
			unsigned long *receivedArray = NULL;
            
            if(spdFile->getPulseVersion() == 1)
            {
                pulseArray = new SPDPulseH5V1[spdFile->getPulseBlockSize()];
                pulseType = pulseUtils.createSPDPulseH5V1DataTypeMemory();
            }
            else if(spdFile->getPulseVersion() == 2)
            {
                pulseArray = new SPDPulseH5V2[spdFile->getPulseBlockSize()];
                pulseType = pulseUtils.createSPDPulseH5V2DataTypeMemory();
            }
            else
            {
                throw SPDIOException("SPD Pulse version was not recognised.");
            }
            
			if(spdFile->getPointVersion() == 1)
            {
                pointType = ptsUtils.createSPDPointV1DataTypeMemory();
            }
            else if(spdFile->getPointVersion() == 2)
            {
                pointType = ptsUtils.createSPDPointV2DataTypeMemory();
            }
            else
            {
                throw SPDIOException("SPD Point version was not recognised.");
            }
			
			boost::uint_fast64_t ptStartIdx = 0;
			boost::uint_fast64_t numOfPoints = 0;
			boost::uint_fast64_t transStartIdx = 0;
			boost::uint_fast64_t numOfTransVals = 0;
			boost::uint_fast64_t receivedStartIdx = 0;
			boost::uint_fast64_t numOfReceivedVals = 0;
			
			boost::uint_fast64_t ptIdx = 0;
			boost::uint_fast64_t transIdx = 0;
			boost::uint_fast64_t receivedIdx = 0;
			
			SPDPulse *pulse = NULL;
			SPDPoint *point = NULL;
            
			boost::uint_fast32_t feedback = multipleOfBlocks/10;
			boost::uint_fast16_t feedbackCounter = 0;
			
			cout << "Started (Read Data) ." << flush;
			for( boost::uint_fast64_t i = 0; i < multipleOfBlocks; ++i)
			{
				if((feedback > 10) && ((i % feedback) == 0))
				{
					cout << "." << feedbackCounter << "." << flush;
					feedbackCounter += 10;
				}
				pulseOffset[0] = i * spdFile->getPulseBlockSize();
				pulsesDataspace.selectHyperslab(H5S_SELECT_SET, pulseCount, pulseOffset);
				pulsesDataset.read(pulseArray, *pulseType, pulseMemspace, pulsesDataspace);
				ptStartIdx = 0;
				numOfPoints = 0;
				transStartIdx = 0;
				numOfTransVals = 0;
				receivedStartIdx = 0;
				numOfReceivedVals = 0;
				
                if(spdFile->getPulseVersion() == 1)
                {
                    SPDPulseH5V1 *pulseObj = NULL;
                    for( boost::uint_fast32_t j = 0; j < spdFile->getPulseBlockSize(); ++j)
                    {
                        pulseObj = &((SPDPulseH5V1 *)pulseArray)[j];
                        if(j == 0)
                        {
                            ptStartIdx = pulseObj->ptsStartIdx;
                            transStartIdx = pulseObj->transmittedStartIdx;
                            receivedStartIdx = pulseObj->receivedStartIdx;
                        }
                        numOfPoints += pulseObj->numberOfReturns;
                        numOfTransVals += pulseObj->numOfTransmittedBins;
                        numOfReceivedVals += pulseObj->numOfReceivedBins;
                    }
                }
                else if(spdFile->getPulseVersion() == 2)
                {
                    SPDPulseH5V2 *pulseObj = NULL;
                    for( boost::uint_fast32_t j = 0; j < spdFile->getPulseBlockSize(); ++j)
                    {
                        pulseObj = &((SPDPulseH5V2 *)pulseArray)[j];
                        if(j == 0)
                        {
                            ptStartIdx = pulseObj->ptsStartIdx;
                            transStartIdx = pulseObj->transmittedStartIdx;
                            receivedStartIdx = pulseObj->receivedStartIdx;
                        }
                        numOfPoints += pulseObj->numberOfReturns;
                        numOfTransVals += pulseObj->numOfTransmittedBins;
                        numOfReceivedVals += pulseObj->numOfReceivedBins;
                    }
                }
                
				if(numOfPoints > 0)
				{
					// Read Points.
                    if(spdFile->getPointVersion() == 1)
                    {
                        pointsArray = new SPDPointH5V1[numOfPoints];
                    }
                    else if(spdFile->getPointVersion() == 2)
                    {
                        pointsArray = new SPDPointH5V2[numOfPoints];
                    }
                    
					pointOffset[0] = ptStartIdx;
					pointCount[0] = numOfPoints;
					pointsDataspace.selectHyperslab(H5S_SELECT_SET, pointCount, pointOffset);
					pointDims[0] = numOfPoints;
					pointMemspace = DataSpace(rank, pointDims);
					pointOffset_out[0] = 0;
					pointCount_out[0] = numOfPoints;
					pointMemspace.selectHyperslab( H5S_SELECT_SET, pointCount_out, pointOffset_out );
					pointsDataset.read(pointsArray, *pointType, pointMemspace, pointsDataspace);
				}
				
				if(numOfTransVals > 0)
				{
					// Read Transmitted Vals.
					transmittedArray = new unsigned long[numOfTransVals];
					transOffset[0] = transStartIdx;
					transCount[0] = numOfTransVals;
					transmittedDataspace.selectHyperslab(H5S_SELECT_SET, transCount, transOffset);
					transDims[0] = numOfTransVals;
					transMemspace = DataSpace(rank, transDims);
					transOffset_out[0] = 0;
					transCount_out[0] = numOfTransVals;
					transMemspace.selectHyperslab( H5S_SELECT_SET, transCount_out, transOffset_out );
					transmittedDataset.read(transmittedArray, PredType::NATIVE_ULONG, transMemspace, transmittedDataspace);
				}
				
				if(numOfReceivedVals > 0)
				{
					// Read Received Vals.
					receivedArray = new unsigned long[numOfReceivedVals];
					receivedOffset[0] = receivedStartIdx;
					receivedCount[0] = numOfReceivedVals;
					receivedDataspace.selectHyperslab(H5S_SELECT_SET, receivedCount, receivedOffset);
					receivedDims[0] = numOfReceivedVals;
					receivedMemspace = DataSpace(rank, receivedDims);
					receivedOffset_out[0] = 0;
					receivedCount_out[0] = numOfReceivedVals;
					receivedMemspace.selectHyperslab( H5S_SELECT_SET, receivedCount_out, receivedOffset_out );
					receivedDataset.read(receivedArray, PredType::NATIVE_ULONG, receivedMemspace, receivedDataspace);
				}
				
				ptIdx = 0;
				transIdx = 0;
				receivedIdx = 0;
				
				for( boost::uint_fast32_t j = 0; j < spdFile->getPulseBlockSize(); ++j)
				{
                    if(spdFile->getPulseVersion() == 1)
                    {
                        pulse = pulseUtils.createSPDPulseCopyFromH5(&((SPDPulseH5V1 *)pulseArray)[j]);
                    }
                    else if(spdFile->getPulseVersion() == 2)
                    {
                        pulse = pulseUtils.createSPDPulseCopyFromH5(&((SPDPulseH5V2 *)pulseArray)[j]);
                    }
                    
					if(pulse->numberOfReturns > 0)
					{
						pulse->pts = new vector<SPDPoint*>();
                        
                        for(boost::uint_fast16_t n = 0; n < pulse->numberOfReturns; ++n)
                        {
                            if(spdFile->getPointVersion() == 1)
                            {
                                point = ptsUtils.createSPDPointCopy(&((SPDPointH5V1 *)pointsArray)[ptIdx++]);
                            }
                            else if(spdFile->getPointVersion() == 2)
                            {
                                point = ptsUtils.createSPDPointCopy(&((SPDPointH5V2 *)pointsArray)[ptIdx++]);
                            }
                            pulse->pts->push_back(point);
                        }
					}
					
					if(pulse->numOfTransmittedBins > 0)
					{
						pulse->transmitted = new boost::uint_fast32_t[pulse->numOfTransmittedBins];
						for(boost::uint_fast16_t n = 0; n < pulse->numOfTransmittedBins; ++n)
						{
							pulse->transmitted[n] = transmittedArray[transIdx++];
						}
					}
					
					if(pulse->numOfReceivedBins > 0)
					{
						pulse->received = new boost::uint_fast32_t[pulse->numOfReceivedBins];
						for(boost::uint_fast16_t n = 0; n < pulse->numOfReceivedBins; ++n)
						{
							pulse->received[n] = receivedArray[receivedIdx++];
						}
					}
					
					if(changeIdxMethod)
					{
						this->defineIdxCoords(spdFile, pulse);
						
						if(first)
						{
							xMin = pulse->xIdx;
							xMax = pulse->xIdx;
							yMin = pulse->yIdx;
							yMax = pulse->yIdx;
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
						}
					}
					
					if(convertCoords)
					{
						this->transformPulseCoords(spdFile, pulse);
						
						if(first)
						{
							xMin = pulse->xIdx;
							xMax = pulse->xIdx;
							yMin = pulse->yIdx;
							yMax = pulse->yIdx;
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
						}
					}
					
					processor->processImportedPulse(spdFile, pulse);
				}
				
				if(numOfPoints > 0)
				{
                    if(spdFile->getPointVersion() == 1)
                    {
                        delete[] reinterpret_cast<SPDPointH5V1*>(pointsArray);
                    }
                    else if(spdFile->getPointVersion() == 2)
                    {
                        delete[] reinterpret_cast<SPDPointH5V2*>(pointsArray);
                    }
				}
				
				if(numOfTransVals > 0)
				{
					delete[] transmittedArray;
				}
				
				if(numOfReceivedVals > 0)
				{
					delete[] receivedArray;
				}
			}
            
            if(spdFile->getPulseVersion() == 1)
            {
                delete[] reinterpret_cast<SPDPulseH5V1*>(pulseArray);
            }
            else if(spdFile->getPulseVersion() == 2)
            {
                delete[] reinterpret_cast<SPDPulseH5V2*>(pulseArray);
            }
			
			if(remainingPulses > 0)
			{
				pulseOffset[0] = numOfPulsesInBlocks;
				pulseCount[0]  = remainingPulses;
				pulsesDataspace.selectHyperslab( H5S_SELECT_SET, pulseCount, pulseOffset );
				
				pulseDims[0] = remainingPulses;
				pulseMemspace = DataSpace( rank, pulseDims );
				
				pulseOffset_out[0] = 0;
				pulseCount_out[0]  = remainingPulses;
				pulseMemspace.selectHyperslab( H5S_SELECT_SET, pulseCount_out, pulseOffset_out );
                
                if(spdFile->getPulseVersion() == 1)
                {
                    pulseArray = new SPDPulseH5V1[remainingPulses];
                }
                else if(spdFile->getPulseVersion() == 2)
                {
                    pulseArray = new SPDPulseH5V2[remainingPulses];
                }
                else
                {
                    throw SPDIOException("SPD Pulse version was not recognised.");
                }
                
				pulsesDataset.read(pulseArray, *pulseType, pulseMemspace, pulsesDataspace );
				
				
				ptStartIdx = 0;
				numOfPoints = 0;
				transStartIdx = 0;
				numOfTransVals = 0;
				receivedStartIdx = 0;
				numOfReceivedVals = 0;
				
                if(spdFile->getPulseVersion() == 1)
                {
                    SPDPulseH5V1 *pulseObj = NULL;
                    for( boost::uint_fast32_t j = 0; j < remainingPulses; ++j)
                    {
                        pulseObj = &((SPDPulseH5V1 *)pulseArray)[j];
                        if(j == 0)
                        {
                            ptStartIdx = pulseObj->ptsStartIdx;
                            transStartIdx = pulseObj->transmittedStartIdx;
                            receivedStartIdx = pulseObj->receivedStartIdx;
                        }
                        numOfPoints += pulseObj->numberOfReturns;
                        numOfTransVals += pulseObj->numOfTransmittedBins;
                        numOfReceivedVals += pulseObj->numOfReceivedBins;
                    }
                }
                else if(spdFile->getPulseVersion() == 2)
                {
                    SPDPulseH5V2 *pulseObj = NULL;
                    for( boost::uint_fast32_t j = 0; j < remainingPulses; ++j)
                    {
                        pulseObj = &((SPDPulseH5V2 *)pulseArray)[j];
                        if(j == 0)
                        {
                            ptStartIdx = pulseObj->ptsStartIdx;
                            transStartIdx = pulseObj->transmittedStartIdx;
                            receivedStartIdx = pulseObj->receivedStartIdx;
                        }
                        numOfPoints += pulseObj->numberOfReturns;
                        numOfTransVals += pulseObj->numOfTransmittedBins;
                        numOfReceivedVals += pulseObj->numOfReceivedBins;
                    }
                }
                
				
				if(numOfPoints > 0)
				{
					// Read Points.
					if(spdFile->getPointVersion() == 1)
                    {
                        pointsArray = new SPDPointH5V1[numOfPoints];
                    }
                    else if(spdFile->getPointVersion() == 2)
                    {
                        pointsArray = new SPDPointH5V2[numOfPoints];
                    }
                    
					pointOffset[0] = ptStartIdx;
					pointCount[0] = numOfPoints;
					pointsDataspace.selectHyperslab(H5S_SELECT_SET, pointCount, pointOffset);
					pointDims[0] = numOfPoints;
					pointMemspace = DataSpace(rank, pointDims);
					pointOffset_out[0] = 0;
					pointCount_out[0] = numOfPoints;
					pointMemspace.selectHyperslab( H5S_SELECT_SET, pointCount_out, pointOffset_out );
					pointsDataset.read(pointsArray, *pointType, pointMemspace, pointsDataspace);
				}
				
				if(numOfTransVals > 0)
				{
					// Read Transmitted Vals.
					transmittedArray = new unsigned long[numOfTransVals];
					transOffset[0] = transStartIdx;
					transCount[0] = numOfTransVals;
					transmittedDataspace.selectHyperslab(H5S_SELECT_SET, transCount, transOffset);
					transDims[0] = numOfTransVals;
					transMemspace = DataSpace(rank, transDims);
					transOffset_out[0] = 0;
					transCount_out[0] = numOfTransVals;
					transMemspace.selectHyperslab( H5S_SELECT_SET, transCount_out, transOffset_out );
					transmittedDataset.read(transmittedArray, PredType::NATIVE_ULONG, transMemspace, transmittedDataspace);
				}
				
				if(numOfReceivedVals > 0)
				{
					// Read Received Vals.
					receivedArray = new unsigned long[numOfReceivedVals];
					receivedOffset[0] = receivedStartIdx;
					receivedCount[0] = numOfReceivedVals;
					receivedDataspace.selectHyperslab(H5S_SELECT_SET, receivedCount, receivedOffset);
					receivedDims[0] = numOfReceivedVals;
					receivedMemspace = DataSpace(rank, receivedDims);
					receivedOffset_out[0] = 0;
					receivedCount_out[0] = numOfReceivedVals;
					receivedMemspace.selectHyperslab( H5S_SELECT_SET, receivedCount_out, receivedOffset_out );
					receivedDataset.read(receivedArray, PredType::NATIVE_ULONG, receivedMemspace, receivedDataspace);
				}
				
				ptIdx = 0;
				transIdx = 0;
				receivedIdx = 0;
				
				for( boost::uint_fast32_t j = 0; j < remainingPulses; ++j)
				{
					if(spdFile->getPulseVersion() == 1)
                    {
                        pulse = pulseUtils.createSPDPulseCopyFromH5(&((SPDPulseH5V1 *)pulseArray)[j]);
                    }
                    else if(spdFile->getPulseVersion() == 2)
                    {
                        pulse = pulseUtils.createSPDPulseCopyFromH5(&((SPDPulseH5V2 *)pulseArray)[j]);
                    }
					
					if(pulse->numberOfReturns > 0)
					{
						pulse->pts = new vector<SPDPoint*>();
                        
                        for(boost::uint_fast16_t n = 0; n < pulse->numberOfReturns; ++n)
                        {
                            if(spdFile->getPointVersion() == 1)
                            {
                                point = ptsUtils.createSPDPointCopy(&((SPDPointH5V1 *)pointsArray)[ptIdx++]);
                            }
                            else if(spdFile->getPointVersion() == 2)
                            {
                                point = ptsUtils.createSPDPointCopy(&((SPDPointH5V2 *)pointsArray)[ptIdx++]);
                            }
                            pulse->pts->push_back(point);
                        }
					}
					
					if(pulse->numOfTransmittedBins > 0)
					{
						pulse->transmitted = new boost::uint_fast32_t[pulse->numOfTransmittedBins];
						for(boost::uint_fast16_t n = 0; n < pulse->numOfTransmittedBins; ++n)
						{
							pulse->transmitted[n] = transmittedArray[transIdx++];
						}
					}
					
					if(pulse->numOfReceivedBins > 0)
					{
						pulse->received = new boost::uint_fast32_t[pulse->numOfReceivedBins];
						for(boost::uint_fast16_t n = 0; n < pulse->numOfReceivedBins; ++n)
						{
							pulse->received[n] = receivedArray[receivedIdx++];
						}
					}
					
					if(changeIdxMethod)
					{
						this->defineIdxCoords(spdFile, pulse);
						
						if(first)
						{
							xMin = pulse->xIdx;
							xMax = pulse->xIdx;
							yMin = pulse->yIdx;
							yMax = pulse->yIdx;
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
						}
					}
					
					if(convertCoords)
					{
						this->transformPulseCoords(spdFile, pulse);
						
						if(first)
						{
							xMin = pulse->xIdx;
							xMax = pulse->xIdx;
							yMin = pulse->yIdx;
							yMax = pulse->yIdx;
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
						}
					}
					processor->processImportedPulse(spdFile, pulse);
				}
				
				if(numOfPoints > 0)
				{
                    if(spdFile->getPointVersion() == 1)
                    {
                        delete[] reinterpret_cast<SPDPointH5V1*>(pointsArray);
                    }
                    else if(spdFile->getPointVersion() == 2)
                    {
                        delete[] reinterpret_cast<SPDPointH5V2*>(pointsArray);
                    }
				}
				
				if(numOfTransVals > 0)
				{
					delete[] transmittedArray;
				}
				
				if(numOfReceivedVals > 0)
				{
					delete[] receivedArray;
				}
                
                if(spdFile->getPulseVersion() == 1)
                {
                    delete[] reinterpret_cast<SPDPulseH5V1*>(pulseArray);
                }
                else if(spdFile->getPulseVersion() == 2)
                {
                    delete[] reinterpret_cast<SPDPulseH5V2*>(pulseArray);
                }
            }
			delete pulseType;
			delete pointType;
            
			cout << ".Complete\n";
			
			if(convertCoords | changeIdxMethod)
			{
				spdFile->setBoundingBox(xMin, xMax, yMin, yMax);
			}
			
			spdInFile->close();
		}
		catch( FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataTypeIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch(SPDIOException &e)
		{
			throw e;
		}
	}
	
	bool SPDFileReader::isFileType(string fileType)
	{
		if(fileType == "SPD")
		{
			return true;
		}
		return false;
	}
		
	void SPDFileReader::readHeaderInfo(string, SPDFile *spdFile) throw(SPDIOException)
	{
		float inFloatDataValue[1];
		double inDoubleDataValue[1];
		int in16bitintDataValue[1];
		unsigned int in16bitUintDataValue[1];
		unsigned long in32bitUintDataValue[1];
		unsigned long long in64bitUintDataValue[1];
		
		unsigned int numLinesStr = 1;
		hid_t nativeStrType;
		char **strData = NULL;
		DataType strDataType;
		
		hsize_t dimsValue[1];
		dimsValue[0] = 1;
		DataSpace singleValueDataSpace(1, dimsValue);
		
		hsize_t	dims1Str[1];
		dims1Str[0] = numLinesStr;
		
		StrType strTypeAll(0, H5T_VARIABLE);
		
		const H5std_string spdFilePath( spdFile->getFilePath() );
		try 
		{
			Exception::dontPrint();
			
			// Create File..
			H5File *spdH5File = new H5File( spdFilePath, H5F_ACC_RDONLY );
			
			if((H5T_STRING!=H5Tget_class(strTypeAll.getId())) || (!H5Tis_variable_str(strTypeAll.getId())))
			{
				throw SPDIOException("The string data type defined is not variable.");
			}
			
            try 
            {
                DataSet datasetMajorVersion = spdH5File->openDataSet( SPDFILE_DATASETNAME_MAJOR_VERSION );
                datasetMajorVersion.read(in16bitUintDataValue, PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setMajorSPDVersion(in16bitUintDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("The SPD major version header value was not provided.");
            }
            
            try 
            {
                DataSet datasetMinorVersion = spdH5File->openDataSet( SPDFILE_DATASETNAME_MINOR_VERSION );
                datasetMinorVersion.read(in16bitUintDataValue, PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setMinorSPDVersion(in16bitUintDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("The SPD minor version header value was not provided.");
            }
            
            try 
            {
                DataSet datasetPointVersion = spdH5File->openDataSet( SPDFILE_DATASETNAME_POINT_VERSION );
                datasetPointVersion.read(in16bitUintDataValue, PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setPointVersion(in16bitUintDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("The SPD point version header value was not provided.");
            }
            
            try 
            {
                DataSet datasetPulseVersion = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_VERSION );
                datasetPulseVersion.read(in16bitUintDataValue, PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setPulseVersion(in16bitUintDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("The SPD pulse version header value was not provided.");
            }
                        
            try 
            {
                DataSet datasetSpatialReference = spdH5File->openDataSet( SPDFILE_DATASETNAME_SPATIAL_REFERENCE );
                strDataType = datasetSpatialReference.getDataType();
                strData = new char*[numLinesStr];
                if((nativeStrType=H5Tget_native_type(strDataType.getId(), H5T_DIR_DEFAULT))<0)
                {
                    throw SPDIOException("Could not define a native string type");
                }
                datasetSpatialReference.read((void*)strData, strDataType);
                spdFile->setSpatialReference(string(strData[0]));
                delete strData[0];
                delete[] strData;
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Spatial reference header value is not represent.");
            }

            try 
            {
                DataSet datasetFileType = spdH5File->openDataSet( SPDFILE_DATASETNAME_FILE_TYPE );
                datasetFileType.read(in16bitUintDataValue, PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setFileType(in16bitUintDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                try 
                {
                    DataSet datasetBinSize = spdH5File->openDataSet( SPDFILE_DATASETNAME_BIN_SIZE );
                    DataSet datasetNumberBinsX = spdH5File->openDataSet( SPDFILE_DATASETNAME_NUMBER_BINS_X );			
                    DataSet datasetNumberBinsY = spdH5File->openDataSet( SPDFILE_DATASETNAME_NUMBER_BINS_Y );
                    spdFile->setFileType(SPD_SEQ_TYPE);
                } 
                catch ( Exception &e) 
                {
                    spdFile->setFileType(SPD_UPD_TYPE);
                }
            }
            
            try 
            {
                DataSet datasetIndexType = spdH5File->openDataSet( SPDFILE_DATASETNAME_INDEX_TYPE );
                datasetIndexType.read(in16bitUintDataValue, PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setIndexType(in16bitUintDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                spdFile->setIndexType(SPD_NO_IDX);
                cerr << "Warning: Index type header value not provided. Defaulting to non-indexed file.\n";
            }
            
            try 
            {
                DataSet datasetDiscreteDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_DISCRETE_PT_DEFINED );
                datasetDiscreteDefined.read(in16bitintDataValue, PredType::NATIVE_INT, singleValueDataSpace);
                spdFile->setDiscretePtDefined(in16bitintDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Discrete Point Defined header value not provided.");
            }
            
            try 
            {
                DataSet datasetDecomposedDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_DECOMPOSED_PT_DEFINED );
                datasetDecomposedDefined.read(in16bitintDataValue, PredType::NATIVE_INT, singleValueDataSpace);
                spdFile->setDecomposedPtDefined(in16bitintDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Decomposed Point Defined header value not provided.");
            }
            
            try 
            {
                DataSet datasetTransWaveformDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_TRANS_WAVEFORM_DEFINED );
                datasetTransWaveformDefined.read(in16bitintDataValue, PredType::NATIVE_INT, singleValueDataSpace);
                spdFile->setTransWaveformDefined(in16bitintDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Transmitted Waveform Defined header value not provided.");
            }
            
            try 
            {
                DataSet datasetReceiveWaveformDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_RECEIVE_WAVEFORM_DEFINED );
                datasetReceiveWaveformDefined.read(in16bitintDataValue, PredType::NATIVE_INT, singleValueDataSpace);
                spdFile->setReceiveWaveformDefined(in16bitintDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Received Waveform Defined header value not provided.");
            }
            
            try 
            {
                DataSet datasetGeneratingSoftware = spdH5File->openDataSet( SPDFILE_DATASETNAME_GENERATING_SOFTWARE );
                strDataType = datasetGeneratingSoftware.getDataType();
                strData = new char*[numLinesStr];
                if((nativeStrType=H5Tget_native_type(strDataType.getId(), H5T_DIR_DEFAULT))<0)
                {
                    throw SPDIOException("Could not define a native string type");
                }
                datasetGeneratingSoftware.read((void*)strData, strDataType);
                spdFile->setGeneratingSoftware(string(strData[0]));
                delete strData[0];
                delete[] strData;
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Generating software header value not provided.");
            }
            
            try 
            {
                DataSet datasetSystemIdentifier = spdH5File->openDataSet( SPDFILE_DATASETNAME_SYSTEM_IDENTIFIER );
                strDataType = datasetSystemIdentifier.getDataType();
                strData = new char*[numLinesStr];
                if((nativeStrType=H5Tget_native_type(strDataType.getId(), H5T_DIR_DEFAULT))<0)
                {
                    throw SPDIOException("Could not define a native string type");
                }
                datasetSystemIdentifier.read((void*)strData, strDataType);
                spdFile->setSystemIdentifier(string(strData[0]));
                delete strData[0];
                delete[] strData;
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("System identifier header value not provided.");
            }
            
            try 
            {
                DataSet datasetFileSignature = spdH5File->openDataSet( SPDFILE_DATASETNAME_FILE_SIGNATURE );
                strDataType = datasetFileSignature.getDataType();
                strData = new char*[numLinesStr];
                if((nativeStrType=H5Tget_native_type(strDataType.getId(), H5T_DIR_DEFAULT))<0)
                {
                    throw SPDIOException("Could not define a native string type");
                }
                datasetFileSignature.read((void*)strData, strDataType);
                spdFile->setFileSignature(string(strData[0]));
                delete strData[0];
                delete[] strData;
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("File signature header value not provided.");
            }
            
            try 
            {
                DataSet datasetYearOfCreation = spdH5File->openDataSet( SPDFILE_DATASETNAME_YEAR_OF_CREATION );
                DataSet datasetMonthOfCreation = spdH5File->openDataSet( SPDFILE_DATASETNAME_MONTH_OF_CREATION );
                DataSet datasetDayOfCreation = spdH5File->openDataSet( SPDFILE_DATASETNAME_DAY_OF_CREATION );
                DataSet datasetHourOfCreation = spdH5File->openDataSet( SPDFILE_DATASETNAME_HOUR_OF_CREATION );
                DataSet datasetMinuteOfCreation = spdH5File->openDataSet( SPDFILE_DATASETNAME_MINUTE_OF_CREATION );
                DataSet datasetSecondOfCreation = spdH5File->openDataSet( SPDFILE_DATASETNAME_SECOND_OF_CREATION );
                
                datasetYearOfCreation.read(in16bitUintDataValue, PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setYearOfCreation(in16bitUintDataValue[0]);
                
                datasetMonthOfCreation.read(in16bitUintDataValue, PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setMonthOfCreation(in16bitUintDataValue[0]);
                
                datasetDayOfCreation.read(in16bitUintDataValue, PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setDayOfCreation(in16bitUintDataValue[0]);
                
                datasetHourOfCreation.read(in16bitUintDataValue, PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setHourOfCreation(in16bitUintDataValue[0]);
                
                datasetMinuteOfCreation.read(in16bitUintDataValue, PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setMinuteOfCreation(in16bitUintDataValue[0]);
                
                datasetSecondOfCreation.read(in16bitUintDataValue, PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setSecondOfCreation(in16bitUintDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Date of file creation header values not provided.");
            }
            
            try 
            {
                DataSet datasetYearOfCapture = spdH5File->openDataSet( SPDFILE_DATASETNAME_YEAR_OF_CAPTURE );
                DataSet datasetMonthOfCapture = spdH5File->openDataSet( SPDFILE_DATASETNAME_MONTH_OF_CAPTURE );
                DataSet datasetDayOfCapture = spdH5File->openDataSet( SPDFILE_DATASETNAME_DAY_OF_CAPTURE );
                DataSet datasetHourOfCapture = spdH5File->openDataSet( SPDFILE_DATASETNAME_HOUR_OF_CAPTURE );
                DataSet datasetMinuteOfCapture = spdH5File->openDataSet( SPDFILE_DATASETNAME_MINUTE_OF_CAPTURE );
                DataSet datasetSecondOfCapture = spdH5File->openDataSet( SPDFILE_DATASETNAME_SECOND_OF_CAPTURE );
                
                datasetYearOfCapture.read(in16bitUintDataValue, PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setYearOfCapture(in16bitUintDataValue[0]);
                
                datasetMonthOfCapture.read(in16bitUintDataValue, PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setMonthOfCapture(in16bitUintDataValue[0]);
                
                datasetDayOfCapture.read(in16bitUintDataValue, PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setDayOfCapture(in16bitUintDataValue[0]);
                
                datasetHourOfCapture.read(in16bitUintDataValue, PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setHourOfCapture(in16bitUintDataValue[0]);
                
                datasetMinuteOfCapture.read(in16bitUintDataValue, PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setMinuteOfCapture(in16bitUintDataValue[0]);
                
                datasetSecondOfCapture.read(in16bitUintDataValue, PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setSecondOfCapture(in16bitUintDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Date/Time of capture header values not provided.");
            }
            
            try 
            {
                DataSet datasetNumberOfPoints = spdH5File->openDataSet( SPDFILE_DATASETNAME_NUMBER_OF_POINTS );
                datasetNumberOfPoints.read(in64bitUintDataValue, PredType::NATIVE_ULLONG, singleValueDataSpace);
                spdFile->setNumberOfPoints(in64bitUintDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Number of points header value not provided.");
            }
            
            try 
            {
                DataSet datasetNumberOfPulses = spdH5File->openDataSet( SPDFILE_DATASETNAME_NUMBER_OF_PULSES );
                datasetNumberOfPulses.read(in64bitUintDataValue, PredType::NATIVE_ULLONG, singleValueDataSpace);
                spdFile->setNumberOfPulses(in64bitUintDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Number of pulses header value not provided.");
            }
            
            try 
            {
                DataSet datasetUserMetaData = spdH5File->openDataSet( SPDFILE_DATASETNAME_USER_META_DATA );
                strDataType = datasetUserMetaData.getDataType();
                strData = new char*[numLinesStr];
                if((nativeStrType=H5Tget_native_type(strDataType.getId(), H5T_DIR_DEFAULT))<0)
                {
                    throw SPDIOException("Could not define a native string type");
                }
                datasetUserMetaData.read((void*)strData, strDataType);
                spdFile->setUserMetaField(string(strData[0]));
                delete strData[0];
                delete[] strData;
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("User metadata header value not provided.");
            }
            
            try 
            {
                DataSet datasetXMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_X_MIN );
                DataSet datasetXMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_X_MAX );
                DataSet datasetYMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_Y_MIN );
                DataSet datasetYMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_Y_MAX );
                DataSet datasetZMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_Z_MIN );
                DataSet datasetZMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_Z_MAX );
                
                datasetXMin.read(inDoubleDataValue, PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setXMin(inDoubleDataValue[0]);
                
                datasetXMax.read(inDoubleDataValue, PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setXMax(inDoubleDataValue[0]);
                
                datasetYMin.read(inDoubleDataValue, PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setYMin(inDoubleDataValue[0]);
                
                datasetYMax.read(inDoubleDataValue, PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setYMax(inDoubleDataValue[0]);
                
                datasetZMin.read(inDoubleDataValue, PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setZMin(inDoubleDataValue[0]);
                
                datasetZMax.read(inDoubleDataValue, PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setZMax(inDoubleDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Dataset bounding volume header values not provided.");
            }
            
            try 
            {
                DataSet datasetZenithMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_ZENITH_MIN );
                DataSet datasetZenithMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_ZENITH_MAX );
                DataSet datasetAzimuthMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_AZIMUTH_MIN );
                DataSet datasetAzimuthMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_AZIMUTH_MAX );
                DataSet datasetRangeMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_RANGE_MIN );
                DataSet datasetRangeMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_RANGE_MAX );
                
                datasetZenithMin.read(inDoubleDataValue, PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setZenithMin(inDoubleDataValue[0]);
                
                datasetZenithMax.read(inDoubleDataValue, PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setZenithMax(inDoubleDataValue[0]);
                
                datasetAzimuthMax.read(inDoubleDataValue, PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setAzimuthMax(inDoubleDataValue[0]);
                
                datasetAzimuthMin.read(inDoubleDataValue, PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setAzimuthMin(inDoubleDataValue[0]);
                
                datasetRangeMax.read(inDoubleDataValue, PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setRangeMax(inDoubleDataValue[0]);
                
                datasetRangeMin.read(inDoubleDataValue, PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setRangeMin(inDoubleDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Bounding spherical volume header values not provided.");
            }
            
            try 
            {
                DataSet datasetScanlineMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_SCANLINE_MIN );
                DataSet datasetScanlineMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_SCANLINE_MAX );
                DataSet datasetScanlineIdxMin = spdH5File->openDataSet( SPDFILE_DATASETNAME_SCANLINE_IDX_MIN );
                DataSet datasetScanlineIdxMax = spdH5File->openDataSet( SPDFILE_DATASETNAME_SCANLINE_IDX_MAX );
                
                datasetScanlineMin.read(inDoubleDataValue, PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setScanlineMin(inDoubleDataValue[0]);
                
                datasetScanlineMax.read(inDoubleDataValue, PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setScanlineMax(inDoubleDataValue[0]);
                
                datasetScanlineIdxMin.read(inDoubleDataValue, PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setScanlineIdxMax(inDoubleDataValue[0]);
                
                datasetScanlineIdxMax.read(inDoubleDataValue, PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setScanlineIdxMin(inDoubleDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                spdFile->setScanlineMin(0);
                spdFile->setScanlineMax(0);
                spdFile->setScanlineIdxMax(0);
                spdFile->setScanlineIdxMin(0);
            }
            
            if(spdFile->getFileType() != SPD_UPD_TYPE)
            {
                try 
                {
                    DataSet datasetBinSize = spdH5File->openDataSet( SPDFILE_DATASETNAME_BIN_SIZE );
                    datasetBinSize.read(inFloatDataValue, PredType::NATIVE_FLOAT, singleValueDataSpace);
                    spdFile->setBinSize(inFloatDataValue[0]);
                } 
                catch ( Exception &e) 
                {
                    throw SPDIOException("Bin size header value not provided.");
                }
                
                try 
                {
                    DataSet datasetNumberBinsX = spdH5File->openDataSet( SPDFILE_DATASETNAME_NUMBER_BINS_X );
                    datasetNumberBinsX.read(in32bitUintDataValue, PredType::NATIVE_ULONG, singleValueDataSpace);
                    spdFile->setNumberBinsX(in32bitUintDataValue[0]);
                } 
                catch ( Exception &e) 
                {
                    throw SPDIOException("Number of X bins header value not provided.");
                }

                try 
                {
                    DataSet datasetNumberBinsY = spdH5File->openDataSet( SPDFILE_DATASETNAME_NUMBER_BINS_Y );
                    datasetNumberBinsY.read(in32bitUintDataValue, PredType::NATIVE_ULONG, singleValueDataSpace);
                    spdFile->setNumberBinsY(in32bitUintDataValue[0]);
                } 
                catch ( Exception &e) 
                {
                    throw SPDIOException("Number of Y bins header value not provided.");
                }
            }
            
            try 
            {
                DataSet datasetPulseRepFreq = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_REPETITION_FREQ );
                datasetPulseRepFreq.read(inFloatDataValue, PredType::NATIVE_FLOAT, singleValueDataSpace);
                spdFile->setPulseRepetitionFreq(inFloatDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Pulse repetition frequency header value not provided.");
            }

            try 
            {
                DataSet datasetBeamDivergence = spdH5File->openDataSet( SPDFILE_DATASETNAME_BEAM_DIVERGENCE );
                datasetBeamDivergence.read(inFloatDataValue, PredType::NATIVE_FLOAT, singleValueDataSpace);
                spdFile->setBeamDivergence(inFloatDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Beam divergence header value not provided.");
            }
            
            try 
            {
                DataSet datasetSensorHeight = spdH5File->openDataSet( SPDFILE_DATASETNAME_SENSOR_HEIGHT );
                datasetSensorHeight.read(inDoubleDataValue, PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setSensorHeight(inDoubleDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Sensor height header value not provided.");
            }
            
            try 
            {
                DataSet datasetFootprint = spdH5File->openDataSet( SPDFILE_DATASETNAME_FOOTPRINT );
                datasetFootprint.read(inFloatDataValue, PredType::NATIVE_FLOAT, singleValueDataSpace);
                spdFile->setFootprint(inFloatDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Footprint header value not provided.");
            }
            
            try 
            {
                DataSet datasetMaxScanAngle = spdH5File->openDataSet( SPDFILE_DATASETNAME_MAX_SCAN_ANGLE );
                datasetMaxScanAngle.read(inFloatDataValue, PredType::NATIVE_FLOAT, singleValueDataSpace);
                spdFile->setMaxScanAngle(inFloatDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Max scan angle header value not provided.");
            }

            try 
            {
                DataSet datasetRGBDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_RGB_DEFINED );
                datasetRGBDefined.read(in16bitintDataValue, PredType::NATIVE_INT, singleValueDataSpace);
                spdFile->setRGBDefined(in16bitintDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("RGB defined header value not provided.");
            }
            
            try 
            {
                DataSet datasetPulseBlockSize = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_BLOCK_SIZE );
                datasetPulseBlockSize.read(in16bitUintDataValue, PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setPulseBlockSize(in16bitUintDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Pulse block size header value not provided.");
            }
            
            try 
            {
                DataSet datasetPointsBlockSize = spdH5File->openDataSet( SPDFILE_DATASETNAME_POINT_BLOCK_SIZE );
                datasetPointsBlockSize.read(in16bitUintDataValue, PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setPointBlockSize(in16bitUintDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Point block size header value not provided.");
            }
            
            try 
            {
                DataSet datasetReceivedBlockSize = spdH5File->openDataSet( SPDFILE_DATASETNAME_RECEIVED_BLOCK_SIZE );
                datasetReceivedBlockSize.read(in16bitUintDataValue, PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setReceivedBlockSize(in16bitUintDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Received waveform block size header value not provided.");
            }
            
            try 
            {
                DataSet datasetTransmittedBlockSize = spdH5File->openDataSet( SPDFILE_DATASETNAME_TRANSMITTED_BLOCK_SIZE );
                datasetTransmittedBlockSize.read(in16bitUintDataValue, PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setTransmittedBlockSize(in16bitUintDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Transmitted waveform block size header value not provided.");
            }
            
            try 
            {
                DataSet datasetWaveformBitRes = spdH5File->openDataSet( SPDFILE_DATASETNAME_WAVEFORM_BIT_RES );
                datasetWaveformBitRes.read(in16bitUintDataValue, PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setWaveformBitRes(in16bitUintDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Waveform bit resolution header value not provided.");
            }
            
            try 
            {
                DataSet datasetTemporalBinSpacing = spdH5File->openDataSet( SPDFILE_DATASETNAME_TEMPORAL_BIN_SPACING );
                datasetTemporalBinSpacing.read(inDoubleDataValue, PredType::NATIVE_DOUBLE, singleValueDataSpace);
                spdFile->setTemporalBinSpacing(inDoubleDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Temporal bin spacing header value not provided.");
            }
            
            try 
            {
                DataSet datasetReturnNumsSynGen = spdH5File->openDataSet( SPDFILE_DATASETNAME_RETURN_NUMBERS_SYN_GEN );
                datasetReturnNumsSynGen.read(in16bitintDataValue, PredType::NATIVE_INT, singleValueDataSpace);
                spdFile->setReturnNumsSynGen(in16bitintDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Return number synthetically generated header value not provided.");
            }
            
            try 
            {
                DataSet datasetHeightDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_HEIGHT_DEFINED );
                datasetHeightDefined.read(in16bitintDataValue, PredType::NATIVE_INT, singleValueDataSpace);
                spdFile->setHeightDefined(in16bitintDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Height fields defined header value not provided.");
            }
            
            try 
            {
                DataSet datasetSensorSpeed = spdH5File->openDataSet( SPDFILE_DATASETNAME_SENSOR_SPEED );
                datasetSensorSpeed.read(inFloatDataValue, PredType::NATIVE_FLOAT, singleValueDataSpace);
                spdFile->setSensorSpeed(inFloatDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Sensor speed header value not provided.");
            }
            
            try 
            {
                DataSet datasetSensorScanRate = spdH5File->openDataSet( SPDFILE_DATASETNAME_SENSOR_SCAN_RATE );
                datasetSensorScanRate.read(inFloatDataValue, PredType::NATIVE_FLOAT, singleValueDataSpace);
                spdFile->setSensorScanRate(inFloatDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Sensor Scan Rate header value not provided.");
            }
            
            try 
            {
                DataSet datasetPointDensity = spdH5File->openDataSet( SPDFILE_DATASETNAME_POINT_DENSITY );
                datasetPointDensity.read(inFloatDataValue, PredType::NATIVE_FLOAT, singleValueDataSpace);
                spdFile->setPointDensity(inFloatDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Point density header value not provided.");
            }
            
            try 
            {
                DataSet datasetPulseDensity = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_DENSITY );
                datasetPulseDensity.read(inFloatDataValue, PredType::NATIVE_FLOAT, singleValueDataSpace);
                spdFile->setPulseDensity(inFloatDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Pulse density header value not provided.");
            }
            
            try 
            {
                DataSet datasetPulseCrossTrackSpacing = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_CROSS_TRACK_SPACING );
                datasetPulseCrossTrackSpacing.read(inFloatDataValue, PredType::NATIVE_FLOAT, singleValueDataSpace);
                spdFile->setPulseCrossTrackSpacing(inFloatDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Cross track spacing header value not provided.");
            }
            
            try 
            {
                DataSet datasetPulseAlongTrackSpacing = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_ALONG_TRACK_SPACING );
                datasetPulseAlongTrackSpacing.read(inFloatDataValue, PredType::NATIVE_FLOAT, singleValueDataSpace);
                spdFile->setPulseAlongTrackSpacing(inFloatDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Along track spacing header value not provided.");
            }
            
            try 
            {
                DataSet datasetOriginDefined = spdH5File->openDataSet( SPDFILE_DATASETNAME_ORIGIN_DEFINED );
                datasetOriginDefined.read(in16bitintDataValue, PredType::NATIVE_INT, singleValueDataSpace);
                spdFile->setOriginDefined(in16bitintDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Origin defined header value not provided.");
            }
            
            try 
            {
                DataSet datasetPulseAngularSpacingAzimuth = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_ANGULAR_SPACING_AZIMUTH );
                datasetPulseAngularSpacingAzimuth.read(inFloatDataValue, PredType::NATIVE_FLOAT, singleValueDataSpace);
                spdFile->setPulseAngularSpacingAzimuth(inFloatDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Angular azimuth spacing header value not provided.");
            }
            
            try 
            {
                DataSet datasetPulseAngularSpacingZenith = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_ANGULAR_SPACING_ZENITH );
                datasetPulseAngularSpacingZenith.read(inFloatDataValue, PredType::NATIVE_FLOAT, singleValueDataSpace);
                spdFile->setPulseAngularSpacingZenith(inFloatDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                throw SPDIOException("Angular Zenith spacing header value not provided.");
            }
            
            try 
            {
                DataSet datasetPulseIndexMethod = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_INDEX_METHOD );
                datasetPulseIndexMethod.read(in16bitintDataValue, PredType::NATIVE_INT, singleValueDataSpace);
                spdFile->setPulseIdxMethod(in16bitintDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                spdFile->setPulseIdxMethod(SPD_FIRST_RETURN);
                cerr << "Method of indexing header value not provided. Default: First Return\n";
            }
            
            try 
            {
                DataSet datasetSensorApertureSize = spdH5File->openDataSet( SPDFILE_DATASETNAME_SENSOR_APERTURE_SIZE );
                datasetSensorApertureSize.read(inFloatDataValue, PredType::NATIVE_FLOAT, singleValueDataSpace);
                spdFile->setSensorApertureSize(inFloatDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                //ignore
                spdFile->setSensorApertureSize(0);
            }
            
            try 
            {
                DataSet datasetPulseEnergy = spdH5File->openDataSet( SPDFILE_DATASETNAME_PULSE_ENERGY );
                datasetPulseEnergy.read(inFloatDataValue, PredType::NATIVE_FLOAT, singleValueDataSpace);
                spdFile->setPulseEnergy(inFloatDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                //ignore
                spdFile->setPulseEnergy(0);
            }
            
            try 
            {
                DataSet datasetFieldOfView = spdH5File->openDataSet( SPDFILE_DATASETNAME_FIELD_OF_VIEW );
                datasetFieldOfView.read(inFloatDataValue, PredType::NATIVE_FLOAT, singleValueDataSpace);
                spdFile->setFieldOfView(inFloatDataValue[0]);
            } 
            catch ( Exception &e) 
            {
                //ignore
                spdFile->setFieldOfView(0);
            }
            
            try 
            {
                DataSet datasetNumOfWavelengths = spdH5File->openDataSet( SPDFILE_DATASETNAME_NUM_OF_WAVELENGTHS );
                datasetNumOfWavelengths.read(in16bitUintDataValue, PredType::NATIVE_UINT, singleValueDataSpace);
                spdFile->setNumOfWavelengths(in16bitUintDataValue[0]);
                
                if(in16bitUintDataValue[0] > 0)
                {
                    float *inFloatDataValues = new float[in16bitUintDataValue[0]];
                    hsize_t dimsValue[1];
                    dimsValue[0] = in16bitUintDataValue[0];
                    DataSpace valuesDataSpace(1, dimsValue);
                    DataSet datasetWavelengths = spdH5File->openDataSet( SPDFILE_DATASETNAME_WAVELENGTHS );
                    datasetWavelengths.read(inFloatDataValues, PredType::NATIVE_FLOAT, valuesDataSpace);
                    vector<float> wavelengths;
                    for(unsigned int i = 0; i < in16bitUintDataValue[0]; ++i)
                    {
                        wavelengths.push_back(inFloatDataValues[i]);
                    }
                    spdFile->setWavelengths(wavelengths);
                    
                    DataSet datasetBandwidths = spdH5File->openDataSet( SPDFILE_DATASETNAME_BANDWIDTHS );
                    datasetWavelengths.read(inFloatDataValues, PredType::NATIVE_FLOAT, valuesDataSpace);
                    vector<float> bandwidths;
                    for(unsigned int i = 0; i < in16bitUintDataValue[0]; ++i)
                    {
                        bandwidths.push_back(inFloatDataValues[i]);
                    }
                    spdFile->setBandwidths(bandwidths);
                    delete[] inFloatDataValues;
                }
                else
                {
                    vector<float> wavelengths;
                    spdFile->setWavelengths(wavelengths);
                    vector<float> bandwidths;
                    spdFile->setBandwidths(bandwidths);
                }
                
            } 
            catch ( Exception &e) 
            {
                DataSet datasetWavelength = spdH5File->openDataSet( SPDFILE_DATASETNAME_WAVELENGTH );
                datasetWavelength.read(inFloatDataValue, PredType::NATIVE_FLOAT, singleValueDataSpace);
                spdFile->setNumOfWavelengths(1);
                vector<float> wavelengths;
                wavelengths.push_back(inFloatDataValue[0]);
                spdFile->setWavelengths(wavelengths);
                vector<float> bandwidths;
                bandwidths.push_back(0);
                spdFile->setBandwidths(bandwidths);
            }
            
            
            

			spdH5File->close();
			delete spdH5File;
			
		}
		catch( FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataTypeIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
	}
	
	void SPDFileReader::readRefHeaderRow(H5File *spdInFile, boost::uint_fast32_t row, unsigned long long *binOffsets, unsigned long *numPtsInBin, boost::uint_fast32_t numXBins) throw(SPDIOException)
	{
		try
		{
			Exception::dontPrint();
			
			// Read data 
			DataSet plsPerBinDSet = spdInFile->openDataSet( SPDFILE_DATASETNAME_PLS_PER_BIN );
			DataSpace plsPerBinDSpace = plsPerBinDSet.getSpace();
			
			DataSet binOffsetsDset = spdInFile->openDataSet( SPDFILE_DATASETNAME_BIN_OFFSETS );
			DataSpace binOffsetsDSpace = binOffsetsDset.getSpace();
			
			hsize_t offsetDims[2];
			offsetDims[0] = row;
			offsetDims[1] = 0;
			hsize_t selectionSize[2];
			selectionSize[0]  = 1;
			selectionSize[1]  = numXBins;
			plsPerBinDSpace.selectHyperslab( H5S_SELECT_SET, selectionSize, offsetDims );
			binOffsetsDSpace.selectHyperslab( H5S_SELECT_SET, selectionSize, offsetDims );
			
			hsize_t memSpaceDims[1]; 
			memSpaceDims[0] = numXBins;
			DataSpace memSpace( 1, memSpaceDims ); // has rank == 1
			
			hsize_t offsetMemSpace[1];
			hsize_t selectionMemSpace[1];
			offsetMemSpace[0] = 0;
			selectionMemSpace[0]  = numXBins;
			memSpace.selectHyperslab( H5S_SELECT_SET, selectionMemSpace, offsetMemSpace );
			
			binOffsetsDset.read( binOffsets, PredType::NATIVE_ULLONG, memSpace, binOffsetsDSpace );
			plsPerBinDSet.read( numPtsInBin, PredType::NATIVE_ULONG, memSpace, plsPerBinDSpace );
		}
		catch( FileIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSetIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataSpaceIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch( DataTypeIException &e )
		{
			throw SPDIOException(e.getCDetailMsg());
		}
		catch(SPDIOException &e)
		{
			throw e;
		}
	}
	
	
	SPDFileReader::~SPDFileReader()
	{
		
	}
}


