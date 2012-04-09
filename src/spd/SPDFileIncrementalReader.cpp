/*
 *  SPDFileIncrementalReader.cpp
 *  spdlib_prj
 *
 *  Created by Pete Bunting on 11/10/2009.
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

#include "spd/SPDFileIncrementalReader.h"

namespace spdlib
{

	SPDFileIncrementalReader::SPDFileIncrementalReader():spdFile(NULL), spdInFile(NULL), pulseType(NULL), pointType(NULL), fileOpened(false)
	{
	
	}
	
	SPDFileIncrementalReader::SPDFileIncrementalReader(const SPDFileIncrementalReader &spdReader) throw(SPDException) : spdFile(NULL), spdInFile(NULL), pulseType(NULL), pointType(NULL), fileOpened(false)
	{
		if(fileOpened)
		{
			throw SPDException("Cannot copy reader as file open");
		}
		
		this->spdFile = spdReader.spdFile;
		this->spdInFile = spdReader.spdInFile;
		this->fileOpened = false;
	}
	
	SPDFileIncrementalReader& SPDFileIncrementalReader::operator=(const SPDFileIncrementalReader& spdReader) throw(SPDException)
	{
		if(fileOpened)
		{
			throw SPDException("Cannot copy reader as file open");
		}
		this->spdFile = spdReader.spdFile;
		this->spdInFile = spdReader.spdInFile;
		this->fileOpened = false;
		return *this;
	}
	
    /*UPDATED*/
	bool SPDFileIncrementalReader::open(SPDFile *spdFile) throw(SPDIOException)
	{
		try
		{
			Exception::dontPrint();
			SPDFileReader spdFileReader;
			SPDPointUtils ptsUtils;
			SPDPulseUtils pulseUtils;
			
			this->spdFile = spdFile;
			spdFileReader.readHeaderInfo(spdFile->getFilePath(), spdFile);
			spdInFile = new H5File( spdFile->getFilePath(), H5F_ACC_RDONLY );
			fileOpened = true;
			if(spdFile->getPulseVersion() == 1)
            {
                pulseType = pulseUtils.createSPDPulseH5V1DataTypeMemory();
            }
            else if(spdFile->getPulseVersion() == 2)
            {
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
		
		return fileOpened;
	}
    
    /*UPDATED*/
    void SPDFileIncrementalReader::readPulseData(list<SPDPulse*> *pulses, boost::uint_fast64_t offset, boost::uint_fast64_t numPulses) throw(SPDIOException)
	{
        if(!fileOpened)
		{
			throw SPDIOException("Input file is not open..");
		}
		
        try
        {
            Exception::dontPrint();
            SPDPointUtils ptsUtils;
            SPDPulseUtils pulseUtils;
            
            if(numPulses > 0)
            {
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
                pulseCount[0]  = numPulses;
                pulsesDataspace.selectHyperslab( H5S_SELECT_SET, pulseCount, pulseOffset );
                
                hsize_t pulseDims[1]; 
                pulseDims[0] = numPulses;
                DataSpace pulseMemspace( rank, pulseDims );
                
                hsize_t pulseOffset_out[1];
                hsize_t pulseCount_out[1];
                pulseOffset_out[0] = 0;
                pulseCount_out[0]  = numPulses;
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
                unsigned long *transmittedArray = NULL;
                unsigned long *receivedArray = NULL;
                
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
                
                pulseOffset[0] = offset;
                pulseCount[0]  = numPulses;
                pulsesDataspace.selectHyperslab( H5S_SELECT_SET, pulseCount, pulseOffset );
                
                pulseDims[0] = numPulses;
                pulseMemspace = DataSpace( rank, pulseDims );
                
                pulseOffset_out[0] = 0;
                pulseCount_out[0]  = numPulses;
                pulseMemspace.selectHyperslab( H5S_SELECT_SET, pulseCount_out, pulseOffset_out );
                                
                if(spdFile->getPulseVersion() == 1)
                {
                    pulseArray = new SPDPulseH5V1[numPulses];
                }
                else if(spdFile->getPulseVersion() == 2)
                {
                    pulseArray = new SPDPulseH5V2[numPulses];
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
                    for( boost::uint_fast32_t j = 0; j < numPulses; ++j)
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
                    for( boost::uint_fast32_t j = 0; j < numPulses; ++j)
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
                
                for( boost::uint_fast32_t j = 0; j < numPulses; ++j)
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
                        pulse->pts->reserve(pulse->numberOfReturns);
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
    
	/*UPDATED*/
	void SPDFileIncrementalReader::readPulseData(vector<SPDPulse*> *pulses, boost::uint_fast64_t offset, boost::uint_fast64_t numPulses) throw(SPDIOException)
	{
		if(!fileOpened)
		{
			throw SPDIOException("Input file is not open..");
		}
		
        try
        {
            Exception::dontPrint();
            SPDPointUtils ptsUtils;
            SPDPulseUtils pulseUtils;
            
            if(numPulses > 0)
            {
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
                pulseCount[0]  = numPulses;
                pulsesDataspace.selectHyperslab( H5S_SELECT_SET, pulseCount, pulseOffset );
                
                hsize_t pulseDims[1]; 
                pulseDims[0] = numPulses;
                DataSpace pulseMemspace( rank, pulseDims );
                
                hsize_t pulseOffset_out[1];
                hsize_t pulseCount_out[1];
                pulseOffset_out[0] = 0;
                pulseCount_out[0]  = numPulses;
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
                unsigned long *transmittedArray = NULL;
                unsigned long *receivedArray = NULL;
                
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
                
                pulseOffset[0] = offset;
                pulseCount[0]  = numPulses;
                pulsesDataspace.selectHyperslab( H5S_SELECT_SET, pulseCount, pulseOffset );
                
                pulseDims[0] = numPulses;
                pulseMemspace = DataSpace( rank, pulseDims );
                
                pulseOffset_out[0] = 0;
                pulseCount_out[0]  = numPulses;
                pulseMemspace.selectHyperslab( H5S_SELECT_SET, pulseCount_out, pulseOffset_out );
                
                if(spdFile->getPulseVersion() == 1)
                {
                    pulseArray = new SPDPulseH5V1[numPulses];
                }
                else if(spdFile->getPulseVersion() == 2)
                {
                    pulseArray = new SPDPulseH5V2[numPulses];
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
                    for( boost::uint_fast32_t j = 0; j < numPulses; ++j)
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
                    for( boost::uint_fast32_t j = 0; j < numPulses; ++j)
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
                
                for( boost::uint_fast32_t j = 0; j < numPulses; ++j)
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
                        pulse->pts->reserve(pulse->numberOfReturns);
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
	
	void SPDFileIncrementalReader::readRefHeaderRow(boost::uint_fast32_t row, unsigned long long *binOffsets, unsigned long *numPtsInBin) throw(SPDIOException)
	{
		if(!fileOpened)
		{
			throw SPDIOException("Input file is not open..");
		}
        
        if(spdFile->getFileType() == SPD_UPD_TYPE)
        {
            throw SPDIOException("This function is only available for files with a spatial index.");
        }
		
		try
		{
			Exception::dontPrint();
			
			/* Read data */
			DataSet plsPerBinDSet = spdInFile->openDataSet( SPDFILE_DATASETNAME_PLS_PER_BIN );
			DataSpace plsPerBinDSpace = plsPerBinDSet.getSpace();
			
			DataSet binOffsetsDset = spdInFile->openDataSet( SPDFILE_DATASETNAME_BIN_OFFSETS );
			DataSpace binOffsetsDSpace = binOffsetsDset.getSpace();
			
			hsize_t offsetDims[2];
			offsetDims[0] = row;
			offsetDims[1] = 0;
			hsize_t selectionSize[2];
			selectionSize[0]  = 1;
			selectionSize[1]  = spdFile->getNumberBinsX();
			plsPerBinDSpace.selectHyperslab( H5S_SELECT_SET, selectionSize, offsetDims );
			binOffsetsDSpace.selectHyperslab( H5S_SELECT_SET, selectionSize, offsetDims );
			
			hsize_t memSpaceDims[1]; 
			memSpaceDims[0] = spdFile->getNumberBinsX();
			DataSpace memSpace( 1, memSpaceDims ); // has rank == 1
			
			hsize_t offsetMemSpace[1];
			hsize_t selectionMemSpace[1];
			offsetMemSpace[0] = 0;
			selectionMemSpace[0]  = spdFile->getNumberBinsX();
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
    
	/*UPDATED*/
	void SPDFileIncrementalReader::readPulseDataRow(boost::uint_fast32_t row, list<SPDPulse*> **pulses) throw(SPDIOException)
	{
		if(!fileOpened)
		{
			throw SPDIOException("Input file is not open..");
		}
        
        if(spdFile->getFileType() == SPD_UPD_TYPE)
        {
            throw SPDIOException("This function is only available for files with a spatial index.");
        }
        else if(spdFile->getFileType() == SPD_SEQ_TYPE)
        {
            try
            {
                Exception::dontPrint();
                SPDPointUtils ptsUtils;
                SPDPulseUtils pulseUtils;
                
                unsigned long *plsInBins = new unsigned long[spdFile->getNumberBinsX()];
                unsigned long long *offsets = new unsigned long long[spdFile->getNumberBinsX()];
                
                this->readRefHeaderRow(row, offsets, plsInBins);
                            
                boost::uint_fast64_t totalNumPulses = 0;
                for(boost::uint_fast32_t i = 0; i < spdFile->getNumberBinsX(); ++i)
                {
                    totalNumPulses += plsInBins[i];
                }
                            
                if(totalNumPulses > 0)
                {
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
                    pulseCount[0]  = totalNumPulses;
                    pulsesDataspace.selectHyperslab( H5S_SELECT_SET, pulseCount, pulseOffset );
                    
                    hsize_t pulseDims[1]; 
                    pulseDims[0] = totalNumPulses;
                    DataSpace pulseMemspace( rank, pulseDims );
                    
                    hsize_t pulseOffset_out[1];
                    hsize_t pulseCount_out[1];
                    pulseOffset_out[0] = 0;
                    pulseCount_out[0]  = totalNumPulses;
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
                    unsigned long *transmittedArray = NULL;
                    unsigned long *receivedArray = NULL;
                    
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
                    
                    pulseOffset[0] = offsets[0];
                    pulseCount[0]  = totalNumPulses;
                    pulsesDataspace.selectHyperslab( H5S_SELECT_SET, pulseCount, pulseOffset );
                    
                    pulseDims[0] = totalNumPulses;
                    pulseMemspace = DataSpace( rank, pulseDims );
                    
                    pulseOffset_out[0] = 0;
                    pulseCount_out[0]  = totalNumPulses;
                    pulseMemspace.selectHyperslab( H5S_SELECT_SET, pulseCount_out, pulseOffset_out );
                                        
                    if(spdFile->getPulseVersion() == 1)
                    {
                        pulseArray = new SPDPulseH5V1[totalNumPulses];
                    }
                    else if(spdFile->getPulseVersion() == 2)
                    {
                        pulseArray = new SPDPulseH5V2[totalNumPulses];
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
                        for( boost::uint_fast32_t j = 0; j < totalNumPulses; ++j)
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
                        for( boost::uint_fast32_t j = 0; j < totalNumPulses; ++j)
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
                    
                    boost::uint_fast32_t start = 0;
                    boost::uint_fast32_t end = 0;
                    
                    for(boost::uint_fast32_t i = 0; i < spdFile->getNumberBinsX(); ++i)
                    {
                        start = end;
                        end += plsInBins[i];
                        
                        for( boost::uint_fast32_t j = start; j < end; ++j)
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
                                pulse->pts->reserve(pulse->numberOfReturns);
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

                            pulses[i]->push_back(pulse);
                            
                        }
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
                
                delete[] plsInBins;
                delete[] offsets;
                
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
        else if(spdFile->getFileType() == SPD_NONSEQ_TYPE)
        {
            try
            {
                SPDPointUtils ptsUtils;
                SPDPulseUtils pulseUtils;
                
                unsigned long *plsInBins = new unsigned long[spdFile->getNumberBinsX()];
                unsigned long long *offsets = new unsigned long long[spdFile->getNumberBinsX()];
                
                this->readRefHeaderRow(row, offsets, plsInBins);
                
                for(boost::uint_fast64_t col = 0; col < spdFile->getNumberBinsX(); ++col)
                {
                    this->readPulseData(pulses[col], offsets[col], plsInBins[col]);
                }
                
                delete[] plsInBins;
                delete[] offsets;
                
            }
            catch(SPDIOException &e)
            {
                throw e;
            }
        }
        else
        {
            throw SPDIOException("SPD File type was not recognised.");
        }
	}
	
    /*UPDATED*/
	void SPDFileIncrementalReader::readPulseDataRow(boost::uint_fast32_t row, vector<SPDPulse*> **pulses) throw(SPDIOException)
	{
		if(!fileOpened)
		{
			throw SPDIOException("Input file is not open..");
		}
        
        if(spdFile->getFileType() == SPD_UPD_TYPE)
        {
            throw SPDIOException("This function is only available for files with a spatial index.");
        }
        else if(spdFile->getFileType() == SPD_SEQ_TYPE)
        {
            try
            {
                Exception::dontPrint();
                SPDPointUtils ptsUtils;
                SPDPulseUtils pulseUtils;
                
                unsigned long *plsInBins = new unsigned long[spdFile->getNumberBinsX()];
                unsigned long long *offsets = new unsigned long long[spdFile->getNumberBinsX()];
                
                this->readRefHeaderRow(row, offsets, plsInBins);
                
                boost::uint_fast64_t totalNumPulses = 0;
                for(boost::uint_fast32_t i = 0; i < spdFile->getNumberBinsX(); ++i)
                {
                    totalNumPulses += plsInBins[i];
                }
                
                if(totalNumPulses > 0)
                {
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
                    pulseCount[0]  = totalNumPulses;
                    pulsesDataspace.selectHyperslab( H5S_SELECT_SET, pulseCount, pulseOffset );
                    
                    hsize_t pulseDims[1]; 
                    pulseDims[0] = totalNumPulses;
                    DataSpace pulseMemspace( rank, pulseDims );
                    
                    hsize_t pulseOffset_out[1];
                    hsize_t pulseCount_out[1];
                    pulseOffset_out[0] = 0;
                    pulseCount_out[0]  = totalNumPulses;
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
                    unsigned long *transmittedArray = NULL;
                    unsigned long *receivedArray = NULL;
                    
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
                    
                    pulseOffset[0] = offsets[0];
                    pulseCount[0]  = totalNumPulses;
                    pulsesDataspace.selectHyperslab( H5S_SELECT_SET, pulseCount, pulseOffset );
                    
                    pulseDims[0] = totalNumPulses;
                    pulseMemspace = DataSpace( rank, pulseDims );
                    
                    pulseOffset_out[0] = 0;
                    pulseCount_out[0]  = totalNumPulses;
                    pulseMemspace.selectHyperslab( H5S_SELECT_SET, pulseCount_out, pulseOffset_out );
                    
                    if(spdFile->getPulseVersion() == 1)
                    {
                        pulseArray = new SPDPulseH5V1[totalNumPulses];
                    }
                    else if(spdFile->getPulseVersion() == 2)
                    {
                        pulseArray = new SPDPulseH5V2[totalNumPulses];
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
                        for( boost::uint_fast32_t j = 0; j < totalNumPulses; ++j)
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
                        for( boost::uint_fast32_t j = 0; j < totalNumPulses; ++j)
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
                    
                    boost::uint_fast32_t start = 0;
                    boost::uint_fast32_t end = 0;
                    
                    for(boost::uint_fast32_t i = 0; i < spdFile->getNumberBinsX(); ++i)
                    {
                        start = end;
                        end += plsInBins[i];
                        
                        for( boost::uint_fast32_t j = start; j < end; ++j)
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
                                pulse->pts->reserve(pulse->numberOfReturns);
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
                            
                            pulses[i]->push_back(pulse);
                            
                        }
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
                
                delete[] plsInBins;
                delete[] offsets;
                
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
        else if(spdFile->getFileType() == SPD_NONSEQ_TYPE)
        {
            try
            {
                SPDPointUtils ptsUtils;
                SPDPulseUtils pulseUtils;
                
                unsigned long *plsInBins = new unsigned long[spdFile->getNumberBinsX()];
                unsigned long long *offsets = new unsigned long long[spdFile->getNumberBinsX()];
                
                this->readRefHeaderRow(row, offsets, plsInBins);
                
                for(boost::uint_fast64_t col = 0; col < spdFile->getNumberBinsX(); ++col)
                {
                    this->readPulseData(pulses[col], offsets[col], plsInBins[col]);
                }
                
                delete[] plsInBins;
                delete[] offsets;
                
            }
            catch(SPDIOException &e)
            {
                throw e;
            }
        }
        else
        {
            throw SPDIOException("SPD File type was not recognised.");
        }
	}
	
    /*UPDATED*/
	void SPDFileIncrementalReader::readPulseData(list<SPDPulse*> *pulses, boost::uint_fast32_t row, boost::uint_fast32_t startCol, boost::uint_fast32_t endCol) throw(SPDIOException)
	{
		if(!fileOpened)
		{
			throw SPDIOException("Input file is not open..");
		}
		
		if(startCol > endCol)
		{
			throw SPDIOException("The starting column should be before the ending column.");
		}
		
		if(startCol > spdFile->getNumberBinsX())
		{
			throw SPDIOException("The starting column less than the number of columns within the SPDFile.");
		}
		
		if(endCol > spdFile->getNumberBinsX())
		{
			throw SPDIOException("The starting column less than the number of columns within the SPDFile.");
		}
		
        if(spdFile->getFileType() == SPD_UPD_TYPE)
        {
            throw SPDIOException("This function is only available for files with a spatial index.");
        }
        else if(spdFile->getFileType() == SPD_SEQ_TYPE)
        {
            try
            {
                unsigned long *plsInBins = new unsigned long[spdFile->getNumberBinsX()];
                unsigned long long *offsets = new unsigned long long[spdFile->getNumberBinsX()];
                
                this->readRefHeaderRow(row, offsets, plsInBins);
                
                boost::uint_fast64_t offset = offsets[startCol];
                boost::uint_fast64_t numPts = 0;
                
                for(boost::uint_fast32_t i = startCol; i < endCol; ++i)
                {
                    numPts = numPts + plsInBins[i];
                }
                
                this->readPulseData(pulses, offset, numPts);
                
                delete[] plsInBins;
                delete[] offsets;
            }
            catch(SPDIOException &e)
            {
                throw e;
            }
        }
        else if(spdFile->getFileType() == SPD_NONSEQ_TYPE)
        {
            try
            {
                SPDPointUtils ptsUtils;
                SPDPulseUtils pulseUtils;
                
                unsigned long *plsInBins = new unsigned long[spdFile->getNumberBinsX()];
                unsigned long long *offsets = new unsigned long long[spdFile->getNumberBinsX()];
                
                this->readRefHeaderRow(row, offsets, plsInBins);
                
                for(boost::uint_fast64_t col = startCol; col < endCol; ++col)
                {
                    this->readPulseData(pulses, offsets[col], plsInBins[col]);
                }
                
                delete[] plsInBins;
                delete[] offsets;
                
            }
            catch(SPDIOException &e)
            {
                throw e;
            }
        }
        else
        {
            throw SPDIOException("SPD File type was not recognised.");
        }
	}
	
    /*UPDATED*/
	void SPDFileIncrementalReader::readPulseData(vector<SPDPulse*> *pulses, boost::uint_fast32_t row, boost::uint_fast32_t startCol, boost::uint_fast32_t endCol) throw(SPDIOException)
	{
		if(!fileOpened)
		{
			throw SPDIOException("Input file is not open..");
		}
		
		if(startCol > endCol)
		{
			throw SPDIOException("The starting column should be before the ending column.");
		}
		
		if(startCol > spdFile->getNumberBinsX())
		{
			throw SPDIOException("The starting column less than the number of columns within the SPDFile.");
		}
		
		if(endCol > spdFile->getNumberBinsX())
		{
			throw SPDIOException("The starting column less than the number of columns within the SPDFile.");
		}
		
        if(spdFile->getFileType() == SPD_UPD_TYPE)
        {
            throw SPDIOException("This function is only available for files with a spatial index.");
        }
        else if(spdFile->getFileType() == SPD_SEQ_TYPE)
        {
            try
            {
                unsigned long *plsInBins = new unsigned long[spdFile->getNumberBinsX()];
                unsigned long long *offsets = new unsigned long long[spdFile->getNumberBinsX()];
                
                this->readRefHeaderRow(row, offsets, plsInBins);
                
                boost::uint_fast64_t offset = offsets[startCol];
                boost::uint_fast64_t numPts = 0;
                
                for(boost::uint_fast32_t i = startCol; i < endCol; ++i)
                {
                    numPts = numPts + plsInBins[i];
                }
                
                this->readPulseData(pulses, offset, numPts);
                
                delete[] plsInBins;
                delete[] offsets;
            }
            catch(SPDIOException &e)
            {
                throw e;
            }
        }
        else if(spdFile->getFileType() == SPD_NONSEQ_TYPE)
        {
            try
            {
                SPDPointUtils ptsUtils;
                SPDPulseUtils pulseUtils;
                
                unsigned long *plsInBins = new unsigned long[spdFile->getNumberBinsX()];
                unsigned long long *offsets = new unsigned long long[spdFile->getNumberBinsX()];
                
                this->readRefHeaderRow(row, offsets, plsInBins);
                
                for(boost::uint_fast64_t col = startCol; col < endCol; ++col)
                {
                    this->readPulseData(pulses, offsets[col], plsInBins[col]);
                }
                
                delete[] plsInBins;
                delete[] offsets;
                
            }
            catch(SPDIOException &e)
            {
                throw e;
            }
        }
        else
        {
            throw SPDIOException("SPD File type was not recognised.");
        }
	}
	
    /*UPDATED*/
	void SPDFileIncrementalReader::readPulseDataBlock(list<SPDPulse*> ***pulses, boost::uint_fast32_t *bbox) throw(SPDIOException)
	{
		if(!fileOpened)
		{
			throw SPDIOException("Input file is not open..");
		}
        
        boost::uint_fast32_t startRow = bbox[1];
        boost::uint_fast32_t endRow = bbox[3];
        boost::uint_fast32_t startCol = bbox[0];
        boost::uint_fast32_t endCol = bbox[2];
        
        boost::uint_fast32_t numCols = endCol - startCol;
        
        if(spdFile->getFileType() == SPD_UPD_TYPE)
        {
            throw SPDIOException("This function is only available for files with a spatial index.");
        }
        else if(spdFile->getFileType() == SPD_SEQ_TYPE)
        {
            try
            {
                // Read PTS..
                Exception::dontPrint();
                
                SPDPointUtils ptsUtils;
                SPDPulseUtils pulseUtils;
                
                unsigned long *plsInBins = new unsigned long[spdFile->getNumberBinsX()];
                unsigned long long *offsets = new unsigned long long[spdFile->getNumberBinsX()];
                
                boost::uint_fast64_t totalNumPulses = 0;
                
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
                hsize_t pulseCount[1];
                
                hsize_t pulseDims[1]; 
                DataSpace pulseMemspace;
                
                hsize_t pulseOffset_out[1];
                hsize_t pulseCount_out[1];
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
                unsigned long *transmittedArray = NULL;
                unsigned long *receivedArray = NULL;
                
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
                
                boost::uint_fast32_t rowIdx = 0;
                
                for(boost::uint_fast32_t row = startRow; row < endRow; ++row)
                {
                    this->readRefHeaderRow(row, offsets, plsInBins);
                    
                    totalNumPulses = 0;
                    for(boost::uint_fast32_t i = startCol; i < endCol; ++i)
                    {
                        totalNumPulses += plsInBins[i];
                    }
                                    
                    if(totalNumPulses > 0)
                    {
                        pulseOffset[0] = offsets[startCol];
                        pulseCount[0]  = totalNumPulses;
                        pulsesDataspace.selectHyperslab( H5S_SELECT_SET, pulseCount, pulseOffset );
                        
                        pulseDims[0] = totalNumPulses;
                        pulseMemspace = DataSpace( rank, pulseDims );
                        
                        pulseOffset_out[0] = 0;
                        pulseCount_out[0]  = totalNumPulses;
                        pulseMemspace.selectHyperslab( H5S_SELECT_SET, pulseCount_out, pulseOffset_out );
                        
                        if(spdFile->getPulseVersion() == 1)
                        {
                            pulseArray = new SPDPulseH5V1[totalNumPulses];
                        }
                        else if(spdFile->getPulseVersion() == 2)
                        {
                            pulseArray = new SPDPulseH5V2[totalNumPulses];
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
                            for( boost::uint_fast32_t j = 0; j < totalNumPulses; ++j)
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
                            for( boost::uint_fast32_t j = 0; j < totalNumPulses; ++j)
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
                        
                        boost::uint_fast32_t start = 0;
                        boost::uint_fast32_t end = 0;
                        
                        for(boost::uint_fast32_t colIdx = 0; colIdx < numCols; ++colIdx)
                        {
                            start = end;
                            end += plsInBins[(startCol+colIdx)];
                        
                            for( boost::uint_fast32_t j = start; j < end; ++j)
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
                                    pulse->pts->reserve(pulse->numberOfReturns);
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

                                pulses[rowIdx][colIdx]->push_back(pulse);
                            }
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
                    
                    ++rowIdx;
                }		
                delete[] plsInBins;
                delete[] offsets;
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
        else if(spdFile->getFileType() == SPD_NONSEQ_TYPE)
        {
            try
            {
                SPDPointUtils ptsUtils;
                SPDPulseUtils pulseUtils;
                
                unsigned long *plsInBins = new unsigned long[spdFile->getNumberBinsX()];
                unsigned long long *offsets = new unsigned long long[spdFile->getNumberBinsX()];
                
                boost::uint_fast64_t i = 0;
                boost::uint_fast64_t j = 0;
                for(boost::uint_fast64_t row = startRow; row < endRow; ++row)
                {
                    this->readRefHeaderRow(row, offsets, plsInBins);
                    
                    for(boost::uint_fast64_t col = startCol; col < endCol; ++col)
                    {
                        this->readPulseData(pulses[i][j], offsets[col], plsInBins[col]);
                        ++j;
                    }
                    j = 0;
                    ++i;
                }
                
                delete[] plsInBins;
                delete[] offsets;
                
            }
            catch(SPDIOException &e)
            {
                throw e;
            }
        }
        else
        {
            throw SPDIOException("SPD File type was not recognised.");
        }
	}
	
    /*UPDATED*/
	void SPDFileIncrementalReader::readPulseDataBlock(vector<SPDPulse*> ***pulses, boost::uint_fast32_t *bbox) throw(SPDIOException)
	{
		if(!fileOpened)
		{
			throw SPDIOException("Input file is not open..");
		}
        
        boost::uint_fast32_t startRow = bbox[1];
        boost::uint_fast32_t endRow = bbox[3];
        boost::uint_fast32_t startCol = bbox[0];
        boost::uint_fast32_t endCol = bbox[2];
        
        boost::uint_fast32_t numCols = endCol - startCol;
        
        if(spdFile->getFileType() == SPD_UPD_TYPE)
        {
            throw SPDIOException("This function is only available for files with a spatial index.");
        }
        else if(spdFile->getFileType() == SPD_SEQ_TYPE)
        {
            try
            {
                // Read PTS..
                Exception::dontPrint();
                
                SPDPointUtils ptsUtils;
                SPDPulseUtils pulseUtils;
                
                unsigned long *plsInBins = new unsigned long[spdFile->getNumberBinsX()];
                unsigned long long *offsets = new unsigned long long[spdFile->getNumberBinsX()];
                
                boost::uint_fast64_t totalNumPulses = 0;
                
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
                hsize_t pulseCount[1];
                
                hsize_t pulseDims[1]; 
                DataSpace pulseMemspace;
                
                hsize_t pulseOffset_out[1];
                hsize_t pulseCount_out[1];
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
                unsigned long *transmittedArray = NULL;
                unsigned long *receivedArray = NULL;
                
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
                
                boost::uint_fast32_t rowIdx = 0;
                
                for(boost::uint_fast32_t row = startRow; row < endRow; ++row)
                {
                    this->readRefHeaderRow(row, offsets, plsInBins);
                    
                    totalNumPulses = 0;
                    for(boost::uint_fast32_t i = startCol; i < endCol; ++i)
                    {
                        totalNumPulses += plsInBins[i];
                    }
                    
                    if(totalNumPulses > 0)
                    {
                        pulseOffset[0] = offsets[startCol];
                        pulseCount[0]  = totalNumPulses;
                        pulsesDataspace.selectHyperslab( H5S_SELECT_SET, pulseCount, pulseOffset );
                        
                        pulseDims[0] = totalNumPulses;
                        pulseMemspace = DataSpace( rank, pulseDims );
                        
                        pulseOffset_out[0] = 0;
                        pulseCount_out[0]  = totalNumPulses;
                        pulseMemspace.selectHyperslab( H5S_SELECT_SET, pulseCount_out, pulseOffset_out );
                        
                        if(spdFile->getPulseVersion() == 1)
                        {
                            pulseArray = new SPDPulseH5V1[totalNumPulses];
                        }
                        else if(spdFile->getPulseVersion() == 2)
                        {
                            pulseArray = new SPDPulseH5V2[totalNumPulses];
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
                            for( boost::uint_fast32_t j = 0; j < totalNumPulses; ++j)
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
                            for( boost::uint_fast32_t j = 0; j < totalNumPulses; ++j)
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
                        
                        boost::uint_fast32_t start = 0;
                        boost::uint_fast32_t end = 0;
                        
                        for(boost::uint_fast32_t colIdx = 0; colIdx < numCols; ++colIdx)
                        {
                            start = end;
                            end += plsInBins[(startCol+colIdx)];
                            
                            for( boost::uint_fast32_t j = start; j < end; ++j)
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
                                    pulse->pts->reserve(pulse->numberOfReturns);
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
                                
                                pulses[rowIdx][colIdx]->push_back(pulse);
                            }
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
                    
                    ++rowIdx;
                }		
                delete[] plsInBins;
                delete[] offsets;
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
        else if(spdFile->getFileType() == SPD_NONSEQ_TYPE)
        {
            try
            {
                SPDPointUtils ptsUtils;
                SPDPulseUtils pulseUtils;
                
                unsigned long *plsInBins = new unsigned long[spdFile->getNumberBinsX()];
                unsigned long long *offsets = new unsigned long long[spdFile->getNumberBinsX()];
                
                boost::uint_fast64_t i = 0;
                boost::uint_fast64_t j = 0;
                for(boost::uint_fast64_t row = startRow; row < endRow; ++row)
                {
                    this->readRefHeaderRow(row, offsets, plsInBins);
                    
                    for(boost::uint_fast64_t col = startCol; col < endCol; ++col)
                    {
                        this->readPulseData(pulses[i][j], offsets[col], plsInBins[col]);
                        ++j;
                    }
                    j = 0;
                    ++i;
                }
                
                delete[] plsInBins;
                delete[] offsets;
                
            }
            catch(SPDIOException &e)
            {
                throw e;
            }
        }
        else
        {
            throw SPDIOException("SPD File type was not recognised.");
        }
	}
    
    /*UPDATED*/
	void SPDFileIncrementalReader::readPulseDataBlock(list<SPDPulse*> ***pulses, boost::uint_fast32_t *bbox, boost::uint_fast32_t xOff, boost::uint_fast32_t yOff) throw(SPDIOException)
	{
		if(!fileOpened)
		{
			throw SPDIOException("Input file is not open..");
		}
        
        boost::uint_fast32_t startRow = bbox[1];
        boost::uint_fast32_t endRow = bbox[3];
        boost::uint_fast32_t startCol = bbox[0];
        boost::uint_fast32_t endCol = bbox[2];
        
        boost::uint_fast32_t numCols = endCol - startCol;
        
        if(spdFile->getFileType() == SPD_UPD_TYPE)
        {
            throw SPDIOException("This function is only available for files with a spatial index.");
        }
        else if(spdFile->getFileType() == SPD_SEQ_TYPE)
        {
            try
            {
                // Read PTS..
                Exception::dontPrint();
                
                SPDPointUtils ptsUtils;
                SPDPulseUtils pulseUtils;
                
                unsigned long *plsInBins = new unsigned long[spdFile->getNumberBinsX()];
                unsigned long long *offsets = new unsigned long long[spdFile->getNumberBinsX()];
                
                boost::uint_fast64_t totalNumPulses = 0;
                
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
                hsize_t pulseCount[1];
                
                hsize_t pulseDims[1]; 
                DataSpace pulseMemspace;
                
                hsize_t pulseOffset_out[1];
                hsize_t pulseCount_out[1];
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
                unsigned long *transmittedArray = NULL;
                unsigned long *receivedArray = NULL;
                
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
                
                boost::uint_fast32_t rowIdx = 0;
                
                for(boost::uint_fast32_t row = startRow; row < endRow; ++row)
                {
                    this->readRefHeaderRow(row, offsets, plsInBins);
                    
                    totalNumPulses = 0;
                    for(boost::uint_fast32_t i = startCol; i < endCol; ++i)
                    {
                        totalNumPulses += plsInBins[i];
                    }
                    
                    if(totalNumPulses > 0)
                    {
                        pulseOffset[0] = offsets[startCol];
                        pulseCount[0]  = totalNumPulses;
                        pulsesDataspace.selectHyperslab( H5S_SELECT_SET, pulseCount, pulseOffset );
                        
                        pulseDims[0] = totalNumPulses;
                        pulseMemspace = DataSpace( rank, pulseDims );
                        
                        pulseOffset_out[0] = 0;
                        pulseCount_out[0]  = totalNumPulses;
                        pulseMemspace.selectHyperslab( H5S_SELECT_SET, pulseCount_out, pulseOffset_out );
                        
                        if(spdFile->getPulseVersion() == 1)
                        {
                            pulseArray = new SPDPulseH5V1[totalNumPulses];
                        }
                        else if(spdFile->getPulseVersion() == 2)
                        {
                            pulseArray = new SPDPulseH5V2[totalNumPulses];
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
                            for( boost::uint_fast32_t j = 0; j < totalNumPulses; ++j)
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
                            for( boost::uint_fast32_t j = 0; j < totalNumPulses; ++j)
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
                        
                        boost::uint_fast32_t start = 0;
                        boost::uint_fast32_t end = 0;
                        
                        for(boost::uint_fast32_t colIdx = 0; colIdx < numCols; ++colIdx)
                        {
                            start = end;
                            end += plsInBins[(startCol+colIdx)];
                            
                            for( boost::uint_fast32_t j = start; j < end; ++j)
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
                                    pulse->pts->reserve(pulse->numberOfReturns);
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
                                
                                pulses[rowIdx+yOff][colIdx+xOff]->push_back(pulse);
                            }
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
                    
                    ++rowIdx;
                }		
                delete[] plsInBins;
                delete[] offsets;
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
        else if(spdFile->getFileType() == SPD_NONSEQ_TYPE)
        {
            try
            {
                SPDPointUtils ptsUtils;
                SPDPulseUtils pulseUtils;
                
                unsigned long *plsInBins = new unsigned long[spdFile->getNumberBinsX()];
                unsigned long long *offsets = new unsigned long long[spdFile->getNumberBinsX()];
                
                boost::uint_fast64_t i = yOff;
                boost::uint_fast64_t j = xOff;
                for(boost::uint_fast64_t row = startRow; row < endRow; ++row)
                {
                    this->readRefHeaderRow(row, offsets, plsInBins);
                    
                    for(boost::uint_fast64_t col = startCol; col < endCol; ++col)
                    {
                        this->readPulseData(pulses[i][j], offsets[col], plsInBins[col]);
                        ++j;
                    }
                    j = 0;
                    ++i;
                }
                
                delete[] plsInBins;
                delete[] offsets;
                
            }
            catch(SPDIOException &e)
            {
                throw e;
            }
        }
        else
        {
            throw SPDIOException("SPD File type was not recognised.");
        }
	}
	
    /*UPDATED*/
	void SPDFileIncrementalReader::readPulseDataBlock(vector<SPDPulse*> ***pulses, boost::uint_fast32_t *bbox, boost::uint_fast32_t xOff, boost::uint_fast32_t yOff) throw(SPDIOException)
	{
		if(!fileOpened)
		{
			throw SPDIOException("Input file is not open..");
		}
        
        boost::uint_fast32_t startRow = bbox[1];
        boost::uint_fast32_t endRow = bbox[3];
        boost::uint_fast32_t startCol = bbox[0];
        boost::uint_fast32_t endCol = bbox[2];
        
        boost::uint_fast32_t numCols = endCol - startCol;
        
        //cout << "Rows: " << startRow << " - " << endRow << " = " << endRow - startRow << endl;
        //cout << "Cols: " << startCol << " - " << endCol << " = " << endCol - startCol << endl;
        
        if(spdFile->getFileType() == SPD_UPD_TYPE)
        {
            throw SPDIOException("This function is only available for files with a spatial index.");
        }
        else if(spdFile->getFileType() == SPD_SEQ_TYPE)
        {
            try
            {
                // Read PTS..
                Exception::dontPrint();
                
                SPDPointUtils ptsUtils;
                SPDPulseUtils pulseUtils;
                
                unsigned long *plsInBins = new unsigned long[spdFile->getNumberBinsX()];
                unsigned long long *offsets = new unsigned long long[spdFile->getNumberBinsX()];
                
                boost::uint_fast64_t totalNumPulses = 0;
                
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
                hsize_t pulseCount[1];
                
                hsize_t pulseDims[1]; 
                DataSpace pulseMemspace;
                
                hsize_t pulseOffset_out[1];
                hsize_t pulseCount_out[1];
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
                unsigned long *transmittedArray = NULL;
                unsigned long *receivedArray = NULL;
                
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
                
                boost::uint_fast32_t rowIdx = 0;
                
                for(boost::uint_fast32_t row = startRow; row < endRow; ++row)
                {
                    this->readRefHeaderRow(row, offsets, plsInBins);
                    
                    totalNumPulses = 0;
                    for(boost::uint_fast32_t i = startCol; i < endCol; ++i)
                    {
                        totalNumPulses += plsInBins[i];
                    }
                    
                    if(totalNumPulses > 0)
                    {
                        pulseOffset[0] = offsets[startCol];
                        pulseCount[0]  = totalNumPulses;
                        pulsesDataspace.selectHyperslab( H5S_SELECT_SET, pulseCount, pulseOffset );
                        
                        pulseDims[0] = totalNumPulses;
                        pulseMemspace = DataSpace( rank, pulseDims );
                        
                        pulseOffset_out[0] = 0;
                        pulseCount_out[0]  = totalNumPulses;
                        pulseMemspace.selectHyperslab( H5S_SELECT_SET, pulseCount_out, pulseOffset_out );
                        
                        if(spdFile->getPulseVersion() == 1)
                        {
                            pulseArray = new SPDPulseH5V1[totalNumPulses];
                        }
                        else if(spdFile->getPulseVersion() == 2)
                        {
                            pulseArray = new SPDPulseH5V2[totalNumPulses];
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
                            for( boost::uint_fast32_t j = 0; j < totalNumPulses; ++j)
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
                            for( boost::uint_fast32_t j = 0; j < totalNumPulses; ++j)
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
                        
                        boost::uint_fast32_t start = 0;
                        boost::uint_fast32_t end = 0;
                        
                        for(boost::uint_fast32_t colIdx = 0; colIdx < numCols; ++colIdx)
                        {
                            start = end;
                            end += plsInBins[(startCol+colIdx)];
                            
                            for( boost::uint_fast32_t j = start; j < end; ++j)
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
                                    pulse->pts->reserve(pulse->numberOfReturns);
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
                                
                                pulses[rowIdx+yOff][colIdx+xOff]->push_back(pulse);
                            }
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
                    
                    ++rowIdx;
                }		
                delete[] plsInBins;
                delete[] offsets;
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
        else if(spdFile->getFileType() == SPD_NONSEQ_TYPE)
        {
            try
            {
                SPDPointUtils ptsUtils;
                SPDPulseUtils pulseUtils;
                
                unsigned long *plsInBins = new unsigned long[spdFile->getNumberBinsX()];
                unsigned long long *offsets = new unsigned long long[spdFile->getNumberBinsX()];
                
                boost::uint_fast64_t i = yOff;
                boost::uint_fast64_t j = xOff;
                for(boost::uint_fast64_t row = startRow; row < endRow; ++row)
                {
                    this->readRefHeaderRow(row, offsets, plsInBins);
                    
                    for(boost::uint_fast64_t col = startCol; col < endCol; ++col)
                    {
                        this->readPulseData(pulses[i][j], offsets[col], plsInBins[col]);
                        ++j;
                    }
                    j = 0;
                    ++i;
                }
                
                delete[] plsInBins;
                delete[] offsets;
                
            }
            catch(SPDIOException &e)
            {
                throw e;
            }
        }
        else
        {
            throw SPDIOException("SPD File type was not recognised.");
        }
	}
	
	void SPDFileIncrementalReader::readPulseDataBlock(list<SPDPulse*> *pulses, boost::uint_fast32_t *bbox) throw(SPDIOException)
	{
		if(!fileOpened)
		{
			throw SPDIOException("Input file is not open..");
		}
		
        boost::uint_fast32_t startRow = bbox[1];
        boost::uint_fast32_t endRow = bbox[3];
        boost::uint_fast32_t startCol = bbox[0];
        boost::uint_fast32_t endCol = bbox[2];
        
        if(spdFile->getFileType() == SPD_UPD_TYPE)
        {
            throw SPDIOException("This function is only available for files with a spatial index.");
        }
        else if(spdFile->getFileType() == SPD_SEQ_TYPE)
        {
            try
            {
                // Read PTS..
                Exception::dontPrint();
                
                H5File *spdInFile = NULL;
                spdInFile = new H5File( spdFile->getFilePath(), H5F_ACC_RDONLY );
                
                SPDPointUtils ptsUtils;
                SPDPulseUtils pulseUtils;
                
                unsigned long *plsInBins = new unsigned long[spdFile->getNumberBinsX()];
                unsigned long long *offsets = new unsigned long long[spdFile->getNumberBinsX()];
                
                boost::uint_fast64_t totalNumPulses = 0;
                
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
                hsize_t pulseCount[1];
                
                hsize_t pulseDims[1]; 
                DataSpace pulseMemspace;
                
                hsize_t pulseOffset_out[1];
                hsize_t pulseCount_out[1];
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
                unsigned long *transmittedArray = NULL;
                unsigned long *receivedArray = NULL;
                
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
                
                for(boost::uint_fast32_t row = startRow; row < endRow; ++row)
                {
                    this->readRefHeaderRow(row, offsets, plsInBins);
                    
                    totalNumPulses = 0;
                    for(boost::uint_fast32_t i = startCol; i < endCol; ++i)
                    {
                        totalNumPulses += plsInBins[i];
                    }
                    
                    if(totalNumPulses > 0)
                    {
                        pulseOffset[0] = 0;
                        pulseCount[0]  = totalNumPulses;
                        pulsesDataspace.selectHyperslab( H5S_SELECT_SET, pulseCount, pulseOffset );
                        pulseDims[0] = totalNumPulses;
                        pulseMemspace = DataSpace( rank, pulseDims );
                        pulseOffset_out[0] = 0;
                        pulseCount_out[0]  = totalNumPulses;
                        pulseMemspace.selectHyperslab( H5S_SELECT_SET, pulseCount_out, pulseOffset_out );
                        
                        if(spdFile->getPulseVersion() == 1)
                        {
                            pulseArray = new SPDPulseH5V1[totalNumPulses];
                        }
                        else if(spdFile->getPulseVersion() == 2)
                        {
                            pulseArray = new SPDPulseH5V2[totalNumPulses];
                        }
                        pulseOffset[0] = offsets[startCol];
                        pulseCount[0]  = totalNumPulses;
                        pulsesDataspace.selectHyperslab( H5S_SELECT_SET, pulseCount, pulseOffset );
                        
                        pulseDims[0] = totalNumPulses;
                        pulseMemspace = DataSpace( rank, pulseDims );
                        
                        pulseOffset_out[0] = 0;
                        pulseCount_out[0]  = totalNumPulses;
                        pulseMemspace.selectHyperslab( H5S_SELECT_SET, pulseCount_out, pulseOffset_out );
                        
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
                            for( boost::uint_fast32_t j = 0; j < totalNumPulses; ++j)
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
                            for( boost::uint_fast32_t j = 0; j < totalNumPulses; ++j)
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
                        
                        for( boost::uint_fast32_t j = 0; j < totalNumPulses; ++j)
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
                                pulse->pts->reserve(pulse->numberOfReturns);
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
                }		
                delete[] plsInBins;
                delete[] offsets;
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
        else if(spdFile->getFileType() == SPD_NONSEQ_TYPE)
        {
            try
            {
                SPDPointUtils ptsUtils;
                SPDPulseUtils pulseUtils;
                
                unsigned long *plsInBins = new unsigned long[spdFile->getNumberBinsX()];
                unsigned long long *offsets = new unsigned long long[spdFile->getNumberBinsX()];
                
                for(boost::uint_fast64_t row = startRow; row < endRow; ++row)
                {
                    this->readRefHeaderRow(row, offsets, plsInBins);
                    
                    for(boost::uint_fast64_t col = startCol; col < endCol; ++col)
                    {
                        this->readPulseData(pulses, offsets[col], plsInBins[col]);
                    }
                }
                
                delete[] plsInBins;
                delete[] offsets;
                
            }
            catch(SPDIOException &e)
            {
                throw e;
            }
        }
        else
        {
            throw SPDIOException("SPD File type was not recognised.");
        }
	}
	
	void SPDFileIncrementalReader::readPulseDataBlock(vector<SPDPulse*> *pulses, boost::uint_fast32_t *bbox) throw(SPDIOException)
	{
		if(!fileOpened)
		{
			throw SPDIOException("Input file is not open..");
		}
		
        boost::uint_fast32_t startRow = bbox[1];
        boost::uint_fast32_t endRow = bbox[3];
        boost::uint_fast32_t startCol = bbox[0];
        boost::uint_fast32_t endCol = bbox[2];
        
        if(spdFile->getFileType() == SPD_UPD_TYPE)
        {
            throw SPDIOException("This function is only available for files with a spatial index.");
        }
        else if(spdFile->getFileType() == SPD_SEQ_TYPE)
        {
            try
            {
                // Read PTS..
                Exception::dontPrint();
                
                H5File *spdInFile = NULL;
                spdInFile = new H5File( spdFile->getFilePath(), H5F_ACC_RDONLY );
                
                SPDPointUtils ptsUtils;
                SPDPulseUtils pulseUtils;
                
                unsigned long *plsInBins = new unsigned long[spdFile->getNumberBinsX()];
                unsigned long long *offsets = new unsigned long long[spdFile->getNumberBinsX()];
                
                boost::uint_fast64_t totalNumPulses = 0;
                
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
                hsize_t pulseCount[1];
                
                hsize_t pulseDims[1]; 
                DataSpace pulseMemspace;
                
                hsize_t pulseOffset_out[1];
                hsize_t pulseCount_out[1];
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
                unsigned long *transmittedArray = NULL;
                unsigned long *receivedArray = NULL;
                
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
                
                for(boost::uint_fast32_t row = startRow; row < endRow; ++row)
                {
                    this->readRefHeaderRow(row, offsets, plsInBins);
                    
                    totalNumPulses = 0;
                    for(boost::uint_fast32_t i = startCol; i < endCol; ++i)
                    {
                        totalNumPulses += plsInBins[i];
                    }
                    
                    if(totalNumPulses > 0)
                    {
                        pulseOffset[0] = 0;
                        pulseCount[0]  = totalNumPulses;
                        pulsesDataspace.selectHyperslab( H5S_SELECT_SET, pulseCount, pulseOffset );
                        pulseDims[0] = totalNumPulses;
                        pulseMemspace = DataSpace( rank, pulseDims );
                        pulseOffset_out[0] = 0;
                        pulseCount_out[0]  = totalNumPulses;
                        pulseMemspace.selectHyperslab( H5S_SELECT_SET, pulseCount_out, pulseOffset_out );
                        
                        if(spdFile->getPulseVersion() == 1)
                        {
                            pulseArray = new SPDPulseH5V1[totalNumPulses];
                        }
                        else if(spdFile->getPulseVersion() == 2)
                        {
                            pulseArray = new SPDPulseH5V2[totalNumPulses];
                        }
                        pulseOffset[0] = offsets[startCol];
                        pulseCount[0]  = totalNumPulses;
                        pulsesDataspace.selectHyperslab( H5S_SELECT_SET, pulseCount, pulseOffset );
                        
                        pulseDims[0] = totalNumPulses;
                        pulseMemspace = DataSpace( rank, pulseDims );
                        
                        pulseOffset_out[0] = 0;
                        pulseCount_out[0]  = totalNumPulses;
                        pulseMemspace.selectHyperslab( H5S_SELECT_SET, pulseCount_out, pulseOffset_out );
                        
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
                            for( boost::uint_fast32_t j = 0; j < totalNumPulses; ++j)
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
                            for( boost::uint_fast32_t j = 0; j < totalNumPulses; ++j)
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
                        
                        for( boost::uint_fast32_t j = 0; j < totalNumPulses; ++j)
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
                                pulse->pts->reserve(pulse->numberOfReturns);
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
                }		
                delete[] plsInBins;
                delete[] offsets;
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
        else if(spdFile->getFileType() == SPD_NONSEQ_TYPE)
        {
            try
            {
                SPDPointUtils ptsUtils;
                SPDPulseUtils pulseUtils;
                
                unsigned long *plsInBins = new unsigned long[spdFile->getNumberBinsX()];
                unsigned long long *offsets = new unsigned long long[spdFile->getNumberBinsX()];
                
                for(boost::uint_fast64_t row = startRow; row < endRow; ++row)
                {
                    this->readRefHeaderRow(row, offsets, plsInBins);
                    
                    for(boost::uint_fast64_t col = startCol; col < endCol; ++col)
                    {
                        this->readPulseData(pulses, offsets[col], plsInBins[col]);
                    }
                }
                
                delete[] plsInBins;
                delete[] offsets;
                
            }
            catch(SPDIOException &e)
            {
                throw e;
            }
        }
        else
        {
            throw SPDIOException("SPD File type was not recognised.");
        }
	}
	
    /*UPDATED*/
	void SPDFileIncrementalReader::readPulseDataInGeoEnv(list<SPDPulse*> *pulses, OGREnvelope *env) throw(SPDIOException)
	{
		if(!fileOpened)
		{
			throw SPDIOException("Input file is not open..");
		}
        
        if(spdFile->getFileType() == SPD_UPD_TYPE)
        {
            throw SPDIOException("This function is only available for files with a spatial index.");
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
            tmpDist = env->MinX - spdFile->getXMin();
            if(tmpDist < 0)
            {
                startCol = 0;
            }
            else
            {
                numCols = numeric_cast<boost::uint_fast32_t>(tmpDist/spdFile->getBinSize());
                startCol = numCols;
            }
            
            // Define End Column
            tmpDist = spdFile->getXMax() - env->MaxX;
            if(tmpDist < 0)
            {
                endCol = spdFile->getNumberBinsX();
            }
            else
            {
                numCols = numeric_cast<boost::uint_fast32_t>(tmpDist/spdFile->getBinSize());
                endCol = spdFile->getNumberBinsX() - numCols;
            }
            
            // Define Starting Row
            tmpDist = spdFile->getYMax() - env->MaxY;
            if(tmpDist < 0)
            {
                startRow = 0;
            }
            else
            {
                numRows = numeric_cast<boost::uint_fast32_t>(tmpDist/spdFile->getBinSize());
                startRow = numRows;
            }
            
            // Define End Row
            tmpDist = env->MinY - spdFile->getYMin();
            if(tmpDist < 0)
            {
                endRow = spdFile->getNumberBinsY();
            }
            else
            {
                numRows = numeric_cast<boost::uint_fast32_t>(tmpDist/spdFile->getBinSize());
                endRow = spdFile->getNumberBinsY() - numRows;
            }
        }
        catch(negative_overflow& e) 
        {
            throw SPDIOException(e.what());
        }
        catch(positive_overflow& e) 
        {
            throw SPDIOException(e.what());
        }
        catch(bad_numeric_cast& e) 
        {
            throw SPDIOException(e.what());
        }
        
        if(endCol <= startCol)
        {
            throw SPDIOException("Define subset is not within the input file (X Axis).");
        }
        
        if(endRow <= startRow)
        {
            throw SPDIOException("Define subset is not within the input file (Y Axis).");
        }
   
        try
        {
            for(boost::uint_fast32_t rows = startRow; rows < endRow; ++rows)
            {
                this->readPulseData(pulses, rows, startCol, endCol);
            }
        }
        catch(SPDIOException &e)
        {
            throw e;
        }
	}
	
    /*UPDATED*/
	void SPDFileIncrementalReader::readPulseDataInGeoEnv(vector<SPDPulse*> *pulses, OGREnvelope *env) throw(SPDIOException)
	{
		if(!fileOpened)
		{
			throw SPDIOException("Input file is not open..");
		}
        
        if(spdFile->getFileType() == SPD_UPD_TYPE)
        {
            throw SPDIOException("This function is only available for files with a spatial index.");
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
            tmpDist = env->MinX - spdFile->getXMin();
            if(tmpDist < 0)
            {
                startCol = 0;
            }
            else
            {
                numCols = numeric_cast<boost::uint_fast32_t>(tmpDist/spdFile->getBinSize());
                startCol = numCols;
            }
            
            // Define End Column
            tmpDist = spdFile->getXMax() - env->MaxX;
            if(tmpDist < 0)
            {
                endCol = spdFile->getNumberBinsX();
            }
            else
            {
                numCols = numeric_cast<boost::uint_fast32_t>(tmpDist/spdFile->getBinSize());
                endCol = spdFile->getNumberBinsX() - numCols;
            }
            
            // Define Starting Row
            tmpDist = spdFile->getYMax() - env->MaxY;
            if(tmpDist < 0)
            {
                startRow = 0;
            }
            else
            {
                numRows = numeric_cast<boost::uint_fast32_t>(tmpDist/spdFile->getBinSize());
                startRow = numRows;
            }
            
            // Define End Row
            tmpDist = env->MinY - spdFile->getYMin();
            if(tmpDist < 0)
            {
                endRow = spdFile->getNumberBinsY();
            }
            else
            {
                numRows = numeric_cast<boost::uint_fast32_t>(tmpDist/spdFile->getBinSize());
                endRow = spdFile->getNumberBinsY() - numRows;
            }
        }
        catch(negative_overflow& e) 
        {
            throw SPDIOException(e.what());
        }
        catch(positive_overflow& e) 
        {
            throw SPDIOException(e.what());
        }
        catch(bad_numeric_cast& e) 
        {
            throw SPDIOException(e.what());
        }
        
        if(endCol <= startCol)
        {
            throw SPDIOException("Define subset is not within the input file (X Axis).");
        }
        
        if(endRow <= startRow)
        {
            throw SPDIOException("Define subset is not within the input file (Y Axis).");
        }
        
        try
        {
            for(boost::uint_fast32_t rows = startRow; rows < endRow; ++rows)
            {
                this->readPulseData(pulses, rows, startCol, endCol);
            }
        }
        catch(SPDIOException &e)
        {
            throw e;
        }		
	}
    
    /*UPDATED*/
    void SPDFileIncrementalReader::readPulseDataInGeom(list<SPDPulse*> *pulses, OGRGeometry *geom) throw(SPDIOException)
	{
		if(!fileOpened)
		{
			throw SPDIOException("Input file is not open..");
		}
        
        if(spdFile->getFileType() == SPD_UPD_TYPE)
        {
            throw SPDIOException("This function is only available for files with a spatial index.");
        }
		
        OGREnvelope geomEnv;
        geom->getEnvelope(&geomEnv);
        
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
            }
        }
        catch(negative_overflow& e) 
        {
            throw SPDIOException(e.what());
        }
        catch(positive_overflow& e) 
        {
            throw SPDIOException(e.what());
        }
        catch(bad_numeric_cast& e) 
        {
            throw SPDIOException(e.what());
        }
        
        if(endCol <= startCol)
        {
            throw SPDIOException("Define subset is not within the input file (X Axis).");
        }
        
        if(endRow <= startRow)
        {
            throw SPDIOException("Define subset is not within the input file (Y Axis).");
        }
        
        try
        {
            OGRPoint *pt = new OGRPoint();
            for(boost::uint_fast32_t rows = startRow; rows < endRow; ++rows)
            {
                this->readPulseData(pulses, rows, startCol, endCol);
                for(list<SPDPulse*>::iterator iterPulses = pulses->begin(); iterPulses != pulses->end(); )
                {
                    pt->setX((*iterPulses)->xIdx);
                    pt->setY((*iterPulses)->yIdx);
                    
                    if(geom->Contains(pt))
                    {
                        ++iterPulses; // Next Pulse...
                    }
                    else 
                    {
                        SPDPulseUtils::deleteSPDPulse(*iterPulses);
                        iterPulses = pulses->erase(iterPulses);
                    }
                }
            }
            delete pt;
        }
        catch(SPDIOException &e)
        {
            throw e;
        }			
	}
	
    /*UPDATED*/
	void SPDFileIncrementalReader::readPulseDataInGeom(vector<SPDPulse*> *pulses, OGRGeometry *geom) throw(SPDIOException)
	{
		if(!fileOpened)
		{
			throw SPDIOException("Input file is not open..");
		}
        
        if(spdFile->getFileType() == SPD_UPD_TYPE)
        {
            throw SPDIOException("This function is only available for files with a spatial index.");
        }
		
        OGREnvelope geomEnv;
        geom->getEnvelope(&geomEnv);
        
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
            }
        }
        catch(negative_overflow& e) 
        {
            throw SPDIOException(e.what());
        }
        catch(positive_overflow& e) 
        {
            throw SPDIOException(e.what());
        }
        catch(bad_numeric_cast& e) 
        {
            throw SPDIOException(e.what());
        }
        
        if(endCol <= startCol)
        {
            throw SPDIOException("Define subset is not within the input file (X Axis).");
        }
        
        if(endRow <= startRow)
        {
            throw SPDIOException("Define subset is not within the input file (Y Axis).");
        }
        
        try
        {
            OGRPoint *pt = new OGRPoint();
            for(boost::uint_fast32_t rows = startRow; rows < endRow; ++rows)
            {
                this->readPulseData(pulses, rows, startCol, endCol);
                for(vector<SPDPulse*>::iterator iterPulses = pulses->begin(); iterPulses != pulses->end(); )
                {
                    pt->setX((*iterPulses)->xIdx);
                    pt->setY((*iterPulses)->yIdx);
                    
                    if(geom->Contains(pt))
                    {
                        ++iterPulses; // Next Pulse...
                    }
                    else 
                    {
                        SPDPulseUtils::deleteSPDPulse(*iterPulses);
                        iterPulses = pulses->erase(iterPulses);
                    }
                }
            }
            delete pt;
        }
        catch(SPDIOException &e)
        {
            throw e;
        }
	}

    /*UPDATED*/
	void SPDFileIncrementalReader::calcGeoEnv(OGREnvelope *env) throw(SPDIOException)
	{
		SPDPointUtils ptsUtils;
		SPDPulseUtils pulseUtils;
		
		bool first = true;
		double xMin = 0;
		double xMax = 0;
		double yMin = 0;
		double yMax = 0;
		
		try 
		{
			Exception::dontPrint();
			
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
			
			void *pulseArray = NULL;
            
            if(spdFile->getPulseVersion() == 1)
            {
                pulseArray = new SPDPulseH5V1[spdFile->getPulseBlockSize()];
            }
            else if(spdFile->getPulseVersion() == 2)
            {
                pulseArray = new SPDPulseH5V2[spdFile->getPulseBlockSize()];
            }
            			
			for( boost::uint_fast64_t i = 0; i < multipleOfBlocks; ++i)
			{
				pulseOffset[0] = i * spdFile->getPulseBlockSize();
				pulsesDataspace.selectHyperslab(H5S_SELECT_SET, pulseCount, pulseOffset);
				pulsesDataset.read(pulseArray, *pulseType, pulseMemspace, pulsesDataspace);

                
                if(spdFile->getPulseVersion() == 1)
                {
                    SPDPulseH5V1 *pulseObj = NULL;
                    for( boost::uint_fast32_t j = 0; j < spdFile->getPulseBlockSize(); ++j)
                    {
                        pulseObj = &((SPDPulseH5V1 *)pulseArray)[j];
                        if(first)
                        {
                            xMin = pulseObj->xIdx;
                            xMax = pulseObj->xIdx;
                            yMin = pulseObj->yIdx;
                            yMax = pulseObj->yIdx;
                        }
                        else 
                        {
                            if(pulseObj->xIdx < xMin)
                            {
                                xMin = pulseObj->xIdx;
                            }
                            else if(pulseObj->xIdx > xMax)
                            {
                                xMax = pulseObj->xIdx;
                            }
                            
                            if(pulseObj->yIdx < yMin)
                            {
                                yMin = pulseObj->yIdx;
                            }
                            else if(pulseObj->yIdx > yMax)
                            {
                                yMax = pulseObj->yIdx;
                            }
                        }
                    }
                }
                else if(spdFile->getPulseVersion() == 2)
                {
                    SPDPulseH5V2 *pulseObj = NULL;
                    for( boost::uint_fast32_t j = 0; j < spdFile->getPulseBlockSize(); ++j)
                    {
                        pulseObj = &((SPDPulseH5V2 *)pulseArray)[j];
                        if(first)
                        {
                            xMin = pulseObj->xIdx;
                            xMax = pulseObj->xIdx;
                            yMin = pulseObj->yIdx;
                            yMax = pulseObj->yIdx;
                        }
                        else 
                        {
                            if(pulseObj->xIdx < xMin)
                            {
                                xMin = pulseObj->xIdx;
                            }
                            else if(pulseObj->xIdx > xMax)
                            {
                                xMax = pulseObj->xIdx;
                            }
                            
                            if(pulseObj->yIdx < yMin)
                            {
                                yMin = pulseObj->yIdx;
                            }
                            else if(pulseObj->yIdx > yMax)
                            {
                                yMax = pulseObj->yIdx;
                            }
                        }
                    }
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
            
			pulsesDataset.read(pulseArray, *pulseType, pulseMemspace, pulsesDataspace );
            
            if(spdFile->getPulseVersion() == 1)
            {
                SPDPulseH5V1 *pulseObj = NULL;
                for( boost::uint_fast32_t j = 0; j < remainingPulses; ++j)
                {
                    pulseObj = &((SPDPulseH5V1 *)pulseArray)[j];
                    if(first)
                    {
                        xMin = pulseObj->xIdx;
                        xMax = pulseObj->xIdx;
                        yMin = pulseObj->yIdx;
                        yMax = pulseObj->yIdx;
                    }
                    else 
                    {
                        if(pulseObj->xIdx < xMin)
                        {
                            xMin = pulseObj->xIdx;
                        }
                        else if(pulseObj->xIdx > xMax)
                        {
                            xMax = pulseObj->xIdx;
                        }
                        
                        if(pulseObj->yIdx < yMin)
                        {
                            yMin = pulseObj->yIdx;
                        }
                        else if(pulseObj->yIdx > yMax)
                        {
                            yMax = pulseObj->yIdx;
                        }
                    }
                }
            }
            else if(spdFile->getPulseVersion() == 2)
            {
                SPDPulseH5V2 *pulseObj = NULL;
                for( boost::uint_fast32_t j = 0; j < remainingPulses; ++j)
                {
                    pulseObj = &((SPDPulseH5V2 *)pulseArray)[j];
                    if(first)
                    {
                        xMin = pulseObj->xIdx;
                        xMax = pulseObj->xIdx;
                        yMin = pulseObj->yIdx;
                        yMax = pulseObj->yIdx;
                    }
                    else 
                    {
                        if(pulseObj->xIdx < xMin)
                        {
                            xMin = pulseObj->xIdx;
                        }
                        else if(pulseObj->xIdx > xMax)
                        {
                            xMax = pulseObj->xIdx;
                        }
                        
                        if(pulseObj->yIdx < yMin)
                        {
                            yMin = pulseObj->yIdx;
                        }
                        else if(pulseObj->yIdx > yMax)
                        {
                            yMax = pulseObj->yIdx;
                        }
                    }
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
		
		env->MinX = xMin;
		env->MaxX = xMax;
		env->MinY = yMin;
		env->MaxY = yMax;
	}
	
	void SPDFileIncrementalReader::readQKRow(float *data, boost::uint_fast32_t row) throw(SPDIOException)
	{
		if(!fileOpened)
		{
			throw SPDIOException("Input file is not open..");
		}
        
        if(spdFile->getFileType() == SPD_UPD_TYPE)
        {
            throw SPDIOException("This function is only available for files with a spatial index.");
        }
		
		try
		{
			Exception::dontPrint();
			
			/* Read data */
			DataSet plsQKImgDSet = spdInFile->openDataSet( SPDFILE_DATASETNAME_QKLIMAGE );
			DataSpace plsQKImgDSpace = plsQKImgDSet.getSpace();
			
			hsize_t offsetDims[2];
			offsetDims[0] = row;
			offsetDims[1] = 0;
			hsize_t selectionSize[2];
			selectionSize[0]  = 1;
			selectionSize[1]  = spdFile->getNumberBinsX();
			plsQKImgDSpace.selectHyperslab( H5S_SELECT_SET, selectionSize, offsetDims );
			
			hsize_t memSpaceDims[1]; 
			memSpaceDims[0] = spdFile->getNumberBinsX();
			DataSpace memSpace( 1, memSpaceDims ); // has rank == 1
			
			hsize_t offsetMemSpace[1];
			hsize_t selectionMemSpace[1];
			offsetMemSpace[0] = 0;
			selectionMemSpace[0]  = spdFile->getNumberBinsX();
			memSpace.selectHyperslab( H5S_SELECT_SET, selectionMemSpace, offsetMemSpace );
			
			plsQKImgDSet.read( data, PredType::NATIVE_FLOAT, memSpace, plsQKImgDSpace );
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
	
	void SPDFileIncrementalReader::close() throw(SPDIOException)
	{
		try
		{
			Exception::dontPrint();
			
			if(fileOpened)
			{
				spdInFile->close();
				delete spdInFile;
				fileOpened = false;
				delete pointType;
				delete pulseType;
			}
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
	
	SPDFileIncrementalReader::~SPDFileIncrementalReader()
	{
		if(fileOpened)
		{
			try 
			{
				this->close();
			}
			catch (SPDIOException &e) 
			{
				cout << "WARNING: " << e.what() << endl;
			}
		}
	}
}

