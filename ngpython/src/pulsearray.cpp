/*
 *  pulsearray.cpp
 *  SPDLIB
 *
 *  Created by Sam Gillingham on 22/01/2014.
 *  Copyright 2013 SPDLib. All rights reserved.
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

#include "pulsearray.h"
#include "pointarray.h"

PulseArrayIndices::PulseArrayIndices()
{

}

PulseArrayIndices::~PulseArrayIndices()
{

}

void addPulseFields(RecArrayCreator *pCreator)
{
    pCreator->addField("pulseID", NPY_ULONG);
    pCreator->addField("gpsTime", NPY_ULONG);
    pCreator->addField("x0", NPY_DOUBLE);
    pCreator->addField("y0", NPY_DOUBLE);
    pCreator->addField("z0", NPY_FLOAT);
    pCreator->addField("h0", NPY_FLOAT);
    pCreator->addField("xIdx", NPY_DOUBLE);
    pCreator->addField("yIdx", NPY_DOUBLE);
    pCreator->addField("azimuth", NPY_FLOAT);
    pCreator->addField("zenith", NPY_FLOAT);
    pCreator->addField("numberOfReturns", NPY_USHORT);
    pCreator->addField("numOfTransmittedBins", NPY_USHORT);
    pCreator->addField("numOfReceivedBins", NPY_USHORT);
    pCreator->addField("rangeToWaveformStart", NPY_FLOAT);
    pCreator->addField("amplitudePulse", NPY_FLOAT);
    pCreator->addField("widthPulse", NPY_FLOAT);
    pCreator->addField("user", NPY_UINT);
    pCreator->addField("sourceID", NPY_USHORT);
    pCreator->addField("edgeFlightLineFlag", NPY_USHORT);
    pCreator->addField("scanDirectionFlag", NPY_USHORT);
    pCreator->addField("scanAngleRank", NPY_FLOAT);
    pCreator->addField("scanline", NPY_UINT);
    pCreator->addField("scanlineIdx", NPY_USHORT);
    pCreator->addField("receiveWaveNoiseThreshold", NPY_FLOAT);
    pCreator->addField("transWaveNoiseThres", NPY_FLOAT);
    pCreator->addField("wavelength", NPY_FLOAT);
    pCreator->addField("receiveWaveGain", NPY_FLOAT);
    pCreator->addField("receiveWaveOffset", NPY_FLOAT);
    pCreator->addField("transWaveGain", NPY_FLOAT);
    pCreator->addField("transWaveOffset", NPY_FLOAT);
    // 'fake' fields
    pCreator->addField("startPtsIdx", NPY_UINT);
    pCreator->addField("blockX", NPY_UINT);
    pCreator->addField("blockY", NPY_UINT);
    pCreator->addField("thisPulseIdx", NPY_UINT);
    pCreator->addField("startTransmittedIdx", NPY_UINT);
    pCreator->addField("startReceivedIdx", NPY_UINT);
}

PulseArrayIndices* getPulseIndices(PyObject *pArray)
{
    PulseArrayIndices *indices = new PulseArrayIndices();
    indices->pulseID.setField(pArray, "pulseID");
    indices->gpsTime.setField(pArray, "gpsTime");
    indices->x0.setField(pArray, "x0");
    indices->y0.setField(pArray, "y0");
    indices->z0.setField(pArray, "z0");
    indices->h0.setField(pArray, "h0");
    indices->xIdx.setField(pArray, "xIdx");
    indices->yIdx.setField(pArray, "yIdx");
    indices->azimuth.setField(pArray, "azimuth");
    indices->zenith.setField(pArray, "zenith");
    indices->numberOfReturns.setField(pArray, "numberOfReturns");
    indices->numOfTransmittedBins.setField(pArray, "numOfTransmittedBins");
    indices->numOfReceivedBins.setField(pArray, "numOfReceivedBins");
    indices->rangeToWaveformStart.setField(pArray, "rangeToWaveformStart");
    indices->amplitudePulse.setField(pArray, "amplitudePulse");
    indices->widthPulse.setField(pArray, "widthPulse");
    indices->user.setField(pArray, "user");
    indices->sourceID.setField(pArray, "sourceID");
    indices->edgeFlightLineFlag.setField(pArray, "edgeFlightLineFlag");
    indices->scanDirectionFlag.setField(pArray, "scanDirectionFlag");
    indices->scanAngleRank.setField(pArray, "scanAngleRank");
    indices->scanline.setField(pArray, "scanline");
    indices->scanlineIdx.setField(pArray, "scanlineIdx");
    indices->receiveWaveNoiseThreshold.setField(pArray, "receiveWaveNoiseThreshold");
    indices->transWaveNoiseThres.setField(pArray, "transWaveNoiseThres");
    indices->wavelength.setField(pArray, "wavelength");
    indices->receiveWaveGain.setField(pArray, "receiveWaveGain");
    indices->receiveWaveOffset.setField(pArray, "receiveWaveOffset");
    indices->transWaveGain.setField(pArray, "transWaveGain");
    indices->transWaveOffset.setField(pArray, "transWaveOffset");

    indices->startPtsIdx.setField(pArray, "startPtsIdx");
    indices->blockX.setField(pArray, "blockX");
    indices->blockY.setField(pArray, "blockY");
    indices->thisPulseIdx.setField(pArray, "thisPulseIdx");
    indices->startTransmittedIdx.setField(pArray, "startTransmittedIdx");
    indices->startReceivedIdx.setField(pArray, "startReceivedIdx");
    return indices;
}

PulsePointConverter::PulsePointConverter()
{
    addPulseFields(&m_pulseCreator);
    addPointFields(&m_pointCreator);
    m_ppulseIndices = NULL;
    m_ppointIndices = NULL;
}

PulsePointConverter::~PulsePointConverter()
{
    if( m_ppulseIndices != NULL )
        delete m_ppulseIndices;
    if( m_ppointIndices != NULL )
        delete m_ppointIndices;
}

void PulsePointConverter::convertCPPPulseArrayToRecArrays(std::vector<spdlib::SPDPulse*> ***pulses,
        boost::uint_fast32_t xSize, boost::uint_fast32_t ySize,
        PyObject **pPulseArray, PyObject **pPointArray, PyObject **pTransmittedArray/*=NULL*/, PyObject **pReceivedArray/*=NULL*/)
{
    // first need to scan through and get total size of arrays to be created
    npy_intp nPulses = 0, nPoints = 0, nTransmitted = 0, nReceived = 0;
    for(boost::uint_fast32_t i = 0; i < ySize; ++i)
    {
        for(boost::uint_fast32_t j = 0; j < xSize; ++j)
        {
            for(std::vector<spdlib::SPDPulse*>::iterator iterPls = pulses[i][j]->begin(); iterPls != pulses[i][j]->end(); ++iterPls)
            {
                nPulses++;
                nPoints += (*iterPls)->pts->size();
                nTransmitted += (*iterPls)->numOfTransmittedBins;
                nReceived += (*iterPls)->numOfReceivedBins;
            }
        }
    }

    *pPulseArray = m_pulseCreator.createArray(nPulses);
    *pPointArray = m_pointCreator.createArray(nPoints);

    if( pTransmittedArray != NULL )
    {
        *pTransmittedArray = PyArray_SimpleNew(1, &nTransmitted, NPY_INT32);
    }
    if( pReceivedArray != NULL )
    {
        *pReceivedArray = PyArray_SimpleNew(1, &nReceived, NPY_INT32);
    }

    // get the indices of our fields - assume these won't
    // be different for each array created
    if( m_ppulseIndices == NULL )
        m_ppulseIndices = getPulseIndices(*pPulseArray);
    if( m_ppointIndices == NULL )
        m_ppointIndices = getPointIndices(*pPointArray);

    npy_intp nPulseCount = 0, nPointCount = 0, nTransmittedCount = 0, nReceivedCount = 0;
    void *pRecord;
    for(boost::uint_fast32_t i = 0; i < ySize; ++i)
    {
        for(boost::uint_fast32_t j = 0; j < xSize; ++j)
        {
            npy_uint thisPulseIdx = 0;
            for(std::vector<spdlib::SPDPulse*>::iterator iterPls = pulses[i][j]->begin(); iterPls != pulses[i][j]->end(); ++iterPls)
            {
                // add all the points first
                npy_intp nStartPoint = nPointCount;
                for(std::vector<spdlib::SPDPoint*>::iterator iterPts = (*iterPls)->pts->begin(); iterPts != (*iterPls)->pts->end(); ++iterPts)
                {
                    pRecord = PyArray_GETPTR1((PyArrayObject*)*pPointArray, nPointCount);
                    copyPointToRecord(pRecord, (*iterPts), m_ppointIndices);
                    nPointCount++;
                }
                // now we know the start and end indices of the points we can add the pulse
                pRecord = PyArray_GETPTR1((PyArrayObject*)*pPulseArray, nPulseCount);
                if( nPointCount > nStartPoint )
                {
                    copyPulseToRecord(pRecord, (*iterPls), m_ppulseIndices, nStartPoint, nPointCount - nStartPoint, j, i, thisPulseIdx);
                }
                else
                {
                    copyPulseToRecord(pRecord, (*iterPls), m_ppulseIndices, 0, 0, j, i, thisPulseIdx);
                }

                // copy the transmitted
                if( pTransmittedArray != NULL )
                {
                    npy_intp nStartTransmitted = nTransmittedCount;
                    for( boost::uint_fast16_t n = 0; n < (*iterPls)->numOfTransmittedBins; n++ )
                    {
                        *(npy_int32*)PyArray_GETPTR1((PyArrayObject*)*pTransmittedArray, nTransmittedCount) = (*iterPls)->transmitted[n];
                        nTransmittedCount++;
                    }
                    m_ppulseIndices->startTransmittedIdx.setValue(pRecord, nStartTransmitted);
                    // numberOfTransmitted already set
                }
                else
                {
                    m_ppulseIndices->startTransmittedIdx.setValue(pRecord, 0);
                }

                // copy the received
                if( pReceivedArray != NULL )
                {
                    npy_intp nStartReceived = nReceivedCount;
                    for( boost::uint_fast16_t n = 0; n < (*iterPls)->numOfReceivedBins; n++ )
                    {
                        *(npy_int32*)PyArray_GETPTR1((PyArrayObject*)*pReceivedArray, nReceivedCount) = (*iterPls)->received[n];
                        nReceivedCount++;
                    }
                    m_ppulseIndices->startReceivedIdx.setValue(pRecord, nStartReceived);
                    // numberOfReceived already set
                }
                else
                {
                    m_ppulseIndices->startReceivedIdx.setValue(pRecord, 0);
                }


                nPulseCount++;
                thisPulseIdx++;
            }

        }
    }
}

void PulsePointConverter::convertRecArraysToCPPPulseArray(PyObject *pPulseArray, PyObject *pPointArray, std::vector<spdlib::SPDPulse*> ***pulses)
{
    // assume created in convertCPPPulseArrayToRecArrays above
    assert(m_ppulseIndices != NULL );
    assert(m_ppointIndices != NULL );

    void *pRecord;
    npy_intp nNumPulses = PyArray_DIM((PyArrayObject*)pPulseArray, 0);
    for( npy_intp nPulseCount = 0; nPulseCount < nNumPulses; nPulseCount++ )
    {
        pRecord = PyArray_GETPTR1((PyArrayObject*)pPulseArray, nPulseCount);
        npy_uint startPtsIdx = GetPulseStartPtsIdx(pRecord);
        npy_uint nNumberOfReturns = GetPulseNumberOfReturns(pRecord);
        npy_uint thisPulseIdx = GetPulseThisPulseIdx(pRecord);
        npy_uint blockX = GetPulseBlockX(pRecord);
        npy_uint blockY = GetPulseBlockY(pRecord);

        spdlib::SPDPulse* pCurrPulse = pulses[blockY][blockX]->at(thisPulseIdx);
        copyRecordToPulse(pCurrPulse, pRecord, m_ppulseIndices);

        if( nNumberOfReturns != 0 )
        {
            // we have points to copy out
            npy_uint nNewNumberOfReturns = 0; // they may have deleted some
            for( npy_uint i = 0; i < nNumberOfReturns; i++ )
            {
                pRecord = PyArray_GETPTR1((PyArrayObject*)pPointArray, startPtsIdx + i );
                if( !getPointRecordDeleteMe(pRecord,  m_ppointIndices) )
                {
                    copyRecordToPoint(pCurrPulse->pts->at(nNewNumberOfReturns), pRecord, m_ppointIndices);
                    nNewNumberOfReturns++;
                }
            }

            // they have deleted some
            if( nNewNumberOfReturns < nNumberOfReturns )
            {
                // TODO: do we have to do any other sort of deletion?
                pCurrPulse->pts->resize(nNewNumberOfReturns);
                pCurrPulse->numberOfReturns = nNewNumberOfReturns;
            }

            // TOOD: adding points
        }
    }
}

#if PY_MAJOR_VERSION >= 3
PyObject* pulsearray_init()
#else
void pulsearray_init()
#endif
{
    import_array();
#if PY_MAJOR_VERSION >= 3
    return NULL;
#endif
}
