/*
 *  pulsearray.h
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
#ifndef __PULSEARRAY_H__
#define __PULSEARRAY_H__

#include "recarray.h"
#include "pointarray.h"
#include "spd/SPDPulse.h"

class PulseArrayIndices
{
public:
    PulseArrayIndices();
    ~PulseArrayIndices();

    RecArrayField<npy_ulong> pulseID;
    RecArrayField<npy_ulong> gpsTime;
    RecArrayField<npy_double> x0;
    RecArrayField<npy_double> y0;
    RecArrayField<npy_float> z0;
    RecArrayField<npy_float> h0;
    RecArrayField<npy_double> xIdx;
    RecArrayField<npy_double> yIdx;
    RecArrayField<npy_float> azimuth;
    RecArrayField<npy_float> zenith;
    RecArrayField<npy_ushort> numberOfReturns;
    // TODO: transmitted
    // TODO: received
    RecArrayField<npy_ushort> numOfTransmittedBins;
    RecArrayField<npy_ushort> numOfReceivedBins;
    RecArrayField<npy_float> rangeToWaveformStart;
    RecArrayField<npy_float> amplitudePulse;
    RecArrayField<npy_float> widthPulse;
    RecArrayField<npy_uint> user;
    RecArrayField<npy_ushort> sourceID;
    RecArrayField<npy_ushort> edgeFlightLineFlag;
    RecArrayField<npy_ushort> scanDirectionFlag;
    RecArrayField<npy_float> scanAngleRank;
    RecArrayField<npy_uint> scanline;
    RecArrayField<npy_ushort> scanlineIdx;
    RecArrayField<npy_float> receiveWaveNoiseThreshold;
    RecArrayField<npy_float> transWaveNoiseThres;
    RecArrayField<npy_float> wavelength;
    RecArrayField<npy_float> receiveWaveGain;
    RecArrayField<npy_float> receiveWaveOffset;
    RecArrayField<npy_float> transWaveGain;
    RecArrayField<npy_float> transWaveOffset;
    // index into associated points array
    RecArrayField<npy_uint> startPtsIdx;
    RecArrayField<npy_uint> nPoints; // TODO: is this the same as numberOfReturns?
    // index of this pulse in the original 2d array
    RecArrayField<npy_uint> blockX;
    RecArrayField<npy_uint> blockY;
    // index of this pulse in the vector of pulses for this bin
    RecArrayField<npy_uint> thisPulseIdx;
};

void addPulseFields(RecArrayCreator *pCreator);
PulseArrayIndices* getPulseIndices(PyObject *pArray);
inline void copyPulseToRecord(void *pRecord, spdlib::SPDPulse *pulse, PulseArrayIndices *indices, 
        npy_uint startPtsIdx, npy_uint nPoints, npy_uint blockX, npy_uint blockY, npy_uint thisPulseIdx)
{
    indices->pulseID.setValue(pRecord, pulse->pulseID);
    indices->gpsTime.setValue(pRecord, pulse->gpsTime);
    indices->x0.setValue(pRecord, pulse->x0);
    indices->y0.setValue(pRecord, pulse->y0);
    indices->z0.setValue(pRecord, pulse->z0);
    indices->h0.setValue(pRecord, pulse->h0);
    indices->xIdx.setValue(pRecord, pulse->xIdx);
    indices->yIdx.setValue(pRecord, pulse->yIdx);
    indices->azimuth.setValue(pRecord, pulse->azimuth);
    indices->zenith.setValue(pRecord, pulse->zenith);
    indices->numberOfReturns.setValue(pRecord, pulse->numberOfReturns);
    indices->numOfTransmittedBins.setValue(pRecord, pulse->numOfTransmittedBins);
    indices->numOfReceivedBins.setValue(pRecord, pulse->numOfReceivedBins);
    indices->rangeToWaveformStart.setValue(pRecord, pulse->rangeToWaveformStart);
    indices->amplitudePulse.setValue(pRecord, pulse->amplitudePulse);
    indices->widthPulse.setValue(pRecord, pulse->widthPulse);
    indices->user.setValue(pRecord, pulse->user);
    indices->sourceID.setValue(pRecord, pulse->sourceID);
    indices->edgeFlightLineFlag.setValue(pRecord, pulse->edgeFlightLineFlag);
    indices->scanDirectionFlag.setValue(pRecord, pulse->scanDirectionFlag);
    indices->scanAngleRank.setValue(pRecord, pulse->scanAngleRank);
    indices->scanline.setValue(pRecord, pulse->scanline);
    indices->scanlineIdx.setValue(pRecord, pulse->scanlineIdx);
    indices->receiveWaveNoiseThreshold.setValue(pRecord, pulse->receiveWaveNoiseThreshold);
    indices->transWaveNoiseThres.setValue(pRecord, pulse->transWaveNoiseThres);
    indices->wavelength.setValue(pRecord, pulse->wavelength);
    indices->receiveWaveGain.setValue(pRecord, pulse->receiveWaveGain);
    indices->receiveWaveOffset.setValue(pRecord, pulse->receiveWaveOffset);
    indices->transWaveGain.setValue(pRecord, pulse->transWaveGain);
    indices->transWaveOffset.setValue(pRecord, pulse->transWaveOffset);

    // 'fake' fields
    indices->startPtsIdx.setValue(pRecord, startPtsIdx);
    indices->nPoints.setValue(pRecord, nPoints);
    indices->blockX.setValue(pRecord, blockX);
    indices->blockY.setValue(pRecord, blockY);
    indices->thisPulseIdx.setValue(pRecord, thisPulseIdx);
}

inline void copyRecordToPulse(spdlib::SPDPulse *pulse, void *pRecord, PulseArrayIndices *indices)
{
    pulse->pulseID = indices->pulseID.getValue(pRecord);
    pulse->gpsTime = indices->gpsTime.getValue(pRecord);
    pulse->x0 = indices->x0.getValue(pRecord);
    pulse->y0 = indices->y0.getValue(pRecord);
    pulse->z0 = indices->z0.getValue(pRecord);
    pulse->h0 = indices->h0.getValue(pRecord);
    pulse->xIdx = indices->xIdx.getValue(pRecord);
    pulse->yIdx = indices->yIdx.getValue(pRecord);
    pulse->azimuth = indices->azimuth.getValue(pRecord);
    pulse->zenith = indices->zenith.getValue(pRecord);
    pulse->numberOfReturns = indices->numberOfReturns.getValue(pRecord);
    pulse->numOfTransmittedBins = indices->numOfTransmittedBins.getValue(pRecord);
    pulse->numOfReceivedBins = indices->numOfReceivedBins.getValue(pRecord);
    pulse->rangeToWaveformStart = indices->rangeToWaveformStart.getValue(pRecord);
    pulse->amplitudePulse = indices->amplitudePulse.getValue(pRecord);
    pulse->widthPulse = indices->widthPulse.getValue(pRecord);
    pulse->user = indices->user.getValue(pRecord);
    pulse->sourceID = indices->sourceID.getValue(pRecord);
    pulse->edgeFlightLineFlag = indices->edgeFlightLineFlag.getValue(pRecord);
    pulse->scanDirectionFlag = indices->scanDirectionFlag.getValue(pRecord);
    pulse->scanAngleRank = indices->scanAngleRank.getValue(pRecord);
    pulse->scanline = indices->scanline.getValue(pRecord);
    pulse->scanlineIdx = indices->scanlineIdx.getValue(pRecord);
    pulse->receiveWaveNoiseThreshold = indices->receiveWaveNoiseThreshold.getValue(pRecord);
    pulse->transWaveNoiseThres = indices->transWaveNoiseThres.getValue(pRecord);
    pulse->wavelength = indices->wavelength.getValue(pRecord);
    pulse->receiveWaveGain = indices->receiveWaveGain.getValue(pRecord);
    pulse->receiveWaveOffset = indices->receiveWaveOffset.getValue(pRecord);
    pulse->transWaveGain = indices->transWaveGain.getValue(pRecord);
    pulse->transWaveOffset = indices->transWaveOffset.getValue(pRecord);
}

// handles the conversion of the pulse C++ stuff
// into linked numpy pulse and point arrays
class PulsePointConverter
{
public:
    PulsePointConverter();
    ~PulsePointConverter();

    // methods for getting 'fake' records out of a pulse
    npy_uint GetPulseStartPtsIdx(void *pRecord)
    {
        return m_ppulseIndices->startPtsIdx.getValue(pRecord);
    }

    npy_uint GetPulseNPoints(void *pRecord)
    {
        return m_ppulseIndices->nPoints.getValue(pRecord);
    }

    npy_uint GetPulseThisPulseIdx(void *pRecord)
    {
        return m_ppulseIndices->thisPulseIdx.getValue(pRecord);
    }

    npy_uint GetPulseBlockX(void *pRecord)
    {
        return m_ppulseIndices->blockX.getValue(pRecord);
    }

    npy_uint GetPulseBlockY(void *pRecord)
    {
        return m_ppulseIndices->blockY.getValue(pRecord);
    }

    // methods to convert to/from C++ arrays    
    void convertCPPPulseArrayToRecArrays(std::vector<spdlib::SPDPulse*> ***pulses, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize,
        PyObject **pPulseArray, PyObject **pPointArray);

    void convertRecArraysToCPPPulseArray(PyObject *pPulseArray, PyObject *pPointArray, std::vector<spdlib::SPDPulse*> ***pulses);

private:
    RecArrayCreator m_pulseCreator;
    RecArrayCreator m_pointCreator;
    PulseArrayIndices *m_ppulseIndices;
    PointArrayIndices *m_ppointIndices;
};

#if PY_MAJOR_VERSION >= 3
PyObject* pulsearray_init();
#else
void pulsearray_init();
#endif

#endif // __PULSEARRAY_H__
