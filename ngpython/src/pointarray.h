/*
 *  pointarray.h
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
#ifndef __POINTARRAY_H__
#define __POINTARRAY_H__

#include "recarray.h"
#include "spd/SPDPoint.h"

class PointArrayIndices
{
public:
    PointArrayIndices();
    ~PointArrayIndices();

    RecArrayField<npy_ushort> returnID;
    RecArrayField<npy_ulong> gpsTime;
    RecArrayField<npy_double> x;
    RecArrayField<npy_double> y;
    RecArrayField<npy_float> z;
    RecArrayField<npy_float> height;
    RecArrayField<npy_float> range;
    RecArrayField<npy_float> amplitudeReturn;
    RecArrayField<npy_float> widthReturn;
    RecArrayField<npy_ushort> red;
    RecArrayField<npy_ushort> green;
    RecArrayField<npy_ushort> blue;
    RecArrayField<npy_ushort> classification;
    RecArrayField<npy_uint> user;
    RecArrayField<npy_ushort> modelKeyPoint;
    RecArrayField<npy_ushort> lowPoint;
    RecArrayField<npy_ushort> overlap;
    RecArrayField<npy_ushort> ignore;
    RecArrayField<npy_ushort> wavePacketDescIdx;
    RecArrayField<npy_uint> waveformOffset;
};

void addPointFields(RecArrayCreator *pCreator);
PointArrayIndices* getPointIndices(PyObject *pArray);

inline void copyPointToRecord(void *pRecord, spdlib::SPDPoint *point, PointArrayIndices *indices)
{
    indices->returnID.setValue(pRecord, point->returnID);
    indices->gpsTime.setValue(pRecord, point->gpsTime);
    indices->x.setValue(pRecord, point->x);
    indices->y.setValue(pRecord, point->y);
    indices->z.setValue(pRecord, point->z);
    indices->height.setValue(pRecord, point->height);
    indices->range.setValue(pRecord, point->range);
    indices->amplitudeReturn.setValue(pRecord, point->amplitudeReturn);
    indices->widthReturn.setValue(pRecord, point->widthReturn);
    indices->red.setValue(pRecord, point->red);
    indices->green.setValue(pRecord, point->green);
    indices->blue.setValue(pRecord, point->blue);
    indices->classification.setValue(pRecord, point->classification);
    indices->user.setValue(pRecord, point->user);
    indices->modelKeyPoint.setValue(pRecord, point->modelKeyPoint);
    indices->lowPoint.setValue(pRecord, point->lowPoint);
    indices->overlap.setValue(pRecord, point->overlap);
    indices->ignore.setValue(pRecord, point->ignore);
    indices->wavePacketDescIdx.setValue(pRecord, point->wavePacketDescIdx);
    indices->waveformOffset.setValue(pRecord, point->waveformOffset);
}

inline void copyRecordToPoint(spdlib::SPDPoint *point, void *pRecord, PointArrayIndices *indices)
{
    point->returnID = indices->returnID.getValue(pRecord);
    point->gpsTime = indices->gpsTime.getValue(pRecord);
    point->x = indices->x.getValue(pRecord);
    point->y = indices->y.getValue(pRecord);
    point->z = indices->z.getValue(pRecord);
    point->height = indices->height.getValue(pRecord);
    point->range = indices->range.getValue(pRecord);
    point->amplitudeReturn = indices->amplitudeReturn.getValue(pRecord);
    point->widthReturn = indices->widthReturn.getValue(pRecord);
    point->red = indices->red.getValue(pRecord);
    point->green = indices->green.getValue(pRecord);
    point->blue = indices->blue.getValue(pRecord);
    point->classification = indices->classification.getValue(pRecord);
    point->user = indices->user.getValue(pRecord);
    point->modelKeyPoint = indices->modelKeyPoint.getValue(pRecord);
    point->lowPoint = indices->lowPoint.getValue(pRecord);
    point->overlap = indices->overlap.getValue(pRecord);
    point->ignore = indices->ignore.getValue(pRecord);
    point->wavePacketDescIdx = indices->wavePacketDescIdx.getValue(pRecord);
    point->waveformOffset = indices->waveformOffset.getValue(pRecord);
}

PyMODINIT_FUNC pointarray_init();

#endif // __POINTARRAY_H__
