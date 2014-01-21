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
PointArrayIndices getPointIndices(PyObject *pArray);
void copyPointToRecord(void *pRecord, spdlib::SPDPoint *point, PointArrayIndices &indices);
void copyRecordToPoint(spdlib::SPDPoint *point, void *pRecord, PointArrayIndices &indices);

PyMODINIT_FUNC pointarray_init();

#endif // __POINTARRAY_H__
