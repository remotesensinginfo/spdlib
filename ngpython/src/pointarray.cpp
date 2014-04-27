/*
 *  pointarray.cpp
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

#include "pointarray.h"

PointArrayIndices::PointArrayIndices()
{

}

PointArrayIndices::~PointArrayIndices()
{

}

void addPointFields(RecArrayCreator *pCreator)
{
    pCreator->addField("returnID", NPY_USHORT);
    pCreator->addField("gpsTime", NPY_ULONG);
    pCreator->addField("x", NPY_DOUBLE);
    pCreator->addField("y", NPY_DOUBLE);
    pCreator->addField("z", NPY_FLOAT);
    pCreator->addField("height", NPY_FLOAT);
    pCreator->addField("range", NPY_FLOAT);
    pCreator->addField("amplitudeReturn", NPY_FLOAT);
    pCreator->addField("widthReturn", NPY_FLOAT);
    pCreator->addField("red", NPY_USHORT);
    pCreator->addField("green", NPY_USHORT);
    pCreator->addField("blue", NPY_USHORT);
    pCreator->addField("classification", NPY_USHORT);
    pCreator->addField("user", NPY_UINT);
    pCreator->addField("modelKeyPoint", NPY_USHORT);
    pCreator->addField("lowPoint", NPY_USHORT);
    pCreator->addField("overlap", NPY_USHORT);
    pCreator->addField("ignore", NPY_USHORT);
    pCreator->addField("wavePacketDescIdx", NPY_USHORT);
    pCreator->addField("waveformOffset", NPY_UINT);

    pCreator->addField("deleteMe", NPY_BOOL);
}

PointArrayIndices* getPointIndices(PyObject *pArray)
{
    PointArrayIndices *indices = new PointArrayIndices();
    indices->returnID.setField(pArray, "returnID");
    indices->gpsTime.setField(pArray, "gpsTime");
    indices->x.setField(pArray, "x");
    indices->y.setField(pArray, "y");
    indices->z.setField(pArray, "z");
    indices->height.setField(pArray, "height");
    indices->range.setField(pArray, "range");
    indices->amplitudeReturn.setField(pArray, "amplitudeReturn");
    indices->widthReturn.setField(pArray, "widthReturn");
    indices->red.setField(pArray, "red");
    indices->green.setField(pArray, "green");
    indices->blue.setField(pArray, "blue");
    indices->classification.setField(pArray, "classification");
    indices->user.setField(pArray, "user");
    indices->modelKeyPoint.setField(pArray, "modelKeyPoint");
    indices->lowPoint.setField(pArray, "lowPoint");
    indices->overlap.setField(pArray, "overlap");
    indices->ignore.setField(pArray, "ignore");
    indices->wavePacketDescIdx.setField(pArray, "wavePacketDescIdx");
    indices->waveformOffset.setField(pArray, "waveformOffset");

    indices->deleteMe.setField(pArray, "deleteMe");
    return indices;
}

PyMODINIT_FUNC pointarray_init()
{
    import_array();
#if PY_MAJOR_VERSION >= 3
    return NULL;
#endif
}
