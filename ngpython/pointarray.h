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

void pointarray_init();

#endif // __POINTARRAY_H__
