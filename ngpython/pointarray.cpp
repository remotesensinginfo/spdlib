
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
}

PointArrayIndices getPointIndices(PyObject *pArray)
{
    PointArrayIndices indices;
    indices.returnID.setField(pArray, "returnID");
    indices.gpsTime.setField(pArray, "gpsTime");
    indices.x.setField(pArray, "x");
    indices.y.setField(pArray, "y");
    indices.z.setField(pArray, "z");
    indices.height.setField(pArray, "height");
    indices.range.setField(pArray, "range");
    indices.amplitudeReturn.setField(pArray, "amplitudeReturn");
    indices.widthReturn.setField(pArray, "widthReturn");
    indices.red.setField(pArray, "red");
    indices.green.setField(pArray, "green");
    indices.blue.setField(pArray, "blue");
    indices.classification.setField(pArray, "classification");
    indices.user.setField(pArray, "user");
    indices.modelKeyPoint.setField(pArray, "modelKeyPoint");
    indices.lowPoint.setField(pArray, "lowPoint");
    indices.overlap.setField(pArray, "overlap");
    indices.ignore.setField(pArray, "ignore");
    indices.wavePacketDescIdx.setField(pArray, "wavePacketDescIdx");
    indices.waveformOffset.setField(pArray, "waveformOffset");
    return indices;
}

void copyPointToRecord(void *pRecord, spdlib::SPDPoint *point, PointArrayIndices &indices)
{
    indices.returnID.setValue(pRecord, point->returnID);
    indices.gpsTime.setValue(pRecord, point->gpsTime);
    indices.x.setValue(pRecord, point->x);
    indices.y.setValue(pRecord, point->y);
    indices.z.setValue(pRecord, point->z);
    indices.height.setValue(pRecord, point->height);
    indices.range.setValue(pRecord, point->range);
    indices.amplitudeReturn.setValue(pRecord, point->amplitudeReturn);
    indices.widthReturn.setValue(pRecord, point->widthReturn);
    indices.red.setValue(pRecord, point->red);
    indices.green.setValue(pRecord, point->green);
    indices.blue.setValue(pRecord, point->blue);
    indices.classification.setValue(pRecord, point->classification);
    indices.user.setValue(pRecord, point->user);
    indices.modelKeyPoint.setValue(pRecord, point->modelKeyPoint);
    indices.lowPoint.setValue(pRecord, point->lowPoint);
    indices.overlap.setValue(pRecord, point->overlap);
    indices.ignore.setValue(pRecord, point->ignore);
    indices.wavePacketDescIdx.setValue(pRecord, point->wavePacketDescIdx);
    indices.waveformOffset.setValue(pRecord, point->waveformOffset);
}

PyMODINIT_FUNC pointarray_init()
{
    import_array();
#if PY_MAJOR_VERSION >= 3
    return NULL;
#endif
}
