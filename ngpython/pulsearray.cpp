
#include "pulsearray.h"

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
}

PulseArrayIndices getPulseIndices(PyObject *pArray)
{
    PulseArrayIndices indices;
    indices.pulseID.setField(pArray, "pulseID");
    indices.gpsTime.setField(pArray, "gpsTime");
    indices.x0.setField(pArray, "x0");
    indices.y0.setField(pArray, "y0");
    indices.z0.setField(pArray, "z0");
    indices.h0.setField(pArray, "h0");
    indices.xIdx.setField(pArray, "xIdx");
    indices.yIdx.setField(pArray, "yIdx");
    indices.azimuth.setField(pArray, "azimuth");
    indices.zenith.setField(pArray, "zenith");
    indices.numberOfReturns.setField(pArray, "numberOfReturns");
    indices.numOfTransmittedBins.setField(pArray, "numOfTransmittedBins");
    indices.numOfReceivedBins.setField(pArray, "numOfReceivedBins");
    indices.rangeToWaveformStart.setField(pArray, "rangeToWaveformStart");
    indices.amplitudePulse.setField(pArray, "amplitudePulse");
    indices.widthPulse.setField(pArray, "widthPulse");
    indices.user.setField(pArray, "user");
    indices.sourceID.setField(pArray, "sourceID");
    indices.edgeFlightLineFlag.setField(pArray, "edgeFlightLineFlag");
    indices.scanDirectionFlag.setField(pArray, "scanDirectionFlag");
    indices.scanAngleRank.setField(pArray, "scanAngleRank");
    indices.scanline.setField(pArray, "scanline");
    indices.scanlineIdx.setField(pArray, "scanlineIdx");
    indices.receiveWaveNoiseThreshold.setField(pArray, "receiveWaveNoiseThreshold");
    indices.transWaveNoiseThres.setField(pArray, "transWaveNoiseThres");
    indices.wavelength.setField(pArray, "wavelength");
    indices.receiveWaveGain.setField(pArray, "receiveWaveGain");
    indices.receiveWaveOffset.setField(pArray, "receiveWaveOffset");
    indices.transWaveGain.setField(pArray, "transWaveGain");
    indices.transWaveOffset.setField(pArray, "transWaveOffset");
    return indices;
}

void copyPulseToRecord(void *pRecord, spdlib::SPDPulse *pulse, PulseArrayIndices &indices)
{
    indices.pulseID.setValue(pRecord, pulse->pulseID);
    indices.gpsTime.setValue(pRecord, pulse->gpsTime);
    indices.x0.setValue(pRecord, pulse->x0);
    indices.y0.setValue(pRecord, pulse->y0);
    indices.z0.setValue(pRecord, pulse->z0);
    indices.h0.setValue(pRecord, pulse->h0);
    indices.xIdx.setValue(pRecord, pulse->xIdx);
    indices.yIdx.setValue(pRecord, pulse->yIdx);
    indices.azimuth.setValue(pRecord, pulse->azimuth);
    indices.zenith.setValue(pRecord, pulse->zenith);
    indices.numberOfReturns.setValue(pRecord, pulse->numberOfReturns);
    indices.numOfTransmittedBins.setValue(pRecord, pulse->numOfTransmittedBins);
    indices.numOfReceivedBins.setValue(pRecord, pulse->numOfReceivedBins);
    indices.rangeToWaveformStart.setValue(pRecord, pulse->rangeToWaveformStart);
    indices.amplitudePulse.setValue(pRecord, pulse->amplitudePulse);
    indices.widthPulse.setValue(pRecord, pulse->widthPulse);
    indices.user.setValue(pRecord, pulse->user);
    indices.sourceID.setValue(pRecord, pulse->sourceID);
    indices.edgeFlightLineFlag.setValue(pRecord, pulse->edgeFlightLineFlag);
    indices.scanDirectionFlag.setValue(pRecord, pulse->scanDirectionFlag);
    indices.scanAngleRank.setValue(pRecord, pulse->scanAngleRank);
    indices.scanline.setValue(pRecord, pulse->scanline);
    indices.scanlineIdx.setValue(pRecord, pulse->scanlineIdx);
    indices.receiveWaveNoiseThreshold.setValue(pRecord, pulse->receiveWaveNoiseThreshold);
    indices.transWaveNoiseThres.setValue(pRecord, pulse->transWaveNoiseThres);
    indices.wavelength.setValue(pRecord, pulse->wavelength);
    indices.receiveWaveGain.setValue(pRecord, pulse->receiveWaveGain);
    indices.receiveWaveOffset.setValue(pRecord, pulse->receiveWaveOffset);
    indices.transWaveGain.setValue(pRecord, pulse->transWaveGain);
    indices.transWaveOffset.setValue(pRecord, pulse->transWaveOffset);
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
