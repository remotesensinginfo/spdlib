#ifndef __PULSEARRAY_H__
#define __PULSEARRAY_H__

#include "recarray.h"
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
    // TODO: pts
    // TODO: transmistted
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
};

void addPulseFields(RecArrayCreator *pCreator);
PulseArrayIndices getPulseIndices(PyObject *pArray);
void copyPulseToRecord(void *pRecord, spdlib::SPDPulse *pulse, PulseArrayIndices &indices);

#if PY_MAJOR_VERSION >= 3
PyObject* pulsearray_init();
#else
void pulsearray_init();
#endif

#endif // __PULSEARRAY_H__
