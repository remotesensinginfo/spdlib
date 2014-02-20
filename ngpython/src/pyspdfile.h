/*
 *  pyspdfile.h
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

#ifndef PYSPDFILE_H
#define PYSPDFILE_H

#include "recarray.h"
#include "spd/SPDFile.h"

PyMODINIT_FUNC pyspdfile_init(PyObject *module, PyObject *error);

class SPDFileArrayIndices
{
public:
    SPDFileArrayIndices()
    {
    }
    ~SPDFileArrayIndices()
    {
    }

    // numba currently not supporting strings so leave for now
    //RecArrayField<npy_char> FilePath;           // string
    //RecArrayField<npy_char> SpatialReference;   // string
    RecArrayField<npy_uint16> IndexType;
    RecArrayField<npy_uint16> FileType;
    RecArrayField<npy_int16> DiscretePtDefined;
    RecArrayField<npy_int16> DecomposedPtDefined;
    RecArrayField<npy_int16> TransWaveformDefined;
    RecArrayField<npy_int16> ReceiveWaveformDefined;
    RecArrayField<npy_uint16> MajorSPDVersion;
    RecArrayField<npy_uint16> MinorSPDVersion;
    RecArrayField<npy_uint16> PointVersion;
    RecArrayField<npy_uint16> PulseVersion;
    //RecArrayField<npy_char> GeneratingSoftware; // string
    //RecArrayField<npy_char> SystemIdentifier;   // string
    //RecArrayField<npy_char> FileSignature;      // string
    RecArrayField<npy_uint16> YearOfCreation;
    RecArrayField<npy_uint16> MonthOfCreation;
    RecArrayField<npy_uint16> DayOfCreation;
    RecArrayField<npy_uint16> HourOfCreation;
    RecArrayField<npy_uint16> MinuteOfCreation;
    RecArrayField<npy_uint16> SecondOfCreation;
    RecArrayField<npy_uint16> YearOfCapture;
    RecArrayField<npy_uint16> MonthOfCapture;
    RecArrayField<npy_uint16> DayOfCapture;
    RecArrayField<npy_uint16> HourOfCapture;
    RecArrayField<npy_uint16> MinuteOfCapture;
    RecArrayField<npy_uint16> SecondOfCapture;
    RecArrayField<npy_uint64> NumberOfPoints;
    RecArrayField<npy_uint64> NumberOfPulses;
    //RecArrayField<npy_char> UserMetaField;      // string
    RecArrayField<npy_double> XMin;
    RecArrayField<npy_double> XMax;
    RecArrayField<npy_double> YMin;
    RecArrayField<npy_double> YMax;
    RecArrayField<npy_double> ZMin;
    RecArrayField<npy_double> ZMax;
    RecArrayField<npy_double> ZenithMin;
    RecArrayField<npy_double> ZenithMax;
    RecArrayField<npy_double> AzimuthMin;
    RecArrayField<npy_double> AzimuthMax;
    RecArrayField<npy_double> RangeMin;
    RecArrayField<npy_double> RangeMax;
    RecArrayField<npy_double> ScanlineMin;
    RecArrayField<npy_double> ScanlineMax;
    RecArrayField<npy_double> ScanlineIdxMin;
    RecArrayField<npy_double> ScanlineIdxMax;
    RecArrayField<npy_double> BinSize;
    RecArrayField<npy_uint32> NumberBinsX;
    RecArrayField<npy_uint32> NumberBinsY;
    RecArrayField<npy_float> Wavelengths;       // array
    RecArrayField<npy_float> Bandwidths;        // array
    RecArrayField<npy_uint16> NumOfWavelengths;
    RecArrayField<npy_float> PulseRepetitionFreq;
    RecArrayField<npy_float> BeamDivergence;
    RecArrayField<npy_double> SensorHeight;
    RecArrayField<npy_float> Footprint;
    RecArrayField<npy_float> MaxScanAngle;
    RecArrayField<npy_int16> RGBDefined;
    RecArrayField<npy_uint16> PulseBlockSize;
    RecArrayField<npy_uint16> ReceivedBlockSize;
    RecArrayField<npy_uint16> TransmittedBlockSize;
    RecArrayField<npy_uint16> WaveformBitRes;
    RecArrayField<npy_double> TemporalBinSpacing;
    RecArrayField<npy_int16> ReturnNumsSynGen;
    RecArrayField<npy_int16> HeightDefined;
    RecArrayField<npy_float> SensorSpeed;
    RecArrayField<npy_float> SensorScanRate;
    RecArrayField<npy_float> PointDensity;
    RecArrayField<npy_float> PulseDensity;
    RecArrayField<npy_float> PulseCrossTrackSpacing;
    RecArrayField<npy_float> PulseAlongTrackSpacing;
    RecArrayField<npy_int16> OriginDefined;
    RecArrayField<npy_float> PulseAngularSpacingAzimuth;
    RecArrayField<npy_float> PulseAngularSpacingZenith;
    RecArrayField<npy_int16> PulseIdxMethod;
    RecArrayField<npy_float> SensorApertureSize;
    RecArrayField<npy_float> PulseEnergy;
    RecArrayField<npy_float> FieldOfView;

    // fake
    RecArrayField<npy_float> processingBinSize;
};

void addSPDFileFields(RecArrayCreator *pCreator, spdlib::SPDFile *pFile);
SPDFileArrayIndices* getSPDFileIndices(PyObject *pArray);
PyObject* createSPDFileArray(spdlib::SPDFile *pFile, float binSize);

#endif //PYSPDFILE_H

