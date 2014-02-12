/*
 *  pyspdfile.cpp
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

#include "spd/SPDFile.h"
#include "spd/SPDFileReader.h"
#include "spd/SPDFileIncrementalReader.h"

#include <Python.h>
#include "structmember.h"

#include "pulsearray.h"
#include "pointarray.h"
#include "pyspdfile.h"

// global error pointer - set in the pyspdfile_init method
PyObject *PySPDError;

// Python object wrapping a spdlib::SPDFile*
typedef struct
{
    PyObject_HEAD
    spdlib::SPDFile *pFile;
} PySPDFile;

// destructor - delete pFile
static void
PySPDFile_dealloc(PySPDFile *self)
{
    delete self->pFile;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

// init method - construct a spdlib::SPDFile
// assume this has been set to NULL in the new method
static int
PySPDFile_init(PySPDFile *self, PyObject *args, PyObject *kwds)
{
    char *pszFname = NULL;

    if( !PyArg_ParseTuple(args, "s", &pszFname ) )
    {
        return -1;
    }

    self->pFile = new spdlib::SPDFile(pszFname);
    if( self->pFile == NULL )
    {
        PyErr_SetString( PySPDError, "Unable to create SPDFile" );
        return -1;
    }
    return 0;
}

// OK because I am lazy I created these crazy macros
// to create get/set methods for this object. This
// is possible due to the way that Pete has named his
// functions regularly. Thanks Pete!

// Note: ## is the preprocesser append operator

// Get and Set of a string attribute.
#if PY_MAJOR_VERSION >= 3
    #define CREATE_GET_SET_STRING(BASE) static PyObject * \
        PySPDFile_get ## BASE(PySPDFile *self, void *closure) \
        { \
            std::string strVal = self->pFile->get ## BASE(); \
            return PyUnicode_FromString(strVal.c_str()); \
        } \
        static int \
        PySPDFile_set ## BASE(PySPDFile *self, PyObject *value, void *closure)\
        { \
            if( value == NULL ) \
            { \
                PyErr_SetString(PySPDError, "Can't delete attributes"); \
                return -1; \
            } \
            if( !PyUnicode_Check(value) ) \
            { \
                PyErr_SetString(PySPDError, "Must be a string"); \
                return -1; \
            } \
            PyObject *bytes = PyUnicode_AsEncodedString(value, NULL, NULL); \
            self->pFile->set ## BASE(PyBytes_AsString(bytes)); \
            Py_DECREF(bytes); \
            return 0; \
        }
#else
    #define CREATE_GET_SET_STRING(BASE) static PyObject * \
        PySPDFile_get ## BASE(PySPDFile *self, void *closure) \
        { \
            std::string strVal = self->pFile->get ## BASE(); \
            return PyString_FromString(strVal.c_str()); \
        } \
        static int \
        PySPDFile_set ## BASE(PySPDFile *self, PyObject *value, void *closure)\
        { \
            if( value == NULL ) \
            { \
                PyErr_SetString(PySPDError, "Can't delete attributes"); \
                return -1; \
            } \
            if( !PyString_Check(value) ) \
            { \
                PyErr_SetString(PySPDError, "Must be a string"); \
                return -1; \
            } \
            self->pFile->set ## BASE(PyString_AsString(value)); \
            return 0; \
        }
#endif

// Get and Set of an integer attribute.
#if PY_MAJOR_VERSION >= 3
#define CREATE_GET_SET_INT(BASE) static PyObject * \
        PySPDFile_get ## BASE(PySPDFile *self, void *closure) \
        { \
            long nVal = self->pFile->get ## BASE(); \
            return PyLong_FromLong(nVal); \
        } \
        static int \
        PySPDFile_set ## BASE(PySPDFile *self, PyObject *value, void *closure)\
        { \
            if( value == NULL ) \
            { \
                PyErr_SetString(PySPDError, "Can't delete attributes"); \
                return -1; \
            } \
            if( !PyLong_Check(value) ) \
            { \
                PyErr_SetString(PySPDError, "Must be an int"); \
                return -1; \
            } \
            self->pFile->set ## BASE(PyLong_AsLong(value)); \
            return 0; \
        }
#else
#define CREATE_GET_SET_INT(BASE) static PyObject * \
        PySPDFile_get ## BASE(PySPDFile *self, void *closure) \
        { \
            long nVal = self->pFile->get ## BASE(); \
            return PyInt_FromLong(nVal); \
        } \
        static int \
        PySPDFile_set ## BASE(PySPDFile *self, PyObject *value, void *closure)\
        { \
            if( value == NULL ) \
            { \
                PyErr_SetString(PySPDError, "Can't delete attributes"); \
                return -1; \
            } \
            if( !PyInt_Check(value) ) \
            { \
                PyErr_SetString(PySPDError, "Must be an int"); \
                return -1; \
            } \
            self->pFile->set ## BASE(PyInt_AsLong(value)); \
            return 0; \
        }
#endif

// Get and Set of a double attribute.
#define CREATE_GET_SET_DOUBLE(BASE) static PyObject * \
        PySPDFile_get ## BASE(PySPDFile *self, void *closure) \
        { \
            double dVal = self->pFile->get ## BASE(); \
            return PyFloat_FromDouble(dVal); \
        } \
        static int \
        PySPDFile_set ## BASE(PySPDFile *self, PyObject *value, void *closure)\
        { \
            if( value == NULL ) \
            { \
                PyErr_SetString(PySPDError, "Can't delete attributes"); \
                return -1; \
            } \
            if( !PyFloat_Check(value) ) \
            { \
                PyErr_SetString(PySPDError, "Must be a float"); \
                return -1; \
            } \
            self->pFile->set ## BASE(PyFloat_AsDouble(value)); \
            return 0; \
        }

// Get and Set of a std::vector<float> attribute
// Note: should I be doing this as an array
#define CREATE_GET_SET_FLOAT_VEC(BASE) static PyObject * \
        PySPDFile_get ## BASE(PySPDFile *self, void *closure) \
        { \
            std::vector<float> *vec = self->pFile->get ## BASE(); \
            PyObject *p = PyTuple_New(vec->size()); \
            for( std::vector<float>::size_type n = 0; n < vec->size(); n++) \
            { \
                PyObject *o = PyFloat_FromDouble(vec->at(n)); \
                PyTuple_SET_ITEM(p, n, o); \
            } \
            return p; \
        } \
        static int \
        PySPDFile_set ## BASE(PySPDFile *self, PyObject *value, void *closure)\
        { \
            if( value == NULL ) \
            { \
                PyErr_SetString(PySPDError, "Can't delete attributes"); \
                return -1; \
            } \
            if( !PySequence_Check(value) ) \
            { \
                PyErr_SetString(PySPDError, "Must be a sequence"); \
                return -1; \
            } \
            Py_ssize_t size = PySequence_Size(value); \
            std::vector<float> vec(size); \
            for( Py_ssize_t n = 0; n < size; n++ ) \
            { \
                PyObject *o = PySequence_GetItem(value, n); \
                if( !PyFloat_Check(o) ) \
                { \
                    PyErr_SetString(PySPDError, "Must be a sequence of floats" ); \
                    Py_DECREF(o); \
                    return -1; \
                } \
                vec[n] = PyFloat_AsDouble(o); \
                Py_DECREF(o); \
            } \
            self->pFile->set ## BASE(vec); \
            return 0; \
        }

// Creates the entry in the PyGetSetDef array.
// Note: # is the preprocessor 'stringification' operator
#define GETSETDEF(BASE) {(char*)#BASE, (getter)PySPDFile_get ## BASE, (setter)PySPDFile_set ## BASE, NULL, NULL}

// Now create all the get/set methods for the attributes
CREATE_GET_SET_STRING(FilePath)
CREATE_GET_SET_STRING(SpatialReference)
CREATE_GET_SET_INT(IndexType)
CREATE_GET_SET_INT(FileType)
CREATE_GET_SET_INT(DiscretePtDefined)
CREATE_GET_SET_INT(DecomposedPtDefined)
CREATE_GET_SET_INT(TransWaveformDefined)
CREATE_GET_SET_INT(ReceiveWaveformDefined)
CREATE_GET_SET_INT(MajorSPDVersion)
CREATE_GET_SET_INT(MinorSPDVersion)
CREATE_GET_SET_INT(PointVersion)
CREATE_GET_SET_INT(PulseVersion)
CREATE_GET_SET_STRING(GeneratingSoftware)
CREATE_GET_SET_STRING(SystemIdentifier)
CREATE_GET_SET_STRING(FileSignature)
CREATE_GET_SET_INT(YearOfCreation)
CREATE_GET_SET_INT(MonthOfCreation)
CREATE_GET_SET_INT(DayOfCreation)
CREATE_GET_SET_INT(HourOfCreation)
CREATE_GET_SET_INT(MinuteOfCreation)
CREATE_GET_SET_INT(SecondOfCreation)
CREATE_GET_SET_INT(YearOfCapture)
CREATE_GET_SET_INT(MonthOfCapture)
CREATE_GET_SET_INT(DayOfCapture)
CREATE_GET_SET_INT(HourOfCapture)
CREATE_GET_SET_INT(MinuteOfCapture)
CREATE_GET_SET_INT(SecondOfCapture)
CREATE_GET_SET_INT(NumberOfPoints)
CREATE_GET_SET_INT(NumberOfPulses)
CREATE_GET_SET_STRING(UserMetaField)
CREATE_GET_SET_DOUBLE(XMin)
CREATE_GET_SET_DOUBLE(XMax)
CREATE_GET_SET_DOUBLE(YMin)
CREATE_GET_SET_DOUBLE(YMax)
CREATE_GET_SET_DOUBLE(ZMin)
CREATE_GET_SET_DOUBLE(ZMax)
CREATE_GET_SET_DOUBLE(ZenithMin)
CREATE_GET_SET_DOUBLE(ZenithMax)
CREATE_GET_SET_DOUBLE(AzimuthMin)
CREATE_GET_SET_DOUBLE(AzimuthMax)
CREATE_GET_SET_DOUBLE(RangeMin)
CREATE_GET_SET_DOUBLE(RangeMax)
CREATE_GET_SET_DOUBLE(ScanlineMin)
CREATE_GET_SET_DOUBLE(ScanlineMax)
CREATE_GET_SET_DOUBLE(ScanlineIdxMin)
CREATE_GET_SET_DOUBLE(ScanlineIdxMax)
CREATE_GET_SET_DOUBLE(BinSize)
CREATE_GET_SET_INT(NumberBinsX)
CREATE_GET_SET_INT(NumberBinsY)
CREATE_GET_SET_FLOAT_VEC(Wavelengths)
CREATE_GET_SET_FLOAT_VEC(Bandwidths)
CREATE_GET_SET_INT(NumOfWavelengths)
CREATE_GET_SET_DOUBLE(PulseRepetitionFreq)
CREATE_GET_SET_DOUBLE(BeamDivergence)
CREATE_GET_SET_DOUBLE(SensorHeight)
CREATE_GET_SET_DOUBLE(Footprint)
CREATE_GET_SET_DOUBLE(MaxScanAngle)
CREATE_GET_SET_INT(RGBDefined)
CREATE_GET_SET_INT(PulseBlockSize)
CREATE_GET_SET_INT(ReceivedBlockSize)
CREATE_GET_SET_INT(TransmittedBlockSize)
CREATE_GET_SET_INT(WaveformBitRes)
CREATE_GET_SET_DOUBLE(TemporalBinSpacing)
CREATE_GET_SET_INT(ReturnNumsSynGen)
CREATE_GET_SET_INT(HeightDefined)
CREATE_GET_SET_DOUBLE(SensorSpeed)
CREATE_GET_SET_DOUBLE(SensorScanRate)
CREATE_GET_SET_DOUBLE(PointDensity)
CREATE_GET_SET_DOUBLE(PulseDensity)
CREATE_GET_SET_DOUBLE(PulseCrossTrackSpacing)
CREATE_GET_SET_DOUBLE(PulseAlongTrackSpacing)
CREATE_GET_SET_INT(OriginDefined)
CREATE_GET_SET_DOUBLE(PulseAngularSpacingAzimuth)
CREATE_GET_SET_DOUBLE(PulseAngularSpacingZenith)
CREATE_GET_SET_INT(PulseIdxMethod)
CREATE_GET_SET_DOUBLE(SensorApertureSize)
CREATE_GET_SET_DOUBLE(PulseEnergy)
CREATE_GET_SET_DOUBLE(FieldOfView)

// Create the PyGetSetDef entries with the help of our macro
static PyGetSetDef PySPDFile_getseters[] = {
    GETSETDEF(FilePath),
    GETSETDEF(SpatialReference),
    GETSETDEF(IndexType),
    GETSETDEF(FileType),
    GETSETDEF(DiscretePtDefined),
    GETSETDEF(DecomposedPtDefined),
    GETSETDEF(TransWaveformDefined),
    GETSETDEF(ReceiveWaveformDefined),
    GETSETDEF(MajorSPDVersion),
    GETSETDEF(MinorSPDVersion),
    GETSETDEF(PointVersion),
    GETSETDEF(PulseVersion),
    GETSETDEF(GeneratingSoftware),
    GETSETDEF(SystemIdentifier),
    GETSETDEF(FileSignature),
    GETSETDEF(YearOfCreation),
    GETSETDEF(MonthOfCreation),
    GETSETDEF(DayOfCreation),
    GETSETDEF(HourOfCreation),
    GETSETDEF(MinuteOfCreation),
    GETSETDEF(SecondOfCreation),
    GETSETDEF(YearOfCapture),
    GETSETDEF(MonthOfCapture),
    GETSETDEF(DayOfCapture),
    GETSETDEF(HourOfCapture),
    GETSETDEF(MinuteOfCapture),
    GETSETDEF(SecondOfCapture),
    GETSETDEF(NumberOfPoints),
    GETSETDEF(NumberOfPulses),
    GETSETDEF(UserMetaField),
    GETSETDEF(XMin),
    GETSETDEF(XMax),
    GETSETDEF(YMin),
    GETSETDEF(YMax),
    GETSETDEF(ZMin),
    GETSETDEF(ZMax),
    GETSETDEF(ZenithMin),
    GETSETDEF(ZenithMax),
    GETSETDEF(AzimuthMin),
    GETSETDEF(AzimuthMax),
    GETSETDEF(RangeMin),
    GETSETDEF(RangeMax),
    GETSETDEF(ScanlineMin),
    GETSETDEF(ScanlineMax),
    GETSETDEF(ScanlineIdxMin),
    GETSETDEF(ScanlineIdxMax),
    GETSETDEF(BinSize),
    GETSETDEF(NumberBinsX),
    GETSETDEF(NumberBinsY),
    GETSETDEF(Wavelengths),
    GETSETDEF(Bandwidths),
    GETSETDEF(NumOfWavelengths),
    GETSETDEF(PulseRepetitionFreq),
    GETSETDEF(BeamDivergence),
    GETSETDEF(SensorHeight),
    GETSETDEF(Footprint),
    GETSETDEF(MaxScanAngle),
    GETSETDEF(RGBDefined),
    GETSETDEF(PulseBlockSize),
    GETSETDEF(ReceivedBlockSize),
    GETSETDEF(TransmittedBlockSize),
    GETSETDEF(WaveformBitRes),
    GETSETDEF(TemporalBinSpacing),
    GETSETDEF(ReturnNumsSynGen),
    GETSETDEF(HeightDefined),
    GETSETDEF(SensorSpeed),
    GETSETDEF(SensorScanRate),
    GETSETDEF(PointDensity),
    GETSETDEF(PulseDensity),
    GETSETDEF(PulseCrossTrackSpacing),
    GETSETDEF(PulseAlongTrackSpacing),
    GETSETDEF(OriginDefined),
    GETSETDEF(PulseAngularSpacingAzimuth),
    GETSETDEF(PulseAngularSpacingZenith),
    GETSETDEF(PulseIdxMethod),
    GETSETDEF(SensorApertureSize),
    GETSETDEF(PulseEnergy),
    GETSETDEF(FieldOfView),
    {NULL} // sentinel
};

// methods on the object
static PyObject*
PySPDFile_setBoundingBox(PySPDFile *self, PyObject *args)
{
    double xMin, xMax, yMin, yMax;
    if( !PyArg_ParseTuple(args, "dddd", &xMin, &xMax, &yMin, &yMax ) )
        return NULL;

    self->pFile->setBoundingBox(xMin, xMax, yMin, yMax);

    Py_RETURN_NONE;
}

static PyObject*
PySPDFile_setBoundingVolume(PySPDFile *self, PyObject *args)
{
    double xMin, xMax, yMin, yMax, zMin, zMax;
    if( !PyArg_ParseTuple(args, "dddddd", &xMin, &xMax, &yMin, &yMax, &zMin, &zMax ) )
        return NULL;

    self->pFile->setBoundingVolume(xMin, xMax, yMin, yMax, zMin, zMax);

    Py_RETURN_NONE;
}

static PyObject*
PySPDFile_setBoundingBoxSpherical(PySPDFile *self, PyObject *args)
{
    double zenithMin, zenithMax, azimuthMin, azimuthMax;
    if( !PyArg_ParseTuple(args, "dddd", &zenithMin, &zenithMax, &azimuthMin, &azimuthMax) )
        return NULL;

    self->pFile->setBoundingBoxSpherical(zenithMin, zenithMax, azimuthMin, azimuthMax);

    Py_RETURN_NONE;
}

static PyObject*
PySPDFile_setBoundingVolumeSpherical(PySPDFile *self, PyObject *args)
{
    double zenithMin, zenithMax, azimuthMin, azimuthMax, rangeMin, rangeMax;
    if( !PyArg_ParseTuple(args, "dddddd", &zenithMin, &zenithMax, &azimuthMin, &azimuthMax, &rangeMin, &rangeMax) )
        return NULL;

    self->pFile->setBoundingVolumeSpherical(zenithMin, zenithMax, azimuthMin, azimuthMax, rangeMin, rangeMax);

    Py_RETURN_NONE;
}

static PyObject*
PySPDFile_setBoundingBoxScanline(PySPDFile *self, PyObject *args)
{
    double scanlineMin, scanlineMax, scanlineIdxMin, scanlineIdxMax;
    if( !PyArg_ParseTuple(args, "dddd", &scanlineMin, &scanlineMax, &scanlineIdxMin, &scanlineIdxMax) )
        return NULL;

    self->pFile->setBoundingBoxScanline(scanlineMin, scanlineMax, scanlineIdxMin, scanlineIdxMax);

    Py_RETURN_NONE;
}

static PyObject*
PySPDFile_readHeader(PySPDFile *self, PyObject *args)
{
    try
    {
        std::string filepath = self->pFile->getFilePath();
        spdlib::SPDFileReader reader;
        reader.readHeaderInfo(filepath, self->pFile);
    }
    catch(spdlib::SPDException &e)
    {
        PyErr_SetString(PySPDError, e.what());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
PySPDFile_readBlock(PySPDFile *self, PyObject *args)
{
    PyObject *pSeq;
    if( !PyArg_ParseTuple(args, "O", &pSeq ) )
        return NULL;

    if( !PySequence_Check(pSeq) )
    {
        PyErr_SetString(PySPDError, "Must pass a sequnce" );
        return NULL;
    }

    if( PySequence_Size(pSeq) != 4 )
    {
        PyErr_SetString(PySPDError, "Expected a 4 element sequence" );
        return NULL;
    }

    boost::uint_fast32_t bboxArr[4];
    for( int n = 0; n < 4; n++ )
    {
        PyObject *o = PySequence_GetItem(pSeq, n); 
#if PY_MAJOR_VERSION >= 3
        if( !PyLong_Check(o) ) 
#else
        if( !PyInt_Check(o) ) 
#endif
        { 
            PyErr_SetString(PySPDError, "Must be a sequence of ints" ); 
            Py_DECREF(o); 
            return NULL;
        } 
#if PY_MAJOR_VERSION >= 3
        bboxArr[n] = PyLong_AsLong(o);
#else
        bboxArr[n] = PyInt_AsLong(o); 
#endif
        Py_DECREF(o); 
    }

    boost::uint_fast32_t xBlockSize = bboxArr[2] - bboxArr[0];
    boost::uint_fast32_t yBlockSize = bboxArr[3] - bboxArr[1];

    // Create C++ list
    std::vector<spdlib::SPDPulse*> ***pulses = new std::vector<spdlib::SPDPulse*>**[yBlockSize];
    for(boost::uint_fast32_t i = 0; i < yBlockSize; ++i)
    {
        pulses[i] = new std::vector<spdlib::SPDPulse*>*[xBlockSize];
        for(boost::uint_fast32_t j = 0; j < xBlockSize; ++j)
        {
            pulses[i][j] = new std::vector<spdlib::SPDPulse*>();
        }
    }
    spdlib::SPDPulseUtils pulseUtils;

    try
    {
        // Create incremental reader
        spdlib::SPDFileIncrementalReader incReader;

        // Open incremental UPD reader
        incReader.open(self->pFile);

        // Read SPD Pulse data from SPD file
        incReader.readPulseDataBlock(pulses, bboxArr);

        // close the incremental reader
        incReader.close();

    }
    catch(spdlib::SPDException &e)
    {
        PyErr_SetString(PySPDError, e.what());
        // Delete pulses list.
        for(boost::uint_fast32_t i = 0; i < yBlockSize; ++i)
        {
            for(boost::uint_fast32_t j = 0; j < xBlockSize; ++j)
            {
                if(pulses[i][j]->size() > 0)
                {
                    for(std::vector<spdlib::SPDPulse*>::iterator iterPulses = pulses[i][j]->begin(); iterPulses != pulses[i][j]->end(); ++iterPulses)
                    {
                        pulseUtils.deleteSPDPulse(*iterPulses);
                    }
                    pulses[i][j]->clear();
                }
            }
            delete[] pulses[i];
        }
        delete[] pulses;
        return NULL;
    }

    // create rec arrays
    PyObject *pPulseArray, *pPointArray;
    PulsePointConverter converter;
    converter.convertCPPPulseArrayToRecArrays(pulses, xBlockSize, yBlockSize, &pPulseArray, &pPointArray);

    // Delete pulses list.
    for(boost::uint_fast32_t i = 0; i < yBlockSize; ++i)
    {
        for(boost::uint_fast32_t j = 0; j < xBlockSize; ++j)
        {
            if(pulses[i][j]->size() > 0)
            {
                for(std::vector<spdlib::SPDPulse*>::iterator iterPulses = pulses[i][j]->begin(); iterPulses != pulses[i][j]->end(); ++iterPulses)
                {
                    pulseUtils.deleteSPDPulse(*iterPulses);
                }
                pulses[i][j]->clear();
            }
        }
        delete[] pulses[i];
    }
    delete[] pulses;

    // create a tuple holding the results
    PyObject *pResults = PyTuple_New(2);
    if( pResults == NULL )
    {
        PyErr_SetString(PySPDError, "Unable to create result tuple");
        Py_DECREF(pPulseArray);
        Py_DECREF(pPointArray);
        return NULL;
    }
    PyTuple_SetItem(pResults, 0, pPulseArray);
    PyTuple_SetItem(pResults, 1, pPointArray);

    return pResults;
}

// our table of methods
static PyMethodDef PySPDFile_methods[] = {
    {"setBoundingBox", (PyCFunction)PySPDFile_setBoundingBox, METH_VARARGS, NULL},
    {"setBoundingVolume", (PyCFunction)PySPDFile_setBoundingVolume, METH_VARARGS, NULL},
    {"setBoundingBoxSpherical", (PyCFunction)PySPDFile_setBoundingBoxSpherical, METH_VARARGS, NULL},
    {"setBoundingVolumeSpherical", (PyCFunction)PySPDFile_setBoundingVolumeSpherical, METH_VARARGS, NULL},
    {"setBoundingBoxScanline", (PyCFunction)PySPDFile_setBoundingBoxScanline, METH_VARARGS, NULL},
    {"readHeader", (PyCFunction)PySPDFile_readHeader, METH_NOARGS, NULL},
    {"readBlock", (PyCFunction)PySPDFile_readBlock, METH_VARARGS, NULL},
    {NULL}  /* Sentinel */
};

static PyTypeObject PySPDFileType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "_spdpy2.SPDFile",         /*tp_name*/
    sizeof(PySPDFile),             /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)PySPDFile_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "SPDFile objects",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    PySPDFile_methods,             /* tp_methods */
    0,             /* tp_members */
    PySPDFile_getseters,           /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)PySPDFile_init,      /* tp_init */
    0,                         /* tp_alloc */
    0,                 /* tp_new */
};

PyMODINIT_FUNC pyspdfile_init(PyObject *module, PyObject *error)
{
    import_array();
    
    PySPDError = error;

    PySPDFileType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&PySPDFileType) < 0)
#if PY_MAJOR_VERSION >= 3
        return NULL;
#else
        return;
#endif

    Py_INCREF(&PySPDFileType);
    PyModule_AddObject(module, "SPDFile", (PyObject *)&PySPDFileType);
#if PY_MAJOR_VERSION >= 3
    return NULL;
#endif
}

void addSPDFileFields(RecArrayCreator *pCreator, spdlib::SPDFile *pFile)
{
    //pCreator->addField("FilePath", 'S', pFile->getFilePath().size());
    //pCreator->addField("SpatialReference", 'S', pFile->getSpatialReference().size());
    pCreator->addField("IndexType", NPY_UINT16);
    pCreator->addField("FileType", NPY_UINT16);
    pCreator->addField("DiscretePtDefined", NPY_INT16);
    pCreator->addField("DecomposedPtDefined", NPY_INT16);
    pCreator->addField("TransWaveformDefined", NPY_INT16);
    pCreator->addField("ReceiveWaveformDefined", NPY_INT16);
    pCreator->addField("MajorSPDVersion", NPY_UINT16);
    pCreator->addField("MinorSPDVersion", NPY_UINT16);
    pCreator->addField("PointVersion", NPY_UINT16);
    pCreator->addField("PulseVersion", NPY_UINT16);
    //pCreator->addField("GeneratingSoftware", 'S', pFile->getGeneratingSoftware().size());
    //pCreator->addField("SystemIdentifier", 'S', pFile->getSystemIdentifier().size());
    //pCreator->addField("FileSignature", 'S', pFile->getGeneratingSoftware().size());
    pCreator->addField("YearOfCreation", NPY_UINT16);
    pCreator->addField("MonthOfCreation", NPY_UINT16);
    pCreator->addField("DayOfCreation", NPY_UINT16);
    pCreator->addField("HourOfCreation", NPY_UINT16);
    pCreator->addField("MinuteOfCreation", NPY_UINT16);
    pCreator->addField("SecondOfCreation", NPY_UINT16);
    pCreator->addField("YearOfCapture", NPY_UINT16);
    pCreator->addField("MonthOfCapture", NPY_UINT16);
    pCreator->addField("DayOfCapture", NPY_UINT16);
    pCreator->addField("HourOfCapture", NPY_UINT16);
    pCreator->addField("MinuteOfCapture", NPY_UINT16);
    pCreator->addField("SecondOfCapture", NPY_UINT16);
    pCreator->addField("NumberOfPoints", NPY_UINT64);
    pCreator->addField("NumberOfPulses", NPY_UINT64);
    //pCreator->addField("UserMetaField", 'S', pFile->getUserMetaField().size());
    pCreator->addField("XMin", NPY_DOUBLE);
    pCreator->addField("XMax", NPY_DOUBLE);
    pCreator->addField("YMin", NPY_DOUBLE);
    pCreator->addField("YMax", NPY_DOUBLE);
    pCreator->addField("ZMin", NPY_DOUBLE);
    pCreator->addField("ZMax", NPY_DOUBLE);
    pCreator->addField("ZenithMin", NPY_DOUBLE);
    pCreator->addField("ZenithMax", NPY_DOUBLE);
    pCreator->addField("AzimuthMin", NPY_DOUBLE);
    pCreator->addField("AzimuthMax", NPY_DOUBLE);
    pCreator->addField("RangeMin", NPY_DOUBLE);
    pCreator->addField("RangeMax", NPY_DOUBLE);
    pCreator->addField("ScanlineMin", NPY_DOUBLE);
    pCreator->addField("ScanlineMax", NPY_DOUBLE);
    pCreator->addField("ScanlineIdxMin", NPY_DOUBLE);
    pCreator->addField("ScanlineIdxMax", NPY_DOUBLE);
    pCreator->addField("BinSize", NPY_DOUBLE);
    pCreator->addField("NumberBinsX", NPY_UINT32);
    pCreator->addField("NumberBinsY", NPY_UINT32);
    //pCreator->addField("Wavelengths", NPY_FLOAT, pFile->getWavelengths()->size());
    //pCreator->addField("Bandwidths", NPY_FLOAT, pFile->getBandwidths()->size());
    pCreator->addField("Wavelengths", NPY_FLOAT);
    pCreator->addField("Bandwidths", NPY_FLOAT);

    pCreator->addField("NumOfWavelengths", NPY_UINT16);
    pCreator->addField("PulseRepetitionFreq", NPY_FLOAT);
    pCreator->addField("BeamDivergence", NPY_FLOAT);
    pCreator->addField("SensorHeight", NPY_DOUBLE);
    pCreator->addField("Footprint", NPY_FLOAT);
    pCreator->addField("MaxScanAngle", NPY_FLOAT);
    pCreator->addField("RGBDefined", NPY_INT16);
    pCreator->addField("PulseBlockSize", NPY_UINT16);
    pCreator->addField("ReceivedBlockSize", NPY_UINT16);
    pCreator->addField("TransmittedBlockSize", NPY_UINT16);
    pCreator->addField("WaveformBitRes", NPY_UINT16);
    pCreator->addField("TemporalBinSpacing", NPY_DOUBLE);
    pCreator->addField("ReturnNumsSynGen", NPY_INT16);
    pCreator->addField("HeightDefined", NPY_INT16);
    pCreator->addField("SensorSpeed", NPY_FLOAT);
    pCreator->addField("SensorScanRate", NPY_FLOAT);
    pCreator->addField("PointDensity", NPY_FLOAT);
    pCreator->addField("PulseDensity", NPY_FLOAT);
    pCreator->addField("PulseCrossTrackSpacing", NPY_FLOAT);
    pCreator->addField("PulseAlongTrackSpacing", NPY_FLOAT);
    pCreator->addField("OriginDefined", NPY_INT16);
    pCreator->addField("PulseAngularSpacingAzimuth", NPY_FLOAT);
    pCreator->addField("PulseAngularSpacingZenith", NPY_FLOAT);
    pCreator->addField("PulseIdxMethod", NPY_INT16);
    pCreator->addField("SensorApertureSize", NPY_FLOAT);
    pCreator->addField("PulseEnergy", NPY_FLOAT);
    pCreator->addField("FieldOfView", NPY_FLOAT);

    // fake
    pCreator->addField("processingBinSize", NPY_FLOAT);
}

SPDFileArrayIndices* getSPDFileIndices(PyObject *pArray)
{
    SPDFileArrayIndices* indices = new SPDFileArrayIndices();
    //indices->FilePath.setField(pArray, "FilePath");
    //indices->SpatialReference.setField(pArray, "SpatialReference");
    indices->IndexType.setField(pArray, "IndexType");
    indices->FileType.setField(pArray, "FileType");
    indices->DiscretePtDefined.setField(pArray, "DiscretePtDefined");
    indices->DecomposedPtDefined.setField(pArray, "DecomposedPtDefined");
    indices->TransWaveformDefined.setField(pArray, "TransWaveformDefined");
    indices->ReceiveWaveformDefined.setField(pArray, "ReceiveWaveformDefined");
    indices->MajorSPDVersion.setField(pArray, "MajorSPDVersion");
    indices->MinorSPDVersion.setField(pArray, "MinorSPDVersion");
    indices->PointVersion.setField(pArray, "PointVersion");
    indices->PulseVersion.setField(pArray, "PulseVersion");
    //indices->GeneratingSoftware.setField(pArray, "GeneratingSoftware");
    //indices->SystemIdentifier.setField(pArray, "SystemIdentifier");
    //indices->FileSignature.setField(pArray, "FileSignature");
    indices->YearOfCreation.setField(pArray, "YearOfCreation");
    indices->MonthOfCreation.setField(pArray, "MonthOfCreation");
    indices->DayOfCreation.setField(pArray, "DayOfCreation");
    indices->HourOfCreation.setField(pArray, "HourOfCreation");
    indices->MinuteOfCreation.setField(pArray, "MinuteOfCreation");
    indices->SecondOfCreation.setField(pArray, "SecondOfCreation");
    indices->YearOfCapture.setField(pArray, "YearOfCapture");
    indices->MonthOfCapture.setField(pArray, "MonthOfCapture");
    indices->DayOfCapture.setField(pArray, "DayOfCapture");
    indices->HourOfCapture.setField(pArray, "HourOfCapture");
    indices->MinuteOfCapture.setField(pArray, "MinuteOfCapture");
    indices->SecondOfCapture.setField(pArray, "SecondOfCapture");
    indices->NumberOfPoints.setField(pArray, "NumberOfPoints");
    indices->NumberOfPulses.setField(pArray, "NumberOfPulses");
    //indices->UserMetaField.setField(pArray, "UserMetaField");
    indices->XMin.setField(pArray, "XMin");
    indices->XMax.setField(pArray, "XMax");
    indices->YMin.setField(pArray, "YMin");
    indices->YMax.setField(pArray, "YMax");
    indices->ZMin.setField(pArray, "ZMin");
    indices->ZMax.setField(pArray, "ZMax");
    indices->ZenithMin.setField(pArray, "ZenithMin");
    indices->ZenithMax.setField(pArray, "ZenithMax");
    indices->AzimuthMin.setField(pArray, "AzimuthMin");
    indices->AzimuthMax.setField(pArray, "AzimuthMax");
    indices->RangeMin.setField(pArray, "RangeMin");
    indices->RangeMax.setField(pArray, "RangeMax");
    indices->ScanlineMin.setField(pArray, "ScanlineMin");
    indices->ScanlineMax.setField(pArray, "ScanlineMax");
    indices->ScanlineIdxMin.setField(pArray, "ScanlineIdxMin");
    indices->ScanlineIdxMax.setField(pArray, "ScanlineIdxMax");
    indices->BinSize.setField(pArray, "BinSize");
    indices->NumberBinsX.setField(pArray, "NumberBinsX");
    indices->NumberBinsY.setField(pArray, "NumberBinsY");
    indices->Wavelengths.setField(pArray, "Wavelengths");
    indices->Bandwidths.setField(pArray, "Bandwidths");
    indices->NumOfWavelengths.setField(pArray, "NumOfWavelengths");
    indices->PulseRepetitionFreq.setField(pArray, "PulseRepetitionFreq");
    indices->BeamDivergence.setField(pArray, "BeamDivergence");
    indices->SensorHeight.setField(pArray, "SensorHeight");
    indices->Footprint.setField(pArray, "Footprint");
    indices->MaxScanAngle.setField(pArray, "MaxScanAngle");
    indices->RGBDefined.setField(pArray, "RGBDefined");
    indices->PulseBlockSize.setField(pArray, "PulseBlockSize");
    indices->ReceivedBlockSize.setField(pArray, "ReceivedBlockSize");
    indices->TransmittedBlockSize.setField(pArray, "TransmittedBlockSize");
    indices->WaveformBitRes.setField(pArray, "WaveformBitRes");
    indices->TemporalBinSpacing.setField(pArray, "TemporalBinSpacing");
    indices->ReturnNumsSynGen.setField(pArray, "ReturnNumsSynGen");
    indices->HeightDefined.setField(pArray, "HeightDefined");
    indices->SensorSpeed.setField(pArray, "SensorSpeed");
    indices->SensorScanRate.setField(pArray, "SensorScanRate");
    indices->PointDensity.setField(pArray, "PointDensity");
    indices->PulseDensity.setField(pArray, "PulseDensity");
    indices->PulseCrossTrackSpacing.setField(pArray, "PulseCrossTrackSpacing");
    indices->PulseAlongTrackSpacing.setField(pArray, "PulseAlongTrackSpacing");
    indices->OriginDefined.setField(pArray, "OriginDefined");
    indices->PulseAngularSpacingAzimuth.setField(pArray, "PulseAngularSpacingAzimuth");
    indices->PulseAngularSpacingZenith.setField(pArray, "PulseAngularSpacingZenith");
    indices->PulseIdxMethod.setField(pArray, "PulseIdxMethod");
    indices->SensorApertureSize.setField(pArray, "SensorApertureSize");
    indices->PulseEnergy.setField(pArray, "PulseEnergy");
    indices->FieldOfView.setField(pArray, "FieldOfView");

    // fake
    indices->processingBinSize.setField(pArray, "processingBinSize");
    return indices;
}

PyObject* createSPDFileArray(spdlib::SPDFile *pFile, float binSize)
{
    RecArrayCreator spdfileCreator;
    addSPDFileFields(&spdfileCreator, pFile);

    // length 1
    PyObject *pArray = spdfileCreator.createArray(1);
    
    SPDFileArrayIndices *pIndices = getSPDFileIndices(pArray);

    // get the first element 
    void *pRecord = PyArray_GETPTR1(pArray, 0);

    //pIndices->FilePath.setValueArray(pRecord, pFile->getFilePath().c_str());
    //pIndices->SpatialReference.setValueArray(pRecord, pFile->getSpatialReference().c_str());
    pIndices->IndexType.setValue(pRecord, pFile->getIndexType());
    pIndices->FileType.setValue(pRecord, pFile->getFileType());
    pIndices->DiscretePtDefined.setValue(pRecord, pFile->getDiscretePtDefined());
    pIndices->DecomposedPtDefined.setValue(pRecord, pFile->getDecomposedPtDefined());
    pIndices->TransWaveformDefined.setValue(pRecord, pFile->getTransWaveformDefined());
    pIndices->ReceiveWaveformDefined.setValue(pRecord, pFile->getReceiveWaveformDefined());
    pIndices->MajorSPDVersion.setValue(pRecord, pFile->getMajorSPDVersion());
    pIndices->MinorSPDVersion.setValue(pRecord, pFile->getMinorSPDVersion());
    pIndices->PointVersion.setValue(pRecord, pFile->getPointVersion());
    pIndices->PulseVersion.setValue(pRecord, pFile->getPulseVersion());
    //pIndices->GeneratingSoftware.setValueArray(pRecord, pFile->getGeneratingSoftware().c_str());
    //pIndices->SystemIdentifier.setValueArray(pRecord, pFile->getSystemIdentifier().c_str());
    //pIndices->FileSignature.setValueArray(pRecord, pFile->getFileSignature().c_str());
    pIndices->YearOfCreation.setValue(pRecord, pFile->getYearOfCreation());
    pIndices->MonthOfCreation.setValue(pRecord, pFile->getMonthOfCreation());
    pIndices->DayOfCreation.setValue(pRecord, pFile->getDayOfCreation());
    pIndices->HourOfCreation.setValue(pRecord, pFile->getHourOfCreation());
    pIndices->MinuteOfCreation.setValue(pRecord, pFile->getMinuteOfCreation());
    pIndices->SecondOfCreation.setValue(pRecord, pFile->getSecondOfCreation());
    pIndices->YearOfCapture.setValue(pRecord, pFile->getYearOfCapture());
    pIndices->MonthOfCapture.setValue(pRecord, pFile->getMonthOfCapture());
    pIndices->DayOfCapture.setValue(pRecord, pFile->getDayOfCapture());
    pIndices->HourOfCapture.setValue(pRecord, pFile->getHourOfCapture());
    pIndices->MinuteOfCapture.setValue(pRecord, pFile->getMinuteOfCapture());
    pIndices->SecondOfCapture.setValue(pRecord, pFile->getSecondOfCapture());
    pIndices->NumberOfPoints.setValue(pRecord, pFile->getNumberOfPoints());
    pIndices->NumberOfPulses.setValue(pRecord, pFile->getNumberOfPulses());
    //pIndices->UserMetaField.setValueArray(pRecord, pFile->getUserMetaField().c_str());
    pIndices->XMin.setValue(pRecord, pFile->getXMin());
    pIndices->XMax.setValue(pRecord, pFile->getXMax());
    pIndices->YMin.setValue(pRecord, pFile->getYMin());
    pIndices->YMax.setValue(pRecord, pFile->getYMax());
    pIndices->ZMin.setValue(pRecord, pFile->getZMin());
    pIndices->ZMax.setValue(pRecord, pFile->getZMax());
    pIndices->ZenithMin.setValue(pRecord, pFile->getZenithMin());
    pIndices->ZenithMax.setValue(pRecord, pFile->getZenithMax());
    pIndices->AzimuthMin.setValue(pRecord, pFile->getAzimuthMin());
    pIndices->AzimuthMax.setValue(pRecord, pFile->getAzimuthMax());
    pIndices->RangeMin.setValue(pRecord, pFile->getRangeMin());
    pIndices->RangeMax.setValue(pRecord, pFile->getRangeMax());
    pIndices->ScanlineMin.setValue(pRecord, pFile->getScanlineMin());
    pIndices->ScanlineMax.setValue(pRecord, pFile->getScanlineMax());
    pIndices->ScanlineIdxMin.setValue(pRecord, pFile->getScanlineIdxMin());
    pIndices->ScanlineIdxMax.setValue(pRecord, pFile->getScanlineIdxMax());
    pIndices->BinSize.setValue(pRecord, pFile->getBinSize());
    pIndices->NumberBinsX.setValue(pRecord, pFile->getNumberBinsX());
    pIndices->NumberBinsY.setValue(pRecord, pFile->getNumberBinsY());

    std::vector<float> *pWavelengths = pFile->getWavelengths();
    //pIndices->Wavelengths.setValueArray(pRecord, &(*pWavelengths)[0]);
    std::vector<float> *pBandWidths = pFile->getBandwidths();
    //pIndices->Bandwidths.setValueArray(pRecord, &(*pBandWidths)[0]);
    pIndices->Wavelengths.setValue(pRecord, (*pWavelengths)[0]);
    pIndices->Bandwidths.setValue(pRecord, (*pBandWidths)[0]);

    pIndices->NumOfWavelengths.setValue(pRecord, pFile->getNumOfWavelengths());
    pIndices->PulseRepetitionFreq.setValue(pRecord, pFile->getPulseRepetitionFreq());
    pIndices->BeamDivergence.setValue(pRecord, pFile->getBeamDivergence());
    pIndices->SensorHeight.setValue(pRecord, pFile->getSensorHeight());
    pIndices->Footprint.setValue(pRecord, pFile->getFootprint());
    pIndices->MaxScanAngle.setValue(pRecord, pFile->getMaxScanAngle());
    pIndices->RGBDefined.setValue(pRecord, pFile->getRGBDefined());
    pIndices->PulseBlockSize.setValue(pRecord, pFile->getPulseBlockSize());
    pIndices->ReceivedBlockSize.setValue(pRecord, pFile->getReceivedBlockSize());
    pIndices->TransmittedBlockSize.setValue(pRecord, pFile->getTransmittedBlockSize());
    pIndices->WaveformBitRes.setValue(pRecord, pFile->getWaveformBitRes());
    pIndices->TemporalBinSpacing.setValue(pRecord, pFile->getTemporalBinSpacing());
    pIndices->ReturnNumsSynGen.setValue(pRecord, pFile->getReturnNumsSynGen());
    pIndices->HeightDefined.setValue(pRecord, pFile->getHeightDefined());
    pIndices->SensorSpeed.setValue(pRecord, pFile->getSensorSpeed());
    pIndices->SensorScanRate.setValue(pRecord, pFile->getSensorScanRate());
    pIndices->PointDensity.setValue(pRecord, pFile->getPointDensity());
    pIndices->PulseDensity.setValue(pRecord, pFile->getPulseDensity());
    pIndices->PulseCrossTrackSpacing.setValue(pRecord, pFile->getPulseCrossTrackSpacing());
    pIndices->PulseAlongTrackSpacing.setValue(pRecord, pFile->getPulseAlongTrackSpacing());
    pIndices->OriginDefined.setValue(pRecord, pFile->getOriginDefined());
    pIndices->PulseAngularSpacingAzimuth.setValue(pRecord, pFile->getPulseAngularSpacingAzimuth());
    pIndices->PulseAngularSpacingZenith.setValue(pRecord, pFile->getPulseAngularSpacingZenith());
    pIndices->PulseIdxMethod.setValue(pRecord, pFile->getPulseIdxMethod());
    pIndices->SensorApertureSize.setValue(pRecord, pFile->getSensorApertureSize());
    pIndices->PulseEnergy.setValue(pRecord, pFile->getPulseEnergy());
    pIndices->FieldOfView.setValue(pRecord, pFile->getFieldOfView());

    // fake
    pIndices->processingBinSize.setValue(pRecord, binSize);

    delete pIndices;
    return pArray;
}

