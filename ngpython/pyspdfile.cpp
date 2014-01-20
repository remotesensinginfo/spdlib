
#include "spd/SPDFile.h"
#include "spd/SPDFileReader.h"
#include "spd/SPDFileIncrementalReader.h"

#include <Python.h>
#include "structmember.h"

#include "pulsearray.h"
#include "pointarray.h"

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
CREATE_GET_SET_INT(DayOfCreation)
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
    GETSETDEF(DayOfCreation),
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
    convertCPPPulseArrayToRecArrays(pulses, xBlockSize, yBlockSize, &pPulseArray, &pPointArray);

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
    "spdpy2.SPDFile",         /*tp_name*/
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


