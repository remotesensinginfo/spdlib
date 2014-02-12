/*
 *  spdpy2module.cpp
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

#include <Python.h>
#include "structmember.h"

#include "pyspdfile.h"
#include "pulsearray.h"
#include "pointarray.h"
#include "spd/SPDProcessingException.h"
#include "spd/SPDDataBlockProcessor.h"
#include "spd/SPDProcessDataBlocks.h"

/* An exception object for this module */
/* created in the init function */
/* used accross the other source files */
struct SPDPy2State
{
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct SPDPy2State*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct SPDPy2State _state;
#endif

static PyObject *
spdpy2_createPulseArray(PyObject *self, PyObject *args)
{
    PyObject *pObj;
    if( !PyArg_ParseTuple(args, "O", &pObj))
        return NULL;

    RecArrayCreator creator;
    addPulseFields(&creator);
    PyObject *pArray = NULL;
#if PY_MAJOR_VERSION >= 3
    if( PyLong_Check(pObj) )
#else
    if( PyInt_Check(pObj) )
#endif
    {
#if PY_MAJOR_VERSION >= 3
        pArray = creator.createArray(PyLong_AsLong(pObj));
#else
        pArray = creator.createArray(PyInt_AsLong(pObj));
#endif
    }
    else if( PySequence_Check(pObj))
    {
        int nd = PySequence_Size(pObj);
        npy_intp *dims = new npy_intp[nd];
        for( int n = 0; n < nd; n++ )
        {
            PyObject *pElement = PySequence_GetItem(pObj, n);
#if PY_MAJOR_VERSION >= 3
            if( !PyLong_Check(pElement))
#else
            if( !PyInt_Check(pElement))
#endif
            {
                PyErr_SetString(GETSTATE(self)->error, "sequence must be all ints");
                delete dims;
                Py_DECREF(pElement);
                return NULL;
            }
#if PY_MAJOR_VERSION >= 3
            dims[n] = PyLong_AsLong(pElement);
#else
            dims[n] = PyInt_AsLong(pElement);
#endif
            Py_DECREF(pElement);
        }
        pArray = creator.createArray(nd, dims);
        delete dims;
    }
    else
    {
        PyErr_SetString(GETSTATE(self)->error, "expected an int or a sequence");
        return NULL;
    }

    return pArray;
}

static PyObject *
spdpy2_createPointArray(PyObject *self, PyObject *args)
{
    PyObject *pObj;
    if( !PyArg_ParseTuple(args, "O", &pObj))
        return NULL;

    RecArrayCreator creator;
    addPointFields(&creator);
    PyObject *pArray = NULL;
#if PY_MAJOR_VERSION >= 3
    if( PyLong_Check(pObj) )
#else
    if( PyInt_Check(pObj) )
#endif
    {
#if PY_MAJOR_VERSION >= 3
        pArray = creator.createArray(PyLong_AsLong(pObj));
#else
        pArray = creator.createArray(PyInt_AsLong(pObj));
#endif
    }
    else if( PySequence_Check(pObj))
    {
        int nd = PySequence_Size(pObj);
        npy_intp *dims = new npy_intp[nd];
        for( int n = 0; n < nd; n++ )
        {
            PyObject *pElement = PySequence_GetItem(pObj, n);
#if PY_MAJOR_VERSION >= 3
            if( !PyLong_Check(pElement))
#else
            if( !PyInt_Check(pElement))
#endif
            {
                PyErr_SetString(GETSTATE(self)->error, "sequence must be all ints");
                delete dims;
                Py_DECREF(pElement);
                return NULL;
            }
#if PY_MAJOR_VERSION >= 3
            dims[n] = PyLong_AsLong(pElement);
#else
            dims[n] = PyInt_AsLong(pElement);
#endif
            Py_DECREF(pElement);
        }
        pArray = creator.createArray(nd, dims);
        delete dims;
    }
    else
    {
        PyErr_SetString(GETSTATE(self)->error, "expected an int or a sequence");
        return NULL;
    }

    return pArray;
}

// Convert a float *** to a numpy array to a numpy array
// Note to Pete: a 3D C array would have made this easier
// since we could have just wrapped it with PyArray_SimpleNewFromData
PyObject* ConvertFloatCPPArrayToNumpy(float ***imageDataBlock, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, 
            boost::uint_fast32_t numImgBands)
{
    npy_intp dims[] = {ySize, xSize, numImgBands};
    PyObject *pImageData = PyArray_SimpleNew(3, dims, NPY_FLOAT);

    for(boost::uint_fast32_t i = 0; i < ySize; ++i)
    {
        for(boost::uint_fast32_t j = 0; j < xSize; ++j)
        {
            for(boost::uint_fast32_t z = 0; z < numImgBands; ++z)
            {
                *((float*)PyArray_GETPTR3(pImageData, i, j, z)) = imageDataBlock[i][j][z];
            }
        }
    }

    return pImageData;
}

// the other way
void ConvertNumpyToFloatCPPArray(float ***imageDataBlock, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, 
            boost::uint_fast32_t numImgBands, PyObject *pImageData)
{
    for(boost::uint_fast32_t i = 0; i < ySize; ++i)
    {
        for(boost::uint_fast32_t j = 0; j < xSize; ++j)
        {
            for(boost::uint_fast32_t z = 0; z < numImgBands; ++z)
            {
                 imageDataBlock[i][j][z] = *((float*)PyArray_GETPTR3(pImageData, i, j, z));
            }
        }
    }
}

// handle the centre points
PyObject *ConvertCenPtsCPPArrayToNumpy(spdlib::SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize)
{
    npy_intp dims[] = {2, ySize, xSize};
    PyObject *pCenData = PyArray_SimpleNew(3, dims, NPY_DOUBLE);

    for(boost::uint_fast32_t i = 0; i < ySize; ++i)
    {
        for(boost::uint_fast32_t j = 0; j < xSize; ++j)
        {
            *((double*)PyArray_GETPTR3(pCenData, 0, i, j)) = cenPts[i][j]->x;
            *((double*)PyArray_GETPTR3(pCenData, 1, i, j)) = cenPts[i][j]->y;
        }
    }

    return pCenData;
}

// to the users function
const Py_ssize_t FILE_INDEX = 0;
const Py_ssize_t PULSES_INDEX = 1;
const Py_ssize_t POINTS_INDEX = 2;

// For use in the blockProcessor function
class SPDDataBlockProcessorPython : public spdlib::SPDDataBlockProcessor
{
public:
    SPDDataBlockProcessorPython(PyObject *pApplyFn, PyObject *pOtherInputs, bool bHaveOutputSPD,
                    bool bHaveInputImage, bool bHaveOutputImage, bool passCenPts)
    {
        m_pApplyFn = pApplyFn;
        m_pOtherInputs = pOtherInputs;
        m_bHaveOutputSPD = bHaveOutputSPD;
        m_bHaveInputImage = bHaveInputImage;
        m_bHaveOutputImage = bHaveOutputImage;
        // assume that this just needs to be done once
        // and stays the same for each block
        m_bHaveCreatedFileArray = false;

        // work out the indices of the optional params
        Py_ssize_t nLastIndex = POINTS_INDEX;
        if( m_bHaveInputImage || m_bHaveOutputImage )
        {
            nLastIndex++;
            m_nImageDataIndex = nLastIndex;
        }
        else
        {
            m_nImageDataIndex = -1;
        }

        if( passCenPts )
        {
            nLastIndex++;
            m_nCenPtsIndex = nLastIndex;
        }
        else
        {
            m_nCenPtsIndex = -1;
        }

        if( m_pOtherInputs != Py_None )
        {
            nLastIndex++;
            m_nOtherInputsIndex = nLastIndex;
        }
        else
        {
            m_nOtherInputsIndex = -1;
        }

        // create the tuple to pass to the user function
        m_pUserParams = PyTuple_New(nLastIndex + 1);
        if( m_pUserParams == NULL )
        {
            throw spdlib::SPDProcessingException("Unable to create parameter tuple");
        }

        // might as well set the user data since this doesn't change
        if( m_nOtherInputsIndex != -1 )
        {
            Py_INCREF(m_pOtherInputs);
            PyTuple_SetItem(m_pUserParams, m_nOtherInputsIndex, m_pOtherInputs);
        }
    }
    ~SPDDataBlockProcessorPython()
    {
        Py_DECREF(m_pUserParams);
    }

    void processDataBlockImage(spdlib::SPDFile *inSPDFile, std::vector<spdlib::SPDPulse*> ***pulses, 
            float ***imageDataBlock, spdlib::SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, 
            boost::uint_fast32_t numImgBands, float binSize) throw(spdlib::SPDProcessingException)
    {
        // convert file into numpy array on the first call
        if( !m_bHaveCreatedFileArray )
        {
            PyObject *pFileArray = createSPDFileArray(inSPDFile);
            PyTuple_SetItem(m_pUserParams, FILE_INDEX, pFileArray);
            m_bHaveCreatedFileArray = true;
        }

        // convert data to numpy array
        PyObject *pPulseArray, *pPointArray;
        m_pulseConverter.convertCPPPulseArrayToRecArrays(pulses, xSize, ySize, &pPulseArray, &pPointArray);

        // grab the image data - wrap with a numpy array
        PyObject *pImageData;
        if( m_bHaveInputImage )
        {
            pImageData = ConvertFloatCPPArrayToNumpy(imageDataBlock, xSize, ySize, numImgBands);
        }
        else
        {
            // blank is fine
            npy_intp dims[] = {ySize, xSize, numImgBands};
            pImageData = PyArray_SimpleNew(3, dims, NPY_FLOAT);
        }

        // set the various elements in the tuple
        // note PyTuple_SetItem steals a reference and 
        // presumeably decrefs anything already there
        PyTuple_SetItem(m_pUserParams, PULSES_INDEX, pPulseArray);
        PyTuple_SetItem(m_pUserParams, POINTS_INDEX, pPointArray);
        PyTuple_SetItem(m_pUserParams, m_nImageDataIndex, pImageData);
        if( m_nCenPtsIndex != -1 )
        {
            PyObject *pCenPtsData = ConvertCenPtsCPPArrayToNumpy(cenPts, xSize, ySize);
            PyTuple_SetItem(m_pUserParams, m_nCenPtsIndex, pCenPtsData);
        }
        // otherinputs set in constructor

        // call python function 
        PyObject *pResult = PyObject_CallObject(m_pApplyFn, m_pUserParams);

        if( pResult == NULL )
        {
            // error
            PyErr_Print();
            throw spdlib::SPDProcessingException("Python error");
        }
        Py_DECREF(pResult);

        // copy data back
        if( m_bHaveOutputSPD )
        {
            m_pulseConverter.convertRecArraysToCPPPulseArray(pPulseArray, pPointArray, pulses);
        }
        if( m_bHaveOutputImage )
        {
            ConvertNumpyToFloatCPPArray(imageDataBlock, xSize, ySize, numImgBands, pImageData);
        }
    }

    void processDataBlock(spdlib::SPDFile *inSPDFile, std::vector<spdlib::SPDPulse*> ***pulses, 
            spdlib::SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, boost::uint_fast32_t ySize, 
            float binSize) throw(spdlib::SPDProcessingException)
    {
        // convert file into numpy array on the first call
        if( !m_bHaveCreatedFileArray )
        {
            PyObject *pFileArray = createSPDFileArray(inSPDFile);
            PyTuple_SetItem(m_pUserParams, FILE_INDEX, pFileArray);
            m_bHaveCreatedFileArray = true;
        }

        // convert data to numpy array
        PyObject *pPulseArray, *pPointArray;
        m_pulseConverter.convertCPPPulseArrayToRecArrays(pulses, xSize, ySize, &pPulseArray, &pPointArray);

        // set the various elements in the tuple
        // note PyTuple_SetItem steals a reference and 
        // presumeably decrefs anything already there
        PyTuple_SetItem(m_pUserParams, PULSES_INDEX, pPulseArray);
        PyTuple_SetItem(m_pUserParams, POINTS_INDEX, pPointArray);
        if( m_nCenPtsIndex != -1 )
        {
            PyObject *pCenPtsData = ConvertCenPtsCPPArrayToNumpy(cenPts, xSize, ySize);
            PyTuple_SetItem(m_pUserParams, m_nCenPtsIndex, pCenPtsData);
        }
        // otherinputs set in constructor

        // call python function 
        PyObject *pResult = PyObject_CallObject(m_pApplyFn, m_pUserParams);

        if( pResult == NULL )
        {
            // error
            PyErr_Print();
            throw spdlib::SPDProcessingException("Python error");
        }
        Py_DECREF(pResult);

        // copy data back
        if( m_bHaveOutputSPD )
        {
            m_pulseConverter.convertRecArraysToCPPPulseArray(pPulseArray, pPointArray, pulses);
        }
    }

    void processDataBlockImage(spdlib::SPDFile *inSPDFile, std::vector<spdlib::SPDPulse*> *pulses, 
            float ***imageDataBlock, spdlib::SPDXYPoint ***cenPts, boost::uint_fast32_t xSize, 
            boost::uint_fast32_t ySize, boost::uint_fast32_t numImgBands) throw(spdlib::SPDProcessingException)
    {
        throw spdlib::SPDProcessingException("Not implemented as not required.");
    }

    void processDataBlock(spdlib::SPDFile *inSPDFile, std::vector<spdlib::SPDPulse*> *pulses) throw(spdlib::SPDProcessingException)
    {
        throw spdlib::SPDProcessingException("Not implemented as not required.");
    }

    std::vector<std::string> getImageBandDescriptions() throw(spdlib::SPDProcessingException)
    {
        // TODO
        return std::vector<std::string>();
    }

    void setHeaderValues(spdlib::SPDFile *spdFile) throw(spdlib::SPDProcessingException)
    {
        // TODO
    }
private:
    PyObject *m_pApplyFn;
    PyObject *m_pOtherInputs;
    bool m_bHaveOutputSPD;
    bool m_bHaveInputImage;
    bool m_bHaveOutputImage;
    PulsePointConverter m_pulseConverter;
    Py_ssize_t m_nImageDataIndex;
    Py_ssize_t m_nCenPtsIndex;
    Py_ssize_t m_nOtherInputsIndex;
    PyObject *m_pUserParams;
    bool m_bHaveCreatedFileArray;
};

static PyObject *
spdpy2_blockProcessor(PyObject *self, PyObject *args)
{
    PyObject *pApplyFn, *pControls, *pOtherInputs;
    const char *pszInputSPDFile, *pszInputImageFile, *pszOutputSPDFile, *pszOutputImageFile;
    int nPassCenPts;
    if( !PyArg_ParseTuple(args, "OszzzOiO", &pApplyFn, &pszInputSPDFile, &pszInputImageFile, 
                &pszOutputSPDFile, &pszOutputImageFile, &pControls, &nPassCenPts, &pOtherInputs))
        return NULL;

    // get the control values
    PyObject *pVal = PyObject_GetAttrString(pControls, "overlap");
#if PY_MAJOR_VERSION >= 3
    if( ( pVal == NULL ) || !PyLong_Check(pVal) )
#else
    if( ( pVal == NULL ) || !PyInt_Check(pVal) )
#endif
    {
        PyErr_SetString(GETSTATE(self)->error, "Controls object must have an int overlap field");
        Py_XDECREF(pVal);
        return NULL;
    }
#if PY_MAJOR_VERSION >= 3
    boost::uint_fast32_t overlap = PyLong_AsLong(pVal);
#else
    boost::uint_fast32_t overlap = PyInt_AsLong(pVal);
#endif
    Py_DECREF(pVal);

    pVal = PyObject_GetAttrString(pControls, "blockXSize");
#if PY_MAJOR_VERSION >= 3
    if( ( pVal == NULL ) || !PyLong_Check(pVal) )
#else
    if( ( pVal == NULL ) || !PyInt_Check(pVal) )
#endif
    {
        PyErr_SetString(GETSTATE(self)->error, "Controls object must have an int blockXSize field");
        Py_XDECREF(pVal);
        return NULL;
    }
#if PY_MAJOR_VERSION >= 3
    boost::uint_fast32_t blockXSize = PyLong_AsLong(pVal);
#else
    boost::uint_fast32_t blockXSize = PyInt_AsLong(pVal);
#endif
    Py_DECREF(pVal);

    pVal = PyObject_GetAttrString(pControls, "blockYSize");
#if PY_MAJOR_VERSION >= 3
    if( ( pVal == NULL ) || !PyLong_Check(pVal) )
#else
    if( ( pVal == NULL ) || !PyInt_Check(pVal) )
#endif
    {
        PyErr_SetString(GETSTATE(self)->error, "Controls object must have an int blockYSize field");
        Py_XDECREF(pVal);
        return NULL;
    }
#if PY_MAJOR_VERSION >= 3
    boost::uint_fast32_t blockYSize = PyLong_AsLong(pVal);
#else
    boost::uint_fast32_t blockYSize = PyInt_AsLong(pVal);
#endif
    Py_DECREF(pVal);

    pVal = PyObject_GetAttrString(pControls, "printProgress");
    if( ( pVal == NULL ) || !PyBool_Check(pVal) )
    {
        PyErr_SetString(GETSTATE(self)->error, "Controls object must have a bool printProgress field");
        Py_XDECREF(pVal);
        return NULL;
    }
    bool printProgress = (pVal == Py_True);
    Py_DECREF(pVal);    

    pVal = PyObject_GetAttrString(pControls, "keepMinExtent");
    if( ( pVal == NULL ) || !PyBool_Check(pVal) )
    {
        PyErr_SetString(GETSTATE(self)->error, "Controls object must have a bool keepMinExtent field");
        Py_XDECREF(pVal);
        return NULL;
    }
    bool keepMinExtent = (pVal == Py_True);
    Py_DECREF(pVal);    

    pVal = PyObject_GetAttrString(pControls, "processingResolution");
#if PY_MAJOR_VERSION >= 3
    if( ( pVal == NULL ) || (!PyFloat_Check(pVal) && !PyLong_Check(pVal) ) )
#else
    if( ( pVal == NULL ) || (!PyFloat_Check(pVal) && !PyInt_Check(pVal) ) )
#endif
    {
        PyErr_SetString(GETSTATE(self)->error, "Controls object must have an int or float processingResolution field");
        Py_XDECREF(pVal);
        return NULL;
    }
    float processingResolution = PyFloat_AS_DOUBLE(pVal);
    Py_DECREF(pVal);

    pVal = PyObject_GetAttrString(pControls, "numImgBands");
#if PY_MAJOR_VERSION >= 3
    if( ( pVal == NULL ) || !PyLong_Check(pVal) )
#else
    if( ( pVal == NULL ) || !PyInt_Check(pVal) )
#endif
    {
        PyErr_SetString(GETSTATE(self)->error, "Controls object must have an int numImgBands field");
        Py_XDECREF(pVal);
        return NULL;
    }
#if PY_MAJOR_VERSION >= 3
    boost::uint_fast16_t numImgBands = PyLong_AsLong(pVal);
#else
    boost::uint_fast16_t numImgBands = PyInt_AsLong(pVal);
#endif
    Py_DECREF(pVal);

    pVal = PyObject_GetAttrString(pControls, "gdalFormat");
#if PY_MAJOR_VERSION >= 3
    if( ( pVal == NULL ) || !PyUnicode_Check(pVal) )
#else
    if( ( pVal == NULL ) || !PyString_Check(pVal) )
#endif    
    {
        PyErr_SetString(GETSTATE(self)->error, "controls object must have an string gdalFormat field");
        Py_XDECREF(pVal);
        return NULL;
    }
#if PY_MAJOR_VERSION >= 3
    PyObject *bytes = PyUnicode_AsEncodedString(pVal, NULL, NULL);
    char *gdalFormat = PyBytes_AsString(bytes);
    Py_DECREF(bytes);
#else   
    char *gdalFormat = PyString_AsString(pVal);
#endif
    Py_DECREF(pVal);

    // check combo makes sense
    if( pszInputSPDFile == NULL )
    {
        PyErr_SetString(GETSTATE(self)->error, "outputs object must have a valid SDPFile");
        return NULL;
    }
    if( ( pszInputImageFile != NULL ) && ( pszOutputImageFile != NULL ) )
    {
        PyErr_SetString(GETSTATE(self)->error, "can't have an input and output image file");
        return NULL;
    }

    spdlib::SPDFile *spdInFile = NULL;
    spdlib::SPDDataBlockProcessor *blockProcessor = NULL;
    try
    {
        spdInFile = new spdlib::SPDFile(pszInputSPDFile);
        blockProcessor = new SPDDataBlockProcessorPython(pApplyFn, pOtherInputs, pszOutputSPDFile != NULL,
                            pszInputImageFile != NULL, pszOutputImageFile != NULL, nPassCenPts != 0);

        spdlib::SPDProcessDataBlocks processBlocks = spdlib::SPDProcessDataBlocks(blockProcessor, 
            overlap, blockXSize, blockYSize, printProgress, keepMinExtent);

        if( ( pszOutputSPDFile != NULL ) && ( pszInputImageFile == NULL ) && ( pszOutputImageFile == NULL ) )
        {
            processBlocks.processDataBlocksGridPulsesOutputSPD(spdInFile, pszOutputSPDFile, processingResolution);
        }
        else if( ( pszOutputSPDFile != NULL ) &&  ( pszInputImageFile != NULL ) )
        {
            processBlocks.processDataBlocksGridPulsesInputImage(spdInFile, pszOutputSPDFile, pszInputImageFile );
        }
        else if( ( pszOutputSPDFile == NULL ) && ( pszInputImageFile == NULL ) && ( pszOutputImageFile == NULL ) )
        {
            processBlocks.processDataBlocksGridPulses(spdInFile, processingResolution);
        }
        else if( ( pszOutputSPDFile == NULL ) && ( pszInputImageFile == NULL ) && ( pszOutputImageFile != NULL ) )
        {
            processBlocks.processDataBlocksGridPulsesOutputImage(spdInFile, pszOutputImageFile, 
                                processingResolution, numImgBands, gdalFormat);
        }
        else
        {
            throw spdlib::SPDProcessingException("Unsupported combination of input and output files");
        }

        delete blockProcessor;
        delete spdInFile;
    }
    catch (spdlib::SPDException &e)
    {
        PyErr_SetString(GETSTATE(self)->error, e.what());
        delete blockProcessor;
        delete spdInFile;
        return NULL;
    }
    catch (std::exception &e)
    {
        PyErr_SetString(GETSTATE(self)->error, e.what());
        delete blockProcessor;
        delete spdInFile;
        return NULL;
    }


    Py_RETURN_NONE;
}

static PyMethodDef module_methods[] = {
    {"createPulseArray", (PyCFunction)spdpy2_createPulseArray, METH_VARARGS,
        "create a Pulse array. Pass the required shape of the array."},
    {"createPointArray", (PyCFunction)spdpy2_createPointArray, METH_VARARGS,
        "create a Point array. Pass the required shape of the array."},
    {"blockProcessor", (PyCFunction)spdpy2_blockProcessor, METH_VARARGS,
        "For use by the applier module. Used SPDDataBlockProcessor to process blocks"},
    {NULL}  /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3

static int spdpy2_traverse(PyObject *m, visitproc visit, void *arg) 
{
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int spdpy2_clear(PyObject *m) 
{
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_spdpy2",
        NULL,
        sizeof(struct SPDPy2State),
        module_methods,
        NULL,
        spdpy2_traverse,
        spdpy2_clear,
        NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC 
PyInit__spdpy2(void)

#else
#define INITERROR return

PyMODINIT_FUNC
init_spdpy2(void) 
#endif
{
    // initialize the numpy/recaray stuff
    import_array();
    recarray_init();
    pulsearray_init();
    pointarray_init();

    PyObject* m;

#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule3("_spdpy2", module_methods,
                       "New Generation Python Bindings for SPDLib");
#endif
    if( m == NULL )
        INITERROR;

    struct SPDPy2State *state = GETSTATE(m);

    state->error = PyErr_NewException((char*)"_spdpy2.error", NULL, NULL);
    if( state->error == NULL )
    {
        Py_DECREF(m);
        INITERROR;
    }

    pyspdfile_init(m, state->error);

#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}


