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

static PyMethodDef module_methods[] = {
    {"createPulseArray", (PyCFunction)spdpy2_createPulseArray, METH_VARARGS,
        "create a Pulse array. Pass the required shape of the array."},
    {"createPointArray", (PyCFunction)spdpy2_createPointArray, METH_VARARGS,
        "create a Point array. Pass the required shape of the array."},
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


