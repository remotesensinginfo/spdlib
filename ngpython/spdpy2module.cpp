
#include <Python.h>
#include "structmember.h"

#include "pyspdfile.h"
#include "pulsearray.h"
#include "pointarray.h"

/* An exception object for this module */
/* created in the init function */
/* used accross the other source files */
PyObject *PySPDError;

static PyObject *
spdpy2_createPulseArray(PyObject *self, PyObject *args)
{
    PyObject *pObj;
    if( !PyArg_ParseTuple(args, "O", &pObj))
        return NULL;

    RecArrayCreator creator;
    addPulseFields(&creator);
    PyObject *pArray = NULL;
    if( PyInt_Check(pObj) )
    {
        pArray = creator.createArray(PyInt_AsLong(pObj));
    }
    else if( PySequence_Check(pObj))
    {
        int nd = PySequence_Size(pObj);
        npy_intp *dims = new npy_intp[nd];
        for( int n = 0; n < nd; n++ )
        {
            PyObject *pElement = PySequence_GetItem(pObj, n);
            if( !PyInt_Check(pElement))
            {
                PyErr_SetString(PySPDError, "sequence must be all ints");
                delete dims;
                Py_DECREF(pElement);
                return NULL;
            }
            dims[n] = PyInt_AsLong(pElement);
            Py_DECREF(pElement);
        }
        pArray = creator.createArray(nd, dims);
        delete dims;
    }
    else
    {
        PyErr_SetString(PySPDError, "expected an int or a sequence");
        return NULL;
    }

    return pArray;
}

static PyMethodDef module_methods[] = {
    {"createPulseArray", (PyCFunction)spdpy2_createPulseArray, METH_VARARGS,
        "create a Pulse array. Pass the required shape of the array."},
    {NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initspdpy2(void) 
{
    // initialize the numpy/recaray stuff
    import_array();
    recarray_init();
    pulsearray_init();
    pointarray_init();

    PyObject* m;

    m = Py_InitModule3("spdpy2", module_methods,
                       "New Generation Python Bindings for SPDLib");

    PySPDError = PyErr_NewException((char*)"spdpy2.error", NULL, NULL);
    Py_INCREF(PySPDError);
    PyModule_AddObject(m, "error", PySPDError);

    pyspdfile_init(m);
}


