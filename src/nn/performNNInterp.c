/*
 *  performNNInterp.c
 *
 *
 * This file is part of PyLidar
 * Copyright (C) 2015 John Armston, Pete Bunting, Neil Flood, Sam Gillingham
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

/*

#include <math.h>

#include <Python.h>
#include "numpy/arrayobject.h"

#include "spd/nn/nn.h"

 // An exception object for this module
 // created in the init function
struct PyNNInterpState
{
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct PyNNInterpState*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct PyNNInterpState _state;
#endif

static PyObject *pynninterp_naturalneighbour(PyObject *self, PyObject *args)
{
    //std::cout.precision(12);
    PyObject *pXVals, *pYVals, *pZVals, *pXGrid, *pYGrid;
    PyObject *pOutArray;
    npy_intp nRows, nCols, nVals, i, j, nPtsOutGrid, idx;
    point *inPts, *gPts;
    double minWeight, meanX, meanY, varX, varY;
    
    if( !PyArg_ParseTuple(args, "OOOOO:NaturalNeighbour", &pXVals, &pYVals, &pZVals, &pXGrid, &pYGrid))
        return NULL;
    
    if( !PyArray_Check(pXVals) || !PyArray_Check(pYVals) || !PyArray_Check(pZVals) || !PyArray_Check(pXGrid) || !PyArray_Check(pYGrid) )
    {
        PyErr_SetString(GETSTATE(self)->error, "All arguments must be numpy arrays");
        return NULL;
    }
    
    // Check dimensions match
    if( (PyArray_DIM(pXVals, 0) != PyArray_DIM(pYVals, 0)) | (PyArray_DIM(pXVals, 0) != PyArray_DIM(pZVals, 0)))
    {
        PyErr_SetString(GETSTATE(self)->error, "Training X, Y and Z arrays must all be of the same length");
        return NULL;
    }
    
    if( (PyArray_DIM(pXGrid, 0) != PyArray_DIM(pYGrid, 0)) | (PyArray_DIM(pXGrid, 1) != PyArray_DIM(pYGrid, 1)))
    {
        PyErr_SetString(GETSTATE(self)->error, "X and Y grids must have the same dimensions");
        return NULL;
    }
    
    // TODO: check types ok
    
    nRows = PyArray_DIM(pXGrid, 0);
    nCols = PyArray_DIM(pXGrid, 1);
    
    nVals = PyArray_DIM(pXVals, 0);
    
    // Create output
    pOutArray = PyArray_EMPTY(2, PyArray_DIMS(pXGrid), NPY_DOUBLE, 0);
    if( pOutArray == NULL )
    {
        PyErr_SetString(GETSTATE(self)->error, "Failed to create array");
        return NULL;
    }
    
    if( PyArray_DIM(pXVals, 0) < 3 )
    {
        PyErr_SetString(GETSTATE(self)->error, "Not enough points, need at least 3.");
        return NULL;
    }
    
    i = 0;
    j = 0;
    if( nVals < 100 )
    {
        // check that these small number of points aren't all within a line
        meanX = 0;
        meanY = 0;
        
        varX = 0;
        varY = 0;
        
        for(i = 0; i < nVals; ++i)
        {
            meanX += *((double*)PyArray_GETPTR1(pXVals, i));
            meanY += *((double*)PyArray_GETPTR1(pYVals, i));
        }
        
        meanX = meanX / nVals;
        meanY = meanY / nVals;
        
        for(i = 0; i < nVals; ++i)
        {
            varX += *((double*)PyArray_GETPTR1(pXVals, i)) - meanX;
            varY += *((double*)PyArray_GETPTR1(pYVals, i)) - meanY;
        }
        
        varX = fabs(varX / nVals);
        varY = fabs(varY / nVals);
        
        if((varX < 4) || (varY < 4))
        {
            PyErr_SetString(GETSTATE(self)->error, "Points are all within a line.");
            return NULL;
        }
    }
    
    // BUILD POINT ARRAYS
    inPts = malloc(nVals * sizeof(point));
    for(i = 0; i < nVals; ++i)
    {
        inPts[i].x = *((double*)PyArray_GETPTR1(pXVals, i));
        inPts[i].y = *((double*)PyArray_GETPTR1(pYVals, i));
        inPts[i].z = *((double*)PyArray_GETPTR1(pZVals, i));
    }
    
    nPtsOutGrid = nRows * nCols;
    idx = 0;
    gPts = malloc(nPtsOutGrid * sizeof(point));
    for(i = 0; i < nRows; ++i)
    {
        for(j = 0; j < nCols; ++j)
        {
            gPts[idx].x = *((double*)PyArray_GETPTR2(pXGrid, i, j));
            gPts[idx].y = *((double*)PyArray_GETPTR2(pYGrid, i, j));
            gPts[idx].z = 0.0;
            ++idx;
        }
    }
    
    minWeight = 0.0;
    nnpi_interpolate_points(nVals, inPts, minWeight, nPtsOutGrid, gPts);
    
    free(inPts);
    
    
    // POPULATE GRID
    idx = 0;
    for(i  = 0; i < nRows; ++i)
    {
        for(j = 0; j < nCols; ++j)
        {
            *((double*)PyArray_GETPTR2(pOutArray, i, j)) = gPts[idx++].z;
        }
    }
    
    free(gPts);
    
    return pOutArray;
}


static PyObject *pynninterp_linear(PyObject *self, PyObject *args)
{
    //std::cout.precision(12);
    PyObject *pXVals, *pYVals, *pZVals, *pXGrid, *pYGrid;
    PyObject *pOutArray;
    npy_intp nRows, nCols, nVals, i, j, nPtsOutGrid, idx;
    point *inPts, *gPts;
    double minWeight, meanX, meanY, varX, varY;
    
    if( !PyArg_ParseTuple(args, "OOOOO:Linear", &pXVals, &pYVals, &pZVals, &pXGrid, &pYGrid))
        return NULL;
    
    if( !PyArray_Check(pXVals) || !PyArray_Check(pYVals) || !PyArray_Check(pZVals) || !PyArray_Check(pXGrid) || !PyArray_Check(pYGrid) )
    {
        PyErr_SetString(GETSTATE(self)->error, "All arguments must be numpy arrays");
        return NULL;
    }
    
    // Check dimensions match
    if( (PyArray_DIM(pXVals, 0) != PyArray_DIM(pYVals, 0)) | (PyArray_DIM(pXVals, 0) != PyArray_DIM(pZVals, 0)))
    {
        PyErr_SetString(GETSTATE(self)->error, "Training X, Y and Z arrays must all be of the same length");
        return NULL;
    }
    
    if( (PyArray_DIM(pXGrid, 0) != PyArray_DIM(pYGrid, 0)) | (PyArray_DIM(pXGrid, 1) != PyArray_DIM(pYGrid, 1)))
    {
        PyErr_SetString(GETSTATE(self)->error, "X and Y grids must have the same dimensions");
        return NULL;
    }
    
    // TODO: check types ok
    
    nRows = PyArray_DIM(pXGrid, 0);
    nCols = PyArray_DIM(pXGrid, 1);
    
    nVals = PyArray_DIM(pXVals, 0);
    
    // Create output
    pOutArray = PyArray_EMPTY(2, PyArray_DIMS(pXGrid), NPY_DOUBLE, 0);
    if( pOutArray == NULL )
    {
        PyErr_SetString(GETSTATE(self)->error, "Failed to create array");
        return NULL;
    }
    
    if( PyArray_DIM(pXVals, 0) < 3 )
    {
        PyErr_SetString(GETSTATE(self)->error, "Not enough points, need at least 3.");
        return NULL;
    }
    
    i = 0;
    j = 0;
    if( nVals < 100 )
    {
        // check that these small number of points aren't all within a line
        meanX = 0;
        meanY = 0;
        
        varX = 0;
        varY = 0;
        
        
        for(i = 0; i < nVals; ++i)
        {
            meanX += *((double*)PyArray_GETPTR1(pXVals, i));
            meanY += *((double*)PyArray_GETPTR1(pYVals, i));
        }
        
        meanX = meanX / nVals;
        meanY = meanY / nVals;
        
        for(i = 0; i < nVals; ++i)
        {
            varX += *((double*)PyArray_GETPTR1(pXVals, i)) - meanX;
            varY += *((double*)PyArray_GETPTR1(pYVals, i)) - meanY;
        }
        
        varX = fabs(varX / nVals);
        varY = fabs(varY / nVals);
        
        if((varX < 4) || (varY < 4))
        {
            PyErr_SetString(GETSTATE(self)->error, "Points are all within a line.");
            return NULL;
        }
    }
    
    // BUILD POINT ARRAYS
    inPts = malloc(nVals * sizeof(point));
    for(i = 0; i < nVals; ++i)
    {
        inPts[i].x = *((double*)PyArray_GETPTR1(pXVals, i));
        inPts[i].y = *((double*)PyArray_GETPTR1(pYVals, i));
        inPts[i].z = *((double*)PyArray_GETPTR1(pZVals, i));
    }
    
    nPtsOutGrid = nRows * nCols;
    idx = 0;
    gPts = malloc(nPtsOutGrid * sizeof(point));
    for(i  = 0; i < nRows; ++i)
    {
        for(j = 0; j < nCols; ++j)
        {
            gPts[idx].x = *((double*)PyArray_GETPTR2(pXGrid, i, j));
            gPts[idx].y = *((double*)PyArray_GETPTR2(pYGrid, i, j));
            gPts[idx].z = 0.0;
            ++idx;
        }
    }
    
    lpi_interpolate_points(nVals, inPts, nPtsOutGrid, gPts);
    
    free(inPts);
    
    
    // POPULATE GRID
    idx = 0;
    for(i  = 0; i < nRows; ++i)
    {
        for(j = 0; j < nCols; ++j)
        {
            *((double*)PyArray_GETPTR2(pOutArray, i, j)) = gPts[idx++].z;
        }
    }
    
    free(gPts);
    
    return pOutArray;
}

// Our list of functions in this module
static PyMethodDef PyNNInterpMethods[] = {
    {"NaturalNeighbour", pynninterp_naturalneighbour, METH_VARARGS,
        "Perform Natural Neighbour Interpolation\n"
        "call signature: arr = NaturalNeighbour(xvals, yvals, zvals, xgrid, ygrid)\n"
        "where:\n"
        "  xvals is a 1d array of the x values of the points\n"
        "  yvals is a 1d array of the y values of the points\n"
        "  zvals is a 1d array of the z values of the points\n"
        "xvals, yvals and zvals should have the same length\n"
        "  xgrid is a 2d array of x coordinates to interpolate at\n"
        "  ygrid is a 2d array of y coordinates to interpolate at\n"
        "xgrid and xgrid must be the same shape"},
    {"Linear", pynninterp_linear, METH_VARARGS,
        "Perform Linear (TIN) Interpolation\n"
        "call signature: arr = Linear(xvals, yvals, zvals, xgrid, ygrid)\n"
        "where:\n"
        "  xvals is a 1d array of the x values of the points\n"
        "  yvals is a 1d array of the y values of the points\n"
        "  zvals is a 1d array of the z values of the points\n"
        "xvals, yvals and zvals should have the same length\n"
        "  xgrid is a 2d array of x coordinates to interpolate at\n"
        "  ygrid is a 2d array of y coordinates to interpolate at\n"
        "xgrid and xgrid must be the same shape"},
    {NULL}        // Sentinel
};

#if PY_MAJOR_VERSION >= 3

static int pynninterp_traverse(PyObject *m, visitproc visit, void *arg)
{
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int pynninterp_clear(PyObject *m)
{
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "pynninterp",
    NULL,
    sizeof(struct PyNNInterpState),
    PyNNInterpMethods,
    NULL,
    pynninterp_traverse,
    pynninterp_clear,
    NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC
PyInit_pynninterp(void)

#else
#define INITERROR return

PyMODINIT_FUNC
initpynninterp(void)
#endif
{
    PyObject *pModule;
    struct PyNNInterpState *state;
    
    // initialize the numpy stuff
    import_array();
    
#if PY_MAJOR_VERSION >= 3
    pModule = PyModule_Create(&moduledef);
#else
    pModule = Py_InitModule("pynninterp", PyNNInterpMethods);
#endif
    if( pModule == NULL )
        INITERROR;
    
    state = GETSTATE(pModule);
    
    // Create and add our exception type
    state->error = PyErr_NewException("pynn.error", NULL, NULL);
    if( state->error == NULL )
    {
        Py_DECREF(pModule);
        INITERROR;
    }
    
#if PY_MAJOR_VERSION >= 3
    return pModule;
#endif
}
*/
