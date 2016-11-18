/*
 *  performNNInterp.c
 *
 *
 * This file is part of SPDLib; although earlier version was created for pylidar
 * Copyright (C) 2016 Pete Bunting
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

#include "spd/SPDException.h"

#include <iostream>
#include <math.h>

#include "spd/nn/nn.h"


namespace spdlib{ namespace nn{

    void interpNaturalNeighbour(double *ptsX, double *ptsY, double *ptsZ, unsigned long nPts, double *outXPts, double *outYPts, double *outZPts, unsigned long nOutPts)
    {
       
        if( nPts < 100 )
        {
            // check that these small number of points aren't all within a line
            double meanX = 0;
            double meanY = 0;
            
            for(unsigned long i = 0; i < nPts; ++i)
            {
                meanX += ptsX[i];
                meanY += ptsY[i];
            }
            
            meanX = meanX / nPts;
            meanY = meanY / nPts;
            
            double varX = 0;
            double varY = 0;
            
            for(unsigned long i = 0; i < nPts; ++i)
            {
                varX += ptsX[i] - meanX;
                varY += ptsY[i] - meanY;
            }
            
            varX = fabs(varX / nPts);
            varY = fabs(varY / nPts);
            
            if((varX < 4) || (varY < 4))
            {
                //throw spdlib::throw SPDProcessingException("Points are all within a line.");
            }
        }
        
        point *inPts, *gPts;
        // BUILD POINT ARRAYS
        inPts = new point[nPts];//malloc(nPts * sizeof(point));
        for(unsigned long i = 0; i < nPts; ++i)
        {
            inPts[i].x = ptsX[i];
            inPts[i].y = ptsY[i];
            inPts[i].z = ptsZ[i];
        }
        
        gPts = (point*) malloc(nOutPts * sizeof(point));
        for(unsigned long i = 0; i < nOutPts; ++i)
        {
            gPts[i].x = outXPts[i];
            gPts[i].y = outYPts[i];
            gPts[i].z = 0.0;
        }
        
        float minWeight = 0.0;
        nnpi_interpolate_points(nPts, inPts, minWeight, nOutPts, gPts);
        
        delete[] inPts;
        
        for(unsigned long i = 0; i < nOutPts; ++i)
        {
            outZPts[i] = gPts[i].z;
        }
        
        free(gPts);    
    }

    /*
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

    */
}}

