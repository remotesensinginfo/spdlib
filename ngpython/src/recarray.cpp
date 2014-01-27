/*
 *  recarray.cpp
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

#include "recarray.h"

PyMODINIT_FUNC recarray_init()
{
    // this must be called per .cpp file to init function pointers etc
    import_array();
#if PY_MAJOR_VERSION >= 3
    return NULL;
#endif
}

// constructor
RecArrayCreator::RecArrayCreator()
{
    m_DTypeList = NULL;
    m_pDescr = NULL;
}

RecArrayCreator::~RecArrayCreator()
{
    // destroy our list
    Py_XDECREF(m_DTypeList);
    // description - this causes a crash. Not sure if it needs to be deallocated...
    //Py_XDECREF(m_pDescr);
}

// Adds a field to our list for creation of array
// eType should be one of the values from ndarraytypes.h
// note not all types supported - you may have to use the other
// flavour of addField for unusual types - notably strings.
void RecArrayCreator::addField(const char *pszName, NPY_TYPES eType, int nLength) throw(RecArrayException)
{
    char cKind;
    int nBytes;
    // convert to kind and size
    switch(eType)
    {
        case NPY_BOOL:
            cKind = 'b';
            nBytes = 1;
            break;
        case NPY_BYTE:
            cKind = 'i';
            nBytes = 1;
            break;
        case NPY_UBYTE:
            cKind = 'u';
            nBytes = 1;
            break;
        case NPY_SHORT:
            cKind = 'i';
            nBytes = 2;
            break;
        case NPY_USHORT:
            cKind = 'u';
            nBytes = 2;
            break;
        case NPY_INT:
            cKind = 'i';
            nBytes = 4;
            break;
        case NPY_UINT:
            cKind = 'u';
            nBytes = 4;
            break;
        case NPY_LONG:
            cKind = 'i';
            nBytes = 8;
            break;
        case NPY_ULONG:
            cKind = 'u';
            nBytes = 8;
            break;
        case NPY_FLOAT:
            cKind = 'f';
            nBytes = 4;
            break;
        case NPY_DOUBLE:
            cKind = 'f';
            nBytes = 8;
            break;
        default:
            throw RecArrayException("Data type not supported");
            break;
    }
    // now we have the kind and bytes call the other method
    addField(pszName, cKind, nBytes, nLength);
}

// Adds a field to our list for creation of array
// takes kind and bytes description. 
// see http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html "Array-protocol type strings"
// set nLength to values greater than 1 for sub arrays
void RecArrayCreator::addField(const char *pszName, char cKind, int nBytes, int nLength) throw(RecArrayException)
{
    // if we don't have a list yet, create one
    if( m_DTypeList == NULL )
    {
        m_DTypeList = PyList_New(0);
    }

    // create a new tuple for this field
    PyObject *pTuple = PyTuple_New(3);

    // first element of tuple is the name
#if PY_MAJOR_VERSION >= 3
    PyObject *pValue = PyUnicode_FromString(pszName);
#else
    PyObject *pValue = PyString_FromString(pszName);
#endif
    PyTuple_SetItem(pTuple, 0, pValue);

    // second is the type string
    char szCode[10];
    sprintf( szCode, "%c%d", cKind, nBytes );
#if PY_MAJOR_VERSION >= 3
    pValue = PyUnicode_FromString(szCode);
#else
    pValue = PyString_FromString(szCode);
#endif
    PyTuple_SetItem(pTuple, 1, pValue);

    // third is the length - if 1 it is ignored
#if PY_MAJOR_VERSION >= 3
    pValue = PyLong_FromLong(nLength);
#else
    pValue = PyInt_FromLong(nLength);
#endif
    PyTuple_SetItem(pTuple, 2, pValue);

    // append this tuple to the list
    PyList_Append(m_DTypeList, pTuple);
}

// create the array with fields as specified by addField()
PyObject *RecArrayCreator::createArray(int nd, npy_intp *dims)
{
    // Convert to a PyArray_Descr object
    // see http://stackoverflow.com/questions/214549/how-to-create-a-numpy-record-array-from-c
    // this assumes once they call here they have finished adding fields...
    if( m_pDescr == NULL )
    {
        if( !PyArray_DescrConverter(m_DTypeList, &m_pDescr) )
        {
            throw RecArrayException("Unable to convert array description");
        }
    }

    // create a 1-d array with this descr
    // steals ref to descr
    PyObject *pArray = PyArray_SimpleNewFromDescr(nd, dims, m_pDescr);
    if( pArray == NULL )
    {
        throw RecArrayException("Unable to create array");
    }

    // return the array - pass back to Python
    // or Py_DECREF when finished
    return pArray;
}

// overloaded function
PyObject *RecArrayCreator::createArray(npy_intp nLength)
{
    return createArray(1, &nLength);
}

//-----------------------------------------------------

// Helper function to get information about a named field within an array
void getFieldDescription(PyObject *pArray, const char *pszName, int *pnOffset, char *pcKind, int *pnSize, int *pnLength)throw(RecArrayException)
{
    if( ! PyArray_Check(pArray) )
    {
        throw RecArrayException("Must pass array type");
    }

    PyArray_Descr *pDescr = PyArray_DESCR(pArray);
    if( pDescr == NULL )
    {
        throw RecArrayException("Cannot get array description");
    }

    if( ( pDescr->byteorder != '|' ) && ( pDescr->byteorder != '=' ) )
    {
        throw RecArrayException("Cannot handle exotic byte order yet");
    }

    if( ( pDescr->fields == NULL ) || !PyDict_Check(pDescr->fields) )
    {
        throw RecArrayException("Cannot obtain the fields");
    }

    // go through each of the fields looking for the right name
    // see http://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html
    // "this data-type-descriptor has fields described by a Python dictionary whose keys 
    // are names (and also titles if given) and whose values are tuples that describe the fields."
    // "A field is described by a tuple composed of another data- type-descriptor and a byte offset"
    PyObject *pKey, *pValue;
    Py_ssize_t pos = 0;
    bool bFound = false;
    while( PyDict_Next(pDescr->fields, &pos, &pKey, &pValue) )
    {
#if PY_MAJOR_VERSION >= 3
        PyObject *bytesKey = PyUnicode_AsEncodedString(pKey, NULL, NULL);
        char *pszElementName = PyBytes_AsString(bytesKey);
#else
        char *pszElementName = PyString_AsString(pKey);
#endif
        if( strcmp( pszElementName, pszName ) == 0 )
        {
            // matches
            bFound = true;
            // byte offset
            PyObject *pOffset = PyTuple_GetItem(pValue, 1);
            // description
            PyArray_Descr *pSubDescr = (PyArray_Descr*)PyTuple_GetItem(pValue, 0);
            if( pSubDescr != NULL )
            {
                if( ( pSubDescr->kind == 'V' ) && ( pSubDescr->subarray != NULL ) )
                {
                    // is a sub array
#if PY_MAJOR_VERSION >= 3
                    *pnOffset = PyLong_AsLong(pOffset);
#else
                    *pnOffset = PyInt_AsLong(pOffset);
#endif
                    *pcKind = pSubDescr->subarray->base->kind;
                    *pnSize = pSubDescr->subarray->base->elsize;
                    if( PyTuple_Size(pSubDescr->subarray->shape) != 1 )
                    {
                        throw RecArrayException("Can only handle 1-d sub arrays");
                    }
#if PY_MAJOR_VERSION >= 3
                    *pnLength = PyLong_AsLong( PyTuple_GetItem(pSubDescr->subarray->shape, 0) );
#else
                    *pnLength = PyInt_AsLong( PyTuple_GetItem(pSubDescr->subarray->shape, 0) );
#endif
                }
                else
                {
                    // is a single item
#if PY_MAJOR_VERSION >= 3
                    *pnOffset = PyLong_AsLong(pOffset);
#else
                    *pnOffset = PyInt_AsLong(pOffset);
#endif
                    *pcKind = pSubDescr->kind;
                    *pnSize = pSubDescr->elsize;
                    *pnLength = 1;
                }
            }
#if PY_MAJOR_VERSION >= 3
            Py_DECREF(bytesKey);
#endif
            break;
        }
#if PY_MAJOR_VERSION >= 3
        Py_DECREF(bytesKey);
#endif
    }
    if( !bFound )
    {
        throw RecArrayException("Couldn't find field");
    }
}
