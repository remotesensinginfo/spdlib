
#include "recarray.h"

void recarray_init()
{
    // this must be called per .cpp file to init function pointers etc
    import_array();
}

// constructor
RecArrayCreator::RecArrayCreator()
{
    m_DTypeList = NULL;
}

RecArrayCreator::~RecArrayCreator()
{
    // destroy our list
    Py_XDECREF(m_DTypeList);
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
    PyObject *pValue = PyString_FromString(pszName);
    PyTuple_SetItem(pTuple, 0, pValue);

    // second is the type string
    char szCode[10];
    sprintf( szCode, "%c%d", cKind, nBytes );
    pValue = PyString_FromString(szCode);
    PyTuple_SetItem(pTuple, 1, pValue);

    // third is the length - if 1 it is ignored
    pValue = PyInt_FromLong(nLength);
    PyTuple_SetItem(pTuple, 2, pValue);

    // append this tuple to the list
    PyList_Append(m_DTypeList, pTuple);
}

// create the array with fields as specified by addField()
PyObject *RecArrayCreator::createArray(int nd, npy_intp *dims)
{
    // Convert to a PyArray_Descr object
    // see http://stackoverflow.com/questions/214549/how-to-create-a-numpy-record-array-from-c
    PyArray_Descr *pDescr;
    if( !PyArray_DescrConverter(m_DTypeList, &pDescr) )
    {
        throw RecArrayException("Unable to convert array description");
    }

    // create a 1-d array with this descr
    // steals ref to descr
    PyObject *pArray = PyArray_SimpleNewFromDescr(nd, dims, pDescr);
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
        char *pszElementName = PyString_AsString(pKey);
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
                    *pnOffset = PyInt_AsLong(pOffset);
                    *pcKind = pSubDescr->subarray->base->kind;
                    *pnSize = pSubDescr->subarray->base->elsize;
                    if( PyTuple_Size(pSubDescr->subarray->shape) != 1 )
                    {
                        throw RecArrayException("Can only handle 1-d sub arrays");
                    }
                    *pnLength = PyInt_AsLong( PyTuple_GetItem(pSubDescr->subarray->shape, 0) );
                }
                else
                {
                    // is a single item
                    *pnOffset = PyInt_AsLong(pOffset);
                    *pcKind = pSubDescr->kind;
                    *pnSize = pSubDescr->elsize;
                    *pnLength = 1;
                }
            }
            break;
        }
    }
    if( !bFound )
    {
        throw RecArrayException("Couldn't find field");
    }
}
