
#ifndef RECARRAY_H
#define RECARRAY_H

#include <Python.h>
#include "numpy/arrayobject.h"

#include <exception>
#include <string>
#include <utility>
#include <vector>

// exception class
class RecArrayException : public std::exception
{
public:
    RecArrayException() : exception() {msgs = "A RecArrayException has been created";};
    RecArrayException(const char *message) : exception() {msgs = std::string(message);};
    RecArrayException(std::string message): exception() {msgs = message;};
    virtual ~RecArrayException() throw() {};
    virtual const char* what() const throw() {return msgs.c_str();}
protected:
    std::string msgs;
};

// must call this in module initilisation function
// so we have full access to numpy API
PyMODINIT_FUNC recarray_init();

// class for creating structred arrays
class RecArrayCreator
{
public:
    RecArrayCreator();
    ~RecArrayCreator();

    // add a field. Either by enum
    void addField(const char *pszName, NPY_TYPES eType, int nLength=1) throw(RecArrayException);
    // or kind and n bytes
    void addField(const char *pszName, char cKind, int nBytes, int nLength=1) throw(RecArrayException);

    // create a 1-d array of given size
    PyObject *createArray(npy_intp nLength);
    // create the array of given shape
    PyObject *createArray(int nd, npy_intp *dims);

private:
    PyObject *m_DTypeList;
};

// helper method used by RecArrayField to get info about a field
void getFieldDescription(PyObject *pArray, const char *pszName, int *pnOffset, char *pcKind, int *pnSize, int *pnLength)throw(RecArrayException);

// template class to access a field. Should set as type one of the types from npy_common.h:
// npy_byte, npy_ubyte, npy_ushort, npy_uint, npy_ulong, npy_float, npy_double, npy_short, npy_int, npy_long, npy_bool
// a simple check is done to see the size matches the field

template <class T>
class RecArrayField
{
public:
    // constructor
    RecArrayField(PyObject *pArray, const char *pszName)throw(RecArrayException)
    {
        setField(pArray, pszName);
    }
    RecArrayField()
    {
        m_nOffset = -1;
        m_cKind = '\0';
        m_nSize = -1;
        m_nLength = -1;
    }

    void setField(PyObject *pArray, const char *pszName)throw(RecArrayException)
    {
        // store info on the field
        getFieldDescription(pArray, pszName, &m_nOffset, &m_cKind, &m_nSize, &m_nLength);
        // do a simple check - can be easily fooled
        if( sizeof(T) != m_nSize )
        {
            throw RecArrayException("size mismatch");
        }
    }

    // set a scalar value into the field
    void setValue(void *pRow, T data)throw(RecArrayException)
    {
        if( m_nLength != 1 )
        {
            throw RecArrayException("should use setValueArray instead");
        }
        memcpy( (char*)pRow + m_nOffset, &data, m_nSize );
    }
    // return a scalar value
    T getValue(void *pRow)throw(RecArrayException)
    {
        if( m_nLength != 1 )
        {
            throw RecArrayException("should use getValueArray instead");
        }
        T val;
        memcpy( &val, (char*)pRow + m_nOffset, m_nSize );
        return val;
    }
    // set an sub array 
    void setValueArray(void *pRow, T *pData)
    {
        memcpy( (char*)pRow + m_nOffset, pData, m_nSize * m_nLength );
    }
    // copy a subarray into pData - must be the right size!
    void getValueArray(void *pRow, T *pData)
    {
        // could just return pointer but worried about
        // non aligned memory access on SPARC, ARM etc
        memcpy( pData, (char*)pRow + m_nOffset, m_nSize * m_nLength );
    }

    int getOffset() { return m_nOffset; }
    char getKind()  { return m_cKind; }
    int getSize()   { return m_nSize; }
    int getLength() { return m_nLength; }

private:
    int m_nOffset;
    char m_cKind;
    int m_nSize;
    int m_nLength;
};

#endif // RECARRAY_H_
