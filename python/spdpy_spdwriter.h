/*
 *  spdpy_spdwriter.h
 *
 *  Functions which provide access to SPD/UPD files
 *  from within Python.
 *
 *  Created by Pete Bunting on 26/02/2011.
 *  Copyright 2011 SPDLib. All rights reserved.
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
 */

#ifndef SPDPy_SPDWriter_H
#define SPDPy_SPDWriter_H

#include <iostream>
#include <string>
#include <list>
#include <vector>

#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/class.hpp>
#include <boost/python/init.hpp>
#include <boost/python/list.hpp>

#include "spd/SPDCommon.h"
#include "spd/SPDFile.h"
#include "spd/SPDPoint.h"
#include "spd/SPDPulse.h"
#include "spd/SPDException.h"
#include "spd/SPDFileWriter.h"

#include "spdpy_common.h"

using namespace spdlib;
using namespace std;

namespace spdlib_py
{
    class SPDPySeqWriter
    {
    public:
        SPDPySeqWriter(): readerOpen(false), spdWriter(NULL)
        {
            readerOpen = false;
            spdWriter = NULL;
        };

        bool open(SPDFile spdFile, string outputFile) throw(SPDException)
        {
            try
            {               
                spdFileOut = new SPDFile(outputFile);
                spdFileOut->copyAttributesFrom(&spdFile);
                
                if(spdFileOut->getNumberBinsX() == 0)
                {
                    throw SPDException("The number of bins in the X axis needs to be > 0.");
                }
                
                if(spdFileOut->getNumberBinsY() == 0)
                {
                    throw SPDException("The number of bins in the Y axis needs to be > 0.");
                }
                
                if(spdFileOut->getBinSize() == 0)
                {
                    throw SPDException("The bin size needs to be > 0.");
                }
                
                spdWriter = new SPDSeqFileWriter();
                spdWriter->open(spdFileOut, outputFile);
                readerOpen = true;
            }
            catch(SPDException &e)
            {
                cout << "ERROR Opening SPD File: " << e.what() << endl;
                throw e;
            }
            return true;
        };

        void writeDataColumn(boost::python::list pulses, uint_fast32_t col, uint_fast32_t row)throw(SPDException)
        {
            try
            {
                if(readerOpen)
                {
                    vector<SPDPulse*> *outPulses = convertPyList2Vector(&pulses);

                    spdWriter->writeDataColumn(outPulses, col, row);
                    
                    delete outPulses;
                }
                else
                {
                    throw SPDException("Reader is not open for writing.");
                }
            }
            catch(SPDException &e)
            {
                cout << "ERROR Writing to SPD File: " << e.what() << endl;
                throw e;
            }
        };

        void writeDataRow(boost::python::list pulses, uint_fast32_t row)throw(SPDException)
        {
            try
            {
                if(readerOpen)
                {
                    
                    // Number of bins in list
                    boost::python::ssize_t nCols = boost::python::len(pulses);
                    
                    // Interate through bins
                    for(boost::python::ssize_t i = 0; i < nCols; ++i)
                    {
                        
                        // Get column of pulses
                        boost::python::list tmpPulses = boost::python::extract<boost::python::list>(pulses[i]);                        
                        
                        // Convert python sublist to a vector
                        vector<SPDPulse*> *outPulses = convertPyList2Vector(&tmpPulses);
                        
                        // Write data column pulses
                        spdWriter->writeDataColumn(outPulses, i, row);
                        
                        // Clean up
                        delete outPulses;
                    
                    }
                
                }
                else
                {
                    throw SPDException("Reader is not open for writing.");
                }
            }
            catch(SPDException &e)
            {
                cout << "ERROR Writing to SPD File: " << e.what() << endl;
                throw e;
            }
        };

        void close(SPDFile spdFile) throw(SPDException)
        {
            try
            {
                spdFileOut->copyAttributesFrom(&spdFile);
                spdWriter->finaliseClose();
                delete spdWriter;
                delete spdFileOut;
                readerOpen = false;
            }
            catch(SPDException &e)
            {
                cout << "ERROR Closing SPD File: " << e.what() << endl;
                throw e;
            }
        }

        ~SPDPySeqWriter()
        {
            if(readerOpen)
            {
                try
                {
                    spdWriter->finaliseClose();
                    delete spdWriter;
                    delete spdFileOut;
                    readerOpen = false;
                }
                catch(SPDException &e)
                {
                    cerr << "WARNING: error when closing spd write\n";
                }
            }
        };
    private:
        bool readerOpen;
        SPDSeqFileWriter *spdWriter;
        SPDFile *spdFileOut;
    };

    class SPDPyNonSeqWriter
    {
    public:
        SPDPyNonSeqWriter(): readerOpen(false), spdWriter(NULL), spdFileOut(NULL)
        {
            readerOpen = false;
            spdWriter = NULL;
        };
        
        bool open(SPDFile spdFile, string outputFile) throw(SPDException)
        {
            try
            {
                spdFileOut = new SPDFile(outputFile);
                spdFileOut->copyAttributesFrom(&spdFile);
                
                if(spdFileOut->getNumberBinsX() == 0)
                {
                    throw SPDException("The number of bins in the X axis needs to be > 0.");
                }
                
                if(spdFileOut->getNumberBinsY() == 0)
                {
                    throw SPDException("The number of bins in the Y axis needs to be > 0.");
                }
                
                if(spdFileOut->getBinSize() == 0)
                {
                    throw SPDException("The bin size needs to be > 0.");
                }
                
                spdWriter = new SPDNonSeqFileWriter();
                spdWriter->open(spdFileOut, outputFile);
                readerOpen = true;
            }
            catch(SPDException &e)
            {
                cout << "ERROR Openning SPD File: " << e.what() << endl;
                throw e;
            }
            return true;
        };
        
        void writeDataColumn(boost::python::list pulses, uint_fast32_t col, uint_fast32_t row)throw(SPDException)
        {
            try
            {
                if(readerOpen)
                {
                    vector<SPDPulse*> *outPulses = convertPyList2Vector(&pulses);
                    
                    spdWriter->writeDataColumn(outPulses, col, row);
 
                    delete outPulses;
                }
                else
                {
                    throw SPDException("Reader is not open for writing.");
                }
            }
            catch(SPDException &e)
            {
                cout << "ERROR Writing to SPD File: " << e.what() << endl;
                throw e;
            }
        };

        void writeDataRow(boost::python::list pulses, uint_fast32_t row)throw(SPDException)
        {
            try
            {
                if(readerOpen)
                {
                    
                    // Number of bins in list
                    boost::python::ssize_t nCols = boost::python::len(pulses);
                    
                    // Interate through bins
                    for(boost::python::ssize_t i = 0; i < nCols; ++i)
                    {
                        
                        // Get column of pulses
                        boost::python::list tmpPulses = boost::python::extract<boost::python::list>(pulses[i]);                        
                        
                        // Convert python sublist to a vector
                        vector<SPDPulse*> *outPulses = convertPyList2Vector(&tmpPulses);
                        
                        // Write data column pulses
                        spdWriter->writeDataColumn(outPulses, i, row);
                        
                        // Clean up
                        delete outPulses;
                    
                    }
                
                }
                else
                {
                    throw SPDException("Reader is not open for writing.");
                }
            }
            catch(SPDException &e)
            {
                cout << "ERROR Writing to SPD File: " << e.what() << endl;
                throw e;
            }
        };
        
        void close(SPDFile spdFile) throw(SPDException)
        {
            try
            {
                spdFileOut->copyAttributesFrom(&spdFile);
                spdWriter->finaliseClose();
                delete spdWriter;
                delete spdFileOut;
                readerOpen = false;
            }
            catch(SPDException &e)
            {
                cout << "ERROR Closing SPD File: " << e.what() << endl;
                throw e;
            }
        }
        
        ~SPDPyNonSeqWriter()
        {
            if(readerOpen)
            {
                try
                {
                    spdWriter->finaliseClose();
                    delete spdWriter;
                    delete spdFileOut;
                    readerOpen = false;
                }
                catch(SPDException &e)
                {
                    cerr << "WARNING: error when closing spd write\n";
                }
            }
        };
    private:
        bool readerOpen;
        SPDNonSeqFileWriter *spdWriter;
        SPDFile *spdFileOut;
    };

}

#endif


