/*
 *  SPDFileIncrementalReader.h
 *  spdlib
 *
 *  Created by Pete Bunting on 11/10/2009.
 *  Copyright 2009 SPDLib. All rights reserved.
 *
 *  This file is part of SPDLib.
 *
 *  Permission is hereby granted, free of charge, to any person
 *  obtaining a copy of this software and associated documentation
 *  files (the "Software"), to deal in the Software without restriction,
 *  including without limitation the rights to use, copy, modify,
 *  merge, publish, distribute, sublicense, and/or sell copies of the
 *  Software, and to permit persons to whom the Software is furnished
 *  to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be
 *  included in all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 *  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 *  OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 *  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
 *  ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 *  CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 *  WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 */


#ifndef SPDFileIncrementalReader_H
#define SPDFileIncrementalReader_H

#include <iostream>
#include <list>

#include "ogrsf_frmts.h"

#include <boost/cstdint.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include "H5Cpp.h"

#include "spd/SPDFile.h"
#include "spd/SPDFileReader.h"
#include "spd/SPDPoint.h"
#include "spd/SPDPulse.h"
#include "spd/SPDIOException.h"

// mark all exported classes/functions with DllExport to have
// them exported by Visual Studio
#undef DllExport
#ifdef _MSC_VER
    #ifdef libspdio_EXPORTS
        #define DllExport   __declspec( dllexport )
    #else
        #define DllExport   __declspec( dllimport )
    #endif
#else
    #define DllExport
#endif

namespace spdlib
{
	class DllExport SPDFileIncrementalReader
	{
	public:
		SPDFileIncrementalReader();
		SPDFileIncrementalReader(const SPDFileIncrementalReader &spdReader) throw(SPDException);
		SPDFileIncrementalReader& operator=(const SPDFileIncrementalReader& spdReader) throw(SPDException);
		/**
		 * Open the SPD file.
		 * This function needs to be executed but the other functions in this class can be used.
		 */
		bool open(SPDFile *spdFile) throw(SPDIOException);
        /**
		 * Read the file index for a row
		 * row - the row of the file to be read in
		 * binOffsets - the offsets for each bin
		 * numPtsInBin - the number of points in each bin.
		 */
		void readRefHeaderRow(boost::uint_fast32_t row, unsigned long long *binOffsets, unsigned long *numPtsInBin) throw(SPDIOException);
		/**
		 * Read a row of data
		 * row - the row of the file to be read in
		 * pulses - pulses to std::list<SPDPulse*>[cols]
		 */
		void readPulseDataRow(boost::uint_fast32_t row, std::list<SPDPulse*> **pulses) throw(SPDIOException);
		/**
		 * Read a row of data
		 * row - the row of the file to be read in
		 * pulses - pulses to std::vector<SPDPulse*>[cols]
		 */
		void readPulseDataRow(boost::uint_fast32_t row, std::vector<SPDPulse*> **pulses) throw(SPDIOException);		
		/**
		 * Read a given number of pulses starting from the offset
		 * pulses - pulses to std::list<SPDPulse*>*
		 * row - the row of the file to be read in
		 * startCol - the column (within the row) to start reading.
		 * endCol - the column (within the row) to end reading.
		 */
		void readPulseData(std::list<SPDPulse*> *pulses, boost::uint_fast32_t row, boost::uint_fast32_t startCol, boost::uint_fast32_t endCol) throw(SPDIOException);
		/**
		 * Read a given number of pulses starting from the offset
		 * pulses - pulses to std::vector<SPDPulse*>*
		 * row - the row of the file to be read in
		 * startCol - the column (within the row) to start reading.
		 * endCol - the column (within the row) to end reading.
		 */
		void readPulseData(std::vector<SPDPulse*> *pulses, boost::uint_fast32_t row, boost::uint_fast32_t startCol, boost::uint_fast32_t endCol) throw(SPDIOException);
		/**
		 * Read a given number of pulses starting from the offset
		 * pulses - pulses to std::list<SPDPulse*>*
		 * offset - the offset in the list of pulses (in the HDF/SPD file)
		 * numPts - the number of points to be read.
		 */
		void readPulseData(std::list<SPDPulse*> *pulses, boost::uint_fast64_t offset, boost::uint_fast64_t numPts) throw(SPDIOException);
		/**
		 * Read a given number of pulses starting from the offset
		 * pulses - pulses to std::vector<SPDPulse*>*
		 * offset - the offset in the list of pulses (in the HDF/SPD file)
		 * numPts - the number of points to be read.
		 */
		void readPulseData(std::vector<SPDPulse*> *pulses, boost::uint_fast64_t offset, boost::uint_fast64_t numPts) throw(SPDIOException);
		/**
		 * Read a block of data
		 * pulses - points to std::list<SPDPulse*>[rows][cols]
		 * bbox - [startX, startY, endX, endY]
		 */
		void readPulseDataBlock(std::list<SPDPulse*> ***pulses, boost::uint_fast32_t *bbox) throw(SPDIOException);
		/**
		 * Read a block of data
		 * pulses - points to std::vector<SPDPulse*>[rows][cols]
		 * bbox - [startX, startY, endX, endY]
		 */
		void readPulseDataBlock(std::vector<SPDPulse*> ***pulses, boost::uint_fast32_t *bbox) throw(SPDIOException);
        /**
		 * Read a block of data
		 * pulses - points to std::list<SPDPulse*>[rows][cols]
		 * bbox - [startX, startY, endX, endY]
         * xOff - start offset for adding data to pulses
         * yOff - start offset for adding data to pulses
		 */
		void readPulseDataBlock(std::list<SPDPulse*> ***pulses, boost::uint_fast32_t *bbox, boost::uint_fast32_t xOff, boost::uint_fast32_t yOff) throw(SPDIOException);
		/**
		 * Read a block of data
		 * pulses - points to std::vector<SPDPulse*>[rows][cols]
		 * bbox - [startX, startY, endX, endY]
         * xOff - start offset for adding data to pulses
         * yOff - start offset for adding data to pulses
		 */
		void readPulseDataBlock(std::vector<SPDPulse*> ***pulses, boost::uint_fast32_t *bbox, boost::uint_fast32_t xOff, boost::uint_fast32_t yOff) throw(SPDIOException);
		/**
		 * Read a block of data into a single array
		 * pulses - points to std::list<SPDPulse*>
		 * bbox - [startX, startY, endX, endY]
		 */
		void readPulseDataBlock(std::list<SPDPulse*> *pulses, boost::uint_fast32_t *bbox) throw(SPDIOException);
		/**
		 * Read a block of data into a single array
		 * pulses - points to std::vector<SPDPulse*>
		 * bbox - [startX, startY, endX, endY]
		 */
		void readPulseDataBlock(std::vector<SPDPulse*> *pulses, boost::uint_fast32_t *bbox) throw(SPDIOException);
		/**
		 * Read in pulses within a geographic envelope without using index
		 * pts - points to std::list<SPDPoint*>
		 * env - geographic envelope
		 */
		void readPulseDataInGeoEnv(std::list<SPDPulse*> *pulses, OGREnvelope *env) throw(SPDIOException);
		/**
		 * Read in pulses within a geographic envelope without using index
		 * pts - points to std::vector<SPDPoint*>
		 * env - geographic envelope
		 */
		void readPulseDataInGeoEnv(std::vector<SPDPulse*> *pulses, OGREnvelope *env) throw(SPDIOException);
        /**
		 * Read in pulses within a geometry without using index
		 * pts - points to std::list<SPDPoint*>
		 * geom - OGRGeometry defining the area from which the pulses are to be returned.
		 */
		void readPulseDataInGeom(std::list<SPDPulse*> *pulses, OGRGeometry *geom) throw(SPDIOException);
		/**
		 * Read in pulses within a geometry without using index
		 * pts - points to std::vector<SPDPoint*>
		 * geom - OGRGeometry defining the area from which the pulses are to be returned.
		 */
		void readPulseDataInGeom(std::vector<SPDPulse*> *pulses, OGRGeometry *geom) throw(SPDIOException);
		/**
		 * Calculate the file extent
		 * env - geographic envelope
		 */
		void calcGeoEnv(OGREnvelope *env) throw(SPDIOException);
		/**
		 * Get the a row of the quicklook image
		 */
		void readQKRow(float *data, boost::uint_fast32_t row) throw(SPDIOException);
		/**
		 * Close the SPD file.
		 * This function needs to be executed when you have finished reading from the file.
		 */
		void close() throw(SPDIOException);
		~SPDFileIncrementalReader();	
	private:
		SPDFile *spdFile;
		H5::H5File *spdInFile;
		H5::CompType *pulseType;
		H5::CompType *pointType;
		bool fileOpened;
	};
}

#endif



