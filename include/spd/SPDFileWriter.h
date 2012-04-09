 /*
  *  SPDFileWriter.h
  *  spdlib
  *
  *  Created by Pete Bunting on 28/11/2010.
  *  Copyright 2010 SPDLib. All rights reserved.
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

#ifndef SPDFileWriter_H
#define SPDFileWriter_H

#include <iostream>
#include <string>
#include <list>

#include "spd/SPDFile.h"
#include "spd/SPDPoint.h"
#include "spd/SPDPulse.h"
#include "spd/SPDIOException.h"
#include "spd/SPDDataExporter.h"
#include "spd/SPDCommon.h"

#include "H5Cpp.h"

using namespace std;
using namespace H5;

namespace spdlib
{
    class SPDFileWriter
    {
    public:
        SPDFileWriter(){};
        virtual ~SPDFileWriter(){};
    protected:
        virtual void writeHeaderInfo(H5File *spdOutH5File, SPDFile *spdFile)throw(SPDIOException);
        virtual void updateHeaderInfo(H5File *spdOutH5File, SPDFile *spdFile)throw(SPDIOException);
    };
    
    
	class SPDSeqFileWriter : public SPDDataExporter, SPDFileWriter
	{
	public:
		SPDSeqFileWriter();
		SPDSeqFileWriter(const SPDDataExporter &dataExporter) throw(SPDException);
		SPDSeqFileWriter(const SPDSeqFileWriter &dataExporter) throw(SPDException);
        SPDDataExporter* getInstance();
		bool open(SPDFile *spdFile, string outputFile) throw(SPDIOException);
		void writeDataColumn(list<SPDPulse*> *pls,boost::uint_fast32_t col,boost::uint_fast32_t row)throw(SPDIOException);
		void writeDataColumn(vector<SPDPulse*> *pls,boost::uint_fast32_t col,boost::uint_fast32_t row)throw(SPDIOException);
		void finaliseClose() throw(SPDIOException);
		bool requireGrid();
		bool needNumOutPts();
		SPDSeqFileWriter& operator=(const SPDSeqFileWriter& dataExporter) throw(SPDException);
		~SPDSeqFileWriter();
	private:
		H5File *spdOutH5File;
		DataSet* pulsesDataset;
		CompType* spdPulseDataType;
		DataSet* pointsDataset;
		CompType* spdPointDataType;
		DataSet *datasetPlsPerBin;
		DataSet *datasetBinsOffset;
		DataSet* receivedDataset;
		DataSet* transmittedDataset;
		DataSet *datasetQuicklook;
        vector<SPDPulse*> *plsBuffer;
        float *qkBuffer;
        unsigned long *plsInColBuf;
        unsigned long long *plsOffsetBuf;
        boost::uint_fast32_t bufIdxCol;
        boost::uint_fast32_t bufIdxRow;
        boost::uint_fast64_t numPulsesForBuf;
        boost::uint_fast64_t numPulses;
        boost::uint_fast64_t numPts;
        boost::uint_fast64_t numTransVals;
        boost::uint_fast64_t numReceiveVals;
		bool firstColumn;
        bool firstPulse;
        boost::uint_fast32_t nextCol;
        boost::uint_fast32_t nextRow;
        boost::uint_fast32_t numCols;
        boost::uint_fast32_t numRows;
        double xMinWritten;
        double yMinWritten;
        float zMinWritten;
        double xMaxWritten;
        double yMaxWritten;
        float zMaxWritten;
        double azMinWritten;
        double zenMinWritten;
        double ranMinWritten;
        double azMaxWritten;
        double zenMaxWritten;
        double ranMaxWritten;
        bool firstReturn;
	};
    
    class SPDNonSeqFileWriter : public SPDDataExporter, SPDFileWriter
	{
	public:
		SPDNonSeqFileWriter();
		SPDNonSeqFileWriter(const SPDDataExporter &dataExporter) throw(SPDException);
		SPDNonSeqFileWriter(const SPDSeqFileWriter &dataExporter) throw(SPDException);
        SPDDataExporter* getInstance();
		bool open(SPDFile *spdFile, string outputFile) throw(SPDIOException);
		void writeDataColumn(list<SPDPulse*> *pls,boost::uint_fast32_t col,boost::uint_fast32_t row)throw(SPDIOException);
		void writeDataColumn(vector<SPDPulse*> *pls,boost::uint_fast32_t col,boost::uint_fast32_t row)throw(SPDIOException);
		void finaliseClose() throw(SPDIOException);
		bool requireGrid();
		bool needNumOutPts();
		SPDNonSeqFileWriter& operator=(const SPDNonSeqFileWriter& dataExporter) throw(SPDException);
		~SPDNonSeqFileWriter();
	private:
		H5File *spdOutH5File;
		DataSet* pulsesDataset;
		CompType* spdPulseDataType;
		DataSet* pointsDataset;
		CompType* spdPointDataType;
		DataSet *datasetPlsPerBin;
		DataSet *datasetBinsOffset;
		DataSet* receivedDataset;
		DataSet* transmittedDataset;
		DataSet *datasetQuicklook;
        vector<SPDPulse*> *plsBuffer;
        boost::uint_fast64_t numPulses;
        boost::uint_fast64_t numPts;
        boost::uint_fast64_t numTransVals;
        boost::uint_fast64_t numReceiveVals;
		bool firstColumn;
        bool firstPulse;
        unsigned long long plsOffset;
        boost::uint_fast32_t numCols;
        boost::uint_fast32_t numRows;
        double xMinWritten;
        double yMinWritten;
        float zMinWritten;
        double xMaxWritten;
        double yMaxWritten;
        float zMaxWritten;
        double azMinWritten;
        double zenMinWritten;
        double ranMinWritten;
        double azMaxWritten;
        double zenMaxWritten;
        double ranMaxWritten;
        bool firstReturn;
	};
    
    class SPDNoIdxFileWriter : public SPDDataExporter, SPDFileWriter
	{
	public:
		SPDNoIdxFileWriter();
		SPDNoIdxFileWriter(const SPDDataExporter &dataExporter) throw(SPDException);
		SPDNoIdxFileWriter(const SPDNoIdxFileWriter &dataExporter) throw(SPDException);
        SPDDataExporter* getInstance();
		bool open(SPDFile *spdFile, string outputFile) throw(SPDIOException);
		void writeDataColumn(list<SPDPulse*> *pls,boost::uint_fast32_t col,boost::uint_fast32_t row)throw(SPDIOException);
		void writeDataColumn(vector<SPDPulse*> *pls,boost::uint_fast32_t col,boost::uint_fast32_t row)throw(SPDIOException);
		void finaliseClose() throw(SPDIOException);
		bool requireGrid();
		bool needNumOutPts();
		SPDNoIdxFileWriter& operator=(const SPDNoIdxFileWriter& dataExporter) throw(SPDException);
		~SPDNoIdxFileWriter();
	private:
		H5File *spdOutH5File;
		DataSet* pulsesDataset;
		CompType* spdPulseDataType;
		DataSet* pointsDataset;
		CompType* spdPointDataType;
		DataSet* receivedDataset;
		DataSet* transmittedDataset;
        boost::uint_fast64_t numPulses;
        boost::uint_fast64_t numPts;
        boost::uint_fast64_t numTransVals;
        boost::uint_fast64_t numReceiveVals;
		vector<SPDPulse*> *plsBuffer;
        double xMinWritten;
        double yMinWritten;
        float zMinWritten;
        double xMaxWritten;
        double yMaxWritten;
        float zMaxWritten;
        double azMinWritten;
        double zenMinWritten;
        double ranMinWritten;
        double azMaxWritten;
        double zenMaxWritten;
        double ranMaxWritten;
        bool firstReturn;
        bool firstPulse;
        bool firstWaveform;
	};
}

#endif





