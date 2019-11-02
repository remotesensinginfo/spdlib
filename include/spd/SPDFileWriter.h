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
    class DllExport SPDFileWriter
    {
    public:
        SPDFileWriter(){};
        virtual ~SPDFileWriter(){};
    protected:
        virtual void writeHeaderInfo(H5::H5File *spdOutH5File, SPDFile *spdFile);
        virtual void updateHeaderInfo(H5::H5File *spdOutH5File, SPDFile *spdFile);
        virtual void updateHeaderInfo(SPDFile *spdFile);
        virtual void readHeaderInfo(H5::H5File *spdH5File, SPDFile *spdFile) ;
    };
    
    
	class DllExport SPDSeqFileWriter : public SPDDataExporter, SPDFileWriter
	{
	public:
		SPDSeqFileWriter();
		SPDSeqFileWriter(const SPDDataExporter &dataExporter) ;
		SPDSeqFileWriter(const SPDSeqFileWriter &dataExporter) ;
        SPDDataExporter* getInstance();
		bool open(SPDFile *spdFile, std::string outputFile) ;
        bool reopen(SPDFile *spdFile, std::string outputFile) ;
		void writeDataColumn(std::list<SPDPulse*> *pls, boost::uint_fast32_t col, boost::uint_fast32_t row);
		void writeDataColumn(std::vector<SPDPulse*> *pls, boost::uint_fast32_t col, boost::uint_fast32_t row);
		void finaliseClose() ;
		bool requireGrid();
		bool needNumOutPts();
		SPDSeqFileWriter& operator=(const SPDSeqFileWriter& dataExporter) ;
		~SPDSeqFileWriter();
	private:
		H5::H5File *spdOutH5File;
		H5::DataSet* pulsesDataset;
		H5::CompType* spdPulseDataType;
		H5::DataSet* pointsDataset;
		H5::CompType* spdPointDataType;
		H5::DataSet *datasetPlsPerBin;
		H5::DataSet *datasetBinsOffset;
		H5::DataSet* receivedDataset;
		H5::DataSet* transmittedDataset;
		H5::DataSet *datasetQuicklook;
        std::vector<SPDPulse*> *plsBuffer;
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
        double scanlineMinWritten;
        double scanlineMaxWritten;
        double scanlineIdxMinWritten;
        double scanlineIdxMaxWritten;
        bool firstReturn;
	};
    
    class DllExport SPDNonSeqFileWriter : public SPDDataExporter, SPDFileWriter
	{
	public:
		SPDNonSeqFileWriter();
		SPDNonSeqFileWriter(const SPDDataExporter &dataExporter) ;
		SPDNonSeqFileWriter(const SPDSeqFileWriter &dataExporter) ;
        SPDDataExporter* getInstance();
		bool open(SPDFile *spdFile, std::string outputFile) ;
        bool reopen(SPDFile *spdFile, std::string outputFile) ;
		void writeDataColumn(std::list<SPDPulse*> *pls, boost::uint_fast32_t col, boost::uint_fast32_t row);
		void writeDataColumn(std::vector<SPDPulse*> *pls, boost::uint_fast32_t col, boost::uint_fast32_t row);
		void finaliseClose() ;
		bool requireGrid();
		bool needNumOutPts();
		SPDNonSeqFileWriter& operator=(const SPDNonSeqFileWriter& dataExporter) ;
		~SPDNonSeqFileWriter();
	private:
		H5::H5File *spdOutH5File;
        H5::DataSet* pulsesDataset;
		H5::CompType* spdPulseDataType;
		H5::DataSet* pointsDataset;
		H5::CompType* spdPointDataType;
		H5::DataSet *datasetPlsPerBin;
		H5::DataSet *datasetBinsOffset;
		H5::DataSet* receivedDataset;
		H5::DataSet* transmittedDataset;
		H5::DataSet *datasetQuicklook;
        std::vector<SPDPulse*> *plsBuffer;
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
        double scanlineMinWritten;
        double scanlineMaxWritten;
        double scanlineIdxMinWritten;
        double scanlineIdxMaxWritten;
        bool firstReturn;
	};
    
    class DllExport SPDNoIdxFileWriter : public SPDDataExporter, SPDFileWriter
	{
	public:
		SPDNoIdxFileWriter();
		SPDNoIdxFileWriter(const SPDDataExporter &dataExporter) ;
		SPDNoIdxFileWriter(const SPDNoIdxFileWriter &dataExporter) ;
        SPDDataExporter* getInstance();
		bool open(SPDFile *spdFile, std::string outputFile) ;
        bool reopen(SPDFile *spdFile, std::string outputFile) ;
		void writeDataColumn(std::list<SPDPulse*> *pls, boost::uint_fast32_t col, boost::uint_fast32_t row);
		void writeDataColumn(std::vector<SPDPulse*> *pls, boost::uint_fast32_t col, boost::uint_fast32_t row);
		void finaliseClose() ;
		bool requireGrid();
		bool needNumOutPts();
		SPDNoIdxFileWriter& operator=(const SPDNoIdxFileWriter& dataExporter) ;
		~SPDNoIdxFileWriter();
	private:
		H5::H5File *spdOutH5File;
		H5::DataSet* pulsesDataset;
		H5::CompType* spdPulseDataType;
		H5::DataSet* pointsDataset;
		H5::CompType* spdPointDataType;
		H5::DataSet* receivedDataset;
		H5::DataSet* transmittedDataset;
        boost::uint_fast64_t numPulses;
        boost::uint_fast64_t numPts;
        boost::uint_fast64_t numTransVals;
        boost::uint_fast64_t numReceiveVals;
		std::vector<SPDPulse*> *plsBuffer;
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
        bool reOpenedFile;
        double scanlineMinWritten;
        double scanlineMaxWritten;
        double scanlineIdxMinWritten;
        double scanlineIdxMaxWritten;
	};
}

#endif





